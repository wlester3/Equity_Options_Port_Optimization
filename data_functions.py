
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fredapi import Fred
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm  # If you have a normal approximation
from datetime import datetime
import pandas_market_calendars as mcal
import QuantLib as ql

'''
Data Input and Manipulation:
'''

def get_trading_days(start_date, end_date):
    nyse = mcal.get_calendar('NYSE')

    # Fetch the trading days
    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)
    return trading_days


def connect_apis():
    with open('API_Keys.txt') as key_file:
        key_data=key_file.readlines()
        fred_key,poly_key=key_data[0].strip().split(': ')[1],key_data[1].strip().split(': ')[1]
    fred=Fred(api_key=fred_key)
    return(fred,poly_key)

def get_price_df(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    adj_close = data['Adj Close']
    return adj_close

def get_log_returns(price_df):
    price_df = price_df.ffill()
    log_returns = np.log(price_df / price_df.shift(1))
    log_returns.dropna(inplace=True)

    return log_returns

def days_to_expiration(ticker, symbol):
    date_part = symbol[len(ticker):]  # Assumes date part starts right after the ticker
    exp_year = int(date_part[0:2]) + 2000
    exp_month = int(date_part[2:4])
    exp_day = int(date_part[4:6])
    return exp_day, exp_month, exp_year

def get_options_data(symbol):
    ticker = yf.Ticker(symbol)
    opts = ticker.option_chain()
    calls = opts.calls
    puts = opts.puts

    calls[['Day', 'Month', 'Year']] = calls['contractSymbol'].apply(lambda x: days_to_expiration(symbol, x)).apply(pd.Series)
    puts[['Day', 'Month', 'Year']] = puts['contractSymbol'].apply(lambda x: days_to_expiration(symbol, x)).apply(pd.Series)
    
    return calls, puts

'''
Initial Optimzation Model:
'''

def portfolio_performance(mean_rets, cov_rets, weights):
    returns = np.sum(mean_rets * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_rets, weights))) * np.sqrt(252)
    return returns, std_dev

def neg_sharpe_ratio(weights, mean_rets, cov_rets, rf):
    p_ret, p_std = portfolio_performance(mean_rets, cov_rets, weights)
    return -(p_ret - rf) / p_std

def get_init_port(return_df, risk_free_rate, leverage_limit=1.0):
    mean_rets = return_df.mean()
    cov_rets = return_df.cov()
    
    constraints = {'type': 'eq', 'fun': lambda x: 1 - np.sum(np.abs(x))}
    
    bounds = tuple((-leverage_limit, leverage_limit) for _ in range(len(return_df.columns)))
    
    init_weights = np.array([leverage_limit / len(return_df.columns)] * len(return_df.columns))

    opt_results = minimize(neg_sharpe_ratio, init_weights, args=(mean_rets, cov_rets, risk_free_rate),
                           method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = opt_results.x
    portfolio_return, portfolio_std_dev = portfolio_performance(mean_rets, cov_rets, optimal_weights)
    
    return (optimal_weights, portfolio_return, portfolio_std_dev)


'''
Geometric Brownian Motion (GBM) Methods:
'''

def simulate_gbm(S0, mu, sigma, T, dt=1/252, N=1000):
    num_steps = int(T / dt)  # Number of time steps for the given number of years
    t = np.linspace(0, T, num_steps + 1)  # Time vector from 0 to T years
    
    # Standard normal random numbers
    W = np.random.standard_normal(size=(N, num_steps))
    W = np.cumsum(W, axis=1) * np.sqrt(dt)  # Cumulative sum to create Brownian paths

    # Prepend zeros to make W start at zero
    W = np.hstack((np.zeros((N, 1)), W))
    
    # GBM formula with dt adjusted for daily steps in a year
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)  # Geometric Brownian Motion formula
    
    return S


def simulate_gbm_portfolio(returns_df,stock_df,weights,T=1,dt=1/252,N=10000):
    mean_returns = returns_df.mean() * 252  # Annualize mean
    std_returns = returns_df.std() * np.sqrt(252)  # Annualize std deviation
    S0 = stock_df.iloc[-1] 

    portfolio_paths = np.zeros((N, int(T/dt)+1))
    
    for i, (mean, std, weight) in enumerate(zip(mean_returns, std_returns, weights)):
        S = simulate_gbm(S0.iloc[i], mean, std, T, dt, N)
        portfolio_paths += weight * S  # Weighted sum of asset paths
    
    return pd.DataFrame(portfolio_paths, columns=np.arange(0, T+dt, dt))

'''
Merton Jump Diffusion:
'''

def estimate_merton_params(ret_series):
    z_scores=(ret_series-np.mean(ret_series))/(np.std(ret_series))
    jump_threshold = 3 

    is_jump = np.abs(z_scores) > jump_threshold
    jumps = ret_series[is_jump]

    lambda_ = len(jumps) 
    m = np.mean(jumps)
    v = np.std(jumps)
    return lambda_,m,v


def simulate_merton(S0, mu, sigma, T, lambda_, m, v, dt=1/252, N=1000):
    num_steps = int(T / dt)  # Number of time steps for the given number of years
    t = np.linspace(0, T, num_steps + 1)  # Time vector from 0 to T years
    
    W = np.random.standard_normal(size=(N, num_steps))
    W = np.cumsum(W, axis=1) * np.sqrt(dt)  # Cumulative sum to create Brownian paths
    W = np.hstack((np.zeros((N, 1)), W))

    J = np.random.normal(loc=m, scale=v, size=(N, num_steps))
    Nj = np.random.poisson(lam=lambda_ * dt, size=(N, num_steps))
    J_total = np.multiply(Nj, J)
    J_total = np.hstack((np.zeros((N, 1)), np.cumsum(J_total, axis=1)))

    X = (mu - 0.5 * sigma**2) * t + sigma * W + J_total
    S = S0 * np.exp(X)  # Merton model formula
    
    return S


def simulate_merton_portfolio(returns_df, stock_df, weights, T=1, dt=1/252, N=10000):
    params = [list(estimate_merton_params(returns_df[col])) for col in returns_df]
    mean_m,mean_v=np.mean([parm[2] for parm in params if parm[0]>0.0]),np.mean([parm[2] for parm in params if parm[2]>0.0])
    for param in params:
        if param[2]==0.0:
            param[2]=mean_v

    S0 = stock_df.iloc[-1]

    portfolio_paths = np.zeros((N, int(T/dt)+1))

    for i, ((lambda_, m, v), weight) in enumerate(zip(params, weights)):
        mu = returns_df.iloc[:, i].mean() * 252
        sigma = returns_df.iloc[:, i].std() * np.sqrt(252)
        S = simulate_merton(S0[i], mu, sigma, 1, lambda_, m, v, dt, N)
        portfolio_paths += weight * S  # Weighted sum of asset paths

    return pd.DataFrame(portfolio_paths, columns=np.linspace(0, T, int(T/dt)+1))

def negative_log_likelihood(params, data):
    mu, sigma, beta = params
    likelihood_acc = 0
    S = data[0]
    for S_next in data[1:]:
        predicted_sigma = sigma * S**(beta)
        likelihood_acc += norm.logpdf(S_next, loc=S + mu * S, scale=predicted_sigma)
        S = S_next
    return -likelihood_acc

'''
Constant Elasticity of Variance (CEV) Model:
'''

def estimate_cev_params(stock_series):
    initial_guess = [0.0001, 0.2, 0.5]
    bounds = [(None, None), (1e-5, None), (0, 2)]  # Bounds for mu, sigma, beta
    # Minimize the negative log-likelihood
    result = minimize(negative_log_likelihood, initial_guess, args=(stock_series,), bounds=bounds)
    mu_est, sigma_est, gamma_est = result.x
    print(f'Estimated {stock_series.name} CEV Parameters: mu={mu_est}, std={sigma_est}, gamma={gamma_est}')
    return [mu_est, sigma_est, gamma_est]


def simulate_cev(S0, mu, sigma, gamma, T=1, dt=1/252, N=10000):
    num_steps = int(T / dt)  # Number of time steps for the given number of years
    sqrt_dt = np.sqrt(dt)  # Square root of dt for Brownian motion scaling
    S = np.zeros((N, num_steps + 1))
    S[:, 0] = S0  # Set the initial stock price

    gaussian_increments = np.random.normal(size=(N, num_steps))

    for i in range(1, num_steps + 1):
        S[:, i] = S[:, i-1] * (1 + mu * dt) + sigma * (S[:, i-1] ** gamma) * gaussian_increments[:, i-1] * sqrt_dt

    return S

def simulate_cev_portfolio(stock_df, weights, T=1, dt=1/252, N=10000):
    params=[estimate_cev_params(stock_df[tick]) for tick in stock_df.columns]
    
    S0 = stock_df.iloc[-1]
    portfolio_paths = np.zeros((N, int(T/dt)+1))

    for i, ((cev_mu, cev_std, gammy), weight) in enumerate(zip(params, weights)):
        cev_mu = cev_mu * 252
        cev_std = cev_std * np.sqrt(252)
        S = simulate_cev(S0[i], cev_mu, cev_std, gammy)
        portfolio_paths += weight * S

    return pd.DataFrame(portfolio_paths, columns=np.linspace(0, T, int(T/dt)+1))

'''
Heston Model:
'''

def estimate_heston_params(ticker,end_date,stock_df,risk_free_rate):
    calls,puts=get_options_data(ticker)

    today=ql.Date(end_date.day,end_date.month,end_date.year)
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    day_count = ql.Actual365Fixed()
    try:
        dividend_rate=yf.Ticker(ticker).info['dividendYield']
    except KeyError:
        dividend_rate=0

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(stock_df[ticker].iloc[-1]))
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, day_count))
    dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_rate, day_count))

    v0_init = 0.01  # initial variance
    kappa_init = 0.2  # rate of mean reversion
    theta_init = 0.02  # long-term variance
    sigma_init = 0.5  # volatility of volatility
    rho_init = -0.75  # correlation between the two Brownian motions

    process = ql.HestonProcess(rate_handle, dividend_handle, spot_handle, v0_init, kappa_init, theta_init, sigma_init, rho_init)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model,128)

    helpers = []
    options_data = [(calls, ql.Option.Call), (puts, ql.Option.Put)]

    for option_df, opt_type in options_data:
        for index, row in option_df.iterrows():
            strike = row['strike']
            mid_price = row['lastPrice']
            maturity_date = ql.Date(row['Day'], row['Month'], row['Year'])
            underlying=stock_df[ticker].asof(row['lastTradeDate'].strftime('%Y-%m-%d'))
            last_date=ql.Date(row['lastTradeDate'].day,row['lastTradeDate'].month,row['lastTradeDate'].year)
            t = maturity_date-last_date
            
            payoff = ql.PlainVanillaPayoff(opt_type, strike)
            exercise = ql.EuropeanExercise(maturity_date)
            helper = ql.HestonModelHelper(ql.Period(int(t), ql.Days), calendar, underlying, strike, ql.QuoteHandle(ql.SimpleQuote(mid_price)), rate_handle, dividend_handle)
            helper.setPricingEngine(engine)
            helpers.append(helper)

    cost_thresh=1.3
    helpers=[help for help in helpers if (np.abs(help.calibrationError())<cost_thresh)]


    for i, helper in enumerate(helpers):
        cost_pre = helper.calibrationError()
        #print(f"Pre-calibration cost for helper {i}: {cost_pre}")

    method = ql.LevenbergMarquardt()
    model.calibrate(helpers, method, ql.EndCriteria(1000, 500, 1e-8, 1e-8, 1e-8))

    theta, kappa, sigma, rho, v0 = model.params()
    print(f"Estimated {ticker} Heston Parameters: theta={theta}, kappa={kappa}, sigma={sigma}, rho={rho}, v0={v0}")
    return theta, kappa, sigma, rho, v0


def simulate_heston(params, S0, r, T=1, dt=1/252, N=1000):
    theta, kappa, sigma, rho, v0 = params
    num_steps = int(T / dt)+1  # Ensure num_steps is an integer

    # Define the risk-free rate curve
    riskFreeRate = ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(r)), ql.Actual365Fixed())
    riskFreeRateHandle = ql.YieldTermStructureHandle(riskFreeRate)

    # Setup the Heston process
    heston_process = ql.HestonProcess(riskFreeRateHandle,
                                      ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), 0, ql.Actual365Fixed())),
                                      ql.QuoteHandle(ql.SimpleQuote(S0)),
                                      v0, kappa, theta, sigma, rho)

    # Create the Heston model
    heston_model = ql.HestonModel(heston_process)

    # Setup the random number generator
    rng = ql.UniformRandomGenerator()
    dimension = 2  # Two dimensions for the Heston model (asset price and variance)
    sequence_generator = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(dimension * num_steps, rng))  # Correct dimensionality
    time_grid = ql.TimeGrid(T, num_steps)
    path_generator = ql.GaussianMultiPathGenerator(
        heston_process, time_grid, sequence_generator, False)
    
    # Generate the paths
    paths = []
    for i in range(N):
        sample_path = path_generator.next()
        multi_path = sample_path.value()
        asset_path = [multi_path[0][j] for j in range(num_steps)]  # Path for the stock price
        paths.append(asset_path)
    
    return pd.DataFrame(paths)

def simulate_heston_portfolio(stock_df, weights, risk_free_rate, T=1, dt=1/252, N=10000):
    params = [estimate_heston_params(tick, stock_df.index[-1], stock_df, risk_free_rate) for tick in stock_df.columns]
    
    S0 = stock_df.iloc[-1]
    num_steps = int(T / dt) + 1  # Adjusted to match the number of columns in the resulting DataFrame
    portfolio_paths = np.zeros((N, num_steps))

    for i, (param, weight) in enumerate(zip(params, weights)):
        S = simulate_heston(param, S0[i], risk_free_rate, T, dt, N)
        portfolio_paths += weight * S.values[:, :num_steps]  # Ensure the same number of steps in each path

    return pd.DataFrame(portfolio_paths, columns=np.linspace(0, T, num_steps))

'''
Simulation Portfolio Optimization
'''

def calculate_sharpe_ratio_for_path(weights, returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, returns)
    excess_return = portfolio_return - risk_free_rate
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_std_dev = np.sqrt(portfolio_variance)
    sharpe_ratio = excess_return / portfolio_std_dev
    return sharpe_ratio

def negative_average_sharpe(weights,tickers,tick_sim_ret,risk_free_rate):
    num_paths = tick_sim_ret[tickers[0]].shape[0]
    sharpe_ratios = []
    
    for path_index in range(num_paths):
        path_returns = np.array([tick_sim_ret[tick][path_index, :] for tick in tickers])
        mean_returns = np.mean(path_returns, axis=1)
        cov_matrix = np.cov(path_returns)
    
        sharpe_ratio = calculate_sharpe_ratio_for_path(weights, mean_returns, cov_matrix,risk_free_rate)
        sharpe_ratios.append(sharpe_ratio)
    
    average_sharpe = np.mean(sharpe_ratios)
    
    return -average_sharpe

def calculate_simulate_portfolio(tickers,end_date,stock_df,returns_df,risk_free_rate,leverage_limit,sim_num):
    tick_sim_ret={tick:None for tick in tickers}

    for tick in tickers:
        tick_price=stock_df[tick].iloc[-1]
        tick_ret=np.mean(returns_df[tick])*252
        tick_std=np.std(returns_df[tick])*np.sqrt(252)

        tick_gbm=simulate_gbm(tick_price,tick_ret,tick_std,1,N=sim_num)
        t_lambda,t_m,t_v=estimate_merton_params(returns_df[tick])
        tick_merton=simulate_merton(tick_price,tick_ret,tick_std,1,t_lambda,t_m,t_v,1/252,N=sim_num)
        tick_cev_params=estimate_cev_params(stock_series=stock_df[tick])
        tick_cev=simulate_cev(tick_price,tick_cev_params[0]*252,tick_cev_params[1]*np.sqrt(252), gamma=tick_cev_params[2], T=1,N=sim_num)
        tick_heston_param=estimate_heston_params(tick,end_date,stock_df,risk_free_rate)
        tick_heston=simulate_heston(tick_heston_param, tick_price, risk_free_rate,N=sim_num)

        tick_sim=np.vstack((tick_gbm, tick_merton,tick_cev,tick_heston))
        tick_returns = np.log(tick_sim[:, 1:]/tick_sim[:, :-1])
        tick_sim_ret[tick]=tick_returns

    constraints = {'type': 'eq', 'fun': lambda x: 1 - np.sum(np.abs(x))}
    bounds = tuple((-leverage_limit, leverage_limit) for _ in tickers)  

    initial_weights = np.array([leverage_limit / len(tickers)] * len(tickers))

    result = minimize(lambda w: negative_average_sharpe(w, tickers, tick_sim_ret, risk_free_rate), initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result