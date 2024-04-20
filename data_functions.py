
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fredapi import Fred
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm  # If you have a normal approximation
from datetime import datetime
import pandas_market_calendars as mcal

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

def options_chain(symbol, start_date, end_date):
    tk = yf.Ticker(symbol)
    exps = tk.options
    
    exps = [e for e in exps if start_date <= pd.to_datetime(e) <= end_date]

    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt_calls = pd.DataFrame(opt.calls)
        opt_puts = pd.DataFrame(opt.puts)
        opt_combined = pd.concat([opt_calls, opt_puts])
        opt_combined['expirationDate'] = e
        options = pd.concat([options, opt_combined], ignore_index=True)

    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days=1)
    options['dte'] = (options['expirationDate'] - pd.to_datetime(datetime.datetime.now().date())).dt.days / 365
    
    options['CALL'] = options['contractSymbol'].str[4:].apply(lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric, errors='coerce')
    options['mark'] = (options['bid'] + options['ask']) / 2
    
    options = options.drop(columns=['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])
    
    return options


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

def get_init_port(return_df, risk_free_rate):
    mean_rets = return_df.mean()
    cov_rets = return_df.cov()
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(return_df.columns)))
    init_weights = np.ones(len(return_df.columns)) / len(return_df.columns)
    
    opt_results = minimize(neg_sharpe_ratio, init_weights, args=(mean_rets, cov_rets, risk_free_rate),
                           method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = opt_results.x
    portfolio_return, portfolio_std_dev = portfolio_performance(mean_rets, cov_rets, optimal_weights)
    
    return (optimal_weights, portfolio_return, portfolio_std_dev)


'''
Simulation Methods:
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