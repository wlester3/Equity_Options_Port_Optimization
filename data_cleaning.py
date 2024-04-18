
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fredapi import Fred
import yfinance as yf
from scipy.optimize import minimize

'''
Data Input and Manipulation:
'''

def connect_apis():
    with open('API_Keys.txt') as key_file:
        key_data=key_file.readlines()
        fred_key,poly_key=key_data[0].strip().split(': ')[1],key_data[1].strip().split(': ')[1]
    fred=Fred(api_key=fred_key)
    poly=None
    return(fred,poly)

def get_price_df(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    adj_close = data['Adj Close']
    return adj_close

def get_log_returns(price_df):
    price_df = price_df.ffill()
    log_returns = np.log(price_df / price_df.shift(1))
    log_returns.dropna(inplace=True)

    return log_returns

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


def simulate_gbm_portfolio(returns_df,stock_df,weights,T=1,dt=1/252,N=1000):
    mean_returns = returns_df.mean() * 252  # Annualize mean
    std_returns = returns_df.std() * np.sqrt(252)  # Annualize std deviation
    S0 = stock_df.iloc[-1] 

    portfolio_paths = np.zeros((N, int(T/dt)+1))
    
    for i, (mean, std, weight) in enumerate(zip(mean_returns, std_returns, weights)):
        S = simulate_gbm(S0.iloc[i], mean, std, T, dt, N)
        portfolio_paths += weight * S  # Weighted sum of asset paths
    
    return pd.DataFrame(portfolio_paths, columns=np.arange(0, T+dt, dt))