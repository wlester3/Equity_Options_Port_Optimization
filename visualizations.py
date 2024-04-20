
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fredapi import Fred
import yfinance as yf


def plot_cumulative_returns(log_returns):
    # Calculate cumulative log returns
    cumulative_log_returns = log_returns.cumsum()
    cumulative_returns = np.exp(cumulative_log_returns) - 1
    
    plt.figure(figsize=(12, 8))
    for column in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[column], label=column)

    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend(title="Stocks", loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=len(cumulative_returns.columns)/2)
    plt.grid(True)
    plt.show()



def plot_paths(S, method,with_mean=False):
    plt.figure(figsize=(10, 6))
    plt.plot(S.T, lw=1)
    if with_mean:
        plt.plot(S.mean(axis=0), 'k', lw=2, label='Mean Path')
        plt.legend()
    plt.title(f'{method} Simulations')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.show()
