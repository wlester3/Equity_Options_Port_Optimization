import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pykalman import KalmanFilter
import yfinance as yf

def transition_matrix(params):
    kappa, theta, sigma, rho = params
    return np.array([[1 - kappa, 0],
                     [0, 1 - rho]])

def observation_matrix(params):
    return np.array([1, 0])

def likelihood(params):
    kf = KalmanFilter(transition_matrices=transition_matrix(params),
                      observation_matrices=observation_matrix(params),
                      initial_state_mean=np.zeros(2),
                      initial_state_covariance=np.eye(2),
                      observation_covariance=1,
                      transition_covariance=np.diag([params[2]**2, params[3]**2]))
    try:
        log_likelihood = kf.loglikelihood(returns.values)
        return -log_likelihood
    except np.linalg.LinAlgError:
        return np.inf  # Return a large number to signify error

# Example usage
data = yf.download("WMT", start="2023-01-01", end="2024-01-01")
returns = np.log(data['Adj Close']).diff().dropna()
initial_guess = [0.2, 0.04, 0.2, 0.1]
bounds = [(0.1, 5), (0.0001, 0.1), (0.1, 2), (-1, 1)]

result = minimize(likelihood, initial_guess, bounds=bounds, method='L-BFGS-B')
print('Optimized Parameters:', result.x)
