"""
Task 4: Portfolio Optimization
=============================

Simple portfolio optimization using Modern Portfolio Theory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_data():
    """Load data for optimization"""
    try:
        returns_df = pd.read_csv("all_returns.csv", index_col=0, parse_dates=True)
        forecast_data = pd.read_csv("ARIMA_forecast.csv", index_col=0, parse_dates=True)
        return returns_df, forecast_data
    except FileNotFoundError:
        print("Data files not found. Please run previous tasks first.")
        return None, None

def calculate_expected_returns(returns_df, forecast_data):
    """Calculate expected returns"""
    bnd_return = returns_df['BND'].mean() * 252
    spy_return = returns_df['SPY'].mean() * 252
    
    # Simple forecast return calculation
    tsla_forecast_return = 0.15  # Assume 15% annual return
    
    return pd.Series({
        'TSLA': tsla_forecast_return,
        'BND': bnd_return,
        'SPY': spy_return
    })

def optimize_portfolio(expected_returns, cov_matrix):
    """Optimize portfolio weights"""
    n_assets = len(expected_returns)
    
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def portfolio_return(weights):
        return np.sum(weights * expected_returns)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Minimize volatility
    result = minimize(portfolio_volatility, [1/n_assets]*n_assets, 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = result.x
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        
        print(f"\nOptimal Portfolio:")
        for i, asset in enumerate(expected_returns.index):
            print(f"  {asset}: {weights[i]:.4f} ({weights[i]*100:.2f}%)")
        print(f"Expected Return: {ret:.4f}")
        print(f"Volatility: {vol:.4f}")
        
        return weights, ret, vol
    else:
        print("Optimization failed")
        return None, None, None

def main():
    print("=" * 80)
    print("TASK 4: PORTFOLIO OPTIMIZATION")
    print("=" * 80)
    
    returns_df, forecast_data = load_data()
    if returns_df is None:
        return
    
    expected_returns = calculate_expected_returns(returns_df, forecast_data)
    cov_matrix = returns_df.cov() * 252
    
    weights, ret, vol = optimize_portfolio(expected_returns, cov_matrix)
    
    if weights is not None:
        # Save results
        weights_df = pd.DataFrame({'Weights': weights}, index=expected_returns.index)
        weights_df.to_csv("portfolio_weights.csv")
        print("\nResults saved to portfolio_weights.csv")
    
    print("\nTASK 4 COMPLETED!")

if __name__ == "__main__":
    main()
