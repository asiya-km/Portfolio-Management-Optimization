"""
Task 5: Strategy Backtesting
============================

Simple backtesting of optimized portfolio against benchmark.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Load data for backtesting"""
    try:
        returns_df = pd.read_csv("all_returns.csv", index_col=0, parse_dates=True)
        weights_df = pd.read_csv("portfolio_weights.csv", index_col=0)
        return returns_df, weights_df
    except FileNotFoundError:
        print("Data files not found. Please run previous tasks first.")
        return None, None

def create_benchmark():
    """Create 60% SPY / 40% BND benchmark"""
    return {'SPY': 0.60, 'BND': 0.40, 'TSLA': 0.00}

def calculate_portfolio_returns(returns_df, weights, period=252):
    """Calculate portfolio returns for last year"""
    # Use last year of data
    recent_returns = returns_df.tail(period)
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(index=recent_returns.index, dtype=float)
    
    for date in recent_returns.index:
        daily_return = 0
        for asset, weight in weights.items():
            if asset in recent_returns.columns:
                daily_return += weight * recent_returns.loc[date, asset]
        portfolio_returns[date] = daily_return
    
    return portfolio_returns

def calculate_metrics(returns):
    """Calculate performance metrics"""
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    return {
        'Total_Return': total_return,
        'Annualized_Return': annualized_return,
        'Volatility': volatility,
        'Sharpe_Ratio': sharpe_ratio
    }

def plot_results(strategy_returns, benchmark_returns):
    """Plot backtesting results"""
    strategy_cumulative = (1 + strategy_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_cumulative.index, strategy_cumulative, 
             label='Optimized Portfolio', linewidth=2)
    plt.plot(benchmark_cumulative.index, benchmark_cumulative, 
             label='Benchmark (60% SPY, 40% BND)', linewidth=2)
    plt.title('Backtesting Results - Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    print("=" * 80)
    print("TASK 5: STRATEGY BACKTESTING")
    print("=" * 80)
    
    # Load data
    returns_df, weights_df = load_data()
    if returns_df is None:
        return
    
    # Create weights
    strategy_weights = weights_df['Weights'].to_dict()
    benchmark_weights = create_benchmark()
    
    print(f"Strategy weights: {strategy_weights}")
    print(f"Benchmark weights: {benchmark_weights}")
    
    # Calculate returns
    strategy_returns = calculate_portfolio_returns(returns_df, strategy_weights)
    benchmark_returns = calculate_portfolio_returns(returns_df, benchmark_weights)
    
    # Calculate metrics
    strategy_metrics = calculate_metrics(strategy_returns)
    benchmark_metrics = calculate_metrics(benchmark_returns)
    
    print("\nSTRATEGY PERFORMANCE:")
    for metric, value in strategy_metrics.items():
        if 'Return' in metric:
            print(f"  {metric}: {value*100:.2f}%")
        else:
            print(f"  {metric}: {value:.4f}")
    
    print("\nBENCHMARK PERFORMANCE:")
    for metric, value in benchmark_metrics.items():
        if 'Return' in metric:
            print(f"  {metric}: {value*100:.2f}%")
        else:
            print(f"  {metric}: {value:.4f}")
    
    # Plot results
    plot_results(strategy_returns, benchmark_returns)
    
    # Compare performance
    print("\nPERFORMANCE COMPARISON:")
    print("-" * 30)
    return_diff = (strategy_metrics['Annualized_Return'] - benchmark_metrics['Annualized_Return']) * 100
    sharpe_diff = strategy_metrics['Sharpe_Ratio'] - benchmark_metrics['Sharpe_Ratio']
    
    print(f"Return difference: {return_diff:.2f}%")
    print(f"Sharpe ratio difference: {sharpe_diff:.3f}")
    
    if return_diff > 0:
        print("✓ Strategy outperformed benchmark")
    else:
        print("✗ Strategy underperformed benchmark")
    
    # Save results
    comparison_df = pd.DataFrame({
        'Strategy': strategy_metrics,
        'Benchmark': benchmark_metrics
    })
    comparison_df.to_csv("backtest_results.csv")
    print("\nResults saved to backtest_results.csv")
    
    print("\nTASK 5 COMPLETED!")

if __name__ == "__main__":
    main()
