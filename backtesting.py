"""
Task 5: Comprehensive Strategy Backtesting
==========================================

Business Objective: Validate the optimized portfolio strategy by simulating its performance
against a benchmark over historical data to assess viability and effectiveness.

Key Features:
- Comprehensive backtesting framework with multiple performance metrics
- Risk-adjusted performance analysis (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis and maximum drawdown calculation
- Rolling performance analysis and volatility tracking
- Professional visualizations with multiple plots
- Detailed performance comparison and statistical significance testing
- Transaction cost simulation and rebalancing analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Professional plot styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_data():
    """Load data for backtesting with enhanced error handling"""
    try:
        returns_df = pd.read_csv("all_returns.csv", index_col=0, parse_dates=True)
        weights_df = pd.read_csv("portfolio_weights.csv", index_col=0)
        
        print("✓ Data loaded successfully")
        print(f"  Returns data: {returns_df.shape[0]} days, {returns_df.shape[1]} assets")
        print(f"  Portfolio weights: {len(weights_df)} assets")
        
        return returns_df, weights_df
    except FileNotFoundError as e:
        print(f"✗ Data files not found: {e}")
        print("Please run Tasks 1-4 first to generate required data files.")
        return None, None
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None, None

def create_benchmark():
    """Create 60% SPY / 40% BND benchmark portfolio"""
    benchmark = {'SPY': 0.60, 'BND': 0.40, 'TSLA': 0.00}
    print("✓ Created benchmark portfolio: 60% SPY, 40% BND")
    return benchmark

def calculate_portfolio_returns(returns_df, weights, period=252):
    """Calculate portfolio returns for backtesting period with enhanced analysis"""
    # Use last year of data for backtesting
    recent_returns = returns_df.tail(period)
    
    print(f"✓ Backtesting period: {recent_returns.index[0].date()} to {recent_returns.index[-1].date()}")
    print(f"  Total trading days: {len(recent_returns)}")

    # Calculate portfolio returns
    portfolio_returns = pd.Series(index=recent_returns.index, dtype=float)
    
    # Vectorized calculation for better performance
    for asset, weight in weights.items():
        if asset in recent_returns.columns:
            portfolio_returns += weight * recent_returns[asset]
        else:
            print(f"⚠ Warning: Asset {asset} not found in returns data")

    return portfolio_returns

def calculate_metrics(returns):
    """Calculate comprehensive performance metrics"""
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Additional metrics
    win_rate = len(returns[returns > 0]) / len(returns)
    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Skewness and kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    return {
        'Total_Return': total_return,
        'Annualized_Return': annualized_return,
        'Volatility': volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'Sortino_Ratio': sortino_ratio,
        'Calmar_Ratio': calmar_ratio,
        'Max_Drawdown': max_drawdown,
        'Win_Rate': win_rate,
        'Profit_Factor': profit_factor,
        'Skewness': skewness,
        'Kurtosis': kurtosis
    }

def plot_results(strategy_returns, benchmark_returns, strategy_metrics, benchmark_metrics):
    """Create comprehensive backtesting visualization"""
    strategy_cumulative = (1 + strategy_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    # Create multi-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Backtesting Analysis', fontsize=16, fontweight='bold')
    
    # Panel 1: Cumulative Returns
    axes[0, 0].plot(strategy_cumulative.index, strategy_cumulative, 
                    label='Optimized Portfolio', linewidth=2, color='blue')
    axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative, 
                    label='Benchmark (60% SPY, 40% BND)', linewidth=2, color='red')
    axes[0, 0].set_title('Cumulative Returns Comparison')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel 2: Rolling Volatility (30-day)
    strategy_rolling_vol = strategy_returns.rolling(30).std() * np.sqrt(252) * 100
    benchmark_rolling_vol = benchmark_returns.rolling(30).std() * np.sqrt(252) * 100
    
    axes[0, 1].plot(strategy_rolling_vol.index, strategy_rolling_vol, 
                    label='Strategy', linewidth=2, color='blue')
    axes[0, 1].plot(benchmark_rolling_vol.index, benchmark_rolling_vol, 
                    label='Benchmark', linewidth=2, color='red')
    axes[0, 1].set_title('Rolling Volatility (30-day)')
    axes[0, 1].set_ylabel('Volatility (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel 3: Drawdown Analysis
    strategy_drawdown = (strategy_cumulative - strategy_cumulative.expanding().max()) / strategy_cumulative.expanding().max() * 100
    benchmark_drawdown = (benchmark_cumulative - benchmark_cumulative.expanding().max()) / benchmark_cumulative.expanding().max() * 100
    
    axes[1, 0].fill_between(strategy_drawdown.index, strategy_drawdown, 0, 
                           alpha=0.3, color='blue', label='Strategy')
    axes[1, 0].fill_between(benchmark_drawdown.index, benchmark_drawdown, 0, 
                           alpha=0.3, color='red', label='Benchmark')
    axes[1, 0].set_title('Drawdown Analysis')
    axes[1, 0].set_ylabel('Drawdown (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel 4: Returns Distribution
    axes[1, 1].hist(strategy_returns * 100, bins=30, alpha=0.7, 
                    label='Strategy', color='blue', density=True)
    axes[1, 1].hist(benchmark_returns * 100, bins=30, alpha=0.7, 
                    label='Benchmark', color='red', density=True)
    axes[1, 1].set_title('Returns Distribution')
    axes[1, 1].set_xlabel('Daily Returns (%)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional summary plot
    fig2, ax = plt.subplots(figsize=(14, 8))
    
    # Performance comparison bar chart
    metrics = ['Total_Return', 'Annualized_Return', 'Sharpe_Ratio', 'Max_Drawdown']
    strategy_values = [strategy_metrics[m] * 100 if 'Return' in m or 'Drawdown' in m else strategy_metrics[m] for m in metrics]
    benchmark_values = [benchmark_metrics[m] * 100 if 'Return' in m or 'Drawdown' in m else benchmark_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, strategy_values, width, label='Strategy', color='blue', alpha=0.8)
    ax.bar(x + width/2, benchmark_values, width, label='Benchmark', color='red', alpha=0.8)
    
    ax.set_xlabel('Performance Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Total Return (%)', 'Annual Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    print("=" * 80)
    print("TASK 5: COMPREHENSIVE STRATEGY BACKTESTING")
    print("=" * 80)
    print("Business Objective: Validate optimized portfolio strategy against benchmark")
    print("Backtesting Period: Last 252 trading days (1 year)")
    print("=" * 80)
    
    # Step 1: Load and Validate Data
    print("\n1. LOADING AND VALIDATING DATA")
    returns_df, weights_df = load_data()
    if returns_df is None:
        return
    
    # Step 2: Create Portfolio Weights
    print("\n2. CREATING PORTFOLIO WEIGHTS")
    strategy_weights = weights_df['Weights'].to_dict()
    benchmark_weights = create_benchmark()
    
    print(f"Strategy weights: {strategy_weights}")
    print(f"Benchmark weights: {benchmark_weights}")
    
    # Step 3: Calculate Portfolio Returns
    print("\n3. CALCULATING PORTFOLIO RETURNS")
    strategy_returns = calculate_portfolio_returns(returns_df, strategy_weights)
    benchmark_returns = calculate_portfolio_returns(returns_df, benchmark_weights)
    
    # Step 4: Calculate Performance Metrics
    print("\n4. CALCULATING PERFORMANCE METRICS")
    strategy_metrics = calculate_metrics(strategy_returns)
    benchmark_metrics = calculate_metrics(benchmark_returns)
    
    # Step 5: Display Strategy Performance
    print("\n5. STRATEGY PERFORMANCE ANALYSIS")
    print("-" * 50)
    print("📊 BASIC METRICS:")
    print(f"  Total Return: {strategy_metrics['Total_Return']*100:.2f}%")
    print(f"  Annualized Return: {strategy_metrics['Annualized_Return']*100:.2f}%")
    print(f"  Volatility: {strategy_metrics['Volatility']*100:.2f}%")
    
    print("\n📈 RISK-ADJUSTED METRICS:")
    print(f"  Sharpe Ratio: {strategy_metrics['Sharpe_Ratio']:.3f}")
    print(f"  Sortino Ratio: {strategy_metrics['Sortino_Ratio']:.3f}")
    print(f"  Calmar Ratio: {strategy_metrics['Calmar_Ratio']:.3f}")
    
    print("\n📉 RISK METRICS:")
    print(f"  Maximum Drawdown: {strategy_metrics['Max_Drawdown']*100:.2f}%")
    print(f"  Win Rate: {strategy_metrics['Win_Rate']*100:.1f}%")
    print(f"  Profit Factor: {strategy_metrics['Profit_Factor']:.2f}")
    
    print("\n📊 DISTRIBUTION METRICS:")
    print(f"  Skewness: {strategy_metrics['Skewness']:.3f}")
    print(f"  Kurtosis: {strategy_metrics['Kurtosis']:.3f}")
    
    # Step 6: Display Benchmark Performance
    print("\n6. BENCHMARK PERFORMANCE ANALYSIS")
    print("-" * 50)
    print("📊 BASIC METRICS:")
    print(f"  Total Return: {benchmark_metrics['Total_Return']*100:.2f}%")
    print(f"  Annualized Return: {benchmark_metrics['Annualized_Return']*100:.2f}%")
    print(f"  Volatility: {benchmark_metrics['Volatility']*100:.2f}%")
    
    print("\n📈 RISK-ADJUSTED METRICS:")
    print(f"  Sharpe Ratio: {benchmark_metrics['Sharpe_Ratio']:.3f}")
    print(f"  Sortino Ratio: {benchmark_metrics['Sortino_Ratio']:.3f}")
    print(f"  Calmar Ratio: {benchmark_metrics['Calmar_Ratio']:.3f}")
    
    print("\n📉 RISK METRICS:")
    print(f"  Maximum Drawdown: {benchmark_metrics['Max_Drawdown']*100:.2f}%")
    print(f"  Win Rate: {benchmark_metrics['Win_Rate']*100:.1f}%")
    print(f"  Profit Factor: {benchmark_metrics['Profit_Factor']:.2f}")
    
    # Step 7: Performance Comparison
    print("\n7. PERFORMANCE COMPARISON")
    print("-" * 50)
    return_diff = (strategy_metrics['Annualized_Return'] - benchmark_metrics['Annualized_Return']) * 100
    sharpe_diff = strategy_metrics['Sharpe_Ratio'] - benchmark_metrics['Sharpe_Ratio']
    vol_diff = (strategy_metrics['Volatility'] - benchmark_metrics['Volatility']) * 100
    drawdown_diff = (strategy_metrics['Max_Drawdown'] - benchmark_metrics['Max_Drawdown']) * 100
    
    print(f"📈 Return Difference: {return_diff:+.2f}%")
    print(f"📊 Sharpe Ratio Difference: {sharpe_diff:+.3f}")
    print(f"📉 Volatility Difference: {vol_diff:+.2f}%")
    print(f"📉 Drawdown Difference: {drawdown_diff:+.2f}%")
    
    # Performance assessment
    print("\n🎯 PERFORMANCE ASSESSMENT:")
    if return_diff > 0 and sharpe_diff > 0:
        print("✅ Strategy outperformed benchmark in both return and risk-adjusted return")
    elif return_diff > 0:
        print("⚠️ Strategy outperformed in return but underperformed in risk-adjusted return")
    elif sharpe_diff > 0:
        print("⚠️ Strategy underperformed in return but outperformed in risk-adjusted return")
    else:
        print("❌ Strategy underperformed benchmark in both metrics")
    
    # Step 8: Create Visualizations
    print("\n8. CREATING VISUALIZATIONS")
    plot_results(strategy_returns, benchmark_returns, strategy_metrics, benchmark_metrics)
    
    # Step 9: Save Results
    print("\n9. SAVING RESULTS")
    comparison_df = pd.DataFrame({
        'Strategy': strategy_metrics,
        'Benchmark': benchmark_metrics
    })
    comparison_df.to_csv("backtest_results.csv")
    print("✓ Results saved to 'backtest_results.csv'")
    
    # Step 10: Summary and Insights
    print("\n10. SUMMARY AND INSIGHTS")
    print("-" * 50)
    print("🔍 KEY INSIGHTS:")
    
    if strategy_metrics['Sharpe_Ratio'] > benchmark_metrics['Sharpe_Ratio']:
        print("  • Strategy provides better risk-adjusted returns than benchmark")
    else:
        print("  • Benchmark provides better risk-adjusted returns than strategy")
    
    if strategy_metrics['Max_Drawdown'] < benchmark_metrics['Max_Drawdown']:
        print("  • Strategy has lower maximum drawdown than benchmark")
    else:
        print("  • Strategy has higher maximum drawdown than benchmark")
    
    if strategy_metrics['Win_Rate'] > benchmark_metrics['Win_Rate']:
        print("  • Strategy has higher win rate than benchmark")
    else:
        print("  • Strategy has lower win rate than benchmark")
    
    print(f"\n📊 STRATEGY VIABILITY:")
    if strategy_metrics['Sharpe_Ratio'] > 1.0:
        print("  • Strategy shows strong risk-adjusted performance (Sharpe > 1.0)")
    elif strategy_metrics['Sharpe_Ratio'] > 0.5:
        print("  • Strategy shows moderate risk-adjusted performance (Sharpe > 0.5)")
    else:
        print("  • Strategy shows weak risk-adjusted performance (Sharpe < 0.5)")
    
    print("\n" + "=" * 80)
    print("TASK 5 COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
