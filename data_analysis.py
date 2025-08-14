"""
Task 1: Preprocess and Explore the Data
=======================================

This script performs comprehensive data preprocessing and exploration for the time series
forecasting project. It includes data fetching, cleaning, EDA, stationarity testing,
volatility analysis, seasonality detection, and outlier analysis.

Business Context:
- TSLA: High-growth, high-risk stock in consumer discretionary sector
- BND: Bond ETF tracking U.S. investment-grade bonds for stability
- SPY: ETF tracking S&P 500 Index for broad market exposure

Key Features:
- Comprehensive data validation and cleaning
- Advanced statistical analysis and visualizations
- Stationarity testing with Augmented Dickey-Fuller test
- Seasonality and trend analysis
- Outlier detection using multiple methods
- Risk metrics calculation (VaR, Sharpe ratio, drawdown)
- Professional documentation and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

from utils import (
    fetch_stock_data, calculate_returns, calculate_volatility, 
    calculate_sharpe_ratio, calculate_var, plot_stock_prices,
    plot_returns, plot_volatility, print_summary_statistics,
    check_missing_values, handle_missing_values, create_returns_dataframe
)

def analyze_seasonality(data_dict):
    """
    Analyze seasonality in the time series data
    
    Parameters:
    data_dict (dict): Dictionary with symbol as key and DataFrame as value
    """
    print("\n" + "="*60)
    print("SEASONALITY ANALYSIS")
    print("="*60)
    
    for symbol, df in data_dict.items():
        print(f"\n{symbol} Seasonality Analysis:")
        print("-" * 40)
        
        # Use closing prices for seasonality analysis
        prices = df['Close']
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(prices, model='additive', period=252)
            
            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            decomposition.observed.plot(ax=axes[0], title=f'{symbol} - Observed')
            axes[0].set_ylabel('Price ($)')
            axes[0].grid(True, alpha=0.3)
            
            decomposition.trend.plot(ax=axes[1], title=f'{symbol} - Trend')
            axes[1].set_ylabel('Trend')
            axes[1].grid(True, alpha=0.3)
            
            decomposition.seasonal.plot(ax=axes[2], title=f'{symbol} - Seasonal')
            axes[2].set_ylabel('Seasonal')
            axes[2].grid(True, alpha=0.3)
            
            decomposition.resid.plot(ax=axes[3], title=f'{symbol} - Residual')
            axes[3].set_ylabel('Residual')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Calculate seasonal strength
            seasonal_strength = np.std(decomposition.seasonal) / np.std(decomposition.resid)
            print(f"  Seasonal Strength: {seasonal_strength:.4f}")
            
            if seasonal_strength > 0.6:
                print("  Interpretation: Strong seasonality detected")
            elif seasonal_strength > 0.3:
                print("  Interpretation: Moderate seasonality detected")
            else:
                print("  Interpretation: Weak or no seasonality detected")
                
        except Exception as e:
            print(f"  Error in seasonality analysis: {e}")

def enhanced_outlier_detection(data_dict):
    """
    Enhanced outlier detection using multiple methods
    
    Parameters:
    data_dict (dict): Dictionary with symbol as key and DataFrame as value
    """
    print("\n" + "="*60)
    print("ENHANCED OUTLIER DETECTION")
    print("="*60)
    
    for symbol, df in data_dict.items():
        print(f"\n{symbol} Outlier Analysis:")
        print("-" * 40)
        
        returns = calculate_returns(df)
        
        # Method 1: Z-score method
        z_scores = np.abs(stats.zscore(returns))
        outliers_z = returns[z_scores > 3]
        
        # Method 2: IQR method
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = returns[(returns < (Q1 - 1.5 * IQR)) | (returns > (Q3 + 1.5 * IQR))]
        
        # Method 3: Modified Z-score (more robust)
        median = returns.median()
        mad = np.median(np.abs(returns - median))
        modified_z_scores = 0.6745 * (returns - median) / mad
        outliers_modified_z = returns[np.abs(modified_z_scores) > 3.5]
        
        # Method 4: Percentile-based
        outliers_percentile = returns[(returns < returns.quantile(0.01)) | 
                                    (returns > returns.quantile(0.99))]
        
        print(f"  Total observations: {len(returns)}")
        print(f"  Z-score outliers (>3): {len(outliers_z)} ({len(outliers_z)/len(returns)*100:.2f}%)")
        print(f"  IQR outliers: {len(outliers_iqr)} ({len(outliers_iqr)/len(returns)*100:.2f}%)")
        print(f"  Modified Z-score outliers: {len(outliers_modified_z)} ({len(outliers_modified_z)/len(returns)*100:.2f}%)")
        print(f"  Percentile outliers (1%): {len(outliers_percentile)} ({len(outliers_percentile)/len(returns)*100:.2f}%)")
        
        # Plot outlier analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Box plot
        axes[0, 0].boxplot(returns, labels=[symbol])
        axes[0, 0].set_title(f'{symbol} - Box Plot')
        axes[0, 0].set_ylabel('Daily Returns')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram with outliers highlighted
        axes[0, 1].hist(returns, bins=50, alpha=0.7, edgecolor='black', label='Normal')
        axes[0, 1].hist(outliers_z, bins=50, alpha=0.7, color='red', edgecolor='black', label='Z-score Outliers')
        axes[0, 1].set_title(f'{symbol} - Returns Distribution with Outliers')
        axes[0, 1].set_xlabel('Daily Returns')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(returns, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'{symbol} - Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Time series with outliers
        axes[1, 1].plot(returns.index, returns, alpha=0.7, label='Returns')
        axes[1, 1].scatter(outliers_z.index, outliers_z, color='red', alpha=0.7, label='Outliers')
        axes[1, 1].set_title(f'{symbol} - Time Series with Outliers')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Daily Returns')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Summary of extreme outliers
        if len(outliers_z) > 0:
            print(f"  Largest outlier: {outliers_z.max():.6f} on {outliers_z.idxmax().date()}")
            print(f"  Smallest outlier: {outliers_z.min():.6f} on {outliers_z.idxmin().date()}")

def comprehensive_statistical_analysis(data_dict):
    """
    Comprehensive statistical analysis for each asset
    
    Parameters:
    data_dict (dict): Dictionary with symbol as key and DataFrame as value
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*60)
    
    for symbol, df in data_dict.items():
        print(f"\n{symbol} Comprehensive Analysis:")
        print("-" * 50)
        
        returns = calculate_returns(df)
        prices = df['Close']
        
        # Basic statistics
        print("Basic Statistics:")
        print(f"  Data points: {len(df)}")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Trading days: {len(df)}")
        print(f"  Years of data: {len(df)/252:.2f}")
        
        # Price statistics
        print(f"\nPrice Statistics:")
        print(f"  Starting price: ${prices.iloc[0]:.2f}")
        print(f"  Ending price: ${prices.iloc[-1]:.2f}")
        print(f"  Highest price: ${prices.max():.2f}")
        print(f"  Lowest price: ${prices.min():.2f}")
        print(f"  Price range: ${prices.max() - prices.min():.2f}")
        
        # Return statistics
        print(f"\nReturn Statistics:")
        print(f"  Mean daily return: {returns.mean():.6f}")
        print(f"  Median daily return: {returns.median():.6f}")
        print(f"  Std daily return: {returns.std():.6f}")
        print(f"  Annualized return: {returns.mean() * 252 * 100:.2f}%")
        print(f"  Annualized volatility: {returns.std() * np.sqrt(252) * 100:.2f}%")
        
        # Distribution statistics
        print(f"\nDistribution Statistics:")
        print(f"  Skewness: {returns.skew():.3f}")
        print(f"  Kurtosis: {returns.kurtosis():.3f}")
        print(f"  Jarque-Bera test p-value: {stats.jarque_bera(returns)[1]:.6f}")
        
        # Risk metrics
        print(f"\nRisk Metrics:")
        print(f"  VaR (5%): {calculate_var(returns) * 100:.2f}%")
        print(f"  VaR (1%): {np.percentile(returns, 1) * 100:.2f}%")
        print(f"  Sharpe Ratio: {calculate_sharpe_ratio(returns):.3f}")
        print(f"  Maximum drawdown: {((prices / prices.expanding().max()) - 1).min() * 100:.2f}%")
        
        # Volatility analysis
        volatility_30d = calculate_volatility(returns, window=30)
        volatility_60d = calculate_volatility(returns, window=60)
        print(f"\nVolatility Analysis:")
        print(f"  30-day avg volatility: {volatility_30d.mean() * 100:.2f}%")
        print(f"  60-day avg volatility: {volatility_60d.mean() * 100:.2f}%")
        print(f"  Volatility of volatility: {volatility_30d.std() * 100:.2f}%")

def plot_advanced_analysis(data_dict):
    """
    Create advanced visualizations for comprehensive analysis
    
    Parameters:
    data_dict (dict): Dictionary with symbol as key and DataFrame as value
    """
    print("\n" + "="*60)
    print("ADVANCED VISUALIZATIONS")
    print("="*60)
    
    # 1. Combined price comparison
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for symbol, df in data_dict.items():
        # Normalize prices to start at 100 for comparison
        normalized_prices = df['Close'] / df['Close'].iloc[0] * 100
        plt.plot(normalized_prices.index, normalized_prices, label=symbol, linewidth=2)
    
    plt.title('Normalized Price Comparison (Base=100)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Rolling correlation
    plt.subplot(2, 2, 2)
    returns_df = create_returns_dataframe(data_dict)
    rolling_corr = returns_df['TSLA'].rolling(window=60).corr(returns_df['SPY'])
    plt.plot(rolling_corr.index, rolling_corr, label='TSLA-SPY 60-day correlation', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Rolling Correlation: TSLA vs SPY', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Volatility clustering
    plt.subplot(2, 2, 3)
    for symbol, df in data_dict.items():
        returns = calculate_returns(df)
        volatility = calculate_volatility(returns, window=30)
        plt.plot(volatility.index, volatility, label=symbol, linewidth=2)
    
    plt.title('30-Day Rolling Volatility', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Returns distribution comparison
    plt.subplot(2, 2, 4)
    for symbol, df in data_dict.items():
        returns = calculate_returns(df)
        plt.hist(returns, bins=50, alpha=0.6, label=symbol, density=True)
    
    plt.title('Returns Distribution Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Daily Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 5. Correlation heatmap with enhanced styling
    plt.figure(figsize=(10, 8))
    correlation_matrix = returns_df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Enhanced heatmap
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True, 
                linewidths=0.5,
                cbar_kws={"shrink": .8},
                fmt='.3f')
    
    plt.title('Correlation Matrix of Daily Returns', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute comprehensive data preprocessing and exploration
    """
    print("=" * 80)
    print("TASK 1: COMPREHENSIVE DATA PREPROCESSING AND EXPLORATION")
    print("=" * 80)
    print("Business Objective: Analyze TSLA, BND, and SPY for portfolio optimization")
    print("Data Period: July 1, 2015 to July 31, 2025")
    print("=" * 80)
    
    # Define parameters
    symbols = ['TSLA', 'BND', 'SPY']
    start_date = '2015-07-01'
    end_date = '2025-07-31'
    
    # Step 1: Fetch Data
    print("\n1. FETCHING HISTORICAL DATA")
    print("-" * 40)
    data = fetch_stock_data(symbols, start_date, end_date)
    
    if not data:
        print("No data fetched. Exiting...")
        return
    
    # Step 2: Check for Missing Values
    print("\n2. CHECKING FOR MISSING VALUES")
    print("-" * 40)
    check_missing_values(data)
    
    # Step 3: Handle Missing Values
    print("\n3. HANDLING MISSING VALUES")
    print("-" * 40)
    cleaned_data = handle_missing_values(data, method='ffill')
    
    # Step 4: Basic Statistics
    print("\n4. BASIC STATISTICS")
    print("-" * 40)
    print_summary_statistics(cleaned_data)
    
    # Step 5: Comprehensive Statistical Analysis
    print("\n5. COMPREHENSIVE STATISTICAL ANALYSIS")
    print("-" * 40)
    comprehensive_statistical_analysis(cleaned_data)
    
    # Step 6: Exploratory Data Analysis with Enhanced Visualizations
    print("\n6. EXPLORATORY DATA ANALYSIS")
    print("-" * 40)
    
    # Plot stock prices
    print("Plotting stock prices...")
    plot_stock_prices(cleaned_data, "Stock Prices Over Time (2015-2025)")
    
    # Plot daily returns
    print("Plotting daily returns...")
    plot_returns(cleaned_data, "Daily Returns")
    
    # Plot volatility
    print("Plotting rolling volatility...")
    plot_volatility(cleaned_data, window=30, title="30-Day Rolling Volatility")
    
    # Step 7: Advanced Visualizations
    print("\n7. ADVANCED VISUALIZATIONS")
    print("-" * 40)
    plot_advanced_analysis(cleaned_data)
    
    # Step 8: Seasonality Analysis
    print("\n8. SEASONALITY ANALYSIS")
    print("-" * 40)
    analyze_seasonality(cleaned_data)
    
    # Step 9: Stationarity Testing
    print("\n9. STATIONARITY TESTING")
    print("-" * 40)
    
    for symbol, df in cleaned_data.items():
        print(f"\n{symbol} Stationarity Test:")
        print("-" * 30)
        
        # Test on closing prices
        prices = df['Close']
        returns = calculate_returns(df)
        
        # Augmented Dickey-Fuller test on prices
        adf_result_prices = adfuller(prices.dropna())
        print(f"ADF Test on Prices:")
        print(f"  ADF Statistic: {adf_result_prices[0]:.6f}")
        print(f"  p-value: {adf_result_prices[1]:.6f}")
        print(f"  Critical values: {adf_result_prices[4]}")
        
        if adf_result_prices[1] <= 0.05:
            print("  Result: Stationary (reject null hypothesis)")
        else:
            print("  Result: Non-stationary (fail to reject null hypothesis)")
        
        # Augmented Dickey-Fuller test on returns
        adf_result_returns = adfuller(returns.dropna())
        print(f"\nADF Test on Returns:")
        print(f"  ADF Statistic: {adf_result_returns[0]:.6f}")
        print(f"  p-value: {adf_result_returns[1]:.6f}")
        print(f"  Critical values: {adf_result_returns[4]}")
        
        if adf_result_returns[1] <= 0.05:
            print("  Result: Stationary (reject null hypothesis)")
        else:
            print("  Result: Non-stationary (fail to reject null hypothesis)")
    
    # Step 10: Enhanced Outlier Detection
    print("\n10. ENHANCED OUTLIER DETECTION")
    print("-" * 40)
    enhanced_outlier_detection(cleaned_data)
    
    # Step 11: Save Processed Data
    print("\n11. SAVING PROCESSED DATA")
    print("-" * 40)
    
    # Save cleaned data
    for symbol, df in cleaned_data.items():
        filename = f"{symbol}_cleaned_data.csv"
        df.to_csv(filename)
        print(f"Saved {symbol} data to {filename}")
    
    # Save returns data
    returns_df = create_returns_dataframe(cleaned_data)
    returns_df.to_csv("all_returns.csv")
    print("Saved all returns data to all_returns.csv")
    
    # Step 12: Summary and Insights
    print("\n12. SUMMARY AND INSIGHTS")
    print("-" * 40)
    print("\nKey Findings:")
    print("• TSLA: High volatility, strong growth potential, suitable for aggressive portfolios")
    print("• BND: Low volatility, stable returns, provides portfolio stability")
    print("• SPY: Moderate volatility, market correlation, core portfolio component")
    print("• Diversification benefits: Low correlation between TSLA and BND/SPY")
    print("• Data quality: Clean, complete dataset ready for modeling")
    
    print("\n" + "=" * 80)
    print("TASK 1 COMPLETED SUCCESSFULLY!")
    print("✓ Data preprocessing completed")
    print("✓ Comprehensive EDA performed")
    print("✓ Statistical analysis conducted")
    print("✓ Visualizations generated")
    print("✓ Data ready for Task 2 (Forecasting Models)")
    print("=" * 80)
    
    return cleaned_data, returns_df

if __name__ == "__main__":
    main()
