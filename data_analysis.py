"""
Task 1: Preprocess and Explore the Data
=======================================

This script performs comprehensive data preprocessing and exploration for the time series
forecasting project. It includes data fetching, cleaning, EDA, stationarity testing,
and volatility analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from utils import (
    fetch_stock_data, calculate_returns, calculate_volatility, 
    calculate_sharpe_ratio, calculate_var, plot_stock_prices,
    plot_returns, plot_volatility, print_summary_statistics,
    check_missing_values, handle_missing_values, create_returns_dataframe
)

def main():
    print("=" * 80)
    print("TASK 1: DATA PREPROCESSING AND EXPLORATION")
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
    
    # Step 5: Exploratory Data Analysis
    print("\n5. EXPLORATORY DATA ANALYSIS")
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
    
    # Step 6: Detailed Analysis for Each Asset
    print("\n6. DETAILED ASSET ANALYSIS")
    print("-" * 40)
    
    for symbol, df in cleaned_data.items():
        print(f"\n{symbol} Detailed Analysis:")
        print("-" * 30)
        
        # Calculate returns
        returns = calculate_returns(df)
        
        # Basic statistics
        print(f"Data points: {len(df)}")
        print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Mean daily return: {returns.mean():.6f}")
        print(f"Std daily return: {returns.std():.6f}")
        print(f"Skewness: {returns.skew():.3f}")
        print(f"Kurtosis: {returns.kurtosis():.3f}")
        
        # Risk metrics
        print(f"VaR (5%): {calculate_var(returns):.6f}")
        print(f"Sharpe Ratio: {calculate_sharpe_ratio(returns):.3f}")
        
        # Plot distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(df.index, df['Close'])
        plt.title(f'{symbol} - Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{symbol} - Returns Distribution')
        plt.xlabel('Daily Returns')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        volatility = calculate_volatility(returns, window=30)
        plt.plot(volatility.index, volatility)
        plt.title(f'{symbol} - 30-Day Rolling Volatility')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        stats.probplot(returns, dist="norm", plot=plt)
        plt.title(f'{symbol} - Q-Q Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Step 7: Stationarity Testing
    print("\n7. STATIONARITY TESTING")
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
    
    # Step 8: Correlation Analysis
    print("\n8. CORRELATION ANALYSIS")
    print("-" * 40)
    
    returns_df = create_returns_dataframe(cleaned_data)
    
    # Correlation matrix
    correlation_matrix = returns_df.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Daily Returns')
    plt.tight_layout()
    plt.show()
    
    # Step 9: Outlier Detection
    print("\n9. OUTLIER DETECTION")
    print("-" * 40)
    
    for symbol in symbols:
        if symbol in cleaned_data:
            returns = calculate_returns(cleaned_data[symbol])
            
            # Z-score method
            z_scores = np.abs(stats.zscore(returns))
            outliers_z = returns[z_scores > 3]
            
            # IQR method
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            outliers_iqr = returns[(returns < (Q1 - 1.5 * IQR)) | (returns > (Q3 + 1.5 * IQR))]
            
            print(f"\n{symbol} Outlier Analysis:")
            print(f"  Total observations: {len(returns)}")
            print(f"  Outliers (Z-score > 3): {len(outliers_z)} ({len(outliers_z)/len(returns)*100:.2f}%)")
            print(f"  Outliers (IQR method): {len(outliers_iqr)} ({len(outliers_iqr)/len(returns)*100:.2f}%)")
            
            if len(outliers_z) > 0:
                print(f"  Largest outlier (Z-score): {outliers_z.max():.6f}")
                print(f"  Smallest outlier (Z-score): {outliers_z.min():.6f}")
    
    # Step 10: Save Processed Data
    print("\n10. SAVING PROCESSED DATA")
    print("-" * 40)
    
    # Save cleaned data
    for symbol, df in cleaned_data.items():
        filename = f"{symbol}_cleaned_data.csv"
        df.to_csv(filename)
        print(f"Saved {symbol} data to {filename}")
    
    # Save returns data
    returns_df.to_csv("all_returns.csv")
    print("Saved all returns data to all_returns.csv")
    
    print("\n" + "=" * 80)
    print("TASK 1 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return cleaned_data, returns_df

if __name__ == "__main__":
    main()
