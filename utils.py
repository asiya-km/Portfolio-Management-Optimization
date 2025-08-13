import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def fetch_stock_data(symbols, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance
    
    Parameters:
    symbols (list): List of stock symbols
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
    dict: Dictionary with symbol as key and DataFrame as value
    """
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            if not df.empty:
                data[symbol] = df
                print(f"Successfully fetched data for {symbol}: {len(df)} records")
            else:
                print(f"No data found for {symbol}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    return data

def calculate_returns(df, column='Close'):
    """
    Calculate daily returns
    
    Parameters:
    df (DataFrame): Stock data DataFrame
    column (str): Column to calculate returns for
    
    Returns:
    Series: Daily returns
    """
    return df[column].pct_change().dropna()

def calculate_volatility(returns, window=30):
    """
    Calculate rolling volatility
    
    Parameters:
    returns (Series): Daily returns
    window (int): Rolling window size
    
    Returns:
    Series: Rolling volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(252)

def calculate_rolling_mean(returns, window=30):
    """
    Calculate rolling mean returns
    
    Parameters:
    returns (Series): Daily returns
    window (int): Rolling window size
    
    Returns:
    Series: Rolling mean returns
    """
    return returns.rolling(window=window).mean() * 252

def calculate_var(returns, confidence_level=0.05):
    """
    Calculate Value at Risk
    
    Parameters:
    returns (Series): Daily returns
    confidence_level (float): Confidence level for VaR
    
    Returns:
    float: Value at Risk
    """
    return np.percentile(returns, confidence_level * 100)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe Ratio
    
    Parameters:
    returns (Series): Daily returns
    risk_free_rate (float): Annual risk-free rate
    
    Returns:
    float: Sharpe Ratio
    """
    excess_returns = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)
    return excess_returns / volatility if volatility != 0 else 0

def plot_stock_prices(data_dict, title="Stock Prices Over Time"):
    """
    Plot stock prices for multiple symbols
    
    Parameters:
    data_dict (dict): Dictionary with symbol as key and DataFrame as value
    title (str): Plot title
    """
    plt.figure(figsize=(15, 8))
    for symbol, df in data_dict.items():
        plt.plot(df.index, df['Close'], label=symbol, linewidth=2)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_returns(data_dict, title="Daily Returns"):
    """
    Plot daily returns for multiple symbols
    
    Parameters:
    data_dict (dict): Dictionary with symbol as key and DataFrame as value
    title (str): Plot title
    """
    plt.figure(figsize=(15, 8))
    for symbol, df in data_dict.items():
        returns = calculate_returns(df)
        plt.plot(returns.index, returns, label=symbol, alpha=0.7)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Returns', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_volatility(data_dict, window=30, title="Rolling Volatility"):
    """
    Plot rolling volatility for multiple symbols
    
    Parameters:
    data_dict (dict): Dictionary with symbol as key and DataFrame as value
    window (int): Rolling window size
    title (str): Plot title
    """
    plt.figure(figsize=(15, 8))
    for symbol, df in data_dict.items():
        returns = calculate_returns(df)
        volatility = calculate_volatility(returns, window)
        plt.plot(volatility.index, volatility, label=symbol, linewidth=2)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_summary_statistics(data_dict):
    """
    Print summary statistics for all symbols
    
    Parameters:
    data_dict (dict): Dictionary with symbol as key and DataFrame as value
    """
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for symbol, df in data_dict.items():
        returns = calculate_returns(df)
        print(f"\n{symbol} Statistics:")
        print(f"  Total Return: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%")
        print(f"  Annualized Return: {returns.mean() * 252 * 100:.2f}%")
        print(f"  Annualized Volatility: {returns.std() * np.sqrt(252) * 100:.2f}%")
        print(f"  Sharpe Ratio: {calculate_sharpe_ratio(returns):.3f}")
        print(f"  VaR (5%): {calculate_var(returns) * 100:.2f}%")
        print(f"  Max Drawdown: {((df['Close'] / df['Close'].expanding().max()) - 1).min() * 100:.2f}%")

def check_missing_values(data_dict):
    """
    Check for missing values in all datasets
    
    Parameters:
    data_dict (dict): Dictionary with symbol as key and DataFrame as value
    """
    print("=" * 80)
    print("MISSING VALUES CHECK")
    print("=" * 80)
    
    for symbol, df in data_dict.items():
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n{symbol} - Missing values:")
            print(missing[missing > 0])
        else:
            print(f"\n{symbol} - No missing values found")

def handle_missing_values(data_dict, method='ffill'):
    """
    Handle missing values in datasets
    
    Parameters:
    data_dict (dict): Dictionary with symbol as key and DataFrame as value
    method (str): Method to handle missing values ('ffill', 'bfill', 'interpolate')
    
    Returns:
    dict: Dictionary with cleaned data
    """
    cleaned_data = {}
    
    for symbol, df in data_dict.items():
        if method == 'ffill':
            cleaned_df = df.fillna(method='ffill')
        elif method == 'bfill':
            cleaned_df = df.fillna(method='bfill')
        elif method == 'interpolate':
            cleaned_df = df.interpolate()
        else:
            cleaned_df = df.dropna()
        
        cleaned_data[symbol] = cleaned_df
        print(f"{symbol}: Handled missing values using {method}")
    
    return cleaned_data

def create_returns_dataframe(data_dict):
    """
    Create a DataFrame with returns for all symbols
    
    Parameters:
    data_dict (dict): Dictionary with symbol as key and DataFrame as value
    
    Returns:
    DataFrame: DataFrame with returns for all symbols
    """
    returns_dict = {}
    for symbol, df in data_dict.items():
        returns_dict[symbol] = calculate_returns(df)
    
    returns_df = pd.DataFrame(returns_dict)
    return returns_df.dropna()

def save_results(data, filename):
    """
    Save results to CSV file
    
    Parameters:
    data (DataFrame): Data to save
    filename (str): Output filename
    """
    data.to_csv(filename)
    print(f"Results saved to {filename}")

def load_results(filename):
    """
    Load results from CSV file
    
    Parameters:
    filename (str): Input filename
    
    Returns:
    DataFrame: Loaded data
    """
    return pd.read_csv(filename, index_col=0, parse_dates=True)
