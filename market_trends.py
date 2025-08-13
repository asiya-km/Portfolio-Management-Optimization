"""
Task 3: Forecast Future Market Trends
====================================

Uses the best model from Task 2 to forecast TSLA prices for 6-12 months.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_best_model():
    """Load the best model from Task 2"""
    try:
        comparison = pd.read_csv("model_comparison.csv", index_col=0)
        best_model = comparison.columns[comparison.loc['RMSE'].idxmin()]
        print(f"Best model: {best_model}")
        return best_model
    except:
        print("Using ARIMA as default")
        return 'ARIMA'

def generate_forecast_dates(last_date, periods=252):
    """Generate future trading dates"""
    future_dates = []
    current_date = last_date
    
    while len(future_dates) < periods:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # Monday to Friday
            future_dates.append(current_date)
    
    return pd.DatetimeIndex(future_dates[:periods])

def arima_forecast(data, periods=252):
    """Generate ARIMA forecast with confidence intervals"""
    try:
        auto_model = auto_arima(data, seasonal=True, m=12, 
                               suppress_warnings=True, error_action='ignore')
        
        forecast = auto_model.predict(n_periods=periods)
        conf_int = auto_model.predict(n_periods=periods, return_conf_int=True)[1]
        
        return forecast, conf_int
    except Exception as e:
        print(f"ARIMA error: {e}")
        return None, None

def lstm_forecast(data, periods=252):
    """Generate LSTM forecast"""
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
        look_back = 60
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X).reshape(-1, look_back, 1), np.array(y)
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(0.001), loss='mse')
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # Generate forecasts
        last_sequence = scaled_data[-look_back:].reshape(1, look_back, 1)
        forecasts = []
        
        for _ in range(periods):
            next_pred = model.predict(last_sequence)
            forecasts.append(next_pred[0, 0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[0, -1, 0] = next_pred[0, 0]
        
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = scaler.inverse_transform(forecasts).flatten()
        
        # Simple confidence intervals
        std_dev = np.std(data) * 0.1
        conf_int = np.column_stack([
            forecasts - 1.96 * std_dev,
            forecasts + 1.96 * std_dev
        ])
        
        return forecasts, conf_int
    except Exception as e:
        print(f"LSTM error: {e}")
        return None, None

def analyze_trends(forecast_values, conf_int, forecast_dates):
    """Analyze forecast trends"""
    print("\nTREND ANALYSIS:")
    print("-" * 30)
    
    start_price = forecast_values[0]
    end_price = forecast_values[-1]
    total_change = ((end_price - start_price) / start_price) * 100
    
    print(f"Forecast period: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
    print(f"Starting price: ${start_price:.2f}")
    print(f"Ending price: ${end_price:.2f}")
    print(f"Total change: {total_change:.2f}%")
    
    if total_change > 5:
        trend = "Bullish"
    elif total_change > 0:
        trend = "Slightly Bullish"
    elif total_change > -5:
        trend = "Slightly Bearish"
    else:
        trend = "Bearish"
    
    print(f"Trend: {trend}")
    
    # Volatility
    returns = np.diff(forecast_values) / forecast_values[:-1]
    volatility = np.std(returns) * np.sqrt(252) * 100
    print(f"Forecasted volatility: {volatility:.2f}%")
    
    return {'trend': trend, 'change': total_change, 'volatility': volatility}

def plot_forecast(historical_data, forecast_values, conf_int, forecast_dates, model_name):
    """Plot forecast with confidence intervals"""
    plt.figure(figsize=(15, 8))
    
    plt.plot(historical_data.index, historical_data, 
             label='Historical Data', color='blue', linewidth=2)
    plt.plot(forecast_dates, forecast_values, 
             label=f'{model_name} Forecast', color='red', linewidth=2, linestyle='--')
    
    if conf_int is not None:
        plt.fill_between(forecast_dates, conf_int[:, 0], conf_int[:, 1], 
                         alpha=0.3, color='red', label='95% Confidence Interval')
    
    plt.title(f'TSLA Stock Price Forecast - {model_name} Model (12 Months)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    print("=" * 80)
    print("TASK 3: FORECAST FUTURE MARKET TRENDS")
    print("=" * 80)
    
    # Load data
    try:
        tsla_data = pd.read_csv("TSLA_cleaned_data.csv", index_col=0, parse_dates=True)
        print("Loaded TSLA data successfully")
    except FileNotFoundError:
        print("TSLA data not found. Please run Task 1 first.")
        return
    
    # Get best model
    best_model = load_best_model()
    
    # Prepare data
    tsla_prices = tsla_data['Close']
    forecast_periods = 252  # 12 months
    forecast_dates = generate_forecast_dates(tsla_prices.index[-1], forecast_periods)
    
    print(f"Forecasting {forecast_periods} periods")
    print(f"Forecast period: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
    
    # Generate forecast
    if best_model == 'ARIMA':
        forecast_values, conf_int = arima_forecast(tsla_prices, forecast_periods)
    else:
        forecast_values, conf_int = lstm_forecast(tsla_prices, forecast_periods)
    
    if forecast_values is None:
        print("Failed to generate forecast")
        return
    
    # Plot results
    plot_forecast(tsla_prices, forecast_values, conf_int, forecast_dates, best_model)
    
    # Analyze trends
    trend_analysis = analyze_trends(forecast_values, conf_int, forecast_dates)
    
    # Save results
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast_Price': forecast_values,
        'Lower_CI': conf_int[:, 0] if conf_int is not None else forecast_values,
        'Upper_CI': conf_int[:, 1] if conf_int is not None else forecast_values
    })
    
    forecast_df.to_csv(f"{best_model}_forecast.csv", index=False)
    print(f"Forecast saved to {best_model}_forecast.csv")
    
    print("\nTASK 3 COMPLETED!")
    return forecast_df, trend_analysis

if __name__ == "__main__":
    main()
