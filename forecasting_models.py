"""
Task 2: Time Series Forecasting Models
=====================================

Implements ARIMA/SARIMA and LSTM models for TSLA price forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def build_arima_model(data):
    """Build and fit ARIMA model"""
    try:
        auto_model = auto_arima(data, seasonal=True, m=12, 
                               suppress_warnings=True, error_action='ignore')
        print(f"Best ARIMA parameters: {auto_model.order}")
        return auto_model
    except Exception as e:
        print(f"ARIMA error: {e}")
        return None

def prepare_lstm_data(data, look_back=60):
    """Prepare data for LSTM"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X).reshape(-1, look_back, 1), np.array(y), scaler

def build_lstm_model(look_back):
    """Build LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model

def main():
    print("=" * 80)
    print("TASK 2: TIME SERIES FORECASTING MODELS")
    print("=" * 80)
    
    # Load TSLA data
    try:
        tsla_data = pd.read_csv("TSLA_cleaned_data.csv", index_col=0, parse_dates=True)
        print("Loaded TSLA data successfully")
    except FileNotFoundError:
        print("TSLA data not found. Please run Task 1 first.")
        return
    
    # Split data (train: 2015-2023, test: 2024-2025)
    tsla_prices = tsla_data['Close']
    split_date = '2024-01-01'
    train_data = tsla_prices[tsla_prices.index < split_date]
    test_data = tsla_prices[tsla_prices.index >= split_date]
    
    print(f"Training: {len(train_data)} obs, Test: {len(test_data)} obs")
    
    # Model 1: ARIMA
    print("\n1. ARIMA/SARIMA MODEL")
    print("-" * 30)
    arima_model = build_arima_model(train_data)
    
    if arima_model:
        arima_forecast = arima_model.predict(n_periods=len(test_data))
        arima_metrics = evaluate_model(test_data.values, arima_forecast, "ARIMA")
    else:
        arima_metrics = None
    
    # Model 2: LSTM
    print("\n2. LSTM MODEL")
    print("-" * 30)
    
    look_back = 60
    X_train, y_train, scaler = prepare_lstm_data(train_data.values, look_back)
    
    lstm_model = build_lstm_model(look_back)
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Prepare test data for LSTM
    test_sequences = []
    for i in range(look_back, len(test_data)):
        test_sequences.append(test_data.iloc[i-look_back:i].values)
    
    if test_sequences:
        X_test = np.array(test_sequences).reshape(-1, look_back, 1)
        X_test_scaled = scaler.transform(X_test.reshape(-1, look_back)).reshape(-1, look_back, 1)
        
        lstm_forecast_scaled = lstm_model.predict(X_test_scaled)
        lstm_forecast = scaler.inverse_transform(lstm_forecast_scaled)
        
        forecast_dates = test_data.index[look_back:]
        actual_values = test_data.iloc[look_back:].values
        
        lstm_metrics = evaluate_model(actual_values, lstm_forecast.flatten(), "LSTM")
        
        # Plot results
        plt.figure(figsize=(15, 6))
        plt.plot(train_data.index, train_data, label='Training Data', color='blue')
        plt.plot(test_data.index, test_data, label='Actual Test Data', color='red')
        
        if arima_metrics:
            plt.plot(test_data.index, arima_forecast, label='ARIMA Forecast', color='green', linestyle='--')
        
        plt.plot(forecast_dates, lstm_forecast, label='LSTM Forecast', color='orange', linestyle='--')
        plt.title('TSLA Price Forecasting Results')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Model comparison
        if arima_metrics and lstm_metrics:
            print("\nMODEL COMPARISON:")
            print("-" * 30)
            comparison = pd.DataFrame({
                'ARIMA': arima_metrics,
                'LSTM': lstm_metrics
            })
            print(comparison)
            
            best_model = 'ARIMA' if arima_metrics['RMSE'] < lstm_metrics['RMSE'] else 'LSTM'
            print(f"\nBest model (RMSE): {best_model}")
            
            # Save results
            comparison.to_csv("model_comparison.csv")
            return best_model
    else:
        print("Insufficient test data for LSTM")
        return None

if __name__ == "__main__":
    main()
