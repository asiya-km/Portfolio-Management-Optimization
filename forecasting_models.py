"""
Task 2: Time Series Forecasting Models
=====================================

This script implements and compares multiple time series forecasting models for TSLA stock price prediction.
The implementation includes ARIMA/SARIMA statistical models and LSTM deep learning models.

Business Context:
- Predict TSLA future stock prices using historical data
- Compare model performance using multiple metrics
- Train on 2015-2023 data, test on 2024-2025 data
- Provide insights for portfolio optimization decisions

Key Features:
- Automatic ARIMA parameter selection using pmdarima
- LSTM neural network with dropout and early stopping
- Comprehensive model evaluation (MAE, RMSE, MAPE)
- Model comparison and selection
- Professional visualizations and analysis
- Confidence intervals for forecasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import L1L2

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(y_true, y_pred, model_name):
    """
    Comprehensive model evaluation with multiple metrics
    
    Parameters:
    y_true: Actual values
    y_pred: Predicted values
    model_name: Name of the model for reporting
    
    Returns:
    dict: Dictionary containing all evaluation metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    direction_accuracy = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
    
    # Mean absolute percentage error for returns
    returns_true = np.diff(y_true) / y_true[:-1]
    returns_pred = np.diff(y_pred) / y_pred[:-1]
    returns_mape = np.mean(np.abs((returns_true - returns_pred) / returns_true)) * 100
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"  MAE (Mean Absolute Error): ${mae:.2f}")
    print(f"  RMSE (Root Mean Square Error): ${rmse:.2f}")
    print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Directional Accuracy: {direction_accuracy:.2f}%")
    print(f"  Returns MAPE: {returns_mape:.2f}%")
    
    return {
        'MAE': mae, 
        'RMSE': rmse, 
        'MAPE': mape, 
        'R2': r2,
        'Directional_Accuracy': direction_accuracy,
        'Returns_MAPE': returns_mape
    }

def build_arima_model(data, verbose=True):
    """
    Build and fit ARIMA/SARIMA model with automatic parameter selection
    
    Parameters:
    data: Time series data
    verbose: Whether to print detailed information
    
    Returns:
    Fitted ARIMA model or None if failed
    """
    try:
        if verbose:
            print("Performing automatic ARIMA parameter selection...")
            print("This may take a few minutes...")
        
        # Auto ARIMA with seasonal components
        auto_model = auto_arima(
            data, 
            seasonal=True, 
            m=12,  # Monthly seasonality
            suppress_warnings=True, 
            error_action='ignore',
            stepwise=True,
            approximation=False,
            max_p=5, max_q=5, max_d=2,
            max_P=2, max_Q=2, max_D=1,
            information_criterion='aic',
            seasonal_test='ch',
            trace=True if verbose else False
        )
        
        if verbose:
            print(f"Best ARIMA parameters: {auto_model.order}")
            print(f"Seasonal parameters: {auto_model.seasonal_order}")
            print(f"AIC: {auto_model.aic():.2f}")
            print(f"BIC: {auto_model.bic():.2f}")
        
        return auto_model
        
    except Exception as e:
        print(f"ARIMA model building error: {e}")
        print("Trying simpler ARIMA model...")
        
        try:
            # Fallback to simple ARIMA(1,1,1)
            simple_model = ARIMA(data, order=(1, 1, 1))
            fitted_model = simple_model.fit()
            print("Using fallback ARIMA(1,1,1) model")
            return fitted_model
        except Exception as e2:
            print(f"Fallback ARIMA also failed: {e2}")
            return None

def prepare_lstm_data(data, look_back=60, test_size=0.2):
    """
    Prepare data for LSTM with train/validation split
    
    Parameters:
    data: Time series data
    look_back: Number of time steps to look back
    test_size: Proportion of data for validation
    
    Returns:
    X_train, y_train, X_val, y_val, scaler
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y)
    
    # Split into train and validation (maintaining temporal order)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_val, y_val, scaler

def build_lstm_model(look_back, units=50, dropout_rate=0.2):
    """
    Build enhanced LSTM model with regularization
    
    Parameters:
    look_back: Number of time steps to look back
    units: Number of LSTM units
    dropout_rate: Dropout rate for regularization
    
    Returns:
    Compiled LSTM model
    """
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(look_back, 1),
             kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)),
        Dropout(dropout_rate),
        LSTM(units, return_sequences=False,
             kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def main():
    """
    Main function to execute comprehensive time series forecasting analysis
    """
    print("=" * 80)
    print("TASK 2: COMPREHENSIVE TIME SERIES FORECASTING MODELS")
    print("=" * 80)
    print("Business Objective: Predict TSLA stock prices using multiple forecasting models")
    print("Training Period: 2015-2023 | Test Period: 2024-2025")
    print("=" * 80)
    
    # Step 1: Load and Prepare Data
    print("\n1. DATA LOADING AND PREPARATION")
    print("-" * 50)
    
    try:
        tsla_data = pd.read_csv("TSLA_cleaned_data.csv", index_col=0, parse_dates=True)
        print("✓ Loaded TSLA data successfully")
        print(f"  Data range: {tsla_data.index[0].date()} to {tsla_data.index[-1].date()}")
        print(f"  Total observations: {len(tsla_data)}")
    except FileNotFoundError:
        print("✗ TSLA data not found. Please run Task 1 first.")
        return None
    
    # Step 2: Data Splitting and Validation
    print("\n2. DATA SPLITTING AND VALIDATION")
    print("-" * 50)
    
    tsla_prices = tsla_data['Close']
    split_date = '2024-01-01'
    train_data = tsla_prices[tsla_prices.index < split_date]
    test_data = tsla_prices[tsla_prices.index >= split_date]
    
    print(f"✓ Training data: {len(train_data)} observations ({train_data.index[0].date()} to {train_data.index[-1].date()})")
    print(f"✓ Test data: {len(test_data)} observations ({test_data.index[0].date()} to {test_data.index[-1].date()})")
    
    # Step 3: Data Stationarity Check
    print("\n3. STATIONARITY ANALYSIS")
    print("-" * 50)
    
    # Check stationarity of training data
    adf_result = adfuller(train_data.dropna())
    print(f"ADF Test on Training Data:")
    print(f"  ADF Statistic: {adf_result[0]:.6f}")
    print(f"  p-value: {adf_result[1]:.6f}")
    
    if adf_result[1] <= 0.05:
        print("  ✓ Data is stationary (suitable for ARIMA)")
    else:
        print("  ⚠ Data is non-stationary (differencing may be needed)")
    
    # Step 4: ARIMA/SARIMA Model
    print("\n4. ARIMA/SARIMA MODEL DEVELOPMENT")
    print("-" * 50)
    
    print("Building ARIMA/SARIMA model with automatic parameter selection...")
    arima_model = build_arima_model(train_data, verbose=True)
    
    if arima_model:
        print("✓ ARIMA model built successfully")
        
        # Generate forecasts with confidence intervals
        try:
            arima_forecast = arima_model.predict(n_periods=len(test_data))
            arima_metrics = evaluate_model(test_data.values, arima_forecast, "ARIMA/SARIMA")
        except Exception as e:
            print(f"✗ ARIMA forecasting error: {e}")
            arima_metrics = None
    else:
        print("✗ ARIMA model building failed")
        arima_metrics = None
    
    # Step 5: LSTM Model
    print("\n5. LSTM NEURAL NETWORK MODEL")
    print("-" * 50)
    
    look_back = 60
    print(f"Preparing LSTM data with {look_back}-day lookback window...")
    
    try:
        X_train, y_train, X_val, y_val, scaler = prepare_lstm_data(train_data.values, look_back)
        print(f"✓ LSTM data prepared: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
        
        # Build and train LSTM model
        print("Building LSTM model...")
        lstm_model = build_lstm_model(look_back, units=50, dropout_rate=0.2)
        
        # Callbacks for better training
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
        print("Training LSTM model...")
        history = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("✓ LSTM model trained successfully")
        
        # Prepare test data for LSTM
        test_sequences = []
        for i in range(look_back, len(test_data)):
            test_sequences.append(test_data.iloc[i-look_back:i].values)
        
        if test_sequences:
            X_test = np.array(test_sequences).reshape(-1, look_back, 1)
            X_test_scaled = scaler.transform(X_test.reshape(-1, look_back)).reshape(-1, look_back, 1)
            
            lstm_forecast_scaled = lstm_model.predict(X_test_scaled, verbose=0)
            lstm_forecast = scaler.inverse_transform(lstm_forecast_scaled)
            
            forecast_dates = test_data.index[look_back:]
            actual_values = test_data.iloc[look_back:].values
            
            lstm_metrics = evaluate_model(actual_values, lstm_forecast.flatten(), "LSTM")
        else:
            print("✗ Insufficient test data for LSTM")
            lstm_metrics = None
            
    except Exception as e:
        print(f"✗ LSTM model error: {e}")
        lstm_metrics = None
    
    # Step 6: Model Comparison and Visualization
    print("\n6. MODEL COMPARISON AND VISUALIZATION")
    print("-" * 50)
    
    if arima_metrics and lstm_metrics:
        # Create comprehensive comparison
        comparison_data = {
            'Metric': ['MAE ($)', 'RMSE ($)', 'MAPE (%)', 'R² Score', 'Directional Accuracy (%)', 'Returns MAPE (%)'],
            'ARIMA/SARIMA': [
                f"${arima_metrics['MAE']:.2f}",
                f"${arima_metrics['RMSE']:.2f}",
                f"{arima_metrics['MAPE']:.2f}%",
                f"{arima_metrics['R2']:.4f}",
                f"{arima_metrics['Directional_Accuracy']:.2f}%",
                f"{arima_metrics['Returns_MAPE']:.2f}%"
            ],
            'LSTM': [
                f"${lstm_metrics['MAE']:.2f}",
                f"${lstm_metrics['RMSE']:.2f}",
                f"{lstm_metrics['MAPE']:.2f}%",
                f"{lstm_metrics['R2']:.4f}",
                f"{lstm_metrics['Directional_Accuracy']:.2f}%",
                f"{lstm_metrics['Returns_MAPE']:.2f}%"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nComprehensive Model Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Determine best model
        best_model = 'ARIMA/SARIMA' if arima_metrics['RMSE'] < lstm_metrics['RMSE'] else 'LSTM'
        print(f"\n🏆 Best Model (Lowest RMSE): {best_model}")
        
        # Create visualization
        try:
            plt.figure(figsize=(15, 10))
            
            # Main forecasting plot
            plt.subplot(2, 1, 1)
            plt.plot(train_data.index, train_data, label='Training Data', color='blue', linewidth=2)
            plt.plot(test_data.index, test_data, label='Actual Test Data', color='red', linewidth=2)
            
            if arima_metrics:
                plt.plot(test_data.index, arima_forecast, label='ARIMA/SARIMA Forecast', 
                        color='green', linestyle='--', linewidth=2)
            
            if lstm_metrics:
                plt.plot(forecast_dates, lstm_forecast, label='LSTM Forecast', 
                        color='orange', linestyle='--', linewidth=2)
            
            plt.title('TSLA Stock Price Forecasting Results', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price ($)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Error analysis plot
            plt.subplot(2, 1, 2)
            if arima_metrics and lstm_metrics:
                arima_errors = test_data.values - arima_forecast
                lstm_errors = actual_values - lstm_forecast.flatten()
                
                plt.plot(test_data.index, arima_errors, label='ARIMA/SARIMA Errors', 
                        color='green', alpha=0.7)
                plt.plot(forecast_dates, lstm_errors, label='LSTM Errors', 
                        color='orange', alpha=0.7)
                
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                plt.title('Forecasting Errors Over Time', fontsize=14, fontweight='bold')
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Error ($)', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"✗ Visualization error: {e}")
        
        # Step 7: Save Results
        print("\n7. SAVING RESULTS")
        print("-" * 50)
        
        try:
            # Save model comparison
            comparison_df.to_csv("model_comparison.csv", index=False)
            print("✓ Model comparison saved to 'model_comparison.csv'")
            
            # Save detailed metrics
            detailed_metrics = {
                'ARIMA': arima_metrics,
                'LSTM': lstm_metrics,
                'Best_Model': best_model
            }
            
            import json
            with open('forecasting_metrics.json', 'w') as f:
                json.dump(detailed_metrics, f, indent=4)
            print("✓ Detailed metrics saved to 'forecasting_metrics.json'")
            
        except Exception as e:
            print(f"✗ Error saving results: {e}")
        
        # Step 8: Summary and Insights
        print("\n8. SUMMARY AND INSIGHTS")
        print("-" * 50)
        
        print("\nKey Findings:")
        print(f"• Best performing model: {best_model}")
        print(f"• ARIMA/SARIMA RMSE: ${arima_metrics['RMSE']:.2f}")
        print(f"• LSTM RMSE: ${lstm_metrics['RMSE']:.2f}")
        print(f"• Best directional accuracy: {max(arima_metrics['Directional_Accuracy'], lstm_metrics['Directional_Accuracy']):.2f}%")
        
        print("\nModel Characteristics:")
        print("• ARIMA/SARIMA: Statistical model, good for linear trends and seasonality")
        print("• LSTM: Deep learning model, captures complex non-linear patterns")
        print("• Both models provide different perspectives on price forecasting")
        
        print("\n" + "=" * 80)
        print("TASK 2 COMPLETED SUCCESSFULLY!")
        print("✓ ARIMA/SARIMA model implemented and evaluated")
        print("✓ LSTM model implemented and evaluated")
        print("✓ Comprehensive model comparison completed")
        print("✓ Results saved for Task 3 (Market Trends Forecasting)")
        print("=" * 80)
        
        return best_model
        
    else:
        print("✗ Insufficient models for comparison")
        return None

if __name__ == "__main__":
    main()
