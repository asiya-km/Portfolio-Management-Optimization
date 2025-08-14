"""
Task 3: Forecast Future Market Trends
====================================

This script uses the best-performing model from Task 2 to forecast TSLA stock prices
for 6-12 months into the future, providing comprehensive trend analysis and market insights.

Business Context:
- Generate future price predictions for TSLA using the optimal model
- Analyze market trends, volatility, and identify opportunities/risks
- Provide confidence intervals to assess forecast uncertainty
- Support portfolio optimization decisions with forward-looking insights

Key Features:
- Multi-period forecasting (6-12 months)
- Confidence intervals for uncertainty quantification
- Trend analysis and market direction assessment
- Volatility forecasting and risk assessment
- Market opportunity and risk identification
- Professional visualizations with multiple timeframes
- Comprehensive market insights and recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L1L2

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_best_model():
    """
    Load the best performing model from Task 2 results
    
    Returns:
    str: Name of the best model ('ARIMA/SARIMA' or 'LSTM')
    """
    try:
        # Try to load from JSON file first (more detailed)
        import json
        with open('forecasting_metrics.json', 'r') as f:
            metrics = json.load(f)
            best_model = metrics.get('Best_Model', 'ARIMA/SARIMA')
            print(f"✓ Loaded best model from Task 2: {best_model}")
            return best_model
    except FileNotFoundError:
        try:
            # Fallback to CSV file
            comparison = pd.read_csv("model_comparison.csv")
            # Find the model with lowest RMSE
            rmse_row = comparison[comparison['Metric'] == 'RMSE ($)']
            if not rmse_row.empty:
                arima_rmse = float(rmse_row['ARIMA/SARIMA'].iloc[0].replace('$', ''))
                lstm_rmse = float(rmse_row['LSTM'].iloc[0].replace('$', ''))
                best_model = 'ARIMA/SARIMA' if arima_rmse < lstm_rmse else 'LSTM'
                print(f"✓ Determined best model from comparison: {best_model}")
                return best_model
        except Exception as e:
            print(f"⚠ Could not load model comparison: {e}")
    
    print("⚠ Using ARIMA/SARIMA as default model")
    return 'ARIMA/SARIMA'

def generate_forecast_dates(last_date, periods=252):
    """
    Generate future trading dates (excluding weekends and holidays)
    
    Parameters:
    last_date: Last date in historical data
    periods: Number of trading days to forecast
    
    Returns:
    pd.DatetimeIndex: Future trading dates
    """
    future_dates = []
    current_date = last_date
    
    while len(future_dates) < periods:
        current_date += timedelta(days=1)
        # Exclude weekends (Saturday=5, Sunday=6)
        if current_date.weekday() < 5:
            future_dates.append(current_date)
    
    return pd.DatetimeIndex(future_dates[:periods])

def calculate_confidence_intervals(forecast_values, historical_volatility, confidence_level=0.95):
    """
    Calculate confidence intervals for forecasts
    
    Parameters:
    forecast_values: Predicted values
    historical_volatility: Historical volatility of returns
    confidence_level: Confidence level (default 0.95 for 95%)
    
    Returns:
    tuple: Lower and upper confidence intervals
    """
    from scipy import stats
    
    # Calculate z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Calculate confidence intervals
    std_dev = historical_volatility * np.sqrt(np.arange(1, len(forecast_values) + 1))
    lower_ci = forecast_values - z_score * std_dev
    upper_ci = forecast_values + z_score * std_dev
    
    return lower_ci, upper_ci

def arima_forecast(data, periods=252):
    """
    Generate ARIMA/SARIMA forecast with confidence intervals
    
    Parameters:
    data: Historical time series data
    periods: Number of periods to forecast
    
    Returns:
    tuple: (forecast_values, confidence_intervals)
    """
    try:
        print("Building ARIMA/SARIMA model for forecasting...")
        
        # Build ARIMA model with automatic parameter selection
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
            trace=False
        )
        
        print(f"✓ ARIMA model built with parameters: {auto_model.order}")
        
        # Generate forecast
        forecast = auto_model.predict(n_periods=periods)
        
        # Generate confidence intervals
        try:
            conf_int = auto_model.predict(n_periods=periods, return_conf_int=True)[1]
            print("✓ Generated forecast with confidence intervals")
        except:
            # Fallback confidence intervals
            print("⚠ Using fallback confidence intervals")
            historical_volatility = np.std(np.diff(data) / data[:-1])
            lower_ci, upper_ci = calculate_confidence_intervals(forecast, historical_volatility)
            conf_int = np.column_stack([lower_ci, upper_ci])
        
        return forecast, conf_int
        
    except Exception as e:
        print(f"✗ ARIMA forecasting error: {e}")
        print("Trying fallback ARIMA model...")
        
        try:
            # Fallback to simple ARIMA(1,1,1)
            simple_model = ARIMA(data, order=(1, 1, 1))
            fitted_model = simple_model.fit()
            forecast = fitted_model.forecast(steps=periods)
            
            # Simple confidence intervals
            historical_volatility = np.std(np.diff(data) / data[:-1])
            lower_ci, upper_ci = calculate_confidence_intervals(forecast, historical_volatility)
            conf_int = np.column_stack([lower_ci, upper_ci])
            
            print("✓ Fallback ARIMA model successful")
            return forecast, conf_int
            
        except Exception as e2:
            print(f"✗ Fallback ARIMA also failed: {e2}")
            return None, None

def lstm_forecast(data, periods=252):
    """
    Generate LSTM forecast with confidence intervals
    
    Parameters:
    data: Historical time series data
    periods: Number of periods to forecast
    
    Returns:
    tuple: (forecast_values, confidence_intervals)
    """
    try:
        print("Building LSTM model for forecasting...")
        
        # Data preparation
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
        look_back = 60
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X).reshape(-1, look_back, 1), np.array(y)
        
        # Build enhanced LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1),
                 kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)),
            Dropout(0.2),
            LSTM(50, return_sequences=False,
                 kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train with early stopping
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        print("Training LSTM model...")
        model.fit(X, y, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
        print("✓ LSTM model trained successfully")
        
        # Generate forecasts using recursive prediction
        last_sequence = scaled_data[-look_back:].reshape(1, look_back, 1)
        forecasts = []
        
        for _ in range(periods):
            next_pred = model.predict(last_sequence, verbose=0)
            forecasts.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Inverse transform forecasts
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = scaler.inverse_transform(forecasts).flatten()
        
        # Calculate confidence intervals
        historical_volatility = np.std(np.diff(data) / data[:-1])
        lower_ci, upper_ci = calculate_confidence_intervals(forecasts, historical_volatility)
        conf_int = np.column_stack([lower_ci, upper_ci])
        
        print("✓ Generated LSTM forecast with confidence intervals")
        return forecasts, conf_int
        
    except Exception as e:
        print(f"✗ LSTM forecasting error: {e}")
        return None, None

def analyze_trends(forecast_values, conf_int, forecast_dates, historical_data):
    """
    Comprehensive trend analysis and market insights
    
    Parameters:
    forecast_values: Predicted price values
    conf_int: Confidence intervals
    forecast_dates: Future dates
    historical_data: Historical price data for comparison
    
    Returns:
    dict: Comprehensive analysis results
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE TREND ANALYSIS & MARKET INSIGHTS")
    print("="*60)
    
    # Basic trend analysis
    start_price = forecast_values[0]
    end_price = forecast_values[-1]
    total_change = ((end_price - start_price) / start_price) * 100
    
    print(f"\n📊 FORECAST PERIOD: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
    print(f"   Starting price: ${start_price:.2f}")
    print(f"   Ending price: ${end_price:.2f}")
    print(f"   Total change: {total_change:+.2f}%")
    
    # Trend classification
    if total_change > 10:
        trend = "Strongly Bullish 🚀"
        sentiment = "Very Positive"
    elif total_change > 5:
        trend = "Bullish 📈"
        sentiment = "Positive"
    elif total_change > 0:
        trend = "Slightly Bullish ↗️"
        sentiment = "Slightly Positive"
    elif total_change > -5:
        trend = "Slightly Bearish ↘️"
        sentiment = "Slightly Negative"
    elif total_change > -10:
        trend = "Bearish 📉"
        sentiment = "Negative"
    else:
        trend = "Strongly Bearish 🔻"
        sentiment = "Very Negative"
    
    print(f"   Market Trend: {trend}")
    print(f"   Market Sentiment: {sentiment}")
    
    # Volatility analysis
    returns = np.diff(forecast_values) / forecast_values[:-1]
    forecast_volatility = np.std(returns) * np.sqrt(252) * 100
    
    # Historical volatility for comparison
    historical_returns = np.diff(historical_data) / historical_data[:-1]
    historical_volatility = np.std(historical_returns) * np.sqrt(252) * 100
    
    print(f"\n📈 VOLATILITY ANALYSIS:")
    print(f"   Forecasted volatility: {forecast_volatility:.2f}%")
    print(f"   Historical volatility: {historical_volatility:.2f}%")
    
    if forecast_volatility > historical_volatility * 1.2:
        volatility_trend = "Increasing ⬆️"
    elif forecast_volatility < historical_volatility * 0.8:
        volatility_trend = "Decreasing ⬇️"
    else:
        volatility_trend = "Stable ➡️"
    
    print(f"   Volatility trend: {volatility_trend}")
    
    # Confidence interval analysis
    if conf_int is not None:
        ci_width = np.mean(conf_int[:, 1] - conf_int[:, 0])
        ci_width_pct = (ci_width / np.mean(forecast_values)) * 100
        
        print(f"\n🎯 UNCERTAINTY ANALYSIS:")
        print(f"   Average confidence interval width: ${ci_width:.2f}")
        print(f"   CI width as % of price: {ci_width_pct:.2f}%")
        
        if ci_width_pct > 20:
            uncertainty_level = "High ⚠️"
        elif ci_width_pct > 10:
            uncertainty_level = "Medium ⚡"
        else:
            uncertainty_level = "Low ✅"
        
        print(f"   Uncertainty level: {uncertainty_level}")
    
    # Market opportunities and risks
    print(f"\n💡 MARKET OPPORTUNITIES & RISKS:")
    
    opportunities = []
    risks = []
    
    if total_change > 5:
        opportunities.append("Strong upward price momentum expected")
        opportunities.append("Potential for significant capital gains")
    elif total_change > 0:
        opportunities.append("Moderate growth potential")
    
    if forecast_volatility > historical_volatility:
        risks.append("Higher volatility expected - increased risk")
        opportunities.append("Higher volatility may present trading opportunities")
    
    if conf_int is not None and ci_width_pct > 15:
        risks.append("High forecast uncertainty - proceed with caution")
    
    if total_change < -5:
        risks.append("Significant downside risk expected")
        opportunities.append("Potential buying opportunities at lower prices")
    
    # Print opportunities
    if opportunities:
        print("   🟢 Opportunities:")
        for opp in opportunities:
            print(f"      • {opp}")
    
    # Print risks
    if risks:
        print("   🔴 Risks:")
        for risk in risks:
            print(f"      • {risk}")
    
    # Key price levels
    print(f"\n🎯 KEY PRICE LEVELS:")
    print(f"   Support level (lower CI): ${np.min(conf_int[:, 0]):.2f}")
    print(f"   Resistance level (upper CI): ${np.max(conf_int[:, 1]):.2f}")
    print(f"   Target price (forecast end): ${end_price:.2f}")
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Maximum forecasted price: ${np.max(forecast_values):.2f}")
    print(f"   Minimum forecasted price: ${np.min(forecast_values):.2f}")
    print(f"   Average forecasted price: ${np.mean(forecast_values):.2f}")
    
    return {
        'trend': trend,
        'sentiment': sentiment,
        'total_change': total_change,
        'forecast_volatility': forecast_volatility,
        'historical_volatility': historical_volatility,
        'volatility_trend': volatility_trend,
        'uncertainty_level': uncertainty_level if 'uncertainty_level' in locals() else 'Unknown',
        'opportunities': opportunities,
        'risks': risks,
        'support_level': np.min(conf_int[:, 0]) if conf_int is not None else None,
        'resistance_level': np.max(conf_int[:, 1]) if conf_int is not None else None,
        'target_price': end_price
    }

def plot_forecast(historical_data, forecast_values, conf_int, forecast_dates, model_name, analysis_results):
    """
    Create comprehensive forecast visualization
    
    Parameters:
    historical_data: Historical price data
    forecast_values: Predicted values
    conf_int: Confidence intervals
    forecast_dates: Future dates
    model_name: Name of the forecasting model
    analysis_results: Results from trend analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'TSLA Stock Price Forecast Analysis - {model_name} Model', 
                 fontsize=16, fontweight='bold')
    
    # Main forecast plot
    axes[0, 0].plot(historical_data.index, historical_data, 
                    label='Historical Data', color='blue', linewidth=2)
    axes[0, 0].plot(forecast_dates, forecast_values, 
                    label=f'{model_name} Forecast', color='red', linewidth=2, linestyle='--')
    
    if conf_int is not None:
        axes[0, 0].fill_between(forecast_dates, conf_int[:, 0], conf_int[:, 1], 
                               alpha=0.3, color='red', label='95% Confidence Interval')
    
    axes[0, 0].set_title('Price Forecast with Confidence Intervals', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Volatility comparison
    historical_returns = np.diff(historical_data) / historical_data[:-1]
    forecast_returns = np.diff(forecast_values) / forecast_values[:-1]
    
    # Calculate rolling volatility
    historical_vol = pd.Series(historical_returns).rolling(window=30).std() * np.sqrt(252) * 100
    forecast_vol = pd.Series(forecast_returns).rolling(window=30).std() * np.sqrt(252) * 100
    
    axes[0, 1].plot(historical_data.index[30:], historical_vol, 
                    label='Historical Volatility', color='blue', linewidth=2)
    axes[0, 1].plot(forecast_dates[30:], forecast_vol, 
                    label='Forecasted Volatility', color='red', linewidth=2, linestyle='--')
    
    axes[0, 1].set_title('Volatility Comparison (30-day Rolling)', fontweight='bold')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Volatility (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confidence interval width over time
    if conf_int is not None:
        ci_width = conf_int[:, 1] - conf_int[:, 0]
        ci_width_pct = (ci_width / forecast_values) * 100
        
        axes[1, 0].plot(forecast_dates, ci_width_pct, 
                        color='purple', linewidth=2)
        axes[1, 0].set_title('Forecast Uncertainty Over Time', fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('CI Width (% of Price)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add uncertainty level annotation
        avg_uncertainty = np.mean(ci_width_pct)
        axes[1, 0].axhline(y=avg_uncertainty, color='red', linestyle='--', 
                          alpha=0.7, label=f'Average: {avg_uncertainty:.1f}%')
        axes[1, 0].legend()
    
    # Price distribution
    axes[1, 1].hist(forecast_values, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].axvline(x=np.mean(forecast_values), color='blue', linestyle='--', 
                      linewidth=2, label=f'Mean: ${np.mean(forecast_values):.2f}')
    axes[1, 1].axvline(x=analysis_results['target_price'], color='green', linestyle='--', 
                      linewidth=2, label=f'Target: ${analysis_results["target_price"]:.2f}')
    
    axes[1, 1].set_title('Forecasted Price Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Price ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional summary plot
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    # Plot with key levels
    ax.plot(historical_data.index, historical_data, 
            label='Historical Data', color='blue', linewidth=2)
    ax.plot(forecast_dates, forecast_values, 
            label=f'{model_name} Forecast', color='red', linewidth=2, linestyle='--')
    
    if conf_int is not None:
        ax.fill_between(forecast_dates, conf_int[:, 0], conf_int[:, 1], 
                       alpha=0.3, color='red', label='95% Confidence Interval')
        
        # Add key levels
        ax.axhline(y=analysis_results['support_level'], color='green', linestyle=':', 
                  alpha=0.7, label=f'Support: ${analysis_results["support_level"]:.2f}')
        ax.axhline(y=analysis_results['resistance_level'], color='orange', linestyle=':', 
                  alpha=0.7, label=f'Resistance: ${analysis_results["resistance_level"]:.2f}')
    
    ax.set_title(f'TSLA Stock Price Forecast - {model_name} Model\n'
                f'Trend: {analysis_results["trend"]} | '
                f'Target: ${analysis_results["target_price"]:.2f}', 
                fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute comprehensive market trend forecasting analysis
    """
    print("=" * 80)
    print("TASK 3: COMPREHENSIVE MARKET TRENDS FORECASTING")
    print("=" * 80)
    print("Business Objective: Forecast TSLA stock prices for 6-12 months")
    print("Focus: Market trends, volatility, opportunities, and risks")
    print("=" * 80)
    
    # Step 1: Data Loading and Validation
    print("\n1. DATA LOADING AND VALIDATION")
    print("-" * 50)
    
    try:
        tsla_data = pd.read_csv("TSLA_cleaned_data.csv", index_col=0, parse_dates=True)
        print("✓ Loaded TSLA data successfully")
        print(f"  Data range: {tsla_data.index[0].date()} to {tsla_data.index[-1].date()}")
        print(f"  Total observations: {len(tsla_data)}")
    except FileNotFoundError:
        print("✗ TSLA data not found. Please run Task 1 first.")
        return None
    
    # Step 2: Model Selection
    print("\n2. MODEL SELECTION")
    print("-" * 50)
    
    best_model = load_best_model()
    print(f"✓ Selected forecasting model: {best_model}")
    
    # Step 3: Forecast Configuration
    print("\n3. FORECAST CONFIGURATION")
    print("-" * 50)
    
    tsla_prices = tsla_data['Close']
    forecast_periods = 252  # 12 months (trading days)
    forecast_dates = generate_forecast_dates(tsla_prices.index[-1], forecast_periods)
    
    print(f"✓ Forecast periods: {forecast_periods} trading days")
    print(f"✓ Forecast period: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
    print(f"✓ Current price: ${tsla_prices.iloc[-1]:.2f}")
    
    # Step 4: Generate Forecast
    print("\n4. FORECAST GENERATION")
    print("-" * 50)
    
    print(f"Generating forecast using {best_model} model...")
    
    if 'ARIMA' in best_model:
        forecast_values, conf_int = arima_forecast(tsla_prices, forecast_periods)
    else:
        forecast_values, conf_int = lstm_forecast(tsla_prices, forecast_periods)
    
    if forecast_values is None:
        print("✗ Failed to generate forecast")
        return None
    
    print("✓ Forecast generated successfully")
    print(f"  Forecast start: ${forecast_values[0]:.2f}")
    print(f"  Forecast end: ${forecast_values[-1]:.2f}")
    
    # Step 5: Comprehensive Trend Analysis
    print("\n5. COMPREHENSIVE TREND ANALYSIS")
    print("-" * 50)
    
    analysis_results = analyze_trends(forecast_values, conf_int, forecast_dates, tsla_prices)
    
    # Step 6: Advanced Visualizations
    print("\n6. ADVANCED VISUALIZATIONS")
    print("-" * 50)
    
    try:
        plot_forecast(tsla_prices, forecast_values, conf_int, forecast_dates, best_model, analysis_results)
        print("✓ Comprehensive visualizations created")
    except Exception as e:
        print(f"⚠ Visualization error: {e}")
    
    # Step 7: Save Results and Generate Reports
    print("\n7. RESULTS SAVING AND REPORTING")
    print("-" * 50)
    
    try:
        # Save forecast data
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast_Price': forecast_values,
            'Lower_CI': conf_int[:, 0] if conf_int is not None else forecast_values,
            'Upper_CI': conf_int[:, 1] if conf_int is not None else forecast_values
        })
        
        forecast_filename = f"{best_model.replace('/', '_')}_forecast.csv"
        forecast_df.to_csv(forecast_filename, index=False)
        print(f"✓ Forecast data saved to '{forecast_filename}'")
        
        # Save analysis results
        import json
        analysis_filename = f"{best_model.replace('/', '_')}_analysis.json"
        with open(analysis_filename, 'w') as f:
            json.dump(analysis_results, f, indent=4, default=str)
        print(f"✓ Analysis results saved to '{analysis_filename}'")
        
        # Generate summary report
        generate_summary_report(forecast_df, analysis_results, best_model)
        
    except Exception as e:
        print(f"⚠ Error saving results: {e}")
    
    # Step 8: Portfolio Implications
    print("\n8. PORTFOLIO IMPLICATIONS")
    print("-" * 50)
    
    print("\n📊 FORECAST SUMMARY FOR PORTFOLIO OPTIMIZATION:")
    print(f"   • Expected return: {analysis_results['total_change']:+.2f}%")
    print(f"   • Market sentiment: {analysis_results['sentiment']}")
    print(f"   • Volatility trend: {analysis_results['volatility_trend']}")
    print(f"   • Uncertainty level: {analysis_results['uncertainty_level']}")
    
    if analysis_results['opportunities']:
        print(f"   • Key opportunities: {len(analysis_results['opportunities'])} identified")
    
    if analysis_results['risks']:
        print(f"   • Key risks: {len(analysis_results['risks'])} identified")
    
    print(f"\n🎯 RECOMMENDATIONS FOR TASK 4 (Portfolio Optimization):")
    print(f"   • Use forecasted TSLA return: {analysis_results['total_change']:+.2f}%")
    print(f"   • Consider volatility: {analysis_results['forecast_volatility']:.2f}%")
    print(f"   • Account for uncertainty: {analysis_results['uncertainty_level']}")
    
    # Step 9: Final Summary
    print("\n" + "=" * 80)
    print("TASK 3 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("✓ Future market trends forecasted (6-12 months)")
    print("✓ Confidence intervals calculated")
    print("✓ Comprehensive trend analysis completed")
    print("✓ Market opportunities and risks identified")
    print("✓ Professional visualizations created")
    print("✓ Results saved for Task 4 (Portfolio Optimization)")
    print("=" * 80)
    
    return forecast_df, analysis_results

def generate_summary_report(forecast_df, analysis_results, model_name):
    """
    Generate a comprehensive summary report
    """
    print("\n📋 GENERATING SUMMARY REPORT...")
    
    report = f"""
MARKET TRENDS FORECASTING SUMMARY REPORT
========================================

Model Used: {model_name}
Forecast Period: {forecast_df['Date'].iloc[0].date()} to {forecast_df['Date'].iloc[-1].date()}

KEY FINDINGS:
-------------
• Market Trend: {analysis_results['trend']}
• Market Sentiment: {analysis_results['sentiment']}
• Expected Return: {analysis_results['total_change']:+.2f}%
• Forecasted Volatility: {analysis_results['forecast_volatility']:.2f}%
• Historical Volatility: {analysis_results['historical_volatility']:.2f}%
• Volatility Trend: {analysis_results['volatility_trend']}
• Uncertainty Level: {analysis_results['uncertainty_level']}

PRICE TARGETS:
--------------
• Target Price: ${analysis_results['target_price']:.2f}
• Support Level: ${analysis_results['support_level']:.2f}
• Resistance Level: ${analysis_results['resistance_level']:.2f}

OPPORTUNITIES ({len(analysis_results['opportunities'])}):
{chr(10).join([f"• {opp}" for opp in analysis_results['opportunities']])}

RISKS ({len(analysis_results['risks'])}):
{chr(10).join([f"• {risk}" for risk in analysis_results['risks']])}

PORTFOLIO IMPLICATIONS:
----------------------
• This forecast provides the expected return for TSLA in portfolio optimization
• Volatility estimates help in risk assessment
• Uncertainty levels guide position sizing decisions
• Market sentiment influences overall portfolio strategy

NEXT STEPS:
-----------
• Use forecasted return in Task 4 (Portfolio Optimization)
• Consider confidence intervals for risk management
• Monitor actual vs. forecasted performance
• Recalibrate models based on new data
"""
    
    # Save report
    report_filename = f"{model_name.replace('/', '_')}_forecast_report.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    print(f"✓ Summary report saved to '{report_filename}'")
    
    # Print key highlights
    print("\n🎯 KEY HIGHLIGHTS:")
    print(f"   • {analysis_results['trend']}")
    print(f"   • {analysis_results['total_change']:+.2f}% expected return")
    print(f"   • {analysis_results['uncertainty_level']} uncertainty")
    print(f"   • {len(analysis_results['opportunities'])} opportunities identified")
    print(f"   • {len(analysis_results['risks'])} risks identified")

if __name__ == "__main__":
    main()
