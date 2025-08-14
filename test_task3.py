"""
Simple test script for Task 3 functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample TSLA data for testing"""
    print("Creating sample TSLA data for testing...")
    
    # Generate sample dates
    start_date = datetime(2015, 7, 1)
    end_date = datetime(2025, 7, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate sample price data (simplified)
    np.random.seed(42)
    base_price = 200
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    
    # Filter to trading days (Mon-Fri)
    data = data[data['Date'].dt.weekday < 5].copy()
    data.set_index('Date', inplace=True)
    
    print(f"✓ Created sample data with {len(data)} trading days")
    print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"  Price range: ${data['Close'].min():.2f} to ${data['Close'].max():.2f}")
    
    return data

def test_forecasting_functions():
    """Test the forecasting functions without matplotlib"""
    print("\n" + "="*60)
    print("TESTING TASK 3 FORECASTING FUNCTIONS")
    print("="*60)
    
    # Create sample data
    tsla_data = create_sample_data()
    
    # Test forecast date generation
    print("\n1. Testing forecast date generation...")
    forecast_dates = generate_forecast_dates(tsla_data.index[-1], 252)
    print(f"✓ Generated {len(forecast_dates)} forecast dates")
    print(f"  Forecast period: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
    
    # Test ARIMA forecasting
    print("\n2. Testing ARIMA forecasting...")
    try:
        from pmdarima import auto_arima
        forecast_values, conf_int = arima_forecast(tsla_data['Close'], 60)  # 60 days for testing
        if forecast_values is not None:
            print("✓ ARIMA forecast successful")
            print(f"  Forecast range: ${forecast_values[0]:.2f} to ${forecast_values[-1]:.2f}")
        else:
            print("✗ ARIMA forecast failed")
    except Exception as e:
        print(f"✗ ARIMA test failed: {e}")
    
    # Test trend analysis
    print("\n3. Testing trend analysis...")
    if 'forecast_values' in locals() and forecast_values is not None:
        analysis_results = analyze_trends(forecast_values, conf_int, forecast_dates[:60], tsla_data['Close'])
        print("✓ Trend analysis completed")
        print(f"  Trend: {analysis_results['trend']}")
        print(f"  Expected return: {analysis_results['total_change']:+.2f}%")
    else:
        print("⚠ Skipping trend analysis (no forecast available)")
    
    # Save test results
    print("\n4. Saving test results...")
    try:
        if 'forecast_values' in locals() and forecast_values is not None:
            forecast_df = pd.DataFrame({
                'Date': forecast_dates[:60],
                'Forecast_Price': forecast_values,
                'Lower_CI': conf_int[:, 0] if conf_int is not None else forecast_values,
                'Upper_CI': conf_int[:, 1] if conf_int is not None else forecast_values
            })
            forecast_df.to_csv("test_forecast_results.csv", index=False)
            print("✓ Test results saved to 'test_forecast_results.csv'")
    except Exception as e:
        print(f"⚠ Error saving results: {e}")
    
    print("\n" + "="*60)
    print("TASK 3 TESTING COMPLETED")
    print("="*60)

def generate_forecast_dates(last_date, periods=252):
    """Generate future trading dates (excluding weekends and holidays)"""
    future_dates = []
    current_date = last_date
    
    while len(future_dates) < periods:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # Exclude weekends
            future_dates.append(current_date)
    
    return pd.DatetimeIndex(future_dates[:periods])

def arima_forecast(data, periods=252):
    """Generate ARIMA/SARIMA forecast with confidence intervals"""
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
            max_p=3, max_q=3, max_d=1,  # Reduced for testing
            max_P=1, max_Q=1, max_D=1,
            information_criterion='aic',
            seasonal_test='ch',
            trace=False
        )
        
        print(f"✓ ARIMA model built with parameters: {auto_model.order}")
        
        # Generate forecast
        forecast = auto_model.predict(n_periods=periods)
        
        # Simple confidence intervals
        historical_volatility = np.std(np.diff(data) / data[:-1])
        std_dev = historical_volatility * np.sqrt(np.arange(1, len(forecast) + 1))
        lower_ci = forecast - 1.96 * std_dev
        upper_ci = forecast + 1.96 * std_dev
        conf_int = np.column_stack([lower_ci, upper_ci])
        
        print("✓ Generated forecast with confidence intervals")
        return forecast, conf_int
        
    except Exception as e:
        print(f"✗ ARIMA forecasting error: {e}")
        return None, None

def analyze_trends(forecast_values, conf_int, forecast_dates, historical_data):
    """Comprehensive trend analysis and market insights"""
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
    
    return {
        'trend': trend,
        'sentiment': sentiment,
        'total_change': total_change,
        'forecast_volatility': forecast_volatility,
        'historical_volatility': historical_volatility,
        'volatility_trend': volatility_trend,
        'opportunities': opportunities,
        'risks': risks,
        'support_level': np.min(conf_int[:, 0]) if conf_int is not None else None,
        'resistance_level': np.max(conf_int[:, 1]) if conf_int is not None else None,
        'target_price': end_price
    }

if __name__ == "__main__":
    test_forecasting_functions()
