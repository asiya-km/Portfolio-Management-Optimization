# Time Series Forecasting for Portfolio Management Optimization

## Project Overview

This project implements a comprehensive time series forecasting and portfolio optimization system for GMF Investments, analyzing TSLA, BND, and SPY data from 2015-2025.

## Business Context

**GMF Investments** leverages cutting-edge technology for personalized portfolio management, aiming to:
- Predict market trends using time series forecasting
- Optimize asset allocation using Modern Portfolio Theory
- Enhance portfolio performance while managing risks

## Project Structure

```
├── requirements.txt              # Dependencies
├── utils.py                     # Utility functions
├── data_analysis.py             # Task 1: Data preprocessing
├── forecasting_models.py        # Task 2: ARIMA & LSTM models
├── market_trends.py             # Task 3: Future forecasting
├── portfolio_optimization.py    # Task 4: MPT optimization
├── backtesting.py               # Task 5: Strategy validation
└── main.py                      # Main execution script
```

## Data Sources

- **TSLA**: Tesla Inc. (high-growth, high-risk)
- **BND**: Vanguard Total Bond Market ETF (stability)
- **SPY**: S&P 500 ETF (diversified exposure)
- **Period**: July 1, 2015 - July 31, 2025
- **Source**: Yahoo Finance API

## Task Breakdown

### Task 1: Data Analysis
- Fetch and clean historical data
- Perform EDA with volatility analysis
- Test stationarity (ADF test)
- Calculate risk metrics (VaR, Sharpe ratio)

### Task 2: Forecasting Models
- **ARIMA/SARIMA**: Statistical time series model
- **LSTM**: Deep learning neural network
- Model comparison using MAE, RMSE, MAPE
- Train on 2015-2023, test on 2024-2025

### Task 3: Market Trends
- Generate 6-12 month price forecasts
- Analyze trend directions
- Calculate confidence intervals
- Identify opportunities and risks

### Task 4: Portfolio Optimization
- Implement Modern Portfolio Theory
- Generate Efficient Frontier
- Find optimal weights (Max Sharpe, Min Volatility)
- Use forecasted returns for TSLA, historical for BND/SPY

### Task 5: Backtesting
- Simulate strategy performance
- Compare against 60% SPY / 40% BND benchmark
- Calculate performance metrics
- Validate strategy effectiveness

## Key Technologies

- **Python**: Primary language
- **YFinance**: Financial data API
- **Pandas/NumPy**: Data manipulation
- **StatsModels**: ARIMA modeling
- **TensorFlow**: LSTM implementation
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Visualization

## Usage

### Installation:
```bash
pip install -r requirements.txt
```

### Run Complete Analysis:
```bash
python main.py
```

### Run Individual Tasks:
```bash
python data_analysis.py          # Task 1
python forecasting_models.py     # Task 2
python market_trends.py          # Task 3
python portfolio_optimization.py # Task 4
python backtesting.py            # Task 5
```

## Expected Outcomes

### Skills Developed:
- API usage and data wrangling
- Time series modeling (ARIMA, LSTM)
- Portfolio optimization (MPT)
- Strategy backtesting
- Data visualization

### Knowledge Gained:
- Asset class characteristics
- Efficient Market Hypothesis
- Stationarity and model selection
- Portfolio theory and optimization
- Risk-adjusted performance evaluation

## Key Insights

### Asset Characteristics:
- **TSLA**: High volatility, growth potential
- **BND**: Low volatility, income generation
- **SPY**: Moderate risk, market exposure

### Model Performance:
- ARIMA: Good for short-term forecasts
- LSTM: Captures complex patterns
- Selection depends on horizon and data

### Portfolio Strategy:
- Diversification reduces risk
- Optimal weights depend on risk tolerance
- Regular rebalancing is essential

## Limitations

- Past performance ≠ future results
- Forecasts less reliable over longer horizons
- Models may miss structural changes
- Transaction costs affect real performance

## Future Enhancements

- Additional models (Prophet, XGBoost)
- Real-time data integration
- Advanced risk management
- Web application dashboard
- Alternative assets inclusion

## Conclusion

This project provides a comprehensive framework for quantitative finance analysis, combining time series forecasting with portfolio optimization. It demonstrates practical application of statistical and machine learning techniques for investment decision-making.

The system successfully integrates data analysis, forecasting, optimization, and validation, providing valuable insights for portfolio management while acknowledging the limitations and uncertainties inherent in financial markets.
