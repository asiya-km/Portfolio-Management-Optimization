# Time Series Forecasting for Portfolio Management Optimization

## Project Overview
This project implements a comprehensive time series forecasting and portfolio optimization system for GMF Investments. The system analyzes historical financial data for Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY) to provide data-driven portfolio management insights.

## Business Objective
Guide Me in Finance (GMF) Investments leverages cutting-edge technology and data-driven insights to provide clients with tailored investment strategies. This system integrates advanced time series forecasting models to predict market trends, optimize asset allocation, and enhance portfolio performance.

## Project Structure
```
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── data_analysis.py         # Task 1: Data preprocessing and exploration
├── forecasting_models.py    # Task 2: Time series forecasting models
├── market_trends.py         # Task 3: Future market trend forecasting
├── portfolio_optimization.py # Task 4: Portfolio optimization using MPT
├── backtesting.py           # Task 5: Strategy backtesting
├── main.py                  # Main execution script
└── utils.py                 # Utility functions
```

## Key Features
- **Data Analysis**: Comprehensive EDA with volatility analysis and stationarity testing
- **Forecasting Models**: ARIMA/SARIMA and LSTM models for price prediction
- **Portfolio Optimization**: Modern Portfolio Theory implementation with Efficient Frontier
- **Backtesting**: Historical performance simulation and benchmark comparison
- **Risk Management**: VaR, Sharpe Ratio, and volatility analysis

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the complete analysis:
```bash
python main.py
```

Or run individual tasks:
```bash
python data_analysis.py      # Task 1
python forecasting_models.py # Task 2
python market_trends.py      # Task 3
python portfolio_optimization.py # Task 4
python backtesting.py        # Task 5
```

## Data Sources
- **TSLA**: Tesla Inc. stock data (high-growth, high-risk)
- **BND**: Vanguard Total Bond Market ETF (stability, low-risk)
- **SPY**: S&P 500 ETF (diversified, moderate-risk)
- **Period**: July 1, 2015 to July 31, 2025
- **Source**: YFinance API

## Expected Outcomes
- Time series forecasting models with performance metrics
- Optimized portfolio weights based on Modern Portfolio Theory
- Backtesting results comparing strategy vs benchmark performance
- Comprehensive risk analysis and market insights

## Key Insights
- Understanding of different asset class characteristics
- Application of Efficient Market Hypothesis principles
- Stationarity testing and model selection
- Portfolio optimization using the Efficient Frontier
- Risk-adjusted performance evaluation
