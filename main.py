"""
Main Execution Script
====================

Runs all tasks in sequence for the Time Series Forecasting for Portfolio Management Optimization project.
"""

import os
import sys
import time
from datetime import datetime

def run_task(task_name, script_name):
    """Run a specific task"""
    print(f"\n{'='*80}")
    print(f"RUNNING {task_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Import and run the task
        if task_name == "TASK 1: DATA ANALYSIS":
            import data_analysis
            data_analysis.main()
        elif task_name == "TASK 2: FORECASTING MODELS":
            import forecasting_models
            forecasting_models.main()
        elif task_name == "TASK 3: MARKET TRENDS":
            import market_trends
            market_trends.main()
        elif task_name == "TASK 4: PORTFOLIO OPTIMIZATION":
            import portfolio_optimization
            portfolio_optimization.main()
        elif task_name == "TASK 5: BACKTESTING":
            import backtesting
            backtesting.main()
        
        end_time = time.time()
        print(f"\n{task_name} completed in {end_time - start_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"Error running {task_name}: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'yfinance', 'pandas', 'numpy', 'matplotlib', 'seaborn',
        'scikit-learn', 'statsmodels', 'pmdarima', 'tensorflow',
        'scipy', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main execution function"""
    print("=" * 80)
    print("TIME SERIES FORECASTING FOR PORTFOLIO MANAGEMENT OPTIMIZATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        print("Dependencies check failed. Exiting...")
        return
    
    print("All dependencies satisfied.")
    
    # Define tasks
    tasks = [
        ("TASK 1: DATA ANALYSIS", "data_analysis.py"),
        ("TASK 2: FORECASTING MODELS", "forecasting_models.py"),
        ("TASK 3: MARKET TRENDS", "market_trends.py"),
        ("TASK 4: PORTFOLIO OPTIMIZATION", "portfolio_optimization.py"),
        ("TASK 5: BACKTESTING", "backtesting.py")
    ]
    
    # Run tasks
    successful_tasks = 0
    total_tasks = len(tasks)
    
    for task_name, script_name in tasks:
        print(f"\n{'='*80}")
        print(f"PROGRESS: {successful_tasks}/{total_tasks} tasks completed")
        print(f"{'='*80}")
        
        if run_task(task_name, script_name):
            successful_tasks += 1
        else:
            print(f"Failed to complete {task_name}")
            print("Continuing with next task...")
    
    # Final summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {total_tasks}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {total_tasks - successful_tasks}")
    print(f"Success rate: {successful_tasks/total_tasks*100:.1f}%")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_tasks == total_tasks:
        print("\n🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
        print("\nGenerated files:")
        generated_files = [
            "TSLA_cleaned_data.csv", "BND_cleaned_data.csv", "SPY_cleaned_data.csv",
            "all_returns.csv", "model_comparison.csv", "ARIMA_forecast.csv",
            "LSTM_forecast.csv", "portfolio_weights.csv", "backtest_results.csv"
        ]
        
        for file in generated_files:
            if os.path.exists(file):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (not found)")
        
        print("\nNext steps:")
        print("1. Review the generated plots and analysis")
        print("2. Examine the CSV files for detailed results")
        print("3. Consider running individual tasks for specific analysis")
        
    else:
        print(f"\n⚠️  {total_tasks - successful_tasks} task(s) failed")
        print("Please check the error messages above and try running failed tasks individually")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
