import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

def evaluate_sarima_accuracy(test_steps=4):
    print("Loading demand data for evaluation...")
    df_demand = pd.read_csv("../data/fact_demand_weekly.csv")
    df_demand["week"] = pd.to_datetime(df_demand["week"])
    
    # Pick a single representative SKU and Warehouse for instant evaluation
    sku = "SKU-1001"
    wh = "WH-BWD"
    
    sub = df_demand[(df_demand["sku_id"] == sku) & (df_demand["wh_id"] == wh)].sort_values("week")
    print(f"Evaluating SARIMA accuracy for {sku} at {wh} (holding out last {test_steps} weeks)...")
    
    # Train-test split: hold out last 4 weeks
    train = sub.iloc[:-test_steps].set_index("week")["demand_qty"].asfreq("W-SUN")
    test = sub.iloc[-test_steps:].set_index("week")["demand_qty"].asfreq("W-SUN")
    
    try:
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52),
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=test_steps)
        preds = forecast.predicted_mean.values
        actuals = test.values
        
        # Calculate Metrics
        mae = np.mean(np.abs(actuals - preds))
        rmse = np.sqrt(np.mean((actuals - preds) ** 2))
        mape = np.mean(np.abs((actuals - preds) / np.maximum(actuals, 1))) * 100
        
        print("\n==============================")
        print(" 📈 SARIMA FORECAST EVALUATION ")
        print("==============================")
        print(f"Target SKU & Warehouse : {sku} @ {wh}")
        print(f"Mean Absolute Error (MAE) : {mae:.2f} units")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f} units")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print("==============================")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    evaluate_sarima_accuracy()