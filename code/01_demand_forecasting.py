import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

def run_forecasting(forecast_steps=4):
    print("Loading demand data...")
    df_demand = pd.read_csv("../data/fact_demand_weekly.csv")
    df_demand["week"] = pd.to_datetime(df_demand["week"])
    
    sku_list = df_demand["sku_id"].unique()
    wh_list = df_demand["wh_id"].unique()
    
    forecast_results = []
    
    print(f"Running SARIMA demand forecasting for {len(sku_list) * len(wh_list)} SKU-Warehouse combinations...")
    
    for sku in sku_list:
        for wh in wh_list:
            sub = df_demand[(df_demand["sku_id"] == sku) & (df_demand["wh_id"] == wh)].sort_values("week")
            ts = sub.set_index("week")["demand_qty"].asfreq("W-SUN")
            
            try:
                # SARIMA (1, 1, 1) x (1, 1, 1, 52)
                model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52),
                                enforce_stationarity=False, enforce_invertibility=False)
                results = model.fit(disp=False)
                forecast = results.get_forecast(steps=forecast_steps)
                pred_mean = forecast.predicted_mean
            except Exception as e:
                # Fallback to simple moving average if SARIMA fails to converge
                last_week = ts.index[-1]
                future_dates = pd.date_range(start=last_week + pd.Timedelta(weeks=1), periods=forecast_steps, freq="W-SUN")
                pred_mean = pd.Series([ts.tail(4).mean()] * forecast_steps, index=future_dates)
            
            last_date = ts.index[-1].strftime("%Y-%m-%d")
            for step_idx, (fc_date, fc_val) in enumerate(pred_mean.items(), start=1):
                forecast_results.append({
                    "forecast_run_date": last_date,
                    "sku_id": sku,
                    "wh_id": wh,
                    "target_week": fc_date.strftime("%Y-%m-%d"),
                    "horizon_weeks": step_idx,
                    "forecast_qty": max(0, round(float(fc_val), 2))
                })
                
    df_fc = pd.DataFrame(forecast_results)
    output_path = "../data/output_demand_forecast.csv"
    df_fc.to_csv(output_path, index=False)
    print(f"SARIMA demand forecasting complete. Output saved to {output_path}.")
    return df_fc

if __name__ == "__main__":
    run_forecasting()
