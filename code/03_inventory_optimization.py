import pandas as pd
import numpy as np

def optimize_inventory():
    print("Loading inventory and demand data...")
    df_demand = pd.read_csv("../data/fact_demand_weekly.csv")
    df_sku = pd.read_csv("../data/dim_sku.csv")
    df_inv = pd.read_csv("../data/fact_inventory_snapshot.csv")
    
    Z_dict = {"A": 2.054, "B": 1.645, "C": 1.282}
    
    stats = df_demand.groupby(["sku_id", "wh_id"]).agg(
        avg_demand=("demand_qty", "mean"),
        std_demand=("demand_qty", "std")
    ).reset_index()
    
    merged = df_inv.merge(stats, on=["sku_id", "wh_id"]).merge(df_sku, on="sku_id")
    
    merged["lead_time_weeks"] = 3.0
    merged["z"] = merged["abc_class"].map(Z_dict)
    merged["safety_stock"] = np.ceil(merged["z"] * np.sqrt(merged["lead_time_weeks"] * (merged["std_demand"]**2)))
    merged["reorder_point"] = np.ceil(merged["avg_demand"] * merged["lead_time_weeks"] + merged["safety_stock"])
    merged["eoq"] = np.ceil(np.sqrt(2 * merged["avg_demand"] * 52 * 6000 / (0.22 * merged["unit_cost"])) / merged["moq"]) * merged["moq"]
    merged["max_level"] = merged["reorder_point"] + merged["eoq"]
    
    merged["position"] = merged["on_hand_qty"] + merged["on_order_qty"]
    merged["reorder_qty"] = np.where(merged["position"] <= merged["reorder_point"],
                                    np.ceil((merged["max_level"] - merged["position"]) / merged["moq"]) * merged["moq"], 0)
                                    
    print("Inventory optimization complete.")
    print(merged[["sku_id", "wh_id", "safety_stock", "reorder_point", "reorder_qty"]].head())

if __name__ == "__main__":
    optimize_inventory()
