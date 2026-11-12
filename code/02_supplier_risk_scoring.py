import pandas as pd
import numpy as np

def calculate_supplier_scores():
    print("Loading purchase order and supplier data...")
    df_po = pd.read_csv("../data/fact_po_line.csv")
    df_sup = pd.read_csv("../data/dim_supplier.csv")
    
    agg = df_po.groupby("supplier_id").agg(
        on_time_rate=("on_time_flag", "mean"),
        fill_rate=("in_full_flag", "mean"),
        total_ordered=("ordered_qty", "sum"),
        total_received=("received_qty", "sum"),
        total_rejected=("rejected_qty", "sum"),
        mean_lt=("lead_time_days", "mean"),
        contract_lt=("contract_lead_time_days", "mean"),
        paid_price=("unit_price", "mean"),
        contract_price=("contract_price", "mean")
    ).reset_index()
    
    agg["rejection_rate"] = agg["total_rejected"] / np.maximum(1, agg["total_received"])
    agg["score_otd"] = agg["on_time_rate"] * 100
    agg["score_qual"] = np.clip(100 * (1 - agg["rejection_rate"] / 0.05), 0, 100)
    agg["score_lt"] = np.clip(100 * (1 - np.maximum(0, (agg["mean_lt"] / agg["contract_lt"] - 1)) / 0.30), 0, 100)
    agg["score_cost"] = np.clip(100 * (1 - np.maximum(0, (agg["paid_price"] / agg["contract_price"] - 1)) / 0.06), 0, 100)
    agg["score_fill"] = np.clip(100 * agg["fill_rate"], 0, 100)
    
    agg["composite_score"] = (
        0.30 * agg["score_otd"] +
        0.25 * agg["score_qual"] +
        0.15 * agg["score_lt"] +
        0.10 * agg["score_cost"] +
        0.20 * agg["score_fill"]
    ).round(1)
    
    def assign_tier(score):
        if score >= 85: return "Preferred"
        elif score >= 70: return "Approved"
        elif score >= 55: return "Watchlist"
        else: return "Critical"
        
    agg["risk_tier"] = agg["composite_score"].apply(assign_tier)
    print("Supplier risk scoring complete.")
    print(agg[["supplier_id", "composite_score", "risk_tier"]].head())

if __name__ == "__main__":
    calculate_supplier_scores()
