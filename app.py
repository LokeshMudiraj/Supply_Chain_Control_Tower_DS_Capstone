import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Supply Chain Control Tower - Data Science", layout="wide")

st.title("🛡️ Enterprise Supply Chain Control Tower")
st.markdown("### Data Science & Predictive Analytics Dashboard")

@st.cache_data
def load_all_data():
    df_demand = pd.read_csv("data/fact_demand_weekly.csv")
    df_sku = pd.read_csv("data/dim_sku.csv")
    df_wh = pd.read_csv("data/dim_warehouse.csv")
    df_sup_scores = pd.read_csv("data/output_supplier_risk_scores.csv")
    df_recs = pd.read_csv("data/output_inventory_recommendations.csv")
    df_fc = pd.read_csv("data/output_demand_forecast.csv")
    return df_demand, df_sku, df_wh, df_sup_scores, df_recs, df_fc

df_demand, df_sku, df_wh, df_sup_scores, df_recs, df_fc = load_all_data()

st.sidebar.header("Navigation")
page = st.sidebar.radio("Select View", ["Executive Summary", "Demand Forecasting", "Supplier Risk Scoring", "Inventory Optimization & ROP"])

if page == "Executive Summary":
    st.subheader("Executive Control Tower Risk Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rev_risk = df_recs["revenue_at_risk"].sum() / 1e7
        st.metric("Revenue at Risk", f"₹{rev_risk:.2f} Cr")
    with col2:
        crit_count = len(df_recs[df_recs["risk_band"] == "Critical"])
        st.metric("Critical SKU-Locs", f"{crit_count} of 360")
    with col3:
        excess_val = (df_recs[df_recs["risk_band"] == "Excess"]["on_hand_qty"] * df_recs.merge(df_sku, on="sku_id")["unit_cost"]).sum() / 1e7
        st.metric("Excess Inventory Value", f"₹{excess_val:.2f} Cr")
    with col4:
        at_risk_sup = len(df_sup_scores[df_sup_scores["risk_tier"].isin(["Watchlist", "Critical"])])
        st.metric("Suppliers at Risk", f"{at_risk_sup} Suppliers")

    st.markdown("---")
    st.markdown("### Risk Band Distribution Across 360 SKU-Locations")
    st.bar_chart(df_recs["risk_band"].value_counts())
    
    st.markdown("### Top SKU-Locations Needing Urgent Action")
    st.dataframe(df_recs[["sku_id", "wh_id", "on_hand_qty", "reorder_point_qty", "recommended_reorder_qty", "risk_band", "revenue_at_risk"]].sort_values("revenue_at_risk", ascending=False).head(10))

elif page == "Demand Forecasting":
    st.subheader("ML Demand Forecasting Module")
    sku_sel = st.selectbox("Select SKU", df_sku["sku_id"].unique())
    wh_sel = st.selectbox("Select Warehouse", df_wh["wh_id"].unique())
    
    hist = df_demand[(df_demand["sku_id"] == sku_sel) & (df_demand["wh_id"] == wh_sel)].tail(26)
    fc = df_fc[(df_fc["sku_id"] == sku_sel) & (df_fc["wh_id"] == wh_sel)]
    
    st.markdown(f"#### Demand Trajectory & 4-Week Forecast for {sku_sel} at {wh_sel}")
    st.dataframe(fc[["target_week", "horizon_weeks", "forecast_qty"]])

elif page == "Supplier Risk Scoring":
    st.subheader("Multi-Factor Supplier Risk Scorecard")
    st.dataframe(df_sup_scores.sort_values("composite_score"))

elif page == "Inventory Optimization & ROP":
    st.subheader("Inventory Replenishment & Safety Stock Recommendations")
    
    # Drop the eoq_qty column if it exists in the dataframe
    df_display = df_recs.drop(columns=["eoq_qty"], errors="ignore")
    
    st.dataframe(df_display)
