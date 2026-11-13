
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Supply Chain Control Tower",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

BG_COLOR = "#0e1117"
CARD_COLOR = "#181f2b"
TEXT_COLOR = "#f0f2f6"
ACCENT = "#38bdf8"

RISK_COLORS = {
    "Critical": "#ef4444",
    "High": "#f97316",
    "Medium": "#facc15",
    "Low": "#22c55e",
    "Excess": "#8b5cf6"
}

CHART_TEMPLATE = "plotly_dark"


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }

    [data-testid="stSidebar"] {
        background-color: #141b26;
    }

    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #f0f2f6;
        margin-bottom: 0;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 25px;
    }

    [data-testid="stMetric"] {
        background-color: #181f2b;
        border: 1px solid #293548;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    [data-testid="stMetricValue"] {
        color: #f0f2f6;
    }

    h1, h2, h3 {
        color: #f0f2f6;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #293548;
        border-radius: 10px;
    }

    .section-caption {
        color: #94a3b8;
        font-size: 13px;
    }

    hr {
        border-color: #293548;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. DATA LOADING
# ============================================================

@st.cache_data
def load_all_data():

    files = {
        "demand": "fact_demand_weekly.csv",
        "sku": "dim_sku.csv",
        "warehouse": "dim_warehouse.csv",
        "supplier": "output_supplier_risk_scores.csv",
        "inventory": "output_inventory_recommendations.csv",
        "forecast": "output_demand_forecast.csv"
    }

    data = {}

    for key, filename in files.items():

        filepath = DATA_DIR / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Missing dataset: {filepath}"
            )

        data[key] = pd.read_csv(filepath)

    return data


try:
    data = load_all_data()

    df_demand = data["demand"]
    df_sku = data["sku"]
    df_wh = data["warehouse"]
    df_sup_scores = data["supplier"]
    df_recs = data["inventory"]
    df_fc = data["forecast"]

except Exception as e:

    st.error(f"Unable to load project datasets: {e}")
    st.stop()


# ============================================================
# 4. DATA VALIDATION AND PREPROCESSING
# ============================================================

required_columns = {
    "demand": ["sku_id", "wh_id", "week", "demand_qty"],
    "sku": ["sku_id"],
    "warehouse": ["wh_id"],
    "supplier": ["composite_score", "risk_tier"],
    "inventory": [
        "sku_id", "wh_id", "on_hand_qty",
        "reorder_point_qty", "recommended_reorder_qty",
        "risk_band", "revenue_at_risk"
    ],
    "forecast": ["sku_id", "wh_id", "target_week", "forecast_qty"]
}

datasets = {
    "demand": df_demand,
    "sku": df_sku,
    "warehouse": df_wh,
    "supplier": df_sup_scores,
    "inventory": df_recs,
    "forecast": df_fc
}

for name, columns in required_columns.items():

    missing = [
        col for col in columns
        if col not in datasets[name].columns
    ]

    if missing:
        st.error(
            f"{name} dataset is missing required columns: {missing}"
        )
        st.stop()


df_demand["week"] = pd.to_datetime(
    df_demand["week"], errors="coerce"
)

df_fc["target_week"] = pd.to_datetime(
    df_fc["target_week"], errors="coerce"
)

# Correctly enrich inventory records with SKU unit costs.
# This avoids multiplying or misaligning inventory values.

if "unit_cost" in df_sku.columns:

    sku_cost = (
        df_sku[["sku_id", "unit_cost"]]
        .drop_duplicates(subset=["sku_id"])
    )

    df_inventory = df_recs.merge(
        sku_cost,
        on="sku_id",
        how="left",
        validate="many_to_one"
    )

else:

    df_inventory = df_recs.copy()
    df_inventory["unit_cost"] = np.nan


# Calculate inventory value.

df_inventory["inventory_value"] = (
    df_inventory["on_hand_qty"]
    * df_inventory["unit_cost"]
)

# Stock position relative to reorder point.

df_inventory["stock_gap"] = (
    df_inventory["reorder_point_qty"]
    - df_inventory["on_hand_qty"]
)


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def format_crores(value):

    if pd.isna(value):
        return "₹0.00 Cr"

    return f"₹{value / 1e7:.2f} Cr"


def chart_layout(fig, height=380):

    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=35, r=25, t=65, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False
    )

    fig.update_yaxes(
        gridcolor="#293548",
        zeroline=False
    )

    return fig


def show_chart(fig):

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def show_download(df, filename):

    st.download_button(
        label="⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv"
    )


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📦 SUPPLY CHAIN")
st.sidebar.caption("Enterprise Analytics Platform")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Executive Summary",
        "Demand Forecasting",
        "Supplier Risk Analysis",
        "Inventory Optimization",
        "Data Quality"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Dataset Overview")

st.sidebar.metric(
    "Total Transactions",
    f"{len(df_demand):,}"
)

st.sidebar.metric(
    "SKU Locations",
    f"{len(df_recs):,}"
)

st.sidebar.metric(
    "Suppliers",
    f"{len(df_sup_scores):,}"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Supply Chain Control Tower | Data Science & Predictive Analytics"
)


# ============================================================
# 7. MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ Enterprise Supply Chain Control Tower</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Real-Time Operational Intelligence | '
    'Demand Forecasting | Supplier Risk | Inventory Optimization</div>',
    unsafe_allow_html=True
)


# ============================================================
# 8. EXECUTIVE SUMMARY
# ============================================================

if page == "Executive Summary":

    st.header("Executive Intelligence Overview")

    st.caption(
        "High-level visibility into supply chain risk, "
        "inventory exposure and operational performance."
    )

    # ---------------- KPI CALCULATIONS ----------------

    total_revenue_risk = df_recs[
        "revenue_at_risk"
    ].sum()

    critical_count = (
        df_recs["risk_band"]
        .eq("Critical")
        .sum()
    )

    supplier_risk_count = (
        df_sup_scores["risk_tier"]
        .isin(["Critical", "Watchlist"])
        .sum()
    )

    total_inventory = df_recs[
        "on_hand_qty"
    ].sum()

    excess_inventory = df_inventory[
        df_inventory["risk_band"] == "Excess"
    ]

    if excess_inventory["unit_cost"].notna().any():

        excess_value = excess_inventory[
            "inventory_value"
        ].sum()

    else:

        excess_value = np.nan

    # ---------------- KPI CARDS ----------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Revenue at Risk",
            format_crores(total_revenue_risk)
        )

    with c2:
        st.metric(
            "Critical SKU Locations",
            f"{critical_count:,}",
            f"{critical_count / max(len(df_recs), 1) * 100:.1f}% of locations"
        )

    with c3:
        st.metric(
            "Excess Inventory Value",
            format_crores(excess_value)
            if pd.notna(excess_value)
            else "Cost data unavailable"
        )

    with c4:
        st.metric(
            "Suppliers at Risk",
            f"{supplier_risk_count:,}"
        )

    st.markdown("---")

    # ---------------- RISK DISTRIBUTION ----------------

    st.subheader("Supply Chain Risk Distribution")

    col1, col2 = st.columns(2)

    risk_counts = (
        df_recs["risk_band"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("Risk Band")
        .reset_index(name="SKU Locations")
    )

    with col1:

        fig = px.pie(
            risk_counts,
            names="Risk Band",
            values="SKU Locations",
            hole=0.58,
            color="Risk Band",
            color_discrete_map=RISK_COLORS,
            title="SKU Location Risk Composition"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        show_chart(chart_layout(fig))

    with col2:

        fig = px.bar(
            risk_counts,
            x="Risk Band",
            y="SKU Locations",
            color="Risk Band",
            color_discrete_map=RISK_COLORS,
            title="Number of Locations by Risk Level",
            text="SKU Locations"
        )

        fig.update_traces(
            textposition="outside"
        )

        show_chart(chart_layout(fig))

    # ---------------- INVENTORY OVERVIEW ----------------

    st.subheader("Inventory Health Overview")

    col1, col2 = st.columns(2)

    with col1:

        inventory_status = (
            df_recs["risk_band"]
            .value_counts()
            .reset_index()
        )

        inventory_status.columns = [
            "Risk Band", "Count"
        ]

        fig = px.bar(
            inventory_status,
            x="Risk Band",
            y="Count",
            color="Risk Band",
            color_discrete_map=RISK_COLORS,
            title="Inventory Health by Risk Category"
        )

        show_chart(chart_layout(fig))

    with col2:

        fig = px.scatter(
            df_inventory,
            x="on_hand_qty",
            y="reorder_point_qty",
            color="risk_band",
            color_discrete_map=RISK_COLORS,
            hover_data=["sku_id", "wh_id"],
            title="Stock Level vs Reorder Point",
            labels={
                "on_hand_qty": "Available Stock",
                "reorder_point_qty": "Reorder Point",
                "risk_band": "Risk Band"
            }
        )

        show_chart(chart_layout(fig))

    # ---------------- URGENT ACTIONS ----------------

    st.subheader("🚨 Priority Action Queue")

    urgent = df_recs.sort_values(
        "revenue_at_risk",
        ascending=False
    ).head(10)

    st.dataframe(
        urgent,
        use_container_width=True,
        hide_index=True
    )

    show_download(
        urgent,
        "priority_action_queue.csv"
    )


# ============================================================
# 9. DEMAND FORECASTING
# ============================================================

elif page == "Demand Forecasting":

    st.header("Demand Forecasting & Predictive Analytics")

    st.caption(
        "Analyze historical demand patterns and compare "
        "actual demand with predicted demand."
    )

    c1, c2 = st.columns(2)

    with c1:

        selected_sku = st.selectbox(
            "Select SKU",
            sorted(df_sku["sku_id"].dropna().unique())
        )

    with c2:

        selected_wh = st.selectbox(
            "Select Warehouse",
            sorted(df_wh["wh_id"].dropna().unique())
        )

    hist = df_demand[
        (df_demand["sku_id"] == selected_sku)
        & (df_demand["wh_id"] == selected_wh)
    ].sort_values("week").tail(26)

    forecast = df_fc[
        (df_fc["sku_id"] == selected_sku)
        & (df_fc["wh_id"] == selected_wh)
    ].sort_values("target_week")

    st.markdown("---")

    if hist.empty:

        st.warning("No historical demand data available.")

    else:

        # Demand KPIs

        avg_demand = hist["demand_qty"].mean()

        max_demand = hist["demand_qty"].max()

        total_demand = hist["demand_qty"].sum()

        k1, k2, k3 = st.columns(3)

        k1.metric(
            "Total Historical Demand",
            f"{total_demand:,.0f} Units"
        )

        k2.metric(
            "Average Weekly Demand",
            f"{avg_demand:,.1f} Units"
        )

        k3.metric(
            "Peak Weekly Demand",
            f"{max_demand:,.0f} Units"
        )

        # Historical vs Forecast Chart

        st.subheader("Historical Demand vs Future Forecast")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=hist["week"],
                y=hist["demand_qty"],
                mode="lines+markers",
                name="Historical Demand",
                line=dict(color="#38bdf8", width=3)
            )
        )

        if not forecast.empty:

            fig.add_trace(
                go.Scatter(
                    x=forecast["target_week"],
                    y=forecast["forecast_qty"],
                    mode="lines+markers",
                    name="Predicted Demand",
                    line=dict(
                        color="#f97316",
                        width=3,
                        dash="dash"
                    )
                )
            )

        fig.update_layout(
            title=f"Demand Forecast: {selected_sku} | {selected_wh}",
            xaxis_title="Week",
            yaxis_title="Demand Units"
        )

        show_chart(chart_layout(fig, 450))

        # Forecast table

        st.subheader("Forecast Output")

        if not forecast.empty:

            display_cols = [
                col for col in [
                    "target_week",
                    "horizon_weeks",
                    "forecast_qty"
                ]
                if col in forecast.columns
            ]

            st.dataframe(
                forecast[display_cols],
                use_container_width=True,
                hide_index=True
            )

            show_download(
                forecast,
                "demand_forecast.csv"
            )

        else:

            st.info("No forecast available for this SKU and warehouse.")

    # Demand by SKU

    st.markdown("---")

    st.subheader("Top 10 SKUs by Historical Demand")

    sku_demand = (
        df_demand.groupby("sku_id")["demand_qty"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    fig = px.bar(
        sku_demand,
        x="sku_id",
        y="demand_qty",
        color="demand_qty",
        color_continuous_scale="Blues",
        title="Highest Demand SKUs"
    )

    show_chart(chart_layout(fig))


# ============================================================
# 10. SUPPLIER RISK ANALYSIS
# ============================================================

elif page == "Supplier Risk Analysis":

    st.header("Supplier Risk Intelligence")

    st.caption(
        "Evaluate supplier reliability and identify "
        "suppliers requiring operational attention."
    )

    # Supplier KPIs

    total_suppliers = len(df_sup_scores)

    avg_score = df_sup_scores[
        "composite_score"
    ].mean()

    critical_suppliers = df_sup_scores[
        df_sup_scores["risk_tier"] == "Critical"
    ].shape[0]

    watchlist_suppliers = df_sup_scores[
        df_sup_scores["risk_tier"] == "Watchlist"
    ].shape[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Suppliers", f"{total_suppliers:,}")

    c2.metric("Average Risk Score", f"{avg_score:.2f}")

    c3.metric("Critical Suppliers", f"{critical_suppliers:,}")

    c4.metric("Watchlist Suppliers", f"{watchlist_suppliers:,}")

    st.markdown("---")

    # Risk Score Distribution

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df_sup_scores,
            x="composite_score",
            nbins=15,
            color_discrete_sequence=["#38bdf8"],
            title="Supplier Composite Score Distribution"
        )

        show_chart(chart_layout(fig))

    with col2:

        tier_counts = (
            df_sup_scores["risk_tier"]
            .value_counts()
            .reset_index()
        )

        tier_counts.columns = [
            "Risk Tier", "Supplier Count"
        ]

        fig = px.pie(
            tier_counts,
            names="Risk Tier",
            values="Supplier Count",
            hole=0.55,
            title="Supplier Risk Tier Composition"
        )

        show_chart(chart_layout(fig))

    # Supplier ranking

    st.subheader("Supplier Risk Ranking")

    sort_order = st.selectbox(
        "Rank suppliers by",
        ["Highest Risk", "Lowest Risk"]
    )

    ascending = sort_order == "Highest Risk"

    ranked = df_sup_scores.sort_values(
        "composite_score",
        ascending=ascending
    )

    st.dataframe(
        ranked,
        use_container_width=True,
        hide_index=True
    )

    show_download(
        ranked,
        "supplier_risk_analysis.csv"
    )


# ============================================================
# 11. INVENTORY OPTIMIZATION
# ============================================================

elif page == "Inventory Optimization":

    st.header("Inventory Optimization & Replenishment")

    st.caption(
        "Identify stock shortages, excess inventory "
        "and recommended replenishment quantities."
    )

    # Inventory KPIs

    total_stock = df_recs[
        "on_hand_qty"
    ].sum()

    total_reorder = df_recs[
        "recommended_reorder_qty"
    ].sum()

    shortage_count = (
        df_recs["on_hand_qty"]
        < df_recs["reorder_point_qty"]
    ).sum()

    excess_count = (
        df_recs["risk_band"] == "Excess"
    ).sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total On-Hand Stock",
        f"{total_stock:,.0f}"
    )

    c2.metric(
        "Recommended Reorder",
        f"{total_reorder:,.0f}"
    )

    c3.metric(
        "Below Reorder Point",
        f"{shortage_count:,}"
    )

    c4.metric(
        "Excess Stock Locations",
        f"{excess_count:,}"
    )

    st.markdown("---")

    # Stock vs ROP analysis

    st.subheader("Stock Level vs Reorder Point")

    fig = go.Figure()

    top_items = df_recs.nlargest(
        15,
        "recommended_reorder_qty"
    ).copy()

    fig.add_trace(
        go.Bar(
            x=top_items["sku_id"],
            y=top_items["on_hand_qty"],
            name="On-Hand Stock",
            marker_color="#38bdf8"
        )
    )

    fig.add_trace(
        go.Bar(
            x=top_items["sku_id"],
            y=top_items["reorder_point_qty"],
            name="Reorder Point",
            marker_color="#ef4444"
        )
    )

    fig.add_trace(
        go.Bar(
            x=top_items["sku_id"],
            y=top_items["recommended_reorder_qty"],
            name="Recommended Reorder",
            marker_color="#facc15"
        )
    )

    fig.update_layout(
        barmode="group",
        title="Top 15 SKU Replenishment Requirements",
        xaxis_title="SKU",
        yaxis_title="Units"
    )

    show_chart(chart_layout(fig, 450))


    # Inventory value

    if df_inventory["unit_cost"].notna().any():

        st.subheader("Inventory Value by Risk Category")

        value_by_risk = (
            df_inventory.groupby("risk_band")[
                "inventory_value"
            ]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            value_by_risk,
            x="risk_band",
            y="inventory_value",
            color="risk_band",
            color_discrete_map=RISK_COLORS,
            title="Inventory Capital Exposure"
        )

        show_chart(chart_layout(fig))

    # Detailed inventory recommendations

    st.subheader("Replenishment Recommendations")

    risk_filter = st.multiselect(
        "Filter Risk Categories",
        options=sorted(
            df_recs["risk_band"].dropna().unique()
        ),
        default=sorted(
            df_recs["risk_band"].dropna().unique()
        )
    )

    filtered_inventory = df_recs[
        df_recs["risk_band"].isin(risk_filter)
    ]

    display_inventory = filtered_inventory.drop(
        columns=["eoq_qty"],
        errors="ignore"
    )

    st.dataframe(
        display_inventory,
        use_container_width=True,
        hide_index=True
    )

    show_download(
        display_inventory,
        "inventory_recommendations.csv"
    )


# ============================================================
# 12. DATA QUALITY MONITORING
# ============================================================

elif page == "Data Quality":

    st.header("Data Quality & Dataset Monitoring")

    st.caption(
        "Monitor missing values, duplicates and dataset completeness."
    )

    dataset_dict = {
        "Weekly Demand": df_demand,
        "SKU Dimension": df_sku,
        "Warehouse Dimension": df_wh,
        "Supplier Risk": df_sup_scores,
        "Inventory Recommendations": df_recs,
        "Demand Forecast": df_fc
    }

    quality_results = []

    for name, df in dataset_dict.items():

        total_cells = df.size

        missing_cells = df.isnull().sum().sum()

        duplicates = df.duplicated().sum()

        completeness = (
            (1 - missing_cells / total_cells) * 100
            if total_cells > 0 else 100
        )

        quality_results.append({
            "Dataset": name,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Missing Values": missing_cells,
            "Duplicate Rows": duplicates,
            "Completeness (%)": round(completeness, 2)
        })

    quality_df = pd.DataFrame(quality_results)

    # Quality KPIs

    total_rows = sum(
        len(df) for df in dataset_dict.values()
    )

    total_missing = quality_df[
        "Missing Values"
    ].sum()

    total_duplicates = quality_df[
        "Duplicate Rows"
    ].sum()

    overall_completeness = (
        100 * (
            1 - total_missing /
            max(sum(df.size for df in dataset_dict.values()), 1)
        )
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Dataset Rows", f"{total_rows:,}")

    c2.metric("Missing Values", f"{total_missing:,}")

    c3.metric("Duplicate Rows", f"{total_duplicates:,}")

    c4.metric(
        "Data Completeness",
        f"{overall_completeness:.2f}%"
    )

    st.markdown("---")

    st.subheader("Dataset Quality Report")

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True
    )

    # Completeness chart

    fig = px.bar(
        quality_df,
        x="Dataset",
        y="Completeness (%)",
        color="Completeness (%)",
        color_continuous_scale="RdYlGn",
        range_y=[0, 100],
        title="Data Completeness Across Project Datasets"
    )

    show_chart(chart_layout(fig))

    # Missing value analysis

    st.subheader("Missing Values by Column")

    selected_dataset = st.selectbox(
        "Choose Dataset",
        list(dataset_dict.keys())
    )

    selected_df = dataset_dict[selected_dataset]

    missing_df = (
        selected_df.isnull()
        .sum()
        .reset_index()
    )

    missing_df.columns = [
        "Column",
        "Missing Values"
    ]

    missing_df = missing_df[
        missing_df["Missing Values"] > 0
    ].sort_values(
        "Missing Values",
        ascending=False
    )

    if missing_df.empty:

        st.success(
            "No missing values found in this dataset."
        )

    else:

        fig = px.bar(
            missing_df,
            x="Missing Values",
            y="Column",
            orientation="h",
            color="Missing Values",
            color_continuous_scale="Reds",
            title="Missing Values by Feature"
        )

        show_chart(chart_layout(fig))

    show_download(
        quality_df,
        "data_quality_report.csv"
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:15px;">
        <b>Enterprise Supply Chain Control Tower</b><br>
        Powered by Python | Pandas | NumPy | Plotly | Streamlit<br>
        Demand Forecasting • Supplier Risk • Inventory Optimization
    </div>
    """,
    unsafe_allow_html=True
)
