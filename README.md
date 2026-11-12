# Supply Chain Control Tower - Data Science Capstone

This project contains the complete Data Science & Predictive Analytics implementation for the Enterprise Supply Chain Control Tower (Meridian Consumer Products Ltd).

## Folder Structure
```
Supply_Chain_Control_Tower_DS_Capstone/
├── README.md                              # Full setup instructions
├── app.py                                 # Streamlit UI dashboard
├── requirements.txt                       # Dependencies list
├── data/                                  # Synthetic datasets & ML outputs
│   ├── dim_sku.csv                        # 60 SKUs
│   ├── dim_warehouse.csv                  # 6 RDCs
│   ├── dim_supplier.csv                   # 40 Suppliers
│   ├── fact_demand_weekly.csv             # 156 weeks demand history (56,160 rows)
│   ├── fact_po_line.csv                   # 2,500 closed PO lines
│   ├── fact_inventory_snapshot.csv        # 360 SKU-location snapshots
│   ├── output_demand_forecast.csv         # 4-week ahead ML forecast
│   ├── output_supplier_risk_scores.csv    # Multi-factor composite supplier scores
│   └── output_inventory_recommendations.csv# Safety stock, ROP, EOQ & risk bands
└── code/                                  # Pure DS execution scripts
    ├── 01_demand_forecasting.py           # ML demand forecast engine
    ├── 02_supplier_risk_scoring.py        # Supplier scoring engine
    └── 03_inventory_optimization.py       # (s, S) inventory optimization engine
```

## How to Run

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Python DS Scripts** (optional):
   ```bash
   cd code
   python 01_demand_forecasting.py
   python 02_supplier_risk_scoring.py
   python 03_inventory_optimization.py
   ```

3. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
