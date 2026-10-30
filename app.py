"""
AIRWISE — Explainable Air Quality Prediction System
==================================================
Master Streamlit Application Entrypoint.
"""

# Import lightgbm first for Windows C-runtime stability
import lightgbm as lgb
import streamlit as st
import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Page Configuration
st.set_page_config(
    page_title="AIRWISE — Air Quality Prediction & Explainability",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 5px solid #2B6CB0;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #718096;
        margin-bottom: 0.3rem;
    }
    .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1A365D;
    }
    .card-caption {
        font-size: 0.8rem;
        color: #718096;
        margin-top: 0.2rem;
    }
    .info-box {
        background-color: #EBF8FF;
        border-left: 4px solid #3182CE;
        padding: 1rem;
        border-radius: 6px;
        color: #2B6CB0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/air-quality.png", width=64)
    st.title("AIRWISE System")
    st.caption("Next-Hour AQI Forecasting with SHAP Explainability")
    st.markdown("---")
    st.markdown("**Station:** Aotizhongxin, Beijing")
    st.markdown("**Dataset:** UCI Multi-Site Air Quality (35,064 hourly records)")
    st.markdown("**AQI Standard:** US EPA 40 CFR Part 58")
    st.markdown("---")
    st.markdown("### Research Team")
    st.markdown("- Harsh Ranjan (23BDS0078)\n- Himanshu Kumar (23BDS0069)\n- Mohammad Ibrahim Ahmad (23BDS0073)")
    st.caption("Vellore Institute of Technology | Academic Year 2026–27")

# Main Content
st.markdown("<div class='main-header'>AIRWISE: Explainable Air Quality Index Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Multi-Modal Ensemble Machine Learning & Transparent Game-Theoretic SHAP Interpretability for Urban Atmospheric Intelligence</div>", unsafe_allow_html=True)

# Load evaluation and dataset metadata
metrics_dir = os.path.join(os.path.dirname(__file__), "outputs", "metrics")
comp_path = os.path.join(metrics_dir, "model_comparison.csv")
eval_path = os.path.join(metrics_dir, "detailed_evaluation.json")

best_model = "LightGBM"
best_mae = 13.68
best_rmse = 22.25
best_r2 = 0.9553

if os.path.exists(comp_path):
    comp_df = pd.read_csv(comp_path)
    best_row = comp_df.sort_values("MAE").iloc[0]
    best_model = str(best_row["Model"])
    best_mae = float(best_row["MAE"])
    best_rmse = float(best_row["RMSE"])
    best_r2 = float(best_row["R2"])

# Top Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #3182CE;">
        <div class="card-title">Top Performing Model</div>
        <div class="card-value">{best_model}</div>
        <div class="card-caption">Gradient Boosted Decision Trees</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #38A169;">
        <div class="card-title">Test MAE (Next-Hour)</div>
        <div class="card-value">{best_mae:.2f}</div>
        <div class="card-caption">AQI Points (Ground Truth 0-500)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #805AD5;">
        <div class="card-title">Test RMSE</div>
        <div class="card-value">{best_rmse:.2f}</div>
        <div class="card-caption">Strict Chronological Hold-out</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #DD6B20;">
        <div class="card-title">Variance Explained (R²)</div>
        <div class="card-value">{best_r2:.4f}</div>
        <div class="card-caption">Outperforms Naive Persistence</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Architectural Framework & Workflow
st.subheader("System Architecture & Processing Pipeline")
st.markdown("""
AIRWISE bridges advanced machine learning regression with clinical public health utility through **three foundational pillars**:
1. **Mathematical Rigor & Zero Data Leakage:** Implements continuous US EPA 40 CFR Part 58 breakpoints. All median/mode imputations, scaling parameters, and target horizons are strictly computed on training partitions.
2. **Multi-Modal Ensemble Benchmarking:** Compares 5 foundational algorithms (Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost) plus naive persistence baselines and feature ablation studies.
3. **Unified Game-Theoretic SHAP Explainability:** Translates complex tree ensembles into real-time local waterfall breakdowns and accessible natural-language public advisories.
""")

# Pipeline Columns
pcol1, pcol2, pcol3, pcol4 = st.columns(4)
with pcol1:
    st.markdown("### 1. Ingestion & Audit")
    st.info("**UCI Beijing Dataset**\n- 35,064 hourly records\n- 6 criteria air pollutants\n- 5 weather variables\n- Wind direction vectors")

with pcol2:
    st.markdown("### 2. Feature Pipeline")
    st.success("**Temporal & Physical**\n- Shifted target: $AQI_{t+1}$\n- Cyclical sine/cosine time\n- Multi-scale rolling averages\n- 80/10/10 chronological split")

with pcol3:
    st.markdown("### 3. Model Training")
    st.warning("**5 ML Ensembles**\n- Linear Regression\n- Random Forest\n- XGBoost\n- LightGBM (Champion)\n- CatBoost")

with pcol4:
    st.markdown("### 4. Explainability")
    st.error("**SHAP & Advisories**\n- Global beeswarm plots\n- Local waterfall breakdowns\n- Automated natural language\n- Clinical public health warnings")

st.markdown("---")

# Quick Navigation Guide
st.subheader("Explore Application Pages")
ncol1, ncol2, ncol3 = st.columns(3)

with ncol1:
    st.markdown("#### 📊 [1. Live Dashboard](Dashboard)")
    st.write("Examine real-time atmospheric KPI cards, multi-pollutant historical trends, and AQI distribution across Beijing seasons.")

    st.markdown("#### 🔬 [2. AQI Prediction](AQI_Prediction)")
    st.write("Forecast next-hour AQI interactively using real or custom sensor inputs with instant on-the-fly local SHAP reasoning.")

with ncol2:
    st.markdown("#### 📈 [3. Model Performance](Model_Performance)")
    st.write("Compare all 5 algorithms, analyze ablation study results (Pollution Only vs Multi-modal), and inspect residual diagnostics.")

    st.markdown("#### 🧠 [4. SHAP Explainability](SHAP_Explainability)")
    st.write("Deep dive into global feature importance beeswarms, bar rankings, and comparative clean vs hazardous case studies.")

with ncol3:
    st.markdown("#### 🗄️ [5. Data Explorer](Data_Explorer)")
    st.write("Inspect raw records, missing value audits, correlation heatmaps, and distribution histograms.")

    st.markdown("#### ℹ️ [6. About the Project](About_Project)")
    st.write("Learn about the project methodology, academic context, VIT team credentials, and GitHub repository links.")
