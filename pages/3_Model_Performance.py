"""
AIRWISE Model Performance & Benchmark Page
==========================================
Comprehensive multi-model comparative analytics, ablation experiments,
and interactive residual diagnostics on chronologically held-out test data.
"""

# Import lightgbm first for Windows C-runtime stability
import lightgbm as lgb
import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.visualization import (
    build_plotly_comparison,
    build_plotly_timeseries,
    build_plotly_scatter
)

st.set_page_config(page_title="AIRWISE — Model Performance", page_icon="📈", layout="wide")

st.title("📈 Model Performance & Comparative Benchmark")
st.caption("Rigorous evaluation on 3,507 chronologically unseen hourly records (October 2016 – February 2017)")

metrics_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics")
preds_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "predictions", "test_predictions.csv")
comp_path = os.path.join(metrics_dir, "model_comparison.csv")
det_metrics_path = os.path.join(metrics_dir, "detailed_model_metrics.csv")
ablation_path = os.path.join(metrics_dir, "ablation_results.csv")
cat_path = os.path.join(metrics_dir, "category_error_breakdown.csv")

# 1. Comparison Table
st.subheader("1. Multi-Model Benchmark Table")

if os.path.exists(det_metrics_path):
    det_df = pd.read_csv(det_metrics_path)
    st.dataframe(
        det_df.style.highlight_min(subset=["MAE", "RMSE", "MAPE (%)", "Median AE"], color="#C6F6D5")
              .highlight_max(subset=["R2", "Explained Variance"], color="#C6F6D5"),
        use_container_width=True
    )
elif os.path.exists(comp_path):
    comp_df = pd.read_csv(comp_path)
    st.dataframe(comp_df, use_container_width=True)
else:
    st.warning("Performance tables not found.")

st.markdown("---")

# 2. Interactive Metric Charts
st.subheader("2. Comparative Performance Visualizer")
col_metric, col_chart = st.columns([1, 3])

if os.path.exists(comp_path):
    comp_df = pd.read_csv(comp_path)
    with col_metric:
        selected_metric = st.selectbox(
            "Select Evaluation Metric",
            options=["MAE", "RMSE", "R2", "Training Time (s)"],
            index=0
        )
        st.info(f"""
        **Metric Definition:**
        - **MAE:** Mean Absolute Error (average magnitude of forecast discrepancy in AQI points).
        - **RMSE:** Root Mean Squared Error (penalizes large severe episode misses).
        - **R²:** Proportion of true variance explained by the model (1.0 = perfect).
        """)

    with col_chart:
        fig_comp = build_plotly_comparison(comp_df, metric=selected_metric)
        st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

# 3. Ablation Study
st.subheader("3. Ablation Experiment: Meteorological Integration")
st.markdown("""
To determine the empirical value of fusing meteorological and temporal telemetry with raw pollution readings,
an ablation experiment was conducted comparing **Experiment A (Pollution Only)** against **Experiment B (Multi-Modal Full Pipeline)**:
""")

if os.path.exists(ablation_path):
    abl_df = pd.read_csv(ablation_path)
    abl_col1, abl_col2 = st.columns([1, 1])

    with abl_col1:
        st.dataframe(abl_df, use_container_width=True)
        # Calculate improvement
        p_mae = abl_df.iloc[0]["MAE"]
        f_mae = abl_df.iloc[1]["MAE"]
        p_rmse = abl_df.iloc[0]["RMSE"]
        f_rmse = abl_df.iloc[1]["RMSE"]
        mae_gain = (p_mae - f_mae) / p_mae * 100
        rmse_gain = (p_rmse - f_rmse) / p_rmse * 100

        st.success(f"""
        **Ablation Key Findings:**
        - Adding weather, time, and wind vectors reduced MAE by **{mae_gain:.2f}%** (from {p_mae:.2f} to {f_mae:.2f}).
        - RMSE was reduced by **{rmse_gain:.2f}%** (from {p_rmse:.2f} to {f_rmse:.2f}).
        - Confirms that atmospheric variables (wind dispersion, temperature inversions) significantly modulate urban pollution.
        """)

    with abl_col2:
        fig_abl = go.Figure()
        fig_abl.add_trace(go.Bar(name="MAE", x=abl_df["Feature Configuration"], y=abl_df["MAE"], marker_color="#2B6CB0"))
        fig_abl.add_trace(go.Bar(name="RMSE", x=abl_df["Feature Configuration"], y=abl_df["RMSE"], marker_color="#E53E3E"))
        fig_abl.update_layout(
            barmode="group",
            title="Ablation Comparison: Error Reduction via Weather & Wind Vectors",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig_abl, use_container_width=True)

st.markdown("---")

# 4. Interactive Test Tracking & Residual Diagnostics
st.subheader("4. Time-Series Tracking & Residual Diagnostics")

if os.path.exists(preds_path):
    preds_df = pd.read_csv(preds_path)
    models_available = [c for c in preds_df.columns if c not in ["Actual_AQI", "datetime"]]

    t_col1, t_col2 = st.columns([1, 2])
    with t_col1:
        chosen_model = st.selectbox("Inspect Model Predictions", options=models_available, index=0)
        start_hour = st.slider("Window Start Hour", min_value=0, max_value=len(preds_df) - 168, value=500, step=24)
        window_len = st.select_slider("Window Length (Hours)", options=[48, 72, 168, 336, 720], value=168)

    with t_col2:
        fig_ts = build_plotly_timeseries(preds_df, model_name=chosen_model, start=start_hour, length=window_len)
        st.plotly_chart(fig_ts, use_container_width=True)

    # Scatter and Residual Plots
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        fig_scat = build_plotly_scatter(preds_df, chosen_model)
        st.plotly_chart(fig_scat, use_container_width=True)

    with d_col2:
        # Residual Plot
        res_vals = preds_df["Actual_AQI"] - preds_df[chosen_model]
        fig_res = px.scatter(
            x=preds_df[chosen_model],
            y=res_vals,
            opacity=0.35,
            color_discrete_sequence=["#D69E2E"],
            labels={"x": f"Predicted AQI ({chosen_model})", "y": "Residual (Actual - Predicted)"},
            title=f"Residual Diagnostics: {chosen_model}"
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="#E53E3E")
        fig_res.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_res, use_container_width=True)

# 5. Category-Level Performance Breakdown
if os.path.exists(cat_path):
    st.markdown("---")
    st.subheader("5. Performance Breakdown Across AQI Severity Classes")
    c_df = pd.read_csv(cat_path)
    st.dataframe(c_df, use_container_width=True)
