"""
AIRWISE Data Explorer Page
==========================
Interactive exploration of the UCI Beijing Multi-Site Air Quality Dataset
for Aotizhongxin monitoring station (35,064 hourly records).
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="AIRWISE — Data Explorer", page_icon="🗄️", layout="wide")

st.title("🗄️ Data Explorer & Atmospheric Telemetry Audit")
st.caption("Inspect raw and processed records, audit missing values, and analyze multi-variable correlations")

@st.cache_data
def load_dataset():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features_aotizhongxin.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

df = load_dataset()

if df is None:
    st.error("Processed features dataset not found.")
    st.stop()

# 1. Dataset Preview
st.subheader("1. Processed Dataset Preview (35,063 Valid Rows × 34 Columns)")
st.dataframe(df.head(100), use_container_width=True)

st.markdown("---")

# 2. Summary Statistics & Audit
st.subheader("2. Parametric Summary Statistics")
num_cols = ["PM2.5_lag1", "PM10_lag1", "SO2_lag1", "NO2_lag1", "CO_lag1", "O3_lag1", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "target_aqi_next_hour"]
st.dataframe(df[num_cols].describe().T.style.format("{:.2f}"), use_container_width=True)

st.markdown("---")

# 3. Missing Value Audit
st.subheader("3. Missing Value Audit (Prior to Median/Mode Imputation)")
missing_csv = os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics", "missing_values_summary.csv")

if os.path.exists(missing_csv):
    miss_df = pd.read_csv(missing_csv)
    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        st.dataframe(miss_df, use_container_width=True)
    with m_col2:
        fig_miss = px.bar(
            miss_df,
            x="Feature",
            y="Missing_Percentage",
            color="Missing_Percentage",
            color_continuous_scale="Reds",
            title="Percentage Missing Across Raw Features",
            labels={"Missing_Percentage": "Missing (%)"}
        )
        fig_miss.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig_miss, use_container_width=True)

st.markdown("---")

# 4. Correlation Matrix
st.subheader("4. Atmospheric Correlation Heatmap")
corr_cols = ["PM2.5_lag1", "PM10_lag1", "SO2_lag1", "NO2_lag1", "CO_lag1", "O3_lag1", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "target_aqi_next_hour"]
corr_matrix = df[corr_cols].corr()

fig_corr = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title="Pearson Correlation Matrix: Criteria Pollutants, Weather Telemetry & Target AQI",
    aspect="auto",
    zmin=-1,
    zmax=1
)
fig_corr.update_layout(height=550)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# 5. Distribution Visualizer
st.subheader("5. Feature Distribution Visualizer")
selected_feat = st.selectbox("Select Feature to Plot Distribution", options=num_cols, index=0)
fig_dist = px.histogram(
    df,
    x=selected_feat,
    nbins=50,
    marginal="box",
    color_discrete_sequence=["#1A365D"],
    title=f"Empirical Distribution: {selected_feat}"
)
fig_dist.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig_dist, use_container_width=True)
