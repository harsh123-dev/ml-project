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
        data = pd.read_csv(csv_path)
        data["datetime"] = pd.to_datetime(data["datetime"])
        return data
    return None

df = load_dataset()

if df is None:
    st.error("Processed features dataset not found.")
    st.stop()

# Interactive Filters (Section 13 Page 5 requirement)
st.sidebar.header("🔍 Dataset Filters")

min_date = df["datetime"].min().date()
max_date = df["datetime"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
    mask = (df["datetime"].dt.date >= start_d) & (df["datetime"].dt.date <= end_d)
    filtered_df = df[mask].copy()
else:
    filtered_df = df.copy()

st.sidebar.caption(f"Showing **{len(filtered_df):,}** of **{len(df):,}** records")

if filtered_df.empty:
    st.warning("No atmospheric records match the selected date filter. Please broaden the date window.")
    st.stop()

# 1. Dataset Preview
st.subheader(f"1. Processed Telemetry Records ({len(filtered_df):,} Selected Observations)")
st.dataframe(filtered_df.head(100), use_container_width=True)

st.markdown("---")

# 2. Summary Statistics & Audit
st.subheader("2. Parametric Descriptive Statistics")
num_cols = ["PM2.5_lag1", "PM10_lag1", "SO2_lag1", "NO2_lag1", "CO_lag1", "O3_lag1", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "AQI", "AQI_target"]
st.dataframe(filtered_df[num_cols].describe().T.style.format("{:.2f}"), use_container_width=True)

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
            miss_df[miss_df["MissingPct"] > 0],
            x="Column",
            y="MissingPct",
            color="MissingPct",
            color_continuous_scale="Reds",
            title="Percentage Missing Across Raw Features (Prior to Imputation)",
            labels={"MissingPct": "Missing (%)", "Column": "Feature"}
        )
        fig_miss.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig_miss, use_container_width=True)

st.markdown("---")

# 4. Correlation Matrix
st.subheader("4. Atmospheric Telemetry Correlation Heatmap")
corr_cols = ["PM2.5_lag1", "PM10_lag1", "SO2_lag1", "NO2_lag1", "CO_lag1", "O3_lag1", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "AQI", "AQI_target"]
corr_matrix = filtered_df[corr_cols].corr()

fig_corr = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title="Pearson Correlation: Criteria Pollutants, Meteorology & Target AQI",
    aspect="auto",
    zmin=-1,
    zmax=1
)
fig_corr.update_layout(height=520)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# 5. Seasonal Patterns and Diurnal Trends
st.subheader("5. Seasonal & Diurnal Atmospheric Dynamics")
s_col1, s_col2 = st.columns(2)

with s_col1:
    monthly_aqi = filtered_df.groupby("month")["AQI"].mean().reset_index()
    fig_month = px.bar(
        monthly_aqi,
        x="month",
        y="AQI",
        color="AQI",
        color_continuous_scale="Viridis",
        title="Mean AQI by Month (Winter Inversion Peak vs Summer Cleansing)",
        labels={"month": "Month of Year", "AQI": "Average AQI Sub-Index"}
    )
    fig_month.update_layout(template="plotly_white", height=360)
    st.plotly_chart(fig_month, use_container_width=True)

with s_col2:
    hourly_aqi = filtered_df.groupby("hour")["AQI"].mean().reset_index()
    fig_hour = px.line(
        hourly_aqi,
        x="hour",
        y="AQI",
        markers=True,
        title="Diurnal AQI Curve by Hour of Day",
        labels={"hour": "Hour of Day (0–23)", "AQI": "Average AQI Sub-Index"}
    )
    fig_hour.update_layout(template="plotly_white", height=360)
    st.plotly_chart(fig_hour, use_container_width=True)

st.markdown("---")

# 6. Distribution Visualizer
st.subheader("6. Feature Distribution Visualizer")
selected_feat = st.selectbox("Select Variable to Plot Distribution", options=num_cols, index=0)
fig_dist = px.histogram(
    filtered_df,
    x=selected_feat,
    nbins=50,
    marginal="box",
    color_discrete_sequence=["#1A365D"],
    title=f"Empirical Distribution: {selected_feat}"
)
fig_dist.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig_dist, use_container_width=True)
