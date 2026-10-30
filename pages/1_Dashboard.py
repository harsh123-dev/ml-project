"""
AIRWISE Dashboard Page
======================
Live atmospheric telemetry, historical trends, pollutant tracking,
and seasonal AQI category distributions for Aotizhongxin station.
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.aqi_calculator import get_aqi_category

st.set_page_config(page_title="AIRWISE — Live Dashboard", page_icon="📊", layout="wide")

st.title("📊 Atmospheric Dashboard & Trends")
st.caption("Aotizhongxin Monitoring Station, Beijing | UCI Multi-Site Air Quality Dataset")

@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features_aotizhongxin.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    return None

df = load_data()

if df is None:
    st.error("Processed features dataset not found. Please run feature engineering pipeline.")
    st.stop()

# Latest Observation KPIs
latest = df.iloc[-1]
aqi_cat = get_aqi_category(latest["target_aqi_next_hour"])

st.subheader("Latest Recorded Atmospheric Telemetry")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        label="Observed AQI (PM2.5)",
        value=f"{latest['target_aqi_next_hour']:.1f}",
        delta=f"{aqi_cat['category']}",
        delta_color="off"
    )
with kpi2:
    st.metric(label="PM2.5 Concentration", value=f"{latest['PM2.5_lag1']:.1f} µg/m³")
with kpi3:
    st.metric(label="PM10 Concentration", value=f"{latest['PM10_lag1']:.1f} µg/m³")
with kpi4:
    st.metric(label="Ambient Temperature", value=f"{latest['TEMP']:.1f} °C")
with kpi5:
    st.metric(label="Wind Speed", value=f"{latest['WSPM']:.1f} m/s")

st.markdown("---")

# Historical Trend Exploration
st.subheader("Hourly Air Quality & Pollutant Trajectories")

col_ctrl1, col_ctrl2 = st.columns([1, 3])
with col_ctrl1:
    year_selected = st.selectbox("Select Year", options=[2013, 2014, 2015, 2016, 2017], index=3)
    month_selected = st.slider("Select Month", min_value=1, max_value=12, value=11)

filtered_df = df[(df["datetime"].dt.year == year_selected) & (df["datetime"].dt.month == month_selected)]

if filtered_df.empty:
    st.warning("No records available for the selected period.")
else:
    # 1. AQI Time Series with Severity Thresholds
    fig_aqi = go.Figure()
    fig_aqi.add_trace(go.Scatter(
        x=filtered_df["datetime"],
        y=filtered_df["target_aqi_next_hour"],
        mode="lines",
        name="Hourly AQI",
        line=dict(color="#1A365D", width=2.2)
    ))
    # Add horizontal reference bands for US EPA levels
    fig_aqi.add_hrect(y0=0, y1=50, fillcolor="#00E400", opacity=0.1, line_width=0, annotation_text="Good (0-50)")
    fig_aqi.add_hrect(y0=50, y1=100, fillcolor="#FFFF00", opacity=0.1, line_width=0, annotation_text="Moderate (51-100)")
    fig_aqi.add_hrect(y0=100, y1=150, fillcolor="#FF7E00", opacity=0.1, line_width=0, annotation_text="USG (101-150)")
    fig_aqi.add_hrect(y0=150, y1=200, fillcolor="#FF0000", opacity=0.1, line_width=0, annotation_text="Unhealthy (151-200)")
    fig_aqi.add_hrect(y0=200, y1=300, fillcolor="#8F3F97", opacity=0.1, line_width=0, annotation_text="Very Unhealthy (201-300)")
    fig_aqi.add_hrect(y0=300, y1=500, fillcolor="#7E0023", opacity=0.15, line_width=0, annotation_text="Hazardous (301-500)")

    fig_aqi.update_layout(
        title=f"Hourly AQI Evolution ({year_selected}-{month_selected:02d}) with EPA Standard Severity Bands",
        xaxis_title="Timestamp",
        yaxis_title="AQI Sub-Index",
        template="plotly_white",
        hovermode="x unified",
        height=420
    )
    st.plotly_chart(fig_aqi, use_container_width=True)

    # 2. Multi-Pollutant Comparison
    st.subheader("Multi-Pollutant Trajectories")
    pollutants_selected = st.multiselect(
        "Select Pollutants to Overlay",
        options=["PM2.5_lag1", "PM10_lag1", "SO2_lag1", "NO2_lag1", "CO_lag1", "O3_lag1"],
        default=["PM2.5_lag1", "PM10_lag1", "NO2_lag1"]
    )

    if pollutants_selected:
        fig_poll = go.Figure()
        for p in pollutants_selected:
            clean_name = p.replace("_lag1", "")
            fig_poll.add_trace(go.Scatter(
                x=filtered_df["datetime"],
                y=filtered_df[p],
                mode="lines",
                name=f"{clean_name} (µg/m³)"
            ))
        fig_poll.update_layout(
            title=f"Atmospheric Pollutant Dynamics ({year_selected}-{month_selected:02d})",
            xaxis_title="Timestamp",
            yaxis_title="Concentration (µg/m³)",
            template="plotly_white",
            hovermode="x unified",
            height=380
        )
        st.plotly_chart(fig_poll, use_container_width=True)

st.markdown("---")

# Seasonal AQI Category Breakdown
st.subheader("Longitudinal AQI Category Distribution (Full 4-Year Record)")
col_dist1, col_dist2 = st.columns(2)

with col_dist1:
    cat_counts = df["target_aqi_next_hour"].apply(lambda v: get_aqi_category(v)["category"]).value_counts()
    cat_df = pd.DataFrame({"Category": cat_counts.index, "Hours": cat_counts.values})
    color_map = {
        "Good": "#00E400",
        "Moderate": "#FFFF00",
        "Unhealthy for Sensitive Groups": "#FF7E00",
        "Unhealthy": "#FF0000",
        "Very Unhealthy": "#8F3F97",
        "Hazardous": "#7E0023"
    }
    fig_pie = px.pie(
        cat_df,
        names="Category",
        values="Hours",
        title="Distribution of EPA AQI Categories (35,064 Hours)",
        color="Category",
        color_discrete_map=color_map,
        hole=0.4
    )
    fig_pie.update_layout(template="plotly_white")
    st.plotly_chart(fig_pie, use_container_width=True)

with col_dist2:
    # Temperature vs AQI Scatter
    fig_weather = px.scatter(
        df.sample(2000, random_state=42),
        x="TEMP",
        y="target_aqi_next_hour",
        color="WSPM",
        color_continuous_scale="Blues",
        labels={"TEMP": "Temperature (°C)", "target_aqi_next_hour": "AQI Sub-Index", "WSPM": "Wind Speed (m/s)"},
        title="Meteorological Interaction: Temp & Wind Speed vs AQI"
    )
    fig_weather.update_layout(template="plotly_white")
    st.plotly_chart(fig_weather, use_container_width=True)
