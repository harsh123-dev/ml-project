"""
AIRWISE Interactive AQI Prediction Page
=======================================
Enables real-time next-hour AQI inference across all 5 ensemble algorithms
with instant game-theoretic SHAP local explainability and medical health advisories.
"""

# Import lightgbm first for Windows C-runtime stability
import lightgbm as lgb
import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prediction import AirQualityPredictor, ALL_WIND_DIRECTIONS
from src.aqi_calculator import get_aqi_category

st.set_page_config(page_title="AIRWISE — AQI Prediction", page_icon="🔬", layout="wide")

st.title("🔬 Next-Hour AQI Forecasting & Local SHAP Reasoning")
st.caption("Perform single-observation inference using trained machine learning models with instant feature attribution")

@st.cache_resource
def get_predictor():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    shap_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "shap")
    return AirQualityPredictor(models_dir=models_dir, shap_dir=shap_dir)

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Failed to initialize predictor: {e}")
    st.stop()

# Sidebar Settings
st.sidebar.header("Inference Settings")
selected_model = st.sidebar.selectbox(
    "Select ML Algorithm",
    options=["LightGBM", "XGBoost", "Random Forest", "CatBoost", "Linear Regression"],
    index=0,
    help="LightGBM is the champion model achieving test MAE 13.68."
)

selected_standard = st.sidebar.radio(
    "AQI Standard",
    options=["US_EPA", "CPCB"],
    index=0,
    help="US EPA 40 CFR Part 58 (Default) or CPCB India Standard."
)

# Preset buttons
st.subheader("1. Atmospheric & Telemetry Inputs")
preset_col1, preset_col2, preset_col3 = st.columns(3)

preset_mode = None
if preset_col1.button("🌿 Preset: Clean Summer Air"):
    st.session_state["pm25"] = 14.0
    st.session_state["pm10"] = 28.0
    st.session_state["so2"] = 3.0
    st.session_state["no2"] = 18.0
    st.session_state["co"] = 400.0
    st.session_state["o3"] = 65.0
    st.session_state["temp"] = 28.5
    st.session_state["pres"] = 1005.0
    st.session_state["dewp"] = 16.0
    st.session_state["rain"] = 0.0
    st.session_state["wspm"] = 4.5
    st.session_state["wd"] = "NW"
    st.session_state["hour"] = 14
    st.session_state["month"] = 7

if preset_col2.button("🔄 Preset: Typical Autumn Day (Moderate)"):
    st.session_state["pm25"] = 78.0
    st.session_state["pm10"] = 112.0
    st.session_state["so2"] = 15.0
    st.session_state["no2"] = 52.0
    st.session_state["co"] = 1200.0
    st.session_state["o3"] = 42.0
    st.session_state["temp"] = 14.0
    st.session_state["pres"] = 1018.0
    st.session_state["dewp"] = 0.0
    st.session_state["rain"] = 0.0
    st.session_state["wspm"] = 1.6
    st.session_state["wd"] = "NE"
    st.session_state["hour"] = 10
    st.session_state["month"] = 10

if preset_col3.button("⚠️ Preset: Severe Winter Smog Event"):
    st.session_state["pm25"] = 285.0
    st.session_state["pm10"] = 340.0
    st.session_state["so2"] = 45.0
    st.session_state["no2"] = 125.0
    st.session_state["co"] = 3800.0
    st.session_state["o3"] = 12.0
    st.session_state["temp"] = -4.0
    st.session_state["pres"] = 1028.0
    st.session_state["dewp"] = -11.0
    st.session_state["rain"] = 0.0
    st.session_state["wspm"] = 0.8
    st.session_state["wd"] = "SSE"
    st.session_state["hour"] = 21
    st.session_state["month"] = 1

# Input Form
with st.form("prediction_form"):
    tab1, tab2, tab3 = st.tabs(["🧪 Air Pollutants", "🌤️ Weather Variables", "🕒 Temporal Context"])

    with tab1:
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=1000.0, value=st.session_state.get("pm25", 85.0), step=1.0)
            pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=1500.0, value=st.session_state.get("pm10", 115.0), step=1.0)
        with pcol2:
            so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, max_value=1000.0, value=st.session_state.get("so2", 18.0), step=1.0)
            no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, max_value=1000.0, value=st.session_state.get("no2", 54.0), step=1.0)
        with pcol3:
            co = st.number_input("CO (µg/m³)", min_value=0.0, max_value=20000.0, value=st.session_state.get("co", 1400.0), step=50.0)
            o3 = st.number_input("O3 (µg/m³)", min_value=0.0, max_value=1000.0, value=st.session_state.get("o3", 45.0), step=1.0)

    with tab2:
        wcol1, wcol2, wcol3 = st.columns(3)
        with wcol1:
            temp = st.number_input("Temperature (°C)", min_value=-40.0, max_value=55.0, value=st.session_state.get("temp", 12.5), step=0.5)
            pres = st.number_input("Barometric Pressure (hPa)", min_value=950.0, max_value=1050.0, value=st.session_state.get("pres", 1018.0), step=0.5)
        with wcol2:
            dewp = st.number_input("Dew Point (°C)", min_value=-50.0, max_value=40.0, value=st.session_state.get("dewp", -2.0), step=0.5)
            rain = st.number_input("Precipitation / Rain (mm)", min_value=0.0, max_value=100.0, value=st.session_state.get("rain", 0.0), step=0.1)
        with wcol3:
            wspm = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=st.session_state.get("wspm", 1.8), step=0.2)
            wd = st.selectbox("Prevailing Wind Direction", options=ALL_WIND_DIRECTIONS, index=ALL_WIND_DIRECTIONS.index(st.session_state.get("wd", "NE")))

    with tab3:
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=st.session_state.get("hour", 14))
        with tcol2:
            month = st.slider("Month of Year (1-12)", min_value=1, max_value=12, value=st.session_state.get("month", 11))
        with tcol3:
            day_of_week = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x], index=2)

    submit_btn = st.form_submit_button("🚀 Run Forecast & Compute SHAP Reasoning", use_container_width=True)

if submit_btn:
    raw_obs = {
        "PM2.5": pm25, "PM10": pm10, "SO2": so2, "NO2": no2, "CO": co, "O3": o3,
        "TEMP": temp, "PRES": pres, "DEWP": dewp, "RAIN": rain, "WSPM": wspm,
        "wd": wd, "hour": hour, "month": month, "day_of_week": day_of_week
    }

    with st.spinner("Generating prediction and game-theoretic SHAP explanations..."):
        res = predictor.predict(
            raw_obs,
            model_name=selected_model,
            include_shap=True,
            standard=selected_standard
        )

    st.markdown("---")
    st.subheader("2. Forecast Results & Public Health Indicator")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.markdown(f"""
        <div style="background-color: {res['color']}22; border-left: 8px solid {res['color']}; padding: 1.5rem; border-radius: 10px;">
            <div style="font-size: 0.9rem; text-transform: uppercase; color: #4A5568; font-weight: 700;">Forecasted Next-Hour AQI</div>
            <div style="font-size: 3.2rem; font-weight: 900; color: {res['color']}; line-height: 1.1;">{res['predicted_aqi']}</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #2D3748; margin-top: 0.5rem;">{res['category']}</div>
            <div style="font-size: 0.85rem; color: #718096; margin-top: 0.3rem;">Standard: {res['standard']} | Algorithm: {res['model_used']}</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"### Health Advisory ({res['category']})")
        st.info(f"**Medical Guidance:** {res['advisory']}")
        st.caption("Target definition: Next-hour PM2.5-derived Air Quality Index ($AQI_{t+1}$).")

    # Local SHAP Explainability
    if "explanation" in res and "top_contributions" in res["explanation"]:
        st.markdown("---")
        st.subheader("3. Transparent SHAP Reasoning & Attribution")

        expl = res["explanation"]
        st.markdown(expl["narrative"])

        # Waterfall / Contribution Bar Chart
        contribs = expl["top_contributions"]
        c_df = pd.DataFrame(contribs)
        c_df["Impact"] = c_df["shap"].apply(lambda s: "Exacerbating (+)" if s > 0 else "Mitigating (-)")
        c_df["Color"] = c_df["shap"].apply(lambda s: "#E53E3E" if s > 0 else "#3182CE")

        fig_contrib = px.bar(
            c_df,
            x="shap",
            y="feature",
            orientation="h",
            color="Impact",
            color_discrete_map={"Exacerbating (+)": "#E53E3E", "Mitigating (-)": "#3182CE"},
            labels={"shap": "SHAP Impact (AQI Points)", "feature": "Atmospheric / Weather Feature"},
            title="Local Feature Attribution: Top Forces Pushing AQI Higher vs Lower"
        )
        fig_contrib.update_layout(template="plotly_white", height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_contrib, use_container_width=True)
