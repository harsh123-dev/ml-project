"""
AIRWISE SHAP Explainability & Transparency Page
===============================================
Deep dive into game-theoretic Shapley values, global feature attribution,
and comparative case studies explaining why the machine learning models make their forecasts.
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="AIRWISE — SHAP Explainability", page_icon="🧠", layout="wide")

st.title("🧠 SHAP Explainability & Game-Theoretic Transparency")
st.caption("Interpreting non-linear tree ensemble decisions via Shapley Additive exPlanations (Lundberg & Lee, 2017)")

shap_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "shap")
summary_json_path = os.path.join(shap_dir, "shap_summary.json")
ranking_csv_path = os.path.join(shap_dir, "shap_importance_ranking.csv")
beeswarm_img = os.path.join(shap_dir, "shap_summary_beeswarm.png")
bar_img = os.path.join(shap_dir, "shap_summary_bar.png")
waterfall1_img = os.path.join(shap_dir, "shap_waterfall_case1.png")
waterfall2_img = os.path.join(shap_dir, "shap_waterfall_case2.png")

# Educational Banner
st.info("""
**What is SHAP?**
Rooted in cooperative game theory (Shapley, 1953), SHAP calculates the exact marginal contribution of each atmospheric feature toward shifting the prediction away from the baseline average.
- **Positive SHAP (+):** Feature drove next-hour AQI higher (exacerbating pollution).
- **Negative SHAP (-):** Feature lowered next-hour AQI (mitigating or cleansing the air).
""")

# 1. Global Feature Explanations
st.subheader("1. Global Explainability Across the Entire Test Partition")

tab_bee, tab_bar, tab_rank = st.tabs(["🐝 SHAP Beeswarm Plot", "📊 Mean |SHAP| Bar Plot", "📋 Importance Ranking Table"])

with tab_bee:
    if os.path.exists(beeswarm_img):
        st.image(beeswarm_img, caption="SHAP Beeswarm: Dot position indicates impact (+/- AQI points); color represents raw feature value (Red = High, Blue = Low).", use_container_width=True)
    else:
        st.warning("Beeswarm image not found.")

with tab_bar:
    if os.path.exists(bar_img):
        st.image(bar_img, caption="Global Feature Importance: Mean Absolute SHAP Value across test samples.", use_container_width=True)
    else:
        st.warning("Bar summary image not found.")

with tab_rank:
    if os.path.exists(ranking_csv_path):
        rank_df = pd.read_csv(ranking_csv_path)
        st.dataframe(rank_df, use_container_width=True)

st.markdown("---")

# 2. Comparative Episode Case Studies
st.subheader("2. Local Case Studies: Extreme Smog Episode vs Pristine Air Episode")

case_data = None
if os.path.exists(summary_json_path):
    with open(summary_json_path, "r") as f:
        case_data = json.load(f).get("case_studies", {})

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("### ⚠️ Severe Pollution Episode (Hazardous AQI)")
    if os.path.exists(waterfall1_img):
        st.image(waterfall1_img, use_container_width=True)
    if case_data and "high_pollution_case" in case_data:
        hp = case_data["high_pollution_case"]
        st.markdown(f"""
        **Forecasted AQI:** `{hp['prediction']:.1f}` ({hp['category']})
        **Baseline Expected AQI:** `{hp['base_value']:.1f}`
        """)
        st.success(hp["narrative"])

with col_c2:
    st.markdown("### 🌿 Clean Air Episode (Good AQI)")
    if os.path.exists(waterfall2_img):
        st.image(waterfall2_img, use_container_width=True)
    if case_data and "clean_air_case" in case_data:
        ca = case_data["clean_air_case"]
        st.markdown(f"""
        **Forecasted AQI:** `{ca['prediction']:.1f}` ({ca['category']})
        **Baseline Expected AQI:** `{ca['base_value']:.1f}`
        """)
        st.info(ca["narrative"])

st.markdown("---")

# 3. Domain Atmospheric Interpretations
st.subheader("3. Atmospheric & Physical Interpretations of Model Decisions")

exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    st.markdown("#### 1. Particulate Inertia ($PM_{2.5}$ Lags)")
    st.write("""
    $PM_{2.5}$ lag-1 is consistently the dominant driver. Fine aerosols exhibit high atmospheric residence time (days to weeks) and low dry deposition velocity, making current airborne loading the primary determinant of next-hour concentrations.
    """)

with exp_col2:
    st.markdown("#### 2. Wind Ventilation Effect ($WSPM$)")
    st.write("""
    Wind speed consistently exhibits strong negative SHAP values. High wind velocities generate mechanical turbulence and advection, dispersing particulate accumulation out of the Beijing mountain-valley basin.
    """)

with exp_col3:
    st.markdown("#### 3. Thermal Inversion & Pressure")
    st.write("""
    High barometric pressure and low temperatures in winter trap stagnant air masses near the urban surface, suppressing convective boundary layer mixing and driving intense smog spikes.
    """)
