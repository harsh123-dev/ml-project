"""
AIRWISE About & Academic Documentation Page
===========================================
Research overview, academic credentials, dataset provenance,
standards compliance, and open-source licensing.
"""

import streamlit as st

st.set_page_config(page_title="AIRWISE — About the Project", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About the AIRWISE Project")
st.caption("Explainable Air Quality Index Prediction Using Ensemble Machine Learning and Weather Data")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("1. Project Abstract & Objectives")
    st.markdown("""
    Rapid urban industrialization and complex boundary-layer meteorology make short-term air quality forecasting a critical public health necessity. Conventional numerical dispersion models require intense supercomputing resources and struggle with high-frequency local atmospheric fluctuations.

    **AIRWISE** addresses these challenges by developing a production-grade, highly interpretable machine learning system for **next-hour Air Quality Index ($AQI_{t+1}$)** prediction. Fusing historical criteria air pollutants ($PM_{2.5}$, $PM_{10}$, $SO_2$, $NO_2$, $CO$, $O_3$) with meteorological observations (temperature, atmospheric pressure, dew point, rainfall, wind speed, and cyclical wind vectors), the system benchmarks 5 distinct regression algorithms and unlocks decision-making transparency via **game-theoretic SHAP explainability**.
    """)

    st.subheader("2. Regulatory Standards Compliance")
    st.markdown("""
    - **Primary Standard:** **United States Environmental Protection Agency (US EPA)**
      - *Citation:* 40 CFR Part 58, Appendix G — Technical Assistance Document for the Reporting of Daily Air Quality (AQI).
      - Continuous piecewise linear interpolation between official breakpoints $(0, 50, 100, 150, 200, 300, 500)$.
    - **Secondary Standard:** **Central Pollution Control Board (CPCB), India**
      - Ministry of Environment, Forest and Climate Change (2014 Standard).
    """)

    st.subheader("3. Dataset Provenance")
    st.markdown("""
    - **Source:** [UCI Machine Learning Repository: Beijing Multi-Site Air Quality Dataset](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
    - **Primary Station:** **Aotizhongxin** (Olympic Sports Center, Chaoyang District, Beijing).
    - **Period:** March 1, 2013 to February 28, 2017 (35,064 continuous hourly observations).
    - **Data Partitioning:** Chronological 80% Training (2013–2016), 10% Validation (2016), 10% Test (Oct 2016 – Feb 2017). Zero data leakage across splits.
    """)

with col2:
    st.subheader("Research Team")
    st.markdown("""
    **Vellore Institute of Technology**
    *Department of Data Science*
    *Academic Year 2026–27*

    - **Harsh Ranjan** — `23BDS0078`
    - **Himanshu Kumar** — `23BDS0069`
    - **Mohammad Ibrahim Ahmad** — `23BDS0073`
    """)

    st.markdown("---")
    st.subheader("Open Source Codebase")
    st.markdown("""
    - **GitHub Repository:**
      [`harsh123-dev/ml-project`](https://github.com/harsh123-dev/ml-project)
    - **License:** [MIT License](https://opensource.org/licenses/MIT)
    """)

    st.markdown("---")
    st.subheader("Disclaimer")
    st.caption("""
    AIRWISE is developed strictly for academic research and educational evaluation. While the underlying algorithms adhere to standard US EPA formulas, operational public health directives should rely on certified governmental meteorological advisories.
    """)
