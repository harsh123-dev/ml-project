# AIRWISE: Explainable Air Quality Index Prediction Using Ensemble Machine Learning and Weather Data

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Pytest-44%2F44%20Passed-brightgreen.svg)]()
[![Standard](https://img.shields.io/badge/Standard-US%20EPA%2040%20CFR%2058-orange.svg)](https://www.epa.gov/criteria-air-pollutants)
[![Explainability](https://img.shields.io/badge/XAI-SHAP%20TreeExplainer-purple.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Academic Information & Authors
- **Harsh Ranjan** — `23BDS0078`
- **Himanshu Kumar** — `23BDS0069`
- **Mohammad Ibrahim Ahmad** — `23BDS0073`
- **Department / School:** School of Computer Science & Engineering (SCOPE)
- **Institution:** Vellore Institute of Technology (VIT), Vellore
- **Academic Year:** 2026–27
- **GitHub Repository:** [`harsh123-dev/ml-project`](https://github.com/harsh123-dev/ml-project)

---

## Executive Abstract
Urban atmospheric particulate matter ($PM_{2.5}$) poses severe risks to cardiopulmonary human health and regional climate stability. While traditional numerical chemical transport models (e.g., CMAQ, WRF-Chem) require supercomputing clusters and struggle with real-time localized variations, classical machine learning models often act as opaque "black boxes" lacking clinical trust.

**AIRWISE** is an end-to-end, production-grade, highly interpretable machine learning system engineered to forecast the **next-hour Air Quality Index ($AQI_{t+1}$)** for the **Aotizhongxin monitoring station** in Beijing (UCI Machine Learning Repository; 35,064 continuous hourly observations from March 2013 through February 2017). By fusing criteria air pollutants ($PM_{2.5}$, $PM_{10}$, $SO_2$, $NO_2$, $CO$, $O_3$) with high-resolution boundary-layer weather telemetry (temperature, barometric pressure, dew point, rainfall, wind speed, and 16-point wind direction vectors), the system benchmarks 5 distinct algorithms (**Linear Regression, Random Forest, XGBoost, LightGBM, and CatBoost**) against a naive persistence baseline and validates the feature synergy through a formal ablation experiment. Full game-theoretic transparency is achieved via **SHAP (Shapley Additive exPlanations)**, producing global beeswarm distributions, local waterfall breakdowns, and automated natural-language medical health advisories.

---

## Mathematical Formulations

### 1. US EPA AQI Sub-Index Calculation (40 CFR Part 58, Appendix G)
The primary forecast target ($AQI_{t+1}$) is the mathematically rigorous PM2.5-derived sub-index:
$$I_p = \frac{I_{hi} - I_{lo}}{BP_{hi} - BP_{lo}} \times (C_p - BP_{lo}) + I_{lo}$$
where:
- $C_p$ is the pollutant concentration rounded to $0.1\ \mu\text{g/m}^3$ per EPA protocol.
- $[BP_{lo}, BP_{hi}]$ is the contiguous concentration breakpoint bracket.
- $[I_{lo}, I_{hi}]$ is the corresponding regulatory AQI category index range $(0\text{--}50, 51\text{--}100, 101\text{--}150, 151\text{--}200, 201\text{--}300, 301\text{--}500)$.

### 2. Cyclical Temporal Encodings
To preserve boundary continuity across midnight ($23\text{h} \to 0\text{h}$) and year-end ($12\text{m} \to 1\text{m}$), cyclical trigonometric projections are applied:
$$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
$$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
$$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$

### 3. SHAP Additive Feature Attribution
Using cooperative coalitional game theory (Lundberg & Lee, 2017), the complex ensemble forecast $f(x)$ is decomposed additively into marginal payoffs:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
where:
- $\phi_0 = \mathbb{E}[f(x)]$ is the historical expected baseline AQI ($143.87$ points).
- $\phi_i \in \mathbb{R}$ is the exact Shapley contribution of feature $i$ in AQI points.

---

## System Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │ UCI Beijing Multi-Site Dataset (Aotizhongxin Station)   │
                  │   35,064 records (March 1, 2013 – February 28, 2017)   │
                  └───────────────────────────┬────────────────────────────┘
                                              │ Ingestion & Audit
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ Mathematical Data Cleaning & Preprocessing             │
                  │ - Strict Chronological Split (80% Train, 10% Val, 10% Test) │
                  │ - Train-Fitted Median/Mode Imputation (Zero Data Leakage)│
                  │ - Physical Outlier Screening & Timestamp Ordering      │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ Multi-Modal Feature Engineering Pipeline                │
                  │ - Target: Shifted AQI_{t+1} (US EPA 40 CFR Part 58)    │
                  │ - Multi-scale Rolling Proxies (3h, 6h, 12h, 24h)       │
                  │ - Cyclical Sine/Cosine Time Encodings (Hour, Month, DOW)│
                  │ - 16-Compass Point Wind Direction One-Hot Encodings    │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ Model Training & Multi-Algorithm Benchmarking          │
                  │ 1. Linear Regression (Baseline)                        │
                  │ 2. Random Forest Regressor                             │
                  │ 3. XGBoost Regressor                                   │
                  │ 4. LightGBM Regressor (Champion)                       │
                  │ 5. CatBoost Regressor                                  │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
      ┌─────────────────────────────┐                  ┌───────────────────────────────┐
      │ Game-Theoretic Explainability│                  │ Interactive Streamlit System  │
      │ - SHAP TreeExplainer        │                  │ - Live Atmospheric Dashboard │
      │ - Global Beeswarm & Bar     │                  │ - Interactive Prediction & XAI│
      │ - Local Waterfall Episodes  │                  │ - Model Benchmark Visualizer  │
      │ - Natural Language Advisory │                  │ - Data Explorer & Telemetry   │
      └─────────────────────────────┘                  └───────────────────────────────┘
```

---

## Empirical Benchmark Results

### 1. Multi-Model Performance Comparison
Evaluated on **3,507 chronologically held-out test hours** (October 5, 2016 to February 28, 2017):

| Model | MAE (AQI) | RMSE (AQI) | $R^2$ Score | MAPE (%) | Median AE | Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM (Champion)** | **13.678** | **22.246** | **0.9553** | **18.36%** | **8.63** | **0.47 s** |
| **XGBoost** | 13.813 | 22.495 | 0.9543 | 18.55% | 8.66 | 0.67 s |
| **CatBoost** | 13.869 | 22.696 | 0.9534 | 18.37% | 8.63 | 0.86 s |
| **Random Forest** | 14.178 | 23.066 | 0.9519 | 18.85% | 8.85 | 6.84 s |
| **Linear Regression** | 19.140 | 26.620 | 0.9359 | 27.26% | 14.93 | 0.05 s |
| *Persistence Baseline ($AQI_t$)* | 14.444 | 24.475 | 0.9459 | 18.72% | 8.33 | — |

*Key Takeaways:*
- **LightGBM** achieves the lowest MAE (**13.678**) and highest $R^2$ (**0.9553**), training in under 0.5 seconds.
- All gradient-boosted decision tree ensembles consistently outperform the naive persistence baseline ($R^2 = 0.9459$), proving genuine physical learning rather than simple lag regurgitation.

### 2. Ablation Experiment (Meteorological Integration)
To quantify the exact empirical value of fusing meteorological observations with pollution telemetry:

| Feature Configuration | Input Dimensions | MAE | RMSE | $R^2$ Score | MAE Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Experiment A (Pollution Only)** | 11 Features | 14.366 | 23.472 | 0.9502 | *Baseline* |
| **Experiment B (Multi-Modal Full)** | 32 Features | **13.813** | **22.495** | **0.9543** | **+3.85% Gain** |

*Findings:* Integrating boundary-layer meteorology (wind speed, direction, temperature, barometric pressure) significantly improves peak-episode forecasting accuracy, reducing RMSE from 23.472 to 22.495.

### 3. Performance Across US EPA Severity Categories (LightGBM)
| Severity Category | Test Samples | Distribution | MAE (AQI) | RMSE (AQI) |
|:---|:---:|:---:|:---:|:---:|
| **Good (0–50)** | 661 | 18.85% | 12.490 | 21.474 |
| **Moderate (51–100)** | 655 | 18.68% | 13.754 | 20.875 |
| **Unhealthy for Sensitive Groups (101–150)** | 303 | 8.64% | 14.022 | 19.243 |
| **Unhealthy (151–200)** | 971 | 27.69% | **9.534** | **16.777** |
| **Very Unhealthy (201–300)** | 588 | 16.77% | 16.214 | 25.766 |
| **Hazardous (301–500)** | 329 | 9.38% | 23.291 | 33.508 |

---

## SHAP Explainability & Physical Atmospheric Insights

```
Global Mean Absolute SHAP Ranking:
1. PM2.5_lag1   (48.24 AQI points) ── Dominant auto-regressive aerosol loading
2. PM2.5_roll6  ( 4.12 AQI points) ── Multi-hour temporal inertia
3. TEMP         ( 2.85 AQI points) ── Surface thermal inversions trap pollutants
4. WSPM         ( 2.14 AQI points) ── Mechanical wind ventilation & dispersion
5. NO2_lag1     ( 1.76 AQI points) ── Traffic exhaust & secondary nitrate precursor
```

### Physical Interpretations
1. **Aerosol Persistence:** $PM_{2.5}$ lag-1 is the primary contributor due to the long atmospheric residence time (days to weeks) of sub-2.5-micron fine particulate matter.
2. **Ventilation Flushing Effect:** Wind speed ($WSPM$) consistently exhibits negative SHAP values, confirming that strong winds mechanically flush trapped aerosol accumulations out of the Beijing basin.
3. **Winter Thermal Inversions:** Low temperature ($TEMP$) paired with high barometric pressure ($PRES$) produces positive SHAP attribution, reflecting planetary boundary-layer compression during Beijing winter smog events.

---

## Project Structure

```
ml project/
├── config/
│   └── config.yaml                 # Master pipeline configurations
├── data/
│   ├── raw/                        # UCI Beijing 12-station raw CSVs
│   └── processed/                  # Cleaned, engineered feature datasets
├── models/
│   ├── lightgbm.pkl                # Serialized LightGBM champion model
│   ├── xgboost.pkl                 # Serialized XGBoost model
│   ├── random_forest.pkl           # Serialized Random Forest model
│   ├── catboost.pkl                # Serialized CatBoost model
│   ├── linear_regression.pkl       # Serialized Linear Regression baseline
│   ├── scaler.joblib               # Chronological train-fitted StandardScaler
│   ├── feature_names.json          # Canonical 32-feature vector specification
│   └── split_metadata.json         # Chronological date range boundaries
├── notebooks/
│   └── exploratory_data_analysis.ipynb # 16-section comprehensive EDA notebook
├── outputs/
│   ├── charts/                     # 9 high-resolution evaluation figures (PNG)
│   ├── metrics/                    # Benchmark CSVs, JSON audits, evaluation report
│   ├── predictions/                # test_predictions.csv
│   └── shap/                       # Beeswarm, bar, and waterfall plots
├── pages/
│   ├── 1_Dashboard.py              # Atmospheric KPIs & historical trends
│   ├── 2_AQI_Prediction.py         # Real-time inference & local SHAP reasoning
│   ├── 3_Model_Performance.py      # Benchmark tables, ablation, residuals
│   ├── 4_SHAP_Explainability.py    # Global & local game-theoretic interpretability
│   ├── 5_Data_Explorer.py          # Missing value audits & correlation heatmaps
│   └── 6_About_Project.py          # Academic details, team, and provenance
├── src/
│   ├── __init__.py
│   ├── aqi_calculator.py           # US EPA 40 CFR 58 & CPCB sub-index calculator
│   ├── data_loader.py              # Station data discovery & integrity auditing
│   ├── data_cleaning.py            # Train-fitted imputation & outlier screening
│   ├── feature_engineering.py      # Cyclical encodings, lags, chronological split
│   ├── lgb_wrapper.py              # Robust LightGBM C++ wrapper with ABI fix
│   ├── model_training.py           # Multi-model training & ablation pipeline
│   ├── model_evaluation.py         # Diagnostic evaluation & category metrics
│   ├── prediction.py               # Production inference & input validation
│   ├── shap_explainer.py           # SHAP TreeExplainer & narrative generator
│   └── visualization.py            # Matplotlib charts & Plotly builders
├── tests/
│   ├── conftest.py                 # Pytest C-runtime initialization hook
│   ├── test_app.py                 # 10 integration tests for Streamlit app, 6 pages, & interactive actions
│   ├── test_aqi.py                 # 5 unit tests for EPA/CPCB AQI breakpoints
│   ├── test_cleaning.py            # 4 unit tests for leakage-free cleaning & sorting
│   ├── test_data.py                # 3 unit tests for dataset integrity
│   ├── test_features.py            # 3 unit tests for feature transformations
│   ├── test_models.py              # 4 unit tests for model persistence & metrics
│   ├── test_notebook.py            # 2 integration tests for EDA notebook cell execution & structure
│   └── test_prediction.py          # 13 unit tests for inference, validation, & concurrent thread safety
├── app.py                          # Master Streamlit entrypoint
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── README.md                       # Comprehensive documentation
```

---

## Installation & Execution Guide

### 1. Clone & Set Up Python Environment
```powershell
# Clone the repository
git clone https://github.com/harsh123-dev/ml-project.git
cd "ml project"

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install tested dependencies
pip install -r requirements.txt
```

### 2. Run the Full Machine Learning Pipeline
```powershell
# 1. Train all 5 models and run ablation experiment
python src/model_training.py

# 2. Evaluate performance and generate markdown report
python src/model_evaluation.py

# 3. Generate all 9 high-resolution evaluation figures
python src/visualization.py

# 4. Compute SHAP values and generate explainability plots
python src/shap_explainer.py
```

### 3. Run the Unit and Integration Test Suite (44 Tests)
```powershell
python -m pytest tests/ -v
```
*Expected Result: `44 passed in ~12s`*

### 4. Launch the Interactive Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Standards Compliance & Citations
1. **US EPA:** United States Environmental Protection Agency. (2018). *Technical Assistance Document for the Reporting of Daily Air Quality – the Air Quality Index (AQI)*. EPA-454/B-18-007. Office of Air Quality Planning and Standards. 40 CFR Part 58, Appendix G.
2. **UCI Dataset:** Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X. (2017). *Cautionary tales on air-quality improvement in Beijing*. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 473(2205), 20170457.
3. **SHAP Framework:** Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 4765–4774.

---

## License
Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for full details.
