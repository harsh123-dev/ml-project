# AIRWISE Model Evaluation Report

## 1. Executive Summary
- **Primary Forecast Target:** Next-hour PM2.5-derived Air Quality Index ($AQI_{t+1}$)
- **Station:** Aotizhongxin (UCI Beijing Air Quality Dataset)
- **Test Samples:** 3,507 chronologically held-out hours (Oct 2016 – Feb 2017)
- **Top Performing Model:** **LightGBM** (MAE: 13.678, RMSE: 22.246, $R^2$: 0.9553)

## 2. Multi-Model Performance Comparison
| Model | MAE | RMSE | $R^2$ | MAPE (%) | Median AE | Max Error |
|:------|:---:|:----:|:-----:|:--------:|:---------:|:---------:|
| Linear Regression | 19.140 | 26.620 | 0.9359 | 27.26% | 14.930 | 271.750 |
| Random Forest | 14.178 | 23.066 | 0.9519 | 18.85% | 8.850 | 261.540 |
| XGBoost | 13.813 | 22.495 | 0.9543 | 18.55% | 8.660 | 266.550 |
| LightGBM | 13.678 | 22.246 | 0.9553 | 18.36% | 8.630 | 269.330 |
| CatBoost | 13.869 | 22.696 | 0.9534 | 18.37% | 8.630 | 264.170 |
| Persistence Baseline (t) | 14.444 | 24.475 | 0.9459 | 18.72% | 8.330 | 266.290 |

## 3. Performance Across AQI Severity Categories (LightGBM)
| Category | Samples | Distribution | MAE | RMSE |
|:---------|:-------:|:------------:|:---:|:----:|
| Good | 661 | 18.85% | 12.490 | 21.474 |
| Moderate | 655 | 18.68% | 13.754 | 20.875 |
| Unhealthy for Sensitive Groups | 303 | 8.64% | 14.022 | 19.243 |
| Unhealthy | 971 | 27.69% | 9.534 | 16.777 |
| Very Unhealthy | 588 | 16.77% | 16.214 | 25.766 |
| Hazardous | 329 | 9.38% | 23.291 | 33.508 |
