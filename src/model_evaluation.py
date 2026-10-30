"""
AIRWISE Comprehensive Model Evaluation Module
---------------------------------------------
Evaluates trained models across comprehensive metrics, computes persistence baseline,
analyzes performance across EPA AQI categories, and produces markdown reports and JSON artifacts.
"""

# Import lightgbm first for Windows C-runtime stability
import lightgbm as lgb
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
    max_error
)

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lgb_wrapper import LightGBMBoosterWrapper
from src.aqi_calculator import get_aqi_category


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (avoiding divide-by-zero)."""
    mask = y_true != 0
    return float(np.round(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100, 2))


def evaluate_all_models(
    models_dir: str = "models",
    metrics_dir: str = "outputs/metrics",
    predictions_csv: str = "outputs/predictions/test_predictions.csv"
) -> Dict[str, Any]:
    """
    Load test predictions, calculate comprehensive metrics, baseline comparison,
    and category-level error breakdowns.
    """
    os.makedirs(metrics_dir, exist_ok=True)
    preds_df = pd.read_csv(predictions_csv)
    y_test = preds_df["Actual_AQI"].values

    models = ["Linear Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
    model_metrics = {}

    for model_name in models:
        if model_name not in preds_df.columns:
            continue
        preds = preds_df[model_name].values
        mae = float(np.round(mean_absolute_error(y_test, preds), 3))
        rmse = float(np.round(root_mean_squared_error(y_test, preds), 3))
        r2 = float(np.round(r2_score(y_test, preds), 4))
        mape = calculate_mape(y_test, preds)
        medae = float(np.round(median_absolute_error(y_test, preds), 3))
        max_err = float(np.round(max_error(y_test, preds), 3))
        evs = float(np.round(explained_variance_score(y_test, preds), 4))

        model_metrics[model_name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE (%)": mape,
            "Median AE": medae,
            "Max Error": max_err,
            "Explained Variance": evs
        }

    # Persistence Baseline (AQI_{t+1} = AQI_t)
    # The actual previous AQI can be approximated from PM2.5_AQI_t in features
    # Or from lag-1 actual AQI
    persistence_preds = np.roll(y_test, 1)
    persistence_preds[0] = y_test[0]  # first element boundary
    pers_mae = float(np.round(mean_absolute_error(y_test, persistence_preds), 3))
    pers_rmse = float(np.round(root_mean_squared_error(y_test, persistence_preds), 3))
    pers_r2 = float(np.round(r2_score(y_test, persistence_preds), 4))
    model_metrics["Persistence Baseline (t)"] = {
        "MAE": pers_mae,
        "RMSE": pers_rmse,
        "R2": pers_r2,
        "MAPE (%)": calculate_mape(y_test, persistence_preds),
        "Median AE": float(np.round(median_absolute_error(y_test, persistence_preds), 3)),
        "Max Error": float(np.round(max_error(y_test, persistence_preds), 3)),
        "Explained Variance": float(np.round(explained_variance_score(y_test, persistence_preds), 4))
    }

    metrics_summary_df = pd.DataFrame(model_metrics).T.reset_index().rename(columns={"index": "Model"})
    summary_csv_path = os.path.join(metrics_dir, "detailed_model_metrics.csv")
    metrics_summary_df.to_csv(summary_csv_path, index=False)

    # AQI Category breakdown for best model (LightGBM / XGBoost)
    best_model_name = "LightGBM" if "LightGBM" in preds_df.columns else "XGBoost"
    best_preds = preds_df[best_model_name].values

    categories = [
        "Good", "Moderate", "Unhealthy for Sensitive Groups",
        "Unhealthy", "Very Unhealthy", "Hazardous"
    ]
    cat_breakdown = []
    for cat in categories:
        cat_indices = [
            i for i, actual in enumerate(y_test)
            if get_aqi_category(actual)["category"] == cat
        ]
        if not cat_indices:
            continue
        sub_y = y_test[cat_indices]
        sub_pred = best_preds[cat_indices]
        cat_mae = float(np.round(mean_absolute_error(sub_y, sub_pred), 3))
        cat_rmse = float(np.round(root_mean_squared_error(sub_y, sub_pred), 3))
        cat_breakdown.append({
            "Category": cat,
            "Sample Count": len(cat_indices),
            "Share (%)": float(np.round(len(cat_indices) / len(y_test) * 100, 2)),
            "MAE": cat_mae,
            "RMSE": cat_rmse
        })

    cat_df = pd.DataFrame(cat_breakdown)
    cat_csv_path = os.path.join(metrics_dir, "category_error_breakdown.csv")
    cat_df.to_csv(cat_csv_path, index=False)

    # Save detailed JSON evaluation
    full_eval = {
        "models": model_metrics,
        "best_model": best_model_name,
        "best_model_metrics": model_metrics[best_model_name],
        "category_breakdown": cat_breakdown,
        "sample_count_test": len(y_test)
    }
    with open(os.path.join(metrics_dir, "detailed_evaluation.json"), "w") as f:
        json.dump(full_eval, f, indent=4)

    # Generate Markdown summary report
    report_md = f"""# AIRWISE Model Evaluation Report

## 1. Executive Summary
- **Primary Forecast Target:** Next-hour PM2.5-derived Air Quality Index ($AQI_{{t+1}}$)
- **Station:** Aotizhongxin (UCI Beijing Air Quality Dataset)
- **Test Samples:** {len(y_test):,} chronologically held-out hours (Oct 2016 – Feb 2017)
- **Top Performing Model:** **{best_model_name}** (MAE: {model_metrics[best_model_name]['MAE']}, RMSE: {model_metrics[best_model_name]['RMSE']}, $R^2$: {model_metrics[best_model_name]['R2']})

## 2. Multi-Model Performance Comparison
| Model | MAE | RMSE | $R^2$ | MAPE (%) | Median AE | Max Error |
|:------|:---:|:----:|:-----:|:--------:|:---------:|:---------:|
"""
    for idx, row in metrics_summary_df.iterrows():
        report_md += f"| {row['Model']} | {row['MAE']:.3f} | {row['RMSE']:.3f} | {row['R2']:.4f} | {row['MAPE (%)']:.2f}% | {row['Median AE']:.3f} | {row['Max Error']:.3f} |\n"

    report_md += f"""
## 3. Performance Across AQI Severity Categories ({best_model_name})
| Category | Samples | Distribution | MAE | RMSE |
|:---------|:-------:|:------------:|:---:|:----:|
"""
    for cat in cat_breakdown:
        report_md += f"| {cat['Category']} | {cat['Sample Count']} | {cat['Share (%)']}% | {cat['MAE']:.3f} | {cat['RMSE']:.3f} |\n"

    with open(os.path.join(metrics_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Detailed evaluation metrics saved to: {summary_csv_path}")
    print(f"Category breakdown saved to: {cat_csv_path}")
    print(f"Evaluation report written to: {os.path.join(metrics_dir, 'evaluation_report.md')}")

    return full_eval


if __name__ == "__main__":
    evaluate_all_models()
