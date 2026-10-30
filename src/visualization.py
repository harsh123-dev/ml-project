"""
AIRWISE Visualizations Module
-----------------------------
Generates publication-quality static charts (saved to outputs/charts/)
and interactive Plotly figures for the Streamlit dashboard:
1. MAE Comparison
2. RMSE Comparison
3. R2 Comparison
4. Combined Multi-metric Comparison
5. Ablation Study (Pollution Only vs Multi-modal)
6. Actual vs Predicted Scatter Plot
7. Actual vs Predicted Time-Series Tracking
8. Residual Diagnostics Plot
9. Feature Importance Ranking
"""

import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_airwise")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import joblib
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set elegant academic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PRIMARY_COLOR = "#1A365D"
ACCENT_COLOR = "#2B6CB0"
SECONDARY_COLOR = "#2E8540"
HIGHLIGHT_COLOR = "#C53030"
COLOR_PALETTE = ["#1A365D", "#2B6CB0", "#319795", "#2E8540", "#D69E2E", "#DD6B20"]


def generate_all_evaluation_charts(
    metrics_dir: str = "outputs/metrics",
    predictions_csv: str = "outputs/predictions/test_predictions.csv",
    models_dir: str = "models",
    output_dir: str = "outputs/charts"
) -> Dict[str, str]:
    """
    Generate all 9 evaluation figures and save them to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_charts = {}

    comparison_csv = os.path.join(metrics_dir, "model_comparison.csv")
    ablation_csv = os.path.join(metrics_dir, "ablation_results.csv")
    preds_df = pd.read_csv(predictions_csv)
    comp_df = pd.read_csv(comparison_csv)
    ablation_df = pd.read_csv(ablation_csv)

    # 1. MAE Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(comp_df["Model"], comp_df["MAE"], color=COLOR_PALETTE[:len(comp_df)], width=0.55, edgecolor="none")
    ax.set_title("Mean Absolute Error (MAE) Comparison across Models (Next-Hour AQI)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("MAE (AQI Points)", fontsize=11, fontweight="medium")
    ax.set_ylim(0, max(comp_df["MAE"]) * 1.25)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "1_mae_comparison.png")
    plt.savefig(p1)
    plt.close()
    generated_charts["mae_comparison"] = p1

    # 2. RMSE Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(comp_df["Model"], comp_df["RMSE"], color=COLOR_PALETTE[:len(comp_df)], width=0.55, edgecolor="none")
    ax.set_title("Root Mean Squared Error (RMSE) Comparison across Models", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("RMSE (AQI Points)", fontsize=11, fontweight="medium")
    ax.set_ylim(0, max(comp_df["RMSE"]) * 1.25)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "2_rmse_comparison.png")
    plt.savefig(p2)
    plt.close()
    generated_charts["rmse_comparison"] = p2

    # 3. R2 Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(comp_df["Model"], comp_df["R2"], color=COLOR_PALETTE[:len(comp_df)], width=0.55, edgecolor="none")
    ax.set_title("Coefficient of Determination ($R^2$) Comparison across Models", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("$R^2$ Score", fontsize=11, fontweight="medium")
    ax.set_ylim(0.85, 1.0)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    p3 = os.path.join(output_dir, "3_r2_comparison.png")
    plt.savefig(p3)
    plt.close()
    generated_charts["r2_comparison"] = p3

    # 4. Combined Multi-metric Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    metrics = [("MAE", "AQI Points"), ("RMSE", "AQI Points"), ("R2", "Score")]
    for i, (m, unit) in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(comp_df["Model"], comp_df[m], color=COLOR_PALETTE[:len(comp_df)], width=0.55)
        ax.set_title(f"{m} ({unit})", fontsize=12, fontweight="bold")
        if m == "R2":
            ax.set_ylim(0.85, 1.0)
        else:
            ax.set_ylim(0, max(comp_df[m]) * 1.25)
        for b in bars:
            h = b.get_height()
            fmt = f"{h:.4f}" if m == "R2" else f"{h:.2f}"
            ax.annotate(fmt, xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
    plt.suptitle("AIRWISE Comprehensive Model Performance Benchmark", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    p4 = os.path.join(output_dir, "4_combined_model_comparison.png")
    plt.savefig(p4, bbox_inches="tight")
    plt.close()
    generated_charts["combined_comparison"] = p4

    # 5. Ablation Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(ablation_df))
    width = 0.35
    b1 = ax.bar(x - width/2, ablation_df["MAE"], width, label="MAE (lower is better)", color="#2B6CB0")
    b2 = ax.bar(x + width/2, ablation_df["RMSE"], width, label="RMSE (lower is better)", color="#E53E3E")
    ax.set_title("Ablation Study: Pollution Only vs Multi-Modal Features", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_df["Feature Configuration"], fontsize=10, fontweight="medium")
    ax.set_ylabel("Error (AQI Points)", fontsize=11)
    ax.legend(frameon=True)
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    p5 = os.path.join(output_dir, "5_ablation_comparison.png")
    plt.savefig(p5)
    plt.close()
    generated_charts["ablation_comparison"] = p5

    # 6. Actual vs Predicted Scatter Plot (LightGBM)
    best_model = "LightGBM" if "LightGBM" in preds_df.columns else "XGBoost"
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    y_act = preds_df["Actual_AQI"].values
    y_pr = preds_df[best_model].values
    ax.scatter(y_act, y_pr, alpha=0.25, color="#1A365D", s=18, edgecolors="none")
    max_val = max(np.max(y_act), np.max(y_pr))
    ax.plot([0, max_val], [0, max_val], color="#E53E3E", linestyle="--", linewidth=1.8, label="Ideal 1:1 Identity Line")
    ax.set_title(f"Actual vs Predicted AQI — {best_model} (Test Set, N={len(y_act)})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Ground Truth AQI ($AQI_{t+1}$)", fontsize=11, fontweight="medium")
    ax.set_ylabel(f"Predicted AQI ({best_model})", fontsize=11, fontweight="medium")
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    plt.tight_layout()
    p6 = os.path.join(output_dir, "6_actual_vs_predicted_scatter.png")
    plt.savefig(p6)
    plt.close()
    generated_charts["actual_vs_predicted_scatter"] = p6

    # 7. Actual vs Predicted Time-Series (High-Variance Episode: 168 hours = 1 week)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    slice_start = 500
    slice_end = 668  # 168 hours = 7 continuous days
    sub_df = preds_df.iloc[slice_start:slice_end]
    hours = np.arange(len(sub_df))
    ax.plot(hours, sub_df["Actual_AQI"], label="Actual AQI", color="#1A365D", linewidth=2.0)
    ax.plot(hours, sub_df[best_model], label=f"Predicted ({best_model})", color="#E53E3E", linewidth=1.8, linestyle="--")
    if "Linear Regression" in sub_df.columns:
        ax.plot(hours, sub_df["Linear Regression"], label="Linear Regression (Baseline)", color="#718096", linewidth=1.2, alpha=0.7)
    ax.set_title(f"Hourly AQI Forecast Tracking over 7 Consecutive Days (168-Hour Test Window)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Elapsed Time (Hours)", fontsize=11)
    ax.set_ylabel("AQI ($PM_{2.5}$ Sub-Index)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    p7 = os.path.join(output_dir, "7_actual_vs_predicted_timeseries.png")
    plt.savefig(p7)
    plt.close()
    generated_charts["actual_vs_predicted_timeseries"] = p7

    # 8. Residual Plot
    residuals = y_act - y_pr
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300, gridspec_kw={"width_ratios": [2, 1]})
    ax1.scatter(y_pr, residuals, alpha=0.3, color="#2B6CB0", s=18, edgecolors="none")
    ax1.axhline(0, color="#E53E3E", linestyle="--", linewidth=1.8)
    ax1.set_title(f"Residuals vs Predicted AQI ({best_model})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted AQI", fontsize=11)
    ax1.set_ylabel("Residual ($Actual - Predicted$)", fontsize=11)

    sns.histplot(residuals, kde=True, ax=ax2, color="#1A365D", bins=40)
    ax2.axvline(0, color="#E53E3E", linestyle="--", linewidth=1.8)
    ax2.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Residual", fontsize=11)
    plt.tight_layout()
    p8 = os.path.join(output_dir, "8_residual_plot.png")
    plt.savefig(p8)
    plt.close()
    generated_charts["residual_plot"] = p8

    # 9. Feature Importance (XGBoost)
    xgb_path = os.path.join(models_dir, "xgboost.pkl")
    feat_names_path = os.path.join(models_dir, "feature_names.json")
    if os.path.exists(xgb_path) and os.path.exists(feat_names_path):
        xgb_model = joblib.load(xgb_path)
        with open(feat_names_path, "r") as f:
            feat_names = json.load(f)
        importances = xgb_model.feature_importances_
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
        bars = ax.barh(imp_df["Feature"], imp_df["Importance"], color="#2B6CB0", height=0.6)
        ax.set_title("Top 15 Most Influential Features (XGBoost Gain Importance)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Relative Feature Importance Score", fontsize=11)
        for b in bars:
            w = b.get_width()
            ax.annotate(f"{w:.3f}", xy=(w, b.get_y() + b.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        p9 = os.path.join(output_dir, "9_feature_importance.png")
        plt.savefig(p9)
        plt.close()
        generated_charts["feature_importance"] = p9

    print(f"All {len(generated_charts)} evaluation figures generated in: {output_dir}")
    return generated_charts


# Interactive Plotly chart builders for Streamlit
def build_plotly_comparison(comp_df: pd.DataFrame, metric: str = "MAE") -> go.Figure:
    fig = px.bar(
        comp_df,
        x="Model",
        y=metric,
        color="Model",
        color_discrete_sequence=["#1A365D", "#2B6CB0", "#319795", "#2E8540", "#D69E2E"],
        text_auto=".3f",
        title=f"Model Comparison: {metric}"
    )
    fig.update_layout(showlegend=False, template="plotly_white", height=420)
    return fig


def build_plotly_timeseries(preds_df: pd.DataFrame, model_name: str, start: int = 0, length: int = 168) -> go.Figure:
    sub = preds_df.iloc[start:start + length]
    x_axis = sub["datetime"] if "datetime" in sub.columns else np.arange(len(sub))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=sub["Actual_AQI"], name="Ground Truth AQI", line=dict(color="#1A365D", width=2.5)))
    fig.add_trace(go.Scatter(x=x_axis, y=sub[model_name], name=f"Predicted ({model_name})", line=dict(color="#E53E3E", width=2, dash="dash")))
    fig.update_layout(
        title=f"Real-Time Forecast Tracking: {model_name} (Continuous Window)",
        xaxis_title="Timeline",
        yaxis_title="AQI Sub-Index",
        template="plotly_white",
        hovermode="x unified",
        height=450
    )
    return fig


def build_plotly_scatter(preds_df: pd.DataFrame, model_name: str) -> go.Figure:
    fig = px.scatter(
        preds_df,
        x="Actual_AQI",
        y=model_name,
        opacity=0.35,
        color_discrete_sequence=["#2B6CB0"],
        labels={"Actual_AQI": "Ground Truth AQI", model_name: f"Predicted AQI ({model_name})"},
        title=f"Actual vs Predicted AQI Scatter: {model_name}"
    )
    max_v = max(preds_df["Actual_AQI"].max(), preds_df[model_name].max())
    fig.add_shape(
        type="line", line=dict(dash="dash", color="#E53E3E", width=2),
        x0=0, x1=max_v, y0=0, y1=max_v
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig


if __name__ == "__main__":
    generate_all_evaluation_charts()
