"""
AIRWISE SHAP Explainability Module
----------------------------------
Computes global and local Shapley values using shap.TreeExplainer on tree ensembles.
Generates publication plots (beeswarm, bar, waterfall case studies) and natural-language
interpretability narratives translating quantitative SHAP contributions into human reasoning.
"""

import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_airwise")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import joblib
import shap
from typing import Dict, Any, List, Tuple

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import partition_dataset_chronological
from src.aqi_calculator import get_aqi_category


def generate_natural_language_explanation(
    prediction: float,
    base_value: float,
    feature_names: List[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Generate an intuitive, human-understandable explanation from SHAP values for a single prediction.
    """
    cat_info = get_aqi_category(prediction)
    cat_name = cat_info["category"]
    advisory = cat_info["advisory"]

    # Pair features with their SHAP value and actual value
    contributions = []
    for name, f_val, s_val in zip(feature_names, feature_values, shap_values):
        contributions.append({
            "feature": name,
            "value": float(f_val),
            "shap": float(s_val),
            "abs_shap": abs(float(s_val))
        })

    contributions_sorted = sorted(contributions, key=lambda x: x["abs_shap"], reverse=True)
    pushing_up = [c for c in contributions if c["shap"] > 0]
    pushing_up = sorted(pushing_up, key=lambda x: x["shap"], reverse=True)[:top_k]

    pulling_down = [c for c in contributions if c["shap"] < 0]
    pulling_down = sorted(pulling_down, key=lambda x: abs(x["shap"]), reverse=True)[:top_k]

    # Narrative construction
    narrative_paragraphs = []
    intro = (
        f"The model forecasts a next-hour AQI of **{prediction:.1f}** ({cat_name}), "
        f"departing from the historical baseline expected AQI of **{base_value:.1f}** "
        f"by **{prediction - base_value:+.1f} AQI points**."
    )
    narrative_paragraphs.append(intro)

    if pushing_up:
        up_details = [
            f"**{c['feature']}** ({c['value']:.1f}) added **+{c['shap']:.1f}** AQI points"
            for c in pushing_up
        ]
        up_text = "The primary drivers exacerbating pollution levels are: " + "; ".join(up_details) + "."
        narrative_paragraphs.append(up_text)

    if pulling_down:
        down_details = [
            f"**{c['feature']}** ({c['value']:.1f}) reduced predicted AQI by **{c['shap']:.1f}** points"
            for c in pulling_down
        ]
        down_text = "Mitigating atmospheric and meteorological factors include: " + "; ".join(down_details) + "."
        narrative_paragraphs.append(down_text)

    # Health advisory
    health_text = f"**Health Advisory ({cat_name}):** {advisory}"
    narrative_paragraphs.append(health_text)

    return {
        "prediction": prediction,
        "base_value": base_value,
        "category": cat_name,
        "advisory": advisory,
        "pushing_up": pushing_up,
        "pulling_down": pulling_down,
        "top_contributions": contributions_sorted[:top_k * 2],
        "narrative": "\n\n".join(narrative_paragraphs)
    }


def compute_and_save_shap_explanations(
    models_dir: str = "models",
    features_csv: str = "data/processed/features_aotizhongxin.csv",
    meta_json: str = "data/processed/feature_metadata.json",
    output_dir: str = "outputs/shap",
    sample_size: int = 600
) -> Dict[str, Any]:
    """
    Run TreeExplainer on XGBoost, save global beeswarm & bar plots,
    and save waterfall plots for clean vs hazardous episodes.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Model and Data
    model_path = os.path.join(models_dir, "xgboost.pkl")
    xgb_model = joblib.load(model_path)

    with open(meta_json, "r") as f:
        meta = json.load(f)
    feature_cols = meta["experiment_b_features"]

    df = pd.read_csv(features_csv)
    splits = partition_dataset_chronological(df, feature_cols=feature_cols, scale_features=False, models_dir="")
    X_test = pd.DataFrame(splits["X_test"], columns=feature_cols)
    y_test = splits["y_test"].values

    # Select representative sample from test set for fast, stable TreeExplainer
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    sample_indices = np.sort(sample_indices)
    X_sample = X_test.iloc[sample_indices]

    print("Step 1: Calculating SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_explanation = explainer(X_sample)
    shap_values = shap_explanation.values
    base_value = float(explainer.expected_value)

    # 2. Beeswarm Plot
    print("Step 2: Generating SHAP Beeswarm summary plot...")
    plt.figure(figsize=(10, 8), dpi=300)
    shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
    plt.title("SHAP Beeswarm: Impact of Multi-Modal Features on Next-Hour AQI", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    beeswarm_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, bbox_inches="tight")
    plt.close()

    # 3. Bar Plot (Global Importance)
    print("Step 3: Generating SHAP Bar summary plot...")
    plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.bar(shap_explanation, max_display=15, show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "shap_summary_bar.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()

    # 4. Local Waterfall Case Studies
    # Case 1: High AQI Episode (Severe Pollution)
    # Case 2: Low AQI Episode (Clean Air)
    sample_preds = xgb_model.predict(X_sample)

    high_idx_local = int(np.argmax(sample_preds))
    low_idx_local = int(np.argmin(sample_preds))

    # Case 1 Waterfall
    print("Step 4: Generating Waterfall Case 1 (Severe Pollution Episode)...")
    plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.waterfall(shap_explanation[high_idx_local], max_display=12, show=False)
    plt.title(f"Local Explanation: High AQI Episode (Forecast: {sample_preds[high_idx_local]:.1f})", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    waterfall1_path = os.path.join(output_dir, "shap_waterfall_case1.png")
    plt.savefig(waterfall1_path, bbox_inches="tight")
    plt.close()

    # Case 2 Waterfall
    print("Step 5: Generating Waterfall Case 2 (Clean Air Episode)...")
    plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.waterfall(shap_explanation[low_idx_local], max_display=12, show=False)
    plt.title(f"Local Explanation: Clean Air Episode (Forecast: {sample_preds[low_idx_local]:.1f})", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    waterfall2_path = os.path.join(output_dir, "shap_waterfall_case2.png")
    plt.savefig(waterfall2_path, bbox_inches="tight")
    plt.close()

    # 5. Generate Natural Language Explanations for Case Studies
    nl_case1 = generate_natural_language_explanation(
        prediction=float(sample_preds[high_idx_local]),
        base_value=base_value,
        feature_names=feature_cols,
        feature_values=X_sample.iloc[high_idx_local].values,
        shap_values=shap_values[high_idx_local]
    )

    nl_case2 = generate_natural_language_explanation(
        prediction=float(sample_preds[low_idx_local]),
        base_value=base_value,
        feature_names=feature_cols,
        feature_values=X_sample.iloc[low_idx_local].values,
        shap_values=shap_values[low_idx_local]
    )

    # 6. Global importance summary table
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_Abs_SHAP": np.round(mean_abs_shap, 4)
    }).sort_values("Mean_Abs_SHAP", ascending=False)
    importance_df.to_csv(os.path.join(output_dir, "shap_importance_ranking.csv"), index=False)

    # 7. Serialize summary bundle
    shap_artifacts = {
        "base_value": base_value,
        "sample_size": len(X_sample),
        "case_studies": {
            "high_pollution_case": nl_case1,
            "clean_air_case": nl_case2
        },
        "top_features": importance_df.head(10).to_dict(orient="records")
    }
    with open(os.path.join(output_dir, "shap_summary.json"), "w") as f:
        json.dump(shap_artifacts, f, indent=4)

    # Also save the explainer object and sample shap explanation for Streamlit on-demand explanations
    joblib.dump({
        "explainer": explainer,
        "feature_names": feature_cols,
        "base_value": base_value,
        "sample_X": X_sample.iloc[:100],
        "sample_shap_values": shap_values[:100]
    }, os.path.join(output_dir, "shap_bundle.joblib"))

    print(f"SHAP artifacts successfully saved to: {output_dir}")
    return shap_artifacts


if __name__ == "__main__":
    compute_and_save_shap_explanations()
