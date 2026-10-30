"""
AIRWISE Machine Learning Training Pipeline
------------------------------------------
Trains and serializes all five regression models:
1. Linear Regression (Baseline)
2. Random Forest Regressor
3. XGBoost Regressor
4. LightGBM Regressor
5. CatBoost Regressor

Evaluates on chronological validation and test sets and performs the ablation experiment.
"""

# Import lightgbm FIRST before any other native numerical libraries to ensure proper C++ runtime binding
import lightgbm as lgb
import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Set loky max cpu count to suppress loky core inspection warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.data_loader import load_station_data
from src.data_cleaning import clean_air_quality_data
from src.feature_engineering import build_feature_dataset, partition_dataset_chronological


from src.lgb_wrapper import LightGBMBoosterWrapper


def train_and_evaluate_models(
    standard: str = "US_EPA",
    models_dir: str = "models",
    metrics_dir: str = "outputs/metrics",
    predictions_dir: str = "outputs/predictions"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run full training pipeline for all 5 models and the ablation experiment.

    Returns:
    --------
    comparison_df : pd.DataFrame
        Table of MAE, RMSE, R2, and Training Time for all 5 models.
    ablation_df : pd.DataFrame
        Ablation study comparing 'Pollution Only' vs 'Pollution + Weather + Time + Wind'.
    artifacts : Dict[str, Any]
        Trained model instances and test data.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    print("Step 1: Loading and cleaning raw dataset...")
    df_raw = load_station_data(station_name="Aotizhongxin")
    df_clean, clean_meta = clean_air_quality_data(df_raw)

    print(f"Step 2: Engineering features using AQI standard: {standard}...")
    feat_df, feat_meta = build_feature_dataset(df_clean, standard=standard)

    exp_b_features = feat_meta["experiment_b_features"]
    exp_a_features = feat_meta["experiment_a_features"]

    print("Step 3: Partitioning dataset chronologically (80% train, 10% val, 10% test)...")
    splits_b = partition_dataset_chronological(
        feat_df,
        feature_cols=exp_b_features,
        scale_features=True,
        models_dir=models_dir
    )

    X_train = np.array(splits_b["X_train"], dtype=np.float32, order="C", copy=True)
    y_train = np.array(splits_b["y_train"], dtype=np.float32, order="C", copy=True)
    X_val = np.array(splits_b["X_val"], dtype=np.float32, order="C", copy=True)
    y_val = np.array(splits_b["y_val"], dtype=np.float32, order="C", copy=True)
    X_test = np.array(splits_b["X_test"], dtype=np.float32, order="C", copy=True)
    y_test = np.array(splits_b["y_test"], dtype=np.float32, order="C", copy=True)

    X_train_scaled = np.array(splits_b["X_train_scaled"], dtype=np.float32, order="C", copy=True)
    X_test_scaled = np.array(splits_b["X_test_scaled"], dtype=np.float32, order="C", copy=True)

    # Initialize 5 models (LightGBM trained first before CatBoost library is loaded)
    models_dict = {
        "LightGBM": {
            "model": LightGBMBoosterWrapper(
                n_estimators=120,
                learning_rate=0.08,
                num_leaves=31,
                max_depth=6,
                random_state=42
            ),
            "use_scaled": False
        },
        "Linear Regression": {
            "model": LinearRegression(fit_intercept=True),
            "use_scaled": True
        },
        "Random Forest": {
            "model": RandomForestRegressor(
                n_estimators=100,
                max_depth=16,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            "use_scaled": False
        },
        "XGBoost": {
            "model": XGBRegressor(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1
            ),
            "use_scaled": False
        },
        "CatBoost": {
            "model": None,  # Lazy loaded after LightGBM completes
            "use_scaled": False
        }
    }

    results = []
    trained_models = {}
    test_preds_df = pd.DataFrame({"Actual_AQI": y_test})

    print("\nStep 4: Training all five models...")
    for model_name, info in models_dict.items():
        if model_name == "CatBoost":
            from catboost import CatBoostRegressor
            estimator = CatBoostRegressor(
                iterations=120,
                learning_rate=0.08,
                depth=6,
                random_seed=42,
                verbose=0
            )
        else:
            estimator = info["model"]

        use_scaled = info["use_scaled"]

        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        print(f"  Training {model_name}...")
        t0 = time.time()
        estimator.fit(X_tr, y_train)
        train_time = time.time() - t0

        preds = estimator.predict(X_te)
        test_preds_df[model_name] = np.round(preds, 2)

        mae = float(np.round(mean_absolute_error(y_test, preds), 3))
        rmse = float(np.round(root_mean_squared_error(y_test, preds), 3))
        r2 = float(np.round(r2_score(y_test, preds), 4))

        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Training Time (s)": float(np.round(train_time, 2))
        })

        trained_models[model_name] = estimator

        # Serialize individual model
        slug = model_name.lower().replace(" ", "_")
        model_file = os.path.join(models_dir, f"{slug}.pkl")
        joblib.dump(estimator, model_file)

    comparison_df = pd.DataFrame(results)
    comparison_csv_path = os.path.join(metrics_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nModel comparison table saved to: {comparison_csv_path}")
    print(comparison_df.to_string(index=False))

    # Step 5: Ablation Study
    print("\nStep 5: Executing Ablation Study (Pollution Only vs Full Model)...")
    splits_a = partition_dataset_chronological(
        feat_df,
        feature_cols=exp_a_features,
        scale_features=False,
        models_dir=""
    )
    X_train_a = np.array(splits_a["X_train"], dtype=np.float32, order="C", copy=True)
    X_test_a = np.array(splits_a["X_test"], dtype=np.float32, order="C", copy=True)

    # Use XGBoost for the ablation comparison (consistent reference)
    ablation_xgb = XGBRegressor(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    ablation_xgb.fit(X_train_a, y_train)
    preds_a = ablation_xgb.predict(X_test_a)

    mae_a = float(np.round(mean_absolute_error(y_test, preds_a), 3))
    rmse_a = float(np.round(root_mean_squared_error(y_test, preds_a), 3))
    r2_a = float(np.round(r2_score(y_test, preds_a), 4))

    # Exp B metrics from XGBoost
    xgb_metrics = comparison_df[comparison_df["Model"] == "XGBoost"].iloc[0]

    ablation_records = [
        {
            "Feature Configuration": "Pollution Only",
            "MAE": mae_a,
            "RMSE": rmse_a,
            "R2": r2_a
        },
        {
            "Feature Configuration": "Pollution + Weather + Time + Wind",
            "MAE": xgb_metrics["MAE"],
            "RMSE": xgb_metrics["RMSE"],
            "R2": xgb_metrics["R2"]
        }
    ]
    ablation_df = pd.DataFrame(ablation_records)
    ablation_csv_path = os.path.join(metrics_dir, "ablation_results.csv")
    ablation_df.to_csv(ablation_csv_path, index=False)
    print(f"Ablation results saved to: {ablation_csv_path}")
    print(ablation_df.to_string(index=False))

    # Save ablation model
    joblib.dump(ablation_xgb, os.path.join(models_dir, "ablation_pollution_only.pkl"))
    with open(os.path.join(models_dir, "ablation_features.json"), "w") as f:
        json.dump(exp_a_features, f, indent=4)

    # Step 6: Save predictions CSV
    test_preds_path = os.path.join(predictions_dir, "test_predictions.csv")
    test_preds_df["datetime"] = feat_df.iloc[splits_b["split_indices"]["test_range"][0]:]["datetime"].values
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"Test predictions saved to: {test_preds_path}")

    # Metadata record
    training_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "standard": standard,
        "station": "Aotizhongxin",
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "features_count_exp_b": len(exp_b_features),
        "features_count_exp_a": len(exp_a_features),
        "best_model_by_rmse": comparison_df.sort_values("RMSE").iloc[0]["Model"],
        "best_model_by_mae": comparison_df.sort_values("MAE").iloc[0]["Model"],
        "best_model_by_r2": comparison_df.sort_values("R2", ascending=False).iloc[0]["Model"],
        "models_trained": list(trained_models.keys())
    }
    with open(os.path.join(metrics_dir, "training_metadata.json"), "w") as f:
        json.dump(training_summary, f, indent=4)

    return comparison_df, ablation_df, {
        "models": trained_models,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": exp_b_features,
        "test_preds_df": test_preds_df
    }


if __name__ == "__main__":
    train_and_evaluate_models()
