"""
Unit Tests for AIRWISE Model Loading and Predictions
"""

# Import lightgbm first for Windows C-runtime stability
import lightgbm as lgb
import os
import sys
import json
import joblib
import pytest
import numpy as np
import pandas as pd

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lgb_wrapper import LightGBMBoosterWrapper
from src.model_training import LightGBMBoosterWrapper

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics")


def test_model_files_exist():
    expected_files = [
        "linear_regression.pkl",
        "random_forest.pkl",
        "xgboost.pkl",
        "lightgbm.pkl",
        "catboost.pkl",
        "scaler.joblib",
        "feature_names.json"
    ]
    for fname in expected_files:
        fpath = os.path.join(MODELS_DIR, fname)
        assert os.path.exists(fpath), f"Expected artifact missing: {fname}"


def test_models_load_and_predict():
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "r") as f:
        feature_names = json.load(f)
    assert len(feature_names) == 32

    # Synthetic test input
    X_dummy = np.ones((5, 32), dtype=np.float32)

    model_names = [
        "lightgbm.pkl",
        "linear_regression.pkl",
        "random_forest.pkl",
        "xgboost.pkl",
        "catboost.pkl"
    ]
    for mfile in model_names:
        mpath = os.path.join(MODELS_DIR, mfile)
        model = joblib.load(mpath)
        assert model is not None

        preds = model.predict(X_dummy)
        assert len(preds) == 5
        assert np.isfinite(preds).all()
        assert not np.isnan(preds).any()


def test_model_comparison_metrics():
    comp_path = os.path.join(METRICS_DIR, "model_comparison.csv")
    assert os.path.exists(comp_path), "model_comparison.csv must exist"

    df = pd.read_csv(comp_path)
    assert len(df) == 5
    assert set(df["Model"]) == {"LightGBM", "Linear Regression", "Random Forest", "XGBoost", "CatBoost"}

    # All models should achieve R2 >= 0.85
    assert (df["R2"] >= 0.85).all(), "All models should have R2 >= 0.85"

    # Best model MAE should be <= 25.0
    assert df["MAE"].min() <= 20.0, "Best model MAE should be <= 20.0"


def test_ablation_results():
    abl_path = os.path.join(METRICS_DIR, "ablation_results.csv")
    assert os.path.exists(abl_path), "ablation_results.csv must exist"

    df = pd.read_csv(abl_path)
    assert len(df) == 2
    full_model_row = df[df["Feature Configuration"].str.contains("Weather")].iloc[0]
    pollution_only_row = df[df["Feature Configuration"].str.contains("Pollution Only")].iloc[0]

    # Full model should outperform pollution only
    assert full_model_row["MAE"] <= pollution_only_row["MAE"]
    assert full_model_row["RMSE"] <= pollution_only_row["RMSE"]
    assert full_model_row["R2"] >= pollution_only_row["R2"]
