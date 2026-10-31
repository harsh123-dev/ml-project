"""
AIRWISE Production Prediction & Inference Pipeline
--------------------------------------------------
Serves single-observation and batch predictions using any of the 5 trained models.
Handles rigorous input validation, cyclical feature encoding, wind one-hot encoding,
feature scaling, AQI categorization, and on-demand local SHAP explanations.
"""

# Import lightgbm first for Windows C-runtime stability
import lightgbm as lgb
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lgb_wrapper import LightGBMBoosterWrapper
from src.aqi_calculator import get_aqi_category
from src.shap_explainer import generate_natural_language_explanation

POLLUTANT_BOUNDS = {
    "PM2.5": (0.0, 1000.0),
    "PM10": (0.0, 1500.0),
    "SO2": (0.0, 1000.0),
    "NO2": (0.0, 1000.0),
    "CO": (0.0, 20000.0),
    "O3": (0.0, 1000.0)
}

WEATHER_BOUNDS = {
    "TEMP": (-50.0, 60.0),
    "PRES": (900.0, 1100.0),
    "DEWP": (-50.0, 50.0),
    "RAIN": (0.0, 200.0),
    "WSPM": (0.0, 50.0)
}

ALL_WIND_DIRECTIONS = [
    "E", "ENE", "ESE", "N", "NE", "NNE", "NNW", "NW",
    "S", "SE", "SSE", "SSW", "SW", "W", "WNW", "WSW"
]


class AirQualityPredictor:
    """
    Inference engine managing model loading, feature transformation,
    and prediction generation for the AIRWISE system.
    """
    def __init__(self, models_dir: str = "models", shap_dir: str = "outputs/shap"):
        self.models_dir = models_dir
        self.shap_dir = shap_dir
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.shap_bundle = None
        self._load_artifacts()

    def _load_artifacts(self):
        # 1. Feature names
        feat_path = os.path.join(self.models_dir, "feature_names.json")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature names file missing: {feat_path}")
        with open(feat_path, "r") as f:
            self.feature_names = json.load(f)

        # 2. Scaler
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        # 3. Models
        model_files = {
            "LightGBM": "lightgbm.pkl",
            "Linear Regression": "linear_regression.pkl",
            "Random Forest": "random_forest.pkl",
            "XGBoost": "xgboost.pkl",
            "CatBoost": "catboost.pkl"
        }
        for name, filename in model_files.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)

        # 4. Optional SHAP bundle
        bundle_path = os.path.join(self.shap_dir, "shap_bundle.joblib")
        if os.path.exists(bundle_path):
            try:
                self.shap_bundle = joblib.load(bundle_path)
            except Exception:
                self.shap_bundle = None

    def validate_inputs(self, raw_input: Dict[str, Any]) -> None:
        """
        Validate input ranges and types against physical real-world atmospheric limits.
        """
        for pollutant, (lo, hi) in POLLUTANT_BOUNDS.items():
            if pollutant in raw_input:
                val = float(raw_input[pollutant])
                if val < lo or val > hi:
                    raise ValueError(f"Value for '{pollutant}' ({val}) is outside valid physical range [{lo}, {hi}].")

        for var, (lo, hi) in WEATHER_BOUNDS.items():
            if var in raw_input:
                val = float(raw_input[var])
                if val < lo or val > hi:
                    raise ValueError(f"Value for '{var}' ({val}) is outside valid atmospheric range [{lo}, {hi}].")

        if "wd" in raw_input and raw_input["wd"] is not None:
            wd = str(raw_input["wd"]).strip().upper()
            if wd not in ALL_WIND_DIRECTIONS:
                raise ValueError(f"Wind direction '{wd}' is not recognized. Valid options: {ALL_WIND_DIRECTIONS}")

    def transform_single_input(self, raw_input: Dict[str, Any]) -> np.ndarray:
        """
        Transform raw observation inputs into the 32-feature vector required by the models.
        """
        self.validate_inputs(raw_input)

        row = {}

        # 1. Direct pollutant values at time t
        for p in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]:
            row[f"{p}_lag1"] = float(raw_input.get(p, 50.0))

        # 2. Weather features at time t
        for w in ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]:
            row[w] = float(raw_input.get(w, 0.0))

        # 3. Rolling proxies (if rolling window not provided, default to current lag1)
        row["PM2.5_roll3"] = float(raw_input.get("PM2.5_roll3", row["PM2.5_lag1"]))
        row["PM2.5_roll6"] = float(raw_input.get("PM2.5_roll6", row["PM2.5_lag1"]))
        row["PM2.5_roll12"] = float(raw_input.get("PM2.5_roll12", row["PM2.5_lag1"]))
        row["PM2.5_roll24"] = float(raw_input.get("PM2.5_roll24", row["PM2.5_lag1"]))

        # 4. Cyclical time encoding
        hour = int(raw_input.get("hour", 12))
        month = int(raw_input.get("month", 6))
        dow = int(raw_input.get("day_of_week", 2))

        row["hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
        row["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
        row["month_sin"] = float(np.sin(2 * np.pi * month / 12.0))
        row["month_cos"] = float(np.cos(2 * np.pi * month / 12.0))
        row["dow_sin"] = float(np.sin(2 * np.pi * dow / 7.0))
        row["dow_cos"] = float(np.cos(2 * np.pi * dow / 7.0))

        # 5. Wind direction one-hot encoding
        raw_wd = raw_input.get("wd")
        active_wd = str(raw_wd).strip().upper() if raw_wd is not None else "NE"
        if active_wd not in ALL_WIND_DIRECTIONS:
            active_wd = "NE"
        for wd in ALL_WIND_DIRECTIONS:
            row[f"wd_{wd}"] = 1.0 if wd == active_wd else 0.0

        # Construct vector in exact order
        feature_vector = np.array([row[col] for col in self.feature_names], dtype=np.float32).reshape(1, -1)
        return feature_vector

    def predict(
        self,
        raw_input: Dict[str, Any],
        model_name: str = "LightGBM",
        include_shap: bool = True,
        standard: str = "US_EPA"
    ) -> Dict[str, Any]:
        """
        Generate prediction and optional SHAP explanation for a single observation.
        """
        if model_name not in self.models:
            valid_models = list(self.models.keys())
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {valid_models}")

        model = self.models[model_name]
        X_vec = self.transform_single_input(raw_input)

        # Scale features if Linear Regression
        if model_name == "Linear Regression":
            if self.scaler is None:
                raise RuntimeError("Scaler not loaded for Linear Regression.")
            X_df = pd.DataFrame(X_vec, columns=self.feature_names)
            X_eval = self.scaler.transform(X_df)
        else:
            X_eval = X_vec

        pred_raw = float(model.predict(X_eval)[0])
        # AQI cannot be negative and realistically caps at 500
        predicted_aqi = float(np.round(np.clip(pred_raw, 0.0, 500.0), 1))

        # Category and Advisory
        category_info = get_aqi_category(predicted_aqi, standard=standard)

        response = {
            "predicted_aqi": predicted_aqi,
            "category": category_info["category"],
            "color": category_info["color"],
            "advisory": category_info["advisory"],
            "model_used": model_name,
            "standard": standard,
            "features_used": self.feature_names
        }

        # Compute SHAP explanation if requested and available
        if include_shap and self.shap_bundle is not None:
            try:
                explainer = self.shap_bundle["explainer"]
                base_value = float(self.shap_bundle["base_value"])
                # TreeExplainer works on unscaled features
                shap_res = explainer(X_vec)
                shap_vals = shap_res.values[0]

                nl_explanation = generate_natural_language_explanation(
                    prediction=predicted_aqi,
                    base_value=base_value,
                    feature_names=self.feature_names,
                    feature_values=X_vec[0],
                    shap_values=shap_vals,
                    top_k=3
                )
                nl_explanation["explainer_reference_model"] = "XGBoost"
                if model_name != "XGBoost":
                    nl_explanation["attribution_note"] = (
                        f"Note: Local SHAP feature attribution is referenced against the validated XGBoost "
                        f"explainability baseline model, matching the experimental research report."
                    )
                response["explanation"] = nl_explanation
            except Exception as e:
                response["explanation"] = {"error": f"Could not compute SHAP: {str(e)}"}

        return response


def get_default_sample_observation() -> Dict[str, Any]:
    """Return a realistic sample observation from Aotizhongxin station."""
    return {
        "PM2.5": 85.0,
        "PM10": 115.0,
        "SO2": 18.0,
        "NO2": 54.0,
        "CO": 1400.0,
        "O3": 45.0,
        "TEMP": 12.5,
        "PRES": 1018.0,
        "DEWP": -2.0,
        "RAIN": 0.0,
        "WSPM": 1.8,
        "wd": "NE",
        "hour": 14,
        "month": 11,
        "day_of_week": 3
    }


if __name__ == "__main__":
    predictor = AirQualityPredictor()
    sample = get_default_sample_observation()
    for m in ["LightGBM", "XGBoost", "Random Forest", "Linear Regression", "CatBoost"]:
        res = predictor.predict(sample, model_name=m, include_shap=(m == "XGBoost"))
        print(f"[{m}] Predicted AQI: {res['predicted_aqi']} | Category: {res['category']} | Color: {res['color']}")
        if "explanation" in res and "narrative" in res["explanation"]:
            print(f"  Narrative snippet: {res['explanation']['narrative'][:120]}...\n")
