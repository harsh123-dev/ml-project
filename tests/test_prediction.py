"""
Unit Tests for AIRWISE Prediction Pipeline and Inference Service
"""

# Import lightgbm first for Windows C-runtime stability
import lightgbm as lgb
import os
import sys
import pytest
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prediction import AirQualityPredictor, get_default_sample_observation


@pytest.fixture(scope="module")
def predictor():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    shap_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "shap")
    return AirQualityPredictor(models_dir=models_dir, shap_dir=shap_dir)


def test_predictor_initialization(predictor):
    assert predictor is not None
    assert len(predictor.models) == 5
    assert len(predictor.feature_names) == 32
    assert predictor.scaler is not None


def test_prediction_valid_sample(predictor):
    sample = get_default_sample_observation()
    for model_name in ["LightGBM", "XGBoost", "Random Forest", "Linear Regression", "CatBoost"]:
        res = predictor.predict(sample, model_name=model_name, include_shap=False)
        assert "predicted_aqi" in res
        assert "category" in res
        assert "color" in res
        assert "advisory" in res
        assert 0.0 <= res["predicted_aqi"] <= 500.0
        assert res["category"] in [
            "Good", "Moderate", "Unhealthy for Sensitive Groups",
            "Unhealthy", "Very Unhealthy", "Hazardous"
        ]


def test_prediction_with_shap(predictor):
    sample = get_default_sample_observation()
    res = predictor.predict(sample, model_name="XGBoost", include_shap=True)
    assert "explanation" in res
    expl = res["explanation"]
    assert "narrative" in expl
    assert "pushing_up" in expl
    assert "pulling_down" in expl
    assert len(expl["narrative"]) > 50


def test_input_validation_negative_pollutant(predictor):
    bad_sample = get_default_sample_observation()
    bad_sample["PM2.5"] = -10.0
    with pytest.raises(ValueError, match="outside valid physical range"):
        predictor.predict(bad_sample)


def test_input_validation_extreme_weather(predictor):
    bad_sample = get_default_sample_observation()
    bad_sample["PRES"] = 500.0  # Unrealistic atmospheric pressure
    with pytest.raises(ValueError, match="outside valid atmospheric range"):
        predictor.predict(bad_sample)


def test_input_validation_invalid_wind_direction(predictor):
    bad_sample = get_default_sample_observation()
    bad_sample["wd"] = "INVALID_DIRECTION"
    with pytest.raises(ValueError, match="Wind direction"):
        predictor.predict(bad_sample)


def test_edge_case_zero_concentrations(predictor):
    zero_sample = {
        "PM2.5": 0.0, "PM10": 0.0, "SO2": 0.0, "NO2": 0.0, "CO": 0.0, "O3": 0.0,
        "TEMP": 20.0, "PRES": 1013.25, "DEWP": 10.0, "RAIN": 0.0, "WSPM": 5.0,
        "wd": "N", "hour": 12, "month": 6, "day_of_week": 1
    }
    res = predictor.predict(zero_sample, model_name="LightGBM")
    assert 0.0 <= res["predicted_aqi"] <= 500.0


def test_edge_case_extreme_smog(predictor):
    smog_sample = {
        "PM2.5": 600.0, "PM10": 800.0, "SO2": 150.0, "NO2": 200.0, "CO": 8000.0, "O3": 20.0,
        "TEMP": -5.0, "PRES": 1030.0, "DEWP": -10.0, "RAIN": 0.0, "WSPM": 0.5,
        "wd": "SSE", "hour": 22, "month": 12, "day_of_week": 4
    }
    res = predictor.predict(smog_sample, model_name="LightGBM")
    assert res["predicted_aqi"] >= 200.0  # Should be Very Unhealthy or Hazardous
    assert res["category"] in ["Very Unhealthy", "Hazardous"]
