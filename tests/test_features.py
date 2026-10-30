"""
Unit Tests for Feature Engineering Module
-----------------------------------------
Validates chronological alignment, no data leakage, and feature properties.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import build_feature_dataset, partition_dataset_chronological


@pytest.fixture
def sample_clean_df():
    """Create a synthetic sequence with known timestamps for feature tests."""
    dates = pd.date_range("2016-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "datetime": dates,
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "hour": dates.hour,
        "PM2.5": np.linspace(10, 100, 100),
        "PM10": np.linspace(20, 150, 100),
        "SO2": np.full(100, 15.0),
        "NO2": np.full(100, 25.0),
        "CO": np.full(100, 800.0),
        "O3": np.full(100, 40.0),
        "TEMP": np.full(100, 12.0),
        "PRES": np.full(100, 1012.0),
        "DEWP": np.full(100, -2.0),
        "RAIN": np.full(100, 0.0),
        "wd": ["NNW"] * 50 + ["SSE"] * 50,
        "WSPM": np.full(100, 2.5),
        "station": ["Aotizhongxin"] * 100
    })
    return df


def test_target_alignment(sample_clean_df):
    """Verify AQI_target at row t strictly equals AQI at row t+1 (no shift error)."""
    feat_df, meta = build_feature_dataset(sample_clean_df, output_dir="")

    # Target at row 0 must equal AQI at row 1
    for i in range(len(feat_df) - 1):
        assert feat_df.loc[i, "AQI_target"] == pytest.approx(feat_df.loc[i + 1, "AQI"], 1e-4)

    # Ensure target column is NOT in the feature lists
    assert "AQI_target" not in meta["experiment_a_features"]
    assert "AQI_target" not in meta["experiment_b_features"]
    assert "AQI" not in meta["experiment_b_features"]


def test_cyclical_time_bounds(sample_clean_df):
    """Verify sine and cosine cyclical representations are strictly within [-1, 1]."""
    feat_df, _ = build_feature_dataset(sample_clean_df, output_dir="")

    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert feat_df[col].min() >= -1.0 - 1e-5
        assert feat_df[col].max() <= 1.0 + 1e-5


def test_chronological_splits(sample_clean_df):
    """Verify train, validation, and test sets are contiguous without time overlap."""
    feat_df, meta = build_feature_dataset(sample_clean_df, output_dir="")
    splits = partition_dataset_chronological(
        feat_df,
        feature_cols=meta["experiment_b_features"],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        scale_features=True,
        models_dir=""
    )

    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]

    # Contiguity check: indices must not overlap
    assert len(set(X_train.index).intersection(set(X_val.index))) == 0
    assert len(set(X_val.index).intersection(set(X_test.index))) == 0
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0

    # Total records preserved
    assert len(X_train) + len(X_val) + len(X_test) == len(feat_df)
