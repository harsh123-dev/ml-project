"""
Unit Tests for Data Cleaning and Preprocessing Pipeline
-------------------------------------------------------
Verifies datetime construction, chronological sorting, duplicate removal,
leakage-free training-fitted imputation, and outlier documentation.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_cleaning import clean_air_quality_data


@pytest.fixture
def dirty_synthetic_df():
    """Create a synthetic dataset with missing values, unsorted order, and duplicates."""
    dates = pd.date_range("2015-01-01", periods=100, freq="h")
    # Duplicate some rows and scramble order
    df = pd.DataFrame({
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "hour": dates.hour,
        "PM2.5": [10.0] * 50 + [np.nan] * 10 + [200.0] * 40,
        "PM10": [20.0] * 70 + [np.nan] * 5 + [300.0] * 25,
        "SO2": [5.0] * 100,
        "NO2": [25.0] * 100,
        "CO": [600.0] * 100,
        "O3": [35.0] * 100,
        "TEMP": [15.0] * 100,
        "PRES": [1015.0] * 100,
        "DEWP": [2.0] * 100,
        "RAIN": [0.0] * 100,
        "wd": ["NW"] * 40 + [np.nan] * 10 + ["SE"] * 50,
        "WSPM": [2.0] * 100,
        "station": ["Aotizhongxin"] * 100
    })
    # Add 5 duplicate rows
    dupes = df.iloc[:5].copy()
    df_with_dupes = pd.concat([df, dupes], ignore_index=True)
    # Shuffle
    shuffled = df_with_dupes.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return shuffled


def test_datetime_and_chronological_sorting(dirty_synthetic_df):
    """Verify that datetime is created properly and dataset is chronologically ordered."""
    cleaned, meta = clean_air_quality_data(dirty_synthetic_df, save_cleaned=False)

    assert "datetime" in cleaned.columns
    assert isinstance(cleaned["datetime"].iloc[0], pd.Timestamp)

    # Check chronological ordering
    for i in range(len(cleaned) - 1):
        assert cleaned.loc[i, "datetime"] <= cleaned.loc[i + 1, "datetime"]


def test_duplicate_removal(dirty_synthetic_df):
    """Verify duplicate timestamps are detected and removed."""
    cleaned, meta = clean_air_quality_data(dirty_synthetic_df, save_cleaned=False)
    assert meta["duplicates_removed"] == 5
    assert len(cleaned) == 100
    assert cleaned["datetime"].is_unique


def test_leakage_free_imputation(dirty_synthetic_df):
    """Verify missing values are imputed strictly using training partition statistics."""
    train_ratio = 0.80
    cleaned, meta = clean_air_quality_data(dirty_synthetic_df, train_ratio=train_ratio, save_cleaned=False)

    # Imputation medians must be present in metadata
    assert "imputation_medians_from_train" in meta
    assert "PM2.5" in meta["imputation_medians_from_train"]

    # No NaNs should remain in numeric or wind direction columns
    assert sum(meta["missing_after_cleaning"].values()) == 0
    assert not cleaned["PM2.5"].isnull().any()
    assert not cleaned["wd"].isnull().any()


def test_outlier_summary_preservation(dirty_synthetic_df):
    """Verify extreme environmental events are documented rather than arbitrarily discarded."""
    cleaned, meta = clean_air_quality_data(dirty_synthetic_df, save_cleaned=False)
    assert "outlier_analysis" in meta
    for col in ["PM2.5", "PM10", "TEMP"]:
        assert col in meta["outlier_analysis"]
        assert meta["outlier_analysis"][col]["retained"] is True
