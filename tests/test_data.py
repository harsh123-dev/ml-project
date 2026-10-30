"""
Unit Tests for Data Loading Module
----------------------------------
Verifies dataset presence, column completeness, station filtering, and audit statistics.
"""

import os
import json
import pytest
import pandas as pd
from src.data_loader import load_station_data, audit_dataset_integrity, ensure_dataset


def test_ensure_dataset():
    folder = ensure_dataset("data/raw")
    assert os.path.isdir(folder)


def test_load_station_data():
    df = load_station_data("Aotizhongxin", "data/raw")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 35064

    expected_cols = [
        "year", "month", "day", "hour",
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM", "station"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_audit_dataset_integrity():
    stats = audit_dataset_integrity("Aotizhongxin", "data/raw", "outputs/metrics")
    assert stats["total_monitoring_stations"] == 12
    assert stats["total_records_all_stations"] == 420768
    assert stats["selected_station"] == "Aotizhongxin"
    assert stats["selected_station_records"] == 35064
    assert os.path.exists("outputs/metrics/dataset_statistics.json")
    assert os.path.exists("outputs/metrics/missing_values_summary.csv")
