"""
Unit Tests for AQI Calculator Module
------------------------------------
Tests formula correctness, boundary values, NaN handling, and category mappings.
"""

import numpy as np
import pandas as pd
import pytest
from src.aqi_calculator import (
    calculate_pm25_aqi,
    calculate_pm25_aqi_scalar,
    get_aqi_category,
    US_EPA_BREAKPOINTS,
    CPCB_BREAKPOINTS
)


def test_epa_exact_boundaries():
    """Verify exact boundary values according to US EPA PM2.5 breakpoints."""
    assert calculate_pm25_aqi(0.0, standard="US_EPA") == 0.0
    assert calculate_pm25_aqi(12.0, standard="US_EPA") == 50.0
    assert calculate_pm25_aqi(12.1, standard="US_EPA") == 51.0
    assert calculate_pm25_aqi(35.4, standard="US_EPA") == 100.0
    assert calculate_pm25_aqi(35.5, standard="US_EPA") == 101.0
    assert calculate_pm25_aqi(55.4, standard="US_EPA") == 150.0
    assert calculate_pm25_aqi(55.5, standard="US_EPA") == 151.0
    assert calculate_pm25_aqi(150.4, standard="US_EPA") == 200.0
    assert calculate_pm25_aqi(150.5, standard="US_EPA") == 201.0
    assert calculate_pm25_aqi(250.4, standard="US_EPA") == 300.0
    assert calculate_pm25_aqi(250.5, standard="US_EPA") == 301.0
    assert calculate_pm25_aqi(350.4, standard="US_EPA") == 400.0
    assert calculate_pm25_aqi(350.5, standard="US_EPA") == 401.0
    assert calculate_pm25_aqi(500.4, standard="US_EPA") == 500.0


def test_cpcb_exact_boundaries():
    """Verify exact boundary values according to India CPCB PM2.5 breakpoints."""
    assert calculate_pm25_aqi(0.0, standard="CPCB") == 0.0
    assert calculate_pm25_aqi(30.0, standard="CPCB") == 50.0
    assert calculate_pm25_aqi(60.0, standard="CPCB") == 100.0
    assert calculate_pm25_aqi(90.0, standard="CPCB") == 200.0
    assert calculate_pm25_aqi(120.0, standard="CPCB") == 300.0
    assert calculate_pm25_aqi(250.0, standard="CPCB") == 400.0
    assert calculate_pm25_aqi(500.0, standard="CPCB") == 500.0


def test_edge_cases():
    """Verify negative, extreme, and NaN handling."""
    # Negative concentration should be clipped to 0
    assert calculate_pm25_aqi(-15.0, standard="US_EPA") == 0.0
    assert calculate_pm25_aqi(-5.0, standard="CPCB") == 0.0

    # Values exceeding 500 ug/m3 capped at 500.0
    assert calculate_pm25_aqi(650.0, standard="US_EPA") == 500.0
    assert calculate_pm25_aqi(800.0, standard="CPCB") == 500.0

    # NaN / None handling
    assert np.isnan(calculate_pm25_aqi(np.nan, standard="US_EPA"))
    assert np.isnan(calculate_pm25_aqi(None, standard="US_EPA"))


def test_vectorized_series():
    """Verify pd.Series and numpy array vectorization matches scalar calculations."""
    data = [0.0, 10.0, 25.0, 50.0, 120.0, np.nan, 200.0]
    series = pd.Series(data)

    res_series = calculate_pm25_aqi(series, standard="US_EPA")
    assert isinstance(res_series, pd.Series)
    assert len(res_series) == len(data)

    for orig, computed in zip(data, res_series):
        if np.isnan(orig):
            assert np.isnan(computed)
        else:
            scalar_val = calculate_pm25_aqi_scalar(orig, standard="US_EPA")
            assert pytest.approx(computed, 0.01) == scalar_val


def test_category_lookup():
    """Verify categories and color indicators."""
    cat_good = get_aqi_category(25.0, standard="US_EPA")
    assert cat_good["category"] == "Good"
    assert cat_good["color"] == "#00E400"

    cat_mod = get_aqi_category(75.0, standard="US_EPA")
    assert cat_mod["category"] == "Moderate"

    cat_sens = get_aqi_category(125.0, standard="US_EPA")
    assert cat_sens["category"] == "Unhealthy for Sensitive Groups"

    cat_unhealthy = get_aqi_category(175.0, standard="US_EPA")
    assert cat_unhealthy["category"] == "Unhealthy"

    cat_very_unhealthy = get_aqi_category(250.0, standard="US_EPA")
    assert cat_very_unhealthy["category"] == "Very Unhealthy"

    cat_hazardous = get_aqi_category(450.0, standard="US_EPA")
    assert cat_hazardous["category"] == "Hazardous"

    cat_nan = get_aqi_category(np.nan)
    assert cat_nan["category"] == "Unknown"
