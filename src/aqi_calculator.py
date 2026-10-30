"""
AIRWISE Air Quality Index (AQI) Calculation Module
--------------------------------------------------
Provides mathematically verified, standard-compliant computation of the
PM2.5-derived Air Quality Index (AQI) sub-index.

Official Standards Supported:
1. US EPA Standard:
   Source: US EPA 40 CFR Part 58, Appendix G - Technical Assistance Document
   for the Reporting of Daily Air Quality – the Air Quality Index (AQI).
2. CPCB India Standard:
   Source: Central Pollution Control Board (CPCB), Ministry of Environment,
   Forest and Climate Change, Government of India (2014 Report).

Formula:
   Ip = [(I_hi - I_lo) / (BP_hi - BP_lo)] * (Cp - BP_lo) + I_lo
Where:
   Ip    = AQI sub-index for pollutant p
   Cp    = Concentration of pollutant p
   BP_lo = Breakpoint concentration lower bound (<= Cp)
   BP_hi = Breakpoint concentration upper bound (>= Cp)
   I_lo  = AQI value corresponding to BP_lo
   I_hi  = AQI value corresponding to BP_hi
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, Any

# Breakpoint tables: (C_low, C_high, I_low, I_high, category, color, description)
US_EPA_BREAKPOINTS = [
    (0.0, 12.0, 0, 50, "Good", "#00E400", "Air quality is satisfactory, and air pollution poses little or no risk."),
    (12.1, 35.4, 51, 100, "Moderate", "#D4AC0D", "Air quality is acceptable; some pollutants may moderately affect sensitive individuals."),
    (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups", "#FF7E00", "Members of sensitive groups may experience health effects. General public is less likely to be affected."),
    (55.5, 150.4, 151, 200, "Unhealthy", "#FF0000", "Everyone may begin to experience health effects; sensitive groups may experience more serious effects."),
    (150.5, 250.4, 201, 300, "Very Unhealthy", "#8F3F97", "Health alert: The risk of health effects is increased for everyone."),
    (250.5, 350.4, 301, 400, "Hazardous", "#7E0023", "Health warning of emergency conditions: Everyone is more likely to be affected."),
    (350.5, 500.4, 401, 500, "Hazardous", "#7E0023", "Health warning of emergency conditions: Serious risk of respiratory effects.")
]

CPCB_BREAKPOINTS = [
    (0.0, 30.0, 0, 50, "Good", "#00E400", "Minimal impact on health."),
    (31.0, 60.0, 51, 100, "Satisfactory", "#92D050", "Minor breathing discomfort to sensitive people."),
    (61.0, 90.0, 101, 200, "Moderate", "#FFFF00", "Breathing discomfort to people with lungs, asthma and heart diseases."),
    (91.0, 120.0, 201, 300, "Poor", "#FF7E00", "Breathing discomfort to most people on prolonged exposure."),
    (121.0, 250.0, 301, 400, "Very Poor", "#FF0000", "Respiratory illness on prolonged exposure."),
    (250.1, 500.0, 401, 500, "Severe", "#7E0023", "Affects healthy people and seriously impacts those with existing diseases.")
]


def calculate_pm25_aqi_scalar(pm25: float, standard: str = "US_EPA") -> float:
    """
    Calculate the PM2.5-derived AQI sub-index for a single concentration value.

    Parameters:
    -----------
    pm25 : float
        PM2.5 concentration in ug/m^3.
    standard : str
        AQI standard: 'US_EPA' or 'CPCB'.

    Returns:
    --------
    float
        Calculated AQI value (0 to 500, or np.nan if input is NaN).
    """
    if pm25 is None or np.isnan(pm25):
        return np.nan

    # Non-negative constraint
    c = max(0.0, float(pm25))

    breakpoints = US_EPA_BREAKPOINTS if standard.upper() == "US_EPA" else CPCB_BREAKPOINTS

    # US EPA specifies rounding PM2.5 to 1 decimal place
    c_eval = np.round(c, 1)

    for bp_lo, bp_hi, i_lo, i_hi, _, _, _ in breakpoints:
        if bp_lo <= c_eval <= bp_hi:
            aqi = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (c_eval - bp_lo) + i_lo
            return float(np.round(aqi, 2))

    # Fallback to contiguous interval search if slightly outside due to rounding
    for i, (bp_lo, bp_hi, i_lo, i_hi, _, _, _) in enumerate(breakpoints):
        prev_hi = breakpoints[i - 1][1] if i > 0 else 0.0
        if prev_hi < c <= bp_hi:
            aqi = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (c - bp_lo) + i_lo
            return float(np.round(aqi, 2))

    # Values exceeding the maximum breakpoint (500 ug/m3)
    return 500.0


def calculate_pm25_aqi(
    pm25_data: Union[float, np.ndarray, pd.Series],
    standard: str = "US_EPA"
) -> Union[float, np.ndarray, pd.Series]:
    """
    Vectorized calculation of PM2.5 AQI sub-index.

    Parameters:
    -----------
    pm25_data : float, np.ndarray, or pd.Series
        PM2.5 concentration values.
    standard : str
        'US_EPA' or 'CPCB'.

    Returns:
    --------
    Same type as pm25_data containing the computed AQI sub-index values.
    """
    if isinstance(pm25_data, (int, float)):
        return calculate_pm25_aqi_scalar(pm25_data, standard=standard)

    is_series = isinstance(pm25_data, pd.Series)
    c_arr = np.asarray(pm25_data, dtype=float)
    aqi_arr = np.full_like(c_arr, np.nan)

    valid_mask = ~np.isnan(c_arr)
    c_valid = np.clip(c_arr[valid_mask], 0.0, None)

    # US EPA specifies rounding PM2.5 to nearest 0.1 ug/m3
    c_eval = np.round(c_valid, 1)

    breakpoints = US_EPA_BREAKPOINTS if standard.upper() == "US_EPA" else CPCB_BREAKPOINTS

    calculated = np.full_like(c_eval, np.nan)

    for bp_lo, bp_hi, i_lo, i_hi, _, _, _ in breakpoints:
        mask = (c_eval >= bp_lo) & (c_eval <= bp_hi)
        calculated[mask] = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (c_eval[mask] - bp_lo) + i_lo

    # Check any unassigned due to edge rounding gaps
    unassigned = np.isnan(calculated)
    if np.any(unassigned):
        for i, (bp_lo, bp_hi, i_lo, i_hi, _, _, _) in enumerate(breakpoints):
            prev_hi = breakpoints[i - 1][1] if i > 0 else 0.0
            mask = unassigned & (c_valid > prev_hi) & (c_valid <= bp_hi)
            calculated[mask] = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (c_valid[mask] - bp_lo) + i_lo

    # Cap above highest breakpoint
    max_bp = breakpoints[-1][1]
    calculated[c_valid > max_bp] = 500.0

    aqi_arr[valid_mask] = np.round(calculated, 2)

    if is_series:
        return pd.Series(aqi_arr, index=pm25_data.index, name="AQI")
    return aqi_arr


def get_aqi_category(aqi_value: float, standard: str = "US_EPA") -> Dict[str, Any]:
    """
    Determine the category, color, and health implications for a given AQI value.

    Parameters:
    -----------
    aqi_value : float
        Numerical AQI value.
    standard : str
        'US_EPA' or 'CPCB'.

    Returns:
    --------
    dict with keys: 'category', 'color', 'advisory', 'standard', 'aqi'
    """
    if aqi_value is None or np.isnan(aqi_value):
        return {
            "aqi": None,
            "category": "Unknown",
            "color": "#808080",
            "advisory": "No data available.",
            "standard": standard
        }

    aqi_val = float(aqi_value)
    breakpoints = US_EPA_BREAKPOINTS if standard.upper() == "US_EPA" else CPCB_BREAKPOINTS

    for _, _, i_lo, i_hi, category, color, advisory in breakpoints:
        if i_lo <= aqi_val <= i_hi:
            return {
                "aqi": aqi_val,
                "category": category,
                "color": color,
                "advisory": advisory,
                "standard": standard
            }

    if aqi_val > 500:
        last = breakpoints[-1]
        return {
            "aqi": aqi_val,
            "category": last[4],
            "color": last[5],
            "advisory": last[6],
            "standard": standard
        }

    first = breakpoints[0]
    return {
        "aqi": aqi_val,
        "category": first[4],
        "color": first[5],
        "advisory": first[6],
        "standard": standard
    }
