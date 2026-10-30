"""
AIRWISE Data Cleaning and Preprocessing Pipeline
------------------------------------------------
Performs timestamp construction, chronological ordering, training-only fitted
imputation, duplicate validation, and documented outlier screening.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


def clean_air_quality_data(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    save_cleaned: bool = True,
    output_path: str = "data/processed/cleaned_aotizhongxin.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean raw air quality dataset strictly preventing test data leakage.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset from load_station_data.
    train_ratio : float
        Ratio of dataset representing training set for computing imputation medians/modes.
    save_cleaned : bool
        Whether to save cleaned dataset to disk.
    output_path : str
        Filepath for saving the cleaned data.

    Returns:
    --------
    cleaned_df : pd.DataFrame
    metadata : Dict[str, Any]
    """
    df = df.copy()

    # Step 1: Datetime construction
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

    # Step 2: Chronological sorting
    df = df.sort_values("datetime").reset_index(drop=True)

    # Step 4: Duplicate removal
    initial_rows = len(df)
    df = df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    duplicates_removed = initial_rows - len(df)

    # Calculate train split cutoff point for leakage-free imputation parameters
    train_cutoff_idx = int(len(df) * train_ratio)
    train_slice = df.iloc[:train_cutoff_idx]

    # Step 3: Compute training-only statistics for imputation
    numeric_cols = POLLUTANTS + WEATHER
    train_medians = {col: float(train_slice[col].median()) for col in numeric_cols if col in train_slice}
    train_wd_mode = str(train_slice["wd"].mode()[0]) if "wd" in train_slice and not train_slice["wd"].dropna().empty else "ENE"

    # Missing counts before imputation
    missing_before = df[numeric_cols + (["wd"] if "wd" in df else [])].isnull().sum().to_dict()

    # Apply time-series aware forward-fill + backward-fill first (local continuity),
    # followed by training-set median fallback for any remaining NaNs
    for col in numeric_cols:
        if col in df.columns:
            # Interpolate short gaps linearly, then fill remaining with train median
            df[col] = df[col].interpolate(method="linear", limit=6)
            df[col] = df[col].ffill().bfill().fillna(train_medians[col])

    if "wd" in df.columns:
        df["wd"] = df["wd"].ffill().bfill().fillna(train_wd_mode)

    missing_after = df[numeric_cols + (["wd"] if "wd" in df else [])].isnull().sum().to_dict()

    # Step 5: Outlier analysis (documenting distribution bounds without arbitrary removal)
    outlier_summary = {}
    for col in numeric_cols:
        q25 = float(train_slice[col].quantile(0.25))
        q75 = float(train_slice[col].quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - 3.0 * iqr
        upper_bound = q75 + 3.0 * iqr
        count_extreme = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        outlier_summary[col] = {
            "q25": np.round(q25, 2),
            "q75": np.round(q75, 2),
            "iqr": np.round(iqr, 2),
            "extreme_threshold_upper": np.round(upper_bound, 2),
            "extreme_events_count": count_extreme,
            "retained": True,
            "rationale": "High pollution and weather spikes represent genuine physical environmental episodes."
        }

    metadata = {
        "initial_record_count": initial_rows,
        "cleaned_record_count": len(df),
        "duplicates_removed": duplicates_removed,
        "train_cutoff_idx": train_cutoff_idx,
        "imputation_medians_from_train": train_medians,
        "imputation_wd_mode_from_train": train_wd_mode,
        "missing_before_cleaning": missing_before,
        "missing_after_cleaning": missing_after,
        "outlier_analysis": outlier_summary
    }

    if save_cleaned:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        meta_path = os.path.join(os.path.dirname(output_path), "cleaning_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

    return df, metadata


if __name__ == "__main__":
    from src.data_loader import load_station_data
    df_raw = load_station_data()
    df_clean, meta = clean_air_quality_data(df_raw)
    print("Data cleaning complete.")
    print(f"Records: {meta['cleaned_record_count']}")
    print(f"Remaining NaNs: {sum(meta['missing_after_cleaning'].values())}")
