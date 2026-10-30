"""
AIRWISE Feature Engineering and Dataset Partitioning Module
-----------------------------------------------------------
Implements chronological target creation (t+1 forecast), historical pollutant lags,
meteorological observations at time t, cyclical time encoding, wind one-hot encoding,
and strictly chronological train/validation/test partitioning.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.aqi_calculator import calculate_pm25_aqi

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
ALL_WIND_DIRECTIONS = [
    "E", "ENE", "ESE", "N", "NE", "NNE", "NNW", "NW",
    "S", "SE", "SSE", "SSW", "SW", "W", "WNW", "WSW"
]


def build_feature_dataset(
    df: pd.DataFrame,
    standard: str = "US_EPA",
    output_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Construct model features and target variable from cleaned data.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset containing datetime, pollutants, weather, and wind direction.
    standard : str
        'US_EPA' or 'CPCB'.
    output_dir : str
        Directory to save processed feature dataset and metadata.

    Returns:
    --------
    feat_df : pd.DataFrame
        Full dataset containing features, datetime, current AQI, and next-hour AQI_target.
    metadata : Dict[str, Any]
        Feature metadata, column sets, and configuration.
    """
    df = df.copy()

    # Step 1: Calculate current AQI sub-index
    df["AQI"] = calculate_pm25_aqi(df["PM2.5"], standard=standard)

    # Step 2: Target variable is strictly NEXT-HOUR AQI (t+1)
    # The model receives observations up to time t and predicts AQI at t+1.
    df["AQI_target"] = df["AQI"].shift(-1)

    # Step 3: Historical pollutant features at time t (lag1 relative to t+1 target)
    for p in POLLUTANTS:
        df[f"{p}_lag1"] = df[p]

    # Additional rolling mean feature over preceding 6 hours for PM2.5 persistence
    df["PM2.5_roll6"] = df["PM2.5"].rolling(window=6, min_periods=1).mean()

    # Step 4: Meteorological features available at time t
    # (TEMP, PRES, DEWP, RAIN, WSPM) already present in df

    # Step 5: Cyclical time features (sine & cosine transforms)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Step 6: Wind direction one-hot encoding with fixed columns
    for wd in ALL_WIND_DIRECTIONS:
        df[f"wd_{wd}"] = (df["wd"] == wd).astype(float)

    # The last row has NaN for target due to shift(-1)
    df_valid = df.dropna(subset=["AQI_target"]).reset_index(drop=True)

    # Define Feature Sets
    # Experiment A: Pollution Only + Cyclical Time
    pollution_features = [f"{p}_lag1" for p in POLLUTANTS] + ["PM2.5_roll6", "hour_sin", "hour_cos", "month_sin", "month_cos"]

    # Experiment B: Full Model (Pollution + Weather + Time + Wind)
    wind_features = [f"wd_{wd}" for wd in ALL_WIND_DIRECTIONS]
    full_features = pollution_features + WEATHER + wind_features

    metadata = {
        "standard": standard,
        "target_col": "AQI_target",
        "horizon_hours": 1,
        "experiment_a_features": pollution_features,
        "experiment_b_features": full_features,
        "weather_features": WEATHER,
        "wind_features": wind_features,
        "all_wind_directions": ALL_WIND_DIRECTIONS,
        "total_valid_records": len(df_valid)
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        feat_path = os.path.join(output_dir, "features_aotizhongxin.csv")
        df_valid.to_csv(feat_path, index=False)
        meta_path = os.path.join(output_dir, "feature_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

    return df_valid, metadata


def partition_dataset_chronological(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "AQI_target",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    scale_features: bool = True,
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Split time-series data chronologically into train, validation, and test sets.
    Fits StandardScaler strictly on training set.

    Returns:
    --------
    dict containing:
        'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test',
        'X_train_scaled', 'X_val_scaled', 'X_test_scaled',
        'scaler', 'split_indices', 'feature_cols'
    """
    n = len(df)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    scaler = None
    X_train_scaled = None
    X_val_scaled = None
    X_test_scaled = None

    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_cols,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=feature_cols,
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=feature_cols,
            index=X_test.index
        )

        if models_dir:
            os.makedirs(models_dir, exist_ok=True)
            joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    split_indices = {
        "train_range": (0, train_end - 1),
        "val_range": (train_end, val_end - 1),
        "test_range": (val_end, n - 1),
        "train_dates": (str(train_df["datetime"].min()), str(train_df["datetime"].max())),
        "val_dates": (str(val_df["datetime"].min()), str(val_df["datetime"].max())),
        "test_dates": (str(test_df["datetime"].min()), str(test_df["datetime"].max())),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df)
    }

    if models_dir:
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, "split_metadata.json"), "w") as f:
            json.dump(split_indices, f, indent=4)
        with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
            json.dump(feature_cols, f, indent=4)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "split_indices": split_indices,
        "feature_cols": feature_cols
    }


if __name__ == "__main__":
    from src.data_loader import load_station_data
    from src.data_cleaning import clean_air_quality_data

    raw = load_station_data()
    cleaned, _ = clean_air_quality_data(raw)
    feat_df, meta = build_feature_dataset(cleaned)
    splits = partition_dataset_chronological(feat_df, meta["experiment_b_features"])

    print("Feature dataset built:")
    print(f"Total records: {len(feat_df)}")
    print(f"Features in Exp B: {len(meta['experiment_b_features'])}")
    print(f"Train split: {splits['split_indices']['train_dates']} ({splits['split_indices']['train_samples']} samples)")
    print(f"Val split:   {splits['split_indices']['val_dates']} ({splits['split_indices']['val_samples']} samples)")
    print(f"Test split:  {splits['split_indices']['test_dates']} ({splits['split_indices']['test_samples']} samples)")
