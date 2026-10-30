"""
AIRWISE Data Loading and Integrity Verification Module
------------------------------------------------------
Handles discovery, loading, and integrity audits for the UCI Beijing Multi-Site
Air Quality Dataset (Aotizhongxin primary station).
"""

import os
import glob
import json
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

UCI_URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"


def ensure_dataset(raw_dir: str = "data/raw") -> str:
    """
    Ensure the Beijing dataset files are present on disk.
    If missing, downloads and extracts them automatically.

    Returns:
    --------
    str: Path to the directory containing station CSVs.
    """
    os.makedirs(raw_dir, exist_ok=True)

    # Check if CSV files are already extracted
    csv_files = glob.glob(os.path.join(raw_dir, "**", "PRSA_Data_*.csv"), recursive=True)
    if csv_files:
        return os.path.dirname(csv_files[0])

    print("Dataset not detected locally. Downloading official UCI Beijing dataset...")
    zip_dest = os.path.join(raw_dir, "beijing_air_quality.zip")

    try:
        urllib.request.urlretrieve(UCI_URL, zip_dest)
        print(f"Downloaded ZIP file ({os.path.getsize(zip_dest)} bytes). Extracting...")
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(raw_dir)

        inner_zip = os.path.join(raw_dir, "PRSA2017_Data_20130301-20170228.zip")
        if os.path.exists(inner_zip):
            with zipfile.ZipFile(inner_zip, "r") as zf_inner:
                zf_inner.extractall(raw_dir)

        csv_files = glob.glob(os.path.join(raw_dir, "**", "PRSA_Data_*.csv"), recursive=True)
        if csv_files:
            return os.path.dirname(csv_files[0])
    except Exception as e:
        print(f"Automatic download encountered an issue: {e}")
        print("Please ensure the CSV files are placed in data/raw/")

    return raw_dir


def load_station_data(
    station_name: str = "Aotizhongxin",
    raw_dir: str = "data/raw"
) -> pd.DataFrame:
    """
    Load raw hourly observations for a specific monitoring station.

    Parameters:
    -----------
    station_name : str
        Target monitoring station (default: 'Aotizhongxin').
    raw_dir : str
        Directory containing raw data files.

    Returns:
    --------
    pd.DataFrame: Raw dataframe for the selected station.
    """
    folder = ensure_dataset(raw_dir)
    pattern = os.path.join(folder, f"*PRSA_Data_{station_name}_*.csv")
    matches = glob.glob(pattern)

    if not matches:
        # Recursive fallback
        pattern_rec = os.path.join(raw_dir, "**", f"*PRSA_Data_{station_name}_*.csv")
        matches = glob.glob(pattern_rec, recursive=True)

    if not matches:
        raise FileNotFoundError(
            f"No raw dataset file found for station '{station_name}' in '{raw_dir}'."
        )

    filepath = matches[0]
    df = pd.read_csv(filepath)
    return df


def audit_dataset_integrity(
    station_name: str = "Aotizhongxin",
    raw_dir: str = "data/raw",
    output_metrics_dir: str = "outputs/metrics"
) -> Dict[str, Any]:
    """
    Compute comprehensive dataset statistics and integrity metrics.
    Saves JSON and CSV summaries.
    """
    os.makedirs(output_metrics_dir, exist_ok=True)
    folder = ensure_dataset(raw_dir)

    all_csvs = glob.glob(os.path.join(folder, "*PRSA_Data_*.csv"))
    if not all_csvs:
        all_csvs = glob.glob(os.path.join(raw_dir, "**", "*PRSA_Data_*.csv"), recursive=True)

    total_records_all_stations = 0
    stations_found = []
    for f in all_csvs:
        sub_df = pd.read_csv(f, usecols=["station"])
        total_records_all_stations += len(sub_df)
        stations_found.extend(sub_df["station"].unique().tolist())
    stations_found = sorted(list(set(stations_found)))

    # Selected station
    df_station = load_station_data(station_name=station_name, raw_dir=raw_dir)
    selected_count = len(df_station)

    # Missing counts
    missing_before = df_station.isnull().sum().to_dict()
    total_missing_cells = int(df_station.isnull().sum().sum())

    # Date range
    df_station["dt"] = pd.to_datetime(df_station[["year", "month", "day", "hour"]])
    start_date = str(df_station["dt"].min())
    end_date = str(df_station["dt"].max())
    duplicates_count = int(df_station.duplicated(subset=["dt"]).sum())

    # Range / Invalid check (negative pollutant values)
    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    negative_values = {}
    for p in pollutants:
        if p in df_station.columns:
            negative_values[p] = int((df_station[p] < 0).sum())

    stats = {
        "dataset_name": "Beijing Multi-Site Air Quality Dataset (UCI)",
        "source_url": UCI_URL,
        "total_monitoring_stations": len(stations_found),
        "monitoring_stations_list": stations_found,
        "total_records_all_stations": total_records_all_stations,
        "selected_station": station_name,
        "selected_station_records": selected_count,
        "date_range_start": start_date,
        "date_range_end": end_date,
        "duplicate_timestamps": duplicates_count,
        "total_missing_values_before_cleaning": total_missing_cells,
        "missing_values_by_column": missing_before,
        "negative_values_by_pollutant": negative_values
    }

    # Save to JSON
    json_path = os.path.join(output_metrics_dir, "dataset_statistics.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=4)

    # Save missing summary to CSV
    missing_df = pd.DataFrame({
        "Column": list(missing_before.keys()),
        "MissingCount": list(missing_before.values()),
        "MissingPct": [np.round((v / selected_count) * 100, 2) for v in missing_before.values()]
    })
    missing_df.to_csv(os.path.join(output_metrics_dir, "missing_values_summary.csv"), index=False)

    return stats


if __name__ == "__main__":
    audit = audit_dataset_integrity()
    print("Dataset audit complete:")
    print(f"Total stations: {audit['total_monitoring_stations']}")
    print(f"Total records: {audit['total_records_all_stations']}")
    print(f"Selected station: {audit['selected_station']} ({audit['selected_station_records']} records)")
    print(f"Date range: {audit['date_range_start']} to {audit['date_range_end']}")
