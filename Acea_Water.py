# Acea Water Prediction — unified pipeline (Google Cloud /content ready)
# Based on Kaggle's Acea Water Prediction challenge (Acea Group).
# This pipeline standardizes multiple hydrological datasets (aquifers, rivers, lakes, springs),
# applies preprocessing, and trains predictive models to forecast water levels and flows.

import os
import re
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Configure base path
BASE_PATHS = ["/content/acea-water-prediction", "/content"]
ARTIFACT_DIR = "/content/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

EXPECTED_FILES = [
    "Aquifer_Auser.csv",
    "Aquifer_Petrignano.csv",
    "Aquifer_Doganella.csv",
    "Aquifer_Luco.csv",
    "Water_Spring_Amiata.csv",
    "Water_Spring_Madonna_di_Canneto.csv",
    "Water_Spring_Lupa.csv",
    "River_Arno.csv",
    "Lake_Bilancino.csv",
]

# Dataset configuration
DATASETS_CFG = {
    "Aquifer_Auser.csv": {
        "type": "aquifer",
        "rename_map": {
            "Date": "date",
            "Rainfall_Gallicano": "rainfall_gallicano",
            "Rainfall_Pontetetto": "rainfall_pontetetto",
            "Rainfall_Monte_Serra": "rainfall_monteserra",
            "Rainfall_Orentano": "rainfall_orentano",
            "Rainfall_Borgo_a_Mozzano": "rainfall_borgo",
            "Rainfall_Piaggione": "rainfall_piaggione",
            "Rainfall_Calavorno": "rainfall_calavorno",
            "Rainfall_Croce_Arcana": "rainfall_crocearcana",
            "Rainfall_Tereglio_Coreglia_Antelminelli": "rainfall_tereglio",
            "Rainfall_Fabbriche_di_Vallico": "rainfall_vallico",
            "Depth_to_Groundwater_SAL": "depth_to_groundwater",
            "Depth_to_Groundwater_LT2": "well_lt2",
            "Depth_to_Groundwater_PAG": "well_pag",
            "Depth_to_Groundwater_CoS": "well_cos",
            "Depth_to_Groundwater_DIEC": "well_diec",
            "Temperature_Orentano": "temp_orentano",
            "Temperature_Monte_Serra": "temp_monteserra",
            "Temperature_Ponte_a_Moriano": "temp_moriano",
            "Temperature_Lucca_Orto_Botanico": "temp_lucca",
            "Volume_POL": "volume_pol",
            "Volume_CC1": "volume_cc1",
            "Volume_CC2": "volume_cc2",
            "Volume_CSA": "volume_csa",
            "Volume_CSAL": "volume_csal",
            "Hydrometry_Monte_S_Quirico": "hydrometry_quirico",
            "Hydrometry_Piaggione": "hydrometry_piaggione",
        },
        "target": "depth_to_groundwater",
        "require_cols": ["date", "depth_to_groundwater"],
        "use_cols": [
            "date", "depth_to_groundwater",
            "rainfall_gallicano", "rainfall_pontetetto", "rainfall_monteserra",
            "rainfall_orentano", "rainfall_borgo", "rainfall_piaggione",
            "rainfall_calavorno", "rainfall_crocearcana", "rainfall_tereglio", "rainfall_vallico",
            "temp_orentano", "temp_monteserra", "temp_moriano", "temp_lucca",
            "volume_pol", "volume_cc1", "volume_cc2", "volume_csa", "volume_csal",
            "hydrometry_quirico", "hydrometry_piaggione",
            "well_lt2", "well_pag", "well_cos", "well_diec"
        ],
    },
    "Aquifer_Petrignano.csv": {
        "type": "aquifer",
        "rename_map": {
            "Date": "date",
            "Rainfall_Bastia_Umbra": "rainfall",
            "Depth_to_Groundwater_P24": "depth_p24",
            "Depth_to_Groundwater_P25": "depth_to_groundwater",
            "Temperature_Bastia_Umbra": "temperature",
            "Temperature_Petrignano": "temperature_petrignano",
            "Volume_C10_Petrignano": "drainage_volume",
            "Hydrometry_Fiume_Chiascio_Petrignano": "river_hydrometry",
        },
        "target": "depth_to_groundwater",
        "require_cols": ["date", "depth_to_groundwater"],
        "use_cols": ["date", "rainfall", "temperature", "drainage_volume", "river_hydrometry", "depth_to_groundwater"],
    },
    "Aquifer_Doganella.csv": {
        "type": "aquifer",
        "rename_map": {
            "Date": "date",
            "Rainfall_Monte_Castello": "rainfall",
            "Temperature_Monte_Castello": "temperature",
            "Depth_to_Groundwater_Pozzo_1": "well_1",
            "Depth_to_Groundwater_Pozzo_2": "well_2",
            "Depth_to_Groundwater_Pozzo_3": "well_3",
            "Depth_to_Groundwater_Pozzo_4": "well_4",
            "Depth_to_Groundwater_Pozzo_5": "well_5",
            "Depth_to_Groundwater_Pozzo_6": "well_6",
            "Depth_to_Groundwater_Pozzo_7": "well_7",
            "Depth_to_Groundwater_Pozzo_8": "well_8",
            "Depth_to_Groundwater_Pozzo_9": "depth_to_groundwater",
            "Volume_Cumulative": "drainage_volume",
        },
        "target": "depth_to_groundwater",
        "require_cols": ["date", "depth_to_groundwater"],
        "use_cols": [
            "date", "rainfall", "temperature", "drainage_volume",
            "well_1", "well_2", "well_3", "well_4", "well_5", "well_6", "well_7", "well_8", "depth_to_groundwater"
        ],
    },
    "Aquifer_Luco.csv": {
        "type": "aquifer",
        "rename_map": {
            "Date": "date",
            "Rainfall_Pieve_di_Santo_Stefano": "rainfall",
            "Temperature_Pieve_di_Santo_Stefano": "temperature",
            "Depth_to_Groundwater_Podere_Casetta": "depth_to_groundwater",
            "Volume_Cumulative": "drainage_volume",
        },
        "target": "depth_to_groundwater",
        "require_cols": ["date", "depth_to_groundwater"],
        "use_cols": ["date", "rainfall", "temperature", "drainage_volume", "depth_to_groundwater"],
    },
    "Water_Spring_Amiata.csv": {
        "type": "spring",
        "rename_map": {
            "Date": "date",
            "Rainfall_Mount_Amiata": "rainfall",
            "Temperature_Mount_Amiata": "temperature",
            "Depth_to_Groundwater_SGA": "depth_to_groundwater",
            "Hydrometry_Albegna": "river_hydrometry",
            "Volume_Cumulative": "drainage_volume",
            "Flow_Rate_Bugnano": "flow_bugnano",
            "Flow_Rate_Arbure": "flow_arbure",
            "Flow_Rate_Ermicciolo": "flow_ermicciolo",
            "Flow_Rate_Galleria_Alta": "flow_galleria_alta",
        },
        "target": "flow_ermicciolo",
        "require_cols": ["date", "flow_ermicciolo"],
        "use_cols": ["date", "rainfall", "temperature", "depth_to_groundwater", "river_hydrometry", "drainage_volume", "flow_ermicciolo"],
    },
    "Water_Spring_Madonna_di_Canneto.csv": {
        "type": "spring",
        "rename_map": {
            "Date": "date",
            "Rainfall_Settefrati": "rainfall",
            "Temperature_Settefrati": "temperature",
            "Flow_Rate_Madonna_di_Canneto": "flow_rate",
        },
        "target": "flow_rate",
        "require_cols": ["date", "flow_rate"],
        "use_cols": ["date", "rainfall", "temperature", "flow_rate"],
    },
    "Water_Spring_Lupa.csv": {
        "type": "spring",
        "rename_map": {
            "Date": "date",
            "Rainfall_Terni": "rainfall",
            "Flow_Rate_Lupa": "flow_rate",
        },
        "target": "flow_rate",
        "require_cols": ["date", "flow_rate"],
        "use_cols": ["date", "rainfall", "flow_rate"],
    },
    "River_Arno.csv": {
        "type": "river",
        "rename_map": {
            "Date": "date",
            "Hydrometry_Nave_di_Rosano": "hydrometry",
        },
        "target": "hydrometry",
        "require_cols": ["date", "hydrometry"],
        "use_cols": ["date", "hydrometry", "rainfall"],
    },
    "Lake_Bilancino.csv": {
        "type": "lake",
        "rename_map": {
            "Date": "date",
            "Rainfall_Mugello": "rainfall",
            "Temperature_Mugello": "temperature",
            "Lake_Level": "lake_level",
            "Lake_Outflow": "lake_outflow",
        },
        "target": "lake_level",
        "require_cols": ["date", "lake_level"],
        "use_cols": ["date", "rainfall", "temperature", "lake_level", "lake_outflow"],
    },
}

# Lupa-specific preprocessing
def lupa_replace_nonnegatives_with_daymonth_mean(df):
    """
    For Water_Spring_Lupa.csv:
    Replace every non-negative flow_rate value with the average of the same day/month across all years.
    Negative values remain unchanged.
    """
    df = df.copy()
    if "date" not in df.columns or "flow_rate" not in df.columns:
        return df

    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Compute reference means by (month, day), ignoring NaNs
    ref = (
        df.groupby(["month", "day"])["flow_rate"]
          .mean()
          .rename("ref_mean")
          .reset_index()
    )
    df = df.merge(ref, on=["month", "day"], how="left")

    # Replace only non-negative values
    mask = df["flow_rate"].notna() & (df["flow_rate"] >= 0)
    df.loc[mask, "flow_rate"] = df.loc[mask, "ref_mean"]

    df = df.drop(columns=["ref_mean", "month", "day"])
    return df

def preprocess_lupa(df):
    """
    Apply Lupa-specific preprocessing:
    - Replace non-negative flow_rate values with day/month averages across years.
    """
    return lupa_replace_nonnegatives_with_daymonth_mean(df)

# Main runner
def run_one(name, cfg, path):
    print(f"\n=== Processing {name} ({cfg['type']}) ===")
    df = pd.read_csv(path)

    # Standardize
    df = robust_rename(df, cfg["rename_map"])
    df = coalesce_station_columns(df)
    df = parse_and_sort_date(df)
    planned = [c for c in cfg["use_cols"] if c in df.columns]
    if planned:
        df = df[planned].copy()

    # Drop rows missing required columns
    req = [c for c in cfg["require_cols"] if c in df.columns]
    if req:
        df = df.dropna(subset=req)

    # Apply Lupa-specific preprocessing
    if name == "Water_Spring_Lupa.csv":
        df = preprocess_lupa(df)

    # Forward-fill exogenous
    df = forward_fill_exogenous(df, cfg["target"])

    # Clip outliers on exogenous + derived features (exclude date/target)
    df = clip_outliers(df, exclude_cols=["date", cfg["target"]], zmax=4.0)

    # Time features
    df = add_time_features(df)

    # Lags and rolling windows
    exo_for_lags = [c for c in ["rainfall", "temperature", "drainage_volume", "river_hydrometry", "lake_outflow"] if c in df.columns]
    df = add_lag_rolling(df, cfg["target"], exo_for_lags, lags=(1,3,7,14,30), windows=(3,7,14,30))

    # Train/test split
    train_df, test_df = split_train_test(df, cfg["target"], test_frac=0.2, min_train=365)

    # Select features and drop NaNs from lag/rolling
    feat_cols = select_numeric_features(train_df, cfg["target"])
    train_df = train_df.dropna(subset=feat_cols + [cfg["target"]]).copy()
    test_df = test_df.dropna(subset=feat_cols + [cfg["target"]]).copy()

    if len(train_df) < 200 or len(test_df) < 30:
        print(f"[{name}] Warning: small train/test after cleaning (train={len(train_df)}, test={len(test_df)}).")

    # Train and evaluate
    X_train, y_train = train_df[feat_cols], train_df[cfg["target"]]
    X_test, y_test = test_df[feat_cols], test_df[cfg["target"]]

    model = choose_model(cfg["type"])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print(f"[{name}] MAE={mae:.4f} | RMSE={rmse:.4f}")

def main():
    print("Starting Acea Water Prediction pipeline (Google Cloud /content)...")
    found_map = list_found_files()

    for expected_name, cfg in DATASETS_CFG.items():
        path = found_map.get(expected_name)
        if not path:
            print(f"[{expected_name}] Skipped: file not found in {BASE_PATHS}.")
            continue
        try:
            run_one(expected_name, cfg, path)
        except Exception as e:
            print(f"[{expected_name}] Error during processing: {e}")

if __name__ == "__main__":
    main()
    