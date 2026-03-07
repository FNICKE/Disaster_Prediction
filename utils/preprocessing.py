"""
utils/preprocessing.py
-----------------------
Data loading, feature engineering, and preprocessing for the Flood Prediction System.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# ─── Feature column definitions ───────────────────────────────────────────────
FEATURE_COLUMNS = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement",
    "Deforestation", "Urbanization", "ClimateChange", "DamsQuality",
    "Siltation", "AgriculturalPractices", "Encroachments",
    "IneffectiveDisasterPreparedness", "DrainageSystems",
    "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors",
]
TARGET_COLUMN = "FloodProbability"


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-driven interaction features that boost model accuracy.
    These capture non-linear risk combinations without blowing up dimensionality.
    """
    df = df.copy()

    # --- Interaction features ---
    # High monsoon + poor drainage = compounding risk
    df["Monsoon_x_Drainage"] = df["MonsoonIntensity"] * df["TopographyDrainage"]

    # Urbanisation + poor river mgmt = flash-flood amplifier
    df["Urban_x_RiverMgmt"] = df["Urbanization"] * df["RiverManagement"]

    # Climate change + deforestation (long-term erosion risk)
    df["Climate_x_Deforest"] = df["ClimateChange"] * df["Deforestation"]

    # Aggregate "Governance" score: planning + political
    df["GovernanceRisk"] = (df["InadequatePlanning"] + df["PoliticalFactors"]) / 2.0

    # Aggregate "Infrastructure" score: dams + drainage + infrastructure
    df["InfraRisk"] = (df["DamsQuality"] + df["DrainageSystems"] + df["DeterioratingInfrastructure"]) / 3.0

    # Aggregate "Environmental" score
    df["EnvRisk"] = (df["Deforestation"] + df["WetlandLoss"] + df["Siltation"]) / 3.0

    return df


def load_and_preprocess(
    file_path: str,
    test_size: float = 0.20,
    random_state: int = 42,
    threshold: float = 0.5,
    models_dir: str = "models",
):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    file_path    : path to flood.csv
    test_size    : fraction for the test split
    random_state : reproducibility seed
    threshold    : binarisation threshold for FloodProbability
    models_dir   : directory where scaler & feature names are saved

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    scaler                            : fitted StandardScaler (also saved to disk)
    feature_names                     : list[str] – columns used by the model
    """
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(file_path)
    print(f"[Data]  Loaded  : {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Keep only expected feature columns + target (drop extras like 'id' if present)
    keep_cols = [c for c in FEATURE_COLUMNS if c in df.columns] + [TARGET_COLUMN]
    df = df[keep_cols].copy()

    # 2. Clean ────────────────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna()
    if len(df) < before:
        print(f"[Data]  Dropped {before - len(df)} rows with NaN values.")

    # 3. Binarise target ──────────────────────────────────────────────────────
    y = (df[TARGET_COLUMN] > threshold).astype(int)
    flood_pct = y.mean() * 100
    print(f"[Data]  Target  : {flood_pct:.1f}% flood (class 1) / {100 - flood_pct:.1f}% no-flood (class 0)")

    # 4. Feature engineering ──────────────────────────────────────────────────
    df = _engineer_features(df)
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols]
    feature_names = list(X.columns)
    print(f"[Data]  Features: {len(feature_names)} (after engineering)")

    # 5. Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Data]  Split   : train={X_train.shape[0]:,} | test={X_test.shape[0]:,}")

    # 6. Scale ────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 7. Persist artefacts ────────────────────────────────────────────────────
    scaler_path  = os.path.join(models_dir, "scaler.pkl")
    names_path   = os.path.join(models_dir, "feature_names.pkl")
    joblib.dump(scaler,        scaler_path)
    joblib.dump(feature_names, names_path)
    print(f"[Data]  Saved   : {scaler_path}")
    print(f"[Data]  Saved   : {names_path}")

    return X_train, X_test, y_train, y_test, scaler, feature_names
