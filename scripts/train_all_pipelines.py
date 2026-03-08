"""
scripts/train_all_pipelines.py
-------------------------------
Trains all non-flood disaster prediction models from their datasets.
Four models are trained: Earthquake, Forest Fire, Landslide, Tsunami.
(Flood model is managed separately via utils/preprocessing.py)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

os.makedirs('models', exist_ok=True)

# ─────────────────────────────────────────────────
# 1. EARTHQUAKE
# ─────────────────────────────────────────────────
def train_earthquake():
    print("\n" + "=" * 55)
    print("  TRAINING: Earthquake Risk Model")
    print("=" * 55)
    # The user provided a new dataset for these models: earthquake_data_tsunami.csv
    # It contains earthquakes with magnitude >= 6.5
    df = pd.read_csv('datasets/earthquake_data_tsunami.csv')
    df = df.dropna(subset=['magnitude', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude'])
    
    # Feature set requested by user
    features = ['magnitude', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude']
    X = df[features].copy()
    
    # Target: Magnitude >= 7.0 for Severe classification natively mapped to High Risk
    y = (df['magnitude'] >= 7.0).astype(int)
    print(f"  Dataset size: {len(df)} | Class balance — High Risk: {y.sum()} ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, preds, target_names=['Low Risk', 'High Risk']))

    joblib.dump(model,   'models/earthquake_model.pkl')
    joblib.dump(scaler,  'models/earthquake_scaler.pkl')
    joblib.dump(features,'models/earthquake_features.pkl')
    print("  ✓ Earthquake model saved.\n")


# ─────────────────────────────────────────────────
# 2. FOREST FIRE
# ─────────────────────────────────────────────────
def train_forestfire():
    print("\n" + "=" * 55)
    print("  TRAINING: Forest Fire Model")
    print("=" * 55)
    df = pd.read_csv('datasets/forestfires.csv')
    df = df.dropna()

    features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
    X = df[features].copy()

    # Target: area > 0 means a fire actually burned an area
    y = (df['area'] > 0).astype(int)
    print(f"  Dataset size: {len(df)} | Fire happened: {y.sum()} ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, preds, target_names=['No Fire', 'Fire']))

    joblib.dump(model,    'models/forestfire_model.pkl')
    joblib.dump(scaler,   'models/forestfire_scaler.pkl')
    joblib.dump(features, 'models/forestfire_features.pkl')
    print("  ✓ Forest Fire model saved.\n")


# ─────────────────────────────────────────────────
# 3. LANDSLIDE
# ─────────────────────────────────────────────────
def train_landslide():
    print("\n" + "=" * 55)
    print("  TRAINING: Landslide Risk Model")
    print("=" * 55)
    df = pd.read_csv('datasets/landslides.csv')
    df['population'] = pd.to_numeric(df['population'], errors='coerce').fillna(0)
    df['distance']   = pd.to_numeric(df['distance'],   errors='coerce').fillna(0)
    df = df.dropna(subset=['latitude', 'longitude'])

    features = ['latitude', 'longitude', 'population', 'distance']
    
    # The dataset only contains positive occurrences. 
    # Create the positive samples:
    X_pos = df[features].copy()
    y_pos = np.ones(len(X_pos), dtype=int)
    
    # Synthesize negative samples (areas where landslides did NOT occur)
    np.random.seed(42)
    n_samples = len(X_pos)
    
    # Synthetic safe zones
    syn_lat = np.random.uniform(-90, 90, n_samples)
    syn_lon = np.random.uniform(-180, 180, n_samples)
    syn_pop = np.random.exponential(50000, n_samples) # random population
    syn_dist = np.random.uniform(50, 500, n_samples)  # further distance from faults = safer
    
    X_neg = pd.DataFrame({
        'latitude': syn_lat,
        'longitude': syn_lon,
        'population': syn_pop,
        'distance': syn_dist
    })
    y_neg = np.zeros(n_samples, dtype=int)
    
    # Combine
    X = pd.concat([X_pos, X_neg], ignore_index=True)
    y = np.concatenate([y_pos, y_neg])
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X = X.iloc[idx].reset_index(drop=True)
    y = y[idx]

    print(f"  Combined Dataset size: {len(X)} | Landslide Risk Target Balance: {y.sum()} / {len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, preds, target_names=['Safe', 'Landslide Risk']))

    joblib.dump(model,    'models/landslide_model.pkl')
    joblib.dump(scaler,   'models/landslide_scaler.pkl')
    joblib.dump(features, 'models/landslide_features.pkl')
    print("  ✓ Landslide model saved.\n")


# ─────────────────────────────────────────────────
# 4. TSUNAMI
# ─────────────────────────────────────────────────
def train_tsunami():
    print("\n" + "=" * 55)
    print("  TRAINING: Tsunami Risk Model")
    print("=" * 55)
    df = pd.read_csv('datasets/tsunami_dataset.csv')
    df = df.dropna(subset=['EQ_MAGNITUDE', 'EQ_DEPTH'])

    features = ['EQ_MAGNITUDE', 'EQ_DEPTH']
    X = df[features].copy()

    # Target: earthquake magnitude >= 7.0 triggers tsunami risk
    # This is scientifically sound — marine quake >= 7.0 is primary tsunami trigger criterion
    y = (df['EQ_MAGNITUDE'] >= 7.0).astype(int)
    print(f"  Dataset size: {len(df)} | High Risk (M>=7.0): {y.sum()} ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, preds, target_names=['Low Risk', 'High Risk']))

    joblib.dump(model,    'models/tsunami_model.pkl')
    joblib.dump(scaler,   'models/tsunami_scaler.pkl')
    joblib.dump(features, 'models/tsunami_features.pkl')
    print("  ✓ Tsunami model saved.\n")


# ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🔥 Starting Disaster Prediction Model Training...")
    train_earthquake()
    train_forestfire()
    train_landslide()
    train_tsunami()
    print("\n✅ All Non-Flood models trained and saved successfully!")
    print("   → Run 'python backend/app.py' to start the API server.\n")
