"""
scripts/train_flood_pipeline.py
--------------------------------
Full ML training pipeline for Flood Prediction.
Trains 3 models, runs GridSearchCV on XGBoost, picks the best by AUC-ROC,
and saves the final model + artefacts to models/.

Run from project root:
    python scripts/train_flood_pipeline.py [--data datasets/flood.csv]
"""

import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from xgboost import XGBClassifier

from utils.preprocessing import load_and_preprocess


# ─── Pretty printing helpers ──────────────────────────────────────────────────
SEP = "=" * 60

def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def train_evaluate(name, model, X_train, y_train, X_test, y_test, cv):
    """Train, cross-validate, and evaluate a model. Returns test AUC."""
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    # CV AUC on training set
    cv_aucs = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_mean, cv_std = cv_aucs.mean(), cv_aucs.std()

    # Test metrics
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    test_auc    = roc_auc_score(y_test, y_pred_prob)
    test_acc    = accuracy_score(y_test, y_pred)

    print(f"\n[{name}]")
    print(f"  Train time   : {elapsed:.1f}s")
    print(f"  CV  AUC      : {cv_mean:.4f} ± {cv_std:.4f}  (5-fold)")
    print(f"  Test AUC     : {test_auc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["No Flood", "Flood"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix (rows=true, cols=pred):")
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    return test_auc


def main(data_path: str, models_dir: str = "models"):
    os.makedirs(models_dir, exist_ok=True)

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    section("1. Loading & Preprocessing Data")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(
        file_path=data_path,
        models_dir=models_dir,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 2. Baseline – Logistic Regression ─────────────────────────────────────
    section("2. Logistic Regression")
    lr = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
    lr_auc = train_evaluate("Logistic Regression", lr, X_train, y_train, X_test, y_test, cv)

    # ── 3. Random Forest ──────────────────────────────────────────────────────
    section("3. Random Forest Classifier")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=5,
        random_state=42, n_jobs=-1, class_weight="balanced"
    )
    rf_auc = train_evaluate("Random Forest", rf, X_train, y_train, X_test, y_test, cv)

    # ── 4. XGBoost with GridSearchCV ─────────────────────────────────────────
    section("4. XGBoost (with GridSearchCV tuning)")
    xgb_base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",           # fast histogram method
    )

    param_grid = {
        "n_estimators"  : [200, 400],
        "max_depth"     : [4, 6],
        "learning_rate" : [0.05, 0.1],
        "subsample"     : [0.8],
        "colsample_bytree": [0.8],
    }

    print("\n  Running GridSearchCV (this may take a few minutes)…")
    gscv = GridSearchCV(
        xgb_base, param_grid,
        cv=3, scoring="roc_auc",
        n_jobs=-1, verbose=0,
        refit=True,
    )
    gscv.fit(X_train, y_train)
    print(f"  Best params : {gscv.best_params_}")
    print(f"  Best CV AUC : {gscv.best_score_:.4f}")
    xgb_best = gscv.best_estimator_

    xgb_auc = train_evaluate("XGBoost (tuned)", xgb_best, X_train, y_train, X_test, y_test, cv)

    # ── 5. Pick best model ────────────────────────────────────────────────────
    section("5. Model Selection")
    scores = {
        "Logistic Regression": (lr_auc,   lr),
        "Random Forest"      : (rf_auc,   rf),
        "XGBoost (tuned)"    : (xgb_auc,  xgb_best),
    }

    best_name = max(scores, key=lambda k: scores[k][0])
    best_auc, best_model = scores[best_name]

    print(f"\n  {'Model':<25} {'Test AUC':>10}")
    print("  " + "-" * 38)
    for name, (auc, _) in sorted(scores.items(), key=lambda x: -x[1][0]):
        marker = "  ← SELECTED" if name == best_name else ""
        print(f"  {name:<25} {auc:>10.4f}{marker}")

    # ── 6. Save artefacts ─────────────────────────────────────────────────────
    section("6. Saving Artefacts")
    model_path = os.path.join(models_dir, "flood_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"  Saved model → {model_path}")
    print(f"  Best model  : {best_name}")
    print(f"  Best AUC    : {best_auc:.4f}")
    print(f"\n✅ Training complete! Run `python backend/app.py` to start the API.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flood Prediction Training Pipeline")
    parser.add_argument("--data",       default="datasets/flood.csv",
                        help="Path to flood.csv (default: datasets/flood.csv)")
    parser.add_argument("--models_dir", default="models",
                        help="Directory to save model artefacts (default: models/)")
    args = parser.parse_args()
    main(args.data, args.models_dir)
