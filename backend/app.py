"""
backend/app.py
--------------
Flask REST API for Flood Prediction.

Run from project root:
    python backend/app.py

Endpoints:
    GET  /health    → server + model status
    GET  /features  → list of expected feature names (in order)
    POST /predict   → flood risk prediction

POST /predict accepts two JSON formats:
  1. Array format  : {"features": [val1, val2, ..., val26]}
  2. Named format  : {"data": {"MonsoonIntensity": 80, "Urbanization": 60, ...}}
"""

import os
import sys
import logging

# Allow loading sibling packages when running from project root or backend/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins (allows React frontend to call this API)

# ─── Model loading ────────────────────────────────────────────────────────────
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def _load_artifacts():
    model_path   = os.path.join(MODELS_DIR, "flood_model.pkl")
    scaler_path  = os.path.join(MODELS_DIR, "scaler.pkl")
    names_path   = os.path.join(MODELS_DIR, "feature_names.pkl")

    missing = [p for p in [model_path, scaler_path, names_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}\n"
            "Run 'python scripts/train_flood_pipeline.py' first."
        )

    model         = joblib.load(model_path)
    scaler        = joblib.load(scaler_path)
    feature_names = joblib.load(names_path)
    log.info("Model loaded: %s", type(model).__name__)
    log.info("Expecting %d features: %s", len(feature_names), feature_names)
    return model, scaler, feature_names

try:
    MODEL, SCALER, FEATURE_NAMES = _load_artifacts()
    MODEL_READY = True
    MODEL_TYPE  = type(MODEL).__name__
except FileNotFoundError as e:
    log.warning("⚠  Model not ready: %s", e)
    MODEL, SCALER, FEATURE_NAMES = None, None, []
    MODEL_READY = False
    MODEL_TYPE  = "N/A"


# ─── Helper ───────────────────────────────────────────────────────────────────
def _risk_label(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Moderate"
    elif probability < 0.80:
        return "High"
    else:
        return "Very High"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"     : "ok" if MODEL_READY else "model_not_loaded",
        "model_ready": MODEL_READY,
        "model_type" : MODEL_TYPE,
        "n_features" : len(FEATURE_NAMES),
        "message"    : (
            "Backend is ready. Send POST /predict with your features."
            if MODEL_READY
            else "Train the model first: python scripts/train_flood_pipeline.py"
        ),
    }), 200


@app.route("/features", methods=["GET"])
def features():
    """Returns the ordered list of feature names the model expects."""
    return jsonify({
        "feature_count": len(FEATURE_NAMES),
        "features"     : FEATURE_NAMES,
        "note"         : (
            "When using array format in /predict, values must be in this exact order. "
            "Engineered features (e.g. Monsoon_x_Drainage) are computed server-side "
            "— only supply the 20 raw base features if using named format."
        ),
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded. Run training script first."}), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        # ── Format 1: raw array ───────────────────────────────────────────────
        if "features" in payload:
            raw = payload["features"]
            if not isinstance(raw, list):
                return jsonify({"error": "'features' must be a JSON array."}), 400
            if len(raw) != len(FEATURE_NAMES):
                return jsonify({
                    "error": f"Expected {len(FEATURE_NAMES)} features, got {len(raw)}.",
                    "expected_features": FEATURE_NAMES,
                }), 400
            input_arr = np.array(raw, dtype=float).reshape(1, -1)

        # ── Format 2: named dict ──────────────────────────────────────────────
        elif "data" in payload:
            data = payload["data"]
            if not isinstance(data, dict):
                return jsonify({"error": "'data' must be a JSON object."}), 400

            # Build array in feature order
            import pandas as pd
            from utils.preprocessing import _engineer_features, FEATURE_COLUMNS

            # Validate raw base features
            missing_cols = [c for c in FEATURE_COLUMNS if c not in data]
            if missing_cols:
                return jsonify({
                    "error": f"Missing raw feature(s): {missing_cols}",
                    "required_raw_features": FEATURE_COLUMNS,
                }), 400

            df_input = pd.DataFrame([{c: data[c] for c in FEATURE_COLUMNS}])
            df_input = _engineer_features(df_input)
            # Order columns exactly as training
            df_input = df_input[FEATURE_NAMES]
            input_arr = df_input.values.astype(float)

        else:
            return jsonify({
                "error": "Provide either 'features' (array) or 'data' (named dict).",
                "formats": {
                    "array" : {"features": [80, 60, 70, "..."]},
                    "named" : {"data": {"MonsoonIntensity": 80, "Urbanization": 60, "...": "..."}},
                },
            }), 400

        # ── Scale & predict ───────────────────────────────────────────────────
        scaled      = SCALER.transform(input_arr)
        prediction  = int(MODEL.predict(scaled)[0])
        probability = float(MODEL.predict_proba(scaled)[0][1])
        risk_level  = _risk_label(probability)

        log.info("Prediction: %d | Probability: %.4f | Risk: %s", prediction, probability, risk_level)

        return jsonify({
            "prediction" : prediction,          # 0 = No Flood, 1 = Flood Risk
            "probability": round(probability, 4),
            "risk_level" : risk_level,           # Low / Moderate / High / Very High
            "label"      : "Flood Risk" if prediction == 1 else "No Flood",
        }), 200

    except ValueError as e:
        return jsonify({"error": f"Invalid feature values: {e}"}), 400
    except Exception as e:
        log.exception("Unexpected error during prediction.")
        return jsonify({"error": str(e)}), 500


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting Flood Prediction API on http://127.0.0.1:%d", port)
    app.run(debug=True, host="127.0.0.1", port=port)
