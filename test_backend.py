"""
test_backend.py
---------------
Quick integration test for the Flask backend.
Start the API first:   python backend/app.py
Then run:              python test_backend.py
"""

import sys
import json
import requests

BASE_URL = "http://127.0.0.1:5000"
PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = []


def test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append(True)
    except AssertionError as e:
        print(f"  {FAIL}  {name}")
        print(f"         → {e}")
        results.append(False)
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         → Unexpected error: {e}")
        results.append(False)


# ─── Test 1: Health check ─────────────────────────────────────────────────────
def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    body = r.json()
    assert body.get("status") == "ok", f"status={body.get('status')}"
    assert body.get("model_ready") is True, "model_ready is not True"

# ─── Test 2: Features list ────────────────────────────────────────────────────
def test_features():
    r = requests.get(f"{BASE_URL}/features", timeout=5)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    body = r.json()
    assert "features" in body, "No 'features' key in response"
    assert isinstance(body["features"], list), "'features' is not a list"
    assert body["feature_count"] > 0, "feature_count is 0"

# ─── Test 3: Predict – array format (high risk) ───────────────────────────────
def test_predict_high_risk_array():
    # All high values → should predict flood (1)
    features = [90, 85, 80, 75, 90, 85, 70, 75, 80, 85,
                80, 75, 90, 85, 80, 75, 90, 85, 80, 75,
                90*85, 90*80, 85*75, (80+75)/2, (70+75+75)/3, (75+85+75)/3]

    # Auto-pad or trim to expected length
    r0 = requests.get(f"{BASE_URL}/features", timeout=5)
    n  = r0.json()["feature_count"]
    features = (features + [80] * 30)[:n]   # pad/trim to exact size

    r = requests.post(f"{BASE_URL}/predict",
                      json={"features": features},
                      headers={"Content-Type": "application/json"},
                      timeout=5)
    assert r.status_code == 200, f"HTTP {r.status_code} — {r.text}"
    body = r.json()
    assert "prediction"  in body, "Missing 'prediction'"
    assert "probability" in body, "Missing 'probability'"
    assert "risk_level"  in body, "Missing 'risk_level'"
    assert body["prediction"] in [0, 1], "prediction not 0 or 1"
    assert 0.0 <= body["probability"] <= 1.0, "probability out of [0,1]"
    print(f"         → {body}")

# ─── Test 4: Predict – named dict format (low risk) ──────────────────────────
def test_predict_low_risk_named():
    low_risk_data = {
        "MonsoonIntensity"              : 10,
        "TopographyDrainage"            : 10,
        "RiverManagement"               : 10,
        "Deforestation"                 : 10,
        "Urbanization"                  : 10,
        "ClimateChange"                 : 10,
        "DamsQuality"                   : 10,
        "Siltation"                     : 10,
        "AgriculturalPractices"         : 10,
        "Encroachments"                 : 10,
        "IneffectiveDisasterPreparedness": 10,
        "DrainageSystems"               : 10,
        "CoastalVulnerability"          : 10,
        "Landslides"                    : 10,
        "Watersheds"                    : 10,
        "DeterioratingInfrastructure"   : 10,
        "PopulationScore"               : 10,
        "WetlandLoss"                   : 10,
        "InadequatePlanning"            : 10,
        "PoliticalFactors"              : 10,
    }
    r = requests.post(f"{BASE_URL}/predict",
                      json={"data": low_risk_data},
                      headers={"Content-Type": "application/json"},
                      timeout=5)
    assert r.status_code == 200, f"HTTP {r.status_code} — {r.text}"
    body = r.json()
    assert "prediction"  in body, "Missing 'prediction'"
    assert "probability" in body, "Missing 'probability'"
    print(f"         → {body}")

# ─── Test 5: Bad request – wrong feature count ───────────────────────────────
def test_bad_feature_count():
    r = requests.post(f"{BASE_URL}/predict",
                      json={"features": [1, 2, 3]},   # too few
                      timeout=5)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"

# ─── Test 6: Bad request – no payload ────────────────────────────────────────
def test_no_payload():
    r = requests.post(f"{BASE_URL}/predict",
                      data="not json",
                      headers={"Content-Type": "text/plain"},
                      timeout=5)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"


# ─── Run all tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Flood Prediction API – Integration Tests")
    print("=" * 50)

    test("GET  /health                  (ok + model_ready)", test_health)
    test("GET  /features                (returns feature list)",  test_features)
    test("POST /predict  [array]        (high-risk sample)",      test_predict_high_risk_array)
    test("POST /predict  [named dict]   (low-risk sample)",       test_predict_low_risk_named)
    test("POST /predict  [bad count]    (should return 400)",     test_bad_feature_count)
    test("POST /predict  [no payload]   (should return 400)",     test_no_payload)

    print("\n" + "=" * 50)
    passed = sum(results)
    total  = len(results)
    emoji  = "🎉" if passed == total else "⚠️ "
    print(f"  {emoji}  {passed}/{total} tests passed.")
    print("=" * 50 + "\n")
    sys.exit(0 if passed == total else 1)
