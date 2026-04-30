# src/predicting_maternal_health_risk/serving/predict.py

from pathlib import Path
import joblib
import pandas as pd

# Adjust path if your model is saved elsewhere
MODEL_PATH = Path("data/06_models/xgb_model.json")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH.resolve()}. "
        "Run `kedro run --pipeline mt` first."
    )

MODEL = joblib.load(MODEL_PATH)

FEATURES = {
    "age": {"type": "int", "min": 15, "max": 50},
    "systolic_bp": {"type": "float", "min": 80, "max": 200},
    "diastolic_bp": {"type": "float", "min": 40, "max": 140},
    "blood_sugar": {"type": "float", "min": 2, "max": 30},
    "body_temp": {"type": "float", "min": 34, "max": 42},
    "heart_rate": {"type": "int", "min": 40, "max": 200},
}

HIGH_RISK_THRESHOLD = 0.60
MID_RISK_THRESHOLD = 0.55

# Load model once at import time
with open(MODEL_PATH, "rb") as f:
    MODEL = joblib.load(f)


def preprocess_input(patient: dict) -> pd.DataFrame:
    """
    Turn a raw patient dict (strings from form) into a one-row
    DataFrame with numeric values in correct order.
    """
    row = {}
    for name, meta in FEATURES.items():
        raw_val = patient.get(name)
        if raw_val is None or raw_val == "":
            raise ValueError(f"Missing value for {name}")

        if meta["type"] == "int":
            val = int(raw_val)
        else:
            val = float(raw_val)

        # Optional: basic range clipping
        if "min" in meta:
            val = max(meta["min"], val)
        if "max" in meta:
            val = min(meta["max"], val)

        row[name] = val

    df = pd.DataFrame([row])
    return df


def predict_risk(patient: dict) -> dict:
    """
    patient: dict of raw form values (strings)
    returns: dict with label and probabilities
    """
    X = preprocess_input(patient)
    proba = MODEL.predict_proba(X)[0]  # array-like, shape (n_classes,)

    # Adjust indices / order if your y labels are encoded differently
    p_low = float(proba[0])
    p_mid = float(proba[1])
    p_high = float(proba[2])

    if p_high >= HIGH_RISK_THRESHOLD:
        label = "High risk"
    elif p_high >= MID_RISK_THRESHOLD:
        label = "Medium risk"
    else:
        label = "Low risk"

    return {
        "risk_label": label,
        "risk_probs": {
            "low": round(p_low, 4),
            "mid": round(p_mid, 4),
            "high": round(p_high, 4),
        },
    }