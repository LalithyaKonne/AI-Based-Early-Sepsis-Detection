import os
import joblib
from pathlib import Path

# Load the pre‑trained model once at import time
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "final_sepsis_model.pkl"

if MODEL_PATH.is_file():
    saved_model = joblib.load(MODEL_PATH)
else:
    saved_model = None
    print(f"⚠️ Model file not found at {MODEL_PATH}")

# Helper to ensure the model is loaded before any prediction
def get_model():
    if saved_model is None:
        raise RuntimeError("Model not loaded – check that final_sepsis_model.pkl exists.")
    return saved_model
