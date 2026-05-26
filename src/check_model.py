import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(BASE_DIR, "models", "final_sepsis_model.pkl")

if os.path.exists(MODEL_FILE):
    saved_model = joblib.load(MODEL_FILE)
    print(f"Current Threshold: {saved_model.get('threshold', 'N/A')}")
    # Also check a few params of the XGB model if possible
    xgb = saved_model['model']['xgb']
    print(f"XGB n_estimators: {xgb.n_estimators}")
    print(f"XGB learning_rate: {xgb.learning_rate}")
else:
    print("Model file not found.")
