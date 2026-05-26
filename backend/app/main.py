import os
import joblib
import pandas as pd
import numpy as np
import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from pydantic import BaseModel
from werkzeug.utils import secure_filename
import io
import shap

# Environment configuration
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "final_sepsis_model.pkl")
RESULTS_DIR = os.getenv("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"))
EXCEL_FILE = os.path.join(RESULTS_DIR, "patient_predictions.xlsx")
DATA_PATH = os.getenv("DATA_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-key")

app = FastAPI(title="Sepsis Prediction API", version="1.0")

# CORS – allow Vercel frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT settings
class Settings(BaseModel):
    authjwt_secret_key: str = JWT_SECRET
    authjwt_token_location: set = {"headers"}
    authjwt_access_token_expires: int = 3600  # 1 hour

@AuthJWT.load_config
def get_config():
    return Settings()

# Exception handler for auth errors
@app.exception_handler(AuthJWTException)
def auth_exception_handler(request: Request, exc: AuthJWTException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})

# Load model globally
if os.path.exists(MODEL_PATH):
    saved_model = joblib.load(MODEL_PATH)
else:
    saved_model = None

# Utility functions (mirrored from src/api.py)
def save_to_excel(patient_id: str, prediction: str, prob: float, severity_level: str):
    result_data = {
        "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Patient ID": [patient_id],
        "Prediction": [prediction],
        "Probability": [prob],
        "Severity": [severity_level],
    }
    df_new = pd.DataFrame(result_data)
    try:
        if os.path.exists(EXCEL_FILE):
            df_existing = pd.read_excel(EXCEL_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(EXCEL_FILE, index=False)
        else:
            df_new.to_excel(EXCEL_FILE, index=False)
    except Exception as e:
        print(f"Failed to save to Excel: {e}")

# Placeholder imports – these functions exist in src/main.py
from src.main import get_prediction_data, get_realtime_timeline

# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Login – returns JWT token
class LoginPayload(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(payload: LoginPayload, Authorize: AuthJWT = Depends()):
    users_file = os.path.join(DATA_PATH, "users.csv")
    if not os.path.exists(users_file):
        raise HTTPException(status_code=500, detail="User database not found")
    import csv
    from werkzeug.security import check_password_hash
    with open(users_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["username"] == payload.username:
                if check_password_hash(row["password_hash"], payload.password):
                    access_token = Authorize.create_access_token(subject=payload.username)
                    return {"access_token": access_token, "role": row.get("role", "user")}
                else:
                    raise HTTPException(status_code=401, detail="Invalid password")
    raise HTTPException(status_code=404, detail="User not found")

# Protected route example
@app.get("/protected")
def protected(Authorize: AuthJWT = Depends()):
    Authorize.jwt_required()
    current_user = Authorize.get_jwt_subject()
    return {"msg": f"You are authorized as {current_user}"}

# Patient data endpoint
@app.get("/patient/{patient_id}")
def get_patient_data(patient_id: str, Authorize: AuthJWT = Depends()):
    Authorize.jwt_required()
    if not saved_model:
        raise HTTPException(status_code=500, detail="Model not trained")
    matched_file_path = None
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.lower() == f"{patient_id.lower()}.psv":
                matched_file_path = os.path.join(root, file)
                break
        if matched_file_path:
            break
    if not matched_file_path:
        raise HTTPException(status_code=404, detail="Patient not found")
    df_raw = pd.read_csv(matched_file_path, sep="|")
    df_raw["PatientID"] = patient_id
    result = process_patient_dataframe(df_raw, patient_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

# Upload prediction endpoint
@app.post("/predict_upload")
def predict_upload(file: UploadFile = File(...), Authorize: AuthJWT = Depends()):
    Authorize.jwt_required()
    if not saved_model:
        raise HTTPException(status_code=500, detail="Model not trained")
    filename = secure_filename(file.filename)
    patient_id = os.path.splitext(filename)[0]
    try:
        content = await file.read()
        ext = filename.lower().split(".")[-1]
        if ext in ["csv", "psv"]:
            text = content.decode("utf-8", errors="ignore")
            sep = "," if ext == "csv" else "|"
            df_raw = pd.read_csv(io.StringIO(text), sep=sep)
        elif ext in ["xlsx", "xls"]:
            df_raw = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        df_raw["PatientID"] = patient_id
        result = process_patient_dataframe(df_raw, patient_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Raw PSV prediction endpoint
class RawPayload(BaseModel):
    data: str
    patient_id: str = None

@app.post("/predict_raw")
def predict_raw(payload: RawPayload, Authorize: AuthJWT = Depends()):
    Authorize.jwt_required()
    if not saved_model:
        raise HTTPException(status_code=500, detail="Model not trained")
    psv_content = payload.data
    patient_id = payload.patient_id or f"manual_{datetime.datetime.now().strftime('%Y%H%M%S')}"
    try:
        df_raw = pd.read_csv(io.StringIO(psv_content), sep="|")
        df_raw["PatientID"] = patient_id
        result = process_patient_dataframe(df_raw, patient_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper used by both API and CLI
def process_patient_dataframe(df_raw: pd.DataFrame, patient_id: str) -> Dict[str, Any]:
    if not saved_model:
        return {"error": "Model not trained"}
    # Reuse logic from src.main
    res = get_prediction_data(df_raw, saved_model, patient_id)
    if "error" in res:
        return res
    prob = res["prob"]
    prediction = "Sepsis" if prob >= saved_model.get("threshold", 0.3) else "No Sepsis"
    severity = res["severity"]
    organ = res["organ"]
    vitals = res["vitals"]
    # Save to Excel
    save_to_excel(patient_id, prediction, float(prob), severity)
    # Timeline data
    risk_timeline, hr_timeline = get_realtime_timeline(res["df_processed"], saved_model)
    return {
        "patient_id": patient_id,
        "prediction": prediction,
        "probability": round(float(prob), 4),
        "severity": severity,
        "organ": organ,
        "vitals": vitals,
        "risk_timeline": risk_timeline,
        "hr_timeline": hr_timeline,
        "shap": res.get("shap", []),
        "shap_base_value": res.get("shap_base_value", 0.0),
        "icu_hours": res.get("icu_hours", 0),
        "recommendations": res.get("recommendations", []),
    }
