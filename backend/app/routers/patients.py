from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd
import io
import sys

# Add project src to path and import prediction utilities
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
from main import get_prediction_data, get_realtime_timeline  # noqa: E402

# Import the pre‑loaded model
from ..utils import saved_model

router = APIRouter()

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"

def _load_patient_file(patient_id: str) -> pd.DataFrame:
    """Load a CSV or XLSX file for an existing patient."""
    for suffix in (".csv", ".xlsx"):
        candidate = DATA_DIR / f"{patient_id}{suffix}"
        if candidate.is_file():
            return pd.read_csv(candidate) if suffix == ".csv" else pd.read_excel(candidate)
    raise FileNotFoundError(f"Patient file not found for id {patient_id}")

@router.get("/patient/{patient_id}")
async def get_existing_patient(patient_id: str):
    try:
        df = _load_patient_file(patient_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    result = get_prediction_data(df, saved_model, patient_id)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"]) 
    return JSONResponse(content=result)

@router.post("/predict_upload")
async def predict_upload(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename.lower()
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {e}")
    if "PatientID" not in df.columns:
        raise HTTPException(status_code=400, detail="Uploaded file must contain a 'PatientID' column")
    patient_id = str(df["PatientID"].iloc[0])
    result = get_prediction_data(df, saved_model, patient_id)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"]) 
    return JSONResponse(content=result)

@router.post("/predict_raw")
async def predict_raw(payload: dict):
    data_str = payload.get("data")
    patient_id = payload.get("patient_id")
    if not data_str or not patient_id:
        raise HTTPException(status_code=400, detail="Missing 'data' or 'patient_id'")
    try:
        # Split pipe‑separated values into a list
        values = [v.strip() for v in data_str.split("|")]
        df = pd.DataFrame([values])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse data string: {e}")
    result = get_prediction_data(df, saved_model, patient_id)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"]) 
    return JSONResponse(content=result)
