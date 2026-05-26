from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import datetime
import io
import pandas as pd

# For PDF generation – using reportlab (installed via requirements)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

router = APIRouter()

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"

# Helper to list PDF reports
def _list_pdfs():
    if not RESULTS_DIR.is_dir():
        return []
    return [p.name for p in RESULTS_DIR.glob("*.pdf")]

@router.get("/reports/history")
async def report_history():
    files = _list_pdfs()
    return JSONResponse(content={"reports": files})

@router.get("/reports/{filename}")
async def download_report(filename: str):
    file_path = RESULTS_DIR / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path=file_path, media_type="application/pdf", filename=filename)

@router.post("/reports/generate")
async def generate_report(payload: dict):
    """Expect payload to contain the prediction result dict (as returned by /predict_raw)
    and optionally a "patient_id" field. The function creates a simple PDF
    summarising the key fields and stores it in the results folder.
    """
    patient_id = payload.get("patient_id") or payload.get("patient_id")
    if not patient_id:
        raise HTTPException(status_code=400, detail="Missing patient_id in payload")
    # Create PDF in memory
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, f"Sepsis Prediction Report – Patient {patient_id}")
    y -= 40
    c.setFont("Helvetica", 12)
    # Add main fields
    prob = payload.get("prob")
    severity = payload.get("severity")
    organ = payload.get("organ")
    vitals = payload.get("vitals", {})
    c.drawString(50, y, f"Sepsis Probability: {prob:.2%}" if prob is not None else "Sepsis Probability: N/A")
    y -= 20
    c.drawString(50, y, f"Severity: {severity or 'N/A'}")
    y -= 20
    c.drawString(50, y, f"Affected Organ: {organ or 'N/A'}")
    y -= 30
    c.drawString(50, y, "Vital Signs:")
    y -= 20
    for k, v in vitals.items():
        c.drawString(70, y, f"{k}: {v}")
        y -= 15
    # Footer with timestamp
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 30, f"Generated on {datetime.datetime.utcnow().isoformat()} UTC")
    c.showPage()
    c.save()
    buf.seek(0)
    # Save to results folder
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"report_{patient_id}_{timestamp}.pdf"
    out_path = RESULTS_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(buf.read())
    return JSONResponse(content={"report_file": filename})
