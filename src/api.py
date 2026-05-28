import os
import sys
# Ensure the project root (one level up) is on the Python import path so that imports from the `src` package work when running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import pandas as pd
import numpy as np
import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Local imports – after adjusting sys.path the `src` package can be imported
from src.data_loader import load_data
from src.main import get_prediction_data, get_realtime_timeline
import shap
import threading
import sys
import io

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Allow React frontend to access API

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_FILE = os.path.join(BASE_DIR, "models", "final_sepsis_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EXCEL_FILE = os.path.join(RESULTS_DIR, "patient_predictions.xlsx")
EARLY_HOURS = 6

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load Model once globally
saved_model = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None

# Training State
training_status = {
    "is_training": False,
    "progress": 0,
    "logs": [],
    "result": None,
    "error": None
}

class LoggerWriter:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout

    def write(self, message):
        if message.strip():
            training_status["logs"].append(message.strip())
        self.original_stdout.write(message)
        
    def flush(self):
        self.original_stdout.flush()

def save_to_excel(patient_id, prediction, prob, severity_level):
    result_data = {
        "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Patient ID": [patient_id],
        "Prediction": [prediction],
        "Probability": [prob],
        "Severity": [severity_level]
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

def process_patient_dataframe(df_raw, patient_id):
    if not saved_model:
        return {"error": "Model not trained"}
        
    # DIRECTLY CALL main.py helpers to ensure 100% equivalence
    res = get_prediction_data(df_raw, saved_model, patient_id)
    if "error" in res:
        return res
        
    prob = res["prob"]
    prediction = "Sepsis" if prob >= saved_model.get("threshold", 0.3) else "No Sepsis"
    sev = res["severity"]
    organ = res["organ"]
    vitals = res["vitals"]
    
    # Save to Excel
    save_to_excel(patient_id, prediction, float(prob), sev)

    # 3. Real-Time Risk Timeline & HR Timeline from main.py
    risk_timeline, hr_timeline = get_realtime_timeline(res["df_processed"], saved_model)

    # 4. SHAP Values
    shap_vals = []
    base_value = 0.0
    try:
        model = saved_model["model"]
        xgb_m = model["xgb"]
        if hasattr(xgb_m, "named_steps"):
            xgb_m = list(xgb_m.named_steps.values())[-1]
        explainer = shap.TreeExplainer(xgb_m)
        s_values = explainer.shap_values(res["X_new"])[0]
        
        ev = explainer.expected_value
        base_value = float(ev[0]) if isinstance(ev, (np.ndarray, list)) else float(ev)
        
        if hasattr(res["X_new"], "columns"):
            feature_names = res["X_new"].columns.tolist()
        else:
            # Fallback to model's stored features or generic names if numpy
            feature_names = saved_model.get("features", [f"Feature_{i}" for i in range(res["X_new"].shape[1])])
            
        impacts = [{"feature": fn, "value": float(sv), "abs_impact": abs(float(sv))} for fn, sv in zip(feature_names, s_values)]
        
        impacts.sort(key=lambda x: x["abs_impact"], reverse=True)
        top_n = 9
        top_features = impacts[:top_n]
        other_value = sum(x["value"] for x in impacts[top_n:])
        
        shap_vals = top_features + [{"feature": f"{len(impacts) - top_n} other features", "value": other_value, "abs_impact": abs(other_value)}]
        
    except Exception as e:
        shap_vals = []
        base_value = 0.0

    # 5. Clinical Decision Support Recommendations
    recs = ["Monitor vitals closely every hour."]
    if prob >= saved_model.get("threshold", 0.3):
        recs.append("Consider early antibiotics administration.")
    if float(vitals["map"]) < 65:
        recs.append("Administer IV fluids for hypotension.")
    recs.append("Check lactate levels immediately.")

    return {
        "patient_id": patient_id,
        "prediction": prediction,
        "probability": round(float(prob), 4),
        "severity": sev,
        "organ": organ,
        "vitals": vitals,
        "risk_timeline": risk_timeline,
        "hr_timeline": hr_timeline,
        "shap": shap_vals,
        "shap_base_value": base_value,
        "icu_hours": int(res["icu_hours"]),
        "recommendations": recs
    }

@app.route('/api/patient/<patient_id>', methods=['GET'])
def get_patient_data(patient_id):
    if not saved_model:
        return jsonify({"error": "Model not trained"}), 500

    matched_file_path = None
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.lower() == f"{patient_id.lower()}.psv":
                matched_file_path = os.path.join(root, file)
                break
        if matched_file_path:
            break

    if not matched_file_path:
        return jsonify({"error": "Patient not found"}), 404

    df_raw = pd.read_csv(matched_file_path, sep="|")
    df_raw["PatientID"] = patient_id

    result = process_patient_dataframe(df_raw, patient_id)
    if "error" in result:
        return jsonify(result), 400
        
    return jsonify(result)

@app.route('/api/predict_upload', methods=['POST'])
def predict_upload():
    if not saved_model:
        return jsonify({"error": "Model not trained"}), 500
        
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(file.filename)
    patient_id = os.path.splitext(filename)[0]
    
    try:
        filename_lower = filename.lower()
        if filename_lower.endswith('.csv') or filename_lower.endswith('.psv'):
            content = file.read()
            # Decode file contents
            try:
                text_content = content.decode('utf-8')
            except Exception:
                text_content = content.decode('latin-1')
                
            # Auto-sniff separator
            sep = ','
            if filename_lower.endswith('.psv'):
                sep = '|'
            else:
                first_line = text_content.split('\n')[0] if '\n' in text_content else text_content
                if '|' in first_line:
                    sep = '|'
                elif ';' in first_line:
                    sep = ';'
                elif '\t' in first_line:
                    sep = '\t'
            df_raw = pd.read_csv(io.StringIO(text_content), sep=sep)
        elif filename_lower.endswith('.xlsx') or filename_lower.endswith('.xls'):
            df_raw = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Use .csv or .xlsx"}), 400
            
        df_raw["PatientID"] = patient_id
        
        result = process_patient_dataframe(df_raw, patient_id)
        if "error" in result:
            return jsonify(result), 400
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/api/predict_raw', methods=['POST'])
def predict_raw():
    if not saved_model:
        return jsonify({"error": "Model not trained"}), 500
        
    # Robust JSON parsing (handle empty body and malformed JSON)
    import json
    raw_body = request.get_data(as_text=True)
    if not raw_body:
        return jsonify({"error": "Empty request body"}), 400
    try:
        data = json.loads(raw_body)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON payload: {str(e)}"}), 400
    if not data or 'data' not in data:
        return jsonify({"error": "No data provided"}), 400
    
    psv_content = data['data']
    patient_id = data.get('patient_id', f"manual_{datetime.datetime.now().strftime('%Y%H%M%S')}")
    
    try:
        from io import StringIO
        df_raw = pd.read_csv(StringIO(psv_content), sep='|')
        df_raw["PatientID"] = patient_id
        
        result = process_patient_dataframe(df_raw, patient_id)
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error processing PSV data: {str(e)}"}), 500

def run_training_job():
    global saved_model
    import matplotlib
    matplotlib.use('Agg') # Disable interactive plots 
    
    from main import train_model # Import from main to use existing training logic
    original_stdout = sys.stdout
    sys.stdout = LoggerWriter(original_stdout)
    
    try:
        training_status["is_training"] = True
        training_status["progress"] = 10
        training_status["logs"] = ["Starting training pipeline..."]
        
        # Override show to avoid blocking
        import matplotlib.pyplot as plt
        original_show = plt.show
        plt.show = lambda *args, **kwargs: None
        
        train_model()
        
        plt.show = original_show
        
        # Reload the saved model in the API memory
        saved_model = joblib.load(MODEL_FILE)
        
        training_status["progress"] = 100
        training_status["result"] = "Training successful. Model deployed."
    except Exception as e:
        training_status["error"] = str(e)
    finally:
        training_status["is_training"] = False
        sys.stdout = original_stdout

@app.route('/api/train', methods=['POST'])
def start_training():
    if training_status["is_training"]:
        return jsonify({"message": "Training already in progress."}), 400
        
    training_status["is_training"] = True
    training_status["progress"] = 0
    training_status["logs"] = []
    training_status["result"] = None
    training_status["error"] = None
    
    thread = threading.Thread(target=run_training_job)
    thread.start()
    return jsonify({"message": "Training started successfully."}), 202

@app.route('/api/train_status', methods=['GET'])
def get_training_status():
    return jsonify(training_status)

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Missing username or password"}), 400
    
    username = data['username']
    password = data['password']
    
    users_file = os.path.join(DATA_PATH, "users.csv")
    if not os.path.exists(users_file):
        return jsonify({"error": "User database not found"}), 500
        
    import csv
    from werkzeug.security import check_password_hash

    with open(users_file, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["username"] == username:
                if check_password_hash(row["password_hash"], password):
                    return jsonify({"message": "Login successful", "role": row["role"], "username": username}), 200
                else:
                    return jsonify({"error": "Invalid password"}), 401
                    
    return jsonify({"error": "User not found"}), 404

# Global error handler to ensure JSON responses on unexpected errors
@app.errorhandler(Exception)
def handle_unexpected_error(e):
    # Log the error (could be expanded later)
    print(f"Unexpected error: {e}")
    return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
