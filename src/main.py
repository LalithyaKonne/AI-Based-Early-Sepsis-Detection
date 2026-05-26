import os
import pandas as pd
import joblib
import numpy as np

from data_loader import load_data
from preprocessing import preprocess
from features import extract_features
from train_models import train_best_models
from utils import severity, most_affected_organ
from visualize import (
    plot_feature_importance, plot_roc_curve,
    plot_confusion_matrix, plot_precision_recall
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report, precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_FILE = os.path.join(BASE_DIR, "models", "final_sepsis_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data")

EARLY_HOURS = 12


# ==========================================
# TRAIN MODEL
# ==========================================
def train_model():

    print("\nLoading dataset...")
    dataA = load_data(os.path.join(BASE_DIR, "data", "training_setA"))
    dataB = load_data(os.path.join(BASE_DIR, "data", "training_setB"))

    data = pd.concat([dataA, dataB], ignore_index=True)
    print("Preprocessing...")
    data = preprocess(data)

    print("Extracting features...")
    X_raw = extract_features(data, hours=EARLY_HOURS)

    print("Scaling features...")
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns, index=X_raw.index)

    y = data.groupby("PatientID")["SepsisLabel"].max()
    y = y.loc[X.index]

    print("\nClass Distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining Models...\n")
    models_dict = train_best_models(X_train, y_train)

    xgb = models_dict["xgb"]
    lgb = models_dict["lgb"]
    selector = models_dict.get("selector")

    # Transformation for Test Set
    if selector:
        X_test_selected = selector.transform(X_test)
    else:
        X_test_selected = X_test

    # Predictions
    X_test_xgb = xgb.imputer.transform(X_test_selected)
    X_test_lgb = lgb.imputer.transform(X_test_selected)

    xgb_prob = xgb.predict_proba(X_test_xgb)[:, 1]
    lgb_prob = lgb.predict_proba(X_test_lgb)[:, 1]

    final_prob = 0.65 * xgb_prob + 0.35 * lgb_prob

    # Metrics
    auc = roc_auc_score(y_test, final_prob)
    auprc = average_precision_score(y_test, final_prob)

    print(f"\nAUC: {auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

    # Threshold tuning
    precisions, recalls, thresholds = precision_recall_curve(y_test, final_prob)

    best_threshold = 0.5
    best_score = 0

    for t in thresholds:

        preds = (final_prob >= t).astype(int)

        recall = recall_score(y_test, preds)
        precision = precision_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Refined medical objective: push recall while protecting precision
        score = (0.60 * recall) + (0.25 * precision) + (0.15 * f1)

        if score > best_score:
            best_score = score
            best_threshold = t


    # Threshold clamping as requested by user (0.30 - 0.50)
    if best_threshold > 0.50:
        best_threshold = 0.50
    elif best_threshold < 0.30:
        best_threshold = 0.30

    preds = (final_prob >= best_threshold).astype(int)

    print("\nFinal Performance")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"F1 Score:  {f1_score(y_test, preds):.4f}")

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # 📈 Generate Visualizations
    print("\nGenerating training visualizations...")
    plot_feature_importance(xgb, models_dict["selected_features"])
    plot_roc_curve(y_test, final_prob)
    plot_confusion_matrix(y_test, preds)
    plot_precision_recall(y_test, final_prob)

    # Save model
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

    joblib.dump({
        "model": models_dict,
        "features": X.columns.tolist(),
        "threshold": float(best_threshold),
        "scaler": scaler
    }, MODEL_FILE)

    print(f"\nModel saved at: {MODEL_FILE}")


# ==========================================
# PREDICT
# ==========================================
def predict_patient():

    if not os.path.exists(MODEL_FILE):
        print("❌ Train model first!")
        return

    saved_model = joblib.load(MODEL_FILE)

    patient_id = input("Enter Patient ID: ")

    file_path = None

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.lower() == f"{patient_id.lower()}.psv":
                file_path = os.path.join(root, file)
                break

    if file_path is None:
        print("❌ Patient not found")
        return

    df = pd.read_csv(file_path, sep="|")
    df["PatientID"] = patient_id

    df = preprocess(df)
    X_raw = extract_features(df, hours=EARLY_HOURS)
    
    scaler = saved_model.get("scaler")
    if scaler:
        X = pd.DataFrame(scaler.transform(X_raw), columns=X_raw.columns, index=X_raw.index)
    else:
        X = X_raw

    models_dict = saved_model["model"]
    xgb = models_dict["xgb"]
    lgb = models_dict["lgb"]
    selector = models_dict.get("selector")

    if selector:
        X_selected = selector.transform(X)
    else:
        X_selected = X

    X_xgb = xgb.imputer.transform(X_selected)
    X_lgb = lgb.imputer.transform(X_selected)

    xgb_prob = xgb.predict_proba(X_xgb)[0, 1]
    lgb_prob = lgb.predict_proba(X_lgb)[0, 1]

    prob = 0.65 * xgb_prob + 0.35 * lgb_prob

    threshold = saved_model["threshold"]

    prediction = "SEPSIS" if prob >= threshold else "NO SEPSIS"

    sev = severity(prob)
    organ = most_affected_organ(X.iloc[0]) if prob >= threshold else "N/A"

    print("\n================================")
    print("   SEPSIS PREDICTION RESULT")
    print("================================")
    print(f"Patient ID : {patient_id}")
    print(f"Prediction : {prediction}")
    print(f"Probability: {prob:.4f}")
    print(f"Severity   : {sev}")
    print(f"Organ      : {organ}")
    print("================================")


# ==========================================
# MAIN MENU
# ==========================================
def main():

    print("\n===================================")
    print("   AI-Based Early Sepsis Detection")
    print("===================================")
    print("1️⃣ Train Model")
    print("2️⃣ Predict Patient")
    print("===================================")

    choice = input("Enter choice (1/2): ")

    if choice == "1":
        train_model()
    elif choice == "2":
        predict_patient()
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()


# ==========================================
# API SUPPORT FUNCTIONS
# ==========================================

def get_prediction_data(df, saved_model, patient_id):
    """
    Helper for API to process raw patient data and return full prediction payload.
    """
    try:
        # Preprocess
        df["PatientID"] = patient_id
        df_processed = preprocess(df)
        
        # Extract Features (using the full window available)
        X_raw = extract_features(df_processed, hours=df_processed["ICULOS"].max())

        # Align features with the ones seen during training
        expected_features = saved_model.get("features", [])
        if expected_features and not X_raw.empty:
            # Reindex to ensure all columns exist and are in the same order
            X_raw = X_raw.reindex(columns=expected_features, fill_value=0)
            
        scaler = saved_model.get("scaler")
        if scaler:
            X = pd.DataFrame(scaler.transform(X_raw), columns=X_raw.columns, index=X_raw.index)
        else:
            X = X_raw
            
        models_dict = saved_model.get("model", saved_model)
        xgb = models_dict.get("xgb")
        lgb = models_dict.get("lgb")
        
        if not xgb or not lgb:
            return {"error": "Invalid model structure: missing xgb or lgb"}
        # Transform & Predict
        selector = models_dict.get("selector")
        if selector:
            X_selected = selector.transform(X)
        else:
            X_selected = X
            
        X_xgb = xgb.imputer.transform(X_selected)
        X_lgb = lgb.imputer.transform(X_selected)
        
        xgb_prob = xgb.predict_proba(X_xgb)[0, 1]
        lgb_prob = lgb.predict_proba(X_lgb)[0, 1]
        prob = 0.65 * xgb_prob + 0.35 * lgb_prob
        
        # Get latest vitals
        last_row = df.iloc[-1]
        vitals = {
            "hr": float(last_row["HR"]) if "HR" in last_row and not pd.isna(last_row["HR"]) else 0,
            "temp": float(last_row["Temp"]) if "Temp" in last_row and not pd.isna(last_row["Temp"]) else 0,
            "resp": float(last_row["Resp"]) if "Resp" in last_row and not pd.isna(last_row["Resp"]) else 0,
            "map": float(last_row["MAP"]) if "MAP" in last_row and not pd.isna(last_row["MAP"]) else 0,
            "o2sat": float(last_row["O2Sat"]) if "O2Sat" in last_row and not pd.isna(last_row["O2Sat"]) else 0,
            "wbc": float(last_row["WBC"]) if "WBC" in last_row and not pd.isna(last_row["WBC"]) else 0,
            "creatinine": float(last_row["Creatinine"]) if "Creatinine" in last_row and not pd.isna(last_row["Creatinine"]) else 0,
            "platelets": float(last_row["Platelets"]) if "Platelets" in last_row and not pd.isna(last_row["Platelets"]) else 0
        }
        
        return {
            "prob": float(prob),
            "severity": severity(prob),
            "organ": most_affected_organ(X.iloc[0]) if prob >= saved_model.get("threshold", 0.4) else "N/A",
            "vitals": vitals,
            "df_processed": df_processed,
            "X_new": X_selected,
            "icu_hours": int(df_processed["ICULOS"].max())
        }
    except Exception as e:
        import traceback
        print(f"Error in get_prediction_data: {e}")
        traceback.print_exc()
        return {"error": str(e)}


def get_realtime_timeline(df_processed, saved_model):
    """
    Generates historical risk and heart rate timelines for GUI plotting.
    """
    risk_timeline = []
    hr_timeline = []
    
    models_dict = saved_model["model"]
    xgb = models_dict["xgb"]
    lgb = models_dict["lgb"]
    
    # We use a moving window to simulate how risk evolved
    indices = df_processed.index.tolist()
    
    # To speed up, we sample points if the history is very long
    step = 1 if len(indices) < 24 else len(indices) // 12
    
    for i in range(1, len(indices) + 1, step):
        df_slice = df_processed.iloc[:i]
        iculos = df_slice["ICULOS"].iloc[-1]
        
        # Extract features for this slice
        X_slice_raw = extract_features(df_slice, hours=iculos)
        
        # Align features
        expected_features = saved_model.get("features", [])
        if expected_features:
            X_slice_raw = X_slice_raw.reindex(columns=expected_features, fill_value=0)
            
        scaler = saved_model.get("scaler")
        if scaler:
            X_slice = pd.DataFrame(scaler.transform(X_slice_raw), columns=X_slice_raw.columns, index=X_slice_raw.index)
        else:
            X_slice = X_slice_raw
            
        selector = models_dict.get("selector")
        if selector:
            X_selected = selector.transform(X_slice)
        else:
            X_selected = X_slice
            
        X_xgb = xgb.imputer.transform(X_selected)
        X_lgb = lgb.imputer.transform(X_selected)
        
        xgb_p = xgb.predict_proba(X_xgb)[0, 1]
        lgb_p = lgb.predict_proba(X_lgb)[0, 1]
        prob = 0.65 * xgb_p + 0.35 * lgb_p
        
        risk_timeline.append({
            "time": int(iculos),
            "prob": round(float(prob), 4)
        })
        
        hr_timeline.append({
            "time": int(iculos),
            "value": float(df_slice["HR"].iloc[-1]) if "HR" in df_slice and not pd.isna(df_slice["HR"].iloc[-1]) else 0
        })
        
    return risk_timeline, hr_timeline
