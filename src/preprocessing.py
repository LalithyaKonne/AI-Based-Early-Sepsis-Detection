import pandas as pd
import numpy as np

STANDARD_COLUMNS = {
    "hr": "HR", "o2sat": "O2Sat", "temp": "Temp", "sbp": "SBP", "map": "MAP", "dbp": "DBP",
    "resp": "Resp", "etco2": "EtCO2", "baseexcess": "BaseExcess", "hco3": "HCO3", "fio2": "FiO2",
    "ph": "pH", "paco2": "PaCO2", "sao2": "SaO2", "ast": "AST", "bun": "BUN", "alkalinephos": "Alkalinephos",
    "calcium": "Calcium", "chloride": "Chloride", "creatinine": "Creatinine", "bilirubin_direct": "Bilirubin_direct",
    "glucose": "Glucose", "lactate": "Lactate", "magnesium": "Magnesium", "phosphate": "Phosphate",
    "potassium": "Potassium", "bilirubin_total": "Bilirubin_total", "hct": "Hct", "hgb": "Hgb",
    "ptt": "PTT", "wbc": "WBC", "fibrinogen": "Fibrinogen", "platelets": "Platelets", "age": "Age",
    "gender": "Gender", "unit1": "Unit1", "unit2": "Unit2", "hospadmtime": "HospAdmTime", "iculos": "ICULOS",
    "sepsislabel": "SepsisLabel"
}

def preprocess(df):
    df = df.copy()

    # 1. Standardize column names case-insensitively
    df.columns = [STANDARD_COLUMNS.get(col.lower(), col) for col in df.columns]

    # 2. Coerce string/object numeric values to proper floats
    for col in df.columns:
        if col not in ["PatientID", "SepsisLabel"]:
            if df[col].dtype == object:
                # Replace commas with dots in case of European number formats
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Ensure ICULOS column exists
    if "ICULOS" not in df.columns:
        if len(df) == 1:
            df["ICULOS"] = 1
        else:
            df["ICULOS"] = np.arange(1, len(df) + 1)

    # 4. Add missing-value indicators (VERY IMPORTANT)
    for col in df.columns:
        if col not in ["PatientID", "SepsisLabel"]:
            df[col + "_missing"] = df[col].isna().astype(int)

    # 5. Forward-backward fill
    df = df.groupby("PatientID", group_keys=False).apply(
        lambda g: g.ffill().bfill()
    )

    # 6. Replace remaining missing values
    df = df.fillna(0)

    # Drop useless column
    if "HospAdmTime" in df.columns:
        df = df.drop(columns=["HospAdmTime"])

    return df
