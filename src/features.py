import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Temporal Feature Engineering
# -----------------------------
def add_temporal_features(df):

    temp_features = df.groupby("PatientID").apply(
        lambda g: pd.Series({
            "HR_slope": (g["HR"].iloc[-1] - g["HR"].iloc[0]) if len(g) > 1 and "HR" in g else 0,
            "Temp_slope": (g["Temp"].iloc[-1] - g["Temp"].iloc[0]) if len(g) > 1 and "Temp" in g else 0,
            "MAP_drop": (g["MAP"].iloc[0] - g["MAP"].iloc[-1]) if len(g) > 1 and "MAP" in g else 0,
            "HR_volatility": g["HR"].std() if len(g) > 1 and "HR" in g else 0,
            "MAP_velocity": (g["MAP"].diff().mean()) if len(g) > 1 and "MAP" in g else 0,
            "Platelet_drop": (g["Platelets"].iloc[0] - g["Platelets"].iloc[-1]) if len(g) > 1 and "Platelets" in g else 0
        })
    )

    return temp_features


# -----------------------------
# Main Feature Extraction
# -----------------------------
def extract_features(df, hours=6):

    df_early = df[df["ICULOS"] <= hours].copy()

    # -----------------------
    # Statistical features
    # -----------------------
    agg_funcs = ["mean", "std", "max", "min"]

    X_stat = df_early.groupby("PatientID").agg(agg_funcs)
    X_stat.columns = ["_".join(c) for c in X_stat.columns]


    # -----------------------
    # Trend features
    # -----------------------
    def compute_slope(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]


    X_trend = df_early.groupby("PatientID").agg(
        lambda x: compute_slope(x) if x.name not in ["SepsisLabel", "ICULOS"] else 0
    )

    X_trend = X_trend.drop(columns=["SepsisLabel", "ICULOS"], errors="ignore")
    X_trend.columns = [c + "_trend" for c in X_trend.columns]


    # -----------------------
    # Range features
    # -----------------------
    X_var = df_early.groupby("PatientID").agg(lambda x: x.max() - x.min())
    X_var = X_var.drop(columns=["SepsisLabel", "ICULOS"], errors="ignore")
    X_var.columns = [c + "_range" for c in X_var.columns]


    # -----------------------
    # Clinical features
    # -----------------------
    X_extra = df_early.groupby("PatientID").apply(
        lambda g: pd.Series({
            "Shock_Risk_Flag": int(((g["HR"] / (g["MAP"] + 1e-6)) > 1).any()) if "HR" in g and "MAP" in g else 0,
            "Shock_Index_mean": (g["HR"] / (g["MAP"] + 1e-6)).mean() if "HR" in g and "MAP" in g else 0,
            "Shock_Index_max": (g["HR"] / (g["MAP"] + 1e-6)).max() if "HR" in g and "MAP" in g else 0,

            "Instability_Index": (
                ((g["HR"].diff() > 0) & (g["MAP"].diff() < 0)).sum() / len(g)
                if len(g) > 1 and "HR" in g and "MAP" in g else 0
            ),

            "Fever_Flag": int((g["Temp"] > 38).any()) if "Temp" in g else 0,
            "Hypotension_Flag": int((g["MAP"] < 65).any()) if "MAP" in g else 0,
            "Renal_Risk_Flag": int((g["Creatinine"] > 1.5).any()) if "Creatinine" in g else 0,
            "Coagulation_Risk_Flag": int((g["Platelets"] < 150).any()) if "Platelets" in g else 0,

            "Tachycardia": int((g["HR"] > 100).any()) if "HR" in g else 0,
            "Tachypnea": int((g["Resp"] > 22).any()) if "Resp" in g else 0,
            "Hypothermia": int((g["Temp"] < 36).any()) if "Temp" in g else 0,
            "Severe_Hypotension": int((g["MAP"] < 60).any()) if "MAP" in g else 0,

            "HR_rolling_mean_6h": g["HR"].rolling(6, min_periods=1).mean().iloc[-1] if "HR" in g else 0,
            "MAP_rolling_mean_6h": g["MAP"].rolling(6, min_periods=1).mean().iloc[-1] if "MAP" in g else 0,
            "HR_trend_last_3h": (g["HR"].iloc[-1] - g["HR"].iloc[-4]) if len(g) >= 4 and "HR" in g else 0,
            "MAP_trend_last_3h": (g["MAP"].iloc[-1] - g["MAP"].iloc[-4]) if len(g) >= 4 and "MAP" in g else 0
        })
    )


    # -----------------------
    # Temporal features
    # -----------------------
    X_temporal = add_temporal_features(df_early)


    # -----------------------
    # Final feature set
    # -----------------------
    X = pd.concat([X_stat, X_trend, X_var, X_extra, X_temporal], axis=1)

    # -----------------------
    # Basic Interaction Features (Phase 1)
    # -----------------------
    if "HR_mean" in X.columns and "MAP_mean" in X.columns:
        X["HR_MAP_ratio"] = X["HR_mean"] / (X["MAP_mean"] + 1e-6)

    if "BUN_mean" in X.columns and "Creatinine_mean" in X.columns:
        X["BUN_Creatinine_ratio"] = X["BUN_mean"] / (X["Creatinine_mean"] + 1e-6)

    # -----------------------
    # Enhanced Interaction Features (Phase 3)
    # -----------------------
    if "HR_mean" in X.columns and "MAP_mean" in X.columns:
        X["Shock_Index"] = X["HR_mean"] / (X["MAP_mean"] + 1e-6)

    if "Resp_mean" in X.columns and "O2Sat_mean" in X.columns:
        X["Resp_O2_ratio"] = X["Resp_mean"] / (X["O2Sat_mean"] + 1e-6)

    if "Temp_mean" in X.columns and "HR_mean" in X.columns:
        X["Temp_HR_product"] = X["Temp_mean"] * X["HR_mean"]

    if "Creatinine_mean" in X.columns and "BUN_mean" in X.columns:
        X["Kidney_Risk_Index"] = X["Creatinine_mean"] * X["BUN_mean"]

    X = X.fillna(0)
    return X