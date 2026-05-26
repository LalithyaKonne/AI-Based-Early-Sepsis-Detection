import numpy as np

ORGAN_FEATURES = {
    "Kidney": ["Creatinine_mean", "BUN_mean"],
    "Liver": ["Bilirubin_total_mean", "AST_mean", "ALT_mean"],
    "Respiratory": ["Resp_mean", "O2Sat_mean"],
    "Cardiovascular": ["SBP_mean", "DBP_mean", "MAP_mean"]
}

def most_affected_organ(row):
    scores = {}
    for organ, feats in ORGAN_FEATURES.items():
        vals = [row[f] for f in feats if f in row]
        scores[organ] = np.nanmean(vals) if vals else 0
    return max(scores, key=scores.get)


# -----------------------------
# Sepsis severity (calibrated)
# -----------------------------
def severity(prob):
    """
    Convert sepsis probability into severity level
    """
    if prob < 0.05:
        return "Low"
    elif prob < 0.2:
        return "Medium"
    else:
        return "High"
