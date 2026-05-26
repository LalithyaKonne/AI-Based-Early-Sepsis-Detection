import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# ==============================
# DATA BALANCING
# ==============================
def balance_data(X, y):
    print("Applying SMOTE balancing...")
    smote = SMOTE(
        sampling_strategy=0.8,
        k_neighbors=3,
        random_state=42
    )
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


# ==============================
# XGBoost Model
# ==============================
def train_xgboost(X, y):
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scale_pos_weight = (len(y) - sum(y)) / sum(y)

    model = XGBClassifier(
        n_estimators=1400,
        max_depth=10,
        learning_rate=0.015,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=1.5,
        min_child_weight=4,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_imputed, y)
    model.imputer = imputer
    return model


# ==============================
# LightGBM Model
# ==============================
def train_lightgbm(X, y):
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    print("Training LightGBM...")
    model = LGBMClassifier(
        n_estimators=900,
        learning_rate=0.02,
        num_leaves=60,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_imputed, y)
    model.imputer = imputer
    return model


# ==============================
# Train Both Models
# ==============================
def train_best_models(X, y):
    # BALANCE DATA FIRST
    X_bal, y_bal = balance_data(X, y)

    from sklearn.feature_selection import SelectFromModel

    print("Performing feature selection...")
    selector = SelectFromModel(
        XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        threshold="mean"
    )

    selector.fit(X_bal, y_bal)
    
    # Store feature names before transformation
    feature_names = X.columns.tolist()
    
    X_bal_selected = selector.transform(X_bal)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"Selected {len(selected_features)} features out of {len(feature_names)}")

    # Models expect numpy arrays from selector
    xgb_model = train_xgboost(X_bal_selected, y_bal)
    lgb_model = train_lightgbm(X_bal_selected, y_bal)

    return {
        "xgb": xgb_model,
        "lgb": lgb_model,
        "selector": selector,
        "selected_features": selected_features
    }


# ==============================
# Ensemble Prediction
# ==============================
def ensemble_predict(models, X):
    X_xgb = models["xgb"].imputer.transform(X)
    X_lgb = models["lgb"].imputer.transform(X)

    xgb_prob = models["xgb"].predict_proba(X_xgb)[:, 1]
    lgb_prob = models["lgb"].predict_proba(X_lgb)[:, 1]

    # Better weighted ensemble
    final_prob = 0.65 * xgb_prob + 0.35 * lgb_prob

    return final_prob