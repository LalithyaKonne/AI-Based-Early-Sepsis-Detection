import shap
import matplotlib.pyplot as plt

def explain_patient(model, X):

    try:

        base_model = model

        # Extract model from pipeline
        if hasattr(base_model, "named_steps"):
            base_model = list(base_model.named_steps.values())[-1]

        # Extract model from calibrated model
        if hasattr(base_model, "calibrated_classifiers_"):
            base_model = base_model.calibrated_classifiers_[0].estimator

        explainer = shap.TreeExplainer(base_model)

        shap_values = explainer.shap_values(X)

        print("\nGenerating SHAP Explanation...")

        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[0],
            X.iloc[0]
        )

        plt.show()

    except Exception as e:

        print("SHAP explanation skipped:", e)
