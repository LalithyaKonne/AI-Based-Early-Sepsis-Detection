import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

import numpy as np

def plot_feature_importance(model, feature_names, top_n=20):

    if hasattr(model, "feature_importances_"):

        importances = model.feature_importances_

        indices = np.argsort(importances)[-top_n:]

        plt.figure(figsize=(8,6))

        plt.barh(range(len(indices)), importances[indices])

        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

        plt.xlabel("Importance")
        plt.title("Top Feature Importance")

        plt.show()

    else:
        print("Model does not support feature importance.")
# -----------------------------
# ROC Curve
# -----------------------------
def plot_roc_curve(y_true, y_prob):

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()
    plt.grid(True)

    plt.show()


# -----------------------------
# Confusion Matrix
# -----------------------------
def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Sepsis","Sepsis"],
        yticklabels=["No Sepsis","Sepsis"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.show()


# -----------------------------
# Precision Recall Curve
# -----------------------------
def plot_precision_recall(y_true, y_prob):
    from sklearn.metrics import average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}", color="green")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Academic Metrics)")

    plt.legend()
    plt.grid(True)

    plt.show()
