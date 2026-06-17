import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)


DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_model.pkl"

CONFUSION_MATRIX_PATH = "reports/figures/confusion_matrix.png"
ROC_CURVE_PATH = "reports/figures/roc_curve.png"
EVALUATION_REPORT_PATH = "reports/evaluation_report.txt"


def load_data(data_path: str) -> pd.DataFrame:
    """Load raw dataset."""
    return pd.read_csv(data_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw churn dataset."""
    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop customerID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convert target variable
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


def load_model(model_path: str):
    """Load trained model."""
    return joblib.load(model_path)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    report = classification_report(y_test, y_pred)

    return metrics, report, y_pred, y_proba


def save_evaluation_report(metrics: dict, report: str):
    """Save evaluation report as text file."""
    os.makedirs("reports", exist_ok=True)

    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n\n")

        for metric_name, score in metrics.items():
            f.write(f"{metric_name}: {score:.4f}\n")

        f.write("\nClassification Report\n")
        f.write("=====================\n")
        f.write(report)

    print(f"Evaluation report saved to: {EVALUATION_REPORT_PATH}")


def plot_confusion_matrix(y_test: pd.Series, y_pred):
    """Plot and save confusion matrix."""
    os.makedirs("reports/figures", exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.show()

    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


def plot_roc_curve(y_test: pd.Series, y_proba):
    """Plot and save ROC curve."""
    os.makedirs("reports/figures", exist_ok=True)

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=300)
    plt.show()

    print(f"ROC curve saved to: {ROC_CURVE_PATH}")


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    df = preprocess_data(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    print("Evaluating model...")
    metrics, report, y_pred, y_proba = evaluate_model(model, X_test, y_test)

    print("\nEvaluation Metrics")
    print("==================")
    for metric_name, score in metrics.items():
        print(f"{metric_name}: {score:.4f}")
    
    print("\nClassification Report")
    print("=====================")
    print(report)

    save_evaluation_report(metrics, report)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)

    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()