import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_model.pkl"
RESULTS_PATH = "reports/model_results.csv"


def load_data(data_path: str) -> pd.DataFrame:
    """Load raw dataset."""
    return pd.read_csv(data_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw churn dataset."""
    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with 0
    # In this dataset, missing TotalCharges often appears when tenure is 0.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop customer ID because it is not useful for prediction
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convert target variable to binary
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return preprocessor


def evaluate_model(model_name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate trained model and return metrics."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    return metrics


def train_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """Train multiple models and return results and trained pipelines."""
    preprocessor = create_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []
    trained_pipelines = {}

    for model_name, model in models.items():
        print(f"Training: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(model_name, pipeline, X_test, y_test)
        results.append(metrics)

        trained_pipelines[model_name] = pipeline

        print(f"Finished: {model_name}")
        print(metrics)
        print("-" * 50)

    results_df = pd.DataFrame(results)

    return results_df, trained_pipelines


def select_best_model(results_df: pd.DataFrame, trained_pipelines: dict):
    """Select best model based on ROC-AUC."""
    best_model_name = results_df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    best_model = trained_pipelines[best_model_name]

    return best_model_name, best_model


def save_outputs(best_model: Pipeline, results_df: pd.DataFrame):
    """Save trained model and evaluation results."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Results saved to: {RESULTS_PATH}")


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

    print("Training models...")
    results_df, trained_pipelines = train_models(X_train, y_train, X_test, y_test)

    print("Model comparison:")
    print(results_df.sort_values("roc_auc", ascending=False))

    best_model_name, best_model = select_best_model(results_df, trained_pipelines)

    print(f"Best model: {best_model_name}")

    save_outputs(best_model, results_df)

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()