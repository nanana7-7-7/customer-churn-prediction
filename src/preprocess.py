import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load raw dataset."""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw churn dataset."""
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df