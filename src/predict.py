import joblib
import pandas as pd


MODEL_PATH = "models/churn_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    """Load trained model."""
    model = joblib.load(model_path)
    return model


def predict_churn(input_data: dict, model_path: str = MODEL_PATH) -> dict:
    """
    Predict churn probability for one customer.

    Parameters
    ----------
    input_data : dict
        Customer information.
    model_path : str
        Path to trained model.

    Returns
    -------
    dict
        Prediction result.
    """
    model = load_model(model_path)

    input_df = pd.DataFrame([input_data])

    churn_probability = model.predict_proba(input_df)[:, 1][0]
    churn_prediction = model.predict(input_df)[0]

    if churn_probability >= 0.7:
        risk_level = "High"
    elif churn_probability >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    result = {
        "prediction": int(churn_prediction),
        "churn_probability": float(churn_probability),
        "risk_level": risk_level
    }

    return result


def main():
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1026.0
    }

    result = predict_churn(sample_customer)

    print("Prediction Result")
    print("=================")
    print(f"Prediction: {result['prediction']}")
    print(f"Churn Probability: {result['churn_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")

    if result["prediction"] == 1:
        print("Result: This customer is likely to churn.")
    else:
        print("Result: This customer is likely to stay.")


if __name__ == "__main__":
    main()