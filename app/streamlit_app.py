import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/churn_model.pkl"


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


@st.cache_resource
def load_model():
    """Load trained model."""
    model = joblib.load(MODEL_PATH)
    return model


def predict_churn(input_data: dict):
    """Predict churn probability."""
    model = load_model()

    input_df = pd.DataFrame([input_data])

    churn_probability = model.predict_proba(input_df)[:, 1][0]
    churn_prediction = model.predict(input_df)[0]

    return churn_prediction, churn_probability


def get_risk_level(probability: float) -> str:
    """Convert probability to risk level."""
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"


st.title("Customer Churn Prediction App")
st.write(
    """
    顧客情報を入力すると、学習済みの機械学習モデルが  
    **解約確率** と **リスクレベル** を予測します。
    """
)

st.divider()

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure Months", min_value=0, max_value=72, value=12)

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0,
        step=10.0
    )
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])


st.subheader("Service Information")

col3, col4 = st.columns(2)

with col3:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col4:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


input_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}


st.divider()

if st.button("Predict Churn Risk"):
    prediction, probability = predict_churn(input_data)
    risk_level = get_risk_level(probability)

    st.header("Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{probability:.2%}"
    )

    if risk_level == "High":
        st.error("Risk Level: High")
    elif risk_level == "Medium":
        st.warning("Risk Level: Medium")
    else:
        st.success("Risk Level: Low")

    if prediction == 1:
        st.write("この顧客は **解約する可能性が高い** と予測されました。")
    else:
        st.write("この顧客は **継続する可能性が高い** と予測されました。")

    st.subheader("Input Data")
    st.dataframe(pd.DataFrame([input_data]))