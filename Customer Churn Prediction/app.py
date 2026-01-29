import streamlit as st
import numpy as np
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ---------------- UI ----------------
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn using ML model.")

st.sidebar.header("Customer Details")

tenure = st.sidebar.number_input("Tenure (months)", 0, 100, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1500.0)

senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)
payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

# ---------------- Encoding ----------------
senior_encoded = 1 if senior == "Yes" else 0

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}

contract_encoded = contract_map[contract]
payment_encoded = payment_map[payment]

# ---------------- Input (⚠️ EXACT 6 FEATURES) ----------------
# ORDER MUST MATCH TRAINING DATA
input_data = np.array([[
    tenure,
    monthly_charges,
    total_charges,
    senior_encoded,
    contract_encoded,
    payment_encoded
]])

# ---------------- Scaling ----------------
input_scaled = scaler.transform(input_data)

# ---------------- Prediction ----------------
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")

    st.write(f"Churn Probability: **{probability:.2%}**")

st.markdown("---")
st.caption("ML Powered Customer Churn Prediction System")
