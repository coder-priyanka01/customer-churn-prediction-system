import streamlit as st
import pickle
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background-color: #020617;
    padding: 2rem;
    border-radius: 12px;
}
h1, h2, h3 {
    color: #e5e7eb;
}
label {
    color: #cbd5f5 !important;
}
.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    width: 100%;
    height: 3em;
    font-size: 16px;
}
.stButton button:hover {
    background-color: #1e40af;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model & Scaler ----------------
@st.cache_resource
def load_files():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_files()

# ---------------- Title ----------------
st.title("📊 Customer Churn Prediction System")
st.write("Enter customer details below (0 = No, 1 = Yes)")

st.divider()

# ---------------- Input Form ----------------
with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.number_input("Gender (0=Female,1=Male)", 0, 1)
        SeniorCitizen = st.number_input("Senior Citizen", 0, 1)
        Partner = st.number_input("Partner", 0, 1)
        Dependents = st.number_input("Dependents", 0, 1)
        tenure = st.number_input("Tenure (months)", 0, 100)

    with col2:
        PhoneService = st.number_input("Phone Service", 0, 1)
        MultipleLines = st.number_input("Multiple Lines", 0, 1)
        InternetService = st.number_input("Internet Service (0=No,1=DSL,2=Fiber)", 0, 2)
        OnlineSecurity = st.number_input("Online Security", 0, 1)
        OnlineBackup = st.number_input("Online Backup", 0, 1)

    with col3:
        DeviceProtection = st.number_input("Device Protection", 0, 1)
        TechSupport = st.number_input("Tech Support", 0, 1)
        StreamingTV = st.number_input("Streaming TV", 0, 1)
        StreamingMovies = st.number_input("Streaming Movies", 0, 1)
        Contract = st.number_input("Contract (0=Month,1=1yr,2=2yr)", 0, 2)

    MonthlyCharges = st.number_input("Monthly Charges", 0.0)
    TotalCharges = st.number_input("Total Charges", 0.0)
    PaperlessBilling = st.number_input("Paperless Billing", 0, 1)
    PaymentMethod = st.number_input("Payment Method (0=E-check,1=Mailed,2=Bank,3=Card)", 0, 3)

    submit = st.form_submit_button("🔍 Predict Churn")

# ---------------- Prediction ----------------
if submit:
    input_data = np.array([[
        gender, SeniorCitizen, Partner, Dependents, tenure,
        PhoneService, MultipleLines, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies,
        Contract, MonthlyCharges, TotalCharges,
        PaperlessBilling, PaymentMethod
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.divider()

    if prediction == 1:
        st.error("❌ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")

# ---------------- Footer ----------------
st.caption("© 2026 | Customer Churn Prediction System")
