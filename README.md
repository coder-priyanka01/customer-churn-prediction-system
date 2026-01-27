# 📊 Customer Churn Prediction & Business Intelligence System

This project is an **end-to-end Machine Learning application** that predicts whether a customer is likely to **churn (leave the service)**. It covers the complete ML lifecycle — from data preprocessing and model training to deployment using **Streamlit**.

---

## 🚀 Project Overview

Customer churn prediction helps businesses identify customers who are at risk of leaving so they can take preventive actions.

This system:

* Trains a machine learning model on customer data
* Saves the trained model and scaler using **joblib**
* Deploys the model as a **web application** using Streamlit
* Predicts churn along with probability

---

## 🧠 Machine Learning Details

* **Problem Type:** Binary Classification
* **Target Variable:** Churn (Yes / No)
* **Model:** Trained ML classification model (saved as `model.pkl`)
* **Scaler:** Feature scaling using StandardScaler (`scaler.pkl`)

### 🔢 Input Features (Exact Order Used in Training)

1. Tenure (months)
2. Monthly Charges
3. Total Charges
4. Senior Citizen (0 = No, 1 = Yes)
5. Contract Type

   * Month-to-month → 0
   * One year → 1
   * Two year → 2
6. Payment Method

   * Electronic check → 0
   * Mailed check → 1
   * Bank transfer → 2
   * Credit card → 3

⚠️ **Important:** Feature order must exactly match the training data.

---

## 📁 Project Structure

```
├── Customer_Churn_Prediction_&_Business_Intelligence_System.ipynb
├── app.py
├── model.pkl
├── scaler.pkl
├── README.md
```

---

## 🖥️ Streamlit Web App (`app.py`)

The Streamlit app allows users to:

* Enter customer details via sidebar
* Scale inputs using the trained scaler
* Predict churn using the saved ML model
* Display churn probability

### Key Libraries Used

* streamlit
* numpy
* joblib

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install streamlit numpy scikit-learn joblib
```

### 2️⃣ Run Streamlit App

```bash
streamlit run app.py
```

### 3️⃣ Open in Browser

Streamlit will open automatically or visit:

```
http://localhost:8501
```

---

## 📊 Output

* **Prediction:** Churn / No Churn
* **Probability:** Likelihood of customer churn

---

## 💡 Business Use Case

* Retention strategy planning
* Customer risk segmentation
* Revenue loss prevention
* Decision support system

---

## 🛠️ Tools & Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* Joblib
* Streamlit

---

## 📌 Future Improvements

* Add more customer features
* Use advanced models (XGBoost, Random Forest)
* Connect with live database
* Add dashboards & BI insights

---

## 👩‍💻 Author

**Priyanka**
Machine Learning & Data Science Project
