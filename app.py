import streamlit as st
import pandas as pd
import joblib
import pickle
import os

# Load paths
pkl_path = "pkl"


model = joblib.load(os.path.join(pkl_path, "salary_model.pkl"))
with open(os.path.join(pkl_path, "label_encoders.pkl"), "rb") as f:
    le_dict = pickle.load(f)
with open(os.path.join(pkl_path, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(pkl_path, "feature_order.pkl"), "rb") as f:
    feature_order = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction App")

with st.form("prediction_form"):
    st.subheader("Enter Employee Details:")

    sex = st.selectbox("Gender", ["Male", "Female"])
    unit = st.selectbox("Department", ["HR", "Finance", "IT", "Marketing", "Operations", "Sales"])
    designation = st.selectbox("Designation", [
        "Analyst", "Executive", "Manager", "Senior Analyst", "Senior Manager", "Team Lead"
    ])
    age = st.slider("Age", 18, 65, 30)
    past_exp = st.number_input("Past Experience (in years)", min_value=0.0, step=0.5)
    ratings = st.slider("Performance Rating (1-5)", 1, 5, 3)
    years_exp = st.number_input("Current Job Experience (in years)", min_value=0.0, step=0.5)

    submit = st.form_submit_button("Predict Salary")

# Predict salary
if submit:
    df = pd.DataFrame([{
        "SEX": sex,
        "UNIT": unit,
        "DESIGNATION": designation,
        "AGE": age,
        "PAST EXP": past_exp,
        "RATINGS": ratings,
        "years_experience": years_exp
    }])

    for col in le_dict:
        df[col] = le_dict[col].transform(df[col])

    df["TOTAL_EXPERIENCE"] = df["years_experience"] + df["PAST EXP"]

    num_cols = ["AGE", "PAST EXP", "RATINGS", "years_experience", "TOTAL_EXPERIENCE"]
    df[num_cols] = scaler.transform(df[num_cols])

    df_final = df[feature_order]

    salary = model.predict(df_final)[0]
    st.success(f"💰 Predicted Salary: ₹{salary:,.2f}") 
