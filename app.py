import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Prediction App")
st.write("Enter the patient's diagnostic clinical measurements below to evaluate diabetes risk.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("mlp_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

IMPUTE_MEDIANS = {
    'Glucose': 121.5,
    'BloodPressure': 72.0,
    'SkinThickness': 29.0,
    'Insulin': 125.0,
    'BMI': 32.3
}

# Collect Inputs from User
st.subheader("Patient Diagnostic Measurements")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, help="Number of times pregnant")
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120, help="Plasma glucose concentration (0 = missing/unknown)")
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70, help="Diastolic blood pressure (0 = missing/unknown)")
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, help="Triceps skin fold thickness (0 = missing/unknown)")

with col2:
    insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, max_value=900, value=80, help="2-Hour serum insulin (0 = missing/unknown)")
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1, help="Body Mass Index (0.0 = missing/unknown)")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, help="Genetic predisposition score")
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=33)

# Prediction Button Logic
if st.button("🔍 Predict Risk", use_container_width=True):
    # Handle zero inputs in physiological parameters (impute with median if 0 entered)
    proc_glucose = IMPUTE_MEDIANS['Glucose'] if glucose == 0 else glucose
    proc_bp = IMPUTE_MEDIANS['BloodPressure'] if blood_pressure == 0 else blood_pressure
    proc_skin = IMPUTE_MEDIANS['SkinThickness'] if skin_thickness == 0 else skin_thickness
    proc_insulin = IMPUTE_MEDIANS['Insulin'] if insulin == 0 else insulin
    proc_bmi = IMPUTE_MEDIANS['BMI'] if bmi == 0.0 else bmi

    # Assemble feature vector in exact dataset order
    input_features = np.array([[pregnancies, proc_glucose, proc_bp, proc_skin, proc_insulin, proc_bmi, dpf, age]])

    # Scale inputs using saved StandardScaler
    scaled_features = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    diabetic_prob = probabilities[1] * 100

    # Display Result
    st.markdown("---")
    st.subheader("Diagnostic Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Diabetes**")
        st.write(f"The model estimates a **{diabetic_prob:.1f}% probability** of diabetes onset.")
    else:
        st.success(f"✅ **Low Risk / Non-Diabetic**")
        st.write(f"The model estimates a **{(100 - diabetic_prob):.1f}% probability** of being non-diabetic.")

    # Show confidence score bar
    st.progress(int(diabetic_prob))