import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os
import joblib

# Get absolute base path
base_path = os.path.dirname(os.path.abspath(__file__))

# Paths to model and preprocessor
model_path = os.path.join(base_path, "heart_attack_model.pkl")
preprocessor_path = os.path.join(base_path, "heart_attack_preprocessor.pkl")

# Debug: print the paths
import streamlit as st
st.write("Model Path:", model_path)
st.write("Preprocessor Path:", preprocessor_path)

# Check if files exist
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
else:
    model = joblib.load(model_path)

if not os.path.exists(preprocessor_path):
    st.error(f"Preprocessor file not found at {preprocessor_path}")
else:
    preprocessor = joblib.load(preprocessor_path)

# 🧠 Define UI
st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")
st.title("🫀 Heart Attack Risk Predictor")
st.markdown("Use lifestyle, demographic, and clinical inputs to predict heart attack risk.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        sleep_hours = st.slider("Sleep Hours/Day", 0, 12, 6)
        sedentary_hours = st.slider("Sedentary Hours/Day", 0.0, 15.0, 6.0)
        activity_days = st.slider("Physical Activity Days/Week", 0, 7, 3)
        exercise_hours = st.slider("Exercise Hours/Week", 0, 14, 3)
        systolic_bp = st.slider("Systolic BP", 80, 200, 120)
        diastolic_bp = st.slider("Diastolic BP", 50, 130, 80)
        obesity = st.selectbox("Obesity", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        income = st.slider("Income ($)", 1000, 100000, 30000)
        diet = st.selectbox("Diet", ["Healthy", "Unhealthy"])

    with col2:
        cholesterol = st.slider("Cholesterol", 100, 400, 200)
        heart_rate = st.slider("Heart Rate", 50, 150, 70)
        bmi = st.slider("BMI", 10.0, 50.0, 22.0)
        triglycerides = st.slider("Triglycerides", 50, 500, 150)
        diabetes = st.selectbox("Diabetes", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        smoking = st.selectbox("Smoking", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        family_history = st.selectbox("Family History", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        previous_heart_problems = st.selectbox("Previous Heart Problems", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        alcohol_consumption = st.selectbox("Alcohol Consumption", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
        medication_use = st.selectbox("Medication Use", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        hemisphere = st.selectbox("Hemisphere", ["Northern Hemisphere", "Southern Hemisphere"])
        continent = st.selectbox("Continent", ["Asia", "Europe", "North America", "South America", "Africa", "Oceania"])
        country = st.text_input("Country", "United States")

    submitted = st.form_submit_button("Predict Risk")

# 🔄 If user submits the form
if submitted:
    # Build single-row DataFrame with correct column names
    input_dict = {
        'Age': age,
        'Sex': sex,
        'Sleep Hours Per Day': sleep_hours,
        'Sedentary Hours Per Day': sedentary_hours,
        'Physical Activity Days Per Week': activity_days,
        'Exercise Hours Per Week': exercise_hours,
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp,
        'Obesity': obesity,
        'Income': income,
        'Diet': diet,
        'Cholesterol': cholesterol,
        'Heart Rate': heart_rate,
        'BMI': bmi,
        'Triglycerides': triglycerides,
        'Diabetes': diabetes,
        'Smoking': smoking,
        'Family History': family_history,
        'Previous Heart Problems': previous_heart_problems,
        'Alcohol Consumption': alcohol_consumption,
        'Stress Level': stress_level,
        'Medication Use': medication_use,
        'Hemisphere': hemisphere,
        'Continent': continent,
        'Country': country
    }

    input_df = pd.DataFrame([input_dict])

    # 🌀 Transform with preprocessor
    input_processed = preprocessor.transform(input_df)

    # 🤖 Predict
    pred = model.predict(input_processed)[0]
    prob = model.predict_proba(input_processed)[0][1]

    # 📣 Output
    st.markdown("---")
    st.subheader("🩺 Prediction Result")
    st.write(f"**Heart Attack Risk:** {'High' if pred == 1 else 'Low'}")
    st.write(f"**Risk Probability:** {prob * 100:.2f}%")

    if pred == 1:
        st.warning("⚠️ High Risk: Recommend lifestyle changes and clinical evaluation.")
    else:
        st.success("✅ Low Risk: Keep maintaining a healthy lifestyle.")
