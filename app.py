import streamlit as st
import pandas as pd
import joblib
import numpy as np


model = joblib.load('cardio_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("❤️ Cardiovascular Disease   Predictor❤️💕")
st.write("Enter patient details below to predict the risk of heart disease.")


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", options=[(1, "Female"), (2, "Male")], format_func=lambda x: x[1])[0]
    height = st.number_input("Height (cm)", value=160)
    weight = st.number_input("Weight (kg)", value=60.0)
    ap_hi = st.number_input("Systolic Blood Pressure", value=120)

with col2:
    ap_lo = st.number_input("Diastolic Blood Pressure", value=80)
    chol = st.selectbox("Cholesterol Level", options=[(1, "Normal"), (2, "Above Normal"), (3, "High")], format_func=lambda x: x[1])[0]
    gluc = st.selectbox("Glucose Level", options=[(1, "Normal"), (2, "Above Normal"), (3, "High")], format_func=lambda x: x[1])[0]
    smoke = st.checkbox("Smoker?")
    alco = st.checkbox("Consumes Alcohol?")
    active = st.checkbox("Physically Active?")


if st.button("Predict Results"):
    # Convert inputs to the format the model expects
    features = np.array([[age*365, gender, height, weight, ap_hi, ap_lo, chol, gluc, int(smoke), int(alco), int(active)]])
    scaled_features = scaler.transform(features)
    
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)[0][1]

    if prediction[0] == 1:
        st.error(f"High Risk Detected! (Probability: {probability:.2%})")
        st.write("Recommendation: Please consult a doctor for a thorough check-up.")
    else:
        st.success(f"Low Risk (Probability: {probability:.2%})")
        st.write("Recommendation: Maintain a healthy lifestyle and regular exercise.")