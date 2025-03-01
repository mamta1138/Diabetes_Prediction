import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('model.pkl')

# Label Encoding mappings (use the same from training)
gender_map = {"Female": 0, "Male": 1}
smoking_map = {"never": 0, "No Info": 1, "current": 2, "former": 3, "ever": 4}

# Streamlit App UI
st.title("Diabetes Prediction App")

# User input fields
gender = st.radio("Select Gender:", list(gender_map.keys()))
age = st.number_input("Enter Age:", min_value=1, max_value=120, value=30)
hypertension = st.radio("Hypertension (0 = No, 1 = Yes):", [0, 1])
heart_disease = st.radio("Heart Disease (0 = No, 1 = Yes):", [0, 1])
smoking_history = st.selectbox("Smoking History:", list(smoking_map.keys()))
bmi = st.number_input("BMI:", min_value=10.0, max_value=50.0, value=25.0)
hba1c = st.number_input("HbA1c Level:", min_value=3.0, max_value=15.0, value=5.5)
blood_glucose = st.number_input("Blood Glucose Level:", min_value=50, max_value=300, value=120)

# Convert inputs using label encoding
gender_encoded = gender_map[gender]
smoking_encoded = smoking_map[smoking_history]

# Create DataFrame for prediction
input_data = pd.DataFrame([[gender_encoded, age, hypertension, heart_disease, smoking_encoded, bmi, hba1c, blood_glucose]],
                          columns=["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"])

# Predict when the user clicks the button
if st.button("Predict"):
    prediction = model.predict(input_data)  # Directly using raw numerical input
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.subheader(f"Prediction: **{result}**")

