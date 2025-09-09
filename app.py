from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('pipeline.joblib')

st.title("🩺 Early Diabetes Risk Prediction")

# Input fields
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
age_group = st.selectbox("Age Group", options=[1, 2, 3, 4, 5])
phys_activity = st.selectbox("Physically Active?", options=[0, 1])

# Feature engineering
bmi_physact = bmi * phys_activity
agegroup_sq = age_group ** 2

# Predict
if st.button("Predict"):
    input_data = np.array([[bmi, age_group, bmi_physact, agegroup_sq]])
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
