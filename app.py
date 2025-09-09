import streamlit as st
import pandas as pd
import joblib

# ✅ Import all classes used in the pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ✅ Load the model safely
try:
    model = joblib.load('pipeline.joblib')
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ✅ Streamlit UI
st.title("🩺 Early Diabetes Risk Prediction")
st.write("Upload patient data to predict diabetes risk.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.write("### Input Data Preview", input_df.head())

        # ✅ Make predictions
        predictions = model.predict(input_df)
        prediction_probs = model.predict_proba(input_df)[:, 1]

        # ✅ Display results
        results_df = input_df.copy()
        results_df["Diabetes Risk"] = predictions
        results_df["Risk Probability"] = prediction_probs

        st.write("### Prediction Results", results_df)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
