import streamlit as st
import pickle
import numpy as np
import os

# Load Random Forest Model
rf_path = os.path.join(os.path.dirname(__file__), '..', 'random_forest', 'random_forest_model.pkl')
with open(rf_path, 'rb') as f:
    rf_model = pickle.load(f)

# Load XGBoost Model
xgb_path = os.path.join(os.path.dirname(__file__), '..', 'XGBModel', 'xgb_model.pkl')
with open(xgb_path, 'rb') as f:
    xgb_model = pickle.load(f)

# Load encoders
encoder_path = os.path.join(os.path.dirname(__file__), 'encoders.pkl')
with open(encoder_path, 'rb') as f:
    encoders = pickle.load(f)

# Category options
sponsor_options = encoders['sponsor_type'].keys()
gender_options = encoders['gender'].keys()
condition_options = encoders['condition'].keys()
location_options = encoders['location'].keys()
phase_options = encoders['phase'].keys()
masking_options = encoders['masking'].keys()
study_design_options = encoders['study_design'].keys()
intervention_type_options = encoders['intervention_type'].keys()

st.title("Clinical Trial Outcome Prediction App")

st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose a model", ['Random Forest', 'XGBoost'])

st.header("Enter Trial Details")

# Inputs
enrollment = st.number_input("Enrollment (in 1000s)", min_value=0.0, max_value=10000.0, value=500.0)
duration = st.number_input("Duration (in days)", min_value=0.0, max_value=5000.0, value=365.0)

phase = st.selectbox("Phase", list(phase_options))
sponsor_type = st.selectbox("Sponsor Type", list(sponsor_options))
gender = st.selectbox("Gender", list(gender_options))
condition = st.selectbox("Condition", list(condition_options))
location = st.selectbox("Location", list(location_options))
masking = st.selectbox("Masking", list(masking_options))
study_design = st.selectbox("Study Design", list(study_design_options))
intervention_type = st.selectbox("Intervention Type", list(intervention_type_options))

if st.button("Predict Outcome"):
    try:
        # Manual Encoding
        phase_encoded = encoders['phase'][phase]
        sponsor_encoded = encoders['sponsor_type'][sponsor_type]
        gender_encoded = encoders['gender'][gender]
        condition_encoded = encoders['condition'][condition]
        location_encoded = encoders['location'][location]
        masking_encoded = encoders['masking'][masking]
        study_design_encoded = encoders['study_design'][study_design]
        intervention_type_encoded = encoders['intervention_type'][intervention_type]

        # Prepare input
        features = np.array([
            enrollment,
            duration,
            phase_encoded,
            condition_encoded,
            intervention_type_encoded,
            study_design_encoded,
            sponsor_encoded,
            gender_encoded,
            location_encoded,
            masking_encoded
        ]).reshape(1, -1)

        # Model prediction
        model = rf_model if model_choice == 'Random Forest' else xgb_model

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        st.success(f"Prediction: {'Success' if pred == 1 else 'Failure'}")
        st.info(f"Probability of Success: {prob:.4f}")

    except Exception as e:
        st.error(f"Error in prediction: {e}")
