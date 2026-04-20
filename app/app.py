import streamlit as st
import pandas as pd
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import preprocess_data

# Load model
model = joblib.load("models/model.pkl")

st.title("🌱 Irrigation Need Prediction")

st.write("Enter the details below:")

# --- User Inputs ---
temperature = st.number_input("Temperature", min_value=0.0)
humidity = st.number_input("Humidity", min_value=0.0)
soil_moisture = st.number_input("Soil Moisture", min_value=0.0)

soil_type = st.selectbox("Soil Type", ["Sandy", "Clay", "Loamy", "Silty"])
crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Maize", "Cotton", "Barley", "Soybean"])
season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
irrigation_type = st.selectbox("Irrigation Type", ["Drip", "Sprinkler", "Flood", "Manual"])
water_source = st.selectbox("Water Source", ["Canal", "Well", "River", "Tank"])
region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
mulching = st.selectbox("Mulching Used", ["Yes", "No"])

# --- Predict ---
if st.button("Predict"):

    input_df = pd.DataFrame([{
        "Temperature": temperature,
        "Humidity": humidity,
        "Soil_Moisture": soil_moisture,
        "Soil_Type": soil_type,
        "Crop_Type": crop_type,
        "Season": season,
        "Irrigation_Type": irrigation_type,
        "Water_Source": water_source,
        "Region": region,
        "Mulching_Used": mulching
    }])

    # Preprocess
    input_processed = preprocess_data(input_df)

    # Align columns (IMPORTANT)
    model_columns = model.feature_name_
    input_processed = input_processed.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_processed)

    st.success(f"💧 Irrigation Need: {prediction[0]}")