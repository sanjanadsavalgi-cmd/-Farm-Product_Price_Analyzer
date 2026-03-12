import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("farm_price_model.pkl")

st.title("🌾 Farm Product Price Predictor")

# ----------------------------
# User Inputs
# ----------------------------

crop = st.text_input("Crop")
temperature = st.number_input("Temperature (°C)", value=25.0)
rainfall = st.number_input("Rainfall (mm)", value=100.0)
season = st.text_input("Season")
state = st.text_input("State")

# ----------------------------
# Predict Button
# ----------------------------

if st.button("Predict Price"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Crop": [crop],
        "Temperature": [temperature],
        "Rainfall": [rainfall],
        "Season": [season],
        "State": [state]
    })

    # Convert categorical variables
    input_data = pd.get_dummies(input_data)

    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Price: ₹ {prediction:.2f}")
    except:
        st.error("Input format does not match the trained model.")
