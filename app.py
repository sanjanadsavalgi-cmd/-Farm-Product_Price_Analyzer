import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------------------
# Load Model & Columns
# ----------------------------
model = joblib.load("farm_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🌾 Farm Product Price Predictor")

# ----------------------------
# Dropdown Inputs (Improved UI)
# ----------------------------

crop = st.selectbox("Select Crop", ["Tomato", "Potato", "Onion", "Rice", "Wheat"])
season = st.selectbox("Select Season", ["Summer", "Winter", "Monsoon"])
state = st.selectbox("Select State", ["Karnataka", "Maharashtra", "Punjab"])

temperature = st.slider("Temperature (°C)", 10, 45, 25)
rainfall = st.slider("Rainfall (mm)", 0, 500, 100)

# ----------------------------
# Predict Button
# ----------------------------

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "Crop": [crop],
        "Temperature": [temperature],
        "Rainfall": [rainfall],
        "Season": [season],
        "State": [state]
    })

    # Convert categorical variables
    input_data = pd.get_dummies(input_data)

    # Match training columns
    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0

    input_data = input_data[model_columns]

    # Prediction
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Price: ₹ {prediction:.2f}")

    # ----------------------------
    # Dynamic Graph 📊
    # ----------------------------

    st.subheader("📊 Price Analysis")

    # Example comparison data (you can replace with real dataset later)
    labels = ["Low", "Average", "High", "Predicted"]
    values = [prediction * 0.7, prediction * 0.9, prediction * 1.2, prediction]

    fig, ax = plt.subplots()
    ax.bar(labels, values)

    ax.set_ylabel("Price (₹)")
    ax.set_title(f"{crop} Price Comparison")

    st.pyplot(fig)
