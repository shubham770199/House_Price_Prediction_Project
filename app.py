import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/final_model.pkl")

# App title
st.title("🏠 House Price Prediction App")
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 0 8px rgba(0,0,0,0.05);
    }
    .st-bx {
        font-size: 16px;
        color: #333333;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.write("Enter the details below to predict the house price:")

# Input form
area = st.number_input("Area (sqft)", min_value=1000, max_value=20000, value=3000)
bedrooms = st.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5, 6])
bathrooms = st.selectbox("Number of Bathrooms", [1, 2, 3, 4])
stories = st.selectbox("Number of Stories", [1, 2, 3, 4])
mainroad = st.radio("Main Road Access", ["yes", "no"])
guestroom = st.radio("Guest Room", ["yes", "no"])
basement = st.radio("Basement", ["yes", "no"])
hotwaterheating = st.radio("Hot Water Heating", ["yes", "no"])
airconditioning = st.radio("Air Conditioning", ["yes", "no"])
parking = st.selectbox("Parking Spaces", [0, 1, 2, 3])
prefarea = st.radio("Preferred Area", ["yes", "no"])
furnishingstatus = st.selectbox("Furnishing Status", ["unfurnished", "semi-furnished", "furnished"])

# Convert to numerical format
def convert_yes_no(value):
    return 1 if value == "yes" else 0

input_data = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": convert_yes_no(mainroad),
    "guestroom": convert_yes_no(guestroom),
    "basement": convert_yes_no(basement),
    "hotwaterheating": convert_yes_no(hotwaterheating),
    "airconditioning": convert_yes_no(airconditioning),
    "parking": parking,
    "prefarea": convert_yes_no(prefarea),
    "furnishingstatus_semi-furnished": 1 if furnishingstatus == "semi-furnished" else 0,
    "furnishingstatus_unfurnished": 1 if furnishingstatus == "unfurnished" else 0,
}

# Predict button
if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted House Price: ₹{int(prediction):,}")
