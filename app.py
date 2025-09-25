# app.py
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("AIModel_For_House.pkl", "rb") as f:
    model = pickle.load(f)

# Load the fitted scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Page config
st.set_page_config(
    page_title="House Price Prediction Model",
    page_icon="🏠",
    layout="wide"
)
# st.image("https://www.google.com/imgres?q=House%20Price%20Prediction%20Model&imgurl=https%3A%2F%2Fmiro.medium.com%2Fv2%2Fresize%3Afit%3A1400%2F1*w0csxxx277bzLKNBDdkQcw.jpeg&imgrefurl=https%3A%2F%2Fmedium.com%2F%40khananns24%2Fmaking-a-house-price-predicting-model-b44ee812e60c&docid=UqwN0R8lWx4aRM&tbnid=5AwK1pl2Q2Q1yM&vet=12ahUKEwjzirrcwPSPAxW-X_EDHX89EBYQM3oECBYQAA..i&w=1400&h=1050&hcb=2&itg=1&ved=2ahUKEwjzirrcwPSPAxW-X_EDHX89EBYQM3oECBYQAA", use_container_width=True)

# Title and description
st.title("🏠 California House Price Prediction App")
st.write("""
Predict house prices based on California housing dataset features.
This app is built by **Mirza Yasir Abdullah Baig**.
""")

# Sidebar: Input Features
st.sidebar.header("Input Features")

def user_input_features():
    MedInc = st.sidebar.number_input("Median Income (MedInc)", min_value=0.0, value=3.0)
    HouseAge = st.sidebar.number_input("House Age (HouseAge)", min_value=0.0, value=30.0)
    AveRooms = st.sidebar.number_input("Average Rooms (AveRooms)", min_value=0.0, value=5.0)
    AveBedrms = st.sidebar.number_input("Average Bedrooms (AveBedrms)", min_value=0.0, value=1.0)
    Population = st.sidebar.number_input("Population", min_value=0.0, value=1000.0)
    AveOccup = st.sidebar.number_input("Average Occupancy (AveOccup)", min_value=0.0, value=3.0)
    Latitude = st.sidebar.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0)
    
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Sidebar: About Me
st.sidebar.header("About Me")
st.sidebar.write("""
**Mirza Yasir Abdullah Baig**  
- [LinkedIn](https://www.linkedin.com/in/mirza-yasir-abdullah-baig/)  
- [Kaggle](https://www.kaggle.com/code/mirzayasirabdullah07)  
- [GitHub](https://github.com/mirzayasirabdullahbaig07)  
""")
st.sidebar.write("This project predicts California house prices using a Linear Regression model trained on the California housing dataset.")

# Prediction button
if st.button("Predict House Price"):
    # Transform input features using the saved scaler
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    st.subheader("Prediction")
    st.write(f"Predicted Median House Value: **${prediction[0]*100000:.2f}**")

# # Show input features
# st.subheader("Input Features")
# st.write(input_df)
