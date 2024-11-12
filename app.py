# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="🏠 House Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('model/house_price_model.joblib')
    return model

model = load_model()

# Define the input fields based on the features used in the model
def user_input_features():
    st.sidebar.header('User Input Parameters')
    
    area = st.sidebar.number_input('Area (sq ft)', min_value=0, max_value=10000, value=1500)
    bedrooms = st.sidebar.number_input('Number of Bedrooms', min_value=0, max_value=20, value=3)
    bathrooms = st.sidebar.number_input('Number of Bathrooms', min_value=0, max_value=10, value=2)
    stories = st.sidebar.number_input('Number of Stories', min_value=0, max_value=10, value=2)

    # Binary categorical inputs
    mainroad = st.sidebar.selectbox('Main Road Access', ['Yes', 'No'])
    guestroom = st.sidebar.selectbox('Guest Room', ['Yes', 'No'])
    basement = st.sidebar.selectbox('Basement', ['Yes', 'No'])
    hotwaterheating = st.sidebar.selectbox('Hot Water Heating', ['Yes', 'No'])
    airconditioning = st.sidebar.selectbox('Air Conditioning', ['Yes', 'No'])
    parking = st.sidebar.number_input('Parking Spaces', min_value=0, max_value=10, value=1)
    prefarea = st.sidebar.selectbox('Preferred Area', ['Yes', 'No'])
    furnishingstatus = st.sidebar.selectbox('Furnishing Status', ['Furnished', 'Semi-Furnished', 'Unfurnished'])

    # Create a dictionary for input data
    input_dict = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'parking': parking,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus,
    }
    
    features = pd.DataFrame(input_dict, index=[0])
    return features

# Main function to run the app
def main():
    st.title("🏠 House Price Prediction App")
    st.write(""" 
    ### Predict the Sale Price of a House 
    Enter the house details in the sidebar, and the app will predict the sale price based on the trained model.
    """)
    
    # Get user input
    input_df = user_input_features()
    
    st.subheader('User Input parameters')
    st.write(input_df)
    
    # Predict
    if st.button('Predict'):
        try:
            # Convert binary categorical inputs to numeric
            input_df['mainroad'] = input_df['mainroad'].map({'Yes': 1, 'No': 0}).astype(int)
            input_df['guestroom'] = input_df['guestroom'].map({'Yes': 1, 'No': 0}).astype(int)
            input_df['basement'] = input_df['basement'].map({'Yes': 1, 'No': 0}).astype(int)
            input_df['hotwaterheating'] = input_df['hotwaterheating'].map({'Yes': 1, 'No': 0}).astype(int)
            input_df['airconditioning'] = input_df['airconditioning'].map({'Yes': 1, 'No': 0}).astype(int)
            input_df['prefarea'] = input_df['prefarea'].map({'Yes': 1, 'No': 0}).astype(int)

            # Ensure parking is treated as an integer
            input_df['parking'] = input_df['parking'].astype(int)

            # Make prediction
            prediction = model.predict(input_df)[0]
            st.subheader('Predicted Sale Price')
            st.write(f"₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
