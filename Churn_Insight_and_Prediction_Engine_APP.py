#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('random_forest_churn_model.pkl')

# Define feature names
feature_names = [
    'Tenure', 'HourSpendOnApp', 'OrderCount', 'WarehouseToHome',
    'SatisfactionScore', 'NumberOfDeviceRegistered', 'NumberOfAddress',
    'OrderAmountHikeFromlastYear', 'CouponUsed', 'DaySinceLastOrder',
    'Interaction', 'AvgHoursPerOrder'
]

# Initialize StandardScaler (should match the scaler used during training)
scaler = StandardScaler()

# Streamlit UI
st.title("Customer Churn Prediction")

# User inputs
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert inputs to DataFrame
input_df = pd.DataFrame([inputs], columns=feature_names)

# Standardize the inputs
input_scaled = scaler.fit_transform(input_df)  # Replace with the scaler used during training

# Make prediction
prediction = model.predict(input_scaled)

# Display result
if prediction[0] == 1:
    st.write("The model predicts that the customer will churn.")
else:
    st.write("The model predicts that the customer will not churn.")


# In[ ]:




