# churn-prediction-app
Churn Insight and Prediction Engine  A Streamlit app for predicting customer churn using a Random Forest model. Enter customer data to get churn predictions. Validated with cross-validation and SHAP for feature insights. Available on Streamlit Community Cloud.
Churn Insight and Prediction Engine

This project is a machine learning-based application designed to predict customer churn for e-commerce platforms. The solution identifies at-risk customers based on behavioral, demographic, and transactional data, helping businesses implement targeted retention strategies.
Project Overview

The Churn Insight and Prediction Engine uses a Random Forest classifier to predict whether a customer is likely to churn. The model was trained and tuned using customer behavior data, achieving strong performance with high precision and recall metrics. Key features like tenure, app usage hours, satisfaction scores, and transaction history contribute to the prediction.
Features

    Data Processing: Cleaned and preprocessed data with imputation and standardization.
    Exploratory Data Analysis (EDA): Visualized key patterns and correlations.
    Model Training: Built and optimized using Random Forest with hyperparameter tuning.
    Model Interpretation: Leveraged SHAP values for interpretability.
    Deployment: Streamlit web application for real-time predictions.

How to Run

    Clone the repository:

    bash

git clone https://github.com/yourusername/churn-insight-prediction-engine.git

Install the required packages:

bash

pip install -r requirements.txt

Run the Streamlit app:

bash

    streamlit run app.py

Project Structure

    app.py: The main Streamlit app script.
    model/: Contains the trained model (.pkl) and scaler.
    notebooks/: Jupyter notebooks used during EDA and model development.
    data/: Sample dataset for testing the application.

License

This project is licensed under the MIT License - see the LICENSE file for details.
