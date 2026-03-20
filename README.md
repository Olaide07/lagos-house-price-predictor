# Lagos House Price Predictor

A machine learning web application that predicts house prices in Lagos based on property features such as location, number of bedrooms, and amenities.
---

## Live App

[Click here to use the app](https://your-app-link.streamlit.app)
## Live app coming soon
---
## Project Overview

This project uses machine learning to estimate house prices in Lagos, Nigeria. It provides an interactive interface where users can input property details and receive a predicted price instantly.
---
## Features
- Predict house prices based on:
  - Bedrooms and Bathrooms
  - Location
  - Property amenities (Pool, BQ, Estate, etc.)
  - Luxury score
- Interactive web interface using Streamlit
- Feature importance visualization
- Real-time predictions
---
## Machine Learning Approach

- **Model Used:** Random Forest Regressor  
- **Target Transformation:** Log transformation (`log1p`)  
- **Evaluation Metrics:**
  - MAE: ₦236M  
  - RMSE: ₦523M  
---
## Tech Stack
- Python  
- Pandas & NumPy  
- Scikit-learn  
- Streamlit  
- Matplotlib  
- Joblib  
---
## Project Structure
Lagos-House-Price-Predictor/
│
├── lagos_app.py # Streamlit application
├── final_model.pkl # Trained machine learning model
├── X_columns.pkl # Feature columns used for prediction
├── requirements.txt # Project dependencies
└── README.md # Project documentation
