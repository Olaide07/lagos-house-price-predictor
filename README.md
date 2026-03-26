# Lagos House Price Predictor

A machine learning web application that predicts house prices across Lagos using property features and location-based intelligence.

This project goes beyond basic modeling by incorporating domain knowledge of Lagos real estate pricing to improve prediction realism.
---
## Live App

 https://lagos-house-price-predictor.streamlit.app/
---

## Project Overview

Predicting house prices in Lagos is complex due to:

- Large price differences across locations  
- Limited luxury property data  
- Imbalanced datasets  

This project addresses these challenges by combining machine learning with **domain-aware feature engineering**.

---

## Key Features

- Location-aware pricing using custom **Location Ranking**
- Feature engineering with interaction features (`Final_Power`)
- XGBoost regression model for better performance
- Interactive web app built with Streamlit
- Deployed on Streamlit Cloud
---

## Model Insight

Initial models (Random Forest) struggled to:

- Differentiate between high-end and mid-tier locations  
- Capture the impact of luxury areas like Banana Island  

### Solution

This was solved by:
- Creating a **Location Ranking system**  
- Engineering interaction features between location and property size  
- Switching to **XGBoost**, which handles complex relationships better  

 Result: More realistic predictions across Lagos locations

---

## Sample Predictions

| Location        | Predicted Price |
|----------------|---------------|
| Banana Island  | ₦600M – ₦1B+ |
| Ikoyi          | ₦400M – ₦700M |
| Ajah           | ₦100M – ₦300M |

---

## Tech Stack

- Python  
- Pandas & NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Joblib  

---

```

## Project Structure
lagos-house-price-predictor/
│
├── lagos_app.py # Streamlit application
├── final_model.pkl # Trained model
├── X_columns.pkl # Feature columns
├── requirements.txt # Dependencies
└── README.md # Project documentation

```


---

## Disclaimer

Prices are estimates based on historical data and may vary depending on market conditions.

---

##  Author

**Olaide Ajibade**

- Mechanical Engineer | Data Scientist  
- Passionate about AI, Machine Learning & Real Estate Analytics  

---

## Future Improvements

- Add geographic coordinates (lat/lon)
- Integrate real-time property listings
- Build API version (FastAPI)
- Add prediction confidence intervals

---
