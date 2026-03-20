import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Lagos House Price Predictor", layout="centered")

# ===============================
# LOAD MODEL & COLUMNS
# ===============================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "final_model.pkl"))
X_columns = joblib.load(os.path.join(BASE_DIR, "X_columns.pkl"))

# ===============================
# TITLE & DESCRIPTION
# ===============================
st.title("🏠 Lagos House Price Predictor")
st.write(
    "This app predicts house prices in Lagos using a machine learning model trained on real estate data."
)

# ===============================
# MODEL PERFORMANCE
# ===============================
st.markdown("### 📊 Model Performance")
st.write("MAE: ₦236M | RMSE: ₦523M")

# ===============================
# USER INPUTS
# ===============================
st.sidebar.header("Input Property Details")

bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1, 10, 2)

has_pool = st.sidebar.selectbox("Has Pool?", ["No", "Yes"])
has_bq = st.sidebar.selectbox("Has BQ?", ["No", "Yes"])
new_build = st.sidebar.selectbox("New Build?", ["No", "Yes"])
in_estate = st.sidebar.selectbox("In Estate?", ["No", "Yes"])

luxury_score = st.sidebar.slider("Luxury Score", 1, 100, 50)

# Get locations dynamically
locations = [col.replace("Location_", "") for col in X_columns if col.startswith("Location_")]
location = st.sidebar.selectbox("Location", locations)

# ===============================
# CREATE INPUT DATA
# ===============================
input_dict = {
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Has_Pool": 1 if has_pool == "Yes" else 0,
    "Has_BQ": 1 if has_bq == "Yes" else 0,
    "New_Build": 1 if new_build == "Yes" else 0,
    "In_Estate": 1 if in_estate == "Yes" else 0,
    "Luxury_Score": luxury_score,
}

# Add location columns
for loc in locations:
    input_dict[f"Location_{loc}"] = 1 if loc == location else 0

# Fill missing columns
for col in X_columns:
    if col not in input_dict:
        input_dict[col] = 0

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])[X_columns]

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_price(model, data):
    pred_log = model.predict(data)
    return np.expm1(pred_log)[0]

# ===============================
# PREDICT BUTTON
# ===============================
if st.button("Predict Price"):
    price = predict_price(model, input_df)

    st.success(f"💰 Estimated House Price: ₦{price:,.0f}")
    st.info("⚠️ This is an estimate based on historical data.")

# ===============================
# FEATURE IMPORTANCE
# ===============================
if st.checkbox("Show Feature Importance"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": X_columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        fig, ax = plt.subplots()
        ax.barh(feat_df["Feature"], feat_df["Importance"])
        ax.invert_yaxis()
        ax.set_title("Top 10 Important Features")

        st.pyplot(fig)
    else:
        st.warning("Feature importance not available for this model.")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Built by Olaide | Machine Learning Project")