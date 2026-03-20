import streamlit as st
import pandas as pd
import joblib
import os

# ===============================
# LOAD MODEL
# ===============================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "final_model.pkl"))
X_columns = joblib.load(os.path.join(BASE_DIR, "X_columns.pkl"))

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Lagos House Price Predictor", layout="centered")

st.title("🏠 Lagos House Price Predictor")
st.write("Predict house prices across Lagos using machine learning")

# ===============================
# USER INPUT
# ===============================
st.sidebar.header("Enter Property Details")

bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1, 10, 2)

has_pool = st.sidebar.selectbox("Has Pool?", ["No", "Yes"])
has_bq = st.sidebar.selectbox("Has BQ?", ["No", "Yes"])
new_build = st.sidebar.selectbox("New Build?", ["No", "Yes"])
in_estate = st.sidebar.selectbox("In Estate?", ["No", "Yes"])

# Extract locations dynamically
locations = [col.replace("Location_", "") for col in X_columns if col.startswith("Location_")]
location = st.sidebar.selectbox("Location", locations)

# ===============================
# FEATURE ENGINEERING (MATCH TRAINING)
# ===============================
location_rank = {
    "Banana Island": 10,
    "Ikoyi": 9,
    "Victoria Island": 7,
    "Apapa": 5,
    "Ikeja": 5,
    "Lekki": 4,
    "Yaba": 3,
    "Ajah": 2,
    "Ikorodu": 1
}

loc_rank = location_rank.get(location, 3)

# ===============================
# BUILD INPUT DATA
# ===============================
input_dict = {
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Has_Pool": 1 if has_pool == "Yes" else 0,
    "Has_BQ": 1 if has_bq == "Yes" else 0,
    "New_Build": 1 if new_build == "Yes" else 0,
    "In_Estate": 1 if in_estate == "Yes" else 0,

    # FINAL FEATURES
    "Location_Rank": loc_rank,
    "Final_Power": bedrooms * (loc_rank ** 3),
    "Bed_Bath_Ratio": bathrooms / (bedrooms + 1)
}

# Add one-hot encoding for location
for loc in locations:
    input_dict[f"Location_{loc}"] = 1 if loc == location else 0

# Fill missing columns
for col in X_columns:
    if col not in input_dict:
        input_dict[col] = 0

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])[X_columns]

# ===============================
# PREDICTION
# ===============================
def predict_price(model, data):
    return model.predict(data)[0]

# ===============================
# BUTTON
# ===============================
if st.button("Predict Price"):
    price = predict_price(model, input_df)

    st.success(f"💰 Estimated Price: ₦{price:,.0f}")
    st.info("⚠️ Prices are estimates based on historical Lagos housing data.")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Built by Olaide | Machine Learning Project")
    





    

       
