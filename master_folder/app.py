import streamlit as st
import pandas as pd
import joblib
import os

st.title("Tourism Package Purchase Prediction")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model not found. Please ensure 'models/best_model.joblib' is pushed to the repository. Error: {e}")
    st.stop()

# Application Inputs
st.header("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    durationofpitch = st.number_input("Duration of Pitch", min_value=1, max_value=100, value=15)
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    numberoffollowups = st.number_input("Number of Followups", min_value=1, max_value=10, value=3)
    productpitched = st.selectbox("Product Pitched", ["Basic", "Wellness", "Standard", "Deluxe", "Super Deluxe"])

with col2:
    preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5])
    maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    numberoftrips = st.number_input("Number of Trips", min_value=1, max_value=20, value=2)
    passport = st.selectbox("Passport", [0, 1])
    pitchsatisfactionscore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    owncar = st.selectbox("Own Car", [0, 1])
    numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthlyincome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)

if st.button("Predict Purchase Likelihood"):
    data = {
        "Age": [age],
        "TypeofContact": [typeofcontact],
        "CityTier": [citytier],
        "DurationOfPitch": [durationofpitch],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [numberofpersonvisiting],
        "NumberOfFollowups": [numberoffollowups],
        "ProductPitched": [productpitched],
        "PreferredPropertyStar": [preferredpropertystar],
        "MaritalStatus": [maritalstatus],
        "NumberOfTrips": [numberoftrips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitchsatisfactionscore],
        "OwnCar": [owncar],
        "NumberOfChildrenVisiting": [numberofchildrenvisiting],
        "Designation": [designation],
        "MonthlyIncome": [monthlyincome]
    }
    
    # Save inputs into a dataframe
    input_df = pd.DataFrame(data)
    
    # Predict
    prediction = model.predict(input_df)
    
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("🎉 The customer is **LIKELY** to purchase the Wellness Tourism Package!")
    else:
        st.warning("⚠️ The customer is **UNLIKELY** to purchase the package.")
