import streamlit as st
import pandas as pd
import joblib

# LOAD MODEL
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Stunting Risk Prediction")
st.write("Prediction Based on Socioeconomic Factors")

st.subheader("Input Data")

stunting = st.number_input("Stunting Prevalence (%)", 0.0, 100.0)
overweight = st.number_input("Overweight Prevalence (%)", 0.0, 100.0)
gdp = st.number_input("GDP per Capita", 0.0)
poverty = st.number_input("Poverty Rate (%)", 0.0, 100.0)
male_lit = st.number_input("Male Literacy Rate (%)", 0.0, 100.0)
female_lit = st.number_input("Female Literacy Rate (%)", 0.0, 100.0)
fertility = st.number_input("Fertility Rate", 0.0, 10.0)
urban = st.number_input("Urban Population Percent (%)", 0.0, 100.0)

if st.button("Predict"):

    # Column order must match numeric_cols used during scaler.fit_transform
    data = pd.DataFrame([[
        stunting,
        overweight,
        gdp,
        poverty,
        male_lit,
        female_lit,
        fertility,
        urban
    ]],
    columns=[
        'Stunting_Prevalence',
        'Overweight_Prevalence',
        'GDP_per_Capita',
        'Poverty_Rate',
        'Male_Literacy_Rate',
        'Female_Literacy_Rate',
        'Fertility_Rate',
        'Urban_Population_Percent'
    ])

    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]

    if pred == 0:
        result = "LOW RISK"
    elif pred == 1:
        result = "MEDIUM RISK"
    else:
        result = "HIGH RISK"

    st.success(f"Predicted Stunting Risk: {result}")
