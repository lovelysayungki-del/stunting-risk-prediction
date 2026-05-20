import streamlit as st
import pandas as pd
import joblib

# LOAD MODEL
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Stunting Risk Prediction")
st.write("Prediction Based on Socioeconomic Factors")

st.subheader("Input Data")

# INPUT USER
stunting = st.number_input(
    "Stunting Prevalence (%)",
    0.0,100.0)

poverty = st.number_input(
    "Poverty Rate (%)",
    0.0,100.0)

fertility = st.number_input(
    "Fertility Rate",
    0.0,10.0)

female_lit = st.number_input(
    "Female Literacy Rate (%)",
    0.0,100.0)

gdp = st.number_input(
    "GDP per Capita",
    0.0)

urban = st.number_input(
    "Urban Population Percent (%)",
    0.0,100.0)

# BUTTON PREDICT
if st.button("Predict"):

    data = pd.DataFrame([[
        stunting,
        poverty,
        fertility,
        female_lit,
        gdp,
        urban
    ]],
    columns=[
        'Stunting_Prevalence',
        'Poverty_Rate',
        'Fertility_Rate',
        'Female_Literacy_Rate',
        'GDP_per_Capita',
        'Urban_Population_Percent'
    ])

    # NORMALISASI
    scaled = scaler.transform(data)

    # PREDIKSI
    pred = model.predict(scaled)[0]

    # OUTPUT KELAS
    if pred == 0:
        result = "LOW RISK"
    elif pred == 1:
        result = "MEDIUM RISK"
    else:
        result = "HIGH RISK"

    st.success(
        f"Predicted Stunting Risk: {result}"
    )