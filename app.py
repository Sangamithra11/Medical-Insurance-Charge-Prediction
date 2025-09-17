import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pickle

st.title("Insurance Charge Prediction App")


with open("Medical_Insurance_cost_prediction.pkl", "rb") as f:
    model = pickle.load(f)
model_features = ['age', 'bmi', 'children',
                  'sex_male',
                  'smoker_yes',
                  'region_northwest','region_southeast','region_southwest']

st.title("Insurance Charge Prediction App")

age = st.number_input("Age", min_value=0, max_value=120, value=19)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=0.0, value=18.5)
children = st.number_input("Number of Children", min_value=0, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict"):

    input_df = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })


    input_df = pd.get_dummies(input_df,columns=['sex','smoker','region'],drop_first=True)


    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_features]

    st.write("Final input for model:", input_df)

    prediction = model.predict(input_df)
    st.success(f"Predicted Insurance Charges:{prediction[0]:.2f}")
