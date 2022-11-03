import streamlit as st
import pandas as pd
import joblib

# Pipelines
def pre_processing(data):
    X_variables = ['age',  'hours_per_week', 'education_num']
    return data[X_variables]

def post_processing(prediction):
    if len(prediction)==1:
        return prediction[:, 1][0]
    else:
        return prediction[:, 1]

def app_prediction_function(input_data, model):
    return post_processing(model.predict_proba(pre_processing(input_data)))
    
# Streamlit Web Interface    
st.header("Income level Prediction Web App")

# Inputs
age = st.number_input("Enter Age")
education_num = st.number_input("Enter Education Number")
hours_per_week = st.number_input("Enter Hors Per Week")

# Action button to initiate prediction 
if st.button("Predict"):
    
    # Load model
    model_file = 'model_rf_test.joblib'
    model = joblib.load(model_file)
    print(model)
    
    # Feature Dataset (row)
    input_data = pd.DataFrame([{'age':age, 'education_num':education_num, 'hours_per_week':hours_per_week}])
    
    # Predict
    prediction = app_prediction_function(input_data, model)
    
    # Output prediction
    st.text(f"Predicted Porbability: {prediction}")