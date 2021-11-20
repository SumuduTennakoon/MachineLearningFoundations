import numpy as np
import pandas as pd

# Classifier algorithms
from sklearn.ensemble import RandomForestClassifier

from flask import Flask, jsonify, request

import joblib
import json
    
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

app = Flask(__name__)

# Load model
model_file = 'model_rf_test.joblib'
model = joblib.load(model_file)
print(model)
    
@app.route("/")
def index():
    return "Greetings from Prediction API"

@app.route("/classifier", methods=['POST'])
def classifier():
    if request.method == 'POST': 
        input_data =  request.form.to_dict()
        print(input_data)
        print(type(input_data))
        input_data = pd.DataFrame([input_data])
        print(input_data)
        prediction = app_prediction_function(input_data, model)
        return jsonify({'prediction': prediction})  
        
if __name__ == '__main__':
    app.run(debug=True, port=5001)