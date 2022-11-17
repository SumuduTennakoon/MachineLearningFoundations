import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Streamlit: https://streamlit.io/
# streamlit run web_app.py

# Pipelines
def pre_process(data):   

    input_columns = ['age', 'hours_per_week', 'workclass', 'education', 'marital_status', 'occupation', 'sex']
    data = data[input_columns]
    
    ##########
    # Cleaning
    ##########
    # Create Unique ID for Each Row
    data['ID'] = data.index+1

    # Remove leading and trailing spaces in string values
    def remove_spaces(data, columns):
        for column in columns:
            data[column] = data[column].str.strip()
        return data

    string_columns = ['workclass', 'education', 'marital_status', 'occupation', 'sex']
    data = remove_spaces(data, string_columns)

    # Drop unwanted column
    data.drop(labels='Unnamed: 0', axis=1, inplace=True, errors='ignore')

    # Drop rows with missing values
    data.dropna(how='any', axis=0, inplace=True)

    #####################################
    # Numeric to Caregorical (Binning)
    #####################################

    # age
    labels = ['<20', '20-30', '30-40', '40-50', '50-60', '>60']
    bin_edges = [0, 20, 30, 40, 50, 60, np.inf]
    data['age_group'] = pd.cut(x=data['age'], bins=bin_edges, labels=labels)

    # hours_per_week
    labels = ['<25', '20-35', '35-45', '45-60', '>60']
    bin_edges = [0, 20, 35, 45, 60, np.inf]
    data['hours_per_week_group'] = pd.cut(x=data['hours_per_week'], bins=bin_edges, labels=labels)

    #####################################
    # Re-group Catergorical Column Values
    #####################################

    # education
    data['education_group'] = data['education'].replace({
        'Preschool':'school', '1st-4th':'school', '5th-6th':'school', '7th-8th':'school', '9th':'school',
        '10th':'h_school', '11th':'h_school', '12th':'h_school', 'HS-grad':'h_school',
        'Some-college':'university_eq', 'Assoc-acdm':'university_eq', 'Assoc-voc':'university_eq', 
        'Bachelors':'university', 'Masters':'university', 'Doctorate':'university', 'Prof-school':'university'
    })

    #sex
    data['is_male'] = np.where(data['sex']=='Male', 1,0)

    # workclass
    data['workclass_group'] = data['workclass'].replace({'?':'Other', 'Without-pay':'Other', 'Never-worked':'Other', 'Local-gov':'Local-State-gov', 'State-gov':'Local-State-gov'})

    # marital_status
    data['marital_status_group'] = data['marital_status'].replace({'Divorced':'Divorced-Separated-Widowed-Absent', 'Separated':'Divorced-Separated-Widowed-Absent', 'Widowed':'Divorced-Separated-Widowed-Absent', 
        'Married-spouse-absent':'Divorced-Separated-Widowed-Absent',
        'Married-civ-spouse':'Married-civ-AF-spouse', 'Married-AF-spouse':'Married-civ-AF-spouse'
    })

    # occupation
    data['occupation_group'] = data['occupation'].replace({'Prof-specialty':'Exec-managerial-Prof-specialty', 'Exec-managerial':'Exec-managerial-Prof-specialty', 
        'Protective-serv':'Armed-Forces-Protective-serv', 'Armed-Forces':'Armed-Forces-Protective-serv',
        'Priv-house-serv':'Priv-house-serv-Handlers-cleaners-Other', 'Handlers-cleaners':'Priv-house-serv-Handlers-cleaners-Other', 
        'Other-service':'Priv-house-serv-Handlers-cleaners-Other', '?':'Priv-house-serv-Handlers-cleaners-Other',
        'Farming-fishing':'Farming-fishing-Machine-op-inspct', 'Machine-op-inspct':'Farming-fishing-Machine-op-inspct',
    })

    ################
    # One-hot Encode
    ################

    # Get Dummies
    categorical_columns = ['workclass_group', 'marital_status_group', 'occupation_group', 'is_male', 'age_group', 'hours_per_week_group', 'education_group']
    dummy_columns_df = pd.get_dummies(data[categorical_columns], drop_first=False)

    # Merge Dummy Values with the Data
    data = pd.concat([data, dummy_columns_df], axis=1)

    ################
    # Select Columns
    ################
    X_variables = ['is_male',
                    'workclass_group_Federal-gov',
                    'workclass_group_Local-State-gov',
                    'workclass_group_Private',
                    'workclass_group_Self-emp-inc',
                    'workclass_group_Self-emp-not-inc',
                    'marital_status_group_Married-civ-AF-spouse',
                    'marital_status_group_Never-married',
                    'occupation_group_Craft-repair',
                    'occupation_group_Farming-fishing-Machine-op-inspct',
                    'occupation_group_Priv-house-serv-Handlers-cleaners-Other',
                    'occupation_group_Exec-managerial-Prof-specialty',
                    'occupation_group_Armed-Forces-Protective-serv',
                    'occupation_group_Sales',
                    'occupation_group_Tech-support',
                    'occupation_group_Transport-moving',
                    'age_group_20-30',
                    'age_group_30-40',
                    'age_group_40-50',
                    'age_group_50-60',
                    'age_group_>60',
                    'hours_per_week_group_20-35',
                    'hours_per_week_group_35-45',
                    'hours_per_week_group_45-60',
                    'hours_per_week_group_>60',
                    'education_group_university',
                    'education_group_school',
                    'education_group_university_eq'
    ]
    
    #############################
    # Assign 0 to missing columns
    #############################
    for x in list(set(X_variables) - set(data.columns)):
        data[x] = 0
        
    return data[X_variables]

def score(input_data, model):
    return model.predict_proba(input_data)

def post_process(prediction):
    output = []
    for i in range(len(prediction)):
        if prediction[i][1]>prediction[i][0]:
            output.append(F">50K ({prediction[i][1]:.2f})")
        else:
            output.append(F"<=50K ({prediction[i][0]:.2f})")
    
    if len(output)==1:
        return output[0]
    else:
        return output

def app_prediction_function(input_data, model):
    return post_process(model.predict_proba(pre_process(input_data)))
    
# Streamlit Web Interface    
st.header("Income level Prediction Web App")

# Inputs
sex = st.selectbox(
    'Select Gender',
    [' Male', ' Female']
)

age = st.number_input("Enter Age")

hours_per_week = st.number_input("Enter Hours Per Week")

workclass = st.selectbox(
    'Select Work Class',
    ['State-gov', ' Self-emp-not-inc', 'Private', 'Federal-gov',
       'Local-gov', ' ?', 'Self-emp-inc', 'Without-pay','Never-worked']
)

education = st.selectbox(
    'Select Heighest Education Level',
    ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th',
       'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th']
)


marital_status = st.selectbox(
    'Select Marital_Status',
    ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 
    'Separated', 'Married-AF-spouse', 'Widowed']
)

occupation = st.selectbox(
    'Select Cccupation',
    ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 
    'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
    'Tech-support', ' ?', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv']
)

# Action button to initiate prediction 
if st.button("Predict"):
    
    # Load model
    model_file = 'model_rf_test.joblib'
    model = joblib.load(model_file)
    print(model)
    
    # Feature Dataset (row)
    input_data = pd.DataFrame([{
                                "age": age,
                                "workclass": workclass,
                                "education": education,
                                "marital_status": marital_status,
                                "occupation": occupation,
                                "sex": sex,
                                "hours_per_week": hours_per_week }]
    )

    print(input_data)
    
    # Predict
    prediction = app_prediction_function(input_data, model)
    
    # Output prediction
    st.text(f"Predicted Porbability: {prediction}")