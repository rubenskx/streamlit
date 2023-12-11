import streamlit as st
import numpy as np
import pickle
import scipy.stats as stats
import pandas as pd

with open('mymodel.pkl', 'rb') as f:
    model = pickle.load(f)


with open('encoders.pkl', 'rb') as l:
    encoder = pickle.load(l)


manufacturer_name = st.text_input("Enter the manufacturer name:", 'BMW')
transmission = st.text_input("Enter the transmission mode:", 'manual')
color = st.text_input("Enter the color of the car:", 'white')
odometer_value = st.text_input("Enter the odometer value:", '115000')
year_produced = st.text_input("Enter the year produced:", '2000')
engine_fuel = st.text_input("Enter the engine fuel type:", 'gasoline')
engine_capacity = st.text_input("Enter the engine capacity:", '2.0')
body_type = st.text_input("Enter the body type:", 'sedan')
has_warranty = st.checkbox("Does the car have a warranty?", True)
ownership = st.text_input("Enter the ownership status:", 'owned')
type_of_drive = st.text_input("Enter the type of drive:", 'front')


def preprocessing(new_data):
    columns = ["manufacturer_name", "transmission", "color", "engine_fuel",
               "body_type", "ownership", "type_of_drive"]
    for column in columns:
        new_data[column] = encoder[column].transform(new_data[column])

    new_data['has_warranty'] = new_data['has_warranty'].map({'True': 1, 'False': 0})
    return new_data


def predict():

    row = np.array([manufacturer_name, transmission, color, odometer_value, year_produced,
                   engine_fuel, engine_capacity, body_type, has_warranty, ownership, type_of_drive])
    new_data_df = pd.DataFrame([row], columns=['manufacturer_name', 'transmission', 'color', 'odometer_value',
                                               'year_produced', 'engine_fuel', 'engine_capacity', 'body_type',
                                               'has_warranty', 'ownership', 'type_of_drive'])
    new_row = np.array(preprocessing(new_data_df))
    new_row = new_row.reshape(1, -1)
    print(new_row)
    prediction = model.predict(new_row)
    st.success(f'The total price is: {prediction[0]}')


trigger = st.button('Predict', on_click=predict)
