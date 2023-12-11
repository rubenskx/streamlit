import gradio as gr
import numpy as np
import pickle
import scipy.stats as stats
import pandas as pd

# Load your model and encoder
with open('mymodel.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as l:
    encoder = pickle.load(l)


def preprocessing(new_data):
    columns = ["manufacturer_name", "transmission", "color", "engine_fuel",
               "body_type", "ownership", "type_of_drive"]
    for column in columns:
        new_data[column] = encoder[column].transform([new_data[column]])[0]
    new_data['has_warranty'] = new_data['has_warranty'].map(
        {'True': 1, 'False': 0})
    return new_data


def predict(manufacturer_name, transmission, color, odometer_value, year_produced,
            engine_fuel, engine_capacity, body_type, has_warranty, ownership, type_of_drive):
    row = np.array([manufacturer_name, transmission, color, odometer_value, year_produced,
                    engine_fuel, engine_capacity, body_type, has_warranty, ownership, type_of_drive])
    new_data_df = pd.DataFrame([row], columns=['manufacturer_name', 'transmission', 'color', 'odometer_value',
                                               'year_produced', 'engine_fuel', 'engine_capacity', 'body_type',
                                               'has_warranty', 'ownership', 'type_of_drive'])
    new_row = preprocessing(new_data_df)
    new_row = new_row.values.reshape(1, -1)
    print(new_row)
    prediction = model.predict(new_row)
    return f'The total price is: {prediction[0]}'


# Create the Gradio interface
iface = gr.Interface(fn=predict,
                     inputs=["text", "text", "text", "text", "text",
                             "text", "text", "text", "checkbox", "text", "text"],
                     outputs="text")

iface.launch()
