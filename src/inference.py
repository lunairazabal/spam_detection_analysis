# import os
# import pickle

# MODELS_PATH = 'models'

# def transform_data(data):
#     # utilizar el mismo pipeline de ETL que se creó en el notebook de ETL
#     # transformar los datos
#     return data

# def predict(data_processed, model_type='ml'):
#     # cargar el modelo
#     if model_type == 'ml':
#         model = pickle.load(open(os.path.join(MODELS_PATH, 'xg_boost_model.pkl'), 'rb'))
#     else:
#         raise ValueError("Tipo de modelo no reconocido. Utiliza 'ml' para modelos de machine learning.")

#     prediction = model.predict(data_processed)

#     return prediction

from inference import transform_data, predict
import streamlit as st



# cuando le den al boton de predecir llamamos primero a transform_data y luego a predict

text_input = st.text_input('Introduce un texto en inglés')

if btn_predict:
    # transformar los datos
    data = transform_data(text_input)

    # predecir
    prediction = predict(data)

    # mostrar el resultado
    st.write(prediction)