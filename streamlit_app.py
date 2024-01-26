from inference import transform_data, predict
import streamlit as st
import pandas as pd
import numpy as np

#add a picture
from PIL import Image
image = Image.open('C:/Users/lunai/OneDrive/Escritorio/capstone1/spam_detection_analysis/spamtdetc3.png')

st.image(image, use_column_width=True)

#st.title("Spam Guardian")


# Solicitar texto en inglés
st.markdown('## Put your SMS here:')
text_input = st.text_input('')

st.write('<p style="text-align:right;">160 characters</p>', unsafe_allow_html=True)


# Botón para predecir
btn_predict = st.button('Predict')

# Verificar si se hizo clic en el botón
if btn_predict:
    # Transformar los datos
    data = transform_data(text_input)

    # Predecir
    prediction = predict(data)

    # Mapear la predicción a etiquetas más descriptivas (ajusta esto según tu lógica)
    if prediction == 1:
        result_label = "Spam"
    else:
        result_label = "No Spam"

    # Mostrar el resultado
    st.write(f"The prediction is: {result_label}")
    