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