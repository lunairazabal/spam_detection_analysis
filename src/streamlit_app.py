from inference import transform_data, predict
import streamlit as st
import xgboost as xgb

st.title("Spam Guardian")
#add a picture
from PIL import Image
image = Image.open('C:\Users\lunai\OneDrive\Escritorio\capstone1\Imagen de WhatsApp 2023-12-28 a las 12.02.36_6d2ecfe6.jpg')

# Solicitar texto en inglés
text_input = st.text_input('Introduce un texto en inglés')

# Botón para predecir
btn_predict = st.button('Predecir')

# Verificar si se hizo clic en el botón
if btn_predict:
    # transformar los datos
    data = transform_data(text_input)

    # predecir
    prediction = predict(data)

    # mostrar el resultado
    st.write(prediction)


# from inference import transform_data, predict
# import streamlit as st
# from PIL import Image

# # Cargar una imagen en la aplicación
# image = Image.open('Imagen de WhatsApp 2023-12-28 a las 12.02.36_6d2ecfe6.jpg')
# st.image(image, use_column_width=True)

# st.title("Spam Guardian")

# # Inicializar btn_predict
# btn_predict = None

# # Cuando le den al botón de predecir llamamos primero a transform_data y luego a predict
# text_input = st.text_input('Introduce un texto en inglés')
# if st.button('Predecir'):
#     btn_predict = True

# if btn_predict:
#     # Transformar los datos
#     data = transform_data(text_input)

#     # Predecir
#     prediction = predict(data)

#     # Mostrar el resultado
#     st.write(prediction)
