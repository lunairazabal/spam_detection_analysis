# from inference import transform_data, predict
# import streamlit as st

# #put and image in the app
# from PIL import Image
# image = Image.open('Imagen de WhatsApp 2023-12-28 a las 12.02.36_6d2ecfe6.jpg')
# st.image(image, use_column_width=True)

# st.title("Spam Guardian")



# # cuando le den al boton de predecir llamamos primero a transform_data y luego a predict

# text_input = st.text_input('Introduce un texto en inglés')
# btn_predict=st.button('Predecir')

# if btn_predict:
#     # transformar los datos
#     data = transform_data(text_input)

#     # predecir
#     prediction = predict(data)

#     # mostrar el resultado
#     st.write(prediction)

from inference import transform_data, predict
import streamlit as st
from PIL import Image

# Establecer la configuración de la página para ocupar todo el ancho
st.set_page_config(layout="wide")

# Cargar una imagen en la aplicación
image = Image.open('Imagen de WhatsApp 2023-12-28 a las 12.02.36_6d2ecfe6.jpg')
st.image(image, use_column_width=True)

# Crear una columna para el título y el cuadro de entrada de texto
col1, col2 = st.beta_columns(2)

# Título
with col1:
    st.title("Spam Guardian")

# Cuadro de entrada de texto
with col2:
    text_input = st.text_input('Introduce un texto en inglés')

# Botón de predecir
if st.button('Predecir'):
    # Transformar los datos
    data = transform_data(text_input)

    # Predecir
    prediction = predict(data)

    # Mostrar el resultado
    st.write(prediction)
