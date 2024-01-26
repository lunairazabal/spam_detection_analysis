import streamlit as st
import os
import re
import nltk
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import xgboost
import pandas as pd
from io import StringIO
import numpy as np
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import nltk


def transform_data(data):
    # Agrega aquí la lógica de transformación de datos si es necesaria
    print(type(data))
    print("Transformación de datos realizada.")
    
    # Llama a la función transformations y aplica las transformaciones al data
   # Llama a la función transformations y aplica las transformaciones al data
    trans1 = str(data).lower()
    trans2 = transformations(trans1)
    trans3= word_tokenize(trans2)
    trans4 = lemmatize_text(trans3)
    trans5 = remove_stopwords(trans4)

    
    num_features = 30  # Ajusta esto según la dimensión correcta

    # Convertir las características a una matriz NumPy con la misma dimensión
    data_array = np.zeros(num_features)

    # Llenar la matriz con las características transformadas
    for feature_idx, count in enumerate(data_array):
        data_array[feature_idx] = count

    return data_array

def transformations(data):
    if data == "NA":
        return "NA"
    else:
        data = re.sub(r'\b(u)\b', 'you', data)
        data = re.sub(r'\b(2)\b', 'to', data)
        data = re.sub(r'\b(i\' m)\b', 'i am', data)
        data = re.sub(r'\b(cant)\b', 'can not', data)
        data = re.sub(r'\b(thats)\b', 'that is', data)
        data = re.sub(r'\b(im)\b', 'i am', data)
        data = re.sub(r'\b(ill)\b', 'i will', data)
        data = re.sub(r'\b(ur)\b', 'your', data)
        data = re.sub(r'\b(4)\b', 'for', data)
        data = re.sub(r'\b(n)\b', 'no', data)
        data = re.sub(r'\b(i\'ll)\b', 'i will', data)
        data = re.sub(r'\b(ü)\b', 'you', data)
        data = re.sub(r'\b(&)\b', 'and', data)
        data = re.sub(r'\b(txt)\b', 'text', data)
        data = re.sub(r'\b(1)\b', 'one', data)
        data = re.sub(r'\b(po)\b', 'purchase order', data)
        data = re.sub(r'\b(dont)\b', 'do not', data)
        data = re.sub(r'\b(don\'t)\b', 'do not', data)
        data = re.sub(r'\b(lor)\b', 'lot', data)
        data = re.sub(r'\b(msg)\b', 'message', data)
        data = re.sub(r'\b(wat)\b', 'what', data)
        data = re.sub(r'\b(na)\b', 'not available', data)
        data = re.sub(r'\b(pls)\b', 'please', data)
        data = re.sub(r'\b(wkly)\b', 'weekly', data)
        data = re.sub(r'\b(i\'d)\b', 'i would', data)
        data = re.sub(r'\b(2nd)\b', 'second', data)
        data = re.sub(r'\b(usf)\b', 'university of south florida', data)
        data = re.sub(r'\b(hor)\b', 'hour', data)
        data = re.sub(r'\b(fr)\b', 'for', data)
        data = re.sub(r'\b(gt)\b', 'get', data)
        data = re.sub(r'\b(lt)\b', 'let', data)
        data = re.sub(r'\b(comp)\b', 'composition', data)
        data = re.sub(r'\b(go)\b', 'gone', data)
        data = re.sub(r'\b(wan)\b', 'want', data)
        data = re.sub(r'\b(mob)\b', 'mobile', data)
        data = re.sub(r'\b(\'s)\b', 'is', data)
        data = re.sub(r'\b(m)\b', 'i am', data)
        data = re.sub(r'\b(ll)\b', 'will', data)
        data = re.sub(r'\b(its)\b', 'it is', data)
        data = re.sub(r'\b(itis)\b', 'it is', data)
        data = re.sub(r'\b(i\'i)\b', 'i', data)
        data = re.sub(r'\b(cos)\b', 'because', data)
        return data

nlp = spacy.load('en_core_web_sm')
import spacy

def word_tokenize(text, language='en_core_web_sm'):
    """
    Tokeniza un texto en inglés utilizando spaCy.

    Parámetros:
    - text: Texto a ser tokenizado.
    - language: Modelo de lenguaje de spaCy a utilizar (por defecto, 'en_core_web_sm').

    Retorna:
    - Lista de tokens.
    """
    nlp = spacy.load(language)
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

# Inicializar el lematizador de spaCy
nlp = spacy.load('en_core_web_sm')

# Supongamos que ya tienes el DataFrame 'sms' con una columna 'Tokens' que contiene listas de palabras
# sms['Tokens'] = ...

# Definir la función de lematización con spaCy
def lemmatize_text(word_list):
    text = ' '.join(word_list)
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return lemmas

from nltk.corpus import stopwords

# Descargar la lista de stopwords en inglés si no lo has hecho antes
nltk.download('stopwords')

# Definir la lista de stopwords en inglés
stop_words = set(stopwords.words('english'))

def remove_stopwords(lemmas):
    """
    Elimina las stopwords de una lista de lemas.

    Parámetros:
    - lemas: Lista de lemas.

    Retorna:
    - Lista de lemas sin stopwords.
    """
    filtered_lemmas = [lemma for lemma in lemmas if lemma.lower() not in stop_words]
    return filtered_lemmas

import pickle
import xgboost
import numpy as np
def predict(data, model_type='ml'):
    # Cargar el modelo y predecir
    if model_type == 'ml':
        model_path = './models/xgb_app.dat'  # Coloca directamente el path del modelo
        
        # Cargar el modelo
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            print("Modelo cargado correctamente.")

        # Transformar los datos
        data_processed = transform_data(data)

        # Realizar la predicción
        prediction = model.predict(np.array([data_processed]))
        print("Predicción realizada:", prediction)

        # Devolver el resultado como "Spam" o "No Spam"
        result = "Spam" if prediction[0] == 1 else "No Spam"
        return result
    else:
        raise ValueError("Tipo de modelo no reconocido. Utiliza 'ml' para modelos de machine learning.")


# def predict(data, model_type='ml'):
#     data_transformed = transform_data(data)
    
#     # Cargar el modelo y predecir
#     if model_type == 'ml':
#         model_path = './models/xgb_app.dat'  # Coloca directamente el path del modelo

#         # Cargar el modelo
#         with open(model_path, 'rb') as model_file:
#             model = pickle.load(model_file)
#             print("Modelo cargado correctamente.")

#         # Tokenizar y lematizar el texto
#         tokens = word_tokenize(data_transformed)
#         lemmatized_tokens = lemmatize_text(tokens)

#         # Crear un diccionario de bolsa de palabras
#         bag_of_words = {word: lemmatized_tokens.count(word) for word in set(lemmatized_tokens)}

#         # Crear una lista de tuplas (indice, valor) para DMatrix
#         data_tuples = [(feature_idx, count) for feature_idx, (word, count) in enumerate(bag_of_words.items())]

#         # Obtener las dimensiones del conjunto de datos
#         num_features = len(data_tuples)
#         num_rows = 1  # Solo una fila de datos

#         # Crear un array de dos dimensiones para DMatrix
#         data_array = np.zeros((num_rows, num_features))

#         # Llenar el array con los valores de las tuplas
#         for feature_idx, count in data_tuples:
#             data_array[0, feature_idx] = count

#         # Crear DMatrix
#         data_matrix = xgb.DMatrix(data_array)

#         # Realizar la predicción
#         prediction = model.predict(data_matrix)
#         print("Predicción realizada:", prediction)
#         return prediction
