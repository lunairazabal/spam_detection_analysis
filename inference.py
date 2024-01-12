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

def transform_data(data):
    # Agrega aquí la lógica de transformación de datos si es necesaria
    print(type(data))
    print("Transformación de datos realizada.")
    
    # Llama a la función transformations y aplica las transformaciones al data
   # Llama a la función transformations y aplica las transformaciones al data
    trans1 = str(data).lower()
    trans2 = transformations(trans1)
    trans3 = get_wordnet_pos(trans2)
    trans4 = lemmatize_words(trans3)
    
    # Convierte la lista de palabras lematizadas a una cadena
    trans5 = ' '.join(trans4)
    
    # Llama a la función remove_stopwords
    trans6 = remove_stopwords(trans5)

    print(trans6)

    return trans6

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
    
def get_wordnet_pos(word):
    """Mapea las etiquetas POS de NLTK a etiquetas de WordNet."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_words(words):
    """Realiza la lematización de una lista de palabras."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]


stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)



def predict(data, model_type='ml'):
    # Cargar el modelo y predecir
    if model_type == 'ml':
        model_path = './models/xgb_app.dat'  # Puedes colocar directamente el path del modelo
        
        # Cargar el modelo
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            print("Modelo cargado correctamente.")

        # Cargar los datos
        data_processed = transform_data(data)
        print("Datos cargados y transformados correctamente.")

        # Convertir los datos procesados en una lista de palabras
        words = data_processed.split()

        # Crear un diccionario de bolsa de palabras
        bag_of_words = {word: words.count(word) for word in set(words)}

        # Crear una lista de tuplas (indice, valor) para DMatrix
        data_tuples = [(feature_idx, count) for feature_idx, (word, count) in enumerate(bag_of_words.items())]

        # Obtener las dimensiones del conjunto de datos
        num_features = len(data_tuples)
        num_rows = 1  # Solo una fila de datos

        # Crear un array de dos dimensiones para DMatrix
        data_array = np.zeros((num_rows, num_features))

        # Llenar el array con los valores de las tuplas
        for feature_idx, count in data_tuples:
            data_array[0, feature_idx] = count

        # Crear DMatrix
        data_matrix = xgboost.DMatrix(data_array)
