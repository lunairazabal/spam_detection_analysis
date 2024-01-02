import os
import pickle

MODELS_PATH = 'models'

import re
import string
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import graphviz 
from sklearn import tree
from sklearn import metrics
from wordcloud import WordCloud
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
DATA_DIR = os.path.join( '..', 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

def transform_data(data):
    # data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'processed'))
    # data = pd.read_csv(os.path.join(RAW_DATA_DIR, 'data'))
    #transformar los datos

    # utilizar el mismo pipeline de ETL que se creó en el notebook de ETL
    # transformar los datos
    return data

def predict(data_processed, model_type='ml'):
    # cargar el modelo
    if model_type == 'ml':
        model = pickle.load(open(os.path.join(MODELS_PATH, 'xgb_app.dat'), 'rb'))
    else:
        raise ValueError("Tipo de modelo no reconocido. Utiliza 'ml' para modelos de machine learning.")

    prediction = model.predict(data_processed)

    return prediction



