
import os
import pickle
import xgboost

DATA_DIR = os.path.join('..', 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_PATH = os.path.join('..', 'models')

# def transform_data(data):
#     # utilizar el mismo pipeline de ETL que se creó en el notebook de ETL
#     # transformar los datos
#     return data

# def predict(data_processed, model_type='ml'):
#     # Cargar el modelo y predecir
#     if model_type == 'ml':
#         model_path = os.path.join(MODELS_PATH, 'xgb_app.dat')
#         data_path = os.path.join(PROCESSED_DATA_DIR, 'dataset_procesado.csv')

#         # Carga el modelo
#         with open(model_path, 'rb') as model_file:
#             model = pickle.load(model_file)

#         # Carga los datos
#         data = xgboost.DMatrix(data_path)
#     else:
#         raise ValueError("Tipo de modelo no reconocido. Utiliza 'ml' para modelos de machine learning.")

#     prediction = model.predict(data)

#     return prediction

import os
import pickle
import xgboost
import pandas as pd

DATA_DIR = os.path.join('..', 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_PATH = os.path.join('..', 'models')

def transform_data(data):
    # Utilizar el mismo pipeline de ETL que se creó en el notebook de ETL
    # Transformar los datos
    return data

def predict(data_processed, model_type='ml'):
    # Cargar el modelo y predecir
    if model_type == 'ml':
        model_path = os.path.join(MODELS_PATH, 'xgb_app.dat')
        data_path = os.path.join(RAW_DATA_DIR, 'sms_test.csv')

        # Carga el modelo
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        # Carga los datos
        data = pd.read_csv(data_path)
        data_processed = transform_data(data)  # Asegúrate de tener una función transform_data adecuada

        # Puedes continuar con el procesamiento específico si es necesario

        # Puedes ajustar el modelo.predict según lo que necesites hacer con los datos procesados
        prediction = model.predict(xgboost.DMatrix(data_processed))
    else:
        raise ValueError("Tipo de modelo no reconocido. Utiliza 'ml' para modelos de machine learning.")

    return prediction
