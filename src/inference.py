import os
import pickle

MODELS_PATH = 'models'

def transform_data(data):
    # utilizar el mismo pipeline de ETL que se creó en el notebook de ETL
    # transformar los datos
    return data

def predict(data_processed, model_type='ml'):
    # cargar el modelo
    if model_type == 'ml':
        model = pickle.load(open(os.path.join(MODELS_PATH, 'xgb.app.pkl'), 'rb'))
    else:
        raise ValueError("Tipo de modelo no reconocido. Utiliza 'ml' para modelos de machine learning.")

    prediction = model.predict(data_processed)

    return prediction



