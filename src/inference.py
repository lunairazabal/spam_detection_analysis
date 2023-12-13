import os

MODELS_PATH = 'models'

def transform_data(data):
    # utilizar el mismo pipeline de etl que se creo en el notebook de etl
    # transformar los datos
    return data

def predict(data_processed, model_type='dl'):
    # cargar el model
    # predecir
    if dl:
        model = pickle.load(open(os.path.join(MODELS_PATH, 'model_dl.h5'), 'rb'))
    else:
        model = pickle.load(open(os.path.join(MODELS_PATH, 'model.pkl'), 'rb'))

    prediction = model.predict(data_processed)

    return prediction