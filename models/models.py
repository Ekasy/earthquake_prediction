import os
import joblib


# model name to model file (should placed to data folder) mapper
_models = {
    'random_forest': 'random_forest.pkl',
}


def get_allowed_models() -> list:
    return list(_models.keys())


def load_all_models() -> dict:
    models = {}
    for model_name, model_file in _models.items():
        models[model_name] = os.path.join(os.environ.get('PROJECT_DIR'), 'models/data', model_file)
    return models


def load_model(model_name: str):
    if model_file := _models.get(model_name):
        path = os.path.join(os.environ.get('PROJECT_DIR'), 'models', 'data', model_file)
        return joblib.load(path)
    return None
