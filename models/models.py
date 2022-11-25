import models.model_wrappers as mw


# model name to model file (should placed to data folder) mapper
_models = {
    'random_forest': {
        'model_files': ['random_forest.pkl',],
        'model_wrapper': mw.JoblibWrapper
    },
}


def get_allowed_models() -> list:
    return list(_models.keys())


def get_model(model_name: str):
    if model_node := _models.get(model_name):
        return model_node['model_wrapper'](model_node['model_files'])
    return None
