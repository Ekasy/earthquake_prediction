import models.model_wrappers as mw


# model name to model file (should placed to data folder) mapper
_models = {
    'random_forest': {
        'columns': [
            'Date',
            'Time',
            'Latitude',
            'Longitude',
        ],
        'model_files': ['random_forest.pkl',],
        'model_wrapper': mw.JoblibWrapper,
    },
    'torch_neural_network': {
        'columns': [],
        'model_files': ['torch_magnitude.pth', 'torch_depth.pth'],
        'model_wrapper': mw.TorchWrapper,
    },
    'xgboost': {
        'columns': [],
        'model_files': ['final_model_Magnitude_seed', 'final_model_Depth_seed'],
        'additional_args': {
            'seed_order': [687, 749, 577, 158, 755, 904, 15, 187, 940, 606],
        },
        'model_wrapper': mw.XGBoostWrapper,
    },
}


def get_allowed_models() -> list:
    return list(_models.keys())


def get_model(model_name: str):
    if model_node := _models.get(model_name):
        return model_node['model_wrapper'](model_node['model_files'], model_node['columns'], model_node.get('additional_args'))
    return None
