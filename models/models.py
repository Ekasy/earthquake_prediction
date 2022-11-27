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
        'columns': [
            'Latitude',
            'Longitude',
            'Type',
            'Depth',
            'Magnitude',
            'Magnitude Type',
            'Root Mean Square',
            'Source',
            'Location Source',
            'Magnitude Source', 
            'Status',
        ],
        'model_files': ['xgboost'],
        'additional_args': {
            'seed_order': [687, 749, 577, 158, 755, 904, 15, 187, 940, 606],
            'xgb_params': {
                'eta': 0.1,
                'max_depth': 4,
                'max_bin': 100,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_lambda': 0.5,
                'min_child_weight': 0.01,
                'booster': 'gbtree',
                'tree_method': 'hist',
                'eval_metric': 'rmse',
                'objective': 'reg:squarederror',
                'verbosity': 0,
                'nthread': 12,
                'gamma': 0.5
            },
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
