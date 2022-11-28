import torch

import pandas as pd
import numpy as np
import torch.nn as nn
import xgboost as xgb


class NN(nn.Module):
    def __init__(self, input_size, n_hidden_neurons):
        super(NN, self).__init__()

        self.fc1 = nn.Linear(input_size, n_hidden_neurons[0])
        self.act1 = torch.nn.ReLU()
        self.fc2 = nn.Linear(n_hidden_neurons[0], 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x.reshape(-1)


class GBModel:
    def __init__(self, n_models=1, early_stopping_rounds=40, use_cv_train=True, cv_folds=5, params_list=None,
                 random_seed=42, model_name="default_gb_model", save_models_path=None):
        self.params_list = params_list
        self.random_seed = random_seed
        self.n_models = n_models

        self.boosting_regressor = xgb
        self.boosters = [None for i in range(self.n_models)]
        self.customize_data = xgb.DMatrix
        self.customize_data_predict = xgb.DMatrix
        self.train_rmse_mean = 'train-rmse-mean'
        self.eval_params = lambda dtrain, dvalid: {
            'evals': [(dtrain, 'train'), (dvalid, 'valid')]
        }
        if self.params_list is None:
            self.random_seed = 42
            self.seeds = np.random.choice(range(1000), self.n_models, replace=False)

            self.params_list = [{
                'eta': 0.1,
                'eval_metric': 'rmse',
                'max_depth': 4,
                'nthread': 8,
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'random_seed': self.seeds[i],
            } for i in range(self.n_models)]
        else:
            self.seeds = [
                self.params_list[i]['random_seed']
                for i in range(self.n_models)
            ]

        self.boost_type = 'xgb'
        self.early_stopping_rounds = early_stopping_rounds
        self.use_cv_train = use_cv_train
        self.cv_folds = cv_folds
        self.model_name = model_name
        self.save_models_path = save_models_path
        self.num_trees_list = False
        self.evals = []

    def get_customize_data_predict(self):
        return self.customize_data_predict
