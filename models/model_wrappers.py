import datetime
import joblib
import os
import time
import torch

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from models.helpers import GBModel


class BaseWrapper:
    _TARGET_COLUMNS = [
        'Magnitude',
        'Depth',
    ]

    def __init__(self, model_files: list, columns: list, additional_args):
        '''
        model_files - list of model_paths where:
                            model_files[0] - single model / magnitude predictor model
                            model_files[1] - not exist / depth predictor model
        '''
        self._columns = columns
        self._model_files = model_files
        self._additional_args = additional_args
        self._models = []
        self._dataset = None

    def _read_dataset(self, dataset_file):
        dataset = pd.read_csv(dataset_file)
        if not self._columns:
            return dataset

        if set(self._columns) - set(dataset.columns):
            raise Exception(f'Check dataset columns: {", ".join(col for col in self._columns)} are required!')
        return dataset[self._columns]

    def _load(self, path):
        raise NotImplementedError()

    def load(self):
        for model_file in self._model_files:
            path = os.path.join(os.environ.get('PROJECT_DIR'), 'models', 'data', model_file)
            self._models.append(self._load(path))

    def prepare(self, dataset: str):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()


class JoblibWrapper(BaseWrapper):
    def __init__(self, model_files: list, columns: list, additional_args):
        super().__init__(model_files, columns, additional_args)

    def _load(self, path):
        return joblib.load(path)

    def prepare(self, dataset_file: str):
        self._dataset = self._read_dataset(dataset_file)
        timestamp = []
        for d, t in zip(self._dataset['Date'], self._dataset['Time']):
            try:
                ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
                timestamp.append(time.mktime(ts.timetuple()))
            except ValueError:
                timestamp.append('ValueError')

        timeStamp = pd.Series(timestamp)
        self._dataset['Timestamp'] = timeStamp.values
        final_data = self._dataset.drop(['Date', 'Time'], axis=1)
        self._dataset = final_data[final_data.Timestamp != 'ValueError']

    def predict(self):
        predictions = []
        for model in self._models:
            predict = model.predict(self._dataset)
            predictions.append(predict.ravel())

        return np.reshape(np.array(predictions), (-1, len(self._TARGET_COLUMNS)))


class TorchWrapper(BaseWrapper):
    _LIMIT_FOR_LABEL_ENCODING = 2
    _N_EPOCHS=100
    _BATCH_SIZE=247

    def __init__(self, model_files: list, columns: list, additional_args):
        super().__init__(model_files, columns, additional_args)
        self._loaders = None

    def _load(self, path):
        model = torch.load(path, map_location=torch.device('cpu'))
        model.eval()
        return model

    def _create_tensor_ds(self, X_train, y_train):
        X_tnsr = torch.tensor(X_train.values, dtype=torch.float32)
        y_tnsr = torch.tensor(y_train.values, dtype=torch.float32)
        return TensorDataset(X_tnsr, y_tnsr)

    def prepare(self, dataset_file: str):
        self._dataset = self._read_dataset(dataset_file)
        X = self._dataset.drop(self._TARGET_COLUMNS, axis=1)
        y_m = self._dataset['Magnitude']
        y_d = self._dataset['Depth']

        X = StandardScaler().fit_transform(X)

        mask = np.isnan(X)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        X = X[np.arange(idx.shape[0])[:,None], idx]

        ds_m = self._create_tensor_ds(pd.DataFrame(X), y_m)
        ds_d = self._create_tensor_ds(pd.DataFrame(X), y_d)

        self._loaders = [
            DataLoader(dataset = ds_m, batch_size=self._BATCH_SIZE, shuffle=True),
            DataLoader(dataset = ds_d, batch_size=self._BATCH_SIZE, shuffle=True)
        ]

    def _get_predictions(self, loader, model):
        saved_preds = []
        with torch.no_grad():
            for x, _ in loader:
                scores = model(x)
                saved_preds += scores.tolist()
        return saved_preds

    def predict(self):
        predictions = []
        for model, loader in zip(self._models, self._loaders):
            predict = self._get_predictions(loader, model)
            predictions.append(predict)
        return np.array(predictions).transpose()


class XGBoostWrapper(BaseWrapper):
    def __init__(self, model_files: list, columns: list, additional_args):
        super().__init__(model_files, columns, additional_args)
        self._customize_data_predict = xgb.DMatrix

    def load(self):
        for model_file in self._model_files:
            local_models = []
            for seed in self._additional_args['seed_order']:
                path = os.path.join(os.environ.get('PROJECT_DIR'), 'models', 'data', f'{model_file}_{seed}')
                bst = xgb.Booster({'nthread': 4})
                bst.load_model(path)
                local_models.append(bst)
            self._models.append(local_models)

    def prepare(self, dataset: str):
        self._dataset = pd.read_csv(dataset)
        self._dataset = self._dataset.drop(['Magnitude', 'Depth'], axis=1)

    def predict(self):
        predictions = []
        for local_models in self._models:
            data_to_predict = self._customize_data_predict(self._dataset)
            score = pd.Series(np.zeros((self._dataset.shape[0],)), index=self._dataset.index)
            for model in local_models:
                prediction = model.predict(data_to_predict)
                score += prediction
            score /= len(self._additional_args['seed_order'])
            predictions.append(score)
        return np.array(predictions).transpose()

