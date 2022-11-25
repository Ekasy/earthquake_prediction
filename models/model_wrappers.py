import joblib
import os

import numpy as np
import pandas as pd



class BaseWrapper:
    _COLUMNS = [
        'Timestamp',
        'Latitude',
        'Longitude',
    ]

    _TARGET_COLUMNS = [
        'Magnitude',
        'Depth',
    ]

    def __init__(self, model_files: list):
        '''
        models - list of model_paths where:
                            models[0] - single model / magnitude predictor model
                            models[1] - not exist / depth predictor model
        '''
        self._model_files = model_files
        self._models = []
        self._dataset = None

    def _read_dataset(self, dataset_file):
        dataset = pd.read_csv(dataset_file)
        if set(self._COLUMNS) - set(dataset.columns):
            raise Exception(f'Check dataset columns: {", ".join(col for col in self._COLUMNS)} are required!')
        return dataset[self._COLUMNS]

    def load(self):
        raise NotImplementedError()

    def prepare(self, dataset: str):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()


class JoblibWrapper(BaseWrapper):
    def __init__(self, model_files: list):
        super().__init__(model_files)

    def load(self):
        for model_file in self._model_files:
            path = os.path.join(os.environ.get('PROJECT_DIR'), 'models', 'data', model_file)
            self._models.append(joblib.load(path))

    def prepare(self, dataset_file: str):
        self._dataset = self._read_dataset(dataset_file)

    def predict(self):
        predictions = []
        for model in self._models:
            predict = model.predict(self._dataset)
            predictions.append(predict.ravel())

        return np.reshape(np.array(predictions), (-1, len(self._TARGET_COLUMNS)))
