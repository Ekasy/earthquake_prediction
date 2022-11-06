import pandas as pd

import uuid
import os
import time


COLUMNS = [
    'Timestamp',
    'Latitude',
    'Longitude',
]


def _read_dataset(dataset_file):
    dataset = pd.read_csv(dataset_file)
    if set(COLUMNS) - set(dataset.columns):
        raise Exception(f'Check dataset columns: {", ".join(col for col in COLUMNS)} are required!')
    return dataset[COLUMNS]


def _make_predict(dataset, model):
    predictions = model.predict(dataset)
    df = pd.DataFrame(predictions, columns=['Magnitude', 'Depth'])

    predictions_file_name = f'{int(time.time())}_{uuid.uuid4()}'
    predictions_folder = os.path.join(os.environ.get('PROJECT_DIR'), 'predictions')
    if not os.path.exists(predictions_folder):
        os.makedirs('predictions')

    df.to_csv(os.path.join(predictions_folder, predictions_file_name))
    print(f'See {predictions_file_name} for predictions')


def load_dataset_and_predict(dataset_file, model):
    ds = _read_dataset(dataset_file)
    _make_predict(ds, model)
