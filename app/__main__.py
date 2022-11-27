import argparse
import os
import sys
import time
import uuid

import warnings
warnings.filterwarnings('ignore')

# add root package and set project directory to environment
sys.path.insert(0, os.getenv('PROJECT_DIR'))

import pandas as pd

# local packages
from models.models import get_allowed_models, get_model
from models.helpers import NN


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='data for predict')
    parser.add_argument('--model_name', required=True, choices=get_allowed_models(), help='which model should to use for make prediction')
    return parser.parse_args()


def main():
    args = args_parse()
    if model := get_model(args.model_name):
        model.load()
        model.prepare(args.data)
        predictions = model.predict()

        df = pd.DataFrame(predictions, columns=['Magnitude', 'Depth'])

        predictions_file_name = f'{int(time.time())}_{uuid.uuid4()}'
        predictions_folder = os.path.join(os.environ.get('PROJECT_DIR'), 'predictions')
        if not os.path.exists(predictions_folder):
            os.makedirs('predictions')

        df.to_csv(os.path.join(predictions_folder, predictions_file_name))
        print(f'See {predictions_file_name} for predictions')
        return

    raise Exception(f'Model {args.model_name} not found')


if __name__ == '__main__':
    main()
