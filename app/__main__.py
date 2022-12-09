import argparse
import datetime
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
    parser.add_argument('--longitude_min', required=True, type=float, help='Min value for longitude')
    parser.add_argument('--longitude_max', required=True, type=float, help='Max value for longitude')
    parser.add_argument('--latitude_min', required=True, type=float, help='Min value for latitude')
    parser.add_argument('--latitude_max', required=True, type=float, help='Max value for latitude')
    parser.add_argument('--days', type=int, default=30, help='Days number to predict to future')
    parser.add_argument('--max_magnitude', type=float, default=8.0, help='Max magnitude for raise alert')
    return parser.parse_args()


def main():
    args = args_parse()

    assert args.longitude_min < args.longitude_max and args.longitude_min > 0.0 and args.longitude_max < 180.0, 'longitude must be in [0; 180] interval'
    assert args.latitude_min < args.latitude_max and args.latitude_min > -90.0 and args.latitude_max < 90.0, 'latitude must be in [-90; 90] interval'
    assert args.days > 0, 'days should be positive number'

    if model := get_model(args.model_name):
        model.load()
        try:
            model.prepare(args.data, args.longitude_min, args.longitude_max, args.latitude_min, args.latitude_max, args.days)
            predictions = model.predict()
        except ValueError:
            print('No points in inputed field')
            exit(1)

        df = pd.DataFrame(predictions, columns=['Magnitude', 'Depth'])

        predictions_file_name = f'{int(time.time())}_{uuid.uuid4()}'
        predictions_folder = os.path.join(os.environ.get('PROJECT_DIR'), 'predictions')
        if not os.path.exists(predictions_folder):
            os.makedirs('predictions')

        df.to_csv(os.path.join(predictions_folder, predictions_file_name))
        print(f'See {predictions_file_name} for predictions')

        for _, row in model.union_data(df).iterrows():
            magnitude = row['Magnitude']
            if magnitude > args.max_magnitude:
                str_ts = datetime.datetime.fromtimestamp(row["Timestamp"]).strftime('%Y.%m.%d')
                print(f'Warning! {str_ts} magnitude will be {magnitude} at ({row["Longitude"]}, {row["Latitude"]})')
        return

    raise Exception(f'Model {args.model_name} not found')


if __name__ == '__main__':
    main()
