import argparse
import os
import sys

import warnings
warnings.filterwarnings('ignore')

# add root package and set project directory to environment
sys.path.insert(0, os.getenv('PROJECT_DIR'))

# local packages
from models.models import get_allowed_models, load_model

from make_predict import load_dataset_and_predict


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='data for predict')
    parser.add_argument('--model_name', required=True, choices=get_allowed_models(), help='which model should to use for make prediction')
    return parser.parse_args()


def main():
    args = args_parse()
    if model := load_model(args.model_name):
        load_dataset_and_predict(args.data, model)
        return
    raise Exception(f'Model {args.model_name} not found')


if __name__ == '__main__':
    main()
