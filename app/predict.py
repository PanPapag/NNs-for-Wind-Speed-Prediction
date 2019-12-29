import argparse
import util
import os

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from util import mean_absolute_percentage_error

MODEL_PATH = '../models/WindDenseNN1.h5'
ACTUAL_PATH = '../datasets/actual.csv'
PREDICTED_PATH = '../datasets/predicted.csv'

def make_args_parser():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(usage='%(prog)s [-h] [-i <input file>]')
    # fill parser with information about program arguments
    parser.add_argument('-i', '--input', action='store',
                        default=None,  metavar='',
                        help='relative path to the input file')
    # return an ArgumentParser object
    return parser.parse_args()

def print_args(args):
    print("Running with the following configuration")
    # get the __dict__ attribute of args using vars() function
    args_map = vars(args)
    for key in args_map:
        print('\t', key, '-->', args_map[key])
    # print an empty line
    print()

def main():
    # parse and print arguments
    args = make_args_parser()
    print_args(args)
    # load input file as well as the timestamps
    X, ts = util.load_file(args.input)
    # load and compile model
    model = util.load_and_compile_model(MODEL_PATH)
    # print model summary
    print("------------------------- Model Summary -------------------------")
    model.summary()
    # predict model output
    y_pred = model.predict(X, batch_size=32)
    # load actual values
    y_true, _ = util.load_file(ACTUAL_PATH)
    # Compute MAE, MAPE and MSE
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    # Check if file already exists. If so delete it
    if os.path.exists(PREDICTED_PATH):
        os.remove(PREDICTED_PATH)
    # Open file an write from scratch
    with open(PREDICTED_PATH, 'a') as file:
        file.write('MAE: {:.4f}\tMAPE: {:.4f}\tMSE: {:.4f}\n'.format(mae,mape,mse))
        df = pd.concat([ts, pd.DataFrame(y_pred)], axis=1)
        df.to_csv(file, index=False, header=False)

if __name__ == '__main__':
    main()
