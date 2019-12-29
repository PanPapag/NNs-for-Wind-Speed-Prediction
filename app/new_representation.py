import argparse
import util
import time
import os

import numpy as np
import pandas as pd

from keras.models import load_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from util import mean_absolute_percentage_error

MODEL_PATH = '../models/WindDenseNN1.h5'
OUTPUT_PATH = '../datasets/new_representations.csv'

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
    # load model
    model = load_model(MODEL_PATH)
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    # print model summary
    print("------------------------- Model Summary -------------------------")
    model.summary()


if __name__ == '__main__':
    main()
