import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

# Define Neural Network model
def get_model():
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=128, activation='relu'))
    model.add(Dense(7, activation='sigmoid'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Load pretrained keras model
def load_model(path):
    model = load_model(path)
    return model

'''
    - Loads a file into a pandas dataframe
    - Names its columns
    - Discards the first one into a different df
    - Returns a tuple of the values of the initial df
        and the one with the timestamps
'''
def load_file(path):
    # load file into a pandas dataframe
    df = pd.read_csv(path, encoding='utf-8', sep=',')
    # name each column
    dynamic_columns = []
    for i in range(df.shape[1] - 1):
        dynamic_columns.append('df_' + str(i+1))
    df.columns = ['Timestamp'] + dynamic_columns
    # Get the two different dataframes. The one with timestamps only and the
    # one without the timestamps
    timestamps_df = df['Timestamp']
    values_df = df.drop('Timestamp', axis=1)
    # return info in the form of a tuple
    return values_df, timestamps_df

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(np.divide((y_true - y_pred), y_true, \
        out=np.zeros_like((y_true - y_pred)), where=y_true!=0))) * 100
