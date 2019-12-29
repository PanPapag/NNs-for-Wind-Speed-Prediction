import numpy as np
import pandas as pd

from keras.models import Model, load_model

# Define a new model which has only one layer, the first layer
# of the pretrained one
def get_intermediate_layer_model(model, layer_name):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model
# Load and compile pretrained keras model
def load_and_compile_model(path):
    model = load_model(path)
    model.compile(loss='mean_squared_error', optimizer='adam')
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
    # name columns
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
