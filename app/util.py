from keras.models import Sequential
from keras.layers import Dense

# Define Neural Network model
def get_model():
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=128, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    # compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model
