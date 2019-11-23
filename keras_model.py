from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.constraints import max_norm


def create_model(env):
    dropout_prob = 0.2
    num_units = 256
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.input_shape))
    model.add(Dense(num_units))
    model.add(Activation('relu'))
    model.add(Dense(num_units))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_prob))
    model.add(Dense(env.action_size))
    print(model.summary())
    return model