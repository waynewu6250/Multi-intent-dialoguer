import keras
from keras.layers import Input, Conv1D, MaxPool1D, Flatten, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model, load_model


def CAE_model(input_shape=(768,1), filters=[16, 8, 1], kernel_size=3):

    x_input = Input(shape=input_shape)
    x = Conv1D(filters[0], kernel_size, activation="relu", padding="same")(x_input)
    x = MaxPool1D(2, padding="same")(x)
    x = Conv1D(filters[1], kernel_size, activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = MaxPool1D(2, padding="same")(x)
    x = Conv1D(filters[2], kernel_size, activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    encoded = MaxPool1D(2, padding="same")(x)

    encoder = Model(x_input, encoded)

    x = Conv1D(filters[2], kernel_size, activation="relu", padding="same")(encoded)
    #x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters[1], 2, activation='relu', padding="same")(x)
    #x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters[0], kernel_size, activation='relu', padding="same")(x)
    #x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(x_input, decoded)

    return autoencoder