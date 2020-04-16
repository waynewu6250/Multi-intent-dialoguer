import keras
from keras.layers import Input, Conv1D, MaxPool1D, Flatten, UpSampling1D, BatchNormalization, LSTM, RepeatVector, Lambda, Dense, Activation, Reshape
from keras.models import Model, load_model
from keras.engine.topology import Layer
import keras.backend as K
import numpy as np
import tensorflow as tf

class Attention(Layer):
    """
    Compute attention weights between x and y
    return shape (N, T)
    """

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):

        self.time_steps = input_shape[0][1]

        self.W1 = self.add_weight(shape=(input_shape[0][-1]+input_shape[0][-1], 32),
                                  initializer='glorot_normal', name='W_attention1')
        self.b1 = self.add_weight(shape=(32,), 
                                  initializer='zero', 
                                  name='b_attention1')
        self.W2 = self.add_weight(shape=(32, 1),
                                  initializer='glorot_normal', name='W_attention2')
        self.b2 = self.add_weight(shape=(1,), 
                                  initializer='zero', 
                                  name='b_attention2')
        
        # self.mask = K.variable(np.arange(self.time_steps))

        self.built = True

    def compute_mask(self, input_tensor, mask=None):
        return None
    
    def call(self, input_tensor):
        
        x = input_tensor[0] #(N, T, H)
        y = input_tensor[1] #(N, H)
        x_length = input_tensor[2] #(N, 1)

        # Mask
        # self.mask = K.expand_dims(self.mask, axis=0)
        # x_length = K.repeat_elements(x_length, self.time_steps, axis=1)
        # self.mask = self.mask >= x_length
        # self.mask = K.cast(self.mask, 'float32') * -float('inf')
        # self.mask = K.expand_dims(self.mask, axis=-1)

        # Attention Mechanism
        y = K.repeat_elements(K.expand_dims(y, axis=-2), self.time_steps, axis=1)

        concat = K.concatenate([x, y], axis=-1) #(N, T, 2H)
        h1 = K.tanh(K.dot(concat, self.W1) + self.b1)
        h2 = K.relu(K.dot(h1, self.W2) + self.b2)

        #h2 = h2 + self.mask
        
        a = K.exp(h2)
        a /= K.cast(K.sum(a, axis=-2, keepdims=True) + K.epsilon(), K.floatx())

        return a

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1])

class WeightedSum(Layer):
    """
    Compute weighted sum between x and weights
    return shape (N, H)
    """
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def compute_mask(self, input_tensor, mask=None):
        return None
    
    def call(self, input_tensor):

        x = input_tensor[0]
        weights = input_tensor[1]
        x = x * weights
        return K.sum(x, axis=-2)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

class CustomLoss(Layer):
    """
    Compute losses: reconstruction loss + negative loss
    return loss
    """

    def __init__(self, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
    
    def call(self, input_tensor):

        z_s, r_s, weights = input_tensor[0], input_tensor[1], input_tensor[2]
        weights = K.squeeze(weights, axis=-1)
        loss = K.mean(K.sum(K.square(z_s-r_s), axis=-1)) - 0.01*K.sum(K.sum(weights * K.log(weights)))
        self.add_loss(loss, inputs = input_tensor)
        return loss

def CAE_model(input_shape=(20,768), filters=[16, 8, 1], kernel_size=3):
    
    x_input = Input(shape=input_shape)
    x_length = Input(shape=(1,))

    y_s = Lambda(lambda x: K.mean(x, axis=-2))(x_input) # NxH
    attn_weights = Attention(name='att_weights')([x_input, y_s, x_length]) # NxT
    z_s = WeightedSum(name='weighted_sum')([x_input, attn_weights]) # NxH
    z_s = Lambda(lambda x: K.expand_dims(x, axis=-1), name='average_vector')(z_s) # NxHx1

    x = Conv1D(filters[0], kernel_size, activation="relu", padding="same")(z_s)
    x = MaxPool1D(2, padding="same")(x)
    x = Conv1D(filters[1], kernel_size, activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = MaxPool1D(2, padding="same")(x)
    x = Conv1D(filters[2], kernel_size, activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    encoded = MaxPool1D(2, padding="same")(x)

    encoder = Model([x_input, x_length], encoded)

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
    
    loss = CustomLoss()([z_s, decoded, attn_weights])
    autoencoder = Model([x_input, x_length], loss)

    return autoencoder