from keras_bert import load_trained_model_from_checkpoint
from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths
from keras_bert import Tokenizer

from keras.layers import Input, Conv1D, MaxPool1D, Flatten, UpSampling1D, BatchNormalization, LSTM, RepeatVector, Lambda, Dense, Activation, Reshape
from keras.models import Model, load_model

import keras.backend as K
from keras.engine.topology import Layer

class MyTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            else:
                R.append('[UNK]')
        return R

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

        self.built = True

    def compute_mask(self, input_tensor, mask=None):
        return None
    
    def call(self, input_tensor):
        
        x = input_tensor[0] #(N, T, H)
        y = input_tensor[1] #(N, H)

        # Attention Mechanism
        y = K.repeat_elements(K.expand_dims(y, axis=-2), self.time_steps, axis=1)

        concat = K.concatenate([x, y], axis=-1) #(N, T, 2H)
        h1 = K.tanh(K.dot(concat, self.W1) + self.b1)
        h2 = K.relu(K.dot(h1, self.W2) + self.b2)
        
        a = K.exp(h2)
        a /= K.cast(K.sum(a, axis=-2, keepdims=True) + K.epsilon(), K.floatx())
        a = K.squeeze(a, axis=-1)

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
        x = x * K.expand_dims(weights, axis=-1)
        return K.sum(x, axis= 1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

class WeightedEmbedding(Layer):
    """
    Compute weighted sum between cluster embeddings and scores
    return shape (N, H)
    """
    def __init__(self, hidden_dim, **kwargs):
        super(WeightedEmbedding, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
    
    def compute_mask(self, input_tensor, mask=None):
        return None
    
    def build(self, input_shape):
        
        n_clusters = input_shape[1]
        self.W = self.add_weight(shape=(n_clusters, self.hidden_dim),
                                 initializer='glorot_uniform', name='W_embedding')
        self.built = True

    def call(self, input_tensor):
        return K.dot(input_tensor, self.W)

class CustomLoss(Layer):
    """
    Compute losses: reconstruction loss + negative loss
    return loss
    """

    def __init__(self, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)

    def norm(self, tensor):
        return K.cast(K.epsilon() + K.sqrt(K.sum(K.square(tensor), axis=-1, keepdims=True)), K.floatx())
    
    def call(self, input_tensor):

        z_s, r_s = input_tensor[0], input_tensor[1]

        z_s = z_s / self.norm(z_s)
        r_s = r_s / self.norm(r_s)

        #pos = K.sum(z_s*r_s, axis=1) / (self.norm(z_s)*self.norm(r_s))
        #neg = K.sum(n_s*r_s, axis=1) / (self.norm(n_s)*self.norm(r_s))
        pos = K.sum(z_s*r_s, axis=-1, keepdims=False)
        loss = K.cast(K.sum(K.maximum(0., (1. - pos ))), K.floatx())
        
        self.add_loss(loss, inputs = input_tensor)
        return loss
        # z_s = z_s / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(z_s), axis=-1, keepdims=True)), K.floatx())
        # z_n = z_n / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(z_n), axis=-1, keepdims=True)), K.floatx())
        # r_s = r_s / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(r_s), axis=-1, keepdims=True)), K.floatx())

        # steps = z_n.shape[1]

        # pos = K.sum(z_s*r_s, axis=-1, keepdims=True)
        # pos = K.repeat_elements(pos, steps, axis=-1)
        # r_s = K.expand_dims(r_s, dim=-2)
        # r_s = K.repeat_elements(r_s, steps, axis=1)
        # neg = K.sum(z_n*r_s, axis=-1)

        # loss = K.cast(K.sum(T.maximum(0., (1. - pos + neg)), axis=-1, keepdims=True), K.floatx())

    def compute_mask(self, input_tensor, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return 1
    

def SCBert(opt):

    # Load bert model
    model_path = get_pretrained(PretrainedList.multi_cased_base)
    paths = get_checkpoint_paths(model_path)

    bert_model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=opt.maxlen)

    for l in bert_model.layers:
        l.trainable = False

    x_input = Input(shape=(opt.maxlen,), name='x_input')
    x_segment = Input(shape=(opt.maxlen,), name='x_segment')
    
    word_embeddings = bert_model([x_input, x_segment])

    # Negative samples
    # neg_input = Input(shape=(opt.maxlen,))
    # neg_segment = Input(shape=(opt.maxlen,))
    # neg_embeddings = bert_model([neg_input, neg_segment]) # (Nxneg)xTxH
    # N_neg, T, H = neg_embeddings.shape
    # neg_embeddings = Lambda(lambda x: x, output_shape=lambda s:s)(neg_embeddings)
    # neg_embeddings = Reshape((opt.neg_size, T, H))(neg_embeddings) # NxnegxTxH
    # neg_embeddings = Lambda(lambda x: K.mean(x, axis=-2, keepdims=False))(neg_embeddings) # NxTxH
    # n_s = Lambda(lambda x: K.mean(x, axis=-2, keepdims=False))(neg_embeddings) # NxH
    
    # Sentence Representation
    y_s = Lambda(lambda x: K.mean(x, axis=-2))(word_embeddings) # NxH
    attn_weights = Attention(name='att_weights')([word_embeddings, y_s]) # NxT
    
    z_s = WeightedSum(name='weighted_sum')([word_embeddings, attn_weights]) # NxH

    # Get probability score for clusters
    preds = Dense(opt.n_clusters)(z_s)
    scores = Activation('softmax', name='scores')(preds)
    r_s = WeightedEmbedding(opt.hidden_dim)(scores) # NxH
    
    # Calculate loss
    loss = CustomLoss()([z_s, r_s])

    model = Model(inputs=[x_input, x_segment], outputs=loss)

    return model










    











