from keras.preprocessing.sequence import pad_sequences
from keras_bert import get_base_dict
from keras import optimizers
from sklearn.model_selection import train_test_split
import pickle
import copy
import numpy as np
import collections

from model import MyTokenizer, SCBert
from config import opt

def set_dict(data):
    vocab = get_base_dict()
    for tokens in data:
        for tok in tokens.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

def load_data(X):
    
    input_ids = pad_sequences(X, maxlen=opt.maxlen, dtype="long", truncating="post", padding="post")
    
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return input_ids, attention_masks

def neg_sampling(X, seg):
    
    N, T = X.shape
    samples = np.zeros((N, opt.neg_size, T))
    segs = np.zeros((N, opt.neg_size, T))
    for i in range(len(X)):
        indices = np.random.choice(len(X), opt.neg_size)
        samples[i] = X[indices]
        segs[i] = seg[indices]

    return samples, segs

def main():

    # Data
    with open(opt.se_dic_path_for_sc, 'rb') as f:
        dic = pickle.load(f)
    with open(opt.se_path_for_sc, 'rb') as f:
        data = pickle.load(f)
    X, y = zip(*data)
    vocab = set_dict(X)

    tokenizer = MyTokenizer(vocab)
    results = [tokenizer.encode(tokens.split()) for tokens in X]
    token_ids, token_segs = zip(*results)
    
    token_ids, mask = load_data(token_ids)
    token_segs, _ = load_data(token_segs)

    token_ids = np.stack(token_ids, axis=0)
    token_segs = np.stack(token_segs, axis=0)

    y = np.array(y)
    
    np.random.seed(0)
    indices = np.random.permutation(len(token_ids))
    train_size = np.floor(0.9*len(token_ids)).astype(int)

    X_train = token_ids[indices[:train_size]]
    X_test = token_ids[indices[train_size:]]
    seg_train = token_segs[indices[:train_size]]
    seg_test = token_segs[indices[train_size:]]
    y_train = y[indices[:train_size]]
    y_test = y[indices[train_size:]]

    # X_train, X_test, y_train, y_test = train_test_split(token_ids, y, test_size=0.1, random_state=42)
    
    neg_X_train, neg_seg_train = neg_sampling(X_train, seg_train)
    neg_X_test, neg_seg_test = neg_sampling(X_test, seg_test)

    model = SCBert(opt)
    adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer = adam, loss=None)
    model.summary()
    
    model.fit(x = [token_ids, token_segs],
              epochs=10,
              batch_size=100,
              shuffle=True,
              validation_split=0.1)
        
    model.save('checkpoints-scbert/model-val.h5')



if __name__ == "__main__":
    main()
