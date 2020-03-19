from keras.preprocessing.sequence import pad_sequences
from keras_bert import get_base_dict
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

def main():

    # Data
    with open(opt.se_dic_path_for_sc, 'rb') as f:
        dic = pickle.load(f)
    with open(opt.se_path_for_sc, 'rb') as f:
        data = pickle.load(f)
    X, y = zip(*data)
    vocab = set_dict(X)

    tokenizer = MyTokenizer(vocab)
    token_ids = [tokenizer.encode(tokens.split())[0] for tokens in X]

    X_train, X_test, y_train, y_test = train_test_split(token_ids, y, test_size=0.1, random_state=42)
    X_train, mask_train = load_data(X_train)
    X_test, mask_test = load_data(X_test)

    model = SCBert(opt)

if __name__ == "__main__":
    main()
