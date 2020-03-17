import pickle
import numpy as np
import torch
import torch.nn as nn

import keras
from keras.layers import Input, Conv1D, MaxPool1D, Flatten, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model, load_model
import h5py
from sklearn.cluster import KMeans
from sklearn import metrics

from model import DCEC
from config import opt

class PerformClustering:
    
    def __init__(self, dic_path, embeddings_path, label_id=None):
        '''
        data:
        1. embeddings: id: [(sent1,emb1), (sent2,emb2), (sent3,emb3), ...]
        2. data: [emb1, emb2, emb3, ...]

        dictinary:
        1. dic: label: label_id
        2. reverse_dic: label_id: label
        3. emb2sent: emb: sentence
        4. emb2id: emb: sentence_id
        5. labels_ref: sentence_id: label_id

        '''
        
        self.dic_path = dic_path
        self.embeddings_path = embeddings_path

        with open(self.dic_path, 'rb') as f:
            self.dic = pickle.load(f)
        self.reverse_dic = {v: k for k,v in self.dic.items()}
        self.embeddings = torch.load(self.embeddings_path)
        self.emb2sent, self.emb2id, self.data, self.labels_ref = self.prepare_data()
    
    def prepare_data(self):
        
        num_data = sum([len(value) for value in self.embeddings.values()])
        emb2sent = {}
        emb2id = {}
        labels_ref = {}
        data = np.zeros((num_data, 768))
        
        number = 0
        for key, value in self.embeddings.items():
            for (sent, emb) in value:
                emb2sent[tuple(emb)] = sent
                emb2id[tuple(emb)] = key
                data[number] = emb
                labels_ref[number] = key
                number += 1
        
        return emb2sent, emb2id, data, labels_ref
    
    def random_split(self, ratio):
        
        indices = np.random.permutation(len(self.data))
        train_size = int(ratio*len(self.data))
        x_train = self.data[indices[:train_size],:][:,:,np.newaxis]
        x_test = self.data[indices[train_size:],:][:,:,np.newaxis]
        
        return (x_train, x_test)


def train(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)
    
    cluster = PerformClustering(opt.dic_path, opt.embedding_path)
    data = cluster.random_split(0.8)
    
    print("1. Get data ready!")

    model = DCEC(opt.input_shape, opt.filters, opt.kernel_size, opt.n_clusters, opt.weights, data, opt.alpha, pretrain=False)
    model.compile(loss=['kld', 'binary_crossentropy'], optimizer='adam')
    print("3. Compile model!")
    
    model.fit(data, opt)

    with open('labels.pkl', 'wb') as f:
        pickle.dump(model.cur_label, f)
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
    


def test(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)

    cluster = PerformClustering(opt.dic_path, opt.embedding_path)

    with open('labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)

    x_train, x_test = data
    x_test = x_test.squeeze(-1)

    print('Cluster Number:', opt.cluster_id)

    for emb in x_test[np.where(labels == opt.cluster_id)]:
        sent = cluster.emb2sent[tuple(emb.tolist())]
        idd = cluster.emb2id[tuple(emb.tolist())]
        print('{}: {}'.format(idd, sent))
        

if __name__ == '__main__':
    import fire
    fire.Fire()

    
    



