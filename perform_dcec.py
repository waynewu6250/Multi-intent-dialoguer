import pickle
import numpy as np
import torch
import torch.nn as nn

import keras
from keras.layers import Input, Conv1D, MaxPool1D, Flatten, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model, load_model
import keras.backend as K
import h5py
from sklearn.cluster import KMeans
from sklearn import metrics
import collections

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
        self.emb2sent, self.emb2id, self.data, self.attdata, self.lengths, self.labels_ref = self.prepare_data()
    
    def prepare_data(self):
        
        num_data = sum([len(value) for value in self.embeddings.values()])
        emb2sent = {}
        emb2id = {}
        labels_ref = {}
        data = np.zeros((num_data, 768))
        attdata = np.zeros((num_data, 20, 768))
        lengths = np.zeros((num_data, 1))
        
        number = 0
        for key, value in self.embeddings.items():
            for (sent, emb, word_emb) in value:
                emb2sent[tuple(emb)] = sent
                emb2id[tuple(emb)] = key
                
                data[number] = emb
                attdata[number] = word_emb
                lengths[number] = 20-len(sent.split(" "))
                
                labels_ref[number] = key
                number += 1
        
        return emb2sent, emb2id, data, attdata, lengths.astype(np.int32), labels_ref
    
    def random_split(self, ratio):
        
        indices = np.random.permutation(len(self.data))
        train_size = int(ratio*len(self.data))
        emb_train, emb_test = self.data[indices[:train_size],:], self.data[indices[train_size:],:]
        att_train, att_test = self.attdata[indices[:train_size],:,:], self.attdata[indices[train_size:],:,:]
        l_train, l_test = self.lengths[indices[:train_size]], self.lengths[indices[train_size:]]
        
        return emb_train, emb_test, att_train, att_test, l_train, l_test


def train(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)
    
    cluster = PerformClustering(opt.dic_path, opt.embedding_path)
    data = cluster.random_split(0.8)
    emb_train, emb_test, att_train, att_test, l_train, l_test = data
    
    print("1. Get data ready!")

    model = DCEC(opt.input_shape, opt.filters, opt.kernel_size, opt.n_clusters, opt.weights, data, opt.alpha, pretrain=False)
    model.compile(loss='kld', optimizer='adam')
    print("3. Compile model!")
    
    model.fit(data, opt)

    emb_train, emb_test, att_train, att_test, l_train, l_test = data

    # 1. Attention weights
    test_func = K.function(model.model.input, model.model.get_layer(name='att_weights').output)
    att_weights = test_func([att_test, l_test])

    # 2. Cluster center
    test_func = K.function(model.model.input, model.model.get_layer(name='cluster').weights)
    cluster_centers = test_func([att_test, l_test])
    
    q = model.model.predict([att_test, l_test])
    cur_label = np.argmax(q, axis = 1)

    with open(opt.cluster_label_path, 'wb') as f:
        pickle.dump(cur_label, f)
    with open(opt.cluster_data_path, 'wb') as f:
        pickle.dump(data, f)
    with open(opt.cluster_weight_path, 'wb') as f:
        pickle.dump(att_weights, f)

    
    true_label = np.array([cluster.emb2id[tuple(emb.tolist())] for emb in emb_test])
    with open('clustering_labels/se_true.pkl', 'wb') as f:
        pickle.dump(true_label, f)
    with open('clustering_labels/se_pred.pkl', 'wb') as f:
        pickle.dump(cur_label, f)

    


def test(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)

    cluster = PerformClustering(opt.dic_path, opt.embedding_path)

    with open(opt.cluster_data_path, 'rb') as f:
        data = pickle.load(f)
    with open(opt.cluster_label_path, 'rb') as f:
        labels = pickle.load(f)
    with open(opt.cluster_weight_path, 'rb') as f:
        att_weights = pickle.load(f)
    
    emb_train, emb_test, att_train, att_test, l_train, l_test = data
    att_weights = att_weights.squeeze(-1)

    #print('Cluster Number:', opt.cluster_id)
    cache = collections.defaultdict(list)
    for cluster_id in range(opt.n_clusters):
        idds = []
        sents = []
        
        # Calculate original ids
        if cluster_id not in labels:
            continue
        
        chosen = np.where(labels == cluster_id)
        for emb, weights in zip(emb_test[chosen], att_weights[chosen]):
            
            index = np.argsort(weights)[-3:]
            print(np.sort(weights)[-3:])

            idd = cluster.emb2id[tuple(emb.tolist())]
            sent = cluster.emb2sent[tuple(emb.tolist())]

            toks = sent.split(' ')
            toks = np.array(toks+['<PAD>']*(20-len(toks)))

            idds.append(idd)
            sents.append((idd,sent,toks[index]))
        
        # Cluster:
        uid = np.unique(idds)
        lengths = [len(np.where(idds == uid[i])) for i in range(len(uid))]
        cache[uid[np.argmax(lengths)]].append(sents)

    cache = sorted(cache.items(), key = lambda x: x[0])

    with open('clustering_results/result_atis_att.txt', 'w') as f:
        for key, value in cache:
            f.write("-"*15)
            f.write("\n Original Label: {} \n".format(key))
            for i, sents in enumerate(value):
                f.write("*"*5+"Mini-cluster {}".format(i)+"*"*5+"\n")
                for idd, sent, words in sents:
                    f.write("Ground Truth: {}, Attention Words: {} | {}\n".format(idd, " ".join(words[::-1]), sent))
            f.write("-"*15+"\n\n")
        

if __name__ == '__main__':
    import fire
    fire.Fire()

    
    



