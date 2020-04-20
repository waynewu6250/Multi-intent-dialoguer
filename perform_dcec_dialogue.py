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
        
        num_data = len(self.embeddings)
        emb2sent = {}
        emb2id = {}
        labels_ref = {}
        data = np.zeros((num_data, 768))
        attdata = np.zeros((num_data, 25, 768))
        lengths = np.zeros((num_data, 1))
        
        number = 0
        for original_sentence, embedding, word_embeddings, key in self.embeddings:
            
            emb2sent[tuple(embedding)] = original_sentence
            emb2id[tuple(embedding)] = key
            
            data[number] = embedding
            attdata[number] = word_embeddings
            lengths[number] = 25-len(original_sentence.split(" "))
            
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
    
    cluster = PerformClustering(opt.woz_dic_path, opt.woz_embedding_path)
    data = cluster.random_split(0.8)
    emb_train, emb_test, att_train, att_test, l_train, l_test = data
    
    print("1. Get data ready!")

    model = DCEC((25, 768), opt.filters, opt.kernel_size, opt.n_clusters, opt.weights, data, opt.alpha, pretrain=True)
    model.compile(loss='kld', optimizer='adam')
    print("3. Compile model!")
    
    model.fit(data, opt)

    emb_train, emb_test, att_train, att_test, l_train, l_test = data

    # 1. Attention weights
    test_func = K.function(model.model.input, model.model.get_layer(name='att_weights').output)
    att_weights = test_func([cluster.attdata, cluster.lengths])

    # 2. Cluster center
    test_func = K.function(model.model.input, model.model.get_layer(name='cluster').weights)
    cluster_centers = test_func([cluster.attdata, cluster.lengths])
    
    q = model.model.predict([cluster.attdata, cluster.lengths])
    cur_label = np.argmax(q, axis = 1)

    with open(opt.cluster_label_path, 'wb') as f:
        pickle.dump(cur_label, f)
    with open(opt.cluster_data_path, 'wb') as f:
        pickle.dump(data, f)
    with open(opt.cluster_weight_path, 'wb') as f:
        pickle.dump(att_weights, f)
    


def test(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)

    cluster = PerformClustering(opt.woz_dic_path, opt.woz_embedding_path)

    with open(opt.cluster_data_path, 'rb') as f:
        data = pickle.load(f)
    with open(opt.cluster_label_path, 'rb') as f:
        labels = pickle.load(f)
    with open(opt.cluster_weight_path, 'rb') as f:
        att_weights = pickle.load(f)
    with open(opt.woz_dialogue_id_path, 'rb') as f:
        dialogue_id = pickle.load(f)
    
    emb_train, emb_test, att_train, att_test, l_train, l_test = data
    att_weights = att_weights.squeeze(-1)

    ########################## dialogue id ##########################
    # USE ALL DATA
    with open('clustering_results/result_ft_woz_dialogue.txt', 'w') as f:

        for i, (emb, weights, pred_label) in enumerate(zip(cluster.data, att_weights, labels)):
            
            index = np.argsort(weights)[-3:]

            real_label = cluster.emb2id[tuple(emb.tolist())]
            sent = cluster.emb2sent[tuple(emb.tolist())]

            toks = sent.split(' ')
            toks = np.array(toks+['<PAD>']*(25-len(toks)))

            f.write("Dialogue Num: {} ||| Ground Truth {}, Predicted Label {}, Attention Words: {} | {} \
                     \n".format(dialogue_id[i], real_label, pred_label, " ".join(toks[index][::-1]), sent))
    ########################## dialogue id ##########################

    ########################## calculate sentences in same clusters ##########################
    # # USE TEST DATA
    # cache = collections.defaultdict(list)
    # for cluster_id in range(opt.n_clusters):
    #     idds = []
    #     sents = []
        
    #     # Calculate original ids
    #     if cluster_id not in labels:
    #         continue
        
    #     chosen = np.where(labels == cluster_id)
    #     for emb, weights in zip(emb_test[chosen], att_weights[chosen]):
            
    #         index = np.argsort(weights)[-3:]

    #         idd = cluster.emb2id[tuple(emb.tolist())]
    #         sent = cluster.emb2sent[tuple(emb.tolist())]

    #         toks = sent.split(' ')
    #         toks = np.array(toks+['<PAD>']*(25-len(toks)))

    #         idds += list(idd)
    #         sents.append((idd,sent,toks[index]))
        
    #     # Cluster:
    #     uid = np.unique(idds)
    #     lengths = [len(np.where(idds == uid[i])) for i in range(len(uid))]
    #     cache[uid[np.argmax(lengths)]].append(sents)

    # cache = sorted(cache.items(), key = lambda x: x[0])

    # with open('clustering_results/result_ft_woz.txt', 'w') as f:
    #     for key, value in cache:
    #         f.write("-"*15)
    #         f.write("\n Original Label: {} \n".format(key))
    #         for i, sents in enumerate(value):
    #             f.write("*"*5+"Mini-cluster {}".format(i)+"*"*5+"\n")
    #             for idd, sent, words in sents:
    #                 f.write("Ground Truth: {}, Attention Words: {} | {}\n".format(idd, " ".join(words[::-1]), sent))
    #         f.write("-"*15+"\n\n")
    ########################## calculate sentences in same clusters ##########################


        

if __name__ == '__main__':
    import fire
    fire.Fire()

    
    



