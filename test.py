#%%

import torch
from config import opt
from sklearn.cluster import KMeans
import numpy as np
import pickle

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


print(opt.embedding_path)
cluster = PerformClustering(opt.dic_path, opt.embedding_path)
data = cluster.random_split(0.8)
emb_train, emb_test, att_train, att_test, l_train, l_test = data

kmeans = KMeans(n_clusters=20, random_state=0).fit(emb_test)
kmeans.labels_

true_label = np.array([cluster.emb2id[tuple(emb.tolist())] for emb in emb_test])
with open('clustering_labels/atis_true_kmeans.pkl', 'wb') as f:
    pickle.dump(true_label, f)
with open('clustering_labels/atis_pred_kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans.labels_, f)
print(len(np.unique(true_label)))



# %%
