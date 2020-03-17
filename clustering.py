#%%
import torch
import torch.nn as nn
#from config import opt
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import itertools

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
        if label_id != None:
            self.emb2sent, self.emb2id, self.data, self.labels_ref = self.prepare_data_single(label_id)
        else:
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
    
    def prepare_data_single(self, label_id):

        num_data = len(self.embeddings[label_id])
        data = np.zeros((num_data, 768))
        emb2sent = {}
        emb2id = {}
        labels_ref = {}
        number = 0
        for (sent, emb) in self.embeddings[label_id]:
            emb2sent[tuple(emb)] = sent
            emb2id[tuple(emb)] = 0
            data[number] = emb
            labels_ref[number] = 0
            number += 1

        return emb2sent, emb2id, data, labels_ref
    
    def nearestneighbor(self, n):

        nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        return distances, indices
    
    def accuracy_measure(self, indices):
        correct = 0
        incorrect = 0
        for indice in indices:
            summ = sum([self.labels_ref[i] for i in indice])
            if int((summ / len(indice))) == self.labels_ref[indice[0]]:
                correct += 1
            else:
                incorrect += 1
        print("Accuracy: %4f" % (correct / (correct+incorrect)))
    
    def explore_neighbor(self, indice):
        print()
        for i in indice:
            print(i, ": ", self.emb2sent[tuple(self.data[i])], "\n")
    
    def explore_cluster(self, indices, label_id):
        cooc = np.zeros((len(indices), len(indices)))
        for i in range(len(indices)):
            cooc[i][i] = 1
            for i, j in itertools.combinations(indices[i],2):
                cooc[i][j] = 1
                cooc[j][i] = 1
        
        for i in range(len(indices)):
            if cooc[label_id][i] == 1:
                print(i, ": ", self.emb2sent[tuple(self.data[i])], "\n")


#%%
# if __name__ == "__main__":

dic_path = "/nethome/twu367/Multi-intent-dialoguer/data/semantic/intent2id_se.pkl"
embedding_path = "/nethome/twu367/Multi-intent-dialoguer/results/se_embeddings.pth"

# All neighbor analysis
sentence_id = 5
cluster = PerformClustering(dic_path, embedding_path)
print(len(cluster.data))
distances, indices = cluster.nearestneighbor(50)
cluster.accuracy_measure(indices)
print("Explore the neighbor for sentence: \n{}".format(cluster.emb2sent[tuple(cluster.data[sentence_id])]))
cluster.explore_neighbor(indices[sentence_id])
print()

#%%
# Particular label analysis
label_id = 0
sentence_id = 5
print("Label for {}: {}".format(label_id, cluster.reverse_dic[label_id]))
cluster = PerformClustering(dic_path, embedding_path, label_id=label_id)
print(len(cluster.data))
distances, indices = cluster.nearestneighbor(50)
cluster.accuracy_measure(indices)
cluster.explore_neighbor(indices[sentence_id])
print()

#%%
# Perform cluster analysis
print("Explore the neighbor for sentence: \n{}".format(cluster.emb2sent[tuple(cluster.data[sentence_id])]))
cluster.explore_cluster(indices, label_id)
    




# %%
