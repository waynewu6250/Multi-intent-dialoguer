import torch
import torch.nn as nn
from config import opt
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors

def perform():
    
    with open(opt.se_dic_path, 'rb') as f:
        dic = pickle.load(f)
    reverse_dic = {v: k for k,v in dic.items()}

    embeddings = torch.load(opt.se_embedding_path)
    
    sent2id = {}
    emb2id = {}
    num_data = sum([len(value) for value in embeddings.values()])
    data = np.zeros((num_data, 768))
    
    number = 0
    for key, value in embeddings.items():
        for (sent, emb) in value:
            emb2set[tuple(emb)] = sent
            emb2id[tuple(emb)] = key
            data[number] = emb
            number += 1
    
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    

if __name__ == '__main__':
    perform()
    