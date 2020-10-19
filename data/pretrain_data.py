import torch as t
from torch.autograd import Variable
import numpy as np
import pandas as pd
import re
import pickle
import h5py
import json
import os
import csv
import spacy
from train_data import Data
import time 

class PretrainData(Data):

    def __init__(self, data_path, neg_path, rawdata_path, done=True):

        super(PretrainData, self).__init__(data_path, rawdata_path, None)
        self.train_data = self.prepare_pretrain(done, neg_path)
    
    def prepare_pretrain(self, done, neg_path):
        """
        train_data:
        
        a list of dialogues
        for each dialogue:
            [(sent1+sent2, 0/1), 
             (sent2+sent3, 0/1),...]
        """

        if done:
            with open(self.rawdata_path, "rb") as f:
                train_data = pickle.load(f)
            return train_data
        
        ptime = time.time()

        with open(self.data_path, "rb") as f:
            train_data = pickle.load(f)

        # dialogue corpus
        with open(neg_path, 'rb') as f:
            other_data = pickle.load(f)
        texts = []
        for data in train_data:
            for (text, label, _) in data:
                texts.append(text)
        
        all_data = []
        counter = 0
        for data in train_data:
            prev_text = None
            for (text, label, _) in data:
                if prev_text:
                    print('parsed {} data'.format(counter))
                    if np.random.rand(1)[0] < 0.7:
                        # positive sample
                        encoded = self.tokenizer.encode_plus(prev_text, text_pair=text, return_tensors='pt')
                        label = 1
                    else:
                        # negative sample
                        idx = np.random.choice(len(texts), 1)
                        encoded = self.tokenizer.encode_plus(texts[idx[0]], text_pair=text, return_tensors='pt')
                        label = 0
                    sample = (encoded['input_ids'], encoded['token_type_ids'], encoded['attention_mask'], label)
                    all_data.append(sample)
                    counter += 1

                prev_text = text
                
        
        with open(self.rawdata_path, "wb") as f:
            pickle.dump(all_data, f)
        
        print("Process time: ", time.time()-ptime)
        
        return all_data
    
    
if __name__ == "__main__":
    # e2e dataset
    # data_path = "e2e_dialogue/dialogue_data_raw.pkl"
    # neg_path = "sgd_dialogue/dialogue_data_raw.pkl"
    # rawdata_path = "e2e_dialogue/dialogue_data_pretrain.pkl"

    # sgd dataset
    data_path = "sgd_dialogue/dialogue_data_raw.pkl"
    neg_path = "e2e_dialogue/dialogue_data_raw.pkl"
    rawdata_path = "sgd_dialogue/dialogue_data_pretrain.pkl"
    
    data = PretrainData(data_path, neg_path, rawdata_path, done=False)
    for i,j,k,l in data.train_data[:10]:
        print(i)
        print(l)