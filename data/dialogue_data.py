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

class MULTIWOZData(Data):

    def __init__(self, data_path, done=True):

        super(MULTIWOZData, self).__init__(data_path)
        self.df, self.data, self.labels = self.prepare_text(done)

    #==================================================#
    #                   Prepare Text                   #
    #==================================================#
    
    def prepare_text(self, done):

        if done:
            with open("MULTIWOZ2.1/raw_table.pkl", "rb") as f:
                raw_table = pickle.load(f)
            with open("MULTIWOZ2.1/raw_data.pkl", "rb") as f:
                data = pickle.load(f)
            with open("MULTIWOZ2.1/raw_labels.pkl", "rb") as f:
                labels = pickle.load(f)
            return raw_table, data, labels

        ptime = time.time()

        def check(item):
            if item not in value['goal'] or len(value['goal'][item])==0:
                return 0
            else: return value['goal'][item]
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(columns=['filename', 'text', 'taxi', 'police', 'hospital', 'hotel', 'topic', 'attraction', 'train', 'message', 'restaurant'])
        
        for key, value in data.items():
            texts = " | ".join([dic['text'] for dic in value["log"]])
            element = pd.DataFrame([{'filename': key,
                                    'text': texts,
                                    'taxi': check('taxi'),
                                    'police': check('police'),
                                    'hospital': check('hospital'),
                                    'hotel': check('hotel'),
                                    'topic': check('topic'),
                                    'attraction': check('attraction'),
                                    'train': check('train'),
                                    'message': check('message'),
                                    'restaurant': check('restaurant'),
                                }])
            df = df.append(element)
        
        # Prepare texts
        texts = [(filename, text.split(' | ')) for filename, text in zip(df.filename, df.text)]

        # Prepare labels
        subdf = df[['filename', 'taxi', 'police', 'hospital', 'hotel', 'attraction', 'train', 'restaurant']]
        labels = {}
        for i in range(len(subdf)):
            label = []
            for col in subdf.iloc[i].keys():
                if subdf.iloc[i][col] and col != 'filename':
                    label.append(col)
            labels[subdf.iloc[i]['filename']] = label
        
        with open("MULTIWOZ2.1/raw_table.pkl", "wb") as f:
            pickle.dump(df, f)
        with open("MULTIWOZ2.1/raw_data.pkl", "wb") as f:
            pickle.dump(texts, f)
        with open("MULTIWOZ2.1/raw_labels.pkl", "wb") as f:
            pickle.dump(labels, f)
        
        print("Process time: ", time.time()-ptime)
        
        return df, texts, labels
    
    
if __name__ == "__main__":
    data = MULTIWOZData("../raw_datasets/MULTIWOZ2.1/data.json", done=True)