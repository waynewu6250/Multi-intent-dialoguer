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

    def __init__(self, data_path, rawdata_path, intent2id_path, done=True):

        super(MULTIWOZData, self).__init__(data_path, rawdata_path, intent2id_path)
        #self.df, self.data, self.labels = self.prepare_text(done)
        self.train_data, self.intent2id = self.prepare_dialogue(done)
        self.num_labels = len(self.intent2id)

    #==================================================#
    #                   Prepare Text                   #
    #==================================================#

    def text_prepare(self, text, mode):
        """
            text: a string       
            return: modified string
        """
        def prepare(text):
            text = text.lower() # lowercase text
            text = re.sub(self.REPLACE_BY_SPACE_RE, ' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
            text = re.sub(self.BAD_SYMBOLS_RE, '', text) # delete symbols which are in BAD_SYMBOLS_RE from text
            text = re.sub(r"[ ]+", " ", text)
            text = re.sub(r"\!+", "!", text)
            text = re.sub(r"\,+", ",", text)
            text = re.sub(r"\?+", "?", text)
            return text
        
        texts = text.split('[SEP]')
        text = "[SEP]".join([prepare(text) for text in texts])
        
        if mode == "Bert":
            text = "[CLS] " + text + " [SEP]"
            tokenized_text = self.tokenizer.tokenize(text)
            tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            text = tokenized_ids
        return text
    
    def prepare_dialogue(self, done):
        """
        train_data:
        
        a list of dialogues
        for each dialogue:
            [(sent1, [label1, label2]), 
             (sent2, [label2]),...]
        """

        if done:
            with open(self.rawdata_path, "rb") as f:
                train_data = pickle.load(f)
            with open(self.intent2id_path, "rb") as f:
                intent2id = pickle.load(f)
            return train_data, intent2id
        
        ptime = time.time()

        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        dialogues = []
        for key, value in data.items():
            
            topic = []
            for k, v in value["goal"].items():
                if len(v) > 0 and k not in ['message', 'topic']:
                    topic.append(k)
            
            # dialogue
            dialogue = []
            for dic in value["log"]:
                labels = []
                for key, value in dic['metadata'].items():
                    if len("".join(list(value['semi'].values()))) != 0:
                        labels.append(key)
                dialogue.append((dic['text'], labels))
            dialogues.append((topic, dialogue))
        
        train_data = []
        intent2id = {}
        counter = 0
        yes = 0
        for topic, texts in dialogues:
            data=[]
            prev_text = ''
            prev_col = []
            for text, col in texts:
                full_text = prev_text+' [SEP] '+text
                full_col = prev_col+col
                prev_text = text
                prev_col = col
                
                if not col:
                    full_col += topic
                
                # set up intent2id
                full_col = set(full_col)
                for col in full_col:
                    if col not in intent2id:
                        intent2id[col] = counter
                        counter += 1
                
                data.append((self.text_prepare(full_text,'Bert'), [intent2id[col] for col in full_col]))
            train_data.append(data[1:])
            print(yes)
            yes += 1
        
        with open(self.rawdata_path, "wb") as f:
            pickle.dump(train_data, f)
        with open(self.intent2id_path, "wb") as f:
            pickle.dump(intent2id, f)
        
        print("Process time: ", time.time()-ptime)
        
        return train_data, intent2id
        

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
    data_path = "../raw_datasets/MULTIWOZ2.1/data.json"
    rawdata_path = "MULTIWOZ2.1/dialogue_data.pkl"
    intent2id_path = "MULTIWOZ2.1/intent2id.pkl"
    data = MULTIWOZData(data_path, rawdata_path, intent2id_path, done=True)
    print(data.train_data[0])