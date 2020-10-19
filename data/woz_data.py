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

    def __init__(self, data_path, turn_path, rawdata_path, done=True):

        super(MULTIWOZData, self).__init__(data_path, rawdata_path, None)
        self.turn_path = turn_path
        #self.df, self.data, self.labels = self.prepare_text(done)
        self.turn_data_all = self.prepare_dialogue(done)
        self.anum_labels = len(self.turn_data_all['aintent2id'])
        self.snum_labels = len(self.turn_data_all['slot2id'])
        self.vnum_labels = len(self.turn_data_all['value2id'])

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
    
    def build_ids(self, items, item2id, counter):
        for item in items:
            if item not in item2id:
                item2id[item] = (counter, self.text_prepare(item, 'Bert')) # counter
                counter += 1
        items = [item2id[item][0] for item in items]
        return items, item2id, counter
    
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
                turn_data_all = pickle.load(f)
            return turn_data_all
        
        ptime = time.time()

        aintent2id = {}
        acounter = 0
        slot2id = {}
        scounter = 0
        value2id = {}
        vcounter = 0

        with open(self.data_path, 'r') as f:
            data = json.load(f)
        with open(self.turn_path, 'r') as f:
            slot_values = json.load(f)
        
        dialogues = []
        for key, value in data.items():
            print('Parsing ', key)
            
            topic = []
            for k, v in value["goal"].items():
                if len(v) > 0 and k not in ['message', 'topic']:
                    topic.append(k)
            
            # dialogue
            svs = slot_values[key.split('.')[0]]
            if len(value['log'])//2 != len(svs):
                continue
            
            dialogue = []
            turn_num = 1
            for i, dic in enumerate(value["log"]):
                if (i+1) % 2 == 0:
                    texts = self.tokenizer.encode_plus(prev_text, text_pair=dic['text'], return_tensors='pt')

                    slot_pairs = []
                    if svs[str(turn_num)] != 'No Annotation':
                        for k,v in svs[str(turn_num)].items():
                            for pair in v:
                                slot_pairs.append(tuple([k]+pair))
                        aintents, slots, values = zip(*slot_pairs)
                        aintents, aintent2id, acounter = self.build_ids(aintents, aintent2id, acounter)
                        slots, slot2id, scounter = self.build_ids(slots, slot2id, scounter)
                        values, value2id, vcounter = self.build_ids(values, value2id, vcounter)
                        slot_pairs = list(zip(aintents, slots, values))

                    dialogue.append((texts['input_ids'], slot_pairs))
                    turn_num += 1
                
                prev_text = dic['text']
                
            dialogues.append((topic, dialogue))
        
        turn_data_all = {'turns': dialogues,
                         'aintent2id': aintent2id,
                         'slot2id': slot2id,
                         'value2id': value2id}
        with open(self.rawdata_path, "wb") as f:
            pickle.dump(turn_data_all, f)
        
        
        print("Process time: ", time.time()-ptime)
        
        return turn_data_all
        

    def prepare_text(self, done):
        """Depreciated.
        """

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
    turn_path = "../raw_datasets/MULTIWOZ2.1/dialogue_acts.json"
    rawdata_path = "MULTIWOZ2.1/turns.pkl"
    data = MULTIWOZData(data_path, turn_path, rawdata_path, done=False)
    print(data.turn_data_all['turns'][0])