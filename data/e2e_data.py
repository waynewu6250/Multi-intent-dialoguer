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

class E2EData(Data):

    def __init__(self, data_path, rawdata_path, intent2id_path, done=True):

        super(E2EData, self).__init__(data_path, rawdata_path, intent2id_path)
        self.train_data, self.intent2id = self.prepare_dialogue(done)
        self.num_labels = len(self.intent2id)
    
    def prepare(self, data_path, intent2id, counter):

        print('Parsing file: ', data_path)

        all_data = []
        data = []
        prev_id = '1'

        with open(self.data_path+data_path, 'r') as f:
        
            for i, line in enumerate(f):
                if i == 0:
                    continue
                
                infos = line.split('\t')
                dialogue_id = infos[0]
                message_id = infos[1]
                speaker = infos[3]
                text = infos[4]
                intents = []
                slots = []
                for act in infos[5:]:
                    if act[:act.find('(')] != '':
                        intents.append(act[:act.find('(')])
                    s = re.findall('\((.*)\)', act)
                    if s:
                        slots.append(s[0].split(';'))
                
                # single intent
                intents = "@".join(sorted(intents))
                if intents not in intent2id:
                    intent2id[intents] = counter
                    counter += 1
                intents = intent2id[intents]
                
                # multi intents
                # for intent in intents:
                #     if intent not in intent2id:
                #         intent2id[intent] = counter
                #         counter += 1
                # intents = [intent2id[intent] for intent in intents]
                
                if data and prev_id != dialogue_id:
                    all_data.append(data)
                    data = []
                    prev_id = dialogue_id
                
                data.append((self.text_prepare(text, 'Bert'), intents, slots))
        
        return all_data, counter
    
    def prepare_dialogue(self, done):
        """
        train_data:
        
        a list of dialogues
        for each dialogue:
            [(sent1, [label1, label2], [slot1, slot2]), 
             (sent2, [label2], [slot2]),...]
        """

        if done:
            with open(self.rawdata_path, "rb") as f:
                train_data = pickle.load(f)
            with open(self.intent2id_path, "rb") as f:
                intent2id = pickle.load(f)
            return train_data, intent2id
        
        ptime = time.time()

        # if os.path.exists(self.intent2id_path):
        #     with open(self.intent2id_path, "rb") as f:
        #         intent2id = pickle.load(f)
        #     counter = len(intent2id)
        # else:
        intent2id = {}
        counter = 0
        
        all_data = []
        for data_path in os.listdir(self.data_path):
            data, counter = self.prepare(data_path, intent2id, counter)
            all_data += data
        
        with open(self.rawdata_path, "wb") as f:
            pickle.dump(all_data, f)
        with open(self.intent2id_path, "wb") as f:
            pickle.dump(intent2id, f)
        
        print("Process time: ", time.time()-ptime)
        
        return all_data, intent2id
    
    
if __name__ == "__main__":
    data_path = "../raw_datasets/e2e_dialogue/"
    rawdata_path = "e2e_dialogue/dialogue_data.pkl"
    intent2id_path = "e2e_dialogue/intent2id.pkl"

    data = E2EData(data_path, rawdata_path, intent2id_path, done=False)
    print(data.train_data[100])
    print(data.intent2id)