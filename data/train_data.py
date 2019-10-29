import torch as t
from torch.autograd import Variable
import numpy as np
import re
import pickle
import h5py
import json
import os
import csv
import spacy
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import time

class Data:

    def __init__(self, data_path):

        self.data_path = data_path
        self.REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')

    #==================================================#
    #                   Text Prepare                   #
    #==================================================#
    
    #pure virtual function
    def prepare_text(self):
        raise NotImplementedError("Please define virtual function!!")

    # prepare text
    def text_prepare(self, text, mode):
        """
            text: a string       
            return: modified string
        """
        
        text = text.lower() # lowercase text
        text = re.sub(self.REPLACE_BY_SPACE_RE, ' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = re.sub(self.BAD_SYMBOLS_RE, '', text) # delete symbols which are in BAD_SYMBOLS_RE from text
        text = re.sub(r"[ ]+", " ", text)
        text = re.sub(r"\!+", "!", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"\?+", "?", text)
        if mode == "Bert":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        
        return text
    

class ATISData(Data):

    def __init__(self, data_path, mode, input_path=None, embedding_path=None, done=True):

        super(ATISData, self).__init__(data_path)
        
        self.raw_data, self.intent2id = self.prepare_text(mode, done)
        self.intents = [data[1] for data in self.raw_data]
        self.num_labels = len(self.intent2id)

        if mode == "Starspace":
            # Run the following to get starspace embedding
            # > ./starspace train -trainFile data.txt -model modelSaveFile -label '#'
            self.embedding_path = embedding_path
            self.input_path = input_path
            self.write_files()
            self.load_embeddings()
        
        if mode == "Bert":
            pass


    #==================================================#
    #                   Prepare Text                   #
    #==================================================#
    
    def prepare_text(self, mode, done):

        if done:
            with open("raw_data.pkl", "rb") as f:
                raw_data = pickle.load(f)
            with open("intent2id.pkl", "rb") as f:
                intent2id = pickle.load(f)
            return raw_data, intent2id

        ptime = time.time()

        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        raw_data = []
        intent2id = {}
        counter = 0
        for sample in data['rasa_nlu_data']['common_examples']:
            if sample['intent'] not in intent2id:
                intent2id[sample['intent']] = counter
                counter += 1
            raw_data.append((self.text_prepare(sample['text'], mode), intent2id[sample['intent']], sample['entities']))
        
        with open("raw_data.pkl", "wb") as f:
            pickle.dump(raw_data, f)
        with open("intent2id.pkl", "wb") as f:
            pickle.dump(intent2id, f)
        
        print("Process time: ", time.time()-ptime)
        
        return raw_data, intent2id
    
    #==================================================#
    #                       Bert                       #
    #==================================================#

    
    
    
    #==================================================#
    #                    Starspace                     #
    #==================================================#

    def write_files(self):
        if mode == "yes":
            with open(self.input_path, 'w') as f:
                for text, intent, _ in self.raw_data:
                    f.write(text+" __label__{}".format(intent)+"\n")

    def load_embeddings(self):
        
        # Load embeddings
        self.word_embeddings = {}
        with open(self.embedding_path) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                self.word_embeddings[row[0]] = [float(i) for i in row[1:]]
        
        # Embed the texts into sentence embeddings
        self.embedded_data = np.zeros((len(self.raw_data), 100))
        for i, data in enumerate(self.raw_data):
            self.embedded_data[i,:] = np.mean([self.word_embeddings[txt] for txt in data[0].split()], axis=0)
    

if __name__ == "__main__":
    data = ATISData("../raw_datasets/ATIS/test.json", "Bert")





        




