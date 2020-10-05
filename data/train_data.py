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
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import time

class Data:

    def __init__(self, data_path, rawdata_path, intent2id_path):

        self.data_path = data_path
        self.rawdata_path = rawdata_path
        self.intent2id_path = intent2id_path
        self.REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

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
            text = "[CLS] " + text + " [SEP]"
            tokenized_text = self.tokenizer.tokenize(text)
            tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            text = tokenized_ids
        return text
    

############################################################################

class ATISData(Data):

    def __init__(self, data_path, rawdata_path, intent2id_path, mode, input_path=None, embedding_path=None, done=True):

        super(ATISData, self).__init__(data_path, rawdata_path, intent2id_path)
        
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
            with open(self.rawdata_path, "rb") as f:
                raw_data = pickle.load(f)
            with open(self.intent2id_path, "rb") as f:
                intent2id = pickle.load(f)
            return raw_data, intent2id

        ptime = time.time()

        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        raw_data = []
        if os.path.exists(self.intent2id_path):
            with open(self.intent2id_path, "rb") as f:
                intent2id = pickle.load(f)
            counter = len(intent2id)
        else:
            intent2id = {}
            counter = 0
        
        for sample in data['rasa_nlu_data']['common_examples']:
            if sample['intent'] not in intent2id:
                intent2id[sample['intent']] = counter
                counter += 1
            raw_data.append((self.text_prepare(sample['text'], mode), intent2id[sample['intent']], sample['entities']))
        
        with open(self.rawdata_path, "wb") as f:
            pickle.dump(raw_data, f)
        with open(self.intent2id_path, "wb") as f:
            pickle.dump(intent2id, f)
        
        print("Process time: ", time.time()-ptime)
        
        return raw_data, intent2id
    
    #==================================================#
    #                    Starspace                     #
    #==================================================#

    def write_files(self):
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
    


############################################################################


class SemanticData(Data):

    def __init__(self, data_path, rawdata_path, intent2id_path, done=True):

        super(SemanticData, self).__init__(data_path, rawdata_path, intent2id_path)
        self.raw_data, self.intent2id = self.prepare_text(done)
    
    def prepare_text(self, done):

        if done:
            with open(self.rawdata_path, "rb") as f:
                raw_data = pickle.load(f)
            with open(self.intent2id_path, "rb") as f:
                intent2id = pickle.load(f)
            return raw_data, intent2id
        
        data = pd.read_csv(self.data_path, sep='\t', names = ["question", "question2", "info"])
        data["intent"] = data["info"].apply(lambda x: "@".join(set(sorted(re.findall(r'\[IN:(\w+)', x)))))

        ptime = time.time()
        
        raw_data = []
        intent2id = {}
        counter = 0
        
        for i, (text, intents) in enumerate(zip(data["question"].values, data["intent"].values)):
            # single intent:
            # if intent not in intent2id:
            #     intent2id[intent] = counter
            #     counter += 1
            # raw_data.append((self.text_prepare(text, "Bert"), intent2id[intent]))

            # multi intents
            intents = intents.split('@')
            for intent in intents:
                if intent not in intent2id:
                    intent2id[intent] = counter
                    counter += 1
            raw_data.append((self.text_prepare(text, "Bert"), [intent2id[intent] for intent in intents]))
            
            print("Finish: ", i)
        
        with open(self.rawdata_path, "wb") as f:
            pickle.dump(raw_data, f)
        with open(self.intent2id_path, "wb") as f:
            pickle.dump(intent2id, f)
        
        print("Process time: ", time.time()-ptime)
        
        return raw_data, intent2id





if __name__ == "__main__":
    # ATIS
    # data_path = "../raw_datasets/ATIS/train.json"
    # rawdata_path = "atis/raw_data.pkl"
    # intent2id_path = "atis/intent2id.pkl"
    # data = ATISData(data_path, rawdata_path, intent2id_path, "Bert", done=False)
    
    # semantic
    data_path = "../raw_datasets/top-dataset-semantic-parsing/train.tsv"
    rawdata_path = "semantic/raw_data_multi_se.pkl"
    intent2id_path = "semantic/intent2id_multi_se.pkl"
    data = SemanticData(data_path, rawdata_path, intent2id_path, done=False)
    
    #print(data.raw_data)
    print(data.intent2id)





        




