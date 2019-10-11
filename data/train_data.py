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

class Data:

    def __init__(self, data_path, embedding_path, mode):

        self.data_path = data_path
        self.embedding_path = embedding_path
        self.REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
        self.raw_data = self.prepare_text(mode)
        self.intents = [data[1] for data in self.raw_data]
        self.num_labels = len(set(self.intents))
        
        # Run the following to get starspace embedding
        # > ./starspace train -trainFile data.txt -model modelSaveFile -label '#'
        
        #self.load_embeddings()

    #==================================================#
    #                   Text Prepare                   #
    #==================================================#

    def text_prepare(self, text):
        """
            text: a string       
            return: modified string
        """
        #nlp = spacy.load('en')
        text = text.lower() # lowercase text
        text = re.sub(self.REPLACE_BY_SPACE_RE, ' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = re.sub(self.BAD_SYMBOLS_RE, '', text) # delete symbols which are in BAD_SYMBOLS_RE from text
        text = re.sub(r"[ ]+", " ", text)
        text = re.sub(r"\!+", "!", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"\?+", "?", text)
        #text = " ".join(nlp.tokenizer(text).text)
        
        return text
    
    #==================================================#
    #            Prepare Text for Starspace            #
    #==================================================#
    
    def prepare_text(self, mode="no"):

        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        raw_data = []
        self.word2id = {}
        counter = 0
        for sample in data['rasa_nlu_data']['common_examples']:
            if sample['intent'] not in self.word2id:
                self.word2id[sample['intent']] = counter
                counter += 1
            raw_data.append((self.text_prepare(sample['text']), self.word2id[sample['intent']], sample['entities']))

        if mode == "yes":
            with open('val.txt', 'w') as f:
                for text, intent, _ in raw_data:
                    f.write(text+" __label__{}".format(intent)+"\n")
        
        return raw_data
    
    #==================================================#
    #                 Load Embeddings                  #
    #==================================================#

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
    data = Data("../raw_datasets/ATIS/test.json", "modelSaveFile.tsv", "yes")


    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=22, random_state=0, init='k-means++').fit(data.embedded_data)
    # print(kmeans.labels_)





        




