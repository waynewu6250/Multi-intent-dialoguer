import csv
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import BertTokenizer
import pickle
import numpy as np
np.random.seed(0)

PAD = "__PAD__"
UNK = "__UNK__"
START = "__START__"
END = "__END__"

class ClusterDataset(Dataset):
    def __init__(self, rawdata_path, intent2id_path, fname, view1_col='view1_col', view2_col='view2_col', label_col='cluster_id', done=True, max_sent=10, train=True):
        """
        Args:
            fname: str, training data file
            view1_col: str, the column corresponding to view 1 input
            view2_col: str, the column corresponding to view 2 input
            label_col: str, the column corresponding to label
        """
        self.rawdata_path = rawdata_path
        self.intent2id_path = intent2id_path
        self.max_sent = max_sent

        if done:
            with open(self.rawdata_path, "rb") as f:
                final_data = pickle.load(f)
                self.data = final_data['data']
                self.labels = final_data['labels']
                if train:
                    self.data = self.data[:int(0.9*len(self.data))]
                    self.labels = self.labels[:int(0.9*len(self.labels))]
                else:
                    self.data = self.data[int(0.9*len(self.data)):]
                    self.labels = self.labels[int(0.9*len(self.labels)):]
            with open(self.intent2id_path, "rb") as f:
                self.label_to_id = pickle.load(f)
        else:
            self.preprocess()
    
    def preprocess(self):
        """
        Preprocess data
        """
        def tokens_to_idices(tokens):
            text = " ".join(["[CLS]"] + tokens + ["[SEP]"])
            tokenized_text = self.tokenizer.tokenize(text)
            token_idices = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            return token_idices

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        id_to_label = [UNK]
        label_to_id = {UNK: 0}
        data = []
        labels = []
        v1_utts = []  # needed for displaying cluster samples
        self.trn_idx, self.tst_idx = [], []
        self.trn_idx_no_unk = []
        
        with open(fname, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                view1_text, view2_text = row[view1_col], row[view2_col]
                label = row[label_col]
                if 'UNK' == label:
                    label = UNK
                if '<cust_' not in view1_text:
                    view2_sents = sent_tokenize(view2_text.lower())
                else:
                    view2_sents = view2_text.split("> <")
                    for i in range(len(view2_sents) - 1):
                        view2_sents[i] = view2_sents[i] + '>'
                        view2_sents[i+1] = '<' + view2_sents[i + 1]
                v1_utts.append(view1_text)
                
                v1_tokens = view1_text.lower().split()
                v2_tokens = [sent.lower().split() for sent in view2_sents]
                v2_tokens = v2_tokens[:self.max_sent]

                v1_token_idices = tokens_to_idices(v1_tokens)
                v2_token_idices = [tokens_to_idices(tokens) for tokens in v2_tokens]
                v2_token_idices = [idices for idices in v2_token_idices if len(idices) > 0]
                if len(v1_token_idices) == 0 or len(v2_token_idices) == 0:
                    continue
                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id)
                    id_to_label.append(label)
                
                data.append((v1_token_idices, v2_token_idices))
                labels.append(label_to_id[label])

                if label == UNK and np.random.random_sample() < .1:
                    self.tst_idx.append(len(data)-1)
                else:
                    self.trn_idx.append(len(data)-1)
                    if label != UNK:
                        self.trn_idx_no_unk.append(len(data) - 1)
        
        self.v1_utts = v1_utts
        self.id_to_label = id_to_label
        self.label_to_id = label_to_id
        self.data = data
        self.labels = labels

        final_data = {'data': data, 'labels':labels}
        with open(self.rawdata_path, "wb") as f:
            pickle.dump(final_data, f)
        with open(self.intent2id_path, "wb") as f:
            pickle.dump(label_to_id, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]

def get_dataloader(rawdata_path, intent2id_path, fname, view1_col='view1_col', view2_col='view2_col', label_col='cluster_id'):
    dataset = ClusterDataset(rawdata_path, intent2id_path, fname, view1_col, view2_col, label_col, done=True)
    batch_size = opt.dialog_batch_size if opt.dialog_data_mode else opt.batch_size
    return DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=False)

if __name__ == '__main__':
    # data: [(u1, [u2, u3, ..., un]), ...]
    # label: [l1, l2, ..., ln]

    # twitter
    # rawdata_path = 'cluster/rawdata.pkl'
    # intent2id_path = 'cluster/intent2id.pkl'
    # fname = '../raw_datasets/intent_cluster/twitter_air/airlines_processed.csv'
    # view1_col = 'first_utterance'
    # view2_col = 'context'
    # label_col = 'tag'

    # askubuntu
    rawdata_path = 'cluster/rawdata_ubuntu.pkl'
    intent2id_path = 'cluster/intent2id_ubuntu.pkl'
    fname = '../raw_datasets/intent_cluster/ubuntu/askubuntu_processed.csv'
    view1_col = 'view1'
    view2_col = 'view2'
    label_col = 'label'

    dataset = ClusterDataset(rawdata_path, intent2id_path, fname, view1_col, view2_col, label_col, done=True)
    count = 0
    for label in dataset.labels:
        if label != 0:
            count += 1
    print(count)


