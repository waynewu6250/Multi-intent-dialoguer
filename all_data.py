import torch as t
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.model_selection import train_test_split
from config import opt
import numpy as np

class ATISDataset(Dataset):

    def __init__(self, data, labels, opt):
        self.data = data
        self.labels = labels
        self.num_data = len(self.data)
        self.maxlen = opt.maxlen

    def __getitem__(self, index):

        caps = self.data[index]
        label = self.labels[index]

        if len(caps) > self.maxlen:
            caps = caps[:self.maxlen]
        padding = [0] * (self.maxlen - len(caps))
        caps += padding

        assert len(caps) == self.maxlen
        caps = t.tensor(caps)

        labels = [t.from_numpy(np.array(label))]

        return caps, labels[0], index
    
    def __len__(self):
        return len(self.data)

def get_dataloader(data, labels, opt):
    dataset = ATISDataset(data, labels, opt)
    return DataLoader(dataset, 
                      batch_size=opt.batch_size, 
                      shuffle=False)

if __name__ == '__main__':
    
    with open(opt.data_path, 'rb') as f:
        data = pickle.load(f)
    X, y, entities = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    trainloader = get_dataloader(X_train, y_train, opt)
    
    