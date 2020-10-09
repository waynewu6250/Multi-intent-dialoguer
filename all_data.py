import torch as t
from torch.utils.data import Dataset, DataLoader
import pickle
from config import opt
from sklearn.model_selection import train_test_split
import numpy as np

class CoreDataset(Dataset):

    def __init__(self, data, labels, masks, num_labels, opt):
        self.data = data
        self.labels = labels
        self.masks = masks
        self.num_data = len(self.data)
        self.maxlen = opt.maxlen
        self.num_labels = num_labels
        self.mode = opt.data_mode

    def __getitem__(self, index):

        # caps
        caps = self.data[index]
        caps = t.tensor(caps)

        # labels
        label = self.labels[index]
        if self.mode == 'single':
            labels = t.from_numpy(np.array(label))
        else:
            label = t.LongTensor(np.array(label))
            labels = t.zeros(self.num_labels).scatter_(0, label, 1)

        # masks
        masks = t.tensor(self.masks[index])

        return caps, labels, masks

    def __len__(self):
        return len(self.data)

def get_dataloader(data, labels, masks, num_labels, opt):
    dataset = CoreDataset(data, labels, masks, num_labels, opt)
    return DataLoader(dataset, 
                      batch_size=opt.batch_size, 
                      shuffle=False)

class DialogueDataset(Dataset):
    
    def __init__(self, data, labels, masks, segs, num_labels, opt):
        self.data = data
        self.labels = labels
        self.masks = masks
        self.segs = segs
        self.num_data = len(self.data)
        self.maxlen = opt.maxlen
        self.num_labels = num_labels

    def __getitem__(self, index):

        # caps
        caps = self.data[index]
        caps = t.tensor(caps)

        # labels
        label = self.labels[index]
        label = t.LongTensor(np.array(label))
        labels = t.zeros(self.num_labels).scatter_(0, label, 1)

        # masks
        masks = t.tensor(self.masks[index])

        # segments
        segs = t.tensor(self.segs[index])

        return caps, labels, masks, segs

    def __len__(self):
        return len(self.data)

def get_dataloader_dialogue(data, labels, masks, segs, num_labels, opt):
    dataset = DialogueDataset(data, labels, masks, segs, num_labels, opt)
    return DataLoader(dataset, 
                      batch_size=opt.batch_size, 
                      shuffle=False)

if __name__ == '__main__':
    
    with open(opt.data_path, 'rb') as f:
        data = pickle.load(f)
    X, y, entities = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    
    