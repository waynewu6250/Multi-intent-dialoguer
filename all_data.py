import torch as t
from torch.utils.data import Dataset, DataLoader
import pickle
from config import opt
from sklearn.model_selection import train_test_split
import numpy as np

class CoreDataset(Dataset):

    def __init__(self, data, labels, masks, num_labels, opt, segs=None):
        self.data = data
        self.labels = labels
        self.masks = masks
        self.num_data = len(self.data)
        self.maxlen = opt.maxlen
        self.num_labels = num_labels
        self.mode = opt.data_mode
        self.sentence_mode = opt.sentence_mode
        self.segs = segs

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

        if self.sentence_mode == 'one':
            return caps, labels, masks
        else:
            # segments
            segs = t.tensor(self.segs[index]) 
            return caps, labels, masks, segs

    def __len__(self):
        return len(self.data)

def get_dataloader(data, labels, masks, num_labels, opt, segs=None):
    dataset = CoreDataset(data, labels, masks, num_labels, opt, segs)
    return DataLoader(dataset, 
                      batch_size=opt.batch_size, 
                      shuffle=False)

######################################################################

class DSTDataset(Dataset):

    def __init__(self, x_data, x_masks, y_data, y_masks, y_labels, num_labels, opt):
        self.x_data = x_data
        self.x_masks = x_masks
        self.y_data = y_data
        self.y_masks = y_masks
        self.y_labels = y_labels
        self.num_data = len(self.x_data)
        self.num_labels = num_labels

    def __getitem__(self, index):

        # caps
        x_caps = t.tensor(self.x_data[index])
        
        y_caps_raw = self.y_data[index]
        y_caps = t.zeros(5, 10).long()
        for i, y_cap in enumerate(y_caps_raw):
            y_caps[i] = t.tensor(y_cap)

        # masks
        x_masks = t.tensor(self.x_masks[index])

        y_masks_raw = self.y_data[index]
        y_masks = t.zeros(5, 10).long()
        for i, y_mask in enumerate(y_masks_raw):
            y_masks[i] = t.tensor(y_mask)

        # labels
        label = self.y_labels[index]
        label = t.LongTensor(np.array(label))
        labels = t.zeros(self.num_labels).scatter_(0, label, 1)

        return x_caps, x_masks, y_caps, y_masks, labels

    def __len__(self):
        return len(self.x_data)

def get_dataloader_zsl(x_data, x_masks, y_data, y_masks, y_labels, num_labels, opt):
    dataset = ZSLDataset(x_data, x_masks, y_data, y_masks, y_labels, num_labels, opt)
    return DataLoader(dataset, 
                      batch_size=opt.batch_size, 
                      shuffle=False)

######################################################################

if __name__ == '__main__':
    
    with open(opt.data_path, 'rb') as f:
        data = pickle.load(f)
    X, y, entities = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    
    