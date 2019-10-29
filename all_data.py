import torch as t
from torch.utils.data import Dataset, DataLoader
import pickle
from config import opt
from sklearn.model_selection import train_test_split
import numpy as np

class ATISDataset(Dataset):

    def __init__(self, data, labels, masks, opt):
        self.data = data
        self.labels = labels
        self.masks = masks
        self.num_data = len(self.data)
        self.maxlen = opt.maxlen

    def __getitem__(self, index):

        # caps
        caps = self.data[index]
        caps = t.tensor(caps)

        # labels
        label = self.labels[index]
        labels = [t.from_numpy(np.array(label))]

        # masks
        masks = t.tensor(self.masks[index])

        return caps, labels[0], masks

    def __len__(self):
        return len(self.data)

def get_dataloader(data, labels, masks, opt):
    dataset = ATISDataset(data, labels, masks, opt)
    return DataLoader(dataset, 
                      batch_size=opt.batch_size, 
                      shuffle=False)

if __name__ == '__main__':
    
    with open(opt.data_path, 'rb') as f:
        data = pickle.load(f)
    X, y, entities = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    
    