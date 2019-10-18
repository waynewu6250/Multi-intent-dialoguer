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

    def __getitem__(self, index):

        caps = self.data[index]
        label = self.labels[index]
        return caps, label, index
    
    def __len__(self):
        return len(self.data)


def get_collate_fn(max_length):
    def collate_fn(imgs_caps):
        
        caps, labels, indexes = zip(*imgs_caps)
        
        # Calculate caption length
        lengths = [min(len(c), max_length) for c in caps]
        batch_length = max(lengths)

        # Captions into tensor
        captions_t = t.LongTensor(len(caps), batch_length).fill_(0)
        for i,cap in enumerate(caps):
            captions_t[i, :lengths[i]].copy_(t.LongTensor(cap))
        
        labels = t.from_numpy(np.array(labels))

        return captions_t, labels, indexes
    return collate_fn

def get_dataloader(data, labels, opt):
    dataset = ATISDataset(data, labels, opt)
    return DataLoader(dataset, 
                      batch_size=opt.batch_size, 
                      shuffle=False, 
                      collate_fn=get_collate_fn(opt.maxlen))

if __name__ == '__main__':
    
    with open(opt.data_path, 'rb') as f:
        data = pickle.load(f)
    X, y, entities = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, )

    trainloader = get_dataloader(X_train, y_train, opt)
    
    