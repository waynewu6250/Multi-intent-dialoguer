import torch as t
from torch.utils.data import Dataset, DataLoader
from config import opt

class ATISDataset(Dataset):

    def __init__(self,opt):
        self.data = t.load(opt.caption_path)
        self.captions_list = self.data['captions_list']
        self.word2ix = self.data['word2ix']
        self.ix2word = self.data['ix2word']
        self.id2ix = self.data['id2ix']
        self.ix2id = self.data['ix2id']

        self.pad = self.word2ix['<PAD>']
        self.unk = self.word2ix['<UNK>']
        self.eos = self.word2ix['<EOS>']

        self.imgs = t.load(opt.img_feature_path)

    def __getitem__(self, index):
        img = self.imgs[index]
        caption = self.captions_list[index]
        return img, caption, index
    
    def __len__(self):
        return len(self.imgs)


def get_collate_fn(pad, eos, max_length=50):
    def collate_fn(imgs_caps):
        # Sort by caption lengths
        imgs_caps = sorted(imgs_caps, key = lambda x: len(x[1]), reverse=True)
        imgs, captions, indexes = zip(*imgs_caps)
        imgs = t.cat([img.unsqueeze(0) for img in imgs], 0)
        
        # Calculate caption length
        lengths = [min(len(c) + 1, max_length) for c in captions]
        batch_length = max(lengths)

        # Captions into tensor
        captions_t = t.LongTensor(batch_length, len(captions)).fill_(pad)
        for i,cap in enumerate(captions):
            end_cap = lengths[i]-1
            if end_cap < batch_length:
                captions_t[end_cap,i] = eos
            captions_t[:end_cap,i].copy_(t.LongTensor(cap))
        
        return imgs, (captions_t, lengths), indexes
    return collate_fn

def get_dataloader(opt):
    dataset = AllDataset(opt)
    return DataLoader(dataset, 
                      batch_size=opt.batch_size, 
                      shuffle=False, 
                      collate_fn=get_collate_fn(dataset.pad,dataset.eos))