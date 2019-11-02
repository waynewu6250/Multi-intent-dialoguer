import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, BertAdam
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import copy
import numpy as np

from model import BertEmbedding
from all_data import get_dataloader
from config import opt

def load_data(path):
    
    with open(path, 'rb') as f:
        data = pickle.load(f)

    X, y, _ = zip(*data)
    input_ids = pad_sequences(X, maxlen=opt.maxlen, dtype="long", truncating="post", padding="post")
    
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return input_ids, y, attention_masks

def train(**kwargs):
    
    # attributes
    for k, v in kwargs.items():
        setattr(opt, k, v)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    # dataset
    with open(opt.dic_path, 'rb') as f:
        dic = pickle.load(f)
    
    X_train, y_train, mask_train = load_data(opt.train_path)
    X_test, y_test, mask_test = load_data(opt.test_path)

    train_loader = get_dataloader(X_train, y_train, mask_train, opt)
    val_loader = get_dataloader(X_test, y_test, mask_test, opt)
    
    # model
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    
    model = BertEmbedding(config, len(dic))
    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))
        print("Pretrained model has been loaded.\n")
    model = model.to(device)

    # optimizer, criterion
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    
    optimizer = BertAdam(optimizer_grouped_parameters,lr=opt.learning_rate_bert, warmup=.1)
    criterion = nn.CrossEntropyLoss().to(device)
    best_loss = 100

    # Start training
    for epoch in range(opt.epochs):
        print("====== epoch %d / %d: ======"% (epoch, opt.epochs))

        # Training Phase
        total_train_loss = 0
        train_corrects = 0
        
        for (captions_t, labels, masks) in train_loader:

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            #train_loss = model(captions_t, masks, labels)

            pooled_output, outputs = model(captions_t, masks)
            train_loss = criterion(outputs, labels)

            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss
            train_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)
        
        print('Total train loss: {:.4f} '.format(total_train_loss / train_loader.dataset.num_data))
        print('Train accuracy: {:.4f}'.format(train_corrects.double() / train_loader.dataset.num_data))

        # Validation Phase
        total_val_loss = 0
        val_corrects = 0
        for (captions_t, labels, masks) in val_loader:

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                pooled_output, outputs = model(captions_t, masks)
            val_loss = criterion(outputs, labels)

            total_val_loss += val_loss
            val_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)

        print('Total val loss: {:.4f} '.format(total_val_loss / val_loader.dataset.num_data))
        print('Val accuracy: {:.4f}'.format(val_corrects.double() / val_loader.dataset.num_data))
        if total_val_loss < best_loss:
            print('saving with loss of {}'.format(total_val_loss),
                  'improved over previous {}'.format(best_loss))
            best_loss = total_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), "checkpoints/epoch-%s.pth"%epoch)
        
        print()

def test(**kwargs):

    # attributes
    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    # dataset
    with open(opt.dic_path, 'rb') as f:
        dic = pickle.load(f)
    reverse_dic = {v: k for k,v in dic.items()}
    X_test, y_test, mask_test = load_data(opt.test_path)
    
    # model
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    
    model = BertEmbedding(config, len(dic))
    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))
        print("Pretrained model has been loaded.\n")
    model = model.to(device)

    # Run classification
    index = np.random.randint(0, len(X_test), 1)[0]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids = X_test[index]
    attention_masks = mask_test[index]
    #print(" ".join(tokenizer.convert_ids_to_tokens(input_ids)))
    
    # User defined
    text = "I miss my luggage"
    print(text)
    text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    tokenized_ids = np.array(tokenizer.convert_tokens_to_ids(tokenized_text))[np.newaxis,:]
    
    input_ids = pad_sequences(tokenized_ids, maxlen=opt.maxlen, dtype="long", truncating="post", padding="post").squeeze(0)
    attention_masks = [float(i>0) for i in input_ids]
    
    captions_t = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    mask = torch.LongTensor(attention_masks).unsqueeze(0).to(device)
    with torch.no_grad():
        pooled_output, outputs = model(captions_t, mask)
    print("Predicted label: ", reverse_dic[torch.max(outputs, 1)[1].item()])
    # print("Real label: ", reverse_dic[y_test[index]])



if __name__ == '__main__':
    import fire
    fire.Fire()
    


            








        








    


    