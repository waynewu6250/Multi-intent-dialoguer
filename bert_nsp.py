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
import collections

from model import BertEmbedding
from all_data import get_dataloader_dialogue
from config import opt

def load_data(X):
    
    input_ids = pad_sequences(X, maxlen=opt.maxlen, dtype="long", truncating="post", padding="post")
    
    attention_masks = []
    segments = []
    for seq in input_ids:
        # mask
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
        # segments
        seq = np.array(seq)
        seg = np.zeros_like(seq)
        
        pivot = np.where(seq==102)[0]
        if len(pivot) == 0:
            pass
        elif len(pivot) == 1:
            seg[pivot[0]:] = 1.0
        elif len(pivot) == 2:
            seg[pivot[0]+1:pivot[1]+1] = 1.0
        segments.append(seg)
        
    return input_ids, attention_masks, segments

def train(**kwargs):
    
    # attributes
    for k, v in kwargs.items():
        setattr(opt, k, v)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    # dataset
    with open(opt.woz_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(opt.woz_dic_path, 'rb') as f:
        dic = pickle.load(f)

    all_data = []
    dialogue_id = {}
    dialogue_counter = 0
    counter = 0
    for data in train_data:
        for instance in data:
            all_data.append(instance)
            dialogue_id[counter] = dialogue_counter
            counter += 1
        dialogue_counter += 1

    X, y = zip(*all_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    X_train, mask_train, seg_train = load_data(X_train)
    X_test, mask_test, seg_test = load_data(X_test)
    train_loader = get_dataloader_dialogue(X_train, y_train, mask_train, seg_train, len(dic), opt)
    val_loader = get_dataloader_dialogue(X_test, y_test, mask_test, seg_test, len(dic), opt)
    
    # model
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    
    model = BertEmbedding(config, len(dic))
    if opt.woz_model_path:
        model.load_state_dict(torch.load(opt.woz_model_path))
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
    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
    best_loss = 10000

    # Start training
    for epoch in range(opt.epochs):
        print("====== epoch %d / %d: ======"% (epoch, opt.epochs))

        # Training Phase
        total_train_loss = 0
        train_corrects = 0
        
        for (captions_t, labels, masks, segs) in train_loader:

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            segs = segs.to(device)

            optimizer.zero_grad()
            #train_loss = model(captions_t, masks, labels)

            _, _, outputs = model(captions_t, masks, segs)
            train_loss = criterion(outputs, labels)
            print('Current loss: ', train_loss.item())

            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss
            for i, logits in enumerate(outputs):
                log = logits[torch.where(labels[i]==1)[0]]
                log = torch.sigmoid(log)
                log = torch.sum(log) / len(log)
                train_corrects += log
        
        num_batches = train_loader.dataset.num_data // opt.batch_size
        print('Total train loss: {:.4f} '.format(total_train_loss / num_batches))
        print('Train accuracy: {:.4f}'.format(train_corrects.double() / train_loader.dataset.num_data))

        # Validation Phase
        total_val_loss = 0
        val_corrects = 0
        for (captions_t, labels, masks, segs) in val_loader:

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            segs = segs.to(device)
            
            with torch.no_grad():
                _, pooled_output, outputs = model(captions_t, masks, segs)
            val_loss = criterion(outputs, labels)

            total_val_loss += val_loss
            for i, logits in enumerate(outputs):
                log = logits[torch.where(labels[i]==1)[0]]
                log = torch.sigmoid(log)
                log = torch.sum(log) / len(log)
                val_corrects += log
        
        num_batches = val_loader.dataset.num_data // opt.batch_size
        print('Total val loss: {:.4f} '.format(total_val_loss / num_batches))
        print('Val accuracy: {:.4f}'.format(val_corrects.double() / val_loader.dataset.num_data))

        print('saving with loss of {}'.format(total_val_loss / num_batches))
        best_loss = total_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(model.state_dict(), "checkpoints/epoch-woz-%s.pth"%epoch)
        
        print()

def test(**kwargs):

    #dataset
    with open(opt.woz_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(opt.woz_dic_path, 'rb') as f:
        dic = pickle.load(f)
    reverse_dic = {v: k for k,v in dic.items()}

    all_data = []
    dialogue_id = {}
    dialogue_counter = 0
    counter = 0
    for data in train_data:
        for instance in data:
            all_data.append(instance)
            dialogue_id[counter] = dialogue_counter
            counter += 1
        dialogue_counter += 1
    with open(opt.woz_dialogue_id_path, 'wb') as f:
        pickle.dump(dialogue_id, f)
    

    X, y = zip(*all_data)

    model_path = opt.woz_model_path
    embedding_path = opt.woz_embedding_path

    # attributes
    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False
    
    # model
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    
    model = BertEmbedding(config, len(dic))
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print("Pretrained model has been loaded.\n")
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Store embeddings
    if opt.mode == "embedding":
        X_train, mask_train, seg_train = load_data(X)
        train_loader = get_dataloader_dialogue(X_train, y, mask_train, seg_train, len(dic), opt)

        results = []
        for i, (captions_t, labels, masks, segs) in enumerate(train_loader):
            
            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            segs = segs.to(device)
            with torch.no_grad():
                
                # Skip the last sentence in each dialogue at this time

                seq = torch.zeros(1, 25).to(device)
                mask = torch.zeros(1, 25).to(device)
                seg = torch.zeros(1, 25).to(device)

                for ii in range(len(labels)):
                    
                    pivot = torch.where(captions_t[ii]==102)[0]
                    if len(pivot) > 0 and pivot[0] < 25:
                        seq[0][:pivot[0]+1] = captions_t[ii][:pivot[0]+1]
                        mask[0][:pivot[0]+1] = (captions_t[ii][:pivot[0]+1]>0).long()
                        seg[0][:pivot[0]+1] = segs[ii][:pivot[0]+1]
                    else:
                        seq[0] = captions_t[ii][:25]
                        mask[0] = (captions_t[ii][:25]>0).long()
                        seg[0] = segs[ii][:25]
                    
                    hidden_states, pooled_output, outputs = model(seq.long(), mask.long(), seg.long())
                    
                    key = torch.where(labels[ii]==1)[0].data.cpu().numpy()
                    
                    embedding = pooled_output[0].data.cpu().numpy().reshape(-1)
                    word_embeddings = hidden_states[-1][0].data.cpu().numpy()
                    
                    tokens = tokenizer.convert_ids_to_tokens(seq[0].data.cpu().numpy())
                    tokens = [token for token in tokens if token != "[CLS]" and token != "[SEP]" and token != "[PAD]"]
                    original_sentence = " ".join(tokens)
                    
                    results.append((original_sentence, embedding, word_embeddings, key))

                print("Saving Data: %d" % i)
            if i == 100:
                break

        torch.save(results, embedding_path)
    
    # Run test classification
    elif opt.mode == "data":

        index = np.random.randint(0, len(X_test), 1)[0]
        input_ids = X_test[index]
        attention_masks = mask_test[index]
        print(" ".join(tokenizer.convert_ids_to_tokens(input_ids)))

        captions_t = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        mask = torch.LongTensor(attention_masks).unsqueeze(0).to(device)
        with torch.no_grad():
            pooled_output, outputs = model(captions_t, mask)
        print("Predicted label: ", reverse_dic[torch.max(outputs, 1)[1].item()])
        print("Real label: ", reverse_dic[y_test[index]])


if __name__ == '__main__':
    import fire
    fire.Fire()
    


            








        








    


    