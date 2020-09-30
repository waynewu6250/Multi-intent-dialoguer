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
from tqdm import tqdm

from model import BertEmbedding
from all_data import get_dataloader
from config import opt

def load_data(X):
    
    input_ids = pad_sequences(X, maxlen=opt.maxlen, dtype="long", truncating="post", padding="post")
    
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return input_ids, attention_masks

def train(**kwargs):
    
    # attributes
    for k, v in kwargs.items():
        setattr(opt, k, v)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    # dataset
    with open(opt.dic_path, 'rb') as f:
        dic = pickle.load(f)
    with open(opt.train_path, 'rb') as f:
        train_data = pickle.load(f)
    if opt.test_path:
        with open(opt.test_path, 'rb') as f:
            test_data = pickle.load(f)

    if opt.datatype == "atis":
        # ATIS Dataset
        X_train, y_train, _ = zip(*train_data)
        X_test, y_test, _ = zip(*test_data)
    elif opt.datatype == "semantic":
        # Semantic parsing Dataset
        X, y = zip(*train_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    elif opt.datatype == "e2e":
        # Microsoft Dialogue Dataset
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
        X, y, _ = zip(*all_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train, mask_train = load_data(X_train)
    X_test, mask_test = load_data(X_test)
    
    # length = int(len(X_train)*0.1)
    # X_train = X_train[:length]
    # y_train = y_train[:length]
    # mask_train = mask_train[:length]
    
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
    best_accuracy = 0

    # Start training
    for epoch in range(opt.epochs):
        print("====== epoch %d / %d: ======"% (epoch+1, opt.epochs))

        # Training Phase
        total_train_loss = 0
        train_corrects = 0
        
        for (captions_t, labels, masks) in tqdm(train_loader):

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            #train_loss = model(captions_t, masks, labels)

            _, _, outputs = model(captions_t, masks)
            train_loss = criterion(outputs, labels)

            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss
            train_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)
        
        train_acc = train_corrects.double() / train_loader.dataset.num_data
        print('Total train loss: {:.4f} '.format(total_train_loss / train_loader.dataset.num_data))
        print('Train accuracy: {:.4f}'.format(train_acc))

        # Validation Phase
        total_val_loss = 0
        val_corrects = 0
        for (captions_t, labels, masks) in val_loader:

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                _, pooled_output, outputs = model(captions_t, masks)
            val_loss = criterion(outputs, labels)

            total_val_loss += val_loss
            val_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)

        val_acc = val_corrects.double() / val_loader.dataset.num_data
        print('Total val loss: {:.4f} '.format(total_val_loss / val_loader.dataset.num_data))
        print('Val accuracy: {:.4f}'.format(val_acc))
        if total_val_loss < best_loss:
            print('saving with loss of {}'.format(total_val_loss),
                  'improved over previous {}'.format(best_loss))
            best_loss = total_val_loss
            best_accuracy = val_acc

            torch.save(model.state_dict(), 'checkpoints/best_{}.pth'.format(opt.datatype))
        
        print()
    print('Best Test Accuracy: {:.4f}'.format(best_accuracy))

def test(**kwargs):

    # ATIS Dataset
    if opt.datatype == "atis":
        with open(opt.atis_train_path, 'rb') as f:
            train_data = pickle.load(f)
        X_train, y_train, _ = zip(*train_data)
        X_train, mask_train = load_data(X_train)
        
        with open(opt.atis_test_path, 'rb') as f:
            test_data = pickle.load(f)
        X_test, y_test, _ = zip(*test_data)
        X_test, mask_test = load_data(X_test)

        dic_path = opt.atis_dic_path
        model_path = opt.atis_model_path
        embedding_path = opt.atis_embedding_path

    # Semantic parsing Dataset
    elif opt.datatype == "semantic":
        with open(opt.se_path, 'rb') as f:
            data = pickle.load(f)
        X_train, y_train = zip(*data)
        X_train, mask_train = load_data(X_train)

        dic_path = opt.se_dic_path
        model_path = opt.se_model_path
        embedding_path = opt.se_embedding_path

    # attributes
    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    # dictionary
    with open(dic_path, 'rb') as f:
        dic = pickle.load(f)
    reverse_dic = {v: k for k,v in dic.items()}
    
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
        
        train_loader = get_dataloader(X_train, y_train, mask_train, opt)

        results = collections.defaultdict(list)

        for i, (captions_t, labels, masks) in enumerate(train_loader):
            
            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                hidden_states, pooled_output, outputs = model(captions_t, masks)
                print("Saving Data: %d" % i)

                for ii in range(len(labels)):
                    key = labels[ii].data.cpu().item()
                    
                    embedding = pooled_output[ii].data.cpu().numpy().reshape(-1)
                    word_embeddings = hidden_states[-1][ii].data.cpu().numpy()
                    
                    tokens = tokenizer.convert_ids_to_tokens(captions_t[ii].data.cpu().numpy())
                    tokens = [token for token in tokens if token != "[CLS]" and token != "[SEP]" and token != "[PAD]"]
                    original_sentence = " ".join(tokens)
                    results[key].append((original_sentence, embedding, word_embeddings))

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
    
    # User defined
    elif opt.mode == "user":
        while True:
            print("Please input the sentence: ")
            text = input()
            print("\n======== Predicted Results ========")
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
            print("=================================")    
    
    





if __name__ == '__main__':
    import fire
    fire.Fire()
    


            








        








    


    