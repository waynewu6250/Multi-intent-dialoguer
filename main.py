import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from sklearn.model_selection import train_test_split
import pickle
import copy

from model import BertEmbedding
from all_data import get_dataloader
from config import opt


def train():
    
    # attributes
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    # dataset
    with open(opt.data_path, 'rb') as f:
        data = pickle.load(f)
    X, y, _ = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = get_dataloader(X_train, y_train, opt)
    val_loader = get_dataloader(X_test, y_test, opt)
    
    # model
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    
    
    model = BertEmbedding(config, max(train_loader.dataset.labels)+1)
    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path), map_location="cpu")
        print("Pretrained model has been loaded.\n")
    model = model.to(device)

    # optimizer, criterion
    optimizer= Adam([{"params": model.bert.parameters(), "lr": opt.learning_rate_bert},
                     {"params": model.classifier.parameters(), "lr": opt.learning_rate_classifier}])
    criterion = nn.CrossEntropyLoss().to(device)
    best_loss = 100

    # Start training
    for epoch in range(opt.epochs):
        print("====== epoch %d / %d: ======"% (epoch, opt.epochs))

        # Training Phase
        total_train_loss = 0
        train_corrects = 0
        
        for (captions_t, labels, _) in train_loader:

            captions_t = captions_t.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            pooled_output, outputs = model(captions_t)
            train_loss = criterion(outputs, labels)
            print(train_loss)

            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss
            train_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)
        
        print('Total train loss: {:.4f} '.format(total_train_loss / train_loader.dataset.num_data))
        print('Train accuracy: {:.4f}'.format(train_corrects.double() / train_loader.dataset.num_data))

        # Validation Phase
        total_val_loss = 0
        val_corrects = 0
        for (captions_t, labels, _) in val_loader:

            captions_t = captions_t.to(device)
            labels = labels.to(device)

            pooled_output, outputs = model(captions_t)
            val_loss = criterion(outputs, labels)

            total_val_loss += val_loss
            val_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)

        print('Total val loss: {:.4f} '.format(total_val_loss / val_loader.dataset.num_data))
        print('Val accuracy: {:.4f}'.format(val_corrects.double() / val_loader.dataset.num_data))
        if val_loss < best_loss:
            print('saving with loss of {}'.format(total_val_loss),
                  'improved over previous {}'.format(best_loss))
            best_loss = total_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            if (epoch+1) % 5 == 0:
                torch.save(model.state_dict(), "checkpoints/chinese-epoch-%s.pth"%epoch)
        
        print()

if __name__ == '__main__':
    train()
    


            








        








    


    