import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertEmbedding(nn.Module):
    
    def __init__(self, config, num_labels=2):
        super(BertEmbedding, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)

        
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output_d = self.dropout(pooled_output)
        logits = self.classifier(pooled_output_d)
        outputs = F.softmax(logits, dim=1)
        return pooled_output, outputs

    # def freeze_bert_encoder(self):
    #     for param in self.bert.parameters():
    #         param.requires_grad = False
    
    # def unfreeze_bert_encoder(self):
        # for param in self.bert.parameters():
        #         param.requires_grad = True
        


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = 'what is a pig'
    zz = tokenizer.tokenize(text)
    tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(zz)])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2
    model = BertEmbedding(config, num_labels)

    pooled_output, outputs = model(tokens_tensor)
    print((pooled_output).shape)
    print(outputs)





