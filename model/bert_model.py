import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from pytorch_pretrained_bert import BertForNextSentencePrediction

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
        # self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)

        # Self-attentive
        self.linear1 = nn.Linear(config.hidden_size, 256)
        self.linear2 = nn.Linear(4*256, config.hidden_size)
        self.tanh = nn.Tanh()
        self.context_vector = nn.Parameter(torch.randn(256, 4), requires_grad=True)

        # Hierarchical
        self.linear = nn.Linear(13*config.hidden_size, config.hidden_size)

    def forward(self, input_ids, mask, seg_tensors=None):
        """
        BERT outputs:
        last_hidden_states: (b, t, h)
        pooled_output: (b, h), from output of a linear classifier + tanh
        hidden_states: 13 x (b, t, h), embed to last layer embedding
        attentions: 12 x (b, num_heads, t, t)
        """
        last_hidden_states, pooled_output, hidden_states, attentions = self.bert(input_ids, token_type_ids=seg_tensors, attention_mask=mask)

        # mode = 'max-pooling'
        # mode = 'self-attentive'
        # mode = 'self-attentive-mean'
        # mode = 'h-max-pooling'
        mode = 'bissect'
        pooled_output = self.transform(last_hidden_states, pooled_output, hidden_states, attentions, mask, mode)
        pooled_output_d = self.dropout(pooled_output)
        logits = self.classifier(pooled_output_d)

        return last_hidden_states, pooled_output, logits
        # loss = self.bert(input_ids, attention_mask=mask, labels=labels)
        # return loss
    
    def transform(self, last_hidden_states, pooled_output, hidden_states, attentions, mask, mode):
        
        if mode == 'max-pooling':
            # Method 1: max pooling
            pooled_output, indexes = torch.max(last_hidden_states * mask[:,:,None], dim=1)
        
        elif mode == 'self-attentive':
            # Method 2: self-attentive network
            b, _, _ = last_hidden_states.shape
            vectors = self.context_vector.unsqueeze(0).repeat(b, 1, 1)

            h = self.tanh(self.linear1(last_hidden_states)) # (b, t, h)
            scores = torch.bmm(h, vectors) # (b, t, 4)
            scores = nn.Softmax(dim=1)(scores) # (b, t, 4)
            outputs = torch.bmm(scores.permute(0, 2, 1), h).view(b, -1) # (b, 4h)
            pooled_output = self.linear2(outputs)
        
        elif mode == 'self-attentive-mean':
            # Method 2: self-attentive network
            b, _, _ = last_hidden_states.shape
            vector = torch.mean(last_hidden_states, dim=1).unsqueeze(2)

            #h = self.tanh(self.linear1(last_hidden_states)) # (b, t, h)
            scores = torch.bmm(last_hidden_states, vector) # (b, t, 1)
            scores = nn.Softmax(dim=1)(scores) # (b, t, 1)
            pooled_output = torch.bmm(scores.permute(0, 2, 1), last_hidden_states).squeeze(1) # (b, h)

        elif mode == 'h-max-pooling':
            # Method 3: hierarchical max pooling
            b, t, h = last_hidden_states.shape
            N = len(hidden_states)
            final_vectors = torch.zeros(b, h, N).to(self.device)
            for i in range(len(hidden_states)):
                outs, _ = torch.max(hidden_states[i] * mask[:,:,None], dim=1)
                final_vectors[:, :, i] = outs
            final_vectors = final_vectors.view(b, -1)
            pooled_output = self.linear(final_vectors)
        
        elif mode == 'bissect':
            # Method 4: bissecting bert
            hidden_states = hidden_states[1:]
            b, t, h = hidden_states[0].shape
            N = len(hidden_states)

            h_states = torch.zeros(b, t, h, N).to(self.device)
            for i in range(N):
                h_states[:, :, :, i] = hidden_states[i]

            final_vectors = torch.zeros(b, t, h).to(self.device)
            # x = (1-torch.eye(N)).unsqueeze(0).repeat(b, 1, 1).to(self.device)

            for i in range(t):
                word_vector = h_states[:, i, :, :] # (b, h, N)
                vector = torch.mean(word_vector, dim=2).unsqueeze(1) # (b, 1, h)

                scores = torch.bmm(vector, word_vector) # (b, 1, N)
                scores = nn.Softmax(dim=2)(scores) # (b, 1, N)
                final_vectors[:, i, :] = torch.bmm(word_vector, scores.permute(0, 2, 1)).squeeze(2) # (b, h)

                # word_vector = word_vector / torch.norm(word_vector, dim=1).unsqueeze(1) # normalize each vector
                # scores = torch.bmm(word_vector.permute(0,2,1), word_vector) # (b, N, N)
                # scores = scores * x # set self attention to be 0s
                # scores = 1 / scores.sum(dim=2)
                # scores = scores / scores.sum(dim=1).unsqueeze(1) # (b, N)
                # scores = scores.unsqueeze(2) # (b, N, 1)
                # final_vectors[:, i, :] = torch.bmm(word_vector, scores).squeeze(2) # (b, h)
                # final_vectors[:, i, :] = hidden_states[-1][:, i, :]
            
            pooled_output, indexes = torch.max(final_vectors * mask[:,:,None], dim=1)
            
        return pooled_output
            


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = 'what is a pig'
    zz = tokenizer.tokenize(text)
    tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(zz)])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2
    model = BertEmbedding(config, num_labels)

    hidden_states, pooled_output, outputs = model(tokens_tensor)
    print((pooled_output).shape)
    print(outputs)





