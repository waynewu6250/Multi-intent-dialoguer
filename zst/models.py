import torch
import random
import itertools

from tqdm import tqdm
from utils import RunningAverage, rindex, pad

from torch import nn
from torch import optim
from torch.nn import functional as F

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification

# Special Tokens
CLS = '[CLS]'
SEP = '[SEP]' 

# Generate examples for a turn
def turn_to_examples(x, y, dic, tokenizer):
    examples = []
    for intent, (num, intent_ids) in dic.items():
        
        candidate = intent_ids[1:]
        # Prepare input_ids
        input_ids = x + candidate

        # Prepare token_type_ids
        sent1_len = rindex(input_ids[:-1], 102) + 1
        sent2_len = len(input_ids) - sent1_len
        token_type_ids = [0] * sent1_len + [1] * sent2_len

        # Prepare label
        label = int(num in y)

        # Update examples list
        examples.append((y, num, input_ids, token_type_ids, label))
    return examples

class Model(nn.Module):
    def __init__(self, tokenizer, bert):
        super(Model, self).__init__()

        self.tokenizer = tokenizer
        self.bert = bert

    @classmethod
    def from_scratch(cls, bert_model, verbose=True):
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        bert = BertForSequenceClassification.from_pretrained(bert_model, num_labels=2)
        
        # print('Loading pretrained weights...')
        # pre_model_dict = torch.load('best_e2e_multi.pth')
        # model_dict = bert.state_dict()
        # pre_model_dict = {k:v for k, v in pre_model_dict.items() if k in model_dict and v.size() == model_dict[k].size}
        # model_dict.update(pre_model_dict)
        # bert.load_state_dict(model_dict)

        model = cls(tokenizer, bert)
        if verbose:
            print('Intialized the model and the tokenizer from scratch')
        return model

    @classmethod
    def from_model_path(cls, output_model_path, verbose=True):
        tokenizer = BertTokenizer.from_pretrained(output_model_path)
        bert = BertForSequenceClassification.from_pretrained(output_model_path, num_labels=2)

        model = cls(tokenizer, bert)
        if verbose:
            print('Restored the model and the tokenizer from {}'.format(output_model_path))
        return model

    def move_to_device(self, args):
        self.bert.to(args.device)
        # if args.n_gpus > 1:
        #     self.bert = torch.nn.DataParallel(self.bert)

    def init_optimizer(self, args, num_train_iters):
        # Optimizer
        param_optimizer = list(self.bert.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.optimizer = BertAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  warmup=args.warmup_proportion,
                                  t_total=num_train_iters)
        self.optimizer.zero_grad()

    def run_train(self, dic, args, X_train, X_test, y_train, y_test):
        model, tokenizer = self.bert, self.tokenizer
        batch_size = args.batch_size
        model.train()

        # Generate training examples
        train_examples = [turn_to_examples(x, y, dic, tokenizer) for x,y in zip(X_train, y_train)]
        train_examples = list(itertools.chain.from_iterable(train_examples))
        print('Generated training examples')

        # Random Oversampling
        # Note that: Most of the constructed examples are negative
        if args.random_oversampling:
            negative_examples, positive_examples = [], []
            for example in train_examples:
                if example[-1] == 0: negative_examples.append(example)
                if example[-1] == 1: positive_examples.append(example)
            nb_negatives, nb_positives = len(negative_examples), len(positive_examples)
            sampled_positive_examples = random.choices(positive_examples, k=int(nb_negatives / 8))
            train_examples = sampled_positive_examples + negative_examples
            print('Did Random Oversampling')
            print('Number of positive examples increased from {} to {}'
                  .format(nb_positives, len(sampled_positive_examples)))

        # Initialize Optimizer
        num_train_iters = args.epochs * len(train_examples) / batch_size / args.gradient_accumulation_steps
        self.init_optimizer(args, num_train_iters)

        train_examples = train_examples[:300000]

        # Main training loop
        iterations = 0
        best_dev_f1 = 0.0
        train_avg_loss = RunningAverage()
        for epoch in range(args.epochs):
            print('Epoch {}'.format(epoch))

            random.shuffle(train_examples)
            train_corrects = 0
            totals = 0
            preds = 0
            total_acc = 0
            pbar = tqdm(range(0, len(train_examples), batch_size))
            for i in pbar:
                iterations += 1

                # Next training batch
                batch = train_examples[i:i+batch_size]
                true_labels, _, input_ids, token_type_ids, labels = list(zip(*batch))
                x_batch = input_ids

                # Padding and Convert to Torch Tensors
                input_ids, _ = pad(input_ids, args.device)
                token_type_ids, _ = pad(token_type_ids, args.device)
                labels = torch.LongTensor(labels).to(args.device)

                # Calculate loss
                loss = model(input_ids, token_type_ids, labels=labels)
                # if args.n_gpus > 1:
                #     loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                train_avg_loss.update(loss.item())

                # Update pbar
                pbar.update(1)
                pbar.set_postfix_str(f'Train Loss: {train_avg_loss()}')

                # parameters update
                if iterations % args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    tc, tt, tp, tacc = self.run_prediction(dic, args, x_batch, true_labels)
                    train_corrects += tc
                    totals += tt
                    preds += tp
                    total_acc += tacc
            
            recall = train_corrects / totals
            precision = train_corrects / preds
            f1 = 2 * (precision*recall) / (precision + recall)
            accuracy = total_acc / len(X_test) * args.gradient_accumulation_steps
            print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
            
            # Evaluate on the dev set and the test set
            vc, vt, vp, acc = self.run_prediction(dic, args, X_test[:1000], y_test[:1000])

            recall = vc / vt
            precision = vc / vp
            f1 = 2 * (precision*recall) / (precision + recall)
            accuracy = acc / len(X_test)
            print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
            self.save(args.output_dir)

            print('Evaluations after epoch {}'.format(epoch))
            if f1 > best_dev_f1:
                best_dev_f1 = f1
                self.save(args.output_dir)
                print('Saved the model')

    def predict_turn(self, x, y, dic, args, threshold=0.5):
        model, tokenizer = self.bert, self.tokenizer
        batch_size = args.batch_size
        was_training = model.training
        model.eval()

        preds = []
        examples = turn_to_examples(x, y, dic, tokenizer)
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            _, num, input_ids, token_type_ids, _ = list(zip(*batch))

            # Padding and Convert to Torch Tensors
            input_ids, _ = pad(input_ids, args.device)
            token_type_ids, _ = pad(token_type_ids, args.device)

            # Forward Pass
            logits = model(input_ids, token_type_ids)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().data.numpy()

            # Update preds
            for j in range(len(batch)):
                if probs[j] >= threshold:
                    preds.append(batch[j][1])

        if was_training:
            model.train()

        return preds

    def run_prediction(self, dic, args, X_test, y_test):
        preds = [self.predict_turn(x, y, dic, args) for (x,y) in zip(X_test, y_test)]
        return self.calc_score(preds, y_test)
    
    def calc_score(self, outputs, labels):

        corrects = 0
        totals = 0
        preds = 0
        acc = 0

        for (logit, label) in zip(outputs, labels):
            correct = len([p for p in logit if p in label])
            pred = len(logit)
            total = len(label)
            corrects = corrects + correct
            totals = totals + total
            preds = preds + pred
            
            if set(logit) == set(label):
                acc += 1

        return corrects, totals, preds, acc

    def save(self, output_model_path, verbose=True):
        model, tokenizer = self.bert, self.tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        output_model_file = output_model_path / WEIGHTS_NAME
        output_config_file = output_model_path / CONFIG_NAME

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(output_model_path)

        if verbose:
            print('Saved the model, the model config and the tokenizer')
