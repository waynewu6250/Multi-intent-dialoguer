# Multi-intent-dialoguer

**Main Research:** <br>

A new framework for spoken language understanding in task-oriented dialgoue systems with a more real scenario.
This research project mainly focuses on the motivations of understanding human utterances and machine interactions, then forming knowledge abstraction for downstream policy learning tasks.

Mainly given a dialogue turn, it will return the updated **slot-values** and **satisfaction level**. <br>
It has the following features & aims: <br>

1. NLU:

    1) **Surface Model**:
        
        a. Multi-intent clustering framework <br>
        b. intent+slot-filling
    
    2) **Pertinence Model**:

        a. Evaluation framework

2. DST:

    1) **State Tracker**


## TODOs

- [ ] Data: Redefine labels in dialogue dataset (some are not clear).
- [ ] Fine-tune: Train Bert model with single sentence datasets and apply to dialogue datasets for clustering.
- [ ] Surface: Check dcec convergence.
- [ ] Surface: Attention words with masking mechanism.
- [ ] Surface: Fewer labels to train as possible.



-----------------------


## 1. Data Processing
Data preprocessing pipeline. <br>

**Associated files and folders**
>
    data/train_data.py
    data/dialogue_data.py

1. Go to `config.py` to select data type
2. Run the following to generate raw_data.pkl for the following use. <br>
    **Single sentence: ATIS/Semantic parsing dataset**
    >
        python data/train_data.py
    **Dialogue: MultiWOZ2.1 dataset**
    >
        python data/dialogue_data.py

-----------------------

## 2. Feature Extractor
To extract contextualized representations, the service fine-tune BERT model to generate the pretrained sentence embeddings. <br>

**Associated files and folders**
>  
    bert_finetune.py
    bert_nsp.py
    finetune_results/
    checkpoints/

### 1) Training

To train single sentence dataset: [atis](https://github.com/howl-anderson/ATIS_dataset)/[top semantic](https://arxiv.org/pdf/1810.07942.pdf)
>
    python bert_finetune.py train --datatype=atis
    python bert_finetune.py train --datatype=semantic

To train MultiWOZ dataset with next-sentence prediction:
>
    python bert_nsp.py train

### 2) Testing
There are three modes for testing the atis/top semantic BERT embeddings:

>
    python bert_finetune.py test --mode=[mode_type]

mode_type:
1. embedding: <br>
    For this mode, it will generate and store the sentence embeddings for every training data
2. data: <br>
    For this mode, it will do original text classification based on the dataset
3. user: <br>
    For this mode, you can type in any kind of sentence and it will classify a specific label for the corresponding sentence.

To extract BERT embeddings on single sentence level from dialogue dataset:
>
    python bert_nsp.py test --mode=embedding
For this mode, it will generate and store the sentence embeddings for every training data

-----------------------

## 3. Surface Model
After obtaining embeddings, we could use them for surface model:

please check here for more details: <br>

1. [intent clustering](nlu.md) <br>
2. intent+slot-filling


-----------------------

## 4. Pertinence Model
After obtaining embeddings, we could use them for pertinence model.






