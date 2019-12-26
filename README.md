# Multi-intent-dialoguer

**Research project:** <br>
Intent clustering mechanism for task-oriented dialogue systems to deal with single or multi-intent recognition.

In this project, we explore techniques for clustering texts with diverse intents. For a given task-oriented training datasets,
our database should be established as multiple intent clusters
with the corresponding intent. For a typical cluster, it should come up with three of the following features, `Intent tag`, `Attention Words`, `Text Generation`. 

## 1. Training
The service uses BERT model to fine tune the pretrained sentence embeddings. First use 
>
    python train_data.py
to generate raw_data.pkl for the following use.

Then to train, type in the following command:
>
    python main.py train --datatype=atis
to generate the embeddings based on [atis dataset](https://github.com/howl-anderson/ATIS_dataset).

>
    python main.py train --datatype=semantic
to generate the embeddings based on semantic parsing dataset from [this paper](https://arxiv.org/pdf/1810.07942.pdf).

## 2. Testing
There are three modes for testing the BERT embeddings:

1. Embeddings

>
    python main.py test --mode=embedding
For this mode, it will generate and store the sentence embeddings for every training data

2. Data

>
    python main.py test --mode=data
For this mode, it will do original text classification based on the dataset

3. User defined

>
    python main.py test --mode=user
For this mode, you can type in any kind of sentence and it will classify a specific label for the corresponding sentence.

## 3.






