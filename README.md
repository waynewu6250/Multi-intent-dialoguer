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

## 3. Clustering
In [cluster.py](https://github.com/waynewu6250/Multi-intent-dialoguer/blob/master/clustering.py), we perform two analysis on the given embeddings, one is k nearest neighbor for each of the sentence embedding and calculate the true label belonging to these neighbors; finally check the accuracy. And we also perform regroup these embeddings into same clusters if each other are neighbors.


|    Neighbor Number     | accuracy |
|      ------------      | -------- | 
| Semantic Parsing: 1    |  0.9692  | 
| Semantic Parsing: 2    |  0.9579  | 
| Semantic Parsing: 50   |  0.8274  | 
| ATIS: 1                |  0.9955  |
| ATIS: 50               |  0.9698  |

As we further delve into some subclusters by choosing only 6 nearest neighbors with same label:
**"FLIGHT"** in ATIS dataset, we could have some example clusters as follows:

**Subcluster 1:**
>

    I want to fly from boston at 8:38 am and arrive in denver at 11:10
    I'd like to leave from boston on tuesday and i'd like to leave sometime in the morning
    hi I'm calling from boston I'd like to make a fight to either orlando or los angeles
    I want to travel from kansas city to chicago round trip leaving wednesday june sixteenth arriving in Chicago
    I need to fly from washington to san francisco but i'd like to stop over at dallas
    I'd like to fly from denver to pittsburgh to atlanta

**Subcluster 2:**
>
    Show me the flights from pittsburgh to los angeles on thursday
    Show me the flights from baltimore to boston
    Show me the flights from denver to las vegas
    Please give me flights from atlanta to boston on wednesday afternoon and thursday morning
    I need a flight from philadelphia to boston
    I would like a flight from boston to san francisco on august seventeenth what are your flights from dallas to baltimore

## 4. Deep Embedding Clustering
>
    python perform_dcec.py train
    python perform_dcec.py test

Checkpoints are in ``checkpoints-dcec/``

Results are in ``clustering_results/``

## 5. Sentence Clustering BERT
>
    python sentence_clustering.py train
    python sentence_clustering.py test

Checkpoints are in ``checkpoints-scbert/``

Results are in ``clustering_results/``






