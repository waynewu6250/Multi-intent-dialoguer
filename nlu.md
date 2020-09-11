# Surface Model

Intent clustering mechanism for task-oriented dialogue systems to deal with single or multi-intent recognition.

In this sub-module, we explore techniques for clustering texts or dialogues with diverse intents. For a given task-oriented training datasets,
our database should be established as multiple intent clusters
with the corresponding intent. For a typical cluster, it should come up with three of the following features, `Intent tag`, `Attention Words`, `Text Generation`.

## 1. Intent Clustering

1. Nearest Neighbor Search
2. Deep Embedding Clustering (DEC)
3. Neural Attention Model (NAM)

**Associated files and folders**
>  
    clustering.py
    perform_dcec.py
    perform_dcec_dialogue.py
    sentence_clustering.py
    clustering_labels/
    clustering_results/
    checkpoints-dcec/
    checkpoints-scbert/
    
### 1) Nearest Neighbor Search

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

### 2) Deep Embedding Clustering
Inspired by the work of [Deep Embedding Clustering](http://proceedings.mlr.press/v48/xieb16.pdf), we further cluster sentence embeddings which are with or without BERT fine-tuning using techniques of mapping bottleneck vectors into p/cluster probability space. <br>

**ATIS/Semantic parsing dataset**

To use:
>
    python perform_dcec.py train
    python perform_dcec.py test

**MULTIWOZ2.1 dataset**
Here, we perform clustering on single utterances or on pair of dialogues (data with postfix: pair)

To use:
>
    python perform_dcec_dialogue.py train
    python perform_dcec_dialogue.py test

Checkpoints are saved in ``checkpoints-dcec/`` <br>
clustering data and corresponding labels are saved in ``clustering_labels`` <br>
Results are in ``clustering_results/``


### 3) Neural Attention Model (NAM)
We could also treat [neural attention model](https://www.comp.nus.edu.sg/~leews/publications/acl17.pdf) as a clustering mean to further construct trainable embedding matrix to minimize reconstruction loss of weighted sum of them and the original sentence embeddings.

To use:
>
    python sentence_clustering.py train
    python sentence_clustering.py test

Checkpoints are in ``checkpoints-scbert/`` <br>
Results are in ``clustering_results/``