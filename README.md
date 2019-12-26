# Multi-intent-dialoguer

**Research project:** <br>
Intent clustering mechanism for task-oriented dialogue systems to deal with single or multi-intent recognition.

In this project, we explore techniques for clustering texts with diverse intents. For a given task-oriented training datasets,
our database should be established as multiple intent clusters
with the corresponding intent. For a typical cluster, it should come up with three of the following features, `Intent tag`, `Attention Words`, `Text Generation`. 

## 1. Training
The service uses BERT model to fine tune pretrained sentence embeddings. To use, 
