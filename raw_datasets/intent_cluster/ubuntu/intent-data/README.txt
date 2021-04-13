# INTRODUCTION
--------------
This folder contains data used in the experiments reported in the paper "Supervised Clustering of Questions into Intents for Dialog System Applications" published at the EMNLP2018 conference:
 – Supervised Clustering of Questions into Intents for Dialog System Applications
   Iryna Haponchyk, Antonio Uva, Seunghak Yu, Olga Uryupina and Alessandro Moschitti

The folder contains two datasets, i.e. Quora and FAQ-Hype datasets.


# THE QUORA INTENT DATASET
--------------------------
The Quora dataset is derived from the Quora dump relased on Jan 24, 2017 and published here: https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs. The original dataset consists of 400,000 of potential question duplicate pairs. The original file report IDs for each question in the pair, the full text of each question, and a duplication flag, 0 or 1, which indicates whether the line truly contains a duplicate pair.

In this work, we used a smaller part of the dataset, adapting it to the task being the target of the paper, i.e. question clustering. This could be done as the original Quora dataset contains many repeated questions spread through different pairs in the dataset. The procedure we used for building the Quora intent dataset is described in detail in section 4.1.1 of our paper - Question clusters from Quora. Briefly, we built the corpus by automatically deriving question clusters from the Quora dataset, complementing the available annotation, given for question pairs, with a transitive closure of the semantic matching property. This way, we obtained questions grouped by their intents.
However, since the Quora data is noisy, additionally, we manually annotated a part of the test data fixing the intent-based automatic question clusters obtained following inconsistent semantic match ground truths. 

The Quora intent dataset contains 1,334 questions distributed in  628 clusters. Thus, there is an average of 2.12 questions per cluster. The data is split into 20 samples organized as follows:
 - 10 samples for training
 -  5 samples for development 
 -  5 samples for test

The train set contains samples 01, 03, 05, 07, 09, 10, 11, 12, 13 and 19.
The development set contains samples 02, 04, 14, 18 and 20.
The test set contains samples 06, 08, 15, 16 and 17.

The train, dev. and test set contains 270, 146 and 212 clusters respectively. The clusters contain different number of questions, ranging from singletons, i.e. 1-element clusters, to groups of 100 questions.

Each sample file follows the format:

qid1<TAB>question1
qid2<TAB>question2
qid3<TAB>question3
..
qidN<TAB>questionN
<BLANKLINE>>
qidN+1<TAB>questionN+1
qidN+2<TAB>questionN+2
...
qidN+M<TAB>questionN+M

where each line reports a question (together with its id), and each group of lines represents a cluster of questions. Different clusters are separated by a blank line. 
The singleton clusters are the dominant group in the Quora intent dataset. This is due to the inclusion of non-duplicate questions that appear in the original Quora dataset.
	
AUTOMATICALLY DERIVING INTENT CLUSTERS
--------------------------------------
Each sample contains many question clusters, with each cluster containing a varying number of questions. 
Clusters have been sampled according to the following procedure:
 - we sample an initial pool of questions from the Quora dataset
 - we compute transitive closure of questions in the original pool to obtain a cluster of questions, i.e. positive examples
 - we add questions from negative pairs that co-occur with selected cluster questions in the original Quora dataset.

This way, we obtained a set of intent-based clusters, together with other nearly-similar questions that seem to have the same intent, but, which in reality, ask for a different thing (i.e. more specific/general questions, etc..). These challenging pairs, added by the authors to the original Quora dataset, serve the purpose of making the duplicate detection task more challenging. 

For example, in sample 02 there is one cluster about presidential election:

    22942   What do you think of the 2016 US presidential election ?
    112086  What do you think about the 2016 US Presidential election ?

And there is also one singleton cluster, like this:

    144611  Who will win the 2020 presidential election and why ?

Clearly, questions in these clusters, although highly related, are kept separated as they refer to two different topics, i.e. 2016 US presidential election vs. 2020 presidential election. For other examples of such questions look at examples (3) and (4) in section 4.1 - Quora corpus.

MANUALLY ANNOTATED INTENT CLUSTERS
----------------------------------
File quora-sample06_08-reannotated.csv contains samples 06 and 08 from the test set that have been re-assessed by human annotators in order to correct inconsistencies.
This file is in a semicolon separated csv format, in which each line corresponds to a question. Each question is marked with an id and an automatically derived cluster id. And, the human annotation is provided in a form of a new composite intent label, and a number of categories and slots.
See section 4.1.2 - Manually annotating intent clusters - for further details.


# FAQ-HYPE DATASET
------------------
The FAQ dataset corpus contains questions, in Italian language, asked by users to a conversational agent about Hype. Hype is an online service offering a free credit card with an IBAN, and an iBanking app to its customers. Unlike Quora, most of the questions are about bank/financial domain. Furthermore, the questions are explicitly assigned to clusters by human annotators and they correspond to intents by construction. The clusters are build based on an original FAQ list and a number of semantic variants used by real users to refer to these FAQs.

Below, we report an example of a question cluster from this dataset:

q1: Non ricordo più la password per accedere all'App (I don't remember the password for the App)
q2: mi sono dimenticato la password (I forgot the password)
q3: reimpostare la password (reset the password)
q4: cambio password (change the password)

As it can be seen, the intent of these questions is to recover the user password. Thus, the cluster contains utterances q1, q2, q3 and q4, which are lexical variations of the same intent. For more details, see section 4.2 – FAQ: Hype Intent corpus.

The FAQ-Hype dataset is much smaller than the Quora dataset. Thus, in contrast to the Quora dataset, we do not not organize it in samples, but split the data into 2 samples: one for training, and one - for testing. 
Overall, the dataset contains 147 questions spread in 28 clusters, which we divided as follows:
 - 19 clusters in the training set (90 questions)
 -  9 clusters in the test set (57 questions)

Differently from the Quora dataset, where many singletons are present, each cluster in the FAQ-Hype dataset contains no less than 3 questions. The majority of clusters have size 4 and 5.

Clusters of questions follow the format:

FAQ=>question1
question2
question3
...
questionN

FAQ=>questionN+1
questionN+2
questionN+3
...
questionN+M

where, like in Quora, different clusters are separated by blank line.

This corpus provides very valuable data for a study, despite of a limited number of questions compared to the Quora dataset, since it contains manual annotation of clusters provided by highly skilled domain experts obtained in a real-world setting (FAQ).
