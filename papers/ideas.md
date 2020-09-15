# Key Idea

## Sequence Tagging Problem (Supervised):
Tag the sentences as three following tags: <br>
\<Problem\>, \<Solution\>, \<NaN\>

1) **Name-entity recognition:** LSTM+CRF <br>

2) **BERT:** <br>
Cut into language segments and do sentence-sentence classification to see if they are relevant (scores)

3) **Topic Model** (Word level)

-----

## Text Summarization (Unsupervised):
Uses autoencoder to summarize important words and cluster them into problems and solutions.


1) **Attention Mechanism:**

    [Multimodal Attention 1](https://arxiv.org/pdf/1612.01887.pdf), [Multimodal Attention 2](https://arxiv.org/pdf/1502.03044.pdf): <br>
    Use text vectors to attend on different regions of images.

    [Multimodal Attention for Explanations](http://openaccess.thecvf.com/content_cvpr_2018/papers/Park_Multimodal_Explanations_Justifying_CVPR_2018_paper.pdf): <br>
    Use Image+Texts+Answer to attend on images.

    [Attention on BERT](https://drive.google.com/file/d/1e0WA8t0T0xvngTuMk01rbMeJySxynGE8/view): Extract attention from BERT model.

2) **Deep Clustering:**

    [Deep Learning Methods for Clustering](https://arxiv.org/pdf/1801.07648.pdf)

    [Deep Embedding Clustering](http://proceedings.mlr.press/v48/xieb16.pdf)

    [Deep Convolutional Embedding Clustering](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf)

    [Unsupervised Aspect Extraction](https://www.comp.nus.edu.sg/~leews/publications/acl17.pdf)

    [Weakly Supervised Aspect Extraction](https://stangelid.github.io/emnlp18oposum.pdf)

    [Text Clustering](https://www.aclweb.org/anthology/D19-5405.pdf)

3) **Text Summarization using autoencoder:**

    [SummAE](https://www.groundai.com/project/summae-zero-shot-abstractive-text-summarization-using-length-agnostic-auto-encoders/1)

    [MeanSum](https://arxiv.org/pdf/1810.05739.pdf): Uses autoencoder to produce summary combining reconstruction loss and summary loss

4) **Unsupervised Object Detection/Segmentation**

    WNet

    DCEC

    [Unsupervised Segmentation](https://kanezaki.github.io/pytorch-unsupervised-segmentation/ICASSP2018_kanezaki.pdf)

    [Instance Embedding](https://towardsdatascience.com/instance-embedding-instance-segmentation-without-proposals-31946a7c53e1): [code](https://github.com/nyoki-mtl/pytorch-discriminative-loss)

    [Unsupervised Object Detection](https://arxiv.org/pdf/1808.04593.pdf)

------

## Model Structure

|     Task     | Embedding | Structure | Source | 
| ------------ |  -------  | --------- | ------ |
| Text classification | ESA/brown cluster | similarity + tree search | https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8588/8611 |
| Text classfication  | BOW | charCNN | http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf |
| Document classification | CNN/RNN | Gated-RNN | https://www.aclweb.org/anthology/D15-1167.pdf |
| Document classification | word2vec | 2-level Attention RNN (sentence+document), attend on trained representating word | https://www.aclweb.org/anthology/N16-1174.pdf |
| Tagging | charCNN + flair | bilstm + self attention | https://github.com/Das-Boot/scifi |
| Tagging | BERT | bilstm + SSVM | https://github.com/rujunhan/EMNLP-2019 |
| Tagging | word2vec | bilstm + similarity clustering | https://www.aclweb.org/anthology/W18-5035.pdf |
| Relations | x | BERT matching the blanks | https://github.com/plkmo/BERT-Relation-Extraction |
| Relations | x | BERT + span classifier + relation classifier | https://arxiv.org/pdf/1909.07755.pdf |
| Relations | x | BERT + relation computing layer (TxT matrix of prob in class) | https://github.com/slczgwh/REDN |
| Commonsense | x | BERT + maximum attention score | https://github.com/SAP-samples/acl2019-commonsense-reasoning |
| Evaluation | BERT+LSTM | unsupervised nce loss between contexts and responses | https://research.fb.com/wp-content/uploads/2020/07/Learning-an-Unreferenced-Metric-for-Online-Dialogue-Evaluation.pdf | 














   