# Key Idea

## Sequence Tagging Problem (Supervised):
Tag the sentences as three following tags: <br>
\<Problem\>, \<Solution\>, \<NaN\>

a) Name-entity recognition: LSTM+CRF <br>
* Ref: 

b) BERT: Cut into language segments and do sentence-sentence classification to see if they are relevant (scores)
* Ref: 

c) Topic Model (Word level)
* Ref: 

-----

## Text Summarization (Unsupervised):
Uses autoencoder to summarize important words and cluster them into problems and solutions.


a) **Attention Mechanism:**

[Multimodal Attention 1](https://arxiv.org/pdf/1612.01887.pdf), [Multimodal Attention 2](https://arxiv.org/pdf/1502.03044.pdf): <br>
Use text fectors to attend on different regions of images.

[Multimodal Attention for Explanations](http://openaccess.thecvf.com/content_cvpr_2018/papers/Park_Multimodal_Explanations_Justifying_CVPR_2018_paper.pdf): <br>
Use Image+Texts+Answer to attend on images.

[Attention on BERT](https://drive.google.com/file/d/1e0WA8t0T0xvngTuMk01rbMeJySxynGE8/view): Extract attention from BERT model.

b) **Deep Clustering:**

[Deep Embedding Clustering](http://proceedings.mlr.press/v48/xieb16.pdf)

[Deep Convolutional Embedding Clustering](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf)

[Text Clustering](https://www.aclweb.org/anthology/D19-5405.pdf)

c) **Text Summarization using autoencoder:**

[SummAE](https://www.groundai.com/project/summae-zero-shot-abstractive-text-summarization-using-length-agnostic-auto-encoders/1)

[MeanSum](https://arxiv.org/pdf/1810.05739.pdf): Uses autoencoder to produce summary combining reconstruction loss and summary loss

d) **Unsupervised Object Detection**

[Unsupervised Object Detection](https://arxiv.org/pdf/1808.04593.pdf)



------
WNet instance segmentation

DCEC change p





   