# Papers & Ideas:

## Datasets
[Dialogue datasets1](https://github.com/AtmaHou/Task-Oriented-Dialogue-Dataset-Survey) <br>
[Dialogue datasets2](https://breakend.github.io/DialogDatasets/)

Single turn
- [x] [ATIS](https://github.com/yvchen/JointSLU/tree/master/data) 
      : single intent + entity <br>
- [x] [SNIPS](https://github.com/waynewu6250/Multi-intent-dialoguer/tree/master/raw_datasets/Benchmark)
      : add 'and' to be multi intents + entity <br>
- [x] [TOP semantic parsing](https://github.com/waynewu6250/Multi-intent-dialoguer/blob/master/raw_datasets/top-dataset-semantic-parsing/train.tsv)
      : multi intents + entity <br>

Multi turn
- [ ] [DSTC4 challenge](http://www.colips.org/workshop/dstc4/DSTC4_pilot_tasks.pdf)
      : multi intents + entity <br>
- [x] [MultiWOZ Corpus](https://www.repository.cam.ac.uk/handle/1810/294507) <br>
- [x] [Schema-Guided Dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)

Multi turn references
- [x] [Microsoft e2e Dialogue Challenge](https://github.com/xiul-msr/e2e_dialog_challenge/tree/master/data) <br>
- [ ] [Movie Booking Dataset](https://github.com/MiuLab/TC-Bot#data) <br>
- [ ] [Stanford Dialog Dataset](http://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip)
- [ ] [Facebook bAbi end-to-end dialog](https://arxiv.org/pdf/1605.07683.pdf) <br>

---------

## Papers:

## 1. Conversational AI reviews
- [x] [Recent NLP paper](https://github.com/mhagiwara/100-nlp-papers)
- [x] [NLP research paper overview](https://www.topbots.com/most-important-ai-nlp-research/)
- [x] [DL NLP review](https://arxiv.org/pdf/1708.02709.pdf)
- [x] [Systematic Review Paper List](https://github.com/sz128/Natural-language-understanding-papers/blob/master/domain-intent-slot.md) <br>
- [x] [Neural Approaches to Conversational AI](https://arxiv.org/pdf/1809.08267.pdf) <br>
- [x] [Chatbot Design review](https://thesai.org/Downloads/Volume6No7/Paper_12-Survey_on_Chatbot_Design_Techniques_in_Speech_Conversation_Systems.pdf) <br>
- [x] [Chatbot Evaluation](https://www.aclweb.org/anthology/W17-5522.pdf) <br>
- [x] [Systematic review on speech recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8632885) <br>
- [x] [CNN Model Review](https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11)

## 2. Task oriented dialogue system

Overview
- [x] [Stanford NLP overview](https://web.stanford.edu/~jurafsky/slp3/26.pdf)
- [ ] [Microsoft NLP Group papers](https://github.com/microsoft/MSR-NLP-Projects)
- [ ] [Intelligent conversational chatbot](https://www.csie.ntu.edu.tw/~yvchen/s105-icb/syllabus.html)
- [ ] [MiuLab work](https://www.csie.ntu.edu.tw/~miulab/#home)
- [ ] [Comparison for task-oriented dialogue system](https://github.com/AtmaHou/Task-Oriented-Dialogue-Research-Progress-Survey) <br>
- [ ] [Spoken Language Understanding papers](https://paperswithcode.com/task/spoken-language-understanding)

NLU
- [x] [NLU services comparison](https://www.aclweb.org/anthology/W17-5522.pdf)
- [x] [Multi-intent benchmark1](https://arxiv.org/pdf/2004.10087.pdf)
- [ ] [BERT pretrained dialogue](https://arxiv.org/pdf/2004.06871.pdf)


Dialogue State Tracking
- [ ] [Hidden Information State model](http://mi.eng.cam.ac.uk/~sjy/papers/ygkm10.pdf)
- [ ] [Bayesian update of dialogue state](http://mi.eng.cam.ac.uk/~sjy/papers/thyo10.pdf)
- [x] [DNN](https://www.aclweb.org/anthology/W13-4073.pdf)
- [x] [2 LSTM network: slot-value & sentence](https://assets.amazon.science/23/98/80671ef545e4927c1716279a9340/flexible-and-scalable-state-tracking-framework-for-goal-oriented-dialogue-systems.pdf)
- [x] [BERT-training1: slot-value & sentence](https://arxiv.org/pdf/2006.01554.pdf): input: r_n-1+u_n, slot_n; output: value_n
- [x] [BERT-training2: direct prediction](https://arxiv.org/pdf/1907.03040.pdf): input: r_n-1+u_n; output: slot_n;
- [x] [BERT-training3: SLU task](https://arxiv.org/pdf/2005.11640v3.pdf): input: u_n, slot&value_n-1; output: slot&value_n (https://github.com/simplc/WCN-BERT)

NLU+DST
- [ ] [Joint training](https://drive.google.com/file/d/1I8iU-dLPRnC7ZxTULTso_gwhj4uQJ23U/view)

NLG
- [x] [Natural Language Generation](https://pdfs.semanticscholar.org/728e/18fbf00f5a80e9a070db4f4416d66c7b28f4.pdf)

## 3. Attention Mechanism
- [x] [Attention on BERT](https://drive.google.com/file/d/1e0WA8t0T0xvngTuMk01rbMeJySxynGE8/view) <br>
- [x] [Multimodal Explanations by attention on images](http://openaccess.thecvf.com/content_cvpr_2018/papers/Park_Multimodal_Explanations_Justifying_CVPR_2018_paper.pdf) <br>
- [x] [Adaptive Attention for Image Captioning](https://arxiv.org/pdf/1612.01887.pdf) <br>
- [x] [Visual Attention for Image Captioning](https://arxiv.org/pdf/1502.03044.pdf) <br>


--------

## Ideas:

## 1. Sequence Tagging (Supervised):
Tag the sentences as three following tags: <br>
\<Problem\>, \<Solution\>, \<NaN\>

1. **Name-entity recognition:** <br>
LSTM+CRF 

2. **BERT:** <br>
Cut into language segments and do sentence-sentence classification to see if they are relevant (scores)

3. **Topic Model** <br>


## 2. Text Summarization (Unsupervised):
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

### 3. Model Structure

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
| Embedding | x | 2 encoders and use concat, diff, dot to fuse. <br> multiple sentence encoder approaches | https://arxiv.org/pdf/1705.02364.pdf |


### 4. Evaluation

|     Task     | Evaluation | Source |
| ------------ | ---------- | ------ |

