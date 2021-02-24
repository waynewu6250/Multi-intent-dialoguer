# Papers & Ideas:

## Datasets
[Single Sentence benchmark](https://github.com/sz128/slot_filling_and_intent_detection_of_SLU) <br>
[Dialogue datasets1](https://github.com/AtmaHou/Task-Oriented-Dialogue-Dataset-Survey) <br>
[Dialogue datasets2](https://breakend.github.io/DialogDatasets/)

Single turn
- [x] [ATIS](https://github.com/yvchen/JointSLU/tree/master/data) 
      : single intent + entity <br>
- [x] [SNIPS](https://github.com/waynewu6250/Multi-intent-dialoguer/tree/master/raw_datasets/SNIPS)
      : add 'and' to be multi intents + entity <br>
- [x] [TOP semantic parsing](https://www.aclweb.org/anthology/D18-1300/)
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
- [x] [CNN Model Review](https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11) <br>
- [ ] [Deep Learning in Spoken and Text-Based Dialog Systems (Book)](https://link.springer.com/chapter/10.1007%2F978-981-10-5209-5_3) <br>
- [ ] [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/pdf/2003.08271.pdf)

## 2. Task oriented dialogue system

Overview
- [x] [Stanford NLP overview](https://web.stanford.edu/~jurafsky/slp3/26.pdf)
- [ ] [Microsoft NLP Group papers](https://github.com/microsoft/MSR-NLP-Projects)
- [ ] [Intelligent conversational chatbot](https://www.csie.ntu.edu.tw/~yvchen/s105-icb/syllabus.html)
- [ ] [MiuLab work](https://www.csie.ntu.edu.tw/~miulab/#home)
- [ ] [Comparison for task-oriented dialogue system](https://github.com/AtmaHou/Task-Oriented-Dialogue-Research-Progress-Survey)
- [ ] [Spoken Language Understanding papers](https://paperswithcode.com/task/spoken-language-understanding)
- [ ] [Dialogue papers](https://paperswithcode.com/task/dialogue)
- [ ] [State of the arts](http://nlpprogress.com/english/dialogue.html)
- [ ] [Meaning representation in NLP (slides)](https://gabrielstanovsky.github.io/assets/invited_talks/job/presentation.pdf)
- [ ] [ConvLab: joint NLU/DST/Policy/Simulator/NLG benchmark](https://github.com/thu-coai/Convlab-2)

### 1) NLU

NLU (Intent + Slot-filling)
- [x] [NLU services comparison](https://www.aclweb.org/anthology/W17-5522.pdf)
- [ ] [Intent detection overview](https://iopscience.iop.org/article/10.1088/1742-6596/1267/1/012059)
- [ ] [Review of Intent Detection Methods in the Human-Machine Dialogue System](https://iopscience.iop.org/article/10.1088/1742-6596/1267/1/012059/pdf)
- [x] [Query intent detection with CNN 2016](http://people.cs.pitt.edu/~hashemi/papers/QRUMS2016_HBHashemi.pdf)
- [x] [Query intent detection with NER, LSTM and similarity 2018](https://ieeexplore.ieee.org/document/8458426)
- [x] [Intent detection with dual pretrained sentence encoders 2020 (ConvRT)](https://arxiv.org/abs/2003.04807)
- [x] [Intent detection with siamese network 2020](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9082602)
- [x] [Slot filling with knowledge K-SAN 2016](https://www.csie.ntu.edu.tw/~yvchen/doc/SLT16_SyntaxSemantics.pdf)
- [x] [Slot filling with focus not attention mechanism 2017](https://arxiv.org/pdf/1608.02097.pdf)
- [x] [Slot filling with TDNN and context embedding 2020](https://www.researchgate.net/publication/342209812_Using_Deep_Time_Delay_Neural_Network_for_Slot_Filling_in_Spoken_Language_Understanding)
- [x] [Slot filling with sparse attention 2017](https://arxiv.org/pdf/1709.10191.pdf)
- [x] [Slot filling with three label embedding 2020](https://arxiv.org/pdf/2003.09831.pdf)
- [x] [Joint task with BERT+self attention+intent/slot attention 2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8907842)
- [x] [Joint task with BERT+self attention+slot gating+CRF 2020](https://dl.acm.org/doi/pdf/10.1145/3379247.3379266)
- [x] [Joint task with graph LSTM 2020](https://ojs.aaai.org//index.php/AAAI/article/view/6499)
- [x] [Joint task with CM-Net 2020](https://arxiv.org/abs/1909.06937)
- [ ] [Joint task with interrelated model](https://www.aclweb.org/anthology/P19-1544.pdf)
- [x] [New concept for semantic frames](https://www.aclweb.org/anthology/2020.acl-main.186.pdf)
- [ ] [Semantic machines](https://arxiv.org/abs/2009.11423)
- [ ] [Visual Dialogue](https://arxiv.org/pdf/2005.07493.pdf)

Multi-intent
- [ ] [Two-stage multi-intent 2017](https://link.springer.com/article/10.1007%2Fs11042-016-3724-4)
- [ ] [Joint task with LSTM](https://www.aclweb.org/anthology/N19-1055.pdf)
- [x] [Multi-intent detection (multilingual)](http://ajiips.com.au/papers/V17.1/v17n1_5-12.pdf)
- [x] [Multi-intent benchmark](https://arxiv.org/pdf/2004.10087.pdf): (https://github.com/LooperXX/AGIF)

Zero-shot/Out of domain
- [ ] [Zero-shot Overview](https://arxiv.org/pdf/1707.00600.pdf)
- [x] [CDSSM (embedding match)](https://www.csie.ntu.edu.tw/~yvchen/doc/ICASSP16_ZeroShot.pdf)
- [x] [Zero-Shot Learning Across Heterogeneous Overlapping Domains (embedding match)](https://assets.amazon.science/5c/3e/0e957e1f4b609c1778e0f5576eb2/zero-shot-learning-across-heterogeneous-overlapping-domains.pdf)
- [x] [Zero-shot Intent detection: word-sense ambiguation (embedding match)](https://www.aclweb.org/anthology/P19-1568.pdf)
- [x] [Zero-shot Intent detection: capsule network](https://www.aclweb.org/anthology/D18-1348.pdf)
- [x] [Zero-shot Intent detection: capsule network 2](https://www.aclweb.org/anthology/D19-1486.pdf)
- [ ] [Zero-shot Intent detection: reading comprehension](http://nlp.cs.washington.edu/zeroshot/)
- [ ] [Zero-shot Intent detection: disentangled intent representation](https://arxiv.org/pdf/2012.01721.pdf)
- [x] [Out of scope: intent dataset 2019](https://www.aclweb.org/anthology/D19-1131.pdf?fbclid=IwAR0mRMf0PQ3IJzD9AeIscsJ6X1DCTWGCIA9dhKCMqagm-0JT64kYo_SJI9s)
- [x] [Out of scope: LMCL, LOF 2019](https://arxiv.org/pdf/1906.00434.pdf)
- [x] [Out of scope: KL divergence 2020](https://dl.acm.org/doi/pdf/10.1145/3397271.3401318)

NLU+DST
- [ ] [Joint training](https://drive.google.com/file/d/1I8iU-dLPRnC7ZxTULTso_gwhj4uQJ23U/view)

Context
- [x] [Context on joint task](https://ieeexplore.ieee.org/document/6639291)
- [x] [Context on joint task: knowledge graph](https://ieeexplore.ieee.org/document/9006162)
- [x] [Context on joint task: CASA-NLU, DiSAN](https://arxiv.org/pdf/1909.08705.pdf)

Cross Domain
- [x] [Cross Domain slot filling](https://arxiv.org/pdf/2003.09831.pdf)
- [x] [Cross Domain DST-delexicalised RNN](https://www.aclweb.org/anthology/P15-2130.pdf)
- [x] [Cross Domain DST-graph attention network](https://ojs.aaai.org//index.php/AAAI/article/view/6250)

Pretrain
- [ ] [BERT pretrained dialogue](https://arxiv.org/pdf/2004.06871.pdf)
- [ ] [ConveRT](https://arxiv.org/pdf/1911.03688.pdf)

Memory
- [x] [HMN: Hetereogenous memory network](https://arxiv.org/pdf/1909.11287.pdf)
- [x] [Mem2seq](https://arxiv.org/pdf/1901.04713.pdf)
- [x] [GLMP: Global-to-local memory pointer network](https://arxiv.org/pdf/1901.04713.pdf)
- [x] [QA with Freebase](https://arxiv.org/pdf/1506.02075.pdf)

Evaluation
- [ ] [Efficient evaluation of dialogue](https://www.amazon.science/publications/efficient-evaluation-of-task-oriented-dialogue-systems)
- [ ] [Towards Unified Dialogue System Evaluation: A Comprehensive Analysis of Current Evaluation Protocols](https://arxiv.org/abs/2006.06110)

Intent Clustering
- [ ] [Intent clustering](https://www.aclweb.org/anthology/D18-1254.pdf)
- [ ] [Dialog intent Induction with AvKmeans](https://arxiv.org/pdf/1908.11487.pdf): (https://github.com/asappresearch/dialog-intent-induction)
- [ ] [Dialog management using intent clustering](https://onlinelibrary.wiley.com/doi/epdf/10.1111/exsy.12630)
- [ ] [Semi-supervised clustering using deep metric learning and graph embedding](https://www.researchgate.net/publication/335382603_Semi-supervised_clustering_with_deep_metric_learning_and_graph_embedding)

### 2) DST

Dialogue State Tracking
- [ ] [Hidden Information State model](http://mi.eng.cam.ac.uk/~sjy/papers/ygkm10.pdf)
- [ ] [Bayesian update of dialogue state](http://mi.eng.cam.ac.uk/~sjy/papers/thyo10.pdf)
- [x] [DNN](https://www.aclweb.org/anthology/W13-4073.pdf)
- [x] [2 LSTM network: slot-value & sentence](https://assets.amazon.science/23/98/80671ef545e4927c1716279a9340/flexible-and-scalable-state-tracking-framework-for-goal-oriented-dialogue-systems.pdf)
- [x] [Neural belief tracking](https://arxiv.org/pdf/1606.03777.pdf)
- [x] [BERT-DST: 1,0](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053975): (https://github.com/laituan245/BERT-Dialog-State-Tracking)
- [x] [BERT-DST: Chao](https://arxiv.org/pdf/1907.03040.pdf): (https://github.com/guanlinchao/bert-dst)
- [x] [BERT-DST: context GLAD-RCFS](https://www.aclweb.org/anthology/N19-1057.pdf)
- [x] [BERT-DST: TRADE generator](https://arxiv.org/pdf/1905.08743.pdf)
- [x] [BERT-DST: DSTQA](https://arxiv.org/pdf/1911.06192.pdf): (https://github.com/alexa/dstqa)
- [x] [BERT-DST: CHAN-DST](https://arxiv.org/pdf/2006.01554.pdf): (https://github.com/smartyfh/CHAN-DST)
- [x] [BERT-DST: WCN-SLU](https://arxiv.org/pdf/2005.11640v3.pdf): (https://github.com/simplc/WCN-BERT)
- [x] [BERT-training1: slot-value & sentence](https://arxiv.org/pdf/2006.01554.pdf): (https://github.com/smartyfh/CHAN-DST)
- [x] [BERT-training2: direct prediction](https://arxiv.org/pdf/1907.03040.pdf): (https://github.com/guanlinchao/bert-dst)
- [x] [BERT-training3: SLU task](https://arxiv.org/pdf/2005.11640v3.pdf): (https://github.com/simplc/WCN-BERT)
- [x] [BERT-training4: direct prediction2](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053975): (https://github.com/laituan245/BERT-Dialog-State-Tracking)
- [ ] [A Simple Language Model for Task-Oriented Dialogue](https://arxiv.org/pdf/2005.00796.pdf)

### 3) NLG

NLG
- [x] [Natural Language Generation](https://pdfs.semanticscholar.org/728e/18fbf00f5a80e9a070db4f4416d66c7b28f4.pdf)
- [ ] [NLG Evaluation survey](https://arxiv.org/pdf/2006.14799.pdf)
- [x] [NLG: ZSDG mapping 2018](https://arxiv.org/pdf/1801.06176.pdf)
- [x] [NLG: SC-GPT few shot 2020](https://arxiv.org/pdf/2002.12328.pdf)

### 4) Others

Text-to-SQL
- [x] [Text-to-SQL](https://arxiv.org/pdf/2012.10309v1.pdf)

Knowledge graph
- [ ] [Fg2seq: Effectively Encoding Knowledge for End-To-End Task-Oriented Dialog](https://ieeexplore.ieee.org/document/9053667)
- [ ] [GraphDialog: Integrating Graph Knowledge into End-to-End Task-Oriented Dialogue Systems](https://arxiv.org/pdf/2010.01447.pdf)

Model Analysis (Noise perturbation)
- [x] [Do Neural Dialog Systems Use the Conversation History Effectively? An Empirical Study](https://arxiv.org/pdf/1906.01603.pdf)


## 3. Attention Mechanism
- [x] [Attention on BERT](https://drive.google.com/file/d/1e0WA8t0T0xvngTuMk01rbMeJySxynGE8/view) <br>
- [x] [Multimodal Explanations by attention on images](http://openaccess.thecvf.com/content_cvpr_2018/papers/Park_Multimodal_Explanations_Justifying_CVPR_2018_paper.pdf) <br>
- [x] [Adaptive Attention for Image Captioning](https://arxiv.org/pdf/1612.01887.pdf) <br>
- [x] [Visual Attention for Image Captioning](https://arxiv.org/pdf/1502.03044.pdf) <br>

## 4. Label Embedding
- [ ] [Label embedding](https://reader.elsevier.com/reader/sd/pii/S0031320319300184?token=17658B5D93506DABE37CD07981324BD915C4C626AD0D5EAB9039D9E2397E8A55034D0F7292AFF0AC7B967508AC2822B1)
- [ ] [Relation Network](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf)
- [x] [Meta Dataset](https://arxiv.org/pdf/1903.03096.pdf)

## 5. Amazon Science in Conversational AI
- [ ] [Work from Alexa](https://www.amazon.science/publications?f0=0000016e-2ff0-da81-a5ef-3ff057f10000&s=0)
- [ ] [Work from Interspeech](https://www.amazon.science/conferences-and-events/interspeech-2020)
- [ ] [Work from Alexa Speech](https://www.aboutamazon.com/news/aws/meet-alexas-speech-coach)
- [ ] [Work from Hakkani-Tur](https://www.amazon.science/author/dilek-hakkani-tur)


----


## Conferences
ML conferences1: http://www.guide2research.com/topconf/machine-learning
ML conferences2: https://blog.csdn.net/devenlau/article/details/82660886
ML conferences ranking: https://blog.csdn.net/cpp12341234/article/details/50886540
Acceptance rate: https://github.com/lixin4ever/Conference-Acceptance-Rate
Deadlines: https://aideadlin.es/?sub=ML,CV,NLP,RO,SP,DM

|      Conference     |  Day   | Submission Deadline | Tier | Target |
| ------------------- | ------ | ------------------- | ---- | ------ |
|  IWSDS 2021         |  5/18  | 1/10 (temp)         |  2   |        |
|  IJCAI 2021         |  8/21  | 1/13 (1/20)         |  1   |        |
|  ICDM 2021          |  7/14  | 1/15                |  2   |        |
|  ACL-IJCNLP 2021    |  8/1   | 1/25 (2/1)          |  1   |    v   |
|  ICML 2021          |  7/18  | 1/28 (2/4)          |  1   |        |
|  SIGIR 2021         |  7/11  | 2/2 (2/9)           |  1   |        |
|  KDD 2021           |  8/14  | 2/8                 |  1   |    v   |
|  ICDAR 2021         |  9/5   | 2/8                 |  2   |        |
|  NAACL SRW          |  6/6   | 2/12                |  w   |        |  
|  UAI 2021           |  8/3   | 2/20                |  2   |    b   |
| ------------------- | ------ | ------------------- | ---- | ------ |
|  ICCV 2021          |  10/10 | 3/17                |  1   |    a   |
|  KR 2021            |  11/6  | 3/24 (3/31)         |  2   |    b   |
|  ECML PKDD 2021     |  9/13  | 3/26 (4/2)          |  2   |        |
|  Interspeech 2021   |  8/30  | 3/26                |  1   |    a   |
|  [RepL4NLP 2021](https://sites.google.com/view/repl4nlp-2021/call-for-papers?authuser=0)      |  8/5   | 4/26                |  w   |    a   |
|  [ACL workshop](https://doc2dial.github.io/workshop2021/)                                     |  8/5   | 4/26                |  w   |    a   |
|  RANLP 2021         |  9/6   | 5/15                |  2   |    a   |
|  ICMI 2021          |  10/18 | 5/26                |  2   |    b   |
|  SIGDIAL 2021       |  7/29  | 4/2                 |  2   |    a   |
|  NIPS 2021          |  12/6  | 5/27 (6/5)          |  1   |    b   |
|  EMNLP 2021 (CONLL) |  11/7  | 6/3                 |  1   |    a   |
|  EMNLP 2021 workshop|  11/7  |                     |  w   |    a   |
|  IWSDS 2021         |  11/15 |                     |  2   |    a   |
|  ASRU 2021          |  12/13 | 7/5                 |  2   |    b   |
|  SLT 2021           |  1/19  | 8/14                |  2   |    a   |
|  AAAI 2021          |  2/2   | 9/1 (9/9)           |  1   |    a   |
|  AAAI student 2021  |  2/2   | 9/18                |  2   |    w   |
|  ICLR 2021          |  5/4   | 10/2                |  1   |    b   |
|  AISTATS 2021       |  4/13  | 10/8 (10/15)        |  2   |        |
|  WWW 2021           |  4/19  | 10/12 (10/19)       |  1   |        |
|  EACL 2021          |  4/21  | 10/7                |  2   |    a   |
|  ICASSP 2021        |  6/6   | 10/21               |  1   |    a   |
|  AAAI 2021 workshop |  2/8   | 11/9                |  w   |    w   |
|  CVPR 2021          |  7/21  | 11/16               |  1   |    t   |
|  NAACL-HLT 2021     |  6/6   | 11/23               |  1   |    a   |
ACML
IEEE transactions on audio, speech and language processing
ICCDE 3

v: ok <br>
a: first to submit <br>
b: second to submit <br>
w: workshop <br>