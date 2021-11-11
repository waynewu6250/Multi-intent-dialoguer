# The Schema-Guided Dialogue Dataset

Please refer to the github link [here](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) to download the data.

***************
(Excerpted from schema-guided README.md)

**Contact -** schema-guided-dst@google.com

## Overview

The Schema-Guided Dialogue (SGD) dataset consists of over 20k annotated
multi-domain, task-oriented conversations between a human and a virtual
assistant. These conversations involve interactions with services and APIs
spanning 20 domains, ranging from banks and events to media, calendar, travel,
and weather. For most of these domains, the dataset contains multiple different
APIs, many of which have overlapping functionalities but different interfaces,
which reflects common real-world scenarios. The wide range of available
annotations can be used for intent prediction, slot filling, dialogue state
tracking, policy imitation learning, language generation, user simulation
learning, among other tasks in large-scale virtual assistants. Besides these,
the dataset has unseen domains and services in the evaluation set to quantify
the performance in zero-shot or few shot settings.

**The dataset is provided "AS IS" without any warranty, express or implied.
Google disclaims all liability for any damages, direct or indirect, resulting
from the use of this dataset.**

## Updates

**07/05/2020** - Test set annotations released. User actions and service calls
made during the dialogue are also released for all dialogues.

**10/14/2019** - DSTC8 challenge concluded. Details about the submissions to the
challenge may be found in the [challenge overview
paper](https://arxiv.org/pdf/2002.01359.pdf).

**10/07/2019** - Test dataset released without the dialogue state annotations.

**07/23/2019** - Train and dev sets are publicly released as part of [DSTC8
challenge](dstc8.md).

## Important Links

* [Paper for dataset and DST baseline](https://arxiv.org/pdf/1909.05855.pdf)
* [DSTC8 overview paper](https://arxiv.org/pdf/2002.01359.pdf)
* [Code for DST
  baseline](https://github.com/google-research/google-research/tree/master/schema_guided_dst)
* [Natural language generation](https://arxiv.org/pdf/2004.15006.pdf)
* [Blog post announcing the
  dataset](https://ai.googleblog.com/2019/10/introducing-schema-guided-dialogue.html)

## Data
The dataset consists of schemas outlining the interface of different APIs, and
annotated dialogues. The dialogues have been generated with the help of a
dialogue simulator and paid crowd-workers. The data collection approach is
summarized in our [paper](https://arxiv.org/pdf/1801.04871.pdf).