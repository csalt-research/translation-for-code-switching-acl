### Translation for Code-Switching

This is the official repository for the paper titled [From Machine Translation to Code-Switching: Generating High-Quality Code-Switched Text](TODO) accepted at ACL 2021.

#### Datasets

There are 4 datasets used in the training - <br>
| Data | Description |
|:----:|:------------|
| PRETRAIN | Hi-En parallel dataset comprising of sentence pairs from OpSub-{Hi,En} and IITB Hi-En Parallel dataset. |
| OpSub-LEX | Hi-CS Synthetic dataset with LEX style of generation on OpSub-Hi |
| OpSub-EMT | Hi-CS Synthetic dataset with EMT style of generation on OpSub-Hi |
| AllCS | Real Hi-CS dataset |

The directory structure for training is like this -

```
data
|- PRETRAIN
|- OpSub-LEX
|- OpSub-EMT
|- AllCS
```

Please contact the authors for the pretraining and Synthetic CS datasets created.

#### Model Architecture

We follow the architecture from the repository [UnsupervisedMT](https://github.com/facebookresearch/UnsupervisedMT). The model comprises of three layers of stacked Transformer encoder and decoder layers, two of which are shared and the remaining layer is private to each language. Monolingual Hindi (i.e. the source language) has its own private encoder and decoder layers while English and Hindi-English CS text jointly make use of the remaining private encoder and decoder layers. In our model, the target language is either English or CS text.


#### Paper Abstract

Generating code-switched text is a problem of growing interest, especially given the scarcity of corpora containing large volumes of real code-switched text. In this work, we adapt a state-of-the-art neural machine translation model to generate Hindi-English code-switched sentences starting from monolingual Hindi sentences. We outline a carefully designed curriculum of pretraining steps, including the use of synthetic code-switched text, that enable the model to generate high-quality code-switched text. Using text generated from our model as data augmentation, we show significant reductions in perplexity on a language modeling task, compared to using text from other generative models of CS text. We also show improvements using our text for a downstream code-switched natural language inference task. Our generated text is further subjected to a rigorous evaluation using a human evaluation study and a range of objective metrics, where we show performance comparable (and sometimes even superior) to code-switched text obtained via crowd workers who are native Hindi speakers.
