### Translation for Code-Switching

This is the official repository for the paper titled [From Machine Translation to Code-Switching: Generating High-Quality Code-Switched Text](TODO) accepted at ACL 2021.

We follow the architecture from the repository [UnsupervisedMT](https://github.com/facebookresearch/UnsupervisedMT)

```
data
|- PRETRAIN_COMBINED
|- OPUS-LEX
|- OPUS-EMT
|- combinedcs
```

#### Paper Abstract

Generating code-switched text is a problem of growing interest, especially given the scarcity of corpora containing large volumes of real code-switched text. In this work, we adapt a state-of-the-art neural machine translation model to generate Hindi-English code-switched sentences starting from monolingual Hindi sentences. We outline a carefully designed curriculum of pretraining steps, including the use of synthetic code-switched text, that enable the model to generate high-quality code-switched text. Using text generated from our model as data augmentation, we show significant reductions in perplexity on a language modeling task, compared to using text from other generative models of CS text. We also show improvements using our text for a downstream code-switched natural language inference task. Our generated text is further subjected to a rigorous evaluation using a human evaluation study and a range of objective metrics, where we show performance comparable (and sometimes even superior) to code-switched text obtained via crowd workers who are native Hindi speakers.