# Model Architecture
The underlying architecture of BERT is a multi-layer Transformer encoder, which is inherently bidirectional in nature. Two models are proposed in the paper.

BERTBASE - 12 Transformer blocks, 12 self-attention heads, 768 is the hidden size
BERTLARGE - 24 transformer blocks, 16 self-attention heads, 1024 is the hidden size
The model size of BERTBASE and Open AI’s GPT was chosen to be the same.


# Input-Output Representations
BERT uses WordPiece embeddings with a 30,000 token vocabulary. The first token of every sequence is ([CLS]). The final hidden state corresponding to the [CLS] token is used as the aggregate sequence representation.
To deal with sentence pairs, BERT uses a special token [SEP] to separate the two sentences. A learned embedding is added to every token indicating whether it is the first or the second sentence. The input embedding for each token is obtained by adding the corresponding token embedding (WordPiece embedding), segment embedding (first / second sentence) and position embedding (as in Transformers).


# BERT pre-training
BERT is pre-trained using two unsupervised tasks.

## Masked LM
The bidirectional model is more powerful than either a left-to-right model or the shallow concatenation of a left-to-right and right-to-left model.
In order to train a deep bidirectional representation, some percentage (15% in the paper) of the input tokens are masked at random, and those masked tokens are predicted using an output softmax over the vocabulary. This is called a masked LM. The masking is performed by replacing the token with a [MASK] token. Now since the [MASK] token does not appear during fine-tuning, the [MASK] token is used 80% of the time. For 10% of the selected tokens (from the 15%) a random token is used to replace it and the token is kept unchanged for the rest 10%. The token is then predicted using cross-entropy loss.

## Next Sentence Prediction (NSP)
To understand the relationship between two sentences (which is not captured by language modelling), a binarized NSP task is formulated. Here, when choosing the sentences A and B (refer to the model pre-training figure above) for each pre-training example, 50% of the time B is the actual next sentence and the rest 50% of the time, a random sentence from the corpus is used. The vector C (without fine-tuning) is used for NSP. This is helpful for tasks like Question Answering and Natural Language Inference.

## Pre-training data
It is useful for BERT to use a document-level corpus rather than a shuffled sentence-level corpus. BERT 9as in the paper) uses the BookCorpus (800M words) and English Wikipedia (2500M words).

## Fine-tuning BERT
Instead of independently encoding text (sentence) pairs and then applying bidirectional cross attention, BERT uses the Transformer model architecture’s self-attention mechanism. Encoding the concatenated text (sentence) pair with self-attention effectively incorporates bidirectional cross attention between the two sentences.

The fine-tuning is performed for all the parameters and the task-specific inputs and outputs of the downstream task are plugged for fine-tuning.

A and B are the sentence pairs in case of paraphrasing
A and B are hypothesis-premise pairs in the entailment task
A and B are question-passage pairs in question answering
A and B are the text and Φ in text classification or sequence tagging task
At the output, for the token-level tasks (sequence tagging, question answering), the token representations are fed into the output layer. For the sentence-level tasks, the representation of the [CLS] token is fed to the output layer for classification.

# Citation
```
@article{DBLP:journals/corr/abs-1810-04805,
  author       = {Jacob Devlin and
                  Ming{-}Wei Chang and
                  Kenton Lee and
                  Kristina Toutanova},
  title        = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
                  Understanding},
  journal      = {CoRR},
  volume       = {abs/1810.04805},
  year         = {2018},
  url          = {http://arxiv.org/abs/1810.04805},
  eprinttype    = {arXiv},
  eprint       = {1810.04805},
  timestamp    = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
