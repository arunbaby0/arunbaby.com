---
title: "Character-Level Language Models"
day: 50
related_dsa_day: 50
related_speech_day: 50
related_agents_day: 50
collection: ml_system_design
categories:
 - ml-system-design
tags:
 - nlp
 - rnn
 - tokenization
 - sequence-modeling
 - lstm
difficulty: Medium
subdomain: "Sequence Modeling"
tech_stack: PyTorch, Andrej Karpathy's minGPT
scale: "Learning language one letter at a time"
companies: OpenAI, DeepMind, Google
---

**"Before machines could write essays, they had to learn to spell."**

## 1. Problem Statement

Modern LLMs (GPT-4) operate on **tokens** (sub-words).
But to understand *why*, we must study the alternatives.
**Character-Level Modeling** is the task of predicting the next character in a sequence.
- Input: `['h', 'e', 'l', 'l']`
- Target: `'o'`

Why build a Char-LM?
1. **Infinite Vocabulary**: No "Unknown Token" `<UNK>` issues. It can generate any word.
2. **Robustness**: Handles typos (`"helo"`) and biological sequences (`"ACTG"`) natively.
3. **Simplicity**: Vocab size is 100 (ASCII), not 50,000 (GPT-2 BPE).

---

## 2. Understanding the Requirements

### 2.1 The Context Problem
Prediction depends on long-range dependencies.
- "The cat sat on the **m**..." -> **a** -> **t**. (Local context).
- "I grew up in France... I speak fluent **F**..." -> **r** -> **e**... (Global context).

A Char-LM must remember history that is 5x longer than a Word-LM (since avg word length is 5 chars).
If a sentence is 20 words, the Char-LM sees 100 steps.

### 2.2 Sparsity vs Density
- **One-Hot Encoding**: Characters are dense. `a` is always vector `[1, 0, ...]`.
- **Embedding**: We still learn a dense vector for 'a', capturing nuances like "vowels cluster together".

---

## 3. High-Level Architecture

We compare **RNN** (Recurrent) vs **Transformer** (Attention).

**RNN Style (The Classic)**:
``
[h] -> [e] -> [l] -> [l]
 | | | |
 v v v v
(S0)-> (S1)-> (S2)-> (S3) -> Predict 'o'
``
State `S3` must verify "We are in the word 'hello'".

**Transformer Style (Modern)**:
Input: `[h, e, l, l]`
Attention: `l` attends to `h`, `e`, `l`.
Output: `Prob(o)`

---

## 4. Component Deep-Dives

### 4.1 Tokenization Trade-offs

| Strategy | Vocab Size | Sequence Length (for 1000 words) | OOV Risk |
|----------|------------|----------------------------------|----------|
| **Character** | ~100 | ~5000 chars | None |
| **Word** | ~1M | 1000 words | High (Rare names) |
| **Subword (BPE)** | ~50k | ~1300 tokens | Low |

**Why BPE won**: It balances the trade-off. It keeps sequence length manageable (for Transformer O(N^2) attention) while handling rare words via characters.

### 4.2 The Softmax Bottleneck
Predicting 1 out of 100 chars is cheap.
Predicting 1 out of 50,000 tokens is expensive (large Matrix Mul at the end).
Char-LMs are incredibly fast at the *final layer*, but incredibly slow at the *layers/inference* (requiring more steps).

---

## 5. Data Flow: Training Pipeline

1. **Raw Text**: "Hello world"
2. **Vectorizer**: `[H, e, l, l, o, _, w, o, r, l, d]` -> `[8, 5, 12, 12, 15, 0, ...]`
3. **Windowing**: Create pairs `(Input, Target)`.
 - `[8, 5, 12]` -> `12` ("Hel" -> "l")
 - `[5, 12, 12]` -> `15` ("ell" -> "o")
4. **Loss Calculation**: Cross Entropy Loss on the prediction.

---

## 6. Implementation: RNN Char-LM

``python
import torch
import torch.nn as nn

class CharRNN(nn.Module):
 def __init__(self, vocab_size, hidden_size, n_layers=1):
 super().__init__()
 self.embedding = nn.Embedding(vocab_size, hidden_size)
 self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
 self.fc = nn.Linear(hidden_size, vocab_size)
 
 def forward(self, x, hidden=None):
 # x: [Batch, Seq_Len] (e.g., indices of chars)
 embeds = self.embedding(x)
 
 # rnn_out: [Batch, Seq_Len, Hidden]
 rnn_out, hidden = self.rnn(embeds, hidden)
 
 # Predict next char for EVERY step in sequence
 logits = self.fc(rnn_out)
 return logits, hidden

 def generate(self, start_char, max_len=100):
 # Inference Loop
 curr_char = torch.tensor([[char_to_ix[start_char]]])
 hidden = None
 out = start_char
 
 for _ in range(max_len):
 logits, hidden = self.forward(curr_char, hidden)
 probs = nn.functional.softmax(logits[0, -1], dim=0)
 next_ix = torch.multinomial(probs, 1).item()
 
 out += ix_to_char[next_ix]
 curr_char = torch.tensor([[next_ix]])
 
 return out
``

---

## 7. Scaling Strategies

### 7.1 Truncated Backpropagation through Time (TBPTT)
You cannot backpropagate through a book with 1 million characters. Gradients vanish or explode.
We process chunks of 100 characters.
**Crucial**: We pass the `hidden` state from Chunk 1 to Chunk 2, but we *detach* the gradient history. The model remembers the context, but doesn't try to learn across the boundary.

---

## 8. Failure Modes

1. **Hallucinated Words**: "The quxijumped over..."
 - Since it spells letter-by-letter, it can invent non-existent words that "sound" pronounceable.
2. **Incoherent Grammar**: It closes parentheses `)` that were never opened `(`.
 - LSTMs struggled with this (counting). Transformers fixed it.

---

## 9. Real-World Case Study: Andrej Karpathy's minGPT

The famous blog post "The Unreasonable Effectiveness of Recurrent Neural Networks" trained a Char-RNN on:
- **Shakespeare**: Resulted in fake plays.
- **Linux Kernel Code**: Resulted in C code that *almost* compiled (including comments and indentation).
This proved that neural nets learn **Syntactic Structure** just from statistical co-occurrence.

---

## 10. Connections to ML Systems

This connects to **Custom Language Modeling in Speech** (Speech ).
- ASR systems use Char-LMs to correct spelling in noisy transcripts.
- If ASR hears "Helo", the Char-LM says "l followed by o is unlikely after He, it should be 'll'".

---

## 11. Cost Analysis

**Training**: Cheap. A Char-RNN trains on a laptop CPU in minutes.
**Inference**: Expensive.
- To generate a 1000-word essay (5000 chars), you run the model 5000 times (serial).
- A Token-LM runs 700 times.
- This 7x latency penalty is why Char-LMs are not used for Chatbots.

---

## 12. Key Takeaways

1. **Granularity Matters**: Breaking text down to atoms (chars) simplifies vocabulary but complicates structure learning.
2. **Embeddings**: Even characters need embeddings. 'A' and 'a' should be close vectors.
3. **Subword Dominance**: BPE won because it is the "Goldilocks" zone—short sequences, manageable vocab.

---

**Originally published at:** [arunbaby.com/ml-system-design/0050-character-level-language-models](https://www.arunbaby.com/ml-system-design/0050-character-level-language-models/)

*If you found this helpful, consider sharing it with others who might benefit.*
