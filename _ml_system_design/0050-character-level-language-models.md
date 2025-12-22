---
title: "Character-Level Language Models"
day: 50
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
related_dsa_day: 50
related_speech_day: 50
related_agents_day: 50
---

**"Before machines could write essays, they had to learn to spell."**

## 1. Introduction: The Atomic Unit of Language

When you read, do you read L-E-T-T-E-R-S or do you read *Words*? Probably words.
Modern LLMs (like GPT-4) read chunks called **Tokens** (roughly syllables).
But the simplest, most elegant way to model language is **Character by Character**.

A **Character-Level Language Model** (Char-LM) predicts the next character in a sequence.
Input: `"hell"` -> Output: `"o"`.

Why study this?
1. **No Out-of-Vocabulary (OOV) Problems**: You never encounter a "word" you don't know. You can generate "supercalifragilistic..." even if you've never seen it, just by knowing phonetic patterns.
2. **Robustness**: It handles typos (`"hrello"`) and new slang (`"rizz"`) gracefully.
3. **Simplicity**: The "Vocabulary" is just 26 letters + punctuation (~100 items), not 50,000 tokens.

---

## 2. Architecture: RNNs and LSTMs

Before Transformers took over, **Recurrent Neural Networks (RNNs)** were the kings of Char-LMs.
Many junior engineers haven't built an RNN, but it's crucial for understanding "State".

### 2.1 The Recurrent Loop
Regular Feed-Forward networks map `Input -> Output`.
RNNs map `Input + Past_State -> Output + New_State`.

```python
# Pseudo-code for a Char-RNN Step
def rnn_step(char_input, hidden_state):
    # Combine input and history
    combined = linear(char_input + hidden_state)
    
    # Update memory
    new_hidden_state = tanh(combined)
    
    # Predict next char
    prediction = softmax(linear(new_hidden_state))
    
    return prediction, new_hidden_state
```

If we feed `"h"`, it updates its hidden state to "remembering 'h'".
Feed `"e"`, it remembers "'h', 'e'".
Feed `"l"`, it remembers "'h', 'e', 'l'".
Now it strongly predicts `"l"` (for "hell") or `"p"` (for "help").

### 2.2 The "Vanishing Gradient" Problem
Simple RNNs forget things quickly. If the sentence is "The clouds in the sky are [?]", the RNN sees "sky" and predicts "blue".
If the sentence is "I grew up in France... (1000 words later) ... I speak fluent [?]", the RNN forgets "France". It guesses "English".

This led to **LSTMs (Long Short-Term Memory)**, which have specific "gates" to keep information for thousands of steps.

---

## 3. Tokenization Trade-offs

Why don't we use Character Models for everything?

| Feature | Character-Level | Word-Level | Subword (BPE/Tokenizer) |
|---------|-----------------|------------|-------------------------|
| **Vocab Size** | Small (~100) | Huge (1M+) | Balanced (50k) |
| **Context Length** | Very Long (1000 steps = 2 sentences) | Short (20 steps = 2 sentences) | Medium |
| **Compute Cost** | Expensive (Many small steps) | Cheap (Few big steps) | Optimal |
| **Meaning** | Hard (Letters have no meaning) | Easy (Words have meaning) | Mix |

**Char-Level is computationally inefficient.**
To process 1,000 words, a Char-LM takes ~5,000 steps. A Token-LM takes ~700 steps. Transformers simply cannot handle sequences that long (Self-Attention is `O(N^2)`).
This is why GPT uses **BPE (Byte Pair Encoding)**, finding a middle ground.

---

## 4. Modern Applications of Char-LMs

Despite the dominance of Token-LMs, Char-LMs are still critical in:

### 4.1 Code Generation
Code has strict syntax. `}` must close `{`.
Char-LMs are surprisingly good at learning syntax (indentation, brackets) because they see every single keystroke.

### 4.2 Language Identification
To guess if text is "French" or "Spanish", you don't need words. You need character patterns.
- Sequence of `eau` -> French.
- Sequence of `ñ` -> Spanish.
FastText (by Facebook) uses character n-grams for this.

### 4.3 Handling "Alien" Languages (Bioinformatics)
DNA sequences (`ACTG`) are essentially char-level languages. There are no "words" in DNA.
Models like **GenomicLM** are Char-LMs.

---

## 5. Training your own "MiniGPT"

Today, you can train a Char-level GPT on your laptop in 10 minutes (thanks to Andrej Karpathy's `nanoGPT`).

1. **Data**: Text file (Shakespeare).
2. **Tokenizer**: `list(set(text))` (Result: `a, b, c...`).
3. **Model**: Small Transformer (Embedding size 64, 4 layers).
4. **Train**: Predict next char.

**Result**:
`"The kyng hath sayd, that thou shalt die!"`
It invents fake Old English words! It learns spelling, grammar, and punctuation purely from examples.

---

## 6. Summary

Understanding Character-Level models removes the "magic" of AI.
It's just statistics.
- "t" is likely followed by "h".
- "th" is likely followed by "e".
- "the" is likely followed by " ".

If you scale this simple pattern matching up to 1 Trillion parameters, you get ChatGPT. But at its core, it's just predicting the next symbol.

---

**Originally published at:** [arunbaby.com/ml-system-design/0050-character-level-language-models](https://www.arunbaby.com/ml-system-design/0050-character-level-language-models/)

*If you found this helpful, consider sharing it with others who might benefit.*
