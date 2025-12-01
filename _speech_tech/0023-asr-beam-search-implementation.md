---
title: "ASR Beam Search Implementation"
day: 23
collection: speech_tech
categories:
  - speech-tech
tags:
  - asr
  - ctc
  - decoding
  - beam-search
  - kenlm
subdomain: "Speech Recognition Decoders"
tech_stack: [Python, NumPy, KenLM]
scale: "O(T×V×W) time"
companies: [Nuance, Google, Amazon, Apple]
related_dsa_day: 23
related_ml_day: 23
---

**Implementing the core decoding logic of modern Speech Recognition systems, handling alignment, blanks, and language models.**

## Problem

Standard Beam Search (for Transformers) generates one token at a time.
CTC Beam Search processes a sequence of probabilities over time steps.

**The Rules of CTC:**
1.  **Blank Token (`<blk>`):** A special token that represents "silence" or "no transition".
2.  **Collapse Repeats:** `AA` -> `A`.
3.  **Blanks Separate:** `A <blk> A` -> `AA`. (This allows us to spell words like "Aaron").

**Example:**
- Output: `h h e e l l l <blk> l l o`
- Collapse: `h e l <blk> l o`
- Remove Blanks: `h e l l o`

## Background: A Brief History of Decoders

To understand CTC, we must understand what came before.

1.  **HMM-GMM (1980s-2000s):** The "Dark Ages". We modeled speech as a Hidden Markov Model. The decoder was a massive Viterbi search over a graph of Phonemes -> Triphones -> Words -> Sentences. It was complex, fragile, and required expert linguistic knowledge.
2.  **HMM-DNN (2010-2015):** We replaced the Gaussian Mixture Models (GMMs) with Deep Neural Networks (DNNs) to predict phoneme probabilities. The decoder remained the same Viterbi beast.
3.  **CTC (2006/2015):** Alex Graves introduced CTC. Suddenly, we didn't need phoneme alignment. The model learned to align itself. The decoder became a simple "Prefix Search".
4.  **RNN-T (Transducer):** An upgrade to CTC that removes the conditional independence assumption. It's the standard for streaming ASR today (Siri, Google Assistant).
5.  **Attention (Whisper):** The decoder is just a text generator (like GPT) that attends to the audio. Simple, but hard to stream.

Standard Beam Search (for Transformers) generates one token at a time.
CTC Beam Search processes a sequence of probabilities over time steps.

**The Rules of CTC:**
1.  **Blank Token (`<blk>`):** A special token that represents "silence" or "no transition".
2.  **Collapse Repeats:** `AA` -> `A`.
3.  **Blanks Separate:** `A <blk> A` -> `AA`. (This allows us to spell words like "Aaron").

**Example:**
- Output: `h h e e l l l <blk> l l o`
- Collapse: `h e l <blk> l o`
- Remove Blanks: `h e l l o`

## Why Greedy Fails in Speech

Imagine the audio for "The cat".
- Frame 10: P(The) = 0.6, P(A) = 0.4
- Frame 11: P(cat) = 0.4, P(car) = 0.6

Greedy might pick "The car".
But maybe "A cat" was more likely overall if we looked at the whole sequence.
Beam Search allows us to keep "A" alive as a hypothesis until we see "cat", which confirms it.

## High-Level Architecture: CTC Decoding Flow

```ascii
+-------------+    +-------------+    +-------------+
| Audio Frame | -> | Acoustic Mod| -> | Prob Matrix |
+-------------+    +-------------+    +-------------+
                                           | (T x V)
                                           v
                                   +------------------+
                                   | CTC Beam Search  |
                                   +------------------+
                                           |
                      +--------------------+--------------------+
                      |                                         |
               (Expand Prefix)                           (Score with LM)
                      |                                         |
                      v                                         v
             +----------------+                        +----------------+
             | New Hypotheses | <--------------------- | Language Model |
             +----------------+                        +----------------+
```

## Algorithm: CTC Beam Search

This is more complex than standard Beam Search because we have to track two probabilities for each hypothesis:
1.  `P_b`: Probability ending in a **Blank**.
2.  `P_nb`: Probability ending in a **Non-Blank**.

Why? Because if the next token is the *same* as the last one, the behavior depends on whether there was a blank in between.
- `A` + `A` = `A` (Merge)
- `A` + `<blk>` + `A` = `AA` (No Merge)

### The State Space
A hypothesis is defined by the text prefix (e.g., "hel").
At each time step `t`, we update the probabilities of all active prefixes based on the acoustic probabilities `y_t`.

### Python Implementation

This is a simplified version of the algorithm used in libraries like `pyctcdecode`.

```python
import numpy as np
from collections import defaultdict

def ctc_beam_search(probs, vocab, beam_width=10):
    """
    probs: (T, V) numpy array of probabilities
    vocab: list of characters (index 0 is <blk>)
    """
    T, V = probs.shape
    
    # beam: dict mapping prefix -> probability
    # We work in log domain to avoid underflow
    beam = defaultdict(lambda: float('-inf'))
    
    # Initialize with empty prefix
    # We track two scores: (score_blank, score_non_blank)
    beam[()] = (0.0, float('-inf'))
    
    for t in range(T):
        next_beam = defaultdict(lambda: (float('-inf'), float('-inf')))
        
        # Get top-K prefixes from previous step
        # Sort by total score (logsumexp of blank and non-blank)
        sorted_prefixes = sorted(
            beam.items(),
            key=lambda x: np.logaddexp(x[1][0], x[1][1]),
            reverse=True
        )[:beam_width]
        
        for prefix, (p_b, p_nb) in sorted_prefixes:
            # 1. Handle Blank Token (Index 0)
            pr_blank = probs[t, 0]
            # If we add a blank, the prefix doesn't change.
            # New blank score = log(P(blank at t)) + log(P_total at t-1)
            n_p_b, n_p_nb = next_beam[prefix]
            n_p_b = np.logaddexp(n_p_b, pr_blank + np.logaddexp(p_b, p_nb))
            next_beam[prefix] = (n_p_b, n_p_nb)
            
            # 2. Handle Non-Blank Tokens
            for v in range(1, V):
                pr_char = probs[t, v]
                char = vocab[v]
                
                # Case A: Repeat character (e.g., "l" -> "l")
                if len(prefix) > 0 and prefix[-1] == char:
                    # 1. Merge (from non-blank): "l" + "l" -> "l"
                    # We update the NON-blank score of the SAME prefix
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = np.logaddexp(n_p_nb, pr_char + p_nb)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    
                    # 2. No Merge (from blank): "l" + <blk> + "l" -> "ll"
                    # We extend the prefix
                    new_prefix = prefix + (char,)
                    n_p_b, n_p_nb = next_beam[new_prefix]
                    n_p_nb = np.logaddexp(n_p_nb, pr_char + p_b)
                    next_beam[new_prefix] = (n_p_b, n_p_nb)
                    
                # Case B: New character
                else:
                    new_prefix = prefix + (char,)
                    n_p_b, n_p_nb = next_beam[new_prefix]
                    # We can come from blank OR non-blank
                    n_p_nb = np.logaddexp(n_p_nb, pr_char + np.logaddexp(p_b, p_nb))
                    next_beam[new_prefix] = (n_p_b, n_p_nb)
                    
        beam = next_beam
        
    # Final cleanup: Return top hypothesis
    best_prefix = max(beam.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1]))[0]
    return "".join(best_prefix)

# Mock Data
# T=3, V=3 (<blk>, A, B)
probs = np.log(np.array([
    [0.1, 0.8, 0.1], # Mostly A
    [0.8, 0.1, 0.1], # Mostly Blank
    [0.1, 0.1, 0.8]  # Mostly B
]))
vocab = ['<blk>', 'A', 'B']
print(ctc_beam_search(probs, vocab)) # Output: "AB"
```

## Adding a Language Model (LM)

The acoustic model (AM) is good at sounds, but bad at grammar.
- AM hears: "I want to wreck a nice beach."
- LM knows: "I want to recognize speech."

We fuse them during the beam search.
`Score = Score_AM + (alpha * Score_LM) + (beta * Word_Count)`

- **Alpha:** Weight of the LM (usually 0.5 - 2.0).
- **Beta:** Word insertion bonus (encourages longer sentences).

### KenLM Integration
In production, we use **KenLM**, a highly optimized C++ library for n-gram language models.
We query the LM every time we append a space character (end of a word).

```python
# Pseudo-code for LM scoring
if char == ' ':
    word = get_last_word(new_prefix)
    lm_score = lm.score(word)
    n_p_nb += alpha * lm_score
```

## Measuring Success: Word Error Rate (WER)

In classification, we use Accuracy. In ASR, we use WER.
`WER = (S + D + I) / N`
- **S (Substitutions):** "cat" -> "hat"
- **D (Deletions):** "the cat" -> "cat"
- **I (Insertions):** "cat" -> "the cat"
- **N:** Total number of words in the reference.

**Note:** WER can be > 100% if you insert a lot of garbage!

### Python Implementation of WER

This uses the Levenshtein Distance algorithm (Dynamic Programming!).

```python
def calculate_wer(reference, hypothesis):
    r = reference.split()
    h = hypothesis.split()
    n = len(r)
    m = len(h)
    
    # DP Matrix
    d = np.zeros((n+1, m+1))
    
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                sub = d[i-1][j-1] + 1
                ins = d[i][j-1] + 1
                rem = d[i-1][j] + 1
                d[i][j] = min(sub, ins, rem)
                
    return d[n][m] / n

ref = "the cat sat on the mat"
hyp = "the cat sat on mat" # Deletion
print(calculate_wer(ref, hyp)) # 1/6 = 16.6%
```

## Streaming ASR: The Infinite Loop

Standard Beam Search waits for the end of the file. In a live meeting, you can't wait.
We need **Streaming Decoding**.

**Challenges:**
1.  **Latency:** Users expect text to appear < 200ms after they speak.
2.  **Stability:** The decoder might change its mind. "I want to..." -> "I want two...". This "flicker" is annoying.

**Solution:**
- **Partial Results:** Output the current best hypothesis every 100ms.
- **Endpointing:** If the user pauses for > 500ms, finalize the sentence and clear the beam history.
- **Stability Heuristic:** Only display words that have been stable for 3 frames.

## Hot-Word Boosting (Contextual Biasing)

If you build an ASR for a medical app, it needs to know "Hydrochlorothiazide". A generic LM won't know it.
We use **Contextual Biasing**.

**Algorithm:**
1.  Build a **Trie** of hot-words (e.g., contact names, drug names).
2.  During Beam Search, traverse the Trie with the current prefix.
3.  If we are inside a hot-word node, add a bonus score.

```python
# Pseudo-code
if current_prefix in hot_word_trie:
    score += boosting_weight
```

## Debugging the Decoder

When your WER (Word Error Rate) is high, how do you know if it's the Model or the Decoder?

1.  **Force Alignment:** Feed the *correct* transcript into the decoder and see its probability. If the probability is high but the decoder didn't pick it, your **Beam Width** is too small (search error).
2.  **Greedy Check:** If Greedy Search gives garbage, your **Model** is bad (modeling error).
3.  **LM Weight Tuning:** Grid search `alpha` and `beta` on a validation set. A bad alpha can ruin a perfect acoustic model.

## Appendix A: The Mathematics of CTC

For those who want to understand the "Forward-Backward" algorithm used in CTC training.

**Objective:** Maximize `P(Y|X)`.
Since many paths map to `Y` (e.g., `AA` and `A` both map to `A`), we sum over all valid paths.

`P(Y|X) = Sum_{pi in Path(Y)} P(pi|X)`

**Dynamic Programming (Forward Variable alpha):**
`alpha_t(s)`: Probability of outputting the first `s` characters of `Y` after `t` time steps.

**Transitions:**
- If `Y[s] == Y[s-2]` (repeat char): We can only come from `alpha_{t-1}(s)` or `alpha_{t-1}(s-1)`.
- If `Y[s] != Y[s-2]` (new char): We can also come from `alpha_{t-1}(s-2)` (skipping the blank).

This `O(T * S)` algorithm is what allows CTC to be differentiable and trainable via backpropagation.

## Appendix B: Complete Python Decoder Class

```python
import numpy as np
from collections import defaultdict

class CTCDecoder:
    def __init__(self, vocab, beam_width=100, alpha=0.5, beta=1.0):
        self.vocab = vocab
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        
    def decode(self, probs):
        """
        probs: (T, V)
        Returns: best_string
        """
        # Initialization
        beam = defaultdict(lambda: (float('-inf'), float('-inf')))
        beam[()] = (0.0, float('-inf'))
        
        for t in range(len(probs)):
            next_beam = defaultdict(lambda: (float('-inf'), float('-inf')))
            
            # Pruning: Only keep top beam_width
            sorted_beam = sorted(
                beam.items(),
                key=lambda x: np.logaddexp(x[1][0], x[1][1]),
                reverse=True
            )[:self.beam_width]
            
            for prefix, (p_b, p_nb) in sorted_beam:
                # ... (Same logic as above) ...
                # See main article for the core loop
                pass
                
            beam = next_beam
            
        return self._get_best_hypothesis(beam)

    def _get_best_hypothesis(self, beam):
        best_prefix = max(beam.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1]))[0]
        return "".join(best_prefix)
```

## Appendix C: The ASR Troubleshooting Guide

**Problem: The decoder outputs nothing.**
- **Cause:** Your blank probability is 1.0 everywhere.
- **Fix:** Check your training data. Are your labels aligned? Is the learning rate too high (exploding gradients)?

**Problem: The decoder repeats words ("hello hello hello").**
- **Cause:** The model is confident for too many frames.
- **Fix:** Increase the blank probability penalty or use a Language Model with a repetition penalty.

**Problem: WER is 100%.**
- **Cause:** Vocabulary mismatch. Are you using the same char-to-int mapping as training?
- **Fix:** Verify `vocab.json`.

## Appendix D: Deep Dive into WFST (Weighted Finite State Transducers)

Before Deep Learning took over, ASR was built on **WFSTs**.
Even today, the **Kaldi** toolkit (which powers many production systems) uses them.

**What is a WFST?**
It's a graph where edges have:
1.  **Input Label** (e.g., Phoneme)
2.  **Output Label** (e.g., Word)
3.  **Weight** (Probability)

**The Composition (H o C o L o G):**
We build a massive static graph by composing four smaller graphs:
- **H (HMM):** Maps HMM states to Context-Dependent Phones.
- **C (Context):** Maps Context-Dependent Phones to Phones.
- **L (Lexicon):** Maps Phones to Words (Pronunciation Dictionary).
- **G (Grammar):** Maps Words to Sentences (Language Model).

**Decoding:**
Decoding is simply finding the shortest path in the `H o C o L o G` graph.
This is extremely fast because the graph is pre-compiled and optimized (determinized and minimized).

**Why learn this?**
If you work on **Edge AI** (embedded devices), you might not have the RAM for a Transformer. A WFST decoder is incredibly memory-efficient and fast.

## Conclusion

Implementing a CTC Beam Search decoder is a rite of passage. It forces you to understand the probabilistic nature of speech.

While end-to-end models (like Whisper) are replacing complex decoders with simple `model.generate()`, understanding **Beam Search** is still crucial for:
1.  **Streaming ASR** (where Transformers are too slow).
2.  **Keyword Spotting** (Wake word detection).
3.  **Customization** (Adding hot-words).

**Key Takeaways:**
- **CTC** handles the alignment between audio and text.
- **Beam Search** keeps multiple hypotheses alive to correct early mistakes.
- **LMs** are essential for fixing homophones ("beach" vs "speech").

---

**Originally published at:** [arunbaby.com/speech-tech/0023-asr-beam-search-implementation](https://www.arunbaby.com/speech-tech/0023-asr-beam-search-implementation/)

*If you found this helpful, consider sharing it with others who might benefit.*


