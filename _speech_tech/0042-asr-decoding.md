---
title: "Automatic Speech Recognition (ASR) Decoding"
day: 42
collection: speech_tech
categories:
  - speech-tech
tags:
  - asr
  - decoding
  - beam-search
  - language-model
difficulty: Hard
---

**"Turning acoustic probabilities into coherent text."**

## 1. Introduction

**Automatic Speech Recognition (ASR)** converts speech audio into text. The **decoding** stage is where we take the acoustic model's output (probabilities over characters/tokens) and produce the final transcription.

**Pipeline:**
```
Audio → Feature Extraction (MFCC) → Acoustic Model → Decoder → Text
```

The decoder's job is to find the most likely text sequence given the acoustic observations.

## 2. The Decoding Problem

**Mathematically:**
$$\hat{W} = \arg\max_W P(W | O)$$

Where:
*   $O$ = Acoustic observations (audio features).
*   $W$ = Word sequence (transcription).

Using Bayes' theorem:
$$P(W | O) = \frac{P(O | W) P(W)}{P(O)}$$

Since $P(O)$ is constant, we maximize:
$$\hat{W} = \arg\max_W P(O | W) \cdot P(W)$$

*   $P(O | W)$ = **Acoustic Model** (AM): How likely is this audio given these words?
*   $P(W)$ = **Language Model** (LM): How likely is this word sequence?

## 3. Decoding Strategies

### 3.1. Greedy Decoding

**Algorithm:**
1.  At each time step, pick the most likely token.
2.  Concatenate to form the output.

```python
def greedy_decode(probs):
    # probs: (T, vocab_size)
    tokens = []
    for t in range(len(probs)):
        token = np.argmax(probs[t])
        tokens.append(token)
    return tokens
```

**Pros:** Fast, simple.
**Cons:** Doesn't consider context. Often produces suboptimal results.

### 3.2. Beam Search

**Algorithm:**
1.  Maintain top-k hypotheses (beams).
2.  At each step, extend each hypothesis with all possible tokens.
3.  Keep only the top-k scoring hypotheses.
4.  Repeat until end-of-sequence.

```python
def beam_search(probs, beam_width=5):
    # probs: (T, vocab_size)
    sequences = [[[], 0.0]]  # (tokens, score)
    
    for t in range(len(probs)):
        all_candidates = []
        for seq, score in sequences:
            for token in range(len(probs[t])):
                new_seq = seq + [token]
                new_score = score + np.log(probs[t][token])
                all_candidates.append((new_seq, new_score))
        
        # Keep top-k
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    return sequences[0][0]  # Return best sequence
```

**Pros:** Considers multiple hypotheses. Usually better than greedy.
**Cons:** Computationally expensive. Still not globally optimal.

### 3.3. CTC Decoding

**Connectionist Temporal Classification (CTC)** is a loss function for sequence-to-sequence tasks where alignment is unknown.

**Key Idea:**
*   Add a "blank" token (ε) to handle alignment.
*   Multiple CTC paths map to the same output (e.g., "aa-ab" and "a-aab" both → "ab").

**CTC Decoding:**
1.  Apply greedy or beam search to CTC output.
2.  Collapse consecutive duplicate tokens.
3.  Remove blank tokens.

```python
def ctc_decode(probs, blank_token=0):
    tokens = []
    prev = -1
    
    for t in range(len(probs)):
        token = np.argmax(probs[t])
        if token != prev and token != blank_token:
            tokens.append(token)
        prev = token
    
    return tokens
```

### 3.4. Attention-Based Decoding

In encoder-decoder models (like Whisper, Seq2Seq), the decoder uses attention to focus on relevant parts of the encoder output.

**Algorithm:**
1.  Encoder processes audio → hidden states.
2.  Decoder autoregressively generates tokens.
3.  At each step, attention weights determine which encoder states to focus on.

```python
def attention_decode(encoder_output, decoder, max_length=100):
    tokens = [BOS_TOKEN]  # Begin of sequence
    
    for _ in range(max_length):
        output = decoder(tokens, encoder_output)
        next_token = np.argmax(output[-1])
        tokens.append(next_token)
        
        if next_token == EOS_TOKEN:
            break
    
    return tokens[1:-1]  # Remove BOS and EOS
```

## 4. Language Model Integration

**Problem:** Acoustic model output may be noisy. Language models can correct grammatical errors.

**Approaches:**

### 4.1. Shallow Fusion

**Idea:** Combine AM and LM scores during beam search.

$$\text{score} = \log P_{AM}(y | x) + \lambda \log P_{LM}(y)$$

Where $\lambda$ is a tunable weight.

```python
def beam_search_with_lm(am_probs, lm, beam_width=5, lm_weight=0.5):
    sequences = [[[], 0.0, 0.0]]  # (tokens, am_score, lm_score)
    
    for t in range(len(am_probs)):
        all_candidates = []
        for seq, am_score, lm_score in sequences:
            for token in range(len(am_probs[t])):
                new_seq = seq + [token]
                new_am_score = am_score + np.log(am_probs[t][token])
                new_lm_score = lm_score + lm.score(new_seq)
                
                combined = new_am_score + lm_weight * new_lm_score
                all_candidates.append((new_seq, new_am_score, new_lm_score, combined))
        
        # Keep top-k by combined score
        sequences = sorted(all_candidates, key=lambda x: x[3], reverse=True)[:beam_width]
    
    return sequences[0][0]
```

### 4.2. Deep Fusion

**Idea:** Train the AM and LM jointly. The LM's hidden states are fed into the AM.

**Benefits:** Tighter integration, better performance.
**Drawbacks:** Requires retraining. Less flexible.

### 4.3. Rescoring (N-Best Rescoring)

**Idea:**
1.  Generate N-best hypotheses with AM only.
2.  Rescore each hypothesis with a powerful LM (e.g., GPT).
3.  Return the highest-scoring hypothesis.

**Benefits:** Can use large, pretrained LMs.
**Drawbacks:** Two-pass (slower).

## 5. CTC Beam Search with Language Model

**Prefix Beam Search:**
Handles CTC's blank tokens and collapsed outputs properly.

**Algorithm:**
1.  Track two scores for each prefix:
    *   **Pb (blank):** Probability of prefix ending with blank.
    *   **Pnb (non-blank):** Probability of prefix not ending with blank.
2.  At each step, extend prefixes by:
    *   Adding a blank (only affects Pb).
    *   Repeating the last character (affects Pb/Pnb transition).
    *   Adding a new character (affects Pnb).

```python
def ctc_beam_search_with_lm(probs, lm, beam_width=10, blank=0, lm_weight=0.5):
    # probs: (T, vocab_size)
    # Initialize
    beams = {(): (1.0, 0.0)}  # prefix: (pb, pnb)
    
    for t in range(len(probs)):
        new_beams = defaultdict(lambda: (0.0, 0.0))
        
        for prefix, (pb, pnb) in beams.items():
            for c in range(len(probs[t])):
                p = probs[t][c]
                
                if c == blank:
                    # Extend with blank
                    key = prefix
                    new_beams[key] = (new_beams[key][0] + (pb + pnb) * p, new_beams[key][1])
                
                elif prefix and c == prefix[-1]:
                    # Repeat character (needs blank before)
                    new_beams[prefix] = (new_beams[prefix][0], new_beams[prefix][1] + pb * p)
                    # Or new character after blank
                    new_key = prefix + (c,)
                    new_beams[new_key] = (new_beams[new_key][0], new_beams[new_key][1] + pnb * p)
                
                else:
                    # New character
                    new_key = prefix + (c,)
                    lm_factor = lm.score(new_key) ** lm_weight
                    new_beams[new_key] = (new_beams[new_key][0], new_beams[new_key][1] + (pb + pnb) * p * lm_factor)
        
        # Prune to beam_width
        beams = dict(sorted(new_beams.items(), key=lambda x: sum(x[1]), reverse=True)[:beam_width])
    
    # Return best prefix
    best_prefix = max(beams, key=lambda x: sum(beams[x]))
    return list(best_prefix)
```

## 6. System Design: Production ASR Decoder

**Scenario:** Build a real-time ASR system for voice assistants.

**Requirements:**
*   **Latency:** < 200ms from end of utterance to transcription.
*   **Accuracy:** WER < 10%.
*   **Streaming:** Start transcribing before the user finishes speaking.

**Architecture:**

**Step 1: Streaming Audio Input**
*   Audio arrives in 20ms chunks.
*   Feature extraction (MFCC/Mel) in real-time.

**Step 2: Streaming Acoustic Model**
*   Use a streaming-compatible architecture (e.g., Conformer with lookahead).
*   Output probabilities every 20ms.

**Step 3: Online Decoding**
*   Run CTC beam search incrementally.
*   Emit partial results as hypotheses stabilize.

**Step 4: Endpointing**
*   Detect when the user stops speaking.
*   Finalize the transcription.

**Step 5: Rescoring (Optional)**
*   Rescore final hypothesis with a larger LM.

## 7. Deep Dive: Weighted Finite State Transducers (WFST)

**WFST** is a mathematical framework for composing AM, LM, and lexicon.

**Components:**
*   **H (HMM):** Maps senones to context-dependent phones.
*   **C (Context):** Maps context-dependent phones to context-independent phones.
*   **L (Lexicon):** Maps phones to words.
*   **G (Grammar):** Language model (n-gram).

**Composition:**
$$\text{Decoding Graph} = H \circ C \circ L \circ G$$

**Benefits:**
*   Efficient decoding using Viterbi algorithm on the composed graph.
*   Caching: Precompute and store the composed graph.

**Tools:**
*   **Kaldi:** Open-source ASR toolkit using WFST.
*   **OpenFst:** Library for finite-state transducers.

## 8. Deep Dive: Streaming vs Non-Streaming

**Non-Streaming (Offline):**
*   Process entire audio at once.
*   Can use bidirectional models (look at future).
*   Higher accuracy.

**Streaming (Online):**
*   Process audio chunk by chunk.
*   Cannot look at future (causality constraint).
*   Lower latency, slightly lower accuracy.

**Hybrid (Look-Ahead):**
*   Allow small lookahead (e.g., 200ms).
*   Balance between latency and accuracy.

## 9. Word Error Rate (WER)

**Definition:**
$$WER = \frac{S + D + I}{N}$$

Where:
*   **S** = Substitutions (wrong word).
*   **D** = Deletions (missing word).
*   **I** = Insertions (extra word).
*   **N** = Total words in reference.

**Example:**
*   Reference: "the cat sat on the mat"
*   Hypothesis: "the cat hat on a mat"
*   Errors: S=1 (sat→hat), S=1 (the→a)
*   WER = 2 / 6 = 33.3%

## 10. Decoding Optimizations

### 10.1. Pruning

**Beam Pruning:** Keep only top-k hypotheses.
**Threshold Pruning:** Discard hypotheses with score < best_score - threshold.
**Histogram Pruning:** Limit hypotheses per frame.

### 10.2. Lattice Generation

Instead of outputting a single transcription, output a **lattice**—a compact representation of all hypotheses.

**Benefits:**
*   Rescoring: Apply different LMs later.
*   Confidence Estimation: Compute confidence scores.
*   N-Best Lists: Extract top-N transcriptions.

### 10.3. GPU Acceleration

*   **Batched Beam Search:** Process multiple utterances in parallel.
*   **CUDA Implementations:** Use GPU for matrix operations.
*   **TensorRT:** Optimize inference.

## 11. Production Case Study: Whisper

**Whisper (OpenAI):**
*   **Architecture:** Encoder-Decoder Transformer.
*   **Decoding:** Greedy or beam search with temperature.
*   **Language Model:** Implicit in the decoder.
*   **Multilingual:** 99 languages.

**Decoding Features:**
*   **Timestamps:** Predict start/end times for each word.
*   **Temperature Fallback:** If greedy fails (repetition), retry with temperature sampling.
*   **Compression Threshold:** Skip segments with too much compression (repetition).

## 12. Interview Questions

1.  **Greedy vs Beam Search:** When would you use each?
2.  **CTC Decoding:** Explain how blank tokens work.
3.  **Language Model Integration:** What is shallow fusion?
4.  **Streaming ASR:** How do you handle causality constraints?
5.  **WFST:** What is the role of the lexicon (L)?
6.  **Calculate WER:** Given reference and hypothesis, compute WER.

## 13. Common Mistakes

*   **Ignoring Blanks in CTC:** Not collapsing consecutive tokens.
*   **Beam Width Too Small:** Missing the correct hypothesis.
*   **LM Weight Too High:** Over-relying on LM, ignoring AM.
*   **Not Handling OOV Words:** Out-of-vocabulary words cause errors.
*   **Ignoring Latency:** Production systems have strict latency requirements.

## 14. Deep Dive: Confidence Estimation

**Problem:** Not all transcriptions are equally reliable. How do we estimate confidence?

**Approaches:**

**1. Posterior Probability:**
*   Use the probability of the best hypothesis.
*   $\text{confidence} = P(W | O)$.

**2. Entropy:**
*   High entropy = low confidence (many competing hypotheses).

**3. Word-Level Confidence:**
*   Compute confidence for each word using lattice posteriors.

**4. Model-Based:**
*   Train a separate model to predict confidence from ASR features.

## 15. Deep Dive: Hot Words (Contextual Biasing)

**Problem:** ASR fails on domain-specific terms (e.g., product names, jargon).

**Solution:** Bias decoding towards expected words.

**Approaches:**
1.  **N-Gram Boosting:** Increase LM probability for hot words.
2.  **Contextual LM:** Condition LM on context (e.g., user's recent queries).
3.  **Shallow Fusion with Boosted LM:** Add bonus score for hot words.

**Example:**
```python
hot_words = ["Alexa", "Siri", "Cortana"]
hot_word_boost = 2.0  # Log-scale boost

def score_with_hotwords(seq, am_score, lm_score):
    bonus = sum(hot_word_boost for word in seq if word in hot_words)
    return am_score + lm_weight * lm_score + bonus
```

## 16. Future Trends

**1. End-to-End LM Integration:**
*   Train AM and LM jointly (e.g., Hybrid Transducers).

**2. LLM for Error Correction:**
*   Use GPT/LLaMA to post-process ASR output.

**3. Multimodal Decoding:**
*   Use video (lip reading) to improve decoding.

**4. On-Device Decoding:**
*   Run decoding on mobile devices (requires efficient implementations).

## 17. Conclusion

ASR decoding is the bridge between acoustic probabilities and human-readable text. The key is balancing:
*   **Accuracy:** Use beam search and language models.
*   **Latency:** Use streaming architectures and pruning.
*   **Flexibility:** Use lattices and rescoring for adaptability.

**Key Takeaways:**
*   **Greedy:** Fast but suboptimal.
*   **Beam Search:** Better quality, higher cost.
*   **CTC:** Handles alignment automatically.
*   **LM Integration:** Shallow fusion, deep fusion, rescoring.
*   **WFST:** Classical approach for composing AM/LM/Lexicon.
*   **Streaming:** Essential for real-time applications.

Mastering decoding is essential for building production ASR systems. The techniques here apply to any sequence-to-sequence task: machine translation, speech synthesis, and beyond.

## 18. Deep Dive: RNN-Transducer (RNN-T) Decoding

**RNN-T** is a popular architecture for streaming ASR (used by Google, Meta).

**Architecture:**
*   **Encoder:** Processes audio, outputs acoustic features.
*   **Prediction Network:** RNN that processes output history.
*   **Joint Network:** Combines encoder + prediction network outputs.
*   **Output:** Probability over vocabulary + blank.

**Decoding Algorithm:**
```python
def rnnt_greedy_decode(encoder_output, predictor, joint):
    T = len(encoder_output)
    U = 0  # Output index
    outputs = []
    pred_state = predictor.initial_state()
    
    t = 0
    while t < T:
        # Get encoder output at time t
        enc_t = encoder_output[t]
        
        # Get prediction network output
        pred_output, pred_state = predictor(outputs[-1:] if outputs else [BOS], pred_state)
        
        # Joint network
        logits = joint(enc_t, pred_output)
        token = np.argmax(logits)
        
        if token == BLANK:
            t += 1  # Advance time
        else:
            outputs.append(token)
            # Don't advance time (can emit multiple tokens at same time step)
    
    return outputs
```

**Key Difference from CTC:**
*   CTC: One output per time step.
*   RNN-T: Can emit multiple tokens per time step (or none).

## 19. Production Case Study: Google Voice Search

**Architecture:**
*   **Model:** Conformer encoder + RNN-T decoder.
*   **Decoding:** Beam search with shallow fusion LM.
*   **Latency:** < 200ms.
*   **WER:** < 5% on conversational speech.

**Optimizations:**
*   **Quantized Model:** INT8 for mobile.
*   **Speculative Decoding:** Start decoding before audio ends.
*   **Hot Word Boosting:** Boost contact names, app names.
*   **Contextual LM:** Condition on user's search history.

## 20. Production Case Study: Amazon Alexa

**Architecture:**
*   **First Pass:** CTC decoding with small LM.
*   **Second Pass:** Rescoring with large LM.
*   **Third Pass:** NLU (intent recognition).

**Latency Breakdown:**
*   Audio capture: 50ms.
*   CTC decoding: 100ms.
*   LM rescoring: 50ms.
*   Total: ~200ms.

**Optimization:**
*   **Endpointing:** Detect speech end quickly.
*   **Prefetch:** Start cloud processing during endpointing.

## 21. Benchmarking ASR Decoders

**Metrics:**
*   **WER:** Word Error Rate.
*   **Latency:** Time from audio end to transcription.
*   **RTF (Real-Time Factor):** Processing time / Audio duration. RTF < 1 means faster than real-time.

**Benchmark (LibriSpeech):**
| Model | WER (test-clean) | RTF (GPU) | RTF (CPU) |
|-------|------------------|-----------|-----------|
| Wav2Vec 2.0 + Greedy | 2.7% | 0.1 | 1.5 |
| Whisper (small) + Beam | 3.0% | 0.2 | 3.0 |
| Conformer + RNN-T | 2.1% | 0.15 | 2.0 |

## 22. Decoding for Low-Resource Languages

**Challenge:** Language models may not exist for low-resource languages.

**Solutions:**
1.  **Multilingual Models:** Use models trained on 100+ languages.
2.  **Cross-Lingual Transfer:** Fine-tune on target language with limited data.
3.  **Character-Level LM:** Train a small character LM.
4.  **Phoneme-Based LM:** Use phoneme sequences instead of words.

## 23. Advanced: Minimum Bayes Risk (MBR) Decoding

**Problem:** Beam search finds the most likely sequence, but not necessarily the one with the lowest WER.

**MBR Approach:**
*   Generate N-best hypotheses.
*   For each hypothesis, compute expected WER against all other hypotheses.
*   Return the hypothesis with lowest expected WER.

**Algorithm:**
```python
def mbr_decode(n_best_list):
    # n_best_list: [(hypothesis, probability), ...]
    best_hyp = None
    best_risk = float('inf')
    
    for hyp, prob in n_best_list:
        risk = 0
        for other_hyp, other_prob in n_best_list:
            risk += other_prob * word_error_rate(hyp, other_hyp)
        
        if risk < best_risk:
            best_risk = risk
            best_hyp = hyp
    
    return best_hyp
```

**Use Case:** When WER is more important than likelihood.

## 24. Deep Dive: Subword Units

**Problem:** Character-level models have long sequences. Word-level models have OOV issues.

**Solution:** Subword units (BPE, SentencePiece).

**Algorithm (BPE):**
1.  Start with characters.
2.  Iteratively merge the most frequent adjacent pair.
3.  Stop when vocabulary size reached.

**Example:**
*   "lower" → "l o w e r" → "lo we r" → "low er" → "lower"

**Benefits:**
*   Handles rare words.
*   Shorter sequences than characters.
*   No OOV (can spell any word).

## 25. Decoding with Word Timestamps

**Problem:** Beyond transcription, we need to know when each word was spoken.

**Approaches:**

**1. CTC Alignment:**
*   After decoding, use forced alignment to find timestamps.

**2. Attention Weights:**
*   Use encoder-decoder attention to infer timestamps.

**3. Whisper-Style:**
*   Predict `<|start_time|>` and `<|end_time|>` tokens.

## 26. Decoding for Dictation vs Conversation

**Dictation:**
*   User speaks clearly, slowly.
*   Expects exact transcription.
*   Punctuation and formatting expected.

**Conversation:**
*   Multiple speakers, overlapping speech.
*   Disfluencies (uh, um, repetitions).
*   Informal language.

**Decoding Differences:**
*   **Dictation:** Use strong LM, inverse text normalization (numbers, dates).
*   **Conversation:** Use disfluency model, speaker diarization.

## 27. Implementation: PyTorch CTC Beam Search

```python
import torch
from torchaudio.models.decoder import ctc_decoder

# Load lexicon and LM
decoder = ctc_decoder(
    lexicon="lexicon.txt",
    tokens="tokens.txt",
    lm="language_model.arpa",
    beam_size=50,
    lm_weight=0.5,
    word_score=-0.2
)

# Decode
emissions = model(audio)  # (T, vocab_size)
hypotheses = decoder(emissions)

# Get best hypothesis
best = hypotheses[0][0]
text = " ".join(best.words)
```

## 28. Interview Strategy: ASR Decoding

**Step-by-Step:**
1.  **Explain the Problem:** AM gives probabilities, need to find best text.
2.  **Greedy:** Simple but suboptimal.
3.  **Beam Search:** Better, introduce beam width.
4.  **CTC Specifics:** Blank token, collapsing.
5.  **LM Integration:** Shallow fusion, rescoring.
6.  **Streaming:** Discuss latency constraints.
7.  **Trade-offs:** Accuracy vs latency vs compute.

## 29. Testing ASR Decoders

**Unit Tests:**
*   Test CTC collapsing: "aa-a-bb" → "ab".
*   Test beam search ranking.
*   Test LM weight behavior.

**Integration Tests:**
*   End-to-end: audio → transcription.
*   Compare with baseline (e.g., Whisper).

**Regression Tests:**
*   Test on standard benchmarks (LibriSpeech, Common Voice).
*   Alert if WER increases.

## 30. Conclusion & Mastery Checklist

**Mastery Checklist:**
- [ ] Implement greedy decoding
- [ ] Implement beam search
- [ ] Implement CTC decoding (with blank collapsing)
- [ ] Integrate language model (shallow fusion)
- [ ] Understand RNN-T decoding
- [ ] Implement N-best list generation
- [ ] Calculate WER correctly
- [ ] Handle streaming ASR
- [ ] Implement hot word boosting
- [ ] Benchmark RTF and WER

ASR decoding is where the magic happens—turning probabilities into words. The techniques you've learned here are the foundation for building voice assistants, transcription services, and real-time translation systems. As LLMs become more integrated with speech, the importance of efficient, accurate decoding will only grow.

**Next Steps:**
*   Implement a CTC decoder from scratch.
*   Add LM integration.
*   Build a streaming decoder.
*   Explore RNN-T for production systems.
*   Study WFST-based decoding (Kaldi).

The journey from "audio in" to "text out" is complex but rewarding. Master it, and you'll have the skills to build world-class speech systems.



---

**Originally published at:** [arunbaby.com/speech-tech/0042-asr-decoding](https://www.arunbaby.com/speech-tech/0042-asr-decoding/)

*If you found this helpful, consider sharing it with others who might benefit.*

