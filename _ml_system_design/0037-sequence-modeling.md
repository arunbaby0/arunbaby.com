---
title: "Sequence Modeling in ML"
day: 37
collection: ml_system_design
categories:
  - ml_system_design
tags:
  - rnn
  - lstm
  - transformer
  - attention
  - time-series
subdomain: "Deep Learning"
tech_stack: [PyTorch, TensorFlow, Transformers, ONNX]
scale: "Billions of tokens"
companies: [OpenAI, Google, Meta, DeepMind]
---

**"Predicting the next word, the next stock price, the next frame."**

## 1. The Problem: Sequential Data

Many real-world problems involve sequences:
- **Text:** "The cat sat on the ___" → "mat"
- **Time Series:** Stock prices, weather, sensor data.
- **Video:** Predict next frame.
- **Audio:** Speech recognition, music generation.

**Challenge:** The output depends on **context** (previous elements in the sequence).

## 2. Evolution of Sequence Models

### 1. Recurrent Neural Networks (RNN)
- **Idea:** Hidden state $h_t$ carries information from previous steps.
- **Equation:** $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$
- **Problem:** Vanishing gradients. Can't remember long-term dependencies.

### 2. LSTM (Long Short-Term Memory)
- **Gates:** Forget, Input, Output gates control information flow.
- **Advantage:** Can remember 100+ steps.
- **Disadvantage:** Sequential processing (can't parallelize).

### 3. Transformer (Attention is All You Need)
- **Self-Attention:** Every token attends to every other token.
- **Parallelization:** Process entire sequence at once.
- **Scalability:** Powers GPT, BERT, LLaMA.

## 3. Transformer Architecture

**Key Components:**

1.  **Self-Attention:**
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    - **Query (Q):** "What am I looking for?"
    - **Key (K):** "What do I have?"
    - **Value (V):** "What information do I provide?"

2.  **Multi-Head Attention:**
    - Run attention multiple times with different learned projections.
    - Allows model to attend to different aspects (syntax, semantics, etc.).

3.  **Positional Encoding:**
    - Transformers have no notion of order.
    - Add sinusoidal encodings: $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$

4.  **Feed-Forward Network:**
    - Two linear layers with ReLU: $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$

## 4. Training Strategies

### 1. Teacher Forcing
- During training, feed ground truth as input (even if model predicted wrong).
- **Pro:** Faster convergence.
- **Con:** Exposure bias (model never sees its own mistakes during training).

### 2. Scheduled Sampling
- Gradually mix ground truth with model predictions during training.
- Start with 100% teacher forcing, decay to 0%.

### 3. Curriculum Learning
- Start with short sequences, gradually increase length.

## 5. Inference Strategies

### 1. Greedy Decoding
- At each step, pick the token with highest probability.
- **Fast** but **suboptimal**.

### 2. Beam Search
- Keep top-$k$ candidates at each step.
- Explore multiple paths, pick the best overall sequence.
- **Better quality** but **slower**.

### 3. Sampling
- Sample from the probability distribution.
- **Temperature:** Control randomness.
  - $T \to 0$: Greedy (deterministic).
  - $T \to \infty$: Uniform (random).

## 6. System Design: Real-Time Translation

**Scenario:** Google Translate (Text-to-Text).

**Architecture:**
1.  **Encoder:** Processes source sentence (English).
2.  **Decoder:** Generates target sentence (French).
3.  **Attention:** Decoder attends to relevant encoder states.

**Optimizations:**
- **Caching:** Cache encoder output (source doesn't change).
- **Batching:** Process multiple requests together.
- **Quantization:** INT8 for 4x speedup.

## 7. Case Study: GPT-3 Serving

**Challenge:** Serve a 175B parameter model with low latency.

**Solutions:**
1.  **Model Parallelism:** Split model across 8 GPUs.
2.  **KV Cache:** Cache key/value tensors from previous tokens.
3.  **Speculative Decoding:** Use a small model to draft, large model to verify.

## 8. Summary

| Model | Pros | Cons |
| :--- | :--- | :--- |
| **RNN** | Simple | Vanishing gradients |
| **LSTM** | Long memory | Sequential (slow) |
| **Transformer** | Parallel, Scalable | $O(N^2)$ memory |

## 9. Deep Dive: Attention Mechanism Math

Let's break down the attention formula step by step.

**Input:** Sequence of vectors $X = [x_1, x_2, ..., x_n]$, each $x_i \in \mathbb{R}^d$.

**Step 1: Linear Projections**
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$
- $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ are learned weight matrices.

**Step 2: Compute Attention Scores**
$$S = \frac{QK^T}{\sqrt{d_k}}$$
- $S \in \mathbb{R}^{n \times n}$. Entry $S_{ij}$ is the "compatibility" between query $i$ and key $j$.
- Divide by $\sqrt{d_k}$ to prevent dot products from becoming too large (which would make softmax saturate).

**Step 3: Softmax**
$$A = \text{softmax}(S)$$
- Each row sums to 1. $A_{ij}$ is the "attention weight" from position $i$ to position $j$.

**Step 4: Weighted Sum**
$$\text{Output} = AV$$
- Each output vector is a weighted combination of all value vectors.

## 10. Deep Dive: KV Caching for Autoregressive Decoding

**Problem:** When generating token $t$, we recompute attention for tokens $1, 2, ..., t-1$ (wasteful!).

**Solution:** Cache the Key and Value matrices.

**Without Caching:**
```python
for t in range(max_len):
    # Recompute K, V for all previous tokens
    K = compute_keys(tokens[:t+1])  # O(t * d^2)
    V = compute_values(tokens[:t+1])
    output = attention(Q, K, V)
```

**With Caching:**
```python
K_cache, V_cache = [], []

for t in range(max_len):
    # Only compute K, V for new token
    k_t = compute_key(tokens[t])  # O(d^2)
    v_t = compute_value(tokens[t])
    
    K_cache.append(k_t)
    V_cache.append(v_t)
    
    K = concat(K_cache)
    V = concat(V_cache)
    output = attention(Q, K, V)
```

**Speedup:** $O(T^2)$ → $O(T)$ per token.

## 11. Deep Dive: Flash Attention

**Problem:** Standard attention requires materializing the $N \times N$ attention matrix in memory.
- For $N = 4096$, that's 16M floats = 64MB per head.
- GPT-3 has 96 heads → 6GB just for attention!

**Flash Attention (Dao et al., 2022):**
- **Idea:** Compute attention in blocks, never materialize the full matrix.
- **Algorithm:**
  1. Divide $Q, K, V$ into blocks.
  2. Load one block of $Q$ and one block of $K$ into SRAM.
  3. Compute partial attention scores.
  4. Accumulate results.
- **Speedup:** 2-4x faster, uses less memory.

## 12. Deep Dive: Positional Encoding

Transformers have no notion of order. We add positional information.

**Sinusoidal Encoding (Original Transformer):**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**Why this works:**
- Different frequencies for different dimensions.
- Model can learn to attend to relative positions.

**Learned Positional Embeddings (BERT, GPT):**
- Treat position as a lookup table.
- More flexible, but limited to max sequence length seen during training.

**Rotary Position Embedding (RoPE):**
- Used in LLaMA, PaLM.
- Rotates query and key vectors based on position.
- Allows extrapolation to longer sequences.

## 13. System Design: Chatbot with Context

**Scenario:** Build a chatbot that remembers conversation history.

**Challenge:** Context grows with each turn. How to handle long conversations?

**Approach 1: Sliding Window**
- Keep only last $N$ tokens (e.g., 2048).
- **Pro:** Fixed memory.
- **Con:** Forgets old context.

**Approach 2: Summarization**
- Periodically summarize old context.
- **Pro:** Retains important info.
- **Con:** Lossy compression.

**Approach 3: Retrieval-Augmented**
- Store conversation in a vector DB.
- Retrieve relevant past messages based on current query.
- **Pro:** Scalable to infinite history.
- **Con:** Requires embedding model + DB.

## 14. Deep Dive: Beam Search Implementation

```python
def beam_search(model, start_token, beam_width=5, max_len=50):
    # Initialize beam with start token
    beams = [(start_token, 0.0)]  # (sequence, log_prob)
    
    for _ in range(max_len):
        candidates = []
        
        for seq, score in beams:
            if seq[-1] == EOS_TOKEN:
                candidates.append((seq, score))
                continue
            
            # Get next token probabilities
            logits = model(seq)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Top-k tokens
            top_k_probs, top_k_tokens = torch.topk(log_probs, beam_width)
            
            for prob, token in zip(top_k_probs, top_k_tokens):
                new_seq = seq + [token.item()]
                new_score = score + prob.item()
                candidates.append((new_seq, new_score))
        
        # Keep top beam_width candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    return beams[0][0]  # Best sequence
```

## 15. Production Optimizations

### 1. Model Quantization
- Convert FP32 → INT8.
- **Speedup:** 4x.
- **Accuracy Loss:** < 1% with careful calibration.

### 2. Distillation
- Train a small "student" model to mimic a large "teacher".
- **DistilBERT:** 40% smaller, 60% faster, 97% of BERT's performance.

### 3. Pruning
- Remove unimportant weights (e.g., magnitude pruning).
- **Sparse Transformers:** 90% sparsity with minimal accuracy loss.

## 16. Summary

| Model | Pros | Cons |
| :--- | :--- | :--- |
| **RNN** | Simple | Vanishing gradients |
| **LSTM** | Long memory | Sequential (slow) |
| **Transformer** | Parallel, Scalable | $O(N^2)$ memory |
| **Flash Attention** | Memory efficient | Complex implementation |

---

**Originally published at:** [arunbaby.com/ml-system-design/0037-sequence-modeling](https://www.arunbaby.com/ml-system-design/0037-sequence-modeling/)
