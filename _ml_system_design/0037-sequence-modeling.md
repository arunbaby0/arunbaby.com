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

- **Sparse Transformers:** 90% sparsity with minimal accuracy loss.

## 16. Deep Dive: Sequence-to-Sequence Models (Seq2Seq)

**Architecture:** Encoder-Decoder with Attention.

**Use Cases:**
- Machine Translation (English → French)
- Summarization (Long article → Short summary)
- Question Answering (Context + Question → Answer)

**Key Innovation: Attention Mechanism (Bahdanau et al., 2015)**
- **Problem:** Fixed-length context vector bottleneck.
- **Solution:** Decoder attends to all encoder states.

**Implementation:**
```python
class Seq2SeqWithAttention(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.encoder = nn.LSTM(vocab_size, hidden_size, bidirectional=True)
        self.decoder = nn.LSTM(vocab_size, hidden_size)
        self.attention = nn.Linear(hidden_size * 3, 1)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, src, tgt):
        # Encode
        encoder_outputs, (h, c) = self.encoder(src)
        
        # Decode with attention
        decoder_hidden = h
        outputs = []
        
        for t in range(tgt.size(0)):
            # Compute attention scores
            scores = self.attention(torch.cat([
                decoder_hidden.expand(encoder_outputs.size(0), -1, -1),
                encoder_outputs
            ], dim=2))
            
            # Attention weights
            attn_weights = F.softmax(scores, dim=0)
            
            # Context vector
            context = (attn_weights * encoder_outputs).sum(dim=0)
            
            # Decoder step
            decoder_input = torch.cat([tgt[t], context], dim=1)
            output, (decoder_hidden, c) = self.decoder(decoder_input.unsqueeze(0), (decoder_hidden, c))
            
            outputs.append(self.output(output))
        
        return torch.cat(outputs, dim=0)
```

## 17. Advanced: Sparse Attention Mechanisms

**Problem:** Full attention is $O(N^2)$. For $N = 100k$ (long documents), this is prohibitive.

**Solutions:**

**1. Longformer (Beltagy et al., 2020):**
- **Local Attention:** Each token attends to $w$ neighbors (sliding window).
- **Global Attention:** A few tokens (e.g., `[CLS]`) attend to everything.
- **Complexity:** $O(N \cdot w)$ instead of $O(N^2)$.

**2. BigBird (Zaheer et al., 2020):**
- **Random Attention:** Each token attends to $r$ random tokens.
- **Block Attention:** Divide sequence into blocks, attend within blocks.
- **Global Attention:** Special tokens attend globally.

**3. Linformer (Wang et al., 2020):**
- **Low-Rank Projection:** Project $K, V$ to lower dimension.
- **Complexity:** $O(N \cdot k)$ where $k \ll N$.

**Code (Longformer-style Local Attention):**
```python
def local_attention(Q, K, V, window_size=256):
    N, d = Q.shape
    
    # Create attention mask (band matrix)
    mask = torch.zeros(N, N)
    for i in range(N):
        start = max(0, i - window_size // 2)
        end = min(N, i + window_size // 2)
        mask[i, start:end] = 1
    
    # Compute attention
    scores = (Q @ K.T) / math.sqrt(d)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    
    return attn @ V
```

## 18. Case Study: AlphaFold (Protein Folding)

**Problem:** Predict 3D protein structure from amino acid sequence.

**Why Sequence Modeling?**
- Protein is a sequence of amino acids (A, C, D, E, ...).
- Structure depends on long-range interactions (residue 10 affects residue 500).

**AlphaFold 2 Architecture:**
1. **Evoformer:** Transformer variant with:
   - **MSA (Multiple Sequence Alignment) Attention:** Attend across evolutionary related sequences.
   - **Pair Representation:** Attend to pairwise residue relationships.
2. **Structure Module:** Predicts 3D coordinates.

**Key Innovation:** Attention over both sequence and structure space.

**Result:** Solved a 50-year-old problem. Accuracy comparable to experimental methods.

## 19. Deep Dive: Mixture of Experts (MoE)

**Idea:** Instead of one giant model, use many small "expert" models. Route each input to the most relevant experts.

**Architecture:**
```
Input → Router (Gating Network) → Top-K Experts → Combine Outputs
```

**Example: Switch Transformer (Google, 2021)**
- 1.6 **Trillion** parameters.
- But only 10B active per token (sparse activation).
- **Routing:** Each token is sent to 1 expert (out of 2048).

**Benefits:**
- **Scalability:** Add more experts without increasing per-token compute.
- **Specialization:** Expert 1 learns math, Expert 2 learns code, etc.

**Challenges:**
- **Load Balancing:** Ensure all experts are used equally.
- **Training Instability:** Router can collapse (send everything to one expert).

**Code:**
```python
class MoELayer(nn.Module):
    def __init__(self, num_experts, hidden_size):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # Route
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-1 expert per token
        expert_idx = torch.argmax(router_probs, dim=-1)
        
        # Apply experts
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i)
            if mask.any():
                output[mask] = expert(x[mask])
        
        return output
```

## 20. Production Serving Architecture

**Scenario:** Serve GPT-3-scale model (175B params) with <1s latency.

**Architecture:**
```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Load Balancer  │
└──────┬──────────┘
       │
   ┌───┴───┐
   ▼       ▼
┌─────┐ ┌─────┐
│ GPU │ │ GPU │  (Model Parallelism: 8 GPUs per replica)
│  1  │ │  2  │
└─────┘ └─────┘
```

**Optimizations:**

**1. Continuous Batching (Orca, 2022):**
- Don't wait for all requests to finish.
- As soon as one finishes, add a new request to the batch.
- **Speedup:** 2-3x higher throughput.

**2. PagedAttention (vLLM, 2023):**
- Store KV cache in paged memory (like OS virtual memory).
- **Benefit:** Reduce memory waste from fragmentation.

**3. Speculative Decoding:**
- Use a small model (1B) to draft 5 tokens.
- Use large model (175B) to verify in parallel.
- **Speedup:** 2-3x for same quality.

## 21. Evaluation Metrics for Sequence Models

**1. Perplexity (Language Modeling):**
$$PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(w_i | w_{<i})\right)$$
- Lower is better.
- **Interpretation:** "How surprised is the model by the test data?"

**2. BLEU (Machine Translation):**
- Measures n-gram overlap between prediction and reference.
- **Range:** 0-100. BLEU > 40 is considered good.

**3. ROUGE (Summarization):**
- Similar to BLEU, but focuses on recall.

**4. Exact Match (QA):**
- Percentage of predictions that exactly match ground truth.

## 22. Common Pitfalls and How to Avoid Them

**Pitfall 1: Exposure Bias (Teacher Forcing)**
- Model trained on ground truth, tested on its own predictions.
- **Fix:** Scheduled sampling or reinforcement learning.

**Pitfall 2: Length Bias in Beam Search**
- Longer sequences have lower cumulative probability.
- **Fix:** Length normalization: $\text{score} = \frac{\log P(y)}{|y|^\alpha}$

**Pitfall 3: Catastrophic Forgetting (Fine-Tuning)**
- Fine-tuning on Task B makes model forget Task A.
- **Fix:** Elastic Weight Consolidation (EWC) or multi-task learning.

**Pitfall 4: OOM (Out of Memory) During Training**
- Gradient accumulation: Simulate large batch with small batches.
- **Fix:** `loss.backward(); if step % 4 == 0: optimizer.step()`

**Pitfall 5: Ignoring Positional Encoding Limits**
- BERT trained on 512 tokens can't handle 1024.
- **Fix:** Use RoPE or ALiBi (Attention with Linear Biases).

- **Fix:** Use RoPE or ALiBi (Attention with Linear Biases).

## 23. Deep Dive: State Space Models (Mamba / S4)

**Problem:** Transformers are $O(N^2)$. RNNs are $O(N)$ but hard to train. Can we get the best of both?

**Solution:** Structured State Space Models (SSMs).

**Key Idea:**
-   Model the sequence as a continuous-time system:
    $$h'(t) = Ah(t) + Bx(t)$$
    $$y(t) = Ch(t)$$
-   **Discretize** it for digital computers.
-   **Training:** Can be computed as a **Convolution** (Parallelizable like Transformers).
-   **Inference:** Can be computed as a **Recurrence** (Constant memory like RNNs).

**Mamba (Gu & Dao, 2023):**
-   Introduces **Selection Mechanism**: The matrices $B$ and $C$ depend on the input $x_t$.
-   Allows the model to "selectively" remember or ignore information.
-   **Performance:** Matches Transformers on language modeling, but with linear scaling $O(N)$.

**Code (Simplified Mamba Block):**
```python
class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model)
        self.x_proj = nn.Linear(d_model, dt_rank + B_rank + C_rank)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # 1. Project
        x_and_res = self.in_proj(x)
        x, res = x_and_res.chunk(2, dim=-1)
        
        # 2. Conv
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = F.silu(x)
        
        # 3. SSM (Selective Scan)
        # This is the core Mamba magic (usually implemented in CUDA)
        y = selective_scan(x, self.x_proj(x))
        
        # 4. Output
        return self.out_proj(y * F.silu(res))
```

## 24. Deep Dive: Reinforcement Learning from Human Feedback (RLHF)

**Problem:** LLMs are trained to predict the next token, not to be helpful or safe.

**Pipeline (InstructGPT / ChatGPT):**

1.  **Supervised Fine-Tuning (SFT):**
    -   Collect human demonstrations (Question + Answer).
    -   Fine-tune base model.

2.  **Reward Modeling (RM):**
    -   Collect comparison data (Model generates A and B; Human says A > B).
    -   Train a Reward Model to predict human preference.
    -   Loss: $\log(\sigma(r(A) - r(B)))$.

3.  **PPO (Proximal Policy Optimization):**
    -   Optimize the SFT model to maximize the reward from RM.
    -   Constraint: Don't drift too far from SFT model (KL Divergence penalty).

**Impact:**
-   Aligns model with human intent.
-   Reduces toxicity and hallucinations (mostly).

## 25. Advanced: Long-Context Transformers (Ring Attention)

**Problem:** How to train on 1 Million tokens?
-   Memory is the bottleneck. Even with Flash Attention, activations don't fit on one GPU.

**Ring Attention (Liu et al., 2023):**
-   **Idea:** Distribute the sequence across multiple GPUs in a ring.
-   Each GPU holds a chunk of Query, Key, Value.
-   **Step 1:** Compute local attention.
-   **Step 2:** Pass Key/Value blocks to the next GPU in the ring.
-   **Step 3:** Compute attention with new KV blocks.
-   **Repeat:** Until all KV blocks have visited all GPUs.

**Result:**
-   Train on sequences length = Number of GPUs $\times$ Memory per GPU.
-   Enables "Needle in a Haystack" retrieval over entire books.

## 26. Interview Questions for Sequence Modeling

**Q1: Why do we divide by $\sqrt{d_k}$ in Attention?**
*Answer:* To scale the dot products. If $d_k$ is large, dot products become huge, pushing Softmax into regions with extremely small gradients (vanishing gradients).

**Q2: What is the difference between Post-Norm and Pre-Norm?**
*Answer:*
-   **Post-Norm:** `LayerNorm(x + Sublayer(x))`. Original Transformer. Harder to train (warmup needed).
-   **Pre-Norm:** `x + Sublayer(LayerNorm(x))`. Used in GPT-3/LLaMA. More stable gradients, easier to train.

**Q3: How does RoPE handle sequence length extrapolation?**
*Answer:* By rotating the vectors, the attention score depends only on relative distance $(m-n)$. This relative property generalizes better to unseen lengths than absolute positional embeddings.

**Q4: Explain the "KV Cache" memory usage.**
*Answer:* Memory = $2 \times \text{Batch} \times \text{SeqLen} \times \text{Layers} \times \text{HiddenSize}$. For long sequences, this dominates memory. PagedAttention helps reduce fragmentation.

**Q5: Why use MoE?**
*Answer:* To decouple model size (parameters) from compute cost (FLOPs). We can have a 1T parameter model but only use 10B parameters per token, enabling massive capacity with reasonable inference latency.

## 27. Ethical Considerations

**1. Hallucinations:**
-   Sequence models are "stochastic parrots". They generate plausible-sounding but potentially false text.
-   **Risk:** Misinformation in medical/legal contexts.
-   **Mitigation:** RAG (Retrieval Augmented Generation) to ground answers in facts.

**2. Bias and Toxicity:**
-   Models learn biases present in training data (internet).
-   **Risk:** Hate speech, stereotypes.
-   **Mitigation:** RLHF, careful data curation, red-teaming.

**3. Dual Use:**
-   Code generation models can write malware.
-   **Mitigation:** Safety guardrails, refusal to answer harmful queries.

-   **Mitigation:** Safety guardrails, refusal to answer harmful queries.

## 28. Common Mistakes in Sequence Modeling

**1. Training on Test Data (Data Leakage):**
-   Accidentally including the test set in the pre-training corpus.
-   **Consequence:** Overestimated performance.
-   **Fix:** De-duplication (MinHash) against evaluation benchmarks.

**2. Ignoring Tokenizer Issues:**
-   Different tokenizers (BPE, WordPiece) handle whitespace and special characters differently.
-   **Consequence:** Poor performance on code or multilingual text.
-   **Fix:** Use a robust tokenizer like Tiktoken or SentencePiece.

**3. Incorrect Masking in Causal Attention:**
-   Allowing tokens to attend to future tokens during training.
-   **Consequence:** Model learns to cheat, fails at inference.
-   **Fix:** Verify the causal mask (upper triangular matrix is $-\infty$).

**4. Underestimating Inference Cost:**
-   Focusing only on training loss.
-   **Consequence:** Model is too slow/expensive to deploy.
-   **Fix:** Monitor FLOPs per token and KV cache size during design.

## 29. Glossary of Terms

-   **Token:** The atomic unit of text processing (word, subword, or character).
-   **Embedding:** A dense vector representation of a token.
-   **Attention:** Mechanism to weigh the importance of different input tokens.
-   **Self-Attention:** Attention applied to the sequence itself.
-   **Cross-Attention:** Attention applied between two sequences (e.g., Encoder-Decoder).
-   **Logits:** The raw, unnormalized scores output by the last layer.
-   **Softmax:** Function to convert logits into probabilities.
-   **Temperature:** A hyperparameter that controls the randomness of sampling.
-   **Beam Search:** A heuristic search algorithm that explores a graph by expanding the most promising node in a limited set.
-   **Perplexity:** A measurement of how well a probability model predicts a sample.

## 30. Further Reading

1. **"Attention Is All You Need" (Vaswani et al., 2017):** The Transformer paper.
2. **"BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019):** Masked language modeling.
3. **"GPT-3: Language Models are Few-Shot Learners" (Brown et al., 2020):** Scaling laws.
4. **"Flash Attention" (Dao et al., 2022):** Memory-efficient attention.
5. **"Switch Transformers" (Fedus et al., 2021):** Mixture of Experts at scale.

## 30. Conclusion

Sequence modeling has transformed from simple statistical methods (n-grams) to the behemoth Large Language Models that define the current AI era. The journey from RNNs to LSTMs and finally to Transformers represents a shift from sequential processing to parallel, attention-based architectures. This evolution has enabled us to model not just language, but code, biology, and even robot actions as sequences. As we look forward, the challenges of infinite context length, efficient inference, and alignment with human values remain the frontier of research. Whether you are building a chatbot, a translation system, or a protein folder, understanding the underlying mechanics of attention and state management is the key to unlocking the potential of sequence models.

## 31. Summary

| Model | Pros | Cons |
| :--- | :--- | :--- |
| **RNN** | Simple | Vanishing gradients |
| **LSTM** | Long memory | Sequential (slow) |
| **Transformer** | Parallel, Scalable | $O(N^2)$ memory |
| **Flash Attention** | Memory efficient | Complex implementation |
| **MoE** | Trillion-scale models | Load balancing challenges |

---

**Originally published at:** [arunbaby.com/ml-system-design/0037-sequence-modeling](https://www.arunbaby.com/ml-system-design/0037-sequence-modeling/)
