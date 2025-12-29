---
title: "Beam Search Decoding"
day: 23
related_dsa_day: 23
related_speech_day: 23
related_agents_day: 23
collection: ml_system_design
categories:
 - ml-system-design
tags:
 - nlp
 - decoding
 - beam-search
 - sampling
 - production
subdomain: "Generative AI & Inference"
tech_stack: [Python, PyTorch, C++]
scale: "O(T×K×V) time, O(T×K) space"
companies: [Google, DeepL, OpenAI, Meta]
---

**The industry-standard algorithm for converting probabilistic model outputs into coherent text sequences.**

## Problem

A sequence model (RNN, LSTM, Transformer) outputs a probability distribution over the vocabulary for the next token, given the history.

`P(y_t | y_1, ..., y_{t-1}, X)`

Our goal is to find the sequence `Y = [y_1, ..., y_T]` that maximizes the total probability:
`argmax_Y P(Y | X) = argmax_Y Π P(y_t | y_{<t}, X)`

Since multiplying probabilities results in tiny numbers (underflow), we work with **Log Probabilities** and sum them:
`argmax_Y Σ log P(y_t | y_{<t}, X)`

### Why not Brute Force?
If the vocabulary size is `V` (e.g., 50,000) and the sequence length is `T` (e.g., 20), the number of possible sequences is `V^T`.
`50,000^20` is larger than the number of atoms in the universe. We cannot check them all.

A sequence model (RNN, LSTM, Transformer) outputs a probability distribution over the vocabulary for the next token, given the history.

`P(y_t | y_1, ..., y_{t-1}, X)`

Our goal is to find the sequence `Y = [y_1, ..., y_T]` that maximizes the total probability:
`argmax_Y P(Y | X) = argmax_Y Π P(y_t | y_{<t}, X)`

Since multiplying probabilities results in tiny numbers (underflow), we work with **Log Probabilities** and sum them:
`argmax_Y Σ log P(y_t | y_{<t}, X)`

### Why not Brute Force?
If the vocabulary size is `V` (e.g., 50,000) and the sequence length is `T` (e.g., 20), the number of possible sequences is `V^T`.
`50,000^20` is larger than the number of atoms in the universe. We cannot check them all.

## High-Level Architecture: The Beam Search Pipeline

``ascii
+-----------+ +------------+ +-----------------+
| Input Seq | --> | Encoder | --> | Hidden States |
+-----------+ +------------+ +-----------------+
 |
 v
 +-----------------+
 | Decoder | <--- (KV Cache)
 +-----------------+
 |
 v
 +-----------------+
 | Logits (V) |
 +-----------------+
 |
 v
 +-----------------+
 | Beam Search | --> Top-K Sequences
 +-----------------+
``

## Strategy 1: Greedy Search

The simplest approach: Always pick the token with the highest probability at each step.

**Algorithm:**
1. Start with `<SOS>` (Start of Sentence).
2. Feed to model -> Get Top-1 token.
3. Append to sequence.
4. Repeat until `<EOS>` (End of Sentence).

**Pros:**
- Fast (O(T)).
- Simple to implement.

**Cons:**
- **Short-sighted:** It makes locally optimal decisions that might lead to a dead end.
- **Example:**
 - Step 1: "The" (0.4), "A" (0.3). Greedy picks "The".
 - Step 2: "The" -> "dog" (0.1). Total score: 0.4 * 0.1 = 0.04.
 - Alternative: "A" -> "cat" (0.9). Total score: 0.3 * 0.9 = 0.27.
 - Greedy missed the better path because it committed too early.

## Strategy 2: Beam Search

Beam Search is a heuristic search algorithm that explores the graph by keeping the **Top-K** most promising sequences at each step. `K` is called the **Beam Width**.

**Algorithm:**
1. **Initialization:** Start with 1 hypothesis: `[<SOS>]` with score 0.0.
2. **Expansion:** For each of the `K` hypotheses, generate the top `V` candidates for the next token.
3. **Scoring:** Calculate the new score for all `K * V` candidates.
 - `NewScore = OldScore + log(P(token))`
4. **Pruning:** Sort all candidates and keep only the top `K`.
5. **Repeat:** Continue until all `K` hypotheses generate `<EOS>` or max length is reached.

### Visualizing the Beam

Imagine a flashlight (beam) shining into a dark cave.
- **Width 1:** A laser pointer (Greedy Search). You only see one path.
- **Width 5:** A flashlight. You see 5 paths.
- **Width Infinity:** The sun. You see everything (BFS), but it burns your CPU.

### Python Implementation

Let's implement a clean, production-ready Beam Search.

``python
import torch
import torch.nn.functional as F
import math

def beam_search_decoder(model, start_token, end_token, k=3, max_len=20):
 # Hypothesis: (score, sequence)
 # Start with one hypothesis
 hypotheses = [(0.0, [start_token])]
 
 # Final completed sequences
 completed_hypotheses = []
 
 for step in range(max_len):
 candidates = []
 
 # 1. Expand each hypothesis
 for score, seq in hypotheses:
 # If this hypothesis is already done, don't expand
 if seq[-1] == end_token:
 completed_hypotheses.append((score, seq))
 continue
 
 # Get model prediction for the next token
 # (In reality, you'd cache the hidden state!)
 input_tensor = torch.tensor([seq])
 with torch.no_grad():
 logits = model(input_tensor) # Shape: [1, seq_len, vocab_size]
 
 # Get log probabilities of the last step
 log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
 
 # 2. Get Top-K candidates for this branch
 top_k_log_probs, top_k_indices = torch.topk(log_probs, k)
 
 for i in range(k):
 token_idx = top_k_indices[i].item()
 token_prob = top_k_log_probs[i].item()
 
 new_score = score + token_prob
 new_seq = seq + [token_idx]
 candidates.append((new_score, new_seq))
 
 # 3. Prune: Keep only global Top-K
 # Sort by score (descending)
 ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
 hypotheses = ordered[:k]
 
 # Early stopping: If all K hypotheses are done
 if not hypotheses:
 break
 
 # Add any remaining running hypotheses to completed
 completed_hypotheses.extend(hypotheses)
 
 # Sort final results
 completed_hypotheses.sort(key=lambda x: x[0], reverse=True)
 
 return completed_hypotheses
``

### Complexity Analysis
- **Time Complexity:** \(O(T \times K \times V)\) (naive) or \(O(T \times K \times \log K)\) (optimized with top-k selection).
- **Space Complexity:** \(O(T \times K)\). We store \(K\) sequences of length \(T\).

## Advanced Beam Search Techniques

Standard Beam Search has issues.
1. **Length Bias:** Longer sentences have more negative terms added to the log-probability (since log(p) < 0). So, Beam Search prefers short sentences.
 - **Fix:** **Length Normalization**. Divide the score by `length^alpha` (usually alpha=0.6 or 0.7).
 - `Score = Sum(log_probs) / (Length)^0.7`

2. **Lack of Diversity:** Often, the top K hypotheses differ only by one word (e.g., "I love dog", "I love dogs", "I love the dog").
 - **Fix:** **Diverse Beam Search (DBS)**. Add a penalty term if hypotheses share the same parent or tokens.
 - **Algorithm:** Divide the beam into `G` groups. Perform beam search for each group, but penalize tokens selected by previous groups.

3. **Constrained Beam Search:**
 - Sometimes you *must* include a specific word in the output (e.g., "Translate this but ensure 'Apple' is capitalized").
 - **Algorithm:** We track the "constraint state" (which words have been met) in the hypothesis. A hypothesis is only valid if it satisfies all constraints by the end.

## Beam Search vs. Sampling (Nucleus/Top-P)

For **Creative Writing** (e.g., ChatGPT writing a poem), Beam Search is bad. It's *too* optimal. It produces boring, repetitive text.
For **Translation/ASR**, Beam Search is king. We want the *correct* translation, not a creative one.

### 1. Temperature
We divide the logits by a temperature `T` before softmax.
- `T < 1`: Makes the distribution sharper (more confident).
- `T > 1`: Makes the distribution flatter (more random).

### 2. Top-K Sampling
Only sample from the top `K` most likely tokens.
- **Problem:** If `K=10`, but only 2 words make sense, we might pick a garbage word.

### 3. Top-P (Nucleus) Sampling
Sample from the smallest set of tokens whose cumulative probability exceeds `P` (e.g., 0.9).
- **Dynamic K:** If the model is unsure, the set is large. If the model is sure, the set is small.

| Feature | Beam Search | Sampling (Top-K / Top-P) |
| :--- | :--- | :--- |
| **Goal** | Maximize Probability | Generate Diversity |
| **Use Case** | Translation, ASR, Summarization | Chatbots, Story Generation |
| **Output** | Deterministic (mostly) | Stochastic (Random) |
| **Risk** | Repetitive loops | Hallucinations |

## Production Engineering: C++ Implementation

In production, Python is too slow. We implement Beam Search in C++ using `std::priority_queue`.

``cpp
#include <queue>
#include <vector>
#include <cmath>
#include <algorithm>

struct Hypothesis {
 std::vector<int> sequence;
 float score;
 
 bool operator<(const Hypothesis& other) const {
 return score < other.score; // Max-heap
 }
};

std::vector<Hypothesis> beam_search(const Model& model, int k) {
 std::priority_queue<Hypothesis> beam;
 beam.push({ {START_TOKEN}, 0.0 });
 
 for (int t = 0; t < MAX_LEN; ++t) {
 std::priority_queue<Hypothesis> next_beam;
 
 // We only pop K times
 int count = 0;
 while (!beam.empty() && count < k) {
 Hypothesis h = beam.top();
 beam.pop();
 count++;
 
 // Expand
 auto logits = model.forward(h.sequence);
 auto top_k_tokens = get_top_k(logits, k);
 
 for (auto token : top_k_tokens) {
 float new_score = h.score + std::log(token.prob);
 next_beam.push({ h.sequence + token.id, new_score });
 }
 }
 beam = next_beam;
 }
 // ... return top results
}
``

### KV Caching & Memory Bandwidth

The bottleneck in Beam Search is not compute (FLOPs), it is **Memory Bandwidth**.
Every step, we need to read the entire model weights from VRAM.
- **KV Caching:** We cache the Key and Value matrices of the attention layers.
- **Memory Usage:** `Batch_Size * Beam_Width * Seq_Len * Hidden_Dim * Layers * 2 (K+V)`.
- **Optimization:** **PagedAttention** (vLLM). Instead of allocating contiguous memory for KV cache (which causes fragmentation), we allocate blocks (pages) on demand, just like an OS manages RAM. This allows 2-4x higher batch sizes.

## System Design Considerations

When deploying Beam Search in production (e.g., serving a Transformer model):

### 1. Latency vs. Width
Increasing `K` improves accuracy but linearly increases compute time.
- **K=1:** Fast, lower quality.
- **K=5:** Standard for Translation.
- **K=10+:** Diminishing returns.

**Optimization:** Use **Adaptive Beam Width**. Start with `K=5`. If the top 2 candidates have very close scores, keep searching. If the top candidate is way ahead, stop early (shrink K to 1).

### 2. Batching
Beam Search is hard to batch because different hypotheses finish at different times.
- **Padding:** We pad finished sequences with `<PAD>` tokens so we can keep doing matrix math on the whole batch.
- **Masking:** We mask out the `<PAD>` tokens so they don't affect the score.

## Case Study: Google Translate

Google Translate uses a variant of Beam Search with:
- **Width:** ~4-6.
- **Length Normalization:** Alpha = 0.6.
- **Coverage Penalty:** Ensures the model translates *all* parts of the source sentence (prevents dropping words).

## Appendix A: Full C++ Production Implementation

Here is a more complete example using `LibTorch` (PyTorch C++ API).

``cpp
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <queue>

// Define a Hypothesis struct
struct Hypothesis {
 std::vector<int64_t> tokens;
 double score;
 
 // For min-heap (we want to pop the lowest score to keep top-k)
 bool operator>(const Hypothesis& other) const {
 return score > other.score;
 }
};

std::vector<int64_t> beam_search_decode(
 torch::jit::script::Module& model, 
 int64_t start_token, 
 int64_t end_token, 
 int k, 
 int max_len
) {
 // Current beam: List of hypotheses
 std::vector<Hypothesis> beam;
 beam.push_back({ {start_token}, 0.0 });
 
 for (int step = 0; step < max_len; ++step) {
 std::vector<Hypothesis> candidates;
 
 for (const auto& h : beam) {
 if (h.tokens.back() == end_token) {
 candidates.push_back(h);
 continue;
 }
 
 // Prepare input tensor
 auto input = torch::tensor(h.tokens).unsqueeze(0); // [1, seq_len]
 
 // Run model (Forward)
 // In real life, use KV cache!
 auto output = model.forward({input}).toTensor();
 auto logits = output.select(1, -1); // Last step: [1, vocab_size]
 auto log_probs = torch::log_softmax(logits, 1);
 
 // Get Top-K
 auto top_k = log_probs.topk(k);
 auto top_k_scores = std::get<0>(top_k)[0];
 auto top_k_indices = std::get<1>(top_k)[0];
 
 for (int i = 0; i < k; ++i) {
 double score = top_k_scores[i].item<double>();
 int64_t idx = top_k_indices[i].item<int64_t>();
 
 std::vector<int64_t> new_tokens = h.tokens;
 new_tokens.push_back(idx);
 
 candidates.push_back({ new_tokens, h.score + score });
 }
 }
 
 // Sort and Prune
 std::sort(candidates.begin(), candidates.end(), [](const Hypothesis& a, const Hypothesis& b) {
 return a.score > b.score; // Descending
 });
 
 if (candidates.size() > k) {
 candidates.resize(k);
 }
 beam = candidates;
 
 // Check if all finished
 bool all_finished = true;
 for (const auto& h : beam) {
 if (h.tokens.back() != end_token) {
 all_finished = false;
 break;
 }
 }
 if (all_finished) break;
 }
 
 return beam[0].tokens;
}
``

## Appendix B: The Ultimate Decoding Glossary

1. **Logits:** Raw, unnormalized scores output by the last layer of the neural network.
2. **Softmax:** Function that converts logits into probabilities (sum to 1).
3. **Log-Probability:** `log(p)`. Used to avoid underflow. Always negative (since p <= 1).
4. **Perplexity:** `exp(-mean(log_probs))`. A measure of how "surprised" the model is. Lower is better.
5. **Entropy:** Measure of randomness in the distribution.
6. **Temperature:** Hyperparameter to control entropy. High T = High Entropy (Random).
7. **Greedy Search:** Beam Search with Width 1.
8. **Beam Width:** Number of hypotheses kept at each step.
9. **Length Penalty:** Normalization term to prevent bias against long sequences.
10. **Coverage Penalty:** Penalty for not attending to source tokens (NMT specific).
11. **Repetition Penalty:** Penalty for generating the same n-gram twice.
12. **Nucleus Sampling (Top-P):** Sampling from the smallest set of tokens with cumulative probability P.
13. **Top-K Sampling:** Sampling from the top K tokens.
14. **Teacher Forcing:** Training technique where we feed the *ground truth* token as input, not the model's prediction.
15. **Exposure Bias:** The mismatch between training (Teacher Forcing) and inference (Autoregressive generation).
16. **BLEU Score:** Metric for translation quality (n-gram overlap).
17. **ROUGE Score:** Metric for summarization quality (recall-oriented).
18. **METEOR:** Metric that considers synonyms and stemming.
19. **WER (Word Error Rate):** Metric for ASR.
20. **KV Cache:** Caching Key/Value matrices to speed up Transformer inference.

## Appendix C: Key Research Papers Summarized

**1. "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2020)**
- **Problem:** Beam Search leads to repetitive, dull text.
- **Solution:** Introduced **Nucleus Sampling (Top-P)**.
- **Key Insight:** Human text is not always "high probability". Humans often use "surprising" words. Beam Search maximizes probability, which is unnatural for creative writing.

**2. "Attention Is All You Need" (Vaswani et al., 2017)**
- **Contribution:** Introduced the Transformer architecture.
- **Relevance:** The Transformer decoder is the standard model used with Beam Search today.

**3. "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014)**
- **Contribution:** Proved that LSTM + Beam Search could beat state-of-the-art SMT (Statistical Machine Translation) systems.

## Appendix D: Full Python Implementation of Nucleus Sampling

While Beam Search is great for accuracy, Nucleus Sampling is the gold standard for creativity (chatbots).

``python
import torch
import torch.nn.functional as F

def top_p_sampling(logits, p=0.9, temperature=1.0):
 """
 logits: [batch_size, vocab_size]
 p: cumulative probability threshold (e.g., 0.9)
 temperature: softmax temperature
 """
 # 1. Apply Temperature
 logits = logits / temperature
 
 # 2. Sort logits in descending order
 sorted_logits, sorted_indices = torch.sort(logits, descending=True)
 
 # 3. Compute cumulative probabilities
 cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
 
 # 4. Create a mask for tokens to remove
 # We want to keep tokens where cumulative_prob <= p
 # But we must always keep the first token (even if its prob > p)
 sorted_indices_to_remove = cumulative_probs > p
 
 # Shift the mask to the right to keep the first token above the threshold
 sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
 sorted_indices_to_remove[..., 0] = 0
 
 # 5. Scatter the mask back to the original indices
 indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
 
 # 6. Set logits of removed tokens to -infinity
 logits[indices_to_remove] = float('-inf')
 
 # 7. Sample from the filtered distribution
 probs = F.softmax(logits, dim=-1)
 next_token = torch.multinomial(probs, num_samples=1)
 
 return next_token
``

## Conclusion

Beam Search is the "Minimum Path Sum" of the NLP world. It's a graph search algorithm designed to find the optimal path through a probabilistic space.

As an ML Engineer, you won't just call `model.generate()`. You will tune `K`, implement length penalties, and optimize the KV-cache to balance the delicate trade-off between **Latency** (cost) and **Quality** (BLEU score).

**Key Takeaways:**
1. **Greedy is fast but risky.**
2. **Beam Search is standard for "correctness" tasks.**
3. **Length Normalization is mandatory.**
4. **KV Caching is essential for speed.**

---

**Originally published at:** [arunbaby.com/ml-system-design/0023-beam-search-decoding](https://www.arunbaby.com/ml-system-design/0023-beam-search-decoding/)

*If you found this helpful, consider sharing it with others who might benefit.*


