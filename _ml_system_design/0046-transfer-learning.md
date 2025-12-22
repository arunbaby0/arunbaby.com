---
title: "Transfer Learning Systems"
day: 46
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - transfer-learning
  - fine-tuning
  - bert
  - resnet
  - system-design
difficulty: Hard
subdomain: "Model Architecture"
tech_stack: PyTorch, Hugging Face, Ray Serve
scale: "Serving 100+ fine-tuned models efficiently"
companies: Hugging Face, Google Cloud, AWS, OpenAI
related_dsa_day: 46
related_ml_day: 46
related_speech_day: 46
related_agents_day: 46
---

**"Standing on the shoulders of giants isn't just a metaphor—it's an engineering requirement."**

## 1. Problem Statement

In the modern ML landscape, training a model from scratch is rarely the answer.
- **Cost**: Training GPT-3 costs ~$4M-$12M.
- **Data**: You rarely have the billions of tokens needed for pre-training.
- **Efficiency**: Why relearn "grammar" or "edges" when open-source models already know them?

**The System Design Problem**:
Design a platform that allows enterprise customers to build custom classifiers (e.g., "Legal Document Classifier", "Spam Detector", "Code Reviewer") using **Transfer Learning**, while minimizing:
1.  **Training Time**: Max 1 hour per model.
2.  **Inference Latency**: <50ms.
3.  **Storage Cost**: We cannot store 100 copies of a 100GB model (10TB) just for 100 customers.

---

## 2. Understanding the Requirements

### 2.1 The Math of Efficiency
Transfer Learning (TL) works by taking a **Teacher Model** (e.g., BERT, ResNet) trained on a massive generic dataset (Wikipedia, ImageNet) and adapting it to a specific task.

There are three main flavors, each with detailed system implications:

1.  **Feature Extraction (Frozen Backbone)**:
    -   Run input through the Backone (BERT).
    -   Get the vector (embedding).
    -   Train a tiny Logistic Regression/MLP on top.
    -   *System*: Very cheap training. Sharing the backbone at inference is easy.

2.  **Full Fine-Tuning**:
    -   Unfreeze all weights. Update everything.
    -   *System*: Expensive training. Result is a completely new 500MB file. Hard to multi-tenant.

3.  **Parameter-Efficient Fine-Tuning (PEFT/Adapters)**:
    -   Inject tiny trainable layers (LoRA adapters) into the frozen backbone.
    -   *System*: The "Holy Grail". Only store 10MB diffs.

### 2.2 Scale Constraints
-   **Base Models**: BERT-Large (340M params), ViT (Vision Transformer).
-   **Throughput**: 10,000 requests/sec aggregate.
-   **Tenancy**: 1,000+ distinct customer models active simultaneously.

---

## 3. High-Level Architecture

We need a layered architecture that separates the "Heavy Lifting" (Base Models) from the "Specific Logic" (Heads/Adapters).

```
[Request: {text: "Sue him!", model_id: "client_A_legal"}]
        |
        v
[Load Balancer / Gateway]
        |
        v
[Model Serving Layer (Ray Serve / TorchServe)]
        |
   +----+--------------------------------+
   |  GPU Worker (Shared Inference)      |
   |                                     |
   |  [ Base Model (BERT) - Frozen ]     |  <-- Loaded Once (VRAM: 2GB)
   |             |                       |
   |             v                       |
   |  [ Adapter Controller ]             |
   |    /        |         \             |
   | [LoRA A]  [LoRA B]  [LoRA C]        |  <-- Swapped Dynamically
   | (Client A)(Client B)(Client C)      |      (VRAM: 10MB each)
   +-------------------------------------+
```

---

## 4. Component Deep-Dives

### 4.1 The Model Registry
This isn't just an S3 bucket. It's a versioned graph.
-   **Parent**: `bert-base-uncased` (SHA256: `a8d2...`)
-   **Child**: `client_A_v1` (Delta Weights + Config) -> Refers to Parent `a8d2...`

When a worker starts, it pulls the Parent. When a request comes for `client_A`, it hot-loads the Child weights.

### 4.2 The Adapter Controller
This is the specialized software component (often custom C++/CUDA).
-   **Function**: Matrix Multiplication routing.
-   Instead of `Y = W * X` (Standard Linear), it computes `Y = W * X + (A * B) * X` where A and B are the low-rank adapter matrices.
-   **Optimization**: Because `A` and `B` are small, we can batch requests from different clients together!
    -   Request 1 (Client A), Request 2 (Client B) -> Both run through BERT backbone together (Batch Size 2).
    -   At the specific layer, split the computation or use specialized CUDA kernels (like **LoRA-Serving** or **vLLM**) to apply per-row adaptors.

---

## 5. Data Flow

1.  **Ingestion**: User uploads 1,000 labeled examples (Instruction Tuning data).
2.  **Preprocessing**: Data is tokenized using the *Base Model's* tokenizer. (Crucial: Adapters can't change the tokenizer).
3.  **Training (ephemeral)**:
    -   Spin up a spot GPU instance.
    -   Load Base Model (Frozen).
    -   Attach new Adapter layers.
    -   Train for 5 epochs (takes ~10 mins).
    -   Extract *only* the Adapter weights.
    -   Save to Registry (Size: 5MB).
4.  **Inference**:
    -   Worker loads Adapter weights into Host RAM.
    -   On request, moves weights to GPU Cache (if not present).
    -   Executes forward pass.

---

## 6. Scaling Strategies

### 6.1 Multi-Tenancy (The "Cold Start" problem)
If a user hasn't sent a request in 24 hours, we offload their adapter from GPU VRAM.
When they return:
-   **Model Load Time**: Loading a 500MB Fine-Tuned model takes 5-10 seconds. (Too slow).
-   **Adapter Load Time**: Loading 5MB LoRA weights takes 50 milliseconds. (Acceptable).
**Conclusion**: PEFT/Adapters enable "Serverless" feel for LLMs.

### 6.2 Caching Strategy
CACHE heavily on:
1.  **Embeddings**: If using Feature Extraction, cache the output of the backbone. If the same email text is classified by 5 different distinct classifiers (Spam, Urgent, Sales, HR, Legal), compute the BERT embedding *once*, then run the 5 lightweight heads.

---

## 7. Implementation: LoRA Injection Conceptual Code

```python
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8):
        super().__init__()
        self.original_layer = original_layer # Frozen
        self.rank = rank
        
        # Low Rank Adaption matrices
        in_dim = original_layer.in_features
        out_dim = original_layer.out_features
        
        # A: Gaussian init, B: Zero init
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        
    def forward(self, x):
        # Path 1: Frozen backbone
        # The original weights don't change
        base_out = self.original_layer(x)
        
        # Path 2: Trainable low-rank branch
        # B @ A is a low-rank matrix that approximates the weight update delta
        adapter_out = (x @ self.lora_A) @ self.lora_B
        
        return base_out + adapter_out

# Usage in System
# To switch clients, we only update lora_A and lora_B
def switch_context(model, new_client_weights):
    model.layer1.lora_A.data = new_client_weights['params_A']
    model.layer1.lora_B.data = new_client_weights['params_B']
```

---

## 8. Monitoring & Metrics

In a Transfer Learning system, typical metrics (Latency, Error Rate) aren't enough. We need **Drift Detection**.

-   **Base Model Drift**: Does the underlying pre-training data (Wikipedia 2020) still represent the world? (e.g., "Covid" meaning change).
-   **Task Drift**: Is the customer's definition of "Spam" changing?
-   **Correlation**: If the Base Model is updated/patched (e.g., security fix), it might break *all 1,000* child adapters. We need rigorous regression testing of the Base Model before upgrades.

---

## 9. Failure Modes

1.  **Catastrophic Forgetting**: (Less relevant here since base is frozen, but crucial if full fine-tuning).
2.  **Tokenizer Mismatch**: User uploads data with Emojis. BERT tokenizer ignores them (mapping to `[UNK]`). Classifier performs poorly.
    -   *Mitigation*: Automated Data Validation step in the pipeline that warns users about `% [UNK]` tokens.
3.  **Noisy Neighbors**: One client sends batch size 128 request, starving the GPU compute for the other 50 clients sharing the backbone.
    -   *Mitigation*: Strict semaphore/queue management on the GPU worker.

---

## 10. Real-World Case Study: Hugging Face Inference API

Hugging Face hosts >100,000 models. They don't have 100,000 GPUs constantly hot.
They use **Shared Backbones** aggressively.
-   If you request `bert-base-finetuned-squad`, they might route you to a generic BERT fleet and apply the distinction (if architecture permits) or use rapid model swapping.
-   For LLMs (Llama-2), they use **LoRA Exchange (LoRAX)** servers allowing one GPU to serve 100s of specialized adapters.

---

## 11. Cost Analysis

**Scenario**: 10 distinct specialized models (7B params each).

**Option A: Dedicated Instances (Full Fine-Tune)**
-   10 x A10 GPUs ($1.50/hr).
-   Cost: $15/hr.
-   Utilization: Low (each model sits idle mostly).

**Option B: Adapter Serving**
-   1 x A10 GPU ($1.50/hr).
-   Base Model (14GB VRAM).
-   10 Adapters (200MB VRAM).
-   Total VRAM: ~15GB. Fits on one card.
-   Cost: $1.50/hr.
-   **Savings: 90%**.

---

## 12. Key Takeaways

1.  **Don't Retrain, Adapt**: Creating new weights from scratch is practically illegal in modern engineering. Use Transfer Learning.
2.  **Freeze the Backbone**: This enables caching, storage savings, and multi-tenant serving.
3.  **PEFT is King**: Techniques like LoRA aren't just "research hacks"; they are fundamental cloud infrastructure enablers that separate compute (Backbone) from logic (Adapter).
4.  **System Design mirrors Model Design**: The mathematical layers of a Neural Net (Frozen vs Trainable) directly dictate the microservices architecture (Shared Fleet vs Dedicated).

---

**Originally published at:** [arunbaby.com/ml-system-design/0046-transfer-learning](https://www.arunbaby.com/ml-system-design/0046-transfer-learning/)

*If you found this helpful, consider sharing it with others who might benefit.*
