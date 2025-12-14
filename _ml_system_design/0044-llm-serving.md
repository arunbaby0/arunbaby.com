---
title: "LLM Serving Infrastructure"
day: 44
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - llm
  - inference
  - infrastructure
  - optimization
difficulty: Hard
---

**"Serving models that think at human scale."**

## 1. Introduction: The LLM Serving Challenge

Large Language Models (LLMs) like GPT-4, Claude, and LLaMA present unique serving challenges:

*   **Size:** 7B to 175B+ parameters (tens to hundreds of GBs).
*   **Latency:** Users expect real-time responses (<1s for first token).
*   **Cost:** GPU compute is expensive ($2-10/hour per A100).
*   **Scale:** Millions of concurrent users.

**Goal:** Serve LLMs efficiently, balancing latency, throughput, and cost.

## 2. LLM Inference Basics

### 2.1. Autoregressive Generation

LLMs generate text token by token:
1.  **Prefill:** Process the input prompt (can be parallelized).
2.  **Decode:** Generate one token at a time (sequential).

**Latency Components:**
*   **Time to First Token (TTFT):** Time to generate the first output token.
*   **Time Per Output Token (TPOT):** Time to generate each subsequent token.
*   **Total Latency:** TTFT + (num_tokens - 1) × TPOT.

### 2.2. Memory Bottleneck

**KV Cache:** During generation, we cache the key-value pairs from previous tokens.

**Memory Usage:**
*   Model weights: 2 bytes per parameter (FP16).
*   KV Cache: 2 × num_layers × hidden_size × 2 bytes per token.

**Example (LLaMA-7B):**
*   Weights: 7B × 2 = 14GB.
*   KV Cache (4096 tokens): 32 layers × 4096 dim × 4096 tokens × 2 × 2 = 4GB.
*   **Total:** ~18GB per request (with long context).

## 3. Key Performance Metrics

**Latency:**
*   **TTFT:** Time to first token (<500ms typical).
*   **TPOT:** Time per output token (<50ms typical).

**Throughput:**
*   **Tokens/second:** How many tokens generated per second.
*   **Requests/second:** How many concurrent requests served.

**Efficiency:**
*   **GPU Utilization:** % of GPU compute used.
*   **Cost per Token:** $/1000 tokens.

## 4. Serving Architecture

### 4.1. Basic Architecture

```
                    ┌─────────────────────┐
                    │   Load Balancer     │
                    │   (HAProxy/Nginx)   │
                    └─────────┬───────────┘
                              │
         ┌────────────────────┴────────────────────┐
         │                    │                    │
┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
│   GPU Node 1    │  │   GPU Node 2    │  │   GPU Node N    │
│  (4x A100 80GB) │  │  (4x A100 80GB) │  │  (4x A100 80GB) │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 4.2. Components

**1. Load Balancer:**
*   Distributes requests across GPU nodes.
*   Health checks, sticky sessions (for streaming).

**2. Inference Server:**
*   Runs the LLM.
*   Examples: vLLM, TensorRT-LLM, TGI (Text Generation Inference).

**3. Request Queue:**
*   Buffers requests during peak load.
*   Prioritization (premium users first).

**4. Caching Layer:**
*   Cache common prompts and responses.
*   Reduces compute for repeated queries.

## 5. Optimization Techniques

### 5.1. Continuous Batching

**Problem:** Static batching waits for all requests in a batch to finish before processing new ones.

**Solution:** **Continuous batching** allows new requests to join the batch as soon as old ones finish.

**Benefits:**
*   Higher GPU utilization.
*   Lower latency for short requests.
*   2-10x throughput improvement.

### 5.2. PagedAttention (vLLM)

**Problem:** KV cache memory is fragmented, leading to inefficient memory usage.

**Solution:** PagedAttention manages KV cache like OS virtual memory:
*   Divide KV cache into fixed-size blocks.
*   Allocate blocks on demand.
*   Share blocks for common prefixes (e.g., system prompts).

**Benefits:**
*   2-4x more concurrent requests.
*   Efficient memory utilization.

### 5.3. Speculative Decoding

**Problem:** Autoregressive decoding is inherently sequential.

**Solution:** Use a smaller, faster "draft" model to generate candidate tokens. Verify with the large model in parallel.

**Algorithm:**
1.  Draft model generates N tokens.
2.  Large model verifies (can be parallelized).
3.  Accept verified tokens, reject or correct others.

**Benefits:**
*   2-3x speedup for when draft model matches well.

### 5.4. Quantization

**Reduce memory and compute by using lower precision:**
*   **FP16:** Standard (2 bytes/param).
*   **INT8:** 4x reduction (1 byte/param), minimal quality loss.
*   **INT4:** 8x reduction, some quality loss.
*   **GPTQ/AWQ:** Post-training quantization methods.

**Example (LLaMA-70B):**
*   FP16: 140GB (2x A100 80GB).
*   INT4: 35GB (1x A100 80GB).

### 5.5. Tensor Parallelism

**Split model across multiple GPUs:**
*   Each GPU holds a portion of each layer.
*   Communicate intermediate results.

**Use Case:** Models too large for single GPU.

### 5.6. Pipeline Parallelism

**Split model by layers across GPUs:**
*   GPU 1: Layers 1-10.
*   GPU 2: Layers 11-20.
*   GPU 3: Layers 21-30.

**Issue:** Bubble time (GPUs waiting for each other).

## 6. Popular LLM Serving Frameworks

### 6.1. vLLM

**Features:**
*   PagedAttention for memory efficiency.
*   Continuous batching.
*   OpenAI-compatible API.
*   High throughput.

**Usage:**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

outputs = llm.generate(["What is AI?"], sampling_params)
print(outputs[0].outputs[0].text)
```

### 6.2. TensorRT-LLM (NVIDIA)

**Features:**
*   NVIDIA-optimized kernels.
*   INT8/INT4 quantization.
*   Tensor parallelism.
*   Flash Attention.

**Best for:** Maximum performance on NVIDIA GPUs.

### 6.3. Text Generation Inference (TGI)

**Features:**
*   By Hugging Face.
*   Docker-based deployment.
*   Supports many models.
*   Streaming support.

### 6.4. Ollama

**Features:**
*   Simple local deployment.
*   Bundled models.
*   Good for development/testing.

### 6.5. ONNX Runtime

**Features:**
*   Cross-platform.
*   CPU and GPU support.
*   Quantization support.

## 7. System Design: ChatGPT-like Service

**Requirements:**
*   Handle 1M+ requests/day.
*   TTFT < 500ms.
*   Support streaming responses.
*   Handle long conversations (context > 4K tokens).

### 7.1. Architecture

```
                    ┌─────────────────────┐
                    │   API Gateway       │
                    │   (Rate Limiting)   │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Request Router    │
                    │   (Model Selection) │
                    └─────────┬───────────┘
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
┌────────▼────────┐                      ┌────────▼────────┐
│   Inference     │                      │   Inference     │
│   Cluster A     │                      │   Cluster B     │
│   (GPT-4)       │                      │   (GPT-3.5)     │
└────────┬────────┘                      └────────┬────────┘
         │                                         │
         └────────────────────┬────────────────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Streaming         │
                    │   Response Server   │
                    └─────────────────────┘
```

### 7.2. Request Flow

1.  **Authentication:** Verify API key, check rate limits.
2.  **Routing:** Route to appropriate model cluster.
3.  **Queueing:** Add to request queue with priority.
4.  **Inference:** Process with continuous batching.
5.  **Streaming:** Stream tokens back to client.

### 7.3. Scaling Strategy

**Horizontal Scaling:**
*   Add more GPU nodes for throughput.
*   Use auto-scaling based on queue depth.

**Vertical Scaling:**
*   Use larger GPUs (A100 → H100).
*   Use tensor parallelism for larger models.

## 8. Caching Strategies

### 8.1. Prompt Caching

**Cache common prompts** (system messages, instructions).

**Implementation:**
*   Hash the prompt prefix.
*   Store KV cache for the prefix.
*   Reuse for new requests with same prefix.

**Benefits:**
*   Lower TTFT for repeated prompts.
*   Reduced compute.

### 8.2. Semantic Caching

**Cache based on semantic similarity:**
*   Embed the query.
*   Find similar cached queries.
*   Return cached response if similarity > threshold.

**Tools:** GPTCache, LangChain caching.

### 8.3. Response Caching

**Cache full responses for common queries:**
*   Hash the full input.
*   Store the response.
*   Return cached response for exact matches.

## 9. Production Monitoring

**Metrics to Track:**
1.  **Latency:** TTFT, TPOT, P50/P95/P99.
2.  **Throughput:** Requests/second, tokens/second.
3.  **GPU Utilization:** Compute and memory.
4.  **Queue Depth:** Requests waiting.
5.  **Error Rate:** Failed requests.

**Dashboards:**
*   Grafana with Prometheus.
*   Custom dashboards in Datadog.

**Alerting:**
*   TTFT P99 > 1s.
*   GPU utilization < 50% (underutilization).
*   Error rate > 1%.

## 10. Cost Optimization

**Cost Drivers:**
*   GPU hours (major cost).
*   Network egress (streaming).
*   Storage (model weights, logs).

**Optimization Strategies:**
1.  **Quantization:** Fit larger models on fewer GPUs.
2.  **Batching:** Higher throughput = lower cost/request.
3.  **Caching:** Avoid redundant computation.
4.  **Spot Instances:** Use for non-critical workloads.
5.  **Auto-Scaling:** Scale down during off-peak hours.

**Example Cost Analysis:**
*   A100 80GB: ~$2/hour.
*   LLaMA-7B: 1000 tokens/second.
*   Cost: $0.002 per 1000 tokens.

## 11. Interview Questions

1.  **Explain continuous batching.** Why is it better than static batching?
2.  **What is PagedAttention?** How does it improve memory efficiency?
3.  **How do you handle long contexts (>4K tokens)?**
4.  **Design a ChatGPT-like service.** What are the key components?
5.  **Compare vLLM, TensorRT-LLM, and TGI.** When would you use each?

## 12. Common Pitfalls

*   **Underestimating Memory:** KV cache grows with sequence length.
*   **Ignoring Cold Start:** First request is slow (model loading).
*   **Not Using Streaming:** Users perceive lower latency with streaming.
*   **Over-Provisioning:** GPU idle time is wasted money.
*   **Ignoring Quantization:** FP16 may be overkill for many applications.

## 13. Future Trends

**1. Specialized Hardware:**
*   NVIDIA H100, AMD MI300.
*   Google TPU v5.
*   Custom ASICs (Groq, Cerebras).

**2. Efficient Architectures:**
*   Mixture of Experts (MoE).
*   State Space Models (Mamba).
*   Sparse Attention.

**3. On-Device LLMs:**
*   Quantized models on mobile.
*   Apple Intelligence, Gemini Nano.

**4. Disaggregated Serving:**
*   Separate prefill and decode phases.
*   Optimize each independently.

## 14. Conclusion

LLM serving is a rapidly evolving field. The key challenges are:
*   **Memory:** KV cache is the bottleneck.
*   **Latency:** Users expect real-time responses.
*   **Cost:** GPU compute is expensive.

**Key Takeaways:**
*   **Continuous Batching:** Essential for throughput.
*   **PagedAttention:** Efficient memory management.
*   **Quantization:** Reduce memory and cost.
*   **Caching:** Avoid redundant computation.
*   **Monitoring:** Track latency, throughput, and utilization.

As LLMs become ubiquitous, efficient serving will become a critical skill. The techniques here are the foundation for building production LLM systems at scale.

## 15. Deep Dive: KV Cache Optimization

**Challenge:** KV cache grows linearly with sequence length.

**For LLaMA-7B, 4K context:**
*   32 layers × 32 heads × 128 dim × 4K tokens × 2 (K+V) × 2 bytes = 4GB per request.

**Optimization Strategies:**

**1. Multi-Query Attention (MQA):**
*   Share K and V heads across attention heads.
*   Reduction: 8x (if 8 Q heads share 1 KV head).

**2. Grouped-Query Attention (GQA):**
*   Compromise between MHA and MQA.
*   LLaMA 2 70B uses 8 KV heads for 64 Q heads.

**3. KV Cache Compression:**
*   Quantize KV cache to INT8.
*   2x reduction, minimal quality loss.

**4. Sliding Window Attention:**
*   Only attend to last N tokens.
*   Mistral uses 4K sliding window.

**5. Sparse Attention:**
*   Only attend to important tokens.
*   Block-sparse or learned patterns.

## 16. Deep Dive: Mixture of Experts (MoE)

**Idea:** Not all parameters are active for each token. Route tokens to specialized experts.

**Architecture:**
*   Multiple FFN "experts" per layer.
*   Router selects top-K experts per token.
*   Only K experts' parameters are active.

**Benefits:**
*   More parameters with similar compute.
*   Mixtral-8x7B: 8 experts, uses 2 per token.

**Serving Challenges:**
*   Expert weights may be on different GPUs.
*   All-to-all communication for routing.
*   Load balancing across experts.

## 17. Production Case Study: OpenAI ChatGPT

**Scale:**
*   Millions of requests per day.
*   Multiple model sizes (GPT-3.5, GPT-4).

**Architecture (Estimated):**
*   Azure infrastructure.
*   A100 clusters with tensor parallelism.
*   Custom inference stack.

**Optimizations:**
*   Speculative decoding for GPT-4.
*   Heavy prompt caching.
*   Priority queuing for Plus subscribers.

## 18. Production Case Study: Anthropic Claude

**Innovations:**
*   Constitutional AI (safety filtering).
*   Long context (100K+ tokens).

**Serving Challenges:**
*   100K context = massive KV cache.
*   Requires memory optimization.

**Solutions (Likely):**
*   Sliding window + summary caching.
*   Efficient attention patterns.

## 19. Advanced: Prefill-Decode Disaggregation

**Observation:** Prefill and decode have different compute profiles.
*   **Prefill:** Compute-bound (parallel).
*   **Decode:** Memory-bound (sequential).

**Solution:** Use different hardware for each phase.
*   **Prefill:** High-compute GPUs (H100).
*   **Decode:** Memory-optimized GPUs (A10G).

**Benefits:**
*   Better resource utilization.
*   Lower cost per token.

## 20. Advanced: Request Scheduling

**Challenge:** Different requests have different priorities and lengths.

**Strategies:**
1.  **FIFO:** Simple, but long requests block short ones.
2.  **Shortest Job First:** Better latency, but starvation risk.
3.  **Priority Queuing:** Premium users first.
4.  **Fair Scheduling:** Balance latency across users.

**Implementation:**
```python
class RequestScheduler:
    def __init__(self):
        self.queues = {
            'premium': [],
            'standard': [],
            'batch': []
        }
    
    def add_request(self, request, priority='standard'):
        self.queues[priority].append(request)
    
    def get_next_request(self):
        for priority in ['premium', 'standard', 'batch']:
            if self.queues[priority]:
                return self.queues[priority].pop(0)
        return None
```

## 21. Benchmarking LLM Serving

**Metrics:**
*   **Tokens/second/GPU:** Throughput efficiency.
*   **TTFT P50/P99:** Latency for first token.
*   **TPOT:** Time per output token.
*   **Maximum Concurrent Requests:** Scalability.

**Benchmark Tools:**
*   **LLMPerf:** Open-source benchmark suite.
*   **vLLM Benchmark:** Built-in benchmarking.
*   **Custom Load Testing:** Locust, k6.

**Example Benchmark (LLaMA-7B on A100 80GB):**
| Framework | Tokens/sec | TTFT (ms) | Max Concurrent |
|-----------|------------|-----------|----------------|
| vLLM | 2000 | 50 | 100 |
| TensorRT-LLM | 2500 | 40 | 120 |
| TGI | 1500 | 80 | 80 |

## 22. Multi-Model Serving

**Scenario:** Serve multiple models on the same infrastructure.

**Approaches:**
1.  **Dedicated Clusters:** Each model on separate GPUs.
2.  **Time-Slicing:** Switch models on same GPU.
3.  **Memory Sharing:** Load multiple models if memory allows.

**Trade-offs:**
*   Dedicated: Simple, no interference, but expensive.
*   Time-Slicing: Flexible, but model loading overhead.
*   Memory Sharing: Efficient, but complex management.

## 23. Disaster Recovery

**Scenarios:**
*   GPU failure.
*   Network partition.
*   Model corruption.

**Strategies:**
1.  **Redundancy:** Multiple replicas of each model.
2.  **Fallback:** Route to smaller model if primary fails.
3.  **Health Checks:** Quick detection and recovery.
4.  **Load Shedding:** Reject requests gracefully under overload.

## 24. Mastery Checklist

**Mastery Checklist:**
- [ ] Explain TTFT vs TPOT
- [ ] Implement continuous batching
- [ ] Understand PagedAttention
- [ ] Deploy vLLM for production
- [ ] Implement prompt caching
- [ ] Configure quantization (INT8/INT4)
- [ ] Set up monitoring dashboards
- [ ] Estimate costs per 1000 tokens
- [ ] Design ChatGPT-like architecture
- [ ] Handle long context (>32K tokens)

## 25. Conclusion

LLM serving is one of the most challenging and rewarding areas in ML infrastructure. The techniques covered here—continuous batching, PagedAttention, quantization, and caching—are the building blocks of production LLM systems.

As models get larger and contexts get longer, these optimizations become even more critical. The engineers who master LLM serving will be essential to the AI infrastructure of the future.

**The key insight:** LLM inference is memory-bound, not compute-bound. Optimize for memory, and throughput will follow.

