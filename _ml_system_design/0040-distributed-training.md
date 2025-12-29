---
title: "Distributed Training Patterns"
day: 40
related_dsa_day: 40
related_speech_day: 40
related_agents_day: 40
collection: ml_system_design
categories:
 - ml-system-design
tags:
 - distributed-systems
 - deep-learning
 - parallelism
 - gpu
difficulty: Hard
---

**"Scaling from one GPU to thousands."**

## 1. The Need for Scale

Modern Deep Learning models are massive.
* **GPT-3:** 175 Billion parameters.
* **PaLM:** 540 Billion parameters.
* **Training Data:** Trillions of tokens.

A single NVIDIA A100 GPU (80GB VRAM) cannot hold these models, let alone train them in a reasonable time.
To train these models, we must distribute the computation across hundreds or thousands of GPUs.

## 2. Taxonomy of Parallelism

There are three main dimensions of parallelism in Deep Learning:

1. **Data Parallelism:** Split the *data* across devices. Replicate the *model*.
2. **Model Parallelism:** Split the *model* across devices. Replicate the *data* (or split it).
 * **Pipeline Parallelism:** Split layers across devices (Inter-layer).
 * **Tensor Parallelism:** Split individual operations (matrices) across devices (Intra-layer).
3. **3D Parallelism:** Combining all the above.

## 3. Data Parallelism (DP)

This is the simplest and most common form.

**Mechanism:**
1. Copy the entire model to every GPU (Worker).
2. Split the global batch into mini-batches. Assign one mini-batch to each GPU.
3. **Forward Pass:** Each GPU computes predictions and loss for its mini-batch.
4. **Backward Pass:** Each GPU computes gradients w.r.t. its local data.
5. **Synchronization:** Gradients are aggregated (averaged) across all GPUs.
6. **Update:** All GPUs update their model weights with the averaged gradients.

### 3.1. Parameter Server (PS) Architecture
* **Workers:** Compute gradients.
* **Parameter Servers:** Store global weights.
* **Flow:** Workers pull weights -> Compute gradients -> Push gradients to PS -> PS updates weights.
* **Bottleneck:** Network bandwidth at the PS.

### 3.2. Ring All-Reduce
Used in **DistributedDataParallel (DDP)** (PyTorch) and **Horovod**.
* No central server.
* GPUs are arranged in a logical ring.
* Each GPU sends data to its neighbor and receives from the other neighbor.
* **Scatter-Reduce:** Chunk gradients and reduce them as they pass around the ring.
* **All-Gather:** Gather the reduced chunks back to all GPUs.
* **Bandwidth Optimal:** Bandwidth usage is constant regardless of the number of GPUs.

## 4. Model Parallelism (MP)

When the model doesn't fit in one GPU's memory.

### 4.1. Pipeline Parallelism (PP)
Split the model layers into stages.
* GPU 1: Layers 1-10
* GPU 2: Layers 11-20
* ...
* **Naive Approach:** GPU 2 waits for GPU 1. Huge "bubble" (idle time).
* **GPipe / PipeDream:** Split the mini-batch into **micro-batches**. Pipeline the execution of micro-batches to fill the bubbles.

### 4.2. Tensor Parallelism (TP)
Split the tensors (matrices) themselves.
* **Example:** Matrix Multiplication `Y = X \cdot A`.
* Split `A` into columns `A_1, A_2`.
* GPU 1 computes `Y_1 = X \cdot A_1`.
* GPU 2 computes `Y_2 = X \cdot A_2`.
* Concatenate `Y = [Y_1, Y_2]`.
* Requires high-bandwidth interconnect (NVLink) because synchronization happens *per layer*. Used in **Megatron-LM**.

## 5. ZeRO (Zero Redundancy Optimizer)

Standard Data Parallelism replicates the **Optimizer States**, **Gradients**, and **Parameters** on every GPU. This is wasteful.

**ZeRO-DP (DeepSpeed):**
* **ZeRO-1:** Shard Optimizer States. (4x memory reduction).
* **ZeRO-2:** Shard Gradients. (8x memory reduction).
* **ZeRO-3:** Shard Parameters. (Memory usage scales linearly with number of GPUs).

With ZeRO-3, parameters are fetched on-demand just before the forward/backward pass of a layer and released immediately after. It effectively allows training trillion-parameter models on limited GPU memory.

## 6. System Design: Designing a Training Cluster

**Scenario:** Build a cluster to train a 100B parameter model.

### 6.1. Hardware
* **Compute:** NVIDIA H100 or A100 GPUs.
* **Interconnect:**
 * **Intra-node:** NVLink / NVSwitch (900 GB/s). Crucial for Tensor Parallelism.
 * **Inter-node:** InfiniBand or RoCE (RDMA over Converged Ethernet) (400 Gbps). Crucial for Data/Pipeline Parallelism.
* **Storage:** High-performance parallel file system (Lustre, GPUDirect Storage) to feed data at line rate.

### 6.2. Network Topology
* **Rail-optimized:** All GPU-0s across nodes communicate, all GPU-1s communicate, etc.
* **Non-Blocking Fat Tree:** Ensures full bisection bandwidth.

### 6.3. Fault Tolerance
* **Checkpointing:** Save model state frequently to S3/HDFS.
* **Elastic Training:** If a node fails, the job should pause, reconfigure (remove bad node), and resume from the last checkpoint automatically (e.g., PyTorch Elastic / TorchRun).

## 7. Communication Primitives (NCCL)

Understanding the underlying collective operations is key.
* **Broadcast:** One sends to all.
* **Scatter:** One splits data and sends parts to all.
* **Gather:** All send to one.
* **All-Gather:** Everyone gathers data from everyone.
* **Reduce:** Aggregate data to one (Sum, Min, Max).
* **All-Reduce:** Aggregate data and distribute result to all. (The workhorse of DP).

## 8. Deep Dive: Gradient Accumulation

If you can't fit a large enough batch size for convergence (e.g., batch size 32) even with parallelism:
* Run forward/backward for batch size 1.
* Don't update weights.
* Accumulate gradients.
* Repeat 32 times.
* Update weights.
* Simulates a larger batch size at the cost of compute time (no extra memory).

## 9. Case Study: Training GPT-3

* **Architecture:** Transformer Decoder.
* **Parallelism:**
 * **Tensor Parallelism:** Within each node (8 GPUs).
 * **Pipeline Parallelism:** Across nodes.
 * **Data Parallelism:** Across replicas of the pipeline.
* **Infrastructure:** Microsoft Azure AI Supercomputer (10,000+ GPUs).
* **Challenges:** Stragglers (slow nodes), Silent Data Corruption (bit flips), Loss Spikes.

## 10. Deep Dive: Collective Communication (Ring All-Reduce)

The efficiency of Data Parallelism hinges on the **All-Reduce** operation.
**Goal:** Every GPU starts with a gradient vector `G_i`. Every GPU ends with the sum `\sum G_i`.

**Naive Approach:**
* All GPUs send their gradients to GPU 0 (Gather).
* GPU 0 sums them.
* GPU 0 sends the sum back to all GPUs (Broadcast).
* **Bottleneck:** GPU 0's bandwidth. Time scales linearly with `N` (number of GPUs).

**Ring All-Reduce:**
* **Topology:** GPU 0 -> GPU 1 -> ... -> GPU N-1 -> GPU 0.
* **Step 1: Scatter-Reduce:**
 * Split the gradient vector into `N` chunks.
 * In each step, GPU `i` sends a chunk to GPU `i+1` and receives a chunk from GPU `i-1`.
 * It adds the received chunk to its local chunk and passes it on.
 * After `N-1` steps, each GPU holds a fully summed chunk of the gradient vector (different chunk for each GPU).
* **Step 2: All-Gather:**
 * Each GPU sends its fully summed chunk to the next neighbor.
 * After `N-1` steps, all GPUs have all the fully summed chunks.
* **Complexity:**
 * Data transmitted per GPU: `2(N-1) \frac{K}{N} \approx 2K`.
 * **Crucially:** Independent of `N`. This allows scaling to thousands of GPUs.

## 11. Deep Dive: ZeRO (Zero Redundancy Optimizer) Details

Let's break down the memory savings for a model with `\Psi` parameters.
Using Mixed Precision (fp16/bf16) and Adam Optimizer.

**Baseline (Standard DDP):**
* **Parameters (fp16):** `2\Psi` bytes.
* **Gradients (fp16):** `2\Psi` bytes.
* **Optimizer States (fp32):**
 * Master Weights: `4\Psi` bytes.
 * Momentum: `4\Psi` bytes.
 * Variance: `4\Psi` bytes.
 * Total Opt States: `12\Psi` bytes.
* **Total per GPU:** `16\Psi` bytes.

**ZeRO-1 (Shard Optimizer States):**
* Partition the `12\Psi` optimizer states across `N` GPUs.
* Each GPU stores `\frac{12\Psi}{N}`.
* **Total per GPU:** `4\Psi + \frac{12\Psi}{N}`.
* **Savings:** ~4x reduction.

**ZeRO-2 (Shard Gradients):**
* Partition the `2\Psi` gradients as well.
* Each GPU stores `\frac{2\Psi + 12\Psi}{N}`.
* **Total per GPU:** `2\Psi + \frac{14\Psi}{N}`.
* **Savings:** ~8x reduction.

**ZeRO-3 (Shard Parameters):**
* Partition the `2\Psi` parameters too.
* **Total per GPU:** `\frac{16\Psi}{N}`.
* **Savings:** Linear reduction with `N`.
* **Trade-off:** Increased communication. Parameters must be broadcasted (all-gather) before each layer's forward/backward pass and discarded immediately.

## 12. Deep Dive: Pipeline Parallelism (1F1B)

**GPipe:**
* Split batch into `M` micro-batches.
* Run all `M` forward passes.
* Run all `M` backward passes.
* **Memory Issue:** Need to store activations for all `M` micro-batches until the backward pass starts.

**1F1B (One Forward One Backward) - PipeDream:**
* Schedule: Forward 1, Forward 2, ..., Backward 1, Forward 3, Backward 2...
* As soon as the first micro-batch finishes its forward pass at the last stage, it starts its backward pass.
* This frees up activation memory much earlier.
* **Bubble:** The idle time at the start and end of the pipeline.
 * Bubble fraction `\approx \frac{P-1}{M}`, where `P` is pipeline stages.
 * To minimize bubble, we need `M \gg P`.

## 13. Deep Dive: Tensor Parallelism (Megatron-LM)

How do we split a Transformer Layer across GPUs?

**MLP Layer (`A \rightarrow GeLU \rightarrow B`):**
* `Y = GeLU(X A)`.
* Split `A` by **columns** (`A_1, A_2`).
* Each GPU computes `Y_i = GeLU(X A_i)`.
* Next matrix `B` is split by **rows** (`B_1, B_2`).
* Compute `Z = [Y_1, Y_2] \cdot \begin{bmatrix} B_1 \\ B_2 \end{bmatrix} = Y_1 B_1 + Y_2 B_2`.
* **All-Reduce:** Sum the results `Y_1 B_1 + Y_2 B_2` across GPUs.
* **Benefit:** Only one All-Reduce needed per MLP block.

**Self-Attention Layer:**
* Split Query (`Q`), Key (`K`), Value (`V`) weight matrices by **columns** (split heads).
* Each GPU computes attention for a subset of heads.
* Output projection matrix `O` is split by **rows**.
* **All-Reduce:** Sum the outputs of the projection.

## 15. Deep Dive: Network Topologies for AI Clusters

The physical wiring of the cluster determines the maximum possible bandwidth and latency.

**1. Fat Tree (Clos Network):**
* **Structure:** A tree where links near the root have higher bandwidth (fatter) than links near the leaves.
* **Non-Blocking:** Guarantees that any node can communicate with any other node at full line rate (if the switch capacity allows).
* **Pros:** Predictable performance, easy to route.
* **Cons:** Expensive (lots of switches and cables).

**2. Torus / Mesh:**
* **Structure:** Grid-like connection. 2D Torus connects neighbors in X and Y (wrapping around). 3D Torus adds Z.
* **Pros:** Cheaper (fewer switches, direct node-to-node links). Good for local communication (stencil patterns).
* **Cons:** Higher latency for distant nodes (multi-hop).

**3. Dragonfly:**
* **Structure:** Groups of routers fully connected within the group, and groups are fully connected to other groups.
* **Pros:** Low diameter (max 3 hops for any pair), highly scalable.
* **Cons:** Complex routing (needs adaptive routing to avoid congestion).

**NVIDIA SuperPOD:** Uses a non-blocking Fat Tree with InfiniBand HDR/NDR to ensure 800 Gbps per GPU.

## 16. Deep Dive: Framework Internals (PyTorch DDP)

How does `DistributedDataParallel` actually work?

**1. Bucketing:**
* Gradients are small (e.g., bias vector). Sending millions of tiny packets kills performance (latency overhead).
* DDP groups parameters into **Buckets** (e.g., 25MB).
* It waits for all gradients in a bucket to be computed, then triggers one All-Reduce for the entire bucket.

**2. Gradient Hooks:**
* DDP registers autograd hooks on every parameter.
* When a gradient is ready, the hook fires.
* The hook copies the gradient into the bucket buffer.

**3. Overlap (Compute-Comm):**
* While the backward pass is computing gradients for layer `L-1`, the All-Reduce for layer `L` (already in bucket) is running asynchronously on the network card.
* **Goal:** Hide communication time behind computation time.

## 17. Case Study: LLaMA 2 Training Infrastructure

**Meta's Research SuperCluster (RSC):**
* **GPUs:** 16,000 NVIDIA A100 (80GB).
* **Interconnect:** InfiniBand 200 Gbps (Fat Tree).
* **Storage:** 175 PB of Pure Storage FlashBlade.
* **Optimization:**
 * **xFormers:** Optimized Attention kernels (FlashAttention).
 * **Checkpointing:** Saved to distributed storage every few hours.
 * **Silent Data Corruption:** Detected via loss spikes. If detected, roll back to previous checkpoint and skip the bad batch.

**Training Stability:**
* **Loss Spikes:** Often caused by "bad" data or numerical instability (fp16 overflow).
* **Fix:** Gradient Clipping (norm 1.0), Weight Decay decoupling (AdamW), and skipping batches with NaN gradients.

## 18. Deep Dive: Asynchronous vs Synchronous SGD

**Synchronous SGD (Standard):**
* Wait for ALL workers to finish.
* Update = Average of all.
* **Pros:** Mathematically equivalent to large-batch SGD. Converges well.
* **Cons:** Straggler problem (fastest worker waits for slowest).

**Asynchronous SGD (Hogwild!):**
* Workers push gradients to PS whenever they are done.
* PS updates weights immediately.
* Workers pull new weights.
* **Pros:** No waiting. High hardware utilization.
* **Cons:** **Stale Gradients**. Worker computes gradient on `W_t`, but by the time it pushes, the global weights are `W_{t+10}`. The gradient is "stale" and points in the wrong direction.
* **Solution:** Rarely used now. Sync SGD with backup workers (ignore slowest 5%) is preferred.

## 19. Interview Questions

1. **Explain All-Reduce.** How does Ring All-Reduce work? What is its complexity? (`2(N-1) \frac{K}{N}`).
2. **Data Parallelism vs. Model Parallelism.** When to use which?
3. **What is the "Stale Gradient" problem?** In Asynchronous SGD, workers might compute gradients on old weights. How does Synchronous SGD fix this?
4. **How does ZeRO-3 work?** How does it handle communication overhead? (Prefetching).
5. **Calculate Memory Footprint.** For a model with `P` parameters, using Adam optimizer and mixed precision.
 * Parameters: `2P` bytes (fp16).
 * Gradients: `2P` bytes (fp16).
 * Optimizer States: `12P` bytes (fp32 copy of params, momentum, variance).
 * Total: `16P` bytes. For 1B params, ~16GB.

## 20. Deep Dive: Gradient Checkpointing (Activation Recomputation)

**Problem:** During backprop, we need activations from the forward pass. For a 100-layer model, storing all activations requires massive memory.
**Solution:** Trade compute for memory.

**Algorithm:**
1. **Forward Pass:** Only save activations at **checkpoints** (e.g., every 10 layers). Discard intermediate activations.
2. **Backward Pass:** When we need activations for layer 15:
 * Re-run forward from checkpoint 10 to layer 15.
 * Compute gradients.
 * Discard the recomputed activations.

**Memory Savings:**
* Without checkpointing: O(N) memory for `N` layers.
* With checkpointing every `\sqrt{N}` layers: O(\sqrt{N}) memory.
* **Cost:** `\sqrt{N}` extra forward passes (33% compute overhead for Transformers).

**PyTorch Implementation:**
``python
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
 def forward(self, x):
 # Checkpoint expensive blocks
 x = checkpoint(self.layer1, x)
 x = checkpoint(self.layer2, x)
 return x
``

## 21. Deep Dive: Mixed Precision Training (FP16/BF16)

**Motivation:** FP32 (32-bit floats) are slow and memory-hungry. FP16 (16-bit) is 2x faster and uses half the memory.
**Challenge:** FP16 has limited range (`6 \times 10^{-8}` to `65504`). Gradients can underflow (become zero) or overflow.

**Solution: Mixed Precision (NVIDIA Apex / PyTorch AMP):**
1. **Master Weights:** Keep FP32 copy of weights.
2. **Forward/Backward:** Use FP16 for matrix multiplications (fast).
3. **Loss Scaling:** Multiply loss by a large number (e.g., 1024) before backprop. This shifts gradients into FP16's representable range.
4. **Unscale Gradients:** Divide gradients by the scale factor before updating FP32 master weights.
5. **Update:** Update FP32 weights, then copy to FP16 for next iteration.

**BFloat16 (BF16):**
* Same exponent range as FP32 (8 bits), but only 7 bits for mantissa.
* **Advantage:** No loss scaling needed. Easier to use.
* **Disadvantage:** Lower precision than FP16 for small numbers.
* **Used in:** Google TPUs, AMD MI250, NVIDIA H100.

## 22. Deep Dive: FlashAttention (Memory-Efficient Attention)

Standard Attention: O(N^2) memory for the attention matrix.
For `N=4096` tokens, this is 16M elements (64MB in FP16).

**FlashAttention (Dao et al., 2022):**
* **Idea:** Never materialize the full `N \times N` attention matrix.
* **Algorithm:**
 1. Tile `Q`, `K`, `V` into blocks that fit in SRAM (on-chip cache).
 2. Compute attention for each block.
 3. Use online softmax (incremental computation) to avoid storing intermediate results in HBM (slow GPU memory).
* **Result:** 2-4x speedup, enables training with 64K context length.

**Impact on Distributed Training:**
* Reduces activation memory, allowing larger batch sizes or longer sequences.
* Critical for LLaMA, GPT-4, Claude.

## 23. Deep Dive: FSDP (Fully Sharded Data Parallel)

**PyTorch FSDP** is Meta's implementation of ZeRO-3.
**Key Features:**
1. **Auto-Wrapping:** Automatically wraps model layers for sharding.
2. **CPU Offloading:** Can offload parameters to CPU RAM when not in use (train 13B model on 1x A100).
3. **Mixed Precision:** Native support for BF16.

**Usage:**
``python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = MyTransformer()
model = FSDP(model, 
 auto_wrap_policy=transformer_auto_wrap_policy,
 mixed_precision=bf16_policy)
``

**When to use FSDP vs DDP:**
* **DDP:** Model fits in one GPU. Simple, fast.
* **FSDP:** Model doesn't fit. Need memory efficiency.

## 24. Production Deployment: Monitoring & Observability

Training a model for weeks/months requires robust monitoring.

**Metrics to Track:**
1. **Loss Curves:** Train/Val loss per step. Detect divergence early.
2. **Gradient Norms:** Sudden spikes indicate instability.
3. **GPU Utilization:** Should be >90%. If low, data loading is the bottleneck.
4. **Communication Time:** Time spent in All-Reduce. Should be <10% of step time.
5. **Throughput:** Tokens/second. Track degradation over time (memory leaks, stragglers).

**Tools:**
* **TensorBoard / Weights & Biases:** Visualize metrics.
* **NVIDIA DCGM (Data Center GPU Manager):** Monitor GPU health (temperature, power, ECC errors).
* **Prometheus + Grafana:** Cluster-wide metrics.

**Alerting:**
* **Loss NaN:** Immediate rollback.
* **GPU Failure:** Auto-restart with elastic training.
* **Slow Node:** Blacklist and redistribute work.

## 25. Production Deployment: Cost Optimization

Training GPT-3 cost ~$5M. How do we reduce this?

**1. Spot Instances:**
* Use AWS/GCP spot instances (70% cheaper).
* **Risk:** Can be preempted.
* **Mitigation:** Frequent checkpointing + elastic training.

**2. Gradient Accumulation:**
* Simulate large batch size without needing more GPUs.
* Example: 8 GPUs, batch size 4 per GPU, accumulate 8 steps = effective batch size 256.

**3. Selective Precision:**
* Use FP16 for most layers, FP32 for sensitive layers (LayerNorm, Softmax).

**4. Data Loading Optimization:**
* **Prefetching:** Load next batch while GPU is computing.
* **Compression:** Store data in compressed format (Parquet, Arrow).
* **Sharding:** Shard dataset across nodes to avoid network bottleneck.

## 27. Deep Dive: Bandwidth Analysis & Bottleneck Detection

**Theoretical Peak Performance:**
For a model with `P` parameters and batch size `B`:
* **Compute:** `2 \times P \times B` FLOPs per forward pass (matrix multiplications).
* **Memory Bandwidth:** `2 \times P` bytes to load weights (fp16).

**Roofline Model:**
* **Arithmetic Intensity (AI):** FLOPs / Bytes.
* For Transformers: `AI \approx \frac{2PB}{2P} = B` (batch size).
* **A100 GPU:**
 * Peak Compute: 312 TFLOPS (fp16).
 * Peak Bandwidth: 2 TB/s.
 * **Ridge Point:** `AI = \frac{312}{2000} = 0.156` FLOPs/Byte.
* If `B < 0.156`, we are **memory-bound**. If `B > 0.156`, we are **compute-bound**.

**Implication:**
* Small batch sizes (B=1, inference) are memory-bound. Need to optimize data loading.
* Large batch sizes (B=256, training) are compute-bound. Need to optimize kernels (FlashAttention).

## 28. Deep Dive: Debugging Distributed Training

**Common Issues:**

**1. Hanging (Deadlock):**
* **Symptom:** Training freezes. No error message.
* **Cause:** One GPU is waiting for All-Reduce, but another GPU crashed before reaching it.
* **Debug:** Set `NCCL_DEBUG=INFO`. Check which GPU is stuck.
* **Fix:** Add timeouts to collective operations.

**2. Loss Divergence:**
* **Symptom:** Loss becomes NaN after a few steps.
* **Cause:** Gradient explosion, bad data, or numerical instability.
* **Debug:**
 * Log gradient norms per layer.
 * Check for NaN/Inf in activations.
 * Reduce learning rate.
* **Fix:** Gradient clipping, mixed precision with loss scaling.

**3. Slow Training:**
* **Symptom:** Throughput is 50% of expected.
* **Cause:** Communication overhead, data loading bottleneck, or stragglers.
* **Debug:**
 * Profile with `torch.profiler` or NVIDIA Nsight.
 * Check GPU utilization (`nvidia-smi dmon`).
 * Measure communication time vs compute time.
* **Fix:**
 * Increase batch size (reduce communication frequency).
 * Use faster interconnect (InfiniBand).
 * Optimize data loading (more workers, prefetching).

## 29. Advanced Topic: Sequence Parallelism

For very long sequences (e.g., 100K tokens), even the sequence dimension doesn't fit in memory.
**Sequence Parallelism (Megatron-LM):**
* Split the sequence across GPUs along the time dimension.
* Each GPU processes a chunk of the sequence.
* **Challenge:** Self-Attention requires the full sequence. Need to gather all chunks for attention, then scatter back.
* **Optimization:** Overlap communication with computation (pipeline the gather/scatter).

## 30. Advanced Topic: Expert Parallelism (MoE)

**Mixture of Experts (MoE):**
* Replace the MLP layer with `N` expert MLPs.
* A router (gating network) decides which expert(s) to use for each token.
* **Benefit:** Increase model capacity without increasing compute (only 1-2 experts are active per token).

**Expert Parallelism:**
* Place each expert on a different GPU.
* Tokens are routed to the appropriate GPU.
* **Challenge:** Load imbalance (some experts get more tokens).
* **Solution:** Auxiliary loss to encourage balanced routing.

**Example: Switch Transformer (Google):**
* 1.6 Trillion parameters.
* 128 experts per layer.
* Trained with expert parallelism + data parallelism.

## 31. Summary & Best Practices

**Choosing the Right Parallelism Strategy:**
* **Model fits in 1 GPU:** Use DDP (Data Parallelism).
* **Model fits in 1 node (8 GPUs):** Use Tensor Parallelism (Megatron).
* **Model doesn't fit in 1 node:** Use Pipeline Parallelism + Tensor Parallelism.
* **Model is HUGE (>100B):** Use 3D Parallelism (DP + TP + PP) or FSDP/ZeRO-3.

**Optimization Checklist:**
1. ✅ Mixed Precision (BF16).
2. ✅ Gradient Checkpointing.
3. ✅ FlashAttention.
4. ✅ Fused Kernels (AdamW, LayerNorm).
5. ✅ Gradient Accumulation (if batch size is limited).
6. ✅ Data Loading (prefetch, multiple workers).
7. ✅ Profiling (find bottlenecks).

## 32. Common Pitfalls


* **OOM (Out of Memory):** Not using gradient checkpointing (activation recomputation) or mixed precision.
* **Communication Overhead:** Using Ethernet instead of InfiniBand for large models.
* **Uneven Load Balancing:** In Pipeline Parallelism, if layers have different compute costs, some GPUs wait.
* **Batch Norm:** Standard Batch Norm only sees the local batch. Need **SyncBatchNorm** to compute statistics across all GPUs.
* **Random Seed:** Forgetting to set different random seeds per worker for data shuffling (all workers see same data).
* **Learning Rate Scaling:** When increasing batch size from 256 to 2048, need to scale LR proportionally (Linear Scaling Rule).

## 33. Real-World Deployment: Kubernetes for ML

**Challenges:**
* GPUs are expensive. Need efficient scheduling.
* Jobs can fail. Need auto-restart.
* Multi-tenancy. Need isolation.

**Kubeflow:**
* Kubernetes-native ML platform.
* **Components:**
 * **TFJob / PyTorchJob:** Operators for distributed training.
 * **Katib:** Hyperparameter tuning.
 * **KFServing:** Model serving.

**Example PyTorchJob:**
``yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
 name: gpt-training
spec:
 pytorchReplicaSpecs:
 Master:
 replicas: 1
 template:
 spec:
 containers:
 - name: pytorch
 image: pytorch/pytorch:2.0
 command: ["python", "train.py"]
 resources:
 limits:
 nvidia.com/gpu: 8
 Worker:
 replicas: 15
 template:
 spec:
 containers:
 - name: pytorch
 image: pytorch/pytorch:2.0
 command: ["python", "train.py"]
 resources:
 limits:
 nvidia.com/gpu: 8
``

## 34. Conclusion

Distributed training is the backbone of modern AI. From GPT to Stable Diffusion, every large model relies on these techniques.
**Key Takeaways:**
* **Start Simple:** Use DDP for models that fit in one GPU.
* **Scale Gradually:** Add Tensor/Pipeline Parallelism as needed.
* **Optimize Aggressively:** Mixed precision, gradient checkpointing, FlashAttention are non-negotiable.
* **Monitor Everything:** Loss, gradients, GPU utilization, communication time.
* **Expect Failures:** Checkpointing and elastic training are essential.

The future of AI depends on our ability to train ever-larger models efficiently. Mastering distributed training is no longer optional—it's a core skill for ML engineers.


---

**Originally published at:** [arunbaby.com/ml-system-design/0040-distributed-training](https://www.arunbaby.com/ml-system-design/0040-distributed-training/)

*If you found this helpful, consider sharing it with others who might benefit.*

