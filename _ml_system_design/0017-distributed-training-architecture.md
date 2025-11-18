---
title: "Distributed Training Architecture"
day: 17
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - distributed-training
  - data-parallelism
  - model-parallelism
  - pipeline-parallelism
  - all-reduce
  - large-scale-sequences
subdomain: "Training Infrastructure"
tech_stack: [PyTorch, TensorFlow, Horovod, NCCL, Kubernetes, Ray, DeepSpeed]
scale: "1K+ GPUs, 10B+ parameters, PB-scale data"
companies: [Google, Meta, OpenAI, DeepMind, Microsoft, Amazon]
related_dsa_day: 17
related_ml_day: 17
related_speech_day: 17
---

**Design distributed training architectures that can efficiently process massive sequential datasets and train billion-parameter models across thousands of GPUs.**

## Problem Statement

You need to design a **distributed training architecture** for large-scale deep learning models that:

1. Trains on **petabytes of sequential data** (text tokens, audio frames, clickstreams).
2. Supports **hundreds of millions to tens of billions of parameters**.
3. Utilizes **hundreds to thousands of GPUs** efficiently.
4. Provides **fault tolerance**, **elastic scaling**, and **observability** suitable for production.

### Functional Requirements

1. **Data ingestion & preprocessing**
   - Stream training data from distributed storage (S3/HDFS/GCS).
   - Handle sharded datasets and multiple epochs.
   - Perform lightweight preprocessing/augmentation online.

2. **Parallel training**
   - Support data parallelism, model parallelism, and pipeline parallelism.
   - Allow hybrid combinations for very large models.

3. **Gradient synchronization**
   - Efficient all-reduce / all-gather for gradients and parameters.
   - Topology-aware communication (intra-node vs inter-node).

4. **Checkpointing & recovery**
   - Periodic checkpoints to distributed storage.
   - Resume after failures without losing significant progress.

5. **Experiment management**
   - Track configs, code versions, metrics, and artifacts.
   - Support hyperparameter sweeps.

6. **Scheduling & orchestration**
   - Submit, pause, resume, and cancel training jobs.
   - Allocate GPUs/TPUs across multiple teams.

### Non-Functional Requirements

1. **Throughput**
   - High GPU utilization (70–90%).
   - Minimize data pipeline and communication stalls.

2. **Scalability**
   - Near-linear scaling when increasing GPU count (e.g., 8 → 64 → 512).

3. **Reliability**
   - Automatic recovery from worker/node failures.
   - Tolerate transient network/storage issues.

4. **Cost efficiency**
   - Reasonable cost per training step / per processed token.
   - Ability to leverage spot/preemptible instances when possible.

5. **Reproducibility**
   - Seed control, deterministic data shuffling.
   - Ability to reproduce critical experiments.

## Understanding the Requirements

Distributed training is required when:

- **Model is too big** for a single GPU (e.g., 10B+ parameters).
- **Dataset is huge** (e.g., trillions of tokens, millions of hours of speech).
- **Training time needs to move from weeks to days or hours.**

This architecture lives at the intersection of:

- **High-performance computing (HPC)** – communication, topology, scheduling.
- **Data engineering** – sharded sequential data pipelines.
- **ML research** – model architectures, training recipes, evaluation.

### Core Challenges

1. **Compute parallelism:** How do we split model and data across GPUs?
2. **Communication overhead:** How do we synchronize parameters/gradients efficiently?
3. **Data pipeline throughput:** How do we keep GPUs fed with data?
4. **Fault tolerance:** How do we handle worker/preemptions gracefully?
5. **Sequential data handling:** How do we stream long sequences efficiently?

### The Sequential Data Connection

Conceptually, this is the same pattern as **Add Two Numbers (Linked List)**, just at a different scale:

| Domain | Sequential Data | State |
|--------|-----------------|-------|
| DSA | Digits in linked lists | Carry |
| Distributed Training | Tokens/audio frames | Optimizer + model state |
| Speech Training | Audio chunks | Streaming encoder state |

In all 3:
- You **stream through long sequences** chunk-by-chunk.
- You maintain **small state** across steps (carry, optimizer state, hidden states).
- You often process data in a **sharded** fashion across machines.

## High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                Distributed Training Architecture                 │
└─────────────────────────────────────────────────────────────────┘

                          Control Plane
                    ┌────────────────────┐
                    │  Orchestrator      │
                    │  - Job scheduler   │
                    │  - Resource mgr    │
                    │  - Elastic scaling │
                    └─────────┬──────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
┌──────────▼─────────┐ ┌─────▼──────┐ ┌────────▼────────┐
│  Config & Params   │ │  Logging   │ │  Experiment     │
│  - Model configs   │ │  & Metrics │ │  Tracking       │
│  - Optimizer cfgs  │ │  (Prom/Graf)││  (MLflow/W&B)   │
└─────────┬──────────┘ └─────┬──────┘ └────────┬────────┘
          │                   │                 │
          └───────────────────┼─────────────────┘
                              │
                         Data Plane
          ┌──────────────────┼──────────────────┐
          │                  │                  │
┌─────────▼────────┐ ┌──────▼───────┐ ┌────────▼────────┐
│  Trainer Group 1 │ │ Trainer Group│ │ Trainer Group N │
│  (Data Parallel) │ │ 2 (Hybrid)   │ │ (Specialized)   │
│  GPUs: 0..7      │ │ GPUs: 8..15  │ │ GPUs: ...       │
└─────────┬────────┘ └──────┬───────┘ └────────┬────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                     ┌───────▼───────┐
                     │   Data Layer  │
                     │   - Sharded   │
                     │     datasets  │
                     │   - Feature   │
                     │     store     │
                     └───────────────┘
```

### Key Components

1. **Data Layer**
   - Sharded datasets in object storage (S3/GCS/HDFS).
   - Optional feature store (pre-computed embeddings, features).

2. **Trainer Groups**
   - Sets of GPUs/nodes cooperating on one training job.
   - May use different parallelism strategies (pure data-parallel, hybrid, etc.).

3. **Communication Layer**
   - NCCL, MPI, or gRPC for collective communication (all-reduce, all-gather).

4. **Control Plane**
   - Orchestrates jobs, scales clusters, schedules resources.
   - Often backed by Kubernetes + a training framework (Ray, Kubeflow, SageMaker, etc.).

5. **Monitoring & Experimentation**
   - Metrics pipelines (Prometheus, Grafana).
   - Experiment tracking (MLflow, Weights & Biases).

## Parallelism Strategies

### 1. Data Parallelism

**Idea:** replicate the model on each worker, shard the data.

- Each worker:
  - Gets a different mini-batch.
  - Computes local gradients.
- Then all workers:
  - **All-reduce** gradients,
  - Apply the update to their own copy of the model.

```python
import torch
import torch.distributed as dist

def train_epoch_data_parallel(model, dataloader, optimizer, rank, world_size):
    model.train()
    for step, batch in enumerate(dataloader):
        inputs = batch['inputs'].to(rank)
        targets = batch['targets'].to(rank)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = compute_loss(outputs, targets)
        loss.backward()

        # Gradient all-reduce
        for param in model.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size

        optimizer.step()
```

**Pros:**
- Simple to reason about.
- Works for most models that fit in one GPU.

**Cons:**
- Limited by single-GPU model memory.
- Communication cost grows with model size.

### 2. Model Parallelism

**Idea:** split the model itself across multiple GPUs.

- Often used when the model is too big for a single GPU.
- Example: tensor parallelism (split matrix multiplications across GPUs).

```python
class TensorParallelLinear(torch.nn.Module):
    \"\"\"Example of a tensor-parallel linear layer over 2 GPUs.\"\"\"\n    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_half = out_features // 2
        self.w0 = torch.nn.Parameter(
            torch.randn(in_features, self.out_half, device='cuda:0')
        )
        self.w1 = torch.nn.Parameter(
            torch.randn(in_features, self.out_half, device='cuda:1')
        )

    def forward(self, x):
        # x initially on cuda:0
        x0 = x.to('cuda:0')
        x1 = x.to('cuda:1')

        y0 = x0 @ self.w0
        y1 = x1 @ self.w1

        # Gather back to one device
        y = torch.cat([y0.to('cuda:0'), y1.to('cuda:0')], dim=-1)
        return y
```

**Pros:**
- Allows training models larger than single-GPU memory.

**Cons:**
- More complex to implement and debug.
- Imbalanced partitions cause stragglers.

### 3. Pipeline Parallelism

**Idea:** Split the network into **stages** and place each on a GPU (or set of GPUs).

```text
Stage 0 (GPU0): layers 0–3
Stage 1 (GPU1): layers 4–7
Stage 2 (GPU2): layers 8–11
...
```

- Micro-batches flow through the pipeline, overlapping compute across stages.
- Schedules like GPipe and 1F1B (one-forward-one-backward) reduce pipeline bubbles.

**Pros:**
- Scales deep models nicely.

**Cons:**
- Requires careful tuning of micro-batch size and scheduling.
- More complex debugging.

### 4. Hybrid Parallelism

Real SOTA systems combine:

- Data parallel across nodes,
- Tensor model parallel across GPUs within node,
- Pipeline parallel across layers.

This is how very large LLMs and giant speech models are trained.

## Data Layer: Handling Large-Scale Sequential Data

### 1. Sharded Datasets

For large corpora (text, audio, click logs), store data as **shards**:

- `data-00000.tfrecord`, `data-00001.tfrecord`, ...
- Each shard contains a manageable number of samples (e.g., 10K–100K).

```python
from torch.utils.data import IterableDataset

class ShardedDataset(IterableDataset):
    \"\"\"Distributed sharded dataset for large-scale sequential data.\"\"\"\n    def __init__(self, shard_paths: list[str], rank: int, world_size: int):
        super().__init__()
        self.shard_paths = shard_paths[rank::world_size]  # simple sharding

    def __iter__(self):
        for shard_path in self.shard_paths:
            yield from self._read_shard(shard_path)

    def _read_shard(self, path: str):
        # Read compressed records (e.g., TFRecord, WebDataset tar)
        # Yield token/audio sequences lazily
        raise NotImplementedError
```

### 2. Sequence Bucketing & Packing

To reduce padding waste when training on sequences:

```python
def bucket_by_length(sequences, bucket_sizes):
    buckets = {b: [] for b in bucket_sizes}
    for seq in sequences:
        length = len(seq)
        for b in bucket_sizes:
            if length <= b:
                buckets[b].append(seq)
                break
    return buckets
```

- Group sequences by length bucket.
- Within each bucket, pad to that bucket size.
- Improves GPU efficiency significantly for long-tail length distributions.

### 3. Streaming Input Pipeline

```python
from torch.utils.data import DataLoader

def build_dataloader(shards, batch_size, rank, world_size):
    dataset = ShardedDataset(shards, rank, world_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        prefetch_factor=4,
    )
    return dataloader
```

Common pitfalls:

- Underestimating I/O latency from cloud storage.
- Not using enough data loader workers.
- Doing heavy CPU-bound preprocessing inside `__getitem__`.

## Communication Layer: Collectives & Topology

### All-Reduce for Gradients

```python
import torch.distributed as dist

def allreduce_gradients(model):
    \"\"\"All-reduce gradients across data-parallel workers.\"\"\"\n    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= dist.get_world_size()
```

### Topologies

- **Ring all-reduce**
  - Good bandwidth utilization.
  - Latency grows with number of nodes.
- **Tree all-reduce**
  - Better latency characteristics.
  - Often used when world size is large.

Frameworks like NCCL dynamically choose strategies based on the cluster topology:

- GPUs within a node (NVLink, PCIe).
- Nodes within a rack (top-of-rack switch).
- Racks within a data center.

## Checkpointing & Fault Tolerance

### Checkpointing

```python
import torch

def save_checkpoint(model, optimizer, step, path):
    state = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'step': step,
    }
    torch.save(state, path)


def load_checkpoint(model, optimizer, path, map_location='cuda'):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])
    return state['step']
```

Best practices:

- Save to **replicated or erasure-coded storage** (S3/GCS/HDFS).
- Keep multiple generations (e.g., last 3–5 checkpoints).
- Include additional metadata (config hash, git commit).

### Fault Tolerance

Scenarios:

- **Worker dies** (e.g., preempted spot instance).
  - Use elastic training (TorchElastic/Ray Train) to allow workers to join/leave.
  - Rebuild process groups on the fly.
- **Node dies**
  - Kubernetes reschedules pods.
  - Training resumes from latest checkpoint.

Important design question:

- **How often do you checkpoint?**
  - Trade-off between:
    - Time spent writing checkpoints.
    - Amount of work lost on failure.

## Monitoring & Metrics

### What to Track

- **Training metrics**
  - Loss, accuracy, perplexity, WER, etc.
  - Learning rate schedules, gradient norms.
- **System metrics**
  - GPU utilization, memory usage.
  - Network bandwidth, all-reduce time.
  - Data loader time vs step time.

```python
class TrainingMetrics:
    def __init__(self):
        self.step_times = []
        self.throughputs = []

    def log_step(self, step_time, samples):
        self.step_times.append(step_time)
        self.throughputs.append(samples / max(step_time, 1e-8))

    @property
    def avg_step_time(self):
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0

    @property
    def avg_throughput(self):
        return sum(self.throughputs) / len(self.throughputs) if self.throughputs else 0
```

Use Prometheus/Grafana or similar for real-time dashboards:

- Per-job, per-node, per-GPU metrics.
- Alerting for:
  - Low GPU utilization,
  - High all-reduce latency,
  - Data loader bottlenecks.

## Failure Modes & Mitigations

### 1. Stragglers

Symptoms:

- Some workers consistently slower.
- Step times dominated by waiting for slowest worker.

Causes:

- Heterogeneous hardware.
- Data skew (some workers get heavier batches).
- Noisy neighbors in shared clusters.

Mitigations:

- Use **dynamic load balancing** for data shards.
- Prefer homogeneous instance types for training clusters.
- Monitor per-worker step time and reassign data if needed.

### 2. Data Pipeline Bottlenecks

Symptoms:

- GPUs idle waiting for data.
- High CPU usage in data loaders.

Mitigations:

- Increase `num_workers` in data loaders.
- Move heavy preprocessing offline.
- Cache preprocessed data on local SSDs.

### 3. Communication Bottlenecks

Symptoms:

- Step time dominated by all-reduce.
- Network saturation.

Mitigations:

- Overlap communication and computation (e.g., gradient bucketing).
- Use hierarchical all-reduce (intra-node then inter-node).
- Consider gradient compression for extremely large clusters.

## Real-World Case Study (Conceptual): GPT-Scale Training

Large language models like GPT, PaLM, LLaMA are trained with:

- **Model size:** 10B–100B+ parameters.
- **Data:** Trillions of tokens.
- **Hardware:** 100s–1000s of GPUs or TPUs.

Parallelism:

- **Tensor parallelism** for large matrix multiplications.
- **Pipeline parallelism** over layers.
- **Data parallelism** across nodes.

Key techniques:

- Mixed-precision training (FP16/BF16).
- ZeRO optimizer sharding (DeepSpeed).
- Gradient checkpointing to reduce memory.
- Sophisticated LR schedules and warmup.

Results:

- Training times on the order of days to weeks (not months).
- Sustained TFLOPs in the tens of percent of theoretical peak.

## Cost Analysis (Back-of-the-Envelope)

Example: 1B-parameter Transformer

Assume:

- 1B parameters
- 1024 tokens per sample
- 1T tokens total
- 128 A100 GPUs at $3/hr each

| Component               | Value           |
|-------------------------|-----------------|
| Tokens/sec/GPU         | ~10,000         |
| Total tokens/sec       | 1.28M           |
| Time to process 1T tok | ~9 days         |
| GPU cost/day           | 128 × $3 = $384 |
| **Total cost**         | **≈ $3,456**    |

Cost levers:

- Larger batch size (within stability limits).
- Better input pipeline (reduce stalls).
- Using cheaper GPU types where possible.
- Spot instances for non-critical runs (with robust checkpointing).

## Key Takeaways

✅ **Distributed training is about parallelizing sequential processing** of huge datasets.

✅ **Data parallelism** is the default; model/pipeline parallelism unlocks enormous models.

✅ **Handling large-scale sequential data** requires sharding, streaming, and careful state management.

✅ **Communication** (all-reduce/all-gather) is often the primary bottleneck at scale.

✅ **Resilience and checkpointing** are non-negotiable at 100s–1000s of GPUs.

✅ **Observability** (throughput, utilization, step times) is key to cost efficiency.

### Connection to Thematic Link: Handling Large-Scale Sequential Data

All three Day 17 topics share the same pattern:

**DSA (Add Two Numbers – Linked List):**
- Process digits sequentially.
- Maintain small carry state.
- Handle arbitrarily long numbers.

**ML System Design (Distributed Training Architecture):**
- Process long sequences of tokens/audio frames.
- Maintain optimizer/model state across steps.
- Scale to petabytes of data and billions of parameters.

**Speech Tech (Distributed Speech Training):**
- Process long-form audio in chunks.
- Maintain streaming encoder state and dataset state across shards.
- Train robust ASR/TTS models at massive scale.

The **sequential, stateful processing model** is universal—from a single linked list on a whiteboard to a thousand-GPU training job in a data center.

---

**Originally published at:** [arunbaby.com/ml-system-design/0017-distributed-training-architecture](https://www.arunbaby.com/ml-system-design/0017-distributed-training-architecture/)

*If you found this helpful, consider sharing it with others who might benefit.*


