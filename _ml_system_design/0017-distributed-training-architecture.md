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
  - all-reduce
  - pipeline-parallelism
  - large-scale-sequences
subdomain: "Training Infrastructure"
tech_stack: [PyTorch, TensorFlow, Horovod, NCCL, Kubernetes, Ray, DeepSpeed]
scale: "1K+ GPUs, 10B+ parameters, PB-scale data"
companies: [Google, Meta, OpenAI, DeepMind, Microsoft, Amazon]
related_dsa_day: 17
related_ml_day: 17
related_speech_day: 17
---

**Design distributed training architectures that efficiently handle large-scale sequential data—from billion-token text corpora to multi-petabyte speech datasets.**

## Problem Statement

Design a **Distributed Training Architecture** for large-scale deep learning models that:

1. Trains on **petabytes of sequential data** (text, audio, clickstreams)
2. Supports **billions of parameters** (Transformer-scale models)
3. Efficiently utilizes **hundreds to thousands of GPUs**
4. Provides **fault tolerance**, **elastic scaling**, and **high throughput**

### Functional Requirements

1. **Data ingestion:** Stream training data from storage (S3/HDFS/GCS) to workers
2. **Parallel training:** Support data, model, and pipeline parallelism
3. **Gradient synchronization:** Efficient all-reduce across GPUs/nodes
4. **Checkpointing:** Periodic checkpoints with recovery support
5. **Hyperparameter configuration:** Learning rate schedules, mixed precision, etc.
6. **Experiment tracking:** Metrics, configs, artifacts
7. **Resilience:** Recover from worker/node failures without losing progress
8. **Scheduling:** Job management, priority queues, resource allocation

### Non-Functional Requirements

1. **Throughput:** 70–90% GPU utilization across the cluster
2. **Scalability:** Linear or near-linear scaling to 1K+ GPUs
3. **Latency:** Low communication overhead for gradient sync
4. **Reliability:** Recoverable from failures within minutes
5. **Cost efficiency:** Optimize dollar-per-training-step
6. **Reproducibility:** Deterministic training under controlled settings

## Understanding the Requirements

Distributed training is required when:

- **Model is too big** for a single GPU (e.g., 10B+ parameters)
- **Dataset is huge** (e.g., 1T tokens, millions of hours of speech)
- **Training time must be reduced** from weeks to days/hours

### Core Challenges

1. **Compute parallelism:** How to split work across GPUs?
2. **Communication overhead:** How to synchronize parameters/gradients efficiently?
3. **Data pipeline throughput:** How to keep GPUs fed with data?
4. **Fault tolerance:** What happens when a node fails?
5. **Sequential data handling:** How to stream sequences (text/audio) efficiently?

### The Sequential Data Connection

This architecture is all about **processing long sequences at scale**:

| Domain | Sequential Data | Operation Pattern |
|--------|-----------------|-------------------|
| DSA | Digits in Add Two Numbers | Carry-based sequential addition |
| ML Training | Tokens/audio frames | Streaming through model & optimizer |
| Speech | Audio segments | Frame-wise processing with state |

In all three, we:
- Process sequences **one chunk at a time**
- Maintain **small state** across steps (e.g., RNN hidden state, optimizer state, carry)
- Stream data efficiently, often across machines

## High-Level Architecture

```
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
│  Parameter Service │ │  Logging   │ │  Experiment     │
│  - Config          │ │  & Metrics │ │  Tracking (ML)  │
└─────────┬──────────┘ └─────┬──────┘ └────────┬────────┘
          │                   │                 │
          └───────────────────┼─────────────────┘
                              │
                         Data Plane
          ┌──────────────────┼──────────────────┐
          │                  │                  │
┌─────────▼────────┐ ┌──────▼───────┐ ┌────────▼────────┐
│  Trainer Group 1 │ │ Trainer Group│ │ Trainer Group N │
│  (Data Parallel) │ │ 2 (Data +    │ │ (Pipeline +     │
│                  │ │  Model Par)  │ │  Data Par)      │
│  GPU 0..7        │ │  GPU 0..7    │ │  GPU 0..7       │
└─────────┬────────┘ └──────┬───────┘ └────────┬────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                     ┌───────▼───────┐
                     │  Data Layer   │
                     │  - Object     │
                     │    Storage    │
                     │  - Feature    │
                     │    Store      │
                     └───────────────┘
```

### Key Components

1. **Data Layer:** Sharded, preprocessed training data (e.g., TFRecords, WebDataset, Parquet)
2. **Trainer Groups:** GPUs performing forward/backward passes
3. **Communication Layer:** NCCL / gRPC / MPI for gradient syncing
4. **Orchestrator:** Schedules jobs, manages resources, handles failures
5. **Experiment Tracking:** Logs metrics, checkpoints, configs

## Parallelism Strategies

### 1. Data Parallelism

- **Each worker** has a full copy of the model.
- **Data is sharded** across workers (different batches).
- After each step, workers **synchronize gradients** (all-reduce).

```python
# Pseudo-code for data parallel training
model = build_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=...)

for step, batch in enumerate(dataloader_shard):
    # 1. Forward
    outputs = model(batch['inputs'])
    loss = loss_fn(outputs, batch['targets'])

    # 2. Backward
    loss.backward()

    # 3. Gradient synchronization
    #    (All-reduce across data-parallel workers)
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size

    # 4. Optimizer step
    optimizer.step()
    optimizer.zero_grad()
```

**Pros:**
- Simple to implement
- Works well up to hundreds of GPUs

**Cons:**
- Memory bound by single-GPU model footprint
- Communication cost increases with model size

### 2. Model Parallelism

- Split the model itself **across GPUs**.
- Useful for **very large models** that don't fit in one GPU.

```python
# Simple tensor model-parallel example (2 GPUs)

class ParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Split weight across GPUs
        self.w0 = torch.nn.Parameter(
            torch.randn(in_features, out_features // 2, device='cuda:0')
        )
        self.w1 = torch.nn.Parameter(
            torch.randn(in_features, out_features // 2, device='cuda:1')
        )

    def forward(self, x):
        # Broadcast input to both GPUs
        x0 = x.to('cuda:0')
        x1 = x.to('cuda:1')

        y0 = x0 @ self.w0
        y1 = x1 @ self.w1

        # Gather outputs
        y = torch.cat([y0.to('cuda:0'), y1.to('cuda:0')], dim=-1)
        return y
```

**Pros:**
- Enables training of huge models

**Cons:**
- Complex to implement and debug
- Load balancing challenges

### 3. Pipeline Parallelism

- Split model layers into **stages** and place each on a different GPU.
- Micro-batches flow through the pipeline.

```text
Stage 0 (GPU0): Layers 0-3
Stage 1 (GPU1): Layers 4-7
Stage 2 (GPU2): Layers 8-11
...
```

**Pros:**
- Scales deep models
- Good for Transformer-style architectures

**Cons:**
- Pipeline bubbles (idle time)
- Complex scheduling (GPipe, 1F1B, etc.)

### 4. Hybrid Parallelism

Real systems often combine:

- **Data parallel** across nodes
- **Model + pipeline parallel** within node

This is how **GPT-3 / PaLM / LLaMA**-scale models are trained.

## Data Layer: Handling Large-Scale Sequential Data

### 1. Sharding Sequential Data

For text/audio:

- Break into **shards**:
  - `shard-0000.tfrecord`
  - `shard-0001.tfrecord`
  - ...
- Each shard contains sequences (e.g., 1K–10K examples).
- Workers **randomly sample shards** to avoid hot spots.

```python
class ShardedDataset(torch.utils.data.IterableDataset):
    \"\"\"Distributed sharded dataset for large-scale sequential data.\"\"\"\n    def __init__(self, shard_paths: list[str], rank: int, world_size: int):
        super().__init__()
        # Assign subset of shards to each worker
        self.shard_paths = shard_paths[rank::world_size]

    def __iter__(self):
        for shard_path in self.shard_paths:
            for sample in self._read_shard(shard_path):
                yield sample

    def _read_shard(self, path: str):
        # Read compressed records (e.g., TFRecord/WebDataset)
        # Yield token/audio sequences lazily
        pass
```

### 2. Sequence Packing & Bucketing

- Group sequences by length (bucketing) to minimize padding.
- Pack multiple shorter sequences into a single training sample.

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

### 3. Streaming Input Pipeline

```python
def build_dataloader(shards, batch_size, rank, world_size):
    dataset = ShardedDataset(shards, rank, world_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        prefetch_factor=4
    )
    return dataloader
```

## Communication Layer: All-Reduce & Collectives

### All-Reduce with NCCL

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

- **Ring all-reduce:** Good bandwidth utilization
- **Tree all-reduce:** Better latency for many workers

Real systems (e.g., NCCL) choose dynamically based on topology.

## Checkpointing & Fault Tolerance

### Checkpoint Strategy

```python
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
- Save to **distributed storage** (S3, GCS, HDFS)
- Use **sharded checkpoints** for huge models
- Maintain **multiple generations** (last N checkpoints)

### Fault Tolerance

1. **Worker failure**:
   - Restart worker
   - Reload last checkpoint
   - Rejoin training group
2. **Node failure**:
   - Kubernetes reschedules pods
   - Use **elastic training** (e.g., TorchElastic, Ray Train)

## Monitoring & Metrics

### Key Metrics

- **Throughput:** samples/sec, tokens/sec
- **GPU utilization:** 0–100%
- **Network bandwidth:** Gbps
- **Step time:** ms/step
- **Gradient all-reduce time**
- **Checkpoint time**

```python
class TrainingMetrics:
    def __init__(self):
        self.step_times = []
        self.throughputs = []

    def log_step(self, step_time, samples):
        self.step_times.append(step_time)
        self.throughputs.append(samples / step_time)

    @property
    def avg_step_time(self):
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0

    @property
    def avg_throughput(self):
        return sum(self.throughputs) / len(self.throughputs) if self.throughputs else 0
```

## Real-World Case Study: GPT-3 / LLaMA-Scale Training

### OpenAI / Meta-Style Setup (Conceptual)

- **Model:** 10B–70B parameter Transformer
- **Data:** Trillions of tokens (text), multi-terabyte corpora
- **Hardware:** 1024+ A100/H100 GPUs

**Parallelism:**
- 8-way tensor parallel (model parallel)
- 16-way pipeline parallel
- 8-way data parallel

**Key Techniques:**
- Mixed precision (FP16/BF16)
- Gradient checkpointing
- ZeRO-style optimizer sharding (DeepSpeed, ZeRO-3)
- Overlapping communication and computation

**Results:**
- Training time reduced from months to weeks
- Sustained 50–80% of theoretical FLOPs
- Billions of tokens processed per hour

## Cost Analysis

### Example: 1B-Parameter Model Training

Assume:
- 1B parameters
- 1024 tokens/sequence
- 1T tokens total
- 128 GPUs (A100, $3/hr each)

| Component | Value |
|----------|-------|
| Tokens/sec/GPU | ~10K |
| Total tokens/sec | 1.28M |
| Time to process 1T tokens | ~9 days |
| GPU cost/day | 128 × $3 = $384 |
| **Total cost** | **~$3,456** |

**Optimization Levers:**
- Larger batch sizes (better hardware utilization)
- Mixed precision (TFLOPs ↑)
- Better data pipeline (avoid I/O bottlenecks)
- Spot instances for non-critical training

## Key Takeaways

✅ **Distributed training is fundamentally about parallelizing sequential processing** of huge datasets.

✅ **Data parallelism** is the workhorse; model/pipeline parallelism unlocks giant models.

✅ **Handling large-scale sequential data** requires sharding, streaming, and stateful processing.

✅ **Communication (all-reduce) is often the bottleneck**—optimize with NCCL/topology-aware algorithms.

✅ **Resilience and checkpointing** are mandatory at 100s–1000s of GPUs.

✅ **Monitoring throughput and utilization** is crucial for cost efficiency.

✅ **Same sequential pattern** as Add Two Numbers: process one chunk at a time, maintain small state (`carry`/optimizer state), aggregate results.

### Connection to Thematic Link: Handling Large-Scale Sequential Data

All three topics share the same core pattern:

**DSA (Add Two Numbers - Linked List):**
- Process digits sequentially
- Maintain carry (small state)
- Scale to arbitrarily long numbers

**ML System Design (Distributed Training Architecture):**
- Process sequences of tokens/audio frames
- Maintain optimizer/model state across steps
- Scale to petabytes of data and billions of parameters

**Speech Tech (Distributed Speech Training):**
- Process long audio sequences
- Maintain streaming state (hidden states, accumulators)
- Scale to millions of hours of speech

The **sequential, stateful processing pattern** is universal—from linked lists to massive distributed training jobs.

---

**Originally published at:** [arunbaby.com/ml-system-design/0017-distributed-training-architecture](https://www.arunbaby.com/ml-system-design/0017-distributed-training-architecture/)

*If you found this helpful, consider sharing it with others who might benefit.*



