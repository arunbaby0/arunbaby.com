---
title: "Compute Allocation for Speech Models"
day: 13
related_dsa_day: 13
related_ml_day: 13
related_agents_day: 13
collection: speech_tech
categories:
 - speech-tech
tags:
 - compute-allocation
 - speech-pipeline
 - optimization
 - real-time-processing
 - inference-optimization
 - asr
 - tts
subdomain: "Speech Infrastructure"
tech_stack: [PyTorch, ONNX, TensorRT, Triton, Kubernetes, CUDA]
scale: "10K+ requests/sec, <100ms latency, multi-model pipeline"
companies: [Google, Amazon, Apple, Microsoft, Meta]
---

**Optimize speech pipeline throughput by allocating compute to bottleneck stages using greedy resource management.**

## Problem Statement

Design a **compute allocation system for speech processing pipelines** that efficiently distributes CPU/GPU resources across multiple stages (feature extraction, acoustic model, language model, post-processing) to maximize throughput while meeting strict latency SLAs.

### Functional Requirements

1. **Multi-stage pipeline:** Allocate resources across 4-6 pipeline stages
2. **Real-time processing:** Meet <100ms latency for streaming ASR
3. **Dynamic scaling:** Adjust allocation based on load and bottlenecks
4. **Multi-model support:** Handle ASR, TTS, speaker recognition, etc.
5. **Heterogeneous compute:** Mix of CPU (feature extraction) and GPU (neural models)
6. **Batch optimization:** Dynamic batching for GPU efficiency
7. **Quality-aware:** Maintain accuracy while optimizing for speed
8. **Cost-efficient:** Minimize cloud spending per request

### Non-Functional Requirements

1. **Latency:** p95 < 100ms for ASR, <200ms for TTS
2. **Throughput:** 10,000+ concurrent requests
3. **Accuracy:** WER < 5% (ASR), MOS > 4.0 (TTS)
4. **Availability:** 99.95% uptime
5. **Cost:** <$0.001 per request
6. **GPU utilization:** >80%
7. **Scalability:** Handle 10x traffic spikes

## Understanding the Problem

Speech processing pipelines are **compute-intensive** and **latency-sensitive**. Poor compute allocation leads to:

- **Bottlenecks:** One slow stage limits entire pipeline throughput
- **Wasted resources:** Over-provisioning fast stages wastes money
- **Latency violations:** Under-provisioning causes SLA breaches
- **Poor GPU utilization:** Inefficient batching leaves GPUs idle

### Typical Speech Pipeline

``
Audio Input (16kHz PCM)
 ↓
┌─────────────────────────────────────────────────────────────┐
│ Speech Pipeline │
├─────────────────────────────────────────────────────────────┤
│ │
│ Stage 1: Feature Extraction (CPU) │
│ - Convert audio to mel spectrograms │
│ - Time: ~5ms per 100ms audio │
│ - Memory: 1MB per request │
│ ↓ │
│ Stage 2: Acoustic Model (GPU) │
│ - Neural network (Conformer/Wav2Vec2) │
│ - Time: ~20ms per 100ms audio (batched) │
│ - Memory: 500MB model + 10MB per request │
│ ↓ │
│ Stage 3: Language Model (GPU/CPU) │
│ - Beam search with n-gram or neural LM │
│ - Time: ~15ms per 100ms audio │
│ - Memory: 2GB model + 5MB per request │
│ ↓ │
│ Stage 4: Post-processing (CPU) │
│ - Punctuation, capitalization, formatting │
│ - Time: ~2ms per request │
│ - Memory: 100KB per request │
│ ↓ │
│ Text Output │
└─────────────────────────────────────────────────────────────┘

Total latency: ~42ms (with perfect pipelining)
Bottleneck: Acoustic Model (47% of time)
``

### The Greedy Optimization Connection

Just like the **Container With Most Water** problem and **Resource Allocation for ML** systems:

| Container Problem | Speech Compute Allocation |
|-------------------|---------------------------|
| Two lines (heights) | Multiple pipeline stages |
| Bottleneck (shorter line) | Slowest stage limits throughput |
| Maximize area | Maximize throughput |
| Greedy: move shorter pointer | Greedy: allocate to bottleneck |
| Width vs height tradeoff | Latency vs throughput tradeoff |

**Core insight:** Identify the bottleneck stage and allocate resources greedily to maximize end-to-end throughput.

## High-Level Architecture

``
┌─────────────────────────────────────────────────────────────────┐
│ Compute Allocation Controller │
│ │
│ ┌──────────────────────┐ ┌──────────────────────┐ │
│ │ Profiler │ │ Optimizer │ │
│ │ - Measure latency │─────▶│ - Identify │ │
│ │ - Track utilization │ │ bottleneck │ │
│ │ - Detect bottleneck │ │ - Reallocation │ │
│ └──────────────────────┘ │ strategy │ │
│ └──────────┬───────────┘ │
│ │ │
│ ▼ │
│ ┌──────────────────────┐ │
│ │ Resource Manager │ │
│ │ - CPU pool │ │
│ │ - GPU pool │ │
│ │ - Batch scheduler │ │
│ └──────────┬───────────┘ │
└────────────────────────────────────────────┼────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Speech Pipeline Workers │
│ │
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│ │ Feature │ │ Acoustic │ │ Language │ │ Post- │ │
│ │ Extract │─▶│ Model │─▶│ Model │─▶│ Process │ │
│ │ │ │ │ │ │ │ │ │
│ │ CPU × N │ │ GPU × M │ │ GPU × K │ │ CPU × P │ │
│ └────────────┘ └────────────┘ └────────────┘ └──────────┘ │
│ │
│ Compute: 4 CPUs → 2 GPUs → 1 GPU → 2 CPUs (example) │
└─────────────────────────────────────────────────────────────────┘
``

### Key Components

1. **Profiler:** Continuously measures stage latencies and resource utilization
2. **Optimizer:** Identifies bottlenecks and computes optimal allocation
3. **Resource Manager:** Executes allocation decisions (spawn/kill workers)
4. **Pipeline Workers:** Actual compute resources running each stage

## Component Deep-Dives

### 1. Pipeline Profiler - Bottleneck Detection

The profiler tracks per-stage metrics to identify bottlenecks.

``python
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime
import numpy as np

@dataclass
class StageMetrics:
 """Metrics for a single pipeline stage."""
 stage_name: str
 latency_ms: deque # Rolling window of latencies
 utilization: float # 0.0 to 1.0
 throughput_rps: float # Requests per second
 queue_size: int
 num_workers: int
 worker_type: str # "CPU" or "GPU"
 
 def __post_init__(self):
 if not isinstance(self.latency_ms, deque):
 self.latency_ms = deque(maxlen=1000) # Last 1000 requests
 
 @property
 def avg_latency_ms(self) -> float:
 """Average latency over window."""
 return np.mean(self.latency_ms) if self.latency_ms else 0.0
 
 @property
 def p95_latency_ms(self) -> float:
 """P95 latency over window."""
 return np.percentile(self.latency_ms, 95) if self.latency_ms else 0.0
 
 @property
 def p99_latency_ms(self) -> float:
 """P99 latency over window."""
 return np.percentile(self.latency_ms, 99) if self.latency_ms else 0.0
 
 @property
 def is_bottleneck(self) -> bool:
 """
 Heuristic: stage is bottleneck if:
 1. High utilization (>80%)
 2. Growing queue
 3. High latency variance
 """
 high_utilization = self.utilization > 0.80
 has_queue = self.queue_size > 10
 high_variance = (
 self.p99_latency_ms > 1.5 * self.avg_latency_ms
 if self.latency_ms else False
 )
 
 return high_utilization and (has_queue or high_variance)


class PipelineProfiler:
 """
 Profiles speech pipeline to identify bottlenecks.
 
 Similar to Container With Most Water:
 - Each stage is a "line" with capacity (height)
 - Bottleneck stage (shortest line) limits throughput (area)
 """
 
 def __init__(self, stages: List[str]):
 self.stages = stages
 self.metrics: Dict[str, StageMetrics] = {
 stage: StageMetrics(
 stage_name=stage,
 latency_ms=deque(maxlen=1000),
 utilization=0.0,
 throughput_rps=0.0,
 queue_size=0,
 num_workers=1,
 worker_type="CPU" if stage in ["feature_extraction", "post_process"] else "GPU"
 )
 for stage in stages
 }
 self.request_count = 0
 self.start_time = datetime.now()
 
 def record_latency(self, stage: str, latency_ms: float):
 """Record latency measurement for a stage."""
 if stage in self.metrics:
 self.metrics[stage].latency_ms.append(latency_ms)
 self.request_count += 1
 
 def update_utilization(self, stage: str, utilization: float):
 """Update utilization measurement."""
 if stage in self.metrics:
 self.metrics[stage].utilization = utilization
 
 def update_queue_size(self, stage: str, queue_size: int):
 """Update queue size."""
 if stage in self.metrics:
 self.metrics[stage].queue_size = queue_size
 
 def identify_bottleneck(self) -> Optional[str]:
 """
 Identify bottleneck stage using greedy heuristic.
 
 Greedy choice: stage with highest "pressure" score.
 Pressure = weighted combination of:
 - Latency (40%)
 - Utilization (30%)
 - Queue size (30%)
 
 Returns:
 Bottleneck stage name or None
 """
 if not self.metrics:
 return None
 
 max_pressure = 0.0
 bottleneck = None
 
 # Normalize metrics for comparison
 max_latency = max(m.avg_latency_ms for m in self.metrics.values())
 max_queue = max(m.queue_size for m in self.metrics.values())
 
 for stage, metrics in self.metrics.items():
 # Calculate pressure score
 latency_score = (
 metrics.avg_latency_ms / max_latency if max_latency > 0 else 0
 )
 util_score = metrics.utilization
 queue_score = (
 metrics.queue_size / max_queue if max_queue > 0 else 0
 )
 
 # Weighted pressure
 pressure = (
 0.40 * latency_score +
 0.30 * util_score +
 0.30 * queue_score
 )
 
 if pressure > max_pressure:
 max_pressure = pressure
 bottleneck = stage
 
 return bottleneck if max_pressure > 0.5 else None
 
 def get_pipeline_summary(self) -> Dict:
 """Get overall pipeline statistics."""
 total_latency = sum(m.avg_latency_ms for m in self.metrics.values())
 
 # Find bottleneck
 bottleneck = self.identify_bottleneck()
 bottleneck_metrics = self.metrics.get(bottleneck) if bottleneck else None
 
 # Calculate end-to-end throughput
 # Limited by bottleneck stage
 if bottleneck_metrics:
 e2e_throughput = (
 bottleneck_metrics.num_workers * 
 (1000.0 / bottleneck_metrics.avg_latency_ms)
 if bottleneck_metrics.avg_latency_ms > 0 else 0
 )
 else:
 e2e_throughput = 0
 
 return {
 "total_requests": self.request_count,
 "avg_latency_ms": total_latency,
 "bottleneck_stage": bottleneck,
 "bottleneck_latency_ms": (
 bottleneck_metrics.avg_latency_ms if bottleneck_metrics else 0
 ),
 "estimated_throughput_rps": e2e_throughput,
 "stage_breakdown": {
 stage: {
 "avg_latency_ms": m.avg_latency_ms,
 "p95_latency_ms": m.p95_latency_ms,
 "utilization": m.utilization,
 "queue_size": m.queue_size,
 "is_bottleneck": m.is_bottleneck,
 }
 for stage, m in self.metrics.items()
 }
 }
``

### 2. Compute Optimizer - Greedy Allocation Strategy

The optimizer decides how to allocate compute resources to maximize throughput.

``python
from typing import Tuple, List
import math

@dataclass
class ComputeResource:
 """A compute resource (CPU core or GPU)."""
 resource_id: str
 resource_type: str # "CPU" or "GPU"
 cost_per_hour: float
 max_batch_size: int = 1 # For GPUs
 
@dataclass
class AllocationPlan:
 """Compute allocation plan for pipeline."""
 stage_allocations: Dict[str, int] # stage -> num_workers
 expected_throughput_rps: float
 expected_latency_ms: float
 estimated_cost_per_hour: float
 

class ComputeOptimizer:
 """
 Greedy optimizer for compute allocation.
 
 Strategy (like Container With Most Water):
 1. Identify bottleneck stage (shortest line)
 2. Allocate more resources to bottleneck (greedy choice)
 3. Repeat until:
 - Throughput target met
 - Budget exhausted
 - Bottleneck shifts to different stage
 """
 
 def __init__(
 self,
 profiler: PipelineProfiler,
 target_throughput_rps: float,
 max_latency_ms: float,
 budget_per_hour: float
 ):
 self.profiler = profiler
 self.target_throughput = target_throughput_rps
 self.max_latency = max_latency_ms
 self.budget = budget_per_hour
 
 # Resource costs (example AWS pricing)
 self.cpu_cost = 0.10 # per core per hour
 self.gpu_cost = 3.00 # per GPU per hour (T4)
 
 def compute_optimal_allocation(self) -> AllocationPlan:
 """
 Compute optimal resource allocation using greedy algorithm.
 
 Greedy approach:
 1. Start with minimal allocation (1 worker per stage)
 2. Iteratively add resources to bottleneck
 3. Stop when target met or budget exhausted
 
 Time: O(N × M) where N=stages, M=max_workers
 Similar to two-pointer approach in container problem
 """
 # Start with baseline allocation
 allocation = {
 stage: 1
 for stage in self.profiler.stages
 }
 
 # Iteratively improve
 max_iterations = 100
 for iteration in range(max_iterations):
 # Simulate current allocation
 throughput, latency, cost = self._simulate_allocation(allocation)
 
 # Check if targets met
 if (throughput >= self.target_throughput and
 latency <= self.max_latency and
 cost <= self.budget):
 # Success!
 return AllocationPlan(
 stage_allocations=allocation,
 expected_throughput_rps=throughput,
 expected_latency_ms=latency,
 estimated_cost_per_hour=cost
 )
 
 # Greedy: add resource to bottleneck
 bottleneck = self._find_bottleneck_stage(allocation)
 if not bottleneck:
 break
 
 # Check if adding resource exceeds budget
 new_cost = self._calculate_incremental_cost(bottleneck, allocation)
 if cost + new_cost > self.budget:
 break # Budget constraint
 
 # Add resource to bottleneck (greedy choice)
 allocation[bottleneck] += 1
 
 # Return best effort allocation
 throughput, latency, cost = self._simulate_allocation(allocation)
 return AllocationPlan(
 stage_allocations=allocation,
 expected_throughput_rps=throughput,
 expected_latency_ms=latency,
 estimated_cost_per_hour=cost
 )
 
 def _simulate_allocation(
 self,
 allocation: Dict[str, int]
 ) -> Tuple[float, float, float]:
 """
 Simulate pipeline performance with given allocation.
 
 Returns:
 (throughput_rps, latency_ms, cost_per_hour)
 """
 # Get baseline metrics from profiler
 summary = self.profiler.get_pipeline_summary()
 
 # Calculate per-stage throughput
 stage_throughputs = {}
 for stage, num_workers in allocation.items():
 metrics = self.profiler.metrics[stage]
 
 if metrics.avg_latency_ms > 0:
 # Throughput = workers / latency
 # With batching for GPU stages
 batch_factor = 1.0
 if metrics.worker_type == "GPU":
 batch_factor = min(8, num_workers * 2) # Assume batch size ~8-16
 
 throughput = (
 num_workers * batch_factor * 1000.0 / metrics.avg_latency_ms
 )
 stage_throughputs[stage] = throughput
 else:
 stage_throughputs[stage] = float('inf')
 
 # End-to-end throughput limited by slowest stage
 min_throughput = min(stage_throughputs.values())
 
 # End-to-end latency is sum of stage latencies
 # (assuming perfect pipelining, otherwise add queuing delays)
 total_latency = sum(
 self.profiler.metrics[stage].avg_latency_ms
 for stage in self.profiler.stages
 )
 
 # Calculate cost
 cost = 0.0
 for stage, num_workers in allocation.items():
 worker_type = self.profiler.metrics[stage].worker_type
 if worker_type == "GPU":
 cost += num_workers * self.gpu_cost
 else:
 cost += num_workers * self.cpu_cost
 
 return min_throughput, total_latency, cost
 
 def _find_bottleneck_stage(self, allocation: Dict[str, int]) -> Optional[str]:
 """
 Find bottleneck stage given current allocation.
 
 Bottleneck = stage with lowest throughput capacity.
 (Like finding shorter line in container problem)
 """
 min_throughput = float('inf')
 bottleneck = None
 
 for stage in self.profiler.stages:
 metrics = self.profiler.metrics[stage]
 num_workers = allocation[stage]
 
 if metrics.avg_latency_ms > 0:
 # Calculate stage throughput
 batch_factor = 1.0
 if metrics.worker_type == "GPU":
 batch_factor = min(8, num_workers * 2)
 
 throughput = (
 num_workers * batch_factor * 1000.0 / metrics.avg_latency_ms
 )
 
 if throughput < min_throughput:
 min_throughput = throughput
 bottleneck = stage
 
 return bottleneck
 
 def _calculate_incremental_cost(
 self,
 stage: str,
 current_allocation: Dict[str, int]
 ) -> float:
 """Calculate cost of adding one more worker to stage."""
 worker_type = self.profiler.metrics[stage].worker_type
 return self.gpu_cost if worker_type == "GPU" else self.cpu_cost
``

### 3. Dynamic Batch Scheduler - GPU Optimization

For GPU stages (acoustic model, language model), batching is critical for efficiency.

``python
import asyncio
from asyncio import Queue
from typing import List
import time

@dataclass
class SpeechRequest:
 """A speech processing request."""
 request_id: str
 audio_data: bytes
 duration_ms: float
 timestamp: float
 
class DynamicBatchScheduler:
 """
 Dynamic batching for GPU inference.
 
 Trade-off:
 - Large batches: Higher throughput, higher latency
 - Small batches: Lower latency, lower throughput
 
 Greedy strategy:
 - Wait for batch to fill up to `target_batch_size`
 - But timeout after `max_wait_ms` to maintain latency SLA
 """
 
 def __init__(
 self,
 target_batch_size: int = 16,
 max_wait_ms: float = 10.0,
 max_queue_size: int = 1000
 ):
 self.target_batch_size = target_batch_size
 self.max_wait_ms = max_wait_ms / 1000.0 # Convert to seconds
 self.queue: Queue[SpeechRequest] = Queue(maxsize=max_queue_size)
 self.batch_count = 0
 
 async def add_request(self, request: SpeechRequest):
 """Add request to batch queue."""
 await self.queue.put(request)
 
 async def get_batch(self) -> List[SpeechRequest]:
 """
 Get next batch using greedy strategy.
 
 Greedy decision:
 1. If batch_size reached: return immediately (maximize throughput)
 2. If timeout: return partial batch (maintain latency SLA)
 3. Else: keep waiting
 
 Returns:
 List of requests (1 to target_batch_size)
 """
 batch = []
 start_time = time.time()
 
 while len(batch) < self.target_batch_size:
 remaining_time = self.max_wait_ms - (time.time() - start_time)
 
 # Timeout check (latency SLA)
 if remaining_time <= 0 and batch:
 break # Return partial batch
 
 try:
 # Wait for next request (with timeout)
 request = await asyncio.wait_for(
 self.queue.get(),
 timeout=max(remaining_time, 0.001)
 )
 batch.append(request)
 
 # Greedy: if we have enough, return immediately
 if len(batch) >= self.target_batch_size:
 break
 
 except asyncio.TimeoutError:
 # Timeout - return what we have
 if batch:
 break
 else:
 continue # Keep waiting if empty
 
 self.batch_count += 1
 return batch
 
 def get_stats(self) -> Dict:
 """Get batching statistics."""
 return {
 "queue_size": self.queue.qsize(),
 "batch_count": self.batch_count,
 "avg_batch_size": "N/A", # Would track in production
 }


# Example usage in acoustic model inference
class AcousticModelWorker:
 """GPU worker for acoustic model inference with batching."""
 
 def __init__(self, model, device="cuda"):
 self.model = model
 self.device = device
 self.scheduler = DynamicBatchScheduler(
 target_batch_size=16,
 max_wait_ms=10.0
 )
 
 async def process_loop(self):
 """Main processing loop."""
 while True:
 # Get batch (greedy batching)
 batch = await self.scheduler.get_batch()
 
 if not batch:
 await asyncio.sleep(0.001)
 continue
 
 # Process batch on GPU
 results = await self._inference_batch(batch)
 
 # Return results to each request
 # ... send results back ...
 
 async def _inference_batch(self, batch: List[SpeechRequest]):
 """Run batched inference on GPU."""
 # Prepare batch
 # Run model
 # Return results
 pass
``

### 4. Resource Manager - Execute Allocation

``python
import subprocess
from typing import Dict, List

class ResourceManager:
 """
 Manages compute resources (spawn/kill workers).
 
 Executes allocation decisions from optimizer.
 """
 
 def __init__(self):
 self.workers: Dict[str, List[subprocess.Popen]] = {}
 for stage in ["feature_extraction", "acoustic_model", "language_model", "post_process"]:
 self.workers[stage] = []
 
 def apply_allocation(self, plan: AllocationPlan):
 """
 Apply allocation plan by spawning/killing workers.
 
 Greedy approach:
 1. Calculate delta (target - current)
 2. Spawn new workers if delta > 0
 3. Kill excess workers if delta < 0
 """
 for stage, target_count in plan.stage_allocations.items():
 current_count = len(self.workers[stage])
 delta = target_count - current_count
 
 if delta > 0:
 # Spawn new workers
 self._spawn_workers(stage, delta)
 elif delta < 0:
 # Kill excess workers
 self._kill_workers(stage, abs(delta))
 
 def _spawn_workers(self, stage: str, count: int):
 """Spawn worker processes."""
 for i in range(count):
 # In production: spawn Kubernetes pod or start process
 # Example: subprocess.Popen(["python", f"{stage}_worker.py"])
 pass
 
 def _kill_workers(self, stage: str, count: int):
 """Gracefully terminate workers."""
 for i in range(count):
 if self.workers[stage]:
 worker = self.workers[stage].pop()
 # worker.terminate()
 # worker.wait(timeout=30)
``

## Data Flow

### Request Processing Flow

``
1. Request arrives
 └─> Load balancer routes to available feature extraction worker

2. Feature Extraction (CPU)
 └─> Extract mel spectrogram (5ms)
 └─> Send to batch scheduler for acoustic model

3. Acoustic Model (GPU) - Batching
 └─> Wait for batch (up to 10ms)
 └─> Process batch of 16 requests (20ms)
 └─> Amortized: ~1.25ms per request (batched)
 └─> Send to language model

4. Language Model (GPU)
 └─> Beam search decoding (15ms)
 └─> Send to post-processing

5. Post-processing (CPU)
 └─> Punctuation, capitalization (2ms)
 └─> Return result

Total: 5ms + 10ms + 1.25ms + 15ms + 2ms ≈ 33ms (with batching)
Without batching: 5ms + 20ms + 15ms + 2ms = 42ms
``

### Monitoring Loop

``python
async def monitoring_loop(
 profiler: PipelineProfiler,
 optimizer: ComputeOptimizer,
 resource_manager: ResourceManager
):
 """
 Continuous monitoring and reallocation loop.
 
 Every 60 seconds:
 1. Check for bottlenecks
 2. Compute optimal allocation
 3. Apply if significantly different
 """
 while True:
 # Get current state
 summary = profiler.get_pipeline_summary()
 
 # Log metrics
 print(f"Bottleneck: {summary['bottleneck_stage']}")
 print(f"Throughput: {summary['estimated_throughput_rps']:.1f} rps")
 print(f"Latency: {summary['avg_latency_ms']:.1f}ms")
 
 # Recompute optimal allocation
 new_plan = optimizer.compute_optimal_allocation()
 
 # Apply if significant change (>20% difference)
 if should_reallocate(new_plan, resource_manager):
 print(f"Reallocating: {new_plan.stage_allocations}")
 resource_manager.apply_allocation(new_plan)
 
 # Wait before next check
 await asyncio.sleep(60)


def should_reallocate(
 new_plan: AllocationPlan,
 resource_manager: ResourceManager
) -> bool:
 """Check if reallocation is worthwhile."""
 # Avoid thrashing - only reallocate if significant change
 for stage, target in new_plan.stage_allocations.items():
 current = len(resource_manager.workers[stage])
 if abs(target - current) >= 2: # At least 2 worker difference
 return True
 return False
``

## Production Deployment

### Multi-Region Architecture

``
 ┌─────────────────┐
 │ Global LB │
 │ (Route53) │
 └────────┬────────┘
 │
 ┌────────────────────┼────────────────────┐
 │ │ │
 ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
 │ us-west │ │ us-east │ │ eu-west │
 │ Region │ │ Region │ │ Region │
 └────┬────┘ └────┬────┘ └────┬────┘
 │ │ │
 ┌────▼─────────┐ ┌───▼──────────┐ ┌───▼──────────┐
 │ Pipeline │ │ Pipeline │ │ Pipeline │
 │ Cluster │ │ Cluster │ │ Cluster │
 │ │ │ │ │ │
 │ • 4 Feature │ │ • 4 Feature │ │ • 4 Feature │
 │ • 2 Acoustic │ │ • 2 Acoustic │ │ • 2 Acoustic │
 │ • 1 LM │ │ • 1 LM │ │ • 1 LM │
 │ • 2 Post │ │ • 2 Post │ │ • 2 Post │
 └──────────────┘ └──────────────┘ └──────────────┘
``

### Kubernetes Deployment

``yaml
# acoustic-model-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
 name: acoustic-model
spec:
 replicas: 2 # Managed by HPA + custom controller
 selector:
 matchLabels:
 app: acoustic-model
 template:
 metadata:
 labels:
 app: acoustic-model
 spec:
 containers:
 - name: model-server
 image: speech-pipeline/acoustic-model:v1.2.3
 resources:
 requests:
 nvidia.com/gpu: 1
 cpu: "4"
 memory: "16Gi"
 limits:
 nvidia.com/gpu: 1
 cpu: "8"
 memory: "32Gi"
 env:
 - name: BATCH_SIZE
 value: "16"
 - name: MAX_WAIT_MS
 value: "10"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
 name: acoustic-model-hpa
spec:
 scaleTargetRef:
 apiVersion: apps/v1
 kind: Deployment
 name: acoustic-model
 minReplicas: 1
 maxReplicas: 10
 metrics:
 - type: Pods
 pods:
 metric:
 name: gpu_utilization
 target:
 type: AverageValue
 averageValue: "80"
 - type: Pods
 pods:
 metric:
 name: queue_size
 target:
 type: AverageValue
 averageValue: "50"
``

### Model Optimization Techniques

``python
import torch
import tensorrt as trt
from onnx import onnx
import onnxruntime as ort

class ModelOptimizer:
 """Optimize models for production inference."""
 
 @staticmethod
 def quantize_model(model: torch.nn.Module, calibration_data):
 """
 Quantize model to INT8 for faster inference.
 
 Benefits:
 - 4x smaller model size
 - 2-4x faster inference
 - Cost: ~1-2% accuracy drop
 """
 model.eval()
 
 # Dynamic quantization (weights only)
 quantized_model = torch.quantization.quantize_dynamic(
 model,
 {torch.nn.Linear, torch.nn.Conv1d},
 dtype=torch.qint8
 )
 
 return quantized_model
 
 @staticmethod
 def export_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, path: str):
 """
 Export to ONNX for deployment.
 
 Benefits:
 - Framework agnostic
 - Optimized runtime (ONNX Runtime)
 - TensorRT compilation
 """
 model.eval()
 torch.onnx.export(
 model,
 dummy_input,
 path,
 input_names=["audio_features"],
 output_names=["logits"],
 dynamic_axes={
 "audio_features": {0: "batch_size", 1: "time"},
 "logits": {0: "batch_size", 1: "time"}
 },
 opset_version=14
 )
 
 @staticmethod
 def compile_tensorrt(onnx_path: str, engine_path: str):
 """
 Compile ONNX model to TensorRT engine.
 
 Benefits:
 - 2-6x faster on NVIDIA GPUs
 - Automatic kernel fusion
 - Mixed precision (FP16)
 """
 # Build TensorRT engine
 logger = trt.Logger(trt.Logger.WARNING)
 builder = trt.Builder(logger)
 network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
 parser = trt.OnnxParser(network, logger)
 
 # Parse ONNX
 with open(onnx_path, 'rb') as model_file:
 parser.parse(model_file.read())
 
 # Build engine
 config = builder.create_builder_config()
 config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
 config.set_flag(trt.BuilderFlag.FP16) # Enable FP16
 
 engine = builder.build_serialized_network(network, config)
 
 # Save engine
 with open(engine_path, 'wb') as f:
 f.write(engine)
 
 return engine_path


# Example usage
def optimize_acoustic_model():
 """Full optimization pipeline."""
 # 1. Load PyTorch model
 model = torch.load("acoustic_model.pt")
 
 # 2. Quantize (optional - for CPU deployment)
 quantized = ModelOptimizer.quantize_model(model, calibration_data=None)
 
 # 3. Export to ONNX
 dummy_input = torch.randn(1, 100, 80) # batch=1, time=100, features=80
 ModelOptimizer.export_to_onnx(model, dummy_input, "acoustic_model.onnx")
 
 # 4. Compile to TensorRT (for GPU deployment)
 ModelOptimizer.compile_tensorrt("acoustic_model.onnx", "acoustic_model.trt")
 
 print("Optimization complete!")
 print("- Original: ~500MB, ~20ms latency")
 print("- Quantized: ~125MB, ~15ms latency")
 print("- TensorRT: ~125MB, ~5ms latency (batched)")
``

## Scaling Strategies

### Vertical Scaling - GPU Selection

| GPU | Memory | FP16 TFLOPS | Cost/hr | Use Case |
|-----|--------|-------------|---------|----------|
| T4 | 16GB | 65 | $0.35 | Small models, inference |
| V100 | 16GB | 125 | $2.50 | Medium models |
| A10 | 24GB | 125 | $0.75 | Cost-efficient inference |
| A100 | 40GB | 312 | $3.00 | Large models, training |

**Greedy choice:** Match GPU to model size and throughput requirements.

### Horizontal Scaling - Auto-scaling Rules

``python
@dataclass
class ScalingRule:
 """Auto-scaling rule for speech pipeline."""
 metric: str
 threshold: float
 scale_up_by: int
 cooldown_seconds: int

scaling_rules = [
 ScalingRule(
 metric="gpu_utilization",
 threshold=85.0,
 scale_up_by=1,
 cooldown_seconds=120
 ),
 ScalingRule(
 metric="queue_size",
 threshold=100,
 scale_up_by=2,
 cooldown_seconds=60
 ),
 ScalingRule(
 metric="p95_latency_ms",
 threshold=150.0,
 scale_up_by=1,
 cooldown_seconds=90
 ),
]
``

## Real-World Case Study: Google Assistant

### Google's Speech Pipeline

Google Assistant processes billions of speech requests daily with <100ms latency.

**Architecture:**
1. **Multi-tiered inference:**
 - On-device: Lightweight model for simple queries
 - Edge: Medium model at regional data centers
 - Cloud: Large model for complex queries

2. **Dynamic model selection:**
 - Greedy choice: use smallest model that meets confidence threshold
 - Fallback to larger model if confidence < 0.9

3. **Batching strategy:**
 - Dynamic batch sizes: 1-32 based on queue
 - Adaptive timeout: 5-20ms based on SLA

4. **Resource allocation:**
 - Per-region optimization
 - TPU v4 pods for large models
 - GPU for medium models
 - CPU for feature extraction

**Results:**
- **p95 latency:** 85ms
- **Throughput:** 100K+ rps per region
- **GPU utilization:** 88%
- **Cost:** <$0.0005 per request

### Key Lessons

1. **Multi-tiered models:** Use appropriate model size for each query
2. **Aggressive batching:** Critical for GPU efficiency
3. **Edge deployment:** Reduces latency and cost
4. **Continuous profiling:** Identify bottlenecks in real-time
5. **Greedy allocation works:** Simple strategy scales to billions of requests

## Cost Analysis

### Cost Breakdown (10K rps speech pipeline)

| Component | Resources | Cost/hr | Cost/request |
|-----------|-----------|---------|--------------|
| Feature extraction | 40 CPUs | `4 | `0.00010 |
| Acoustic model | 10 T4 GPUs | `3.50 | `0.00009 |
| Language model | 5 T4 GPUs | `1.75 | `0.00004 |
| Post-processing | 20 CPUs | `2 | `0.00005 |
| **Total** | | **`11.25/hr** | **`0.00028** |

**Optimization strategies:**

1. **Batching:** Reduces GPU count by 50%
 - Before: 20 GPUs @ `0.35/hr = `7/hr
 - After: 10 GPUs @ `0.35/hr = `3.50/hr
 - Savings: **50%**

2. **Model quantization:** Reduces GPU count by 30%
 - INT8 models are 2-3x faster
 - Need fewer GPUs for same throughput
 - Savings: **30%**

3. **Right-sizing instances:**
 - Use T4 (`0.35/hr) instead of V100 (`2.50/hr)
 - Savings: **86%**

4. **Spot instances:**
 - 70% discount on interruptible workloads
 - Use for batch processing, not real-time
 - Savings: **70%** (for applicable workloads)

**Total optimized cost:** $0.00012 per request (57% reduction)

## Key Takeaways

✅ **Speech pipelines have bottlenecks** - identify and optimize the slowest stage first (greedy)

✅ **Dynamic batching is critical** for GPU efficiency - trade off latency vs throughput

✅ **Continuous profiling** identifies bottlenecks in real-time

✅ **Greedy allocation strategy** - add resources to bottleneck stage iteratively

✅ **Model optimization** (quantization, TensorRT) reduces compute requirements by 50%+

✅ **Multi-region deployment** reduces latency and improves availability

✅ **Right-sizing GPU types** saves 80%+ on costs

✅ **Kubernetes + auto-scaling** enables dynamic resource allocation

✅ **Same principles as DSA** - bottleneck (shorter line) limits throughput (area)

✅ **Same principles as ML systems** - greedy optimization for resource allocation

### Connection to Thematic Link: Greedy Optimization and Resource Management

All three topics converge on the same fundamental insight:

**DSA (Container With Most Water):**
- Two lines with heights h₁, h₂
- Container area = min(h₁, h₂) × width
- Bottleneck: shorter line limits capacity
- **Greedy:** Move pointer at shorter line

**ML System Design (Resource Allocation):**
- Multiple ML jobs competing for GPUs
- System throughput limited by resource bottleneck
- **Greedy:** Allocate to highest-priority job that fits

**Speech Tech (Compute Allocation):**
- Multi-stage pipeline with different latencies
- End-to-end throughput limited by slowest stage
- **Greedy:** Allocate compute to bottleneck stage

### Universal Principle

**The Bottleneck Principle:**
> In any multi-component system, the component with the lowest capacity determines the overall system throughput.

**Greedy Optimization:**
> Iteratively improve the bottleneck until:
> 1. Target performance achieved
> 2. Budget exhausted
> 3. Bottleneck shifts to different component

This principle applies to:
- Algorithm design (two-pointer technique)
- Infrastructure (resource allocation)
- Production systems (pipeline optimization)
- Real-time processing (compute allocation)

**Why it works:**
- **Simple:** Easy to implement and reason about
- **Fast:** O(N) time complexity
- **Effective:** Proven to work at scale (Google, Meta, etc.)
- **Robust:** Handles dynamic workloads and changing bottlenecks

---

**Originally published at:** [arunbaby.com/speech-tech/0013-compute-allocation-for-speech-models](https://www.arunbaby.com/speech-tech/0013-compute-allocation-for-speech-models/)

*If you found this helpful, consider sharing it with others who might benefit.*

