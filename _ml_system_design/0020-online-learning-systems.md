---
title: "Online Learning Systems"
day: 20
related_dsa_day: 20
related_speech_day: 20
related_agents_day: 20
collection: ml_system_design
categories:
 - ml-system-design
tags:
 - online-learning
 - streaming-ml
 - adaptive-models
 - incremental-learning
 - real-time-ml
 - concept-drift
subdomain: "Real-Time Machine Learning"
tech_stack: [Python, River, Scikit-multiflow, Kafka, Redis, TensorFlow, PyTorch]
scale: "1M+ updates/day, <10ms inference, continuous adaptation"
companies: [Google, Meta, Amazon, Netflix, Uber, Spotify]
---

**Design online learning systems that adapt models in real-time using greedy updates—the same adaptive decision-making pattern from Jump Game applied to streaming data.**

## Problem Statement

Design an **Online Learning System** that continuously adapts ML models to new data without full retraining, supporting:

1. **Incremental updates** from streaming data
2. **Real-time adaptation** to distribution shifts
3. **Low-latency inference** (<10ms) and updates
4. **Concept drift detection** and handling
5. **Model versioning** with rollback capability
6. **Scale** to millions of updates per day

### Functional Requirements

1. **Streaming data ingestion:**
 - Ingest labeled samples from event streams (Kafka, Kinesis)
 - Buffer and batch micro-updates
 - Handle out-of-order arrivals

2. **Incremental model updates:**
 - Update model parameters with each new sample (or mini-batch)
 - Support various algorithms (SGD, online gradient descent, online random forests)
 - Maintain model state across updates

3. **Inference serving:**
 - Serve predictions with latest model
 - Low latency (<10ms p95)
 - High throughput (10K+ QPS)

4. **Drift detection:**
 - Monitor distribution shifts
 - Detect concept drift (when model performance degrades)
 - Alert and trigger adaptation strategies

5. **Model versioning and rollback:**
 - Version models by timestamp/update count
 - Store checkpoints periodically
 - Rollback to previous version if performance degrades

6. **Evaluation and monitoring:**
 - Track online metrics (accuracy, loss, calibration)
 - A/B test online vs batch models
 - Compare to baseline (batch-trained model)

### Non-Functional Requirements

1. **Latency:** p95 inference < 10ms, updates < 100ms
2. **Throughput:** 10K+ predictions/sec, 1M+ updates/day
3. **Availability:** 99.9% uptime
4. **Consistency:** Eventually consistent updates across replicas
5. **Resource efficiency:** Minimize memory and compute per update
6. **Freshness:** Model reflects data from last N minutes

## Understanding the Requirements

### When to Use Online Learning

**Good use cases:**
- **Fast-changing environments:** Ad click prediction, fraud detection
- **Personalization:** User preferences evolve over time
- **Limited labeled data:** Start with small dataset, improve continuously
- **Concept drift:** Distribution shifts (seasonal, trends, user behavior changes)

**Not ideal when:**
- **Stable distributions:** Offline training is simpler and often better
- **Complex models:** Deep neural networks are hard to update incrementally
- **Large batch requirements:** Models need full-batch statistics (e.g., batch normalization)

### The Greedy Adaptation Connection

Just like **Jump Game** greedily extends reach at each position:

| Jump Game | Online Learning | Adaptive Speech |
|-----------|-----------------|-----------------|
| Track max reachable index | Track model performance | Track model quality (WER) |
| Greedy: extend reach | Greedy: update weights | Greedy: adapt to speaker |
| Each step updates state | Each sample updates model | Each utterance refines model |
| Forward-looking | Predict future distribution | Anticipate corrections |
| Early termination | Early stopping | Fallback triggers |

All three use **greedy, adaptive strategies** to optimize in dynamic environments.

## High-Level Architecture

``
┌─────────────────────────────────────────────────────────────────┐
│ Online Learning System │
└─────────────────────────────────────────────────────────────────┘

 Data Sources
 ┌─────────────────────────────────────────┐
 │ User interactions │ Feedback loops │
 │ Event streams │ Label corrections│
 └──────────────┬──────────────────────────┘
 │
 ┌──────▼──────┐
 │ Kafka │
 │ (Events + │
 │ Labels) │
 └──────┬──────┘
 │
 ┌──────────────┼──────────────┐
 │ │ │
┌───────▼────────┐ ┌──▼────┐ ┌──────▼──────┐
│ Update │ │Feature│ │ Drift │
│ Service │ │Store │ │ Detector │
│ │ │(Redis)│ │ │
│ - Batch updates│ └───────┘ │ - Monitor │
│ - Gradient │ │ metrics │
│ computation │ │ - Alert │
└───────┬────────┘ └──────┬──────┘
 │ │
 └──────────────┬─────────────┘
 │
 ┌──────▼──────┐
 │ Model │
 │ Store │
 │ │
 │ - Current │
 │ - Versions │
 │ - Checkpts │
 └──────┬──────┘
 │
 ┌──────────────┼──────────────┐
 │ │ │
┌───────▼────────┐ ┌──▼────┐ ┌──────▼──────┐
│ Inference │ │Monitor│ │ A/B Test │
│ Service │ │ │ │ Controller │
│ │ │- Loss │ │ │
│ - Predictions │ │- Acc │ │ - Traffic │
│ - Low latency │ │- Drift│ │ split │
└────────────────┘ └───────┘ └─────────────┘
``

### Key Components

1. **Data Ingestion:** Stream labeled samples from Kafka
2. **Feature Store:** Low-latency feature lookup (Redis)
3. **Update Service:** Apply incremental updates to model
4. **Model Store:** Versioned model storage with checkpoints
5. **Inference Service:** Serve predictions with latest model
6. **Drift Detector:** Monitor for distribution/concept shifts
7. **A/B Test Controller:** Compare online vs batch models

## Component Deep-Dives

### 1. Incremental Update Algorithms

**Online Gradient Descent (for linear models):**

``python
import numpy as np
from typing import Dict, Any

class OnlineLinearModel:
 """
 Online linear model with SGD updates.
 
 Similar to Jump Game:
 - Each update 'extends reach' (improves model)
 - Greedy: always update towards lower error
 - Track 'max reach' (best performance so far)
 """
 
 def __init__(
 self,
 n_features: int,
 learning_rate: float = 0.01,
 regularization: float = 0.01
 ):
 self.weights = np.zeros(n_features)
 self.bias = 0.0
 self.learning_rate = learning_rate
 self.regularization = regularization
 
 # Metrics
 self.update_count = 0
 self.cumulative_loss = 0.0
 
 def predict(self, features: np.ndarray) -> float:
 """Make prediction."""
 return np.dot(self.weights, features) + self.bias
 
 def update(self, features: np.ndarray, label: float):
 """
 Incremental update with one sample (online SGD).
 
 Greedy decision: move weights to reduce error on this sample.
 Like Jump Game extending max_reach.
 """
 # Prediction
 pred = self.predict(features)
 
 # Error
 error = label - pred
 
 # Gradient descent update
 self.weights += self.learning_rate * error * features
 self.weights -= self.learning_rate * self.regularization * self.weights # L2 reg
 self.bias += self.learning_rate * error
 
 # Track metrics
 self.update_count += 1
 self.cumulative_loss += error ** 2
 
 def batch_update(self, features_batch: np.ndarray, labels_batch: np.ndarray):
 """Update with mini-batch (more stable than single samples)."""
 for features, label in zip(features_batch, labels_batch):
 self.update(features, label)
 
 def get_state(self) -> Dict:
 """Get model state for checkpointing."""
 return {
 "weights": self.weights.tolist(),
 "bias": float(self.bias),
 "update_count": self.update_count,
 "avg_loss": self.cumulative_loss / max(1, self.update_count)
 }
 
 def set_state(self, state: Dict):
 """Restore model from checkpoint."""
 self.weights = np.array(state["weights"])
 self.bias = state["bias"]
 self.update_count = state["update_count"]
``

**Online Random Forest (Mondrian Forest):**

``python
class OnlineRandomForest:
 """
 Online random forest using Mondrian trees.
 
 Supports incremental updates without full retraining.
 """
 
 def __init__(self, n_trees: int = 10):
 self.n_trees = n_trees
 self.trees = [MondrianTree() for _ in range(n_trees)]
 
 def update(self, features: np.ndarray, label: int):
 """Update all trees with new sample."""
 for tree in self.trees:
 tree.update(features, label)
 
 def predict(self, features: np.ndarray) -> int:
 """Predict by majority vote."""
 predictions = [tree.predict(features) for tree in self.trees]
 return max(set(predictions), key=predictions.count)
``

### 2. Streaming Data Pipeline

``python
from kafka import KafkaConsumer
import json
from queue import Queue
from threading import Thread

class StreamingDataPipeline:
 """
 Ingest labeled samples from Kafka for online learning.
 """
 
 def __init__(
 self,
 kafka_brokers: List[str],
 topic: str,
 batch_size: int = 32,
 batch_timeout_sec: float = 1.0
 ):
 self.consumer = KafkaConsumer(
 topic,
 bootstrap_servers=kafka_brokers,
 value_deserializer=lambda m: json.loads(m.decode('utf-8'))
 )
 
 self.batch_size = batch_size
 self.batch_timeout_sec = batch_timeout_sec
 self.sample_queue = Queue(maxsize=10000)
 
 self.running = False
 
 def start(self):
 """Start consuming from Kafka."""
 self.running = True
 Thread(target=self._consume_loop, daemon=True).start()
 
 def _consume_loop(self):
 """Consume samples from Kafka and queue them."""
 for message in self.consumer:
 if not self.running:
 break
 
 sample = message.value
 self.sample_queue.put(sample)
 
 def get_batch(self) -> List[Dict]:
 """Get a batch of samples for model update."""
 batch = []
 
 import time
 start_time = time.time()
 
 while len(batch) < self.batch_size:
 if time.time() - start_time > self.batch_timeout_sec:
 break
 
 if not self.sample_queue.empty():
 batch.append(self.sample_queue.get())
 
 return batch
``

### 3. Concept Drift Detection

``python
from collections import deque
import numpy as np

class DriftDetector:
 """
 Detect concept drift using performance monitoring.
 
 Similar to Jump Game checking if we're 'stuck':
 - Monitor if model performance is degrading
 - Trigger adaptation/retraining if drift detected
 """
 
 def __init__(
 self,
 window_size: int = 1000,
 threshold: float = 0.1
 ):
 self.window_size = window_size
 self.threshold = threshold
 
 # Recent performance
 self.recent_errors = deque(maxlen=window_size)
 self.baseline_error = None
 
 def update(self, prediction: float, label: float):
 """Update with new prediction and label."""
 error = abs(prediction - label)
 self.recent_errors.append(error)
 
 # Set baseline from first window
 if self.baseline_error is None and len(self.recent_errors) == self.window_size:
 self.baseline_error = np.mean(self.recent_errors)
 
 def detect_drift(self) -> bool:
 """
 Check if concept drift has occurred.
 
 Returns:
 True if drift detected, False otherwise
 """
 if self.baseline_error is None:
 return False
 
 if len(self.recent_errors) < self.window_size:
 return False
 
 current_error = np.mean(self.recent_errors)
 
 # Drift if error increased significantly
 return current_error > self.baseline_error * (1 + self.threshold)
 
 def reset_baseline(self):
 """Reset baseline after handling drift."""
 if self.recent_errors:
 self.baseline_error = np.mean(self.recent_errors)
``

### 4. Model Versioning

``python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelVersion:
 """Metadata for a model version."""
 version_id: str
 timestamp: float
 update_count: int
 performance_metrics: Dict
 model_state: Dict

class ModelVersionManager:
 """
 Manage model versions for online learning.
 
 Features:
 - Periodic checkpoints
 - Performance-based versioning
 - Rollback capability
 """
 
 def __init__(
 self,
 checkpoint_interval: int = 10000, # Updates between checkpoints
 max_versions: int = 10
 ):
 self.checkpoint_interval = checkpoint_interval
 self.max_versions = max_versions
 
 self.versions: List[ModelVersion] = []
 self.current_version = None
 
 def should_checkpoint(self, update_count: int) -> bool:
 """Check if we should create a checkpoint."""
 return update_count % self.checkpoint_interval == 0
 
 def create_checkpoint(
 self,
 model_state: Dict,
 update_count: int,
 metrics: Dict
 ) -> str:
 """
 Create a new model checkpoint.
 
 Returns:
 Version ID
 """
 version_id = f"v_{update_count}_{int(time.time())}"
 
 version = ModelVersion(
 version_id=version_id,
 timestamp=time.time(),
 update_count=update_count,
 performance_metrics=metrics,
 model_state=model_state
 )
 
 self.versions.append(version)
 self.current_version = version
 
 # Keep only recent versions
 if len(self.versions) > self.max_versions:
 self.versions.pop(0)
 
 return version_id
 
 def rollback(self, version_id: Optional[str] = None) -> Optional[Dict]:
 """
 Rollback to a previous version.
 
 Args:
 version_id: Specific version to rollback to, or None for previous
 
 Returns:
 Model state of the version
 """
 if not self.versions:
 return None
 
 if version_id is None:
 # Rollback to previous version
 if len(self.versions) >= 2:
 target = self.versions[-2]
 else:
 return None
 else:
 # Find specific version
 target = next((v for v in self.versions if v.version_id == version_id), None)
 if not target:
 return None
 
 self.current_version = target
 return target.model_state
``

## Data Flow

### Online Learning Pipeline

``
1. Labeled sample arrives (via Kafka)
 └─> Feature extraction/enrichment
 └─> Add to update buffer

2. Update Service (every N samples or T seconds)
 └─> Get batch from buffer
 └─> Compute gradients/updates
 └─> Apply updates to model
 └─> Update model version

3. Inference Request
 └─> Load latest model version
 └─> Extract features
 └─> Make prediction
 └─> Log prediction for feedback loop

4. Feedback Loop
 └─> Collect true labels (delayed or real-time)
 └─> Send to Kafka as new training samples
 └─> Monitor drift

5. Drift Detection (continuous)
 └─> Compare recent performance to baseline
 └─> If drift detected: alert, increase update frequency, or trigger retraining
``

## Scaling Strategies

### Horizontal Scaling - Distributed Online Learning

``python
import ray

@ray.remote
class DistributedOnlineLearner:
 """
 Distributed online learner using parameter server pattern.
 """
 
 def __init__(self, n_features: int):
 self.model = OnlineLinearModel(n_features)
 self.lock = threading.Lock()
 
 def update(self, features: np.ndarray, label: float):
 """Thread-safe update."""
 with self.lock:
 self.model.update(features, label)
 
 def get_weights(self) -> np.ndarray:
 """Get current weights."""
 with self.lock:
 return self.model.weights.copy()
 
 def predict(self, features: np.ndarray) -> float:
 """Make prediction."""
 return self.model.predict(features)


class ParameterServerSystem:
 """
 Parameter server for distributed online learning.
 
 Workers:
 - Process incoming samples
 - Compute gradients
 - Send updates to parameter server
 
 Parameter Server:
 - Aggregate updates from workers
 - Maintain global model state
 - Serve latest model for inference
 """
 
 def __init__(self, n_features: int, n_workers: int = 4):
 # Create parameter server
 self.param_server = DistributedOnlineLearner.remote(n_features)
 
 # Create workers
 self.workers = [
 DistributedOnlineLearner.remote(n_features)
 for _ in range(n_workers)
 ]
 
 self.n_workers = n_workers
 
 def update_distributed(self, samples: List[Dict]):
 """
 Distribute samples to workers for parallel updates.
 """
 # Distribute samples to workers
 chunk_size = len(samples) // self.n_workers
 
 futures = []
 for i, worker in enumerate(self.workers):
 start = i * chunk_size
 end = start + chunk_size if i < self.n_workers - 1 else len(samples)
 worker_samples = samples[start:end]
 
 # Each worker computes local updates
 for sample in worker_samples:
 futures.append(
 worker.update.remote(sample['features'], sample['label'])
 )
 
 # Wait for all updates
 ray.get(futures)
 
 # Sync workers with parameter server (simplified)
 # In production: use all-reduce or async parameter server
``

### Model Update Strategies

**1. Single-sample updates (pure online):**
``python
for sample in stream:
 model.update(sample['features'], sample['label'])
``
- Pros: Fastest adaptation
- Cons: Noisy updates, unstable

**2. Mini-batch updates:**
``python
batch = []
for sample in stream:
 batch.append(sample)
 if len(batch) >= batch_size:
 model.batch_update(batch)
 batch = []
``
- Pros: More stable, better GPU utilization
- Cons: Slightly delayed adaptation

**3. Timed updates:**
``python
last_update = time.time()
buffer = []

for sample in stream:
 buffer.append(sample)
 
 if time.time() - last_update > update_interval_sec:
 model.batch_update(buffer)
 buffer = []
 last_update = time.time()
``
- Pros: Predictable update schedule
- Cons: Variable batch sizes

## Implementation: Complete System

``python
import logging
from typing import List, Dict, Optional
import time

class OnlineLearningSystem:
 """
 Complete online learning system.
 
 Features:
 - Streaming data ingestion
 - Incremental model updates
 - Drift detection
 - Model versioning
 - Inference serving
 """
 
 def __init__(
 self,
 n_features: int,
 learning_rate: float = 0.01,
 batch_size: int = 32,
 checkpoint_interval: int = 10000
 ):
 # Core model
 self.model = OnlineLinearModel(
 n_features=n_features,
 learning_rate=learning_rate
 )
 
 # Components
 self.drift_detector = DriftDetector(window_size=1000)
 self.version_manager = ModelVersionManager(
 checkpoint_interval=checkpoint_interval
 )
 
 # Update buffer
 self.batch_size = batch_size
 self.update_buffer: List[Dict] = []
 
 self.logger = logging.getLogger(__name__)
 
 # Metrics
 self.total_updates = 0
 self.total_predictions = 0
 self.drift_events = 0
 
 def predict(self, features: np.ndarray) -> float:
 """
 Make prediction with current model.
 
 Args:
 features: Input features
 
 Returns:
 Prediction
 """
 self.total_predictions += 1
 return self.model.predict(features)
 
 def update(self, features: np.ndarray, label: float):
 """
 Queue sample for model update.
 
 Args:
 features: Input features
 label: True label (from feedback)
 """
 # Add to buffer
 self.update_buffer.append({
 "features": features,
 "label": label
 })
 
 # Update model when batch is ready
 if len(self.update_buffer) >= self.batch_size:
 self._apply_updates()
 
 def _apply_updates(self):
 """Apply batched updates to model."""
 batch = self.update_buffer
 self.update_buffer = []
 
 # Apply updates
 for sample in batch:
 self.model.update(sample['features'], sample['label'])
 self.total_updates += 1
 
 # Update drift detector
 pred = self.model.predict(sample['features'])
 self.drift_detector.update(pred, sample['label'])
 
 # Check for drift
 if self.drift_detector.detect_drift():
 self.logger.warning("Concept drift detected!")
 self._handle_drift()
 
 # Checkpoint if needed
 if self.version_manager.should_checkpoint(self.total_updates):
 self._create_checkpoint()
 
 def _handle_drift(self):
 """
 Handle concept drift.
 
 Strategies:
 1. Increase learning rate temporarily
 2. Reset model (if severe drift)
 3. Trigger full retraining
 4. Alert monitoring system
 """
 self.drift_events += 1
 
 # Simple strategy: reset drift detector baseline
 self.drift_detector.reset_baseline()
 
 # Could also:
 # - Increase learning rate
 # - Trigger alert/page
 # - Request full retrain from batch system
 
 self.logger.info(f"Handled drift event #{self.drift_events}")
 
 def _create_checkpoint(self):
 """Create model checkpoint."""
 state = self.model.get_state()
 metrics = {
 "avg_loss": state["avg_loss"],
 "total_updates": self.total_updates,
 "total_predictions": self.total_predictions
 }
 
 version_id = self.version_manager.create_checkpoint(
 model_state=state,
 update_count=self.total_updates,
 metrics=metrics
 )
 
 self.logger.info(f"Created checkpoint: {version_id}")
 
 def rollback(self, version_id: Optional[str] = None):
 """Rollback to previous model version."""
 state = self.version_manager.rollback(version_id)
 
 if state:
 self.model.set_state(state)
 self.logger.info(f"Rolled back to version {version_id}")
 else:
 self.logger.error("Rollback failed")
 
 def get_metrics(self) -> Dict:
 """Get system metrics."""
 return {
 "total_updates": self.total_updates,
 "total_predictions": self.total_predictions,
 "drift_events": self.drift_events,
 "current_version": (
 self.version_manager.current_version.version_id
 if self.version_manager.current_version
 else None
 ),
 "model_performance": self.model.get_state()["avg_loss"]
 }


# Example usage
if __name__ == "__main__":
 logging.basicConfig(level=logging.INFO)
 
 # Create system
 system = OnlineLearningSystem(
 n_features=10,
 learning_rate=0.01,
 batch_size=32
 )
 
 # Simulate streaming data
 for i in range(1000):
 # Generate sample
 features = np.random.randn(10)
 label = np.dot([0.5] * 10, features) + np.random.randn() * 0.1
 
 # Make prediction
 pred = system.predict(features)
 
 # Update model (with delay, simulating feedback loop)
 system.update(features, label)
 
 # Get metrics
 metrics = system.get_metrics()
 print(f"\nSystem metrics: {metrics}")
``

## Monitoring & Metrics

### Key Metrics to Track

**Model Performance:**
- Online loss/error (moving average)
- Online accuracy (for classification)
- Prediction drift (distribution shift)

**System Performance:**
- Update latency (time to apply update)
- Inference latency (time to predict)
- Throughput (updates/sec, predictions/sec)
- Buffer size (samples waiting for update)

**Data Quality:**
- Label delay (time from prediction to label)
- Sample arrival rate
- Feature distribution shifts

### Alerts

- Concept drift detected (performance degradation >10%)
- Update latency >100ms (system overloaded)
- Buffer overflow (can't keep up with data rate)
- Model performance below baseline (trigger rollback)

## Failure Modes

| Failure Mode | Impact | Mitigation |
|-------------|--------|------------|
| **Label delay** | Can't update model | Use semi-supervised or unsupervised proxies |
| **Data quality issues** | Model learns garbage | Input validation, outlier detection |
| **Concept drift** | Model performance degrades | Drift detection, adaptive learning rate |
| **Update lag** | Model falls behind | Increase update frequency, add workers |
| **Catastrophic forgetting** | Model forgets old patterns | Regularization, rehearsal buffers |
| **Model instability** | Oscillating performance | Decrease learning rate, use momentum |

## Real-World Case Study: Netflix Recommendation

### Netflix's Online Learning Approach

Netflix uses online learning for:
- Real-time recommendation updates
- A/B test metric computation
- Personalization based on recent viewing

**Architecture:**
1. **Event stream:** User interactions (plays, pauses, ratings) → Kafka
2. **Feature computation:** Real-time feature updates (watch history, preferences)
3. **Model updates:** Incremental updates every few minutes
4. **Inference:** Serve recommendations with latest model
5. **Evaluation:** Compare online vs batch models via A/B tests

**Results:**
- **<100ms** model update latency
- **Updates every 5 minutes** (vs daily batch retraining)
- **+5% engagement** from real-time personalization
- **Faster adaptation** to trending content

### Key Lessons

1. **Hybrid approach works best:** Batch model as baseline, online updates for fine-tuning
2. **Drift detection is critical:** Monitor and handle distribution shifts
3. **Checkpointing enables rollback:** When online updates degrade performance
4. **A/B testing validates:** Always compare online vs batch models
5. **Greedy updates can be unstable:** Use regularization and momentum

## Cost Analysis

### Infrastructure Costs (1M updates/day)

| Component | Resources | Cost/Month | Notes |
|-----------|-----------|------------|-------|
| Kafka cluster | 3 brokers | $450 | Event streaming |
| Update service | 5 instances | $500 | Apply model updates |
| Inference service | 10 instances | $1,000 | Serve predictions |
| Redis (feature store) | 1 instance | $200 | Fast feature lookup |
| Model storage | S3, 100 GB | $3 | Versioned models |
| Monitoring | Prometheus+Grafana | $100 | Metrics & alerts |
| **Total** | | **`2,253/month** | **`0.07 per 1K updates** |

**Optimization strategies:**
- Batch updates: Reduce update service cost by 50%
- Shared inference cache: Reduce duplicate predictions
- Model compression: Smaller models → faster updates
- Spot instances: 70% cost reduction for update workers

## Key Takeaways

✅ **Online learning enables real-time adaptation** to new data without full retraining

✅ **Greedy updates** (like Jump Game's greedy reach extension) work well for many models

✅ **Drift detection is critical** to maintain model quality over time

✅ **Hybrid approach** (batch baseline + online fine-tuning) often performs best

✅ **Model versioning and rollback** protect against bad updates

✅ **Mini-batch updates** balance stability and adaptation speed

✅ **Monitoring and A/B testing** validate that online learning improves over batch

✅ **Linear models work well**, but deep learning requires careful design (see adaptive speech models)

✅ **Cost vs freshness trade-off** - more frequent updates cost more but improve relevance

✅ **Same greedy pattern** as Jump Game - make locally optimal decisions, adapt forward

### Connection to Thematic Link: Greedy Decisions and Adaptive Strategies

All three topics use **greedy, adaptive optimization**:

**DSA (Jump Game):**
- Greedy: extend max reach at each position
- Adaptive: update strategy based on current state
- Forward-looking: anticipate future reachability

**ML System Design (Online Learning Systems):**
- Greedy: update model with each new sample
- Adaptive: adjust to distribution shifts via incremental learning
- Forward-looking: drift detection predicts future performance

**Speech Tech (Adaptive Speech Models):**
- Greedy: adapt to speaker/noise in real-time
- Adaptive: fine-tune acoustic model based on recent utterances
- Forward-looking: anticipate user corrections and adapt preemptively

The **unifying principle**: make greedy, locally optimal decisions while continuously adapting to new information—essential for systems operating in dynamic, uncertain environments.

---

**Originally published at:** [arunbaby.com/ml-system-design/0020-online-learning-systems](https://www.arunbaby.com/ml-system-design/0020-online-learning-systems/)

*If you found this helpful, consider sharing it with others who might benefit.*





