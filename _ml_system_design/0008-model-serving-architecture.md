---
title: "Model Serving Architecture"
day: 8
related_dsa_day: 8
related_speech_day: 8
related_agents_day: 8
collection: ml_system_design
categories:
 - ml-system-design
tags:
 - model-serving
 - inference
 - deployment
 - scalability
 - latency
subdomain: Model Deployment
tech_stack: [Python, TensorFlow Serving, TorchServe, FastAPI, Docker, Kubernetes, gRPC]
scale: "Millions of predictions/second"
companies: [Google, Meta, Uber, Netflix, Airbnb, Spotify]
---

**Design production-grade model serving systems that deliver predictions at scale with low latency and high reliability.**

## Introduction

**Model serving** is the process of deploying ML models to production and making predictions available to end users or downstream systems.

**Why it's critical:**
- **Bridge training and production:** Trained models are useless without serving
- **Performance matters:** Latency directly impacts user experience
- **Scale requirements:** Handle millions of requests per second
- **Reliability:** Downtime = lost revenue

**Key challenges:**
- Low latency (< 100ms for many applications)
- High throughput (handle traffic spikes)
- Model versioning and rollback
- A/B testing and gradual rollouts
- Monitoring and debugging

---

## Model Serving Architecture Overview

``
┌─────────────────────────────────────────────────────────┐
│ Client Applications │
│ (Web, Mobile, Backend Services) │
└────────────────────┬────────────────────────────────────┘
 │ HTTP/gRPC requests
 ▼
┌─────────────────────────────────────────────────────────┐
│ Load Balancer │
│ (nginx, ALB, GCP Load Balancer) │
└────────────────────┬────────────────────────────────────┘
 │
 ┌──────────┼──────────┐
 ▼ ▼ ▼
 ┌─────────┐ ┌─────────┐ ┌─────────┐
 │ Serving │ │ Serving │ │ Serving │
 │ Instance│ │ Instance│ │ Instance│
 │ 1 │ │ 2 │ │ N │
 └────┬────┘ └────┬────┘ └────┬────┘
 │ │ │
 ▼ ▼ ▼
 ┌────────────────────────────────┐
 │ Model Repository │
 │ (S3, GCS, Model Registry) │
 └────────────────────────────────┘
``

---

## Serving Patterns

### Pattern 1: REST API Serving

**Best for:** Web applications, microservices

``python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from typing import List
import time

app = FastAPI()

# Load model on startup
model = None

@app.on_event("startup")
async def load_model():
 """Load model when server starts"""
 global model
 model = joblib.load('model.pkl')
 print("Model loaded successfully")

class PredictionRequest(BaseModel):
 """Request schema"""
 features: List[float]
 
class PredictionResponse(BaseModel):
 """Response schema"""
 prediction: float
 confidence: float
 model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
 """
 Make prediction
 
 Returns: Prediction with confidence
 """
 try:
 # Convert to numpy array
 features = np.array([request.features])
 
 # Make prediction
 prediction = model.predict(features)[0]
 
 # Get confidence (if available)
 if hasattr(model, 'predict_proba'):
 proba = model.predict_proba(features)[0]
 confidence = float(np.max(proba))
 else:
 confidence = 1.0
 
 return PredictionResponse(
 prediction=float(prediction),
 confidence=confidence,
 model_version="v1.0"
 )
 
 except Exception as e:
 raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
 """Health check endpoint"""
 if model is None:
 raise HTTPException(status_code=503, detail="Model not loaded")
 return {"status": "healthy", "model_loaded": True}

@app.get("/ready")
async def readiness_check():
 """Readiness probe endpoint"""
 # Optionally include lightweight self-test
 return {"ready": model is not None}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
``

**Usage:**
``bash
curl -X POST "http://localhost:8000/predict" \
 -H "Content-Type: application/json" \
 -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
``

### Pattern 2: gRPC Serving

**Best for:** High-performance, low-latency applications

``python
# prediction.proto
"""
syntax = "proto3";

service PredictionService {
 rpc Predict (PredictRequest) returns (PredictResponse);
}

message PredictRequest {
 repeated float features = 1;
}

message PredictResponse {
 float prediction = 1;
 float confidence = 2;
}
"""

# server.py
import grpc
from concurrent import futures
import prediction_pb2
import prediction_pb2_grpc
import numpy as np
import joblib

class PredictionServicer(prediction_pb2_grpc.PredictionServiceServicer):
 """gRPC Prediction Service"""
 
 def __init__(self):
 self.model = joblib.load('model.pkl')
 
 def Predict(self, request, context):
 """Handle prediction request"""
 try:
 # Convert features
 features = np.array([list(request.features)])
 
 # Predict
 prediction = self.model.predict(features)[0]
 
 # Get confidence
 if hasattr(self.model, 'predict_proba'):
 proba = self.model.predict_proba(features)[0]
 confidence = float(np.max(proba))
 else:
 confidence = 1.0
 
 return prediction_pb2.PredictResponse(
 prediction=float(prediction),
 confidence=confidence
 )
 
 except Exception as e:
 context.set_code(grpc.StatusCode.INTERNAL)
 context.set_details(str(e))
 return prediction_pb2.PredictResponse()

def serve():
 """Start gRPC server"""
 server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
 prediction_pb2_grpc.add_PredictionServiceServicer_to_server(
 PredictionServicer(), server
 )
 server.add_insecure_port('[::]:50051')
 server.start()
 print("gRPC server started on port 50051")
 server.wait_for_termination()

if __name__ == '__main__':
 serve()
``

**Performance comparison:**
``
Metric REST API gRPC
───────────────────────────────────
Latency (p50) 15ms 5ms
Latency (p99) 50ms 20ms
Throughput 5K rps 15K rps
Payload size JSON Protocol Buffers (smaller)
``

### Pattern 3: Batch Serving

**Best for:** Offline predictions, large-scale inference

``python
import pandas as pd
import numpy as np
from multiprocessing import Pool
import joblib

class BatchPredictor:
 """
 Batch prediction system
 
 Efficient for processing large datasets
 """
 
 def __init__(self, model_path, batch_size=1000, n_workers=4):
 self.model = joblib.load(model_path)
 self.batch_size = batch_size
 self.n_workers = n_workers
 
 def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
 """
 Predict on large dataset
 
 Args:
 features_df: DataFrame with features
 
 Returns:
 Array of predictions
 """
 n_samples = len(features_df)
 n_batches = (n_samples + self.batch_size - 1) // self.batch_size
 
 predictions = []
 
 for i in range(n_batches):
 start_idx = i * self.batch_size
 end_idx = min((i + 1) * self.batch_size, n_samples)
 
 batch = features_df.iloc[start_idx:end_idx].values
 batch_pred = self.model.predict(batch)
 predictions.extend(batch_pred)
 
 if (i + 1) % 10 == 0:
 print(f"Processed {end_idx}/{n_samples} samples")
 
 return np.array(predictions)
 
 def predict_parallel(self, features_df: pd.DataFrame) -> np.ndarray:
 """
 Parallel batch prediction
 
 Splits data across multiple processes
 """
 # Split data into chunks
 chunk_size = len(features_df) // self.n_workers
 chunks = [
 features_df.iloc[i:i+chunk_size]
 for i in range(0, len(features_df), chunk_size)
 ]
 
 # Process in parallel
 with Pool(self.n_workers) as pool:
 results = pool.map(self._predict_chunk, chunks)
 
 # Combine results
 return np.concatenate(results)
 
 def _predict_chunk(self, chunk_df):
 """Predict on single chunk"""
 return self.model.predict(chunk_df.values)

# Usage
predictor = BatchPredictor('model.pkl', batch_size=10000, n_workers=8)

# Load large dataset
data = pd.read_parquet('features.parquet')

# Predict
predictions = predictor.predict_parallel(data)

# Save results
results_df = data.copy()
results_df['prediction'] = predictions
results_df.to_parquet('predictions.parquet')
``

---

## Model Loading Strategies

### Strategy 1: Eager Loading

``python
class EagerModelServer:
 """
 Load model on server startup
 
 Pros: Fast predictions, simple
 Cons: High startup time, high memory
 """
 
 def __init__(self, model_path):
 print("Loading model...")
 self.model = joblib.load(model_path)
 print("Model loaded!")
 
 def predict(self, features):
 """Make prediction (fast)"""
 return self.model.predict(features)
``

### Strategy 2: Lazy Loading

``python
class LazyModelServer:
 """
 Load model on first request
 
 Pros: Fast startup
 Cons: First request is slow
 """
 
 def __init__(self, model_path):
 self.model_path = model_path
 self.model = None
 
 def predict(self, features):
 """Load model if needed, then predict"""
 if self.model is None:
 print("Loading model on first request...")
 self.model = joblib.load(self.model_path)
 
 return self.model.predict(features)
``

### Strategy 3: Model Caching with Expiration

``python
from datetime import datetime, timedelta
import threading

class CachedModelServer:
 """
 Load model with cache expiration
 
 Automatically reloads model periodically
 """
 
 def __init__(self, model_path, cache_ttl_minutes=60):
 self.model_path = model_path
 self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
 self.model = None
 self.last_loaded = None
 self.lock = threading.Lock()
 
 def _load_model(self):
 """Load model with lock"""
 with self.lock:
 print(f"Loading model from {self.model_path}")
 self.model = joblib.load(self.model_path)
 self.last_loaded = datetime.now()
 
 def predict(self, features):
 """Predict with cache check"""
 # Check if model needs refresh
 if (self.model is None or 
 datetime.now() - self.last_loaded > self.cache_ttl):
 self._load_model()
 
 return self.model.predict(features)
``

---

## Model Versioning & A/B Testing

### Multi-Model Serving

``python
from enum import Enum
from typing import Dict
import random

class ModelVersion(Enum):
 V1 = "v1"
 V2 = "v2"
 V3 = "v3"

class MultiModelServer:
 """
 Serve multiple model versions
 
 Supports A/B testing and gradual rollouts
 """
 
 def __init__(self):
 self.models: Dict[str, any] = {}
 self.traffic_split = {} # version → weight
 
 def load_model(self, version: ModelVersion, model_path: str):
 """Load a specific model version"""
 print(f"Loading {version.value} from {model_path}")
 self.models[version.value] = joblib.load(model_path)
 
 def set_traffic_split(self, split: Dict[str, float]):
 """
 Set traffic distribution
 
 Args:
 split: Dict mapping version to weight
 e.g., {"v1": 0.9, "v2": 0.1}
 """
 # Validate weights sum to 1
 total = sum(split.values())
 assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1, got {total}"
 
 self.traffic_split = split
 
 def select_model(self, user_id: str = None) -> str:
 """
 Select model version based on traffic split
 
 Args:
 user_id: Optional user ID for deterministic routing
 
 Returns:
 Selected model version
 """
 if user_id:
 # Deterministic selection (consistent for same user)
 import hashlib
 hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
 rand_val = (hash_val % 10000) / 10000.0
 else:
 # Random selection
 rand_val = random.random()
 
 # Select based on cumulative weights
 cumulative = 0
 for version, weight in self.traffic_split.items():
 cumulative += weight
 if rand_val < cumulative:
 return version
 
 # Fallback to first version
 return list(self.traffic_split.keys())[0]
 
 def predict(self, features, user_id: str = None):
 """
 Make prediction with version selection
 
 Returns: (prediction, version_used)
 """
 version = self.select_model(user_id)
 model = self.models[version]
 prediction = model.predict(features)
 
 return prediction, version

# Usage
server = MultiModelServer()

# Load models
server.load_model(ModelVersion.V1, 'model_v1.pkl')
server.load_model(ModelVersion.V2, 'model_v2.pkl')

# Start with 90% v1, 10% v2
server.set_traffic_split({"v1": 0.9, "v2": 0.1})

# Make predictions
features = [[1, 2, 3, 4]]
prediction, version = server.predict(features, user_id="user_123")
print(f"Prediction: {prediction}, Version: {version}")

# Gradually increase v2 traffic
server.set_traffic_split({"v1": 0.5, "v2": 0.5})
``

---

## Optimization Techniques

### 1. Model Quantization

``python
import torch
import torch.quantization

def quantize_model(model, example_input):
 """
 Quantize PyTorch model to INT8
 
 Reduces model size by ~4x, speeds up inference
 
 Args:
 model: PyTorch model
 example_input: Sample input for calibration
 
 Returns:
 Quantized model
 """
 # Set model to eval mode
 model.eval()
 
 # Specify quantization configuration
 model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
 
 # Prepare for quantization
 model_prepared = torch.quantization.prepare(model)
 
 # Calibrate with example data
 with torch.no_grad():
 model_prepared(example_input)
 
 # Convert to quantized model
 model_quantized = torch.quantization.convert(model_prepared)
 
 return model_quantized

# Example
model = torch.nn.Sequential(
 torch.nn.Linear(10, 50),
 torch.nn.ReLU(),
 torch.nn.Linear(50, 2)
)

example_input = torch.randn(1, 10)
quantized_model = quantize_model(model, example_input)

# Quantized model is ~4x smaller and faster
print(f"Original size: {get_model_size(model):.2f} MB")
print(f"Quantized size: {get_model_size(quantized_model):.2f} MB")
``

### 2. Batch Inference

``python
import asyncio
from collections import deque
import time

class BatchingPredictor:
 """
 Batch multiple requests for efficient inference
 
 Collects requests and processes them in batches
 """
 
 def __init__(self, model, max_batch_size=32, max_wait_ms=10):
 self.model = model
 self.max_batch_size = max_batch_size
 self.max_wait_ms = max_wait_ms
 self.queue = deque()
 self.processing = False
 
 async def predict(self, features):
 """
 Add request to batch queue
 
 Returns: Future that resolves with prediction
 """
 future = asyncio.Future()
 self.queue.append((features, future))
 
 # Start batch processing if not already running
 if not self.processing:
 asyncio.create_task(self._process_batch())
 
 return await future
 
 async def _process_batch(self):
 """Process accumulated requests as batch"""
 self.processing = True
 
 # Wait for batch to fill or timeout
 await asyncio.sleep(self.max_wait_ms / 1000.0)
 
 if not self.queue:
 self.processing = False
 return
 
 # Collect batch
 batch = []
 futures = []
 
 while self.queue and len(batch) < self.max_batch_size:
 features, future = self.queue.popleft()
 batch.append(features)
 futures.append(future)
 
 # Run batch inference
 batch_array = np.array(batch)
 predictions = self.model.predict(batch_array)
 
 # Resolve futures
 for future, pred in zip(futures, predictions):
 future.set_result(pred)
 
 self.processing = False
 
 # Process remaining queue
 if self.queue:
 asyncio.create_task(self._process_batch())

# Usage
predictor = BatchingPredictor(model, max_batch_size=32, max_wait_ms=10)

async def handle_request(features):
 prediction = await predictor.predict(features)
 return prediction
``

---

## Monitoring & Observability

### Prediction Logging

``python
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class PredictionLog:
 """Log entry for each prediction"""
 timestamp: str
 model_version: str
 features: list
 prediction: float
 confidence: float
 latency_ms: float
 user_id: str = None
 
class MonitoredModelServer:
 """
 Model server with comprehensive monitoring
 """
 
 def __init__(self, model, model_version):
 self.model = model
 self.model_version = model_version
 
 # Setup logging
 self.logger = logging.getLogger('model_server')
 self.logger.setLevel(logging.INFO)
 
 # Metrics
 self.prediction_count = 0
 self.latencies = []
 self.error_count = 0
 
 def predict(self, features, user_id=None):
 """
 Make prediction with logging
 
 Returns: (prediction, confidence, metadata)
 """
 start_time = time.time()
 
 try:
 # Make prediction
 prediction = self.model.predict([features])[0]
 
 # Get confidence
 if hasattr(self.model, 'predict_proba'):
 proba = self.model.predict_proba([features])[0]
 confidence = float(np.max(proba))
 else:
 confidence = 1.0
 
 # Calculate latency
 latency_ms = (time.time() - start_time) * 1000
 
 # Log prediction
 log_entry = PredictionLog(
 timestamp=datetime.now().isoformat(),
 model_version=self.model_version,
 features=features,
 prediction=float(prediction),
 confidence=confidence,
 latency_ms=latency_ms,
 user_id=user_id
 )
 
 self.logger.info(json.dumps(asdict(log_entry)))
 
 # Update metrics
 self.prediction_count += 1
 self.latencies.append(latency_ms)
 
 return prediction, confidence, {'latency_ms': latency_ms}
 
 except Exception as e:
 self.error_count += 1
 self.logger.error(f"Prediction failed: {str(e)}")
 raise
 
 def get_metrics(self):
 """Get serving metrics"""
 if not self.latencies:
 return {}
 
 return {
 'prediction_count': self.prediction_count,
 'error_count': self.error_count,
 'error_rate': self.error_count / max(self.prediction_count, 1),
 'latency_p50': np.percentile(self.latencies, 50),
 'latency_p95': np.percentile(self.latencies, 95),
 'latency_p99': np.percentile(self.latencies, 99),
 }
``

---

## Connection to BST Validation (DSA)

Model serving systems validate predictions similar to BST range checking:

``python
class PredictionBoundsValidator:
 """
 Validate predictions fall within expected ranges
 
 Similar to BST validation with min/max bounds
 """
 
 def __init__(self):
 self.bounds = {} # feature → (min, max)
 
 def set_bounds(self, feature_name, min_val, max_val):
 """Set validation bounds"""
 self.bounds[feature_name] = (min_val, max_val)
 
 def validate_input(self, features):
 """
 Validate input features
 
 Like BST range checking: each value must be in [min, max]
 """
 violations = []
 
 for feature_name, value in features.items():
 if feature_name in self.bounds:
 min_val, max_val = self.bounds[feature_name]
 
 # Range check (like BST validation)
 if value < min_val or value > max_val:
 violations.append({
 'feature': feature_name,
 'value': value,
 'bounds': (min_val, max_val)
 })
 
 return len(violations) == 0, violations
``

---

## Advanced Serving Patterns

### 1. Shadow Mode Deployment

``python
class ShadowModeServer:
 """
 Run new model in shadow mode
 
 New model receives traffic but doesn't affect users
 Predictions are logged for comparison
 """
 
 def __init__(self, production_model, shadow_model):
 self.production_model = production_model
 self.shadow_model = shadow_model
 self.comparison_logs = []
 
 def predict(self, features):
 """
 Make predictions with both models
 
 Returns: Production prediction (shadow runs async)
 """
 import asyncio
 
 # Production prediction (synchronous)
 prod_prediction = self.production_model.predict(features)
 
 # Shadow prediction (async, doesn't block)
 asyncio.create_task(self._shadow_predict(features, prod_prediction))
 
 return prod_prediction
 
 async def _shadow_predict(self, features, prod_prediction):
 """Run shadow model and log comparison"""
 try:
 shadow_prediction = self.shadow_model.predict(features)
 
 # Log comparison
 self.comparison_logs.append({
 'features': features,
 'production': prod_prediction,
 'shadow': shadow_prediction,
 'difference': abs(prod_prediction - shadow_prediction)
 })
 except Exception as e:
 print(f"Shadow prediction failed: {e}")
 
 def get_shadow_metrics(self):
 """Analyze shadow model performance"""
 if not self.comparison_logs:
 return {}
 
 differences = [log['difference'] for log in self.comparison_logs]
 
 return {
 'num_predictions': len(self.comparison_logs),
 'mean_difference': np.mean(differences),
 'max_difference': np.max(differences),
 'agreement_rate': sum(1 for d in differences if d < 0.01) / len(differences)
 }

# Usage
shadow_server = ShadowModeServer(
 production_model=model_v1,
 shadow_model=model_v2
)

# Normal serving
prediction = shadow_server.predict(features)

# Analyze shadow performance
metrics = shadow_server.get_shadow_metrics()
print(f"Shadow agreement rate: {metrics['agreement_rate']:.2%}")
``

### 2. Canary Deployment

``python
class CanaryDeployment:
 """
 Gradual rollout with automated rollback
 
 Monitors metrics and automatically rolls back if issues detected
 """
 
 def __init__(self, stable_model, canary_model):
 self.stable_model = stable_model
 self.canary_model = canary_model
 self.canary_percentage = 0.0
 self.metrics = {
 'stable': {'errors': 0, 'predictions': 0, 'latencies': []},
 'canary': {'errors': 0, 'predictions': 0, 'latencies': []}
 }
 
 def set_canary_percentage(self, percentage):
 """Set canary traffic percentage"""
 assert 0 <= percentage <= 100
 self.canary_percentage = percentage
 print(f"Canary traffic: {percentage}%")
 
 def predict(self, features, user_id=None):
 """
 Predict with canary logic
 
 Routes percentage of traffic to canary
 """
 import random
 import time
 
 # Determine which model to use
 use_canary = random.random() < (self.canary_percentage / 100)
 model_name = 'canary' if use_canary else 'stable'
 model = self.canary_model if use_canary else self.stable_model
 
 # Make prediction with metrics
 start_time = time.time()
 try:
 prediction = model.predict(features)
 latency = time.time() - start_time
 
 # Record metrics
 self.metrics[model_name]['predictions'] += 1
 self.metrics[model_name]['latencies'].append(latency)
 
 return prediction, model_name
 
 except Exception as e:
 # Record error
 self.metrics[model_name]['errors'] += 1
 raise
 
 def check_health(self):
 """
 Check canary health
 
 Returns: (is_healthy, should_rollback, reason)
 """
 canary_metrics = self.metrics['canary']
 stable_metrics = self.metrics['stable']
 
 if canary_metrics['predictions'] < 100:
 # Not enough data yet
 return True, False, "Insufficient data"
 
 # Calculate error rates
 canary_error_rate = canary_metrics['errors'] / canary_metrics['predictions']
 stable_error_rate = stable_metrics['errors'] / max(stable_metrics['predictions'], 1)
 
 # Check if error rate is significantly higher
 if canary_error_rate > stable_error_rate * 2:
 return False, True, f"Error rate too high: {canary_error_rate:.2%}"
 
 # Check latency
 canary_p95 = np.percentile(canary_metrics['latencies'], 95)
 stable_p95 = np.percentile(stable_metrics['latencies'], 95)
 
 if canary_p95 > stable_p95 * 1.5:
 return False, True, f"Latency too high: {canary_p95:.1f}ms"
 
 return True, False, "Healthy"
 
 def auto_rollout(self, target_percentage=100, step=10, check_interval=60):
 """
 Automatically increase canary traffic
 
 Rolls back if health checks fail
 """
 current = 0
 
 while current < target_percentage:
 # Increase canary traffic
 current = min(current + step, target_percentage)
 self.set_canary_percentage(current)
 
 # Wait and check health
 time.sleep(check_interval)
 
 is_healthy, should_rollback, reason = self.check_health()
 
 if should_rollback:
 print(f"❌ Rollback triggered: {reason}")
 self.set_canary_percentage(0) # Rollback to stable
 return False
 
 print(f"✓ Health check passed at {current}%")
 
 print(f"🎉 Canary rollout complete!")
 return True

# Usage
canary = CanaryDeployment(stable_model=model_v1, canary_model=model_v2)

# Start with 5% traffic
canary.set_canary_percentage(5)

# Automatic gradual rollout
success = canary.auto_rollout(target_percentage=100, step=10, check_interval=300)
``

### 3. Multi-Armed Bandit Serving

``python
class BanditModelServer:
 """
 Multi-armed bandit for model selection
 
 Dynamically allocates traffic based on performance
 """
 
 def __init__(self, models: dict):
 """
 Args:
 models: Dict of {model_name: model}
 """
 self.models = models
 self.rewards = {name: [] for name in models.keys()}
 self.counts = {name: 0 for name in models.keys()}
 self.epsilon = 0.1 # Exploration rate
 
 def select_model(self):
 """
 Select model using epsilon-greedy strategy
 
 Returns: model_name
 """
 import random
 
 # Explore: random selection
 if random.random() < self.epsilon:
 return random.choice(list(self.models.keys()))
 
 # Exploit: select best performing model
 avg_rewards = {
 name: np.mean(rewards) if rewards else 0
 for name, rewards in self.rewards.items()
 }
 
 return max(avg_rewards, key=avg_rewards.get)
 
 def predict(self, features, true_label=None):
 """
 Make prediction and optionally update rewards
 
 Args:
 features: Input features
 true_label: Optional ground truth for reward
 
 Returns: (prediction, model_used)
 """
 # Select model
 model_name = self.select_model()
 model = self.models[model_name]
 
 # Make prediction
 prediction = model.predict(features)
 self.counts[model_name] += 1
 
 # Update reward if ground truth available
 if true_label is not None:
 reward = 1.0 if prediction == true_label else 0.0
 self.rewards[model_name].append(reward)
 
 return prediction, model_name
 
 def get_model_stats(self):
 """Get statistics for each model"""
 stats = {}
 
 for name in self.models.keys():
 if self.rewards[name]:
 stats[name] = {
 'count': self.counts[name],
 'avg_reward': np.mean(self.rewards[name]),
 'selection_rate': self.counts[name] / sum(self.counts.values())
 }
 else:
 stats[name] = {
 'count': self.counts[name],
 'avg_reward': 0,
 'selection_rate': 0
 }
 
 return stats

# Usage
bandit = BanditModelServer({
 'model_a': model_a,
 'model_b': model_b,
 'model_c': model_c
})

# Serve with automatic optimization
for features, label in data_stream:
 prediction, model_used = bandit.predict(features, true_label=label)
 
# Check which model performs best
stats = bandit.get_model_stats()
for name, stat in stats.items():
 print(f"{name}: {stat['avg_reward']:.2%} accuracy, {stat['selection_rate']:.1%} traffic")
``

---

## Infrastructure & Deployment

### Containerized Serving with Docker

``dockerfile
# Dockerfile for model serving
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model.pkl .
COPY serve.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
 CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
``

``yaml
# docker-compose.yml
version: '3.8'

services:
 model-server:
 build: .
 ports:
 - "8000:8000"
 environment:
 - MODEL_PATH=/app/model.pkl
 - LOG_LEVEL=INFO
 volumes:
 - ./models:/app/models
 deploy:
 replicas: 3
 resources:
 limits:
 cpus: '2'
 memory: 4G
 healthcheck:
 test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
 interval: 30s
 timeout: 10s
 retries: 3

 load-balancer:
 image: nginx:alpine
 ports:
 - "80:80"
 volumes:
 - ./nginx.conf:/etc/nginx/nginx.conf
 depends_on:
 - model-server
``

### Kubernetes Deployment

``yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
 name: model-serving
spec:
 replicas: 5
 selector:
 matchLabels:
 app: model-serving
 template:
 metadata:
 labels:
 app: model-serving
 version: v1
 spec:
 containers:
 - name: model-server
 image: your-registry/model-serving:v1
 ports:
 - containerPort: 8000
 env:
 - name: MODEL_VERSION
 value: "v1.0"
 resources:
 requests:
 memory: "2Gi"
 cpu: "1000m"
 limits:
 memory: "4Gi"
 cpu: "2000m"
 livenessProbe:
 httpGet:
 path: /health
 port: 8000
 initialDelaySeconds: 30
 periodSeconds: 10
 readinessProbe:
 httpGet:
 path: /ready
 port: 8000
 initialDelaySeconds: 5
 periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
 name: model-serving-service
spec:
 selector:
 app: model-serving
 ports:
 - protocol: TCP
 port: 80
 targetPort: 8000
 type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
 name: model-serving-hpa
spec:
 scaleTargetRef:
 apiVersion: apps/v1
 kind: Deployment
 name: model-serving
 minReplicas: 3
 maxReplicas: 20
 metrics:
 - type: Resource
 resource:
 name: cpu
 target:
 type: Utilization
 averageUtilization: 70
 - type: Resource
 resource:
 name: memory
 target:
 type: Utilization
 averageUtilization: 80
``

---

## Feature Store Integration

``python
class ModelServerWithFeatureStore:
 """
 Model server integrated with feature store
 
 Fetches features on-demand for prediction
 """
 
 def __init__(self, model, feature_store):
 self.model = model
 self.feature_store = feature_store
 
 def predict_from_entity_id(self, entity_id: str):
 """
 Make prediction given entity ID
 
 Fetches features from feature store
 
 Args:
 entity_id: ID to fetch features for
 
 Returns: Prediction
 """
 # Fetch features from feature store
 features = self.feature_store.get_online_features(
 entity_id=entity_id,
 feature_names=[
 'user_age',
 'user_income',
 'user_num_purchases_30d',
 'user_avg_purchase_amount'
 ]
 )
 
 # Convert to array
 feature_array = [
 features['user_age'],
 features['user_income'],
 features['user_num_purchases_30d'],
 features['user_avg_purchase_amount']
 ]
 
 # Make prediction
 prediction = self.model.predict([feature_array])[0]
 
 return {
 'entity_id': entity_id,
 'prediction': float(prediction),
 'features_used': features
 }

# Usage with caching
from functools import lru_cache

class CachedFeatureStore:
 """Feature store with caching"""
 
 def __init__(self, backend):
 self.backend = backend
 
 @lru_cache(maxsize=10000)
 def get_online_features(self, entity_id, feature_names):
 """Cached feature retrieval"""
 return self.backend.get_features(entity_id, feature_names)
``

---

## Cost Optimization

### 1. Request Batching for Cost Reduction

``python
class CostOptimizedServer:
 """
 Optimize costs by batching and caching
 
 Reduces number of model invocations
 """
 
 def __init__(self, model, batch_wait_ms=50, batch_size=32):
 self.model = model
 self.batch_wait_ms = batch_wait_ms
 self.batch_size = batch_size
 self.pending_requests = []
 self.cache = {}
 self.stats = {
 'cache_hits': 0,
 'cache_misses': 0,
 'batches_processed': 0,
 'cost_saved': 0
 }
 
 async def predict_with_caching(self, features, cache_key=None):
 """
 Predict with caching
 
 Args:
 features: Input features
 cache_key: Optional cache key
 
 Returns: Prediction
 """
 # Check cache
 if cache_key and cache_key in self.cache:
 self.stats['cache_hits'] += 1
 return self.cache[cache_key]
 
 self.stats['cache_misses'] += 1
 
 # Add to batch
 future = asyncio.Future()
 self.pending_requests.append((features, future, cache_key))
 
 # Trigger batch processing if needed
 if len(self.pending_requests) >= self.batch_size:
 await self._process_batch()
 
 return await future
 
 async def _process_batch(self):
 """Process accumulated requests as batch"""
 if not self.pending_requests:
 return
 
 # Extract batch
 batch_features = [req[0] for req in self.pending_requests]
 futures = [req[1] for req in self.pending_requests]
 cache_keys = [req[2] for req in self.pending_requests]
 
 # Run batch inference
 predictions = self.model.predict(batch_features)
 
 self.stats['batches_processed'] += 1
 
 # Distribute results
 for pred, future, cache_key in zip(predictions, futures, cache_keys):
 # Cache result
 if cache_key:
 self.cache[cache_key] = pred
 
 # Resolve future
 future.set_result(pred)
 
 # Clear requests
 self.pending_requests = []
 
 # Calculate cost savings (batching is cheaper)
 cost_per_single_request = 0.001 # $0.001 per request
 cost_per_batch = 0.010 # $0.01 per batch
 savings = (len(predictions) * cost_per_single_request) - cost_per_batch
 self.stats['cost_saved'] += savings
 
 def get_cost_stats(self):
 """Get cost optimization statistics"""
 total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
 
 return {
 'total_requests': total_requests,
 'cache_hit_rate': self.stats['cache_hits'] / max(total_requests, 1),
 'batches_processed': self.stats['batches_processed'],
 'avg_batch_size': total_requests / max(self.stats['batches_processed'], 1),
 'estimated_cost_saved': self.stats['cost_saved']
 }
``

### 2. Model Compression for Cheaper Hosting

``python
import torch

def compress_model_for_deployment(model, sample_input):
 """
 Compress model for cheaper hosting
 
 Techniques:
 - Quantization (INT8)
 - Pruning
 - Knowledge distillation
 
 Returns: Compressed model
 """
 # 1. Quantization
 model.eval()
 model_quantized = torch.quantization.quantize_dynamic(
 model,
 {torch.nn.Linear},
 dtype=torch.qint8
 )
 
 # 2. Pruning (remove small weights)
 import torch.nn.utils.prune as prune
 
 for name, module in model_quantized.named_modules():
 if isinstance(module, torch.nn.Linear):
 prune.l1_unstructured(module, name='weight', amount=0.3)
 
 # 3. Verify accuracy
 with torch.no_grad():
 original_output = model(sample_input)
 compressed_output = model_quantized(sample_input)
 
 diff = torch.abs(original_output - compressed_output).mean()
 print(f"Compression error: {diff:.4f}")
 
 return model_quantized

# Compare costs
original_size_mb = get_model_size(model)
compressed_size_mb = get_model_size(compressed_model)

print(f"Size reduction: {original_size_mb:.1f}MB → {compressed_size_mb:.1f}MB")
print(f"Cost savings: ~${(original_size_mb - compressed_size_mb) * 0.10:.2f}/month")
``

---

## Troubleshooting & Debugging

### Prediction Debugging

``python
class DebuggableModelServer:
 """
 Model server with debugging capabilities
 
 Helps diagnose prediction issues
 """
 
 def __init__(self, model):
 self.model = model
 
 def predict_with_debug(self, features, debug=False):
 """
 Make prediction with optional debug info
 
 Returns: (prediction, debug_info)
 """
 debug_info = {}
 
 if debug:
 # Record input stats
 debug_info['input_stats'] = {
 'mean': np.mean(features),
 'std': np.std(features),
 'min': np.min(features),
 'max': np.max(features),
 'nan_count': np.isnan(features).sum()
 }
 
 # Check for anomalies
 debug_info['anomalies'] = self._detect_anomalies(features)
 
 # Make prediction
 prediction = self.model.predict([features])[0]
 
 if debug:
 # Record prediction confidence
 if hasattr(self.model, 'predict_proba'):
 proba = self.model.predict_proba([features])[0]
 debug_info['confidence'] = float(np.max(proba))
 debug_info['class_probabilities'] = proba.tolist()
 
 return prediction, debug_info
 
 def _detect_anomalies(self, features):
 """Detect input anomalies"""
 anomalies = []
 
 # Check for NaN
 if np.any(np.isnan(features)):
 anomalies.append("Contains NaN values")
 
 # Check for extreme values
 z_scores = np.abs((features - np.mean(features)) / (np.std(features) + 1e-8))
 if np.any(z_scores > 3):
 anomalies.append("Contains outliers (z-score > 3)")
 
 return anomalies
 
 def explain_prediction(self, features):
 """
 Explain prediction using SHAP or similar
 
 Returns: Feature importance
 """
 # Simplified explanation (in practice, use SHAP)
 if hasattr(self.model, 'feature_importances_'):
 importances = self.model.feature_importances_
 
 return {
 f'feature_{i}': {'value': features[i], 'importance': imp}
 for i, imp in enumerate(importances)
 }
 
 return {}
``

---

## Key Takeaways

✅ **Multiple serving patterns** - REST, gRPC, batch for different needs 
✅ **Model versioning essential** - Support A/B testing and rollbacks 
✅ **Optimize for latency** - Quantization, batching, caching 
✅ **Monitor everything** - Latency, errors, prediction distribution 
✅ **Validate inputs/outputs** - Catch issues early 
✅ **Scale horizontally** - Add more serving instances 
✅ **Connection to validation** - Like BST range checking 

---

**Originally published at:** [arunbaby.com/ml-system-design/0008-model-serving-architecture](https://www.arunbaby.com/ml-system-design/0008-model-serving-architecture/)

*If you found this helpful, consider sharing it with others who might benefit.*

