---
title: "Batch vs Real-Time Inference"
day: 5
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - inference
  - model-serving
  - architecture
  - real-time
  - batch-processing
subdomain: Model Serving
tech_stack: [Python, TensorFlow Serving, PyTorch, Ray, Celery, Kafka]
scale: "1M+ predictions/day"
companies: [Google, Meta, Netflix, Uber, Airbnb]
related_dsa_day: 5
related_speech_day: 5
---

**How to choose between batch and real-time inference, the architectural decision that shapes your entire ML serving infrastructure.**

## Introduction

After training a model, you need to **serve predictions**. Two fundamental approaches:

1. **Batch Inference:** Precompute predictions for all users/items periodically
2. **Real-Time Inference:** Compute predictions on-demand when requested

**Why this matters:**
- **Different latency requirements** → Different architectures
- **Cost implications** → Batch can be 10-100x cheaper
- **System complexity** → Real-time requires more infrastructure
- **Feature freshness** → Real-time uses latest data

**What you'll learn:**
- When to use batch vs real-time
- Architecture for each approach
- Hybrid systems combining both
- Trade-offs and decision framework
- Production implementation patterns

---

## Problem Definition

Design an ML inference system that serves predictions efficiently.

### Functional Requirements

1. **Prediction Serving**
   - Batch: Generate predictions for all entities periodically
   - Real-time: Serve predictions on-demand with low latency
   - Hybrid: Combine both approaches

2. **Data Freshness**
   - Access to latest features
   - Handle feature staleness
   - Feature computation strategy

3. **Scalability**
   - Handle millions of predictions
   - Scale horizontally
   - Handle traffic spikes

### Non-Functional Requirements

1. **Latency**
   - Batch: Minutes to hours acceptable
   - Real-time: < 100ms for most applications

2. **Throughput**
   - Batch: Process millions of predictions in one run
   - Real-time: 1000s of requests/second

3. **Cost**
   - Optimize compute resources
   - Minimize infrastructure costs

4. **Reliability**
   - 99.9%+ uptime for real-time
   - Graceful degradation
   - Fallback mechanisms

---

## Batch Inference

Precompute predictions periodically (daily, hourly, etc.).

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Batch Inference Pipeline                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐      ┌──────────────┐                │
│  │  Data Lake   │      │  Feature     │                │
│  │  (HDFS/S3)   │─────▶│  Engineering │                │
│  └──────────────┘      └──────────────┘                │
│                               │                          │
│                               ▼                          │
│                        ┌──────────────┐                 │
│                        │  Batch Job   │                 │
│                        │  (Spark/Ray) │                 │
│                        │  - Load model│                 │
│                        │  - Predict   │                 │
│                        └──────────────┘                 │
│                               │                          │
│                               ▼                          │
│                        ┌──────────────┐                 │
│                        │  Write to    │                 │
│                        │  Cache/DB    │                 │
│                        │  (Redis/DDB) │                 │
│                        └──────────────┘                 │
│                               │                          │
│  ┌──────────────┐            │                          │
│  │  Application │◀───────────┘                          │
│  │  Server      │  Lookup predictions                   │
│  └──────────────┘                                        │
│                                                          │
└─────────────────────────────────────────────────────────┘

Flow:
1. Extract features from data warehouse
2. Run batch prediction job
3. Store predictions in fast lookup store
4. Application does simple lookup
```

### Implementation

```python
from typing import List, Dict
import numpy as np
import redis
import json
import time

class BatchInferenceSystem:
    """
    Batch inference system
    
    Precomputes predictions for all users/items
    """
    
    def __init__(self, model, redis_client):
        self.model = model
        self.redis = redis_client
        self.batch_size = 1000
    
    def run_batch_prediction(self, entity_ids: List[str], features_df):
        """
        Run batch prediction for all entities
        
        Args:
            entity_ids: List of user/item IDs
            features_df: DataFrame with features for all entities
        
        Returns:
            Number of predictions generated
        """
        num_predictions = 0
        
        # Process in batches for memory efficiency
        for i in range(0, len(entity_ids), self.batch_size):
            batch_ids = entity_ids[i:i+self.batch_size]
            batch_features = features_df.iloc[i:i+self.batch_size]
            
            # Predict
            predictions = self.model.predict(batch_features.values)
            
            # Store in Redis
            self._store_predictions(batch_ids, predictions)
            
            num_predictions += len(batch_ids)
            
            if num_predictions % 10000 == 0:
                print(f"Processed {num_predictions} predictions...")
        
        return num_predictions
    
    def _store_predictions(self, entity_ids: List[str], predictions: np.ndarray):
        """Store predictions in Redis with TTL"""
        pipeline = self.redis.pipeline()
        
        ttl_seconds = 24 * 3600  # 24 hours
        
        for entity_id, prediction in zip(entity_ids, predictions):
            # Store as JSON
            key = f"pred:{entity_id}"
            value = json.dumps({
                'prediction': float(prediction),
                'timestamp': time.time()
            })
            
            pipeline.setex(key, ttl_seconds, value)
        
        pipeline.execute()
    
    def get_prediction(self, entity_id: str) -> float:
        """
        Lookup precomputed prediction
        
        Fast O(1) lookup
        """
        key = f"pred:{entity_id}"
        value = self.redis.get(key)
        
        if value is None:
            # Prediction not found or expired
            return None
        
        data = json.loads(value)
        return data['prediction']

# Usage
import pandas as pd
import time

# Initialize
redis_client = redis.Redis(host='localhost', port=6379, db=0)
model = load_trained_model()  # Your trained model

batch_system = BatchInferenceSystem(model, redis_client)

# Run batch prediction (e.g., daily cron job)
user_ids = fetch_all_user_ids()  # Get all users
features_df = fetch_user_features(user_ids)  # Get features

num_preds = batch_system.run_batch_prediction(user_ids, features_df)
print(f"Generated {num_preds} predictions")

# Later, application looks up prediction
prediction = batch_system.get_prediction("user_12345")
print(f"Prediction: {prediction}")
```

### Spark-based Batch Inference

For large-scale batch processing:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

class SparkBatchInference:
    """
    Distributed batch inference using PySpark
    
    Scales to billions of predictions
    """
    
    def __init__(self, model_path):
        self.spark = SparkSession.builder \
            .appName("BatchInference") \
            .getOrCreate()
        
        self.model_path = model_path
    
    def predict_spark(self, features_df_spark):
        """
        Distribute prediction across cluster
        
        Args:
            features_df_spark: Spark DataFrame with features
        
        Returns:
            Spark DataFrame with predictions
        """
        model_path = self.model_path
        
        # Define pandas UDF for prediction
        @pandas_udf("double", PandasUDFType.SCALAR)
        def predict_udf(*features):
            # Load model once per executor
            import joblib
            model = joblib.load(model_path)
            
            # Create feature matrix
            X = pd.DataFrame({
                f'feature_{i}': features[i]
                for i in range(len(features))
            })
            
            # Predict
            predictions = model.predict(X.values)
            return pd.Series(predictions)
        
        # Apply UDF
        feature_cols = [col for col in features_df_spark.columns if col.startswith('feature_')]
        
        result_df = features_df_spark.withColumn(
            'prediction',
            predict_udf(*feature_cols)
        )
        
        return result_df
    
    def run_batch_job(self, input_path, output_path):
        """
        Full batch inference pipeline
        
        Args:
            input_path: S3/HDFS path to input data
            output_path: S3/HDFS path to save predictions
        """
        # Read input
        df = self.spark.read.parquet(input_path)
        
        # Predict
        predictions_df = self.predict_spark(df)
        
        # Write output
        predictions_df.write.parquet(output_path, mode='overwrite')
        
        print(f"Batch prediction complete. Output: {output_path}")

# Usage
spark_batch = SparkBatchInference(model_path='s3://models/my_model.pkl')

spark_batch.run_batch_job(
    input_path='s3://data/user_features/',
    output_path='s3://predictions/daily/2025-01-15/'
)
```

---

## Real-Time Inference

Compute predictions on-demand when requested.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│            Real-Time Inference System                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐                                       │
│  │  Load        │◀─── Model Registry                    │
│  │  Balancer    │                                        │
│  └──────┬───────┘                                        │
│         │                                                │
│    ┌────▼─────────────────────────────┐                 │
│    │   Model Serving Instances        │                 │
│    │   ┌─────────┐  ┌─────────┐      │                 │
│    │   │ Model 1 │  │ Model 2 │ ...  │                 │
│    │   │ (GPU)   │  │ (GPU)   │      │                 │
│    │   └─────────┘  └─────────┘      │                 │
│    └────┬─────────────────────────────┘                 │
│         │                                                │
│    ┌────▼──────────┐     ┌──────────────┐              │
│    │ Feature       │────▶│  Feature     │              │
│    │ Service       │     │  Store       │              │
│    │ - Online      │     │  (Redis)     │              │
│    │   features    │     └──────────────┘              │
│    └───────────────┘                                    │
│                                                          │
└─────────────────────────────────────────────────────────┘

Flow:
1. Request arrives with user/item ID
2. Fetch features from feature store
3. Compute additional online features
4. Model predicts
5. Return prediction
```

### Implementation

```python
from fastapi import FastAPI
import numpy as np
from typing import Dict
import torch

app = FastAPI()

class RealTimeInferenceService:
    """
    Real-time inference service
    
    Serves predictions with low latency
    """
    
    def __init__(self, model, feature_store):
        self.model = model
        self.feature_store = feature_store
        
        # Warm up model
        self._warmup()
    
    def _warmup(self):
        """Warm up model with dummy prediction"""
        dummy_features = np.random.randn(1, self.model.input_dim)
        _ = self.model.predict(dummy_features)
    
    def get_features(self, entity_id: str) -> Dict:
        """
        Fetch features for entity
        
        Combines precomputed + real-time features
        """
        # Fetch precomputed features from Redis
        precomputed_raw = self.feature_store.get(f"features:{entity_id}")
        precomputed = {}
        if precomputed_raw:
            try:
                precomputed = json.loads(precomputed_raw)
            except Exception:
                precomputed = {}
        
        if precomputed is None:
            # Fallback: compute features on-the-fly
            precomputed = self._compute_features_fallback(entity_id)
        
        # Add real-time features
        realtime_features = self._compute_realtime_features(entity_id)
        
        # Combine
        features = {**precomputed, **realtime_features}
        
        return features
    
    def _compute_realtime_features(self, entity_id: str) -> Dict:
        """
        Compute features that must be fresh
        
        E.g., time of day, user's current session, etc.
        """
        import datetime
        
        now = datetime.datetime.now()
        
        return {
            'hour_of_day': now.hour,
            'day_of_week': now.weekday(),
            'is_weekend': 1 if now.weekday() >= 5 else 0
        }
    
    def _compute_features_fallback(self, entity_id: str) -> Dict:
        """Fallback feature computation"""
        # Query database, compute on-the-fly
        # This is slower but ensures we can always serve
        return {}
    
    def predict(self, entity_id: str) -> float:
        """
        Real-time prediction
        
        Returns:
            Prediction score
        """
        # Get features
        features = self.get_features(entity_id)
        
        # Convert to numpy array (assuming fixed feature order)
        feature_vector = np.array([
            features.get(f'feature_{i}', 0.0)
            for i in range(self.model.input_dim)
        ]).reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(feature_vector)[0]
        
        return float(prediction)

# FastAPI endpoints
realtime_service = RealTimeInferenceService(model, redis_client)

@app.get("/predict/{entity_id}")
async def predict_endpoint(entity_id: str):
    """
    Real-time prediction endpoint
    
    GET /predict/user_12345
    """
    try:
        prediction = realtime_service.predict(entity_id)
        
        return {
            'entity_id': entity_id,
            'prediction': prediction,
            'timestamp': time.time()
        }
    
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail={'error': str(e), 'entity_id': entity_id})

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

### TensorFlow Serving

Production-grade model serving:

```python
import requests
import json

class TensorFlowServingClient:
    """
    Client for TensorFlow Serving
    
    High-performance model serving
    """
    
    def __init__(self, server_url, model_name, model_version=None):
        self.server_url = server_url
        self.model_name = model_name
        self.model_version = model_version or 'latest'
        
        # Endpoint
        if self.model_version == 'latest':
            self.endpoint = f"{server_url}/v1/models/{model_name}:predict"
        else:
            self.endpoint = f"{server_url}/v1/models/{model_name}/versions/{model_version}:predict"
    
    def predict(self, instances: List[List[float]]) -> List[float]:
        """
        Send prediction request to TF Serving
        
        Args:
            instances: List of feature vectors
        
        Returns:
            List of predictions
        """
        # Prepare request
        payload = {
            "signature_name": "serving_default",
            "instances": instances
        }
        
        # Send request
        response = requests.post(
            self.endpoint,
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            raise Exception(f"Prediction failed: {response.text}")
        
        # Parse response
        result = response.json()
        predictions = result['predictions']
        
        return predictions

# Usage
tf_client = TensorFlowServingClient(
    server_url='http://localhost:8501',
    model_name='recommendation_model',
    model_version='3'
)

# Predict
features = [[0.1, 0.5, 0.3, 0.9]]
predictions = tf_client.predict(features)
print(f"Prediction: {predictions[0]}")
```

---

## Hybrid Approach

Combine batch and real-time for optimal performance.

### Architecture

```
┌────────────────────────────────────────────────────────┐
│              Hybrid Inference System                    │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────────────┐         ┌────────────────┐        │
│  │ Batch Pipeline │         │ Real-Time API  │        │
│  │ (Daily)        │         │                │        │
│  └───────┬────────┘         └───────┬────────┘        │
│          │                          │                  │
│          ▼                          ▼                  │
│  ┌──────────────────────────────────────────┐         │
│  │        Prediction Cache (Redis)          │         │
│  │  ┌────────────┐      ┌────────────┐     │         │
│  │  │ Batch      │      │ Real-time  │     │         │
│  │  │ Predictions│      │ Predictions│     │         │
│  │  │ (TTL: 24h) │      │ (TTL: 1h)  │     │         │
│  │  └────────────┘      └────────────┘     │         │
│  └──────────────────────────────────────────┘         │
│                     ▲                                  │
│                     │                                  │
│              ┌──────┴────────┐                         │
│              │  Application  │                         │
│              │  1. Check cache│                        │
│              │  2. Fallback to│                        │
│              │     real-time  │                        │
│              └───────────────┘                         │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### Implementation

```python
class HybridInferenceSystem:
    """
    Hybrid system: batch + real-time
    
    - Fast path: Use batch predictions if available
    - Slow path: Compute real-time if needed
    """
    
    def __init__(self, batch_system, realtime_system):
        self.batch = batch_system
        self.realtime = realtime_system
        self.cache_hit_counter = 0
        self.cache_miss_counter = 0
    
    def predict(self, entity_id: str, max_staleness_hours: int = 24) -> Dict:
        """
        Get prediction with automatic fallback
        
        Args:
            entity_id: Entity to predict for
            max_staleness_hours: Maximum age of batch prediction
        
        Returns:
            {
                'prediction': float,
                'source': 'batch' | 'realtime',
                'timestamp': float
            }
        """
        # Try batch prediction first
        batch_pred_value = self.batch.get_prediction(entity_id)
        
        if batch_pred_value is not None:
            # If batch system returns only a float, treat as fresh within TTL of Redis
            self.cache_hit_counter += 1
            return {
                'prediction': batch_pred_value,
                'source': 'batch',
                'timestamp': time.time(),
                'cache_hit': True
            }
        
        # Fallback to real-time
        self.cache_miss_counter += 1
        
        realtime_pred = self.realtime.predict(entity_id)
        
        return {
            'prediction': realtime_pred,
            'source': 'realtime',
            'timestamp': time.time(),
            'cache_hit': False
        }
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hit_counter + self.cache_miss_counter
        if total == 0:
            return 0.0
        return self.cache_hit_counter / total

# Usage
hybrid = HybridInferenceSystem(batch_system, realtime_service)

# Predict for user
result = hybrid.predict('user_12345', max_staleness_hours=12)

print(f"Prediction: {result['prediction']}")
print(f"Source: {result['source']}")
print(f"Cache hit rate: {hybrid.get_cache_hit_rate():.2%}")
```

---

## Decision Framework

When to use which approach:

### Use Batch Inference When:

✅ **Latency is not critical** (recommendations, email campaigns)  
✅ **Predictions needed for all entities** (e.g., all users)  
✅ **Features are expensive to compute**  
✅ **Model is large/slow**  
✅ **Cost optimization is priority**  
✅ **Predictions don't change frequently**

**Examples:**
- Daily email recommendations
- Product catalog rankings
- Weekly personalized content
- Batch fraud scoring

### Use Real-Time Inference When:

✅ **Low latency required** (< 100ms)  
✅ **Fresh features critical** (current context)  
✅ **Predictions for small subset** (active users)  
✅ **Immediate user feedback** (search, ads)  
✅ **High-value decisions** (fraud detection)

**Examples:**
- Search ranking
- Ad serving
- Real-time fraud detection
- Live recommendation widgets

### Use Hybrid When:

✅ **Mix of latency requirements**  
✅ **Want cost + performance**  
✅ **Can tolerate some staleness**  
✅ **Variable traffic patterns**  
✅ **Graceful degradation needed**

**Examples:**
- Homepage recommendations (batch) + search (real-time)
- Social feed (batch) + stories (real-time)
- Product pages (batch) + checkout (real-time)

---

## Cost Comparison

```python
class CostAnalyzer:
    """
    Estimate costs for batch vs real-time
    """
    
    def estimate_batch_cost(
        self,
        num_entities: int,
        predictions_per_day: int,
        cost_per_compute_hour: float = 3.0
    ) -> Dict:
        """Estimate daily batch inference cost"""
        
        # Assume 10K predictions/second throughput
        throughput = 10_000
        
        # Total predictions
        total_preds = num_entities * predictions_per_day
        
        # Compute time needed
        compute_seconds = total_preds / throughput
        compute_hours = compute_seconds / 3600
        
        # Cost
        compute_cost = compute_hours * cost_per_compute_hour
        
        # Storage cost (Redis/DDB)
        storage_gb = total_preds * 100 / 1e9  # 100 bytes per prediction
        storage_cost = storage_gb * 0.25  # $0.25/GB/month
        
        total_cost = compute_cost + storage_cost
        
        return {
            'compute_hours': compute_hours,
            'compute_cost': compute_cost,
            'storage_cost': storage_cost,
            'total_daily_cost': total_cost,
            'cost_per_prediction': total_cost / total_preds
        }
    
    def estimate_realtime_cost(
        self,
        requests_per_second: int,
        cost_per_instance_hour: float = 5.0,
        requests_per_instance: int = 100
    ) -> Dict:
        """Estimate real-time serving cost"""
        
        # Number of instances needed
        num_instances = requests_per_second / requests_per_instance
        num_instances = int(np.ceil(num_instances * 1.5))  # 50% headroom
        
        # Daily cost
        daily_hours = 24
        daily_cost = num_instances * cost_per_instance_hour * daily_hours
        
        # Predictions per day
        daily_requests = requests_per_second * 86400
        
        return {
            'num_instances': num_instances,
            'daily_cost': daily_cost,
            'cost_per_prediction': daily_cost / daily_requests
        }

# Compare costs
analyzer = CostAnalyzer()

# Batch: 1M users, predict once/day
batch_cost = analyzer.estimate_batch_cost(
    num_entities=1_000_000,
    predictions_per_day=1
)

print("Batch Inference:")
print(f"  Daily cost: ${batch_cost['total_daily_cost']:.2f}")
print(f"  Cost per prediction: ${batch_cost['cost_per_prediction']:.6f}")

# Real-time: 100 QPS average
realtime_cost = analyzer.estimate_realtime_cost(
    requests_per_second=100
)

print("\nReal-Time Inference:")
print(f"  Daily cost: ${realtime_cost['daily_cost']:.2f}")
print(f"  Cost per prediction: ${realtime_cost['cost_per_prediction']:.6f}")

# Compare
savings = (realtime_cost['daily_cost'] - batch_cost['total_daily_cost']) / realtime_cost['daily_cost'] * 100
print(f"\nBatch is {savings:.1f}% cheaper!")
```

---

## Advanced Patterns

### Multi-Tier Caching

Layer multiple caches for optimal performance.

```python
class MultiTierInferenceSystem:
    """
    Multi-tier caching: Memory → Redis → Compute
    
    Optimizes for different latency/cost profiles
    """
    
    def __init__(self, model, redis_client):
        self.model = model
        self.redis = redis_client
        
        # In-memory cache (fastest)
        self.memory_cache = {}
        self.memory_cache_size = 10000
        
        # Statistics
        self.stats = {
            'memory_hits': 0,
            'redis_hits': 0,
            'compute': 0,
            'total_requests': 0
        }
    
    def predict(self, entity_id: str) -> float:
        """
        Predict with multi-tier caching
        
        Tier 1: In-memory cache (~1ms)
        Tier 2: Redis cache (~5ms)
        Tier 3: Compute prediction (~50ms)
        """
        self.stats['total_requests'] += 1
        
        # Tier 1: Memory cache
        if entity_id in self.memory_cache:
            self.stats['memory_hits'] += 1
            return self.memory_cache[entity_id]
        
        # Tier 2: Redis cache
        redis_key = f"pred:{entity_id}"
        cached = self.redis.get(redis_key)
        
        if cached is not None:
            self.stats['redis_hits'] += 1
            prediction = float(cached)
            
            # Promote to memory cache
            self._add_to_memory_cache(entity_id, prediction)
            
            return prediction
        
        # Tier 3: Compute
        self.stats['compute'] += 1
        prediction = self._compute_prediction(entity_id)
        
        # Write to both caches
        self.redis.setex(redis_key, 3600, str(prediction))  # 1 hour TTL
        self._add_to_memory_cache(entity_id, prediction)
        
        return prediction
    
    def _add_to_memory_cache(self, entity_id: str, prediction: float):
        """Add to memory cache with LRU eviction"""
        if len(self.memory_cache) >= self.memory_cache_size:
            # Simple eviction: remove first item
            # In production, use LRU cache
            self.memory_cache.pop(next(iter(self.memory_cache)))
        
        self.memory_cache[entity_id] = prediction
    
    def _compute_prediction(self, entity_id: str) -> float:
        """Compute prediction from model"""
        # Fetch features
        features = self._get_features(entity_id)
        
        # Predict
        prediction = self.model.predict([features])[0]
        
        return float(prediction)
    
    def _get_features(self, entity_id: str):
        """Fetch features for entity"""
        # Placeholder
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics"""
        total = self.stats['total_requests']
        
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'memory_hit_rate': self.stats['memory_hits'] / total * 100,
            'redis_hit_rate': self.stats['redis_hits'] / total * 100,
            'compute_rate': self.stats['compute'] / total * 100,
            'overall_cache_hit_rate': 
                (self.stats['memory_hits'] + self.stats['redis_hits']) / total * 100
        }

# Usage
system = MultiTierInferenceSystem(model, redis_client)

# Make predictions
for entity_id in ['user_1', 'user_2', 'user_1', 'user_3', 'user_1']:
    prediction = system.predict(entity_id)
    print(f"{entity_id}: {prediction:.4f}")

stats = system.get_cache_stats()
print(f"\nCache hit rate: {stats['overall_cache_hit_rate']:.1f}%")
print(f"Memory: {stats['memory_hit_rate']:.1f}%, Redis: {stats['redis_hit_rate']:.1f}%, Compute: {stats['compute_rate']:.1f}%")
```

### Prediction Warming

Precompute predictions for likely requests.

```python
class PredictionWarmer:
    """
    Warm cache with predictions for likely-to-be-requested entities
    
    Use case: Preload predictions for active users
    """
    
    def __init__(self, model, cache):
        self.model = model
        self.cache = cache
    
    def warm_predictions(
        self,
        entity_ids: List[str],
        batch_size: int = 100
    ):
        """
        Warm cache for list of entities
        
        Args:
            entity_ids: Entities to warm
            batch_size: Batch size for efficient computation
        """
        num_warmed = 0
        
        for i in range(0, len(entity_ids), batch_size):
            batch_ids = entity_ids[i:i+batch_size]
            
            # Batch feature fetching
            features = self._batch_get_features(batch_ids)
            
            # Batch prediction
            predictions = self.model.predict(features)
            
            # Write to cache
            for entity_id, prediction in zip(batch_ids, predictions):
                self.cache.set(f"pred:{entity_id}", float(prediction), ex=3600)
                num_warmed += 1
        
        return num_warmed
    
    def _batch_get_features(self, entity_ids: List[str]):
        """Fetch features for multiple entities"""
        # In production: Batch query to feature store
        return [[0.1] * 5 for _ in entity_ids]
    
    def warm_by_activity(
        self,
        lookback_hours: int = 24,
        top_k: int = 10000
    ):
        """
        Warm cache for most active entities
        
        Args:
            lookback_hours: Look back this many hours for activity
            top_k: Warm top K most active entities
        """
        # Query activity logs
        active_entities = self._get_active_entities(lookback_hours, top_k)
        
        # Warm predictions
        num_warmed = self.warm_predictions(active_entities)
        
        return {
            'num_warmed': num_warmed,
            'lookback_hours': lookback_hours,
            'timestamp': time.time()
        }
    
    def _get_active_entities(self, lookback_hours: int, top_k: int) -> List[str]:
        """Get most active entities from activity logs"""
        # Placeholder: Query activity database
        return [f'user_{i}' for i in range(top_k)]

# Usage: Warm cache every hour for active users
warmer = PredictionWarmer(model, redis_client)

# Warm cache for top 10K active users
result = warmer.warm_by_activity(lookback_hours=1, top_k=10000)
print(f"Warmed {result['num_warmed']} predictions")
```

### Conditional Batch Updates

Update batch predictions conditionally based on staleness/changes.

```python
class ConditionalBatchUpdater:
    """
    Update batch predictions only when necessary
    
    Strategies:
    - Update only if features changed significantly
    - Update only if prediction is stale
    - Update only for active entities
    """
    
    def __init__(self, model, cache, feature_store):
        self.model = model
        self.cache = cache
        self.feature_store = feature_store
    
    def update_if_changed(
        self,
        entity_ids: List[str],
        change_threshold: float = 0.1
    ) -> dict:
        """
        Update predictions only if features changed significantly
        
        Args:
            entity_ids: Entities to check
            change_threshold: Update if features changed by this much
        
        Returns:
            Statistics on updates
        """
        num_checked = 0
        num_updated = 0
        
        for entity_id in entity_ids:
            num_checked += 1
            
            # Get current features
            current_features = self.feature_store.get(f"features:{entity_id}")
            
            # Get cached features (when prediction was made)
            cached_features = self.feature_store.get(f"cached_features:{entity_id}")
            
            # Check if features changed significantly
            if self._features_changed(cached_features, current_features, change_threshold):
                # Recompute prediction
                prediction = self.model.predict([current_features])[0]
                
                # Update cache
                self.cache.set(f"pred:{entity_id}", float(prediction), ex=3600)
                self.feature_store.set(f"cached_features:{entity_id}", current_features)
                
                num_updated += 1
        
        return {
            'num_checked': num_checked,
            'num_updated': num_updated,
            'update_rate': num_updated / num_checked * 100 if num_checked > 0 else 0
        }
    
    def _features_changed(
        self,
        old_features,
        new_features,
        threshold: float
    ) -> bool:
        """Check if features changed significantly"""
        if old_features is None or new_features is None:
            return True
        
        # Compute L2 distance
        diff = np.linalg.norm(np.array(new_features) - np.array(old_features))
        
        return diff > threshold
```

### Graceful Degradation

Handle failures gracefully with fallback strategies.

```python
class GracefulDegradationSystem:
    """
    Inference system with graceful degradation
    
    Fallback chain:
    1. Try real-time prediction
    2. Fallback to batch prediction (if available)
    3. Fallback to default/fallback prediction
    """
    
    def __init__(
        self,
        realtime_service,
        batch_cache,
        default_prediction: float = 0.5
    ):
        self.realtime = realtime_service
        self.batch_cache = batch_cache
        self.default_prediction = default_prediction
        
        # Monitoring
        self.degradation_stats = {
            'realtime': 0,
            'batch_fallback': 0,
            'default_fallback': 0
        }
    
    def predict_with_fallback(
        self,
        entity_id: str,
        max_latency_ms: int = 100
    ) -> dict:
        """
        Predict with fallback strategies
        
        Args:
            entity_id: Entity to predict for
            max_latency_ms: Maximum acceptable latency
        
        Returns:
            {
                'prediction': float,
                'source': str,
                'latency_ms': float
            }
        """
        start = time.perf_counter()
        
        # Try real-time prediction
        try:
            prediction = self.realtime.predict(entity_id)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            if elapsed_ms <= max_latency_ms:
                self.degradation_stats['realtime'] += 1
                return {
                    'prediction': prediction,
                    'source': 'realtime',
                    'latency_ms': elapsed_ms
                }
        except Exception as e:
            print(f"Real-time prediction failed: {e}")
        
        # Fallback 1: Batch cache
        try:
            batch_pred = self.batch_cache.get(f"pred:{entity_id}")
            
            if batch_pred is not None:
                elapsed_ms = (time.perf_counter() - start) * 1000
                self.degradation_stats['batch_fallback'] += 1
                
                return {
                    'prediction': float(batch_pred),
                    'source': 'batch_fallback',
                    'latency_ms': elapsed_ms,
                    'warning': 'Using stale batch prediction'
                }
        except Exception as e:
            print(f"Batch fallback failed: {e}")
        
        # Fallback 2: Default prediction
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.degradation_stats['default_fallback'] += 1
        
        return {
            'prediction': self.default_prediction,
            'source': 'default_fallback',
            'latency_ms': elapsed_ms,
            'warning': 'Using default prediction - service degraded'
        }
    
    def get_health_status(self) -> dict:
        """Get system health metrics"""
        total = sum(self.degradation_stats.values())
        
        if total == 0:
            return {'status': 'no_traffic'}
        
        realtime_rate = self.degradation_stats['realtime'] / total * 100
        
        if realtime_rate > 95:
            status = 'healthy'
        elif realtime_rate > 80:
            status = 'degraded'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'realtime_rate': realtime_rate,
            'batch_fallback_rate': self.degradation_stats['batch_fallback'] / total * 100,
            'default_fallback_rate': self.degradation_stats['default_fallback'] / total * 100,
            'total_requests': total
        }
```

---

## Real-World Case Studies

### Netflix: Hybrid Recommendations

**Challenge:** Personalized recommendations for 200M+ users

**Solution:**
- **Batch:** Precompute top-N recommendations for all users daily
- **Real-time:** Rerank based on current session context
- **Result:** < 100ms latency with personalized results

**Architecture:**
```
Daily Batch Job (Spark)
  ↓
Precompute Top 1000 movies per user
  ↓
Store in Cassandra
  ↓
Real-time API fetches top 1000 + reranks based on:
  - Current time of day
  - Device type
  - Recent viewing history
  ↓
Return Top 20 to UI
```

### Uber: Real-Time ETA Prediction

**Challenge:** Predict arrival time for millions of rides

**Solution:**
- **Real-time only:** ETA must reflect current traffic
- **Strategy:** Fast model (< 50ms inference)
- **Features:** Current location, traffic data, historical patterns

**Why not batch:**
- Traffic changes rapidly
- Each ride is unique
- Requires current GPS coordinates

### LinkedIn: People You May Know

**Challenge:** Suggest connections for 800M+ users

**Solution:**
- **Batch:** Graph algorithms compute connection candidates (weekly)
- **Real-time:** Scoring based on user activity
- **Result:** Balance compute cost with personalization

**Hybrid Strategy:**
```
Weekly Batch:
  - Graph traversal (2nd, 3rd degree connections)
  - Identify ~1000 candidates per user
  - Store in candidate DB

Real-time (on page load):
  - Fetch candidates from DB
  - Score based on:
    * Recent profile views
    * Shared groups/companies
    * Mutual connections
  - Return top 10
```

---

## Monitoring & Observability

### Key Metrics to Track

```python
class InferenceMetrics:
    """
    Track comprehensive inference metrics
    """
    
    def __init__(self):
        self.metrics = {
            'latency_p50': [],
            'latency_p95': [],
            'latency_p99': [],
            'cache_hit_rate': [],
            'error_rate': [],
            'throughput': [],
            'cost_per_prediction': []
        }
    
    def record_prediction(
        self,
        latency_ms: float,
        cache_hit: bool,
        error: bool,
        cost: float
    ):
        """Record single prediction metrics"""
        pass  # Implementation details
    
    def get_dashboard_metrics(self) -> dict:
        """
        Get metrics for monitoring dashboard
        
        Returns:
            Key metrics for alerting
        """
        return {
            'latency_p50_ms': np.median(self.metrics['latency_p50']),
            'latency_p99_ms': np.percentile(self.metrics['latency_p99'], 99),
            'cache_hit_rate': np.mean(self.metrics['cache_hit_rate']) * 100,
            'error_rate': np.mean(self.metrics['error_rate']) * 100,
            'qps': np.mean(self.metrics['throughput']),
            'cost_per_1k_predictions': np.mean(self.metrics['cost_per_prediction']) * 1000
        }
```

### SLA Definition

```python
class InferenceSLA:
    """
    Define and monitor SLA for inference service
    """
    
    def __init__(self):
        self.sla_targets = {
            'p99_latency_ms': 100,
            'availability': 99.9,
            'error_rate': 0.1  # 0.1%
        }
    
    def check_sla_compliance(self, metrics: dict) -> dict:
        """
        Check if current metrics meet SLA
        
        Returns:
            SLA compliance report
        """
        compliance = {}
        
        for metric, target in self.sla_targets.items():
            actual = metrics.get(metric, 0)
            
            if metric == 'error_rate':
                # Lower is better
                meets_sla = actual <= target
            else:
                # Check if within range (e.g., latency or availability)
                meets_sla = actual <= target if 'latency' in metric else actual >= target
            
            compliance[metric] = {
                'target': target,
                'actual': actual,
                'meets_sla': meets_sla,
                'margin': target - actual if 'latency' in metric or 'error' in metric else actual - target
            }
        
        return compliance
```

---

## Key Takeaways

✅ **Batch inference** precomputes predictions, cheaper, higher latency  
✅ **Real-time inference** computes on-demand, expensive, lower latency  
✅ **Hybrid approach** combines both for optimal cost/performance  
✅ **Multi-tier caching** (memory → Redis → compute) optimizes latency  
✅ **Prediction warming** preloads cache for likely requests  
✅ **Conditional updates** reduce unnecessary recomputation  
✅ **Graceful degradation** ensures reliability via fallback strategies  
✅ **Latency vs cost** is the fundamental trade-off  
✅ **Feature freshness** often determines the choice  
✅ **Most systems** use hybrid: batch for bulk, real-time for edge cases  
✅ **Cache hit rate** critical metric for hybrid systems  
✅ **SLA monitoring** ensures service quality  

---

**Originally published at:** [arunbaby.com/ml-system-design/0005-batch-realtime-inference](https://www.arunbaby.com/ml-system-design/0005-batch-realtime-inference/)

*If you found this helpful, consider sharing it with others who might benefit.*

