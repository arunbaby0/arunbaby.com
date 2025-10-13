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

**How to choose between batch and real-time inference—the architectural decision that shapes your entire ML serving infrastructure.**

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

## Key Takeaways

✅ **Batch inference** precomputes predictions—cheaper, higher latency  
✅ **Real-time inference** computes on-demand—expensive, lower latency  
✅ **Hybrid approach** combines both for optimal cost/performance  
✅ **Latency vs cost** is the fundamental trade-off  
✅ **Feature freshness** often determines the choice  
✅ **Most systems** use hybrid: batch for bulk, real-time for edge cases  
✅ **Cache hit rate** critical metric for hybrid systems  
✅ **Graceful degradation** via fallbacks ensures reliability  

---

**Originally published at:** [arunbaby.com/ml-system-design/0005-batch-realtime-inference](https://www.arunbaby.com/ml-system-design/0005-batch-realtime-inference/)

*If you found this helpful, consider sharing it with others who might benefit.*

