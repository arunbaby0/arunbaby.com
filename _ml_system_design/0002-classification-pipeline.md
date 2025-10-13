---
title: "Classification Pipeline Design"
day: 2
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - classification
  - pipeline
  - production-ml
domain: Classification
scale: "1M+ predictions/day"
key_components: [preprocessing, model-serving, monitoring]
companies: [Google, Meta, Uber, Airbnb]
related_dsa_day: 2
related_speech_day: 2
---

**From raw data to production predictions: building a classification pipeline that handles millions of requests with 99.9% uptime.**

## Introduction

Classification is one of the most common machine learning tasks in production: spam detection, content moderation, fraud detection, sentiment analysis, image categorization, and countless others. While training a classifier might take hours in a Jupyter notebook, deploying it to production requires a sophisticated pipeline that handles:

- **Real-time inference** (< 100ms latency)
- **Feature engineering** at scale
- **Model versioning** and A/B testing
- **Data drift** detection and handling
- **Explainability** for debugging and compliance
- **Monitoring** for performance degradation
- **Graceful degradation** when components fail

This post focuses on building an end-to-end classification system that processes millions of predictions daily while maintaining high availability and performance.

**What you'll learn:**
- End-to-end pipeline architecture for production classification
- Feature engineering and feature store patterns
- Model serving strategies and optimization
- A/B testing and model deployment
- Monitoring, alerting, and data drift detection
- Real-world examples from Uber, Airbnb, and Meta

---

## Problem Definition

Design a production classification system (example: spam detection for user messages) that:

### Functional Requirements

1. **Real-time Inference**
   - Classify incoming data in real-time
   - Return predictions within latency budget
   - Handle variable request rates

2. **Multi-class Support**
   - Binary classification (spam/not spam)
   - Multi-class (topic categorization)
   - Multi-label (multiple tags per item)

3. **Feature Processing**
   - Transform raw data into model-ready features
   - Handle missing values and outliers
   - Cache expensive feature computations

4. **Model Updates**
   - Deploy new models without downtime
   - A/B test model versions
   - Rollback bad deployments quickly

5. **Explainability**
   - Provide reasoning for predictions
   - Support debugging and compliance
   - Build user trust

### Non-Functional Requirements

1. **Latency**
   - p50 < 20ms (median)
   - p99 < 100ms (99th percentile)
   - Tail latency critical for user experience

2. **Throughput**
   - 1M predictions per day
   - ~12 QPS average, ~100 QPS peak
   - Horizontal scaling for growth

3. **Availability**
   - 99.9% uptime (< 9 hours downtime/year)
   - Graceful degradation on failures
   - No single points of failure

4. **Accuracy**
   - Maintain > 90% precision
   - Maintain > 85% recall
   - Monitor for drift

### Example Use Case: Spam Detection

- **Input:** User message (text, metadata)
- **Output:** {spam, not_spam, confidence}
- **Scale:** 1M messages/day
- **Latency:** < 50ms p99
- **False positive cost:** High (blocks legitimate messages)
- **False negative cost:** Medium (spam gets through)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Client Application                  │
└────────────────────┬────────────────────────────────┘
                     │ HTTP/gRPC request
                     ▼
┌─────────────────────────────────────────────────────┐
│                   API Gateway                        │
│  • Rate limiting                                     │
│  • Authentication                                    │
│  • Request validation                                │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│             Classification Service                   │
│  ┌──────────────────────────────────────────────┐  │
│  │  1. Input Validation & Preprocessing         │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  2. Feature Engineering                      │  │
│  │     • Feature Store lookup (cached)          │  │
│  │     • Real-time feature computation          │  │
│  │     • Feature transformation                 │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  3. Model Inference                          │  │
│  │     • Model serving (TF/PyTorch)             │  │
│  │     • A/B testing routing                    │  │
│  │     • Prediction caching                     │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  4. Post-processing                          │  │
│  │     • Threshold optimization                 │  │
│  │     • Calibration                            │  │
│  │     • Explainability generation              │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  5. Logging & Monitoring                     │  │
│  │     • Prediction logs → Kafka                │  │
│  │     • Metrics → Prometheus                   │  │
│  │     • Traces → Jaeger                        │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────┘
                     ▼
              Response to client
```

**Latency Budget (100ms total):**
```
Input validation:      5ms
Feature extraction:   25ms  ← Often bottleneck
Model inference:      40ms
Post-processing:      10ms
Logging (async):       0ms
Network overhead:     20ms
Total:               100ms ✓
```

---

## Component 1: Input Validation

### Schema Validation with Pydantic

```python
from pydantic import BaseModel, validator, Field
from typing import Optional
import re

class ClassificationRequest(BaseModel):
    """
    Validate incoming classification requests
    """
    text: str = Field(..., min_length=1, max_length=10000)
    user_id: int = Field(..., gt=0)
    language: Optional[str] = Field(default="en", regex="^[a-z]{2}$")
    metadata: Optional[dict] = Field(default_factory=dict)
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or v.isspace():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()
    
    @validator('text')
    def text_length_check(cls, v):
        if len(v) > 10000:
            # Truncate instead of rejecting
            return v[:10000]
        return v
    
    @validator('metadata')
    def metadata_size_check(cls, v):
        if v and len(str(v)) > 1000:
            raise ValueError('Metadata too large')
        return v
    
    class Config:
        # Example for API docs
        schema_extra = {
            "example": {
                "text": "Check out this amazing offer!",
                "user_id": 12345,
                "language": "en",
                "metadata": {"platform": "web"}
            }
        }


# Usage in API endpoint
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/classify")
async def classify(request: ClassificationRequest):
    try:
        # Pydantic automatically validates
        result = await classifier.predict(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Input Sanitization

```python
import html
import unicodedata

def sanitize_text(text: str) -> str:
    """
    Clean and normalize input text
    """
    # HTML unescape
    text = html.unescape(text)
    
    # Unicode normalization (NFKC = compatibility composition)
    text = unicodedata.normalize('NFKC', text)
    
    # Remove control characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\r\t')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


# Example
text = "Hello\u00A0world"  # Non-breaking space
clean = sanitize_text(text)  # "Hello world"
```

---

## Component 2: Feature Engineering

### Feature Store Pattern

```python
from typing import Dict, Any
import redis
import json
from datetime import timedelta

class FeatureStore:
    """
    Centralized feature storage with caching
    """
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
    
    def get_user_features(self, user_id: int) -> Dict[str, Any]:
        """
        Get cached user features or compute
        """
        cache_key = f"features:user:{user_id}"
        
        # Try cache
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Compute expensive features
        features = self._compute_user_features(user_id)
        
        # Cache for future requests
        self.redis.setex(
            cache_key,
            self.default_ttl,
            json.dumps(features)
        )
        
        return features
    
    def _compute_user_features(self, user_id: int) -> Dict[str, Any]:
        """
        Compute user-level features (expensive)
        """
        # Query database
        user = db.get_user(user_id)
        
        return {
            # Profile features
            'account_age_days': (datetime.now() - user.created_at).days,
            'verified': user.is_verified,
            'follower_count': user.followers,
            
            # Behavioral features (aggregated)
            'messages_sent_7d': self._count_messages(user_id, days=7),
            'spam_reports_received': user.spam_reports,
            'avg_message_length': user.avg_message_length,
            
            # Engagement features
            'reply_rate': user.replies_received / max(user.messages_sent, 1),
            'block_rate': user.blocks_received / max(user.messages_sent, 1)
        }
    
    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """
        Extract real-time text features (fast, no caching needed)
        """
        return {
            # Length features
            'char_count': len(text),
            'word_count': len(text.split()),
            'avg_word_length': sum(len(w) for w in text.split()) / len(text.split()),
            
            # Pattern features
            'url_count': text.count('http'),
            'email_count': text.count('@'),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capital_ratio': sum(c.isupper() for c in text) / len(text),
            
            # Linguistic features
            'unique_word_ratio': len(set(text.lower().split())) / len(text.split()),
            'repeated_char_ratio': self._count_repeated_chars(text) / len(text)
        }
    
    def _count_repeated_chars(self, text: str) -> int:
        """Count characters repeated 3+ times (e.g., 'hellooo')"""
        import re
        matches = re.findall(r'(.)\1{2,}', text)
        return len(matches)
```

### Feature Transformation Pipeline

```python
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class FeatureTransformer:
    """
    Transform raw features into model-ready format
    """
    def __init__(self):
        # Fit on training data
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        
        # Feature names for debugging
        self.numerical_features = [
            'account_age_days', 'follower_count', 'messages_sent_7d',
            'char_count', 'word_count', 'url_count', 'exclamation_count',
            'capital_ratio', 'unique_word_ratio'
        ]
    
    def transform(self, user_features: Dict, text_features: Dict, text: str) -> np.ndarray:
        """
        Combine and transform all features
        """
        # Numerical features
        numerical = np.array([
            user_features.get(f, 0.0) for f in self.numerical_features
        ])
        numerical_scaled = self.scaler.transform(numerical.reshape(1, -1))
        
        # Text features (TF-IDF)
        text_vec = self.tfidf.transform([text]).toarray()
        
        # Concatenate all features
        features = np.concatenate([
            numerical_scaled,
            text_vec
        ], axis=1)
        
        return features[0]  # Return 1D array
    
    def get_feature_names(self) -> list:
        """Get all feature names for explainability"""
        return self.numerical_features + list(self.tfidf.get_feature_names_out())
```

---

## Component 3: Model Serving

### Multi-Model Serving with A/B Testing

```python
from typing import Tuple
import hashlib
import torch

class ModelServer:
    """
    Serve multiple model versions with A/B testing
    """
    def __init__(self):
        # Load models
        self.models = {
            'v1': torch.jit.load('spam_classifier_v1.pt'),
            'v2': torch.jit.load('spam_classifier_v2.pt')
        }
        
        # Traffic split (%)
        self.traffic_split = {
            'v1': 90,
            'v2': 10
        }
        
        # Model metadata
        self.model_info = {
            'v1': {'deployed_at': '2025-01-01', 'training_accuracy': 0.92},
            'v2': {'deployed_at': '2025-01-15', 'training_accuracy': 0.94}
        }
    
    def select_model(self, user_id: int) -> str:
        """
        Consistent hashing for A/B test assignment
        
        Same user always gets same model (important for consistency)
        """
        # Hash user_id to [0, 99]
        hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        bucket = hash_val % 100
        
        # Assign to model based on traffic split
        if bucket < self.traffic_split['v1']:
            return 'v1'
        else:
            return 'v2'
    
    def predict(self, features: np.ndarray, user_id: int) -> Tuple[int, np.ndarray, str]:
        """
        Run inference with selected model
        
        Returns:
            prediction: Class label (0 or 1)
            probabilities: Class probabilities
            model_version: Which model was used
        """
        # Select model
        model_version = self.select_model(user_id)
        model = self.models[model_version]
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            logits = model(features_tensor)
            probabilities = torch.softmax(logits, dim=1).numpy()[0]
            prediction = int(np.argmax(probabilities))
        
        return prediction, probabilities, model_version
```

### Model Caching

```python
from functools import lru_cache
import hashlib

class CachedModelServer:
    """
    Cache predictions for identical inputs
    """
    def __init__(self, model_server: ModelServer, cache_size=10000):
        self.model_server = model_server
        self.cache_size = cache_size
    
    def _feature_hash(self, features: np.ndarray) -> str:
        """Create hash of feature vector"""
        return hashlib.sha256(features.tobytes()).hexdigest()
    
    @lru_cache(maxsize=10000)
    def predict_cached(self, feature_hash: str, user_id: int) -> Tuple:
        """Cached prediction (won't actually work with mutable args, just illustrative)"""
        # In practice, use Redis or Memcached for distributed caching
        pass
    
    def predict(self, features: np.ndarray, user_id: int) -> Tuple:
        """
        Try cache first, fallback to model
        """
        feature_hash = self._feature_hash(features)
        cache_key = f"pred:{feature_hash}:{user_id}"
        
        # Try Redis cache
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Cache miss - run model
        prediction, probabilities, model_version = self.model_server.predict(
            features, user_id
        )
        
        # Cache result (5 minute TTL)
        result = (prediction, probabilities.tolist(), model_version)
        redis_client.setex(cache_key, 300, json.dumps(result))
        
        return result
```

---

## Component 4: Post-Processing

### Threshold Optimization

```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

class ThresholdOptimizer:
    """
    Find optimal classification threshold
    """
    def __init__(self, target_precision=0.95):
        self.target_precision = target_precision
        self.threshold = 0.5  # Default
    
    def optimize(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Find threshold that maximizes recall while maintaining precision
        
        Common in spam detection: high precision required (few false positives)
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Find highest recall where precision >= target
        valid_indices = np.where(precisions >= self.target_precision)[0]
        
        if len(valid_indices) == 0:
            print(f"Warning: Cannot achieve {self.target_precision} precision")
            # Fall back to threshold that maximizes F1
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_idx = np.argmax(f1_scores)
            self.threshold = thresholds[best_idx]
        else:
            # Choose threshold with maximum recall among valid options
            best_idx = valid_indices[np.argmax(recalls[valid_indices])]
            self.threshold = thresholds[best_idx]
        
        print(f"Optimal threshold: {self.threshold:.3f}")
        print(f"Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}")
        
        return self.threshold
    
    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply optimized threshold"""
        return (probabilities >= self.threshold).astype(int)
```

### Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

class CalibratedClassifier:
    """
    Ensure predicted probabilities match actual frequencies
    
    Example: If model predicts 70% spam, ~70% should actually be spam
    """
    def __init__(self, base_model):
# Wrap model with calibration
        self.calibrated_model = CalibratedClassifierCV(
            base_model,
    method='sigmoid',  # or 'isotonic'
    cv=5
)
    
    def fit(self, X, y):
        """Train with calibration"""
        self.calibrated_model.fit(X, y)
    
    def predict_proba(self, X):
        """Return calibrated probabilities"""
        return self.calibrated_model.predict_proba(X)


# Before calibration:
# Predicted 80% spam → Actually 65% spam (overconfident)

# After calibration:
# Predicted 80% spam → Actually 78% spam (calibrated)
```

---

## Component 5: Explainability

### SHAP Values

```python
import shap

class ExplainableClassifier:
    """
    Generate explanations for predictions
    """
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(model)
    
    def explain(self, features: np.ndarray, top_k=3) -> str:
        """
        Generate human-readable explanation
        """
        # Compute SHAP values
        shap_values = self.explainer.shap_values(features.reshape(1, -1))
        
        # Get top contributing features
        feature_contributions = list(zip(
            self.feature_names,
    shap_values[0]
))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Format explanation
        top_features = feature_contributions[:top_k]
        explanation = "Key factors: "
        explanation += ", ".join([
            f"{name} ({value:+.3f})"
            for name, value in top_features
        ])
        
        return explanation


# Example output:
# "Key factors: url_count (+0.234), capital_ratio (+0.156), exclamation_count (+0.089)"
```

### Rule-Based Explanations

```python
def generate_explanation(features: Dict, prediction: int) -> str:
    """
    Simple rule-based explanation (faster than SHAP)
    """
    if prediction == 1:  # Spam
        reasons = []
        
        if features['url_count'] > 2:
            reasons.append("contains multiple URLs")
        
        if features['exclamation_count'] > 3:
            reasons.append("excessive exclamation marks")
        
        if features['capital_ratio'] > 0.5:
            reasons.append("too many capital letters")
        
        if features['repeated_char_ratio'] > 0.1:
            reasons.append("repeated characters")
        
        if not reasons:
            reasons.append("multiple spam indicators detected")
        
        return f"Classified as spam because: {', '.join(reasons)}"
    
    else:  # Not spam
        return "No spam indicators detected"
```

---

## Monitoring & Drift Detection

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
prediction_counter = Counter(
    'classification_predictions_total',
    'Total predictions',
    ['model_version', 'prediction_class']
)

latency_histogram = Histogram(
    'classification_latency_seconds',
    'Prediction latency',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

model_confidence = Histogram(
    'classification_confidence',
    'Prediction confidence',
    ['model_version']
)

class MonitoredClassifier:
    """
    Classifier with built-in monitoring
    """
    def __init__(self, classifier):
        self.classifier = classifier
    
    def predict(self, features, user_id):
        start_time = time.time()
        
        # Run prediction
        prediction, probabilities, model_version = self.classifier.predict(
            features, user_id
        )
        
        # Record metrics
        latency = time.time() - start_time
        latency_histogram.observe(latency)
        
        prediction_counter.labels(
            model_version=model_version,
            prediction_class=prediction
        ).inc()
        
        confidence = max(probabilities)
        model_confidence.labels(model_version=model_version).observe(confidence)
        
        return prediction, probabilities, model_version
```

### Data Drift Detection

```python
from scipy import stats
import numpy as np

class DriftDetector:
    """
    Detect distribution shift in features
    """
    def __init__(self, reference_data: np.ndarray, feature_names: list):
        """
        reference_data: Training data statistics
        """
        self.reference_stats = {
            feature: {
                'mean': reference_data[:, i].mean(),
                'std': reference_data[:, i].std(),
                'min': reference_data[:, i].min(),
                'max': reference_data[:, i].max()
            }
            for i, feature in enumerate(feature_names)
        }
    
    def detect_drift(self, current_data: np.ndarray, feature_names: list) -> dict:
        """
        Compare current data to reference distribution
        
        Returns:
            Dictionary of features with significant drift
        """
        drift_alerts = {}
        
        for i, feature in enumerate(feature_names):
            ref_stats = self.reference_stats[feature]
            current_values = current_data[:, i]
            
            # Statistical tests
            # 1. KS test (distribution shift)
            ks_statistic, ks_pvalue = stats.ks_2samp(
                current_values,
                np.random.normal(ref_stats['mean'], ref_stats['std'], len(current_values))
            )
            
            # 2. Mean shift (Z-score)
            current_mean = current_values.mean()
            z_score = abs(current_mean - ref_stats['mean']) / (ref_stats['std'] + 1e-10)
            
            # Alert if significant drift
            if ks_pvalue < 0.01 or z_score > 3:
                drift_alerts[feature] = {
                    'z_score': z_score,
                    'ks_pvalue': ks_pvalue,
                    'current_mean': current_mean,
                    'reference_mean': ref_stats['mean']
                }
        
        return drift_alerts


# Usage
detector = DriftDetector(training_data, feature_names)

# Check daily
current_batch = get_last_24h_features()
drift = detector.detect_drift(current_batch, feature_names)

if drift:
    send_alert(f"Drift detected in features: {list(drift.keys())}")
    trigger_model_retraining()
```

---

## Deployment Strategies

### Blue-Green Deployment

```python
class BlueGreenDeployment:
    """
    Zero-downtime deployment with instant rollback
    """
    def __init__(self):
        self.models = {
            'blue': None,   # Current production
            'green': None   # New version
        }
        self.active = 'blue'
    
    def deploy_new_version(self, new_model):
        """
        Deploy to green environment
        """
        inactive = 'green' if self.active == 'blue' else 'blue'
        
        # Load new model to inactive environment
        print(f"Loading new model to {inactive}...")
        self.models[inactive] = new_model
        
        # Run smoke tests
        if not self.smoke_test(inactive):
            print("Smoke tests failed! Keeping current version.")
            return False
        
        # Switch traffic
        print(f"Switching traffic from {self.active} to {inactive}")
        self.active = inactive
        
        return True
    
    def smoke_test(self, environment: str) -> bool:
        """
        Basic health checks before switching traffic
        """
        model = self.models[environment]
        
        # Test with sample inputs
        test_cases = load_test_cases()
        
        for input_data, expected_output in test_cases:
            try:
                output = model.predict(input_data)
                if output is None:
                    return False
            except Exception as e:
                print(f"Smoke test failed: {e}")
                return False
        
        return True
    
    def rollback(self):
        """
        Instant rollback to previous version
        """
        old = self.active
        self.active = 'green' if self.active == 'blue' else 'blue'
        print(f"Rolled back from {old} to {self.active}")
```

---

## Complete Example: Spam Classifier Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

app = FastAPI(title="Spam Classification Service")

# Initialize components
feature_store = FeatureStore(redis_client)
feature_transformer = FeatureTransformer()
model_server = ModelServer()
threshold_optimizer = ThresholdOptimizer(target_precision=0.95)
explainer = ExplainableClassifier(model_server.models['v1'], feature_names)

class SpamRequest(BaseModel):
    text: str
    user_id: int

class SpamResponse(BaseModel):
    is_spam: bool
    confidence: float
    explanation: str
    model_version: str
    latency_ms: float

@app.post("/classify", response_model=SpamResponse)
async def classify_message(request: SpamRequest):
    """
    Main classification endpoint
    """
    start_time = time.time()
    
    try:
        # 1. Sanitize input
        clean_text = sanitize_text(request.text)
        
        # 2. Feature engineering (parallel)
        user_features_task = asyncio.create_task(
            asyncio.to_thread(feature_store.get_user_features, request.user_id)
        )
        text_features = feature_store.extract_text_features(clean_text)
        user_features = await user_features_task
        
        # 3. Transform features
        features = feature_transformer.transform(
            user_features,
            text_features,
            clean_text
        )
        
        # 4. Model inference
        prediction, probabilities, model_version = model_server.predict(
            features,
            request.user_id
        )
        
        # 5. Apply threshold
        is_spam = threshold_optimizer.predict(probabilities[1])
        confidence = float(probabilities[1])
        
        # 6. Generate explanation
        explanation = explainer.explain(features)
        
        # 7. Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # 8. Log prediction (async)
        asyncio.create_task(log_prediction(
            request, prediction, confidence, model_version
        ))
        
        return SpamResponse(
            is_spam=bool(is_spam),
            confidence=confidence,
            explanation=explanation,
            model_version=model_version,
            latency_ms=latency_ms
        )
    
    except Exception as e:
        # Log error
        logger.error(f"Classification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Classification failed")

async def log_prediction(request, prediction, confidence, model_version):
    """
    Async logging to Kafka
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_id': request.user_id,
        'text_hash': hashlib.sha256(request.text.encode()).hexdigest(),
        'prediction': int(prediction),
        'confidence': float(confidence),
        'model_version': model_version
    }
    
    kafka_producer.send('predictions', json.dumps(log_entry))
```

---

## Key Takeaways

✅ **Feature stores** centralize feature computation and caching  
✅ **A/B testing** enables safe model rollouts with consistent user assignment  
✅ **Threshold optimization** balances precision/recall for business needs  
✅ **Monitoring** catches drift and performance degradation early  
✅ **Explainability** builds trust and aids debugging
✅ **Deployment strategies** enable zero-downtime updates and instant rollback

---

## Further Reading

**Papers:**
- [Rules of Machine Learning (Google)](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Michelangelo: Uber's ML Platform](https://www.uber.com/blog/michelangelo-machine-learning-platform/)
- [Airbnb's ML Infrastructure](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d)

**Tools:**
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Feast](https://feast.dev/) - Feature store
- [BentoML](https://www.bentoml.com/) - Model serving
- [Evidently](https://www.evidentlyai.com/) - ML monitoring

**Books:**
- *Machine Learning Design Patterns* (Lakshmanan et al.)
- *Designing Machine Learning Systems* (Chip Huyen)

---

**Originally published at:** [arunbaby.com/ml-system-design/0002-classification-pipeline](https://www.arunbaby.com/ml-system-design/0002-classification-pipeline/)

*If you found this helpful, consider sharing it with others who might benefit.*
