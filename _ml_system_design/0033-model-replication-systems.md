---
title: "Model Replication Systems"
day: 33
collection: ml_system_design
categories:
  - ml_system_design
tags:
  - distributed-systems
  - model-serving
  - replication
  - high-availability
subdomain: "Infrastructure"
tech_stack: [Kubernetes, Docker, Terraform, S3, Redis]
scale: "Multi-region, 99.99% uptime"
companies: [Netflix, Uber, Spotify, Amazon]
related_dsa_day: 33
related_speech_day: 33
---

**"Ensuring your ML models are available everywhere, all the time."**

## 1. The Problem: Single Point of Failure

Deploying an ML model to a single server is a recipe for disaster:
- **Server crashes** → Service unavailable.
- **Network partition** → Users in Europe can't reach a US-based server.
- **Traffic spike** → Server overloaded.

**Model Replication** solves this by deploying multiple copies of the model across different servers, regions, or data centers.

## 2. Why Replicate Models?

**1. High Availability (HA):**
- If one replica fails, traffic routes to healthy replicas.
- Target: 99.99% uptime (< 53 minutes downtime per year).

**2. Low Latency:**
- Serve users from the nearest replica.
- US users → US server, EU users → EU server.
- Reduces latency from 200ms to 20ms.

**3. Load Balancing:**
- Distribute inference requests across replicas.
- Prevents any single server from being overwhelmed.

**4. Disaster Recovery:**
- If an entire data center goes down (fire, earthquake), other regions continue serving.

## 3. Replication Strategies

### Strategy 1: Active-Active (Multi-Master)
All replicas actively serve traffic simultaneously.

**Architecture:**
```
       Load Balancer
      /      |      \
  Replica1  Replica2  Replica3
  (US-West) (US-East) (EU-West)
```

**Pros:**
- Full traffic distribution.
- Maximum throughput.

**Cons:**
- Requires synchronization if models have state (rare for inference).

### Strategy 2: Active-Passive (Primary-Standby)
One replica serves traffic. Others are on standby.

**Architecture:**
```
  Primary (Active)
       |
  Standby 1, Standby 2 (Passive)
```

**Pros:**
- Simple to implement.
- Standby can be used for testing new models.

**Cons:**
- Underutilized resources (standby idles).
- Failover delay (10-60 seconds).

### Strategy 3: Geo-Replication
Replicas deployed in different geographic regions.

**Architecture:**
```
US-West Cluster <---> US-East Cluster <---> EU Cluster
```

**Pros:**
- Complies with data residency laws (GDPR: EU data stays in EU).
- Low latency for global users.

**Cons:**
- Higher cost (multi-region infrastructure).
- Complex network topology.

## 4. Model Synchronization Patterns

**Challenge:** How do we ensure all replicas serve the **same** model version?

### Pattern 1: Push-Based Deployment
A central controller pushes new models to all replicas.

**Flow:**
1. Train new model.
2. Upload to central storage (S3).
3. Controller triggers deployment to all replicas.
4. Replicas pull the model and reload.

```python
# Pseudo-code for controller
def deploy_model(model_path, replicas):
    for replica in replicas:
        replica.pull_model(model_path)
        replica.reload()
```

**Pros:**
- Centralized control.
- Ensures consistency.

**Cons:**
- Single point of failure (controller).
- Deployment can be slow (sequential updates).

### Pattern 2: Pull-Based Deployment (Polling)
Each replica periodically checks for new models.

**Flow:**
1. Upload model to S3.
2. Replicas poll S3 every 60 seconds.
3. If new model detected, download and reload.

```python
import time
import hashlib

def poll_for_updates(model_url, current_hash):
    while True:
        new_hash = get_model_hash(model_url)
        if new_hash != current_hash:
            download_model(model_url)
            reload_model()
            current_hash = new_hash
        time.sleep(60)
```

**Pros:**
- Decentralized (no controller).
- Replicas update independently.

**Cons:**
- Polling overhead.
- Inconsistent state (replicas update at different times).

### Pattern 3: Event-Driven Deployment
Replicas subscribe to a message queue (Kafka, SNS). Controller publishes a "new model available" event.

**Flow:**
1. Upload model to S3.
2. Publish message to Kafka topic: `model-updates`.
3. Replicas consume messages and download the new model.

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer('model-updates')
for message in consumer:
    model_url = message.value
    download_model(model_url)
    reload_model()
```

**Pros:**
- Real-time updates.
- Decoupled controller and replicas.

**Cons:**
- Requires message queue infrastructure.
- Ordering guarantees needed.

## 5. Deep Dive: Canary Deployment

**Problem:** A new model might have bugs. Rolling out to all replicas at once is risky.

**Solution:** **Canary Deployment**
1. Deploy new model to **1%** of replicas (canaries).
2. Monitor metrics (latency, error rate, accuracy).
3. If healthy, gradually increase to 10%, 50%, 100%.
4. If unhealthy, rollback immediately.

**Architecture:**
```
Load Balancer
  |
  ├─ 99% traffic → Old Model (v1)
  └─  1% traffic → New Model (v2) [Canary]
```

**Implementation with Kubernetes:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-v1
spec:
  replicas: 99
  template:
    spec:
      containers:
      - name: model
        image: model:v1

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-v2-canary
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: model
        image: model:v2
```

**Monitoring Canary:**
```python
def monitor_canary(canary_metrics, baseline_metrics):
    if canary_metrics['error_rate'] > baseline_metrics['error_rate'] * 1.5:
        rollback()
    elif canary_metrics['p99_latency'] > baseline_metrics['p99_latency'] * 1.2:
        rollback()
    else:
        promote_canary()
```

## 6. Deep Dive: Blue-Green Deployment

**Concept:** Maintain two identical environments: Blue (current) and Green (new).

**Flow:**
1. **Blue** is live (serving 100% traffic).
2. Deploy new model to **Green**.
3. Test **Green** (smoke tests, load tests).
4. Switch traffic from **Blue** to **Green** (atomic switch).
5. If issues arise, switch back to **Blue** instantly.

**Pros:**
- Zero-downtime deployment.
- Instant rollback.

**Cons:**
- Requires 2x resources (both environments running).

## 7. Deep Dive: Model Versioning and Rollback

**Challenge:** How do we track which model version is deployed where?

**Solution: Model Registry**
- Central database of all model versions.
- Each model tagged with: version, timestamp, accuracy metrics, deployment status.

**Schema:**
```sql
CREATE TABLE models (
    model_id UUID PRIMARY KEY,
    version VARCHAR,
    created_at TIMESTAMP,
    accuracy FLOAT,
    deployed_replicas INTEGER,
    status VARCHAR  -- 'training', 'canary', 'production', 'deprecated'
);
```

**Rollback Strategy:**
1. Detect issue (alert triggered).
2. Query model registry for last known good version.
3. Trigger redeployment to all replicas.

```python
def rollback():
    last_good_version = db.query(
        "SELECT version FROM models WHERE status='production' ORDER BY created_at DESC LIMIT 1"
    )
    deploy_model(last_good_version, replicas=all_replicas)
```

## 8. Deep Dive: Handling Stateful Models

Most inference models are **stateless** (same input → same output). But some models maintain state:

**Examples:**
- **Recommendation Systems:** User session history.
- **Chatbots:** Conversation context.
- **Reinforcement Learning:** Agent state.

**Challenge with Replication:**
If a user's first request goes to Replica 1, the second request might go to Replica 2 (which doesn't have the session state).

**Solution 1: Sticky Sessions**
Route all requests from the same user to the same replica.

```nginx
upstream backend {
    ip_hash;  # Hash user IP to the same server
    server replica1:8000;
    server replica2:8000;
}
```

**Cons:** If a replica dies, user sessions are lost.

**Solution 2: Shared State Store (Redis)**
All replicas read/write state to a central Redis cluster.

```python
import redis

redis_client = redis.Redis()

def predict(user_id, features):
    # Load user state
    state = redis_client.get(f"user:{user_id}:state")
    
    # Run inference
    result = model.predict(features, state)
    
    # Update state
    redis_client.set(f"user:{user_id}:state", result['new_state'])
    
    return result['prediction']
```

**Pros:** Stateless replicas, can route to any replica.
**Cons:** Redis becomes a bottleneck.

## 9. Deep Dive: Cross-Region Replication

**Scenario:** Replicate a recommendation model across US, EU, and Asia.

**Challenges:**
1. **Model Artifact Size:** 5 GB model → slow to transfer across continents.
2. **Network Latency:** EU → Asia = 300ms.
3. **Cost:** Cross-region data transfer is expensive ($0.02/GB in AWS).

**Optimization 1: Delta Updates**
Don't transfer the entire model. Only send the changed weights.

```python
def compute_delta(old_model, new_model):
    delta = {}
    for layer in new_model.layers:
        delta[layer.name] = new_model.weights[layer.name] - old_model.weights[layer.name]
    return delta

def apply_delta(model, delta):
    for layer_name, weight_diff in delta.items():
        model.weights[layer_name] += weight_diff
```

**Optimization 2: Model Compression**
Compress the model before transfer (gzip, quantization).

```python
import gzip

# Compress
with open('model.pkl', 'rb') as f_in:
    with gzip.open('model.pkl.gz', 'wb') as f_out:
        f_out.writelines(f_in)

# Transfer compressed file (5 GB → 1 GB)
```

**Optimization 3: Regional Model Stores**
Replicate the model to regional S3 buckets.
- US replicas pull from `s3://models-us-west/`
- EU replicas pull from `s3://models-eu-west/`

```python
def get_model_url(region):
    base_url = f"s3://models-{region}"
    return f"{base_url}/model-v123.pkl"
```

## 10. Deep Dive: Health Checks and Auto-Scaling

**Health Check Types:**
1. **Liveness Probe:** Is the server running?
   - HTTP GET `/health` → 200 OK.
2. **Readiness Probe:** Is the server ready to serve traffic?
   - Check: Model loaded? Database connected?

```python
from fastapi import FastAPI

app = FastAPI()

model = None

@app.on_event("startup")
def load_model():
    global model
    model = load_model_from_s3()

@app.get("/health")
def liveness():
    return {"status": "alive"}

@app.get("/ready")
def readiness():
    if model is None:
        return {"status": "not ready"}, 503
    return {"status": "ready"}
```

**Auto-Scaling:**
Scale replicas based on traffic.

**Kubernetes Horizontal Pod Autoscaler (HPA):**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model
  minReplicas: 3
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

When CPU > 70%, Kubernetes spawns more pods. When CPU < 70%, it scales down.

## 11. Real-World Case Studies

### Case Study 1: Netflix Model Replication
Netflix uses **A/B testing** at scale across global replicas.

**Architecture:**
- Deploy Model A to US-West.
- Deploy Model B to US-East.
- Compare engagement metrics (watch time).
- Winner gets deployed globally.

### Case Study 2: Uber's Michelangelo
Uber's ML platform deploys models to **hundreds of cities**.

**Challenge:** Each city has different demand patterns.
**Solution:** City-specific models.
- `model-san-francisco-v10`
- `model-new-york-v12`

**Replication:**
- Each city's model is replicated 10x within that region's data center.

### Case Study 3: Spotify Recommendations
Spotify serves personalized playlists to 500M users.

**Architecture:**
- 50+ replicas per region.
- Models deployed via Kubernetes.
- Canary deployment for new models (1% → 100%).

## Implementation: Model Replication with Docker + Kubernetes

**Step 1: Dockerize the Model**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model.pkl model.pkl
COPY serve.py serve.py
CMD ["python", "serve.py"]
```

**Step 2: Deploy to Kubernetes**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-inference
spec:
  replicas: 10
  selector:
    matchLabels:
      app: model
  template:
    metadata:
      labels:
        app: model
    spec:
      containers:
      - name: model
        image: my-registry/model:v1
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

**Step 3: Expose via Load Balancer**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  type: LoadBalancer
  selector:
    app: model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

## Top Interview Questions

**Q1: How do you ensure all replicas serve the same model version?**
*Answer:*
- Use a **model registry** with version tracking.
- Deploy atomically (all replicas pull the same version).
- Use checksums/hashes to verify model integrity.

**Q2: What happens if a replica is serving an old model version?**
*Answer:*
- **Monitoring:** Track model version per replica.
- **Alert:** If version mismatch detected, trigger alert.
- **Force Update:** Controller sends a "reload model" command.

**Q3: How do you handle model deployment in a multi-region setup?**
*Answer:*
- **Regional Rollout:** Deploy to one region first (canary).
- **Monitor:** Check error rates, latency.
- **Progressive Rollout:** If healthy, deploy to other regions.

**Q4: What's the difference between horizontal and vertical scaling for model replicas?**
*Answer:*
- **Horizontal:** Add more replicas (more servers). Better for handling high traffic.
- **Vertical:** Increase resources per replica (more CPU/RAM). Better for larger models.

**Q5: How do you minimize downtime during model updates?**
*Answer:*
- **Rolling Update:** Update replicas one at a time.
- **Blue-Green:** Maintain two environments, switch traffic atomically.
- **Canary:** Gradually shift traffic to the new model.

## 12. Deep Dive: Shadow Deployment (Dark Launch)

**Concept:** Deploy a new model alongside the old model, but don't serve its predictions to users. Instead, log predictions for comparison.

**Architecture:**
```
User Request
    ↓
Load Balancer
    ↓
┌───────────────────────┐
│  Primary Model (v1)   │ → Response to User
│  Shadow Model (v2)    │ → Predictions logged (not served)
└───────────────────────┘
```

**Implementation:**
```python
import logging

class ShadowPredictor:
    def __init__(self, primary_model, shadow_model):
        self.primary = primary_model
        self.shadow = shadow_model
    
    def predict(self, features):
        # Primary prediction (served to user)
        primary_pred = self.primary.predict(features)
        
        # Shadow prediction (async logging)
        try:
            shadow_pred = self.shadow.predict(features)
            logging.info(f"Shadow pred: {shadow_pred}, Primary: {primary_pred}")
        except Exception as e:
            logging.error(f"Shadow model failed: {e}")
        
        return primary_pred
```

**Benefits:**
- **Zero Risk:** Users never see shadow model predictions.
- **Real Production Data:** Validate on actual user queries.
- **Performance Comparison:** Compare latency, accuracy in real-time.

**Use Case:** Testing a radically different model architecture without risk.

## 13. Deep Dive: Gradual Traffic Shifting

More sophisticated than binary canary deployment, gradually shift traffic over hours/days.

**Strategy:**
```
Hour 0:  0% v2,  100% v1
Hour 1:  5% v2,   95% v1
Hour 2: 10% v2,   90% v1
Hour 4: 25% v2,   75% v1
Hour 8: 50% v2,   50% v1
Hour 12: 100% v2,  0% v1
```

**Implementation with Feature Flags:**
```python
import random

class FeatureFlag:
    def __init__(self):
        self.rollout_percentage = 0  # Start at 0%
    
    def should_use_new_model(self, user_id):
        # Consistent hashing: same user always gets same experience
        user_hash = hash(user_id) % 100
        return user_hash < self.rollout_percentage
    
    def set_rollout(self, percentage):
        self.rollout_percentage = percentage

# Usage
flag = FeatureFlag()
flag.set_rollout(25)  # 25% of users see new model

def predict(user_id, features):
    if flag.should_use_new_model(user_id):
        return model_v2.predict(features)
    else:
        return model_v1.predict(features)
```

## 14. Deep Dive: Model Performance Monitoring

**Metrics to Track:**

**1. Latency Metrics:**
- **P50 (Median):** 50% of requests complete in X ms.
- **P95:** 95% of requests complete in X ms.
- **P99:** 99% of requests (worst 1%) complete in X ms.

**Why P99 matters:** Even if P50 is 10ms, P99 of 5000ms means some users have terrible experience.

**2. Throughput Metrics:**
- **Requests Per Second (RPS):** How many requests the replica can handle.
- **Saturation:** % of capacity used. If > 80%, scale up.

**3. Error Metrics:**
- **Error Rate:** % of requests that fail.
- **Error Types:** Timeout, Model Error, OOM, Network Error.

**4. Business Metrics:**
- **Click-Through Rate (CTR):** For recommendation models.
- **Conversion Rate:** For ranking models.
- **Revenue Per User:** For personalization models.

**Prometheus Example:**
```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
prediction_counter = Counter(
    'model_predictions_total',
    'Total predictions',
    ['model_version', 'replica_id']
)

# Histograms (for latency)
prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency',
    ['model_version'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

# Gauges (for current state)
active_replicas = Gauge(
    'model_active_replicas',
    'Number of active replicas',
    ['model_version']
)

#Usage
@app.post("/predict")
def predict(features):
    with prediction_latency.labels(model_version='v2').time():
        result = model.predict(features)
    
    prediction_counter.labels(
        model_version='v2',
        replica_id=get_replica_id()
    ).inc()
    
    return result
```

## 15. Deep Dive: Cost Optimization

Replicating models is expensive. How do we minimize cost without sacrificing reliability?

**Strategy 1: Right-Sizing Replicas**
Don't over-provision. Use actual traffic data to determine replica count.

**Formula:**
\\[
\text{Required Replicas} = \frac{\text{Peak RPS}}{\text{RPS per Replica}} \times \text{Safety Factor}
\\]

**Example:**
- Peak traffic: 10,000 RPS
- Each replica handles: 500 RPS
- Safety factor: 1.5 (for spikes)
- Required: $(10,000 / 500) \times 1.5 = 30$ replicas

**Strategy 2: Spot Instances for Non-Critical Replicas**
AWS Spot Instances are 70% cheaper but can be terminated.

```python
# Kubernetes Node Selector for Spot Instances
spec:
  nodeSelector:
    instance-type: spot
  tolerations:
  - key: "spot"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
```

**Use Case:** Use spot instances for 50% of replicas. If terminated, route to on-demand replicas.

**Strategy 3: Model Quantization**
Reduce model size from FP32 to INT8.
- **Size:** 4 GB → 1 GB (4x reduction)
- **Inference Speed:** 2-3x faster
- **Cost:** Fit 4x more replicas per server

```python
import torch

# Quantize model
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Size comparison
original_size = os.path.getsize('model_fp32.pt')
quantized_size = os.path.getsize('model_int8.pt')
print(f"Compression ratio: {original_size / quantized_size:.2f}x")
```

**Strategy 4: Auto-Scaling with Predictive Scaling**
Don't wait for load to spike. Predict it.

```python
def predict_traffic(hour_of_day, day_of_week):
    # Historical average
    baseline = historical_traffic[day_of_week][hour_of_day]
    
    # Scale up 5 minutes before predicted spike
    return int(baseline * 1.2)

# Cron job runs every 5 minutes
current_hour = datetime.now().hour
predicted_rps = predict_traffic(current_hour, datetime.now().weekday())
required_replicas = calculate_replicas(predicted_rps)
scale_to(required_replicas)
```

## 16. Deep Dive: Security Considerations

**1. Model Theft Prevention:**
Large models (GPT-4, Stable Diffusion) are valuable IP. Prevent extraction.

**Attack Vector:** Adversary queries model repeatedly to reverse-engineer weights.
**Defense:** Rate limiting, query auditing.

```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests_per_minute=100):
        self.requests = defaultdict(list)
        self.limit = max_requests_per_minute
    
    def allow_request(self, user_id):
        now = time.time()
        # Remove requests older than 1 minute
        self.requests[user_id] = [
            ts for ts in self.requests[user_id]
            if now - ts < 60
        ]
        
        if len(self.requests[user_id]) >= self.limit:
            return False
        
        self.requests[user_id].append(now)
        return True
```

**2. Data Privacy:**
Ensure replicas don't log sensitive user data.

```python
import re

def sanitize_logs(log_message):
    # Redact emails, phone numbers, SSNs
    log_message = re.sub(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', '[REDACTED_EMAIL]', log_message, flags=re.I)
    log_message = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', log_message)
    return log_message

logging.info(sanitize_logs(f"User query: {user_input}"))
```

**3. Model Integrity:**
Verify that the deployed model hasn't been tampered with.

```python
import hashlib

def verify_model_integrity(model_path, expected_hash):
    with open(model_path, 'rb') as f:
        model_bytes = f.read()
        computed_hash = hashlib.sha256(model_bytes).hexdigest()
    
    if computed_hash != expected_hash:
        raise SecurityError("Model file has been tampered with!")
    
    return True

# On deployment
expected_hash = "abc123...def"  # From model registry
verify_model_integrity('/models/model.pt', expected_hash)
```

## 17. Deep Dive: Disaster Recovery Drills

**Problem:** You have multi-region replication. But have you tested failover?

**DR Drill Strategy:**
1. **Planned Outage:** Intentionally take down a region.
2. **Monitor:** Verify traffic shifts to healthy regions.
3. **Measure:** Recovery time, user impact.
4. **Document:** Update runbooks.

**Chaos Engineering with Netflix's Chaos Monkey:**
```python
import random

def chaos_monkey():
    # Randomly terminate 1 replica every hour
    if random.random() < 0.1:  # 10% chance
        replica_id = random.choice(get_all_replicas())
        logging.warning(f"Chaos Monkey terminating replica {replica_id}")
        terminate_replica(replica_id)
```

**Result:** Forces teams to build resilient systems.

## 18. Production War Stories

**War Story 1: The Silent Canary Failure**
A team deployed a canary model. Metrics looked good (latency, error rate). But revenue dropped 15%.
**Root Cause:** New model was serving *lower quality* recommendations. Users clicked less.
**Lesson:** Track business metrics, not just technical metrics.

**War Story 2: The Cross-Region Sync Disaster**
A model was deployed to EU-West. The EU-East replica was 2 hours behind (polling delay).
**Result:** Users in Paris saw different recommendations than users in Berlin.
**Lesson:** Use event-driven deployment for consistency guarantees.

**War Story 3: The Thundering Herd**
1000 replicas all polled S3 at the same time (every 60 seconds, on the minute).
**Result:** S3 rate limit errors. Models failed to update.
**Lesson:** Add jitter to polling intervals.

```python
import time
import random

def poll_with_jitter(base_interval=60):
    while True:
        # Sleep 60s ± 10s
        jitter = random.uniform(-10, 10)
        time.sleep(base_interval + jitter)
        check_for_updates()
```

## 19. Deep Dive: A/B Testing Infrastructure for Model Replicas

**Problem:** You have model v1 and v2. Which one is better?

**Solution:** A/B test them on real users.

**Architecture:**
```
User Request
    ↓
Feature Flag Service (reads user_id)
    ↓
├─ 50% → Model v1 (Control)
└─ 50% → Model v2 (Treatment)
```

**Implementation:**
```python
import hashlib

class ABTestRouter:
    def __init__(self):
        self.experiments = {}
    
    def create_experiment(self, exp_id, control_model, treatment_model, traffic_split=0.5):
        self.experiments[exp_id] = {
            'control': control_model,
            'treatment': treatment_model,
            'split': traffic_split
        }
    
    def get_model(self, exp_id, user_id):
        exp = self.experiments[exp_id]
        
        # Consistent hashing: same user always gets same variant
        user_hash = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        bucket = (user_hash % 100) / 100.0
        
        if bucket < exp['split']:
            return exp['treatment']
        else:
            return exp['control']

# Usage
router = ABTestRouter()
router.create_experiment(
    exp_id='model_v2_test',
    control_model=model_v1,
    treatment_model=model_v2,
    traffic_split=0.1  # 10% treatment, 90% control
)

def predict(user_id, features):
    model = router.get_model('model_v2_test', user_id)
    return model.predict(features)
```

**Statistical Significance:**
After 1 week, analyze results:
```python
from scipy import stats

def analyze_ab_test(control_metrics, treatment_metrics):
    # T-test for statistical significance
    t_stat, p_value = stats.ttest_ind(
        control_metrics['ctr'],
        treatment_metrics['ctr']
    )
    
    if p_value < 0.05:
        improvement = (treatment_metrics['ctr'].mean() - control_metrics['ctr'].mean()) / control_metrics['ctr'].mean()
        print(f"Treatment is {improvement*100:.1f}% better (p={p_value:.4f})")
    else:
        print("No significant difference")
```

## 20. Deep Dive: Multi-Armed Bandits for Dynamic Model Selection

**Problem:** A/B tests are slow (weeks to reach significance). Can we adapt faster?

**Solution:** Multi-Armed Bandits (Thompson Sampling).

**Concept:**
- Start with equal traffic to all models.
- Gradually shift traffic to the best-performing model.
- **Result:** Maximize reward while exploring.

**Implementation:**
```python
import numpy as np

class ThompsonSamplingBandit:
    def __init__(self, n_models=3):
        # Beta distribution parameters (successes, failures)
        self.alpha = np.ones(n_models)  # Prior: 1 success
        self.beta = np.ones(n_models)   # Prior: 1 failure
    
    def select_model(self):
        # Sample from Beta distributions
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(len(self.alpha))]
        return np.argmax(samples)
    
    def update(self, model_id, reward):
        if reward == 1:  # Success (user clicked)
            self.alpha[model_id] += 1
        else:  # Failure
            self.beta[model_id] += 1

# Usage
bandit = ThompsonSamplingBandit(n_models=3)

for _ in range(1000):  # 1000 requests
    model_id = bandit.select_model()
    prediction = models[model_id].predict(features)
    
    # Get user feedback (click = 1, no click = 0)
    reward = get_user_feedback()
    
    bandit.update(model_id, reward)

print("Final traffic distribution:")
print(f"Model 1: {bandit.alpha[0] / (bandit.alpha[0] + bandit.beta[0])}")
print(f"Model 2: {bandit.alpha[1] / (bandit.alpha[1] + bandit.beta[1])}")
print(f"Model 3: {bandit.alpha[2] / (bandit.alpha[2] + bandit.beta[2])}")
```

**Benefit:** Converges to the best model in days instead of weeks.

## Key Takeaways

1. **Replication is Essential:** Single-server deployments are fragile.
2. **Active-Active is Preferred:** Maximizes resource utilization and throughput.
3. **Canary Deployments:** Reduce risk when rolling out new models.
4. **Stateless is Simpler:** Stateful models require sticky sessions or shared state stores.
5. **Kubernetes is King:** Industry standard for orchestrating model replicas.

## Summary

| Aspect | Insight |
|:---|:---|
| **Goal** | High availability, low latency, fault tolerance |
| **Strategies** | Active-Active, Active-Passive, Geo-Replication |
| **Deployment** | Canary, Blue-Green, Rolling Updates |
| **Challenges** | State management, version sync, cross-region costs |

---

**Originally published at:** [arunbaby.com/ml-system-design/0033-model-replication-systems](https://www.arunbaby.com/ml-system-design/0033-model-replication-systems/)

*If you found this helpful, consider sharing it with others who might benefit.*
