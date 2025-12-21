---
title: "Online Learning Systems"
day: 9
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - online-learning
  - incremental-learning
  - streaming
  - model-updates
  - real-time
subdomain: Model Training
tech_stack: [Python, Kafka, Redis, River, scikit-multiflow, TensorFlow, PyTorch]
scale: "Real-time model updates"
companies: [Google, Meta, Netflix, Uber, Spotify, LinkedIn]
related_dsa_day: 9
related_speech_day: 9
related_agents_day: 9
---

**Design systems that learn continuously from streaming data, adapting to changing patterns without full retraining.**

## Introduction

**Online learning** (incremental learning) updates models continuously as new data arrives, without retraining from scratch.

**Why online learning?**
- **Concept drift:** User behavior changes over time
- **Freshness:** Models stay up-to-date with recent data
- **Efficiency:** No need to retrain on entire dataset
- **Scalability:** Handle unbounded data streams

**Key challenges:**
- Managing model stability vs plasticity
- Handling catastrophic forgetting
- Maintaining low-latency updates
- Ensuring prediction consistency

---

## Online vs Batch Learning

### Comparison

```
┌────────────────┬──────────────────┬──────────────────────┐
│ Aspect         │ Batch Learning   │ Online Learning      │
├────────────────┼──────────────────┼──────────────────────┤
│ Data           │ Fixed dataset    │ Streaming data       │
│ Training       │ Full retrain     │ Incremental updates  │
│ Frequency      │ Daily/weekly     │ Real-time/micro-batch│
│ Memory         │ High (all data)  │ Low (current batch)  │
│ Adaptability   │ Slow             │ Fast                 │
│ Stability      │ High             │ Requires careful tuning│
└────────────────┴──────────────────┴──────────────────────┘
```

### When to Use Online Learning

**Good fits:**
- Recommendation systems (user preferences change)
- Fraud detection (fraud patterns evolve)
- Ad click-through rate prediction
- Search ranking (trending topics)
- Price optimization (market dynamics)

**Poor fits:**
- Image classification (static classes)
- Medical diagnosis (stable conditions)
- Sentiment analysis (language changes slowly)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Sources                          │
│         (User actions, transactions, events)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 Streaming Platform                       │
│                  (Kafka, Kinesis)                        │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Feature  │ │ Training │ │ Serving  │
    │ Pipeline │ │ Service  │ │ Service  │
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
            ┌──────────────────┐
            │  Model Registry  │
            │   (Versioned)    │
            └──────────────────┘
```

---

## Core Implementation

### Basic Online Learning

```python
from river import linear_model, metrics, preprocessing, optim
import numpy as np

class OnlineLearner:
    """
    Simple online learning system
    
    Updates model with each new example
    """
    
    def __init__(self, learning_rate=0.01):
        # Standardize features then logistic regression with SGD optimizer
        self.model = (
            preprocessing.StandardScaler() |
            linear_model.LogisticRegression(optimizer=optim.SGD(lr=learning_rate))
        )
        
        # Track performance
        self.metric = metrics.Accuracy()
        self.predictions = []
    
    def partial_fit(self, X, y):
        """
        Update model with new example
        
        Args:
            X: Feature dict
            y: True label
        
        Returns:
            Updated model
        """
        # Make prediction before updating
        y_pred = self.model.predict_one(X)
        
        # Update metric
        self.metric.update(y, y_pred)
        
        # Update model with new example
        self.model.learn_one(X, y)
        
        return y_pred
    
    def predict(self, X):
        """Make prediction"""
        return self.model.predict_one(X)
    
    def get_metrics(self):
        """Get current performance"""
        return {
            'accuracy': self.metric.get(),
            'n_samples': self.metric.n
        }

# Usage
learner = OnlineLearner()

# Stream of data
for i in range(1000):
    # Simulate incoming data
    X = {'feature1': np.random.randn(), 'feature2': np.random.randn()}
    y = 1 if X['feature1'] + X['feature2'] > 0 else 0
    
    # Update model
    pred = learner.partial_fit(X, y)
    
    if i % 100 == 0:
        metrics = learner.get_metrics()
        print(f"Step {i}: Accuracy = {metrics['accuracy']:.3f}")
```

### Mini-Batch Online Learning

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class MiniBatchOnlineLearner:
    """
    Online learning with mini-batches
    
    Accumulates examples and updates in batches
    """
    
    def __init__(self, input_dim, output_dim, batch_size=32):
        self.batch_size = batch_size
        
        # Neural network model
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Buffer for accumulating examples
        self.buffer = deque(maxlen=batch_size)
    
    def add_example(self, x, y):
        """
        Add example to buffer
        
        Triggers update when buffer is full
        """
        self.buffer.append((x, y))
        
        if len(self.buffer) >= self.batch_size:
            self._update_model()
    
    def _update_model(self):
        """Update model with buffered examples"""
        if not self.buffer:
            return
        
        # Extract batch
        X_batch = torch.stack([x for x, y in self.buffer])
        y_batch = torch.tensor([y for x, y in self.buffer], dtype=torch.long)
        
        # Forward pass
        outputs = self.model(X_batch)
        loss = self.criterion(outputs, y_batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear buffer
        self.buffer.clear()
        
        return loss.item()
    
    def predict(self, x):
        """Make prediction"""
        with torch.no_grad():
            output = self.model(x.unsqueeze(0))
            return torch.argmax(output, dim=1).item()

# Usage
learner = MiniBatchOnlineLearner(input_dim=10, output_dim=2, batch_size=32)

# Stream data
for i in range(1000):
    x = torch.randn(10)
    y = 1 if x.sum() > 0 else 0
    
    learner.add_example(x, y)
```

---

## Handling Concept Drift

### Detection

```python
class DriftDetector:
    """
    Detect concept drift in online learning
    
    Uses sliding window to track performance
    """
    
    def __init__(self, window_size=100, threshold=0.05):
        self.window_size = window_size
        self.threshold = threshold
        
        self.recent_errors = deque(maxlen=window_size)
        self.baseline_error = None
    
    def add_prediction(self, y_true, y_pred):
        """
        Add prediction result
        
        Returns: True if drift detected
        """
        error = 1 if y_true != y_pred else 0
        self.recent_errors.append(error)
        
        # Initialize baseline
        if self.baseline_error is None and len(self.recent_errors) >= self.window_size:
            self.baseline_error = np.mean(self.recent_errors)
            return False
        
        # Check for drift
        if self.baseline_error is not None and len(self.recent_errors) >= self.window_size:
            current_error = np.mean(self.recent_errors)
            
            # Significant increase in error rate
            if current_error > self.baseline_error + self.threshold:
                print(f"⚠️ Drift detected! Error: {self.baseline_error:.3f} → {current_error:.3f}")
                self.baseline_error = current_error  # Update baseline
                return True
        
        return False

# Usage
detector = DriftDetector(window_size=100, threshold=0.05)

for i in range(1000):
    # Simulate concept drift at i=500
    if i < 500:
        y_true = 1
        y_pred = np.random.choice([0, 1], p=[0.1, 0.9])
    else:
        # Distribution changes
        y_true = 1
        y_pred = np.random.choice([0, 1], p=[0.4, 0.6])
    
    drift = detector.add_prediction(y_true, y_pred)
```

### Adaptation Strategies

```python
class AdaptiveOnlineLearner:
    """
    Online learner with adaptive learning rate
    
    Increases learning rate when drift detected
    """
    
    def __init__(self, base_lr=0.01, drift_lr_multiplier=5.0):
        self.base_lr = base_lr
        self.drift_lr_multiplier = drift_lr_multiplier
        self.current_lr = base_lr
        
        self.model = (
            preprocessing.StandardScaler() |
            linear_model.LogisticRegression(optimizer=optim.SGD(lr=self.base_lr))
        )
        self.drift_detector = DriftDetector()
        
        self.drift_mode = False
        self.drift_countdown = 0
    
    def partial_fit(self, X, y):
        """Update model with drift adaptation"""
        # Make prediction
        y_pred = self.model.predict_one(X)
        
        # Check for drift
        drift_detected = self.drift_detector.add_prediction(y, y_pred)
        
        if drift_detected:
            # Enter drift mode: increase learning rate
            self.drift_mode = True
            self.drift_countdown = 100  # Stay in drift mode for 100 samples
            self.current_lr = self.base_lr * self.drift_lr_multiplier
            print(f"📈 Increased learning rate to {self.current_lr}")
        
        # Update model with current learning rate
        # Update with current learning rate by re-wrapping optimizer
        self.model['LogisticRegression'].optimizer = optim.SGD(lr=self.current_lr)
        self.model.learn_one(X, y)
        
        # Decay drift mode
        if self.drift_mode:
            self.drift_countdown -= 1
            if self.drift_countdown <= 0:
                self.drift_mode = False
                self.current_lr = self.base_lr
                print(f"📉 Restored learning rate to {self.current_lr}")
        
        return y_pred
```

---

## Production Patterns

### Pattern 1: Multi-Model Ensemble

```python
class EnsembleOnlineLearner:
    """
    Maintain ensemble of models with different learning rates
    
    Robust to concept drift
    """
    
    def __init__(self, n_models=3):
        # Models with different learning rates
        self.models = [
            (
                preprocessing.StandardScaler() |
                linear_model.LogisticRegression(optimizer=optim.SGD(lr=lr))
            )
            for lr in [0.001, 0.01, 0.1]
        ]
        
        # Track model weights
        self.model_weights = np.ones(n_models) / n_models
        self.model_errors = [deque(maxlen=100) for _ in range(n_models)]
    
    def partial_fit(self, X, y):
        """Update all models"""
        predictions = []
        
        for i, model in enumerate(self.models):
            # Predict
            y_pred = model.predict_one(X)
            predictions.append(y_pred)
            
            # Track error
            error = 1 if y_pred != y else 0
            self.model_errors[i].append(error)
            
            # Update model
            model.learn_one(X, y)
        
        # Update model weights based on recent performance
        self._update_weights()
        
        # Weighted ensemble prediction
        ensemble_pred = self._ensemble_predict(predictions)
        
        return ensemble_pred
    
    def _update_weights(self):
        """Update model weights based on performance"""
        for i in range(len(self.models)):
            if len(self.model_errors[i]) > 0:
                error_rate = np.mean(self.model_errors[i])
                # Weight inversely proportional to error
                self.model_weights[i] = 1 / (error_rate + 0.01)
        
        # Normalize
        self.model_weights /= self.model_weights.sum()
    
    def _ensemble_predict(self, predictions):
        """Weighted voting"""
        # For binary classification
        # Convert predictions to 0/1 probabilities if None
        probs = [1.0 if p == 1 else 0.0 for p in predictions]
        weighted_sum = sum(p * w for p, w in zip(probs, self.model_weights))
        return 1 if weighted_sum >= 0.5 else 0

# Usage
ensemble = EnsembleOnlineLearner(n_models=3)

for X, y in data_stream:
    pred = ensemble.partial_fit(X, y)
```

### Pattern 2: Warm Start from Batch Model

```python
class HybridLearner:
    """
    Start with batch-trained model, then update online
    
    Best of both worlds
    """
    
    def __init__(self, pretrained_model_path):
        # Load pretrained batch model
        self.base_model = self.load_batch_model(pretrained_model_path)
        
        # Online learning on top
        self.online_layer = nn.Linear(self.base_model.output_dim, 2)
        self.optimizer = optim.Adam(self.online_layer.parameters(), lr=0.001)
        
        # Freeze base model initially
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.update_count = 0
        self.unfreeze_after = 1000  # Unfreeze base after 1000 updates
    
    def partial_fit(self, x, y):
        """Update online layer (and optionally base model)"""
        # Forward pass through frozen base
        with torch.no_grad():
            base_features = self.base_model(x)
        
        # Online layer forward pass
        output = self.online_layer(base_features)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(output.unsqueeze(0), torch.tensor([y]))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        
        # Unfreeze base model after warming up
        if self.update_count == self.unfreeze_after:
            print("🔓 Unfreezing base model for fine-tuning")
            for param in self.base_model.parameters():
                param.requires_grad = True
            
            # Lower learning rate for base model
            self.optimizer = optim.Adam([
                {'params': self.base_model.parameters(), 'lr': 0.0001},
                {'params': self.online_layer.parameters(), 'lr': 0.001}
            ])
        
        return torch.argmax(output).item()
```

### Pattern 3: Checkpoint and Rollback

```python
class CheckpointedOnlineLearner:
    """
    Online learner with periodic checkpointing
    
    Allows rollback if performance degrades
    """
    
    def __init__(self, model, checkpoint_interval=1000):
        self.model = model
        self.checkpoint_interval = checkpoint_interval
        
        self.checkpoints = []
        self.performance_history = []
        self.update_count = 0
    
    def partial_fit(self, X, y):
        """Update with checkpointing"""
        # Make prediction
        y_pred = self.model.predict_one(X)
        
        # Track performance
        correct = 1 if y_pred == y else 0
        self.performance_history.append(correct)
        
        # Update model
        self.model.learn_one(X, y)
        self.update_count += 1
        
        # Periodic checkpoint
        if self.update_count % self.checkpoint_interval == 0:
            self._create_checkpoint()
        
        return y_pred
    
    def _create_checkpoint(self):
        """Save model checkpoint"""
        import copy
        
        # Calculate recent performance
        recent_perf = np.mean(self.performance_history[-self.checkpoint_interval:])
        
        checkpoint = {
            'model': copy.deepcopy(self.model),
            'update_count': self.update_count,
            'performance': recent_perf
        }
        
        self.checkpoints.append(checkpoint)
        
        print(f"💾 Checkpoint {len(self.checkpoints)}: "
              f"Performance = {recent_perf:.3f}")
        
        # Check for degradation
        if len(self.checkpoints) > 1:
            prev_perf = self.checkpoints[-2]['performance']
            if recent_perf < prev_perf - 0.1:  # Significant drop
                print("⚠️ Performance dropped, considering rollback...")
                self._maybe_rollback()
    
    def _maybe_rollback(self):
        """Rollback to previous checkpoint if needed"""
        if len(self.checkpoints) < 2:
            return
        
        current_perf = self.checkpoints[-1]['performance']
        best_checkpoint = max(self.checkpoints[:-1], 
                             key=lambda x: x['performance'])
        
        if best_checkpoint['performance'] > current_perf + 0.05:
            print(f"🔄 Rolling back to checkpoint with "
                  f"performance {best_checkpoint['performance']:.3f}")
            self.model = best_checkpoint['model']
```

---

## Streaming Infrastructure

### Kafka Integration

```python
from kafka import KafkaConsumer, KafkaProducer
import json

class OnlineLearningService:
    """
    Online learning service with Kafka
    
    Consumes training data, produces predictions
    """
    
    def __init__(self, model, kafka_bootstrap_servers):
        self.model = model
        
        # Kafka consumer for training data
        self.consumer = KafkaConsumer(
            'training_data',
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        # Kafka producer for predictions
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda m: json.dumps(m).encode('utf-8')
        )
        
        self.update_count = 0
    
    def run(self):
        """Main service loop"""
        print("🚀 Starting online learning service...")
        
        for message in self.consumer:
            # Extract training example
            data = message.value
            X = data['features']
            y = data['label']
            
            # Make prediction before update
            y_pred = self.model.predict_one(X)
            
            # Update model
            self.model.learn_one(X, y)
            self.update_count += 1
            
            # Publish prediction
            result = {
                'id': data['id'],
                'prediction': y_pred,
                'model_version': self.update_count
            }
            self.producer.send('predictions', value=result)
            
            if self.update_count % 100 == 0:
                print(f"Processed {self.update_count} updates")

# Usage
model = linear_model.LogisticRegression()
service = OnlineLearningService(model, ['localhost:9092'])
service.run()
```

---

## Connection to Binary Search (Day 9 DSA)

Online learning uses binary search patterns for hyperparameter optimization:

```python
class OnlineLearningRateOptimizer:
    """
    Optimize learning rate using binary search
    
    Similar to Day 9 DSA: binary search on continuous space
    """
    
    def __init__(self, model, validation_stream):
        self.model = model
        self.validation_stream = validation_stream
    
    def find_optimal_lr(self, min_lr=1e-5, max_lr=1.0, iterations=10):
        """
        Binary search for optimal learning rate
        
        Args:
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            iterations: Number of binary search iterations
        
        Returns:
            Optimal learning rate
        """
        best_lr = min_lr
        best_score = 0
        
        left, right = min_lr, max_lr
        
        for iteration in range(iterations):
            # Try middle point
            mid_lr = (left + right) / 2
            
            # Evaluate this learning rate
            score = self._evaluate_learning_rate(mid_lr)
            
            print(f"Iteration {iteration}: lr={mid_lr:.6f}, score={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_lr = mid_lr
            
            # Adjust search space (simplified heuristic)
            # In practice, use more sophisticated methods
            if score > 0.8:
                # Good performance, try higher learning rate
                left = mid_lr
            else:
                # Poor performance, try lower learning rate
                right = mid_lr
        
        return best_lr, best_score
    
    def _evaluate_learning_rate(self, learning_rate):
        """Evaluate model with given learning rate"""
        import copy
        from itertools import islice
        
        # Copy and set optimizer lr if available
        temp_model = copy.deepcopy(self.model)
        # Attempt to set lr on inner estimator if present
        try:
            temp_model['LogisticRegression'].optimizer = optim.SGD(lr=learning_rate)
        except Exception:
            pass
        
        # Train on sample of validation stream
        correct = 0
        total = 0
        
        for X, y in islice(self.validation_stream, 100):
            y_pred = temp_model.predict_one(X)
            correct += (y_pred == y)
            temp_model.learn_one(X, y)
            total += 1
        
        return correct / total if total > 0 else 0

# Usage
optimizer = OnlineLearningRateOptimizer(model, validation_data)
optimal_lr, score = optimizer.find_optimal_lr()
print(f"Optimal learning rate: {optimal_lr:.6f}")
```

---

## Monitoring & Evaluation

### Real-time Metrics Dashboard

```python
class OnlineLearningMonitor:
    """
    Monitor online learning system health
    
    Track multiple metrics in real-time
    """
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        
        # Metric windows
        self.recent_predictions = deque(maxlen=window_size)
        self.recent_losses = deque(maxlen=window_size)
        self.recent_latencies = deque(maxlen=window_size)
        
        # Counters
        self.total_updates = 0
        self.start_time = time.time()
    
    def log_update(self, y_true, y_pred, loss, latency_ms):
        """Log single update"""
        correct = 1 if y_true == y_pred else 0
        self.recent_predictions.append(correct)
        self.recent_losses.append(loss)
        self.recent_latencies.append(latency_ms)
        
        self.total_updates += 1
    
    def get_metrics(self):
        """Get current metrics"""
        if not self.recent_predictions:
            return {}
        
        uptime_hours = (time.time() - self.start_time) / 3600
        
        return {
            'accuracy': np.mean(self.recent_predictions),
            'avg_loss': np.mean(self.recent_losses),
            'p50_latency': np.percentile(self.recent_latencies, 50),
            'p95_latency': np.percentile(self.recent_latencies, 95),
            'p99_latency': np.percentile(self.recent_latencies, 99),
            'updates_per_second': self.total_updates / (uptime_hours * 3600),
            'total_updates': self.total_updates,
            'uptime_hours': uptime_hours
        }
    
    def print_dashboard(self):
        """Print real-time dashboard"""
        metrics = self.get_metrics()
        
        print("\n" + "="*50)
        print("Online Learning Dashboard")
        print("="*50)
        print(f"Total Updates:      {metrics['total_updates']:,}")
        print(f"Uptime:            {metrics['uptime_hours']:.2f} hours")
        print(f"Updates/sec:       {metrics['updates_per_second']:.1f}")
        print(f"Accuracy:          {metrics['accuracy']:.3f}")
        print(f"Avg Loss:          {metrics['avg_loss']:.4f}")
        print(f"P50 Latency:       {metrics['p50_latency']:.2f}ms")
        print(f"P95 Latency:       {metrics['p95_latency']:.2f}ms")
        print(f"P99 Latency:       {metrics['p99_latency']:.2f}ms")
        print("="*50 + "\n")

# Usage
monitor = OnlineLearningMonitor()

for i in range(10000):
    start = time.time()
    
    # Update model
    y_pred = model.partial_fit(X, y)
    loss = compute_loss(y, y_pred)
    
    latency = (time.time() - start) * 1000
    
    # Log metrics
    monitor.log_update(y, y_pred, loss, latency)
    
    # Print dashboard every 1000 updates
    if i % 1000 == 0:
        monitor.print_dashboard()
```

---

## Advanced Techniques

### 1. Contextual Bandits

```python
import numpy as np

class ContextualBandit:
    """
    Contextual multi-armed bandit for online learning
    
    Learns which model to use based on context
    """
    
    def __init__(self, n_arms, n_features, epsilon=0.1):
        """
        Args:
            n_arms: Number of models/actions
            n_features: Number of context features
            epsilon: Exploration rate
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.epsilon = epsilon
        
        # Linear models for each arm
        self.weights = [np.zeros(n_features) for _ in range(n_arms)]
        self.counts = np.zeros(n_arms)
        self.rewards = [[] for _ in range(n_arms)]
    
    def select_arm(self, context):
        """
        Select arm (model) based on context
        
        Uses epsilon-greedy with linear reward prediction
        
        Args:
            context: Feature vector [n_features]
        
        Returns:
            Selected arm index
        """
        if np.random.random() < self.epsilon:
            # Explore: random arm
            return np.random.randint(self.n_arms)
        
        # Exploit: arm with highest predicted reward
        predicted_rewards = [
            np.dot(context, weights) 
            for weights in self.weights
        ]
        return np.argmax(predicted_rewards)
    
    def update(self, arm, context, reward):
        """
        Update arm's model with observed reward
        
        Uses online gradient descent
        """
        self.counts[arm] += 1
        self.rewards[arm].append(reward)
        
        # Online gradient descent update
        prediction = np.dot(context, self.weights[arm])
        error = reward - prediction
        
        # Update weights: w = w + alpha * error * context
        learning_rate = 1.0 / (1.0 + self.counts[arm])
        self.weights[arm] += learning_rate * error * context
    
    def get_arm_stats(self):
        """Get statistics for each arm"""
        return {
            f'arm_{i}': {
                'count': int(self.counts[i]),
                'avg_reward': np.mean(self.rewards[i]) if self.rewards[i] else 0
            }
            for i in range(self.n_arms)
        }

# Usage: Choose between models based on user context
bandit = ContextualBandit(n_arms=3, n_features=5)

# Simulate online serving
for iteration in range(1000):
    # Get user context
    context = np.random.randn(5)  # User features
    
    # Select model
    model_idx = bandit.select_arm(context)
    
    # Get reward (e.g., click-through rate)
    reward = simulate_reward(model_idx, context)
    
    # Update
    bandit.update(model_idx, context, reward)

print(bandit.get_arm_stats())
```

### 2. Bayesian Online Learning

```python
class BayesianOnlineLearner:
    """
    Bayesian approach to online learning
    
    Maintains uncertainty estimates
    """
    
    def __init__(self, n_features, alpha=1.0, beta=1.0):
        """
        Args:
            n_features: Number of features
            alpha: Prior precision (inverse variance)
            beta: Noise precision
        """
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        
        # Posterior parameters
        self.mean = np.zeros(n_features)
        self.precision = alpha * np.eye(n_features)
        
        self.update_count = 0
    
    def predict(self, X):
        """
        Predict with uncertainty
        
        Returns: (mean, variance)
        """
        mean = np.dot(X, self.mean)
        
        # Predictive variance
        covariance = np.linalg.inv(self.precision)
        variance = 1.0 / self.beta + np.dot(X, np.dot(covariance, X.T))
        
        return mean, variance
    
    def update(self, X, y):
        """
        Bayesian online update
        
        Updates posterior distribution
        """
        # Update precision matrix
        self.precision += self.beta * np.outer(X, X)
        
        # Update mean
        covariance = np.linalg.inv(self.precision)
        self.mean = np.dot(
            covariance,
            self.alpha * self.mean + self.beta * y * X
        )
        
        self.update_count += 1
    
    def get_confidence_interval(self, X, confidence=0.95):
        """
        Get prediction confidence interval
        
        Useful for uncertainty-based exploration
        """
        mean, variance = self.predict(X)
        std = np.sqrt(variance)
        
        # Z-score for confidence level
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        return (mean - z * std, mean + z * std)

# Usage
learner = BayesianOnlineLearner(n_features=10)

for X, y in data_stream:
    # Predict with uncertainty
    mean, variance = learner.predict(X)
    
    print(f"Prediction: {mean:.3f} ± {np.sqrt(variance):.3f}")
    
    # Update
    learner.update(X, y)
```

### 3. Follow-the-Regularized-Leader (FTRL)

```python
class FTRLOptimizer:
    """
    FTRL-Proximal optimizer for online learning
    
    Popular for large-scale online learning (used by Google)
    """
    
    def __init__(self, n_features, alpha=0.1, beta=1.0, lambda1=0.0, lambda2=1.0):
        """
        Args:
            alpha: Learning rate
            beta: Smoothing parameter
            lambda1: L1 regularization
            lambda2: L2 regularization
        """
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # FTRL parameters
        self.z = np.zeros(n_features)  # Accumulated gradient
        self.n = np.zeros(n_features)  # Accumulated squared gradient
        
        self.weights = np.zeros(n_features)
    
    def predict(self, x):
        """Make prediction"""
        return 1.0 / (1.0 + np.exp(-np.dot(x, self.weights)))
    
    def update(self, x, y):
        """
        FTRL update step
        
        More stable than standard online gradient descent
        """
        # Make prediction
        p = self.predict(x)
        
        # Compute gradient
        g = (p - y) * x
        
        # Update accumulated gradients
        sigma = (np.sqrt(self.n + g * g) - np.sqrt(self.n)) / self.alpha
        self.z += g - sigma * self.weights
        self.n += g * g
        
        # Update weights with proximal step
        for i in range(len(self.weights)):
            if abs(self.z[i]) <= self.lambda1:
                self.weights[i] = 0
            else:
                sign = 1 if self.z[i] > 0 else -1
                self.weights[i] = -(self.z[i] - sign * self.lambda1) / (
                    (self.beta + np.sqrt(self.n[i])) / self.alpha + self.lambda2
                )
    
    def get_sparsity(self):
        """Get weight sparsity (fraction of zero weights)"""
        return np.mean(self.weights == 0)

# Usage
optimizer = FTRLOptimizer(n_features=100, lambda1=1.0)  # L1 for sparsity

for x, y in data_stream:
    pred = optimizer.predict(x)
    optimizer.update(x, y)

print(f"Model sparsity: {optimizer.get_sparsity():.1%}")
```

---

## Real-World Case Studies

### Case Study 1: Netflix Recommendation

```python
class NetflixOnlineLearning:
    """
    Simplified Netflix online learning for recommendations
    
    Updates user preferences in real-time based on viewing behavior
    """
    
    def __init__(self, n_users, n_items, n_factors=50):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        
        # Matrix factorization embeddings
        self.user_factors = np.random.randn(n_users, n_factors) * 0.01
        self.item_factors = np.random.randn(n_items, n_factors) * 0.01
        
        # Learning rates
        self.lr = 0.01
        self.reg = 0.01
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        return np.dot(self.user_factors[user_id], self.item_factors[item_id])
    
    def update_from_interaction(self, user_id, item_id, rating):
        """
        Update embeddings from single interaction
        
        Online matrix factorization
        """
        # Predict current rating
        pred = self.predict(user_id, item_id)
        error = rating - pred
        
        # Gradient updates
        user_grad = error * self.item_factors[item_id] - self.reg * self.user_factors[user_id]
        item_grad = error * self.user_factors[user_id] - self.reg * self.item_factors[item_id]
        
        # Update embeddings
        self.user_factors[user_id] += self.lr * user_grad
        self.item_factors[item_id] += self.lr * item_grad
    
    def recommend(self, user_id, n=10):
        """Get top-N recommendations for user"""
        scores = np.dot(self.item_factors, self.user_factors[user_id])
        top_items = np.argsort(-scores)[:n]
        return top_items

# Simulate Netflix streaming
recommender = NetflixOnlineLearning(n_users=1000000, n_items=10000)

# User watches a movie and rates it
recommender.update_from_interaction(user_id=12345, item_id=567, rating=4.5)

# Get real-time recommendations
recommendations = recommender.recommend(user_id=12345, n=10)
```

### Case Study 2: Twitter Timeline Ranking

```python
class TwitterTimelineRanker:
    """
    Online learning for Twitter timeline ranking
    
    Predicts engagement (clicks, likes, retweets) in real-time
    """
    
    def __init__(self):
        # Multiple models for different engagement types
        from sklearn.linear_model import SGDClassifier
        self.click_model = SGDClassifier(
            loss='log_loss',
            learning_rate='optimal',
            alpha=0.0001
        )
        self.like_model = SGDClassifier(
            loss='log_loss',
            learning_rate='optimal',
            alpha=0.0001
        )
        
        self.update_buffer = deque(maxlen=100)
        self.is_initialized = False
    
    def extract_features(self, tweet, user):
        """
        Extract features for ranking
        
        Features:
        - Tweet features: author followers, recency, media type
        - User features: interests, engagement history
        - Interaction features: author-user affinity
        """
        return {
            'author_followers': tweet['author_followers'],
            'tweet_age_minutes': tweet['age_minutes'],
            'has_media': int(tweet['has_media']),
            'user_interest_match': user['interest_similarity'],
            'author_user_affinity': tweet['author_affinity'],
            'tweet_length': len(tweet['text']),
        }
    
    def score_tweet(self, tweet, user):
        """
        Score tweet for ranking
        
        Combines click and like predictions
        """
        features = self.extract_features(tweet, user)
        
        if not self.is_initialized:
            return 0.5  # Random score until initialized
        
        # Predict engagement probabilities
        click_prob = self.click_model.predict_proba([features])[0][1]
        like_prob = self.like_model.predict_proba([features])[0][1]
        
        # Weighted combination
        score = 0.6 * click_prob + 0.4 * like_prob
        
        return score
    
    def update_from_feedback(self, tweet, user, clicked, liked):
        """
        Update models from user feedback
        
        Called when user interacts (or doesn't) with tweet
        """
        features = self.extract_features(tweet, user)
        
        # Add to buffer
        self.update_buffer.append((features, clicked, liked))
        
        # Batch update when buffer is full
        if len(self.update_buffer) >= 100:
            self._batch_update()
    
    def _batch_update(self):
        """Batch update from buffer"""
        features_list = [item[0] for item in self.update_buffer]
        click_labels = [item[1] for item in self.update_buffer]
        like_labels = [item[2] for item in self.update_buffer]
        
        # Partial fit (online learning)
        import numpy as np
        X = self._features_to_matrix(features_list)
        y_click = np.array(click_labels)
        y_like = np.array(like_labels)
        
        self.click_model.partial_fit(X, y_click, classes=np.array([0, 1]))
        self.like_model.partial_fit(X, y_like, classes=np.array([0, 1]))
        
        self.is_initialized = True
        self.update_buffer.clear()
    
    def rank_timeline(self, tweets, user):
        """Rank tweets for user's timeline"""
        scored_tweets = []
        for tweet in tweets:
            score = self.score_tweet(tweet, user)
            scored_tweets.append((tweet, score))
        
        # Sort by score (descending)
        ranked = sorted(scored_tweets, key=lambda x: x[1], reverse=True)
        
        return [tweet for tweet, score in ranked]

# Usage
ranker = TwitterTimelineRanker()

# User views timeline
timeline_tweets = fetch_candidate_tweets(user_id)
ranked_timeline = ranker.rank_timeline(timeline_tweets, user)

# User interacts with tweets
for tweet in ranked_timeline[:10]:
    clicked, liked = show_tweet_to_user(tweet)
    ranker.update_from_feedback(tweet, user, clicked, liked)
```

### Case Study 3: Fraud Detection

```python
class OnlineFraudDetector:
    """
    Online learning for fraud detection
    
    Adapts to evolving fraud patterns in real-time
    """
    
    def __init__(self, window_size=10000):
        self.model = linear_model.SGDClassifier(
            loss='log',
            penalty='l1',  # L1 for feature selection
            alpha=0.0001,
            learning_rate='adaptive',
            eta0=0.01
        )
        
        self.window_size = window_size
        self.recent_transactions = deque(maxlen=window_size)
        
        # Fraud pattern tracking
        self.fraud_patterns = {}
        self.is_initialized = False
    
    def extract_features(self, transaction):
        """
        Extract fraud detection features
        
        Features:
        - Transaction amount, location, time
        - User behavior patterns
        - Merchant risk score
        """
        return {
            'amount': transaction['amount'],
            'amount_z_score': self._get_amount_zscore(transaction),
            'hour_of_day': transaction['timestamp'].hour,
            'is_weekend': int(transaction['timestamp'].weekday() >= 5),
            'distance_from_home': transaction['distance_km'],
            'merchant_risk_score': self._get_merchant_risk(transaction['merchant']),
            'user_velocity': self._get_user_velocity(transaction['user_id']),
        }
    
    def predict(self, transaction):
        """
        Predict if transaction is fraudulent
        
        Returns: (is_fraud, fraud_probability)
        """
        features = self.extract_features(transaction)
        
        if not self.is_initialized:
            # Cold start: use rule-based system
            return self._rule_based_prediction(transaction)
        
        # ML prediction
        features_array = np.array(list(features.values())).reshape(1, -1)
        fraud_prob = self.model.predict_proba(features_array)[0][1]
        
        # Threshold
        is_fraud = fraud_prob > 0.9  # High threshold to minimize false positives
        
        return is_fraud, fraud_prob
    
    def update(self, transaction, is_fraud):
        """
        Update model with labeled transaction
        
        Label comes from:
        - User confirmation
        - Fraud analyst review
        - Chargeback
        """
        features = self.extract_features(transaction)
        features_array = np.array(list(features.values())).reshape(1, -1)
        
        # Update model
        if self.is_initialized:
            self.model.partial_fit(features_array, [is_fraud])
        else:
            # Initialize on first labeled sample
            self.model.fit(features_array, [is_fraud])
            self.is_initialized = True
        
        # Track fraud patterns
        if is_fraud:
            self._update_fraud_patterns(transaction)
        
        # Add to recent window
        self.recent_transactions.append((transaction, is_fraud))
    
    def _get_amount_zscore(self, transaction):
        """Z-score of amount compared to user's history"""
        if not self.recent_transactions:
            return 0.0
        
        user_txns = [
            t['amount'] for t, _ in self.recent_transactions
            if t['user_id'] == transaction['user_id']
        ]
        
        if len(user_txns) < 2:
            return 0.0
        
        mean = np.mean(user_txns)
        std = np.std(user_txns)
        
        if std == 0:
            return 0.0
        
        return (transaction['amount'] - mean) / std
    
    def _update_fraud_patterns(self, transaction):
        """Track emerging fraud patterns"""
        pattern_key = (transaction['merchant'], transaction['location'])
        
        if pattern_key not in self.fraud_patterns:
            self.fraud_patterns[pattern_key] = {
                'count': 0,
                'first_seen': transaction['timestamp']
            }
        
        self.fraud_patterns[pattern_key]['count'] += 1

# Usage
detector = OnlineFraudDetector()

# Real-time transaction processing
for transaction in transaction_stream:
    # Predict
    is_fraud, prob = detector.predict(transaction)
    
    if is_fraud:
        # Block transaction
        block_transaction(transaction)
        
        # Get analyst review
        analyst_label = request_analyst_review(transaction)
        detector.update(transaction, analyst_label)
    else:
        # Allow transaction
        allow_transaction(transaction)
        
        # Update with feedback (if available)
        if has_feedback(transaction):
            label = get_feedback(transaction)
            detector.update(transaction, label)
```

---

## Performance Optimization

### GPU Acceleration

```python
import torch
import torch.nn as nn

class GPUAcceleratedOnlineLearner:
    """
    GPU-accelerated online learning
    
    Uses PyTorch for fast batch updates
    """
    
    def __init__(self, input_dim, hidden_dim=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Neural network model
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
        # Batch buffer for GPU efficiency
        self.batch_buffer = []
        self.batch_size = 128
    
    def add_example(self, x, y):
        """
        Add example to batch buffer
        
        Triggers GPU update when buffer is full
        """
        self.batch_buffer.append((x, y))
        
        if len(self.batch_buffer) >= self.batch_size:
            self._update_batch()
    
    def _update_batch(self):
        """Update model with GPU batch processing"""
        if not self.batch_buffer:
            return
        
        # Prepare batch tensors
        X_batch = torch.tensor(
            [x for x, y in self.batch_buffer],
            dtype=torch.float32,
            device=self.device
        )
        y_batch = torch.tensor(
            [[y] for x, y in self.batch_buffer],
            dtype=torch.float32,
            device=self.device
        )
        
        # Forward pass
        self.model.train()
        outputs = self.model(X_batch)
        loss = self.criterion(outputs, y_batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear buffer
        self.batch_buffer.clear()
        
        return loss.item()
    
    def predict(self, x):
        """Fast GPU prediction"""
        self.model.eval()
        
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            output = self.model(x_tensor.unsqueeze(0))
        
        return output.item()

# Usage
learner = GPUAcceleratedOnlineLearner(input_dim=100)

# Process stream with GPU acceleration
for x, y in data_stream:
    pred = learner.predict(x)
    learner.add_example(x, y)
```

---

## Key Takeaways

✅ **Continuous adaptation** - Learn from streaming data without full retraining  
✅ **Handle concept drift** - Detect and adapt to changing distributions  
✅ **Memory efficient** - Don't need to store all historical data  
✅ **Fast updates** - Incorporate new information in real-time  
✅ **Stability vs plasticity** - Balance learning new patterns vs retaining knowledge  
✅ **Production patterns** - Checkpointing, ensembles, warm starts  
✅ **Binary search optimization** - Find optimal hyperparameters efficiently  

---

**Originally published at:** [arunbaby.com/ml-system-design/0009-online-learning-systems](https://www.arunbaby.com/ml-system-design/0009-online-learning-systems/)

*If you found this helpful, consider sharing it with others who might benefit.*

