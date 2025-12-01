---
title: "Neural Architecture Search"
day: 21
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - neural-architecture-search
  - automl
  - nas
  - reinforcement-learning
  - optimization
  - model-design
subdomain: "AutoML & Model Design"
tech_stack: [Python, PyTorch, TensorFlow, Ray, Optuna, NNI, ENAS, DARTS]
scale: "1000s of architectures evaluated, 100+ GPU days, multi-objective optimization"
companies: [Google, Meta, OpenAI, DeepMind, Microsoft, NVIDIA]
related_dsa_day: 21
related_speech_day: 21
---

**Design neural architecture search systems that automatically discover optimal model architectures using dynamic programming and path optimization—the same principles from grid path counting scaled to exponential search spaces.**

## Problem Statement

Design a **Neural Architecture Search (NAS) System** that:

1. **Automatically discovers** neural network architectures that outperform hand-designed models
2. **Searches efficiently** through exponentially large search spaces
3. **Optimizes multiple objectives** (accuracy, latency, memory, FLOPS)
4. **Scales to production** (finds models deployable on mobile/edge devices)
5. **Supports different domains** (vision, NLP, speech, multi-modal)

### Functional Requirements

1. **Search space definition:**
   - Define architecture space (layers, operations, connections)
   - Support modular search (cells, blocks, stages)
   - Enable constrained search (max latency, max params)

2. **Search strategy:**
   - Reinforcement learning (controller RNN)
   - Evolutionary algorithms (mutation, crossover)
   - Gradient-based (DARTS, differentiable NAS)
   - Random search (baseline)
   - Bayesian optimization

3. **Performance estimation:**
   - Train architectures to evaluate quality
   - Early stopping for bad candidates
   - Weight sharing / one-shot models (ENAS, DARTS)
   - Performance predictors (surrogate models)

4. **Multi-objective optimization:**
   - Accuracy vs latency
   - Accuracy vs model size
   - Accuracy vs FLOPS
   - Pareto frontier identification

5. **Distributed search:**
   - Parallel architecture evaluation
   - Distributed training of candidates
   - Efficient resource allocation

6. **Transfer and reuse:**
   - Transfer architectures across tasks
   - Re-use components from previous searches
   - Meta-learning for search initialization

### Non-Functional Requirements

1. **Efficiency:** Find good architecture in <100 GPU days (vs manual design months)
2. **Quality:** Discovered models competitive with hand-designed ones
3. **Generalizability:** Architectures transfer across datasets
4. **Interpretability:** Understand why architecture works
5. **Reproducibility:** Same search produces same results

## Understanding the Requirements

### Why NAS?

**Manual architecture design is:**
- Time-consuming (months of expert effort)
- Limited by human intuition and expertise
- Hard to optimize for specific constraints (mobile latency, memory)
- Difficult to explore unconventional designs

**NAS automates:**
- Architecture discovery
- Multi-objective optimization
- Hardware-aware design
- Cross-domain transfer

### Scale of the Problem

**Search space size:**
- A simple NAS space with 6 layers, 5 operations per layer: 5^6 = 15,625 architectures
- NASNet search space: ~10^18 possible architectures
- Without smart search, infeasible to evaluate all

**Computational cost:**
- Training one model: 1-10 GPU days
- Naive search (10K architectures): 10K-100K GPU days
- Smart search (NAS): 100-1000 GPU days
- **Goal:** Reduce by 10-100x through efficient search

### The Path Optimization Connection

Just like **Unique Paths** counts paths through a grid using DP:

| Unique Paths | Neural Architecture Search | Speech Arch Search |
|--------------|---------------------------|-------------------|
| m×n grid | Layer×operation space | Encoder×decoder configs |
| Count all paths | Count/evaluate architectures | Evaluate speech models |
| DP optimization | DP/RL search | DP search |
| O(m×n) vs O(2^(m+n)) | Smart search vs brute force | Efficient vs exhaustive |
| Path reconstruction | Architecture construction | Model construction |

Both use **dynamic programming / smart search** to navigate exponentially large spaces efficiently.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Neural Architecture Search System                 │
└─────────────────────────────────────────────────────────────────┘

                       Search Controller
            ┌──────────────────────────────────┐
            │  Strategy: RL / EA / DARTS       │
            │  - Propose architectures         │
            │  - Update based on performance   │
            └──────────────┬───────────────────┘
                           │
                           ↓
                  ┌────────────────┐
                  │  Search Space  │
                  │  - Layers      │
                  │  - Operations  │
                  │  - Connections │
                  └────────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐ ┌──────▼──────┐ ┌────────▼────────┐
│  Architecture  │ │ Performance │ │  Multi-obj      │
│  Evaluator     │ │ Predictor   │ │  Optimizer      │
│                │ │             │ │                 │
│ - Train model  │ │ - Surrogate │ │ - Pareto        │
│ - Measure acc  │ │ - Skip bad  │ │ - Constraints   │
│ - Measure lat  │ │   candidates│ │ - Trade-offs    │
└───────┬────────┘ └──────┬──────┘ └────────┬────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                  ┌────────▼────────┐
                  │  Distributed    │
                  │  Training       │
                  │  - Worker pool  │
                  │  - GPU cluster  │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  Results Store  │
                  │  - Architectures│
                  │  - Metrics      │
                  │  - Models       │
                  └─────────────────┘
```

### Key Components

1. **Search Controller:** Proposes new architectures to try
2. **Search Space:** Defines valid architecture configurations
3. **Architecture Evaluator:** Trains and evaluates architectures
4. **Performance Predictor:** Estimates performance without full training
5. **Multi-objective Optimizer:** Balances accuracy, latency, size
6. **Distributed Training:** Parallel evaluation of architectures
7. **Results Store:** Tracks all evaluated architectures

## Component Deep-Dives

### 1. Search Space Definition

Define what architectures are possible:

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Operation:
    """A single operation in the search space."""
    name: str
    params: Dict

@dataclass
class SearchSpace:
    """
    NAS search space definition.
    
    Similar to Unique Paths grid:
    - Grid dimensions → num_layers, ops_per_layer
    - Paths through grid → architectures through search space
    """
    num_layers: int
    operations: List[Operation]
    connections: str  # "sequential", "skip", "dense"
    
    def count_architectures(self) -> int:
        """
        Count total possible architectures.
        
        Like counting paths in Unique Paths:
        - If sequential: ops_per_layer ^ num_layers
        - If with skip connections: much larger
        """
        if self.connections == "sequential":
            return len(self.operations) ** self.num_layers
        else:
            # With skip connections, combinatorially larger
            return -1  # Too many to count exactly


# Example search space
MOBILENET_SEARCH_SPACE = SearchSpace(
    num_layers=20,
    operations=[
        Operation("conv3x3", {"kernel_size": 3, "stride": 1}),
        Operation("conv5x5", {"kernel_size": 5, "stride": 1}),
        Operation("depthwise_conv3x3", {"kernel_size": 3}),
        Operation("depthwise_conv5x5", {"kernel_size": 5}),
        Operation("maxpool3x3", {"kernel_size": 3, "stride": 1}),
        Operation("skip", {}),
    ],
    connections="skip"
)


def encode_architecture(arch_ops: List[str], search_space: SearchSpace) -> str:
    """
    Encode architecture as string.
    
    Args:
        arch_ops: List of operation names per layer
        
    Returns:
        String encoding (for hashing/caching)
    """
    return "-".join(arch_ops)


def decode_architecture(arch_string: str) -> List[str]:
    """Decode architecture string to operation list."""
    return arch_string.split("-")
```

### 2. Search Strategy - Reinforcement Learning

Use RL controller to generate architectures:

```python
import torch
import torch.nn as nn

class NASController(nn.Module):
    """
    RNN controller that generates architectures.
    
    Similar to DP in Unique Paths:
    - Build architecture layer-by-layer
    - Use previous decisions to inform next
    - Optimize for high reward (accuracy)
    """
    
    def __init__(
        self,
        num_layers: int,
        num_operations: int,
        hidden_size: int = 100
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_operations = num_operations
        
        # RNN to track state across layers
        self.rnn = nn.LSTM(
            input_size=num_operations,
            hidden_size=hidden_size,
            num_layers=1
        )
        
        # Output layer: predict operation for next layer
        self.fc = nn.Linear(hidden_size, num_operations)
    
    def forward(self):
        """
        Generate an architecture.
        
        Returns:
            architecture: List of operation indices
            log_probs: Log probabilities for REINFORCE
        """
        architecture = []
        log_probs = []
        
        # Initial input
        inputs = torch.zeros(1, 1, self.num_operations)
        hidden = None
        
        # Generate layer-by-layer (like DP building solution)
        for layer in range(self.num_layers):
            # RNN step
            output, hidden = self.rnn(inputs, hidden)
            
            # Predict operation for this layer
            logits = self.fc(output.squeeze(0))
            probs = torch.softmax(logits, dim=-1)
            
            # Sample operation
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            architecture.append(action.item())
            log_probs.append(dist.log_prob(action))
            
            # Next input is one-hot of chosen operation
            inputs = torch.zeros(1, 1, self.num_operations)
            inputs[0, 0, action] = 1.0
        
        return architecture, log_probs
    
    def update(self, log_probs: List[torch.Tensor], reward: float, optimizer):
        """
        Update controller using REINFORCE.
        
        Args:
            log_probs: Log probabilities of sampled actions
            reward: Accuracy of generated architecture
            optimizer: Controller optimizer
        """
        # REINFORCE loss: -sum(log_prob * reward)
        policy_loss = []
        for log_prob in log_probs:
            policy_loss.append(-log_prob * reward)
        
        loss = torch.stack(policy_loss).sum()
        
        # Update controller
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Training loop
def train_nas_controller(
    controller: NASController,
    search_space: SearchSpace,
    num_iterations: int = 1000
):
    """
    Train NAS controller to generate good architectures.
    """
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.001)
    
    for iteration in range(num_iterations):
        # Generate architecture
        arch, log_probs = controller()
        
        # Evaluate architecture (train small model)
        reward = evaluate_architecture(arch, search_space)
        
        # Update controller
        controller.update(log_probs, reward, optimizer)
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Best reward = {reward:.3f}")
```

### 3. Search Strategy - Differentiable NAS (DARTS)

DARTS makes architecture search differentiable:

```python
class DARTSSearchSpace(nn.Module):
    """
    Differentiable architecture search.
    
    Key idea: Instead of discrete choice, use weighted combination.
    Learn weights (architecture parameters) via gradient descent.
    """
    
    def __init__(self, num_layers: int, operations: List[nn.Module]):
        super().__init__()
        
        self.num_layers = num_layers
        self.operations = nn.ModuleList(operations)
        
        # Architecture parameters (learnable weights for each operation)
        self.alpha = nn.Parameter(
            torch.randn(num_layers, len(operations))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weighted operations.
        
        Each layer computes weighted sum of all operations.
        """
        for layer in range(self.num_layers):
            # Get architecture weights for this layer
            weights = torch.softmax(self.alpha[layer], dim=0)
            
            # Compute weighted sum of operations
            layer_output = sum(
                w * op(x)
                for w, op in zip(weights, self.operations)
            )
            
            x = layer_output
        
        return x
    
    def get_best_architecture(self) -> List[int]:
        """
        Extract discrete architecture from learned weights.
        
        Choose operation with highest weight per layer.
        """
        architecture = []
        
        for layer in range(self.num_layers):
            weights = torch.softmax(self.alpha[layer], dim=0)
            best_op = torch.argmax(weights).item()
            architecture.append(best_op)
        
        return architecture


# DARTS training (bi-level optimization)
def train_darts(search_space: DARTSSearchSpace, train_data, val_data, epochs: int = 50):
    """
    Train DARTS to find optimal architecture.
    
    Bi-level optimization:
    - Inner loop: optimize model weights
    - Outer loop: optimize architecture parameters
    """
    # Model weights optimizer
    model_optimizer = torch.optim.SGD(
        search_space.parameters(),
        lr=0.025,
        momentum=0.9
    )
    
    # Architecture parameters optimizer
    arch_optimizer = torch.optim.Adam(
        [search_space.alpha],
        lr=0.001
    )
    
    for epoch in range(epochs):
        # Update model weights on train data
        for batch in train_data:
            model_optimizer.zero_grad()
            loss = compute_loss(search_space(batch['x']), batch['y'])
            loss.backward()
            model_optimizer.step()
        
        # Update architecture parameters on val data
        for batch in val_data:
            arch_optimizer.zero_grad()
            loss = compute_loss(search_space(batch['x']), batch['y'])
            loss.backward()
            arch_optimizer.step()
    
    # Extract final architecture
    best_arch = search_space.get_best_architecture()
    return best_arch
```

### 4. Performance Estimation Strategies

**Problem:** Training every architecture fully is too expensive.

**Solutions:**

**a) Early stopping:**

```python
def evaluate_with_early_stopping(
    arch: List[int],
    train_data,
    val_data,
    max_epochs: int = 50,
    patience: int = 5
):
    """
    Train architecture with early stopping.
    
    Stop if validation accuracy doesn't improve for `patience` epochs.
    """
    model = build_model_from_arch(arch)
    optimizer = torch.optim.Adam(model.parameters())
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        # Train
        train_one_epoch(model, train_data, optimizer)
        
        # Validate
        val_acc = evaluate(model, val_data)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stop
        if epochs_without_improvement >= patience:
            break
    
    return best_val_acc
```

**b) Weight sharing (ENAS):**

```python
class SuperNet(nn.Module):
    """
    Super-network containing all possible operations.
    
    Different architectures share weights.
    Train once, evaluate many architectures quickly.
    """
    
    def __init__(self, search_space: SearchSpace):
        super().__init__()
        
        # Create all operations (shared across architectures)
        self.ops = nn.ModuleList([
            create_operation(op)
            for op in search_space.operations
        ])
    
    def forward(self, x: torch.Tensor, architecture: List[int]) -> torch.Tensor:
        """
        Forward pass for specific architecture.
        
        Args:
            x: Input
            architecture: List of operation indices per layer
        """
        for layer, op_idx in enumerate(architecture):
            x = self.ops[op_idx](x)
        
        return x
    
    def evaluate_architecture(self, arch: List[int], val_data) -> float:
        """
        Evaluate architecture without training.
        
        Uses shared weights - much faster than training from scratch.
        """
        self.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_data:
                outputs = self.forward(batch['x'], arch)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == batch['y']).sum().item()
                total_samples += len(batch['y'])
        
        return total_correct / total_samples
```

**c) Performance prediction:**

```python
from sklearn.ensemble import RandomForestRegressor

class PerformancePredictor:
    """
    Predict architecture performance without training.
    
    Train a surrogate model: architecture features → accuracy.
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.trained = False
    
    def extract_features(self, arch: List[int]) -> np.ndarray:
        """
        Extract features from architecture.
        
        Features:
        - Number of each operation type
        - Depth (number of layers)
        - Estimated FLOPs
        - Estimated params
        """
        features = []
        
        # Count each operation type
        from collections import Counter
        op_counts = Counter(arch)
        for op_idx in range(max(arch) + 1):
            features.append(op_counts.get(op_idx, 0))
        
        # Add depth
        features.append(len(arch))
        
        return np.array(features)
    
    def train(self, architectures: List[List[int]], accuracies: List[float]):
        """Train predictor on evaluated architectures."""
        X = np.array([self.extract_features(arch) for arch in architectures])
        y = np.array(accuracies)
        
        self.model.fit(X, y)
        self.trained = True
    
    def predict(self, arch: List[int]) -> float:
        """Predict accuracy for architecture."""
        if not self.trained:
            raise ValueError("Predictor not trained")
        
        features = self.extract_features(arch).reshape(1, -1)
        return self.model.predict(features)[0]
```

## Data Flow

### NAS Pipeline

```
1. Initialize search
   └─> Define search space
   └─> Initialize controller (RL/EA/DARTS)
   └─> Set up distributed workers

2. Search loop (1000-10000 iterations)
   └─> Controller proposes architecture
   └─> (Optional) Performance predictor filters bad candidates
   └─> Evaluate architecture:
       - Train on subset of data
       - Measure accuracy, latency, size
   └─> Update controller based on reward
   └─> Store results

3. Post-processing
   └─> Identify Pareto frontier (best accuracy-latency trade-offs)
   └─> Retrain top candidates on full data
   └─> Final evaluation on test set

4. Deployment
   └─> Export best architecture
   └─> Optimize for target hardware
   └─> Deploy to production
```

## Scaling Strategies

### Distributed Architecture Evaluation

```python
import ray

@ray.remote(num_gpus=1)
class ArchitectureWorker:
    """Worker that evaluates architectures on GPU."""
    
    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space
    
    def evaluate(self, arch: List[int], train_subset, val_subset) -> Dict:
        """
        Evaluate architecture.
        
        Returns:
            Dictionary with accuracy, latency, params, flops
        """
        # Build model
        model = build_model_from_arch(arch, self.search_space)
        
        # Train briefly
        train_model(model, train_subset, epochs=10)
        
        # Evaluate
        accuracy = evaluate(model, val_subset)
        latency = measure_latency(model)
        params = count_parameters(model)
        flops = estimate_flops(model)
        
        return {
            "accuracy": accuracy,
            "latency_ms": latency,
            "params": params,
            "flops": flops
        }


class DistributedNAS:
    """Distributed NAS system."""
    
    def __init__(self, search_space: SearchSpace, num_workers: int = 8):
        self.search_space = search_space
        
        # Create worker pool
        self.workers = [
            ArchitectureWorker.remote(search_space)
            for _ in range(num_workers)
        ]
        
        self.num_workers = num_workers
    
    def search(self, controller: NASController, num_iterations: int = 1000):
        """
        Distributed NAS search.
        
        Args:
            controller: Architecture generator
            num_iterations: Number of architectures to try
        """
        results = []
        
        # Process in batches (parallel evaluation)
        batch_size = self.num_workers
        
        for iteration in range(0, num_iterations, batch_size):
            # Generate batch of architectures
            architectures = []
            log_probs_batch = []
            
            for _ in range(batch_size):
                arch, log_probs = controller()
                architectures.append(arch)
                log_probs_batch.append(log_probs)
            
            # Evaluate in parallel
            futures = [
                self.workers[i % self.num_workers].evaluate.remote(
                    architectures[i],
                    get_train_subset(),
                    get_val_subset()
                )
                for i in range(batch_size)
            ]
            
            eval_results = ray.get(futures)
            
            # Update controller with rewards
            for arch, log_probs, result in zip(architectures, log_probs_batch, eval_results):
                reward = result['accuracy']
                controller.update(log_probs, reward, controller_optimizer)
                results.append((arch, result))
        
        return results
```

## Monitoring & Metrics

### Key Metrics

**Search Progress:**
- Best accuracy found so far
- Number of architectures evaluated
- Search efficiency (good arch per GPU day)
- Diversity of architectures explored

**Architecture Quality:**
- Accuracy vs latency scatter plot
- Pareto frontier (optimal trade-offs)
- Architecture complexity distribution (params, FLOPs)

**Resource Usage:**
- GPU utilization
- Training time per architecture
- Total GPU hours consumed

### Visualization

- Architecture topology graphs
- Performance over search iterations
- Pareto frontier (accuracy vs latency/size)
- Operation frequency (which ops are most common in good models)

## Failure Modes & Mitigations

| Failure Mode | Impact | Mitigation |
|-------------|--------|------------|
| **Search collapse** | Controller generates same arch repeatedly | Entropy regularization, exploration bonus |
| **Overfitting to search** | Arch good on val, bad on test | Proper val/test splits, cross-validation |
| **Poor weight sharing** | ENAS/supernet gives misleading results | Standalone training for top candidates |
| **Hardware mismatch** | Arch fast on A100, slow on mobile | Include target hardware in eval |
| **Expensive search** | 1000s of GPU days | Early stopping, predictor, weight sharing |

## Real-World Case Study: Google's EfficientNet

### Google's NAS Approach

**Goal:** Find architectures that are both accurate and efficient (mobile-friendly).

**Method:**
- Multi-objective NAS optimizing accuracy and FLOPS
- Compound scaling (depth, width, resolution)
- Progressive search (coarse → fine)

**Architecture:**
- Search space: MobileNetV2-based
- Search strategy: Reinforcement learning
- Evaluation: Early stopping + supernet
- Scale: 1000 architectures evaluated, 100 GPU days

**Results:**
- **EfficientNet-B0:** 77.1% top-1 on ImageNet, 390M FLOPs
- **10x more efficient** than previous SOTA (at same accuracy)
- **Transfer learning:** Worked across domains (detection, segmentation)

### Key Lessons

1. **Multi-objective is crucial:** Accuracy alone isn't enough
2. **Progressive search:** Start coarse, refine best candidates
3. **Transfer across tasks:** Good architecture for ImageNet → good for other vision tasks
4. **Hardware-aware:** Include latency/FLOPS in objective
5. **Compound scaling:** After finding base arch, scale systematically

## Cost Analysis

### NAS vs Manual Design

| Approach | Time | GPU Cost | Quality | Notes |
|----------|------|----------|---------|-------|
| Manual design | 3-6 months | 100 GPU days | Good | Expert-dependent |
| Random search | N/A | 1000 GPU days | Poor | Baseline |
| RL-based NAS | 1 month | 200 GPU days | Better | EfficientNet-style |
| DARTS | 1 week | 4 GPU days | Good | Fast but less stable |
| Transfer + fine-tune | 1 week | 10 GPU days | Good | Use existing NAS results |

**ROI Calculation:**
- Manual design: 3 months engineer time ($60K) + 100 GPU days ($30K) = $90K
- NAS: 1 month engineer time ($20K) + 200 GPU days ($60K) = $80K
- **Savings:** $10K + better model

## Advanced Topics

### 1. Once-For-All Networks

Train a single super-network that contains many sub-networks:

```python
class OnceForAllNetwork:
    """
    Train once, deploy many architectures.
    
    Enables instant architecture selection without retraining.
    """
    
    def __init__(self):
        self.supernet = create_supernet()
        self.trained = False
    
    def train_supernet(self, train_data):
        """
        Train supernet to support all sub-architectures.
        
        Progressive shrinking strategy:
        - Train largest network first
        - Progressively train smaller sub-networks
        """
        # Implementation details...
        pass
    
    def extract_subnet(self, target_latency_ms: float):
        """
        Extract sub-network meeting latency constraint.
        
        No training needed!
        """
        # Search for subnet with latency < target
        # Use efficiency predictor
        pass
```

### 2. Hardware-Aware NAS

Include hardware metrics in search:

```python
def hardware_aware_nas(search_space, target_hardware: str):
    """
    Search for architectures optimized for specific hardware.
    
    Args:
        target_hardware: "mobile", "edge_tpu", "nvidia_t4", etc.
    """
    # Measure latency on target hardware
    def measure_latency_on_target(arch):
        model = build_model(arch)
        # Deploy to target, measure
        return measure_on_hardware(model, target_hardware)
    
    # Multi-objective: accuracy + latency on target
    def fitness(arch):
        acc = evaluate_accuracy(arch)
        lat = measure_latency_on_target(arch)
        
        # Combine (accuracy high, latency low)
        return acc - 0.01 * lat  # Weight latency penalty
```

### 3. Transfer NAS

Transfer architectures across tasks:

```python
def transfer_nas(source_task: str, target_task: str):
    """
    Transfer NAS results from source to target task.
    
    Example: ImageNet → COCO detection
    """
    # Load architectures found on source task
    source_archs = load_nas_results(source_task)
    
    # Top-K from source
    top_archs = sorted(source_archs, key=lambda x: x['accuracy'], reverse=True)[:10]
    
    # Fine-tune on target task
    target_results = []
    for arch in top_archs:
        # Build model with source architecture
        model = build_model_from_arch(arch['architecture'])
        
        # Fine-tune on target task
        fine_tune(model, target_task_data)
        
        # Evaluate
        target_acc = evaluate(model, target_task_test_data)
        target_results.append((arch, target_acc))
    
    # Best transferred architecture
    best = max(target_results, key=lambda x: x[1])
    return best
```

## Key Takeaways

✅ **NAS automates architecture design** - discovers models competitive with or better than hand-designed ones

✅ **Search space is exponential** - like paths in a grid, exponentially many architectures

✅ **DP and smart search** reduce complexity - from infeasible to practical

✅ **Multiple search strategies** - RL (flexible), DARTS (fast), evolutionary (robust)

✅ **Weight sharing critical** - enables evaluating 1000s of architectures efficiently

✅ **Multi-objective optimization** - accuracy vs latency vs size

✅ **Hardware-aware NAS** - optimize for target deployment platform

✅ **Transfer learning works** - architectures transfer across tasks

✅ **Same DP principles** as Unique Paths - break into subproblems, build optimal solution

✅ **Production deployment** - once-for-all networks, progressive search, cost-aware

### Connection to Thematic Link: Dynamic Programming and Path Optimization

All three Day 21 topics use **DP for path optimization**:

**DSA (Unique Paths):**
- Count paths in m×n grid using DP
- Recurrence: paths(i,j) = paths(i-1,j) + paths(i,j-1)
- Reduces O(2^(m+n)) to O(m×n)

**ML System Design (Neural Architecture Search):**
- Search through exponential architecture space
- Use DP/RL/gradient-based methods to find optimal
- Build architectures from optimal sub-architectures

**Speech Tech (Speech Architecture Search):**
- Search encoder/decoder configurations
- Use DP to evaluate speech model paths
- Find optimal ASR/TTS architectures

The **unifying principle**: navigate exponentially large search spaces by breaking into subproblems, solving optimally, and building up the final solution—whether counting grid paths, finding neural architectures, or designing speech models.

---

**Originally published at:** [arunbaby.com/ml-system-design/0021-neural-architecture-search](https://www.arunbaby.com/ml-system-design/0021-neural-architecture-search/)

*If you found this helpful, consider sharing it with others who might benefit.*


