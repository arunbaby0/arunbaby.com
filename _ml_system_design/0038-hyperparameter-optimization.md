---
title: "Hyperparameter Optimization"
day: 38
collection: ml_system_design
categories:
  - ml_system_design
tags:
  - hyperparameter-tuning
  - bayesian-optimization
  - optuna
  - ray-tune
  - automl
subdomain: "MLOps"
tech_stack: [Optuna, Ray Tune, Hyperopt, Weights & Biases]
scale: "1000s of trials"
companies: [Google, Meta, OpenAI, Uber]
---

**"Finding the perfect knobs to turn."**

## 1. The Problem: Too Many Knobs

Training a neural network involves many hyperparameters:
- **Learning rate:** 0.001? 0.01? 0.0001?
- **Batch size:** 32? 64? 128?
- **Number of layers:** 3? 5? 10?
- **Dropout rate:** 0.1? 0.3? 0.5?
- **Optimizer:** Adam? SGD? AdamW?

**Challenge:** The search space is **exponential**. For 10 hyperparameters with 5 values each, that's $5^{10} = 9.7$ million combinations!

## 2. Search Strategies

### 1. Grid Search
- **Idea:** Try all combinations.
- **Pros:** Exhaustive, guaranteed to find best in grid.
- **Cons:** Exponentially expensive.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'num_layers': [3, 5, 7]
}

# Total trials: 3 × 3 × 3 = 27
```

### 2. Random Search
- **Idea:** Sample random combinations.
- **Pros:** More efficient than grid search.
- **Insight:** Most hyperparameters don't matter much. Random search explores more of the important ones.

```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64, 128, 256]
}

# Try 20 random combinations
```

### 3. Bayesian Optimization
- **Idea:** Build a probabilistic model of the objective function.
- **Acquisition Function:** Decides where to sample next (balance exploration vs. exploitation).
- **Pros:** Sample-efficient. Converges faster than random search.

## 3. Bayesian Optimization Deep Dive

**Algorithm:**
1.  **Surrogate Model:** Gaussian Process (GP) models $f(\theta) \approx \text{validation accuracy}$.
2.  **Acquisition Function:** Expected Improvement (EI) or Upper Confidence Bound (UCB).
    $$\text{EI}(\theta) = \mathbb{E}[\max(f(\theta) - f(\theta^*), 0)]$$
    Where $\theta^*$ is the current best.
3.  **Optimize Acquisition:** Find $\theta$ that maximizes EI.
4.  **Evaluate:** Train model with $\theta$, observe accuracy.
5.  **Update GP:** Add new observation, repeat.

**Libraries:**
- **Optuna:** Most popular in ML.
- **Hyperopt:** Tree-structured Parzen Estimator (TPE).
- **Ray Tune:** Distributed tuning at scale.

## 4. Optuna Example

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Train model
    model = build_model(lr, batch_size, dropout)
    val_acc = train_and_evaluate(model)
    
    return val_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
```

## 5. Advanced: Multi-Fidelity Optimization

**Problem:** Evaluating each trial is expensive (train for 100 epochs).

**Solution:** **Successive Halving** (Hyperband).
1.  Start with many trials, train for 1 epoch.
2.  Keep top 50%, train for 2 epochs.
3.  Keep top 50%, train for 4 epochs.
4.  Repeat until 1 trial remains, train for 100 epochs.

**Speedup:** 10-100x faster than full evaluation.

## 6. Ray Tune for Distributed Tuning

```python
from ray import tune

def train_model(config):
    model = build_model(config['lr'], config['batch_size'])
    for epoch in range(10):
        loss = train_epoch(model)
        tune.report(loss=loss)

config = {
    'lr': tune.loguniform(1e-5, 1e-1),
    'batch_size': tune.choice([16, 32, 64])
}

analysis = tune.run(
    train_model,
    config=config,
    num_samples=100,
    resources_per_trial={'gpu': 1}
)

print(f"Best config: {analysis.best_config}")
```

## 7. Summary

| Method | Trials Needed | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Grid** | $O(k^n)$ | Exhaustive | Exponential |
| **Random** | $O(100)$ | Simple | Inefficient |
| **Bayesian** | $O(50)$ | Sample-efficient | Complex |
| **Hyperband** | $O(20)$ | Very fast | Needs early stopping |

## 8. Deep Dive: Acquisition Functions

Acquisition functions decide where to sample next in Bayesian Optimization.

### 1. Expected Improvement (EI)
$$\text{EI}(\theta) = \mathbb{E}[\max(f(\theta) - f(\theta^*), 0)]$$
- **Intuition:** How much better can we expect this point to be?
- **Pros:** Balances exploration (high uncertainty) and exploitation (high mean).

### 2. Upper Confidence Bound (UCB)
$$\text{UCB}(\theta) = \mu(\theta) + \kappa \sigma(\theta)$$
- $\mu$: Predicted mean.
- $\sigma$: Predicted std dev (uncertainty).
- $\kappa$: Exploration parameter (typically 2-3).
- **Intuition:** Optimistic estimate. "This could be really good!"

### 3. Probability of Improvement (PI)
$$\text{PI}(\theta) = P(f(\theta) > f(\theta^*))$$
- **Intuition:** What's the chance this beats the current best?
- **Cons:** Too greedy, doesn't care *how much* better.

## 9. Deep Dive: Hyperband Algorithm

**Problem:** Training to convergence is expensive. Can we stop bad trials early?

**Hyperband (Successive Halving + Adaptive Resource Allocation):**

```python
def hyperband(max_iter=81, eta=3):
    # max_iter: max epochs
    # eta: downsampling rate
    
    s_max = int(np.log(max_iter) / np.log(eta))
    B = (s_max + 1) * max_iter
    
    for s in reversed(range(s_max + 1)):
        n = int(np.ceil(B / max_iter / (s + 1) * eta**s))
        r = max_iter * eta**(-s)
        
        # Generate n random configurations
        configs = [random_config() for _ in range(n)]
        
        for i in range(s + 1):
            n_i = int(n * eta**(-i))
            r_i = int(r * eta**i)
            
            # Train each config for r_i epochs
            results = [train(c, r_i) for c in configs]
            
            # Keep top 1/eta
            configs = top_k(configs, results, int(n_i / eta))
    
    return best_config
```

**Example:** `max_iter=81`, `eta=3`
- Round 1: 81 configs, 1 epoch each.
- Round 2: 27 configs (top 1/3), 3 epochs each.
- Round 3: 9 configs, 9 epochs each.
- Round 4: 3 configs, 27 epochs each.
- Round 5: 1 config, 81 epochs.

## 10. Deep Dive: Parallel Hyperparameter Tuning

**Challenge:** Bayesian Optimization is sequential (needs previous results to decide next point).

**Solution 1: Batch Bayesian Optimization**
- Use acquisition function to select top-$k$ points.
- Evaluate them in parallel.
- Update GP with all $k$ results.

**Solution 2: Asynchronous Successive Halving (ASHA)**
- Don't wait for all trials to finish.
- As soon as a trial completes an epoch, decide: promote or kill.

```python
# Ray Tune with ASHA
from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(
    max_t=100,  # Max epochs
    grace_period=10,  # Min epochs before stopping
    reduction_factor=3
)

tune.run(
    train_model,
    config=config,
    num_samples=100,
    scheduler=scheduler,
    resources_per_trial={'gpu': 1}
)
```

## 11. System Design: Hyperparameter Tuning Platform

**Scenario:** Build a platform for 100 ML engineers to tune models.

**Requirements:**
- **Scalability:** 1000s of concurrent trials.
- **Reproducibility:** Track all experiments.
- **Visualization:** Compare trials easily.

**Architecture:**
1.  **Scheduler:** Ray Tune (distributed).
2.  **Tracking:** Weights & Biases (W&B) or MLflow.
3.  **Storage:** S3 for checkpoints.
4.  **Compute:** Kubernetes cluster with autoscaling.

**Code:**
```python
import wandb
from ray import tune

def train_with_logging(config):
    wandb.init(project='hyperparameter-tuning', config=config)
    
    model = build_model(config)
    for epoch in range(100):
        loss = train_epoch(model)
        wandb.log({'loss': loss, 'epoch': epoch})
        tune.report(loss=loss)

tune.run(
    train_with_logging,
    config=search_space,
    num_samples=1000
)
```

## 12. Deep Dive: Transfer Learning for Hyperparameters

**Idea:** If we tuned hyperparameters for Task A, can we use them for Task B?

**Meta-Learning Approach:**
1.  Collect tuning history from many tasks.
2.  Train a model: $f(\text{task features}) \rightarrow \text{good hyperparameters}$.
3.  For new task, predict good starting point.

**Example:** Google Vizier uses this internally.

## 13. Production Considerations

1.  **Cost:** Each trial costs GPU hours. Set a budget.
2.  **Reproducibility:** Always set random seeds.
3.  **Monitoring:** Track resource usage (GPU util, memory).
4.  **Checkpointing:** Save model every N epochs (for Hyperband).
5.  **Early Stopping:** Don't waste time on diverging models.

## 14. Summary

| Method | Trials Needed | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Grid** | $O(k^n)$ | Exhaustive | Exponential |
| **Random** | $O(100)$ | Simple | Inefficient |
| **Bayesian** | $O(50)$ | Sample-efficient | Complex |
| **Hyperband** | $O(20)$ | Very fast | Needs early stopping |
| **ASHA** | $O(20)$ | Parallel | Requires Ray |

---

**Originally published at:** [arunbaby.com/ml-system-design/0038-hyperparameter-optimization](https://www.arunbaby.com/ml-system-design/0038-hyperparameter-optimization/)
