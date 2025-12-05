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

- **Early Stopping:** Don't waste time on diverging models.

## 14. Deep Dive: Population-Based Training (PBT)

**Origin:** DeepMind (2017). Used to train AlphaStar and Waymo agents.

**Concept:**
- Combines **Random Search** (exploration) with **Greedy Selection** (exploitation).
- Instead of fixed hyperparameters, PBT evolves them *during* training.

**Algorithm:**
1.  **Initialize:** Start a population of $N$ models with random hyperparameters.
2.  **Train:** Train all models for $k$ steps.
3.  **Eval:** Evaluate performance.
4.  **Exploit:** Replace the bottom 20% of models with copies of the top 20%.
5.  **Explore:** Perturb the hyperparameters of the copied models (mutation).
    - `lr = lr * random.choice([0.8, 1.2])`
6.  **Repeat:** Continue training.

**Benefits:**
- **Dynamic Schedules:** Discovers complex schedules (e.g., "start with high LR, then decay, then spike").
- **Efficiency:** No wasted compute on bad trials (they get killed).
- **Single Run:** You get a fully trained model at the end, not just a config.

## 15. Deep Dive: Neural Architecture Search (NAS)

Hyperparameters aren't just numbers (LR, Batch Size). They can be the **architecture** itself.

**Search Space:**
- Number of layers.
- Operation type (Conv3x3, Conv5x5, MaxPool).
- Skip connections.

**Algorithms:**
1.  **Reinforcement Learning (RL):**
    - Controller (RNN) generates an architecture string.
    - Train child network, get accuracy (Reward).
    - Update Controller using Policy Gradient.
    - **Cons:** Extremely slow (2000 GPU-days for original NAS).

2.  **Evolutionary Algorithms (EA):**
    - Mutate architectures (add layer, change filter size).
    - Select best, repeat.
    - **Example:** AmoebaNet.

3.  **Differentiable NAS (DARTS):**
    - Relax discrete choices into continuous weights (softmax).
    - Train architecture weights $\alpha$ and model weights $w$ simultaneously using gradient descent.
    - **Pros:** Fast (single GPU-day).

## 16. Deep Dive: The Math of Gaussian Processes (GP)

Bayesian Optimization relies on GPs. What are they?

**Definition:** A GP is a distribution over functions, defined by a **mean function** $m(x)$ and a **covariance function** (kernel) $k(x, x')$.

$$f(x) \sim GP(m(x), k(x, x'))$$

**Kernels:**
- **RBF (Radial Basis Function):** Smooth functions.
  $$k(x, x') = \sigma^2 \exp(-\frac{||x - x'||^2}{2l^2})$$
- **Matern:** Rougher functions (better for deep learning landscapes).

**Posterior Update:**
Given observed data $D = \{(x_i, y_i)\}$, the predictive distribution for a new point $x_*$ is Gaussian:
$$P(f_* | D, x_*) = \mathcal{N}(\mu_*, \Sigma_*)$$

$$\mu_* = K_*^T (K + \sigma_n^2 I)^{-1} y$$
$$\Sigma_* = K_{**} - K_*^T (K + \sigma_n^2 I)^{-1} K_*$$

- $\mu_*$: Predicted value (Exploitation).
- $\Sigma_*$: Uncertainty (Exploration).

## 17. Deep Dive: Tree-Structured Parzen Estimator (TPE)

Optuna uses TPE by default. It's faster than GPs for high dimensions.

**Idea:** Instead of modeling $P(y|x)$ (GP), model $P(x|y)$ and $P(y)$.

1.  **Split Data:** Divide observations into two groups:
    - Top 20% (Good): $l(x)$
    - Bottom 80% (Bad): $g(x)$

2.  **Density Estimation:** Fit Kernel Density Estimators (KDE) to $l(x)$ and $g(x)$.
    - "What do good hyperparameters look like?"
    - "What do bad hyperparameters look like?"

3.  **Acquisition:** Maximize Expected Improvement, which simplifies to maximizing:
    $$\frac{l(x)}{g(x)}$$

**Intuition:** Pick $x$ that is highly likely under the "Good" distribution and unlikely under the "Bad" distribution.

## 18. Case Study: Tuning BERT for Production

**Scenario:** Fine-tuning BERT-Large for Sentiment Analysis.

**Search Space:**
- **Learning Rate:** $1e-5, 2e-5, 3e-5, 5e-5$.
- **Batch Size:** 16, 32.
- **Epochs:** 2, 3, 4.
- **Warmup Steps:** 0, 100, 500.

**Key Findings (RoBERTa paper):**
- **Batch Size:** Larger is better (up to a point).
- **Training Duration:** Training longer with smaller LR is better than short/high LR.
- **Layer-wise LR Decay:** Lower layers (closer to input) capture general features, need smaller LR. Higher layers need larger LR.
  - $\text{LR}_{layer} = \text{LR}_{base} \times \xi^{L - layer}$ where $\xi = 0.95$.

## 19. Case Study: AlphaGo Zero Tuning

**Problem:** Tuning Monte Carlo Tree Search (MCTS) + Neural Network.

**Hyperparameters:**
- **$c_{puct}$:** Exploration constant in MCTS.
- **Dirichlet Noise $\alpha$:** Noise added to root node for exploration.
- **Self-play games:** How many games before retraining?

**Strategy:**
- **Self-Play Evaluation:** New model plays 400 games against old model.
- **Gating:** Only promote if win rate > 55%.
- **Massive Parallelism:** Thousands of TPUs generating self-play data.

## 20. System Design: Scalable Tuning Infrastructure

**Components:**
1.  **Experiment Manager (Katib / Ray Tune):**
    - Stores search space config.
    - Generates trials.
2.  **Trial Runner (Kubernetes Pods):**
    - Pulls Docker image.
    - Runs training code.
    - Reports metrics to Manager.
3.  **Database (MySQL / PostgreSQL):**
    - Stores trial history (params, metrics).
4.  **Dashboard (Vizier / W&B):**
    - Visualizes parallel coordinate plots.

**Scalability Challenges:**
- **Database Bottleneck:** 1000 concurrent trials reporting metrics every second.
  - *Fix:* Buffer metrics in Redis, flush to DB periodically.
- **Pod Startup Latency:** K8s takes 30s to start a pod.
  - *Fix:* Use a pool of warm pods (Ray Actors).

## 21. Deep Dive: Multi-Objective Optimization

**Real World:** We don't just want accuracy. We want:
1.  Maximize Accuracy.
2.  Minimize Latency.
3.  Minimize Model Size.

**Pareto Frontier:**
- A set of solutions where you cannot improve one objective without hurting another.
- **Dominated Solution:** Worse than another solution in *all* objectives.
- **Non-Dominated Solution:** Better in at least one objective.

**Scalarization:**
- Convert to single objective: $L = w_1 \cdot Acc + w_2 \cdot \frac{1}{Lat}$.
- **Problem:** Need to tune weights $w$.

**NSGA-II (Non-dominated Sorting Genetic Algorithm):**
- Used by Optuna for multi-objective search.
- Maintains a population of Pareto-optimal solutions.

## 22. Code: Implementing a Simple Bayesian Optimizer

Let's build a toy BO from scratch using `scikit-learn`.

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

class SimpleBayesianOptimizer:
    def __init__(self, objective_func, bounds):
        self.objective = objective_func
        self.bounds = bounds
        self.X = []
        self.y = []
        self.gp = GaussianProcessRegressor(kernel=Matern(nu=2.5))
        
    def expected_improvement(self, X_candidates):
        mu, sigma = self.gp.predict(X_candidates, return_std=True)
        mu_sample_opt = np.max(self.y)
        
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return ei
        
    def optimize(self, n_iters=10):
        # Initial random samples
        for _ in range(2):
            x = np.random.uniform(self.bounds[0], self.bounds[1], 1).reshape(-1, 1)
            y = self.objective(x)
            self.X.append(x)
            self.y.append(y)
            
        for i in range(n_iters):
            # Fit GP
            self.gp.fit(np.array(self.X).reshape(-1, 1), np.array(self.y))
            
            # Find point with max EI
            X_grid = np.linspace(self.bounds[0], self.bounds[1], 100).reshape(-1, 1)
            ei = self.expected_improvement(X_grid)
            next_x = X_grid[np.argmax(ei)]
            
            # Evaluate
            next_y = self.objective(next_x)
            self.X.append(next_x)
            self.y.append(next_y)
            
            print(f"Iter {i}: Best y = {np.max(self.y):.4f}")

# Usage
def objective(x): return -1 * (x - 2)**2 + 10  # Max at x=2
opt = SimpleBayesianOptimizer(objective, bounds=(-5, 5))
opt.optimize(n_iters=10)
```

## 23. Future Trends: AutoML-Zero

**Goal:** Evolve the *algorithms* themselves, not just parameters.

**Method:**
- Represent ML algorithms as a sequence of basic math operations (add, multiply, sin, cos).
- Use evolutionary algorithms to discover "Gradient Descent" or "Neural Networks" from scratch.
- **Result:** Rediscovered backpropagation and linear regression.

**Implication:** Future ML engineers might tune "Search Space Definitions" rather than models.

## 24. Summary

| **ASHA** | $O(20)$ | Parallel | Requires Ray |

## 25. Deep Dive: Hyperparameter Importance Analysis

After running 100 trials, you want to know: **Which knob actually mattered?**

**Methods:**
1.  **fANOVA (Functional Analysis of Variance):**
    - Decomposes the variance of the objective function into additive components.
    - "60% of variance comes from Learning Rate, 10% from Batch Size, 5% from interaction between LR and Batch Size."
    - **Tool:** `optuna.importance.get_param_importances(study)`.

2.  **SHAP (SHapley Additive exPlanations):**
    - Treats hyperparameter values as "features" and the objective value as the "prediction".
    - Calculates the marginal contribution of each hyperparameter.

3.  **Parallel Coordinate Plots:**
    - Visualizes the high-dimensional relationships.
    - Useful for spotting "bad regions" (e.g., "High LR + Low Batch Size always crashes").

**Actionable Insight:**
- If `num_layers` has 1% importance, stop tuning it! Fix it to a reasonable default and save compute.

## 26. Deep Dive: Handling Categorical & Conditional Hyperparameters

Real-world search spaces are messy.

**Categorical:**
- `optimizer`: ["Adam", "SGD", "RMSprop"]
- **Problem:** GPs assume continuous distance. Distance("Adam", "SGD") is undefined.
- **Solution:** One-hot encoding or using Tree-based models (Random Forests, TPE) which handle splits naturally.

**Conditional (Nested):**
- IF `optimizer == "SGD"` THEN tune `momentum`.
- IF `optimizer == "Adam"` THEN tune `beta1`, `beta2`.
- **Problem:** `momentum` is irrelevant if `optimizer` is Adam.
- **Solution:**
  - **TPE:** Handles this naturally by splitting the tree.
  - **ConfigSpace:** A library specifically for defining DAG-structured search spaces.

## 27. Deep Dive: Warm-Starting Optimization

**Problem:** Every time we tune a new model, we start from scratch (random sampling).
**Reality:** We have tuned 50 similar models before.

**Strategies:**
1.  **Initial Points:**
    - Instead of random initialization, seed the optimizer with the best configs from previous studies.
    - `study.enqueue_trial({'lr': 1e-3, 'batch_size': 32})`.

2.  **Transfer Learning for GPs:**
    - Use data from previous tasks to learn a "prior" for the GP mean function.
    - **Multi-Task Bayesian Optimization:** Model the correlation between Task A and Task B. If they are correlated, observations in A reduce uncertainty in B.

3.  **Meta-Learning (Auto-Sklearn):**
    - Compute meta-features of the dataset (num_rows, num_cols, class_balance).
    - Find nearest neighbors in the "dataset space".
    - Reuse their best hyperparameters.

## 28. Case Study: Tuning XGBoost vs Neural Networks

**XGBoost / LightGBM:**
- **Key Params:** `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`.
- **Landscape:** Rugged but convex-ish locally.
- **Strategy:** Random Search is often "good enough". TPE works very well.
- **Cost:** Fast to train (seconds/minutes). Can run 1000s of trials.

**Neural Networks (ResNet/Transformer):**
- **Key Params:** `lr`, `batch_size`, `optimizer`, `scheduler`.
- **Landscape:** Non-convex, saddle points, noise.
- **Strategy:** Must use Learning Rate Schedules. Tuning the *schedule* is more important than tuning the fixed LR.
- **Cost:** Slow (hours/days). Must use Early Stopping (Hyperband).

## 29. Ethical Considerations: The Carbon Footprint of Tuning

**The Cost:**
- Training a Transformer with NAS can emit 600,000 lbs of CO2 (equivalent to 5 cars' lifetime).
- "Red AI" (buying performance with massive compute) vs "Green AI" (efficiency).

**Mitigation Strategies:**
1.  **Green NAS:** Penalize energy consumption in the objective function.
    $$L = \text{Error} + \lambda \cdot \text{Energy}$$
2.  **Proxy Tasks:** Tune on a subset of data (10%), then transfer to full data.
3.  **Share Configs:** Publish the best hyperparameters so others don't have to re-tune. (Hugging Face Model Cards).

## 30. Further Reading

1.  **"Algorithms for Hyper-Parameter Optimization" (Bergstra et al., 2011):** Introduced TPE.
2.  **"Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization" (Li et al., 2018):** The standard for resource allocation.
3.  **"Google Vizier: A Service for Black-Box Optimization" (Golovin et al., 2017):** How Google does it at scale.
4.  **"Neural Architecture Search with Reinforcement Learning" (Zoph & Le, 2017):** The paper that started the NAS craze.
5.  **"On the Importance of On-Manifold Regularization" (Mixup):** Data augmentation as a hyperparameter.

- **NAS:** $O(1000)$ | Finds architecture | Very expensive |

## 32. Deep Dive: The Future of Tuning - LLMs as Optimizers

**OptiMus (2023):**
- Uses an LLM (GPT-4) to suggest hyperparameters.
- **Prompt:** "I am training a ResNet-50. The loss is oscillating. Current LR is 0.1. What should I try next?"
- **Response:** "Try reducing LR to 0.01 and adding a scheduler."
- **Why it works:** LLMs have read millions of papers and GitHub issues. They have "common sense" about training dynamics that Bayesian Optimization lacks.

**OMNI (OpenAI):**
- Future systems will likely abstract tuning away entirely. You provide data + metric, the system handles the rest.

## 33. Deep Dive: Tuning for Robustness and Fairness

**Robustness (Adversarial Training):**
- **Hyperparams:** Epsilon (perturbation size), Alpha (step size).
- **Trade-off:** Increasing robustness often decreases clean accuracy.
- **Tuning Goal:** Find the Pareto frontier between Accuracy and Robustness.

**Fairness:**
- **Hyperparams:** Regularization strength for fairness constraints (e.g., Equalized Odds).
- **Objective:** Minimize Error + $\lambda \cdot \text{Disparity}$.
- **Tuning:** We need to find the $\lambda$ that satisfies legal/ethical requirements while maximizing utility.

## 34. Code: Grid Search from Scratch

To understand why Grid Search is bad, let's implement it.

```python
import itertools

def grid_search(objective, param_grid):
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    best_score = -float('inf')
    best_params = None
    
    print(f"Total combinations: {len(combinations)}")
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        score = objective(params)
        
        if score > best_score:
            best_score = score
            best_params = params
            
    return best_params, best_score

# Usage
grid = {
    'lr': [0.1, 0.01, 0.001],
    'batch_size': [32, 64, 128],
    'dropout': [0.1, 0.5]
}
# 3 * 3 * 2 = 18 trials.
# If we add one more parameter with 5 options -> 90 trials.
# Exponential explosion!
```

## 35. Production Checklist for Hyperparameter Tuning

Before you launch a tuning job:

1.  [ ] **Define Metric:** Is it Accuracy? F1? AUC? Latency?
2.  [ ] **Define Budget:** How many GPU hours can I afford?
3.  [ ] **Choose Algorithm:**
    - < 10 params: Bayesian Optimization (Optuna).
    - > 10 params: Random Search or Hyperband.
    - Neural Net: Hyperband / ASHA.
4.  [ ] **Set Search Space:**
    - Use Log Scale for LR and Regularization.
    - Don't tune things that don't matter (e.g., random seed).
5.  [ ] **Enable Early Stopping:** Don't waste compute.
6.  [ ] **Log Everything:** Use W&B / MLflow.
7.  [ ] **Verify on Test Set:** Evaluate the *single best* model on the held-out test set.

- **NAS:** $O(1000)$ | Finds architecture | Very expensive |

## 36. Deep Dive: Bayesian Optimization Hyperband (BOHB)

**Problem:**
- **Bayesian Optimization** is great at finding good configs but slow (doesn't kill bad trials).
- **Hyperband** is fast (kills bad trials) but random (doesn't learn from history).

**Solution: BOHB (2018)**
- Combines the best of both.
- Uses **Hyperband** to determine how many resources (epochs) to allocate.
- Uses **Bayesian Optimization (TPE)** to select the configurations to run at each step.
- **Result:** Converges faster than either method alone. SOTA for many problems.

## 37. Deep Dive: The "No Free Lunch" Theorem in Tuning

**Theorem:** Averaged over all possible problems, every optimization algorithm performs equally well (same as random search).

**Implication:**
- There is no "Best Optimizer" for every problem.
- **TPE** might be best for XGBoost.
- **CMA-ES** might be best for Reinforcement Learning.
- **Adam** might be best for CNNs.
- **Lesson:** Try multiple optimizers if you are stuck.

## 38. Deep Dive: Tuning Generative Models (GANs / Diffusion)

Tuning GANs is notoriously hard.

**Challenges:**
- **Mode Collapse:** Generator produces only one image.
- **Non-Convergence:** Discriminator becomes too strong too fast.

**Key Hyperparameters:**
- **Learning Rate Ratio:** Often we set TTUR (Two-Time-Scale Update Rule).
  - $LR_{disc} = 4 \times LR_{gen}$.
- **Beta1:** Momentum. Often set to 0.0 or 0.5 (instead of default 0.9).
- **Gradient Penalty:** Weight $\lambda$ for WGAN-GP.

**Diffusion Models:**
- **Noise Schedule:** Linear vs Cosine.
- **Timesteps:** 1000? 4000?
- **EMA Decay:** Exponential Moving Average of weights (crucial for quality).

## 39. Case Study: Tuning Stable Diffusion

**Goal:** Fine-tune Stable Diffusion on a specific style (e.g., "Disney Style").

**Method: Dreambooth / LoRA.**

**Hyperparameters:**
- **Learning Rate:** Extremely sensitive. $1e-6$ works, $1e-5$ destroys the model.
- **Text Encoder Training:** Train it or freeze it? (Training = better likeness, Freezing = better editing).
- **Prior Preservation Loss:** Weight of the class images (to prevent forgetting what a "dog" looks like).

- **Prior Preservation Loss:** Weight of the class images (to prevent forgetting what a "dog" looks like).

## 40. The Psychology of Tuning

Why do humans struggle with tuning?

1.  **Confirmation Bias:** We try 3 things, one works, and we assume it's the "Golden Config". We stop searching.
2.  **Sunk Cost Fallacy:** "I spent 3 days tuning this ResNet. I can't switch to EfficientNet now."
3.  **Dimensionality Curse:** Humans can visualize 2D/3D. We cannot intuit 10D spaces. We miss interactions (e.g., "LR is only bad if Batch Size is small").

**Lesson:** Trust the algorithm. Don't "babysit" the tuner.

## 41. Checklist for Debugging Tuning Failures

If your tuner isn't finding good results:

1.  [ ] **Is the search space too big?** Prune irrelevant parameters.
2.  [ ] **Are the ranges correct?** Is LR `[1e-5, 1e-1]` or `[1, 10]`? (Common bug).
3.  [ ] **Is the metric noisy?** If running the same config twice gives $\pm 5\%$ accuracy, the tuner is confused. Fix the seed or average over runs.
4.  [ ] **Is the budget too small?** 10 trials is not enough for 10 parameters.
5.  [ ] **Is the model broken?** Does it train with *default* parameters? If not, fix the code first.

## 42. Summary

| Method | Trials Needed | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Grid** | $O(k^n)$ | Exhaustive | Exponential |
| **Random** | $O(100)$ | Simple | Inefficient |
| **Bayesian** | $O(50)$ | Sample-efficient | Complex |
| **Hyperband** | $O(20)$ | Very fast | Needs early stopping |
| **ASHA** | $O(20)$ | Parallel | Requires Ray |
| **PBT** | $O(20)$ | Dynamic schedules | Complex setup |
| **NAS** | $O(1000)$ | Finds architecture | Very expensive |
| **BOHB** | $O(30)$ | Best of both worlds | Complex |
| **LLM** | $O(10)$ | Uses "common sense" | New, experimental |

## 43. Further Reading

1.  **"Algorithms for Hyper-Parameter Optimization" (Bergstra et al., 2011):** The paper that introduced TPE.
2.  **"Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization" (Li et al., 2018):** The standard for resource allocation.
3.  **"Google Vizier: A Service for Black-Box Optimization" (Golovin et al., 2017):** How Google does it at scale.
4.  **"Neural Architecture Search with Reinforcement Learning" (Zoph & Le, 2017):** The paper that started the NAS craze.
5.  **"Optuna: A Next-generation Hyperparameter Optimization Framework" (Akiba et al., 2019):** The define-by-run philosophy.

## 44. Conclusion

Hyperparameter optimization is no longer a "nice to have"—it is a critical component of the modern ML stack. As models grow larger and compute becomes more expensive, the ability to efficiently navigate the search space becomes a competitive advantage. Whether you are using simple Random Search for a baseline or deploying massive Population-Based Training on a Kubernetes cluster, the principles remain the same: **Explore** the unknown, **Exploit** the promising, and **Automate** everything.

---

**Originally published at:** [arunbaby.com/ml-system-design/0038-hyperparameter-optimization](https://www.arunbaby.com/ml-system-design/0038-hyperparameter-optimization/)
```
