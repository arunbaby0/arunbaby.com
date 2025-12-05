---
title: "Speech Hyperparameter Tuning"
day: 38
collection: speech_tech
categories:
  - speech_tech
tags:
  - hyperparameter-tuning
  - asr
  - tts
  - optuna
  - neural-architecture-search
subdomain: "Optimization"
tech_stack: [Optuna, Ray Tune, ESPnet, NeMo]
scale: "1000s of experiments"
companies: [Google, Meta, NVIDIA, Baidu]
---

**"Tuning speech models for peak performance."**

## 1. Speech-Specific Hyperparameters

Speech models have unique hyperparameters beyond standard ML:

**Audio Processing:**
- **Sample Rate:** 8kHz (telephony) vs. 16kHz (standard) vs. 48kHz (high-quality).
- **Window Size:** 25ms? 50ms?
- **Hop Length:** 10ms? 20ms?
- **Num Mel Bins:** 40? 80? 128?

**Model Architecture:**
- **Encoder Type:** LSTM? Transformer? Conformer?
- **Num Layers:** 6? 12? 24?
- **Attention Heads:** 4? 8? 16?

**Training:**
- **SpecAugment:** Mask how many time/frequency bins?
- **CTC vs. Attention:** Which loss weight?

## 2. The Cost Problem

**Challenge:** Speech models are expensive to train.
- **Whisper Large:** 1 week on 256 GPUs.
- **Conformer-XXL:** 3 days on 64 GPUs.

**Implication:** We can't afford 100 trials. Need smart search.

## 3. Multi-Fidelity Tuning for ASR

**Idea:** Use smaller datasets/models as proxies.

**Fidelity Levels:**
1.  **Low:** Train on 1 hour of data, 3 layers, 1 epoch.
2.  **Medium:** Train on 10 hours, 6 layers, 5 epochs.
3.  **High:** Train on 100 hours, 12 layers, 50 epochs.

**Hyperband Strategy:**
- Start 64 trials at low fidelity.
- Promote top 16 to medium.
- Promote top 4 to high.

## 4. Optuna for Speech

```python
import optuna

def objective(trial):
    # Audio hyperparameters
    n_mels = trial.suggest_int('n_mels', 40, 128, step=8)
    win_length = trial.suggest_int('win_length', 20, 50, step=5)
    
    # Model hyperparameters
    num_layers = trial.suggest_int('num_layers', 4, 12)
    d_model = trial.suggest_categorical('d_model', [256, 512, 1024])
    
    # Training hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    
    # Build and train model
    model = build_asr_model(n_mels, win_length, num_layers, d_model)
    wer = train_and_evaluate(model, lr)
    
    return wer  # Minimize WER

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

## 5. Neural Architecture Search (NAS)

**Goal:** Automatically find the best architecture.

**Search Space:**
- **Encoder:** LSTM, GRU, Transformer, Conformer.
- **Decoder:** CTC, Attention, Transducer.
- **Connections:** Skip connections? Residual?

**Search Algorithm:**
- **ENAS (Efficient NAS):** Share weights across architectures.
- **DARTS (Differentiable):** Make architecture choices continuous, use gradient descent.

## 6. Case Study: ESPnet Tuning

**ESPnet** (End-to-End Speech Processing toolkit) has built-in tuning.

```bash
# Define search space in YAML
espnet_tune.py \
  --config conf/tuning.yaml \
  --n-trials 100 \
  --backend optuna
```

**conf/tuning.yaml:**
```yaml
search_space:
  encoder_layers: [4, 6, 8, 12]
  attention_heads: [4, 8]
  dropout: [0.1, 0.2, 0.3]
  learning_rate: [1e-4, 5e-4, 1e-3]
```

## 7. Summary

| Aspect | Strategy |
| :--- | :--- |
| **Search** | Bayesian Optimization (Optuna) |
| **Fidelity** | Hyperband (small data first) |
| **Architecture** | NAS (ENAS, DARTS) |
| **Parallelization** | Ray Tune (multi-GPU) |

## 8. Deep Dive: Audio Augmentation Hyperparameters

**SpecAugment** is crucial for speech models. But how much augmentation?

**Hyperparameters:**
- **Time Masking:** How many time steps to mask? (10? 20? 50?)
- **Frequency Masking:** How many mel bins? (5? 10? 20?)
- **Num Masks:** How many masks per spectrogram? (1? 2? 3?)

**Tuning Strategy:**
```python
def objective(trial):
    time_mask = trial.suggest_int('time_mask', 10, 100, step=10)
    freq_mask = trial.suggest_int('freq_mask', 5, 30, step=5)
    num_masks = trial.suggest_int('num_masks', 1, 3)
    
    augmenter = SpecAugment(time_mask, freq_mask, num_masks)
    model = train_with_augmentation(augmenter)
    
    return model.wer
```

**Insight:** More augmentation helps on small datasets, hurts on large ones.

## 9. Deep Dive: Conformer Architecture Search

**Conformer** is SOTA for ASR. But which variant?

**Search Space:**
- **Num Layers:** 12? 18? 24?
- **d_model:** 256? 512? 1024?
- **Conv Kernel Size:** 15? 31? 63?
- **Attention Heads:** 4? 8? 16?

**Cost:** Training a 24-layer Conformer takes 3 days on 8 GPUs.

**Multi-Fidelity Strategy:**
1.  **Proxy:** Train 6-layer model on 10 hours.
2.  **Correlation:** Check if proxy WER correlates with full model WER.
3.  **Transfer:** Top configs from proxy → Full training.

## 10. Deep Dive: Learning Rate Schedules

Speech models are sensitive to LR schedules.

**Options:**
1.  **Warmup + Decay:**
    - Warmup: Linear increase for 10k steps.
    - Decay: Cosine or exponential.
2.  **Noam Scheduler (Transformer):**
    $$\text{LR} = d_{\text{model}}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})$$
3.  **ReduceLROnPlateau:** Reduce when validation loss plateaus.

**Tuning:**
```python
def objective(trial):
    warmup_steps = trial.suggest_int('warmup_steps', 5000, 25000, step=5000)
    peak_lr = trial.suggest_float('peak_lr', 1e-4, 1e-3, log=True)
    
    scheduler = NoamScheduler(warmup_steps, peak_lr)
    model = train_with_scheduler(scheduler)
    
    return model.wer
```

## 11. System Design: Distributed Tuning for TTS

**Scenario:** Tune a multi-speaker TTS model.

**Challenges:**
- **Long Training:** 1M steps = 1 week on 1 GPU.
- **Many Hyperparameters:** 20+ (encoder, decoder, vocoder).

**Solution:**
1.  **Stage 1:** Tune encoder/decoder (freeze vocoder).
2.  **Stage 2:** Tune vocoder (freeze encoder/decoder).
3.  **Parallelization:** Ray Tune with 64 GPUs.

**Code:**
```python
from ray import tune

def train_tts(config):
    model = build_tts(
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        lr=config['lr']
    )
    
    for step in range(100000):
        loss = train_step(model)
        if step % 1000 == 0:
            tune.report(loss=loss)

config = {
    'encoder_layers': tune.choice([4, 6, 8]),
    'decoder_layers': tune.choice([4, 6]),
    'lr': tune.loguniform(1e-5, 1e-3)
}

tune.run(train_tts, config=config, num_samples=50, resources_per_trial={'gpu': 1})
```

## 12. Deep Dive: Transfer Learning from Pre-Tuned Models

**Idea:** Start from a model that's already tuned for a similar task.

**Example:**
- **Source:** English ASR (tuned on LibriSpeech).
- **Target:** Spanish ASR.
- **Transfer:** Use English hyperparameters as starting point.

**Fine-Tuning Search Space:**
- Keep architecture fixed.
- Only tune learning rate and data augmentation.

**Speedup:** 5x fewer trials needed.

## 13. Production Tuning Workflow

**Step 1: Baseline**
- Train with default hyperparameters.
- Measure WER/MOS.

**Step 2: Coarse Search**
- Use Random Search with 20 trials.
- Identify promising regions.

**Step 3: Fine Search**
- Use Bayesian Optimization with 30 trials.
- Focus on promising region.

**Step 4: Validation**
- Train best config 3 times (different seeds).
- Report mean ± std.

**Step 5: A/B Test**
- Deploy to 5% of users.
- Monitor real-world metrics.

## 14. Deep Dive: Batch Size and Gradient Accumulation

**Problem:** Larger batch sizes improve training stability but require more GPU memory.

**Hyperparameters:**
- **Batch Size:** 8? 16? 32? 64?
- **Gradient Accumulation Steps:** 1? 2? 4? 8?

**Effective Batch Size** = `batch_size × gradient_accumulation_steps × num_gpus`

**Tuning Strategy:**
```python
def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    grad_accum = trial.suggest_categorical('grad_accum', [1, 2, 4])
    
    effective_bs = batch_size * grad_accum
    
    # Adjust learning rate proportionally
    base_lr = 1e-4
    lr = base_lr * (effective_bs / 32)
    
    model = train(batch_size, grad_accum, lr)
    return model.wer
```

**Insight:** Effective batch size of 128-256 works best for most speech models.

## 15. Deep Dive: Optimizer Selection

**Options:**
1.  **Adam:** Default choice. Adaptive learning rates.
2.  **AdamW:** Adam with weight decay decoupling. Better generalization.
3.  **SGD + Momentum:** Simpler, sometimes better for very large models.
4.  **Adafactor:** Memory-efficient (no momentum buffer). Good for TPUs.

**Hyperparameters:**
- **Beta1, Beta2:** Momentum parameters for Adam.
- **Weight Decay:** L2 regularization strength.
- **Epsilon:** Numerical stability constant.

**Tuning:**
```python
def objective(trial):
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
    
    if optimizer_name in ['adam', 'adamw']:
        beta1 = trial.suggest_float('beta1', 0.85, 0.95)
        beta2 = trial.suggest_float('beta2', 0.95, 0.999)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        optimizer = AdamW(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    else:
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        optimizer = SGD(params, lr=lr, momentum=momentum)
    
    return train_with_optimizer(optimizer)
```

## 16. Case Study: Google's Conformer Tuning

**Background:** Google trained Conformer models for production ASR.

**Search Space:**
- 144 hyperparameter combinations.
- Trained on 60,000 hours of audio.

**Key Findings:**
1.  **Convolution Kernel Size:** 31 was optimal (not 15 or 63).
2.  **Dropout:** 0.1 for large datasets, 0.3 for small.
3.  **SpecAugment:** Time mask 100, freq mask 27.

**Cost:** $500,000 in GPU hours.

**Result:** 5% relative WER improvement over baseline.

## 17. Case Study: Meta's Wav2Vec 2.0 Self-Supervised Tuning

**Challenge:** Pre-training on 60,000 hours of unlabeled audio.

**Hyperparameters Tuned:**
- **Masking Probability:** 0.065 (6.5% of time steps masked).
- **Mask Length:** 10 time steps.
- **Contrastive Temperature:** 0.1.
- **Quantizer Codebook Size:** 320.

**Search Method:** Grid search with 20 configurations.

**Key Insight:** Masking probability is the most sensitive hyperparameter. 6.5% is optimal; 5% or 8% degrades performance significantly.

## 18. Deep Dive: Early Stopping Strategies

**Problem:** How do we know when to stop a trial?

**Strategies:**
1.  **Validation Loss Plateau:** Stop if loss doesn't improve for N epochs.
2.  **Hyperband:** Stop bottom 50% of trials at each rung.
3.  **Median Stopping:** Stop if current performance is below median of all trials at this step.

**Optuna Pruner:**
```python
import optuna

def objective(trial):
    model = build_model(trial.params)
    
    for epoch in range(100):
        val_loss = train_epoch(model)
        
        # Report intermediate value
        trial.report(val_loss, epoch)
        
        # Prune if not promising
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_loss

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)
study.optimize(objective, n_trials=100)
```

**Speedup:** 3-5x faster by killing bad trials early.

## 19. Deep Dive: Hyperparameter Importance Analysis

**Question:** Which hyperparameters matter most?

**Optuna Importance:**
```python
import optuna.importance

# After study completes
importance = optuna.importance.get_param_importances(study)

for param, score in importance.items():
    print(f"{param}: {score:.3f}")
```

**Example Output:**
```
learning_rate: 0.45
num_layers: 0.25
dropout: 0.15
batch_size: 0.10
optimizer: 0.05
```

**Insight:** Focus future tuning on top 2-3 hyperparameters.

## 20. Production Deployment: Model Registry

**Problem:** Track 100s of tuning experiments.

**Solution:** MLflow Model Registry.

```python
import mlflow

def objective(trial):
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'num_layers': trial.suggest_int('num_layers', 4, 12)
    }
    
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(params)
        
        # Train model
        model = train(params)
        wer = evaluate(model)
        
        # Log metrics
        mlflow.log_metric('wer', wer)
        
        # Log model
        mlflow.pytorch.log_model(model, 'model')
    
    return wer
```

**Benefits:**
- **Reproducibility:** Every experiment is tracked.
- **Comparison:** Compare trials in UI.
- **Deployment:** Promote best model to production.

## 21. Advanced: Population-Based Training (PBT)

**Idea:** Evolve hyperparameters during training (like genetic algorithms).

**Algorithm:**
1.  Start with N models (population) with random hyperparameters.
2.  Train all models for T steps.
3.  **Exploit:** Replace worst 20% with copies of best 20%.
4.  **Explore:** Perturb hyperparameters of copied models (e.g., `lr *= random.choice([0.8, 1.2])`).
5.  Repeat steps 2-4.

**Ray Tune PBT:**
```python
from ray.tune.schedulers import PopulationBasedTraining

pbt = PopulationBasedTraining(
    time_attr='training_iteration',
    metric='wer',
    mode='min',
    perturbation_interval=5,
    hyperparam_mutations={
        'lr': lambda: np.random.uniform(1e-5, 1e-3),
        'dropout': lambda: np.random.uniform(0.1, 0.5)
    }
)

tune.run(train_model, scheduler=pbt, num_samples=20)
```

**Advantage:** Adapts hyperparameters online. Can find schedules that static tuning misses.

## 22. Deep Dive: Handling Noisy Objectives

**Problem:** WER varies due to randomness (data shuffling, weight initialization).

**Solution:** Run each config multiple times, report mean.

```python
def objective(trial):
    wers = []
    for seed in [42, 123, 456]:
        set_seed(seed)
        model = train(trial.params)
        wers.append(evaluate(model))
    
    return np.mean(wers)
```

**Trade-off:** 3x slower, but more reliable.

**Alternative:** Use larger validation set to reduce variance.

## 24. Deep Dive: Bayesian Optimization Internals for Speech

Speech hyperparameter spaces are often high-dimensional and continuous. Random search is inefficient. Bayesian Optimization (BO) builds a probabilistic model of the objective function $f(x)$ (e.g., WER) and uses it to select the most promising hyperparameters to evaluate next.

**1. Gaussian Processes (GP):**
BO typically uses a GP as a surrogate model. A GP defines a distribution over functions.
- **Prior:** Before seeing any data, we assume $f(x)$ follows a multivariate normal distribution.
- **Posterior:** After observing data points $D = \{(x_1, y_1), ..., (x_n, y_n)\}$, we update the distribution.
- **Mean Function $\mu(x)$:** The expected value of WER at hyperparameter configuration $x$.
- **Covariance Function $k(x, x')$:** Encodes assumptions about smoothness. If $x$ and $x'$ are similar, $f(x)$ and $f(x')$ should be similar.
  - Common kernel: **Matern 5/2** (allows for some roughness, suitable for non-smooth deep learning landscapes).

**2. Acquisition Functions:**
How do we choose the next $x_{n+1}$? We maximize an acquisition function $\alpha(x)$.
- **Expected Improvement (EI):**
  $$EI(x) = \mathbb{E}[\max(f(x^*) - f(x), 0)]$$
  where $f(x^*)$ is the best WER observed so far. This balances exploring high-uncertainty regions and exploiting low-mean regions.
- **Upper Confidence Bound (UCB):**
  $$UCB(x) = \mu(x) - \kappa \sigma(x)$$
  (Note: minus because we minimize WER). $\kappa$ controls the exploration-exploitation trade-off.

**3. Tree-Structured Parzen Estimator (TPE):**
Standard GPs scale cubically $O(n^3)$ with the number of trials. TPE (used by Optuna) scales linearly.
- Instead of modeling $p(y|x)$, TPE models $p(x|y)$ and $p(y)$.
- It defines two densities for hyperparameters $x$:
  - $l(x)$ if $y < y^*$ (promising configs)
  - $g(x)$ if $y \ge y^*$ (bad configs)
- It chooses $x$ to maximize the ratio $l(x) / g(x)$.
- **Why it works for Speech:** Speech pipelines have conditional hyperparameters (e.g., "If optimizer=SGD, tune momentum. If Adam, ignore momentum"). TPE handles this tree structure naturally.

## 25. Deep Dive: Ray Tune Architecture

When tuning massive speech models, we need distributed compute. **Ray Tune** is the industry standard.

**Architecture:**
1.  **Driver:** The script where you define the search space and `tune.run()`.
2.  **Trial Executor:** Manages the lifecycle of trials.
3.  **Search Algorithm:** (e.g., Optuna, HyperOpt) Suggests new configurations.
4.  **Scheduler:** (e.g., ASHA, PBT) Decides whether to pause, stop, or resume trials based on intermediate results.
5.  **Trainable (Actor):** A Ray Actor (process) that runs the training loop.

**Resource Management:**
- Ray abstracts resources (CPU, GPU).
- `resources_per_trial={"cpu": 4, "gpu": 1}`.
- If you have 8 GPUs, Ray Tune runs 8 concurrent trials.
- **Fractional GPUs:** `{"gpu": 0.5}` allows running 2 small trials on one GPU (useful for small ASR models or proxy tasks).

**Fault Tolerance:**
- Speech training takes days. Nodes fail.
- Ray Tune automatically checkpoints trials.
- If a node dies, Ray reschedules the trial on a healthy node and resumes from the last checkpoint.

**Code Example: Custom Stopper**
```python
from ray.tune import Stopper

class WERPlateauStopper(Stopper):
    def __init__(self, patience=5, metric="wer"):
        self.patience = patience
        self.metric = metric
        self.best_wer = float("inf")
        self.no_improve_count = 0

    def __call__(self, trial_id, result):
        current_wer = result[self.metric]
        if current_wer < self.best_wer:
            self.best_wer = current_wer
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        return self.no_improve_count >= self.patience

    def stop_all(self):
        return False
```

## 26. System Design: Auto-Tuning Pipeline for Production

**Scenario:** A company constantly ingests new audio data (call center logs). They need to retrain and retune models weekly.

**Pipeline:**

1.  **Data Ingestion:**
    - New audio lands in S3.
    - Airflow job triggers preprocessing (MFCC extraction, text normalization).

2.  **Proxy Dataset Creation:**
    - Randomly sample 5% of the new data (~100 hours) for hyperparameter tuning.
    - Full dataset (2000 hours) reserved for final training.

3.  **Hyperparameter Search (Ray Tune + K8s):**
    - Spin up ephemeral K8s cluster with Spot Instances (cheaper).
    - Run 50 trials of ASHA on the Proxy Dataset.
    - Search space: LR, SpecAugment, Dropout.
    - Output: Best configuration JSON.

4.  **Full Training:**
    - Launch a distributed training job (PyTorch DDP) on the full dataset using the Best Configuration.
    - No tuning here, just training.

5.  **Evaluation & Gating:**
    - Evaluate on a held-out Golden Set.
    - If WER < Current Production Model, promote to Staging.

6.  **Deployment:**
    - TorchServe loads the new model.
    - Canary deployment to 1% traffic.

**Benefit:** This decouples the expensive "Search" phase (done on small data) from the expensive "Train" phase (done once).

## 27. Deep Dive: Tuning for Edge Deployment

Deploying ASR on mobile devices (Android/iOS) introduces new constraints: **Model Size** and **Latency**.

**Hyperparameters to Tune:**
1.  **Quantization Aware Training (QAT):**
    - **Bit Width:** 8-bit vs 4-bit weights.
    - **Observer Type:** MinMax vs MovingAverage.
2.  **Pruning:**
    - **Sparsity Level:** 50%? 75%? 90%?
    - **Pruning Schedule:** Linear vs Cubic.
3.  **Architecture:**
    - **Depth Multiplier:** Scale down channel dimensions (MobileNet style).
    - **Streaming Chunk Size:** 100ms vs 400ms (Latency vs Accuracy trade-off).

**Multi-Objective Optimization:**
We want to minimize WER *and* minimize Latency.
$$Loss = WER + \lambda \times Latency$$

**Pareto Frontier:**
Instead of a single best model, we want a set of models that represent optimal trade-offs.
- Model A: WER 5%, Latency 200ms.
- Model B: WER 6%, Latency 50ms.

**Optuna Multi-Objective:**
```python
def objective(trial):
    # ... build and evaluate model ...
    wer = evaluate_wer(model)
    latency = measure_latency(model)
    return wer, latency

study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=100)

# Plot Pareto Frontier
optuna.visualization.plot_pareto_front(study)
```

## 28. Deep Dive: Hyperparameters for Low-Resource Speech

When you only have 10 hours of Swahili audio, tuning is different.

**Key Hyperparameters:**
1.  **Dropout:** Needs to be much higher (0.3 - 0.5) to prevent overfitting.
2.  **SpecAugment:** Aggressive masking helps significantly.
3.  **Freezing Layers:**
    - Start with a pre-trained English Wav2Vec 2.0.
    - **Hyperparam:** How many bottom layers to freeze? (Freeze 0? 6? 12?)
    - Tuning often shows freezing the feature extractor (CNN) is crucial, but fine-tuning top Transformer layers is necessary.
4.  **Learning Rate:** Needs to be smaller for fine-tuning ($1e-5$) compared to pre-training ($1e-3$).

**Few-Shot Tuning:**
- Use **MAML (Model-Agnostic Meta-Learning)** to find initial hyperparameters that adapt quickly to new languages.

## 29. Case Study: Tuning Whisper for Code-Switching

**Problem:** "Hinglish" (Hindi + English) ASR.
**Base Model:** OpenAI Whisper Large-v2.

**Tuning Strategy:**
- **LoRA (Low-Rank Adaptation):** Fine-tuning 1.5B parameters is too slow. Tune low-rank matrices instead.
- **Hyperparameters:**
  - **Rank (r):** 8? 16? 64? (Higher = more capacity, slower).
  - **Alpha:** Scaling factor.
  - **Target Modules:** Query/Value projections? Or all linear layers?

**Results:**
- Tuning `r=16` on `q_proj` and `v_proj` yielded best results.
- Tuning all linear layers led to overfitting on the small Hinglish dataset.
- **Learning Rate:** $1e-4$ was optimal (standard fine-tuning uses $1e-5$, but LoRA allows higher LR).

## 30. Advanced: Differentiable Architecture Search (DARTS) Math

NAS is usually discrete (try architecture A, then B). **DARTS** relaxes this to a continuous space.

**Concept:**
- Construct a super-graph containing all possible operations (Conv3x3, Conv5x5, MaxPool, Identity) between nodes.
- Assign a weight $\alpha_o$ to each operation $o$.
- The output of a node is a weighted sum: $\bar{o}(x) = \sum_{o \in O} \frac{\exp(\alpha_o)}{\sum_{o'} \exp(\alpha_{o'})} o(x)$.
- We can now differentiate WER with respect to $\alpha$!

**Bilevel Optimization:**
$$\min_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha)$$
$$\text{s.t. } w^*(\alpha) = \text{argmin}_w \mathcal{L}_{train}(w, \alpha)$$

- Inner loop: Train weights $w$ (standard SGD).
- Outer loop: Update architecture $\alpha$ (gradient descent on validation loss).

**Application to Speech:**
- Used to discover optimal Convolution cells for ASR encoders.
- **Result:** Found architectures that outperform manually designed ResNets with fewer parameters.

## 31. Deep Dive: The Interaction of Hyperparameters

Hyperparameters are not independent.

1.  **Batch Size & Learning Rate:**
    - **Linear Scaling Rule:** If you double batch size, double learning rate.
    - **Square Root Rule:** Multiply LR by $\sqrt{2}$.
    - *Tuning implication:* Don't tune them independently. Tune `base_lr` and scale it dynamically based on `batch_size`.

2.  **Model Depth & Residual Scale:**
    - Deeper models (50+ layers) are harder to train.
    - **DeepNorm / ReZero:** Scale weights by $\frac{1}{\sqrt{2N}}$.
    - *Tuning:* If `num_layers` is a hyperparameter, the initialization scale must also be a function of it.

3.  **Regularization & Data Size:**
    - If you increase `SpecAugment`, you might need to decrease `Dropout`. They both provide regularization. Too much leads to underfitting.

## 32. Best Practices & Pitfalls

**Do's:**
- **Log Everything:** Use W&B / MLflow. You will forget what Config #34 was.
- **Set Random Seeds:** For reproducibility.
- **Use Log Scale:** For LR and Weight Decay. $1e-4$ to $1e-2$ is a huge range linearly, but reasonable logarithmically.
- **Monitor Gradient Norm:** If gradients explode, your LR is too high, regardless of what the tuner says.

**Don'ts:**
- **Don't Tune on Test Set:** The cardinal sin. You will overfit to the test set. Use a Validation set.
- **Don't Grid Search:** It's a waste of compute. Random Search is better. Bayesian is best.
- **Don't Ignore Defaults:** Start with SOTA defaults (e.g., from ESPnet recipes). Tune around them.
- **Don't Tune Everything:** Focus on LR, Batch Size, Regularization. Architecture tuning yields diminishing returns compared to data cleaning.

## 33. Cost-Benefit Analysis of Tuning

**Is it worth it?**

**Scenario:**
- **Baseline Model:** WER 10.0%. Training cost $100.
- **Tuned Model:** WER 9.5%. Tuning cost $2000 (20 trials).

**ROI Calculation:**
- If this is a hobby project: **No.**
- If this is a call center transcribing 1M hours/year:
  - 0.5% WER reduction = 5% fewer human corrections.
  - Human correction cost = $100/hour.
  - Savings = Huge. **Yes.**

**Green AI:**
- Hyperparameter tuning has a massive carbon footprint.
- **Mitigation:** Use Transfer Learning, Multi-Fidelity tuning, and share best configs (Model Cards).

## 34. Future Trends: LLM-driven Tuning

**AutoML-Zero:** Can we evolve the *code* of the algorithm?

**LLM as Tuner:**
- Feed the training logs (loss curves) to GPT-4.
- Ask: "The loss is oscillating. What should I change?"
- GPT-4: "Decrease learning rate by half and increase beta2."
- **Why it works:** LLMs have read millions of ML papers and GitHub issues. They understand the *physics* of training dynamics better than random search.
- **OMNI (OpenAI):** Future systems might just take data + metric and output a deployed API, handling all tuning internally.

## 35. Deep Dive: Troubleshooting Common Tuning Failures

Even with Optuna, things go wrong.

**1. The "Flatline" Loss:**
- **Symptom:** Loss stays constant from epoch 0.
- **Cause:** LR too high (gradients exploded) or too low (stuck in local minima).
- **Fix:** Tune LR on a logarithmic scale from $1e-6$ to $1e-1$.

**2. The "Divergence" Spike:**
- **Symptom:** Loss decreases, then suddenly shoots to NaN.
- **Cause:** Batch size too small for the LR, or bad data batch.
- **Fix:** Gradient Clipping (`clip_grad_norm_`). Tune clipping threshold (1.0 vs 5.0).

**3. The "Overfitting" Gap:**
- **Symptom:** Train loss 0.1, Val loss 5.0.
- **Cause:** Model too big, not enough regularization.
- **Fix:** Increase Dropout, Weight Decay, and SpecAugment.

**4. The "OOM" (Out of Memory):**
- **Symptom:** CUDA OOM error.
- **Cause:** Batch size too large.
- **Fix:** Prune trials that OOM. Ray Tune handles this by marking the trial as failed.

## 36. Deep Dive: The Mathematics of Learning Rate Schedules

Why do we need schedules?

**SGD Update:**
$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

**The Landscape:**
- Early training: Landscape is rough. Large steps help escape local valleys.
- Late training: We are near the minimum. Large steps oscillate. We need to decay $\eta$.

**Cosine Annealing:**
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{T_{cur}}{T_{max}}\pi))$$
- Smooth decay. No sharp drops.
- **Hyperparams:** $T_{max}$ (cycle length), $\eta_{min}$.

**Cyclic Learning Rates (CLR):**
- Oscillate LR between base and max.
- **Intuition:** "Pop" the model out of sharp minima (poor generalization) into flat minima (good generalization).

- **Intuition:** "Pop" the model out of sharp minima (poor generalization) into flat minima (good generalization).

## 37. Deep Dive: The Future - Quantum Hyperparameter Optimization

As classical computers hit Moore's Law limits, Quantum Computing offers a new frontier.

**Quantum Annealing:**
- D-Wave systems can solve optimization problems by finding the ground state of a Hamiltonian.
- **Application:** Finding the optimal discrete architecture (NAS) can be mapped to a QUBO (Quadratic Unconstrained Binary Optimization) problem.
- **Speedup:** Potentially exponential speedup for discrete search spaces.

**Grover's Search:**
- Can search an unstructured database of $N$ items in $O(\sqrt{N})$ time.
- **Implication:** Random search could become quadratically faster.

## 38. Ethical Considerations in Hyperparameter Tuning

Tuning is not value-neutral.

**1. Bias Amplification:**
- If you tune for global WER, the model might sacrifice accuracy on minority accents to improve the majority.
- **Fix:** Tune for **Worst-Case WER** across demographic groups, not Average WER.

**2. Energy Consumption:**
- Training a large Transformer with NAS emits as much CO2 as 5 cars in their lifetime.
- **Responsibility:** Report "CO2e" alongside WER in papers. Use Green algorithms.

**3. Accessibility:**
- Only Big Tech has the compute to tune 100B parameter models.
- **Democratization:** Release pre-tuned checkpoints and "recipes" so smaller labs don't have to re-tune from scratch.
- **Transparency:** Disclose the carbon footprint of your tuning process.

## 39. Further Reading

To dive deeper into the mathematics and systems of hyperparameter tuning, check out these seminal papers:

1.  **"Algorithms for Hyper-Parameter Optimization" (Bergstra et al., 2011):** The paper that introduced TPE and showed Random Search beats Grid Search.
2.  **"Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization" (Li et al., 2018):** The foundation of modern early-stopping strategies.
3.  **"Ray Tune: A Framework for Distributed Hyperparameter Tuning" (Liaw et al., 2018):** The system design behind scalable tuning.
4.  **"Optuna: A Next-generation Hyperparameter Optimization Framework" (Akiba et al., 2019):** Introduced the define-by-run API that we use today.
5.  **"Neural Architecture Search with Reinforcement Learning" (Zoph & Le, 2017):** The paper that started the NAS craze (and burned a lot of GPU hours).
6.  **"SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition" (Park et al., 2019):** Essential reading for speech regularization.

## 40. Summary

| Aspect | Strategy |
| :--- | :--- |
| **Search** | Bayesian Optimization (Optuna) |
| **Fidelity** | Hyperband (small data first) |
| **Architecture** | NAS (ENAS, DARTS) |
| **Parallelization** | Ray Tune (multi-GPU) |
| **Transfer** | Use pre-tuned models |
| **Early Stopping** | Median Pruner |
| **Tracking** | MLflow Registry |
| **Edge** | Multi-objective (WER + Latency) |
| **Production** | Auto-tuning pipelines on K8s |
| **Troubleshooting** | Log-scale LR, Gradient Clipping |
| **Ethics** | Tune for Worst-Case WER (Fairness) |

---

**Originally published at:** [arunbaby.com/speech-tech/0038-speech-hyperparameter-tuning](https://www.arunbaby.com/speech-tech/0038-speech-hyperparameter-tuning/)
