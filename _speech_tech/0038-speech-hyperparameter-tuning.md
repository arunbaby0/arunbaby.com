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

## 14. Summary

| Aspect | Strategy |
| :--- | :--- |
| **Search** | Bayesian Optimization (Optuna) |
| **Fidelity** | Hyperband (small data first) |
| **Architecture** | NAS (ENAS, DARTS) |
| **Parallelization** | Ray Tune (multi-GPU) |
| **Transfer** | Use pre-tuned models |

---

**Originally published at:** [arunbaby.com/speech-tech/0038-speech-hyperparameter-tuning](https://www.arunbaby.com/speech-tech/0038-speech-hyperparameter-tuning/)
