---
title: "Speech Architecture Search"
day: 21
collection: speech_tech
categories:
  - speech-tech
tags:
  - neural-architecture-search
  - automl
  - speech-recognition
  - asr
  - tts
  - model-optimization
  - nas
subdomain: "Speech Model Design"
tech_stack: [PyTorch, ESPnet, NNI, Optuna, Ray, TensorFlow, Kaldi]
scale: "1000s of architectures, 100+ GPU days, multi-lingual optimization"
companies: [Google, Amazon, Apple, Microsoft, Meta, Baidu]
related_dsa_day: 21
related_ml_day: 21
---

**Design neural architecture search systems for speech models that automatically discover optimal ASR/TTS architectures—using dynamic programming and path optimization to navigate exponential search spaces.**

## Problem Statement

Design a **Speech Architecture Search System** that:

1. **Automatically discovers** ASR/TTS architectures optimized for accuracy, latency, and size
2. **Searches efficiently** through speech-specific architecture spaces (encoders, decoders, attention)
3. **Handles speech constraints** (streaming, long sequences, variable-length inputs)
4. **Optimizes for deployment** (mobile, edge, server, different hardware)
5. **Supports multi-objective** optimization (WER, latency, params, multilingual capability)

### Functional Requirements

1. **Speech-specific search spaces:**
   - Encoder architectures (Conformer, Transformer, RNN, CNN)
   - Decoder types (CTC, RNN-T, attention-based)
   - Attention mechanisms (self-attention, cross-attention, relative positional)
   - Feature extraction configs (mel-spec, MFCC, learnable features)

2. **Search strategies:**
   - Reinforcement learning
   - Evolutionary algorithms
   - Differentiable NAS (DARTS for speech)
   - Bayesian optimization
   - Transfer from vision NAS results

3. **Performance estimation:**
   - Train on subset of data (LibriSpeech-100h vs 960h)
   - Early stopping based on validation WER
   - Weight sharing across architectures
   - WER prediction from architecture features

4. **Multi-objective optimization:**
   - WER vs latency (for real-time ASR)
   - WER vs model size (for on-device)
   - WER vs RTF (real-time factor)
   - Multi-lingual capability vs params

5. **Streaming-aware search:**
   - Architectures must support chunk-wise processing
   - Latency measured per chunk, not full utterance
   - Look-ahead constraints (for causal models)

6. **Evaluation:**
   - WER/CER on multiple test sets
   - Latency measurement on target hardware
   - Parameter count and memory footprint
   - Multi-lingual evaluation

### Non-Functional Requirements

1. **Efficiency:** Find good architecture in <50 GPU days
2. **Quality:** WER competitive with hand-designed models
3. **Generalizability:** Transfer across languages and domains
4. **Reproducibility:** Same search produces same results
5. **Practicality:** Discovered models deployable in production

## Understanding the Requirements

### Why Speech Architecture Search?

**Manual speech model design challenges:**
- Requires domain expertise (speech signal processing + deep learning)
- Hard to balance accuracy, latency, and size
- Difficult to optimize for specific hardware (mobile, server)
- Time-consuming to explore alternative designs

**Speech NAS enables:**
- Automated discovery of novel architectures
- Hardware-specific optimization (mobile, edge TPU, server GPU)
- Multi-lingual model optimization
- Systematic exploration of design space

### Speech Architecture Challenges

1. **Long sequences:** Audio is 100s-1000s of frames (vs images ~224×224)
2. **Temporal modeling:** Need strong sequential modeling (RNNs, Transformers)
3. **Streaming requirements:** Many applications need real-time processing
4. **Variable length:** Utterances vary from 1s to 60s+
5. **Multi-lingual:** Same architecture should work across languages

### The Path Optimization Connection

Just like **Unique Paths** uses DP to count paths through a grid:

| Unique Paths | Neural Arch Search | Speech Arch Search |
|--------------|-------------------|-------------------|
| m×n grid | General model space | Speech-specific space |
| Count paths | Evaluate architectures | Evaluate speech models |
| DP: paths(i,j) = paths(i-1,j) + paths(i,j-1) | DP: Build from sub-architectures | DP: Build from encoder/decoder blocks |
| O(m×n) from O(2^(m+n)) | Polynomial from exponential | Efficient from exhaustive |
| Reconstruct optimal path | Extract best architecture | Extract best speech model |

Both use **DP and path optimization** to navigate exponentially large spaces.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Speech Architecture Search System                 │
└─────────────────────────────────────────────────────────────────┘

                    Search Controller
        ┌────────────────────────────────────┐
        │  Strategy: RL / EA / DARTS         │
        │  - Propose speech architectures    │
        │  - Encoder + Decoder + Attention   │
        └──────────────┬─────────────────────┘
                       │
                ┌──────▼──────┐
                │ Speech      │
                │ Search      │
                │ Space       │
                │             │
                │ - Encoder   │
                │ - Decoder   │
                │ - Attention │
                │ - Features  │
                └──────┬──────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼────────┐ ┌──▼────┐ ┌──────▼──────┐
│ Architecture   │ │  WER  │ │ Latency     │
│ Evaluator      │ │Predict│ │ Predictor   │
│                │ │       │ │             │
│ - Train ASR/TTS│ │ - Skip│ │ - Hardware  │
│ - Measure WER  │ │   bad │ │   profile   │
│ - Measure RTF  │ │  archs│ │ - RTF est   │
└───────┬────────┘ └───────┘ └──────┬──────┘
        │                           │
        └─────────────┬─────────────┘
                      │
            ┌─────────▼────────┐
            │ Distributed      │
            │ Training         │
            │ - Worker pool    │
            │ - GPU cluster    │
            │ - Multi-task eval│
            └─────────┬────────┘
                      │
            ┌─────────▼────────┐
            │ Results Database │
            │ - Architectures  │
            │ - WER scores     │
            │ - Latency        │
            │ - Pareto front   │
            └──────────────────┘
```

### Key Components

1. **Search Controller:** Proposes speech architectures
2. **Speech Search Space:** Defines encoder/decoder/attention options
3. **Architecture Evaluator:** Trains and measures WER/latency
4. **Performance Predictors:** Estimate WER and latency without full training
5. **Distributed Training:** Parallel architecture evaluation
6. **Results Database:** Track all evaluated architectures

## Component Deep-Dives

### 1. Speech-Specific Search Space

Define search space for ASR models:

```python
from dataclasses import dataclass
from typing import List
from enum import Enum

class EncoderType(Enum):
    """Encoder architecture options."""
    CONFORMER = "conformer"
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    BLSTM = "blstm"
    CNN_LSTM = "cnn_lstm"
    CONTEXTNET = "contextnet"

class DecoderType(Enum):
    """Decoder architecture options."""
    CTC = "ctc"
    RNN_T = "rnn_t"
    ATTENTION = "attention"
    TRANSFORMER_DECODER = "transformer_decoder"

class AttentionType(Enum):
    """Attention mechanism options."""
    MULTI_HEAD = "multi_head"
    RELATIVE = "relative"
    LOCAL = "local"
    EFFICIENT = "efficient_attention"

@dataclass
class SpeechArchConfig:
    """
    Speech model architecture configuration.
    
    Similar to choosing path in Unique Paths:
    - Each choice (encoder, decoder, etc.) is like a move
    - Combination forms complete architecture (path)
    """
    # Encoder config
    encoder_type: EncoderType
    encoder_layers: int
    encoder_dim: int
    encoder_heads: int  # For attention-based encoders
    
    # Decoder config
    decoder_type: DecoderType
    decoder_layers: int
    decoder_dim: int
    
    # Attention config
    attention_type: AttentionType
    attention_dim: int
    
    # Feature extraction
    n_mels: int
    
    def count_parameters(self) -> int:
        """Estimate parameter count."""
        # Simplified estimation
        encoder_params = self.encoder_layers * (self.encoder_dim ** 2) * 4
        decoder_params = self.decoder_layers * (self.decoder_dim ** 2) * 4
        return encoder_params + decoder_params
    
    def estimate_flops(self, sequence_length: int = 1000) -> int:
        """Estimate FLOPs for sequence of given length."""
        # Encoder (attention is O(L^2 * D))
        encoder_flops = sequence_length ** 2 * self.encoder_dim * self.encoder_layers
        
        # Decoder
        decoder_flops = sequence_length * self.decoder_dim * self.decoder_layers
        
        return encoder_flops + decoder_flops


class SpeechSearchSpace:
    """
    Search space for speech architectures.
    
    Similar to grid in Unique Paths:
    - Dimensions: encoder × decoder × attention × features
    - Each dimension has multiple choices
    - Total space is exponential
    """
    
    def __init__(self):
        # Define choices
        self.encoder_types = list(EncoderType)
        self.decoder_types = list(DecoderType)
        self.encoder_layer_options = [4, 6, 8, 12]
        self.encoder_dim_options = [256, 512, 768]
        self.decoder_layer_options = [1, 2, 4]
        
    def count_total_architectures(self) -> int:
        """
        Count total architectures (like counting paths).
        """
        count = (
            len(self.encoder_types) *
            len(self.encoder_layer_options) *
            len(self.encoder_dim_options) *
            len(self.decoder_types) *
            len(self.decoder_layer_options)
        )
        return count
    
    def sample_random_architecture(self) -> SpeechArchConfig:
        """Sample random architecture from space."""
        import random
        
        return SpeechArchConfig(
            encoder_type=random.choice(self.encoder_types),
            encoder_layers=random.choice(self.encoder_layer_options),
            encoder_dim=random.choice(self.encoder_dim_options),
            encoder_heads=8,  # Fixed for simplicity
            decoder_type=random.choice(self.decoder_types),
            decoder_layers=random.choice(self.decoder_layer_options),
            decoder_dim=256,  # Fixed
            attention_type=AttentionType.MULTI_HEAD,
            attention_dim=256,
            n_mels=80
        )
```

### 2. Architecture Evaluation

```python
import torch
import torch.nn as nn

def build_speech_model(config: SpeechArchConfig) -> nn.Module:
    """
    Build speech model from configuration.
    
    Args:
        config: Architecture configuration
        
    Returns:
        PyTorch model
    """
    # This would integrate with ESPnet or custom implementation
    # Simplified example:
    
    if config.encoder_type == EncoderType.CONFORMER:
        from espnet.nets.pytorch_backend.conformer.encoder import Encoder
        encoder = Encoder(
            idim=config.n_mels,
            attention_dim=config.encoder_dim,
            attention_heads=config.encoder_heads,
            linear_units=config.encoder_dim * 4,
            num_blocks=config.encoder_layers
        )
    elif config.encoder_type == EncoderType.TRANSFORMER:
        from espnet.nets.pytorch_backend.transformer.encoder import Encoder
        encoder = Encoder(
            idim=config.n_mels,
            attention_dim=config.encoder_dim,
            attention_heads=config.encoder_heads,
            linear_units=config.encoder_dim * 4,
            num_blocks=config.encoder_layers
        )
    else:
        # LSTM, CNN-LSTM, etc.
        encoder = create_encoder(config)
    
    # Build decoder
    if config.decoder_type == DecoderType.CTC:
        decoder = nn.Linear(config.encoder_dim, num_tokens)
    elif config.decoder_type == DecoderType.RNN_T:
        decoder = create_rnnt_decoder(config)
    else:
        decoder = create_attention_decoder(config)
    
    # Combine into full model
    model = SpeechModel(encoder=encoder, decoder=decoder)
    
    return model


def evaluate_speech_architecture(
    config: SpeechArchConfig,
    train_subset: str = "librispeech-100h",
    val_subset: str = "librispeech-dev",
    max_epochs: int = 20
) -> Dict:
    """
    Evaluate speech architecture.
    
    Args:
        config: Architecture to evaluate
        train_subset: Training data subset
        val_subset: Validation data
        max_epochs: Max training epochs
        
    Returns:
        Dictionary with WER, latency, params, etc.
    """
    # Build model
    model = build_speech_model(config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Train
    best_wer = train_and_evaluate(
        model,
        train_data=train_subset,
        val_data=val_subset,
        max_epochs=max_epochs
    )
    
    # Measure latency
    latency_ms = measure_inference_latency(model)
    
    # Measure RTF (real-time factor)
    rtf = measure_rtf(model)
    
    return {
        "config": config,
        "wer": best_wer,
        "latency_ms": latency_ms,
        "rtf": rtf,
        "params": num_params,
        "flops": config.estimate_flops()
    }
```

### 3. Search Strategy for Speech

```python
class SpeechNASController:
    """
    NAS controller for speech architectures.
    
    Uses DP-like building:
    - Build encoder → choose decoder → optimize jointly
    - Like building path: choose direction at each step
    """
    
    def __init__(self, search_space: SpeechSearchSpace):
        self.search_space = search_space
        self.evaluated_archs = {}
        self.best_archs = []
    
    def search_with_evolutionary(self, population_size: int = 20, generations: int = 50):
        """
        Evolutionary search for speech architectures.
        
        Similar to exploring paths in Unique Paths:
        - Generate population (multiple paths)
        - Evaluate fitness (WER)
        - Mutate and crossover (create new paths)
        - Select best (optimal paths)
        """
        # Initialize population
        population = [
            self.search_space.sample_random_architecture()
            for _ in range(population_size)
        ]
        
        for generation in range(generations):
            # Evaluate all architectures
            fitness_scores = []
            
            for arch in population:
                if encode_architecture(arch) not in self.evaluated_archs:
                    result = evaluate_speech_architecture(arch)
                    self.evaluated_archs[encode_architecture(arch)] = result
                    fitness = 1.0 / (result['wer'] + 0.01)  # Lower WER = higher fitness
                else:
                    result = self.evaluated_archs[encode_architecture(arch)]
                    fitness = 1.0 / (result['wer'] + 0.01)
                
                fitness_scores.append((arch, fitness, result))
            
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Track best
            self.best_archs.append(fitness_scores[0])
            
            # Selection: keep top 50%
            survivors = [arch for arch, _, _ in fitness_scores[:population_size // 2]]
            
            # Mutation and crossover to create next generation
            offspring = []
            
            while len(offspring) < population_size // 2:
                # Select parents
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutation
                if random.random() < 0.3:
                    child = self._mutate(child)
                
                offspring.append(child)
            
            # New population
            population = survivors + offspring
        
        # Return best architecture found
        best = max(self.best_archs, key=lambda x: x[1])
        return best[0], best[2]
    
    def _crossover(self, arch1: SpeechArchConfig, arch2: SpeechArchConfig) -> SpeechArchConfig:
        """
        Crossover two architectures.
        
        Randomly inherit components from parents.
        """
        return SpeechArchConfig(
            encoder_type=random.choice([arch1.encoder_type, arch2.encoder_type]),
            encoder_layers=random.choice([arch1.encoder_layers, arch2.encoder_layers]),
            encoder_dim=random.choice([arch1.encoder_dim, arch2.encoder_dim]),
            encoder_heads=random.choice([arch1.encoder_heads, arch2.encoder_heads]),
            decoder_type=random.choice([arch1.decoder_type, arch2.decoder_type]),
            decoder_layers=random.choice([arch1.decoder_layers, arch2.decoder_layers]),
            decoder_dim=random.choice([arch1.decoder_dim, arch2.decoder_dim]),
            attention_type=random.choice([arch1.attention_type, arch2.attention_type]),
            attention_dim=random.choice([arch1.attention_dim, arch2.attention_dim]),
            n_mels=random.choice([arch1.n_mels, arch2.n_mels])
        )
    
    def _mutate(self, arch: SpeechArchConfig) -> SpeechArchConfig:
        """
        Mutate architecture.
        
        Randomly change one component.
        """
        mutation_choice = random.randint(0, 3)
        
        new_arch = SpeechArchConfig(**arch.__dict__)
        
        if mutation_choice == 0:
            # Mutate encoder
            new_arch.encoder_layers = random.choice(self.search_space.encoder_layer_options)
        elif mutation_choice == 1:
            # Mutate encoder dim
            new_arch.encoder_dim = random.choice(self.search_space.encoder_dim_options)
        elif mutation_choice == 2:
            # Mutate decoder
            new_arch.decoder_layers = random.choice(self.search_space.decoder_layer_options)
        else:
            # Mutate encoder type
            new_arch.encoder_type = random.choice(self.search_space.encoder_types)
        
        return new_arch
```

### 4. Multi-Objective Optimization

```python
class MultiObjectiveSpeechNAS:
    """
    Multi-objective NAS for speech.
    
    Optimize for:
    - WER (minimize)
    - Latency (minimize)
    - Model size (minimize)
    
    Find Pareto frontier of optimal trade-offs.
    """
    
    def __init__(self, search_space: SpeechSearchSpace):
        self.search_space = search_space
        self.pareto_front = []
    
    def search(self, num_candidates: int = 100):
        """Search for Pareto-optimal architectures."""
        evaluated = []
        
        for i in range(num_candidates):
            # Sample architecture
            arch = self.search_space.sample_random_architecture()
            
            # Evaluate
            result = evaluate_speech_architecture(arch)
            
            evaluated.append({
                "arch": arch,
                "wer": result['wer'],
                "latency": result['latency_ms'],
                "params": result['params']
            })
        
        # Find Pareto frontier
        self.pareto_front = self._compute_pareto_front(evaluated)
        
        return self.pareto_front
    
    def _compute_pareto_front(self, candidates: List[Dict]) -> List[Dict]:
        """
        Compute Pareto frontier.
        
        An architecture is Pareto-optimal if no other architecture
        is better in all objectives.
        """
        pareto = []
        
        for i, cand1 in enumerate(candidates):
            is_dominated = False
            
            for j, cand2 in enumerate(candidates):
                if i == j:
                    continue
                
                # Check if cand2 dominates cand1
                # (better or equal in all objectives, strictly better in at least one)
                if (cand2['wer'] <= cand1['wer'] and
                    cand2['latency'] <= cand1['latency'] and
                    cand2['params'] <= cand1['params'] and
                    (cand2['wer'] < cand1['wer'] or
                     cand2['latency'] < cand1['latency'] or
                     cand2['params'] < cand1['params'])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto.append(cand1)
        
        return pareto
    
    def select_for_target(self, max_latency_ms: float, max_params: int) -> Optional[Dict]:
        """
        Select best architecture meeting constraints.
        
        Args:
            max_latency_ms: Maximum acceptable latency
            max_params: Maximum model size
            
        Returns:
            Best architecture meeting constraints, or None
        """
        candidates = [
            arch for arch in self.pareto_front
            if arch['latency'] <= max_latency_ms and arch['params'] <= max_params
        ]
        
        if not candidates:
            return None
        
        # Return lowest WER among candidates
        return min(candidates, key=lambda x: x['wer'])
```

## Scaling Strategies

### Efficient Evaluation

**1. Progressive training:**

```python
def progressive_evaluation(arch: SpeechArchConfig):
    """
    Evaluate architecture progressively.
    
    Start with small dataset/short training.
    Only continue if promising.
    """
    # Stage 1: Train on LibriSpeech-100h for 5 epochs
    wer_stage1 = quick_train(arch, data="librispeech-100h", epochs=5)
    
    if wer_stage1 > 0.20:  # 20% WER threshold
        return {"wer": wer_stage1, "early_stopped": True}
    
    # Stage 2: Train on LibriSpeech-100h for 20 epochs
    wer_stage2 = quick_train(arch, data="librispeech-100h", epochs=20)
    
    if wer_stage2 > 0.10:
        return {"wer": wer_stage2, "early_stopped": True}
    
    # Stage 3: Full training on LibriSpeech-960h
    wer_final = full_train(arch, data="librispeech-960h", epochs=100)
    
    return {"wer": wer_final, "early_stopped": False}
```

**2. Weight sharing (supernet for speech):**

```python
class SpeechSuperNet(nn.Module):
    """
    Super-network for speech NAS.
    
    Contains all possible operations.
    Different architectures share weights.
    """
    
    def __init__(self, search_space: SpeechSearchSpace):
        super().__init__()
        
        # Create all encoder options
        self.encoders = nn.ModuleDict({
            enc_type.value: create_encoder(enc_type, max_layers=12, max_dim=768)
            for enc_type in EncoderType
        })
        
        # Create all decoder options
        self.decoders = nn.ModuleDict({
            dec_type.value: create_decoder(dec_type)
            for dec_type in DecoderType
        })
    
    def forward(self, audio_features: torch.Tensor, arch: SpeechArchConfig):
        """Forward with specific architecture."""
        # Select encoder
        encoder = self.encoders[arch.encoder_type.value]
        
        # Select decoder
        decoder = self.decoders[arch.decoder_type.value]
        
        # Forward pass
        encoder_out = encoder(audio_features)
        output = decoder(encoder_out)
        
        return output
```

## Real-World Case Study: Google's Speech NAS

### Google's Approach for Mobile ASR

**Goal:** Find ASR architecture for on-device deployment with <100ms latency.

**Search space:**
- Encoder: RNN, LSTM, GRU, Conformer variants
- Layers: 2-8
- Hidden dim: 128-512
- Decoder: CTC, RNN-T

**Search strategy:**
- Reinforcement learning controller
- Multi-objective: WER + latency + model size
- Progressive training (100h → 960h dataset)

**Results:**
- **Discovered architecture:** 4-layer Conformer + RNN-T
- **WER:** 5.2% on LibriSpeech test-clean (vs 6.1% baseline)
- **Latency:** 85ms on Pixel 6 (vs 120ms baseline)
- **Size:** 45M params (vs 80M baseline LSTM)
- **Search cost:** 80 GPU days (vs months of manual tuning)

**Key insights:**
- Conformer with fewer layers beats deep LSTM
- RNN-T decoder better latency than attention for streaming
- Smaller models with better architecture beat larger hand-designed ones

### Lessons Learned

1. **Speech-specific constraints matter:** Streaming, variable length, long sequences
2. **Multi-objective is essential:** Can't just optimize WER
3. **Progressive evaluation saves compute:** 80% of candidates filtered early
4. **Transfer works:** ImageNet NAS insights transfer to speech (depth vs width)
5. **Hardware-in-the-loop:** Measure latency on actual target device

## Cost Analysis

### NAS vs Manual Design

| Approach | Time | GPU Cost | Quality (WER) | Notes |
|----------|------|----------|--------------|-------|
| Manual design | 6 months | 50 GPU days | 6.5% | Expert-dependent |
| Random search | N/A | 500 GPU days | 7.0% | Baseline |
| Evolutionary NAS | 2 months | 100 GPU days | 5.8% | Robust |
| RL-based NAS | 1 month | 80 GPU days | 5.2% | Google's approach |
| DARTS for speech | 2 weeks | 10 GPU days | 6.0% | Fast but less stable |
| Transfer + fine-tune | 1 week | 5 GPU days | 5.5% | Use vision NAS results |

**ROI:**
- Manual: $120K (engineer time) + $15K (GPUs) = $135K
- NAS: $40K (engineer time) + $24K (GPUs) = $64K
- **Savings:** $71K + better model + faster iteration

## Advanced Topics

### 1. Multi-Lingual NAS

Search for architectures that work across languages:

```python
def multi_lingual_nas(languages: List[str] = ["en", "zh", "es"]):
    """
    Search for architecture that works well across languages.
    
    Fitness = average WER across all languages.
    """
    def evaluate_multilingual(arch: SpeechArchConfig) -> float:
        wers = []
        
        for lang in languages:
            wer = train_and_evaluate(
                arch,
                train_data=f"common_voice_{lang}",
                val_data=f"common_voice_{lang}_dev"
            )
            wers.append(wer)
        
        # Average WER across languages
        return sum(wers) / len(wers)
    
    # Search with multi-lingual fitness
    # ... (use evolutionary or RL search)
```

### 2. Streaming-Aware NAS

Optimize for streaming ASR:

```python
def streaming_aware_evaluation(arch: SpeechArchConfig) -> Dict:
    """
    Evaluate architecture for streaming capability.
    
    Metrics:
    - Per-chunk latency (not full utterance)
    - Look-ahead requirement
    - Chunk size vs WER trade-off
    """
    model = build_speech_model(arch)
    
    # Test streaming performance
    chunk_size_ms = 100  # 100ms chunks
    
    chunk_latency = measure_chunk_latency(model, chunk_size_ms)
    streaming_wer = evaluate_streaming_wer(model, chunk_size_ms)
    
    return {
        "chunk_latency_ms": chunk_latency,
        "streaming_wer": streaming_wer,
        "supports_streaming": chunk_latency < chunk_size_ms
    }
```

### 3. Transfer from Vision NAS

Leverage insights from ImageNet NAS:

```python
def transfer_vision_to_speech(vision_arch_config):
    """
    Transfer successful vision architectures to speech.
    
    Example: EfficientNet principles → EfficientConformer
    - Depth scaling
    - Width scaling
    - Compound scaling
    """
    # Extract architectural principles
    depth_factor = vision_arch_config.depth_coefficient
    width_factor = vision_arch_config.width_coefficient
    
    # Apply to speech
    speech_config = SpeechArchConfig(
        encoder_type=EncoderType.CONFORMER,
        encoder_layers=int(6 * depth_factor),
        encoder_dim=int(256 * width_factor),
        encoder_heads=8,
        decoder_type=DecoderType.RNN_T,
        decoder_layers=2,
        decoder_dim=256,
        attention_type=AttentionType.RELATIVE,
        attention_dim=256,
        n_mels=80
    )
    
    return speech_config
```

## Monitoring & Debugging

### Key Metrics

**Search Progress:**
- Best WER found so far vs iterations
- Pareto frontier evolution
- Architecture diversity (entropy of designs explored)
- GPU utilization during search

**Architecture Analysis:**
- Most common encoder/decoder types in top performers
- Depth vs width trade-offs
- Correlation between architecture features and WER

**Resource Tracking:**
- Total GPU hours consumed
- Average training time per architecture
- Early stopping rate (% of archs stopped early)

### Debugging Tools

- Visualize architecture graphs
- Compare top-N architectures side-by-side
- Ablation studies (which components matter most?)
- Error analysis (where do discovered archs fail?)

## Key Takeaways

✅ **Speech NAS automates** architecture design for ASR/TTS models

✅ **Search space is exponential** - like paths in a grid, need smart search

✅ **DP and smart search** make NAS practical - from infeasible to 50-100 GPU days

✅ **Multi-objective optimization** essential - WER, latency, size must be balanced

✅ **Progressive evaluation** saves compute - filter bad candidates early

✅ **Weight sharing** (supernet) enables evaluating 1000s of architectures

✅ **Speech-specific constraints** - streaming, variable length, multi-lingual

✅ **Transfer from vision** accelerates speech NAS

✅ **Hardware-aware search** critical for deployment

✅ **Same DP pattern** as Unique Paths - build optimal solution from sub-solutions

### Connection to Thematic Link: Dynamic Programming and Path Optimization

All three Day 21 topics use **DP to optimize paths through exponential spaces**:

**DSA (Unique Paths):**
- Navigate m×n grid using DP
- Recurrence: paths(i,j) = paths(i-1,j) + paths(i,j-1)
- Build solution from optimal sub-solutions

**ML System Design (Neural Architecture Search):**
- Navigate exponential architecture space
- Use DP/RL/gradient methods to find optimal
- Build full model from optimal components

**Speech Tech (Speech Architecture Search):**
- Navigate encoder×decoder×attention space
- Use DP-inspired search to find optimal speech models
- Build ASR/TTS from optimal sub-architectures

The **unifying principle**: decompose exponentially large search spaces into manageable subproblems, solve optimally using DP or DP-inspired methods, and construct the best overall solution.

---

**Originally published at:** [arunbaby.com/speech-tech/0021-speech-architecture-search](https://www.arunbaby.com/speech-tech/0021-speech-architecture-search/)

*If you found this helpful, consider sharing it with others who might benefit.*


