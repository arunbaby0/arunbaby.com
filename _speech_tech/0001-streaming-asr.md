---
title: "Streaming ASR Architecture"
day: 1
collection: speech_tech
categories:
  - speech-tech
tags:
  - asr
  - streaming
  - real-time
subdomain: Automatic Speech Recognition
tech_stack: [RNN-T, CTC, Conformer]
latency_requirement: "< 200ms first token"
scale: "10M+ concurrent streams"
companies: [Google, Amazon, Apple]
related_dsa_day: 1
related_ml_day: 1
---

**Why batch ASR won't work for voice assistants, and how streaming models transcribe speech as you speak in under 200ms.**

## Introduction

Every time you say "Hey Google" or ask Alexa a question, you're interacting with a streaming Automatic Speech Recognition (ASR) system. Unlike traditional batch ASR systems that wait for you to finish speaking before transcribing, streaming ASR must:

- Emit words **as you speak** (not after)
- Maintain **< 200ms latency** for first token
- Handle **millions of concurrent audio streams**
- Work reliably in **noisy environments**
- Run on both **cloud and edge devices**
- Adapt to different **accents and speaking styles**

This is fundamentally different from batch models like OpenAI's Whisper, which achieve amazing accuracy but require the entire utterance before processing. For interactive voice assistants, this delay is unacceptable users expect immediate feedback.

**What you'll learn:**
- Why streaming requires different model architectures
- RNN-Transducer (RNN-T) and CTC for streaming
- How to maintain state across audio chunks
- Latency optimization techniques (quantization, pruning, caching)
- Scaling to millions of concurrent streams
- Cold start and speaker adaptation
- Real production systems (Google, Amazon, Apple)

---

## Problem Definition

Design a production streaming ASR system that transcribes speech in real-time for a voice assistant platform.

### Functional Requirements

1. **Streaming Transcription**
   - Output tokens incrementally as user speaks
   - No need to wait for end of utterance
   - Partial results updated continuously

2. **Low Latency**
   - **First token latency:** < 200ms (time from start of speech to first word)
   - **Per-token latency:** < 100ms (time between subsequent words)
   - **End-of-utterance latency:** < 500ms (finalized transcript)

3. **No Future Context**
   - Cannot "look ahead" into future audio (non-causal)
   - Limited right context window (e.g., 320ms)
   - Must work with incomplete information

4. **State Management**
   - Maintain conversational context across chunks
   - Remember acoustic and linguistic state
   - Handle variable-length inputs

5. **Multi-Language Support**
   - 20+ languages
   - Automatic language detection
   - Code-switching (mixing languages)

### Non-Functional Requirements

1. **Accuracy**
   - **Clean speech:** WER < 5% (Word Error Rate)
   - **Noisy speech:** WER < 15%
   - **Accented speech:** WER < 10%
   - **Far-field:** WER < 20%

2. **Throughput**
   - 10M concurrent audio streams globally
   - 10k QPS per regional cluster
   - Auto-scaling based on load

3. **Availability**
   - 99.99% uptime (< 1 hour downtime/year)
   - Graceful degradation on failures
   - Multi-region failover

4. **Cost Efficiency**
   - < $0.01 per minute of audio (cloud)
   - < 100ms inference time on edge devices
   - GPU/CPU optimization

### Out of Scope

- Audio storage and archival
- Speaker diarization (who is speaking)
- Speech translation
- Emotion/sentiment detection
- Voice biometric authentication

---

## Streaming vs Batch ASR: Key Differences

### Batch ASR (e.g., Whisper)

```python
def batch_asr(audio):
    # Wait for complete audio
    complete_audio = wait_for_end_of_speech(audio)
    
    # Process entire sequence at once
    # Can use bidirectional models, look at future context
    features = extract_features(complete_audio)
    transcript = model(features)  # Has access to all frames
    
    return transcript

# Latency: duration + processing time
# For 10-second audio: 10 seconds + 2 seconds = 12 seconds
```

**Pros:**
- Can use future context → better accuracy
- Simpler architecture (no state management)
- Can use attention over full sequence

**Cons:**
- High latency (must wait for end)
- Poor user experience for voice assistants
- Cannot provide real-time feedback

### Streaming ASR

```python
def streaming_asr(audio_stream):
    state = initialize_state()
    
    for audio_chunk in audio_stream:  # Process 100ms chunks
        # Can only look at past + limited future
        features = extract_features(audio_chunk)
        tokens, state = model(features, state)  # Causal processing
        
        if tokens:
            yield tokens  # Emit immediately
    
    # Finalize
    final_tokens = finalize(state)
    yield final_tokens

# Latency: ~200ms for first token, ~100ms per subsequent token
# For 10-second audio: 200ms + (tokens * 100ms) ≈ 2-3 seconds total
```

**Pros:**
- Low latency (immediate feedback)
- Better user experience
- Can interrupt/correct in real-time

**Cons:**
- More complex (state management)
- Slightly lower accuracy (no full future context)
- Harder to train

---

## Architecture Overview

```
Audio Input (100ms chunks @ 16kHz)
    ↓
Voice Activity Detection (VAD)
    ├─ Speech detected → Continue
    └─ Silence detected → Skip processing
    ↓
Feature Extraction
    ├─ Mel Filterbank (80 dims)
    ├─ Normalization
    └─ Delta features (optional)
    ↓
Streaming Acoustic Model
    ├─ Encoder (Conformer/RNN)
    ├─ Prediction Network
    └─ Joint Network
    ↓
Decoder (Beam Search)
    ├─ Language Model Fusion
    ├─ Beam Management
    └─ Token Emission
    ↓
Post-Processing
    ├─ Punctuation
    ├─ Capitalization
    └─ Inverse Text Normalization
    ↓
Transcription Output
```

---

## Component 1: Voice Activity Detection (VAD)

### Why VAD is Critical

**Problem:** Processing silence wastes 50-70% of compute.

**Solution:** Filter out non-speech audio before expensive ASR processing.

```python
# Without VAD
total_audio = 10 seconds
speech = 3 seconds (30%)
silence = 7 seconds (70% wasted compute)

# With VAD
processed_audio = 3 seconds (save 70% compute)
```

### VAD Approaches

**Option 1: Energy-Based (Simple)**

```python
def energy_vad(audio_chunk, threshold=0.01):
    """
    Classify based on audio energy
    """
    energy = np.sum(audio_chunk ** 2) / len(audio_chunk)
    return energy > threshold
```

**Pros:** Fast (< 1ms), no model needed  
**Cons:** Fails in noisy environments, no semantic understanding

**Option 2: ML-Based (Robust)**

```python
class SileroVAD:
    """
    Using Silero VAD (open-source, production-ready)
    Model size: 1MB, Latency: ~2ms
    """
    def __init__(self):
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        self.get_speech_timestamps = self.utils[0]
    
    def is_speech(self, audio, sampling_rate=16000):
        """
        Args:
            audio: torch.Tensor, shape (samples,)
            sampling_rate: int
        
        Returns:
            bool: True if speech detected
        """
        speech_timestamps = self.get_speech_timestamps(
            audio, 
            self.model,
            sampling_rate=sampling_rate,
            threshold=0.5
        )
        
        return len(speech_timestamps) > 0

# Usage
vad = SileroVAD()

for audio_chunk in audio_stream:
    if vad.is_speech(audio_chunk):
        # Process with ASR
        process_asr(audio_chunk)
    else:
        # Skip, save compute
        continue
```

**Pros:** Robust to noise, semantic understanding  
**Cons:** Adds 2ms latency, requires model

### Production VAD Pipeline

```python
class ProductionVAD:
    def __init__(self):
        self.vad = SileroVAD()
        self.speech_buffer = []
        self.silence_frames = 0
        self.max_silence_frames = 30  # 300ms of silence
    
    def process_chunk(self, audio_chunk):
        """
        Buffer management with hysteresis
        """
        is_speech = self.vad.is_speech(audio_chunk)
        
        if is_speech:
            # Reset silence counter
            self.silence_frames = 0
            
            # Add to buffer
            self.speech_buffer.append(audio_chunk)
            
            return 'speech', audio_chunk
        
        else:
            # Increment silence counter
            self.silence_frames += 1
            
            # Keep buffering for a bit (hysteresis)
            if self.silence_frames < self.max_silence_frames:
                self.speech_buffer.append(audio_chunk)
                return 'speech', audio_chunk
            
            else:
                # End of utterance
                if self.speech_buffer:
                    complete_utterance = np.concatenate(self.speech_buffer)
                    self.speech_buffer = []
                    return 'end_of_utterance', complete_utterance
                
                return 'silence', None
```

**Key design decisions:**
- **Hysteresis:** Continue processing for 300ms after silence to avoid cutting off speech
- **Buffering:** Accumulate audio for end-of-utterance finalization
- **State management:** Track silence duration to detect utterance boundaries

---

## Component 2: Feature Extraction

### Log Mel Filterbank Features

**Why Mel scale?** Human perception of pitch is logarithmic, not linear.

```python
def extract_mel_features(audio, sr=16000, n_mels=80):
    """
    Extract 80-dimensional log mel filterbank features
    
    Args:
        audio: np.array, shape (samples,)
        sr: sampling rate (Hz)
        n_mels: number of mel bands
    
    Returns:
        features: np.array, shape (time, n_mels)
    """
    # Frame audio: 25ms window, 10ms stride
    frame_length = int(0.025 * sr)  # 400 samples
    hop_length = int(0.010 * sr)     # 160 samples
    
    # Short-Time Fourier Transform
    stft = librosa.stft(
        audio,
        n_fft=512,
        hop_length=hop_length,
        win_length=frame_length,
        window='hann'
    )
    
    # Magnitude spectrum
    magnitude = np.abs(stft)
    
    # Mel filterbank
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=512,
        n_mels=n_mels,
        fmin=0,
        fmax=sr/2
    )
    
    # Apply mel filters
    mel_spec = np.dot(mel_basis, magnitude)
    
    # Log compression (humans perceive loudness logarithmically)
    log_mel = np.log(mel_spec + 1e-6)
    
    # Transpose to (time, frequency)
    return log_mel.T
```

**Output:** 100 frames per second (one every 10ms), each with 80 dimensions

### Normalization

```python
def normalize_features(features, mean=None, std=None):
    """
    Normalize to zero mean, unit variance
    
    Can use global statistics or per-utterance
    """
    if mean is None:
        mean = np.mean(features, axis=0, keepdims=True)
    if std is None:
        std = np.std(features, axis=0, keepdims=True)
    
    normalized = (features - mean) / (std + 1e-6)
    return normalized
```

**Global vs Per-Utterance:**
- **Global normalization:** Use statistics from training data (faster, more stable)
- **Per-utterance normalization:** Adapt to current speaker/environment (better for diverse conditions)

### SpecAugment (Training Only)

```python
def spec_augment(features, time_mask_max=30, freq_mask_max=10):
    """
    Data augmentation for training
    Randomly mask time and frequency bands
    """
    # Time masking
    t_mask_len = np.random.randint(0, time_mask_max)
    t_mask_start = np.random.randint(0, features.shape[0] - t_mask_len)
    features[t_mask_start:t_mask_start+t_mask_len, :] = 0
    
    # Frequency masking
    f_mask_len = np.random.randint(0, freq_mask_max)
    f_mask_start = np.random.randint(0, features.shape[1] - f_mask_len)
    features[:, f_mask_start:f_mask_start+f_mask_len] = 0
    
    return features
```

**Impact:** Improves robustness by 10-20% relative WER reduction

---

## Component 3: Streaming Acoustic Models

### RNN-Transducer (RNN-T)

**Why RNN-T for streaming?**
1. **Naturally causal:** Doesn't need future frames
2. **Emits tokens dynamically:** Can output 0, 1, or multiple tokens per frame
3. **No external alignment:** Learns alignment jointly with transcription

**Architecture:**

```
     Encoder (processes audio)
           ↓
     h_enc[t] (acoustic embedding)
           ↓
     Prediction Network (processes previous tokens)
           ↓
     h_pred[u] (linguistic embedding)
           ↓
     Joint Network (combines both)
           ↓
     Softmax over vocabulary + blank
```

**Implementation:**

```python
import torch
import torch.nn as nn

class StreamingRNNT(nn.Module):
    def __init__(self, vocab_size=1000, enc_dim=512, pred_dim=256, joint_dim=512):
        super().__init__()
        
        # Encoder: audio features → acoustic representation
        self.encoder = ConformerEncoder(
            input_dim=80,
            output_dim=enc_dim,
            num_layers=18,
            num_heads=8
        )
        
        # Prediction network: previous tokens → linguistic representation
        self.prediction_net = nn.LSTM(
            input_size=vocab_size,
            hidden_size=pred_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Joint network: combine acoustic + linguistic
        self.joint_net = nn.Sequential(
            nn.Linear(enc_dim + pred_dim, joint_dim),
            nn.Tanh(),
            nn.Linear(joint_dim, vocab_size + 1)  # +1 for blank token
        )
        
        self.blank_idx = vocab_size
    
    def forward(self, audio_features, prev_tokens, encoder_state=None, predictor_state=None):
        """
        Args:
            audio_features: (batch, time, 80)
            prev_tokens: (batch, seq_len)
            encoder_state: hidden state from previous chunk
            predictor_state: (h, c) from previous tokens
        
        Returns:
            logits: (batch, time, seq_len, vocab_size+1)
            new_encoder_state: updated encoder state
            new_predictor_state: updated predictor state
        """
        # Encode audio
        h_enc, new_encoder_state = self.encoder(audio_features, encoder_state)
        # h_enc: (batch, time, enc_dim)
        
        # Encode previous tokens
        # Convert tokens to one-hot
        prev_tokens_onehot = F.one_hot(prev_tokens, num_classes=self.prediction_net.input_size)
        h_pred, new_predictor_state = self.prediction_net(
            prev_tokens_onehot.float(),
            predictor_state
        )
        # h_pred: (batch, seq_len, pred_dim)
        
        # Joint network: combine all pairs of (time, token_history)
        # Expand dimensions for broadcasting
        h_enc_exp = h_enc.unsqueeze(2)  # (batch, time, 1, enc_dim)
        h_pred_exp = h_pred.unsqueeze(1)  # (batch, 1, seq_len, pred_dim)
        
        # Concatenate
        h_joint = torch.cat([
            h_enc_exp.expand(-1, -1, h_pred.size(1), -1),
            h_pred_exp.expand(-1, h_enc.size(1), -1, -1)
        ], dim=-1)
        # h_joint: (batch, time, seq_len, enc_dim+pred_dim)
        
        # Project to vocabulary
        logits = self.joint_net(h_joint)
        # logits: (batch, time, seq_len, vocab_size+1)
        
        return logits, new_encoder_state, new_predictor_state
```

### Conformer Encoder

**Why Conformer?** Combines convolution (local patterns) + self-attention (long-range dependencies)

```python
class ConformerEncoder(nn.Module):
    def __init__(self, input_dim=80, output_dim=512, num_layers=18, num_heads=8):
        super().__init__()
        
        # Subsampling: 4x downsampling to reduce sequence length
        self.subsampling = Conv2dSubsampling(input_dim, output_dim, factor=4)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(output_dim, num_heads) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x, state=None):
        # x: (batch, time, input_dim)
        
        # Subsampling
        x = self.subsampling(x)
        # x: (batch, time//4, output_dim)
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x, state = block(x, state)
        
        return x, state

class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        
        # Feed-forward module 1
        self.ff1 = FeedForwardModule(dim)
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(dim, num_heads)
        
        # Convolution module
        self.conv = ConvolutionModule(dim, kernel_size=31)
        
        # Feed-forward module 2
        self.ff2 = FeedForwardModule(dim)
        
        # Layer norms
        self.norm_ff1 = nn.LayerNorm(dim)
        self.norm_att = nn.LayerNorm(dim)
        self.norm_conv = nn.LayerNorm(dim)
        self.norm_ff2 = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
    
    def forward(self, x, state=None):
        # Feed-forward 1 (half-step residual)
        residual = x
        x = self.norm_ff1(x)
        x = residual + 0.5 * self.ff1(x)
        
        # Self-attention
        residual = x
        x = self.norm_att(x)
        x, state = self.attention(x, state)
        x = residual + x
        
        # Convolution
        residual = x
        x = self.norm_conv(x)
        x = self.conv(x)
        x = residual + x
        
        # Feed-forward 2 (half-step residual)
        residual = x
        x = self.norm_ff2(x)
        x = residual + 0.5 * self.ff2(x)
        
        # Final norm
        x = self.norm_out(x)
        
        return x, state
```

**Key features:**
- **Macaron-style:** Feed-forward at both beginning and end
- **Depthwise convolution:** Captures local patterns efficiently
- **Relative positional encoding:** Better for variable-length sequences

### Streaming Constraints

**Problem:** Self-attention in Conformer uses entire sequence → not truly streaming

**Solution:** Limited lookahead window

```python
class StreamingAttention(nn.Module):
    def __init__(self, dim, num_heads, left_context=1000, right_context=32):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.left_context = left_context   # Look at past 10 seconds
        self.right_context = right_context  # Look ahead 320ms
    
    def forward(self, x, cache=None):
        # x: (batch, time, dim)
        
        if cache is not None:
            # Concatenate with cached past frames
            x = torch.cat([cache, x], dim=1)
        
        # Apply attention with limited context
        batch_size, seq_len, dim = x.shape
        
        # Create attention mask: can attend to left context + right context
        mask = self.create_streaming_mask(seq_len, self.right_context)
        
        # Attention
        x_att, _ = self.attention(x, x, x, attn_mask=mask)
        
        # Cache for next chunk
        new_cache = x[:, -self.left_context:, :]
        
        # Return only new frames (not cached ones)
        if cache is not None:
            x_att = x_att[:, cache.size(1):, :]
        
        return x_att, new_cache
    
    def create_streaming_mask(self, seq_len, right_context):
        """
        Create mask where each position can attend to:
        - All past positions
        - Up to right_context future positions
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask[:, :right_context] = 0  # Allow right context
        mask = mask.bool()
        return mask
```

---

## Component 4: Decoding and Beam Search

### Greedy Decoding (Fast, Suboptimal)

```python
def greedy_decode(model, audio_features):
    """
    Always pick highest-probability token
    Fast but misses better hypotheses
    """
    tokens = []
    state = None
    
    for frame in audio_features:
        logits, state = model(frame, tokens, state)
        best_token = torch.argmax(logits)
        
        if best_token != BLANK:
            tokens.append(best_token)
    
    return tokens
```

**Pros:** O(T) time, minimal memory  
**Cons:** Can't recover from mistakes, 10-20% worse WER

### Beam Search (Better Accuracy)

```python
class BeamSearchDecoder:
    def __init__(self, beam_size=10, blank_idx=0):
        self.beam_size = beam_size
        self.blank_idx = blank_idx
    
    def decode(self, model, audio_features):
        """
        Maintain top-k hypotheses at each time step
        """
        # Initial beam: empty hypothesis
        beams = [Hypothesis(tokens=[], score=0.0, state=None)]
        
        for frame in audio_features:
            candidates = []
            
            for beam in beams:
                # Get logits for this beam
                logits, new_state = model(frame, beam.tokens, beam.state)
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Extend with each possible token
                for token_idx, log_prob in enumerate(log_probs):
                    if token_idx == self.blank_idx:
                        # Blank: don't emit token, just update score
                        candidates.append(Hypothesis(
                            tokens=beam.tokens,
                            score=beam.score + log_prob,
                            state=beam.state
                        ))
                    else:
                        # Non-blank: emit token
                        candidates.append(Hypothesis(
                            tokens=beam.tokens + [token_idx],
                            score=beam.score + log_prob,
                            state=new_state
                        ))
            
            # Prune to top beam_size hypotheses
            candidates.sort(key=lambda h: h.score, reverse=True)
            beams = candidates[:self.beam_size]
        
        # Return best hypothesis
        return beams[0].tokens

class Hypothesis:
    def __init__(self, tokens, score, state):
        self.tokens = tokens
        self.score = score
        self.state = state
```

**Complexity:** O(T × B × V) where T=time, B=beam size, V=vocabulary size  
**Typical parameters:** B=10, V=1000 → manageable

### Language Model Fusion

**Problem:** Acoustic model doesn't know linguistic patterns (grammar, common phrases)

**Solution:** Integrate language model (LM) scores

```python
def beam_search_with_lm(acoustic_model, lm, audio_features, lm_weight=0.3):
    """
    Combine acoustic model + language model scores
    """
    beams = [Hypothesis(tokens=[], score=0.0, state=None)]
    
    for frame in audio_features:
        candidates = []
        
        for beam in beams:
            logits, new_state = acoustic_model(frame, beam.tokens, beam.state)
            acoustic_log_probs = F.log_softmax(logits, dim=-1)
            
            for token_idx, acoustic_log_prob in enumerate(acoustic_log_probs):
                if token_idx == BLANK:
                    # Blank token
                    combined_score = beam.score + acoustic_log_prob
                    candidates.append(Hypothesis(
                        tokens=beam.tokens,
                        score=combined_score,
                        state=beam.state
                    ))
                else:
                    # Get LM score for this token
                    lm_log_prob = lm.score(beam.tokens + [token_idx])
                    
                    # Combine scores
                    combined_score = (
                        beam.score +
                        acoustic_log_prob +
                        lm_weight * lm_log_prob
                    )
                    
                    candidates.append(Hypothesis(
                        tokens=beam.tokens + [token_idx],
                        score=combined_score,
                        state=new_state
                    ))
            
        candidates.sort(key=lambda h: h.score, reverse=True)
        beams = candidates[:beam_size]
    
    return beams[0].tokens
```

**LM types:**
- **N-gram LM (KenLM):** Fast (< 1ms), large memory (GBs)
- **Neural LM (LSTM/Transformer):** Slower (5-20ms), better quality

**Production choice:** N-gram for first-pass, neural LM for rescoring top hypotheses

---

## Latency Optimization

### Target Breakdown

**Total latency budget: 200ms**
```
VAD:                    2ms
Feature extraction:     5ms
Encoder forward:       80ms  ← Bottleneck
Decoder (beam search): 10ms
Post-processing:        3ms
Network overhead:      20ms
Total:               120ms ✓ (60ms margin)
```

### Technique 1: Model Quantization

**INT8 Quantization:** Convert float32 weights to int8

```python
import torch.quantization as quantization

# Post-training quantization (easiest)
model_fp32 = load_model()
model_fp32.eval()

# Fuse operations (Conv+BN+ReLU → single op)
model_fused = quantization.fuse_modules(
    model_fp32,
    [['conv', 'bn', 'relu']]
)

# Quantize
model_int8 = quantization.quantize_dynamic(
    model_fused,
    {nn.Linear, nn.LSTM, nn.Conv2d},
    dtype=torch.qint8
)

# Save
torch.save(model_int8.state_dict(), 'model_int8.pth')

# Results:
# - Model size: 200MB → 50MB (4x smaller)
# - Inference speed: 80ms → 30ms (2.7x faster)
# - Accuracy: WER 5.2% → 5.4% (0.2% degradation)
```

**Why quantization works:**
- **Smaller memory footprint:** Fits in L1/L2 cache
- **Faster math:** INT8 operations 4x faster than FP32 on CPU
- **Minimal accuracy loss:** Neural networks are surprisingly robust

### Technique 2: Knowledge Distillation

**Train small model to mimic large model**

```python
def distillation_loss(student_logits, teacher_logits, temperature=3.0):
    """
    Soft targets from teacher help student learn better
    """
    # Soften probabilities with temperature
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL divergence
    loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
    loss = loss * (temperature ** 2)
    
    return loss

# Training
teacher = large_model  # 18 layers, 80ms inference
student = small_model  # 8 layers, 30ms inference

for audio, transcript in training_data:
    # Get teacher predictions (no backprop)
    with torch.no_grad():
        teacher_logits = teacher(audio)
    
    # Student predictions
    student_logits = student(audio)
    
    # Distillation loss
    loss = distillation_loss(student_logits, teacher_logits)
    
    # Optimize
    loss.backward()
    optimizer.step()

# Results:
# - Student (8 layers): 30ms, WER 5.8%
# - Teacher (18 layers): 80ms, WER 5.0%
# - Without distillation: 30ms, WER 7.2%
# → Distillation closes the gap!
```

### Technique 3: Pruning

**Remove unimportant weights**

```python
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.4):
    """
    Remove 40% of weights with minimal accuracy loss
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # L1 unstructured pruning
            prune.l1_unstructured(module, name='weight', amount=amount)
            
            # Remove pruning reparameterization
            prune.remove(module, 'weight')
    
    return model

# Results:
# - 40% pruning: WER 5.0% → 5.3%, Speed +20%
# - 60% pruning: WER 5.0% → 6.2%, Speed +40%
```

### Technique 4: Caching

**Cache intermediate results across chunks**

```python
class StreamingASRWithCache:
    def __init__(self, model):
        self.model = model
        self.encoder_cache = None
        self.decoder_state = None
    
    def process_chunk(self, audio_chunk):
        # Extract features (no caching needed, fast)
        features = extract_features(audio_chunk)
        
        # Encoder: reuse cached hidden states
        encoder_out, self.encoder_cache = self.model.encoder(
            features,
            cache=self.encoder_cache
        )
        
        # Decoder: maintain beam state
        tokens, self.decoder_state = self.model.decoder(
            encoder_out,
            state=self.decoder_state
        )
        
        return tokens
    
    def reset(self):
        """Call at end of utterance"""
        self.encoder_cache = None
        self.decoder_state = None
```

**Savings:**
- **Without cache:** Process all frames every chunk → 100ms
- **With cache:** Process only new frames → 30ms (3.3x speedup)

---

## Scaling to Millions of Users

### Throughput Analysis

**Per-stream compute:**
- Encoder: 30ms (after optimization)
- Decoder: 10ms
- Total: 40ms per 100ms audio chunk

**CPU/GPU capacity:**
- CPU (16 cores): ~50 concurrent streams
- GPU (T4): ~200 concurrent streams

**For 10M concurrent streams:**
- GPUs needed: 10M / 200 = 50,000 GPUs
- Cost @ $0.50/hr: $25k/hour = $18M/month

**Way too expensive!** Need further optimization.

### Strategy 1: Batching

**Batch multiple streams together**

```python
def batch_inference(audio_chunks, batch_size=32):
    """
    Process 32 streams simultaneously on GPU
    """
    # Pad to same length
    max_len = max(len(chunk) for chunk in audio_chunks)
    padded = [
        np.pad(chunk, (0, max_len - len(chunk)))
        for chunk in audio_chunks
    ]
    
    # Stack into batch
    batch = torch.tensor(padded)  # (32, max_len, 80)
    
    # Single forward pass
    outputs = model(batch)  # ~40ms for 32 streams
    
    return outputs

# Results:
# - Without batching: 40ms per stream
# - With batching (32): 40ms / 32 = 1.25ms per stream (32x speedup)
# - GPU needed: 10M / (200 × 32) = 1,562 GPUs
# - Cost: $0.78M/month (23x cheaper!)
```

### Strategy 2: Regional Deployment

**Deploy closer to users to reduce latency**

```
North America: 3M users → 500 GPUs → 3 data centers
Europe: 2M users → 330 GPUs → 2 data centers
Asia: 4M users → 660 GPUs → 4 data centers
...

Total: ~1,500 GPUs globally
```

**Benefits:**
- Lower network latency (30ms → 10ms)
- Better fault isolation
- Regulatory compliance (data residency)

### Strategy 3: Hybrid Cloud-Edge

**Run simple queries on-device, complex queries on cloud**

```python
def route_request(audio, user_context):
    # Estimate query complexity
    if is_simple_command(audio):  # "play music", "set timer"
        return on_device_asr(audio)  # 30ms, free, offline
    
    elif is_dictation(audio):  # Long-form transcription
        return cloud_asr(audio)  # 80ms, $0.01/min, high accuracy
    
    else:  # Conversational query
        return cloud_asr(audio)  # Best quality for complex queries
```

**Distribution:**
- 70% simple commands → on-device
- 30% complex queries → cloud
- Effective cloud load: 3M concurrent (70% savings!)

---

## Production Example: Putting It All Together

```python
import asyncio
import websockets
import torch

class ProductionStreamingASR:
    def __init__(self):
        # Load optimized model
        self.model = self.load_optimized_model()
        
        # VAD
        self.vad = SileroVAD()
        
        # Session management
        self.sessions = {}  # session_id → StreamingSession
        
        # Metrics
        self.metrics = Metrics()
    
    def load_optimized_model(self):
        """Load quantized, pruned model"""
        model = StreamingRNNT(vocab_size=1000)
        
        # Load pre-trained weights
        checkpoint = torch.load('rnnt_optimized.pth')
        model.load_state_dict(checkpoint)
        
        # Quantize
        model_quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8
        )
        
        model_quantized.eval()
        return model_quantized
    
    async def handle_stream(self, websocket, path):
        """Handle websocket connection from client"""
        session_id = generate_session_id()
        session = StreamingSession(session_id, self.model, self.vad)
        self.sessions[session_id] = session
        
        try:
            async for message in websocket:
                # Receive audio chunk (binary, 100ms @ 16kHz)
                audio_bytes = message
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Process
                start_time = time.time()
                result = session.process_chunk(audio_float)
                latency = (time.time() - start_time) * 1000  # ms
                
                # Send partial transcript
                if result:
                    await websocket.send(json.dumps({
                        'type': 'partial',
                        'transcript': result['text'],
                        'tokens': result['tokens'],
                        'is_final': result['is_final']
                    }))
                
                # Track metrics
                self.metrics.record_latency(latency)
        
        except websockets.ConnectionClosed:
            # Finalize session
            final_transcript = session.finalize()
            print(f"Session {session_id} ended: {final_transcript}")
        
        finally:
            # Cleanup
            del self.sessions[session_id]
    
    def run(self, host='0.0.0.0', port=8765):
        """Start WebSocket server"""
        start_server = websockets.serve(self.handle_stream, host, port)
        asyncio.get_event_loop().run_until_complete(start_server)
        print(f"Streaming ASR server running on ws://{host}:{port}")
        asyncio.get_event_loop().run_forever()

class StreamingSession:
    def __init__(self, session_id, model, vad):
        self.session_id = session_id
        self.model = model
        self.vad = vad
        
        # State
        self.encoder_cache = None
        self.decoder_state = None
        self.partial_transcript = ""
        self.audio_buffer = []
    
    def process_chunk(self, audio):
        # VAD check
        if not self.vad.is_speech(audio):
            return None
        
        # Extract features
        features = extract_mel_features(audio)
        
        # Encode
        encoder_out, self.encoder_cache = self.model.encoder(
            features,
            cache=self.encoder_cache
        )
        
        # Decode (beam search)
        tokens, self.decoder_state = self.model.decoder(
            encoder_out,
            state=self.decoder_state,
            beam_size=5
        )
        
        # Convert tokens to text
        new_text = self.model.tokenizer.decode(tokens)
        self.partial_transcript += new_text
        
        return {
            'text': new_text,
            'tokens': tokens,
            'is_final': False
        }
    
    def finalize(self):
        """End of utterance processing"""
        # Post-processing
        final_transcript = post_process(self.partial_transcript)
        
        # Reset state
        self.encoder_cache = None
        self.decoder_state = None
        self.partial_transcript = ""
        
        return final_transcript

# Run server
if __name__ == '__main__':
    server = ProductionStreamingASR()
    server.run()
```

---

## Key Takeaways

✅ **RNN-T architecture** enables true streaming without future context  
✅ **Conformer encoder** combines convolution + attention for best accuracy  
✅ **State management** critical for maintaining context across chunks  
✅ **Quantization + pruning** achieve 4x compression, 3x speedup, < 1% WER loss  
✅ **Batching** provides 32x throughput improvement on GPUs  
✅ **Hybrid cloud-edge** reduces cloud load by 70%  
✅ **VAD** saves 50-70% compute by filtering silence

---

## Further Reading

**Papers:**
- [RNN-Transducer (Graves 2012)](https://arxiv.org/abs/1211.3711)
- [Conformer (Google 2020)](https://arxiv.org/abs/2005.08100)
- [ContextNet (Google 2020)](https://arxiv.org/abs/2005.03191)
- [Streaming E2E ASR](https://arxiv.org/abs/1811.06621)

**Open-Source:**
- [ESPnet](https://github.com/espnet/espnet) - End-to-end speech processing
- [SpeechBrain](https://github.com/speechbrain/speechbrain) - PyTorch-based toolkit
- [Kaldi](https://github.com/kaldi-asr/kaldi) - Classic ASR toolkit

**Courses:**
- [Stanford CS224S: Spoken Language Processing](http://web.stanford.edu/class/cs224s/)
- [Coursera: Speech Recognition and Synthesis](https://www.coursera.org/learn/nlp-sequence-models)

---

## Conclusion

Streaming ASR is a fascinating blend of signal processing, deep learning, and systems engineering. The key challenges low latency, high throughput, and maintaining accuracy without future context require careful architectural choices and aggressive optimization.

As voice interfaces become ubiquitous, streaming ASR systems will continue to evolve. Future directions include:
- **Multi-modal models** (audio + video for better accuracy)
- **Personalization** (adapt to individual speaking styles)
- **Emotion recognition** (detect sentiment, stress, sarcasm)
- **On-device models** (< 10MB, < 50ms, works offline)

The fundamentals covered here RNN-T, streaming architectures, optimization techniques will remain relevant as the field advances.

Now go build a voice assistant that feels truly conversational! 🎤🚀

---

**Originally published at:** [arunbaby.com/speech-tech/0001-streaming-asr](https://www.arunbaby.com/speech-tech/0001-streaming-asr/)

*If you found this helpful, consider sharing it with others who might benefit.*
