---
title: "Speech Separation"
day: 11
collection: speech_tech
categories:
  - speech-tech
tags:
  - speech-separation
  - source-separation
  - multi-speaker
  - deep-learning
  - cocktail-party-problem
  - conv-tasnet
  - speaker-diarization
subdomain: Audio Processing
tech_stack: [Python, PyTorch, Conv-TasNet, Dual-Path RNN, WebSocket, NumPy]
scale: "Real-time streaming, < 50ms latency"
companies: [Google, Meta, Zoom, Microsoft, Dolby, Krisp, Descript]
related_dsa_day: 11
related_ml_day: 11
---

**Separate overlapping speakers with 99%+ accuracy: Deep learning solves the cocktail party problem for meeting transcription and voice assistants.**

## Problem Statement

**Speech Separation** (also called **Source Separation** or **Speaker Separation**) is the task of isolating individual speech sources from a mixture of overlapping speakers.

### The Cocktail Party Problem

Humans can focus on a single speaker in a noisy, multi-speaker environment (like a cocktail party). Teaching machines to do the same is a fundamental challenge in speech processing.

**Applications:**
- Meeting transcription with overlapping speech
- Voice assistants in multi-speaker environments
- Hearing aids for selective attention
- Call center audio analysis
- Video conferencing quality improvement

### Problem Formulation

**Input:** Mixed audio with N speakers  
**Output:** N separated audio streams, one per speaker

```
Mixed Audio:
  Speaker 1 + Speaker 2 + ... + Speaker N + Noise

Goal:
  → Separated Speaker 1
  → Separated Speaker 2
  → ...
  → Separated Speaker N
```

---

## Understanding Speech Separation

### Why is Speech Separation Hard?

Let's understand the fundamental challenge with a simple analogy:

**The Paint Mixing Problem**

Imagine you have:
- Red paint (Speaker 1)
- Blue paint (Speaker 2)
- You mix them → Purple paint (Mixed audio)

**Challenge**: Given purple paint, separate back into red and blue!

This seems impossible because mixing is **information-destructive**. But speech separation works because:

1. **Speech has structure**: Not random noise, but patterns (phonemes, pitch, timing)
2. **Speakers differ**: Different voice characteristics (pitch, timbre, accent)
3. **Deep learning**: Can learn these patterns from thousands of examples

### The Human Cocktail Party Effect

At a party with multiple conversations, you can focus on one person and "tune out" others. How?

**Human brain uses:**
- **Spatial cues**: Sound comes from different directions
- **Voice characteristics**: Pitch, timbre, speaking style
- **Linguistic context**: Grammar, meaning help predict words
- **Visual cues**: Lip reading, body language

**ML models use:**
- ❌ No spatial cues (single microphone input)
- ✅ Voice characteristics (learned from data)
- ✅ Temporal patterns (speaking rhythm)
- ✅ Spectral patterns (frequency differences)

### The Core Mathematical Challenge

**Input**: Mixed waveform `M(t) = S1(t) + S2(t) + ... + Sn(t)`
- `M(t)`: What we hear (mixture)
- `S1(t), S2(t), ...`: Individual speakers (what we want)

**Goal**: Find a function `f` such that:
- `f(M)` → `[S1, S2, ..., Sn]`

**Why this is hard:**
1. **Underdetermined problem**: One equation (mixture), N unknowns (sources)
2. **Non-linear mixing**: In reality, it's not just addition (room acoustics, etc.)
3. **Unknown N**: We often don't know how many speakers there are
4. **Permutation ambiguity**: Output order doesn't matter (Speaker 1 could be output 2)

### Challenges

**1. Permutation Problem - The Hardest Part**

When you train a model:

```
Attempt 1:
Ground truth: [Speaker A, Speaker B]
Model output: [Speaker A, Speaker B]  ✓ Matches!

Attempt 2:
Ground truth: [Speaker A, Speaker B]  
Model output: [Speaker B, Speaker A]  ✓ Also correct! Just different order!
```

**The problem**: Standard loss (MSE) would say Attempt 2 is wrong!

```python
# This would incorrectly penalize Attempt 2
loss = mse(output[0], speaker_A) + mse(output[1], speaker_B)
```

**Solution**: Try all permutations, use best one (Permutation Invariant Training)

```python
# Try both orderings, pick better one
loss1 = mse(output[0], speaker_A) + mse(output[1], speaker_B)
loss2 = mse(output[0], speaker_B) + mse(output[1], speaker_A)
loss = min(loss1, loss2)  # Use better permutation
```

**2. Number of Speakers**

| Scenario | Difficulty | Solution |
|----------|-----------|----------|
| Fixed N (always 2 speakers) | Easy | Train model for N=2 |
| Variable N (2-5 speakers) | Hard | Separate approaches: 1) Train multiple models, 2) Train one model + speaker counting |
| Unknown N | Very Hard | Need speaker counting + adaptive separation |

**3. Overlapping Speech**

```
Scenario 1: Sequential (Easy)
Time:    0s     1s     2s     3s     4s
Speaker A: "Hello"      
Speaker B:       "Hi"
          ↑ No overlap, trivial!

Scenario 2: Partial Overlap (Medium)
Time:    0s     1s     2s     3s     4s
Speaker A: "Hello there"
Speaker B:       "Hi how are you"
                 ↑ Some overlap

Scenario 3: Complete Overlap (Hard)
Time:    0s     1s     2s     3s     4s
Speaker A: "Hello there"
Speaker B: "Hi how are you"
          ↑ Both speaking simultaneously!
```

**Why complete overlap is hard:**
- Maximum information loss
- Voices blend in frequency domain
- Harder to find distinguishing features

**4. Quality Metrics**

How do we measure separation quality?

| Metric | What it Measures | Good Value |
|--------|-----------------|------------|
| **SDR** (Signal-to-Distortion Ratio) | Overall quality | > 10 dB |
| **SIR** (Signal-to-Interference) | How well other speakers removed | > 15 dB |
| **SAR** (Signal-to-Artifacts) | Artificial noise introduced | > 10 dB |
| **SI-SDR** (Scale-Invariant SDR) | Quality regardless of volume | > 15 dB |

**Intuition**: Higher dB = Better separation

```
SI-SDR = 0 dB  → No separation (output = input)
SI-SDR = 10 dB → Good separation (10x better signal)
SI-SDR = 20 dB → Excellent (100x better signal!)
```

### Traditional Approaches

**Independent Component Analysis (ICA):**
- Assumes statistical independence
- Works for determined/overdetermined cases
- Limited by linear mixing assumption

**Beamforming:**
- Uses spatial information from microphone array
- Requires known speaker locations
- Hardware-dependent

**Non-Negative Matrix Factorization (NMF):**
- Factorizes spectrogram into basis and activation
- Interpretable but limited capacity

### Deep Learning Revolution

Modern approaches use end-to-end deep learning:
- **TasNet** (Time-domain Audio Separation Network)
- **Conv-TasNet** (Convolutional TasNet)
- **Dual-Path RNN**
- **SepFormer** (Transformer-based)

---

## Solution 1: Conv-TasNet Architecture

### Architecture Overview

Conv-TasNet is the gold standard for speech separation:

```
┌──────────────────────────────────────────────────────────┐
│                    CONV-TASNET ARCHITECTURE               │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Input Waveform                                          │
│  [batch, time]                                           │
│       │                                                   │
│       ▼                                                   │
│  ┌─────────────┐                                        │
│  │   Encoder   │  (1D Conv)                             │
│  │  512 filters│  Learns time-domain basis functions    │
│  └──────┬──────┘                                        │
│         │                                                 │
│         ▼                                                 │
│  [batch, 512, time']                                     │
│         │                                                 │
│         ▼                                                 │
│  ┌─────────────┐                                        │
│  │  Separator  │  (TCN blocks)                          │
│  │  Temporal   │  Estimates masks for each speaker      │
│  │  Convolution│                                         │
│  │  Network    │                                         │
│  └──────┬──────┘                                        │
│         │                                                 │
│         ▼                                                 │
│  [batch, n_speakers, 512, time']                        │
│  (Masks for each speaker)                               │
│         │                                                 │
│         ▼                                                 │
│  ┌─────────────┐                                        │
│  │  Apply Mask │  (Element-wise multiply)               │
│  └──────┬──────┘                                        │
│         │                                                 │
│         ▼                                                 │
│  [batch, n_speakers, 512, time']                        │
│  (Masked representations)                                │
│         │                                                 │
│         ▼                                                 │
│  ┌─────────────┐                                        │
│  │   Decoder   │  (1D Transposed Conv)                  │
│  │  n_speakers │  Reconstructs waveforms                │
│  │  outputs    │                                         │
│  └──────┬──────┘                                        │
│         │                                                 │
│         ▼                                                 │
│  Separated Waveforms                                     │
│  [batch, n_speakers, time]                              │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class ConvTasNet(nn.Module):
    """
    Conv-TasNet for speech separation
    
    Paper: "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking
            for Speech Separation" (Luo & Mesgarani, 2019)
    
    Architecture:
    1. Encoder: Waveform → learned representation
    2. Separator: Mask estimation with TCN
    3. Decoder: Masked representation → waveforms
    """
    
    def __init__(
        self,
        n_src=2,
        n_filters=512,
        kernel_size=16,
        stride=8,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128
    ):
        """
        Args:
            n_src: Number of sources (speakers)
            n_filters: Number of filters in encoder
            kernel_size: Encoder/decoder kernel size
            stride: Encoder/decoder stride
            n_blocks: Number of TCN blocks per repeat
            n_repeats: Number of times to repeat TCN blocks
            bn_chan: Bottleneck channels
            hid_chan: Hidden channels in TCN
            skip_chan: Skip connection channels
        """
        super().__init__()
        
        self.n_src = n_src
        
        # Encoder: waveform → representation
        self.encoder = nn.Conv1d(
            1,
            n_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )
        
        # Separator: Temporal Convolutional Network
        self.separator = TemporalConvNet(
            n_filters,
            n_src,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            skip_chan=skip_chan
        )
        
        # Decoder: representation → waveform
        self.decoder = nn.ConvTranspose1d(
            n_filters,
            1,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )
    
    def forward(self, mixture):
        """
        Separate mixture into sources
        
        Args:
            mixture: [batch, time] mixed waveform
        
        Returns:
            separated: [batch, n_src, time] separated waveforms
        """
        batch_size = mixture.size(0)
        
        # Add channel dimension
        mixture = mixture.unsqueeze(1)  # [batch, 1, time]
        
        # Encode
        encoded = self.encoder(mixture)  # [batch, n_filters, time']
        
        # Estimate masks
        masks = self.separator(encoded)  # [batch, n_src, n_filters, time']
        
        # Apply masks
        masked = encoded.unsqueeze(1) * masks  # [batch, n_src, n_filters, time']
        
        # Decode each source
        separated = []
        
        for src_idx in range(self.n_src):
            src_masked = masked[:, src_idx, :, :]  # [batch, n_filters, time']
            src_waveform = self.decoder(src_masked)  # [batch, 1, time]
            separated.append(src_waveform.squeeze(1))  # [batch, time]
        
        # Stack sources
        separated = torch.stack(separated, dim=1)  # [batch, n_src, time]
        
        # Trim to original length
        if separated.size(-1) != mixture.size(-1):
            separated = separated[..., :mixture.size(-1)]
        
        return separated

class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for mask estimation
    
    Stack of dilated 1D conv blocks with skip connections
    """
    
    def __init__(
        self,
        n_filters,
        n_src,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128
    ):
        super().__init__()
        
        # Layer normalization
        self.layer_norm = nn.GroupNorm(1, n_filters)
        
        # Bottleneck (reduce dimensionality)
        self.bottleneck = nn.Conv1d(n_filters, bn_chan, 1)
        
        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        
        for r in range(n_repeats):
            for b in range(n_blocks):
                dilation = 2 ** b
                self.tcn_blocks.append(
                    TCNBlock(
                        bn_chan,
                        hid_chan,
                        skip_chan,
                        kernel_size=3,
                        dilation=dilation
                    )
                )
        
        # Output projection
        self.output = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(skip_chan, n_filters * n_src, 1),
        )
        
        self.n_filters = n_filters
        self.n_src = n_src
    
    def forward(self, x):
        """
        Estimate masks for each source
        
        Args:
            x: [batch, n_filters, time']
        
        Returns:
            masks: [batch, n_src, n_filters, time']
        """
        batch_size, n_filters, time = x.size()
        
        # Normalize
        x = self.layer_norm(x)
        
        # Bottleneck
        x = self.bottleneck(x)  # [batch, bn_chan, time']
        
        # Accumulate skip connections
        skip_sum = 0
        
        for block in self.tcn_blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        
        # Output masks
        masks = self.output(skip_sum)  # [batch, n_filters * n_src, time']
        
        # Reshape to [batch, n_src, n_filters, time']
        masks = masks.view(batch_size, self.n_src, self.n_filters, time)
        
        # Apply non-linearity (ReLU for masking)
        masks = torch.relu(masks)
        
        return masks

class TCNBlock(nn.Module):
    """
    Single TCN block with dilated depthwise-separable convolution
    """
    
    def __init__(self, in_chan, hid_chan, skip_chan, kernel_size=3, dilation=1):
        super().__init__()
        
        # 1x1 conv
        self.conv1x1_1 = nn.Conv1d(in_chan, hid_chan, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, hid_chan)
        
        # Depthwise conv with dilation
        self.depthwise_conv = nn.Conv1d(
            hid_chan,
            hid_chan,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
            groups=hid_chan  # Depthwise
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, hid_chan)
        
        # 1x1 conv
        self.conv1x1_2 = nn.Conv1d(hid_chan, in_chan, 1)
        
        # Skip connection
        self.skip_conv = nn.Conv1d(hid_chan, skip_chan, 1)
    
    def forward(self, x):
        """
        Args:
            x: [batch, in_chan, time]
        
        Returns:
            output: [batch, in_chan, time]
            skip: [batch, skip_chan, time]
        """
        residual = x
        
        # 1x1 conv
        x = self.conv1x1_1(x)
        x = self.prelu1(x)
        x = self.norm1(x)
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        x = self.prelu2(x)
        x = self.norm2(x)
        
        # Skip connection
        skip = self.skip_conv(x)
        
        # 1x1 conv
        x = self.conv1x1_2(x)
        
        # Residual connection
        output = x + residual
        
        return output, skip

# Example usage
model = ConvTasNet(n_src=2, n_filters=512)

# Mixed waveform (2 speakers)
mixture = torch.randn(4, 16000)  # [batch=4, time=16000 (1 second at 16kHz)]

# Separate
separated = model(mixture)  # [4, 2, 16000]

print(f"Input shape: {mixture.shape}")
print(f"Output shape: {separated.shape}")
print(f"Separated speaker 1: {separated[:, 0, :].shape}")
print(f"Separated speaker 2: {separated[:, 1, :].shape}")
```

### Training with Permutation Invariant Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PermutationInvariantLoss(nn.Module):
    """
    Permutation Invariant Training (PIT) loss
    
    Problem: Model outputs are in arbitrary order
    Solution: Try all permutations, use best one
    
    For 2 speakers:
    - Try (output1→target1, output2→target2)
    - Try (output1→target2, output2→target1)
    - Use permutation with lower loss
    """
    
    def __init__(self, loss_fn='si_sdr'):
        super().__init__()
        self.loss_fn = loss_fn
    
    def forward(self, estimated, target):
        """
        Compute PIT loss
        
        Args:
            estimated: [batch, n_src, time]
            target: [batch, n_src, time]
        
        Returns:
            loss: scalar
        """
        batch_size, n_src, time = estimated.size()
        
        # Generate all permutations
        import itertools
        perms = list(itertools.permutations(range(n_src)))
        
        # Compute loss for each permutation
        perm_losses = []
        
        for perm in perms:
            # Reorder estimated according to permutation
            estimated_perm = estimated[:, perm, :]
            
            # Compute loss
            if self.loss_fn == 'si_sdr':
                loss = self._si_sdr_loss(estimated_perm, target)
            elif self.loss_fn == 'mse':
                loss = F.mse_loss(estimated_perm, target)
            else:
                raise ValueError(f"Unknown loss function: {self.loss_fn}")
            
            perm_losses.append(loss)
        
        # Stack losses
        # [n_perms], take minimum (best permutation)
        perm_losses = torch.stack(perm_losses)
        return perm_losses.min()
    
    def _si_sdr_loss(self, estimated, target):
        """
        Scale-Invariant Signal-to-Distortion Ratio loss
        
        Better than MSE for speech separation
        """
        # Zero-mean
        estimated = estimated - estimated.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        
        # Project estimated onto target
        dot = (estimated * target).sum(dim=-1, keepdim=True)
        target_energy = (target ** 2).sum(dim=-1, keepdim=True) + 1e-8
        projection = dot * target / target_energy
        
        # Noise (estimation error)
        noise = estimated - projection
        
        # SI-SDR
        si_sdr = 10 * torch.log10(
            (projection ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + 1e-8)
        )
        
        # Negative for loss (we want to maximize SI-SDR)
        return -si_sdr.mean()

# Training loop
model = ConvTasNet(n_src=2)
criterion = PermutationInvariantLoss(loss_fn='si_sdr')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_epoch(model, train_loader, criterion, optimizer):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (mixture, target) in enumerate(train_loader):
        # mixture: [batch, time]
        # target: [batch, n_src, time]
        
        # Forward
        estimated = model(mixture)
        
        # Loss with PIT
        loss = criterion(estimated, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)

# Train
# for epoch in range(num_epochs):
#     train_loss = train_epoch(model, train_loader, criterion, optimizer)
#     print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
```

---

## Evaluation Metrics

### Signal-to-Distortion Ratio (SDR)

```python
def compute_sdr(estimated, target):
    """
    Compute Signal-to-Distortion Ratio
    
    SDR = 10 * log10(||target||^2 / ||target - estimated||^2)
    
    Higher is better. Good: > 10 dB, Great: > 15 dB
    """
    target = target - target.mean()
    estimated = estimated - estimated.mean()
    
    signal_power = np.sum(target ** 2)
    distortion = target - estimated
    distortion_power = np.sum(distortion ** 2) + 1e-10
    
    sdr = 10 * np.log10(signal_power / distortion_power)
    
    return sdr

def compute_si_sdr(estimated, target):
    """
    Compute Scale-Invariant SDR
    
    Invariant to scaling of the signal
    """
    # Zero-mean
    estimated = estimated - estimated.mean()
    target = target - target.mean()
    
    # Project estimated onto target
    alpha = np.dot(estimated, target) / (np.dot(target, target) + 1e-10)
    projection = alpha * target
    
    # Noise
    noise = estimated - projection
    
    # SI-SDR
    si_sdr = 10 * np.log10(
        np.sum(projection ** 2) / (np.sum(noise ** 2) + 1e-10)
    )
    
    return si_sdr

def compute_sir(estimated, target, interference):
    """
    Compute Signal-to-Interference Ratio
    
    Measures how well interfering speakers are suppressed
    """
    target = target - target.mean()
    estimated = estimated - estimated.mean()
    
    # Project estimated onto target
    s_target = np.dot(estimated, target) / (np.dot(target, target) + 1e-10) * target
    
    # Interference
    e_interf = 0
    for interf in interference:
        interf = interf - interf.mean()
        e_interf += np.dot(estimated, interf) / (np.dot(interf, interf) + 1e-10) * interf
    
    # SIR
    sir = 10 * np.log10(
        np.sum(s_target ** 2) / (np.sum(e_interf ** 2) + 1e-10)
    )
    
    return sir

# Comprehensive evaluation
def evaluate_separation(model, test_loader):
    """
    Evaluate separation quality
    
    Returns metrics for each source
    """
    model.eval()
    
    all_sdr = []
    all_si_sdr = []
    
    with torch.no_grad():
        for mixture, targets in test_loader:
            # Separate
            estimated = model(mixture)
            
            # Convert to numpy
            estimated_np = estimated.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            batch_size, n_src, time = estimated_np.shape
            
            # Compute metrics for each sample and source
            for b in range(batch_size):
                for s in range(n_src):
                    est = estimated_np[b, s, :]
                    tgt = targets_np[b, s, :]
                    
                    sdr = compute_sdr(est, tgt)
                    si_sdr = compute_si_sdr(est, tgt)
                    
                    all_sdr.append(sdr)
                    all_si_sdr.append(si_sdr)
    
    results = {
        'sdr_mean': np.mean(all_sdr),
        'sdr_std': np.std(all_sdr),
        'si_sdr_mean': np.mean(all_si_sdr),
        'si_sdr_std': np.std(all_si_sdr)
    }
    
    print("="*60)
    print("SEPARATION EVALUATION RESULTS")
    print("="*60)
    print(f"SDR:     {results['sdr_mean']:.2f} ± {results['sdr_std']:.2f} dB")
    print(f"SI-SDR:  {results['si_sdr_mean']:.2f} ± {results['si_sdr_std']:.2f} dB")
    print("="*60)
    
    return results

# Example
# results = evaluate_separation(model, test_loader)
```

---

## Real-Time Separation Pipeline

### Streaming Speech Separation

```python
class StreamingSpeechSeparator:
    """
    Real-time speech separation for streaming audio
    
    Challenges:
    - Causal processing (no future context)
    - Low latency (< 50ms)
    - State management across chunks
    """
    
    def __init__(self, model, chunk_size=4800, overlap=1200):
        """
        Args:
            model: Trained separation model
            chunk_size: Samples per chunk (300ms at 16kHz)
            overlap: Overlap between chunks (75ms at 16kHz)
        """
        self.model = model
        self.model.eval()
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.hop_size = chunk_size - overlap
        
        # Buffer for overlapping
        self.input_buffer = np.zeros(overlap)
        self.output_buffers = [np.zeros(overlap) for _ in range(model.n_src)]
    
    def process_chunk(self, audio_chunk):
        """
        Process single audio chunk
        
        Args:
            audio_chunk: [chunk_size] numpy array
        
        Returns:
            separated_chunks: list of [hop_size] arrays, one per speaker
        """
        # Concatenate with buffer
        full_chunk = np.concatenate([self.input_buffer, audio_chunk])
        
        # Ensure correct size
        if len(full_chunk) < self.chunk_size:
            full_chunk = np.pad(
                full_chunk,
                (0, self.chunk_size - len(full_chunk)),
                mode='constant'
            )
        
        # Convert to tensor
        with torch.no_grad():
            chunk_tensor = torch.from_numpy(full_chunk).float().unsqueeze(0)
            
            # Separate
            separated = self.model(chunk_tensor)  # [1, n_src, chunk_size]
            
            # Convert back to numpy
            separated_np = separated[0].cpu().numpy()  # [n_src, chunk_size]
        
        # Overlap-add
        result_chunks = []
        
        for src_idx in range(self.model.n_src):
            src_audio = separated_np[src_idx]
            
            # Add overlap from previous chunk
            src_audio[:self.overlap] += self.output_buffers[src_idx]
            
            # Extract output (without overlap)
            output_chunk = src_audio[:self.hop_size]
            result_chunks.append(output_chunk)
            
            # Save overlap for next chunk
            self.output_buffers[src_idx] = src_audio[-self.overlap:]
        
        # Update input buffer
        self.input_buffer = audio_chunk[-self.overlap:]
        
        return result_chunks
    
    def reset(self):
        """Reset state for new stream"""
        self.input_buffer = np.zeros(self.overlap)
        self.output_buffers = [np.zeros(self.overlap) for _ in range(self.model.n_src)]

# Example: Real-time separation server
from fastapi import FastAPI, WebSocket
import asyncio

app = FastAPI()

# Load model
model = ConvTasNet(n_src=2)
model.load_state_dict(torch.load('convtasnet_separation.pth'))
separator = StreamingSpeechSeparator(model, chunk_size=4800, overlap=1200)

@app.websocket("/separate")
async def websocket_separation(websocket: WebSocket):
    """
    WebSocket endpoint for real-time separation
    
    Client sends audio chunks, receives separated streams
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            
            # Decode audio (assuming 16-bit PCM)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Separate
            separated_chunks = separator.process_chunk(audio_chunk)
            
            # Send separated streams
            for src_idx, src_chunk in enumerate(separated_chunks):
                # Encode back to 16-bit PCM
                src_bytes = (src_chunk * 32768).astype(np.int16).tobytes()
                
                await websocket.send_json({
                    'speaker_id': src_idx,
                    'audio': src_bytes.hex()
                })
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        separator.reset()
        await websocket.close()

# Run server
# uvicorn.run(app, host='0.0.0.0', port=8000)
```

---

## Integration with Downstream Tasks

### Speech Separation + ASR Pipeline

```python
class SeparationASRPipeline:
    """
    Combined pipeline: Separate speakers → Transcribe each
    
    Use case: Meeting transcription with overlapping speech
    """
    
    def __init__(self, separation_model, asr_model):
        self.separator = separation_model
        self.asr = asr_model
    
    def transcribe_multi_speaker(self, audio):
        """
        Transcribe audio with multiple speakers
        
        Args:
            audio: Mixed audio
        
        Returns:
            List of (speaker_id, transcript) tuples
        """
        # Separate speakers
        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            separated = self.separator(audio_tensor)[0]  # [n_src, time]
        
        # Transcribe each speaker
        transcripts = []
        
        for speaker_id in range(separated.size(0)):
            speaker_audio = separated[speaker_id].cpu().numpy()
            
            # Transcribe
            transcript = self.asr.transcribe(speaker_audio)
            
            transcripts.append({
                'speaker_id': speaker_id,
                'transcript': transcript,
                'audio_length_sec': len(speaker_audio) / 16000
            })
        
        return transcripts
    
    def transcribe_with_diarization(self, audio):
        """
        Transcribe with speaker diarization
        
        Diarization: Who spoke when?
        Separation: Isolate each speaker's audio
        ASR: Transcribe each speaker
        """
        # Separate speakers
        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            separated = self.separator(audio_tensor)[0]  # [n_src, time]
        
        # Speaker diarization on each separated stream
        diarization_results = []
        
        for speaker_id in range(separated.size(0)):
            speaker_audio = separated[speaker_id].cpu().numpy()
            # Voice Activity Detection
            vad_segments = self._detect_voice_activity(speaker_audio)
            
            # Transcribe active segments
            for segment in vad_segments:
                start_idx = int(segment['start'] * 16000)
                end_idx = int(segment['end'] * 16000)
                
                segment_audio = speaker_audio[start_idx:end_idx]
                transcript = self.asr.transcribe(segment_audio)
                
                diarization_results.append({
                    'speaker_id': speaker_id,
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'transcript': transcript
                })
        
        # Sort by start time
        diarization_results.sort(key=lambda x: x['start_time'])
        
        return diarization_results
    
    def _detect_voice_activity(self, audio, frame_duration=0.03):
        """
        Simple energy-based VAD
        
        Returns list of (start, end) segments with voice activity
        """
        import librosa
        
        # Compute energy
        frame_length = int(frame_duration * 16000)
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=frame_length // 2
        )[0]
        
        # Threshold
        threshold = np.mean(energy) * 0.5
        
        # Find voice segments
        is_voice = energy > threshold
        
        segments = []
        in_segment = False
        start = 0
        
        for i, voice in enumerate(is_voice):
            if voice and not in_segment:
                start = i * frame_duration / 2
                in_segment = True
            elif not voice and in_segment:
                end = i * frame_duration / 2
                segments.append({'start': start, 'end': end})
                in_segment = False
        
        return segments

# Example usage
separation_model = ConvTasNet(n_src=2)
separation_model.load_state_dict(torch.load('separation_model.pth'))

# Mock ASR model
class MockASR:
    def transcribe(self, audio):
        return f"Transcribed {len(audio)} samples"

asr_model = MockASR()

pipeline = SeparationASRPipeline(separation_model, asr_model)

# Transcribe multi-speaker audio
audio = np.random.randn(16000 * 10)  # 10 seconds
results = pipeline.transcribe_multi_speaker(audio)

print("Transcription results:")
for result in results:
    print(f"Speaker {result['speaker_id']}: {result['transcript']}")
```

---

## Advanced Topics

### Unknown Number of Speakers

```python
class AdaptiveSeparationModel(nn.Module):
    """
    Separate audio with unknown number of speakers
    
    Approach:
    1. Estimate number of speakers
    2. Separate into estimated number of sources
    3. Filter empty sources
    """
    
    def __init__(self, max_speakers=10):
        super().__init__()
        
        self.max_speakers = max_speakers
        
        # Speaker counting network
        self.speaker_counter = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, max_speakers + 1),  # 0 to max_speakers
            nn.Softmax(dim=-1)
        )
        
        # Separation models for different numbers of speakers
        self.separators = nn.ModuleList([
            ConvTasNet(n_src=n) for n in range(1, max_speakers + 1)
        ])
    
    def forward(self, mixture):
        """
        Separate with adaptive number of sources
        
        Args:
            mixture: [batch, time]
        
        Returns:
            separated: list of [batch, time] tensors (one per active speaker)
        """
        # Estimate number of speakers
        mixture_1d = mixture.unsqueeze(1)  # [batch, 1, time]
        speaker_probs = self.speaker_counter(mixture_1d)  # [batch, max_speakers + 1]
        
        n_speakers = speaker_probs.argmax(dim=-1)  # [batch]
        
        # For simplicity, use max in batch (in practice, process per sample)
        max_n_speakers = n_speakers.max().item()
        
        if max_n_speakers == 0:
            return []
        
        # Separate using appropriate model
        separator = self.separators[max_n_speakers - 1]
        separated = separator(mixture)  # [batch, n_src, time]
        
        return separated

# Example
model = AdaptiveSeparationModel(max_speakers=5)

# Test with 2 speakers
mixture = torch.randn(1, 16000)
separated = model(mixture)

print(f"Estimated sources: {separated.size(1)}")
```

### Multi-Channel Separation

```python
class MultiChannelSeparator(nn.Module):
    """
    Use multiple microphones for better separation
    
    Microphone array provides spatial information
    """
    
    def __init__(self, n_channels, n_src):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_src = n_src
        
        # Encoder for each channel
        self.encoders = nn.ModuleList([
            nn.Conv1d(1, 256, kernel_size=16, stride=8)
            for _ in range(n_channels)
        ])
        
        # Cross-channel attention
        self.cross_channel_attention = nn.MultiheadAttention(
            embed_dim=256 * n_channels,
            num_heads=8
        )
        
        # Separator
        self.separator = TemporalConvNet(
            256 * n_channels,
            n_src,
            n_blocks=8,
            n_repeats=3,
            bn_chan=128,
            hid_chan=512,
            skip_chan=128
        )
        
        # Decoder
        self.decoder = nn.ConvTranspose1d(256, 1, kernel_size=16, stride=8)
    
    def forward(self, multi_channel_mixture):
        """
        Separate using multi-channel input
        
        Args:
            multi_channel_mixture: [batch, n_channels, time]
        
        Returns:
            separated: [batch, n_src, time]
        """
        batch_size, n_channels, time = multi_channel_mixture.size()
        
        # Encode each channel
        encoded_channels = []
        
        for ch in range(n_channels):
            ch_audio = multi_channel_mixture[:, ch:ch+1, :]  # [batch, 1, time]
            ch_encoded = self.encoders[ch](ch_audio)  # [batch, 256, time']
            encoded_channels.append(ch_encoded)
        
        # Concatenate channels
        encoded = torch.cat(encoded_channels, dim=1)  # [batch, 256 * n_channels, time']
        
        # Cross-channel attention
        # Reshape for attention: [time', batch, 256 * n_channels]
        encoded_t = encoded.permute(2, 0, 1)
        attended, _ = self.cross_channel_attention(encoded_t, encoded_t, encoded_t)
        attended = attended.permute(1, 2, 0)  # [batch, 256 * n_channels, time']
        
        # Separate
        masks = self.separator(attended)  # [batch, n_src, 256 * n_channels, time']
        
        # Apply masks and decode
        separated = []
        
        for src_idx in range(self.n_src):
            masked = attended * masks[:, src_idx, :, :]
            
            # Take first 256 channels for decoding
            masked_single = masked[:, :256, :]
            
            src_waveform = self.decoder(masked_single).squeeze(1)
            separated.append(src_waveform)
        
        separated = torch.stack(separated, dim=1)
        
        return separated

# Example: 4-microphone array
model = MultiChannelSeparator(n_channels=4, n_src=2)

# 4-channel input
multi_channel_audio = torch.randn(1, 4, 16000)

separated = model(multi_channel_audio)
print(f"Separated shape: {separated.shape}")  # [1, 2, 16000]
```

---

## Key Takeaways

✅ **Conv-TasNet** - State-of-the-art time-domain separation  
✅ **PIT loss** - Handle output permutation problem  
✅ **SI-SDR metric** - Scale-invariant quality measure  
✅ **Real-time streaming** - Chunk-based processing with overlap-add  
✅ **Integration with ASR** - End-to-end meeting transcription  

**Performance Targets:**
- SI-SDR improvement: > 15 dB
- Real-time factor: < 0.1 (10x faster than real-time)
- Latency: < 50ms for streaming
- Works with 2-5 overlapping speakers

---

**Originally published at:** [arunbaby.com/speech-tech/0011-speech-separation](https://www.arunbaby.com/speech-tech/0011-speech-separation/)

*If you found this helpful, consider sharing it with others who might benefit.*

