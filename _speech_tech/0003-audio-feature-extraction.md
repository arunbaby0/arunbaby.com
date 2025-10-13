---
title: "Audio Feature Extraction for Speech ML"
day: 3
collection: speech_tech
categories:
  - speech-tech
tags:
  - feature-extraction
  - mfcc
  - spectrograms
  - audio-processing
subdomain: Audio Processing
tech_stack: [Librosa, TorchAudio, Kaldi, NumPy]
latency_requirement: "< 10ms"
scale: "1M+ audio files"
companies: [Google, Meta, Amazon, Apple]
related_dsa_day: 3
related_ml_day: 3
---

**How to transform raw audio waveforms into ML-ready features that capture speech characteristics for robust model training.**

## Introduction

Raw audio waveforms are high-dimensional, noisy, and difficult for ML models to learn from directly. **Feature extraction** transforms audio into compact, informative representations that:

- Capture important speech characteristics
- Reduce dimensionality (16kHz audio = 16,000 samples/sec → ~40 features)
- Provide invariance to irrelevant variations (volume, recording device)
- Enable efficient model training

**Why it matters:**
- **Improves accuracy:** Good features → better models
- **Reduces compute:** Lower dimensionality = faster training/inference
- **Enables transfer learning:** Pre-extracted features work across tasks
- **Production efficiency:** Feature extraction can be cached

**What you'll learn:**
- Core audio features (MFCCs, spectrograms, mel-scale)
- Time-domain vs frequency-domain features
- Production-grade extraction pipelines
- Optimization for real-time processing
- Feature engineering for speech tasks

---

## Problem Definition

Design a feature extraction pipeline for speech ML systems.

### Functional Requirements

1. **Feature Types**
   - Time-domain features (energy, zero-crossing rate)
   - Frequency-domain features (spectrograms, MFCCs)
   - Temporal features (deltas, delta-deltas)
   - Learned features (embeddings)

2. **Input Handling**
   - Support multiple sample rates (8kHz, 16kHz, 48kHz)
   - Handle variable-length audio
   - Process both mono and stereo
   - Support batch processing

3. **Output Format**
   - Fixed-size feature vectors
   - Variable-length sequences
   - 2D/3D tensors for neural networks

### Non-Functional Requirements

1. **Performance**
   - Real-time: Extract features < 10ms for 1 sec audio
   - Batch: Process 10K files/hour on single machine
   - Memory: < 100MB RAM for streaming

2. **Quality**
   - Robust to noise
   - Consistent across devices
   - Reproducible (deterministic)

3. **Flexibility**
   - Configurable parameters
   - Support multiple backends (librosa, torchaudio)
   - Easy to extend with new features

---

## Audio Basics

### Waveform Representation

```python
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Load audio
audio, sr = librosa.load('speech.wav', sr=16000)

print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(audio) / sr:.2f} seconds")
print(f"Shape: {audio.shape}")
print(f"Range: [{audio.min():.3f}, {audio.max():.3f}]")

# Visualize waveform
plt.figure(figsize=(12, 4))
time = np.arange(len(audio)) / sr
plt.plot(time, audio)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')
plt.show()
```

**Key properties:**
- **Sample rate (sr):** Samples per second (e.g., 16000 Hz = 16000 samples/sec)
- **Duration:** `len(audio) / sr` seconds
- **Amplitude:** Typically normalized to [-1, 1]

---

## Feature 1: Mel-Frequency Cepstral Coefficients (MFCCs)

**MFCCs** are the most widely used features in speech recognition.

### Why MFCCs?

1. **Mimic human hearing:** Use mel scale (perceptual frequency scale)
2. **Compact:** Represent spectral envelope with 13-40 coefficients
3. **Robust:** Less sensitive to pitch variations
4. **Proven:** Gold standard for ASR for decades

### How MFCCs Work

```
Audio Waveform
    ↓
1. Pre-emphasis (boost high frequencies)
    ↓
2. Frame the signal (25ms windows, 10ms hop)
    ↓
3. Apply window function (Hamming)
    ↓
4. FFT (Fast Fourier Transform)
    ↓
5. Mel filterbank (map to mel scale)
    ↓
6. Log (compress dynamic range)
    ↓
7. DCT (Discrete Cosine Transform)
    ↓
MFCCs (13-40 coefficients per frame)
```

### Implementation

```python
import librosa
import numpy as np

class MFCCExtractor:
    """
    Extract MFCC features from audio
    
    Standard configuration for speech recognition
    """
    
    def __init__(
        self,
        sr=16000,
        n_mfcc=40,
        n_fft=512,
        hop_length=160,  # 10ms at 16kHz
        n_mels=40,
        fmin=20,
        fmax=8000
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCCs
        
        Args:
            audio: Audio waveform (1D array)
        
        Returns:
            MFCCs: (n_mfcc, time_steps)
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        return mfccs  # Shape: (n_mfcc, time)
    
    def extract_with_deltas(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCCs + deltas + delta-deltas
        
        Deltas capture temporal dynamics
        
        Returns:
            Features: (n_mfcc * 3, time_steps)
        """
        # MFCCs
        mfccs = self.extract(audio)
        
        # Delta (first derivative)
        delta = librosa.feature.delta(mfccs)
        
        # Delta-delta (second derivative)
        delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Stack
        features = np.vstack([mfccs, delta, delta2])  # (120, time)
        
        return features

# Usage
extractor = MFCCExtractor()
mfccs = extractor.extract(audio)
print(f"MFCCs shape: {mfccs.shape}")  # (40, time_steps)

# With deltas
features = extractor.extract_with_deltas(audio)
print(f"MFCCs+deltas shape: {features.shape}")  # (120, time_steps)
```

### Visualizing MFCCs

```python
import matplotlib.pyplot as plt

def plot_mfccs(mfccs, sr, hop_length):
    """Visualize MFCC features"""
    plt.figure(figsize=(12, 6))
    
    # Convert frame indices to time
    times = librosa.frames_to_time(
        np.arange(mfccs.shape[1]),
        sr=sr,
        hop_length=hop_length
    )
    
    plt.imshow(
        mfccs,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], 0, mfccs.shape[0]],
        cmap='viridis'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficient')
    plt.title('MFCC Features')
    plt.tight_layout()
    plt.show()

plot_mfccs(mfccs, sr=16000, hop_length=160)
```

---

## Feature 2: Mel-Spectrograms

**Mel-spectrograms** preserve more temporal detail than MFCCs.

### What is a Spectrogram?

A **spectrogram** shows how the frequency content of a signal changes over time.

- **X-axis:** Time
- **Y-axis:** Frequency
- **Color:** Magnitude (energy)

### Mel-Spectrogram vs MFCC

| Aspect | Mel-Spectrogram | MFCC |
|--------|-----------------|------|
| Dimensions | (n_mels, time) | (n_mfcc, time) |
| Information | Full spectrum | Spectral envelope |
| Size | 40-128 bins | 13-40 coefficients |
| Use case | CNNs, deep learning | Traditional ASR |
| Temporal resolution | Higher | Lower (due to DCT) |

### Implementation

```python
class MelSpectrogramExtractor:
    """
    Extract log mel-spectrogram features
    
    Popular for deep learning models (CNNs, Transformers)
    """
    
    def __init__(
        self,
        sr=16000,
        n_fft=512,
        hop_length=160,
        n_mels=80,
        fmin=0,
        fmax=8000
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log mel-spectrogram
        
        Returns:
            Log mel-spectrogram: (n_mels, time_steps)
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale (dB)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel  # Shape: (n_mels, time)
    
    def extract_normalized(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract and normalize to [0, 1]
        
        Better for neural networks
        """
        log_mel = self.extract(audio)
        
        # Normalize to [0, 1]
        log_mel_norm = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
        
        return log_mel_norm

# Usage
mel_extractor = MelSpectrogramExtractor(n_mels=80)
mel_spec = mel_extractor.extract(audio)
print(f"Mel-spectrogram shape: {mel_spec.shape}")  # (80, time_steps)
```

### Visualizing Mel-Spectrogram

```python
def plot_mel_spectrogram(mel_spec, sr, hop_length):
    """Visualize mel-spectrogram"""
    plt.figure(figsize=(12, 6))
    
    librosa.display.specshow(
        mel_spec,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='viridis'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.show()

plot_mel_spectrogram(mel_spec, sr=16000, hop_length=160)
```

---

## Feature 3: Raw Spectrograms (STFT)

**Short-Time Fourier Transform (STFT)** provides the highest frequency resolution.

### Implementation

```python
class STFTExtractor:
    """
    Extract raw STFT features
    
    Used when you need full frequency resolution
    """
    
    def __init__(
        self,
        n_fft=512,
        hop_length=160,
        win_length=400
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract magnitude spectrogram
        
        Returns:
            Spectrogram: (n_fft//2 + 1, time_steps)
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        # Get magnitude
        magnitude = np.abs(stft)
        
        # Convert to dB
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        return magnitude_db  # Shape: (n_fft//2 + 1, time)
    
    def extract_with_phase(self, audio: np.ndarray):
        """
        Extract magnitude and phase
        
        Phase information useful for reconstruction
        """
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        return magnitude, phase

# Usage
stft_extractor = STFTExtractor()
spectrogram = stft_extractor.extract(audio)
print(f"Spectrogram shape: {spectrogram.shape}")  # (257, time_steps)
```

---

## Feature 4: Time-Domain Features

Simple but effective features computed directly from waveform.

### Implementation

```python
class TimeDomainExtractor:
    """
    Extract time-domain features
    
    Fast to compute, useful for simple tasks
    """
    
    def extract_energy(self, audio: np.ndarray, frame_length=400, hop_length=160):
        """
        Frame-wise energy (RMS)
        
        Captures loudness/volume over time
        """
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        return energy
    
    def extract_zero_crossing_rate(self, audio: np.ndarray, frame_length=400, hop_length=160):
        """
        Zero-crossing rate
        
        Measures how often signal crosses zero
        High ZCR → noisy/unvoiced
        Low ZCR → tonal/voiced
        """
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        return zcr
    
    def extract_all(self, audio: np.ndarray):
        """Extract all time-domain features"""
        energy = self.extract_energy(audio)
        zcr = self.extract_zero_crossing_rate(audio)
        
        # Stack features
        features = np.vstack([energy, zcr])  # (2, time)
        
        return features

# Usage
time_extractor = TimeDomainExtractor()
time_features = time_extractor.extract_all(audio)
print(f"Time-domain features shape: {time_features.shape}")  # (2, time_steps)
```

---

## Feature 5: Pitch & Formants

**Pitch** and **formants** are linguistic features important for speech.

### Pitch Extraction

```python
class PitchExtractor:
    """
    Extract fundamental frequency (F0)
    
    Important for:
    - Speaker recognition
    - Emotion detection
    - Prosody modeling
    """
    
    def __init__(self, sr=16000, fmin=80, fmax=400):
        self.sr = sr
        self.fmin = fmin  # Typical male voice
        self.fmax = fmax  # Typical female voice
    
    def extract_f0(self, audio: np.ndarray, hop_length=160):
        """
        Extract pitch (fundamental frequency)
        
        Returns:
            f0: Pitch values (Hz) per frame
            voiced_flag: Boolean array (voiced vs unvoiced)
        """
        # Extract pitch using YIN algorithm
        f0 = librosa.yin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=hop_length
        )
        
        # Detect voiced regions (f0 > 0)
        voiced_flag = f0 > 0
        
        return f0, voiced_flag
    
    def extract_pitch_features(self, audio: np.ndarray):
        """
        Extract pitch statistics
        
        Useful for speaker/emotion recognition
        """
        f0, voiced = self.extract_f0(audio)
        
        # Statistics on voiced frames
        voiced_f0 = f0[voiced]
        
        if len(voiced_f0) > 0:
            features = {
                'mean_pitch': np.mean(voiced_f0),
                'std_pitch': np.std(voiced_f0),
                'min_pitch': np.min(voiced_f0),
                'max_pitch': np.max(voiced_f0),
                'pitch_range': np.max(voiced_f0) - np.min(voiced_f0),
                'voiced_ratio': np.sum(voiced) / len(voiced)
            }
        else:
            features = {k: 0.0 for k in ['mean_pitch', 'std_pitch', 'min_pitch', 'max_pitch', 'pitch_range', 'voiced_ratio']}
        
        return features

# Usage
pitch_extractor = PitchExtractor()
f0, voiced = pitch_extractor.extract_f0(audio)
print(f"Pitch shape: {f0.shape}")

pitch_stats = pitch_extractor.extract_pitch_features(audio)
print(f"Pitch statistics: {pitch_stats}")
```

---

## Production Feature Pipeline

Combine all features into a unified pipeline.

### Unified Feature Extractor

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    sr: int = 16000
    feature_types: List[str] = None  # ['mfcc', 'mel', 'pitch']
    
    # MFCC config
    n_mfcc: int = 40
    
    # Mel-spectrogram config
    n_mels: int = 80
    
    # Common config
    n_fft: int = 512
    hop_length: int = 160  # 10ms
    
    # Normalization
    normalize: bool = True
    
    def __post_init__(self):
        if self.feature_types is None:
            self.feature_types = ['mfcc']

class AudioFeatureExtractor:
    """
    Production-grade audio feature extractor
    
    Supports multiple feature types, caching, and batch processing
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
        # Initialize extractors
        self.mfcc_extractor = MFCCExtractor(
            sr=config.sr,
            n_mfcc=config.n_mfcc,
            n_fft=config.n_fft,
            hop_length=config.hop_length
        )
        
        self.mel_extractor = MelSpectrogramExtractor(
            sr=config.sr,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length
        )
        
        self.pitch_extractor = PitchExtractor(sr=config.sr)
        self.time_extractor = TimeDomainExtractor()
    
    def extract(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features based on config
        
        Args:
            audio: Audio waveform
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        if 'mfcc' in self.config.feature_types:
            mfccs = self.mfcc_extractor.extract_with_deltas(audio)
            if self.config.normalize:
                mfccs = self._normalize(mfccs)
            features['mfcc'] = mfccs
        
        if 'mel' in self.config.feature_types:
            mel = self.mel_extractor.extract(audio)
            if self.config.normalize:
                mel = self._normalize(mel)
            features['mel'] = mel
        
        if 'pitch' in self.config.feature_types:
            f0, voiced = self.pitch_extractor.extract_f0(audio, hop_length=self.config.hop_length)
            features['pitch'] = f0
            features['voiced'] = voiced.astype(np.float32)
        
        if 'time' in self.config.feature_types:
            time_feats = self.time_extractor.extract_all(audio)
            if self.config.normalize:
                time_feats = self._normalize(time_feats)
            features['time'] = time_feats
        
        return features
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features (mean=0, std=1) per coefficient
        """
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True) + 1e-8
        
        normalized = (features - mean) / std
        
        return normalized
    
    def extract_from_file(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract features from audio file
        """
        audio, sr = librosa.load(audio_path, sr=self.config.sr)
        return self.extract(audio)
    
    def extract_batch(self, audio_list: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """
        Extract features from batch of audio
        """
        return [self.extract(audio) for audio in audio_list]
    
    def save_config(self, path: str):
        """Save feature extraction config"""
        with open(path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    @staticmethod
    def load_config(path: str) -> FeatureConfig:
        """Load feature extraction config"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return FeatureConfig(**config_dict)

# Usage
config = FeatureConfig(
    feature_types=['mfcc', 'mel', 'pitch'],
    n_mfcc=40,
    n_mels=80,
    normalize=True
)

extractor = AudioFeatureExtractor(config)

# Extract features
features = extractor.extract(audio)
print("Extracted features:", features.keys())
for name, feat in features.items():
    print(f"  {name}: {feat.shape}")

# Save config for reproducibility
extractor.save_config('feature_config.json')
```

---

## Handling Variable-Length Audio

Different audio clips have different durations. Need to handle this for ML.

### Strategy 1: Padding/Truncation

```python
class VariableLengthHandler:
    """
    Handle variable-length audio
    """
    
    def pad_or_truncate(self, features: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad or truncate features to fixed length
        
        Args:
            features: (n_features, time)
            target_length: Target time dimension
        
        Returns:
            Fixed-length features: (n_features, target_length)
        """
        current_length = features.shape[1]
        
        if current_length < target_length:
            # Pad with zeros
            pad_width = ((0, 0), (0, target_length - current_length))
            features = np.pad(features, pad_width, mode='constant')
        elif current_length > target_length:
            # Truncate (take first target_length frames)
            features = features[:, :target_length]
        
        return features
    
    def create_mask(self, features: np.ndarray, target_length: int) -> np.ndarray:
        """
        Create attention mask for padded features
        
        Returns:
            Mask: (target_length,) - 1 for real frames, 0 for padding
        """
        current_length = features.shape[1]
        
        mask = np.zeros(target_length)
        mask[:min(current_length, target_length)] = 1
        
        return mask
```

### Strategy 2: Temporal Pooling

```python
class TemporalPooler:
    """
    Pool variable-length features to fixed size
    """
    
    def mean_pool(self, features: np.ndarray) -> np.ndarray:
        """
        Average pool over time
        
        Args:
            features: (n_features, time)
        
        Returns:
            Pooled: (n_features,)
        """
        return np.mean(features, axis=1)
    
    def max_pool(self, features: np.ndarray) -> np.ndarray:
        """Max pool over time"""
        return np.max(features, axis=1)
    
    def stats_pool(self, features: np.ndarray) -> np.ndarray:
        """
        Statistical pooling: mean + std
        
        Returns:
            Pooled: (n_features * 2,)
        """
        mean = np.mean(features, axis=1)
        std = np.std(features, axis=1)
        
        return np.concatenate([mean, std])
```

---

## Real-Time Feature Extraction

For streaming applications, need incremental feature extraction.

### Streaming Feature Extractor

```python
from collections import deque

class StreamingFeatureExtractor:
    """
    Extract features from streaming audio
    
    Use case: Real-time ASR, voice assistants
    """
    
    def __init__(
        self,
        sr=16000,
        frame_length_ms=25,
        hop_length_ms=10,
        buffer_duration_ms=500
    ):
        self.sr = sr
        self.frame_length = int(sr * frame_length_ms / 1000)
        self.hop_length = int(sr * hop_length_ms / 1000)
        self.buffer_length = int(sr * buffer_duration_ms / 1000)
        
        # Circular buffer for audio
        self.buffer = deque(maxlen=self.buffer_length)
        
        # Feature extractor
        self.extractor = MFCCExtractor(
            sr=sr,
            hop_length=self.hop_length
        )
    
    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Add new audio chunk to buffer
        
        Args:
            audio_chunk: New audio samples
        """
        self.buffer.extend(audio_chunk)
    
    def extract_latest(self) -> Optional[np.ndarray]:
        """
        Extract features from current buffer
        
        Returns:
            Features or None if buffer too small
        """
        if len(self.buffer) < self.frame_length:
            return None
        
        # Convert buffer to array
        audio = np.array(self.buffer)
        
        # Extract features
        features = self.extractor.extract(audio)
        
        return features
    
    def reset(self):
        """Clear buffer"""
        self.buffer.clear()

# Usage
streaming_extractor = StreamingFeatureExtractor()

# Simulate streaming (100ms chunks)
chunk_size = 1600  # 100ms at 16kHz

for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i+chunk_size]
    
    # Add to buffer
    streaming_extractor.add_audio_chunk(chunk)
    
    # Extract features
    features = streaming_extractor.extract_latest()
    
    if features is not None:
        print(f"Chunk {i//chunk_size}: features shape = {features.shape}")
        # Process features (send to model, etc.)
```

---

## Performance Optimization

### 1. Caching Features

```python
import os
import pickle
import hashlib

class CachedFeatureExtractor:
    """
    Cache extracted features to disk
    
    Avoid re-extracting for same audio
    """
    
    def __init__(self, extractor: AudioFeatureExtractor, cache_dir='./feature_cache'):
        self.extractor = extractor
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, audio_path: str) -> str:
        """Generate cache file path based on audio path hash"""
        path_hash = hashlib.md5(audio_path.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{path_hash}.pkl")
    
    def extract_from_file(self, audio_path: str, use_cache=True) -> Dict[str, np.ndarray]:
        """
        Extract features with caching
        """
        cache_path = self._get_cache_path(audio_path)
        
        # Check cache
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                features = pickle.load(f)
            return features
        
        # Extract features
        features = self.extractor.extract_from_file(audio_path)
        
        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
        
        return features
```

### 2. Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

class ParallelFeatureExtractor:
    """
    Extract features from multiple files in parallel
    """
    
    def __init__(self, extractor: AudioFeatureExtractor, n_workers=4):
        self.extractor = extractor
        self.n_workers = n_workers
    
    def extract_from_files(self, audio_paths: List[str]) -> List[Dict[str, np.ndarray]]:
        """
        Extract features from multiple files in parallel
        """
        with Pool(self.n_workers) as pool:
            features_list = pool.map(
                self.extractor.extract_from_file,
                audio_paths
            )
        
        return features_list

# Usage
parallel_extractor = ParallelFeatureExtractor(extractor, n_workers=8)
audio_files = ['file1.wav', 'file2.wav', ...]  # 1000s of files
features = parallel_extractor.extract_from_files(audio_files)
```

---

## Advanced Feature Types

### 1. Learned Features (Embeddings)

Instead of hand-crafted features, learn representations from data.

```python
import torch
import torch.nn as nn

class AudioEmbeddingExtractor(nn.Module):
    """
    Extract learned audio embeddings
    
    Use pre-trained models (wav2vec, HuBERT) as feature extractors
    """
    
    def __init__(self, model_name='facebook/wav2vec2-base'):
        super().__init__()
        from transformers import Wav2Vec2Model
        
        # Load pre-trained model
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()  # Freeze for feature extraction
    
    def extract(self, audio: np.ndarray, sr=16000) -> np.ndarray:
        """
        Extract contextualized embeddings
        
        Returns:
            Embeddings: (time_steps, hidden_dim)
                typically (time, 768) for base model
        """
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(audio_tensor)
            embeddings = outputs.last_hidden_state[0]  # (time, 768)
        
        return embeddings.numpy()

# Usage - MUCH more powerful than MFCCs for transfer learning
embedding_extractor = AudioEmbeddingExtractor()
embeddings = embedding_extractor.extract(audio)
print(f"Embeddings shape: {embeddings.shape}")  # (time, 768)
```

**Comparison:**

| Feature Type | Dimension | Training Required | Transfer Learning | Accuracy |
|--------------|-----------|-------------------|-------------------|----------|
| MFCCs | 40-120 | No | Poor | Baseline |
| Mel-spectrogram | 80-128 | No | Good | +5-10% |
| Wav2Vec embeddings | 768 | Yes (pre-trained) | Excellent | +15-25% |

### 2. Filter Bank Features (FBank)

Alternative to MFCCs - skip the DCT step.

```python
class FilterbankExtractor:
    """
    Extract log mel-filterbank features
    
    Similar to mel-spectrograms, popular in modern ASR
    """
    
    def __init__(self, sr=16000, n_mels=80, n_fft=512, hop_length=160):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log filter bank energies
        
        Returns:
            FBank: (n_mels, time_steps)
        """
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Log
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel

# FBank vs MFCC:
# - FBank: Keep all mel bins (80-128)
# - MFCC: Compress to 13-40 via DCT
# 
# FBank often works better with neural networks
```

### 3. Prosodic Features

Capture rhythm, stress, and intonation.

```python
class ProsodicFeatureExtractor:
    """
    Extract prosodic features for emotion, speaker ID, etc.
    """
    
    def extract_intensity_contour(self, audio, sr=16000, hop_length=160):
        """
        Intensity (loudness) over time
        """
        intensity = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Convert to dB
        intensity_db = librosa.amplitude_to_db(intensity, ref=np.max)
        
        return intensity_db
    
    def extract_speaking_rate(self, audio, sr=16000):
        """
        Estimate speaking rate (syllables per second)
        
        Approximation: count peaks in energy envelope
        """
        # Energy envelope
        energy = librosa.feature.rms(y=audio, hop_length=160)[0]
        
        # Find peaks (local maxima)
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(energy, distance=10, prominence=0.1)
        
        # Speaking rate
        duration = len(audio) / sr
        syllables_per_sec = len(peaks) / duration
        
        return syllables_per_sec
    
    def extract_all_prosodic(self, audio, sr=16000):
        """Extract all prosodic features"""
        
        # Pitch
        pitch_extractor = PitchExtractor(sr=sr)
        pitch_stats = pitch_extractor.extract_pitch_features(audio)
        
        # Intensity
        intensity = self.extract_intensity_contour(audio, sr)
        
        # Speaking rate
        speaking_rate = self.extract_speaking_rate(audio, sr)
        
        return {
            **pitch_stats,
            'mean_intensity': np.mean(intensity),
            'std_intensity': np.std(intensity),
            'speaking_rate': speaking_rate
        }
```

---

## Feature Quality & Validation

Ensure extracted features are high quality.

### Feature Quality Metrics

```python
class FeatureQualityChecker:
    """
    Validate quality of extracted features
    """
    
    def check_for_nans(self, features: Dict[str, np.ndarray]) -> bool:
        """Check for NaN/Inf values"""
        for name, feat in features.items():
            if np.isnan(feat).any() or np.isinf(feat).any():
                print(f"⚠️  {name} contains NaN/Inf")
                return False
        return True
    
    def check_dynamic_range(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Check dynamic range of features
        
        Low dynamic range → feature not informative
        """
        ranges = {}
        
        for name, feat in features.items():
            feat_range = feat.max() - feat.min()
            ranges[name] = feat_range
            
            if feat_range < 1e-6:
                print(f"⚠️  {name} has very low dynamic range: {feat_range}")
        
        return ranges
    
    def check_feature_statistics(self, features_batch: List[np.ndarray]):
        """
        Check statistics across batch
        
        Ensure features are properly normalized
        """
        # Stack all features
        all_features = np.concatenate(features_batch, axis=1)  # (n_features, total_time)
        
        # Per-feature statistics
        mean_per_feature = np.mean(all_features, axis=1)
        std_per_feature = np.std(all_features, axis=1)
        
        print("Feature Statistics:")
        print(f"  Mean range: [{mean_per_feature.min():.3f}, {mean_per_feature.max():.3f}]")
        print(f"  Std range: [{std_per_feature.min():.3f}, {std_per_feature.max():.3f}]")
        
        # Check if normalized
        if np.abs(mean_per_feature).max() > 0.1:
            print("⚠️  Features not centered (mean far from 0)")
        
        if np.abs(std_per_feature - 1.0).max() > 0.2:
            print("⚠️  Features not standardized (std far from 1)")
```

---

## Connection to Data Preprocessing Pipeline

Feature extraction for speech is analogous to data preprocessing for ML systems (see Day 3 ML).

### Parallel Concepts

| Speech Feature Extraction | ML Data Preprocessing |
|---------------------------|----------------------|
| Handle missing audio | Handle missing values |
| Normalize features (mean=0, std=1) | Normalize numerical features |
| Pad/truncate variable length | Handle variable-length sequences |
| Validate audio quality | Schema validation |
| Cache extracted features | Cache preprocessed data |
| Batch processing | Distributed data processing |

### Unified Preprocessing Framework

```python
class UnifiedPreprocessor:
    """
    Combined preprocessing for multimodal ML
    
    Example: Speech + text + metadata
    """
    
    def __init__(self):
        # Audio features
        self.audio_extractor = AudioFeatureExtractor(
            FeatureConfig(feature_types=['mfcc', 'mel'])
        )
        
        # Text features (from transcripts)
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.text_vectorizer = TfidfVectorizer(max_features=1000)
        
        # Numerical features
        from sklearn.preprocessing import StandardScaler
        self.numerical_scaler = StandardScaler()
    
    def preprocess_sample(self, audio, text, metadata):
        """
        Preprocess multimodal sample
        
        Args:
            audio: Audio waveform
            text: Transcript or description
            metadata: User/item metadata (dict)
        
        Returns:
            Combined feature vector
        """
        # Extract audio features
        audio_features = self.audio_extractor.extract(audio)
        audio_pooled = np.mean(audio_features['mfcc'], axis=1)  # (n_mfcc,)
        
        # Extract text features
        text_features = self.text_vectorizer.transform([text]).toarray()[0]  # (1000,)
        
        # Process metadata
        metadata_array = np.array([
            metadata['user_age'],
            metadata['user_gender'],
            metadata['device_type']
        ])
        metadata_scaled = self.numerical_scaler.transform([metadata_array])[0]
        
        # Concatenate all features
        combined = np.concatenate([
            audio_pooled,      # (40,)
            text_features,     # (1000,)
            metadata_scaled    # (3,)
        ])  # Total: (1043,)
        
        return combined
```

---

## Production Best Practices

### 1. Feature Versioning

Track feature extraction versions for reproducibility.

```python
class VersionedFeatureExtractor:
    """
    Version feature extraction logic
    
    Critical for:
    - A/B testing different features
    - Rollback if new features hurt performance
    - Reproducibility
    """
    
    VERSION = "1.2.0"
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.extractor = AudioFeatureExtractor(config)
    
    def extract_with_metadata(self, audio_path: str):
        """
        Extract features with version metadata
        """
        features = self.extractor.extract_from_file(audio_path)
        
        metadata = {
            'version': self.VERSION,
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'audio_path': audio_path
        }
        
        return {
            'features': features,
            'metadata': metadata
        }
    
    def save_features(self, features, output_path):
        """Save features with version info"""
        np.savez_compressed(
            output_path,
            **features['features'],
            metadata=json.dumps(features['metadata'])
        )
```

### 2. Error Handling

Robust feature extraction handles failures gracefully.

```python
class RobustFeatureExtractor:
    """
    Feature extractor with error handling
    """
    
    def __init__(self, extractor: AudioFeatureExtractor):
        self.extractor = extractor
    
    def extract_safe(self, audio_path: str) -> Optional[Dict]:
        """
        Extract features with error handling
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.extractor.config.sr)
            
            # Validate
            if len(audio) == 0:
                logger.warning(f"Empty audio: {audio_path}")
                return None
            
            if len(audio) < self.extractor.config.sr * 0.1:  # < 100ms
                logger.warning(f"Audio too short: {audio_path}")
                return None
            
            # Extract
            features = self.extractor.extract(audio)
            
            # Quality check
            quality_checker = FeatureQualityChecker()
            if not quality_checker.check_for_nans(features):
                logger.error(f"Feature extraction failed (NaN): {audio_path}")
                return None
            
            return features
        
        except Exception as e:
            logger.error(f"Feature extraction error for {audio_path}: {e}")
            return None
    
    def extract_batch_robust(self, audio_paths: List[str]) -> List[Dict]:
        """
        Extract from batch, skipping failures
        """
        results = []
        failures = []
        
        for path in audio_paths:
            features = self.extract_safe(path)
            if features is not None:
                results.append({'path': path, 'features': features})
            else:
                failures.append(path)
        
        success_rate = len(results) / len(audio_paths)
        logger.info(f"Feature extraction: {len(results)}/{len(audio_paths)} succeeded ({success_rate:.1%})")
        
        if failures:
            logger.warning(f"Failed files: {failures[:10]}")  # Log first 10
        
        return results
```

### 3. Monitoring Feature Quality

Track feature statistics over time to detect issues.

```python
class FeatureMonitor:
    """
    Monitor feature quality in production
    """
    
    def __init__(self, expected_stats: Dict[str, Dict]):
        """
        Args:
            expected_stats: Expected statistics per feature type
                {
                    'mfcc': {'mean_range': [-5, 5], 'std_range': [0.5, 2.0]},
                    'mel': {'mean_range': [-80, 0], 'std_range': [10, 30]}
                }
        """
        self.expected_stats = expected_stats
    
    def validate_features(self, features: Dict[str, np.ndarray]) -> List[str]:
        """
        Validate extracted features against expected statistics
        
        Returns:
            List of warnings
        """
        warnings = []
        
        for feat_name, feat_values in features.items():
            if feat_name not in self.expected_stats:
                continue
            
            expected = self.expected_stats[feat_name]
            
            # Check mean
            actual_mean = np.mean(feat_values)
            expected_mean_range = expected['mean_range']
            
            if not (expected_mean_range[0] <= actual_mean <= expected_mean_range[1]):
                warnings.append(
                    f"{feat_name}: mean {actual_mean:.2f} outside expected range {expected_mean_range}"
                )
            
            # Check std
            actual_std = np.std(feat_values)
            expected_std_range = expected['std_range']
            
            if not (expected_std_range[0] <= actual_std <= expected_std_range[1]):
                warnings.append(
                    f"{feat_name}: std {actual_std:.2f} outside expected range {expected_std_range}"
                )
        
        return warnings
    
    def compute_statistics(self, features_batch: List[Dict[str, np.ndarray]]):
        """
        Compute statistics across batch
        
        Use to establish baseline expected_stats
        """
        stats = {}
        
        # Get feature names from first sample
        feature_names = features_batch[0].keys()
        
        for feat_name in feature_names:
            # Collect all values
            all_values = np.concatenate([
                f[feat_name].flatten() for f in features_batch
            ])
            
            stats[feat_name] = {
                'mean': np.mean(all_values),
                'std': np.std(all_values),
                'min': np.min(all_values),
                'max': np.max(all_values),
                'percentiles': {
                    '25': np.percentile(all_values, 25),
                    '50': np.percentile(all_values, 50),
                    '75': np.percentile(all_values, 75),
                    '95': np.percentile(all_values, 95)
                }
            }
        
        return stats
```

---

## Data Augmentation in Feature Space

Augment features directly for training robustness.

### SpecAugment

```python
class SpecAugment:
    """
    SpecAugment: Data augmentation on spectrograms
    
    Proposed in "SpecAugment: A Simple Data Augmentation Method for ASR" (Google, 2019)
    
    Improves ASR accuracy by 10-20% on many benchmarks
    """
    
    def __init__(
        self,
        time_mask_param=70,
        freq_mask_param=15,
        num_time_masks=2,
        num_freq_masks=2
    ):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
    
    def time_mask(self, spec: np.ndarray) -> np.ndarray:
        """
        Mask random time region
        
        Sets random time frames to zero
        """
        spec = spec.copy()
        time_length = spec.shape[1]
        
        for _ in range(self.num_time_masks):
            t = np.random.randint(0, min(self.time_mask_param, time_length))
            t0 = np.random.randint(0, time_length - t)
            spec[:, t0:t0+t] = 0
        
        return spec
    
    def freq_mask(self, spec: np.ndarray) -> np.ndarray:
        """
        Mask random frequency region
        
        Sets random frequency bins to zero
        """
        spec = spec.copy()
        freq_length = spec.shape[0]
        
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, min(self.freq_mask_param, freq_length))
            f0 = np.random.randint(0, freq_length - f)
            spec[f0:f0+f, :] = 0
        
        return spec
    
    def augment(self, spec: np.ndarray) -> np.ndarray:
        """Apply both time and freq masking"""
        spec = self.time_mask(spec)
        spec = self.freq_mask(spec)
        return spec

# Usage during training
augmenter = SpecAugment()

for audio, label in train_loader:
    # Extract features
    mel_spec = mel_extractor.extract(audio)
    
    # Augment
    mel_spec_aug = augmenter.augment(mel_spec)
    
    # Train model
    train_model(mel_spec_aug, label)
```

---

## Batch Feature Extraction for Training

Extract features for entire dataset efficiently.

### Batch Extraction Pipeline

```python
import os
from pathlib import Path
from tqdm import tqdm
import h5py

class BatchFeatureExtractor:
    """
    Extract features for large audio datasets
    
    Use case: Prepare training data
    - Extract once, train many times
    - Save features to disk (HDF5 format)
    """
    
    def __init__(self, extractor: AudioFeatureExtractor, n_workers=8):
        self.extractor = extractor
        self.n_workers = n_workers
    
    def extract_dataset(
        self,
        audio_dir: str,
        output_path: str,
        max_length_frames: int = 1000
    ):
        """
        Extract features for all audio files in directory
        
        Args:
            audio_dir: Directory containing .wav files
            output_path: HDF5 file to save features
            max_length_frames: Pad/truncate to this length
        """
        # Find all audio files
        audio_files = list(Path(audio_dir).rglob('*.wav'))
        print(f"Found {len(audio_files)} audio files")
        
        # Create HDF5 file
        with h5py.File(output_path, 'w') as hf:
            # Pre-allocate datasets
            # (We'll store features for each type)
            feature_dim = self.extractor.config.n_mfcc * 3  # MFCCs + deltas
            
            features_dataset = hf.create_dataset(
                'features',
                shape=(len(audio_files), feature_dim, max_length_frames),
                dtype='float32'
            )
            
            lengths_dataset = hf.create_dataset(
                'lengths',
                shape=(len(audio_files),),
                dtype='int32'
            )
            
            # Store file paths
            paths_dataset = hf.create_dataset(
                'paths',
                shape=(len(audio_files),),
                dtype=h5py.string_dtype()
            )
            
            # Extract features
            for idx, audio_path in enumerate(tqdm(audio_files)):
                try:
                    # Load audio
                    audio, sr = librosa.load(str(audio_path), sr=self.extractor.config.sr)
                    
                    # Extract features
                    features = self.extractor.extract(audio)
                    
                    # Get MFCCs with deltas
                    mfcc_deltas = features['mfcc']  # (120, time)
                    
                    # Pad or truncate
                    handler = VariableLengthHandler()
                    mfcc_fixed = handler.pad_or_truncate(mfcc_deltas, max_length_frames)
                    
                    # Store
                    features_dataset[idx] = mfcc_fixed
                    lengths_dataset[idx] = min(mfcc_deltas.shape[1], max_length_frames)
                    paths_dataset[idx] = str(audio_path)
                
                except Exception as e:
                    logger.error(f"Failed to process {audio_path}: {e}")
                    # Store zeros for failed files
                    features_dataset[idx] = np.zeros((feature_dim, max_length_frames))
                    lengths_dataset[idx] = 0
                    paths_dataset[idx] = str(audio_path)
        
        print(f"Features saved to {output_path}")

# Usage
batch_extractor = BatchFeatureExtractor(extractor, n_workers=8)
batch_extractor.extract_dataset(
    audio_dir='./data/train/',
    output_path='./features/train_features.h5',
    max_length_frames=1000
)

# Load for training
with h5py.File('./features/train_features.h5', 'r') as hf:
    features = hf['features'][:]  # (N, feature_dim, max_length)
    lengths = hf['lengths'][:]    # (N,)
    paths = hf['paths'][:]        # (N,)
```

---

## Real-World Systems

### Kaldi: Traditional ASR Feature Pipeline

Kaldi is the industry standard for traditional ASR.

**Feature extraction:**
```bash
# Kaldi feature extraction (MFCC + pitch)
compute-mfcc-feats --config=conf/mfcc.conf scp:wav.scp ark:mfcc.ark
compute-and-process-kaldi-pitch-feats scp:wav.scp ark:pitch.ark

# Combine features
paste-feats ark:mfcc.ark ark:pitch.ark ark:features.ark
```

**Configuration (mfcc.conf):**
```
--use-energy=true
--num-mel-bins=40
--num-ceps=40
--low-freq=20
--high-freq=8000
--sample-frequency=16000
```

### PyTorch: Modern Deep Learning Pipeline

```python
import torchaudio
import torch

class TorchAudioExtractor:
    """
    Feature extraction using torchaudio
    
    Benefits:
    - GPU acceleration
    - Differentiable (can backprop through features)
    - Integrated with PyTorch training
    """
    
    def __init__(self, sr=16000, n_mfcc=40, n_mels=80):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        
        # Create transforms (can move to GPU)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={'n_mels': 40, 'n_fft': 512, 'hop_length': 160}
        )
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=512,
            hop_length=160,
            n_mels=n_mels
        )
        
        # Amplitude → dB conversion
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
    
    def to(self, device):
        """
        Move transforms to a device (CPU/GPU) and return self.
        """
        self.mfcc_transform = self.mfcc_transform.to(device)
        self.mel_transform = self.mel_transform.to(device)
        self.db_transform = self.db_transform.to(device)
        return self
    
    def extract(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features (GPU-accelerated if audio on GPU)
        
        Args:
            audio: (batch, time) or (time,)
        
        Returns:
            Dictionary of features
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension
        
        # Extract
        mfccs = self.mfcc_transform(audio)  # (batch, n_mfcc, time)
        mel = self.mel_transform(audio)     # (batch, n_mels, time)
        mel_db = self.db_transform(mel)
        
        return {
            'mfcc': mfccs,
            'mel': mel_db
        }

# Usage with GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

extractor = TorchAudioExtractor().to(device)

# Load audio
audio, sr = torchaudio.load('speech.wav')
audio = audio.to(device)

# Extract (on GPU)
features = extractor.extract(audio)
```

### Google: Production ASR Feature Extraction

**Stack:**
- **Input:** 16kHz audio
- **Features:** 80-bin log mel-filterbank
- **Augmentation:** SpecAugment
- **Normalization:** Per-utterance mean/variance normalization
- **Model:** Transformer encoder-decoder

**Key optimizations:**
- Precompute features for training data
- On-the-fly extraction for inference
- GPU-accelerated extraction for real-time systems

---

## Choosing the Right Features

Different tasks need different features.

### Feature Selection Guide

| Task | Best Features | Why |
|------|---------------|-----|
| **ASR (traditional)** | MFCCs + deltas | Captures phonetic content |
| **ASR (deep learning)** | Mel-spectrograms | Works well with CNNs |
| **Speaker Recognition** | MFCCs + pitch + prosody | Speaker identity in pitch/prosody |
| **Emotion Recognition** | Prosodic + spectral | Emotion in prosody + voice quality |
| **Keyword Spotting** | Mel-spectrograms | Simple, fast with CNNs |
| **Speech Enhancement** | STFT magnitude + phase | Need phase for reconstruction |
| **Voice Activity Detection** | Energy + ZCR | Simple features sufficient |

### Combining Features

```python
class MultiFeatureExtractor:
    """
    Combine multiple feature types
    
    Different features capture different aspects
    """
    
    def __init__(self):
        self.mfcc_ext = MFCCExtractor()
        self.pitch_ext = PitchExtractor()
        self.prosody_ext = ProsodicFeatureExtractor()
    
    def extract_combined(self, audio):
        """
        Extract and combine multiple feature types
        """
        # MFCCs (40, time)
        mfccs = self.mfcc_ext.extract(audio)
        
        # Pitch (time,)
        pitch, voiced = self.pitch_ext.extract_f0(audio)
        pitch = pitch.reshape(1, -1)  # (1, time)
        
        # Energy (1, time)
        energy = librosa.feature.rms(y=audio, hop_length=160)
        
        # Align all features to same time dimension
        min_time = min(mfccs.shape[1], pitch.shape[1], energy.shape[1])
        
        mfccs = mfccs[:, :min_time]
        pitch = pitch[:, :min_time]
        energy = energy[:, :min_time]
        
        # Stack
        combined = np.vstack([mfccs, pitch, energy])  # (42, time)
        
        return combined
```

---

## Key Takeaways

✅ **MFCCs** are standard for speech recognition - compact and robust  
✅ **Mel-spectrograms** work better with deep learning (CNNs, Transformers)  
✅ **Delta features** capture temporal dynamics - critical for accuracy  
✅ **Normalize features** for stable training (mean=0, std=1)  
✅ **Handle variable length** with padding, pooling, or attention masks  
✅ **Cache features** for repeated use - major speedup in training  
✅ **Streaming extraction** possible with circular buffers  
✅ **Parallel processing** speeds up batch feature extraction  
✅ **SpecAugment** improves robustness through feature-space augmentation  
✅ **Monitor feature quality** to detect pipeline issues early  
✅ **Version features** for reproducibility and A/B testing  
✅ **Choose features based on task** - no one-size-fits-all

---

**Originally published at:** [arunbaby.com/speech-tech/0003-audio-feature-extraction](https://www.arunbaby.com/speech-tech/0003-audio-feature-extraction/)

*If you found this helpful, consider sharing it with others who might benefit.*

