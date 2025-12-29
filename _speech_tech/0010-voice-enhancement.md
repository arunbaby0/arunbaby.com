---
title: "Voice Enhancement & Noise Reduction"
day: 10
related_dsa_day: 10
related_ml_day: 10
related_agents_day: 10
collection: speech_tech
categories:
 - speech-tech
tags:
 - noise-reduction
 - speech-enhancement
 - signal-processing
 - deep-learning
 - audio-quality
subdomain: Audio Processing
tech_stack: [Python, PyTorch, librosa, scipy, noisereduce, NVIDIA NeMo]
scale: "Real-time processing, low latency"
companies: [Krisp, Dolby, NVIDIA, Zoom, Microsoft, Google, Meta]
---

**Build systems that enhance voice quality by removing noise, improving intelligibility, and optimizing audio for speech applications.**

## Introduction

**Voice enhancement** improves speech quality by:
- Removing background noise (traffic, wind, keyboard)
- Suppressing reverberation
- Normalizing volume levels
- Enhancing speech intelligibility
- Removing artifacts and distortion

**Critical for:**
- Video conferencing (Zoom, Teams, Meet)
- Voice assistants (Alexa, Siri, Google Assistant)
- Podcast/content creation
- Hearing aids
- Telecommunication
- Speech recognition systems

**Key challenges:**
- Real-time processing (< 50ms latency)
- Preserving speech quality
- Handling diverse noise types
- Low computational cost
- Avoiding artifacts

---

## Problem Formulation

### Input/Output

``
Input: Noisy speech signal
 y(t) = s(t) + n(t)
 where:
 s(t) = clean speech
 n(t) = noise

Output: Enhanced speech signal
 ŝ(t) ≈ s(t)
 
Goal: Minimize ‖ŝ(t) - s(t)‖ while maintaining naturalness
``

### Quality Metrics

``python
import numpy as np
from scipy import signal

def calculate_snr(clean_speech, noisy_speech):
 """
 Calculate Signal-to-Noise Ratio
 
 SNR = 10 * log10(P_signal / P_noise)
 
 Higher is better (typically 10-30 dB)
 """
 signal_power = np.mean(clean_speech ** 2)
 noise = noisy_speech - clean_speech
 noise_power = np.mean(noise ** 2)
 
 if noise_power == 0:
 return float('inf')
 
 snr = 10 * np.log10(signal_power / noise_power)
 return snr

def calculate_pesq(reference, degraded, sr=16000):
 """
 Calculate PESQ (Perceptual Evaluation of Speech Quality)
 
 Range: -0.5 to 4.5 (higher is better)
 Industry standard for speech quality
 """
 from pesq import pesq
 
 # PESQ requires 8kHz or 16kHz
 if sr not in [8000, 16000]:
 raise ValueError("PESQ requires sr=8000 or sr=16000")
 
 mode = 'nb' if sr == 8000 else 'wb'
 score = pesq(sr, reference, degraded, mode)
 
 return score

def calculate_stoi(clean, enhanced, sr=16000):
 """
 Calculate STOI (Short-Time Objective Intelligibility)
 
 Range: 0 to 1 (higher is better)
 Correlates well with speech intelligibility
 """
 from pystoi import stoi
 
 score = stoi(clean, enhanced, sr, extended=False)
 return score

# Usage
clean = np.random.randn(16000) # 1 second at 16kHz
noisy = clean + 0.1 * np.random.randn(16000)

snr = calculate_snr(clean, noisy)
print(f"SNR: {snr:.2f} dB")

# pesq_score = calculate_pesq(clean, noisy, sr=16000)
# print(f"PESQ: {pesq_score:.2f}")
``

---

## Classical Methods

### 1. Spectral Subtraction

**Subtract noise spectrum from noisy spectrum**

``python
import librosa
import numpy as np

class SpectralSubtraction:
 """
 Classic spectral subtraction for noise reduction
 
 Steps:
 1. Estimate noise spectrum (from silence periods)
 2. Subtract from noisy spectrum
 3. Half-wave rectification
 4. Reconstruct signal
 """
 
 def __init__(self, n_fft=512, hop_length=128):
 self.n_fft = n_fft
 self.hop_length = hop_length
 self.noise_profile = None
 
 def estimate_noise(self, noise_audio, sr=16000):
 """
 Estimate noise spectrum from noise-only segment
 
 Args:
 noise_audio: Audio containing only noise
 """
 # STFT of noise
 noise_stft = librosa.stft(
 noise_audio,
 n_fft=self.n_fft,
 hop_length=self.hop_length
 )
 
 # Average magnitude spectrum
 self.noise_profile = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
 
 def enhance(self, noisy_audio, alpha=2.0, beta=0.002):
 """
 Apply spectral subtraction
 
 Args:
 noisy_audio: Noisy speech signal
 alpha: Over-subtraction factor (higher = more aggressive)
 beta: Spectral floor (prevents negative values)
 
 Returns:
 Enhanced audio
 """
 if self.noise_profile is None:
 raise ValueError("Must estimate noise first")
 
 # STFT of noisy signal
 noisy_stft = librosa.stft(
 noisy_audio,
 n_fft=self.n_fft,
 hop_length=self.hop_length
 )
 
 # Magnitude and phase
 mag = np.abs(noisy_stft)
 phase = np.angle(noisy_stft)
 
 # Spectral subtraction
 enhanced_mag = mag - alpha * self.noise_profile
 
 # Half-wave rectification with spectral floor
 enhanced_mag = np.maximum(enhanced_mag, beta * mag)
 
 # Reconstruct with original phase
 enhanced_stft = enhanced_mag * np.exp(1j * phase)
 
 # Inverse STFT
 enhanced_audio = librosa.istft(
 enhanced_stft,
 hop_length=self.hop_length
 )
 
 return enhanced_audio

# Usage
sr = 16000

# Load noisy speech
noisy_speech, _ = librosa.load('noisy_speech.wav', sr=sr)

# Estimate noise from first 0.5 seconds (assumed to be silence)
noise_segment = noisy_speech[:int(0.5 * sr)]

enhancer = SpectralSubtraction()
enhancer.estimate_noise(noise_segment)

# Enhance full audio
enhanced = enhancer.enhance(noisy_speech, alpha=2.0)

# Save result
import soundfile as sf
sf.write('enhanced_speech.wav', enhanced, sr)
``

### 2. Wiener Filtering

**Optimal filter in MMSE sense**

``python
class WienerFilter:
 """
 Wiener filtering for speech enhancement
 
 Minimizes mean squared error between clean and enhanced speech
 """
 
 def __init__(self, n_fft=512, hop_length=128):
 self.n_fft = n_fft
 self.hop_length = hop_length
 self.noise_psd = None
 
 def estimate_noise_psd(self, noise_audio):
 """Estimate noise power spectral density"""
 noise_stft = librosa.stft(
 noise_audio,
 n_fft=self.n_fft,
 hop_length=self.hop_length
 )
 
 # Power spectral density
 self.noise_psd = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
 
 def enhance(self, noisy_audio, a_priori_snr=None):
 """
 Apply Wiener filtering
 
 Wiener gain: H = S / (S + N)
 where S = signal PSD, N = noise PSD
 """
 if self.noise_psd is None:
 raise ValueError("Must estimate noise PSD first")
 
 # STFT
 noisy_stft = librosa.stft(
 noisy_audio,
 n_fft=self.n_fft,
 hop_length=self.hop_length
 )
 
 # Noisy PSD
 noisy_psd = np.abs(noisy_stft) ** 2
 
 # Estimate clean speech PSD
 speech_psd = np.maximum(noisy_psd - self.noise_psd, 0)
 
 # Wiener gain
 wiener_gain = speech_psd / (speech_psd + self.noise_psd + 1e-10)
 
 # Apply gain
 enhanced_stft = wiener_gain * noisy_stft
 
 # Inverse STFT
 enhanced_audio = librosa.istft(
 enhanced_stft,
 hop_length=self.hop_length
 )
 
 return enhanced_audio

# Usage
wiener = WienerFilter()
wiener.estimate_noise_psd(noise_segment)
enhanced = wiener.enhance(noisy_speech)
``

---

## Deep Learning Approaches

### 1. Mask-Based Enhancement

**Learn ideal ratio mask (IRM) or ideal binary mask (IBM)**

``python
import torch
import torch.nn as nn

class MaskEstimationNet(nn.Module):
 """
 Neural network for mask estimation
 
 Predicts time-frequency mask to apply to noisy spectrogram
 """
 
 def __init__(self, n_fft=512, hidden_dim=128):
 super().__init__()
 
 self.n_freq = n_fft // 2 + 1
 
 # Bidirectional LSTM
 self.lstm = nn.LSTM(
 input_size=self.n_freq,
 hidden_size=hidden_dim,
 num_layers=2,
 batch_first=True,
 bidirectional=True
 )
 
 # Mask prediction
 self.mask_fc = nn.Sequential(
 nn.Linear(hidden_dim * 2, hidden_dim),
 nn.ReLU(),
 nn.Dropout(0.2),
 nn.Linear(hidden_dim, self.n_freq),
 nn.Sigmoid() # Mask values in [0, 1]
 )
 
 def forward(self, noisy_mag):
 """
 Args:
 noisy_mag: Noisy magnitude spectrogram [batch, time, freq]
 
 Returns:
 mask: Predicted mask [batch, time, freq]
 """
 # LSTM
 lstm_out, _ = self.lstm(noisy_mag)
 
 # Predict mask
 mask = self.mask_fc(lstm_out)
 
 return mask

class MaskBasedEnhancer:
 """
 Speech enhancement using learned mask
 """
 
 def __init__(self, model, n_fft=512, hop_length=128):
 self.model = model
 self.model.eval()
 
 self.n_fft = n_fft
 self.hop_length = hop_length
 
 def enhance(self, noisy_audio):
 """
 Enhance audio using learned mask
 
 Steps:
 1. Compute noisy spectrogram
 2. Predict mask with neural network
 3. Apply mask
 4. Reconstruct audio
 """
 # STFT
 noisy_stft = librosa.stft(
 noisy_audio,
 n_fft=self.n_fft,
 hop_length=self.hop_length
 )
 
 # Magnitude and phase
 noisy_mag = np.abs(noisy_stft)
 phase = np.angle(noisy_stft)
 
 # Normalize magnitude
 mag_mean = np.mean(noisy_mag)
 mag_std = np.std(noisy_mag)
 noisy_mag_norm = (noisy_mag - mag_mean) / (mag_std + 1e-8)
 
 # Predict mask
 with torch.no_grad():
 # Transpose to [1, time, freq]
 mag_tensor = torch.FloatTensor(noisy_mag_norm.T).unsqueeze(0)
 
 mask = self.model(mag_tensor)
 
 # Back to numpy
 mask = mask.squeeze(0).numpy().T
 
 # Apply mask
 enhanced_mag = noisy_mag * mask
 
 # Reconstruct
 enhanced_stft = enhanced_mag * np.exp(1j * phase)
 enhanced_audio = librosa.istft(
 enhanced_stft,
 hop_length=self.hop_length
 )
 
 return enhanced_audio

# Usage
model = MaskEstimationNet(n_fft=512)
enhancer = MaskBasedEnhancer(model)

# Enhance
enhanced = enhancer.enhance(noisy_speech)
``

### 2. End-to-End Waveform Enhancement

**Direct waveform→waveform mapping**

``python
class ConvTasNet(nn.Module):
 """
 Conv-TasNet for speech enhancement
 
 End-to-end time-domain speech separation
 Based on: "Conv-TasNet: Surpassing Ideal Time-Frequency Masking"
 """
 
 def __init__(self, N=256, L=20, B=256, H=512, P=3, X=8, R=3):
 """
 Args:
 N: Number of filters in autoencoder
 L: Length of filters (ms)
 B: Number of channels in bottleneck
 H: Number of channels in conv blocks
 P: Kernel size in conv blocks
 X: Number of conv blocks in each repeat
 R: Number of repeats
 """
 super().__init__()
 
 # Encoder (waveform → features)
 self.encoder = nn.Conv1d(1, N, L, stride=L//2, padding=L//2)
 
 # Separator (TCN blocks)
 self.separator = self._build_separator(N, B, H, P, X, R)
 
 # Decoder (features → waveform)
 self.decoder = nn.ConvTranspose1d(N, 1, L, stride=L//2, padding=L//2)
 
 def _build_separator(self, N, B, H, P, X, R):
 """Build temporal convolutional network"""
 layers = []
 
 # Layer normalization
 layers.append(nn.LayerNorm(N))
 
 # Bottleneck
 layers.append(nn.Conv1d(N, B, 1))
 
 # TCN blocks
 for r in range(R):
 for x in range(X):
 dilation = 2 ** x
 layers.append(
 TemporalConvBlock(B, H, P, dilation)
 )
 
 # Output projection
 layers.append(nn.PReLU())
 layers.append(nn.Conv1d(B, N, 1))
 
 return nn.Sequential(*layers)
 
 def forward(self, mixture):
 """
 Args:
 mixture: Noisy waveform [batch, 1, samples]
 
 Returns:
 estimated_clean: Enhanced waveform [batch, 1, samples]
 """
 # Encode
 encoded = self.encoder(mixture) # [batch, N, T]
 
 # Separate
 mask = self.separator(encoded) # [batch, N, T]
 
 # Apply mask
 separated = encoded * mask
 
 # Decode
 estimated = self.decoder(separated) # [batch, 1, samples]
 
 return estimated

class TemporalConvBlock(nn.Module):
 """
 Temporal convolutional block with dilated convolutions
 """
 
 def __init__(self, in_channels, hidden_channels, kernel_size, dilation):
 super().__init__()
 
 self.conv1 = nn.Conv1d(
 in_channels, hidden_channels,
 kernel_size, dilation=dilation,
 padding=dilation * (kernel_size - 1) // 2
 )
 self.prelu1 = nn.PReLU()
 self.norm1 = nn.GroupNorm(1, hidden_channels)
 
 self.conv2 = nn.Conv1d(
 hidden_channels, in_channels,
 1
 )
 self.prelu2 = nn.PReLU()
 self.norm2 = nn.GroupNorm(1, in_channels)
 
 def forward(self, x):
 """
 Args:
 x: [batch, channels, time]
 """
 residual = x
 
 out = self.conv1(x)
 out = self.prelu1(out)
 out = self.norm1(out)
 
 out = self.conv2(out)
 out = self.prelu2(out)
 out = self.norm2(out)
 
 return out + residual

# Usage
model = ConvTasNet()
noisy_tensor = torch.randn(1, 1, 16000) # 1 second
enhanced_tensor = model(noisy_tensor)
``

---

## Real-Time Enhancement

### Streaming Enhancement System

``python
import numpy as np
from collections import deque

class StreamingEnhancer:
 """
 Real-time streaming speech enhancement
 
 Requirements:
 - Low latency (< 50ms)
 - Causal processing
 - Minimal buffering
 """
 
 def __init__(self, model, chunk_size=512, overlap=256, sr=16000):
 """
 Args:
 chunk_size: Samples per chunk
 overlap: Overlap between chunks (for smooth transitions)
 """
 self.model = model
 self.chunk_size = chunk_size
 self.overlap = overlap
 self.sr = sr
 
 # Circular buffer for overlap-add
 self.buffer = deque(maxlen=overlap)
 self.output_buffer = deque(maxlen=overlap)
 
 self.processed_chunks = 0
 
 def process_chunk(self, audio_chunk):
 """
 Process single audio chunk
 
 Args:
 audio_chunk: Audio samples [chunk_size]
 
 Returns:
 Enhanced audio chunk
 """
 # Add previous overlap
 if len(self.buffer) > 0:
 input_chunk = np.concatenate([
 np.array(self.buffer),
 audio_chunk
 ])
 else:
 input_chunk = audio_chunk
 
 # Enhance
 enhanced = self._enhance_chunk(input_chunk)
 
 # Overlap-add with linear cross-fade
 if len(self.output_buffer) > 0:
 # Smooth transition
 overlap_region = min(len(self.output_buffer), self.overlap)
 for i in range(overlap_region):
 weight = i / overlap_region
 enhanced[i] = (1 - weight) * self.output_buffer[i] + weight * enhanced[i]
 
 # Save overlap for next chunk
 self.buffer.clear()
 self.buffer.extend(audio_chunk[-self.overlap:])
 
 self.output_buffer.clear()
 self.output_buffer.extend(enhanced[-self.overlap:])
 
 self.processed_chunks += 1
 
 # Return non-overlap part
 return enhanced[:-self.overlap] if len(enhanced) > self.overlap else enhanced
 
 def _enhance_chunk(self, audio_chunk):
 """Enhance using model"""
 # Convert to tensor
 audio_tensor = torch.FloatTensor(audio_chunk).unsqueeze(0).unsqueeze(0)
 
 # Enhance
 with torch.no_grad():
 enhanced_tensor = self.model(audio_tensor)
 
 # Back to numpy
 enhanced = enhanced_tensor.squeeze().numpy()
 
 return enhanced
 
 def get_latency_ms(self):
 """Calculate processing latency"""
 return (self.chunk_size / self.sr) * 1000

# Usage for real-time processing
model = ConvTasNet()
enhancer = StreamingEnhancer(model, chunk_size=512, overlap=256, sr=16000)

print(f"Latency: {enhancer.get_latency_ms():.2f} ms")

# Process audio stream
import sounddevice as sd

def audio_callback(indata, outdata, frames, time, status):
 """Real-time audio callback"""
 # Get input chunk
 input_chunk = indata[:, 0]
 
 # Enhance
 enhanced_chunk = enhancer.process_chunk(input_chunk)
 
 # Output
 outdata[:len(enhanced_chunk), 0] = enhanced_chunk
 
 if status:
 print(f"Status: {status}")

# Start real-time processing
with sd.Stream(
 samplerate=16000,
 channels=1,
 callback=audio_callback,
 blocksize=512
):
 print("Processing audio in real-time... Press Ctrl+C to stop")
 sd.sleep(10000)
``

---

## Multi-Channel Enhancement

### Beamforming

``python
class BeamformerEnhancer:
 """
 Beamforming for multi-microphone enhancement
 
 Uses spatial information to enhance target speech
 """
 
 def __init__(self, n_mics=4, sr=16000):
 self.n_mics = n_mics
 self.sr = sr
 
 def delay_and_sum(self, multi_channel_audio, target_direction=0):
 """
 Delay-and-sum beamforming
 
 Args:
 multi_channel_audio: [n_mics, n_samples]
 target_direction: Target angle in degrees (0 = front)
 
 Returns:
 Enhanced single-channel audio
 """
 n_samples = multi_channel_audio.shape[1]
 
 # Calculate delays for each microphone
 # (Simplified: assumes linear array)
 mic_spacing = 0.05 # 5cm between mics
 speed_of_sound = 343 # m/s
 
 delays = []
 for i in range(self.n_mics):
 distance_diff = i * mic_spacing * np.sin(np.deg2rad(target_direction))
 delay_samples = int(distance_diff / speed_of_sound * self.sr)
 delays.append(delay_samples)
 
 # Align and sum
 aligned_signals = []
 for i, delay in enumerate(delays):
 sig = multi_channel_audio[i]
 if delay > 0:
 # Delay by pre-pending zeros
 padded = np.concatenate([np.zeros(delay, dtype=sig.dtype), sig])
 aligned = padded[:n_samples]
 elif delay < 0:
 # Advance by removing first samples
 aligned = sig[-delay:]
 if aligned.shape[0] < n_samples:
 aligned = np.pad(aligned, (0, n_samples - aligned.shape[0]), mode='constant')
 else:
 aligned = sig
 aligned_signals.append(aligned)
 
 # Sum aligned signals
 enhanced = np.mean(aligned_signals, axis=0)
 
 return enhanced
 
 def mvdr_beamformer(self, multi_channel_audio, noise_segment):
 """
 MVDR (Minimum Variance Distortionless Response) beamformer
 
 Optimal beamformer for known noise covariance
 """
 # Compute noise covariance matrix
 noise_cov = self._compute_covariance(noise_segment)
 
 # Compute signal+noise covariance
 signal_noise_cov = self._compute_covariance(multi_channel_audio)
 
 # MVDR weights
 # w = R_n^{-1} * a / (a^H * R_n^{-1} * a)
 # where a is steering vector
 
 # Simplified: assume steering vector points to channel 0
 steering_vector = np.zeros((self.n_mics, 1))
 steering_vector[0] = 1
 
 # Compute weights
 inv_noise_cov = np.linalg.pinv(noise_cov)
 numerator = inv_noise_cov @ steering_vector
 denominator = steering_vector.T @ inv_noise_cov @ steering_vector
 
 weights = numerator / (denominator + 1e-10)
 
 # Apply weights
 enhanced = weights.T @ multi_channel_audio
 
 return enhanced.squeeze()
 
 def _compute_covariance(self, signal):
 """Compute covariance matrix"""
 # [n_mics, n_samples] → [n_mics, n_mics]
 cov = signal @ signal.T / signal.shape[1]
 return cov

# Usage
beamformer = BeamformerEnhancer(n_mics=4, sr=16000)

# Multi-channel recording
multi_ch_audio = np.random.randn(4, 16000) # 4 mics, 1 second

# Enhance using delay-and-sum
enhanced_ds = beamformer.delay_and_sum(multi_ch_audio, target_direction=0)

# Or using MVDR
noise_segment = multi_ch_audio[:, :8000] # First 0.5 seconds
enhanced_mvdr = beamformer.mvdr_beamformer(multi_ch_audio, noise_segment)
``

---

## Connection to Caching (ML)

Voice enhancement benefits from caching strategies:

``python
class EnhancementCache:
 """
 Cache enhanced audio segments
 
 Connection toML:
 - Cache expensive enhancement operations
 - LRU for frequently accessed segments
 - TTL for time-sensitive applications
 """
 
 def __init__(self, capacity=1000):
 from collections import OrderedDict
 self.cache = OrderedDict()
 self.capacity = capacity
 
 self.hits = 0
 self.misses = 0
 
 def get_enhanced(self, audio_segment, model):
 """
 Get enhanced audio with caching
 
 Args:
 audio_segment: Raw audio
 model: Enhancement model
 
 Returns:
 Enhanced audio
 """
 # Create cache key (hash of audio)
 cache_key = hash(audio_segment.tobytes())
 
 # Check cache
 if cache_key in self.cache:
 self.hits += 1
 self.cache.move_to_end(cache_key) # Mark as recently used
 return self.cache[cache_key]
 
 # Compute enhancement
 self.misses += 1
 enhanced = model.enhance(audio_segment)
 
 # Cache result
 self.cache[cache_key] = enhanced
 
 # Evict if over capacity
 if len(self.cache) > self.capacity:
 self.cache.popitem(last=False)
 
 return enhanced
 
 def get_hit_rate(self):
 """Calculate cache hit rate"""
 total = self.hits + self.misses
 return self.hits / total if total > 0 else 0

# Usage
cache = EnhancementCache(capacity=1000)
model = ConvTasNet()

# Process with caching
for audio_segment in audio_stream:
 enhanced = cache.get_enhanced(audio_segment, model)
 
print(f"Cache hit rate: {cache.get_hit_rate():.2%}")
``

---

## Understanding Audio Enhancement Fundamentals

### Why Enhancement is Critical

Voice enhancement is the foundation of any production speech system. Poor audio quality cascades through the entire pipeline:

``python
class AudioQualityImpactAnalyzer:
 """
 Analyze impact of audio quality on downstream tasks
 
 Demonstrates how SNR affects ASR accuracy, speaker recognition, etc.
 """
 
 def __init__(self, asr_model, speaker_model):
 self.asr_model = asr_model
 self.speaker_model = speaker_model
 
 def evaluate_quality_impact(self, clean_audio, noisy_audio, transcript):
 """
 Compare performance on clean vs noisy audio
 
 Returns:
 Dictionary with metrics for both conditions
 """
 # ASR on clean audio
 clean_prediction = self.asr_model.transcribe(clean_audio)
 clean_wer = self._calculate_wer(transcript, clean_prediction)
 
 # ASR on noisy audio
 noisy_prediction = self.asr_model.transcribe(noisy_audio)
 noisy_wer = self._calculate_wer(transcript, noisy_prediction)
 
 # Speaker embedding quality
 clean_embedding = self.speaker_model.extract_embedding(clean_audio)
 noisy_embedding = self.speaker_model.extract_embedding(noisy_audio)
 
 # Embedding similarity (should be close for same speaker)
 similarity = np.dot(clean_embedding, noisy_embedding) / (
 np.linalg.norm(clean_embedding) * np.linalg.norm(noisy_embedding)
 )
 
 # Calculate SNR
 snr_db = self._calculate_snr(clean_audio, noisy_audio)
 
 return {
 'snr_db': snr_db,
 'clean_wer': clean_wer,
 'noisy_wer': noisy_wer,
 'wer_degradation': noisy_wer - clean_wer,
 'embedding_similarity': similarity,
 'relative_performance': clean_wer / noisy_wer if noisy_wer > 0 else 1.0
 }
 
 def _calculate_wer(self, reference, hypothesis):
 """Calculate Word Error Rate"""
 import editdistance
 
 ref_words = reference.lower().split()
 hyp_words = hypothesis.lower().split()
 
 distance = editdistance.eval(ref_words, hyp_words)
 wer = distance / len(ref_words) if len(ref_words) > 0 else 0
 
 return wer
 
 def _calculate_snr(self, clean, noisy):
 """Calculate Signal-to-Noise Ratio"""
 noise = noisy - clean
 
 signal_power = np.mean(clean ** 2)
 noise_power = np.mean(noise ** 2)
 
 if noise_power == 0:
 return float('inf')
 
 snr = 10 * np.log10(signal_power / noise_power)
 return snr

# Demo impact analysis
print("="*60)
print("AUDIO QUALITY IMPACT ANALYSIS")
print("="*60)

# Simulate different SNR levels
snr_levels = [-5, 0, 5, 10, 15, 20]

for snr_target in snr_levels:
 # Add noise at specific SNR
 noisy = add_noise_at_snr(clean_audio, noise, snr_target)
 
 # Evaluate
 results = analyzer.evaluate_quality_impact(clean_audio, noisy, transcript)
 
 print(f"\nSNR: {snr_target} dB")
 print(f" WER (clean): {results['clean_wer']:.2%}")
 print(f" WER (noisy): {results['noisy_wer']:.2%}")
 print(f" Degradation: {results['wer_degradation']:.2%}")
 print(f" Speaker Sim: {results['embedding_similarity']:.3f}")
``

### Frequency Domain Analysis

Understanding audio in frequency domain is crucial for enhancement:

``python
class FrequencyDomainAnalyzer:
 """
 Analyze and visualize audio in frequency domain
 
 Essential for understanding what noise reduction does
 """
 
 def __init__(self, sr=16000):
 self.sr = sr
 
 def analyze_spectrum(self, audio):
 """
 Compute and visualize spectrum
 
 Returns:
 frequencies, magnitudes, phases
 """
 # Compute FFT
 n_fft = 2048
 fft = np.fft.rfft(audio, n=n_fft)
 
 # Magnitude and phase
 magnitude = np.abs(fft)
 phase = np.angle(fft)
 
 # Frequency bins
 frequencies = np.fft.rfftfreq(n_fft, 1/self.sr)
 
 return frequencies, magnitude, phase
 
 def compare_spectra(self, clean, noisy, enhanced):
 """
 Compare spectra before and after enhancement
 """
 import matplotlib.pyplot as plt
 
 # Compute spectra
 freq_clean, mag_clean, _ = self.analyze_spectrum(clean)
 freq_noisy, mag_noisy, _ = self.analyze_spectrum(noisy)
 freq_enhanced, mag_enhanced, _ = self.analyze_spectrum(enhanced)
 
 # Plot
 fig, axes = plt.subplots(3, 1, figsize=(12, 10))
 
 # Clean
 axes[0].plot(freq_clean, 20 * np.log10(mag_clean + 1e-10))
 axes[0].set_title('Clean Audio Spectrum')
 axes[0].set_ylabel('Magnitude (dB)')
 axes[0].grid(True)
 
 # Noisy
 axes[1].plot(freq_noisy, 20 * np.log10(mag_noisy + 1e-10), color='red')
 axes[1].set_title('Noisy Audio Spectrum')
 axes[1].set_ylabel('Magnitude (dB)')
 axes[1].grid(True)
 
 # Enhanced
 axes[2].plot(freq_enhanced, 20 * np.log10(mag_enhanced + 1e-10), color='green')
 axes[2].set_title('Enhanced Audio Spectrum')
 axes[2].set_xlabel('Frequency (Hz)')
 axes[2].set_ylabel('Magnitude (dB)')
 axes[2].grid(True)
 
 plt.tight_layout()
 plt.savefig('spectrum_comparison.png')
 plt.close()
 
 def compute_spectral_features(self, audio):
 """
 Compute spectral features for quality assessment
 """
 freq, mag, _ = self.analyze_spectrum(audio)
 
 # Spectral centroid
 centroid = np.sum(freq * mag) / np.sum(mag)
 
 # Spectral bandwidth
 bandwidth = np.sqrt(np.sum(((freq - centroid) ** 2) * mag) / np.sum(mag))
 
 # Spectral flatness (Wiener entropy)
 geometric_mean = np.exp(np.mean(np.log(mag + 1e-10)))
 arithmetic_mean = np.mean(mag)
 flatness = geometric_mean / arithmetic_mean
 
 # Spectral rolloff (95% of energy)
 cumsum = np.cumsum(mag)
 rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0][0]
 rolloff = freq[rolloff_idx]
 
 return {
 'centroid_hz': centroid,
 'bandwidth_hz': bandwidth,
 'flatness': flatness,
 'rolloff_hz': rolloff
 }

# Usage
analyzer = FrequencyDomainAnalyzer(sr=16000)

# Analyze audio
features = analyzer.compute_spectral_features(audio)
print("Spectral Features:")
print(f" Centroid: {features['centroid_hz']:.1f} Hz")
print(f" Bandwidth: {features['bandwidth_hz']:.1f} Hz")
print(f" Flatness: {features['flatness']:.3f}")
print(f" Rolloff: {features['rolloff_hz']:.1f} Hz")

# Compare before/after
analyzer.compare_spectra(clean_audio, noisy_audio, enhanced_audio)
``

---

## Advanced Deep Learning Enhancement

### State-of-the-Art Architectures

``python
class ConvTasNetEnhancer(nn.Module):
 """
 Conv-TasNet for speech enhancement
 
 Architecture:
 1. Encoder: Waveform → Feature representation
 2. Separator: Mask estimation using temporal convolutions
 3. Decoder: Masked features → Enhanced waveform
 
 Advantages over STFT-based methods:
 - Operates on raw waveform
 - Learnable basis functions
 - Better phase reconstruction
 """
 
 def __init__(
 self,
 n_src=1,
 n_filters=512,
 kernel_size=16,
 stride=8,
 n_blocks=8,
 n_repeats=3,
 bn_chan=128,
 hid_chan=512,
 skip_chan=128
 ):
 super().__init__()
 
 # Encoder: 1D conv
 self.encoder = nn.Conv1d(
 1,
 n_filters,
 kernel_size=kernel_size,
 stride=stride,
 padding=kernel_size // 2
 )
 
 # Separator: TCN blocks
 self.separator = TemporalConvNet(
 n_filters,
 n_src,
 n_blocks=n_blocks,
 n_repeats=n_repeats,
 bn_chan=bn_chan,
 hid_chan=hid_chan,
 skip_chan=skip_chan
 )
 
 # Decoder: 1D transposed conv
 self.decoder = nn.ConvTranspose1d(
 n_filters,
 1,
 kernel_size=kernel_size,
 stride=stride,
 padding=kernel_size // 2
 )
 
 def forward(self, waveform):
 """
 Enhance waveform
 
 Args:
 waveform: [batch, time]
 
 Returns:
 enhanced: [batch, time]
 """
 # Add channel dimension
 x = waveform.unsqueeze(1) # [batch, 1, time]
 
 # Encode
 encoded = self.encoder(x) # [batch, n_filters, time']
 
 # Separate (estimate mask)
 masks = self.separator(encoded) # [batch, n_src, n_filters, time']
 
 # Apply mask
 masked = encoded.unsqueeze(1) * masks # [batch, n_src, n_filters, time']
 
 # Decode
 enhanced = self.decoder(masked.squeeze(1)) # [batch, 1, time]
 
 # Remove channel dimension
 enhanced = enhanced.squeeze(1) # [batch, time]
 
 # Trim to original length
 if enhanced.shape[-1] != waveform.shape[-1]:
 enhanced = enhanced[..., :waveform.shape[-1]]
 
 return enhanced

class TemporalConvNet(nn.Module):
 """
 Temporal Convolutional Network for Conv-TasNet
 
 Stack of dilated conv blocks with skip connections
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
 
 # Layer norm
 self.layer_norm = nn.GroupNorm(1, n_filters)
 
 # Bottleneck
 self.bottleneck = nn.Conv1d(n_filters, bn_chan, 1)
 
 # TCN blocks
 self.blocks = nn.ModuleList()
 for r in range(n_repeats):
 for b in range(n_blocks):
 dilation = 2 ** b
 self.blocks.append(
 TCNBlock(
 bn_chan,
 hid_chan,
 skip_chan,
 kernel_size=3,
 dilation=dilation
 )
 )
 
 # Output
 self.output = nn.Sequential(
 nn.PReLU(),
 nn.Conv1d(skip_chan, n_filters, 1),
 nn.Sigmoid() # Mask should be [0, 1]
 )
 
 def forward(self, x):
 """
 Args:
 x: [batch, n_filters, time]
 
 Returns:
 masks: [batch, n_src, n_filters, time]
 """
 # Normalize
 x = self.layer_norm(x)
 
 # Bottleneck
 x = self.bottleneck(x) # [batch, bn_chan, time]
 
 # Accumulate skip connections
 skip_sum = 0
 
 for block in self.blocks:
 x, skip = block(x)
 skip_sum = skip_sum + skip
 
 # Output mask
 masks = self.output(skip_sum)
 
 # Unsqueeze for n_src dimension
 masks = masks.unsqueeze(1) # [batch, 1, n_filters, time]
 
 return masks

class TCNBlock(nn.Module):
 """Single TCN block with dilated convolution"""
 
 def __init__(self, in_chan, hid_chan, skip_chan, kernel_size=3, dilation=1):
 super().__init__()
 
 self.conv1 = nn.Conv1d(
 in_chan,
 hid_chan,
 1
 )
 
 self.prelu1 = nn.PReLU()
 
 self.norm1 = nn.GroupNorm(1, hid_chan)
 
 self.depthwise_conv = nn.Conv1d(
 hid_chan,
 hid_chan,
 kernel_size,
 padding=(kernel_size - 1) * dilation // 2,
 dilation=dilation,
 groups=hid_chan
 )
 
 self.prelu2 = nn.PReLU()
 
 self.norm2 = nn.GroupNorm(1, hid_chan)
 
 self.conv2 = nn.Conv1d(hid_chan, in_chan, 1)
 
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
 x = self.conv1(x)
 x = self.prelu1(x)
 x = self.norm1(x)
 
 # Depthwise conv
 x = self.depthwise_conv(x)
 x = self.prelu2(x)
 x = self.norm2(x)
 
 # Skip connection
 skip = self.skip_conv(x)
 
 # Output
 x = self.conv2(x)
 
 # Residual
 output = x + residual
 
 return output, skip

# Training Conv-TasNet
class ConvTasNetTrainer:
 """
 Train Conv-TasNet for speech enhancement
 """
 
 def __init__(self, model, device='cuda'):
 self.model = model.to(device)
 self.device = device
 
 # Optimizer
 self.optimizer = torch.optim.Adam(
 self.model.parameters(),
 lr=1e-3
 )
 
 # Learning rate scheduler
 self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
 self.optimizer,
 mode='min',
 factor=0.5,
 patience=3
 )
 
 def train_epoch(self, train_loader):
 """Train one epoch"""
 self.model.train()
 
 total_loss = 0
 
 for batch_idx, (noisy, clean) in enumerate(train_loader):
 noisy = noisy.to(self.device)
 clean = clean.to(self.device)
 
 # Forward
 enhanced = self.model(noisy)
 
 # Loss: SI-SNR (Scale-Invariant SNR)
 loss = self._si_snr_loss(enhanced, clean)
 
 # Backward
 self.optimizer.zero_grad()
 loss.backward()
 
 # Gradient clipping
 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
 
 self.optimizer.step()
 
 total_loss += loss.item()
 
 if batch_idx % 100 == 0:
 print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
 
 return total_loss / len(train_loader)
 
 def _si_snr_loss(self, estimate, target):
 """
 Scale-Invariant Signal-to-Noise Ratio loss
 
 Better than MSE for speech enhancement
 """
 # Zero-mean
 estimate_zm = estimate - estimate.mean(dim=-1, keepdim=True)
 target_zm = target - target.mean(dim=-1, keepdim=True)
 
 # <s', s>s / ||s||^2
 dot = (estimate_zm * target_zm).sum(dim=-1, keepdim=True)
 target_energy = (target_zm ** 2).sum(dim=-1, keepdim=True)
 projection = dot * target_zm / (target_energy + 1e-8)
 
 # Noise
 noise = estimate_zm - projection
 
 # SI-SNR
 si_snr = 10 * torch.log10(
 (projection ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1) + 1e-8
 )
 
 # Negative for loss (we want to maximize SI-SNR)
 return -si_snr.mean()

# Usage
model = ConvTasNetEnhancer()
trainer = ConvTasNetTrainer(model, device='cuda')

# Train
for epoch in range(num_epochs):
 train_loss = trainer.train_epoch(train_loader)
 val_loss = trainer.validate(val_loader)
 
 print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
 
 trainer.scheduler.step(val_loss)
``

### Real-Time Enhancement with ONNX

``python
class RealTimeONNXEnhancer:
 """
 Real-time enhancement using ONNX Runtime
 
 Optimized for production deployment
 """
 
 def __init__(self, onnx_model_path, chunk_size=4800):
 """
 Args:
 onnx_model_path: Path to exported ONNX model
 chunk_size: Audio chunk size (samples)
 """
 import onnxruntime as ort
 
 self.chunk_size = chunk_size
 
 # Load ONNX model
 self.session = ort.InferenceSession(
 onnx_model_path,
 providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
 )
 
 # Get input/output names
 self.input_name = self.session.get_inputs()[0].name
 self.output_name = self.session.get_outputs()[0].name
 
 # State for streaming
 self.reset_state()
 
 def reset_state(self):
 """Reset streaming state"""
 self.overlap_buffer = np.zeros(self.chunk_size // 2, dtype=np.float32)
 
 def enhance_chunk(self, audio_chunk):
 """
 Enhance single audio chunk with overlap-add
 
 Args:
 audio_chunk: [chunk_size] numpy array
 
 Returns:
 enhanced_chunk: [chunk_size] numpy array
 """
 # Prepare input (batch dimension)
 input_data = audio_chunk.astype(np.float32)[np.newaxis, :]
 
 # Run inference
 enhanced = self.session.run(
 [self.output_name],
 {self.input_name: input_data}
 )[0][0]
 
 # Overlap-add
 overlap_size = len(self.overlap_buffer)
 enhanced[:overlap_size] += self.overlap_buffer
 
 # Save overlap for next chunk
 self.overlap_buffer = enhanced[-overlap_size:].copy()
 
 # Return without overlap region
 return enhanced[:-overlap_size]
 
 def enhance_stream(self, audio_stream):
 """
 Enhance audio stream in real-time
 
 Generator that yields enhanced chunks
 """
 for chunk in audio_stream:
 # Ensure correct size
 if len(chunk) != self.chunk_size:
 # Pad or skip
 continue
 
 # Enhance
 enhanced = self.enhance_chunk(chunk)
 
 yield enhanced

# Export PyTorch model to ONNX
def export_to_onnx(pytorch_model, onnx_path, chunk_size=4800):
 """
 Export trained PyTorch model to ONNX
 """
 pytorch_model.eval()
 
 # Dummy input
 dummy_input = torch.randn(1, chunk_size)
 
 # Export
 torch.onnx.export(
 pytorch_model,
 dummy_input,
 onnx_path,
 input_names=['audio_input'],
 output_names=['audio_output'],
 dynamic_axes={
 'audio_input': {1: 'time'},
 'audio_output': {1: 'time'}
 },
 opset_version=14
 )
 
 print(f"Model exported to {onnx_path}")

# Usage
# Export model
export_to_onnx(trained_model, 'convtasnet_enhancer.onnx')

# Create real-time enhancer
enhancer = RealTimeONNXEnhancer('convtasnet_enhancer.onnx', chunk_size=4800)

# Stream audio
def audio_stream_generator():
 """Generate audio chunks from microphone/file"""
 # Implementation depends on audio source
 pass

# Enhance stream
for enhanced_chunk in enhancer.enhance_stream(audio_stream_generator()):
 # Play or save enhanced audio
 pass
``

---

## Production Quality Assurance

### Automated Quality Metrics

``python
class EnhancementQualityAssurance:
 """
 Automated quality assurance for enhancement pipeline
 
 Monitors:
 - SNR improvement
 - Speech intelligibility
 - Artifacts
 - Latency
 """
 
 def __init__(self):
 self.metrics_history = []
 
 def assess_quality(self, original, enhanced, reference=None):
 """
 Comprehensive quality assessment
 
 Args:
 original: Noisy input
 enhanced: Enhanced output
 reference: Clean reference (if available)
 
 Returns:
 Quality metrics dictionary
 """
 metrics = {}
 
 # SNR improvement (requires reference)
 if reference is not None:
 original_snr = self._compute_snr(original, reference)
 enhanced_snr = self._compute_snr(enhanced, reference)
 metrics['snr_improvement_db'] = enhanced_snr - original_snr
 
 # PESQ (Perceptual Evaluation of Speech Quality)
 from pesq import pesq
 metrics['pesq_original'] = pesq(16000, reference, original, 'wb')
 metrics['pesq_enhanced'] = pesq(16000, reference, enhanced, 'wb')
 metrics['pesq_improvement'] = (
 metrics['pesq_enhanced'] - metrics['pesq_original']
 )
 
 # STOI (Short-Time Objective Intelligibility)
 from pystoi import stoi
 metrics['stoi_original'] = stoi(reference, original, 16000)
 metrics['stoi_enhanced'] = stoi(reference, enhanced, 16000)
 metrics['stoi_improvement'] = (
 metrics['stoi_enhanced'] - metrics['stoi_original']
 )
 
 # Artifact detection (no reference needed)
 metrics['artifact_score'] = self._detect_artifacts(enhanced)
 
 # Spectral distortion
 metrics['spectral_distortion'] = self._compute_spectral_distortion(
 original, enhanced
 )
 
 # Dynamic range
 metrics['dynamic_range_db'] = 20 * np.log10(
 np.max(np.abs(enhanced)) / (np.mean(np.abs(enhanced)) + 1e-8)
 )
 
 # Clipping detection
 metrics['clipping_ratio'] = np.mean(np.abs(enhanced) > 0.99)
 
 # Overall quality score
 metrics['quality_score'] = self._compute_overall_score(metrics)
 
 self.metrics_history.append(metrics)
 
 return metrics
 
 def _compute_snr(self, signal, reference):
 """Compute SNR"""
 noise = signal - reference
 signal_power = np.mean(reference ** 2)
 noise_power = np.mean(noise ** 2)
 
 if noise_power == 0:
 return float('inf')
 
 snr_db = 10 * np.log10(signal_power / noise_power)
 return snr_db
 
 def _detect_artifacts(self, audio):
 """
 Detect musical noise and other artifacts
 
 Returns:
 Artifact score (0-1, lower is better)
 """
 # Compute spectrogram
 S = librosa.stft(audio)
 magnitude = np.abs(S)
 
 # Temporal variation
 temporal_diff = np.diff(magnitude, axis=1)
 temporal_variance = np.var(temporal_diff)
 
 # Spectral variation
 spectral_diff = np.diff(magnitude, axis=0)
 spectral_variance = np.var(spectral_diff)
 
 # High variance indicates artifacts
 artifact_score = (temporal_variance + spectral_variance) / 2
 
 # Normalize to [0, 1]
 artifact_score = np.clip(artifact_score / 100, 0, 1)
 
 return artifact_score
 
 def _compute_spectral_distortion(self, original, enhanced):
 """
 Compute spectral distortion
 
 Measures how much the spectrum changed
 """
 # Compute spectrograms
 S_orig = np.abs(librosa.stft(original))
 S_enh = np.abs(librosa.stft(enhanced))
 
 # Log magnitude
 S_orig_db = librosa.amplitude_to_db(S_orig + 1e-10)
 S_enh_db = librosa.amplitude_to_db(S_enh + 1e-10)
 
 # MSE in log domain
 distortion = np.mean((S_orig_db - S_enh_db) ** 2)
 
 return distortion
 
 def _compute_overall_score(self, metrics):
 """
 Compute overall quality score
 
 Weighted combination of metrics
 """
 score = 0.0
 
 # PESQ improvement (if available)
 if 'pesq_improvement' in metrics:
 score += 0.4 * np.clip(metrics['pesq_improvement'] / 2, 0, 1)
 
 # STOI improvement (if available)
 if 'stoi_improvement' in metrics:
 score += 0.4 * np.clip(metrics['stoi_improvement'], 0, 1)
 
 # Artifact penalty
 score -= 0.2 * metrics['artifact_score']
 
 # Normalize to [0, 1]
 score = np.clip(score, 0, 1)
 
 return score
 
 def generate_report(self):
 """Generate quality assurance report"""
 if not self.metrics_history:
 print("No metrics recorded")
 return
 
 # Aggregate metrics
 avg_metrics = {}
 for key in self.metrics_history[0].keys():
 values = [m[key] for m in self.metrics_history if key in m]
 avg_metrics[key] = np.mean(values)
 
 print("\n" + "="*60)
 print("ENHANCEMENT QUALITY ASSURANCE REPORT")
 print("="*60)
 print(f"Samples Evaluated: {len(self.metrics_history)}")
 print(f"\nAverage Metrics:")
 
 for key, value in avg_metrics.items():
 print(f" {key:30s}: {value:.4f}")
 
 # Pass/fail criteria
 print(f"\n{'Criterion':<30s} {'Status':>10s}")
 print("-" * 42)
 
 checks = [
 ('SNR Improvement', avg_metrics.get('snr_improvement_db', 0) > 3, '>3 dB'),
 ('PESQ Improvement', avg_metrics.get('pesq_improvement', 0) > 0.5, '>0.5'),
 ('STOI Improvement', avg_metrics.get('stoi_improvement', 0) > 0.1, '>0.1'),
 ('Artifact Score', avg_metrics.get('artifact_score', 1) < 0.3, '<0.3'),
 ('Clipping Ratio', avg_metrics.get('clipping_ratio', 1) < 0.01, '<1%'),
 ]
 
 all_passed = True
 for name, passed, threshold in checks:
 status = "✓ PASS" if passed else "✗ FAIL"
 all_passed = all_passed and passed
 print(f" {name:<30s} {status:>10s} ({threshold})")
 
 print("-" * 42)
 print(f" {'Overall Result':<30s} {'✓ PASS' if all_passed else '✗ FAIL':>10s}")
 print("="*60)

# Usage
qa = EnhancementQualityAssurance()

# Evaluate multiple files
for noisy_file, clean_file in test_pairs:
 noisy_audio, _ = librosa.load(noisy_file, sr=16000)
 clean_audio, _ = librosa.load(clean_file, sr=16000)
 
 # Enhance
 enhanced_audio = enhancer.enhance(noisy_audio)
 
 # Assess quality
 metrics = qa.assess_quality(noisy_audio, enhanced_audio, clean_audio)

# Generate report
qa.generate_report()
``

---

## Key Takeaways

✅ **Multiple approaches** - Classical (spectral subtraction, Wiener) and deep learning 
✅ **Quality metrics** - PESQ, STOI, SNR for evaluation 
✅ **Real-time processing** - Streaming with low latency < 50ms 
✅ **Multi-channel** - Beamforming for spatial enhancement 
✅ **Caching benefits** - Reduce computational cost for repeated segments 
✅ **Trade-offs** - Quality vs latency vs computational cost 
✅ **Production considerations** - Monitoring, fallback, quality control 

---

**Originally published at:** [arunbaby.com/speech-tech/0010-voice-enhancement](https://www.arunbaby.com/speech-tech/0010-voice-enhancement/)

*If you found this helpful, consider sharing it with others who might benefit.*


