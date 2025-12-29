---
title: "Real-time Keyword Spotting"
day: 9
related_dsa_day: 9
related_ml_day: 9
related_agents_day: 9
collection: speech_tech
categories:
 - speech-tech
tags:
 - keyword-spotting
 - wake-word-detection
 - real-time
 - edge-ai
 - always-on
subdomain: Speech Recognition
tech_stack: [Python, PyTorch, TensorFlow Lite, ONNX, librosa, sounddevice]
scale: "Always-on, ultra-low latency"
companies: [Google, Amazon, Apple, Microsoft, Nuance, Picovoice]
---

**Build lightweight models that detect specific keywords in audio streams with minimal latency and power consumption for voice interfaces.**

## Introduction

**Keyword spotting (KWS)** detects specific words or phrases in continuous audio streams, enabling voice-activated interfaces.

**Common applications:**
- Wake word detection ("Hey Siri", "Alexa", "OK Google")
- Voice commands ("Play", "Stop", "Next")
- Accessibility features (voice navigation)
- Security (speaker verification)

**Key requirements:**
- **Ultra-low latency:** < 50ms detection time
- **Low power:** Run continuously on battery
- **Small model:** Fit on edge devices (< 1MB)
- **High accuracy:** < 1% false acceptance rate
- **Noise robust:** Work in real-world conditions

---

## Problem Formulation

### Task Definition

Given audio input, classify whether a target keyword is present:

``
Input: Audio waveform (e.g., 1 second, 16kHz = 16,000 samples)
Output: {keyword, no_keyword}

Example:
 Audio: "Hey Siri, what's the weather?"
 Output: keyword="hey_siri", timestamp=0.0s
``

### Challenges

1. **Always-on constraint:** Must run 24/7 without draining battery
2. **False positives:** Accidental activations frustrate users
3. **False negatives:** Missed detections break user experience
4. **Noise robustness:** Background noise, music, TV
5. **Speaker variability:** Different accents, ages, genders

---

## System Architecture

``
┌─────────────────────────────────────────────────────────┐
│ Microphone Input │
└────────────────────┬────────────────────────────────────┘
 │ Continuous audio stream
 ▼
 ┌────────────────────────┐
 │ Audio Preprocessing │
 │ - Noise reduction │
 │ - Normalization │
 └───────────┬────────────┘
 │
 ▼
 ┌────────────────────────┐
 │ Feature Extraction │
 │ - MFCC / Mel-spec │
 │ - Sliding window │
 └───────────┬────────────┘
 │
 ▼
 ┌────────────────────────┐
 │ KWS Model (Tiny NN) │
 │ - CNN / RNN / TCN │
 │ - < 1MB, < 10ms │
 └───────────┬────────────┘
 │
 ▼
 ┌────────────────────────┐
 │ Post-processing │
 │ - Threshold / Smooth │
 │ - Reject false pos │
 └───────────┬────────────┘
 │
 ▼
 ┌────────────────┐
 │ Trigger Event │
 │ (Wake system) │
 └────────────────┘
``

---

## Model Architectures

### Approach 1: CNN-based KWS

**Small convolutional network on spectrograms**

``python
import torch
import torch.nn as nn

class KeywordSpottingCNN(nn.Module):
 """
 Lightweight CNN for keyword spotting
 
 Input: Mel-spectrogram (n_mels, time_steps)
 Output: Keyword confidence score
 
 Model size: ~100KB
 Inference time: ~5ms on CPU
 """
 
 def __init__(self, n_mels=40, n_classes=2):
 super().__init__()
 
 # Convolutional layers
 self.conv1 = nn.Sequential(
 nn.Conv2d(1, 64, kernel_size=3, padding=1),
 nn.BatchNorm2d(64),
 nn.ReLU(),
 nn.MaxPool2d(2)
 )
 
 self.conv2 = nn.Sequential(
 nn.Conv2d(64, 64, kernel_size=3, padding=1),
 nn.BatchNorm2d(64),
 nn.ReLU(),
 nn.MaxPool2d(2)
 )
 
 # Global average pooling
 self.gap = nn.AdaptiveAvgPool2d((1, 1))
 
 # Classifier
 self.fc = nn.Linear(64, n_classes)
 
 def forward(self, x):
 """
 Args:
 x: [batch, 1, n_mels, time_steps]
 
 Returns:
 [batch, n_classes]
 """
 x = self.conv1(x)
 x = self.conv2(x)
 x = self.gap(x)
 x = x.view(x.size(0), -1)
 x = self.fc(x)
 return x

# Create model
model = KeywordSpottingCNN(n_mels=40, n_classes=2)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}") # ~30K parameters

# Estimate model size
model_size_mb = n_params * 4 / (1024 ** 2) # 4 bytes per float32
print(f"Model size: {model_size_mb:.2f} MB")
``

### Approach 2: RNN-based KWS

**Temporal modeling with GRU**

``python
class KeywordSpottingGRU(nn.Module):
 """
 GRU-based keyword spotting
 
 Better for temporal patterns, slightly larger
 """
 
 def __init__(self, n_mels=40, hidden_size=64, n_layers=2, n_classes=2):
 super().__init__()
 
 self.gru = nn.GRU(
 input_size=n_mels,
 hidden_size=hidden_size,
 num_layers=n_layers,
 batch_first=True,
 bidirectional=False # Unidirectional for streaming
 )
 
 self.fc = nn.Linear(hidden_size, n_classes)
 
 def forward(self, x):
 """
 Args:
 x: [batch, time_steps, n_mels]
 
 Returns:
 [batch, n_classes]
 """
 # GRU forward pass
 out, hidden = self.gru(x)
 
 # Use last hidden state
 x = out[:, -1, :]
 
 # Classifier
 x = self.fc(x)
 return x

model_gru = KeywordSpottingGRU(n_mels=40, hidden_size=64)
``

### Approach 3: Temporal Convolutional Network

**Efficient temporal modeling**

``python
class TemporalBlock(nn.Module):
 """Single temporal convolutional block"""
 
 def __init__(self, n_inputs, n_outputs, kernel_size, dilation):
 super().__init__()
 
 self.conv1 = nn.Conv1d(
 n_inputs, n_outputs, kernel_size,
 padding=(kernel_size-1) * dilation // 2,
 dilation=dilation
 )
 self.relu = nn.ReLU()
 self.dropout = nn.Dropout(0.2)
 
 # Residual connection
 self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) \
 if n_inputs != n_outputs else None
 
 def forward(self, x):
 out = self.conv1(x)
 out = self.relu(out)
 out = self.dropout(out)
 
 res = x if self.downsample is None else self.downsample(x)
 return out + res

class KeywordSpottingTCN(nn.Module):
 """
 Temporal Convolutional Network for KWS
 
 Combines efficiency of CNNs with temporal modeling
 """
 
 def __init__(self, n_mels=40, n_classes=2):
 super().__init__()
 
 self.blocks = nn.Sequential(
 TemporalBlock(n_mels, 64, kernel_size=3, dilation=1),
 TemporalBlock(64, 64, kernel_size=3, dilation=2),
 TemporalBlock(64, 64, kernel_size=3, dilation=4),
 )
 
 self.gap = nn.AdaptiveAvgPool1d(1)
 self.fc = nn.Linear(64, n_classes)
 
 def forward(self, x):
 """
 Args:
 x: [batch, time_steps, n_mels]
 
 Returns:
 [batch, n_classes]
 """
 # Transpose for conv1d: [batch, n_mels, time_steps]
 x = x.transpose(1, 2)
 
 # Temporal blocks
 x = self.blocks(x)
 
 # Global average pooling
 x = self.gap(x).squeeze(-1)
 
 # Classifier
 x = self.fc(x)
 return x

model_tcn = KeywordSpottingTCN(n_mels=40)
``

---

## Feature Extraction Pipeline

``python
import librosa
import numpy as np

class KeywordSpottingFeatureExtractor:
 """
 Extract features for keyword spotting
 
 Optimized for real-time processing
 """
 
 def __init__(self, sample_rate=16000, window_size_ms=30, 
 hop_size_ms=10, n_mels=40):
 self.sample_rate = sample_rate
 self.n_fft = int(sample_rate * window_size_ms / 1000)
 self.hop_length = int(sample_rate * hop_size_ms / 1000)
 self.n_mels = n_mels
 
 # Precompute mel filterbank
 self.mel_basis = librosa.filters.mel(
 sr=sample_rate,
 n_fft=self.n_fft,
 n_mels=n_mels,
 fmin=0,
 fmax=sample_rate / 2
 )
 
 def extract(self, audio):
 """
 Extract mel-spectrogram features
 
 Args:
 audio: Audio samples [n_samples]
 
 Returns:
 Mel-spectrogram [n_mels, time_steps]
 """
 # Compute STFT
 stft = librosa.stft(
 audio,
 n_fft=self.n_fft,
 hop_length=self.hop_length,
 window='hann'
 )
 
 # Power spectrogram
 power = np.abs(stft) ** 2
 
 # Apply mel filterbank on power
 mel_power = np.dot(self.mel_basis, power)
 
 # Log compression (power → dB)
 mel_db = librosa.power_to_db(mel_power, ref=np.max)
 
 return mel_db
 
 def extract_from_stream(self, audio_chunk):
 """
 Extract features from streaming audio
 
 Optimized for low latency
 """
 return self.extract(audio_chunk)

# Usage
extractor = KeywordSpottingFeatureExtractor(sample_rate=16000)

# Extract features from 1-second audio
audio = np.random.randn(16000)
features = extractor.extract(audio)
print(f"Feature shape: {features.shape}") # (40, 101)
``

---

## Training Pipeline

### Data Preparation

``python
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

class KeywordSpottingDataset(Dataset):
 """
 Dataset for keyword spotting training
 
 Handles positive (keyword) and negative (non-keyword) examples
 """
 
 def __init__(self, audio_files, labels, feature_extractor, 
 augment=True):
 self.audio_files = audio_files
 self.labels = labels
 self.feature_extractor = feature_extractor
 self.augment = augment
 
 def __len__(self):
 return len(self.audio_files)
 
 def __getitem__(self, idx):
 # Load audio
 audio, sr = librosa.load(
 self.audio_files[idx],
 sr=self.feature_extractor.sample_rate
 )
 
 # Pad or trim to 1 second
 target_length = self.feature_extractor.sample_rate
 if len(audio) < target_length:
 audio = np.pad(audio, (0, target_length - len(audio)))
 else:
 audio = audio[:target_length]
 
 # Data augmentation
 if self.augment:
 audio = self._augment(audio)
 
 # Extract features
 features = self.feature_extractor.extract(audio)
 
 # Convert to tensor
 features = torch.FloatTensor(features).unsqueeze(0) # Add channel dim
 label = torch.LongTensor([self.labels[idx]])
 
 return features, label
 
 def _augment(self, audio):
 """
 Data augmentation
 
 - Add noise
 - Time shift
 - Speed perturbation
 """
 # Add background noise
 noise_level = np.random.uniform(0, 0.005)
 noise = np.random.randn(len(audio)) * noise_level
 audio = audio + noise
 
 # Time shift
 shift = np.random.randint(-1600, 1600) # ±100ms at 16kHz
 audio = np.roll(audio, shift)
 
 # Speed perturbation (simplified)
 speed_factor = np.random.uniform(0.9, 1.1)
 # In practice, use librosa.effects.time_stretch
 
 return audio

# Create dataset
dataset = KeywordSpottingDataset(
 audio_files=['audio1.wav', 'audio2.wav', ...],
 labels=[1, 0, ...], # 1=keyword, 0=no keyword
 feature_extractor=extractor,
 augment=True
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
``

### Training Loop

``python
import torch
import torch.nn as nn

def train_keyword_spotting_model(model, train_loader, val_loader, 
 n_epochs=50, device='cuda'):
 """
 Train keyword spotting model
 
 Args:
 model: PyTorch model
 train_loader: Training data loader
 val_loader: Validation data loader
 n_epochs: Number of epochs
 device: Device to train on
 """
 model = model.to(device)
 
 # Loss and optimizer
 criterion = nn.CrossEntropyLoss()
 optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
 optimizer, mode='max', patience=5
 )
 
 best_val_acc = 0
 
 for epoch in range(n_epochs):
 # Training
 model.train()
 train_loss = 0
 train_correct = 0
 train_total = 0
 
 for features, labels in train_loader:
 features = features.to(device)
 labels = labels.squeeze().to(device)
 
 # Forward pass
 outputs = model(features)
 loss = criterion(outputs, labels)
 
 # Backward pass
 optimizer.zero_grad()
 loss.backward()
 optimizer.step()
 
 # Track metrics
 train_loss += loss.item()
 _, predicted = torch.max(outputs, 1)
 train_correct += (predicted == labels).sum().item()
 train_total += labels.size(0)
 
 train_acc = train_correct / train_total
 
 # Validation
 model.eval()
 val_correct = 0
 val_total = 0
 
 with torch.no_grad():
 for features, labels in val_loader:
 features = features.to(device)
 labels = labels.squeeze().to(device)
 
 outputs = model(features)
 _, predicted = torch.max(outputs, 1)
 
 val_correct += (predicted == labels).sum().item()
 val_total += labels.size(0)
 
 val_acc = val_correct / val_total
 
 # Learning rate scheduling
 scheduler.step(val_acc)
 
 # Save best model
 if val_acc > best_val_acc:
 best_val_acc = val_acc
 torch.save(model.state_dict(), 'best_kws_model.pth')
 
 print(f"Epoch {epoch+1}/{n_epochs}: "
 f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
 
 return model

# Train
model = KeywordSpottingCNN()
trained_model = train_keyword_spotting_model(
 model, train_loader, val_loader, n_epochs=50
)
``

---

## Deployment Optimization

### Model Quantization

``python
def quantize_kws_model(model):
 """
 Apply dynamic quantization to linear layers for edge deployment
 """
 import torch
 import torch.nn as nn
 
 model.eval()
 qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
 return qmodel

# Quantize
model_quantized = quantize_kws_model(model)

# Compare serialized sizes for accurate measurement
import io
import torch

def get_model_size_mb(m):
 buffer = io.BytesIO()
 torch.save(m.state_dict(), buffer)
 return len(buffer.getvalue()) / (1024 ** 2)

original_size = get_model_size_mb(model)
quantized_size = get_model_size_mb(model_quantized)

print(f"Original: {original_size:.2f} MB")
print(f"Quantized: {quantized_size:.2f} MB")
print(f"Compression: {original_size / max(quantized_size, 1e-6):.1f}x")
``

### TensorFlow Lite Conversion

``python
def convert_to_tflite(model, sample_input):
 """
 Convert PyTorch model to TensorFlow Lite
 
 For deployment on mobile/edge devices
 """
 import torch
 import onnx
 import tensorflow as tf
 from onnx_tf.backend import prepare
 
 # Step 1: PyTorch → ONNX
 torch.onnx.export(
 model,
 sample_input,
 'kws_model.onnx',
 input_names=['input'],
 output_names=['output'],
 dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
 )
 
 # Step 2: ONNX → TensorFlow
 onnx_model = onnx.load('kws_model.onnx')
 tf_rep = prepare(onnx_model)
 tf_rep.export_graph('kws_model_tf')
 
 # Step 3: TensorFlow → TFLite
 converter = tf.lite.TFLiteConverter.from_saved_model('kws_model_tf')
 
 # Optimization
 converter.optimizations = [tf.lite.Optimize.DEFAULT]
 converter.target_spec.supported_types = [tf.float16]
 
 tflite_model = converter.convert()
 
 # Save
 with open('kws_model.tflite', 'wb') as f:
 f.write(tflite_model)
 
 print(f"TFLite model size: {len(tflite_model) / 1024:.1f} KB")

# Convert
convert_to_tflite(model, sample_input)
``

---

## Real-time Inference

### Streaming KWS System

``python
import sounddevice as sd
import numpy as np
import torch
from collections import deque

class StreamingKeywordSpotter:
 """
 Real-time keyword spotting system
 
 Continuously monitors audio and detects keywords
 """
 
 def __init__(self, model, feature_extractor, 
 threshold=0.8, cooldown_ms=1000):
 self.model = model
 self.model.eval()
 
 self.feature_extractor = feature_extractor
 self.threshold = threshold
 self.cooldown_samples = int(cooldown_ms * 16000 / 1000)
 
 # Audio buffer (1 second)
 self.buffer_size = 16000
 self.audio_buffer = deque(maxlen=self.buffer_size)
 
 # Detection cooldown
 self.last_detection = -self.cooldown_samples
 self.sample_count = 0
 
 def process_audio_chunk(self, audio_chunk):
 """
 Process incoming audio chunk
 
 Args:
 audio_chunk: Audio samples [n_samples]
 
 Returns:
 (detected, confidence) tuple
 """
 # Add to buffer
 self.audio_buffer.extend(audio_chunk)
 self.sample_count += len(audio_chunk)
 
 # Wait until buffer is full
 if len(self.audio_buffer) < self.buffer_size:
 return False, 0.0
 
 # Check cooldown
 if self.sample_count - self.last_detection < self.cooldown_samples:
 return False, 0.0
 
 # Extract features
 audio = np.array(self.audio_buffer)
 features = self.feature_extractor.extract(audio)
 
 # Add batch and channel dimensions
 features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
 
 # Run inference
 with torch.no_grad():
 output = self.model(features_tensor)
 probs = torch.softmax(output, dim=1)
 confidence = probs[0][1].item() # Probability of keyword
 
 # Check threshold
 if confidence >= self.threshold:
 self.last_detection = self.sample_count
 return True, confidence
 
 return False, confidence
 
 def start_listening(self, callback=None):
 """
 Start continuous listening
 
 Args:
 callback: Function called when keyword detected
 """
 print("🎤 Listening for keywords...")
 
 def audio_callback(indata, frames, time_info, status):
 """Process audio in callback"""
 if status:
 print(f"Audio status: {status}")
 
 # Process chunk
 detected, confidence = self.process_audio_chunk(indata[:, 0])
 
 if detected:
 print(f"✓ Keyword detected! (confidence={confidence:.3f})")
 if callback:
 callback(confidence)
 
 # Start audio stream
 with sd.InputStream(
 samplerate=16000,
 channels=1,
 blocksize=1600, # 100ms chunks
 callback=audio_callback
 ):
 print("Press Ctrl+C to stop")
 sd.sleep(1000000) # Sleep indefinitely

# Usage
model = KeywordSpottingCNN()
model.load_state_dict(torch.load('best_kws_model.pth'))

spotter = StreamingKeywordSpotter(
 model=model,
 feature_extractor=extractor,
 threshold=0.8,
 cooldown_ms=1000
)

def on_keyword_detected(confidence):
 """Callback when keyword detected"""
 print(f"🔔 Activating voice assistant... (conf={confidence:.2f})")
 # Trigger downstream processing

spotter.start_listening(callback=on_keyword_detected)
``

---

## Connection to Binary Search (DSA)

Keyword spotting uses binary search for threshold optimization:

``python
class KeywordThresholdOptimizer:
 """
 Find optimal detection threshold using binary search
 
 Balances false accepts vs false rejects
 """
 
 def __init__(self, model, feature_extractor):
 self.model = model
 self.feature_extractor = feature_extractor
 self.model.eval()
 
 def find_optimal_threshold(self, positive_samples, negative_samples,
 target_far=0.01):
 """
 Binary search for threshold that achieves target FAR
 
 FAR = False Acceptance Rate
 
 Args:
 positive_samples: List of keyword audio samples
 negative_samples: List of non-keyword audio samples
 target_far: Target false acceptance rate (e.g., 0.01 = 1%)
 
 Returns:
 Optimal threshold
 """
 # Get confidence scores for all samples
 pos_scores = self._get_scores(positive_samples)
 neg_scores = self._get_scores(negative_samples)
 
 # Binary search on threshold space [0, 1]
 left, right = 0.0, 1.0
 best_threshold = 0.5
 
 for iteration in range(20): # 20 iterations for precision
 mid = (left + right) / 2
 
 # Calculate FAR at this threshold
 false_accepts = sum(1 for score in neg_scores if score >= mid)
 far = false_accepts / len(neg_scores)
 
 # Calculate FRR at this threshold
 false_rejects = sum(1 for score in pos_scores if score < mid)
 frr = false_rejects / len(pos_scores)
 
 print(f"Iteration {iteration}: threshold={mid:.4f}, "
 f"FAR={far:.4f}, FRR={frr:.4f}")
 
 # Adjust search space
 if far > target_far:
 # Too many false accepts, increase threshold
 left = mid
 else:
 # FAR is good, try lowering threshold to reduce FRR
 right = mid
 best_threshold = mid
 
 return best_threshold
 
 def _get_scores(self, audio_samples):
 """Get confidence scores for audio samples"""
 scores = []
 
 for audio in audio_samples:
 # Extract features
 features = self.feature_extractor.extract(audio)
 features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
 
 # Inference
 with torch.no_grad():
 output = self.model(features_tensor)
 probs = torch.softmax(output, dim=1)
 confidence = probs[0][1].item()
 scores.append(confidence)
 
 return scores

# Usage
optimizer = KeywordThresholdOptimizer(model, extractor)

optimal_threshold = optimizer.find_optimal_threshold(
 positive_samples=keyword_audios,
 negative_samples=background_audios,
 target_far=0.01 # 1% false accept rate
)

print(f"Optimal threshold: {optimal_threshold:.4f}")
``

---

## Advanced Model Architectures

### 1. Attention-Based KWS

``python
import torch
import torch.nn as nn

class AttentionKWS(nn.Module):
 """
 Keyword spotting with attention mechanism
 
 Learns to focus on important parts of audio
 """
 
 def __init__(self, n_mels=40, hidden_dim=128, n_classes=2):
 super().__init__()
 
 # Bidirectional LSTM
 self.lstm = nn.LSTM(
 input_size=n_mels,
 hidden_size=hidden_dim,
 num_layers=2,
 batch_first=True,
 bidirectional=True
 )
 
 # Attention layer
 self.attention = nn.Sequential(
 nn.Linear(hidden_dim * 2, 64),
 nn.Tanh(),
 nn.Linear(64, 1)
 )
 
 # Classifier
 self.fc = nn.Linear(hidden_dim * 2, n_classes)
 
 def forward(self, x):
 """
 Args:
 x: [batch, time_steps, n_mels]
 
 Returns:
 [batch, n_classes]
 """
 # LSTM
 lstm_out, _ = self.lstm(x) # [batch, time, hidden*2]
 
 # Attention scores
 attention_scores = self.attention(lstm_out) # [batch, time, 1]
 attention_weights = torch.softmax(attention_scores, dim=1)
 
 # Weighted sum
 context = torch.sum(lstm_out * attention_weights, dim=1) # [batch, hidden*2]
 
 # Classify
 output = self.fc(context)
 
 return output, attention_weights

# Usage
model = AttentionKWS(n_mels=40, hidden_dim=128)

# Train and visualize attention
x = torch.randn(1, 100, 40) # 1 sample
output, attention = model(x)

# Visualize which parts of audio model focuses on
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(attention[0].detach().numpy())
plt.title('Attention Weights Over Time')
plt.xlabel('Time Step')
plt.ylabel('Attention Weight')
plt.savefig('attention_visualization.png')
``

### 2. Res-Net Based KWS

``python
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
 """Residual block for audio"""
 
 def __init__(self, channels):
 super().__init__()
 self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
 self.bn1 = nn.BatchNorm2d(channels)
 self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
 self.bn2 = nn.BatchNorm2d(channels)
 self.relu = nn.ReLU()
 
 def forward(self, x):
 residual = x
 out = self.relu(self.bn1(self.conv1(x)))
 out = self.bn2(self.conv2(out))
 out += residual
 out = self.relu(out)
 return out

class ResNetKWS(nn.Module):
 """
 ResNet-based keyword spotting
 
 Deeper network for better accuracy
 """
 
 def __init__(self, n_mels=40, n_classes=2):
 super().__init__()
 
 # Initial conv
 self.conv1 = nn.Sequential(
 nn.Conv2d(1, 32, kernel_size=3, padding=1),
 nn.BatchNorm2d(32),
 nn.ReLU()
 )
 
 # Residual blocks
 self.res_blocks = nn.Sequential(
 ResNetBlock(32),
 ResNetBlock(32),
 ResNetBlock(32)
 )
 
 # Pooling
 self.pool = nn.AdaptiveAvgPool2d((1, 1))
 
 # Classifier
 self.fc = nn.Linear(32, n_classes)
 
 def forward(self, x):
 """
 Args:
 x: [batch, 1, n_mels, time_steps]
 """
 x = self.conv1(x)
 x = self.res_blocks(x)
 x = self.pool(x)
 x = x.view(x.size(0), -1)
 x = self.fc(x)
 return x

model_resnet = ResNetKWS(n_mels=40)
``

### 3. Transformer-Based KWS

``python
import torch
import torch.nn as nn
import numpy as np

class TransformerKWS(nn.Module):
 """
 Transformer for keyword spotting
 
 State-of-the-art performance but larger model
 """
 
 def __init__(self, n_mels=40, d_model=128, nhead=4, 
 num_layers=2, n_classes=2):
 super().__init__()
 
 # Input projection
 self.input_proj = nn.Linear(n_mels, d_model)
 
 # Positional encoding
 self.pos_encoder = PositionalEncoding(d_model)
 
 # Transformer encoder
 encoder_layer = nn.TransformerEncoderLayer(
 d_model=d_model,
 nhead=nhead,
 dim_feedforward=d_model * 4,
 dropout=0.1,
 batch_first=True
 )
 self.transformer = nn.TransformerEncoder(
 encoder_layer,
 num_layers=num_layers
 )
 
 # Classifier
 self.fc = nn.Linear(d_model, n_classes)
 
 def forward(self, x):
 """
 Args:
 x: [batch, time_steps, n_mels]
 """
 # Project input
 x = self.input_proj(x)
 
 # Add positional encoding
 x = self.pos_encoder(x)
 
 # Transformer
 x = self.transformer(x)
 
 # Global average pooling
 x = x.mean(dim=1)
 
 # Classify
 x = self.fc(x)
 return x

class PositionalEncoding(nn.Module):
 """Positional encoding for transformer"""
 
 def __init__(self, d_model, max_len=5000):
 super().__init__()
 
 pe = torch.zeros(max_len, d_model)
 position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
 div_term = torch.exp(
 torch.arange(0, d_model, 2).float() * 
 (-np.log(10000.0) / d_model)
 )
 
 pe[:, 0::2] = torch.sin(position * div_term)
 pe[:, 1::2] = torch.cos(position * div_term)
 
 self.register_buffer('pe', pe.unsqueeze(0))
 
 def forward(self, x):
 return x + self.pe[:, :x.size(1)]

model_transformer = TransformerKWS(n_mels=40)
``

---

## Data Augmentation Strategies

### Advanced Audio Augmentation

``python
import librosa
import numpy as np

class AudioAugmenter:
 """
 Comprehensive audio augmentation for KWS training
 
 Improves robustness to real-world conditions
 """
 
 def __init__(self):
 self.sample_rate = 16000
 
 def time_stretch(self, audio, rate=None):
 """
 Stretch/compress audio in time
 
 Args:
 audio: Audio samples
 rate: Stretch factor (0.8-1.2 typical)
 """
 if rate is None:
 rate = np.random.uniform(0.9, 1.1)
 
 return librosa.effects.time_stretch(audio, rate=rate)
 
 def pitch_shift(self, audio, n_steps=None):
 """
 Shift pitch without changing speed
 
 Args:
 n_steps: Semitones to shift (-3 to +3 typical)
 """
 if n_steps is None:
 n_steps = np.random.randint(-2, 3)
 
 return librosa.effects.pitch_shift(
 audio,
 sr=self.sample_rate,
 n_steps=n_steps
 )
 
 def add_background_noise(self, audio, noise_audio, snr_db=None):
 """
 Add background noise at specified SNR
 
 Args:
 noise_audio: Background noise samples
 snr_db: Signal-to-noise ratio in dB (10-30 typical)
 """
 if snr_db is None:
 snr_db = np.random.uniform(10, 30)
 
 # Calculate noise scaling factor
 audio_power = np.mean(audio ** 2)
 noise_power = np.mean(noise_audio ** 2)
 
 snr_linear = 10 ** (snr_db / 10)
 noise_scale = np.sqrt(audio_power / (snr_linear * noise_power))
 
 # Mix audio and noise
 return audio + noise_scale * noise_audio
 
 def room_simulation(self, audio, room_size='medium'):
 """
 Simulate room acoustics (reverb)
 
 Args:
 room_size: 'small', 'medium', or 'large'
 """
 # Room impulse response parameters
 params = {
 'small': {'delay': 0.05, 'decay': 0.3},
 'medium': {'delay': 0.1, 'decay': 0.5},
 'large': {'delay': 0.2, 'decay': 0.7}
 }
 
 delay_samples = int(params[room_size]['delay'] * self.sample_rate)
 decay = params[room_size]['decay']
 
 # Simple reverb simulation
 reverb = np.zeros_like(audio)
 reverb[delay_samples:] = audio[:-delay_samples] * decay
 
 return audio + reverb
 
 def apply_compression(self, audio, threshold_db=-20):
 """
 Dynamic range compression
 
 Makes quiet sounds louder, loud sounds quieter
 """
 threshold = 10 ** (threshold_db / 20)
 compressed = np.copy(audio)
 
 # Compress samples above threshold
 mask = np.abs(audio) > threshold
 compressed[mask] = threshold + (audio[mask] - threshold) * 0.5
 
 return compressed
 
 def augment(self, audio):
 """
 Apply random augmentation pipeline
 
 Returns augmented audio
 """
 # Random selection of augmentations
 aug_functions = [
 lambda x: self.time_stretch(x),
 lambda x: self.pitch_shift(x),
 lambda x: self.apply_compression(x),
 ]
 
 # Apply 1-2 random augmentations
 n_augs = np.random.randint(1, 3)
 for _ in range(n_augs):
 aug_fn = np.random.choice(aug_functions)
 audio = aug_fn(audio)
 
 # Add background noise (always)
 noise = np.random.randn(len(audio)) * 0.005
 audio = self.add_background_noise(audio, noise)
 
 return audio

# Usage in training
augmenter = AudioAugmenter()

# Augment training data
augmented_audio = augmenter.augment(original_audio)
``

---

## Production Deployment Patterns

### Multi-Stage Detection Pipeline

``python
import torch

class MultiStageKWSPipeline:
 """
 Multi-stage KWS for production
 
 Stage 1: Lightweight detector (always-on)
 Stage 2: Accurate model (triggered by stage 1)
 
 Optimizes power consumption vs accuracy
 """
 
 def __init__(self, stage1_model, stage2_model, 
 stage1_threshold=0.7, stage2_threshold=0.9):
 self.stage1_model = stage1_model # Tiny model (~50KB)
 self.stage2_model = stage2_model # Accurate model (~500KB)
 
 self.stage1_threshold = stage1_threshold
 self.stage2_threshold = stage2_threshold
 
 self.stats = {
 'stage1_triggers': 0,
 'stage2_confirms': 0,
 'false_positives': 0
 }
 self.total_chunks = 0
 
 def detect(self, audio_chunk):
 """
 Two-stage detection
 
 Returns: (detected, confidence, stage)
 """
 # Increment processed chunks counter
 self.total_chunks += 1

 # Stage 1: Lightweight screening
 stage1_conf = self._run_stage1(audio_chunk)
 
 if stage1_conf < self.stage1_threshold:
 # Not a keyword, skip stage 2
 return False, stage1_conf, 1
 
 self.stats['stage1_triggers'] += 1
 
 # Stage 2: Accurate verification
 stage2_conf = self._run_stage2(audio_chunk)
 
 if stage2_conf >= self.stage2_threshold:
 self.stats['stage2_confirms'] += 1
 return True, stage2_conf, 2
 else:
 self.stats['false_positives'] += 1
 return False, stage2_conf, 2
 
 def _run_stage1(self, audio_chunk):
 """Run lightweight model"""
 features = extract_features_fast(audio_chunk)
 
 with torch.no_grad():
 output = self.stage1_model(features)
 confidence = torch.softmax(output, dim=1)[0][1].item()
 
 return confidence
 
 def _run_stage2(self, audio_chunk):
 """Run accurate model"""
 features = extract_features_high_quality(audio_chunk)
 
 with torch.no_grad():
 output = self.stage2_model(features)
 confidence = torch.softmax(output, dim=1)[0][1].item()
 
 return confidence
 
 def get_precision(self):
 """Calculate precision of two-stage system"""
 total_detections = self.stats['stage2_confirms'] + self.stats['false_positives']
 if total_detections == 0:
 return 0.0
 
 return self.stats['stage2_confirms'] / total_detections
 
 def get_power_savings(self):
 """Estimate power savings from two-stage approach"""
 # Stage 2 ~10x power of stage 1 (normalized units)
 stage2_invocations = self.stats['stage1_triggers']
 total = max(self.total_chunks, 1)
 cost_stage1 = 1.0
 cost_stage2 = 10.0
 
 energy_two_stage = total * cost_stage1 + stage2_invocations * cost_stage2
 energy_single_stage = total * cost_stage2
 
 savings = 1.0 - (energy_two_stage / energy_single_stage)
 return max(0.0, min(1.0, savings))

# Usage
pipeline = MultiStageKWSPipeline(
 stage1_model=lightweight_model,
 stage2_model=accurate_model
)

# Continuous monitoring
for chunk in audio_stream:
 detected, confidence, stage = pipeline.detect(chunk)
 
 if detected:
 print(f"Keyword detected! (stage={stage}, conf={confidence:.3f})")

print(f"Precision: {pipeline.get_precision():.2%}")
print(f"Power savings: {pipeline.get_power_savings():.2%}")
``

### On-Device Learning

``python
class OnDeviceKWSLearner:
 """
 Personalized KWS with on-device learning
 
 Adapts to user's voice without sending data to cloud
 """
 
 def __init__(self, base_model):
 self.base_model = base_model
 
 # Freeze base model
 for param in self.base_model.parameters():
 param.requires_grad = False
 
 # Add personalization layer
 self.personalization_layer = nn.Linear(
 self.base_model.output_dim,
 2
 )
 
 self.optimizer = torch.optim.SGD(
 self.personalization_layer.parameters(),
 lr=0.01
 )
 
 self.user_examples = []
 self.max_examples = 50 # Limited on-device storage
 
 def collect_user_example(self, audio, label):
 """
 Collect user-specific training example
 
 Args:
 audio: User's audio sample
 label: 1 for keyword, 0 for non-keyword
 """
 features = extract_features(audio)
 
 self.user_examples.append((features, label))
 
 # Keep only recent examples
 if len(self.user_examples) > self.max_examples:
 self.user_examples.pop(0)
 
 def personalize(self, n_epochs=10):
 """
 Personalize model to user
 
 Quick fine-tuning on device
 """
 if len(self.user_examples) < 5:
 print("Not enough user examples yet")
 return
 
 print(f"Personalizing with {len(self.user_examples)} examples...")
 
 for epoch in range(n_epochs):
 total_loss = 0
 
 for features, label in self.user_examples:
 # Extract base features
 with torch.no_grad():
 base_output = self.base_model(features)
 
 # Personalization layer
 output = self.personalization_layer(base_output)
 
 # Loss
 loss = nn.CrossEntropyLoss()(
 output.unsqueeze(0),
 torch.tensor([label])
 )
 
 # Update
 self.optimizer.zero_grad()
 loss.backward()
 self.optimizer.step()
 
 total_loss += loss.item()
 
 if epoch % 5 == 0:
 print(f"Epoch {epoch}: Loss = {total_loss / len(self.user_examples):.4f}")
 
 print("Personalization complete!")
 
 def predict(self, audio):
 """Predict with personalized model"""
 features = extract_features(audio)
 
 with torch.no_grad():
 base_output = self.base_model(features)
 output = self.personalization_layer(base_output)
 confidence = torch.softmax(output, dim=1)[0][1].item()
 
 return confidence

# Usage
learner = OnDeviceKWSLearner(base_model)

# User trains their custom wake word
print("Please say your wake word 5 times...")
for i in range(5):
 audio = record_audio()
 learner.collect_user_example(audio, label=1)

print("Please say 5 non-wake-word phrases...")
for i in range(5):
 audio = record_audio()
 learner.collect_user_example(audio, label=0)

# Personalize on-device
learner.personalize(n_epochs=20)

# Use personalized model
confidence = learner.predict(test_audio)
``

---

## Real-World Integration Examples

### Smart Speaker Integration

``python
import time

class SmartSpeakerKWS:
 """
 KWS integrated with smart speaker
 
 Handles wake word → command processing pipeline
 """
 
 def __init__(self, wake_word_model, command_asr_model):
 self.wake_word_model = wake_word_model
 self.command_asr_model = command_asr_model
 
 self.state = 'listening' # 'listening' or 'processing'
 self.wake_word_detected = False
 self.command_timeout = 5.0 # seconds
 
 async def process_audio_stream(self, audio_stream):
 """
 Main processing loop
 
 Always listening for wake word, then processes command
 """
 wake_word_detector = StreamingKeywordSpotter(
 model=self.wake_word_model,
 feature_extractor=KeywordSpottingFeatureExtractor(sample_rate=16000)
 )
 
 async for chunk in audio_stream:
 if self.state == 'listening':
 # Check for wake word
 detected, confidence = wake_word_detector.process_audio_chunk(chunk)
 
 if detected:
 print("🔊 Wake word detected!")
 await self.play_sound('ding.wav') # Audio feedback
 
 # Switch to command processing
 self.state = 'processing'
 self.wake_word_detected = True
 
 # Start command capture
 command_audio = await self.capture_command(audio_stream)
 
 # Process command
 await self.process_command(command_audio)
 
 # Return to listening
 self.state = 'listening'
 
 async def capture_command(self, audio_stream, timeout=5.0):
 """Capture user command after wake word"""
 command_chunks = []
 start_time = time.time()
 
 async for chunk in audio_stream:
 command_chunks.append(chunk)
 
 # Check timeout
 if time.time() - start_time > timeout:
 break
 
 # Check for silence (end of command)
 if self.is_silence(chunk):
 break
 
 return np.concatenate(command_chunks)
 
 async def process_command(self, command_audio):
 """Process voice command"""
 # Transcribe command
 transcription = self.command_asr_model.transcribe(command_audio)
 print(f"Command: {transcription}")
 
 # Execute command
 response = await self.execute_command(transcription)
 
 # Speak response
 await self.speak(response)
 
 async def execute_command(self, command):
 """Execute voice command"""
 # Command routing
 if 'weather' in command.lower():
 return await self.get_weather()
 elif 'music' in command.lower():
 return await self.play_music()
 elif 'timer' in command.lower():
 return await self.set_timer(command)
 else:
 return "Sorry, I didn't understand that."

# Usage
speaker = SmartSpeakerKWS(wake_word_model, command_asr_model)
await speaker.process_audio_stream(microphone_stream)
``

### Mobile App Integration

``python
class MobileKWSManager:
 """
 KWS manager for mobile apps
 
 Handles battery optimization and background processing
 """
 
 def __init__(self, model_path):
 self.model = self.load_optimized_model(model_path)
 self.is_active = False
 self.battery_saver_mode = False
 
 # Performance tracking
 self.battery_usage = 0
 self.detections = 0
 
 def load_optimized_model(self, model_path):
 """Load quantized model for mobile"""
 # Load TFLite model
 import tensorflow as tf
 interpreter = tf.lite.Interpreter(model_path=model_path)
 interpreter.allocate_tensors()
 
 return interpreter
 
 def start_listening(self, battery_level=100):
 """Start KWS with battery-aware mode"""
 self.is_active = True
 
 # Enable battery saver if low battery
 if battery_level < 20:
 self.enable_battery_saver()
 
 # Start audio capture thread
 self.audio_thread = threading.Thread(target=self._audio_processing_loop)
 self.audio_thread.start()
 
 def enable_battery_saver(self):
 """Enable battery saving mode"""
 self.battery_saver_mode = True
 
 # Reduce processing frequency
 self.chunk_duration_ms = 200 # Longer chunks = less processing
 
 # Lower threshold for stage 1
 self.stage1_threshold = 0.8 # Higher threshold = fewer stage 2 triggers
 
 print("⚡ Battery saver mode enabled")
 
 def _audio_processing_loop(self):
 """Background audio processing"""
 while self.is_active:
 # Capture audio
 audio_chunk = self.capture_audio_chunk()
 
 # Process
 detected, confidence = self.detect_keyword(audio_chunk)
 
 if detected:
 self.detections += 1
 self.trigger_callback(confidence)
 
 # Track battery usage (simplified)
 self.battery_usage += 0.001 # mAh per iteration
 
 # Sleep to save battery
 if self.battery_saver_mode:
 time.sleep(0.1)
 
 def detect_keyword(self, audio_chunk):
 """Run inference on mobile"""
 # Extract features
 features = extract_features(audio_chunk)
 
 # TFLite inference
 input_details = self.model.get_input_details()
 output_details = self.model.get_output_details()
 
 self.model.set_tensor(input_details[0]['index'], features)
 self.model.invoke()
 
 output = self.model.get_tensor(output_details[0]['index'])
 confidence = output[0][1]
 
 return confidence > 0.8, confidence
 
 def get_battery_impact(self):
 """Estimate battery impact"""
 return {
 'total_usage_mah': self.battery_usage,
 'detections': self.detections,
 'usage_per_hour': self.battery_usage * 3600 # Extrapolate
 }

# Usage in mobile app
kws_manager = MobileKWSManager('kws_model.tflite')
kws_manager.start_listening(battery_level=get_battery_level())

# Check battery impact
impact = kws_manager.get_battery_impact()
print(f"Battery usage: {impact['usage_per_hour']:.2f} mAh/hour")
``

---

## Key Takeaways

✅ **Ultra-lightweight models** - < 1MB for edge deployment 
✅ **Real-time processing** - < 50ms latency requirement 
✅ **Always-on capability** - Low power consumption 
✅ **Noise robustness** - Data augmentation and preprocessing critical 
✅ **Binary search optimization** - Find optimal detection thresholds 
✅ **Model compression** - Quantization, pruning for deployment 
✅ **Streaming architecture** - Process continuous audio efficiently 

---

**Originally published at:** [arunbaby.com/speech-tech/0009-keyword-spotting](https://www.arunbaby.com/speech-tech/0009-keyword-spotting/)

*If you found this helpful, consider sharing it with others who might benefit.*

