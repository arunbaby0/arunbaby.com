---
title: "Speech Command Classification"
day: 2
collection: speech_tech
categories:
  - speech-tech
tags:
  - classification
  - intent-recognition
  - voice-commands
subdomain: Speech Classification
tech_stack: [CNN, RNN, Transformer]
latency_requirement: "< 100ms"
scale: "1M+ commands/day"
companies: [Google, Amazon, Apple]
related_dsa_day: 2
related_ml_day: 2
---

**How voice assistants recognize "turn on the lights" from raw audio in under 100ms—without full ASR transcription.**

## Introduction

When you say "Alexa, turn off the lights" or "Hey Google, set a timer," your voice assistant doesn't actually transcribe your speech to text first. Instead, it uses a **direct audio-to-intent classification** system that's:

- **Faster** than ASR + NLU (50-100ms vs 200-500ms)
- **Smaller** models (< 10MB vs 100MB+)
- **Works offline** (on-device inference)
- **More privacy-preserving** (no text sent to cloud)

This approach is perfect for a **limited vocabulary of commands** (30-100 commands) where you care more about speed and privacy than open-ended understanding.

**What you'll learn:**
- Why direct audio→intent beats ASR→NLU for commands
- Audio feature extraction (MFCCs, mel-spectrograms)
- Model architectures (CNN, RNN, Attention)
- Training strategies and data augmentation
- On-device deployment and optimization
- Unknown command handling (OOD detection)
- Real-world examples from Google, Amazon, Apple

---

## Problem Definition

Design a speech command classification system for a voice assistant that:

### Functional Requirements

1. **Multi-class Classification**
   - 30-50 predefined commands
   - Examples: "lights on", "volume up", "play music", "stop timer"
   - Support synonyms and variations

2. **Unknown Detection**
   - Detect and reject out-of-vocabulary audio
   - Handle background conversation
   - Distinguish commands from non-commands

3. **Multi-language Support**
   - 5+ languages initially
   - Shared model or separate models per language

4. **Context Awareness**
   - Optional: Use device state as context
   - Example: "turn it off" depends on what's currently on

### Non-Functional Requirements

1. **Latency**
   - End-to-end < 100ms
   - Includes audio buffering, processing, inference

2. **Model Constraints**
   - Model size < 10MB (on-device)
   - RAM usage < 50MB during inference
   - CPU-only (no GPU on most devices)

3. **Accuracy**
   - > 95% on target commands (clean audio)
   - > 90% on noisy audio
   - < 5% false positive rate

4. **Throughput**
   - 1000 QPS per server (cloud)
   - Single inference on device

---

## Why Not ASR + NLU?

### Traditional Pipeline

```
Audio → ASR → Text → NLU → Intent
"lights on" → ASR (200ms) → "lights on" → NLU (50ms) → {action: "lights", state: "on"}
Total latency: 250ms
```

### Direct Classification

```
Audio → Audio Features → CNN → Intent
"lights on" → Mel-spec (5ms) → CNN (40ms) → {action: "lights", state: "on"}
Total latency: 45ms
```

**Advantages:**
- ✅ 5x faster (45ms vs 250ms)
- ✅ 10x smaller model (5MB vs 50MB)
- ✅ Works offline
- ✅ More private (no text)
- ✅ Fewer points of failure

**Disadvantages:**
- ❌ Limited vocabulary (30-50 commands vs unlimited)
- ❌ Less flexible (new commands need retraining)
- ❌ Can't handle complex queries ("turn on the lights in the living room at 8pm")

**When to use each:**
- **Direct classification:** Simple commands, latency-critical, on-device
- **ASR + NLU:** Complex queries, unlimited vocabulary, cloud-based

---

## Architecture

```
Audio Input (1-2 seconds @ 16kHz)
    ↓
Audio Preprocessing
    ├─ Resampling (if needed)
    ├─ Padding/Trimming to fixed length
    └─ Normalization
    ↓
Feature Extraction
    ├─ MFCCs (40 coefficients)
    or
    ├─ Mel-Spectrogram (40 bins)
    ↓
Neural Network
    ├─ CNN (fastest, on-device)
    or
    ├─ RNN (better temporal modeling)
    or
    ├─ Attention (best accuracy, slower)
    ↓
Softmax Layer (31 classes)
    ├─ 30 command classes
    └─ 1 unknown class
    ↓
Post-processing
    ├─ Confidence thresholding
    ├─ Unknown detection
    └─ Output filtering
    ↓
Prediction: {command: "lights_on", confidence: 0.94}
```

---

## Component 1: Audio Preprocessing

### Fixed-Length Input

**Problem:** Audio clips have variable duration (0.5s - 3s)

**Solution:** Standardize to fixed length (e.g., 1 second)

```python
def preprocess_audio(audio: np.ndarray, sr=16000, target_duration=1.0):
    """
    Ensure all audio clips are same length
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        target_duration: Target duration in seconds
    
    Returns:
        Processed audio of length sr * target_duration
    """
    target_length = int(sr * target_duration)
    
    # Pad if too short
    if len(audio) < target_length:
        pad_length = target_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
    
    # Trim if too long
    elif len(audio) > target_length:
        # Take central portion
        start = (len(audio) - target_length) // 2
        audio = audio[start:start + target_length]
    
    return audio
```

**Why fixed length?**
- Neural networks expect fixed-size inputs
- Enables batching during training
- Simplifies model architecture

**Alternative: Variable-length with padding**
```python
def pad_sequence(audios: list, sr=16000):
    """
    Pad multiple audio clips to longest length
    Used during batched inference
    """
    max_length = max(len(a) for a in audios)
    
    padded = []
    masks = []
    
    for audio in audios:
        pad_length = max_length - len(audio)
        padded_audio = np.pad(audio, (0, pad_length))
        mask = np.ones(len(audio)).tolist() + [0] * pad_length
        
        padded.append(padded_audio)
        masks.append(mask)
    
    return np.array(padded), np.array(masks)
```

### Normalization

```python
def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range
    
    Improves model convergence and generalization
    """
    # Peak normalization
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    return audio


def normalize_rms(audio: np.ndarray, target_rms=0.1) -> np.ndarray:
    """
    Normalize by RMS (root mean square) energy
    
    Better for handling volume variations
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        audio = audio * (target_rms / current_rms)
    
    return audio
```

---

## Component 2: Feature Extraction

### Option 1: MFCCs (Mel-Frequency Cepstral Coefficients)

**MFCCs** capture the spectral envelope of speech, which is important for phonetic content.

```python
import librosa

def extract_mfcc(audio, sr=16000, n_mfcc=40, n_fft=512, hop_length=160):
    """
    Extract MFCC features
    
    Args:
        audio: Waveform
        sr: Sample rate (Hz)
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length between frames (10ms at 16kHz)
    
    Returns:
        MFCCs: (n_mfcc, time_steps)
    """
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=40,          # Number of mel bands
        fmin=20,            # Minimum frequency
        fmax=sr//2          # Maximum frequency (Nyquist)
    )
    
    # Add delta (velocity) and delta-delta (acceleration)
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    
    # Stack all features
    features = np.vstack([mfccs, delta, delta2])  # (120, time)
    
    return features.T  # (time, 120)
```

**Why delta features?**
- **MFCCs:** Spectral shape (what phonemes)
- **Delta:** How spectral shape is changing (dynamics)
- **Delta-delta:** Rate of change (acceleration)

Together they capture both static and dynamic characteristics of speech.

### Option 2: Mel-Spectrogram

**Mel-spectrograms** preserve more temporal resolution than MFCCs.

```python
def extract_mel_spectrogram(audio, sr=16000, n_mels=40, n_fft=512, hop_length=160):
    """
    Extract log mel-spectrogram
    
    Returns:
        Log mel-spectrogram: (time, n_mels)
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=20,
        fmax=sr//2
    )
    
    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel.T  # (time, n_mels)
```

**MFCCs vs Mel-Spectrogram:**

| Feature | MFCCs | Mel-Spectrogram |
|---------|-------|-----------------|
| Size | (time, 13-40) | (time, 40-80) |
| Information | Spectral envelope | Full spectrum |
| Works better with | Small models | CNNs (image-like) |
| Training time | Faster | Slower |
| Accuracy | Slightly lower | Slightly higher |

**Recommendation:** Use **mel-spectrograms with CNNs** for best accuracy.

---

## Component 3: Model Architectures

### Architecture 1: CNN (Fastest for On-Device)

```python
import torch
import torch.nn as nn

class CommandCNN(nn.Module):
    """
    CNN for audio command classification
    
    Treats mel-spectrogram as 2D image
    """
    def __init__(self, num_classes=31, input_channels=1):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Global average pooling (instead of fully-connected)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, 1, time, freq)
        
        x = self.conv1(x)   # → (batch, 32, time/2, freq/2)
        x = self.conv2(x)   # → (batch, 64, time/4, freq/4)
        x = self.conv3(x)   # → (batch, 128, time/8, freq/8)
        
        x = self.gap(x)     # → (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # → (batch, 128)
        
        x = self.classifier(x)  # → (batch, num_classes)
        
        return x

# Model size: ~2MB
# Inference time (CPU): 15ms
# Accuracy: ~93%
```

**Why CNNs work for audio:**
- **Local patterns:** Phonemes have localized frequency patterns
- **Translation invariance:** Command can start at different times
- **Parameter sharing:** Same filters across time/frequency
- **Efficient:** Mostly matrix operations, highly optimized

### Architecture 2: RNN (Better Temporal Modeling)

```python
class CommandRNN(nn.Module):
    """
    RNN for command classification
    
    Better at capturing temporal dependencies
    """
    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, num_classes=31):
        super().__init__()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention mechanism (optional)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, time, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # → (batch, time, hidden*2)
        
        # Attention pooling (instead of taking last time step)
        attention_weights = torch.softmax(
            self.attention(lstm_out),  # → (batch, time, 1)
            dim=1
        )
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # → (batch, hidden*2)
        
        # Classify
        logits = self.classifier(context)  # → (batch, num_classes)
        
        return logits

# Model size: ~5MB
# Inference time (CPU): 30ms
# Accuracy: ~95%
```

### Architecture 3: Attention-Based (Best Accuracy)

```python
class CommandTransformer(nn.Module):
    """
    Transformer for command classification
    
    Best accuracy but slower inference
    """
    def __init__(self, input_dim=40, d_model=128, nhead=4, num_layers=2, num_classes=31):
        super().__init__()
        
        # Input projection
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x: (batch, time, features)
        
        # Project to d_model
        x = self.embedding(x)  # → (batch, time, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer expects (time, batch, d_model)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        # Average pool over time
        x = x.mean(dim=1)  # → (batch, d_model)
        
        # Classify
        logits = self.classifier(x)  # → (batch, num_classes)
        
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Model size: ~8MB
# Inference time (CPU): 50ms
# Accuracy: ~97%
```

### Model Comparison

| Model | Params | Size | CPU Latency | GPU Latency | Accuracy | Best For |
|-------|--------|------|-------------|-------------|----------|----------|
| CNN | 500K | 2MB | 15ms | 3ms | 93% | Mobile devices |
| RNN | 1.2M | 5MB | 30ms | 5ms | 95% | Balanced |
| Transformer | 2M | 8MB | 50ms | 8ms | 97% | Cloud/high-end |

**Production choice:** CNN for on-device, RNN for cloud

---

## Training Strategy

### Data Collection

**Per command, need:**
- 1000-5000 examples
- 100+ speakers (diversity)
- Both genders, various ages
- Different accents
- Background noise variations
- Different recording devices

**Example dataset structure:**
```
data/
├── lights_on/
│   ├── speaker001_01.wav
│   ├── speaker001_02.wav
│   ├── speaker002_01.wav
│   └── ...
├── lights_off/
│   └── ...
├── volume_up/
│   └── ...
└── unknown/
    ├── random_speech/
    ├── music/
    ├── noise/
    └── silence/
```

### Data Augmentation

**Critical for robustness!** Augment during training:

```python
import random

def augment_audio(audio, sr=16000):
    """
    Apply random augmentation
    
    Each training example augmented differently
    """
    augmentations = [
        add_noise,
        time_shift,
        time_stretch,
        pitch_shift,
        add_reverb
    ]
    
    # Apply 1-3 random augmentations
    num_augs = random.randint(1, 3)
    selected = random.sample(augmentations, num_augs)
    
    for aug_fn in selected:
        audio = aug_fn(audio, sr)
    
    return audio


def add_noise(audio, sr, snr_db=random.uniform(5, 20)):
    """Add background noise at specific SNR"""
    # Load random noise sample
    noise = load_random_noise_sample(len(audio))
    
    # Calculate noise power for target SNR
    audio_power = np.mean(audio ** 2)
    noise_power = audio_power / (10 ** (snr_db / 10))
    noise_scaled = noise * np.sqrt(noise_power / np.mean(noise ** 2))
    
    return audio + noise_scaled


def time_shift(audio, sr, shift_max=0.1):
    """Shift audio in time (simulates different reaction times)"""
    shift = int(sr * shift_max * (random.random() - 0.5))
    return np.roll(audio, shift)


def time_stretch(audio, sr, rate=random.uniform(0.9, 1.1)):
    """Change speed without changing pitch"""
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio, sr, n_steps=random.randint(-2, 2)):
    """Shift pitch (simulates different speakers)"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def add_reverb(audio, sr):
    """Add room reverb (simulates different environments)"""
    # Simple reverb using convolution with impulse response
    impulse_response = generate_simple_reverb(sr)
    return np.convolve(audio, impulse_response, mode='same')
```

**Impact:** 2-3x effective dataset size, 10-20% accuracy improvement

### Training Loop

```python
def train_command_classifier(
    model, 
    train_loader, 
    val_loader, 
    epochs=100, 
    lr=0.001
):
    """
    Train speech command classifier
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (audio, labels) in enumerate(train_loader):
            # Extract features
            features = extract_features_batch(audio, sr=16000)
            features = torch.tensor(features, dtype=torch.float32)
            
            # Add channel dimension for CNN
            if len(features.shape) == 3:
                features = features.unsqueeze(1)  # (batch, 1, time, freq)
            
            labels = torch.tensor(labels, dtype=torch.long)
            
            # Forward
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            train_loss += loss.item()
        
        train_acc = train_correct / train_total
        avg_loss = train_loss / len(train_loader)
        
        # Validation
        val_acc = validate(model, val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ New best model: {val_acc:.4f}")
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss={avg_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, "
              f"Val Acc={val_acc:.4f}")
    
    return model


def validate(model, val_loader):
    """Evaluate on validation set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audio, labels in val_loader:
            features = extract_features_batch(audio)
            features = torch.tensor(features).unsqueeze(1)
            labels = torch.tensor(labels)
            
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return correct / total
```

---

## Component 4: Handling Unknown Commands

### Strategy 1: Add "Unknown" Class

```python
# Training data
command_classes = [
    "lights_on", "lights_off", "volume_up", "volume_down",
    "play_music", "stop", "pause", "next", "previous",
    # ... 30 total commands
]

# Collect negative examples
unknown_class = [
    "random_speech",  # Conversations
    "music",          # Background music
    "noise",          # Environmental sounds
    "silence"         # No speech
]

# Labels: 0-29 for commands, 30 for unknown
all_classes = command_classes + ["unknown"]
```

**Collecting unknown data:**
```python
# Record actual user interactions
# Label anything that's NOT a command as "unknown"

unknown_samples = []

for audio in production_audio_stream:
    if not is_valid_command(audio):
        unknown_samples.append(audio)
        
        if len(unknown_samples) >= 10000:
            # Add to training set
            augment_and_save(unknown_samples, label="unknown")
```

### Strategy 2: Confidence Thresholding

```python
def predict_with_threshold(model, audio, threshold=0.7):
    """
    Reject low-confidence predictions as unknown
    """
    # Extract features
    features = extract_mel_spectrogram(audio)
    features = torch.tensor(features).unsqueeze(0).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1)[0]
    
    # Get top prediction
    max_prob, predicted_class = torch.max(probs, 0)
    
    # Threshold check
    if max_prob < threshold:
        return "unknown", float(max_prob)
    
    return command_classes[predicted_class], float(max_prob)
```

### Strategy 3: Out-of-Distribution (OOD) Detection

```python
def detect_ood_with_entropy(probs):
    """
    High entropy = model is uncertain = likely OOD
    """
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    
    # Calibrate threshold on validation set
    # In-distribution: entropy ~0.5
    # Out-of-distribution: entropy > 2.0
    
    if entropy > 2.0:
        return True  # OOD
    return False


def detect_ood_with_mahalanobis(features, class_means, class_covariances):
    """
    Mahalanobis distance to class centroids
    
    Far from all classes = likely OOD
    """
    min_distance = float('inf')
    
    for class_idx in range(len(class_means)):
        mean = class_means[class_idx]
        cov = class_covariances[class_idx]
        
        # Mahalanobis distance
        diff = features - mean
        distance = np.sqrt(diff.T @ np.linalg.inv(cov) @ diff)
        
        min_distance = min(min_distance, distance)
    
    # Threshold: 3-sigma rule
    if min_distance > 3.0:
        return True  # OOD
    return False
```

---

## Model Optimization for Edge Deployment

### Quantization

```python
# Post-training quantization
model_fp32 = CommandCNN(num_classes=31)
model_fp32.load_state_dict(torch.load('model.pth'))
model_fp32.eval()

# Dynamic quantization
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# Save
torch.save(model_int8.state_dict(), 'model_int8.pth')

# Results:
# - Model size: 2MB → 0.5MB (4x smaller)
# - Inference: 15ms → 6ms (2.5x faster)
# - Accuracy: 93.2% → 92.8% (0.4% drop)
```

### Pruning

```python
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """
    Remove 30% of weights with lowest magnitude
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
    
    return model

# Results with 30% pruning:
# - Model size: 2MB → 1.4MB
# - Inference: 15ms → 12ms
# - Accuracy: 93.2% → 92.7%
```

### Knowledge Distillation

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7):
    """
    Train small student to mimic large teacher
    
    Args:
        temperature: Soften probability distributions
        alpha: Weight between soft and hard targets
    """
    # Soft targets from teacher
    soft_targets = torch.softmax(teacher_logits / temperature, dim=1)
    soft_prob = torch.log_softmax(student_logits / temperature, dim=1)
    soft_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0]
    soft_loss = soft_loss * (temperature ** 2)
    
    # Hard targets (ground truth)
    hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
    
    # Combine
    return alpha * soft_loss + (1 - alpha) * hard_loss


# Train student
teacher = CommandTransformer(num_classes=31)  # 8MB, 97% accuracy
student = CommandCNN(num_classes=31)          # 2MB, 93% accuracy

for audio, labels in train_loader:
    # Teacher predictions (frozen)
    with torch.no_grad():
        teacher_logits = teacher(audio)
    
    # Student predictions
    student_logits = student(audio)
    
    # Distillation loss
    loss = distillation_loss(student_logits, teacher_logits, labels)
    
    # Optimize student
    loss.backward()
    optimizer.step()

# Result: Student achieves 95% (vs 93% without distillation)
```

---

## On-Device Deployment

### Export to Mobile Formats

**TensorFlow Lite (Android):**

```python
import tensorflow as tf

# Convert PyTorch to TensorFlow (via ONNX)
# 1. Export PyTorch to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output']
)

# 2. Convert ONNX to TF
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("model.onnx")
tf_model = prepare(onnx_model)
tf_model.export_graph("model_tf")

# 3. Convert TF to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('command_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
```

**Core ML (iOS):**

```python
import coremltools as ct

# Trace PyTorch model
example_input = torch.randn(1, 1, 100, 40)
traced_model = torch.jit.trace(model, example_input)

# Convert to Core ML
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="audio", shape=(1, 1, 100, 40))],
    outputs=[ct.TensorType(name="logits")]
)

# Add metadata
coreml_model.author = "Arun Baby"
coreml_model.short_description = "Speech command classifier"
coreml_model.version = "1.0"

# Save
coreml_model.save("CommandClassifier.mlmodel")
```

### Mobile Inference Code

**Android (Kotlin):**

```kotlin
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer

class CommandClassifier(private val context: Context) {
    private lateinit var interpreter: Interpreter
    
    init {
        // Load model
        val model = loadModelFile("command_classifier.tflite")
        interpreter = Interpreter(model)
    }
    
    fun classify(audio: FloatArray): Pair<String, Float> {
        // Extract features
        val features = extractMelSpectrogram(audio)
        
        // Prepare input
        val inputBuffer = ByteBuffer.allocateDirect(4 * features.size)
        inputBuffer.order(ByteOrder.nativeOrder())
        features.forEach { inputBuffer.putFloat(it) }
        
        // Prepare output
        val output = Array(1) { FloatArray(31) }
        
        // Run inference
        interpreter.run(inputBuffer, output)
        
        // Get top prediction
        val probabilities = output[0]
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val confidence = probabilities[maxIndex]
        
        return Pair(commandNames[maxIndex], confidence)
    }
}
```

**iOS (Swift):**

```swift
import CoreML

class CommandClassifier {
    private var model: CommandClassifierModel!
    
    init() {
        model = try! CommandClassifierModel(configuration: MLModelConfiguration())
    }
    
    func classify(audio: [Float]) -> (command: String, confidence: Double) {
        // Extract features
        let features = extractMelSpectrogram(audio)
        
        // Create MLMultiArray
        let input = try! MLMultiArray(shape: [1, 1, 100, 40], dataType: .float32)
        for i in 0..<features.count {
            input[i] = NSNumber(value: features[i])
        }
        
        // Run inference
        let output = try! model.prediction(audio: input)
        
        // Get top prediction
        let probabilities = output.logits
        let maxIndex = probabilities.argmax()
        let confidence = probabilities[maxIndex]
        
        return (commandNames[maxIndex], Double(confidence))
    }
}
```

---

## Monitoring & Evaluation

### Metrics Dashboard

```python
from dataclasses import dataclass
from typing import List

@dataclass
class ClassificationMetrics:
    """Per-class metrics"""
    precision: float
    recall: float
    f1_score: float
    support: int  # Number of samples
    
def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int):
    """
    Compute detailed metrics per class
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Identify problematic classes
    for i in range(num_classes):
        if report[str(i)]['f1-score'] < 0.85:
            print(f"⚠️  Class {i} ({command_names[i]}) has low F1: {report[str(i)]['f1-score']:.3f}")
            
            # Find most confused class
            confused_with = cm[i].argmax()
            if confused_with != i:
                print(f"   Most confused with class {confused_with} ({command_names[confused_with]})")
    
    return report, cm
```

### Online Monitoring

```python
class OnlineMetricsTracker:
    """
    Track metrics in production
    """
    def __init__(self):
        self.predictions = []
        self.confidences = []
        self.latencies = []
    
    def record(self, prediction: int, confidence: float, latency_ms: float):
        """Record single prediction"""
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.latencies.append(latency_ms)
    
    def get_stats(self, last_n=1000):
        """Get recent statistics"""
        recent_preds = self.predictions[-last_n:]
        recent_confs = self.confidences[-last_n:]
        recent_lats = self.latencies[-last_n:]
        
        # Class distribution
        from collections import Counter
        class_dist = Counter(recent_preds)
        
        return {
            'total_predictions': len(recent_preds),
            'class_distribution': dict(class_dist),
            'avg_confidence': np.mean(recent_confs),
            'low_confidence_rate': sum(c < 0.7 for c in recent_confs) / len(recent_confs),
            'p50_latency': np.percentile(recent_lats, 50),
            'p95_latency': np.percentile(recent_lats, 95),
            'p99_latency': np.percentile(recent_lats, 99)
        }
```

---

## Real-World Examples

### Google Assistant

**"Hey Google" Wake Word:**
- Always-on detection using tiny model (< 1MB)
- Runs on low-power co-processor
- < 10ms latency
- ~ 99.5% accuracy on target phrase

**Command Classification:**
- Separate model for common commands
- Fallback to full ASR for complex queries
- On-device for privacy

### Amazon Alexa

**"Alexa" Wake Word:**
- Multiple-stage detection
- Stage 1: Simple energy detector (< 1ms)
- Stage 2: Keyword spotter (< 10ms)
- Stage 3: Full verification (< 50ms)

**Custom Skills:**
- Slot-filling approach
- Template: "play {song} by {artist}"
- Combined classification + entity extraction

### Apple Siri

**"Hey Siri" Detection:**
- Neural network on Neural Engine (iOS)
- Personalized to user's voice over time
- < 50ms latency
- Works offline

---

## Key Takeaways

✅ **Direct audio→intent** faster than ASR→NLU for limited commands  
✅ **CNNs on mel-spectrograms** work excellently for on-device  
✅ **Data augmentation** critical for robustness (noise, time shift, pitch)  
✅ **Unknown class handling** prevents false activations  
✅ **Quantization** achieves 4x compression with < 1% accuracy loss  
✅ **Threshold tuning** balances precision/recall for business needs

---

## Further Reading

**Papers:**
- [Speech Commands Dataset (Google)](https://arxiv.org/abs/1804.03209)
- [Efficient Keyword Spotting](https://arxiv.org/abs/1711.07128)
- [Hey Snips](https://arxiv.org/abs/1811.07684)

**Datasets:**
- [Google Speech Commands v2](https://www.tensorflow.org/datasets/catalog/speech_commands)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/)

**Tools:**
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Core ML](https://developer.apple.com/documentation/coreml)
- [Librosa](https://librosa.org/) - Audio processing

---

**Originally published at:** [arunbaby.com/speech-tech/0002-speech-classification](https://www.arunbaby.com/speech-tech/0002-speech-classification/)

*If you found this helpful, consider sharing it with others who might benefit.*
