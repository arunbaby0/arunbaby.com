---
title: "Voice Activity Detection (VAD)"
day: 4
collection: speech_tech
categories:
  - speech-tech
tags:
  - vad
  - audio-processing
  - real-time
subdomain: Audio Processing
tech_stack: [Python, WebRTC, Librosa, PyTorch]
latency_requirement: "< 5ms"
scale: "Real-time streaming"
companies: [Google, Amazon, Microsoft, Zoom]
related_dsa_day: 4
related_agents_day: 4
---

**How voice assistants and video conferencing apps detect when you're speaking vs silence, the critical first step in every speech pipeline.**

## Introduction

**Voice Activity Detection (VAD)** is the task of determining which parts of an audio stream contain speech vs non-speech (silence, background noise, music).

VAD is the **gatekeeper** of speech systems:
- Triggers when to start listening (wake word detection)
- Determines when utterance ends (endpoint detection)
- Saves compute by only processing speech frames
- Improves bandwidth by only transmitting speech

**Why it matters:**
- **Power efficiency:** Voice assistants sleep until speech detected
- **Latency:** Know when user finished speaking → respond faster
- **Bandwidth:** Transmit only speech frames in VoIP
- **Accuracy:** Reduce false alarms in ASR systems

**What you'll learn:**
- Energy-based VAD (simple, fast)
- WebRTC VAD (production standard)
- ML-based VAD (state-of-the-art)
- Real-time streaming implementation
- Production deployment considerations

---

## Problem Definition

Design a real-time voice activity detection system.

### Functional Requirements

1. **Detection**
   - Classify each audio frame as speech or non-speech
   - Handle noisy environments
   - Detect speech from multiple speakers

2. **Endpoint Detection**
   - Determine start of speech
   - Determine end of speech
   - Handle pauses within utterances

3. **Real-time Processing**
   - Process audio frames as they arrive
   - Minimal buffering
   - Low latency

### Non-Functional Requirements

1. **Latency**
   - Frame-level detection: < 5ms
   - Endpoint detection: < 100ms after speech ends

2. **Accuracy**
   - True positive rate > 95% (detect speech)
   - False positive rate < 5% (mistake noise for speech)

3. **Robustness**
   - Work in SNR (Signal-to-Noise Ratio) down to 0 dB
   - Handle various noise types (music, traffic, crowds)
   - Adapt to different speakers

---

## Approach 1: Energy-Based VAD

Simplest approach: Speech has higher energy than silence.

### Implementation

```python
import numpy as np
import librosa

class EnergyVAD:
    """
    Energy-based Voice Activity Detection
    
    Pros: Simple, fast, no training required
    Cons: Sensitive to noise, poor in low SNR
    """
    
    def __init__(
        self,
        sr=16000,
        frame_length_ms=20,
        hop_length_ms=10,
        energy_threshold=0.01
    ):
        self.sr = sr
        self.frame_length = int(sr * frame_length_ms / 1000)
        self.hop_length = int(sr * hop_length_ms / 1000)
        self.energy_threshold = energy_threshold
    
    def compute_energy(self, frame):
        """
        Compute frame energy (RMS)
        
        Energy = sqrt(mean(x^2))
        """
        return np.sqrt(np.mean(frame ** 2))
    
    def detect(self, audio):
        """
        Detect speech frames
        
        Args:
            audio: Audio signal
        
        Returns:
            List of booleans (True = speech, False = non-speech)
        """
        # Frame the audio
        frames = librosa.util.frame(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        # Compute energy per frame
        energies = np.array([self.compute_energy(frame) for frame in frames.T])
        
        # Threshold
        is_speech = energies > self.energy_threshold
        
        return is_speech
    
    def get_speech_segments(self, audio):
        """
        Get speech segments (start, end) in seconds
        
        Returns:
            List of (start_time, end_time) tuples
        """
        is_speech = self.detect(audio)
        
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                # Speech started
                start_frame = i
                in_speech = True
            elif not speech and in_speech:
                # Speech ended
                end_frame = i
                in_speech = False
                
                # Convert frames to time
                start_time = start_frame * self.hop_length / self.sr
                end_time = end_frame * self.hop_length / self.sr
                
                segments.append((start_time, end_time))
        
        # Handle case where audio ends during speech
        if in_speech:
            end_time = len(is_speech) * self.hop_length / self.sr
            start_time = start_frame * self.hop_length / self.sr
            segments.append((start_time, end_time))
        
        return segments

# Usage
vad = EnergyVAD(energy_threshold=0.01)

# Load audio
audio, sr = librosa.load('speech_with_silence.wav', sr=16000)

# Detect speech
is_speech = vad.detect(audio)
print(f"Speech frames: {is_speech.sum()} / {len(is_speech)}")

# Get segments
segments = vad.get_speech_segments(audio)
for start, end in segments:
    print(f"Speech from {start:.2f}s to {end:.2f}s")
```

### Adaptive Thresholding

Fixed thresholds fail in varying noise conditions. Use adaptive thresholds.

```python
class AdaptiveEnergyVAD(EnergyVAD):
    """
    Energy VAD with adaptive threshold
    
    Threshold adapts to background noise level
    """
    
    def __init__(self, sr=16000, frame_length_ms=20, hop_length_ms=10):
        super().__init__(sr, frame_length_ms, hop_length_ms)
        self.noise_energy = 0.001  # Initial estimate
        self.alpha = 0.95  # Smoothing factor
    
    def detect(self, audio):
        """Detect with adaptive threshold"""
        frames = librosa.util.frame(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        is_speech = []
        
        for frame in frames.T:
            energy = self.compute_energy(frame)
            
            # Adaptive threshold: 3x noise energy
            threshold = 3.0 * self.noise_energy
            
            if energy > threshold:
                # Likely speech
                is_speech.append(True)
            else:
                # Likely noise/silence
                is_speech.append(False)
                
                # Update noise estimate (during silence only)
                self.noise_energy = self.alpha * self.noise_energy + (1 - self.alpha) * energy
        
        return np.array(is_speech)
```

---

## Approach 2: Zero-Crossing Rate + Energy

Combine energy with zero-crossing rate for better accuracy.

### Implementation

```python
class ZCR_Energy_VAD:
    """
    VAD using Energy + Zero-Crossing Rate
    
    Intuition:
    - Speech: Low ZCR (voiced sounds), moderate to high energy
    - Noise: High ZCR (unvoiced), varying energy
    - Silence: Low energy
    """
    
    def __init__(
        self,
        sr=16000,
        frame_length_ms=20,
        hop_length_ms=10,
        energy_threshold=0.01,
        zcr_threshold=0.1
    ):
        self.sr = sr
        self.frame_length = int(sr * frame_length_ms / 1000)
        self.hop_length = int(sr * hop_length_ms / 1000)
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
    
    def compute_zcr(self, frame):
        """
        Compute zero-crossing rate
        
        ZCR = # of times signal crosses zero / # samples
        """
        signs = np.sign(frame)
        zcr = np.mean(np.abs(np.diff(signs))) / 2
        return zcr
    
    def detect(self, audio):
        """
        Detect using both energy and ZCR
        """
        frames = librosa.util.frame(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        is_speech = []
        
        for frame in frames.T:
            energy = np.sqrt(np.mean(frame ** 2))
            zcr = self.compute_zcr(frame)
            
            # Decision logic
            if energy > self.energy_threshold:
                # High energy: could be speech or noise
                if zcr < self.zcr_threshold:
                    # Low ZCR → likely speech (voiced)
                    is_speech.append(True)
                else:
                    # High ZCR → likely noise
                    is_speech.append(False)
            else:
                # Low energy → silence
                is_speech.append(False)
        
        return np.array(is_speech)
```

---

## Approach 3: WebRTC VAD

Industry-standard VAD used in Chrome, Skype, etc.

### Using WebRTC VAD

```python
# WebRTC VAD requires: pip install webrtcvad
import webrtcvad
import struct

class WebRTCVAD:
    """
    WebRTC Voice Activity Detector
    
    Pros:
    - Production-tested (billions of users)
    - Fast, CPU-efficient
    - Robust to noise
    
    Cons:
    - Only works with specific sample rates (8/16/32/48 kHz)
    - Fixed frame sizes (10/20/30 ms)
    """
    
    def __init__(self, sr=16000, frame_duration_ms=30, aggressiveness=3):
        """
        Args:
            sr: Sample rate (must be 8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame duration (10, 20, or 30 ms)
            aggressiveness: 0-3 (0=least aggressive, 3=most aggressive)
                - Higher = more likely to classify as non-speech
                - Use 3 for noisy environments
        """
        if sr not in [8000, 16000, 32000, 48000]:
            raise ValueError("Sample rate must be 8000, 16000, 32000, or 48000")
        
        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError("Frame duration must be 10, 20, or 30 ms")
        
        self.sr = sr
        self.frame_duration_ms = frame_duration_ms
        self.frame_length = int(sr * frame_duration_ms / 1000)
        
        # Create VAD instance
        self.vad = webrtcvad.Vad(aggressiveness)
    
    def detect(self, audio):
        """
        Detect speech in audio
        
        Args:
            audio: numpy array of int16 samples
        
        Returns:
            List of booleans (True = speech)
        """
        # Convert float to int16 if needed (clip to avoid overflow)
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)
        
        # Frame audio
        num_frames = len(audio) // self.frame_length
        is_speech = []
        
        for i in range(num_frames):
            start = i * self.frame_length
            end = start + self.frame_length
            frame = audio[start:end]
            
            # Convert to bytes
            frame_bytes = struct.pack('%dh' % len(frame), *frame)
            
            # Detect
            speech = self.vad.is_speech(frame_bytes, self.sr)
            is_speech.append(speech)
        
        return np.array(is_speech)
    
    def get_speech_timestamps(self, audio):
        """
        Get speech timestamps
        
        Returns:
            List of (start_time, end_time) in seconds
        """
        is_speech = self.detect(audio)
        
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not speech and in_speech:
                in_speech = False
                start_time = start_frame * self.frame_length / self.sr
                end_time = i * self.frame_length / self.sr
                segments.append((start_time, end_time))
        
        if in_speech:
            start_time = start_frame * self.frame_length / self.sr
            end_time = len(is_speech) * self.frame_length / self.sr
            segments.append((start_time, end_time))
        
        return segments

# Usage
vad = WebRTCVAD(sr=16000, frame_duration_ms=30, aggressiveness=3)

audio, sr = librosa.load('audio.wav', sr=16000)
segments = vad.get_speech_timestamps(audio)

print("Speech segments:")
for start, end in segments:
    print(f"  {start:.2f}s - {end:.2f}s")
```

---

## Approach 4: ML-Based VAD

Use neural networks for state-of-the-art performance.

### CNN-based VAD

```python
import torch
import torch.nn as nn

class CNNVAD(nn.Module):
    """
    CNN-based Voice Activity Detector
    
    Input: Mel-spectrogram (time, freq)
    Output: Speech probability per frame
    """
    
    def __init__(self, n_mels=40):
        super().__init__()
        
        # CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
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
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=64 * (n_mels // 4),
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Classification head
        self.fc = nn.Linear(256, 1)  # Binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch, 1, time, n_mels)
        
        Returns:
            Speech probabilities: (batch, time)
        """
        # CNN
        x = self.conv1(x)  # (batch, 32, time/2, n_mels/2)
        x = self.conv2(x)  # (batch, 64, time/4, n_mels/4)
        
        # Reshape for LSTM
        batch, channels, time, freq = x.size()
        x = x.permute(0, 2, 1, 3)  # (batch, time, channels, freq)
        x = x.reshape(batch, time, channels * freq)
        
        # LSTM
        x, _ = self.lstm(x)  # (batch, time, 256)
        
        # Classification
        x = self.fc(x)  # (batch, time, 1)
        x = self.sigmoid(x)  # (batch, time, 1)
        
        return x.squeeze(-1)  # (batch, time)

# Usage
model = CNNVAD(n_mels=40)

# Example input: mel-spectrogram
mel_spec = torch.randn(1, 1, 100, 40)  # (batch=1, channels=1, time=100, mels=40)

# Predict
speech_prob = model(mel_spec)  # (1, 100) - probability per frame
is_speech = speech_prob > 0.5  # Threshold at 0.5

print(f"Speech probability shape: {speech_prob.shape}")
print(f"Detected speech in {is_speech.sum().item()} / {is_speech.size(1)} frames")
```

### Training ML VAD

```python
class VADTrainer:
    """
    Train VAD model
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for mel_specs, labels in train_loader:
            mel_specs = mel_specs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            predictions = self.model(mel_specs)
            loss = self.criterion(predictions, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for mel_specs, labels in val_loader:
                mel_specs = mel_specs.to(self.device)
                labels = labels.to(self.device)
                
                predictions = self.model(mel_specs)
                predicted = (predictions > 0.5).float()
                
                correct += (predicted == labels).sum().item()
                total += labels.numel()
        
        accuracy = correct / total
        return accuracy
```

---

## Real-Time Streaming VAD

Process audio as it arrives (streaming).

### Streaming Implementation

```python
from collections import deque
import numpy as np
import struct

class StreamingVAD:
    """
    Real-time VAD for streaming audio
    
    Use case: Voice assistants, VoIP, live transcription
    """
    
    def __init__(
        self,
        sr=16000,
        frame_duration_ms=30,
        aggressiveness=3,
        speech_pad_ms=300
    ):
        self.sr = sr
        self.frame_duration_ms = frame_duration_ms
        self.frame_length = int(sr * frame_duration_ms / 1000)
        self.speech_pad_ms = speech_pad_ms
        self.speech_pad_frames = int(speech_pad_ms / frame_duration_ms)
        
        # WebRTC VAD
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # State
        self.buffer = deque(maxlen=10000)  # Audio buffer
        self.speech_frames = 0  # Consecutive speech frames
        self.silence_frames = 0  # Consecutive silence frames
        self.in_speech = False
        
        # Store speech segments
        self.current_speech = []
    
    def add_audio(self, audio_chunk):
        """
        Add audio chunk to buffer
        
        Args:
            audio_chunk: New audio samples (int16)
        """
        self.buffer.extend(audio_chunk)
    
    def process_frame(self):
        """
        Process one frame from buffer
        
        Returns:
            (is_speech, speech_ended, speech_audio)
        """
        if len(self.buffer) < self.frame_length:
            return None, False, None
        
        # Extract frame
        frame = np.array([self.buffer.popleft() for _ in range(self.frame_length)])
        
        # Convert to bytes
        frame_bytes = struct.pack('%dh' % len(frame), *frame)
        
        # Detect
        is_speech = self.vad.is_speech(frame_bytes, self.sr)
        
        # Update state
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            
            if not self.in_speech:
                # Speech just started
                self.in_speech = True
                self.current_speech = []
            
            # Add to current speech
            self.current_speech.extend(frame)
        
        else:
            self.silence_frames += 1
            self.speech_frames = 0
            
            if self.in_speech:
                # Add padding
                self.current_speech.extend(frame)
                
                # Check if speech ended
                if self.silence_frames >= self.speech_pad_frames:
                    # Speech ended
                    self.in_speech = False
                    speech_audio = np.array(self.current_speech)
                    self.current_speech = []
                    
                    return False, True, speech_audio
        
        return is_speech, False, None
    
    def process_stream(self):
        """
        Process all buffered audio
        
        Yields speech segments as they complete
        """
        while len(self.buffer) >= self.frame_length:
            is_speech, speech_ended, speech_audio = self.process_frame()
            
            if speech_ended:
                yield speech_audio

# Usage
streaming_vad = StreamingVAD(sr=16000, frame_duration_ms=30)

# Simulate streaming (process chunks as they arrive)
chunk_size = 480  # 30ms at 16kHz

for chunk_start in range(0, len(audio), chunk_size):
    chunk = audio[chunk_start:chunk_start + chunk_size]
    
    # Add to buffer
    streaming_vad.add_audio(chunk.astype(np.int16))
    
    # Process
    for speech_segment in streaming_vad.process_stream():
        print(f"Speech segment detected: {len(speech_segment)} samples")
        # Send to ASR, save, etc.
```

---

## Production Considerations

### Hangover and Padding

Add padding before/after speech to avoid cutting off words.

```python
class VADWithPadding:
    """
    VAD with pre/post padding
    """
    
    def __init__(
        self,
        vad,
        pre_pad_ms=200,
        post_pad_ms=500,
        sr=16000
    ):
        self.vad = vad
        self.pre_pad_frames = int(pre_pad_ms / 30)  # Assuming 30ms frames
        self.post_pad_frames = int(post_pad_ms / 30)
        self.sr = sr
    
    def detect_with_padding(self, audio):
        """
        Detect speech with padding
        """
        is_speech = self.vad.detect(audio)
        
        # Add pre-padding
        padded = np.copy(is_speech)
        for i in range(len(is_speech)):
            if is_speech[i]:
                # Mark previous frames as speech
                start = max(0, i - self.pre_pad_frames)
                padded[start:i] = True
        
        # Add post-padding
        for i in range(len(is_speech)):
            if is_speech[i]:
                # Mark following frames as speech
                end = min(len(is_speech), i + self.post_pad_frames)
                padded[i:end] = True
        
        return padded
```

### Performance Optimization

```python
import time

class OptimizedVAD:
    """
    Optimized VAD for production
    """
    
    def __init__(self, vad_impl):
        self.vad = vad_impl
        self.stats = {
            'total_frames': 0,
            'speech_frames': 0,
            'processing_time': 0
        }
    
    def detect_with_stats(self, audio):
        """Detect with performance tracking"""
        start = time.perf_counter()
        
        is_speech = self.vad.detect(audio)
        
        end = time.perf_counter()
        
        # Update stats
        self.stats['total_frames'] += len(is_speech)
        self.stats['speech_frames'] += is_speech.sum()
        self.stats['processing_time'] += (end - start)
        
        return is_speech
    
    def get_stats(self):
        """Get performance statistics"""
        if self.stats['total_frames'] == 0:
            return None
        
        speech_ratio = self.stats['speech_frames'] / self.stats['total_frames']
        avg_time_per_frame = self.stats['processing_time'] / self.stats['total_frames']
        
        return {
            'speech_ratio': speech_ratio,
            'avg_latency_ms': avg_time_per_frame * 1000,
            'total_frames': self.stats['total_frames'],
            'speech_frames': self.stats['speech_frames']
        }
```

---

## Integration with ASR Pipeline

VAD as the first stage in speech recognition systems.

### End-to-End Pipeline

```python
class SpeechPipeline:
    """
    Complete speech recognition pipeline with VAD
    
    Pipeline: Audio → VAD → ASR → Text
    """
    
    def __init__(self):
        # VAD
        self.vad = WebRTCVAD(sr=16000, frame_duration_ms=30, aggressiveness=3)
        
        # Placeholder for ASR model
        self.asr_model = None  # Would be actual ASR model
        
        # Buffering
        self.min_speech_duration = 0.5  # seconds
        self.max_speech_duration = 10.0  # seconds
    
    def process_audio_file(self, audio_path):
        """
        Process audio file end-to-end
        
        Returns:
            List of transcriptions
        """
        # Load audio
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Run VAD
        speech_segments = self.vad.get_speech_timestamps(audio)
        
        # Filter by duration
        valid_segments = [
            (start, end) for start, end in speech_segments
            if (end - start) >= self.min_speech_duration and
               (end - start) <= self.max_speech_duration
        ]
        
        transcriptions = []
        
        for start, end in valid_segments:
            # Extract speech segment
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            speech_audio = audio[start_sample:end_sample]
            
            # Run ASR (placeholder)
            # transcript = self.asr_model.transcribe(speech_audio)
            transcript = f"[Speech from {start:.2f}s to {end:.2f}s]"
            
            transcriptions.append({
                'start': start,
                'end': end,
                'duration': end - start,
                'text': transcript
            })
        
        return transcriptions
    
    def process_streaming(self, audio_stream):
        """
        Process streaming audio
        
        Yields transcriptions as speech segments complete
        """
        streaming_vad = StreamingVAD(sr=16000, frame_duration_ms=30)
        
        for chunk in audio_stream:
            streaming_vad.add_audio(chunk)
            
            for speech_segment in streaming_vad.process_stream():
                # Run ASR on completed segment
                # transcript = self.asr_model.transcribe(speech_segment)
                transcript = "[Speech detected]"
                
                yield {
                    'audio': speech_segment,
                    'text': transcript,
                    'timestamp': time.time()
                }

# Usage
pipeline = SpeechPipeline()

# Process file
transcriptions = pipeline.process_audio_file('conversation.wav')
for t in transcriptions:
    print(f"{t['start']:.2f}s - {t['end']:.2f}s: {t['text']}")
```

### Double-Pass VAD for Higher Accuracy

Use aggressive VAD first, then refine with ML model.

```python
class TwoPassVAD:
    """
    Two-pass VAD for improved accuracy
    
    Pass 1: Fast WebRTC VAD (aggressive) → candidate segments
    Pass 2: ML VAD (accurate) → final segments
    """
    
    def __init__(self):
        # Fast pass: WebRTC VAD (aggressive)
        self.fast_vad = WebRTCVAD(sr=16000, frame_duration_ms=30, aggressiveness=3)
        
        # Accurate pass: ML VAD
        self.ml_vad = CNNVAD(n_mels=40)
        self.ml_vad.eval()
    
    def detect(self, audio):
        """
        Two-pass detection
        
        Returns:
            Refined speech segments
        """
        # Pass 1: Fast VAD to get candidate regions
        candidate_segments = self.fast_vad.get_speech_timestamps(audio)
        
        # Pass 2: ML VAD to refine each candidate
        refined_segments = []
        
        for start, end in candidate_segments:
            # Extract segment
            start_sample = int(start * 16000)
            end_sample = int(end * 16000)
            segment_audio = audio[start_sample:end_sample]
            
            # Run ML VAD on segment
            # Convert to mel-spectrogram
            import librosa
            mel_spec = librosa.feature.melspectrogram(
                y=segment_audio,
                sr=16000,
                n_mels=40
            )
            
            # ML model prediction
            # mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0)
            # with torch.no_grad():
            #     predictions = self.ml_vad(mel_tensor)
            #     is_speech_frames = predictions > 0.5
            
            # For now, accept if fast VAD said speech
            refined_segments.append((start, end))
        
        return refined_segments
```

---

## Comparison of VAD Methods

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| **Energy-based** | Simple, fast, no training | Poor in noise | Quiet environments |
| **ZCR + Energy** | Better than energy alone | Still noise-sensitive | Moderate noise |
| **WebRTC VAD** | Fast, robust, production-tested | Fixed aggressiveness | Real-time apps, VoIP |
| **ML-based (CNN)** | Best accuracy, adaptable | Requires training, slower | High-noise, accuracy-critical |
| **ML-based (RNN)** | Temporal modeling | Higher latency | Offline processing |
| **Hybrid (2-pass)** | Balance speed/accuracy | More complex | Production ASR |

---

## Production Deployment

### Latency Budgets

For real-time applications:

```
Voice Assistant Latency Budget:
┌─────────────────────────────────────┐
│ VAD Detection:          5-10ms      │
│ Endpoint Detection:     100-200ms   │
│ ASR Processing:         500-1000ms  │
│ NLU + Dialog:           100-200ms   │
│ TTS Generation:         200-500ms   │
├─────────────────────────────────────┤
│ Total:                  ~1-2 seconds│
└─────────────────────────────────────┘

VAD must be fast to keep overall latency low!
```

### Resource Usage

```python
import psutil
import time

class VADProfiler:
    """
    Profile VAD performance
    """
    
    def __init__(self, vad):
        self.vad = vad
    
    def profile(self, audio, num_runs=100):
        """
        Benchmark VAD
        
        Returns:
            Performance metrics
        """
        latencies = []
        
        # Warm-up
        for _ in range(10):
            self.vad.detect(audio)
        
        # Measure
        process = psutil.Process()
        
        cpu_percent_before = process.cpu_percent()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        for _ in range(num_runs):
            start = time.perf_counter()
            result = self.vad.detect(audio)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # ms
        
        cpu_percent_after = process.cpu_percent()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_fps': 1000 / np.mean(latencies),
            'cpu_usage_pct': cpu_percent_after - cpu_percent_before,
            'memory_mb': memory_after - memory_before
        }

# Usage
profiler = VADProfiler(WebRTCVAD())

audio, sr = librosa.load('test.wav', sr=16000, duration=10.0)
metrics = profiler.profile(audio)

print(f"Mean latency: {metrics['mean_latency_ms']:.2f}ms")
print(f"P95 latency: {metrics['p95_latency_ms']:.2f}ms")
print(f"Throughput: {metrics['throughput_fps']:.0f} frames/sec")
print(f"CPU usage: {metrics['cpu_usage_pct']:.1f}%")
print(f"Memory: {metrics['memory_mb']:.1f} MB")
```

### Mobile/Edge Deployment

Optimize VAD for on-device deployment.

```python
class MobileOptimizedVAD:
    """
    VAD optimized for mobile devices
    
    Quantized model, reduced precision, smaller memory footprint
    """
    
    def __init__(self):
        # Use int8 quantization for mobile
        import torch
        
        self.model = CNNVAD(n_mels=40)
        
        # Quantize model
        # Dynamic quantization applies to Linear/LSTM; Conv2d not supported
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        self.model.eval()
    
    def detect_efficient(self, audio):
        """
        Efficient detection with reduced memory
        
        Process in chunks to reduce peak memory
        """
        chunk_size = 16000  # 1 second chunks
        results = []
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            
            # Process chunk
            # result = self.process_chunk(chunk)
            # results.extend(result)
            pass
        
        return results
```

---

## Monitoring & Debugging

### VAD Quality Metrics

```python
class VADEvaluator:
    """
    Evaluate VAD performance
    
    Metrics:
    - Precision: % of detected speech that is actual speech
    - Recall: % of actual speech that was detected
    - F1 score
    - False alarm rate
    - Miss rate
    """
    
    def __init__(self):
        pass
    
    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> dict:
        """
        Compute VAD metrics
        
        Args:
            predictions: Binary array (1=speech, 0=non-speech)
            ground_truth: Ground truth labels
        
        Returns:
            Dictionary of metrics
        """
        # True positives, false positives, etc.
        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        tn = np.sum((predictions == 0) & (ground_truth == 0))
        fn = np.sum((predictions == 0) & (ground_truth == 1))
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'false_alarm_rate': false_alarm_rate,
            'miss_rate': miss_rate,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }

# Usage
evaluator = VADEvaluator()

# Load ground truth
# ground_truth = load_annotations('test_audio.txt')

# Run VAD
vad = WebRTCVAD()
# predictions = vad.detect(audio)

# Evaluate
# metrics = evaluator.evaluate(predictions, ground_truth)

# print(f"Precision: {metrics['precision']:.3f}")
# print(f"Recall: {metrics['recall']:.3f}")
# print(f"F1 Score: {metrics['f1_score']:.3f}")
# print(f"False Alarm Rate: {metrics['false_alarm_rate']:.3f}")
```

### Debugging Common Issues

**Issue 1: Clipping Speech Beginnings**

```python
# Solution: Increase pre-padding
vad_with_padding = VADWithPadding(
    vad=WebRTCVAD(),
    pre_pad_ms=300,  # Increase from 200ms
    post_pad_ms=500
)
```

**Issue 2: False Positives from Music**

```python
# Solution: Use ML VAD or add music classifier
class MusicFilteredVAD:
    """
    VAD with music filtering
    """
    
    def __init__(self, vad, music_classifier):
        self.vad = vad
        self.music_classifier = music_classifier
    
    def detect(self, audio):
        """Detect speech, filtering out music"""
        # Run VAD
        speech_frames = self.vad.detect(audio)
        
        # Filter music
        is_music = self.music_classifier.predict(audio)
        
        # Combine
        is_speech = speech_frames & (~is_music)
        
        return is_speech
```

**Issue 3: High CPU Usage**

```python
# Solution: Downsample audio or use simpler VAD
class DownsampledVAD:
    """
    VAD with audio downsampling for efficiency
    """
    
    def __init__(self, target_sr=8000):
        self.target_sr = target_sr
        self.vad = WebRTCVAD(sr=8000)  # 8kHz instead of 16kHz
    
    def detect(self, audio, original_sr=16000):
        """Detect with downsampling"""
        # Downsample
        import librosa
        audio_downsampled = librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=self.target_sr
        )
        
        # Run VAD on downsampled audio
        return self.vad.detect(audio_downsampled)
```

---

## Advanced Techniques

### Noise-Robust VAD

Use spectral subtraction for noise reduction before VAD.

```python
class NoiseRobustVAD:
    """
    VAD with noise reduction preprocessing
    """
    
    def __init__(self, vad):
        self.vad = vad
    
    def spectral_subtraction(self, audio, noise_profile):
        """
        Simple spectral subtraction
        
        Args:
            audio: Input audio
            noise_profile: Estimated noise spectrum
        
        Returns:
            Denoised audio
        """
        import librosa
        
        # STFT
        D = librosa.stft(audio)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Subtract noise
        magnitude_clean = np.maximum(magnitude - noise_profile, 0)
        
        # Reconstruct
        D_clean = magnitude_clean * np.exp(1j * phase)
        audio_clean = librosa.istft(D_clean)
        
        return audio_clean
    
    def detect_with_denoising(self, audio):
        """Detect speech after denoising"""
        # Estimate noise from first 0.5 seconds
        noise_segment = audio[:8000]  # 0.5s at 16kHz
        
        import librosa
        noise_spectrum = np.abs(librosa.stft(noise_segment))
        noise_profile = np.median(noise_spectrum, axis=1, keepdims=True)
        
        # Denoise
        audio_clean = self.spectral_subtraction(audio, noise_profile)
        
        # Run VAD on clean audio
        return self.vad.detect(audio_clean)
```

### Multi-Condition Training Data

For ML-based VAD, train on diverse conditions.

```python
class DataAugmentationForVAD:
    """
    Augment training data for robust VAD
    """
    
    def augment(self, clean_speech):
        """
        Create augmented samples
        
        Augmentations:
        - Add various noise types
        - Vary SNR levels
        - Apply room reverberation
        - Change speaker characteristics
        """
        augmented = []
        
        # 1. Add white noise
        noise = np.random.randn(len(clean_speech)) * 0.01
        augmented.append(clean_speech + noise)
        
        # 2. Add babble noise (simulated)
        # babble = load_babble_noise()
        # augmented.append(clean_speech + babble)
        
        # 3. Apply reverberation
        # reverb = apply_reverb(clean_speech)
        # augmented.append(reverb)
        
        return augmented
```

---

## Real-World Deployment Examples

### Zoom/Video Conferencing

**Requirements:**
- Ultra-low latency (< 10ms)
- Adaptive to varying network conditions
- Handle overlapping speech (multiple speakers)

**Solution:**
- WebRTC VAD for speed
- Adaptive aggressiveness based on network bandwidth
- Per-speaker VAD in multi-party calls

### Smart Speakers (Alexa, Google Home)

**Requirements:**
- Always-on (low power)
- Far-field audio (echoes, reverberation)
- Wake word detection + VAD

**Solution:**
- Two-stage: Wake word detector → VAD → ASR
- On-device VAD (WebRTC or lightweight ML)
- Cloud-based refinement for difficult cases

### Call Centers

**Requirements:**
- High accuracy (for analytics)
- Speaker diarization integration
- Post-processing acceptable

**Solution:**
- ML-based VAD with large models
- Two-pass processing
- Combined with speaker diarization

---

## Key Takeaways

✅ **Energy + ZCR** provides simple baseline VAD  
✅ **WebRTC VAD** is production-standard, fast, robust, widely deployed  
✅ **ML-based VAD** achieves best accuracy in noisy conditions  
✅ **Two-pass VAD** balances speed and accuracy for production  
✅ **Streaming processing** enables real-time applications  
✅ **Padding is critical** to avoid cutting off speech (200-500ms)  
✅ **Adaptive thresholds** handle varying noise levels  
✅ **Frame size tradeoff:** Smaller = lower latency, larger = better accuracy  
✅ **Quantization & optimization** essential for mobile/edge deployment  
✅ **Monitor precision/recall** in production to catch degradation  
✅ **Integration with ASR** requires careful endpoint detection logic  
✅ **Noise robustness** via preprocessing or multi-condition training

---

**Originally published at:** [arunbaby.com/speech-tech/0004-voice-activity-detection](https://www.arunbaby.com/speech-tech/0004-voice-activity-detection/)

*If you found this helpful, consider sharing it with others who might benefit.*

