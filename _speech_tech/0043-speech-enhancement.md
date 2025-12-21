---
title: "Speech Enhancement"
day: 43
collection: speech_tech
categories:
  - speech-tech
tags:
  - denoising
  - deep-learning
  - audio-processing
  - signal-processing
difficulty: Hard
---

**"Extracting clear speech from the noise of the real world."**

## 1. Introduction

**Speech Enhancement** is the task of improving the quality and intelligibility of speech signals degraded by noise, reverberation, or other distortions.

**Applications:**
*   **Voice Assistants:** Improve ASR accuracy in noisy environments.
*   **Hearing Aids:** Help hearing-impaired users understand speech.
*   **Video Conferencing:** Remove background noise (Zoom, Teams).
*   **Telecommunications:** Improve call quality.
*   **Forensics:** Enhance speech in recordings.

## 2. Types of Degradation

### 2.1. Additive Noise

**Model:** $y(t) = x(t) + n(t)$
*   $x(t)$: Clean speech.
*   $n(t)$: Noise (fan, traffic, babble).
*   $y(t)$: Noisy speech.

**Noise Types:**
*   **Stationary:** Constant spectrum (fan, AC).
*   **Non-Stationary:** Changing spectrum (babble, music).

### 2.2. Reverberation

**Model:** $y(t) = x(t) * h(t)$
*   $h(t)$: Room impulse response (RIR).
*   Convolution spreads energy over time.

**Effects:**
*   **Early Reflections:** Slight echoes (helpful for perception).
*   **Late Reverberation:** Smearing, reduced intelligibility.

### 2.3. Clipping & Distortion

**Cause:** Microphone saturation, codec artifacts.
**Effect:** Waveform is "cut off" at peaks.

## 3. Classic Signal Processing Approaches

### 3.1. Spectral Subtraction

**Idea:** Estimate noise spectrum, subtract from noisy spectrum.

**Algorithm:**
1.  Estimate noise spectrum $\hat{N}(f)$ from silence regions.
2.  Subtract: $\hat{X}(f) = Y(f) - \alpha \hat{N}(f)$.
3.  Apply flooring to avoid negative values.

**Problems:**
*   **Musical Noise:** Residual tones from random noise estimation errors.
*   **Non-Stationary Noise:** Fails when noise changes rapidly.

### 3.2. Wiener Filtering

**Idea:** Optimal linear filter to minimize MSE between estimated and clean speech.

**Formula:**
$$H(f) = \frac{|X(f)|^2}{|X(f)|^2 + |N(f)|^2} = \frac{\text{SNR}(f)}{\text{SNR}(f) + 1}$$

**Interpretation:**
*   High SNR: $H(f) \approx 1$ (pass signal).
*   Low SNR: $H(f) \approx 0$ (suppress).

### 3.3. Noise Estimation

**VAD-Based:**
*   Detect silence (Voice Activity Detection).
*   Update noise estimate during silence.

**MMSE-Based:**
*   Minimum Mean Square Error estimator.
*   Assumes noise is a random variable.

## 4. Deep Learning Approaches

### 4.1. Masking-Based Methods

**Idea:** Learn a mask $M(t, f)$ to apply to the noisy spectrogram.

$$\hat{X}(t, f) = M(t, f) \cdot Y(t, f)$$

**Mask Types:**
*   **Ideal Binary Mask (IBM):** $M = 1$ if SNR > threshold, else $M = 0$.
*   **Ideal Ratio Mask (IRM):** $M = \frac{|X|^2}{|X|^2 + |N|^2}$.
*   **Complex Ideal Ratio Mask (cIRM):** Operates on complex STFT.

### 4.2. Mapping-Based Methods

**Idea:** Directly map noisy spectrogram to clean spectrogram.

$$\hat{X}(t, f) = f_\theta(Y(t, f))$$

**Model:** CNN, LSTM, or U-Net.

### 4.3. Waveform-Based Methods

**Idea:** Process raw waveform directly (no STFT).

**Models:**
*   **WaveNet:** Dilated convolutions.
*   **Conv-TasNet:** Learned encoder-decoder.
*   **DEMUCS:** U-Net on waveform.

**Pros:** No phase estimation needed.
**Cons:** Computationally expensive.

## 5. Architectures for Speech Enhancement

### 5.1. U-Net

**Architecture:**
*   Encoder: Downsampling convolutions.
*   Decoder: Upsampling convolutions.
*   Skip Connections: Connect encoder to decoder.

**Input:** Noisy spectrogram (magnitude).
**Output:** Enhanced spectrogram (or mask).

```python
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # Decoder
        self.dec3 = self.upconv_block(256, 128)
        self.dec2 = self.upconv_block(256, 64)  # 256 because of skip connection
        self.dec1 = self.upconv_block(128, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        d3 = self.dec3(e3)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return d1
```

### 5.2. Conv-TasNet

**Architecture (Time-domain):**
1.  **Encoder:** 1D convolution to learned representation.
2.  **Separator:** Temporal Convolutional Network (TCN) to estimate mask.
3.  **Decoder:** Transposed convolution to reconstruct waveform.

**Pros:** State-of-the-art for speech separation.
**Cons:** High memory for long audio.

### 5.3. DCCRN (Deep Complex CNN)

**Key Feature:** Operates on complex STFT (real + imaginary).

**Benefit:** Better phase estimation than magnitude-only methods.

## 6. Loss Functions

### 6.1. Mean Squared Error (MSE)

$$L = \frac{1}{T \cdot F} \sum_{t, f} (\hat{X}(t, f) - X(t, f))^2$$

**Pros:** Simple, differentiable.
**Cons:** Doesn't correlate well with perceptual quality.

### 6.2. Scale-Invariant SDR (SI-SDR)

$$\text{SI-SDR} = 10 \log_{10} \frac{||\alpha x||^2}{||\hat{x} - \alpha x||^2}$$

Where $\alpha = \frac{\langle \hat{x}, x \rangle}{||x||^2}$.

**Interpretation:** Higher is better. Measures signal-to-distortion ratio.

### 6.3. Perceptual Loss

**PESQ (Perceptual Evaluation of Speech Quality):**
*   Intrusive metric (requires clean reference).
*   Scores from 1.0 (bad) to 4.5 (excellent).

**STOI (Short-Time Objective Intelligibility):**
*   Correlates with human intelligibility.
*   Range: 0.0 to 1.0.

**Differentiable Approximations:**
*   Train a neural network to approximate PESQ/STOI.
*   Use as a loss function.

## 7. System Design: Real-Time Denoising

**Scenario:** Build a noise suppression module for video conferencing.

**Requirements:**
*   **Latency:** < 20ms.
*   **CPU/GPU:** Must run on laptop CPUs.
*   **Quality:** Preserve speech, remove noise.

**Architecture:**

**Step 1: Frame Processing**
*   Audio arrives in 20ms frames.
*   STFT with 20ms window, 10ms hop.

**Step 2: Neural Network**
*   Lightweight CNN (e.g., 10 layers).
*   Quantized to INT8 for CPU inference.

**Step 3: Apply Mask**
*   Multiply noisy STFT by predicted mask.
*   Inverse STFT to reconstruct waveform.

**Step 4: Overlap-Add**
*   Combine overlapping frames smoothly.

**Step 5: Output**
*   Send enhanced audio to speaker.

## 8. Production Case Study: Zoom Noise Cancellation

**Model:** RNNoise-inspired, enhanced with CNN.

**Features:**
*   **18x real-time:** Processes audio 18x faster than it plays.
*   **CPU-only:** Runs on low-end laptops.
*   **Adaptive:** Learns user's environment over time.

**Training Data:**
*   **Clean:** LibriSpeech, VCTK.
*   **Noise:** AudioSet, FreeSound.
*   **Augmentation:** Mix at various SNRs, add reverberation.

## 9. Production Case Study: Apple AirPods Pro

**Features:**
*   **Active Noise Cancellation (ANC):** Hardware + DSP.
*   **Transparency Mode:** Pass through environment.
*   **Adaptive EQ:** Adjust sound based on ear fit.

**Enhancement:**
*   **Microphones:** 2 external, 1 internal.
*   **Processing:** On-device neural network.
*   **Integration:** Optimized for Siri voice input.

## 10. Datasets

**1. VCTK:**
*   109 speakers, clean speech.
*   Add noise synthetically.

**2. DNS Challenge (Microsoft):**
*   Large-scale, diverse noise.
*   Training and evaluation sets.

**3. CHiME:**
*   Real-world noisy recordings.
*   Multiple noise conditions.

**4. LibriMix:**
*   Mixed speech for separation.
*   Derived from LibriSpeech.

## 11. Evaluation Metrics

**Objective:**
*   **PESQ:** Perceptual quality (1.0-4.5).
*   **STOI:** Intelligibility (0.0-1.0).
*   **SI-SDR:** Signal-to-distortion ratio (dB).
*   **POLQA:** Next-gen PESQ.

**Subjective:**
*   **MOS (Mean Opinion Score):** Human ratings (1-5).
*   **ABX Test:** Which sample sounds better?

## 12. Interview Questions

1.  **Spectral Subtraction:** How does it work? What are its limitations?
2.  **Wiener Filter:** Derive the optimal filter.
3.  **Masking vs Mapping:** What's the difference?
4.  **Real-Time Constraints:** How do you achieve <20ms latency?
5.  **Evaluation:** Explain PESQ and STOI.
6.  **Design:** Design a noise cancellation system for hearing aids.

## 13. Common Mistakes

*   **Ignoring Phase:** Magnitude-only methods produce artifacts.
*   **Training/Test Mismatch:** Training on synthetic noise, testing on real.
*   **Overlooking Latency:** Model too large for real-time.
*   **Suppressing Speech:** Over-aggressive noise removal.
*   **Ignoring Reverberation:** Many systems only handle additive noise.

## 14. Deep Dive: Generative Approaches

### 14.1. Diffusion Models for Speech Enhancement

**Idea:** Learn to reverse the noising process.

**Algorithm:**
1.  **Forward:** Add Gaussian noise to clean speech.
2.  **Reverse:** Train model to predict clean speech from noisy.
3.  **Inference:** Start with noisy speech, iteratively denoise.

**Pros:** High-quality, handles complex degradations.
**Cons:** Slow (many diffusion steps).

### 14.2. GAN-Based Enhancement

**Architecture:**
*   **Generator:** U-Net that enhances speech.
*   **Discriminator:** Classifies real vs enhanced.

**Loss:**
*   Adversarial loss + MSE/SI-SDR.
*   Perceptual loss (from pretrained network).

**Pros:** Sharper, more natural outputs.
**Cons:** Training instability.

## 15. Future Trends

**1. Self-Supervised Learning:**
*   Pretrain on large unlabeled audio.
*   Fine-tune for enhancement.

**2. Multi-Task Learning:**
*   Joint enhancement + ASR.
*   Joint enhancement + diarization.

**3. On-Device Enhancement:**
*   Run on smartphones, earbuds.
*   Neural Processing Units (NPUs).

**4. Personalized Enhancement:**
*   Adapt to user's voice and environment.
*   Few-shot learning.

## 16. Conclusion

Speech enhancement is critical for making AI systems work in the real world. Whether it's helping Siri understand you in a noisy café or enabling clear video calls, enhancement is the first line of defense against acoustic degradation.

**Key Takeaways:**
*   **Classic Methods:** Spectral subtraction, Wiener filtering.
*   **Deep Learning:** Masking (U-Net), waveform (Conv-TasNet).
*   **Metrics:** PESQ, STOI, SI-SDR.
*   **Production:** Latency, CPU efficiency, generalization.
*   **Future:** Diffusion models, on-device processing.

Mastering speech enhancement enables you to build robust speech systems that work in any environment.

## 17. Deep Dive: RNNoise

**RNNoise** is a lightweight, real-time noise suppression algorithm.

**Architecture:**
*   **Input:** 22 features (pitch, spectral bands).
*   **Model:** GRU with 96 units.
*   **Output:** Gains per frequency band.

**Key Innovations:**
*   **Handcrafted Features:** Instead of spectrogram, use pitch, spectral derivative.
*   **Pitch Filtering:** Use pitch information to enhance periodic speech.
*   **Tiny Model:** <100KB, runs on embedded devices.

**Performance:**
*   **18x Real-Time:** On single CPU core.
*   **Quality:** Comparable to larger neural networks.

**Code (C with SIMD):**
```c
// RNNoise inference loop
for (int i = 0; i < frame_size; i++) {
    // Extract features
    float features[22] = compute_features(frame[i]);
    
    // GRU inference
    float gains[22] = gru_forward(features);
    
    // Apply gains to frequency bands
    apply_gains(frame[i], gains);
}
```

## 18. Deep Dive: Dereverberation

**Problem:** Remove room reflections from speech.

**Approaches:**

**1. Weighted Prediction Error (WPE):**
*   Model late reverberation as autoregressive process.
*   Predict reverberant tail, subtract.

**2. Neural Dereverberation:**
*   Train on pairs (reverberant, clean).
*   Similar architecture to denoising.

**3. Beamforming:**
*   Use microphone array to focus on direct sound.
*   Suppress reflections from other directions.

**Metric:** Speech-to-Reverberation Ratio (SRR).

## 19. Multi-Channel Speech Enhancement

**Scenario:** Multiple microphones (phone with 2 mics, smart speaker with 6 mics).

**Algorithm Pipeline:**
1.  **Beamforming:** Combine channels to enhance direction of interest.
2.  **Post-Filter:** Apply single-channel enhancement to beamformed signal.

**Beamforming Types:**
*   **Delay-and-Sum:** Simple, delays based on geometry.
*   **MVDR (Minimum Variance Distortionless Response):** Optimal, requires covariance estimation.
*   **Neural Beamformer:** Learn beamforming weights with neural network.

**Example (MVDR):**
```python
def mvdr_beamformer(stft, steering_vector, noise_covariance):
    # stft: (channels, time, freq)
    # steering_vector: (channels, freq)
    # noise_covariance: (freq, channels, channels)
    
    output = np.zeros((stft.shape[1], stft.shape[2]), dtype=complex)
    
    for f in range(stft.shape[2]):
        Rn_inv = np.linalg.inv(noise_covariance[f])
        d = steering_vector[:, f]
        
        # MVDR weights
        w = Rn_inv @ d / (d.conj().T @ Rn_inv @ d)
        
        # Apply to all time frames
        output[:, f] = w.conj().T @ stft[:, :, f]
    
    return output
```

## 20. Implementation: Real-Time Enhancement Pipeline

**Step-by-Step:**
```python
import numpy as np
from scipy.io import wavfile
import torch

# 1. Load model
model = load_enhancement_model('unet_enhancement.pt')
model.eval()

# 2. Audio parameters
FRAME_SIZE = 512
HOP_SIZE = 256
SAMPLE_RATE = 16000

# 3. Processing loop
def enhance_audio(input_wav, output_wav):
    sr, audio = wavfile.read(input_wav)
    audio = audio.astype(np.float32) / 32768
    
    # STFT
    stft = librosa.stft(audio, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Enhance with model
    with torch.no_grad():
        mag_input = torch.tensor(magnitude).unsqueeze(0).unsqueeze(0)
        mask = model(mag_input).squeeze().numpy()
    
    # Apply mask
    enhanced_magnitude = magnitude * mask
    
    # Inverse STFT
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=HOP_SIZE)
    
    # Save
    wavfile.write(output_wav, sr, (enhanced_audio * 32768).astype(np.int16))
```

## 21. Training a Speech Enhancement Model

**Step 1: Data Preparation**
```python
# Mix clean speech with noise at random SNR
def create_noisy_mixture(clean, noise, snr_db):
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate required noise scale
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = np.sqrt(clean_power / (snr_linear * noise_power))
    
    noisy = clean + noise_scale * noise
    return noisy, clean
```

**Step 2: Define Model (U-Net)**
```python
class EnhancementUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ... more layers
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            # ... more layers
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()  # Output mask in [0, 1]
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        mask = self.decoder(enc)
        return mask
```

**Step 3: Training Loop**
```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    for noisy_batch, clean_batch in dataloader:
        noisy_mag = stft(noisy_batch)
        clean_mag = stft(clean_batch)
        
        # Target mask (IRM)
        target_mask = clean_mag ** 2 / (clean_mag ** 2 + noise_mag ** 2 + 1e-8)
        
        # Forward
        pred_mask = model(noisy_mag)
        
        # Loss
        loss = criterion(pred_mask, target_mask)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 22. Handling Difficult Noise Types

**Music:**
*   Challenge: Music has similar spectral structure to speech.
*   Solution: Train with music as a noise type.

**Babble:**
*   Challenge: Multiple speakers overlap with target.
*   Solution: Speaker separation before enhancement.

**Impulsive Noise (clicks, pops):**
*   Challenge: Short bursts, hard to estimate.
*   Solution: Median filtering + neural enhancement.

**Wind:**
*   Challenge: Low-frequency, fluctuating.
*   Solution: High-pass filter + neural enhancement.

## 23. Integration with ASR

**Pre-Enhancement:**
*   Enhance audio before feeding to ASR.
*   Improves WER in noisy conditions.

**Joint Training:**
*   Train enhancement + ASR end-to-end.
*   Optimize directly for recognition, not perceptual quality.

**Example (Joint Pipeline):**
```
Audio → Enhancement → ASR → Text
         ↑             ↓
       Joint Loss (WER + Perceptual)
```

## 24. Latency Analysis

**Pipeline Latency:**
*   **Frame Size:** 20ms (typical).
*   **STFT:** 10ms (computation).
*   **Neural Network:** 5-20ms (depends on model size).
*   **Inverse STFT:** 5ms.
*   **Total:** 40-55ms (not including buffer delays).

**Reducing Latency:**
*   Smaller models (quantized, pruned).
*   Smaller frame sizes (10ms).
*   GPU/NPU acceleration.

## 25. Deployment Considerations

**Mobile (iOS/Android):**
*   Use TensorFlow Lite or Core ML.
*   Quantize to INT8.
*   Target: <10ms per frame.

**Embedded (Raspberry Pi, STM32):**
*   Use C/C++ with SIMD.
*   Very small model (<100KB).
*   Target: <5ms per frame.

**Cloud:**
*   Batch processing for efficiency.
*   GPU for high-throughput.

## 26. Mastery Checklist

**Mastery Checklist:**
- [ ] Explain spectral subtraction and Wiener filtering
- [ ] Implement a U-Net for speech enhancement
- [ ] Train on noisy/clean pairs
- [ ] Evaluate with PESQ and STOI
- [ ] Implement real-time processing (<20ms latency)
- [ ] Understand CTC/RNN-T integration for ASR
- [ ] Handle different noise types
- [ ] Deploy on mobile (TFLite/Core ML)
- [ ] Implement multi-channel enhancement
- [ ] Understand diffusion-based enhancement

## 27. Conclusion

Speech enhancement is the unsung hero of speech technology. Without it, voice assistants wouldn't work in noisy environments, video calls would be unusable, and hearing aids would be ineffective.

**Key Takeaways:**
*   **Classic Methods:** Spectral subtraction, Wiener filter—foundation of understanding.
*   **Deep Learning:** Masking and mapping with CNNs, U-Nets, and waveform models.
*   **Production:** Real-time constraints, CPU efficiency, generalization to unseen noise.
*   **Metrics:** PESQ (quality), STOI (intelligibility), SI-SDR (distortion).
*   **Multi-Channel:** Beamforming + post-filtering for best results.

The future is on-device, personalized, and multi-modal. As edge AI becomes more powerful, speech enhancement will happen entirely on your device, preserving privacy while delivering crystal-clear audio. Mastering these techniques is essential for any speech engineer.



---

**Originally published at:** [arunbaby.com/speech-tech/0043-speech-enhancement](https://www.arunbaby.com/speech-tech/0043-speech-enhancement/)

*If you found this helpful, consider sharing it with others who might benefit.*

