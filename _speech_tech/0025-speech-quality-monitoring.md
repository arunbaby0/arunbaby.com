---
title: "Speech Quality Monitoring"
day: 25
collection: speech_tech
categories:
  - speech-tech
  - quality-assurance
  - metrics
  - mos
tags:
  - pesq
  - polqa
  - visqol
  - nisqa
  - audio-quality
subdomain: "Evaluation & Metrics"
tech_stack: [Python, Librosa, NISQA, ViSQOL]
scale: "Real-time quality estimation"
companies: [Zoom, Twilio, Discord, Spotify]
related_dsa_day: 25
related_ml_day: 25
related_speech_day: 25
---

**How do we know if the audio sounds "good" without asking a human?**

## The Problem: Subjectivity

In Image Classification, "Accuracy" is objective. Is it a cat? Yes/No.
In Speech, "Quality" is subjective.
- "The audio is intelligible but robotic."
- "The audio is natural but has background noise."
- "The audio cuts in and out (packet loss)."

For decades, the gold standard was **MOS (Mean Opinion Score)**.
1.  Gather 20 humans.
2.  Play them an audio clip.
3.  Ask them to rate it 1 (Bad) to 5 (Excellent).
4.  Average the scores.

**Problem:** This is slow, expensive, and impossible to run in real-time on a Zoom call. We need **Objective Metrics**.

## Intrusive vs. Non-Intrusive Metrics

### 1. Intrusive (Full-Reference)
You have the "Clean" (Reference) audio and the "Degraded" (Test) audio. You compare them.
- **Use Case:** Codec development (MP3 vs AAC), Denoising model training.
- **Metrics:**
    - **PESQ (Perceptual Evaluation of Speech Quality):** The classic telecom standard. Models human hearing.
    - **POLQA (Perceptual Objective Listening Quality Analysis):** The successor to PESQ. Handles wideband (HD) voice better.
    - **ViSQOL (Virtual Speech Quality Objective Listener):** Google's open-source metric. Uses similarity of spectrograms.

### 2. Non-Intrusive (No-Reference)
You *only* have the "Degraded" audio. You don't know what the original sounded like.
- **Use Case:** Real-time monitoring (Zoom, Discord). You receive audio from the internet; you don't have the sender's microphone feed.
- **Metrics:**
    - **P.563:** Old standard.
    - **NISQA (Non-Intrusive Speech Quality Assessment):** Deep Learning based.
    - **DNS-MOS:** Microsoft's Deep Noise Suppression MOS predictor.

## Deep Dive: ViSQOL (Google)

How does a machine "listen"?
ViSQOL aligns the reference and degraded signals in time, then compares their spectrograms using a **Structural Similarity Index (SSIM)**-like approach.

1.  **Spectrogram:** Convert audio to Time-Frequency domain.
2.  **Alignment:** Use dynamic time warping to align the two signals (handling delays/jitter).
3.  **Patch Comparison:** Compare small patches of the spectrograms.
4.  **Mapping:** Map the similarity score to a MOS (1-5).

## Deep Dive: NISQA (Deep Learning for Quality)

**NISQA** is a CNN-LSTM model trained to predict human MOS scores directly from raw audio.

**Architecture:**
- **Input:** Mel-Spectrogram.
- **CNN Layers:** Extract local features (noise, distortion).
- **Self-Attention / LSTM:** Capture temporal dependencies (dropouts, silence).
- **Output:** Predicted MOS (e.g., 3.8).

**Why is this revolutionary?**
It allows **Reference-Free** monitoring. You can run this on the client side (in the browser) to tell the user: "Your microphone quality is poor."

*If you found this helpful, consider sharing it with others who might benefit.*

<div style="opacity: 0.6; font-size: 0.8em; margin-top: 2em;">
  Created with LLM assistance
</div>

## High-Level Architecture: Real-Time Quality Monitor

```ascii
+-----------+     +------------+     +-------------+
| VoIP App  | --> | Edge Calc  | --> | Metrics Agg |
+-----------+     +------------+     +-------------+
(Microphone)      (DNS-MOS/VAD)      (Prometheus)
                                           |
                                           v
+-----------+     +------------+     +-------------+
| Codec Sw  | <-- | Alerting   | <-- | Dashboard   |
+-----------+     +------------+     +-------------+
(Opus Mode)       (Slack/PD)         (Grafana)
```

## System Design: Real-Time Quality Monitor for VoIP

**Scenario:** You are building the quality monitoring system for a Zoom competitor.

**1. Client-Side (Edge):**
- **Packet Loss Rate:** Simple counter.
- **Jitter:** Variance in packet arrival time.
- **Energy Level:** Is the user speaking? (VAD).
- **Lightweight ML:** Run a tiny TFLite model (DNS-MOS) every 10 seconds on a 1-second slice.

**2. Server-Side (Aggregator):**
- Ingest metrics into **Prometheus**.
- **Alerting:** If `Avg MOS < 3.0` for a specific region (e.g., "India-South"), trigger an alert. It might be a network outage.

**3. Feedback Loop:**
- If quality drops, the client automatically switches codecs (e.g., Opus 48kbps -> Opus 12kbps) or enables aggressive packet loss concealment (PLC).

## Deep Dive: The Math of PESQ

PESQ (ITU-T P.862) isn't just a simple subtraction. It simulates the human ear.

**Steps:**
1.  **Level Alignment:** Adjust volume so Reference and Degraded are equally loud.
2.  **Input Filter:** Simulate the frequency response of a telephone handset (300Hz - 3400Hz).
3.  **Auditory Transform:**
    - Convert FFT to **Bark Scale** (perceptual pitch).
    - Convert Amplitude to **Sone Scale** (perceptual loudness).
4.  **Disturbance Calculation:**
    - Subtract the two "Loudness Spectra".
    - `D = |L_ref - L_deg|`.
    - Apply asymmetric masking (we notice added noise more than missing sound).
5.  **Aggregation:** Average the disturbance over time and frequency to get a score ( -0.5 to 4.5).

## Engineering Component: Voice Activity Detection (VAD)

You can't measure quality if no one is talking. Silence is always "perfect quality".
We need a VAD to filter out silence frames.

**Simple Energy-Based VAD (Python):**

```python
import numpy as np

def simple_vad(audio, frame_len=160, threshold=0.01):
    # audio: numpy array of samples
    # frame_len: 10ms at 16kHz
    
    frames = []
    for i in range(0, len(audio), frame_len):
        frame = audio[i:i+frame_len]
        energy = np.sum(frame ** 2) / len(frame)
        
        if energy > threshold:
            frames.append(frame)
            
    return np.concatenate(frames) if frames else np.array([])
```

**Advanced VAD:** WebRTC VAD uses a Gaussian Mixture Model (GMM) to distinguish speech from noise.

## Network Engineering: WebRTC Internals

How does Zoom know your quality is bad? **RTCP (Real-time Transport Control Protocol).**

Every few seconds, the receiver sends an **RTCP Receiver Report (RR)** back to the sender.
**Fields:**
1.  **Fraction Lost:** Percentage of packets lost since last report.
2.  **Cumulative Lost:** Total packets lost.
3.  **Interarrival Jitter:** Variance in packet delay.

**The Quality Estimator:**
`MOS_est = 4.5 - PacketLossPenalty - JitterPenalty - LatencyPenalty`
This is the **E-Model (ITU-T G.107)**. It's a heuristic formula, not a neural net.

## Audio Processing: Packet Loss Concealment (PLC)

When a packet is lost, you have a 20ms gap in audio.
**Option 1: Silence.** (Sounds like a click/pop). Bad.
**Option 2: Repeat.** (Repeat last 20ms). Robotic.
**Option 3: Waveform Similarity Overlap-Add (WSOLA).** Stretch the previous packet to cover the gap.
**Option 4: Deep PLC (Packet Loss Concealment).**
- Use a Generative Model (GAN/RNN) to *predict* what the missing packet should have been based on context.
- **NetEQ (WebRTC):** Uses a jitter buffer and DSP-based PLC to smooth out network bumps.

## Deep Dive: DNS-MOS (Microsoft)

Microsoft's **Deep Noise Suppression (DNS)** dataset is huge. They trained a metric to evaluate it.
**Architecture:**
- **Input:** Log-Mel Spectrogram.
- **Backbone:** ResNet-18 or Polyphonic Inception.
- **Heads:**
    1.  **SIG:** Signal Quality (How natural is the speech?).
    2.  **BAK:** Background Quality (How intrusive is the noise?).
    3.  **OVRL:** Overall Quality.
- **Training:** Trained on P.808 crowdsourced data (humans rating noisy clips).

**Why separate SIG and BAK?**
Sometimes a denoiser removes noise (Good BAK) but makes the voice sound muffled (Bad SIG). We need to balance both.

## Deep Dive: Psychoacoustics

Why do we need complex metrics like PESQ? Why not just use MSE (Mean Squared Error)?
**MSE is terrible for audio.**
- **Phase Shift:** If you shift a signal by 1ms, MSE is huge, but it sounds identical.
- **Masking:** If a loud sound plays at 1000Hz, you can't hear a quiet sound at 1100Hz.
- **Equal Loudness:** A 50Hz tone needs to be much louder than a 1000Hz tone to be heard (Fletcher-Munson curves).

**Perceptual Loss Functions:**
In Deep Learning (e.g., Speech Enhancement), we minimize a "Perceptual Loss".
`Loss = |VGG(Ref) - VGG(Deg)|`
We pass audio through a pre-trained network (like VGGish or wav2vec) and compare the *activations*, not the raw pixels/samples.

## Advanced Metric: STOI (Short-Time Objective Intelligibility)

PESQ measures "Quality" (How pleasant?).
STOI measures "Intelligibility" (Can you understand the words?).

**Use Case:** Cochlear Implants, Hearing Aids.
A signal can be ugly (robotic) but perfectly intelligible (High STOI, Low PESQ).
A signal can be beautiful but mumbled (Low STOI, High PESQ).

**Algorithm:**
1.  Decompose signal into TF-units (Time-Frequency).
2.  Calculate correlation between Reference and Degraded envelopes in each band.
3.  Average the correlations.

## System Design: The "Netflix for Audio" Quality Pipeline

**Scenario:** You are building Spotify's ingestion pipeline.
**Goal:** Reject tracks with bad encoding artifacts.

**Pipeline:**
1.  **Ingest:** Upload WAV/FLAC.
2.  **Transcode:** Convert to Ogg Vorbis (320kbps, 160kbps, 96kbps).
3.  **Quality Check (ViSQOL):** Compare Transcoded vs. Original.
    - If `ViSQOL < 4.5` for 320kbps, something is wrong with the encoder.
4.  **Loudness Normalization (LUFS):**
    - Measure Integrated Loudness (EBU R128).
    - If track is too quiet (-20 LUFS), gain up.
    - If track is too loud (-5 LUFS), gain down to target (-14 LUFS).

## Codec Comparison: Opus vs. AAC vs. EVS

| Feature | Opus | AAC-LD | EVS (5G) |
| :--- | :--- | :--- | :--- |
| **Latency** | Ultra Low (5ms) | Low (20ms) | Low (20ms) |
| **Bitrate** | 6kbps - 510kbps | 32kbps+ | 5.9kbps - 128kbps |
| **Quality** | Excellent | Good | Excellent |
| **Packet Loss** | Built-in FEC | Poor | Channel Aware |
| **Use Case** | Zoom, Discord | FaceTime | VoLTE, 5G Calls |

**Why Opus wins:** It switches modes (SILK for speech, CELT for music) dynamically.

## Appendix C: Subjective Testing (MUSHRA)

When MOS isn't enough, we use **MUSHRA (Multiple Stimuli with Hidden Reference and Anchor)**.
1.  **Hidden Reference:** The original audio (should be rated 100).
2.  **Anchor:** A low-pass filtered version (should be rated 20).
3.  **Test Systems:** The models we are testing.

**Why?** It calibrates the listeners. If a listener rates the Anchor as 80, we disqualify them.

## Deep Dive: Room Acoustics and RT60

Quality isn't just about the codec; it's about the **Room**.
**RT60 (Reverberation Time):** Time it takes for sound to decay by 60dB.
- **Studio:** 0.3s (Dry).
- **Living Room:** 0.5s.
- **Cathedral:** 4.0s (Wet).

**Impact on ASR:**
High RT60 smears the spectrogram. ASR models fail.
**Solution:** Dereverberation (WPE - Weighted Prediction Error).

## Hardware Engineering: Microphone Arrays

How does Alexa hear you from across the room? **Beamforming.**
Using multiple microphones, we can steer the "listening beam" towards the speaker and nullify noise from the TV.

**Metrics:**
1.  **Directivity Index (DI):** Gain in the look direction vs. average gain.
2.  **White Noise Gain (WNG):** Robustness to sensor noise.

**MVDR Beamformer (Minimum Variance Distortionless Response):**
Mathematically minimizes output power while maintaining unity gain in the target direction.

## Advanced Topic: Spatial Audio Quality

With VR/AR (Apple Vision Pro), audio is 3D.
**HRTF (Head-Related Transfer Function):** How your ears/head filter sound based on direction.

**Quality Metrics for Spatial Audio:**
1.  **Localization Accuracy:** Can the user pinpoint the source?
2.  **Timbral Coloration:** Does the HRTF distort the tone?
3.  **Externalization:** Does it sound like it's "out there" or "in your head"?

## Security: Audio Watermarking and Deepfake Detection

**The Threat:** AI Voice Cloning (ElevenLabs).
**The Defense:** Watermarking.
Embed an inaudible signal (spread spectrum) into the audio.

**Detection:**
1.  **Artifact Analysis:** GANs leave traces in the high frequencies.
2.  **Phase Continuity:** Natural speech has specific phase relationships. Vocoders often break them.
3.  **Biometrics:** Verify the "Voice Print" against a known enrollment.

## Accessibility: Hearing Loss Simulation

To ensure quality for *everyone*, we must simulate hearing loss.
**Presbycusis:** Age-related high-frequency loss.
**Recruitment:** Loud sounds become painful quickly.

**Testing:**
Run the audio through a "Hearing Loss Simulator" (Low-pass filter + Dynamic Range Compression) and run STOI.
If STOI drops too much, the content is not accessible.

## Deep Dive: Audio Codec Internals (MDCT)

How does MP3/Opus actually compress audio?
**MDCT (Modified Discrete Cosine Transform).**
It's like FFT, but with overlapping windows (50% overlap) to prevent "blocking artifacts".

**Process:**
1.  **Windowing:** Multiply signal by a window function (Sine/Kaiser).
2.  **MDCT:** Convert to frequency domain.
3.  **Quantization:** Round the float values to integers. **This is where loss happens.**
4.  **Entropy Coding:** Huffman coding to compress the integers.

**Psychoacoustic Model:**
The encoder calculates the **Masking Threshold** for each frequency band.
If the quantization noise is *below* the masking threshold, the human ear can't hear it.
So, we can quantize heavily (low bitrate) without perceived loss.

## Deep Dive: Opus Internals (SILK + CELT)

Opus is the king of VoIP. Why? It's a hybrid.

**1. SILK (Skype):**
- **Type:** LPC (Linear Predictive Coding).
- **Best for:** Speech (Low frequencies).
- **Mechanism:** Models the vocal tract as a tube. Transmits the "excitation" (glottis) and "filter" (throat/mouth).

**2. CELT (Xiph.org):**
- **Type:** MDCT (Transform Coding).
- **Best for:** Music (High frequencies).
- **Mechanism:** Transmits the spectral energy.

**Hybrid Mode:**
Opus sends Low Frequencies (< 8kHz) via SILK and High Frequencies (> 8kHz) via CELT. Best of both worlds.

## Network Engineering: Congestion Control (GCC vs. BBR)

If the network is congested, sending *more* data makes it worse.
We need to lower the bitrate.

**Google Congestion Control (GCC):**
- **Delay-based:** If RTT increases, reduce bitrate.
- **Loss-based:** If packet loss > 2%, reduce bitrate.
- **Kalman Filter:** Predicts the network capacity.

**BBR (Bottleneck Bandwidth and Round-trip propagation time):**
- Probes the network to find the max bandwidth and min RTT.
- Much more aggressive than GCC. Used in QUIC.

## Hardware: MEMS vs. Condenser Microphones

Quality starts at the sensor.
**MEMS (Micro-Electro-Mechanical Systems):**
- **Pros:** Tiny, cheap, solderable (SMD). Used in phones/laptops.
- **Cons:** High noise floor (SNR ~65dB).

**Condenser (Electret):**
- **Pros:** High sensitivity, low noise (SNR > 70dB). Studio quality.
- **Cons:** Large, requires phantom power (48V).

**Clipping:**
If the user screams, the mic diaphragm hits the limit. The signal is "clipped" (flat top).
**Digital Clipping:** Exceeding 0dBFS.
**Solution:** Analog Gain Control (AGC) *before* the ADC.

## Appendix E: The Future of Quality

**Neural Codecs (EnCodec/SoundStream):**
They don't optimize SNR. They optimize Perceptual Quality.
At 3kbps, they sound better than Opus at 12kbps, but the waveform looks *completely different*.
**Implication:** Traditional metrics (SNR, PSNR) are dead. We *must* use Neural Metrics (NISQA, CDPAM).

## Appendix F: Interview Questions

1.  **Q:** "How do you handle packet loss in a real-time audio app?"
    **A:**
    - **Jitter Buffer:** Hold packets for 20-50ms to reorder them.
    - **FEC (Forward Error Correction):** Send redundant data (XOR of previous packets).
    - **PLC (Packet Loss Concealment):** Generate fake audio to fill gaps.

2.  **Q:** "Why is 44.1kHz the standard sample rate?"
    **A:** Nyquist Theorem. Humans hear up to 20kHz. We need `2 * MaxFreq` to reconstruct the signal. `2 * 20kHz = 40kHz`. The extra 4.1kHz is for anti-aliasing filter roll-off.

3.  **Q:** "What is the Cocktail Party Problem?"
    **A:** The ability to focus on one speaker in a noisy room. Humans do it easily (binaural hearing). Machines struggle. Solved using **Blind Source Separation (BSS)** or **Target Speech Extraction (TSE)**.

## Conclusion

Speech Quality Monitoring is moving from "Signal Processing" (PESQ) to "Deep Learning" (NISQA).
Just like in Vision (Perceptual Loss) and NLP (BERTScore), we are learning that **Neural Networks are the best judges of Neural Networks**.


```python
from pesq import pesq
from scipy.io import wavfile

# Load audio (must be 8k or 16k for PESQ)
rate, ref = wavfile.read("reference.wav")
rate, deg = wavfile.read("degraded.wav")

# Calculate PESQ (Wideband)
score = pesq(rate, ref, deg, 'wb')
print(f"PESQ Score: {score:.2f}")
# Output: 3.5 (Fair to Good)
```

## Appendix A: The "Silent" Failure in Speech

In ASR (Speech-to-Text), a common failure is **Hallucination**.
- **Silence** -> Model outputs "Thank you very much."
- **Noise** -> Model outputs "I will kill you." (This actually happens!).

**Quality Monitoring for ASR:**
- **Log-Likelihood:** If the model is confident.
- **Speech-to-Noise Ratio (SNR):** If SNR is too low, don't transcribe.
- **VAD (Voice Activity Detection):** Only send audio that contains speech.

## Conclusion

Speech Quality Monitoring is moving from "Signal Processing" (PESQ) to "Deep Learning" (NISQA).
Just like in Vision (Perceptual Loss) and NLP (BERTScore), we are learning that **Neural Networks are the best judges of Neural Networks**.
