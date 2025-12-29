---
title: "Wake Word Detection"
day: 40
related_dsa_day: 40
related_ml_day: 40
related_agents_day: 40
collection: speech_tech
categories:
 - speech-tech
tags:
 - keyword-spotting
 - edge-ai
 - audio-processing
 - deep-learning
difficulty: Medium
---

**"Hey Siri, Alexa, OK Google: The gateway to voice AI."**

## 1. Introduction

**Wake Word Detection** (or Keyword Spotting - KWS) is the task of detecting a specific phrase in a continuous stream of audio to "wake up" a larger system.
It is the "Always-On" component of voice assistants.

**Constraints:**
1. **Low Power:** Must run 24/7 on battery-powered devices (DSP/MCU).
2. **Low Latency:** Detection must happen instantly.
3. **High Accuracy:**
 * **False Rejection Rate (FRR):** Ignoring the user (Annoying).
 * **False Alarm Rate (FAR):** Waking up randomly (Privacy nightmare).
4. **Small Footprint:** Few KB/MB of memory.

## 2. Anatomy of a KWS System

1. **Feature Extraction:** Convert audio to MFCCs or Log-Mel Spectrograms.
2. **Neural Network:** A small, efficient model classifies the frame sequence.
3. **Posterior Handling:** Smooth the outputs (e.g., moving average).
4. **Decoder:** Trigger if confidence > threshold for `N` frames.

## 3. Model Architectures

### 3.1. DNN / MLP
* **Input:** Flattened context window of frames (e.g., 40 frames left, 10 right).
* **Pros:** Simple, fast.
* **Cons:** Ignores temporal structure, fixed context.

### 3.2. CNN (Convolutional Neural Networks)
* Treat the spectrogram as an image.
* **ResNet-15 / ResNet-8:** Standard image architectures scaled down.
* **TC-ResNet (Temporal Convolution):** Uses 1D convolutions along the time axis. Very efficient.
* **Pros:** Translation invariance (shift in time/frequency), parameter efficient.

### 3.3. CRNN (Convolutional Recurrent Neural Networks)
* **CNN** layers extract local features.
* **RNN/GRU/LSTM** layers capture long-term temporal dependencies.
* **Pros:** Good for longer keywords.
* **Cons:** RNNs are harder to parallelize/quantize than CNNs.

### 3.4. DS-CNN (Depthwise Separable CNN)
* Inspired by MobileNet.
* Separates spatial (frequency) and temporal convolutions.
* Drastically reduces parameters and multiply-accumulates (MACs).
* **State-of-the-art for microcontrollers.**

### 3.5. Conformers & Transformers
* **KWS-Transformer:** Using self-attention for global context.
* **Pros:** High accuracy.
* **Cons:** Heavy computation (O(T^2) attention), usually too heavy for always-on DSPs, but okay for "second stage" verification on the AP (Application Processor).

## 4. Loss Functions

Standard Cross-Entropy is often not enough because "Silence/Background" class dominates the data (imbalanced dataset).

### 4.1. Max-Margin Loss
Encourage the model to maximize the margin between the target keyword score and the runner-up.

### 4.2. Triplet Loss
Learn an embedding space where "Alexa" samples are close together, and "Alexa" vs "Background" are far apart.

## 5. System Design: The Cascaded Architecture

To balance power and accuracy, we use a multi-stage approach.

**Stage 1: The Hardware Gate (DSP)**
* **Model:** Tiny DS-CNN or DNN (e.g., 50KB).
* **Power:** < 1 mW.
* **Goal:** High Recall (Don't miss), Low Precision (Okay to false trigger).
* **Action:** If triggered, wake up the main processor (AP).

**Stage 2: The Software Verification (AP)**
* **Model:** Larger CNN or Conformer (e.g., 5MB).
* **Power:** ~100 mW (runs only when Stage 1 triggers).
* **Goal:** High Precision (Filter out false alarms).
* **Action:** If verified, stream audio to the Cloud.

**Stage 3: Cloud Verification (Optional)**
* **Model:** Massive ASR / Verification model.
* **Goal:** Ultimate check using context.

## 6. Streaming Inference

KWS models must process audio frame-by-frame.
* **Sliding Window:** Re-compute the entire window every shift. (Expensive).
* **Streaming Convolutions:** Cache the "left" context of the convolution so it doesn't need to be recomputed.
* **Ring Buffers:** Efficiently manage audio data without copying.

## 7. Data Augmentation

Crucial for robustness.
* **Noise Injection:** Mix with cafe noise, street noise, TV, music.
* **RIR (Room Impulse Response):** Convolve with RIRs to simulate reverb (bathroom, living room).
* **Pitch Shifting / Time Stretching:** Simulate different speakers and speaking rates.
* **SpecAugment:** Mask blocks of time or frequency in the spectrogram.

## 8. Deep Dive: Keyword Spotting on Microcontrollers (TinyML)

Running on an ARM Cortex-M4 or specialized NPU (Neural Processing Unit).
* **Quantization:** Convert FP32 weights/activations to INT8.
 * Reduces memory by 4x.
 * Speedup on hardware with SIMD instructions (e.g., ARM NEON / CMSIS-NN).
* **Pruning:** Remove near-zero weights.
* **Compiler Optimization:** Fusing layers (Conv + BatchNorm + ReLU) to reduce memory access.

## 9. Evaluation Metrics

* **FRR (False Rejection Rate):** % of times user said "Alexa" but was ignored.
* **FAR (False Alarm Rate):** False positives per hour (e.g., 0.5 FA/hr).
* **ROC Curve:** Plot FRR vs FAR.
* **Latency:** Time from end of keyword to trigger.

## 10. Deep Dive: Feature Extraction (MFCC vs PCEN)

The choice of input features makes or breaks the model in noisy environments.

**MFCC (Mel-Frequency Cepstral Coefficients):**
* Standard for decades.
* Log-Mel Filterbank -> DCT (Discrete Cosine Transform).
* **Issue:** Not robust to gain variations (volume changes).

**PCEN (Per-Channel Energy Normalization):**
* Designed by Google for far-field KWS.
* Replaces the Log compression with a dynamic compression based on an Automatic Gain Control (AGC) mechanism.
* `E(t, f)` is the filterbank energy.
* `M(t, f) = (1-s) M(t-1, f) + s E(t, f)` (Smoothed energy).
* `PCEN(t, f) = (E(t, f) / (M(t, f) + \epsilon))^\alpha + \delta`.
* **Result:** Enhances transients (speech onsets) and suppresses stationary noise (fan hum).

## 11. Deep Dive: TC-ResNet (Temporal Convolutional ResNet)

A dominant architecture for KWS.
**Idea:** Treat audio as a 1D time series with `C` channels (frequency bins), rather than a 2D image.
**Structure:**
* **Input:** `T \times F` (Time x Frequency). Treat `F` as input channels.
* **Conv1D:** Kernel size `(K, 1)`. Convolves only along time.
* **Residual Blocks:** Standard ResNet skip connections.
* **Receptive Field:** Stacking layers increases the receptive field to cover the whole keyword duration (e.g., 1 second).
* **Advantages:**
 * Fewer parameters than 2D CNNs.
 * Matches the physical nature of audio (temporal evolution of spectral content).

## 12. Deep Dive: Acoustic Echo Cancellation (AEC)

**Problem:** "Barge-in". The user says "Alexa, stop!" while the device is playing loud music. The microphone captures the user's voice + the music.
**Solution:** AEC.
1. **Reference Signal:** The device knows what music it is playing (`x(t)`).
2. **Adaptive Filter:** Estimate the room's transfer function (impulse response `h(t)`).
3. **Prediction:** Predict the echo `y(t) = x(t) * h(t)`.
4. **Subtraction:** Subtract `y(t)` from the microphone input `d(t)`. Error `e(t) = d(t) - y(t)`.
5. **Update:** Use LMS (Least Mean Squares) or RLS (Recursive Least Squares) to update `h(t)` to minimize `e(t)`.
6. **KWS Input:** The "clean" error signal `e(t)` is fed to the Wake Word engine.

## 13. Deep Dive: Personalization (Few-Shot Learning)

Users want custom wake words ("Hey Jarvis").
**Challenge:** We cannot train a massive model from scratch for every user (needs 1000s of samples).
**Solution:** Transfer Learning / Embedding Matching.
1. **Base Model:** Train a powerful encoder on a massive dataset to map audio to a fixed-size embedding vector.
2. **Enrollment:** User says "Hey Jarvis" 3 times.
3. **Registration:** Average the 3 embeddings to create a "Prototype" vector for "Hey Jarvis".
4. **Inference:**
 * Compute embedding of current audio frame.
 * Calculate Cosine Similarity with the Prototype.
 * If similarity > threshold, trigger.

## 14. Deep Dive: Federated Learning for KWS

**Privacy:** Users don't want their raw audio sent to the cloud to improve the model.
**Federated Learning:**
1. **Local Training:** The device (e.g., phone) detects a False Alarm (user manually cancels).
2. **On-Device Update:** The model is fine-tuned locally on this negative sample.
3. **Aggregation:** The *weight updates* (gradients) are sent to the server (encrypted), not the audio.
4. **Global Update:** Server averages updates from millions of devices and pushes a new global model.

## 16. Deep Dive: Voice Activity Detection (VAD)

Before the KWS engine even runs, a VAD gatekeeper decides if there is *any* speech at all.
**Goal:** Save power. If silence, don't run the KWS model.

**Types of VAD:**
1. **Energy-Based:**
 * Compute Short-Time Energy.
 * If Energy > Threshold, trigger.
 * **Pros:** Extremely cheap (run on DSP/Analog).
 * **Cons:** Triggers on door slams, wind noise.
2. **Zero-Crossing Rate (ZCR):**
 * Speech oscillates more than noise (usually).
3. **Model-Based (GMM / DNN):**
 * Small GMM (Gaussian Mixture Model) trained on Speech vs Noise.
 * **WebRTC VAD:** Industry standard. Uses GMMs on sub-band energies.

**System Design:**
* **Always-On:** Energy VAD (10 uW).
* **Level 2:** WebRTC VAD (100 uW).
* **Level 3:** KWS Model (1 mW).

## 17. Deep Dive: Beamforming (Microphone Arrays)

Smart speakers have 2-8 microphones. We use them to "steer" the listening beam towards the user and nullify noise sources (TV).

**1. Delay-and-Sum:**
* If user is at angle `\theta`, sound reaches Mic 1 at `t_1` and Mic 2 at `t_2`.
* We shift Mic 2's signal by `\Delta t = t_1 - t_2` so they align.
* Summing them constructively interferes (boosts signal) and destructively interferes for other angles (noise).

**2. MVDR (Minimum Variance Distortionless Response):**
* Mathematically optimal beamformer.
* Minimizes output power (noise) while maintaining gain of 1 in the target direction.
* Requires estimating the **Spatial Covariance Matrix** of the noise.

**3. Blind Source Separation (ICA):**
* Independent Component Analysis.
* Separates mixed signals (Cocktail Party Problem) without knowing geometry.

## 18. Deep Dive: Evaluation Datasets

To build a robust KWS, you need diverse data.

**1. Google Speech Commands:**
* Open source. 65,000 one-second utterances.
* 30 words ("Yes", "No", "Up", "Down", "Marvin").
* Good for benchmarking, bad for production (clean audio).

**2. Hey Snips:**
* Crowdsourced wake word dataset.
* "Hey Snips".
* Contains near-field and far-field.

**3. LibriSpeech:**
* 1000 hours of audiobooks.
* Used for "Negative" data (background speech that should NOT trigger).

**4. Musan:**
* Music, Speech, and Noise dataset.
* Used for augmentation (overlaying noise).

## 19. Deep Dive: Hardware Accelerators (NPU/DSP)

Where does this code run?

**1. Cadence HiFi 4/5 DSP:**
* VLIW (Very Long Instruction Word) architecture.
* Optimized for audio FFTs and matrix math.
* Standard in Alexa/Google Home devices.

**2. ARM Ethos-U55 (NPU):**
* Micro-NPU designed to run alongside Cortex-M.
* Accelerates TensorFlow Lite Micro models.
* Supports INT8 quantization natively.
* 256 MACs/cycle.

**3. Analog Compute (Syntiant):**
* Performs matrix multiplication in flash memory (In-Memory Compute).
* Ultra-low power (< 140 uW for KWS).

## 20. Interview Questions (Advanced)

1. **How do you handle the class imbalance problem in KWS?** (Oversampling, Weighted Loss, Hard Negative Mining).
2. **Why use Depthwise Separable Convolutions?** (Reduce parameters/MACs).
3. **Design a KWS system for a battery-powered toy.** (Focus on Stage 1 DSP, quantization, INT8).
4. **How to detect "Alexa" vs "Alex"?** (Phonetic modeling, sub-word units, or negative training data).
5. **Explain the difference between Streaming and Non-Streaming inference.**

## 21. Deep Dive: Quantization (INT8 Inference)

Converting FP32 models to INT8 reduces memory by 4x and speeds up inference on edge devices.

**Post-Training Quantization (PTQ):**
1. **Calibration:** Run the model on a small calibration dataset (e.g., 1000 samples).
2. **Collect Statistics:** Record min/max values of activations for each layer.
3. **Compute Scale/Zero-Point:**
 * `scale = \frac{max - min}{255}`
 * `zero\_point = -\frac{min}{scale}`
4. **Quantize:** `q = round(\frac{x}{scale} + zero\_point)`
5. **Dequantize (for inference):** `x = (q - zero\_point) \times scale`

**Quantization-Aware Training (QAT):**
* Simulate quantization during training by adding fake quantization nodes.
* Model learns to be robust to quantization noise.
* **Result:** 1-2% accuracy improvement over PTQ.

**TensorFlow Lite Micro Example:**
``python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Calibration
def representative_dataset():
 for data in calibration_data:
 yield [data]

converter.representative_dataset = representative_dataset
tflite_model = converter.convert()
``

## 22. Deep Dive: Pruning (Structured vs Unstructured)

**Unstructured Pruning:**
* Remove individual weights (set to zero).
* **Pros:** High compression (90% sparsity possible).
* **Cons:** Requires sparse matrix libraries (not all hardware supports this efficiently).

**Structured Pruning:**
* Remove entire channels, filters, or attention heads.
* **Pros:** Works on standard hardware (dense operations).
* **Cons:** Lower compression ratio (~50%).

**Magnitude-Based Pruning:**
1. Train model to convergence.
2. Prune weights with smallest magnitude.
3. Fine-tune the pruned model.
4. Repeat (iterative pruning).

**Lottery Ticket Hypothesis:**
* A randomly initialized network contains a "winning ticket" subnetwork that, when trained in isolation, can match the full network's accuracy.
* **Implication:** We can find small, trainable networks by pruning and rewinding to initial weights.

## 23. Deep Dive: Neural Architecture Search (NAS) for KWS

Manually designing DS-CNN is tedious. Can we automate it?

**NAS Approaches:**
1. **Reinforcement Learning (NASNet):**
 * Controller RNN generates architectures.
 * Train each architecture, use accuracy as reward.
 * Update controller to generate better architectures.
 * **Cost:** 1000s of GPU hours.
2. **Differentiable NAS (DARTS):**
 * Represent architecture as a weighted sum of operations.
 * Optimize architecture weights and model weights jointly.
 * **Cost:** 1 GPU day.
3. **Hardware-Aware NAS:**
 * Objective: Maximize accuracy subject to latency < 10ms on ARM Cortex-M4.
 * Use lookup tables for operation latency.

**MicroNets (Google):**
* NAS-designed models for KWS on microcontrollers.
* Achieves 96% accuracy with 20KB model size.

## 24. Production Deployment: On-Device Pipeline

**End-to-End System:**
1. **Audio Capture:** MEMS microphone (16 kHz, 16-bit PCM).
2. **Pre-Processing:**
 * High-Pass Filter (remove DC offset).
 * Pre-Emphasis (`y[n] = x[n] - 0.97 \times x[n-1]`).
3. **Feature Extraction:**
 * STFT (Short-Time Fourier Transform) using Hann window.
 * Mel Filterbank (40 bins).
 * Log compression or PCEN.
4. **Inference:**
 * Run INT8 quantized DS-CNN.
 * Output: Posterior probabilities for [Keyword, Background, Silence].
5. **Posterior Smoothing:**
 * Moving average over 5 frames.
6. **Decoder:**
 * If `P(Keyword) > 0.8` for 3 consecutive frames, trigger.
7. **Action:**
 * Wake up Application Processor.
 * Start streaming audio to cloud ASR.

## 25. Production Deployment: A/B Testing & Metrics

**Metrics:**
* **False Alarm Rate (FAR):** Measured in FA/hour. Target: < 0.5 FA/hr.
* **False Rejection Rate (FRR):** % of true keywords missed. Target: < 5%.
* **Latency:** Time from end of keyword to trigger. Target: < 200ms.

**A/B Testing:**
* Deploy new model to 1% of devices.
* Compare FAR/FRR with baseline.
* **Challenge:** Users don't report false rejections (they just repeat). Need to infer from retry patterns.

**Shadow Mode:**
* Run new model in parallel with production model.
* Log predictions but don't act on them.
* Analyze offline to estimate FAR/FRR before full deployment.

## 26. Case Study: Amazon Alexa Wake Word Engine

**Architecture:**
* **Stage 1 (DSP):** Tiny DNN (50KB). Always-on. Power: 1 mW.
* **Stage 2 (AP):** Larger CNN (5MB). Runs when Stage 1 triggers. Power: 100 mW.
* **Stage 3 (Cloud):** Full ASR + NLU. Verifies intent.

**Training Data:**
* **Positive:** 500K utterances of "Alexa" (crowdsourced + synthetic).
* **Negative:** 10M hours of background audio (TV, music, conversations).
* **Augmentation:** Noise, reverb, codec distortion (Opus, AAC).

**Challenges:**
* **Child Speech:** Higher pitch, different phonetics. Needed separate child voice model.
* **Accents:** Trained separate models for US, UK, India, Australia.
* **Privacy:** All training data anonymized. No raw audio stored, only features.

## 27. Case Study: Google Assistant "Hey Google"

**Innovations:**
* **Personalization:** Uses speaker verification (d-vector) to recognize enrolled user's voice.
* **Hotword-Free Interaction:** "Continued Conversation" mode keeps listening for 8 seconds after response.
* **Multi-Hotword:** Supports "Hey Google" and "OK Google" with a single model (multi-task learning).

**Model:**
* **Architecture:** Conformer (Convolution + Transformer).
* **Size:** 14MB (on-device), 200MB (cloud).
* **Latency:** 150ms (on-device), 50ms (cloud with TPU).

## 29. Deep Dive: Privacy & Security

**Privacy Concerns:**
* **Always Listening:** Microphone is always on. Risk of accidental recording.
* **Cloud Processing:** Audio is sent to servers. Potential for data breaches.

**Solutions:**

**1. Local Processing:**
* Run full ASR on-device (e.g., Apple Siri on iPhone 15 with Neural Engine).
* **Challenge:** Large models (1GB+) don't fit on low-power devices.

**2. Differential Privacy:**
* Add noise to training data or model updates.
* Prevents extracting individual user data from the model.
* **Trade-off:** Slight accuracy degradation.

**3. Secure Enclaves:**
* Process audio in a hardware-isolated environment (ARM TrustZone, Intel SGX).
* Even the OS can't access the audio.

**4. Homomorphic Encryption:**
* Encrypt audio before sending to cloud.
* Server performs inference on encrypted data.
* **Challenge:** 1000x slowdown. Not practical yet.

## 30. Deep Dive: Adversarial Attacks on KWS

**Attack Scenarios:**

**1. Audio Adversarial Examples:**
* Add imperceptible noise to audio that causes misclassification.
* **Example:** "Hey Google" + noise → classified as silence.
* **Defense:** Adversarial training (train on adversarial examples).

**2. Hidden Voice Commands:**
* Embed commands in music or ultrasonic frequencies.
* **DolphinAttack:** Use ultrasound (>20 kHz) to trigger voice assistants.
* **Defense:** Low-pass filter, frequency analysis.

**3. Replay Attacks:**
* Record user saying "Alexa", replay it later.
* **Defense:** Liveness detection (check for acoustic properties of live speech vs recording).

## 31. Edge Deployment: TensorFlow Lite Micro

**TFLite Micro:**
* Runs on microcontrollers with <1MB RAM.
* No OS required (bare metal).
* **Workflow:**
 1. Train model in TensorFlow/Keras.
 2. Convert to TFLite with quantization.
 3. Generate C++ code.
 4. Compile for target MCU (ARM Cortex-M4).

**Example: Arduino Nano 33 BLE Sense:**
* **MCU:** ARM Cortex-M4 (64 MHz, 256KB RAM).
* **Microphone:** MP34DT05 (PDM).
* **Model:** 20KB DS-CNN.
* **Latency:** 50ms.
* **Power:** 5 mW.

## 32. Edge Deployment: Optimization Techniques

**1. Operator Fusion:**
* Combine Conv + BatchNorm + ReLU into a single kernel.
* Reduces memory access (faster).

**2. Weight Clustering:**
* Cluster weights into `K` centroids (e.g., 256).
* Store only the centroid index (8 bits) instead of full weight (32 bits).
* **Compression:** 4x.

**3. Knowledge Distillation:**
* Train a small "student" model to mimic a large "teacher" model.
* Student learns from teacher's soft probabilities (not just hard labels).
* **Result:** Student achieves 95% of teacher's accuracy with 10x fewer parameters.

## 33. Future Trends

**1. Multimodal Wake Words:**
* Combine audio + visual (lip reading) for more robust detection.
* **Use Case:** Noisy environments (construction site).

**2. Contextual Wake Words:**
* "Alexa" only triggers if you're looking at the device (gaze detection).
* Reduces false alarms from TV.

**3. Neuromorphic Computing:**
* Spiking Neural Networks (SNNs) on neuromorphic chips (Intel Loihi).
* **Benefit:** 1000x lower power than traditional DNNs.
* **Challenge:** Training SNNs is hard.

**4. On-Device Personalization:**
* Model adapts to your voice over time (continual learning).
* No cloud updates needed.


## 34. Common Mistakes

* **Not Testing on Real Devices:** Models that work on GPU may fail on MCU due to numerical precision issues.
* **Ignoring Power Consumption:** A model that drains the battery in 2 hours is useless.
* **Overfitting to Clean Data:** Real-world audio is noisy, reverberant, and distorted.
* **Not Handling Edge Cases:** What happens if the user whispers? Shouts? Has a cold?
* **Forgetting Latency:** A 500ms delay between "Alexa" and the response is unacceptable.

## 35. Testing & Validation

**Unit Tests:**
* Test feature extraction (MFCC output matches reference).
* Test model inference (output shape, value ranges).
* Test quantization (INT8 output close to FP32).

**Integration Tests:**
* End-to-end pipeline on recorded audio.
* Measure latency (audio in → trigger out).
* Test on different devices (iPhone, Android, Raspberry Pi).

**Stress Tests:**
* **Noise Robustness:** Add noise at SNR = -5 dB, 0 dB, 5 dB, 10 dB.
* **Reverberation:** Convolve with room impulse responses (T60 = 0.3s, 0.6s, 1.0s).
* **Codec Distortion:** Encode/decode with Opus, AAC, MP3 at various bitrates.
* **Long-Running:** Run for 24 hours. Check for memory leaks, drift.

## 36. Benchmarking Frameworks

**MLPerf Tiny:**
* Industry-standard benchmark for TinyML.
* **Tasks:** Keyword Spotting, Visual Wake Words, Anomaly Detection, Image Classification.
* **Metrics:** Accuracy, Latency, Energy.
* **Leaderboard:** Compare different hardware (Cortex-M4, Cortex-M7, NPUs).

**Example Results (KWS on Google Speech Commands):**
* **ARM Cortex-M4 (80 MHz):**
 * Model: DS-CNN (20KB).
 * Accuracy: 90.5%.
 * Latency: 15ms.
 * Energy: 0.3 mJ.
* **ARM Ethos-U55 (NPU):**
 * Model: DS-CNN (20KB).
 * Accuracy: 90.5%.
 * Latency: 5ms.
 * Energy: 0.05 mJ (6x better).

## 37. Production Monitoring

**Metrics to Track:**
* **False Alarm Rate (FAR):** Aggregate across all devices. Alert if > 0.5 FA/hr.
* **False Rejection Rate (FRR):** Inferred from retry patterns (user says "Alexa" twice).
* **Latency Distribution:** P50, P95, P99. Alert if P95 > 300ms.
* **Device Health:** Battery drain, CPU usage, memory usage.

**Dashboards:**
* **Grafana:** Real-time metrics.
* **Kibana:** Log analysis (search for "wake word triggered").
* **A/B Test Results:** Compare new model vs baseline.

**Incident Response:**
* **High FAR:** Roll back to previous model.
* **High FRR:** Investigate (new accent? new noise source?).
* **High Latency:** Check for CPU throttling, memory leaks.

## 38. Cost Analysis

**Development Costs:**
* **Data Collection:** $500K (crowdsourcing 500K utterances).
* **Annotation:** $100K (labeling, quality control).
* **Training:** $50K (GPU cluster for 2 weeks).
* **Engineering:** $1M (10 engineers for 6 months).
* **Total:** ~$1.65M.

**Operational Costs (per million devices):**
* **Cloud Verification (Stage 3):** `0.01 per query. If 10% of triggers go to cloud, and each device triggers 5 times/day: `0.01 * 0.1 * 5 * 1M = `5K/day = `1.8M/year.
* **Model Updates:** $10K/month (OTA updates, CDN).
* **Monitoring:** $5K/month (Datadog, Grafana Cloud).

**Optimization:**
* Move more processing on-device (reduce cloud costs).
* Use edge caching (reduce CDN costs).

## 39. Ethical Considerations

**Bias:**
* Models trained on US English may not work for Indian English.
* **Solution:** Collect diverse data. Test on all demographics.

**Accessibility:**
* Users with speech impairments may struggle.
* **Solution:** Offer alternative input methods (button press, text).

**Surveillance:**
* Always-on microphones can be abused.
* **Solution:** Hardware mute button. LED indicator when listening.

**Environmental Impact:**
* Training large models consumes energy (carbon footprint).
* **Solution:** Use renewable energy. Optimize models (reduce training time).

## 40. Conclusion

Wake Word Detection is the unsung hero of voice AI. It's the first line of defense, the gatekeeper that decides when to wake up the expensive cloud infrastructure.
**Key Takeaways:**
* **Efficiency is King:** Power, memory, and latency constraints are brutal.
* **Cascaded Architecture:** Use multiple stages (DSP → AP → Cloud) to balance power and accuracy.
* **Quantization & Pruning:** Essential for edge deployment.
* **Robustness:** Test on noisy, reverberant, far-field audio.
* **Privacy:** Process as much as possible on-device.

The next generation of KWS will be multimodal (audio + visual), contextual (gaze-aware), and personalized (adapts to your voice). The challenge is to do all this while consuming less than 1 mW of power.


---

**Originally published at:** [arunbaby.com/speech-tech/0040-wake-word-detection](https://www.arunbaby.com/speech-tech/0040-wake-word-detection/)

*If you found this helpful, consider sharing it with others who might benefit.*

