---
title: "End-to-End Text-to-Speech (TTS)"
day: 39
related_dsa_day: 39
related_ml_day: 39
related_agents_day: 39
collection: speech_tech
categories:
 - speech_tech
tags:
 - tts
 - tacotron
 - wavenet
 - vocoder
 - deep-learning
subdomain: "Synthesis"
tech_stack: [PyTorch, Espnet, Coqui TTS]
scale: "Real-time Synthesis"
companies: [Google, Amazon (Polly), ElevenLabs, Descript]
---

**"Giving machines a voice."**

## 1. The Evolution of TTS

### 1. Concatenative Synthesis (1990s - 2010s)
- **Method:** Record a voice actor reading thousands of sentences. Chop them into phonemes/diphones. Glue them together at runtime.
- **Pros:** Very natural sound for recorded segments.
- **Cons:** "Glitchy" at boundaries. Cannot change emotion or style. Requires massive database (GBs).

### 2. Statistical Parametric Synthesis (HMMs) (2000s - 2015)
- **Method:** Generate acoustic features (F0, spectral envelope) from text using HMMs. Use a vocoder to convert features to audio.
- **Pros:** Flexible, small footprint.
- **Cons:** "Muffled" or "robotic" sound (due to averaging in HMMs).

### 3. Neural / End-to-End TTS (2016 - Present)
- **Method:** Deep Neural Networks map Text `\to` Spectrogram `\to` Waveform.
- **Pros:** Human-level naturalness. Controllable style/emotion.
- **Cons:** Computationally expensive.

## 2. Anatomy of a Modern TTS System

A typical Neural TTS system has two stages:

1. **Acoustic Model (Text `\to` Mel-Spectrogram):**
 - Converts character/phoneme sequence into a time-frequency representation (Mel-spectrogram).
 - Example: **Tacotron 2**, FastSpeech 2, VITS.

2. **Vocoder (Mel-Spectrogram `\to` Waveform):**
 - Inverts the spectrogram back to time-domain audio.
 - Example: **WaveNet**, WaveGlow, HiFi-GAN.

## 3. Deep Dive: Tacotron 2 Architecture

Tacotron 2 (Google, 2017) is the gold standard for high-quality TTS.

### 1. Encoder
- **Input:** Character sequence.
- **Layers:** 3 Convolutional layers (context) + Bi-directional LSTM.
- **Output:** Encoded text features.

### 2. Attention Mechanism
- **Location-Sensitive Attention:** Crucial for TTS.
- Unlike translation (where reordering happens), speech is monotonic.
- The attention weights must move forward linearly.
- Uses previous attention weights as input to calculate current attention.

### 3. Decoder
- **Type:** Autoregressive LSTM.
- **Input:** Previous Mel-frame.
- **Output:** Current Mel-frame.
- **Stop Token:** Predicts when to stop generating.

### 4. Post-Net
- **Purpose:** Refine the Mel-spectrogram.
- **Layers:** 5 Convolutional layers.
- **Residual Connection:** Adds detail to the decoder output.

**Loss Function:** MSE between predicted and ground-truth Mel-spectrograms.

## 4. Deep Dive: Neural Vocoders

The Mel-spectrogram is lossy (phase information is discarded). The Vocoder must "hallucinate" the phase to generate high-fidelity audio.

### 1. Griffin-Lim (Algorithm)
- **Method:** Iterative algorithm to estimate phase.
- **Pros:** Fast, no training.
- **Cons:** Robotic, metallic artifacts.

### 2. WaveNet (Autoregressive)
- **Method:** Predicts sample `x_t` based on `x_{t-1}, x_{t-2}, ...`
- **Architecture:** Dilated Causal Convolutions.
- **Pros:** State-of-the-art quality.
- **Cons:** Extremely slow (sequential generation). 1 second of audio = 24,000 steps.

### 3. WaveGlow (Flow-based)
- **Method:** Normalizing Flows. Maps Gaussian noise to audio.
- **Pros:** Parallel inference (fast). High quality.
- **Cons:** Huge model (hundreds of millions of parameters).

### 4. HiFi-GAN (GAN-based)
- **Method:** Generator produces audio, Discriminator distinguishes real vs fake.
- **Pros:** Very fast (real-time on CPU), high quality.
- **Cons:** Training instability (GANs).
- **Current Standard:** HiFi-GAN is the default for most systems today.

## 5. FastSpeech 2: Non-Autoregressive TTS

**Problem with Tacotron:**
- Autoregressive generation is slow (O(N)).
- Attention failures (skipping or repeating words).

**FastSpeech 2 Solution:**
- **Non-Autoregressive:** Generate all frames in parallel (O(1)).
- **Duration Predictor:** Explicitly predict how many frames each phoneme lasts.
- **Pitch/Energy Predictors:** Explicitly model prosody.

**Architecture:**
- Encoder (Transformer) `\to` Variance Adaptor (Duration/Pitch/Energy) `\to` Decoder (Transformer).

**Pros:**
- Extremely fast training and inference.
- Robust (no skipping/repeating).
- Controllable (can manually adjust speed/pitch).

## 6. System Design: Building a TTS API

**Scenario:** Build a scalable TTS service like Amazon Polly.

**Requirements:**
- **Latency:** < 200ms time-to-first-byte (streaming).
- **Throughput:** 1000 concurrent streams.
- **Voices:** Support multiple speakers/languages.

**Architecture:**

1. **Frontend (Text Normalization):**
 - "Dr. Smith lives on St. John St." `\to` "Doctor Smith lives on Saint John Street".
 - "12:30" `\to` "twelve thirty".
 - **G2P (Grapheme-to-Phoneme):** Convert text to phonemes (using CMU Dict or Model).

2. **Synthesis Engine:**
 - **Model:** FastSpeech 2 (for speed) + HiFi-GAN.
 - **Optimization:** ONNX Runtime / TensorRT.
 - **Streaming:** Chunk text into sentences. Synthesize sentence 1 while user listens.

3. **Caching:**
 - Cache common phrases ("Your ride has arrived").
 - Hit rate for TTS is surprisingly high for navigational/assistant apps.

4. **Scaling:**
 - GPU inference is preferred (T4/A10G).
 - Autoscaling based on queue depth.

## 7. Evaluation Metrics

**1. MOS (Mean Opinion Score):**
- Human raters listen and rate from 1 (Bad) to 5 (Excellent).
- **Ground Truth:** ~4.5.
- **Tacotron 2:** ~4.3.
- **Parametric:** ~3.5.

**2. Intelligibility (Word Error Rate):**
- Feed generated audio into an ASR system.
- Check if the ASR transcribes it correctly.

**3. Latency (RTF - Real Time Factor):**
- Time to generate / Duration of audio.
- RTF < 1.0 means faster than real-time.

## 8. Advanced: Voice Cloning (Zero-Shot TTS)

**Goal:** Generate speech in a target speaker's voice given only a 3-second reference clip.

**Architecture (e.g., Vall-E, XTTS):**
1. **Speaker Encoder:** Compresses reference audio into a fixed-size vector (d-vector).
2. **Conditioning:** Feed d-vector into the TTS model (AdaIN or Concatenation).
3. **Language Modeling:** Treat TTS as a language modeling task (Audio Tokens).

**Vall-E (Microsoft):**
- Uses EnCodec (Audio Codec) to discretize audio.
- Trains a GPT-style model to predict audio tokens from text + acoustic prompt.

## 9. Common Challenges

**1. Text Normalization:**
- "$19.99" -> "nineteen dollars and ninety-nine cents".
- Context dependency: "I read the book" (past) vs "I will read" (future).

**2. Prosody and Emotion:**
- Default TTS is "neutral/newsreader" style.
- Generating "angry" or "whispering" speech requires labeled data or style transfer.

**3. Long-Form Synthesis:**
- Attention mechanisms can drift over long paragraphs.
- **Fix:** Windowed attention or sentence-level splitting.

## 10. Ethical Considerations

**1. Deepfakes:**
- Voice cloning can break biometric auth (banks).
- Used for scams ("Grandma, I'm in jail, send money").
- **Mitigation:** Watermarking audio (inaudible noise).

**2. Copyright:**
- Training on audiobooks without consent.
- **Impact:** Voice actors losing jobs.

## 11. Deep Dive: Tacotron 2 Attention Mechanism

Why does standard Attention fail for TTS?
- In Machine Translation, alignment is soft and can jump (e.g., "red house" -> "maison rouge").
- In TTS, alignment is **monotonic** and **continuous**. You never read the end of the sentence before the beginning.

**Location-Sensitive Attention:**
Standard Bahdanau Attention uses query `s_{i-1}` and values `h_j`.
` e_{i,j} = v^T \tanh(W s_{i-1} + V h_j + b) `

Location-Sensitive Attention adds the **previous alignment** `\alpha_{i-1}` as a feature.
` f_i = F * \alpha_{i-1} `
` e_{i,j} = v^T \tanh(W s_{i-1} + V h_j + U f_{i,j} + b) `

**Effect:**
- The model "knows" where it attended last time.
- It learns to simply shift the attention window forward.
- Prevents "babbling" (repeating the same word forever) or skipping words.

## 12. Deep Dive: WaveNet Dilated Convolutions

WaveNet generates raw audio sample-by-sample (16,000 samples/sec).
To generate sample `x_t`, it needs context from a long history (e.g., 1 second).

**Problem:** Standard convolution with size 3 needs thousands of layers to reach a receptive field of 16,000.

**Solution: Dilated Convolutions:**
- Skip input values with a step size (dilation).
- Layer 1: Dilation 1 (Look at `t, t-1`)
- Layer 2: Dilation 2 (Look at outputs of Layer 1 at `t, t-2`)
- Layer 3: Dilation 4 (Look at outputs of Layer 2 at `t, t-4`)
- ...
- Layer 10: Dilation 512.

**Receptive Field:**
Exponential growth: `2^L`. With 10 layers, we cover 1024 samples. Stack multiple blocks to reach 16,000.

**Conditioning:**
WaveNet is conditioned on the Mel-spectrogram `c`.
` P(x_t | x_{<t}, c) = \text{softmax}(W \cdot \tanh(W_f x + V_f c) \cdot \sigma(W_g x + V_g c)) `
(Gated Activation Unit).

## 13. Deep Dive: HiFi-GAN Architecture

HiFi-GAN (High Fidelity GAN) is the current state-of-the-art vocoder because it's fast and high quality.

**Generator:**
- Input: Mel-spectrogram.
- **Multi-Receptive Field Fusion (MRF):**
 - Instead of one ResNet block, it runs multiple ResNet blocks with different kernel sizes and dilation rates in parallel.
 - Sums their outputs.
 - Allows capturing both fine-grained details (high frequency) and long-term dependencies (low frequency).

**Discriminators:**
1. **Multi-Period Discriminator (MPD):**
 - Reshapes 1D audio into 2D matrices of height `p` (periods 2, 3, 5, 7, 11).
 - Applies 2D convolution.
 - Detects periodic artifacts (metallic sounds).
2. **Multi-Scale Discriminator (MSD):**
 - Operates on raw audio, 2x downsampled, 4x downsampled audio.
 - Ensures structure is correct at different time scales.

**Loss:**
- GAN Loss (Adversarial).
- Feature Matching Loss (Match intermediate layers of discriminator).
- Mel-Spectrogram Loss (L1 distance).

## 14. System Design: Streaming TTS Architecture

**Challenge:** User shouldn't wait 5 seconds for a long paragraph to be synthesized.

**Architecture:**

1. **Text Chunking:**
 - Split text by punctuation (., !, ?).
 - "Hello world! How are you?" -> ["Hello world!", "How are you?"].

2. **Incremental Synthesis:**
 - Send Chunk 1 to TTS Engine.
 - While Chunk 1 is playing, synthesize Chunk 2.

3. **Buffer Management:**
 - Client maintains a jitter buffer (e.g., 200ms).
 - If synthesis is faster than playback (RTF < 1.0), buffer fills up.
 - If synthesis is slower, buffer underruns (stuttering).

4. **Protocol:**
 - **WebSocket / gRPC:** Bi-directional streaming.
 - Server sends binary audio chunks (PCM or Opus encoded).

**Stateful Context:**
- Simply splitting by sentence breaks prosody (pitch resets at start of sentence).
- **Contextual TTS:** Pass the *embedding* of the previous sentence's end state as the initial state for the next sentence.

## 15. Advanced: Style Transfer and Emotion Control

**Global Style Tokens (GST):**
- Learn a bank of "style embeddings" (tokens) during training in an unsupervised way.
- At inference, we can choose a token (e.g., Token 3 might capture "fast/angry", Token 5 "slow/sad").
- We can mix styles: `0.5 \times \text{Happy} + 0.5 \times \text{Whisper}`.

**Reference Audio:**
- Feed a 3-second clip of *expressive* speech.
- Reference Encoder extracts style vector.
- TTS synthesizes new text with that style.

## 16. Case Study: Voice Cloning for Accessibility

**Scenario:** A patient with ALS (Lou Gehrig's disease) is losing their voice. They want to "bank" their voice to use with a TTS system later.

**Process:**
1. **Recording:** Patient records 30-60 minutes of reading scripts while they can still speak.
2. **Fine-Tuning:**
 - Take a pre-trained multi-speaker model (e.g., trained on LibriTTS).
 - Freeze the encoder/decoder layers.
 - Fine-tune the **Speaker Embedding** and last few decoder layers on the patient's data.
3. **Deployment:** Run the model on an iPad (using CoreML/TensorFlow Lite).

**Challenges:**
- **Fatigue:** Patient cannot record for hours. Need data-efficient adaptation (Few-Shot Learning).
- **Dysarthria:** If speech is already slurred, the model will learn the slur. Need "Voice Repair" (mapping slurred speech to healthy speech space).

- **Dysarthria:** If speech is already slurred, the model will learn the slur. Need "Voice Repair" (mapping slurred speech to healthy speech space).

## 17. Deep Dive: VITS (Conditional Variational Autoencoder with Adversarial Learning)

VITS (2021) is the current state-of-the-art "all-in-one" model. It combines Acoustic Model and Vocoder into a single end-to-end network.

**Key Idea:**
- **Training:** It's a VAE.
 - Encoder: Takes **Audio** `\to` Latent `z`.
 - Decoder: Takes Latent `z` `\to` Audio (HiFi-GAN generator).
 - **Prior:** The latent `z` is forced to follow a distribution predicted from **Text**.
- **Inference:**
 - Text Encoder predicts the distribution of `z`.
 - Sample `z`.
 - Decoder generates audio.

**Flow-based Prior:**
- To make the text-to-latent prediction expressive, it uses Normalizing Flows.

**Monotonic Alignment Search (MAS):**
- VITS learns the alignment between text and audio *unsupervised* during training using Dynamic Programming (MAS). No external aligner needed.

**Pros:**
- Higher quality than Tacotron+WaveGlow.
- Faster than autoregressive models.
- No mismatch between acoustic model and vocoder.

## 18. Deep Dive: Prosody Modeling (Pitch, Energy, Duration)

To make speech sound human, we need to control *how* it's said.

**1. Duration:**
- How long is each phoneme?
- **Model:** Predict log-duration for each phoneme.
- **Control:** Multiply predicted durations by 1.2x to speak slower.

**2. Pitch (F0):**
- Fundamental frequency contour.
- **Model:** Predict continuous F0 curve.
- **Control:** Shift F0 mean to make voice higher/lower. Scale variance to make it more expressive/monotone.

**3. Energy:**
- Loudness (L2 norm of frame).
- **Model:** Predict energy per frame.

**Architecture:**
- Add these predictors after the Text Encoder.
- Add the predicted embeddings to the content embedding before the Decoder.

## 19. Deep Dive: Multi-Speaker and Multi-Lingual TTS

**1. Speaker Embeddings (d-vectors):**
- Train a speaker verification model (e.g., GE2E loss).
- Extract the embedding from the last layer.
- Condition the TTS model on this vector (Concatenate or AdaIN).

**2. Code-Switching:**
- "I want to eat *Sushi* today." (English sentence, Japanese word).
- **Challenge:** English TTS doesn't know Japanese phonemes.
- **Solution:** Shared Phoneme Set (IPA).
- **Model:** Train on mixed data. Use a Language ID embedding.

## 20. Deep Dive: Audio Codecs for Generative Audio

With models like **Vall-E** and **AudioLM**, we treat audio generation as language modeling. But audio is continuous.

**Neural Audio Codecs (EnCodec / DAC):**
- **Encoder:** Compresses audio to low-framerate latent.
- **Quantizer (RVQ - Residual Vector Quantization):**
 - Discretizes latent into "codebook indices" (tokens).
 - Hierarchical: Codebook 1 captures coarse structure, Codebook 2 captures residual error, etc.
- **Decoder:** Reconstructs audio from tokens.

**Result:**
- 1 second of audio `\to` 75 tokens.
- Now we can use GPT-4 on these tokens!

## 21. System Design: On-Device TTS Optimization

**Scenario:** Siri/Google Assistant running on a phone without internet.

**Constraints:**
- **Size:** Model < 50MB.
- **Compute:** < 10% CPU usage.

**Techniques:**
1. **Quantization:** Float32 `\to` Int8. (4x smaller).
2. **Pruning:** Remove 50% of weights that are near zero.
3. **Knowledge Distillation:** Train a tiny student model to mimic the large teacher.
4. **Streaming Vocoder:** Use **LPCNet** (combines DSP with small RNN) or **Multi-Band MelGAN** (generates 4 frequency bands in parallel).

## 22. Evaluation: MUSHRA Tests

MOS is simple but subjective. **MUSHRA (Multiple Stimuli with Hidden Reference and Anchor)** is more rigorous.

**Setup:**
- Listener hears:
 - **Reference:** Original recording (Ground Truth).
 - **Anchor:** Low-pass filtered version (Bad quality baseline).
 - **Samples:** Model A, Model B, Model C (blinded).
- Task: Rate all of them from 0-100 relative to Reference.

**Why Anchor?**
- Calibrates the scale. If someone rates the Anchor as 80, their data is discarded.

## 23. Interview Questions

**Q1: Why use Mel-spectrograms instead of linear spectrograms?**
*Answer:* Mel-scale matches human hearing (logarithmic perception of pitch). It compresses the data dimension (e.g., 1024 linear `\to` 80 Mel), making the model easier to train.

**Q2: Autoregressive vs Non-Autoregressive TTS?**
*Answer:*
- **AR (Tacotron):** Higher quality, better prosody, slow, robustness issues.
- **Non-AR (FastSpeech):** Fast, robust, controllable, slightly lower prosody quality (averaged).

**Q3: How to handle OOV words?**
*Answer:* Use a G2P (Grapheme-to-Phoneme) model that predicts pronunciation from spelling, rather than a dictionary lookup.

## 24. Further Reading

1. **"Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" (Shen et al., 2017):** Tacotron 2 paper.
2. **"FastSpeech 2: Fast and High-Quality End-to-End TTS" (Ren et al., 2020).**
3. **"HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (Kong et al., 2020).**

## 25. Conclusion

End-to-End TTS has crossed the "Uncanny Valley". With models like Tacotron 2 and HiFi-GAN, synthesized speech is often indistinguishable from human speech. The focus has now shifted from "quality" to "control" (emotion, style), "efficiency" (on-device TTS), and "adaptation" (zero-shot cloning). As generative audio models (like Vall-E) merge with LLMs, we are entering an era of conversational AI that sounds as human as it thinks.

## 26. Summary

| Component | Role | Examples |
| :--- | :--- | :--- |
| **Frontend** | Text `\to` Phonemes | G2P, Normalization |
| **Acoustic Model** | Phonemes `\to` Mel-Spec | Tacotron 2, FastSpeech 2 |
| **Vocoder** | Mel-Spec `\to` Audio | WaveNet, HiFi-GAN |
| **Speaker Encoder** | Voice Cloning | d-vector, x-vector |

---

**Originally published at:** [arunbaby.com/speech-tech/0039-end-to-end-tts](https://www.arunbaby.com/speech-tech/0039-end-to-end-tts/)
