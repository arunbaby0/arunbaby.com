---
title: "Speech Tokenization"
day: 24
collection: speech_tech
categories:
  - speech-tech
tags:
  - asr
  - tokenization
  - self-supervised-learning
  - hubert
  - wav2vec
subdomain: "Speech Representation Learning"
tech_stack: [Python, PyTorch, Fairseq]
scale: "O(T) time"
companies: [Meta (FAIR), Google, Microsoft]
related_dsa_day: 24
related_ml_day: 24
related_agents_day: 24
---

**The breakthrough that allows us to treat audio like text, enabling GPT-style models for speech.**

## The Challenge: Discretizing the Continuous

In the previous post (ML System Design), we saw how text is broken into discrete tokens (IDs). This works because text is naturally discrete. "Cat" is distinct from "Dog".

Audio is different. It is continuous.
- A waveform is a sequence of floating point numbers.
- A spectrogram is a continuous image.

If we want to apply the massive power of **Large Language Models (LLMs)** to speech—to build a "SpeechGPT"—we first need to convert this continuous signal into a sequence of discrete integers. We need a **Speech Tokenizer**.

This is the technology behind **AudioLM**, **MusicLM**, and **Speech-to-Speech Translation**. It turns "Audio Generation" into "Next Token Prediction".

## The Old Way: Phonemes

For decades, we tried to use **Phonemes** as tokens.
- Audio -> ASR Model -> Phonemes (/k/ /ae/ /t/) -> Integers.
- **Problem:** Phonemes are lossy. They capture *what* was said, but discard *how* it was said (prosody, emotion, speaker identity).
- If you synthesize speech from phonemes, it sounds robotic.

## The New Way: Semantic & Acoustic Tokens

We want tokens that capture:
1.  **Semantics:** The meaning (words).
2.  **Acoustics:** The speaker's voice, pitch, and emotion.

### 1. VQ-VAE (Vector Quantized Variational Autoencoder)

This was the first big step.
- **Encoder:** Compresses audio into a dense vector.
- **Quantizer:** Maps the vector to the nearest neighbor in a "Codebook" (a fixed list of 1024 vectors).
- **Decoder:** Reconstructs audio from the codebook vectors.

The indices of the codebook vectors become our tokens!
- Audio -> [34, 102, 88, 5] -> Audio.

**Pros:** Good reconstruction quality.
**Cons:** Tokens are low-level. They represent "sound textures", not meaning.

### 2. HuBERT (Hidden Unit BERT)

Meta AI changed the game with HuBERT.
- **Idea:** Use k-means clustering on MFCC features to create pseudo-labels. Train a BERT model to predict these cluster IDs from masked audio.
- **Result:** The model learns high-level structure. The tokens correlate strongly with phonemes, even though it was never trained on text!

### 3. SoundStream / EnCodec

These are **Neural Audio Codecs**.
- They use **Residual Vector Quantization (RVQ)**.
- Layer 1 tokens capture the coarse structure (content).
- Layer 2-8 tokens capture the fine details (timbre, noise).
- This allows for high-fidelity compression (better than MP3 at low bitrates) and tokenization.

## High-Level Architecture: The Speech-LLM Pipeline

```ascii
+-----------+    +-------------+    +-------------+    +-------------+
| Raw Audio | -> |   Encoder   | -> |  Quantizer  | -> | Discrete Tok|
+-----------+    +-------------+    +-------------+    +-------------+
                      |                  |                  |
                 (HuBERT/EnCodec)   (Codebook)         [34, 102, 88]
                                                            |
                                                            v
+-----------+    +-------------+    +-------------+    +-------------+
| New Audio | <- |   Vocoder   | <- |  LLM / GPT  | <- | Prompt Toks |
+-----------+    +-------------+    +-------------+    +-------------+
```

## System Design: Building a Speech-LLM

Once we have speech tokens, we can build cool things.

**AudioLM (Google):**
1.  **Semantic Tokens:** Use w2v-BERT to extract high-level meaning tokens.
2.  **Acoustic Tokens:** Use SoundStream to extract low-level audio tokens.
3.  **Transformer:** Train a GPT model to predict the next token.
    - Input: `[Semantic_1, Semantic_2, ..., Acoustic_1, Acoustic_2, ...]`
4.  **Inference:** Prompt with 3 seconds of audio. The model "continues" the speech, maintaining the speaker's voice and recording conditions!

## Deep Dive: How HuBERT Works

HuBERT (Hidden Unit BERT) is self-supervised. It learns from audio without text labels.

**Step 1: Discovery (Clustering)**
- Run MFCC (Mel-frequency cepstral coefficients) on the raw audio.
- Run k-means clustering (k=100) on these MFCC vectors.
- Assign each 20ms frame a cluster ID (0-99). These are the "pseudo-labels".

**Step 2: Prediction (Masked Language Modeling)**
- Mask parts of the audio input (replace with zeros).
- Feed the masked audio into a Transformer.
- Force the model to predict the cluster ID of the masked parts.
- **Loss:** Cross-Entropy between predicted ID and true cluster ID.

**Step 3: Iteration**
- Once the model is trained, use its *internal embeddings* (instead of MFCCs) to run k-means again.
- The new clusters are better. Retrain the model.
- Repeat.

## Neural Audio Codecs: EnCodec & DAC

While HuBERT captures *semantics*, we also need *acoustics* (fidelity).
**EnCodec (Meta)** and **DAC (Descript Audio Codec)** use a VQ-VAE (Vector Quantized Variational Autoencoder) with a twist: **Residual Vector Quantization (RVQ)**.

**The Problem:** A single codebook of size 1024 is not enough to capture high-fidelity audio.
**The Solution (RVQ):**
1.  **Quantizer 1:** Approximates the vector. Residual = Vector - Q1(Vector).
2.  **Quantizer 2:** Approximates the Residual. New Residual = Residual - Q2(Residual).
3.  **Quantizer N:** ...

This gives us a *stack* of tokens for each time step.
`[Token_Layer1, Token_Layer2, ..., Token_Layer8]`
Layer 1 has the "gist". Layer 8 has the "details".

## Deep Dive: The Math of VQ-VAE

The **Vector Quantized Variational Autoencoder** is the heart of modern speech tokenization. Let's break down the math that makes it work.

### 1. The Discretization Bottleneck
We have an encoder `E(x)` that produces a continuous vector `z_e`.
We have a codebook `C = {e_1, ..., e_K}` of `K` vectors.
We want to map `z_e` to the nearest codebook vector `z_q`.

`z_q = argmin_k || z_e - e_k ||_2`

### 2. The Gradient Problem
The `argmin` operation is non-differentiable. You can't backpropagate through a "choice".
**Solution: Straight-Through Estimator (STE).**
- **Forward Pass:** Use `z_q` (the quantized vector).
- **Backward Pass:** Pretend we used `z_e` (the continuous vector). Copy the gradients from decoder to encoder directly.
`dL/dz_e = dL/dz_q`

### 3. The Loss Function
We need to train 3 things: the Encoder, the Decoder, and the Codebook.
`Loss = L_reconstruction + L_codebook + beta * L_commitment`

- **L_reconstruction:** `|| x - D(z_q) ||^2`. Make the output sound like the input.
- **L_codebook:** `|| sg[z_e] - e_k ||^2`. Move the chosen codebook vector closer to the encoder output. (`sg` = stop gradient).
- **L_commitment:** `|| z_e - sg[e_k] ||^2`. Force the encoder to commit to a codebook vector (don't jump around).

### 4. Codebook Collapse
A common failure mode is "Codebook Collapse", where the model uses only 5 out of 1024 tokens. The other 1019 are never chosen, so they never get updated.
**Fix:**
- **K-means Initialization:** Initialize codebook with k-means on the first batch.
- **Random Restart:** If a code vector is dead for too long, re-initialize it to a random active encoder output.

## Advanced Architecture: EnCodec & SoundStream

Meta's **EnCodec** and Google's **SoundStream** are the state-of-the-art. They are not just VQ-VAEs; they are **Neural Audio Codecs**.

### 1. The Encoder-Decoder
- **Convolutional:** Uses 1D Convolutions to downsample the audio.
    - Input: 24kHz audio (24,000 samples/sec).
    - Downsampling factor: 320x.
    - Output: 75 frames/sec.
- **LSTM:** Adds a sequence modeling layer to capture long-term dependencies.

### 2. Residual Vector Quantization (RVQ)
As mentioned, a single codebook is too coarse. RVQ uses a cascade of `N` quantizers (usually 8).
- **Bitrate Control:**
    - If we use 8 quantizers, we get high fidelity (6 kbps).
    - If we use only the first 2 quantizers during decoding, we get lower fidelity but lower bitrate (1.5 kbps).
    - *If you found this helpful, consider sharing it with others who might benefit.*


    - This allows **Bandwidth Scalability**.

### 3. Adversarial Loss (GAN)
MSE (Mean Squared Error) loss produces "blurry" audio (muffled high frequencies).
To fix this, we add a **Discriminator** (a separate neural net) that tries to distinguish real audio from decoded audio.
- **Multi-Scale Discriminator:** Checks audio at different resolutions (raw samples, downsampled).
- **Multi-Period Discriminator:** Checks audio at different periodicities (to capture pitch).

## Generative Audio: AudioLM & MusicLM

Once we have tokens, we can generate audio like text.

### The "Coarse-to-Fine" Generation Strategy
Generating 24,000 samples/sec is hard. Generating 75 tokens/sec is easy.

**AudioLM (Google):**
1.  **Semantic Stage:**
    - Input: Text or Audio Prompt.
    - Output: Semantic Tokens (from w2v-BERT).
    - These tokens capture "The cat sat on the mat" but not the speaker's voice.
2.  **Coarse Acoustic Stage:**
    - Input: Semantic Tokens.
    - Output: The first 3 layers of RVQ tokens (from SoundStream).
    - These capture the speaker identity and prosody.
3.  **Fine Acoustic Stage:**
    - Input: Coarse Acoustic Tokens.
    - Output: The remaining 5 layers of RVQ tokens.
    - These capture the fine details (breath, background noise).

**MusicLM (Google):**
- Same architecture, but conditioned on **MuLan** embeddings (Text-Music joint embedding).
- Prompt: "A calming violin melody backed by a distorted guitar." -> MuLan Embedding -> Semantic Tokens -> Acoustic Tokens -> Audio.

## Tutorial: Training Your Own Speech Tokenizer

Want to build a custom tokenizer for a low-resource language?

**1. Data Preparation:**
- You need 100-1000 hours of raw audio.
- No text transcripts needed! (Self-supervised).
- Clean the audio (remove silence, normalize volume).

**2. Model Configuration (EnCodec):**
- **Channels:** 32 -> 512.
- **Codebook Size:** 1024.
- **Num Quantizers:** 8.
- **Target Bandwidth:** 6 kbps.

**3. Training Loop:**
- **Optimizer:** AdamW (lr=3e-4).
- **Balancer:** You have 5 losses (Reconstruction, Codebook, Commitment, Adversarial, Feature Matching). Balancing them is an art.
    - `L_total = L_rec + 0.1 * L_adv + 1.0 * L_feat + ...`

**4. Evaluation:**
- **ViSQOL:** An objective metric for audio quality (simulates human hearing).
- **MUSHRA:** Subjective human listening test.

## Future Trends: Speech-to-Speech Translation (S2ST)

The "Holy Grail" is to translate speech without converting to text first.
**SeamlessM4T (Meta):**
- Input Audio (English) -> Encoder -> Semantic Tokens.
- Semantic Tokens -> Translator (Transformer) -> Target Semantic Tokens (French).
- Target Semantic Tokens -> Unit HiFi-GAN -> Output Audio (French).

**Why is this better?**
- It preserves **Paralinguistics** (laughter, sighs, tone).
- It handles unwritten languages (Hokkien, Swiss German).

## Appendix A: AudioLM Architecture

Google's AudioLM combines both worlds.

1.  **Semantic Tokens (w2v-BERT):** 25Hz. Captures "what" is said.
2.  **Acoustic Tokens (SoundStream):** 75Hz. Captures "how" it is said.

**Stage 1: Semantic Modeling**
- Predict the next semantic token given history. `p(S_t | S_<t)`

**Stage 2: Coarse Acoustic Modeling**
- Predict the first few layers of acoustic tokens given semantic tokens. `p(A_coarse | S)`

**Stage 3: Fine Acoustic Modeling**
- Predict the fine acoustic tokens given coarse ones. `p(A_fine | A_coarse)`

This hierarchy allows it to generate coherent speech (Stage 1) that sounds high-quality (Stage 3).

## Appendix B: Comparison of Tokenizers

| Feature | MFCC | HuBERT | EnCodec | Whisper |
| :--- | :--- | :--- | :--- | :--- |
| **Type** | Continuous | Discrete (Semantic) | Discrete (Acoustic) | Discrete (Text) |
| **Bitrate** | High | Low | Variable | Very Low |
| **Reconstruction** | Perfect | Poor (Robotic) | Perfect | Impossible (Text only) |
| **Use Case** | Old ASR | Speech Understanding | TTS / Music Gen | ASR / Translation |

## Appendix C: Python Code for RVQ

```python
import torch
import torch.nn as nn

class ResidualVQ(nn.Module):
    def __init__(self, num_quantizers, codebook_size, dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Embedding(codebook_size, dim) for _ in range(num_quantizers)
        ])
        
    def forward(self, x):
        # x: [Batch, Dim]
        residual = x
        quantized_out = 0
        indices = []
        
        for layer in self.layers:
            # Find nearest neighbor in codebook
            # (Simplified: dot product similarity)
            dists = torch.cdist(residual.unsqueeze(1), layer.weight.unsqueeze(0))
            idx = dists.argmin(dim=-1).squeeze(1)
            indices.append(idx)
            
            # Get vector
            quantized = layer(idx)
            quantized_out += quantized
            
            # Update residual
            residual = residual - quantized.detach()
            
        return quantized_out, indices
```

## Case Study: Whisper's Tokenizer

OpenAI's **Whisper** is a unique beast. It's an ASR model, but it uses a **Text Tokenizer** (Byte-Level BPE) directly on audio features? No.
It predicts text tokens from audio embeddings.

**Special Tokens:**
Whisper introduces a brilliant set of special tokens to control the model:
- `<|startoftranscript|>`
- `<|en|>` (Language ID)
- `<|transcribe|>` vs `<|translate|>` (Task ID)
- `<|notimestamps|>` vs `<|0.00|>` ... `<|30.00|>`

**Timestamp Tokens:**
Whisper quantizes time into 1500 tokens (0.02s resolution).
It interleaves text tokens with timestamp tokens:
`"Hello" <|0.00|> " world" <|0.50|>`
This allows it to do **Word-Level Alignment** implicitly.

## The Precursor: Contrastive Predictive Coding (CPC)

Before HuBERT and wav2vec 2.0, there was **CPC** (Oord et al., 2018).
It introduced the idea of **Self-Supervised Learning** for audio.

**Idea:**
1.  Split audio into segments.
2.  Encode past segments into a context vector `c_t`.
3.  Predict the *future* segments `z_{t+k}`.
4.  **Contrastive Loss:** The model must distinguish the *true* future segment from random "negative" segments drawn from other parts of the audio.

**Why it matters:**
CPC proved that you can learn high-quality audio representations without labels. HuBERT improved this by predicting *cluster IDs* instead of raw vectors, which is more stable.

## Challenges in Speech-to-Speech Translation (S2ST)

Translating speech directly to speech (without text) is the frontier.
**Challenges:**
1.  **Data Scarcity:** We have millions of hours of ASR data (Speech -> Text) and MT data (Text -> Text), but very little S2ST data (English Audio -> French Audio).
2.  **One-to-Many Mapping:** "Hello" can be said in infinite ways (happy, sad, loud, quiet). The model has to choose *one* target prosody.
3.  **Latency:** For real-time translation (Skype), we need **Streaming Tokenization**. We can't wait for the full sentence to finish.

**Solution: Unit-based Translation**
Instead of predicting audio waveforms, we predict **Discrete Units** (HuBERT/EnCodec tokens).
This turns the problem into a standard Seq2Seq translation task (like text translation), just with a larger vocabulary (1024 units vs 30k words).

## Deep Dive: HuBERT vs. wav2vec 2.0

These are the two titans of Self-Supervised Speech Learning. How do they differ?

| Feature | wav2vec 2.0 | HuBERT |
| :--- | :--- | :--- |
| **Objective** | Contrastive Loss (Identify true future) | Masked Prediction (Predict cluster ID) |
| **Targets** | Continuous Quantized Vectors | Discrete Cluster IDs (k-means) |
| **Stability** | Hard to train (Codebook collapse) | Stable (Targets are fixed offline) |
| **Performance** | Good | Better (especially for ASR) |
| **Analogy** | "Guess the sound wave" | "Guess the phoneme (cluster)" |

**Why HuBERT won:**
Predicting discrete targets (like BERT predicts words) is easier and more robust than predicting continuous vectors. It forces the model to learn "categories" of sounds rather than exact waveforms.

## Speech Resynthesis: From Tokens to Audio

We have tokens. How do we get audio back?
We need a **Vocoder** (or HiFi-GAN).

**Process:**
1.  **De-quantization:** Look up the codebook vectors for the tokens.
    - `[34, 99]` -> `[Vector_34, Vector_99]`.
2.  **Upsampling:** The tokens are at 75Hz. Audio is at 24kHz. We need to upsample by 320x.
    - Use Transposed Convolutions.
3.  **Refinement:** The raw upsampled signal is robotic.
    - Pass it through a **HiFi-GAN** generator.
    - This neural net adds the "texture" and phase information to make it sound natural.

## Latency Analysis: Streaming vs. Batch

For a real-time voice chat app (like Discord with AI voice), latency is critical.

**1. Batch Processing (Offline)**
- Wait for full sentence.
- Tokenize.
- Process.
- **Latency:** 2-5 seconds. (Unacceptable for chat).

**2. Streaming Processing (Online)**
- **Chunking:** Process audio in 20ms chunks.
- **Causal Convolutions:** The encoder can only look at *past* samples, not future ones.
    - Standard Conv: `Output[t]` depends on `Input[t-k...t+k]`.
    - Causal Conv: `Output[t]` depends on `Input[t-k...t]`.
- **Latency:** 20-40ms. (Real-time).

**Trade-off:** Causal models are slightly worse in quality because they lack future context ("I read..." -> need to know if next word is "book" (red) or "now" (reed)).

## Appendix F: The "Cocktail Party Problem" and Tokenization

Can tokenizers handle overlapping speech?
If two people speak at once, a standard VQ-VAE will produce a "mixed" token that sounds like garbage.
**Solution: Multi-Stream Tokenization.**
- Use a **Source Separation** model (like Conv-TasNet) first to split the audio into 2 streams.
- Tokenize each stream independently.
- Interleave the tokens: `[Speaker1_Token, Speaker2_Token, Speaker1_Token, ...]`.

## Conclusion

Speech Tokenization bridges the gap between Signal Processing and NLP. It allows us to throw away complex DSP pipelines and just say: "It's all tokens."

**Key Takeaways:**
1.  **Discretization** is key to applying Transformers to audio.
2.  **RVQ** allows hierarchical representation (Coarse -> Fine).
3.  **Semantic Tokens** capture meaning; **Acoustic Tokens** capture style.



---

**Originally published at:** [arunbaby.com/speech-tech/0024-speech-tokenization](https://www.arunbaby.com/speech-tech/0024-speech-tokenization/)

*If you found this helpful, consider sharing it with others who might benefit.*

