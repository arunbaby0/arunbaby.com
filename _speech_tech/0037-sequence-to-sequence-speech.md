---
title: "Sequence-to-Sequence Speech Models"
day: 37
related_dsa_day: 37
related_ml_day: 37
related_agents_day: 37
collection: speech_tech
categories:
 - speech_tech
tags:
 - seq2seq
 - attention
 - asr
 - tts
 - speech-translation
subdomain: "End-to-End Models"
tech_stack: [Tacotron, Listen-Attend-Spell, Transformer, Whisper]
scale: "100k+ hours of audio"
companies: [Google, OpenAI, Meta, Baidu]
---

**"From waveforms to words, and back again."**

## 1. The Seq2Seq Paradigm

Traditional speech systems were pipelines:
- **ASR:** Acoustic Model + Language Model + Decoder.
- **TTS:** Text Analysis + Acoustic Model + Vocoder.

**Seq2Seq** unifies this into a single neural network:
- **Input:** Sequence (audio features or text).
- **Output:** Sequence (text or audio features).
- **No hand-crafted features or rules.**

## 2. ASR: Listen, Attend, and Spell (LAS)

**Architecture:**
1. **Listener (Encoder):** Pyramid LSTM processes audio.
 - Reduces time resolution (e.g., 100 frames → 25 frames).
2. **Attender:** Attention mechanism focuses on relevant audio frames.
3. **Speller (Decoder):** LSTM generates characters/words.

**Attention:**
`\alpha_t = \text{softmax}(e_t)`
`e_{t,i} = \text{score}(s_{t-1}, h_i)`
`c_t = \sum_i \alpha_{t,i} h_i`

Where:
- `s_{t-1}`: Previous decoder state.
- `h_i`: Encoder hidden state at time `i`.
- `c_t`: Context vector (weighted sum of encoder states).

## 3. TTS: Tacotron 2

**Goal:** Text → Mel-Spectrogram → Waveform.

**Architecture:**
1. **Encoder:** Character embeddings → LSTM.
2. **Attention:** Decoder attends to encoder states.
3. **Decoder:** Predicts mel-spectrogram frames.
4. **Vocoder (WaveNet/HiFi-GAN):** Mel → Waveform.

**Key Innovation:** Predict multiple frames per step (faster).

## 4. Speech Translation: Direct S2ST

**Traditional:** Audio → ASR → Text → MT → Text → TTS → Audio.
**Direct:** Audio → Audio (no text intermediate).

**Advantages:**
- Preserves prosody (emotion, emphasis).
- Works for unwritten languages.

**Model:** Translatotron (Google).
- Encoder: Audio (Spanish).
- Decoder: Spectrogram (English).

## 5. Attention Mechanisms

### 1. Content-Based Attention
- Decoder decides where to attend based on content.
- **Problem:** Can attend to same location twice (repetition).

### 2. Location-Aware Attention
- Uses previous attention weights to guide current attention.
- Encourages monotonic progression (left-to-right).

### 3. Monotonic Attention
- Enforces strict left-to-right alignment.
- Good for streaming ASR (can't look ahead).

## 6. Challenges

### 1. Exposure Bias
- During training, decoder sees ground truth.
- During inference, decoder sees its own (possibly wrong) predictions.
- **Fix:** Scheduled sampling.

### 2. Alignment Failures
- Attention might skip words or repeat.
- **Fix:** Guided attention (force diagonal alignment early in training).

### 3. Long Sequences
- Attention is O(N^2) in memory.
- **Fix:** Chunked attention, or use CTC for ASR.

## 7. Modern Approach: Transformer-Based

**Whisper (OpenAI):**
- Encoder-Decoder Transformer.
- Trained on 680k hours.
- Handles ASR, Translation, Language ID in one model.

**Advantages:**
- Parallel training (unlike RNN).
- Better long-range dependencies.

## 8. Summary

| Task | Model | Input | Output |
| :--- | :--- | :--- | :--- |
| **ASR** | LAS, Whisper | Audio | Text |
| **TTS** | Tacotron 2 | Text | Audio |
| **ST** | Translatotron | Audio (L1) | Audio (L2) |

## 9. Deep Dive: Attention Alignment Visualization

In ASR, attention should be **monotonic** (left-to-right).

**Good Alignment:**
``
Audio frames: [a1][a2][a3][a4][a5]
Text: [ h ][ e ][ l ][ l ][ o ]
Attention: ████
 ░░░░████
 ░░░░████
 ░░░░████
 ░░░░████
``

**Bad Alignment (Skipping):**
``
Audio frames: [a1][a2][a3][a4][a5]
Text: [ h ][ e ][ l ][ l ][ o ]
Attention: ████
 ░░░░████
 ░░░░████ ← Skipped a3!
``

**Fix:** Guided attention loss forces diagonal alignment early in training.

## 10. Deep Dive: Streaming ASR with Monotonic Attention

**Problem:** Standard attention looks at the entire input. Can't stream.

**Monotonic Chunkwise Attention (MoChA):**
1. At each decoder step, decide: "Should I move to the next audio chunk?"
2. Use a sigmoid gate: `p_{\text{move}} = \sigma(g(h_t, s_{t-1}))`
3. If yes, attend to next chunk. If no, stay.

**Advantage:** Can process audio in real-time (with bounded latency).

## 11. Deep Dive: Tacotron 2 Architecture Details

**Encoder:**
- Character embeddings → 3 Conv layers → Bidirectional LSTM.
- Output: Encoded representation of text.

**Decoder:**
- **Prenet:** 2 FC layers with dropout (helps with generalization).
- **Attention RNN:** 1-layer LSTM.
- **Decoder RNN:** 2-layer LSTM.
- **Output:** Predicts 80-dim mel-spectrogram frame.

**Postnet:**
- 5 Conv layers.
- Refines the mel-spectrogram (adds high-frequency details).

**Stop Token:**
- Decoder also predicts "Should I stop?" at each step.
- Training: Sigmoid BCE loss.

## 12. Deep Dive: Vocoder Evolution

**Goal:** Mel-Spectrogram → Waveform.

**WaveNet (2016):**
- Autoregressive CNN.
- Generates one sample at a time (16kHz = 16,000 samples/sec).
- **Slow:** 1 second of audio takes 10 seconds to generate.

**WaveGlow (2018):**
- Flow-based model.
- Parallel generation (all samples at once).
- **Fast:** Real-time on GPU.

**HiFi-GAN (2020):**
- GAN-based.
- **Fastest:** 167x faster than real-time on V100.
- **Quality:** Indistinguishable from ground truth.

## 13. System Design: Low-Latency TTS

**Scenario:** Voice assistant (Alexa, Siri).

**Requirements:**
- **Latency:** < 300ms from text to first audio chunk.
- **Quality:** Natural, expressive.

**Architecture:**
1. **Text Normalization:** "Dr." → "Doctor", "$100" → "one hundred dollars".
2. **Grapheme-to-Phoneme (G2P):** "read" → /riːd/ or /rɛd/ (context-dependent).
3. **Prosody Prediction:** Predict pitch, duration, energy.
4. **Acoustic Model:** FastSpeech 2 (non-autoregressive, parallel).
5. **Vocoder:** HiFi-GAN.

**Streaming:**
- Generate mel-spectrogram in chunks (e.g., 50ms).
- Vocoder processes each chunk independently.

## 14. Deep Dive: Translatotron (Direct S2ST)

**Challenge:** No parallel S2ST data (Audio_Spanish → Audio_English).

**Solution:** Weak supervision.
1. Use ASR to get Spanish text.
2. Use MT to get English text.
3. Use TTS to get English audio.
4. Train Translatotron on (Audio_Spanish, Audio_English) pairs.

**Architecture:**
- **Encoder:** Processes Spanish audio.
- **Decoder:** Generates English mel-spectrogram.
- **Speaker Encoder:** Preserves source speaker's voice.

## 15. Code: Simple Seq2Seq ASR

``python
import torch
import torch.nn as nn

class Seq2SeqASR(nn.Module):
 def __init__(self, input_dim, hidden_dim, vocab_size):
 super().__init__()
 
 # Encoder: Bi-LSTM
 self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=3, 
 batch_first=True, bidirectional=True)
 
 # Attention
 self.attention = nn.Linear(hidden_dim * 2, 1)
 
 # Decoder: LSTM
 self.decoder = nn.LSTM(vocab_size + hidden_dim * 2, hidden_dim, 
 num_layers=2, batch_first=True)
 
 # Output projection
 self.fc = nn.Linear(hidden_dim, vocab_size)
 
 def forward(self, audio_features, text_input):
 # Encode audio
 encoder_out, _ = self.encoder(audio_features) # (B, T, 2*H)
 
 # Decode
 outputs = []
 hidden = None
 
 for t in range(text_input.size(1)):
 # Compute attention
 attn_scores = self.attention(encoder_out).squeeze(-1) # (B, T)
 attn_weights = torch.softmax(attn_scores, dim=1) # (B, T)
 context = torch.bmm(attn_weights.unsqueeze(1), encoder_out) # (B, 1, 2*H)
 
 # Decoder input: previous char + context
 decoder_input = torch.cat([text_input[:, t:t+1], context], dim=-1)
 
 # Decoder step
 decoder_out, hidden = self.decoder(decoder_input, hidden)
 
 # Predict next char
 logits = self.fc(decoder_out)
 outputs.append(logits)
 
 return torch.cat(outputs, dim=1)
``

## 16. Production Considerations

1. **Model Size:** Quantize to INT8 for mobile deployment.
2. **Latency:** Use non-autoregressive models (FastSpeech, Conformer-CTC).
3. **Personalization:** Fine-tune on user's voice (few-shot learning).
4. **Multilingual:** Train on 100+ languages (mSLAM, Whisper).

## 17. Deep Dive: The Evolution of Seq2Seq in Speech

To understand where we are, we must understand the journey.

**1. HMM-GMM (1980s-2010):**
- **Acoustic Model:** Gaussian Mixture Models (GMM) modeled the probability of audio features given a phoneme state.
- **Sequence Model:** Hidden Markov Models (HMM) modeled the transition between states.
- **Pros:** Mathematically rigorous, worked on small data.
- **Cons:** Assumed independence of frames (false), complex training pipeline.

**2. DNN-HMM (2010-2015):**
- Replaced GMM with Deep Neural Networks (DNN).
- **Impact:** Massive drop in WER (30% relative).
- **Cons:** Still relied on HMM for alignment.

**3. CTC (2006, popularized 2015):**
- **Connectionist Temporal Classification.**
- First "End-to-End" loss.
- Allowed predicting sequences shorter than input without explicit alignment.
- **Cons:** Conditional independence assumption (output at time `t` depends only on input, not previous outputs).

**4. LAS (2016):**
- **Listen-Attend-Spell.**
- Introduced Attention to speech.
- **Pros:** No independence assumption.
- **Cons:** Not streaming (needs full audio to attend).

**5. RNN-T (2012, popularized 2018):**
- **RNN Transducer.**
- Combined CTC (streaming) with LAS (label dependency).
- **Result:** The standard for streaming ASR (Pixel, Siri, Alexa).

**6. Transformer & Conformer (2017-Present):**
- Self-attention replaces RNNs.
- **Conformer:** Combines CNN (local patterns) with Transformer (global context).

## 18. Deep Dive: Connectionist Temporal Classification (CTC)

**Problem:** Audio has `T` frames (e.g., 1000). Text has `L` characters (e.g., 50). `T \gg L`. How do we map them?

**CTC Solution:**
- Introduce a **blank token** `\epsilon`.
- Output sequence length is `T`.
- **Collapse:** Remove repeats and blanks.
 - `h h e e l l l l o` → `hello`
 - `h h \epsilon e e l l \epsilon l l o` → `hello`

**Loss Function:**
`P(Y|X) = \sum_{A \in \mathcal{B}^{-1}(Y)} P(A|X)`
- Sum over all valid alignments `A` that collapse to `Y`.
- Computed efficiently using **Dynamic Programming** (Forward-Backward algorithm).

**Pros:**
- Fast inference (O(1) per step).
- Streaming friendly.

**Cons:**
- Can't model language (e.g., "pair" vs "pear" sounds same). Needs external Language Model.

## 19. Deep Dive: RNN-Transducer (RNN-T)

**Architecture:**
1. **Encoder (Audio):** Processes audio frames `x_t`. Produces `h_t^{enc}`.
2. **Prediction Network (Text):** Processes previous non-blank output `y_{u-1}`. Produces `h_u^{pred}`. (Like an LM).
3. **Joint Network:** Combines them.
 `z_{t,u} = \text{ReLU}(W_{enc} h_t^{enc} + W_{pred} h_u^{pred})`
 `P(y|t,u) = \text{softmax}(W_{out} z_{t,u})`

**Inference (Greedy):**
- If output is non-blank (`y`), feed to Prediction Network, increment `u`. Stay at audio frame `t`.
- If output is blank (`\epsilon`), move to next audio frame `t+1`.

**Why it wins:**
- **Streaming:** Encoder is causal.
- **Accuracy:** Prediction network models label dependencies (unlike CTC).
- **Latency:** Controllable.

## 20. Deep Dive: Transformer vs Conformer

**Transformer:**
- Great at global context (Self-Attention).
- Bad at local fine-grained details (needs deep layers to see local patterns).
- Positional encodings are brittle for varying audio lengths.

**Conformer (Convolution-augmented Transformer):**
- **Macaron Style:** Feed-Forward -> Multi-Head Attention -> Conv Module -> Feed-Forward.
- **Conv Module:** Pointwise Conv -> Gated Linear Unit (GLU) -> Depthwise Conv -> BatchNorm -> Swish -> Pointwise Conv.
- **Why:**
 - **CNNs** capture local features (formants, transitions).
 - **Transformers** capture global semantics.
 - **Result:** SOTA on LibriSpeech.

## 21. Deep Dive: FastSpeech 2 (Non-Autoregressive TTS)

**Problem with Tacotron 2:**
- Autoregressive (slow).
- Unstable attention (skipping/repeating).
- Hard to control prosody (speed, pitch).

**FastSpeech 2 Architecture:**
1. **Encoder:** Feed-Forward Transformer.
2. **Variance Adaptor:**
 - **Duration Predictor:** How many frames does this phoneme last? (Trained on forced alignment).
 - **Pitch Predictor:** Predicts F0 contour.
 - **Energy Predictor:** Predicts volume.
 - Adds embeddings of these predictions to the encoder output.
3. **Length Regulator:** Expands hidden states based on duration (e.g., "a" lasts 5 frames -> repeat vector 5 times).
4. **Decoder:** Feed-Forward Transformer.

**Pros:**
- **Fast:** Parallel generation.
- **Robust:** No attention failures.
- **Controllable:** Can explicitly set "speak 1.2x faster" or "raise pitch".

## 22. Deep Dive: VITS (Conditional Variational Autoencoder)

**VITS (Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech)** is the current SOTA for E2E TTS.

**Key Idea:** Combine Acoustic Model and Vocoder into one flow-based model.

**Architecture:**
- **Posterior Encoder:** Encodes linear spectrogram into latent `z`.
- **Prior Encoder:** Predicts distribution of `z` from text (conditioned on alignment).
- **Decoder (Generator):** Transforms `z` into waveform (HiFi-GAN style).
- **Stochastic Duration Predictor:** Models duration uncertainty.

**Training:**
- **Reconstruction Loss:** Mel-spectrogram L1 loss.
- **KL Divergence:** Between Posterior and Prior.
- **Adversarial Loss:** Discriminator tries to distinguish real vs generated audio.

**Result:** extremely natural, high-fidelity speech.

## 23. System Design: Real-Time Speech Translation System

**Scenario:** "Universal Translator" device. English Audio -> Spanish Audio.

**Constraints:**
- **Latency:** < 2 seconds lag.
- **Compute:** Edge device (limited).

**Architecture Choices:**

**Option A: Cascade (ASR -> MT -> TTS)**
- **Pros:** Modular. Can use SOTA for each.
- **Cons:** Error propagation. Latency adds up. Loss of paralinguistics (tone).

**Option B: Direct S2ST (Audio -> Audio)**
- **Pros:** Fast. Preserves tone.
- **Cons:** Data scarcity. Hard to train.

**Hybrid Design (Production):**
1. **Streaming ASR (RNN-T):** Generates English text stream.
2. **Streaming MT:** Translates partial sentences. "Hello" -> "Hola".
3. **Incremental TTS:** Generates audio for "Hola" immediately.

**Wait-k Policy:**
- MT model waits for `k` words before translating.
- Balances context (accuracy) vs latency.

## 24. Case Study: OpenAI Whisper

**Goal:** Robust ASR that works on "in the wild" audio.

**Data:**
- 680,000 hours of web audio.
- **Weak Supervision:** Transcripts from ASR systems, subtitles (noisy).
- **Multitask:**
 - English Transcription.
 - Any-to-English Translation.
 - Language Identification.
 - Voice Activity Detection.

**Architecture:**
- Standard Encoder-Decoder Transformer.
- **Input:** Log-Mel Spectrogram (30 seconds).
- **Output:** Text tokens.

**Key Features:**
- **Task Tokens:** `<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`.
- **Long-form:** Processes 30s chunks. Uses previous chunk's text as prompt (context).

**Impact:**
- Zero-shot performance on many datasets matches supervised models.
- Proved that **Data Scale > Model Architecture**.

## 25. Case Study: Google USM (Universal Speech Model)

**Goal:** One model for 300+ languages.

**Architecture:**
- **Conformer** (2 Billion parameters).
- **MOST (Multi-Objective Supervised Training):**
 - **BEST-RQ:** Self-supervised loss (BERT-style on audio).
 - **Text-Injection:** Train on text-only data (using shared encoder layers).
 - **ASR:** Supervised loss on labeled audio.

**Result:**
- SOTA on 73 languages.
- Enables ASR for languages with < 10 hours of data.

## 26. Deep Dive: Evaluation Metrics

**ASR:**
- **WER (Word Error Rate):** `\frac{S + D + I}{N}`
 - S: Substitutions, D: Deletions, I: Insertions, N: Total words.
- **CER (Character Error Rate):** For languages without spaces (Chinese).
- **RTF (Real-Time Factor):** `\frac{\text{Processing Time}}{\text{Audio Duration}}`. RTF < 1 means real-time.

**TTS:**
- **MOS (Mean Opinion Score):** Humans rate naturalness 1-5.
- **MCD (Mel Cepstral Distortion):** Objective distance between generated and ground truth spectrograms.

**ST (Speech Translation):**
- **BLEU:** Standard MT metric.

## 27. Code: Implementing Beam Search for Seq2Seq

Greedy decoding (pick max prob) is suboptimal. Beam search explores `k` paths.

``python
def beam_search_decoder(model, encoder_out, beam_width=3, max_len=50):
 # Start with [SOS] token
 # Beam: list of (sequence, score, hidden_state)
 start_token = model.vocab['<sos>']
 beam = [([start_token], 0.0, None)]
 
 for _ in range(max_len):
 candidates = []
 
 for seq, score, hidden in beam:
 if seq[-1] == model.vocab['<eos>']:
 candidates.append((seq, score, hidden))
 continue
 
 # Predict next token
 last_token = torch.tensor([[seq[-1]]])
 output, new_hidden = model.decoder(last_token, hidden, encoder_out)
 log_probs = torch.log_softmax(output, dim=-1)
 
 # Get top k
 topk_probs, topk_ids = log_probs.topk(beam_width)
 
 for i in range(beam_width):
 token = topk_ids[0][i].item()
 prob = topk_probs[0][i].item()
 candidates.append((seq + [token], score + prob, new_hidden))
 
 # Select top k candidates
 beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
 
 # Check if all finished
 if all(c[0][-1] == model.vocab['<eos>'] for c in beam):
 break
 
 return beam[0][0] # Return best sequence
``

## 28. Production: Serving Speech Models with Triton

**NVIDIA Triton Inference Server** is standard for deploying speech models.

**Pipeline:**
1. **Feature Extractor (Python Backend):** Audio -> Mel-Spec.
2. **Encoder (TensorRT):** Mel-Spec -> Hidden States.
3. **Decoder (TensorRT):** Hidden States -> Text (Beam Search).

**Dynamic Batching:**
- Triton groups requests arriving within 5ms into a batch.
- Increases GPU utilization significantly.

**Ensemble Model:**
- Define a DAG of models.
- Client sends audio, Triton handles the flow.

## 29. Deep Dive: End-to-End Speech-to-Speech Translation (S2ST)

**Unit-Based S2ST:**
- Instead of spectrograms, predict **discrete acoustic units** (from HuBERT or k-means).
- **Advantage:** Discrete tokens allow using standard Transformer (like NLP).
- **Vocoder:** Unit HiFi-GAN converts discrete units to waveform.

**SpeechMatrix:**
- Mining parallel speech from 100k hours of multilingual audio.
- Uses LASER embeddings to find matching sentences in different languages.

## 30. Summary

| Task | Model | Input | Output |
| :--- | :--- | :--- | :--- |
| **ASR** | LAS, Whisper | Audio | Text |
| **TTS** | Tacotron 2 | Text | Audio |
| **ST** | Translatotron | Audio (L1) | Audio (L2) |
| **Vocoder** | HiFi-GAN | Mel-Spec | Waveform |
| **Streaming** | RNN-T | Audio Stream | Text Stream |
| **Fast TTS** | FastSpeech 2 | Text | Mel-Spec |

| **Streaming** | RNN-T | Audio Stream | Text Stream |
| **Fast TTS** | FastSpeech 2 | Text | Mel-Spec |

## 31. Deep Dive: Self-Supervised Learning (SSL) in Speech

**Problem:** Labeled data (audio + text) is expensive. Unlabeled audio is free.

**Wav2Vec 2.0 (Meta):**
- **Idea:** Mask parts of the audio latent space and predict the quantized representation of the masked part.
- **Contrastive Loss:** Identify the correct quantized vector among distractors.
- **Result:** Can reach SOTA with only 10 minutes of labeled data (after pre-training on 53k hours).

**HuBERT (Hidden Unit BERT):**
- **Idea:** Offline clustering (k-means) of MFCCs to generate pseudo-labels.
- **Masked Prediction:** BERT-style objective. Predict the cluster ID of masked frames.
- **Result:** More robust than Wav2Vec 2.0.

**Impact on Seq2Seq:**
- Initialize the Encoder with Wav2Vec 2.0 / HuBERT.
- Add a random Decoder.
- Fine-tune on ASR task.
- **Benefit:** Drastic reduction in required labeled data.

## 32. Deep Dive: Multilingual Seq2Seq

**Goal:** One model for 100 languages.

**Challenges:**
- **Data Imbalance:** English has 1M hours, Swahili has 100.
- **Script Diversity:** Latin, Cyrillic, Chinese, Arabic.
- **Phonetic Overlap:** "P" in English `\neq` "P" in French.

**Solutions:**
1. **Language ID Token:** `<|en|>`, `<|fr|>`.
2. **Shared Vocabulary:** SentencePiece (BPE) trained on all languages.
3. **Balancing Sampling:** Up-sample low-resource languages during training.
 `p_l \propto (\frac{N_l}{N_{total}})^\alpha`
 where `\alpha < 1` (e.g., 0.3) flattens the distribution.
4. **Adapter Modules:** Small, language-specific layers inserted into the frozen giant model.

## 33. Deep Dive: End-to-End SLU (Spoken Language Understanding)

**Traditional:** Audio → ASR → Text → NLP → Intent/Slots.
**E2E SLU:** Audio → Intent/Slots.

**Why E2E?**
- **Error Robustness:** ASR might transcribe "play jazz" as "play jas". NLP fails. E2E model learns acoustic features of "jazz".
- **Paralinguistics:** "Yeah right" (sarcastic) -> Negative Sentiment. Text-only NLP misses this.

**Architecture:**
- **Encoder:** Pre-trained Wav2Vec 2.0.
- **Decoder:** Predicts semantic frame directly.
 - `[INTENT: PlayMusic] [ARTIST: The Beatles] [GENRE: Rock]`

**Challenges:**
- Lack of labeled Audio-to-Semantics data.
- **Solution:** Transfer learning from ASR models.

## 34. Deep Dive: Attention Visualization Code

Visualizing attention maps is the best way to debug Seq2Seq models.

``python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_matrix, input_text, output_text):
 """
 attention_matrix: (Output_Len, Input_Len) numpy array
 """
 plt.figure(figsize=(10, 8))
 sns.heatmap(attention_matrix, cmap='viridis', 
 xticklabels=input_text, yticklabels=output_text)
 plt.xlabel('Encoder Input')
 plt.ylabel('Decoder Output')
 plt.show()

# Example usage
# attn = model.get_attention_weights(...)
# plot_attention(attn, audio_frames, predicted_words)
``

**What to look for:**
- **Diagonal:** Good alignment.
- **Vertical Line:** Decoder is stuck on one frame (repeating output).
- **Horizontal Line:** Encoder frame is ignored.
- **Fuzzy:** Weak attention (low confidence).

## 35. Deep Dive: Handling Long-Form Audio

Standard Transformers have O(N^2) attention complexity. 30s audio = 1500 frames. 1 hour = 180,000 frames.

**Strategies:**
1. **Chunking (Whisper):**
 - Slice audio into 30s segments.
 - Transcribe independently.
 - **Problem:** Context cut off at boundaries.
 - **Fix:** Pass previous segment's text as prompt.

2. **Streaming (RNN-T):**
 - Process frame-by-frame. Infinite length.
 - **Problem:** No future context.

3. **Sparse Attention (BigBird / Longformer):**
 - Attend only to local window + global tokens.
 - O(N) complexity.

4. **Block-Processing (Emformer):**
 - Block-wise processing with memory bank for history.

## 36. Future Trends: Audio-Language Models

**SpeechGPT / AudioLM:**
- Treat audio tokens and text tokens as the same thing.
- **Tokenizer:** SoundStream / EnCodec (Neural Audio Codec).
- **Model:** Decoder-only Transformer (GPT).
- **Training:**
 - Text-only data (Web).
 - Audio-only data (Radio).
 - Paired data (ASR).

**Capabilities:**
- **Speech-to-Speech Translation:** "Translate this to French" (Audio input) -> Audio output.
- **Voice Continuation:** Continue speaking in the user's voice.
- **Zero-shot TTS:** "Say 'Hello' in this voice: [Audio Prompt]".

## 37. Ethical Considerations: Voice Cloning & Deepfakes

Seq2Seq TTS (Vall-E) can clone a voice with 3 seconds of audio.

**Risks:**
- **Fraud:** Impersonating CEOs or relatives.
- **Disinformation:** Fake speeches by politicians.
- **Harassment:** Fake audio of individuals.

**Mitigation:**
- **Watermarking:** Embed inaudible signals in generated audio.
- **Detection:** Train classifiers to detect artifacts of synthesis.
- **Regulation:** "Know Your Customer" for TTS APIs.

- **Regulation:** "Know Your Customer" for TTS APIs.

## 38. Deep Dive: The Mathematics of Transformers in Speech

Why did Transformers replace RNNs?

**1. Self-Attention Complexity:**
- **RNN:** O(N) sequential operations. Cannot parallelize.
- **Transformer:** O(1) sequential operations (parallelizable). O(N^2) memory.
- **Benefit:** We can train on 1000 GPUs efficiently.

**2. Positional Encodings in Speech:**
- Text uses absolute sinusoidal encodings.
- **Speech Problem:** Audio length varies wildly. 10s vs 10m.
- **Solution:** **Relative Positional Encoding.**
 - Instead of adding `P_i` to input `X_i`, add a bias `b_{i-j}` to the attention score `A_{ij}`.
 - Allows the model to generalize to audio lengths unseen during training.

**3. Subsampling:**
- Audio is high-frequency (100 frames/sec). Text is low-frequency (3 chars/sec).
- **Conv Subsampling:** First 2 layers of Encoder are strided Convolutions (stride 2x2 = 4x reduction).
- Reduces sequence length `N \to N/4`. Reduces attention cost `N^2 \to (N/4)^2 = N^2/16`.

## 39. Deep Dive: Troubleshooting Seq2Seq Models

**1. The "Hallucination" Problem:**
- **Symptom:** Model outputs "Thank you Thank you Thank you" during silence.
- **Cause:** Decoder language model is too strong; it predicts likely text even without acoustic evidence.
- **Fix:**
 - **Voice Activity Detection (VAD):** Filter out silence.
 - **Coverage Penalty:** Penalize attending to the same frames repeatedly.

**2. The "NaN" Loss:**
- **Symptom:** Training crashes.
- **Cause:** Exploding gradients in LSTM or division by zero in BatchNorm.
- **Fix:**
 - Gradient Clipping (norm 1.0).
 - Warmup learning rate.
 - Check for empty transcripts in data.

**3. The "Babble" Problem:**
- **Symptom:** Output is gibberish.
- **Cause:** CTC alignment failed or Attention didn't converge.
- **Fix:**
 - **Curriculum Learning:** Train on short utterances first, then long.
 - **Guided Attention Loss:** Force diagonal alignment for first few epochs.

- **Guided Attention Loss:** Force diagonal alignment for first few epochs.

## 40. Deep Dive: The Future - Audio-Language Models

The boundary between Speech and NLP is blurring.

**AudioLM (Google):**
- Treats audio as a sequence of discrete tokens (using SoundStream codec).
- Uses a GPT-style decoder to generate audio tokens.
- **Capabilities:**
 - **Speech Continuation:** Given 3s of speech, continue speaking in the same voice and style.
 - **Zero-Shot TTS:** Generate speech from text in a target voice without fine-tuning.

**SpeechGPT (Fudan University):**
- Fine-tunes LLaMA on paired speech-text data.
- **Modality Adaptation:** Teaches the LLM to understand discrete audio tokens.
- **Result:** A chatbot you can talk to, which talks back with emotion and nuance.

**Implication:**
- Seq2Seq models (Encoder-Decoder) might be replaced by Decoder-only LLMs that handle all modalities (Text, Audio, Image) in a unified token space.

## 41. Ethical Considerations in Seq2Seq Speech

**1. Deepfakes & Voice Cloning:**
- Models like Vall-E can clone a voice from 3 seconds of audio.
- **Risk:** Fraud (fake CEO calls), harassment, disinformation.
- **Mitigation:**
 - **Watermarking:** Embed inaudible signals in generated audio.
 - **Detection:** Train classifiers to detect synthesis artifacts.

**2. Bias in ASR:**
- Models trained on LibriSpeech (audiobooks) fail on AAVE (African American Vernacular English) or Indian accents.
- **Fix:** Diverse training data. "Fairness-aware" loss functions that penalize disparity between groups.

**3. Privacy:**
- Smart speakers listen constantly.
- **Fix:** **On-Device Processing.** Run the Seq2Seq model on the phone's NPU, never sending audio to the cloud.

## 42. Further Reading

To master Seq2Seq speech models, these papers are essential:

1. **"Listen, Attend and Spell" (Chan et al., 2016):** The paper that introduced Attention to ASR.
2. **"Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" (Shen et al., 2018):** The Tacotron 2 paper.
3. **"Attention Is All You Need" (Vaswani et al., 2017):** The Transformer paper (foundation of modern speech).
4. **"Conformer: Convolution-augmented Transformer for Speech Recognition" (Gulati et al., 2020):** The current SOTA architecture for ASR.
5. **"Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (Baevski et al., 2020):** The SSL revolution.
6. **"Whisper: Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022):** How scale beats architecture.

## 43. Summary

| Task | Model | Input | Output |
| :--- | :--- | :--- | :--- |
| **ASR** | LAS, Whisper | Audio | Text |
| **TTS** | Tacotron 2 | Text | Audio |
| **ST** | Translatotron | Audio (L1) | Audio (L2) |
| **Vocoder** | HiFi-GAN | Mel-Spec | Waveform |
| **Streaming** | RNN-T | Audio Stream | Text Stream |
| **Fast TTS** | FastSpeech 2 | Text | Mel-Spec |
| **SSL** | Wav2Vec 2.0 | Masked Audio | Quantized Vector |
| **E2E SLU** | SLU-BERT | Audio | Intent/Slots |
| **Troubleshooting** | VAD, Gradient Clipping | - | - |
| **Future** | AudioLM | Discrete Tokens | Discrete Tokens |

---

**Originally published at:** [arunbaby.com/speech-tech/0037-sequence-to-sequence-speech](https://www.arunbaby.com/speech-tech/0037-sequence-to-sequence-speech/)
