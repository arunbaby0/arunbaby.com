---
title: "Sequence-to-Sequence Speech Models"
day: 37
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
1.  **Listener (Encoder):** Pyramid LSTM processes audio.
    - Reduces time resolution (e.g., 100 frames → 25 frames).
2.  **Attender:** Attention mechanism focuses on relevant audio frames.
3.  **Speller (Decoder):** LSTM generates characters/words.

**Attention:**
$$\alpha_t = \text{softmax}(e_t)$$
$$e_{t,i} = \text{score}(s_{t-1}, h_i)$$
$$c_t = \sum_i \alpha_{t,i} h_i$$

Where:
- $s_{t-1}$: Previous decoder state.
- $h_i$: Encoder hidden state at time $i$.
- $c_t$: Context vector (weighted sum of encoder states).

## 3. TTS: Tacotron 2

**Goal:** Text → Mel-Spectrogram → Waveform.

**Architecture:**
1.  **Encoder:** Character embeddings → LSTM.
2.  **Attention:** Decoder attends to encoder states.
3.  **Decoder:** Predicts mel-spectrogram frames.
4.  **Vocoder (WaveNet/HiFi-GAN):** Mel → Waveform.

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
- Attention is $O(N^2)$ in memory.
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
```
Audio frames:  [a1][a2][a3][a4][a5]
Text:          [ h ][ e ][ l ][ l ][ o ]
Attention:     ████
               ░░░░████
                   ░░░░████
                       ░░░░████
                           ░░░░████
```

**Bad Alignment (Skipping):**
```
Audio frames:  [a1][a2][a3][a4][a5]
Text:          [ h ][ e ][ l ][ l ][ o ]
Attention:     ████
               ░░░░████
                           ░░░░████  ← Skipped a3!
```

**Fix:** Guided attention loss forces diagonal alignment early in training.

## 10. Deep Dive: Streaming ASR with Monotonic Attention

**Problem:** Standard attention looks at the entire input. Can't stream.

**Monotonic Chunkwise Attention (MoChA):**
1.  At each decoder step, decide: "Should I move to the next audio chunk?"
2.  Use a sigmoid gate: $p_{\text{move}} = \sigma(g(h_t, s_{t-1}))$
3.  If yes, attend to next chunk. If no, stay.

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
1.  **Text Normalization:** "Dr." → "Doctor", "$100" → "one hundred dollars".
2.  **Grapheme-to-Phoneme (G2P):** "read" → /riːd/ or /rɛd/ (context-dependent).
3.  **Prosody Prediction:** Predict pitch, duration, energy.
4.  **Acoustic Model:** FastSpeech 2 (non-autoregressive, parallel).
5.  **Vocoder:** HiFi-GAN.

**Streaming:**
- Generate mel-spectrogram in chunks (e.g., 50ms).
- Vocoder processes each chunk independently.

## 14. Deep Dive: Translatotron (Direct S2ST)

**Challenge:** No parallel S2ST data (Audio_Spanish → Audio_English).

**Solution:** Weak supervision.
1.  Use ASR to get Spanish text.
2.  Use MT to get English text.
3.  Use TTS to get English audio.
4.  Train Translatotron on (Audio_Spanish, Audio_English) pairs.

**Architecture:**
- **Encoder:** Processes Spanish audio.
- **Decoder:** Generates English mel-spectrogram.
- **Speaker Encoder:** Preserves source speaker's voice.

## 15. Code: Simple Seq2Seq ASR

```python
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
        encoder_out, _ = self.encoder(audio_features)  # (B, T, 2*H)
        
        # Decode
        outputs = []
        hidden = None
        
        for t in range(text_input.size(1)):
            # Compute attention
            attn_scores = self.attention(encoder_out).squeeze(-1)  # (B, T)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_out)  # (B, 1, 2*H)
            
            # Decoder input: previous char + context
            decoder_input = torch.cat([text_input[:, t:t+1], context], dim=-1)
            
            # Decoder step
            decoder_out, hidden = self.decoder(decoder_input, hidden)
            
            # Predict next char
            logits = self.fc(decoder_out)
            outputs.append(logits)
        
        return torch.cat(outputs, dim=1)
```

## 16. Production Considerations

1.  **Model Size:** Quantize to INT8 for mobile deployment.
2.  **Latency:** Use non-autoregressive models (FastSpeech, Conformer-CTC).
3.  **Personalization:** Fine-tune on user's voice (few-shot learning).
4.  **Multilingual:** Train on 100+ languages (mSLAM, Whisper).

## 17. Summary

| Task | Model | Input | Output |
| :--- | :--- | :--- | :--- |
| **ASR** | LAS, Whisper | Audio | Text |
| **TTS** | Tacotron 2 | Text | Audio |
| **ST** | Translatotron | Audio (L1) | Audio (L2) |
| **Vocoder** | HiFi-GAN | Mel-Spec | Waveform |

---

**Originally published at:** [arunbaby.com/speech-tech/0037-sequence-to-sequence-speech](https://www.arunbaby.com/speech-tech/0037-sequence-to-sequence-speech/)
