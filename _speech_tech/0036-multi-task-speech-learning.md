---
title: "Multi-task Speech Learning"
day: 36
collection: speech_tech
categories:
  - speech_tech
tags:
  - multi-task-learning
  - asr
  - translation
  - slu
  - whisper
subdomain: "Modeling"
tech_stack: [PyTorch, Transformer, Whisper, UniSpeech]
scale: "100k+ hours of audio"
companies: [OpenAI, Meta, Google, Microsoft]
---

**"One model to rule them all: ASR, Translation, and Understanding."**

## 1. The Concept: Why Multi-task?

Traditionally, we built separate models:
1.  **ASR:** Audio -> Text.
2.  **ST (Speech Translation):** Audio -> Foreign Text.
3.  **SLU (Spoken Language Understanding):** Audio -> Intent/Slots.
4.  **VAD:** Audio -> Speech/Silence.

**Multi-task Learning (MTL)** trains a single model to perform multiple tasks simultaneously.

**Benefits:**
-   **Regularization:** Learning to translate helps the model understand semantics, which improves ASR.
-   **Data Efficiency:** "Low-resource" tasks (e.g., Swahili ASR) benefit from "High-resource" tasks (e.g., English ASR) via shared representations.
-   **Simplified Deployment:** Deploy one model instead of four.

## 2. Architectures

### 1. Shared Encoder, Separate Decoders
-   **Encoder:** Processes audio (Spectrogram -> Hidden States). Shared across all tasks.
-   **Decoder A:** ASR (predicts English tokens).
-   **Decoder B:** ST (predicts French tokens).
-   **Decoder C:** SLU (predicts Intent labels).
-   **Pros:** Specialized output heads.
-   **Cons:** Increases parameter count with each new task.

### 2. Token-Based Task Specification (The "Whisper" Way)
-   **Single Encoder-Decoder Transformer.**
-   **Task Tokens:** The first token fed to the decoder tells it what to do.
    -   `<|transcribe|>` -> Output English text.
    -   `<|translate|>` -> Output French text.
    -   `<|timestamps|>` -> Output time-aligned text.
-   **Pros:** Extremely flexible. Zero-shot transfer.
-   **Cons:** Balancing tasks during training is tricky.

## 3. Training Strategies

### 1. Loss Weighting
$$L_{total} = \lambda_1 L_{ASR} + \lambda_2 L_{ST} + \lambda_3 L_{SLU}$$
-   **Challenge:** If $L_{ASR}$ is large, the model ignores ST.
-   **Solution:** Dynamic Weight Averaging (DWA) or Uncertainty Weighting (learn $\lambda$ as parameters).

### 2. Gradient Surgery
-   **Problem:** Task A wants to move weights "Left", Task B wants "Right". Gradients conflict.
-   **PCGrad (Project Conflicting Gradients):** If gradients point in opposite directions, project one onto the normal plane of the other. Removes destructive interference.

### 3. Curriculum Learning
-   Start with easy tasks (ASR).
-   Gradually introduce hard tasks (Translation).

## 4. Deep Dive: OpenAI Whisper

**Whisper** is the ultimate example of Multi-task Speech Learning.

-   **Tasks:**
    1.  English Transcription.
    2.  Any-to-English Translation.
    3.  Language Identification.
    4.  Voice Activity Detection (timestamp prediction).
-   **Architecture:** Standard Transformer Encoder-Decoder.
-   **Input:** Log-Mel Spectrogram (30 seconds).
-   **Decoder Prompt:**
    `[<|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>]`

**Key Insight:** By training on 680k hours of weak supervision (internet audio), the model learns robust representations that generalize across tasks.

## 5. Deep Dive: UniSpeech (Microsoft)

**UniSpeech** combines:
1.  **Self-Supervised Learning (SSL):** Contrastive loss on unlabeled audio (like wav2vec 2.0).
2.  **Supervised Learning:** ASR/ST loss on labeled data.

**Unified Representation:**
-   Forces the model to learn phonetically rich representations (for ASR) and semantically rich representations (for Translation).

## 6. System Design: Unified Speech API

**Scenario:** Build an API that takes audio and returns Transcript + Sentiment + Language.

**Approach 1: Pipeline**
`Audio -> ASR Model -> Text -> Sentiment Model -> Label`
-   **Latency:** Sum of latencies.
-   **Error Propagation:** If ASR fails, Sentiment fails.

**Approach 2: Multi-task Model**
`Audio -> [Shared Encoder] -> [ASR Head, Sentiment Head, LID Head]`
-   **Latency:** Encoder (expensive) runs once. Heads (cheap) run in parallel.
-   **Robustness:** Sentiment head works directly on audio features (prosody, tone), not just text.

## 7. Challenges

1.  **Catastrophic Forgetting:** Fine-tuning on Task B makes it forget Task A.
    -   **Fix:** Replay buffers (mix old data with new).
2.  **Negative Transfer:** Task A hurts Task B.
    -   Example: Speaker Identification (needs speaker info) vs. ASR (needs to ignore speaker info).
    -   **Fix:** Task-specific adapters.

## 8. Summary

| Feature | Single-Task | Multi-Task |
| :--- | :--- | :--- |
| **Performance** | High on specific task | High on all (usually) |
| **Data Req** | High labeled data | Can leverage auxiliary data |
| **Deployment** | N models | 1 model |
| **Training** | Simple | Complex (balancing) |

## 9. Deep Dive: Gradient Surgery (PCGrad)

When training on Task A (ASR) and Task B (Translation), the gradients might conflict.
-   $\nabla L_A$ says "increase weight $w$".
-   $\nabla L_B$ says "decrease weight $w$".
-   Result: They cancel out, or the model oscillates.

**PCGrad Algorithm:**
1.  Compute gradients $g_A$ and $g_B$ independently.
2.  Check cosine similarity: if $g_A \cdot g_B < 0$ (angle > 90 degrees), they conflict.
3.  Project $g_A$ onto the normal plane of $g_B$:
    $$g_A' = g_A - \frac{g_A \cdot g_B}{\|g_B\|^2} g_B$$
4.  Do the same for $g_B$.
5.  Update weights with $g_A' + g_B'$.

**Effect:** The optimization trajectory follows a "zigzag" path that satisfies both tasks without destructive interference.

## 10. Deep Dive: Uncertainty Weighting

How do we set $\lambda_1, \lambda_2$ in the loss function?
$$L = \lambda_1 L_{ASR} + \lambda_2 L_{ST}$$

**Homoscedastic Uncertainty:**
-   Assume the task noise is constant.
-   Learn $\sigma_1, \sigma_2$ (variance) as trainable parameters.
-   Loss becomes:
    $$L = \frac{1}{2\sigma_1^2} L_{ASR} + \frac{1}{2\sigma_2^2} L_{ST} + \log \sigma_1 + \log \sigma_2$$
-   If a task is noisy (high loss), the model increases $\sigma$ to reduce its weight.
-   **Result:** Automatic, dynamic balancing.

## 11. Deep Dive: Adapter Modules

Fine-tuning a massive Multi-task model (like Whisper) for a new task is expensive.
**Adapters** allow efficient transfer learning.

**Architecture:**
-   Freeze the pre-trained Transformer layers.
-   Insert small "Adapter" layers between Feed-Forward and Self-Attention blocks.
-   Adapter = `Linear(d -> d/r) -> ReLU -> Linear(d/r -> d)`.
-   Only train the Adapters (few parameters).

**Task-Specific Adapters:**
-   Train one set of adapters for "Medical ASR".
-   Train another set for "Legal ASR".
-   Switch adapters at runtime based on user domain.

## 12. Deep Dive: Whisper's Weak Supervision Strategy

Most ASR models are trained on LibriSpeech (clean, read audio). They fail on real-world noise.
Whisper trained on **680,000 hours** of internet audio.

**Data Filtering:**
1.  **Language ID:** Discard if audio language doesn't match transcript language.
2.  **No Speech:** Discard if VAD detects silence.
3.  **Machine Generated:** Discard if transcript looks like output of another ASR system (to avoid learning errors).

**Result:**
-   Whisper is not SOTA on LibriSpeech (Clean).
-   But it is **SOTA on Robustness** (Accents, Noise, Music).

## 13. Code: PyTorch Multi-task Model

A simple implementation of a shared encoder with multiple heads.

```python
import torch
import torch.nn as nn

class MultiTaskSpeechModel(nn.Module):
    def __init__(self, input_dim, vocab_size, num_intents):
        super().__init__()
        
        # Shared Encoder (e.g., Conformer or LSTM)
        self.encoder = nn.LSTM(input_dim, 512, num_layers=3, batch_first=True, bidirectional=True)
        
        # Task 1: ASR (CTC Head)
        self.asr_head = nn.Linear(1024, vocab_size)
        
        # Task 2: Intent Classification (SLU Head)
        # Use the last hidden state for classification
        self.intent_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_intents)
        )
        
    def forward(self, x):
        # x: (Batch, Time, Feats)
        encoder_out, (h_n, c_n) = self.encoder(x)
        
        # ASR Output: (Batch, Time, Vocab)
        asr_logits = self.asr_head(encoder_out)
        
        # Intent Output: (Batch, Num_Intents)
        # Pool over time (e.g., take last state or mean)
        # Here taking mean of encoder outputs
        context = torch.mean(encoder_out, dim=1)
        intent_logits = self.intent_head(context)
        
        return asr_logits, intent_logits

# Loss Calculation
def compute_loss(asr_logits, asr_targets, intent_logits, intent_targets):
    loss_asr = nn.CTCLoss()(asr_logits.transpose(0, 1), asr_targets, ...)
    loss_intent = nn.CrossEntropyLoss()(intent_logits, intent_targets)
    
    # Simple weighting
    return loss_asr + 0.5 * loss_intent
```

## 14. System Design: Real-Time Translation (Babelfish)

**Scenario:** Build a "Universal Translator" device.
**Input:** Audio (Spanish). **Output:** Audio (English).

**Pipeline:**
1.  **VAD:** Detect speech.
2.  **S2ST Model (Speech-to-Speech Translation):**
    -   **Direct S2ST (Translatotron 2):** Audio -> Spectrogram. No text intermediate.
    -   **Cascaded:** ASR -> MT -> TTS.
3.  **Streaming:**
    -   Use **Wait-k Policy**: Wait for $k$ words before translating.
    -   Trade-off: Latency vs. Context (Accuracy).

## 15. Summary

| Feature | Single-Task | Multi-Task |
| :--- | :--- | :--- |
| **Performance** | High on specific task | High on all (usually) |
| **Data Req** | High labeled data | Can leverage auxiliary data |
| **Deployment** | N models | 1 model |
| **Training** | Simple | Complex (balancing) |
| **Optimization** | Standard SGD | PCGrad, Uncertainty Weighting |

---

**Originally published at:** [arunbaby.com/speech-tech/0036-multi-task-speech-learning](https://www.arunbaby.com/speech-tech/0036-multi-task-speech-learning/)
