---
title: "Multi-task Speech Learning"
day: 36
related_dsa_day: 36
related_ml_day: 36
related_agents_day: 36
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
1. **ASR:** Audio -> Text.
2. **ST (Speech Translation):** Audio -> Foreign Text.
3. **SLU (Spoken Language Understanding):** Audio -> Intent/Slots.
4. **VAD:** Audio -> Speech/Silence.

**Multi-task Learning (MTL)** trains a single model to perform multiple tasks simultaneously.

**Benefits:**
- **Regularization:** Learning to translate helps the model understand semantics, which improves ASR.
- **Data Efficiency:** "Low-resource" tasks (e.g., Swahili ASR) benefit from "High-resource" tasks (e.g., English ASR) via shared representations.
- **Simplified Deployment:** Deploy one model instead of four.

## 2. Architectures

### 1. Shared Encoder, Separate Decoders
- **Encoder:** Processes audio (Spectrogram -> Hidden States). Shared across all tasks.
- **Decoder A:** ASR (predicts English tokens).
- **Decoder B:** ST (predicts French tokens).
- **Decoder C:** SLU (predicts Intent labels).
- **Pros:** Specialized output heads.
- **Cons:** Increases parameter count with each new task.

### 2. Token-Based Task Specification (The "Whisper" Way)
- **Single Encoder-Decoder Transformer.**
- **Task Tokens:** The first token fed to the decoder tells it what to do.
 - `<|transcribe|>` -> Output English text.
 - `<|translate|>` -> Output French text.
 - `<|timestamps|>` -> Output time-aligned text.
- **Pros:** Extremely flexible. Zero-shot transfer.
- **Cons:** Balancing tasks during training is tricky.

## 3. Training Strategies

### 1. Loss Weighting
`L_{total} = \lambda_1 L_{ASR} + \lambda_2 L_{ST} + \lambda_3 L_{SLU}`
- **Challenge:** If `L_{ASR}` is large, the model ignores ST.
- **Solution:** Dynamic Weight Averaging (DWA) or Uncertainty Weighting (learn `\lambda` as parameters).

### 2. Gradient Surgery
- **Problem:** Task A wants to move weights "Left", Task B wants "Right". Gradients conflict.
- **PCGrad (Project Conflicting Gradients):** If gradients point in opposite directions, project one onto the normal plane of the other. Removes destructive interference.

### 3. Curriculum Learning
- Start with easy tasks (ASR).
- Gradually introduce hard tasks (Translation).

## 4. Deep Dive: OpenAI Whisper

**Whisper** is the ultimate example of Multi-task Speech Learning.

- **Tasks:**
 1. English Transcription.
 2. Any-to-English Translation.
 3. Language Identification.
 4. Voice Activity Detection (timestamp prediction).
- **Architecture:** Standard Transformer Encoder-Decoder.
- **Input:** Log-Mel Spectrogram (30 seconds).
- **Decoder Prompt:**
 `[<|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>]`

**Key Insight:** By training on 680k hours of weak supervision (internet audio), the model learns robust representations that generalize across tasks.

## 5. Deep Dive: UniSpeech (Microsoft)

**UniSpeech** combines:
1. **Self-Supervised Learning (SSL):** Contrastive loss on unlabeled audio (like wav2vec 2.0).
2. **Supervised Learning:** ASR/ST loss on labeled data.

**Unified Representation:**
- Forces the model to learn phonetically rich representations (for ASR) and semantically rich representations (for Translation).

## 6. System Design: Unified Speech API

**Scenario:** Build an API that takes audio and returns Transcript + Sentiment + Language.

**Approach 1: Pipeline**
`Audio -> ASR Model -> Text -> Sentiment Model -> Label`
- **Latency:** Sum of latencies.
- **Error Propagation:** If ASR fails, Sentiment fails.

**Approach 2: Multi-task Model**
`Audio -> [Shared Encoder] -> [ASR Head, Sentiment Head, LID Head]`
- **Latency:** Encoder (expensive) runs once. Heads (cheap) run in parallel.
- **Robustness:** Sentiment head works directly on audio features (prosody, tone), not just text.

## 7. Challenges

1. **Catastrophic Forgetting:** Fine-tuning on Task B makes it forget Task A.
 - **Fix:** Replay buffers (mix old data with new).
2. **Negative Transfer:** Task A hurts Task B.
 - Example: Speaker Identification (needs speaker info) vs. ASR (needs to ignore speaker info).
 - **Fix:** Task-specific adapters.

## 8. Summary

| Feature | Single-Task | Multi-Task |
| :--- | :--- | :--- |
| **Performance** | High on specific task | High on all (usually) |
| **Data Req** | High labeled data | Can leverage auxiliary data |
| **Deployment** | N models | 1 model |
| **Training** | Simple | Complex (balancing) |

## 9. Deep Dive: Gradient Surgery (PCGrad)

When training on Task A (ASR) and Task B (Translation), the gradients might conflict.
- `\nabla L_A` says "increase weight `w`".
- `\nabla L_B` says "decrease weight `w`".
- Result: They cancel out, or the model oscillates.

**PCGrad Algorithm:**
1. Compute gradients `g_A` and `g_B` independently.
2. Check cosine similarity: if `g_A \cdot g_B < 0` (angle > 90 degrees), they conflict.
3. Project `g_A` onto the normal plane of `g_B`:
 `g_A' = g_A - \frac{g_A \cdot g_B}{\|g_B\|^2} g_B`
4. Do the same for `g_B`.
5. Update weights with `g_A' + g_B'`.

**Effect:** The optimization trajectory follows a "zigzag" path that satisfies both tasks without destructive interference.

## 10. Deep Dive: Uncertainty Weighting

How do we set `\lambda_1, \lambda_2` in the loss function?
`L = \lambda_1 L_{ASR} + \lambda_2 L_{ST}`

**Homoscedastic Uncertainty:**
- Assume the task noise is constant.
- Learn `\sigma_1, \sigma_2` (variance) as trainable parameters.
- Loss becomes:
 `L = \frac{1}{2\sigma_1^2} L_{ASR} + \frac{1}{2\sigma_2^2} L_{ST} + \log \sigma_1 + \log \sigma_2`
- If a task is noisy (high loss), the model increases `\sigma` to reduce its weight.
- **Result:** Automatic, dynamic balancing.

## 11. Deep Dive: Adapter Modules

Fine-tuning a massive Multi-task model (like Whisper) for a new task is expensive.
**Adapters** allow efficient transfer learning.

**Architecture:**
- Freeze the pre-trained Transformer layers.
- Insert small "Adapter" layers between Feed-Forward and Self-Attention blocks.
- Adapter = `Linear(d -> d/r) -> ReLU -> Linear(d/r -> d)`.
- Only train the Adapters (few parameters).

**Task-Specific Adapters:**
- Train one set of adapters for "Medical ASR".
- Train another set for "Legal ASR".
- Switch adapters at runtime based on user domain.

## 12. Deep Dive: Whisper's Weak Supervision Strategy

Most ASR models are trained on LibriSpeech (clean, read audio). They fail on real-world noise.
Whisper trained on **680,000 hours** of internet audio.

**Data Filtering:**
1. **Language ID:** Discard if audio language doesn't match transcript language.
2. **No Speech:** Discard if VAD detects silence.
3. **Machine Generated:** Discard if transcript looks like output of another ASR system (to avoid learning errors).

**Result:**
- Whisper is not SOTA on LibriSpeech (Clean).
- But it is **SOTA on Robustness** (Accents, Noise, Music).

## 13. Code: PyTorch Multi-task Model

A simple implementation of a shared encoder with multiple heads.

``python
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
``

## 14. System Design: Real-Time Translation (Babelfish)

**Scenario:** Build a "Universal Translator" device.
**Input:** Audio (Spanish). **Output:** Audio (English).

**Pipeline:**
1. **VAD:** Detect speech.
2. **S2ST Model (Speech-to-Speech Translation):**
 - **Direct S2ST (Translatotron 2):** Audio -> Spectrogram. No text intermediate.
 - **Cascaded:** ASR -> MT -> TTS.
3. **Streaming:**
 - Use **Wait-k Policy**: Wait for `k` words before translating.
 - Trade-off: Latency vs. Context (Accuracy).

- Trade-off: Latency vs. Context (Accuracy).

## 15. Deep Dive: Joint CTC-Attention Training

**The Hybrid Approach:**
Most modern ASR systems (ESPnet, WeNet) use **Hybrid CTC/Attention**.

**Why?**
- **Attention:** Good at modeling long-range dependencies and language semantics. Bad at monotonic alignment (can loop or skip).
- **CTC:** Enforces monotonic alignment. Good at timing. Bad at conditional dependence.

**Training Objective:**
`L = \alpha L_{CTC} + (1 - \alpha) L_{Attn}`
- Typically `\alpha = 0.3`.
- The Encoder is shared.
- The CTC head branches off the Encoder.
- The Attention Decoder sits on top of the Encoder.

**Inference (Decoding):**
- **Attention Rescoring:**
 1. Generate top-k hypotheses using CTC (fast).
 2. Rescore them using the Attention Decoder (accurate).
- **Joint Decoding:**
 - Combine scores at each beam search step:
 `Score = \lambda \log P_{CTC}(y|x) + (1-\lambda) \log P_{Attn}(y|x)`

## 16. Deep Dive: Multilingual ASR as Multi-task Learning

Training on 100 languages is effectively a 100-task problem.

**Strategies:**
1. **Language ID (LID) Prediction:**
 - Add an auxiliary head to predict the language.
 - Helps the encoder separate language-specific features (phonemes) from language-agnostic features (speaker voice).

2. **Shared vs. Specific Layers:**
 - **Shared:** Bottom layers (acoustic features are universal).
 - **Specific:** Top layers (phonotactics and grammar vary).
 - **Adapter Modules:** Insert language-specific adapters.

3. **Script Unification:**
 - Map all languages to IPA (International Phonetic Alphabet) or use a shared SentencePiece vocabulary (e.g., 100k tokens covering all scripts).

## 17. Deep Dive: Voice Activity Detection (VAD) Integration

**Problem:** ASR models hallucinate text during silence.
**Solution:** Multi-task learning with VAD.

**Architecture:**
- **Main Task:** ASR (Seq2Seq).
- **Auxiliary Task:** Frame-level binary classification (Speech vs. Silence).
- **Loss:** `L_{ASR} + \lambda L_{VAD}`.

**Benefit:**
- Forces the encoder to learn a robust "speech presence" feature.
- During inference, the VAD head can be used to gate the decoder (don't decode if VAD < 0.5).

## 18. Deep Dive: Speaker Diarization as an Auxiliary Task

**Scenario:** "Who spoke when?"

**E2E ASR-Diarization (E2E-ASR-DA):**
- **Output:** Instead of just text, output `(Speaker_ID, Word)`.
- Example: `<spk:1> Hello <spk:2> Hi there`.

**Serialized Output Training (SOT):**
- For overlapping speech.
- Output Speaker 1's utterance first, then Speaker 2's.
- Separated by a delimiter token `<sep>`.

**Multi-task Benefit:**
- Learning to distinguish speakers helps ASR in "Cocktail Party" scenarios (overlapping speech).

## 19. Case Study: Meta's Massively Multilingual Speech (MMS)

**Goal:** ASR and TTS for **1,100+ languages**.

**Data:**
- **Religious Texts:** The Bible is translated into thousands of languages.
- **Alignment:** Use Forced Alignment on Bible audiobooks to create labeled data.

**Model:**
- **Wav2Vec 2.0** pre-training on 500k hours of unlabeled audio (1,400 languages).
- **Fine-tuning:** Linear layers (adapters) for each language.

**Result:**
- Half the WER of Whisper on low-resource languages.
- Proves that **Self-Supervised Learning + Multi-task Fine-tuning** is the recipe for scale.

## 20. Case Study: Google's Universal Speech Model (USM)

**Architecture:** Conformer (2 Billion parameters).

**Training Objectives (MOST):**
1. **BEST-RQ:** Self-supervised learning (BERT on audio).
2. **Text-Injection:** Train the encoder on text-only data (by upsampling text to match audio length).
3. **ASR:** Supervised learning on 300 languages.
4. **Automatic Speech Translation (AST):** Translate audio directly to English text.

**Key Innovation:**
- **Chunk-wise Attention:** To handle long audio (YouTube videos).
- **Result:** SOTA on YouTube captioning.

## 21. System Design: Scalable Multi-task Training Pipeline

**Challenge:** Training on 10 datasets (ASR, ST, VAD) with different sizes and formats.

**Pipeline:**
1. **Data Loader (The "Mixer"):**
 - Samples batches from different datasets based on a probability distribution `p_i`.
 - **Temperature Sampling:** `p_i \propto |D_i|^{1/T}`.
 - `T=1`: Proportional to dataset size (starves small tasks).
 - `T=\infty`: Uniform sampling (overfits small tasks).
 - `T=5`: Good balance.

2. **Bucketing:**
 - Group audio files by length to minimize padding.
 - Crucial for GPU efficiency.

3. **Distributed Training:**
 - **FSDP (Fully Sharded Data Parallel):** Shard model parameters across GPUs.
 - **Gradient Accumulation:** Simulate large batch sizes.

## 22. Deep Dive: Evaluation Metrics for Multi-task Models

How do you evaluate a "Swiss Army Knife" model?

1. **Average Performance:**
 - Normalize metrics (WER, BLEU, F1) to 0-100 scale.
 - Compute geometric mean.

2. **Pareto Frontier:**
 - Plot ASR Accuracy vs. ST Accuracy.
 - Does improving one hurt the other? (Negative Transfer).

3. **Zero-Shot Transfer:**
 - Train on English ASR + French ASR.
 - Test on English-to-French Translation.
 - (Emergent ability).

## 23. Code: Implementing Uncertainty Weighting

Let's implement the learnable loss weights in PyTorch.

``python
import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
 def __init__(self, num_tasks):
 super().__init__()
 # Learnable log variances (sigma^2)
 # Initialize to 0 (variance = 1)
 self.log_vars = nn.Parameter(torch.zeros(num_tasks))
 
 def forward(self, losses):
 # losses: list of scalar tensors [L1, L2, ...]
 total_loss = 0
 for i, loss in enumerate(losses):
 # Precision = 1 / (2 * sigma^2)
 precision = 0.5 * torch.exp(-self.log_vars[i])
 
 # L = precision * loss + log(sigma)
 # log(sigma) = 0.5 * log_var
 total_loss += precision * loss + 0.5 * self.log_vars[i]
 
 return total_loss

# Usage
mtl_criterion = MultiTaskLoss(num_tasks=2)
optimizer = torch.optim.Adam(
 list(model.parameters()) + list(mtl_criterion.parameters()), 
 lr=1e-4
)

# Training Loop
loss_asr = criterion_asr(pred_asr, target_asr)
loss_st = criterion_st(pred_st, target_st)

loss = mtl_criterion([loss_asr, loss_st])
loss.backward()
optimizer.step()
``

## 24. Future Trends: Foundation Models (SpeechLLM)

**The End of Task-Specific Heads?**

**SpeechLLM (2024+):**
- **Input:** Audio Tokens + Text Tokens.
- **Output:** Text Tokens (or Audio Tokens).
- **Task Specification:** Just a text prompt.
 - "Transcribe this audio."
 - "Who is speaking?"
 - "Is the speaker angry?"
- **Architecture:** Decoder-only Transformer (GPT-4 style).
- **Training:** Next-token prediction on massive mixed-modal data.

**Implication:** Multi-task learning becomes implicit. The model learns tasks as "in-context learning".

- **Training:** Next-token prediction on massive mixed-modal data.
- **Implication:** Multi-task learning becomes implicit. The model learns tasks as "in-context learning".

## 25. Deep Dive: Spoken Language Understanding (SLU) Tasks

SLU goes beyond transcription. It extracts meaning.

**1. Intent Classification:**
- **Input:** Audio.
- **Output:** Class label (e.g., `PlayMusic`, `SetAlarm`).
- **Multi-task Benefit:** ASR learns phonemes; Intent learns semantics. Together, they are robust to noise.

**2. Slot Filling:**
- **Input:** Audio.
- **Output:** Sequence of tags (BIO format).
- `Play [Song: Despacito] by [Artist: Luis Fonsi]`.
- **Architecture:** ASR Encoder -> Slot Decoder (CRF or LSTM).

**3. Sentiment Analysis:**
- **Input:** Audio.
- **Output:** Positive/Negative/Neutral.
- **Why Audio?** Sarcasm ("Yeah, right") is detected via pitch/tone, not text.
- **Fusion:** Concatenate Text Embeddings (from ASR) + Audio Embeddings (from Encoder) -> Classifier.

## 26. Deep Dive: Emotion Recognition as an Auxiliary Task

**SER (Speech Emotion Recognition):**
- Classes: Happy, Sad, Angry, Neutral.

**Why Multi-task with ASR?**
- **ASR helps SER:** Knowing *what* is said helps determine emotion.
- **SER helps ASR:** Emotional speech (shouting, crying) has different acoustic properties. Explicitly modeling emotion helps the encoder normalize these variations.

**Architecture:**
- **Shared Encoder:** Wav2Vec 2.0.
- **ASR Head:** CTC/Attention.
- **SER Head:** Pooling Layer -> Linear -> Softmax.

## 27. Deep Dive: Accent Classification

**Problem:** ASR fails on unseen accents.
**Solution:** Multi-task learning with Accent ID.

**Method:**
1. **Auxiliary Task:** Predict accent (US, UK, Indian, Australian).
2. **Gradient Reversal Layer (GRL):**
 - We want the encoder to be **Accent-Invariant**.
 - Add a GRL before the Accent Classifier.
 - During backprop, flip the gradient sign.
 - The encoder tries to *maximize* accent classification error (remove accent info), while the classifier tries to minimize it.
 - **Result:** Robust, accent-agnostic features.

## 28. Code: Implementing Gradient Surgery (PCGrad)

Here is a simplified implementation of PCGrad in PyTorch.

``python
import torch
import random

class PCGradOptimizer:
 def __init__(self, optimizer):
 self.optimizer = optimizer
 
 def step(self, objectives):
 """
 objectives: list of losses [loss_task1, loss_task2]
 """
 grads = []
 self.optimizer.zero_grad()
 
 # 1. Compute gradients for each task independently
 for loss in objectives:
 loss.backward(retain_graph=True)
 grad_list = []
 for param in self.optimizer.param_groups[0]['params']:
 if param.grad is not None:
 grad_list.append(param.grad.clone())
 param.grad.zero_() # Clear for next task
 grads.append(grad_list)
 
 # 2. Project conflicting gradients
 # Shuffle order to avoid bias
 random.shuffle(grads)
 
 final_grads = [g[:] for g in grads] # Deep copy
 
 for i in range(len(grads)):
 for j in range(len(grads)):
 if i == j: continue
 
 # Flatten gradients to compute dot product
 g_i_flat = torch.cat([g.flatten() for g in grads[i]])
 g_j_flat = torch.cat([g.flatten() for g in grads[j]])
 
 dot_prod = torch.dot(g_i_flat, g_j_flat)
 
 if dot_prod < 0: # Conflict!
 # Project g_i onto normal plane of g_j
 # g_i = g_i - (g_i . g_j) / ||g_j||^2 * g_j
 norm_sq = torch.dot(g_j_flat, g_j_flat)
 scale = dot_prod / norm_sq
 
 for k in range(len(grads[i])):
 final_grads[i][k] -= scale * grads[j][k]
 
 # 3. Apply final gradients
 for i, param in enumerate(self.optimizer.param_groups[0]['params']):
 if param.grad is None:
 param.grad = torch.zeros_like(param)
 
 # Sum projected gradients from all tasks
 for task_idx in range(len(grads)):
 param.grad += final_grads[task_idx][i]
 
 self.optimizer.step()
``

## 29. Checklist for Multi-task Training

Before you start training:

1. [ ] **Data Balance:** Are tasks roughly equal in size? If not, use temperature sampling.
2. [ ] **Loss Scale:** Do losses have similar magnitude? (e.g., ASR loss is 100, Class loss is 1). Normalize them.
3. [ ] **Capacity:** Is the model big enough? Multi-tasking requires more capacity than single-task.
4. [ ] **Scheduling:** Should you start with all tasks, or introduce them sequentially (Curriculum)?
5. [ ] **Evaluation:** Do you have a separate validation set for *each* task?

| **Robustness** | Low | High (Accent/Noise invariant) |

## 31. Deep Dive: Zero-Shot Transfer in Multi-task Models

**The Magic:** Train on Task A and B, test on Task C (which was never seen).

**Example: Zero-Shot Speech Translation**
- **Train:**
 1. English Audio -> English Text (ASR).
 2. English Text -> French Text (MT).
- **Test:** English Audio -> French Text (ST).
- **Mechanism:** If the model learns a shared embedding space for "Audio" and "Text", it can bridge the gap.

**Example: Zero-Shot Language Transfer**
- **Train:**
 1. English ASR.
 2. French ASR.
 3. English-to-Spanish Translation.
- **Test:** French-to-Spanish Translation.
- **Mechanism:** The "Translation" head learns to map semantic concepts to Spanish, regardless of the source language (if the encoder is language-agnostic).

## 32. Deep Dive: The "Curse of Multilingualism"

**Observation:** Adding languages improves performance initially (transfer learning), but eventually degrades it (interference).

**The Capacity Bottleneck:**
- A fixed-size model has limited capacity.
- English takes up 50% of the weights.
- Adding 99 more languages forces them to fight for the remaining 50%.
- **Result:** High-resource languages (English) degrade slightly; low-resource languages improve massively.

**Solution:**
1. **Increase Model Size:** 1B -> 10B parameters.
2. **Mixture of Experts (MoE):**
 - Have 100 "Expert" FFN layers.
 - For each token, route it to the top-2 experts.
 - **Result:** Massive capacity (1 Trillion params) with low inference cost (active params are small).

## 33. Deep Dive: AdapterFusion

**Problem:** We have separate adapters for ASR, ST, and VAD. Can we combine them?

**AdapterFusion (Pfeiffer et al.):**
1. **Train:** Train adapters for each task independently.
2. **Fuse:** Freeze adapters. Learn a **Fusion Layer** (Attention) that combines their outputs.
 `h_{fused} = \text{Attn}(h_{enc}, [h_{ASR}, h_{ST}, h_{VAD}])`
3. **Benefit:** The model dynamically decides: "For this noisy frame, I'll trust the VAD adapter more. For this clear speech, I'll trust the ASR adapter."

## 34. Case Study: Tuning Whisper for Code-Switching

**Scenario:** "Hinglish" (Hindi + English mixed).
- "Main kal market jaaunga to buy vegetables."

**Challenge:**
- Monolingual ASR fails (expects only Hindi or only English).
- Language ID flips rapidly.

**Multi-task Solution:**
1. **Data:** Synthetic Code-Switching.
 - Take English sentence.
 - Replace random nouns with Hindi translations.
 - Generate audio using TTS.
2. **Training:** Fine-tune Whisper on this mixed data.
3. **Result:** The model learns to handle intra-sentence language switching without explicit language tags.

## 35. Future Trends: SpeechLLM and "In-Context" Multi-tasking

**Current:** Explicit heads for ASR, ST.
**Future:** Text Prompting.

**AudioPaLM (Google):**
- Unified vocabulary of Text Tokens and Audio Tokens.
- **Task:** "Translate this audio to German."
- **Input:** `[AudioTokens] [Text: Translate to German]`
- **Output:** `[Text: German Translation]`
- **In-Context Learning:** Provide 3 examples in the prompt, and the model learns the task on the fly without weight updates.

- **In-Context Learning:** Provide 3 examples in the prompt, and the model learns the task on the fly without weight updates.

## 36. Deep Dive: The "Cocktail Party Problem" (Source Separation)

**Scenario:** Two people talking at once.
**Goal:** Separate them into two clean audio streams.

**Multi-task Approach:**
- **Task 1:** Separation (PIT - Permutation Invariant Training).
- **Task 2:** ASR on separated streams.
- **Joint Training:** Backpropagate ASR loss through the separator.
- **Result:** The separator learns to output streams that are "ASR-friendly" (even if they sound slightly unnatural to humans).

## 37. Deep Dive: Audio-Visual Multi-task Learning

**Lip Reading (Visual Speech Recognition):**
- **Input:** Audio + Video of lips.
- **Tasks:**
 1. Audio ASR.
 2. Video ASR.
 3. Audio-Visual Fusion.
- **Benefit:** When audio is noisy (0dB SNR), the model relies on the video stream (lip movement) to disambiguate phonemes (e.g., "P" vs "B").

## 38. Deep Dive: Self-Training (Noisy Student)

**Algorithm:**
1. Train Teacher on labeled data (ASR).
2. Teacher generates pseudo-labels for unlabeled data.
3. **Multi-task Student:**
 - Train Student on Labeled Data (Supervised Loss).
 - Train Student on Unlabeled Data (Consistency Loss).
 - **Augmentation:** Add noise (SpecAugment) to Student input, force it to match Teacher output.
4. Iterate.

**Result:** Massive improvements in robustness without new human labels.

- **Result:** Massive improvements in robustness without new human labels.
- **Augmentation:** Add noise (SpecAugment) to Student input, force it to match Teacher output.

## 39. The Economics of Multi-task Models

**Cost:**
- **Training:** Extremely expensive. Training Whisper-Large took thousands of GPU-days.
- **Inference:** Large models (1B+ params) are slow and require expensive GPUs (A100).

**Benefit:**
- **Maintenance:** Maintaining 1 model is cheaper than maintaining 10 specialized models.
- **Data Efficiency:** You save millions on labeling costs because the model learns from unlabeled data and transfer learning.
- **User Experience:** Seamless switching between languages and tasks (ASR -> Translation) without latency spikes.

**Verdict:** For large tech companies, Multi-task is a no-brainer. For startups, fine-tuning a pre-trained Multi-task model (like Whisper) is the way to go.

## 40. Checklist for Deployment

1. [ ] **Model Size:** Can you afford to run `large-v3` (1.5GB VRAM) or do you need `tiny` (75MB)?
2. [ ] **Quantization:** Use `int8` or `float16` to reduce memory by 2-4x with minimal accuracy loss.
3. [ ] **Batching:** Use dynamic batching (e.g., TorchServe) to saturate the GPU.
4. [ ] **Caching:** Cache common audio queries (hash the audio file).
5. [ ] **Fallback:** If the model fails (low confidence), do you have a fallback (e.g., a simpler model or human-in-the-loop)?

- **Fallback:** If the model fails (low confidence), do you have a fallback (e.g., a simpler model or human-in-the-loop)?
- **Bias:** Multi-task models can amplify bias. If the training data has more male speakers for ASR and female for TTS, the model might associate "male" with "input" and "female" with "output".

## 41. Ethical Considerations

**1. Bias Amplification:**
- Multi-task models trained on internet data (Whisper) inherit internet biases.
- **Example:** Translating "The doctor called the nurse" into a gendered language might default to "Male Doctor" and "Female Nurse" purely based on statistical co-occurrence, even if incorrect.

**2. Representation:**
- Low-resource languages often get "overwritten" by high-resource ones in shared capacity models.
- **Fix:** Ensure strict data balancing and evaluation on *all* languages, not just the top 10.

**3. Dual Use:**
- A model good at Voice Cloning (TTS) and ASR can be used for deepfakes.
- **Safeguards:** Release models with watermarking or restricted licenses.

## 42. Further Reading

1. **"Whisper: Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022):** The bible of multi-task speech.
2. **"UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data" (Wang et al., 2021):** Combining SSL and Supervised learning.
3. **"Gradient Surgery for Multi-Task Learning" (Yu et al., 2020):** The PCGrad paper.
4. **"Massively Multilingual Speech" (Pratap et al., 2023):** Scaling to 1000 languages.
5. **"AudioPaLM: A Large Language Model that Can Speak and Listen" (Rubenstein et al., 2023):** The future of SpeechLLMs.

## 43. Conclusion

Multi-task learning is the key to building **general-purpose speech systems**. Instead of building one model for ASR, one for Translation, and one for VAD, we are moving towards **Unified Foundation Models** that can handle any speech task via prompting. The challenges of gradient conflict and capacity bottlenecks are being solved by techniques like PCGrad and Mixture of Experts. The future is not just multi-task, but **multi-modal** (Speech + Text + Vision).

## 44. Summary

| Feature | Single-Task | Multi-Task |
| :--- | :--- | :--- |
| **Performance** | High on specific task | High on all (usually) |
| **Data Req** | High labeled data | Can leverage auxiliary data |
| **Deployment** | N models | 1 model |
| **Training** | Simple | Complex (balancing) |
| **Optimization** | Standard SGD | PCGrad, Uncertainty Weighting |
| **Robustness** | Low | High (Accent/Noise invariant) |
| **Future** | Specialized Models | Foundation Models (SpeechLLM) |

---

**Originally published at:** [arunbaby.com/speech-tech/0036-multi-task-speech-learning](https://www.arunbaby.com/speech-tech/0036-multi-task-speech-learning/)
