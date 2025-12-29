---
title: "Custom Language Modeling"
day: 50
collection: speech_tech
categories:
  - speech-tech
tags:
  - asr
  - language-model
  - biasing
  - contextual-asr
  - production
difficulty: Hard
subdomain: "ASR Customization"
tech_stack: Kaldi, Whisper, Google Speech API
scale: "Adapting generic models to specific domains"
companies: Nuance, Google Cloud, Amazon Transcribe
related_dsa_day: 50
related_ml_day: 50
related_agents_day: 50
---

**"The model knows 'Apple' the fruit. It needs to learn 'Apple' the stock ticker."**

## 1. Problem Statement

Generic ASR models (Whisper, Google Speech) are trained on general internet data. They perform poorly on **Jargon**.
-   **Medical**: "Administer 50mg of Xarelto." -> "The real toe."
-   **Legal**: "Habeas Corpus." -> "Happy its corpse."
-   **Corporate**: "Met me at the K8s sync." -> "Kate's sink."

**The Problem**: How do we teach a pre-trained ASR model new vocabulary *without* retraining the massive acoustic model?

---

## 2. Fundamentals: The Noisy Channel Model

Recall the ASR equation:
$$P(Text | Audio) \propto P(Audio | Text) \times P(Text)$$

-   **Acoustic Model ($P(Audio | Text)$)**: "Does this sound like 'Xarelto'?" (Maybe).
-   **Language Model ($P(Text)$)**: "Is 'Xarelto' a word?" (Generic LM says No. Prob = 0.000001).

Since the LM probability is near zero, the total score is low. The decoder chooses "The real toe" because `P("The real toe")` is high.
**Solution**: We hack $P(Text)$.

---

## 3. Architecture: Shallow vs Deep Fusion

How do we inject the new words?

### 3.1 Shallow Fusion (The Standard)
We train a small, domain-specific LM (n-gram) on the client's text documents.
During decoding (Beam Search), we interpolate the scores:
$$Score = \log P_{AM} + \alpha \log P_{GenericLM} + \beta \log P_{CustomLM}$$

If $\beta$ is high, the custom model boosts "Xarelto".

### 3.2 Deep Fusion
We inject a specialized neural network layer *inside* the ASR network that attends to a list of custom words. This is harder to implement but more robust.

---

## 4. Implementation Approaches

### 4.1 Class-Based LMs
Instead of hardcoding "John Smith", we train the LM with a placeholder tag: `@NAME`.
-   Training sentence: "Call `@NAME` at 5pm."
-   Runtime: We provide a map `{@NAME: ["John", "Sarah", "Mike"]}`.
The FST (Finite State Transducer) dynamically expands the `@NAME` node into arcs for John, Sarah, Mike.

### 4.2 Contextual Biasing (Attention)
In Transformer ASR (Whisper), we can pass a list of "Hint Strings" in the prompt.
`prompt="Xarelto, Ibuprofen, Tylenol"`
The model's cross-attention mechanism attends to these tokens, increasing their likelihood.

---

## 5. Implementation: Contextual Biasing with Whisper

```python
import whisper

model = whisper.load_model("base")

# 1. Standard Inference
audio = "audio_xarelto.mp3"
result_bad = model.transcribe(audio)
print(result_bad["text"]) 
# Output: "Patient needs the real toe."

# 2. Contextual Prompting
# We prepend the keywords to the decoder's context window.
# It acts like the model "just said" these words, priming it to say them again.
initial_prompt = "Medical Logic: Xarelto, Warfarin, Apixaban."

result_good = model.transcribe(audio, initial_prompt=initial_prompt)
print(result_good["text"])
# Output: "Patient needs Xarelto."
```

---

## 6. Training Considerations

### 6.1 Text Data Augmentation
To train the Custom LM, you need text.
-   **Source**: Technical manuals, past transcripts, email logs.
-   **Normalization**: You must convert "50mg" to "fifty milligrams" to match ASR output space.

### 6.2 Pruning
A custom LM with 1 million words is slow.
Prune the n-grams. Keep only unique jargon. Trust the Generic LM for "the", "cat", "is".

---

## 7. Production Deployment: Dynamic Loading

In a SaaS ASR (like Otter.ai):
1.  User enters a meeting ("Project Apollo Sync").
2.  System loads "Project Apollo" word list (Entities: "Apollo", "Saturn", "Launch").
3.  System compiles a tiny FST on-the-fly (ms).
4.  Decoder graph = `Generic_Graph` composed with `Dynamic_FST`.

This allows **Per-User** customization.

---

## 8. Performance Metrics

**Entity-WER**.
-   Global WER might be 5% with or without customization.
-   But if the 5% error is the *Patient's Name*, the transcript is useless.
-   Measure accuracy specifically on the **Boosted List**.

---

## 9. Failure Modes

1.  **Over-Biasing**:
    -   Boost list: `["Call"]`.
    -   User says: "Tall building".
    -   ASR hears: "Call building".
    -   *Fix*: Tunable parameter `biasing_weight`.
2.  **Phonetic Confusion**:
    -   Boost: `["Resume"]` (Noun).
    -   User: "Resume" (Verb).
    -   ASR gets it right, but downstream NLP gets confused by the tag.

---

## 10. Real-World Case Study: Smart Speakers

**Alexa Contact List**.
When you say "Call Mom", Alexa biased the ASR towards your contacts.
It didn't boost "Mom" for everyone. It boosted "Mom", "Dad", "Arun" for *you*.
This uses **Personalized Language Models (PLM)**.

---

## 11. State-of-the-Art: Neural Biasing

Recent research (Google) uses **GNNs (Graph Neural Networks)** to encode the relationship between entities in the bias list, handling thousands of entities (e.g., a massive Song Library) without degrading latency.

---

## 12. Key Takeaways

1.  **Generic is not enough**: Production ASR requires customization.
2.  **Shallow Fusion is cheap**: No GPU retraining needed. Just text statistical counting.
3.  **Prompt Engineering works for ASR**: Whisper's prompt feature allows 0-shot adaptation.
4.  **Metric Validity**: Optimize for Entity-WER, not just WER.

---

**Originally published at:** [arunbaby.com/speech-tech/0050-custom-language-modeling](https://www.arunbaby.com/speech-tech/0050-custom-language-modeling/)

*If you found this helpful, consider sharing it with others who might benefit.*
