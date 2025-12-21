---
title: "Real-Time Speech-to-Speech Agents"
day: 20
collection: ai_agents
categories:
  - ai-agents
tags:
  - speech-to-speech
  - gpt-4o
  - moshi
  - ultravox
  - multimodal
  - audio-tokenization
  - vq-vae
related_dsa_day: 20
related_ml_day: 20
related_speech_day: 20
  - encodec
difficulty: Medium
---

**"Removing the Text Bottleneck: The Omni Future."**

## 1. Introduction: The Cascade Problem

Standard voice agents use a **Cascade Architecture**:
`Audio -> STT -> Text -> LLM -> Text -> TTS -> Audio`.

This works—it powers Siri, Alexa, and most enterprise bots—but it has two fatal flaws that prevent true "Human-Level" interaction:

### 1.1 The Latency Stackup
* STT Processing: 300ms
* LLM Time-To-First-Token: 400ms
* TTS Generation: 300ms
* Network RTT: 100ms
* **Total:** 1.1 seconds (Best Case ideal).
This floor is hard to break because each system must wait for *some* meaningful input before it starts. The LLM cannot predict the sentiment of the sentence until the STT finishes the sentence.

### 1.2 Information Loss (The Emotional Flatness)
Text is a "Lossy Compression" of speech. By converting audio to text, we strip away:
* **Prosody:** The rhythm and intonation.
* **Emotion:** Sarcasm, Anger, Excitement, Uncertainty.
* **Non-Verbal Cues:** Breaths, laughter, hesitation ("Umm...").

**Example:**
* **User (Sarcastic):** "Oh, *great* job. Really amazing."
* **Text Representation:** "Oh, great job. Really amazing."
* **LLM Brain:** Sees positive sentiment words ("Great", "Amazing"). Replies: "Thanks! I'm glad you liked it."
* **Reality:** The user is angry, and the agent just made it worse.

To fix this, we need **Speech-to-Speech (S2S)** models. Models that "hear" audio and "speak" audio natively, without ever converting to text in the middle.

---

## 2. Anatomy of S2S Models: How do they work?

How can a Transformer, which is designed for discrete text tokens (integers), process continuous audio waves?

### 2.1 Audio Tokenization (The Rosetta Stone)
We use a **Neural Audio Codec** (like Meta's EnCodec or OpenAI's internal codec).
These models use **Residual Vector Quantization (RVQ)**.

1. **Encoder:** Takes raw audio (24khz). Compresses it into a dense latent space.
2. **Quantizer:** Maps the latent vectors to the nearest entry in a fixed "Codebook" (Visualise a dictionary of 2048 distinct "Sound Units").
3. **Result:** Audio becomes a sequence of integers: `[45, 1029, 33, 500...]` just like text is `[101, 205, ...]`.
4. **Rate:** Usually 50 to 100 tokens per second.

### 2.2 The Omni-Model Training
Once audio is tokenized, it is just "Language" to the Transformer.
* **Input Sequence:** `[Text Tokens] + [Image Tokens] + [Audio Tokens]`.
* **Training Objective:** "Next Token Prediction".
* **Output:** The model can predict that after the audio tokens for "Hello", the next audio tokens should be "How are you?".

**GPT-4o (Omni)** is the state of the art. It processes audio, vision, and text in a single transformer.
* *Capability:* It comes closer to human interaction than anything before. It can hear you breathing. It can sing. It can laugh. It detects sarcasm. It can change its tone mid-sentence if you interrupt it.
* *Latency:* ~300ms (Average human response time).

---

## 3. The Landscape: GPT-4o, Moshi, Ultravox

### 3.1 Kyutai Moshi (Open Source S2S)
A French research lab released **Moshi**, the first high-quality open real-time S2S model.
* **Components:** Helium (7B LLM) + Mimi (Audio Codec).
* **Innovation:** **Dual Stream Output.**
 * Standard LLMs output one token at a time.
 * Moshi outputs two streams in parallel:
 1. **Text Token:** The semantic logic ("I am thinking about cats.")
 2. **Audio Token:** The acoustic realization ("Meow.")
 * This allows developers to inspect the "Thought Process" (Text) while playing the Audio.

### 3.2 Ultravox (The Hybrid)
Ultravox is an open-weights architecture fusing Llama 3 with a Whisper Encoder and a Unit-based Vocoder.
* **Approach:** Instead of training from scratch (expensive), it uses a "Projector" to map Whisper's audio embeddings into Llama 3's embedding space.
* **Pros:* You can self-host it. It leverages the reasoning power of Llama 3.
* **Cons:* Not as fluid as GPT-4o yet. It feels like a "Smart Speaker" rather than a "Person".

---

## 4. Engineering S2S Agents: New Paradigms

Working with S2S is fundamentally different from Text Agents.

### 4.1 Prompting "Sound"
S2S models respond to **Audio Prompts**.
* *Technique:* **One-Shot Voice Cloning.** Provide distinct audio examples in the context window.
 * User: `[Audio of Morgan Freeman]` "Speak like this."
 * Model: `[Audio in Morgan Freeman style]` "Okay, I will."
* *Style Tokens:* The model supports meta-tags in the prompt: `[Style: Whispering] [Emotion: Fear]`.

### 4.2 Handling Interruptions (Duplex)
In Cascade, interruption is handled by logic (VAD triggers Stop).
In S2S, the model *hears itself* speaking (if echo cancellation fails) and hears the user.
* **The Problem:** If the model hears the user saying "Wait", does it predict silence (stop talking)?
* **Training:** Models must be fined-tuned on "Turn-Taking" data where interruptions occur, so they learn to yield the floor naturally.
* **Moshi's Solution:** It predicts "Silence Tokens" for its own channel when the User channel is active.

---

## 5. Cost, Safety, and Challenges

### 5.1 Cost
Processing Audio Tokens is much more expensive than Text Tokens.
* **Density:** 1 second of audio might be 50 tokens. To process a 10-second sentence is 500 tokens input + 500 tokens output = 1000 tokens.
* **Comparision:** The text "How are you?" is 3 tokens. The audio "How are you?" is 100 tokens.
* **Price:** GPT-4o Audio pricing is significantly higher (~10x) than text. Using S2S for everything burns money.

### 5.2 Safety (Audio Injection)
A text model can be filtered using Regex (`Stop words`).
An audio model is harder to filter.
* *Scenario:* The user asks the model to "Say 'I hate you' in a happy voice."
* *Scenario:* The model produces a sound that *sounds* like a racial slur but isn't the exact word in the text transcript.
* *Jailbreaks:* "Audio Adversarial Attacks" (statistically noisy audio that sounds like static to humans but forces the model to output harmful content).

### 5.3 Control
It is harder to steer. If you want the agent to say *exactly* "Your balance is $54.20", an S2S model might say "Roughly fifty bucks". In banking, this is unacceptable.
* **Solution:** Hybrid pipelines. Use S2S for chit-chat, switch to Text-to-Speech for strict data reading.

---

## 6. Summary

Speech-to-Speech is the endgame for Voice Agents.

* **Latency** drops to human levels (< 300ms).
* **Emotion** is preserved, enabling true empathy in medical/support agents.
* **Complexity** shifts from "Pipeline Engineering" (gluing STT/TTS) to "Model Deployment" (hosting a massive transformer).

This concludes the exploration of Real-Time & Voice. The journey moves next to **Vision Agents**, learning how to make agents "See" screens, documents, and the physical world.


---

**Originally published at:** [arunbaby.com/ai-agents/0020-speech-to-speech-agents](https://www.arunbaby.com/ai-agents/0020-speech-to-speech-agents/)

*If you found this helpful, consider sharing it with others who might benefit.*

