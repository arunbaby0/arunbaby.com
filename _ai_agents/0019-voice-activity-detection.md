---
title: "Voice Activity Detection (VAD)"
day: 19
collection: ai_agents
categories:
  - ai-agents
tags:
  - vad
  - silero
  - audio-processing
  - webrtc
  - turn-taking
  - end-of-speech
  - zero-crossing-rate
related_dsa_day: 19
related_ml_day: 19
related_speech_day: 19
difficulty: Medium
---

**"The art of knowing when to shut up."**

## 1. Introduction: The Turn-Taking Problem

In a text chat, the user hits "Enter". This is an explicit signal: "I am done. Your turn." It is a deterministic, boolean event.
In voice, there is no "Enter" key. There is only **Silence**.

But silence is ambiguous.
*   *Silence (300ms):* A comma. "I went to the store..." (thinking) "...and bought milk."
*   *Silence (800ms):* A period. "I went to the store." (Done).

This creates the **Turn-Taking Threshold Paradox**:
*   If the agent speaks at 400ms silence, it interrupts the user mid-thought. (Rude / "Aggressive").
*   If the agent waits for 1500ms silence, it feels slow and confused. (Dumb / "Laggy").

**Voice Activity Detection (VAD)** is the algorithmic subsystem responsible for answering two questions:
1.  "Is the user speaking right now?" (Binary State).
2.  "Has the user *finished* speaking?" (State Transition).

---

## 2. The Physics of Sound

How do we detect speech without understanding words? We rely on the physical properties of the waveform.

### 2.1 Amplitude / Energy
The simplest method.
*   **Logic:** Calculate Root Mean Square (RMS) of the audio amplitude. If `RMS > Threshold`, then Speech.
*   **Failures:** A slamming door, typing on a keyboard, a fan, or a cough triggers it. Noise usually has high energy. It cannot distinguish a Human from a Truck.

### 2.2 Zero-Crossing Rate (ZCR)
Speech signals (especially vowels) oscillate at fundamental frequencies using the vocal cords. Noise is often chaotic.
ZCR measures how often the waveform crosses the X-axis (0).
*   *Speech:* Low/Stable ZCR.
*   *Noise:* High/Random ZCR.

### 2.3 Frequency Analysis (FFT)
Human speech lives in specific bands (300Hz - 3400Hz).
*   **Logic:** Run Fast Fourier Transform (FFT). If energy density is high in the 300-3400Hz band and low elsewhere, it's likely speech.
*   **Failures:** Music, TV background noise, or a Podcast playing in the background.

---

## 3. Algorithms: From Gaussian to Neural

### 3.1 WebRTC VAD (The Classic)
Google opened sourced the VAD from Chrome's WebRTC stack.
*   **Algorithm:** Gaussian Mixture Models (GMM). It learns the statistical distribution of "Speech" vs "Noise" frames dynamically.
*   **Pros:** Ultra fast (C++). Runs in the browser using WASM. Almost zero CPU usage.
*   **Cons:** **Not Robust.** It struggles with "Cocktail Party" noise (background chatter). It triggers easily on breathing.

### 3.2 Silero VAD (The Modern Standard)
**Silero** is a pre-trained Neural Network (Enterprise Grade).
*   **Algorithm:** RNN/LSTM architecture trained on thousands of hours of speech and noise datasets.
*   **Pros:** High accuracy. Can distinguish a cough from a word. Low latency (< 30ms chunk processing). Lightweight (runs on CPU).
*   **Output:** A probability score `0.0` to `1.0`.

---

## 4. Engineering the VAD State Machine

Using Silero gives you a probability stream. You must build a **State Machine** on top of it to separate "Bursts" from "Turns".

### 4.1 The Parameters
Implementing VAD is about tuning three critical variables.

1.  **Probability Threshold:**
    *   Sensitivity. `0.5` is standard.
    *   `0.8` eliminates heavy breathing but misses soft whispers.
2.  **Min Speech Duration (start_patience):**
    *   To trigger "Start of Turn", sound must persist for `X` ms.
    *   *Setting:* **250ms**.
    *   *Why:* Prevents "Clicks", "Pops", or short "Uhh" sounds from waking the agent.
3.  **Min Silence Duration (end_patience):**
    *   The most critical param. To trigger "End of Turn", silence must persist for `Y` ms.
    *   **700ms - 1000ms:** Standard conversation.
    *   **500ms:** "Interrupt mode" (Very snappy, high false positive).
    *   **2000ms:** "Dictation mode" (User is thinking heavily).

### 4.2 Handling "Humming" and "Laughing"
Humans make non-speech sounds. "Hahaha", "Mmhmm".
*   Strict VADs filter these out.
*   Conversational Agents *should* hear these. "Mmhmm" is a "Backchannel" signal meaning "I am listening, continue."
*   *Optimization:* Detect Backchannels and **do not** trigger a full LLM response. Just continue speaking or stay silent.

---

## 5. Advanced: Semantic End-of-Turn

VAD is "Acoustic". It relies on sound. It fails on **Structural Incompleteness**.
*   User: "I want to buy..." (Silence 1s).
*   Acoustic VAD: "Turn Over (Silence > 700ms)." Agent interrupts.
*   User semantics: "Sentence incomplete."

**Solution: Hybrid VAD**
We use a small, fast Language Model (or a dedicated model like **TurnGPT**) to score the "Completeness" of the transcript.

*   **Logic:**
    1.  STT Stream: "I want to buy"
    2.  Check: `LLM.predict_completion("I want to buy")`.
    3.  Result: `Low (0.1)`.
    4.  Action: **Extend Silence Timeout** dynamically from 700ms to 2000ms.
    5.  User (after 1.5s): "...a ticket."
    6.  Check: `LLM.predict_completion("I want to buy a ticket")`.
    7.  Result: `High (0.9)`.
    8.  Action: Trigger Turn End immediately.

This allows for **"Pacing"**. The agent waits when you are thinking, and replies instantly when you are done.

---

## 6. Code: Implementing Silero VAD Wrapper

A robust Python wrapper for processing audio chunks.

```python
import torch
import numpy as np

# Load Silero (JIT)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

class VADState:
    SILENCE = 0
    SPEECH = 1

class VADEngine:
    def __init__(self, threshold=0.5, min_silence_duration_ms=700):
        self.model = model
        self.threshold = threshold
        self.min_silence_samples = min_silence_duration_ms * 16 # assuming 16khz
        self.state = VADState.SILENCE
        self.silence_counter = 0
        
    def process_chunk(self, audio_chunk_bytes: bytes):
        """
        Input: 512 bytes of PCM Int16
        Output: Event (START, END, NONE)
        """
        # 1. Convert bytes to float32 tensor
        # Real implementation needs robust float conversion
        audio_int16 = np.frombuffer(audio_chunk_bytes, np.int16)
        # Normalize to -1.0 to 1.0
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_float32)
        
        # 2. Inference
        speech_prob = self.model(tensor, 16000).item()
        
        # 3. State Machine
        if self.state == VADState.SILENCE:
            if speech_prob > self.threshold:
                self.state = VADState.SPEECH
                self.silence_counter = 0
                return "SPEECH_START"
                
        elif self.state == VADState.SPEECH:
            if speech_prob < self.threshold:
                self.silence_counter += len(audio_int16) # Add duration
                if self.silence_counter > self.min_silence_samples:
                    self.state = VADState.SILENCE
                    self.silence_counter = 0
                    return "SPEECH_END"
            else:
                self.silence_counter = 0 # Reset if speech returns
                
        return "NONE"
```

---

## 7. Summary

VAD is the heartbeat of a Voice Agent. It defines the "Rhythm" of conversation.

*   **WebRTC VAD** is fast but noisy.
*   **Silero VAD** is the standard for robust detection.
*   **State Machines** are required to handle the jitter of raw probabilities.
*   **Semantic Analysis** prevents interrupting users who are thinking.

Beyond VAD, the frontier of voice is **Real-Time Speech-to-Speech Agents**, removing text from the loop entirely.
