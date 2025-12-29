---
title: "Speech Pipeline Dependencies"
day: 49
collection: speech_tech
categories:
  - speech-tech
tags:
  - vad
  - speaker-diarization
  - asr
  - pipeline
  - latency
difficulty: Medium
subdomain: "System Integration"
tech_stack: Silero VAD, Pyannote, Whisper
scale: "Real-time meeting transcription"
companies: Zoom, Otter.ai, Gong
related_dsa_day: 49
related_ml_day: 49
related_agents_day: 49
---

**"Garbage in, Garbage out. Silence in, Hallucination out."**

## 1. Problem Statement

A modern "Voice Assistant" is not one model. It is a **Cascade of Models**.
1.  **VAD**: Is someone speaking?
2.  **Diarization**: Who is speaking?
3.  **ASR**: What did they say?
4.  **NLP/Entity**: What does it mean?

**The Problem**: These models depend on each other. If VAD cuts off the first 50ms of a word, ASR fails. If ASR makes a typo, NLP fails. How do we orchestrate this pipeline efficiently?

---

## 2. Fundamentals: The Tightly Coupled Chain

Unlike generic microservices where Service A calls Service B via HTTP, Speech pipelines share **Time Alignment**.
-   VAD output: `User spoke [0.5s - 2.5s]`.
-   Diarization output: `Speaker A at [0.4s - 2.6s]`.
-   The timestamps must match perfectly.
-   **Error Propagation**: A 10% error in VAD can cause a 50% error in Diarization (if it misses the speaker change).

---

## 3. High-Level Architecture

We can view this as a **Stream Processing DAG**.

```mermaid
graph LR
    A[Microphone Stream] --> B{VAD}
    B -- Silence --> C[Discard / Noise Profile Update]
    B -- Speech --> D[Buffer]
    D --> E[Speaker ID (Embedding)]
    D --> F[ASR (Transcription)]
    E --> G[Meeting Transcript Builder]
    F --> G
```

---

## 4. Component Deep-Dives

### 4.1 Voice Activity Detection (VAD)
The Gatekeeper.
-   **Role**: Save compute by ignoring silence. Prevent ASR from hallucinating on noise.
-   **Latency**: Must be < 10ms.
-   **Models**: Silero VAD (RNN), WebRTC VAD (GMM).

### 4.2 Speaker Diarization
The labeler.
-   **Role**: Assign "Speaker 1" vs "Speaker 2".
-   **Complexity**: $O(N^2)$ (Clustering). Hard to do streaming.
-   **Solution**: "Online Diarization" keeps a centroid for active speakers and assigns new frames to nearest centroid.

### 4.3 ASR
The Heavy Lifter.
-   Takes the buffered audio from VAD.
-   Returns text + timestamps.

---

## 5. Data Flow: The "Turn" Concept

Processing audio byte-by-byte is inefficient for ASR. We group audio into **Turns** (Utterances).
1.  VAD detects `Speech Start`.
2.  Pipeline starts Accumulating audio into RAM.
3.  VAD detects `Speech End` (trailing silence > 500ms).
4.  **Trigger**: Send accumulated buffer to Diarization and ASR in parallel.
5.  **Merge**: Combine `Speaker=John` and `Text="Hello"`.

---

## 6. Model Selection & Trade-offs

| Stage | Model Option A (Fast) | Model Option B (Accurate) | Selection Logic |
|-------|-----------------------|---------------------------|-----------------|
| VAD | WebRTC (CPU) | Silero (NN) | Use Silero. Accuracy is vital. Cost is low. |
| Diarization | Speaker Embedding (ResNet) | End-to-End (EEND) | Use Embedding. EEND is too slow for real-time. |
| ASR | Whisper-Tiny | Whisper-Large | Use Tiny for streaming, Large for final correction. |

---

## 7. Implementation: The Pipeline Class

```python
import torch

class SpeechPipeline:
    def __init__(self):
        self.vad_model = load_silero_vad()
        self.asr_model = load_whisper()
        self.buffer = []
        
    def process_frame(self, audio_chunk):
        # 1. Filter Silence
        speech_prob = self.vad_model(audio_chunk, 16000)
        
        if speech_prob > 0.5:
            self.buffer.append(audio_chunk)
            
        elif len(self.buffer) > 0:
            # Trailing silence detected -> End of Turn
            full_audio = torch.cat(self.buffer)
            self.buffer = [] # Reset
            
            # 2. Trigger ASR (The Dependency)
            text = self.asr_model.transcribe(full_audio)
            
            print(f"Turn Complete: {text}")
```

---

## 8. Streaming Implications

In a true streaming pipeline, we cannot wait for "End of Turn".
We use **Speculative Execution**.
1.  ASR runs continuously on partial buffer: `H -> He -> Hel -> Hello`.
2.  Diarization runs every 1 second: `Speaker A`.
3.  **Correction**: If Diarization changes its mind (`Speaker A` -> `Speaker B`), we send a "Correction Event" to the UI to overwrite the previous line.

---

## 9. Quality Metrics

-   **DER (Diarization Error Rate)**: `False Alarm + Missed Detection + Confusion`.
-   **CpWER (Concatenated Person WER)**: WER calculated per speaker. Finding out if the model is biased against Speaker B.

---

## 10. Common Failure Modes

1.  **The "Schrodinger's Word"**: A word at the boundary of a VAD cut.
    -   User: "Important."
    -   VAD cuts at 0.1s.
    -   Audio: "...portant."
    -   ASR: "Portent."
    -   *Fix*: **Padding**. Always keep 200ms of history before the VAD trigger.
2.  **Overlapping Speech**: Two people talk at once.
    -   Standard ASR fails.
    -   Standard Diarization fails.
    -   *Fix*: Source Separation models (rarely used in production due to cost).

---

## 11. State-of-the-Art

**Joint Models (Transducers)**.
Instead of `VAD -> ASR -> NLP`, train one massive Transformer:
Input: Audio.
Output: `<speaker:1> Hello <speaker:2> Hi there <sentiment:pos>`.
This removes the pipeline latency but makes modular upgrades impossible.

---

## 12. Key Takeaways

1.  **VAD is critical**: It is the "Trigger" for the whole DAG. If it's flawed, the system is flawed.
2.  **Padding saves lives**: Never feed exact VAD boundaries to ASR. Add context.
3.  **Latency Budget**: If Total Latency limit is 500ms, and ASR takes 400ms, VAD+Diarization must happen in 100ms.
4.  **Async Design**: Run ASR and Diarization in parallel threads, not sequential, to minimize wall-clock time.

---

**Originally published at:** [arunbaby.com/speech-tech/0049-speech-pipeline-dependencies](https://www.arunbaby.com/speech-tech/0049-speech-pipeline-dependencies/)

*If you found this helpful, consider sharing it with others who might benefit.*
