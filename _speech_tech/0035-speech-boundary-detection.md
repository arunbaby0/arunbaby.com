---
title: "Speech Boundary Detection"
day: 35
collection: speech_tech
categories:
  - speech_tech
tags:
  - vad
  - segmentation
  - forced-alignment
  - phonetics
subdomain: "Audio Processing"
tech_stack: [WebRTC VAD, Pyannote, Montreal Forced Aligner, CTC]
scale: "Streaming, < 10ms latency"
companies: [Zoom, Spotify, Descript, Rev]
---

**"Knowing when to listen and when to stop."**

## 1. The Problem: Segmentation

Speech is a continuous stream. Computers need discrete units.
- **VAD (Voice Activity Detection):** Speech vs. Silence.
- **Speaker Diarization:** Speaker A vs. Speaker B.
- **Word Boundary:** "Ice cream" vs. "I scream".
- **Phoneme Boundary:** Start/End of /s/, /p/, /t/.

**Applications:**
- **Smart Speakers:** Only wake up on speech (save battery).
- **Transcription:** Chop long audio into 30s chunks for ASR.
- **Editing:** "Remove silences" feature in podcast editors.

## 2. Voice Activity Detection (VAD)

The most fundamental boundary.

### 1. Energy-Based (Classical)
- Calculate Short-Time Energy (STE).
- If $Energy > Threshold$, assume speech.
- **Pros:** Ultra-fast, $O(1)$.
- **Cons:** Fails with background noise (AC, traffic).

### 2. Gaussian Mixture Models (GMM)
- Train two GMMs: one for Speech, one for Noise.
- Calculate Log-Likelihood Ratio (LLR).
- **Pros:** Robust to stationary noise.

### 3. Deep Learning VAD (Silero, WebRTC)
- **Model:** LSTM or small CNN.
- **Input:** Mel-spectrogram frames.
- **Output:** Probability of speech $P(speech)$.
- **Latency:** < 10ms.

## 3. Forced Alignment (Word/Phone Boundaries)

Given Audio + Transcript, find the exact timestamp of every word.

**Algorithm:**
1.  **Lexicon:** Convert text to phonemes. "Hello" -> `HH AH L OW`.
2.  **HMM (Hidden Markov Model):**
    - States: Phonemes.
    - Transitions: `HH -> AH -> L -> OW`.
3.  **Viterbi Alignment:** Find the most likely path through the HMM states that aligns with the acoustic features (MFCCs).

**Tool:** **Montreal Forced Aligner (MFA)** is the industry standard.

## 4. CTC Segmentation

Modern End-to-End ASR uses **CTC (Connectionist Temporal Classification)**.

- **CTC Output:** A spike probability for each character/phoneme.
- **Boundary:** The peak of the spike is the center of the unit. The "blank" token represents the boundary.
- **Advantage:** No HMMs needed. Works directly with neural networks.

## 5. System Design: Smart Speaker Wake Word

**Scenario:** "Alexa, play music."

**Pipeline:**
1.  **Hardware VAD:** Low-power DSP checks for energy. (0.1 mW).
2.  **Streaming Wake Word:** Small CNN runs on-device. Checks for "Alexa".
3.  **Boundary Detection:**
    - Start of Command: Immediately after "Alexa".
    - End of Command: Detect > 700ms of silence.
4.  **Cloud ASR:** Send only the command audio to the cloud.

**The "End-of-Query" Problem:**
- User: "Play..." (pause) "...music."
- If timeout is too short, we cut them off.
- If timeout is too long, the system feels sluggish.
- **Solution:** **Endpointing**. Use a model to predict "Is the user done?" based on prosody (pitch drop) and semantics (complete sentence?).

## 6. Deep Dive: Pyannote (Speaker Segmentation)

**Pyannote** is a popular library for diarization.

**Pipeline:**
1.  **Segmentation:** A sliding window model (SincNet) predicts `[Speaker_1, Speaker_2, ...]` activity for every frame.
2.  **Embedding:** Extract x-vectors for each segment.
3.  **Clustering:** Group segments by speaker similarity.

## 7. Real-World Case Studies

### Case Study 1: Spotify Podcast Ad Insertion
- **Problem:** Insert ads at natural breaks.
- **Solution:** Detect "Topic Boundaries".
- **Features:** Long pauses, change in speaker distribution, semantic shift (BERT on transcript).

### Case Study 2: Descript (Audio Editing)
- **Feature:** "Shorten Word Gaps".
- **Tech:** Forced Alignment to find precise start/end of every word.
- **Action:** If `gap > 1.0s`, cut audio to `0.5s` and cross-fade.

## 8. Summary

| Level | Task | Technology |
| :--- | :--- | :--- |
| **Signal** | Speech vs. Silence | WebRTC VAD, Silero |
| **Speaker** | Speaker A vs. B | Pyannote, x-vectors |
| **Linguistic** | Word Timestamps | Montreal Forced Aligner, CTC |
| **Semantic** | Turn-taking | Endpointing Models |

## 9. Deep Dive: Implementing a Production VAD

Let's build a robust VAD using energy + spectral features.

```python
import numpy as np
import librosa

class ProductionVAD:
    def __init__(self, sr=16000, frame_length=0.025, frame_shift=0.010):
        self.sr = sr
        self.frame_length = int(sr * frame_length)
        self.frame_shift = int(sr * frame_shift)
        
        # Thresholds (tuned on dev set)
        self.energy_threshold = 0.02
        self.zcr_threshold = 0.3
        self.spectral_centroid_threshold = 2000
        
    def extract_features(self, audio):
        # Short-Time Energy
        energy = librosa.feature.rms(y=audio, frame_length=self.frame_length, 
                                      hop_length=self.frame_shift)[0]
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length,
                                                   hop_length=self.frame_shift)[0]
        
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, n_fft=self.frame_length, 
            hop_length=self.frame_shift)[0]
        
        return energy, zcr, spectral_centroid
    
    def detect(self, audio):
        energy, zcr, spectral_centroid = self.extract_features(audio)
        
        # Decision logic
        speech_frames = (
            (energy > self.energy_threshold) &
            (zcr < self.zcr_threshold) &
            (spectral_centroid > self.spectral_centroid_threshold)
        )
        
        return speech_frames
    
    def get_speech_segments(self, audio):
        speech_frames = self.detect(audio)
        
        # Convert frame indices to time
        segments = []
        in_speech = False
        start_idx = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start_idx = i
                in_speech = True
            elif not is_speech and in_speech:
                start_time = start_idx * self.frame_shift / self.sr
                end_time = i * self.frame_shift / self.sr
                segments.append((start_time, end_time))
                in_speech = False
        
        return segments

# Usage
vad = ProductionVAD()
audio, sr = librosa.load("audio.wav", sr=16000)
segments = vad.get_speech_segments(audio)
print(f"Speech segments: {segments}")
```

## 10. Deep Dive: WebRTC VAD (Industry Standard)

WebRTC VAD is used in Zoom, Google Meet, etc.

**Algorithm:**
1.  **Gaussian Mixture Model (GMM):** Two GMMs (Speech vs. Noise).
2.  **Features:** 6 spectral features per frame.
3.  **Modes:** Aggressive (0), Normal (1), Conservative (2), Very Conservative (3).

**Python Wrapper:**
```python
import webrtcvad
import struct

def read_wave(path):
    with wave.open(path, 'rb') as wf:
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
    return pcm_data, sample_rate

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n

vad = webrtcvad.Vad(2)  # Mode 2 (Normal)
audio, sample_rate = read_wave('audio.wav')

frames = frame_generator(30, audio, sample_rate)  # 30ms frames
for frame in frames:
    is_speech = vad.is_speech(frame, sample_rate)
    print(f"Speech: {is_speech}")
```

## 11. Deep Dive: Forced Alignment with Montreal Forced Aligner

**Installation:**
```bash
conda install -c conda-forge montreal-forced-aligner
```

**Usage:**
```bash
# 1. Prepare data
# Directory structure:
# corpus/
#   audio1.wav
#   audio1.txt  # Transcript
#   audio2.wav
#   audio2.txt

# 2. Download acoustic model and dictionary
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# 3. Align
mfa align corpus/ english_us_arpa english_us_arpa output/

# 4. Output: TextGrid files with word/phone timestamps
```

**Parsing TextGrid:**
```python
import textgrid

tg = textgrid.TextGrid.fromFile("output/audio1.TextGrid")

# Extract word boundaries
words_tier = tg.getFirst('words')
for interval in words_tier:
    if interval.mark:  # Non-empty
        print(f"{interval.mark}: {interval.minTime:.2f}s - {interval.maxTime:.2f}s")

# Extract phone boundaries
phones_tier = tg.getFirst('phones')
for interval in phones_tier:
    if interval.mark:
        print(f"{interval.mark}: {interval.minTime:.2f}s - {interval.maxTime:.2f}s")
```

## 12. Deep Dive: CTC-Based Segmentation

Modern End-to-End ASR uses CTC. We can extract boundaries from CTC alignments.

```python
import torch
import torch.nn.functional as F

def ctc_segmentation(logits, text, blank_id=0):
    """
    logits: (T, vocab_size) - CTC output probabilities
    text: Ground truth text
    Returns: List of (char, start_frame, end_frame)
    """
    T = logits.shape[0]
    probs = F.softmax(logits, dim=-1)
    
    # Get most likely path (greedy decoding)
    path = torch.argmax(probs, dim=-1)
    
    # Collapse repeated characters and remove blanks
    segments = []
    prev_char = blank_id
    start_frame = 0
    
    for t in range(T):
        char = path[t].item()
        
        if char != blank_id and char != prev_char:
            if prev_char != blank_id:
                segments.append((prev_char, start_frame, t-1))
            start_frame = t
        
        prev_char = char
    
    # Add last segment
    if prev_char != blank_id:
        segments.append((prev_char, start_frame, T-1))
    
    return segments
```

## 13. Deep Dive: Endpointing (End-of-Query Detection)

**Problem:** When should the system stop listening?

**Naive Approach:** Fixed timeout (e.g., 700ms of silence).
**Problem:** Cuts off slow speakers, feels sluggish for fast speakers.

**Adaptive Endpointing:**
```python
class AdaptiveEndpointer:
    def __init__(self):
        self.base_timeout = 0.7  # 700ms
        self.min_timeout = 0.3
        self.max_timeout = 1.5
        
    def compute_timeout(self, speaking_rate, prosody_features):
        # speaking_rate: words per second
        # prosody_features: pitch drop, energy drop
        
        # Slow speakers get longer timeout
        rate_factor = 1.0 / (speaking_rate + 0.1)
        
        # Falling intonation (end of sentence) -> shorter timeout
        pitch_drop = prosody_features['pitch_drop']
        prosody_factor = 1.0 - (pitch_drop * 0.5)
        
        timeout = self.base_timeout * rate_factor * prosody_factor
        timeout = np.clip(timeout, self.min_timeout, self.max_timeout)
        
        return timeout
```

## 14. Deep Dive: Speaker Diarization Boundaries

**Pyannote Pipeline:**
```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Apply on audio file
diarization = pipeline("audio.wav")

# Extract speaker segments
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
```

**Custom Post-Processing:**
```python
def merge_short_segments(diarization, min_duration=1.0):
    """Merge segments shorter than min_duration with neighbors"""
    segments = list(diarization.itertracks(yield_label=True))
    merged = []
    
    i = 0
    while i < len(segments):
        turn, _, speaker = segments[i]
        duration = turn.end - turn.start
        
        if duration < min_duration and i > 0:
            # Merge with previous segment
            prev_turn, _, prev_speaker = merged[-1]
            merged[-1] = (Segment(prev_turn.start, turn.end), _, prev_speaker)
        else:
            merged.append(segments[i])
        
        i += 1
    
    return merged
```

## 15. System Design: Real-Time Podcast Transcription

**Requirements:**
- Transcribe 1-hour podcast in < 5 minutes.
- Accurate speaker labels.
- Word-level timestamps.

**Architecture:**
1.  **VAD:** Silero VAD to remove silence (reduces audio by 30%).
2.  **Diarization:** Pyannote to get speaker segments.
3.  **ASR:** Whisper Large-v2 on each speaker segment.
4.  **Forced Alignment:** MFA to get word timestamps.
5.  **Post-Processing:** Punctuation restoration, capitalization.

**Pipeline Code:**
```python
def transcribe_podcast(audio_path):
    # 1. VAD
    speech_segments = silero_vad(audio_path)
    
    # 2. Diarization
    diarization = pyannote_diarize(audio_path)
    
    # 3. ASR per speaker segment
    transcripts = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_audio = extract_segment(audio_path, turn.start, turn.end)
        text = whisper_transcribe(segment_audio)
        
        # 4. Forced Alignment
        word_timestamps = mfa_align(segment_audio, text)
        
        transcripts.append({
            'speaker': speaker,
            'start': turn.start,
            'end': turn.end,
            'text': text,
            'words': word_timestamps
        })
    
    return transcripts
```

## 16. Production Considerations

1.  **Latency Budget:**
    - VAD: < 10ms
    - Diarization: Can be offline (batch)
    - Forced Alignment: < 1s per minute of audio
2.  **Accuracy vs. Speed:**
    - For live captions: Use streaming VAD + fast ASR (Conformer-S).
    - For archival: Use offline diarization + Whisper Large.
3.  **Edge Deployment:**
    - VAD runs on-device (DSP or NPU).
    - ASR runs in cloud (GPU).

## 17. Summary

| Level | Task | Technology |
| :--- | :--- | :--- |
| **Signal** | Speech vs. Silence | WebRTC VAD, Silero |
| **Speaker** | Speaker A vs. B | Pyannote, x-vectors |
| **Linguistic** | Word Timestamps | Montreal Forced Aligner, CTC |
| **Semantic** | Turn-taking | Endpointing Models |

---

**Originally published at:** [arunbaby.com/speech-tech/0035-speech-boundary-detection](https://www.arunbaby.com/speech-tech/0035-speech-boundary-detection/)
