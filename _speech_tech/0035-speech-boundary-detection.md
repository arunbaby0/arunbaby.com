---
title: "Speech Boundary Detection"
day: 35
related_dsa_day: 35
related_ml_day: 35
related_agents_day: 35
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
- If `Energy > Threshold`, assume speech.
- **Pros:** Ultra-fast, O(1).
- **Cons:** Fails with background noise (AC, traffic).

### 2. Gaussian Mixture Models (GMM)
- Train two GMMs: one for Speech, one for Noise.
- Calculate Log-Likelihood Ratio (LLR).
- **Pros:** Robust to stationary noise.

### 3. Deep Learning VAD (Silero, WebRTC)
- **Model:** LSTM or small CNN.
- **Input:** Mel-spectrogram frames.
- **Output:** Probability of speech `P(speech)`.
- **Latency:** < 10ms.

## 3. Forced Alignment (Word/Phone Boundaries)

Given Audio + Transcript, find the exact timestamp of every word.

**Algorithm:**
1. **Lexicon:** Convert text to phonemes. "Hello" -> `HH AH L OW`.
2. **HMM (Hidden Markov Model):**
 - States: Phonemes.
 - Transitions: `HH -> AH -> L -> OW`.
3. **Viterbi Alignment:** Find the most likely path through the HMM states that aligns with the acoustic features (MFCCs).

**Tool:** **Montreal Forced Aligner (MFA)** is the industry standard.

## 4. CTC Segmentation

Modern End-to-End ASR uses **CTC (Connectionist Temporal Classification)**.

- **CTC Output:** A spike probability for each character/phoneme.
- **Boundary:** The peak of the spike is the center of the unit. The "blank" token represents the boundary.
- **Advantage:** No HMMs needed. Works directly with neural networks.

## 5. System Design: Smart Speaker Wake Word

**Scenario:** "Alexa, play music."

**Pipeline:**
1. **Hardware VAD:** Low-power DSP checks for energy. (0.1 mW).
2. **Streaming Wake Word:** Small CNN runs on-device. Checks for "Alexa".
3. **Boundary Detection:**
 - Start of Command: Immediately after "Alexa".
 - End of Command: Detect > 700ms of silence.
4. **Cloud ASR:** Send only the command audio to the cloud.

**The "End-of-Query" Problem:**
- User: "Play..." (pause) "...music."
- If timeout is too short, we cut them off.
- If timeout is too long, the system feels sluggish.
- **Solution:** **Endpointing**. Use a model to predict "Is the user done?" based on prosody (pitch drop) and semantics (complete sentence?).

## 6. Deep Dive: Pyannote (Speaker Segmentation)

**Pyannote** is a popular library for diarization.

**Pipeline:**
1. **Segmentation:** A sliding window model (SincNet) predicts `[Speaker_1, Speaker_2, ...]` activity for every frame.
2. **Embedding:** Extract x-vectors for each segment.
3. **Clustering:** Group segments by speaker similarity.

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

``python
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
``

## 10. Deep Dive: WebRTC VAD (Industry Standard)

WebRTC VAD is used in Zoom, Google Meet, etc.

**Algorithm:**
1. **Gaussian Mixture Model (GMM):** Two GMMs (Speech vs. Noise).
2. **Features:** 6 spectral features per frame.
3. **Modes:** Aggressive (0), Normal (1), Conservative (2), Very Conservative (3).

**Python Wrapper:**
``python
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

vad = webrtcvad.Vad(2) # Mode 2 (Normal)
audio, sample_rate = read_wave('audio.wav')

frames = frame_generator(30, audio, sample_rate) # 30ms frames
for frame in frames:
 is_speech = vad.is_speech(frame, sample_rate)
 print(f"Speech: {is_speech}")
``

## 11. Deep Dive: Forced Alignment with Montreal Forced Aligner

**Installation:**
``bash
conda install -c conda-forge montreal-forced-aligner
``

**Usage:**
``bash
# 1. Prepare data
# Directory structure:
# corpus/
# audio1.wav
# audio1.txt # Transcript
# audio2.wav
# audio2.txt

# 2. Download acoustic model and dictionary
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# 3. Align
mfa align corpus/ english_us_arpa english_us_arpa output/

# 4. Output: TextGrid files with word/phone timestamps
``

**Parsing TextGrid:**
``python
import textgrid

tg = textgrid.TextGrid.fromFile("output/audio1.TextGrid")

# Extract word boundaries
words_tier = tg.getFirst('words')
for interval in words_tier:
 if interval.mark: # Non-empty
 print(f"{interval.mark}: {interval.minTime:.2f}s - {interval.maxTime:.2f}s")

# Extract phone boundaries
phones_tier = tg.getFirst('phones')
for interval in phones_tier:
 if interval.mark:
 print(f"{interval.mark}: {interval.minTime:.2f}s - {interval.maxTime:.2f}s")
``

## 12. Deep Dive: CTC-Based Segmentation

Modern End-to-End ASR uses CTC. We can extract boundaries from CTC alignments.

``python
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
``

## 13. Deep Dive: Endpointing (End-of-Query Detection)

**Problem:** When should the system stop listening?

**Naive Approach:** Fixed timeout (e.g., 700ms of silence).
**Problem:** Cuts off slow speakers, feels sluggish for fast speakers.

**Adaptive Endpointing:**
``python
class AdaptiveEndpointer:
 def __init__(self):
 self.base_timeout = 0.7 # 700ms
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
``

## 14. Deep Dive: Speaker Diarization Boundaries

**Pyannote Pipeline:**
``python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Apply on audio file
diarization = pipeline("audio.wav")

# Extract speaker segments
for turn, _, speaker in diarization.itertracks(yield_label=True):
 print(f"Speaker {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
``

**Custom Post-Processing:**
``python
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
``

## 15. System Design: Real-Time Podcast Transcription

**Requirements:**
- Transcribe 1-hour podcast in < 5 minutes.
- Accurate speaker labels.
- Word-level timestamps.

**Architecture:**
1. **VAD:** Silero VAD to remove silence (reduces audio by 30%).
2. **Diarization:** Pyannote to get speaker segments.
3. **ASR:** Whisper Large-v2 on each speaker segment.
4. **Forced Alignment:** MFA to get word timestamps.
5. **Post-Processing:** Punctuation restoration, capitalization.

**Pipeline Code:**
``python
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
``

## 16. Production Considerations

1. **Latency Budget:**
 - VAD: < 10ms
 - Diarization: Can be offline (batch)
 - Forced Alignment: < 1s per minute of audio
2. **Accuracy vs. Speed:**
 - For live captions: Use streaming VAD + fast ASR (Conformer-S).
 - For archival: Use offline diarization + Whisper Large.
3. **Edge Deployment:**
 - VAD runs on-device (DSP or NPU).
 - ASR runs in cloud (GPU).

- ASR runs in cloud (GPU).

## 17. Deep Dive: Silence Detection vs. Pause Detection

**Silence:** Complete absence of sound (background noise only).
**Pause:** Brief gap in speech (breathing, hesitation).

**Why Distinguish?**
- **Podcast Editing:** Remove silences (dead air), keep pauses (natural rhythm).
- **Transcription:** Pauses within a sentence shouldn't trigger segmentation.

**Algorithm:**
``python
def classify_gap(audio_segment, duration):
 # Compute RMS energy
 energy = np.sqrt(np.mean(audio_segment**2))
 
 # Thresholds
 SILENCE_ENERGY = 0.01
 PAUSE_DURATION_MAX = 0.5 # 500ms
 
 if energy < SILENCE_ENERGY:
 if duration > PAUSE_DURATION_MAX:
 return "SILENCE"
 else:
 return "PAUSE"
 else:
 return "SPEECH"
``

## 18. Deep Dive: Music vs. Speech Segmentation

**Problem:** Podcast has intro music. Don't transcribe it.

**Features that Distinguish Music from Speech:**
1. **Spectral Flux:** Music has more variation in spectrum over time.
2. **Zero Crossing Rate:** Music (especially instrumental) has higher ZCR.
3. **Harmonic-to-Noise Ratio (HNR):** Speech has lower HNR (more noise-like).
4. **Rhythm:** Music has regular beat patterns.

**Model:**
``python
import librosa

def is_music(audio, sr=16000):
 # Extract features
 spectral_flux = np.mean(librosa.onset.onset_strength(y=audio, sr=sr))
 zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
 
 # Simple classifier (in practice, use a trained model)
 if spectral_flux > 15 and zcr > 0.15:
 return True # Music
 else:
 return False # Speech
``

**Production:** Use a pre-trained classifier (e.g., **Essentia** library).

## 19. Advanced: Neural Endpointing Models

**State-of-the-Art:** Use a Transformer to predict "Is the user done speaking?"

**Architecture:**
``
Audio Features (Mel-Spec) → Conformer Encoder → Binary Classifier
 ↓
 Contextual Features (ASR Partial Hypothesis)
``

**Training Data:**
- Positive Examples: Complete utterances.
- Negative Examples: Utterances with artificial mid-sentence cuts.

**Inference:**
``python
class NeuralEndpointer:
 def __init__(self, model_path):
 self.model = load_model(model_path)
 self.buffer = []
 
 def process_frame(self, audio_frame, partial_transcript):
 self.buffer.append(audio_frame)
 
 # Extract features
 mel_spec = compute_mel_spectrogram(self.buffer)
 text_features = encode_text(partial_transcript)
 
 # Predict
 prob_end = self.model(mel_spec, text_features)
 
 if prob_end > 0.8:
 return "END_OF_QUERY"
 else:
 return "CONTINUE"
``

## 20. Case Study: Zoom's Noise Suppression + VAD

**Challenge:** Distinguish speech from keyboard typing, dog barking, etc.

**Zoom's Approach:**
1. **Noise Suppression:** RNNoise (Recurrent Neural Network for noise reduction).
2. **VAD:** Custom DNN trained on "clean speech" vs. "suppressed noise".
3. **Adaptive Thresholds:** Adjust sensitivity based on SNR (Signal-to-Noise Ratio).

**Result:** 95% accuracy in noisy environments (cafes, airports).

## 21. Evaluation Metrics for Boundary Detection

**1. Frame-Level Accuracy:**
- Precision/Recall on speech frames.
- **Problem:** Doesn't penalize boundary errors (off by 100ms is same as off by 1ms).

**2. Boundary Tolerance:**
- A boundary is "correct" if within ±50ms of ground truth.
- **Metric:** F1-score with tolerance.

**3. Segmentation Error Rate (SER):**
`SER = \frac{FA + Miss + Confusion}{Total\_Frames}`
- **FA (False Alarm):** Silence marked as speech.
- **Miss:** Speech marked as silence.
- **Confusion:** Speaker A marked as Speaker B.

**4. Diarization Error Rate (DER):**
- Standard metric for speaker diarization.
- **SOTA:** DER < 5% on LibriSpeech.

## 22. Common Pitfalls and How to Avoid Them

**Pitfall 1: Fixed Thresholds**
- Energy threshold works in quiet room, fails in noisy cafe.
- **Fix:** Adaptive thresholds based on background noise estimation.

**Pitfall 2: Ignoring Context**
- A 200ms pause might be a breath (keep) or end of sentence (cut).
- **Fix:** Use prosody (pitch contour) and partial transcript to decide.

**Pitfall 3: Over-Segmentation**
- Cutting every pause creates choppy audio.
- **Fix:** Minimum segment duration (e.g., 1 second).

**Pitfall 4: Not Handling Overlapping Speech**
- Two people talking at once.
- **Fix:** Use multi-label VAD (predict multiple speakers simultaneously).

**Pitfall 5: Latency vs. Accuracy Trade-off**
- Waiting for more context improves accuracy but increases latency.
- **Fix:** Use a two-pass system: fast VAD for real-time, slow diarization for archival.

## 23. Advanced: Phoneme Boundary Detection with Deep Learning

**Traditional:** HMM-based forced alignment.
**Modern:** End-to-end neural networks.

**Wav2Vec 2.0 for Phoneme Segmentation:**
``python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

# This model outputs phonemes directly
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits

# Decode to phonemes
predicted_ids = torch.argmax(logits, dim=-1)
phonemes = processor.batch_decode(predicted_ids)
``

**Boundary Extraction:** Use CTC alignment to get start/end frames for each phoneme.

## 24. Deep Dive: Hardware Implementation of VAD

**Constraint:** Always-on VAD must consume < 1mW.

**Architecture:**
1. **Analog VAD:**
 - Uses analog circuits (comparators) to detect energy above noise floor.
 - **Power:** ~10µW.
 - **Accuracy:** Low (triggers on door slams).
2. **Digital VAD (DSP):**
 - Runs on a low-power DSP (e.g., Cadence HiFi).
 - Extracts simple features (ZCR, Energy).
 - **Power:** ~100µW.
3. **Neural VAD (NPU):**
 - Tiny CNN/RNN on specialized NPU (e.g., Syntiant).
 - **Power:** ~1mW.
 - **Accuracy:** High (ignores noise).

**Wake-on-Voice Pipeline:**
``
Mic → Analog VAD (Is there sound?) → DSP (Is it speech?) → NPU (Is it "Alexa"?) → AP (Cloud ASR)
``

## 25. Deep Dive: Data Augmentation for Robust VAD

**Problem:** VAD trained on clean speech fails in noise.

**Augmentation Strategy:**
1. **Noise Injection:** Mix speech with MUSAN dataset (music, speech, noise) at various SNRs (0dB to 20dB).
2. **Reverberation:** Convolve with Room Impulse Responses (RIRs).
3. **SpecAugment:** Mask time/frequency bands in spectrogram.

**Code:**
``python
import torchaudio

def augment_vad_data(speech, noise, rir, snr_db):
 # 1. Apply Reverb
 reverbed = torchaudio.functional.fftconvolve(speech, rir)
 
 # 2. Mix Noise
 speech_power = speech.norm(p=2)
 noise_power = noise.norm(p=2)
 
 snr = 10 ** (snr_db / 20)
 scale = snr * noise_power / speech_power
 noisy_speech = (scale * speech + noise) / (scale + 1)
 
 return noisy_speech
``

## 26. Interview Questions for Speech Boundary Detection

**Q1: How does WebRTC VAD work?**
*Answer:* It uses Gaussian Mixture Models (GMMs) to model speech and noise distributions based on 6 spectral features. It calculates the log-likelihood ratio (LLR) to decide if a frame is speech.

**Q2: What is the difference between VAD and Diarization?**
*Answer:* VAD is binary (Speech vs. Non-Speech). Diarization is multi-class (Speaker A vs. Speaker B vs. Silence). VAD is a prerequisite for Diarization.

**Q3: How do you handle "cocktail party" scenarios (overlapping speech)?**
*Answer:* Standard VAD fails. Use **Overlapped Speech Detection (OSD)** models, often treated as a multi-label classification problem (0, 1, or 2 speakers active).

**Q4: Why is CTC used for segmentation?**
*Answer:* CTC aligns the input sequence (audio frames) with the output sequence (text) without requiring frame-level alignment labels during training. The "spikes" in CTC probability indicate the center of a character/phoneme.

**Q5: How do you evaluate VAD latency?**
*Answer:* Measure the time from the physical onset of speech to the system triggering. For endpointing, measure the time from speech offset to system closing the microphone.

## 27. Future Trends in Boundary Detection

**1. Audio-Visual VAD:**
- Use lip movement to detect speech.
- **Benefit:** Works perfectly in 100dB noise (e.g., concerts).
- **Challenge:** Requires camera, privacy concerns.

**2. Personalized VAD:**
- VAD that only triggers for *your* voice.
- **Mechanism:** Condition VAD on a speaker embedding (d-vector).

**3. Universal Segmentation:**
- Single model that segments Speech, Music, Sound Events (dog, car), and Speaker Identity simultaneously.

- Single model that segments Speech, Music, Sound Events (dog, car), and Speaker Identity simultaneously.

## 28. Deep Dive: SincNet for VAD

**Problem:** Standard CNNs learn arbitrary filters. For audio, we know that band-pass filters are optimal.

**Solution (SincNet):**
- Constrain the first layer of CNN to learn **band-pass filters**.
- Learn only two parameters per filter: low cutoff frequency (`f_1`) and high cutoff frequency (`f_2`).

**Equation:**
`g[n, f_1, f_2] = 2f_2 \text{sinc}(2\pi f_2 n) - 2f_1 \text{sinc}(2\pi f_1 n)`

**Benefits:**
- **Fewer Parameters:** Converges faster.
- **Interpretability:** We can visualize exactly which frequency bands the model is listening to.
- **Robustness:** Better generalization to unseen noise.

**Code:**
``python
class SincConv_fast(nn.Module):
 def __init__(self, out_channels, kernel_size, sample_rate=16000):
 super().__init__()
 self.out_channels = out_channels
 self.kernel_size = kernel_size
 self.sample_rate = sample_rate

 # Initialize filters (mel-scale)
 mel = np.linspace(0, 2595 * np.log10(1 + (sample_rate / 2) / 700), out_channels + 1)
 hz = 700 * (10 ** (mel / 2595) - 1)
 self.min_freq = hz[:-1]
 self.band_width = hz[1:] - hz[:-1]

 # Learnable parameters
 self.min_freq = nn.Parameter(torch.from_numpy(self.min_freq).float())
 self.band_width = nn.Parameter(torch.from_numpy(self.band_width).float())

 def forward(self, x):
 # Generate filters on the fly
 filters = self.get_sinc_filters()
 return F.conv1d(x, filters)
``

## 29. System Design: Building a Scalable VAD Service

**Scenario:** API that accepts audio streams and returns speech segments in real-time.

**Requirements:**
- **Latency:** < 200ms.
- **Throughput:** 10,000 concurrent streams.
- **Cost:** Minimize GPU usage.

**Architecture:**

1. **Protocol:**
 - Use **gRPC** (bidirectional streaming) or **WebSocket**.
 - Client sends chunks of 20ms audio.

2. **Load Balancing:**
 - **Envoy Proxy** for L7 load balancing.
 - Sticky sessions not required (VAD is mostly stateless, or state is small).

3. **Compute Engine:**
 - **CPU vs GPU:** VAD models are small (e.g., Silero is < 1MB).
 - **Decision:** Run on **CPU** (c5.large). Cheaper and easier to scale than GPU for this specific workload.
 - **SIMD:** Use AVX-512 instructions for DSP operations.

4. **Batching:**
 - Even on CPU, batching helps.
 - Accumulate 20ms chunks from 100 users → Run inference → Send results.

5. **Scaling Policy:**
 - Metric: CPU Utilization.
 - Scale out when CPU > 60%.

**API Definition (Protobuf):**
``protobuf
service VadService {
 rpc DetectSpeech(stream AudioChunk) returns (stream SpeechEvent);
}

message AudioChunk {
 bytes data = 1;
 int32 sample_rate = 2;
}

message SpeechEvent {
 enum EventType {
 START_OF_SPEECH = 0;
 END_OF_SPEECH = 1;
 ACTIVE = 2;
 }
 EventType type = 1;
 float timestamp = 2;
}
``

## 30. Further Reading

1. **"WebRTC Voice Activity Detector" (Google):** The VAD used in billions of devices.
2. **"Pyannote.audio: Neural Building Blocks for Speaker Diarization" (Bredin et al., 2020):** State-of-the-art diarization.
3. **"Montreal Forced Aligner" (McAuliffe et al., 2017):** The standard for forced alignment.
4. **"End-to-End Neural Segmentation and Diarization" (Fujita et al., 2019):** Joint modeling.
5. **"Silero VAD" (Silero Team):** Fast, accurate, open-source VAD.

## 31. Ethical Considerations

**1. Privacy and "Always-On" Listening:**
- VAD is the gatekeeper. If it triggers falsely, private conversations are sent to the cloud.
- **Mitigation:** Process VAD and Wake Word strictly on-device. Only stream audio *after* explicit activation.
- **Visual Indicators:** Hardware LEDs must hard-wire to the microphone circuit to indicate recording.

**2. Bias in VAD:**
- VAD models trained on adult male speech may fail for children or high-pitched voices.
- **Impact:** Smart speakers ignoring kids or women.
- **Fix:** Train on diverse datasets (LibriTTS, Common Voice) with balanced demographics.

**3. Surveillance:**
- Advanced diarization can track who said what in a meeting.
- **Risk:** Employee monitoring, chilling effect on free speech.
- **Policy:** Explicit consent, data retention policies (delete after 24h).

## 32. Conclusion

Speech boundary detection is the unsung hero of speech technology. Without accurate VAD, smart speakers would drain batteries listening to silence. Without forced alignment, podcast editors would spend hours manually cutting audio. Without diarization, meeting transcripts would be an incomprehensible wall of text. The field has evolved from simple energy thresholds to sophisticated neural models that understand prosody, semantics, and speaker identity. As we move toward always-on voice interfaces and real-time translation, the demand for low-latency, high-accuracy boundary detection will only grow.

## 33. Summary

| Level | Task | Technology |
| :--- | :--- | :--- |
| **Signal** | Speech vs. Silence | WebRTC VAD, Silero |
| **Speaker** | Speaker A vs. B | Pyannote, x-vectors |
| **Linguistic** | Word Timestamps | Montreal Forced Aligner, CTC |
| **Semantic** | Turn-taking | Endpointing Models |
| **Advanced** | Music vs. Speech | Spectral Features, Essentia |
| **Hardware** | Low Power | Analog VAD, DSP, NPU |

---

**Originally published at:** [arunbaby.com/speech-tech/0035-speech-boundary-detection](https://www.arunbaby.com/speech-tech/0035-speech-boundary-detection/)
