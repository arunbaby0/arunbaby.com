---
title: "Multi-Speaker ASR"
day: 12
related_dsa_day: 12
related_ml_day: 12
related_agents_day: 12
collection: speech_tech
categories:
 - speech-tech
tags:
 - asr
 - multi-speaker
 - speaker-diarization
 - overlapping-speech
 - real-time
 - streaming
subdomain: Speech Recognition
tech_stack: [Python, PyTorch, Whisper, PyAnnote, WebSocket, NumPy, SciPy]
scale: "Real-time, < 300ms latency, 2-10 speakers"
companies: [Google, Meta, Microsoft, Zoom, Otter.ai, Fireflies.ai, AssemblyAI]
---

**Build production multi-speaker ASR systems: Combine speech recognition, speaker diarization, and overlap handling for real-world conversations.**

## Problem Statement

Design a **multi-speaker ASR system** that can:

1. **Recognize speech** from multiple speakers in a conversation
2. **Identify who spoke** each word/sentence (speaker diarization)
3. **Handle overlapping speech** when multiple people speak simultaneously
4. **Work in real-time** with < 300ms latency for live transcription
5. **Scale** to meetings with 2-10 speakers

### Why Is This Hard?

**Single-speaker ASR** (covered in ) assumes:
- ✅ One speaker at a time
- ✅ No speaker changes mid-sentence
- ✅ No overlapping speech

**Real-world conversations** break all these assumptions:

``
Time: 0s 1s 2s 3s 4s
Speaker A: "So I think we should..."
Speaker B: "Wait, can I..."
Speaker C: "Actually..."
 ↑ Overlap! ↑ ↑ Overlap! ↑

Single-speaker ASR would produce: "So I wait can think actually should we..."
Multi-speaker ASR must produce:
 [A, 0.0-1.5s]: "So I think we should"
 [B, 1.2-2.3s]: "Wait, can I"
 [C, 2.8-4.0s]: "Actually"
``

**The core challenges:**

| Challenge | Why It's Hard | Impact |
|-----------|---------------|--------|
| **Speaker changes** | Voice characteristics change suddenly | Acoustic model confused |
| **Overlapping speech** | Multiple audio sources mixed | Can't separate cleanly |
| **Speaker identification** | Need to know *who* said *what* | Requires speaker embeddings |
| **Real-time processing** | Must process while speakers still talking | Latency constraints |
| **Unknown # of speakers** | Don't know speaker count in advance | Can't pre-allocate resources |

### Real-World Use Cases

| Application | Requirements | Challenges |
|-------------|--------------|------------|
| **Meeting transcription** (Zoom, Teams) | 2-10 speakers, real-time | Overlaps, background noise |
| **Call center analytics** | 2 speakers (agent + customer) | Quality monitoring, compliance |
| **Podcast transcription** | 2-5 hosts + guests | High accuracy needed |
| **Courtroom transcription** | Multiple speakers, legal record | 99%+ accuracy, speaker IDs |
| **Medical consultations** | Doctor + patient(s) | HIPAA compliance, accuracy |

---

## Understanding Multi-Speaker ASR

### The Full System Pipeline

``
┌─────────────────────────────────────────────────────────────┐
│ MULTI-SPEAKER ASR PIPELINE │
└─────────────────────────────────────────────────────────────┘

Step 1: AUDIO INPUT
┌────────────────────────────────────────────┐
│ Mixed audio (all speakers combined) │
│ [Speaker A + Speaker B + Speaker C + ...]│
└────────────────┬───────────────────────────┘
 ▼
Step 2: VOICE ACTIVITY DETECTION (VAD)
┌────────────────────────────────────────────┐
│ Find speech regions (vs silence) │
│ Output: [(0.0s, 3.2s), (3.5s, 7.1s), ...] │
└────────────────┬───────────────────────────┘
 ▼
Step 3: SPEAKER DIARIZATION
┌────────────────────────────────────────────┐
│ Cluster speech by speaker │
│ Output: [(A, 0-1.5s), (B, 1.2-2.3s), ...]│
└────────────────┬───────────────────────────┘
 ▼
Step 4: ASR (per speaker segment)
┌────────────────────────────────────────────┐
│ Transcribe each speaker segment │
│ Output: [(A, "Hello"), (B, "Hi"), ...] │
└────────────────┬───────────────────────────┘
 ▼
Step 5: POST-PROCESSING
┌────────────────────────────────────────────┐
│ • Merge overlaps │
│ • Add punctuation │
│ • Format output │
└────────────────────────────────────────────┘
``

Each step has challenges! Let's dig into each.

### Why This Pipeline?

**Why not just run ASR on everything?**

Imagine you have a 1-hour meeting:
- Raw audio: 1 hour
- Actual speech: ~30 minutes (50% silence/pauses)
- Running ASR on silence: **Waste of 30 minutes compute!**

**Why not run ASR first, then diarize?**

``
Option A: ASR → Diarization (BAD)
Problem: ASR produces one continuous text blob
"Hello there hi how are you fine thanks"
↑ Can't tell where speakers change!

Option B: Diarization → ASR (GOOD)
Step 1: Find speaker segments
 [A: 0-2s], [B: 2-4s], [A: 4-6s]
Step 2: Transcribe each segment separately
 A: "Hello there"
 B: "Hi, how are you?"
 A: "Fine, thanks"
↑ Clean separation!
``

**Why separate VAD from diarization?**

- **VAD** is fast (simple energy-based or small model)
- **Diarization** is slow (needs embeddings + clustering)
- Don't waste diarization compute on silence!

**Pipeline efficiency:**

``
1 hour audio
 ↓ VAD (fast, eliminates silence)
30 min speech segments
 ↓ Diarization (slow, but only on speech)
30 min speaker-labeled segments
 ↓ ASR (slowest, but parallelizable)
Transcriptions
``

### The Mathematics Behind Speaker Embeddings

**Key question**: How do we represent a voice mathematically?

**Answer**: Deep learning learns to compress voice characteristics into a fixed-size vector.

**Training process** (simplified):

``
Step 1: Collect data
 Speaker 1: 100 utterances
 Speaker 2: 100 utterances
 ...
 Speaker 10,000: 100 utterances

Step 2: Train neural network
 Input: Audio waveform or spectrogram
 Output: 512-dimensional embedding
 
 Goal: Minimize distance between embeddings of same speaker,
 maximize distance between different speakers

Step 3: Loss function (Triplet Loss)
 Anchor: Speaker A, utterance 1
 Positive: Speaker A, utterance 2 (same speaker)
 Negative: Speaker B, utterance 1 (different speaker)
 
 Loss = max(0, distance(anchor, positive) - distance(anchor, negative) + margin)
 
 This forces:
 - distance(A_utt1, A_utt2) < distance(A_utt1, B_utt1)
``

**Visual intuition:**

``
Before training (random embeddings):
Speaker A utterances: scattered everywhere
Speaker B utterances: scattered everywhere
No clustering!

After training:
Speaker A utterances: tight cluster in embedding space
Speaker B utterances: different tight cluster, far from A
Clear separation!
``

**Why 512 dimensions?**

- **Lower (e.g., 64)**: Not enough capacity to capture all voice variations
- **Higher (e.g., 2048)**: Overfitting, slow, unnecessary
- **512**: Sweet spot (empirically found by researchers)

**What does the embedding capture?**

- Pitch/fundamental frequency
- Formant structure (vocal tract resonances)
- Speaking rate
- Accent/dialect
- Voice quality (breathy, creaky, etc.)

**What it should NOT capture** (ideally):

- Spoken words (content)
- Emotions (though it does somewhat)
- Background noise

### Comparison: Different Diarization Approaches

| Approach | How It Works | Pros | Cons | Use Case |
|----------|--------------|------|------|----------|
| **Clustering-based** | Extract embeddings → Cluster | Simple, interpretable | Needs good embeddings | General purpose |
| **End-to-end neural** | Single model: audio → labels | Best accuracy | Slow, black-box | High-accuracy needs |
| **Online diarization** | Process stream incrementally | Real-time capable | Lower accuracy | Live captions |
| **Supervised (known speakers)** | Match to registered voices | Very accurate for known speakers | Requires enrollment | Authentication, personalization |

**Example scenario: Meeting with known participants**

``python
class KnownSpeakerDiarization:
 """
 When you know who's in the meeting
 
 Much more accurate than unsupervised clustering
 """
 
 def __init__(self):
 self.speaker_profiles = {} # speaker_name → mean embedding
 
 def enroll_speaker(self, speaker_name, audio_samples):
 """
 Register a speaker
 
 Args:
 speaker_name: "Alice", "Bob", etc.
 audio_samples: List of audio clips of this speaker
 """
 # Extract embeddings from all samples
 embeddings = [
 self.extract_embedding(audio)
 for audio in audio_samples
 ]
 
 # Compute mean embedding (speaker profile)
 mean_embedding = np.mean(embeddings, axis=0)
 
 # Store
 self.speaker_profiles[speaker_name] = mean_embedding
 print(f"✓ Enrolled {speaker_name}")
 
 def identify_speaker(self, audio_segment):
 """
 Identify which registered speaker this is
 
 Much more accurate than unsupervised clustering!
 """
 # Extract embedding
 test_embedding = self.extract_embedding(audio_segment)
 
 # Compare with all registered speakers
 best_match = None
 best_similarity = -1
 
 for name, profile_embedding in self.speaker_profiles.items():
 similarity = self._cosine_similarity(test_embedding, profile_embedding)
 
 if similarity > best_similarity:
 best_similarity = similarity
 best_match = name
 
 # Threshold
 if best_similarity > 0.7:
 return best_match, best_similarity
 else:
 return "UNKNOWN", best_similarity

 def extract_embedding(self, audio):
 """Placeholder: replace with real embedding extractor"""
 import numpy as np
 audio = np.asarray(audio)
 return audio[:512] if audio.size >= 512 else np.pad(audio, (0, max(0, 512 - audio.size)))

 def _cosine_similarity(self, a, b):
 import numpy as np
 a = np.asarray(a); b = np.asarray(b)
 denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
 return float(np.dot(a, b) / denom)

# Usage
diarizer = KnownSpeakerDiarization()

# Enroll meeting participants
diarizer.enroll_speaker("Alice", alice_audio_samples)
diarizer.enroll_speaker("Bob", bob_audio_samples)
diarizer.enroll_speaker("Carol", carol_audio_samples)

# Now identify speakers in meeting
for segment in meeting_segments:
 speaker, confidence = diarizer.identify_speaker(segment)
 print(f"{speaker} ({confidence:.2f}): {transcribe(segment)}")
``

**This is how Zoom/Teams could improve**:
- Ask users to speak their name when joining
- Build speaker profile
- Use it for accurate diarization

---

## Component 1: Voice Activity Detection (VAD)

**Goal**: Find when *any* speaker is talking

**Why needed**: Don't waste compute on silence

``python
class VoiceActivityDetector:
 """
 Detect speech vs non-speech
 
 Uses energy + spectral features
 """
 
 def __init__(self, sample_rate=16000, frame_ms=30):
 self.sample_rate = sample_rate
 self.frame_size = int(sample_rate * frame_ms / 1000)
 
 def detect(self, audio):
 """
 Detect speech regions
 
 Args:
 audio: numpy array, shape (samples,)
 
 Returns:
 List of (start_time, end_time) tuples
 """
 import numpy as np
 
 # Split into frames
 frames = self._split_frames(audio)
 
 # Compute features for each frame
 is_speech = []
 for frame in frames:
 # Energy-based detection
 energy = np.mean(frame ** 2)
 
 # Spectral flatness (voice has low flatness)
 flatness = self._spectral_flatness(frame)
 
 # Simple threshold
 speech = (energy > 0.01) and (flatness < 0.5)
 is_speech.append(speech)
 
 # Convert frame-level to time segments
 segments = self._merge_segments(is_speech)
 
 return segments
 
 def _split_frames(self, audio):
 """Split audio into overlapping frames"""
 import numpy as np
 
 frames = []
 hop_size = self.frame_size // 2 # 50% overlap
 
 for i in range(0, len(audio) - self.frame_size, hop_size):
 frame = audio[i:i + self.frame_size]
 frames.append(frame)
 
 return frames
 
 def _spectral_flatness(self, frame):
 """
 Compute spectral flatness
 
 Low for voice (harmonic), high for noise (flat spectrum)
 """
 import numpy as np
 from scipy import signal
 
 # FFT
 fft = np.abs(np.fft.rfft(frame))
 
 # Geometric mean / arithmetic mean
 geometric_mean = np.exp(np.mean(np.log(fft + 1e-10)))
 arithmetic_mean = np.mean(fft)
 
 flatness = geometric_mean / (arithmetic_mean + 1e-10)
 
 return flatness
 
 def _merge_segments(self, is_speech):
 """
 Merge consecutive speech frames into segments
 
 Args:
 is_speech: List of bools per frame
 
 Returns:
 List of (start_time, end_time)
 """
 segments = []
 in_segment = False
 start = 0
 
 frame_duration = self.frame_size / self.sample_rate
 
 for i, speech in enumerate(is_speech):
 if speech and not in_segment:
 # Start new segment
 start = i * frame_duration
 in_segment = True
 elif not speech and in_segment:
 # End segment
 end = i * frame_duration
 segments.append((start, end))
 in_segment = False
 
 # Handle case where last frame is speech
 if in_segment:
 segments.append((start, len(is_speech) * frame_duration))
 
 return segments
``

**Modern approach**: Use pre-trained VAD models (more accurate)

``python
def vad_pretrained(audio, sample_rate=16000):
 """
 Use pre-trained VAD model (Silero VAD)
 
 More accurate than energy-based
 """
 import torch
 
 # Load pre-trained model
 model, utils = torch.hub.load(
 repo_or_dir='snakers4/silero-vad',
 model='silero_vad',
 force_reload=False
 )
 
 (get_speech_timestamps, _, _, _, _) = utils
 
 # Detect speech
 speech_timestamps = get_speech_timestamps(
 audio,
 model,
 sampling_rate=sample_rate,
 threshold=0.5
 )
 
 # Convert to seconds
 segments = [
 (ts['start'] / sample_rate, ts['end'] / sample_rate)
 for ts in speech_timestamps
 ]
 
 return segments
``

---

## Component 2: Speaker Diarization

**Goal**: Cluster speech segments by speaker

**Key idea**: Speakers have unique voice characteristics (embeddings)

### Speaker Embeddings

**Concept**: Convert speech to fixed-size vector that captures speaker identity

``
Speaker A: "Hello" → [0.2, 0.8, -0.3, ...] (512-dim)
Speaker A: "How are you" → [0.21, 0.79, -0.31, ...] (similar!)
Speaker B: "Hi there" → [-0.5, 0.1, 0.9, ...] (different!)
``

**Models**: x-vector, d-vector, ECAPA-TDNN

``python
class SpeakerEmbeddingExtractor:
 """
 Extract speaker embeddings using ECAPA-TDNN
 
 Embeddings capture speaker identity
 """
 
 def __init__(self):
 from speechbrain.pretrained import EncoderClassifier
 
 # Load pre-trained model
 self.model = EncoderClassifier.from_hparams(
 source="speechbrain/spkrec-ecapa-voxceleb",
 savedir="pretrained_models/spkrec-ecapa"
 )
 
 def extract(self, audio, sample_rate=16000):
 """
 Extract speaker embedding
 
 Args:
 audio: numpy array
 
 Returns:
 embedding: numpy array, shape (512,)
 """
 import torch
 
 # Convert to tensor
 audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
 
 # Extract embedding
 with torch.no_grad():
 embedding = self.model.encode_batch(audio_tensor)
 
 # Convert to numpy
 embedding = embedding.squeeze().cpu().numpy()
 
 return embedding

# Usage
extractor = SpeakerEmbeddingExtractor()

# Extract embeddings for each speech segment
segments = [(0.0, 1.5), (1.2, 2.3), (2.8, 4.0)] # From VAD
embeddings = []

for start, end in segments:
 segment_audio = audio[int(start*sr):int(end*sr)]
 embedding = extractor.extract(segment_audio)
 embeddings.append(embedding)

# Now cluster embeddings to identify speakers
``

### Clustering Embeddings

**Goal**: Group similar embeddings (same speaker)

``python
class SpeakerClustering:
 """
 Cluster speaker embeddings
 
 Same speaker → similar embeddings → same cluster
 """
 
 def __init__(self, method='spectral'):
 self.method = method
 
 def cluster(self, embeddings, num_speakers=None):
 """
 Cluster embeddings into speakers
 
 Args:
 embeddings: List of numpy arrays
 num_speakers: If known, specify; else auto-detect
 
 Returns:
 labels: Array of speaker IDs per segment
 """
 import numpy as np
 from sklearn.cluster import SpectralClustering, AgglomerativeClustering
 
 # Convert to matrix
 X = np.array(embeddings)
 
 if num_speakers is None:
 # Auto-detect number of speakers
 num_speakers = self._estimate_num_speakers(X)
 
 # Cluster
 if self.method == 'spectral':
 # Use precomputed affinity for better control
 from sklearn.metrics.pairwise import cosine_similarity
 affinity = cosine_similarity(X)
 clusterer = SpectralClustering(
 n_clusters=num_speakers,
 affinity='precomputed'
 )
 labels = clusterer.fit_predict(affinity)
 return labels
 else:
 clusterer = AgglomerativeClustering(
 n_clusters=num_speakers,
 linkage='average',
 metric='cosine'
 )
 
 labels = clusterer.fit_predict(X)
 return labels
 
 def _estimate_num_speakers(self, embeddings):
 """
 Estimate number of speakers
 
 Use eigengap heuristic or elbow method
 """
 from sklearn.cluster import SpectralClustering
 import numpy as np
 
 # Try different numbers of clusters
 max_speakers = min(10, len(embeddings))
 
 scores = []
 for k in range(2, max_speakers + 1):
 from sklearn.metrics.pairwise import cosine_similarity
 aff = cosine_similarity(embeddings)
 clusterer = SpectralClustering(n_clusters=k, affinity='precomputed')
 labels = clusterer.fit_predict(aff)
 
 # Compute silhouette score
 from sklearn.metrics import silhouette_score
 score = silhouette_score(embeddings, labels, metric='cosine')
 scores.append(score)
 
 # Pick k with highest score
 best_k = np.argmax(scores) + 2
 
 return best_k
``

**Production library**: Use `pyannote.audio` (state-of-the-art)

``python
def diarize_with_pyannote(audio_path):
 """
 Speaker diarization using pyannote.audio
 
 Production-ready, state-of-the-art
 """
 from pyannote.audio import Pipeline
 
 # Load pre-trained pipeline
 pipeline = Pipeline.from_pretrained(
 "pyannote/speaker-diarization",
 use_auth_token="YOUR_HF_TOKEN"
 )
 
 # Run diarization
 diarization = pipeline(audio_path)
 
 # Extract speaker segments
 segments = []
 for turn, _, speaker in diarization.itertracks(yield_label=True):
 segments.append({
 'speaker': speaker,
 'start': turn.start,
 'end': turn.end
 })
 
 return segments

# Example output:
# [
# {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 1.5},
# {'speaker': 'SPEAKER_01', 'start': 1.2, 'end': 2.3},
# {'speaker': 'SPEAKER_00', 'start': 2.8, 'end': 4.0},
# ]
``

---

## Component 3: ASR Per Speaker

**Goal**: Transcribe each speaker segment

``python
class MultiSpeakerASR:
 """
 Complete multi-speaker ASR system
 
 Combines VAD + Diarization + ASR
 """
 
 def __init__(self):
 # Load models
 self.vad = VoiceActivityDetector()
 self.embedding_extractor = SpeakerEmbeddingExtractor()
 self.clustering = SpeakerClustering()
 
 # ASR model (Whisper)
 import whisper
 self.asr_model = whisper.load_model("base")
 
 def transcribe(self, audio, sample_rate=16000):
 """
 Multi-speaker transcription
 
 Returns:
 List of {speaker, start, end, text}
 """
 # Step 1: VAD
 speech_segments = self.vad.detect(audio)
 print(f"Found {len(speech_segments)} speech segments")
 
 # Step 2: Extract embeddings
 embeddings = []
 for start, end in speech_segments:
 segment = audio[int(start*sample_rate):int(end*sample_rate)]
 emb = self.embedding_extractor.extract(segment)
 embeddings.append(emb)
 
 # Step 3: Cluster by speaker
 speaker_labels = self.clustering.cluster(embeddings)
 print(f"Detected {len(set(speaker_labels))} speakers")
 
 # Step 4: Transcribe each segment
 results = []
 for i, (start, end) in enumerate(speech_segments):
 segment = audio[int(start*sample_rate):int(end*sample_rate)]
 
 # Transcribe (Whisper expects float32 numpy audio @16k)
 result = self.asr_model.transcribe(segment, fp16=False)
 text = result['text']
 
 # Add speaker label
 speaker = f"SPEAKER_{speaker_labels[i]}"
 
 results.append({
 'speaker': speaker,
 'start': start,
 'end': end,
 'text': text
 })
 
 return results

# Usage
asr = MultiSpeakerASR()
results = asr.transcribe(audio)

# Output:
# [
# {'speaker': 'SPEAKER_0', 'start': 0.0, 'end': 1.5, 'text': 'Hello everyone'},
# {'speaker': 'SPEAKER_1', 'start': 1.2, 'end': 2.3, 'text': 'Hi there'},
# {'speaker': 'SPEAKER_0', 'start': 2.8, 'end': 4.0, 'text': 'How are you'},
# ]
``

---

## Handling Overlapping Speech

**The hardest problem**: Multiple speakers at once

### Challenge

``
Time: 0s 1s 2s
Speaker A: "Hello there..."
Speaker B: "Hi..."
Audio: [A] [A+B] [A]
 ↑
 Overlapped!
``

**Problem**: Single-channel audio can't separate perfectly

### Approach 1: Overlap Detection + Best Effort

``python
class OverlapHandler:
 """
 Detect overlapping speech and handle gracefully
 """
 
 def detect_overlaps(self, segments):
 """
 Find overlapping segments
 
 Args:
 segments: List of {speaker, start, end}
 
 Returns:
 List of overlap regions
 """
 overlaps = []
 
 for i, seg1 in enumerate(segments):
 for seg2 in segments[i+1:]:
 # Check if overlapping
 if seg1['end'] > seg2['start'] and seg1['start'] < seg2['end']:
 # Compute overlap region
 overlap_start = max(seg1['start'], seg2['start'])
 overlap_end = min(seg1['end'], seg2['end'])
 
 overlaps.append({
 'start': overlap_start,
 'end': overlap_end,
 'speakers': [seg1['speaker'], seg2['speaker']]
 })
 
 return overlaps
 
 def handle_overlap(self, audio, overlap, speakers):
 """
 Handle overlapped region
 
 Options:
 1. Transcribe mixed audio (less accurate)
 2. Mark as [OVERLAP] in transcript
 3. Use source separation (advanced)
 """
 # Option 1: Transcribe mixed audio
 segment = audio[int(overlap['start']*sr):int(overlap['end']*sr)]
 result = asr_model.transcribe(segment)
 
 return {
 'type': 'overlap',
 'speakers': overlap['speakers'],
 'start': overlap['start'],
 'end': overlap['end'],
 'text': result['text'],
 'confidence': 'low' # Mark as uncertain
 }
``

### Approach 2: Multi-Channel Source Separation

**If you have multiple microphones**, you can separate speakers!

``python
class MultiChannelSeparation:
 """
 Use multiple microphones to separate speakers
 
 Requires: Multiple audio channels (e.g., mic array)
 """
 
 def __init__(self):
 # Use beamforming or deep learning separation
 pass
 
 def separate(self, multi_channel_audio):
 """
 Separate speakers using spatial information
 
 Args:
 multi_channel_audio: (channels, samples)
 
 Returns:
 separated_sources: List of (speaker_audio, speaker_id)
 """
 # Advanced: Use Conv-TasNet or similar (from )
 # Here we'll use simple beamforming
 
 from scipy import signal
 
 # Beamforming toward each speaker
 # (Simplified - real implementation is complex)
 
 # For now, just return multi-channel as-is
 # In production, use libraries like:
 # - pyroomacoustics (beamforming)
 # - asteroid (deep learning separation)
 
 return multi_channel_audio
``

---

## Real-Time Streaming

**Challenge**: Process live audio with low latency

### Streaming Architecture

``
User's mic
 ↓
[Capture] → [Buffer] → [VAD] → [Diarization] → [ASR] → [Display]
 20ms 500ms 50ms 100ms 100ms 10ms
 ↑
 Total: ~270ms latency
``

``python
class StreamingMultiSpeakerASR:
 """
 Real-time multi-speaker ASR
 
 Processes audio chunks as they arrive
 """
 
 def __init__(self, chunk_duration=0.5):
 self.chunk_duration = chunk_duration
 self.buffer = []
 self.speaker_history = {} # Track speaker embeddings over time
 
 # Models
 import whisper
 self.vad = VoiceActivityDetector()
 self.embedding_extractor = SpeakerEmbeddingExtractor()
 self.asr_model = whisper.load_model("tiny") # Faster for real-time
 
 async def process_stream(self, audio_stream):
 """
 Process audio stream in real-time
 
 Args:
 audio_stream: Async iterator yielding audio chunks
 """
 async for chunk in audio_stream:
 # Add to buffer
 self.buffer.extend(chunk)
 
 # Process if buffer large enough
 if len(self.buffer) >= int(self.chunk_duration * 16000):
 result = await self._process_chunk()
 
 if result:
 yield result
 
 async def _process_chunk(self):
 """Process buffered audio chunk"""
 import numpy as np
 
 # Get chunk
 chunk = np.array(self.buffer[:int(self.chunk_duration * 16000)])
 
 # Remove from buffer (with overlap for continuity)
 overlap_samples = int(0.1 * 16000) # 100ms overlap
 self.buffer = self.buffer[len(chunk) - overlap_samples:]
 
 # VAD
 if not self._is_speech(chunk):
 return None
 
 # Extract embedding
 embedding = self.embedding_extractor.extract(chunk)
 
 # Identify speaker (match with history)
 speaker_id = self._identify_speaker(embedding)
 
 # Transcribe (async to not block)
 import asyncio
 text = await asyncio.to_thread(self.asr_model.transcribe, chunk, fp16=False)
 
 return {
 'speaker': speaker_id,
 'text': text['text'],
 'timestamp': __import__('time').time()
 }
 
 def _is_speech(self, chunk):
 """Quick speech check"""
 energy = np.mean(chunk ** 2)
 return energy > 0.01
 
 def _identify_speaker(self, embedding):
 """
 Match embedding to known speakers
 
 If new speaker, assign new ID
 """
 import numpy as np
 
 # Compare with known speakers
 best_match = None
 best_similarity = -1
 
 for speaker_id, known_embedding in self.speaker_history.items():
 # Cosine similarity
 similarity = np.dot(embedding, known_embedding) / (
 np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
 )
 
 if similarity > best_similarity:
 best_similarity = similarity
 best_match = speaker_id
 
 # Threshold for same speaker
 if best_similarity > 0.75:
 return best_match
 else:
 # New speaker
 new_id = f"SPEAKER_{len(self.speaker_history)}"
 self.speaker_history[new_id] = embedding
 return new_id

# Usage with WebSocket
import asyncio
import websockets

async def handle_client(websocket, path):
 """Handle incoming audio stream from client"""
 asr = StreamingMultiSpeakerASR()
 
 async for result in asr.process_stream(websocket):
 # Send transcription back to client
 await websocket.send(json.dumps(result))

# Start server
start_server = websockets.serve(handle_client, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
``

---

## Common Failure Modes & Debugging

### Failure Mode 1: Speaker Confusion

**Symptom**: System assigns same utterance to multiple speakers or switches mid-sentence

**Example:**
``
Ground truth:
 [Alice, 0-5s]: "Hello, how are you today?"

System output (WRONG):
 [Alice, 0-2s]: "Hello, how"
 [Bob, 2-5s]: "are you today?"
``

**Root causes:**

1. **Insufficient speech for embedding**
 - Embeddings need 2-3 seconds minimum
 - Short utterances (<1s) have unreliable embeddings
 
2. **Similar voices**
 - Two speakers with similar pitch/timbre
 - System can't distinguish
 
3. **Poor audio quality**
 - Background noise corrupts embeddings
 - Low SNR (<10dB) confuses system

**Solutions:**

``python
class RobustSpeakerIdentification:
 """
 Handle edge cases in speaker identification
 """
 
 def __init__(self, min_segment_duration=2.0):
 self.min_segment_duration = min_segment_duration
 self.speaker_history = [] # Track recent speakers
 
 def identify_with_context(self, audio_segment, duration, prev_speaker=None):
 """
 Identify speaker with contextual hints
 
 Args:
 audio_segment: Audio to identify
 duration: Segment duration in seconds
 prev_speaker: Who spoke last (context)
 
 Returns:
 speaker_id, confidence
 """
 # Check 1: Is segment long enough?
 if duration < self.min_segment_duration:
 # Too short for reliable embedding
 # Use speaker from previous segment (continuity assumption)
 if prev_speaker:
 return prev_speaker, 0.5 # Low confidence
 else:
 return "UNKNOWN", 0.0
 
 # Check 2: Extract embedding
 embedding = self.extract_embedding(audio_segment)
 
 # Check 3: Identify with threshold
 speaker, similarity = self.identify_speaker(embedding)
 
 # Check 4: Apply contextual prior
 if prev_speaker and similarity < 0.75:
 # Ambiguous - bias toward previous speaker (people usually finish sentences)
 return prev_speaker, 0.6
 
 return speaker, similarity
``

### Failure Mode 2: Overlap Mis-attribution

**Symptom**: During overlaps, words from Speaker A attributed to Speaker B

**Example:**
``
Ground truth:
 [Alice, 0-3s]: "I think we should consider this option"
 [Bob, 2-4s]: "Wait, what about the other approach?"

System output (WRONG):
 [Alice, 0-2s]: "I think we should"
 [Bob, 2-4s]: "consider this option Wait, what about the other approach?"
 ↑ These words are Alice's, not Bob's!
``

**Root cause**: Diarization boundaries don't align with actual speaker turns

**Solution**: Post-processing refinement

``python
class OverlapRefiner:
 """
 Refine transcriptions in overlap regions
 """
 
 def refine_overlaps(self, segments, asr_results):
 """
 Use ASR confidence to refine overlap boundaries
 
 Idea: Low-confidence words might be from the other speaker
 """
 refined = []
 
 for i, (seg, result) in enumerate(zip(segments, asr_results)):
 words = result['words'] # Word-level timestamps + confidence
 
 # Check if next segment overlaps
 if i < len(segments) - 1:
 next_seg = segments[i+1]
 
 if self._is_overlapping(seg, next_seg):
 # Refine boundary based on word confidence
 words, next_words = self._split_by_confidence(
 words, seg, next_seg
 )
 
 refined.append({
 'speaker': seg['speaker'],
 'start': seg['start'],
 'end': seg['end'],
 'words': words
 })
 
 return refined
 
 def _split_by_confidence(self, words, seg1, seg2):
 """
 Split words between two overlapping segments
 
 High-confidence words stay, low-confidence might belong to other speaker
 """
 overlap_start = max(seg1['start'], seg2['start'])
 overlap_end = min(seg1['end'], seg2['end'])
 
 seg1_words = []
 seg2_words = []
 
 for word in words:
 # Check if word is in overlap region
 if overlap_start <= word['start'] <= overlap_end:
 # In overlap - check confidence
 if word['confidence'] > 0.8:
 seg1_words.append(word) # Keep in current segment
 else:
 seg2_words.append(word) # Might belong to other speaker
 else:
 seg1_words.append(word)
 
 return seg1_words, seg2_words
``

### Failure Mode 3: Far-Field Audio Degradation

**Symptom**: Accuracy drops significantly when speaker is far from microphone

**Example metrics:**
``
Near-field (< 1m from mic):
 WER: 5%
 Diarization accuracy: 95%

Far-field (> 3m from mic):
 WER: 25% ← 5x worse!
 Diarization accuracy: 70%
``

**Root cause**: 
- Lower SNR (signal-to-noise ratio)
- More reverberation
- Acoustic reflections

**Solutions:**

1. **Beamforming** (if mic array available)
2. **Speech enhancement** pre-processing
3. **Specialized far-field models**

``python
class FarFieldPreprocessor:
 """
 Enhance far-field audio before ASR
 """
 
 def enhance(self, audio, sample_rate=16000):
 """
 Apply far-field enhancements
 
 1. Dereverb (reduce echo)
 2. Denoise
 3. Equalize (boost high frequencies)
 """
 # Step 1: Dereverberation (WPE algorithm)
 enhanced = self._dereverb_wpe(audio, sample_rate)
 
 # Step 2: Noise reduction (spectral subtraction)
 enhanced = self._denoise(enhanced, sample_rate)
 
 # Step 3: Equalization (boost consonants)
 enhanced = self._equalize(enhanced, sample_rate)
 
 return enhanced
 
 def _dereverb_wpe(self, audio, sr):
 """
 Weighted Prediction Error (WPE) dereverberation
 
 Removes room echo/reverberation
 """
 # Simplified - use library like `nara_wpe` in production
 from scipy import signal
 
 # High-pass filter to remove low-freq rumble
 sos = signal.butter(5, 100, 'highpass', fs=sr, output='sos')
 filtered = signal.sosfilt(sos, audio)
 
 return filtered
 
 def _denoise(self, audio, sr):
 """
 Spectral subtraction noise reduction
 """
 import noisereduce as nr
 
 # Estimate noise from first 0.5s (assuming silence/noise)
 reduced = nr.reduce_noise(
 y=audio,
 sr=sr,
 stationary=True,
 prop_decrease=0.8
 )
 
 return reduced
 
 def _equalize(self, audio, sr):
 """
 Boost high frequencies (consonants)
 
 Far-field audio loses high-freq content
 """
 from scipy import signal
 
 # Boost 2-8kHz (consonant region)
 sos = signal.butter(3, [2000, 8000], 'bandpass', fs=sr, output='sos')
 boosted = signal.sosfilt(sos, audio)
 
 # Mix with original (50-50)
 enhanced = 0.5 * audio + 0.5 * boosted
 
 return enhanced
``

### Debugging Tools

``python
class MultiSpeakerASRDebugger:
 """
 Tools for debugging multi-speaker ASR issues
 """
 
 def visualize_diarization(self, segments, audio_duration):
 """
 Visual timeline of speakers
 
 Helps spot issues like:
 - Too many speaker switches
 - Missing speakers
 - Wrong boundaries
 """
 import matplotlib.pyplot as plt
 import numpy as np
 
 fig, ax = plt.subplots(figsize=(15, 3))
 
 # Plot each segment
 for seg in segments:
 speaker_id = int(seg['speaker'].split('_')[1])
 color = plt.cm.tab10(speaker_id)
 
 ax.barh(
 y=speaker_id,
 width=seg['end'] - seg['start'],
 left=seg['start'],
 height=0.8,
 color=color,
 label=seg['speaker']
 )
 
 ax.set_xlabel('Time (seconds)')
 ax.set_ylabel('Speaker')
 ax.set_title('Speaker Diarization Timeline')
 ax.set_xlim(0, audio_duration)
 
 plt.tight_layout()
 plt.savefig('diarization_debug.png')
 print("✓ Saved visualization to diarization_debug.png")
 
 def compute_metrics(self, predicted_segments, ground_truth_segments):
 """
 Compute diarization metrics
 
 DER (Diarization Error Rate) = 
 (False Alarm + Miss + Speaker Confusion) / Total
 """
 from pyannote.metrics.diarization import DiarizationErrorRate
 
 der = DiarizationErrorRate()
 
 # Convert to pyannote format
 pred_annotation = self._to_annotation(predicted_segments)
 gt_annotation = self._to_annotation(ground_truth_segments)
 
 # Compute DER
 error_rate = der(gt_annotation, pred_annotation)
 
 # Detailed breakdown
 details = der.components(gt_annotation, pred_annotation)
 
 return {
 'DER': error_rate,
 'false_alarm': details['false alarm'],
 'missed_detection': details['missed detection'],
 'speaker_confusion': details['confusion']
 }
 
 def _to_annotation(self, segments):
 """Convert segments to pyannote Annotation format"""
 from pyannote.core import Annotation, Segment
 
 annotation = Annotation()
 
 for seg in segments:
 annotation[Segment(seg['start'], seg['end'])] = seg['speaker']
 
 return annotation
``

---

## Production Considerations

### 1. Latency Optimization

**Target**: < 300ms end-to-end for real-time feel

**Breakdown:**
``
Audio capture: 20ms
Buffering: 100ms
VAD: 10ms
Embedding: 50ms
ASR: 100ms
Network: 20ms
Total: 300ms
``

**Optimizations:**
- Use smaller ASR models (Whisper tiny/base)
- Batch embedding extraction
- Pre-compute speaker profiles
- GPU acceleration
- Reduce network round-trips

### 2. Accuracy vs Speed Trade-off

| Model Size | Latency | WER | Use Case |
|------------|---------|-----|----------|
| Whisper tiny | 50ms | 10% | Live captions |
| Whisper base | 100ms | 7% | Meetings |
| Whisper medium | 300ms | 5% | Post-processing |
| Whisper large | 1000ms | 3% | Archival transcription |

### 3. Speaker Persistence

**Challenge**: Same speaker should have consistent ID across session

``python
class SpeakerRegistry:
 """
 Maintain consistent speaker IDs
 
 Matches new embeddings to registered speakers
 """
 
 def __init__(self, similarity_threshold=0.75):
 self.speakers = {} # id -> mean embedding
 self.threshold = similarity_threshold
 
 def register_or_identify(self, embedding):
 """
 Register new speaker or identify existing
 """
 # Check against known speakers
 for speaker_id, known_emb in self.speakers.items():
 similarity = cosine_similarity(embedding, known_emb)
 
 if similarity > self.threshold:
 # Update running average
 self.speakers[speaker_id] = (
 0.9 * known_emb + 0.1 * embedding
 )
 return speaker_id
 
 # New speaker
 new_id = f"SPEAKER_{len(self.speakers) + 1}"
 self.speakers[new_id] = embedding
 return new_id
``

### 4. Monitoring & Debugging

``python
class MultiSpeakerASRMetrics:
 """
 Track system performance
 """
 
 def __init__(self):
 self.metrics = {
 'latency_ms': [],
 'overlap_ratio': 0,
 'speaker_switches_per_minute': 0,
 'wer_per_speaker': {}
 }
 
 def log_latency(self, latency_ms):
 self.metrics['latency_ms'].append(latency_ms)
 
 def report(self):
 import numpy as np
 
 return {
 'p50_latency_ms': np.median(self.metrics['latency_ms']),
 'p95_latency_ms': np.percentile(self.metrics['latency_ms'], 95),
 'overlap_ratio': self.metrics['overlap_ratio'],
 'speaker_switches_per_minute': self.metrics['speaker_switches_per_minute']
 }
``

---

## Key Takeaways

✅ **Multi-speaker ASR** = VAD + Diarization + ASR 
✅ **Speaker embeddings** capture voice identity 
✅ **Clustering** groups segments by speaker 
✅ **Overlaps** are hard - detect and handle gracefully 
✅ **Real-time** requires careful latency optimization 
✅ **State-of-the-art**: Use `pyannote.audio` + Whisper 

**Production tips:**
- Start with `pyannote` + `whisper` (best quality)
- Optimize latency with smaller models if needed
- Handle overlaps explicitly (mark in transcript)
- Maintain speaker consistency across session
- Monitor latency and accuracy per speaker

---

**Originally published at:** [arunbaby.com/speech-tech/0012-multi-speaker-asr](https://www.arunbaby.com/speech-tech/0012-multi-speaker-asr/)

*If you found this helpful, consider sharing it with others who might benefit.*

