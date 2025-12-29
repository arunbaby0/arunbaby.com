---
title: "Speaker Recognition & Verification"
day: 5
related_dsa_day: 5
related_ml_day: 5
related_agents_day: 5
collection: speech_tech
categories:
 - speech-tech
tags:
 - speaker-recognition
 - speaker-verification
 - biometrics
 - embeddings
subdomain: Speaker Technology
tech_stack: [Python, PyTorch, Kaldi, SpeechBrain, ONNX]
latency_requirement: "< 100ms"
scale: "Millions of speakers"
companies: [Google, Amazon, Microsoft, Apple]
---

**How voice assistants recognize who's speaking, the biometric authentication powering "Hey Alexa" and personalized experiences.**

## Introduction

**Speaker Recognition** is the task of identifying or verifying a person based on their voice.

Two main tasks:
1. **Speaker Identification:** Who is speaking? (1:N matching)
2. **Speaker Verification:** Is this person who they claim to be? (1:1 matching)

**Why it matters:**
- **Personalization:** Voice assistants adapt to users
- **Security:** Voice biometric authentication
- **Call centers:** Route calls to correct agent
- **Forensics:** Identify speakers in recordings

**What you'll learn:**
- Speaker embeddings (d-vectors, x-vectors)
- Verification vs identification
- Production deployment patterns
- Anti-spoofing techniques
- Real-world applications

---

## Problem Definition

Design a speaker recognition system.

### Functional Requirements

1. **Enrollment**
 - Capture user's voice samples
 - Extract speaker embedding
 - Store in database

2. **Verification**
 - Given audio + claimed identity
 - Verify if speaker matches

3. **Identification**
 - Given audio only
 - Identify speaker from database

### Non-Functional Requirements

1. **Accuracy**
 - False Acceptance Rate (FAR) < 1%
 - False Rejection Rate (FRR) < 5%
 - Equal Error Rate (EER) < 2%

2. **Latency**
 - Enrollment: < 500ms
 - Verification: < 100ms

3. **Scalability**
 - Support millions of enrolled speakers
 - Fast lookup in embedding space

---

## Speaker Embeddings

Core idea: Map variable-length audio → fixed-size vector that captures speaker identity.

### X-Vectors

State-of-the-art speaker embeddings using time-delay neural networks (TDNN).

``python
import torch
import torch.nn as nn

class XVectorExtractor(nn.Module):
 """
 X-vector architecture for speaker embeddings
 
 Input: Variable-length audio features (mel-spectrogram)
 Output: Fixed 512-dim speaker embedding
 """
 
 def __init__(self, input_dim=40, embedding_dim=512):
 super().__init__()
 
 # Frame-level layers (TDNN)
 self.tdnn1 = nn.Conv1d(input_dim, 512, kernel_size=5, dilation=1)
 self.tdnn2 = nn.Conv1d(512, 512, kernel_size=3, dilation=2)
 self.tdnn3 = nn.Conv1d(512, 512, kernel_size=3, dilation=3)
 self.tdnn4 = nn.Conv1d(512, 512, kernel_size=1, dilation=1)
 self.tdnn5 = nn.Conv1d(512, 1500, kernel_size=1, dilation=1)
 
 # Statistical pooling
 # Computes mean + std over time → fixed size
 
 # Segment-level layers
 self.fc1 = nn.Linear(3000, 512) # 1500 mean + 1500 std
 self.fc2 = nn.Linear(512, embedding_dim)
 
 self.relu = nn.ReLU()
 self.bn = nn.BatchNorm1d(512)
 
 def forward(self, x):
 """
 Args:
 x: (batch, time, features) e.g., (B, T, 40)
 
 Returns:
 embeddings: (batch, embedding_dim)
 """
 # Transpose for Conv1d: (batch, features, time)
 x = x.transpose(1, 2)
 
 # Frame-level processing
 x = self.relu(self.tdnn1(x))
 x = self.relu(self.tdnn2(x))
 x = self.relu(self.tdnn3(x))
 x = self.relu(self.tdnn4(x))
 x = self.relu(self.tdnn5(x))
 
 # Statistical pooling: mean + std over time
 mean = torch.mean(x, dim=2)
 std = torch.std(x, dim=2)
 stats = torch.cat([mean, std], dim=1) # (batch, 3000)
 
 # Segment-level processing
 x = self.relu(self.fc1(stats))
 x = self.bn(x)
 embeddings = self.fc2(x) # (batch, embedding_dim)
 
 # L2 normalize
 embeddings = embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True)
 
 return embeddings

# Usage
model = XVectorExtractor(input_dim=40, embedding_dim=512)
model.eval()

# Extract embedding
mel_spec = torch.randn(1, 300, 40) # 3 seconds of audio
embedding = model(mel_spec) # (1, 512)

print(f"Embedding shape: {embedding.shape}")
print(f"Embedding norm: {torch.norm(embedding):.4f}") # Should be ~1.0
``

### Training Speaker Embeddings

``python
class SpeakerEmbeddingTrainer:
 """
 Train x-vector model using cross-entropy over speaker IDs
 """
 
 def __init__(self, model, num_speakers, device='cuda'):
 self.model = model.to(device)
 self.device = device
 
 # Classification head for training
 self.classifier = nn.Linear(512, num_speakers).to(device)
 
 # Loss
 self.criterion = nn.CrossEntropyLoss()
 
 # Optimizer
 self.optimizer = torch.optim.Adam(
 list(self.model.parameters()) + list(self.classifier.parameters()),
 lr=0.001
 )
 
 def train_step(self, audio_features, speaker_labels):
 """
 Single training step
 
 Args:
 audio_features: (batch, time, features)
 speaker_labels: (batch,) integer speaker IDs
 
 Returns:
 Loss value
 """
 self.model.train()
 self.optimizer.zero_grad()
 
 # Extract embeddings
 embeddings = self.model(audio_features)
 
 # Classify
 logits = self.classifier(embeddings)
 
 # Loss
 loss = self.criterion(logits, speaker_labels)
 
 # Backward
 loss.backward()
 self.optimizer.step()
 
 return loss.item()
 
 def extract_embedding(self, audio_features):
 """Extract embedding for inference (no classification head)"""
 self.model.eval()
 
 with torch.no_grad():
 embedding = self.model(audio_features)
 
 return embedding

# Training loop
trainer = SpeakerEmbeddingTrainer(
 model=XVectorExtractor(),
 num_speakers=10000 # Number of speakers in training set
)

for epoch in range(100):
 for batch in train_loader:
 audio, speaker_ids = batch
 
 loss = trainer.train_step(audio.to(trainer.device), speaker_ids.to(trainer.device))
 
 print(f"Epoch {epoch}, Loss: {loss:.4f}")
``

---

## Speaker Verification

Verify if two audio samples are from the same speaker.

### Cosine Similarity

``python
import numpy as np
import torch

class SpeakerVerifier:
 """
 Speaker verification system
 
 Uses cosine similarity between embeddings
 """
 
 def __init__(self, embedding_extractor, threshold=0.5):
 self.extractor = embedding_extractor
 self.threshold = threshold
 
 def extract_embedding(self, audio):
 """Extract embedding from audio"""
 # Preprocess audio → mel-spectrogram
 features = self._audio_to_features(audio)
 
 # Extract embedding (support trainer-style or raw nn.Module)
 with torch.no_grad():
 if hasattr(self.extractor, 'extract_embedding'):
 emb_tensor = self.extractor.extract_embedding(features)
 else:
 emb_tensor = self.extractor(features)
 
 return emb_tensor.cpu().numpy().flatten()
 
 def _audio_to_features(self, audio):
 """Convert audio to mel-spectrogram"""
 import librosa
 
 # Compute mel-spectrogram
 mel_spec = librosa.feature.melspectrogram(
 y=audio,
 sr=16000,
 n_mels=40,
 n_fft=512,
 hop_length=160
 )
 
 # Log scale
 mel_spec = librosa.power_to_db(mel_spec)
 
 # Transpose: (time, features)
 mel_spec = mel_spec.T
 
 # Convert to tensor
 features = torch.from_numpy(mel_spec).float().unsqueeze(0)
 
 return features
 
 def cosine_similarity(self, emb1, emb2):
 """
 Compute cosine similarity
 
 Returns:
 Similarity score in [-1, 1]
 """
 return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
 
 def verify(self, audio1, audio2):
 """
 Verify if two audio samples are from same speaker
 
 Args:
 audio1, audio2: Audio waveforms
 
 Returns:
 {
 'is_same_speaker': bool,
 'similarity': float,
 'threshold': float
 }
 """
 # Extract embeddings
 emb1 = self.extract_embedding(audio1)
 emb2 = self.extract_embedding(audio2)
 
 # Compute similarity
 similarity = self.cosine_similarity(emb1, emb2)
 
 # Decision
 is_same = similarity >= self.threshold
 
 return {
 'is_same_speaker': bool(is_same),
 'similarity': float(similarity),
 'threshold': self.threshold
 }

# Usage
verifier = SpeakerVerifier(embedding_extractor=trainer, threshold=0.6)

# Load audio samples
audio1, sr1 = librosa.load('speaker1_sample1.wav', sr=16000)
audio2, sr2 = librosa.load('speaker1_sample2.wav', sr=16000)

result = verifier.verify(audio1, audio2)

print(f"Same speaker: {result['is_same_speaker']}")
print(f"Similarity: {result['similarity']:.4f}")
``

### Threshold Selection

``python
class ThresholdOptimizer:
 """
 Find optimal verification threshold
 
 Balances False Acceptance Rate (FAR) and False Rejection Rate (FRR)
 """
 
 def __init__(self):
 pass
 
 def compute_eer(self, genuine_scores, impostor_scores):
 """
 Compute Equal Error Rate (EER)
 
 Args:
 genuine_scores: Similarity scores for same-speaker pairs
 impostor_scores: Similarity scores for different-speaker pairs
 
 Returns:
 {
 'eer': float,
 'threshold': float
 }
 """
 # Try different thresholds
 # Restrict to plausible cosine similarity range [-1, 1]
 thresholds = np.linspace(-1.0, 1.0, 1000)
 
 fars = []
 frrs = []
 
 for threshold in thresholds:
 # False Acceptance: impostor accepted as genuine
 far = np.mean(impostor_scores >= threshold)
 
 # False Rejection: genuine rejected as impostor
 frr = np.mean(genuine_scores < threshold)
 
 fars.append(far)
 frrs.append(frr)
 
 fars = np.array(fars)
 frrs = np.array(frrs)
 
 # Find EER: point where FAR == FRR
 diff = np.abs(fars - frrs)
 eer_idx = np.argmin(diff)
 
 eer = (fars[eer_idx] + frrs[eer_idx]) / 2
 eer_threshold = thresholds[eer_idx]
 
 return {
 'eer': eer,
 'threshold': eer_threshold,
 'far_at_eer': fars[eer_idx],
 'frr_at_eer': frrs[eer_idx]
 }

# Usage
optimizer = ThresholdOptimizer()

# Collect scores from validation set
genuine_scores = [] # Same-speaker pairs
impostor_scores = [] # Different-speaker pairs

# ... collect scores ...

result = optimizer.compute_eer(
 np.array(genuine_scores),
 np.array(impostor_scores)
)

print(f"EER: {result['eer']:.2%}")
print(f"Optimal threshold: {result['threshold']:.4f}")
``

---

## Speaker Identification

Identify which speaker from a database is speaking.

### Database of Speakers

``python
import faiss

class SpeakerDatabase:
 """
 Store and search speaker embeddings
 
 Uses FAISS for efficient similarity search
 """
 
 def __init__(self, embedding_dim=512):
 self.embedding_dim = embedding_dim
 
 # FAISS index for fast similarity search
 self.index = faiss.IndexFlatIP(embedding_dim) # Inner product (cosine similarity)
 
 # Metadata: speaker IDs
 self.speaker_ids = []
 
 def enroll_speaker(self, speaker_id: str, embedding: np.ndarray):
 """
 Enroll a new speaker
 
 Args:
 speaker_id: Unique speaker identifier
 embedding: Speaker embedding (512-dim)
 """
 # Normalize embedding
 embedding = embedding / np.linalg.norm(embedding)
 embedding = embedding.reshape(1, -1).astype('float32')
 
 # Add to index
 self.index.add(embedding)
 
 # Store metadata
 self.speaker_ids.append(speaker_id)
 
 def identify_speaker(self, query_embedding: np.ndarray, top_k=5):
 """
 Identify speaker from database
 
 Args:
 query_embedding: Embedding to search for
 top_k: Return top-k most similar speakers
 
 Returns:
 List of (speaker_id, similarity_score)
 """
 # Normalize query
 query = query_embedding / np.linalg.norm(query_embedding)
 query = query.reshape(1, -1).astype('float32')
 
 # Search
 similarities, indices = self.index.search(query, top_k)
 
 # Format results
 results = []
 for similarity, idx in zip(similarities[0], indices[0]):
 if idx < len(self.speaker_ids):
 results.append({
 'speaker_id': self.speaker_ids[idx],
 'similarity': float(similarity),
 'rank': len(results) + 1
 })
 
 return results
 
 def get_num_speakers(self):
 """Get number of enrolled speakers"""
 return len(self.speaker_ids)

 def save(self, index_path: str, meta_path: str):
 """Persist FAISS index and metadata"""
 faiss.write_index(self.index, index_path)
 import json
 with open(meta_path, 'w') as f:
 json.dump({'speaker_ids': self.speaker_ids}, f)

 def load(self, index_path: str, meta_path: str):
 """Load FAISS index and metadata"""
 self.index = faiss.read_index(index_path)
 import json
 with open(meta_path, 'r') as f:
 meta = json.load(f)
 self.speaker_ids = meta.get('speaker_ids', [])

 def get_embedding(self, speaker_id: str) -> np.ndarray:
 """
 Retrieve enrolled embedding by speaker_id.
 Note: IndexFlatIP does not store vectors retrievably; in production
 store embeddings separately. This function assumes you maintain a
 parallel mapping. Placeholder returns None.
 """
 return None

# Usage
database = SpeakerDatabase(embedding_dim=512)

# Enroll speakers
for speaker_id in ['alice', 'bob', 'charlie']:
 # Extract embedding from enrollment audio
 audio, _ = librosa.load(f'{speaker_id}_enroll.wav', sr=16000)
 embedding = verifier.extract_embedding(audio)
 
 database.enroll_speaker(speaker_id, embedding)

print(f"Enrolled {database.get_num_speakers()} speakers")

# Identify speaker from test audio
test_audio, _ = librosa.load('unknown_speaker.wav', sr=16000)
test_embedding = verifier.extract_embedding(test_audio)

results = database.identify_speaker(test_embedding, top_k=3)

print("Top matches:")
for result in results:
 print(f" {result['rank']}. {result['speaker_id']}: {result['similarity']:.4f}")
``

---

## Production Deployment

### Real-Time Verification API

``python
from fastapi import FastAPI, File, UploadFile
import io

app = FastAPI()

class SpeakerRecognitionService:
 """
 Production speaker recognition service
 """
 
 def __init__(self):
 # Load model
 self.embedding_extractor = load_pretrained_model()
 
 # Load speaker database
 self.database = SpeakerDatabase()
 # Load FAISS index and metadata files
 self.database.load('speaker_database.index', 'speaker_database.meta.json')
 
 # Verifier
 self.verifier = SpeakerVerifier(
 self.embedding_extractor,
 threshold=0.65
 )
 
 def process_audio_bytes(self, audio_bytes: bytes) -> np.ndarray:
 """Convert uploaded audio to waveform"""
 import soundfile as sf
 
 audio, sr = sf.read(io.BytesIO(audio_bytes))
 
 # Resample if needed
 if sr != 16000:
 import librosa
 audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
 
 return audio

service = SpeakerRecognitionService()

@app.post("/enroll")
async def enroll_speaker(
 speaker_id: str,
 audio: UploadFile = File(...)
):
 """
 Enroll new speaker
 
 POST /enroll?speaker_id=alice
 Body: audio file
 """
 # Read audio
 audio_bytes = await audio.read()
 audio_waveform = service.process_audio_bytes(audio_bytes)
 
 # Extract embedding
 embedding = service.verifier.extract_embedding(audio_waveform)
 
 # Enroll
 service.database.enroll_speaker(speaker_id, embedding)
 
 return {
 'status': 'success',
 'speaker_id': speaker_id,
 'total_speakers': service.database.get_num_speakers()
 }

@app.post("/verify")
async def verify_speaker(
 claimed_speaker_id: str,
 audio: UploadFile = File(...)
):
 """
 Verify claimed identity
 
 POST /verify?claimed_speaker_id=alice
 Body: audio file
 """
 # Process audio
 audio_bytes = await audio.read()
 audio_waveform = service.process_audio_bytes(audio_bytes)
 
 # Extract embedding
 query_embedding = service.verifier.extract_embedding(audio_waveform)
 
 # Get enrolled embedding (lookup from database; implement external store in production)
 enrolled_embedding = service.database.get_embedding(claimed_speaker_id)
 if enrolled_embedding is None:
 return {
 'error': 'enrolled embedding not found',
 'claimed_speaker_id': claimed_speaker_id
 }, 404
 
 # Verify
 similarity = service.verifier.cosine_similarity(query_embedding, enrolled_embedding)
 is_verified = similarity >= service.verifier.threshold
 
 return {
 'verified': bool(is_verified),
 'similarity': float(similarity),
 'threshold': service.verifier.threshold,
 'claimed_speaker_id': claimed_speaker_id
 }

@app.post("/identify")
async def identify_speaker(audio: UploadFile = File(...)):
 """
 Identify unknown speaker
 
 POST /identify
 Body: audio file
 """
 # Process audio
 audio_bytes = await audio.read()
 audio_waveform = service.process_audio_bytes(audio_bytes)
 
 # Extract embedding
 embedding = service.verifier.extract_embedding(audio_waveform)
 
 # Identify
 matches = service.database.identify_speaker(embedding, top_k=5)
 
 return {
 'matches': matches
 }
``

---

## Anti-Spoofing

Detect replay attacks and synthetic voices.

``python
class AntiSpoofingDetector:
 """
 Detect spoofing attacks
 
 - Replay attacks (recorded audio)
 - Synthetic voices (TTS, deepfakes)
 """
 
 def __init__(self, model):
 self.model = model
 
 def detect_spoofing(self, audio):
 """
 Detect if audio is spoofed
 
 Returns:
 {
 'is_genuine': bool,
 'confidence': float
 }
 """
 # Extract anti-spoofing features
 # E.g., phase information, low-level acoustic features
 features = self._extract_antispoofing_features(audio)
 
 # Classify
 # is_genuine_prob = self.model.predict(features)
 is_genuine_prob = 0.92 # Placeholder
 
 return {
 'is_genuine': is_genuine_prob > 0.5,
 'confidence': float(is_genuine_prob)
 }
 
 def _extract_antispoofing_features(self, audio):
 """
 Extract features for spoofing detection
 
 - CQCC (Constant Q Cepstral Coefficients)
 - LFCC (Linear Frequency Cepstral Coefficients)
 - Phase information
 """
 # Placeholder
 return None
``

---

## Real-World Applications

### Voice Assistant Personalization

``python
class VoiceAssistantPersonalization:
 """
 Personalize responses based on recognized speaker
 """
 
 def __init__(self, speaker_recognizer):
 self.recognizer = speaker_recognizer
 
 # User preferences
 self.user_preferences = {
 'alice': {'music_genre': 'jazz', 'news_source': 'npr'},
 'bob': {'music_genre': 'rock', 'news_source': 'bbc'},
 }
 
 def process_voice_command(self, audio, command):
 """
 Recognize speaker and personalize response
 """
 # Identify speaker
 embedding = self.recognizer.extract_embedding(audio)
 matches = self.recognizer.database.identify_speaker(embedding, top_k=1)
 
 if matches and matches[0]['similarity'] > 0.7:
 speaker_id = matches[0]['speaker_id']
 
 # Get preferences
 prefs = self.user_preferences.get(speaker_id, {})
 
 # Personalize response based on command
 if 'play music' in command:
 genre = prefs.get('music_genre', 'pop')
 return f"Playing {genre} music for {speaker_id}"
 
 elif 'news' in command:
 source = prefs.get('news_source', 'default')
 return f"Here's news from {source} for {speaker_id}"
 
 return "Generic response for unknown user"
``

---

## Advanced Topics

### Speaker Diarization

Segment audio by speaker ("who spoke when").

``python
class SpeakerDiarizer:
 """
 Speaker diarization: Segment audio by speaker
 
 Process:
 1. VAD: Detect speech segments
 2. Extract embeddings for each segment
 3. Cluster embeddings → speakers
 4. Assign segments to speakers
 """
 
 def __init__(self, embedding_extractor):
 self.extractor = embedding_extractor
 
 def diarize(self, audio, sr=16000, window_sec=2.0):
 """
 Perform speaker diarization
 
 Args:
 audio: Audio waveform
 sr: Sample rate
 window_sec: Window size for embedding extraction
 
 Returns:
 List of (start_time, end_time, speaker_id)
 """
 # Step 1: Segment audio into windows
 window_samples = int(window_sec * sr)
 segments = []
 
 for start in range(0, len(audio) - window_samples, window_samples // 2):
 end = start + window_samples
 segment_audio = audio[start:end]
 
 # Extract embedding
 embedding = self.extractor.extract_embedding(segment_audio)
 
 segments.append({
 'start_time': start / sr,
 'end_time': end / sr,
 'embedding': embedding
 })
 
 # Step 2: Cluster embeddings
 embeddings_matrix = np.array([s['embedding'] for s in segments])
 speaker_labels = self._cluster_embeddings(embeddings_matrix)
 
 # Step 3: Assign labels to segments
 for segment, label in zip(segments, speaker_labels):
 segment['speaker_id'] = f'speaker_{label}'
 
 # Step 4: Merge consecutive segments from same speaker
 merged = self._merge_segments(segments)
 
 return merged
 
 def _cluster_embeddings(self, embeddings, num_speakers=None):
 """
 Cluster embeddings using spectral clustering
 
 Args:
 embeddings: (N, embedding_dim) matrix
 num_speakers: Number of speakers (auto-detect if None)
 
 Returns:
 Speaker labels for each segment
 """
 from sklearn.cluster import SpectralClustering
 
 if num_speakers is None:
 # Auto-detect number of speakers (simplified)
 num_speakers = self._estimate_num_speakers(embeddings)
 
 # Cluster
 clustering = SpectralClustering(
 n_clusters=num_speakers,
 affinity='cosine'
 )
 
 labels = clustering.fit_predict(embeddings)
 
 return labels
 
 def _estimate_num_speakers(self, embeddings):
 """Estimate number of speakers (simplified heuristic)"""
 # Use silhouette score to find optimal clusters
 from sklearn.metrics import silhouette_score
 
 best_score = -1
 best_k = 2
 
 for k in range(2, min(10, len(embeddings) // 5)):
 try:
 from sklearn.cluster import KMeans
 kmeans = KMeans(n_clusters=k, random_state=42)
 labels = kmeans.fit_predict(embeddings)
 score = silhouette_score(embeddings, labels)
 
 if score > best_score:
 best_score = score
 best_k = k
 except:
 break
 
 return best_k
 
 def _merge_segments(self, segments):
 """Merge consecutive segments from same speaker"""
 if not segments:
 return []
 
 merged = []
 current = {
 'start_time': segments[0]['start_time'],
 'end_time': segments[0]['end_time'],
 'speaker_id': segments[0]['speaker_id']
 }
 
 for segment in segments[1:]:
 if segment['speaker_id'] == current['speaker_id']:
 # Same speaker, extend segment
 current['end_time'] = segment['end_time']
 else:
 # Different speaker, save current and start new
 merged.append(current)
 current = {
 'start_time': segment['start_time'],
 'end_time': segment['end_time'],
 'speaker_id': segment['speaker_id']
 }
 
 # Add last segment
 merged.append(current)
 
 return merged

# Usage
diarizer = SpeakerDiarizer(embedding_extractor=trainer)

audio, sr = librosa.load('meeting_audio.wav', sr=16000)
diarization = diarizer.diarize(audio, sr=sr, window_sec=2.0)

print("Speaker diarization results:")
for segment in diarization:
 print(f" {segment['start_time']:.1f}s - {segment['end_time']:.1f}s: {segment['speaker_id']}")
``

### Domain Adaptation

Adapt speaker recognition to new domains/conditions.

``python
class DomainAdaptation:
 """
 Adapt speaker embeddings across domains
 
 Use case: Train on clean speech, deploy on noisy environment
 """
 
 def __init__(self, base_model):
 self.base_model = base_model
 
 def extract_domain_adapted_embedding(
 self,
 audio,
 target_domain='noisy'
 ):
 """
 Extract embedding with domain adaptation
 
 Techniques:
 1. Multi-condition training
 2. Domain adversarial training
 3. Feature normalization
 """
 # Extract base embedding
 features = self._audio_to_features(audio)
 base_embedding = self.base_model(features)
 
 # Apply domain-specific adaptation
 if target_domain == 'noisy':
 # Normalize to reduce noise impact
 adapted = self._normalize_embedding(base_embedding)
 elif target_domain == 'telephone':
 # Adapt for telephony bandwidth
 adapted = self._bandwidth_adaptation(base_embedding)
 else:
 adapted = base_embedding
 
 return adapted
 
 def _normalize_embedding(self, embedding):
 """Length normalization"""
 norm = torch.norm(embedding, p=2, dim=-1, keepdim=True)
 return embedding / norm
 
 def _bandwidth_adaptation(self, embedding):
 """Adapt for limited bandwidth"""
 # Apply transformation learned for telephony
 # In production: learned linear transformation
 return embedding
``

### Multi-Modal Biometrics

Combine speaker recognition with face recognition.

``python
class MultiModalBiometrics:
 """
 Fuse speaker + face recognition for stronger authentication
 
 Fusion strategies:
 1. Score-level fusion
 2. Feature-level fusion
 3. Decision-level fusion
 """
 
 def __init__(self, speaker_verifier, face_verifier):
 self.speaker = speaker_verifier
 self.face = face_verifier
 
 def verify_multimodal(
 self,
 audio,
 face_image,
 claimed_identity: str,
 fusion_method='score'
 ) -> dict:
 """
 Verify using both voice and face
 
 Args:
 audio: Audio sample
 face_image: Face image
 claimed_identity: Claimed identity
 fusion_method: 'score', 'feature', or 'decision'
 
 Returns:
 Verification result
 """
 # Get individual scores
 speaker_result = self.speaker.verify(audio, claimed_identity)
 face_result = self.face.verify(face_image, claimed_identity)
 
 if fusion_method == 'score':
 # Score-level fusion: weighted combination
 combined_score = (
 0.6 * speaker_result['similarity'] +
 0.4 * face_result['similarity']
 )
 
 is_verified = combined_score > 0.7
 
 return {
 'verified': is_verified,
 'combined_score': combined_score,
 'speaker_score': speaker_result['similarity'],
 'face_score': face_result['similarity'],
 'method': 'score_fusion'
 }
 
 elif fusion_method == 'decision':
 # Decision-level fusion: both must pass
 is_verified = (
 speaker_result['is_same_speaker'] and
 face_result['is_same_person']
 )
 
 return {
 'verified': is_verified,
 'speaker_verified': speaker_result['is_same_speaker'],
 'face_verified': face_result['is_same_person'],
 'method': 'decision_fusion'
 }
``

---

## Optimization for Production

### Model Compression

Reduce model size for edge deployment.

``python
class CompressedXVector:
 """
 Compressed x-vector for mobile/edge devices
 
 Techniques:
 1. Quantization (INT8)
 2. Pruning
 3. Knowledge distillation
 """
 
 def __init__(self, base_model):
 self.base_model = base_model
 self.compressed_model = None
 
 def quantize_model(self):
 """
 Quantize model to INT8
 
 Reduces size by 4x with minimal accuracy loss
 """
 import torch.quantization
 
 # Prepare for quantization
 self.base_model.eval()
 self.base_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
 
 # Fuse layers (Conv+BN+ReLU)
 torch.quantization.fuse_modules(
 self.base_model,
 [['conv1', 'bn1', 'relu1']],
 inplace=True
 )
 
 # Prepare
 torch.quantization.prepare(self.base_model, inplace=True)
 
 # Calibrate with sample data
 # In production: use representative dataset
 sample_input = torch.randn(10, 300, 40)
 with torch.no_grad():
 self.base_model(sample_input)
 
 # Convert to quantized model
 self.compressed_model = torch.quantization.convert(self.base_model, inplace=False)
 
 return self.compressed_model
 
 def export_to_onnx(self, output_path='speaker_model.onnx'):
 """
 Export to ONNX for cross-platform deployment
 """
 dummy_input = torch.randn(1, 300, 40)
 
 torch.onnx.export(
 self.compressed_model or self.base_model,
 dummy_input,
 output_path,
 input_names=['mel_spectrogram'],
 output_names=['embedding'],
 dynamic_axes={
 'mel_spectrogram': {1: 'time'}, # Variable length
 }
 )
 
 print(f"Model exported to {output_path}")
``

### Streaming Enrollment

Enroll speakers incrementally from streaming audio.

``python
class StreamingEnrollment:
 """
 Incrementally build speaker profile from multiple utterances
 
 Use case: "Say 'Hey Siri' five times to enroll"
 """
 
 def __init__(self, embedding_extractor, required_utterances=5):
 self.extractor = embedding_extractor
 self.required_utterances = required_utterances
 self.enrollment_sessions = {}
 
 def start_enrollment(self, speaker_id: str):
 """Start new enrollment session"""
 self.enrollment_sessions[speaker_id] = {
 'embeddings': [],
 'started_at': time.time()
 }
 
 def add_utterance(self, speaker_id: str, audio):
 """
 Add enrollment utterance
 
 Returns:
 {
 'progress': int, # Number of utterances collected
 'required': int,
 'complete': bool
 }
 """
 if speaker_id not in self.enrollment_sessions:
 raise ValueError(f"No enrollment session for {speaker_id}")
 
 # Extract embedding
 embedding = self.extractor.extract_embedding(audio)
 
 # Add to session
 session = self.enrollment_sessions[speaker_id]
 session['embeddings'].append(embedding)
 
 progress = len(session['embeddings'])
 complete = progress >= self.required_utterances
 
 return {
 'progress': progress,
 'required': self.required_utterances,
 'complete': complete,
 'speaker_id': speaker_id
 }
 
 def finalize_enrollment(self, speaker_id: str) -> np.ndarray:
 """
 Compute final speaker embedding
 
 Strategy: Average embeddings from all utterances
 """
 session = self.enrollment_sessions[speaker_id]
 
 if len(session['embeddings']) < self.required_utterances:
 raise ValueError(f"Insufficient utterances: {len(session['embeddings'])}/{self.required_utterances}")
 
 # Average embeddings
 embeddings_matrix = np.array(session['embeddings'])
 final_embedding = np.mean(embeddings_matrix, axis=0)
 
 # Normalize
 final_embedding = final_embedding / np.linalg.norm(final_embedding)
 
 # Clean up session
 del self.enrollment_sessions[speaker_id]
 
 return final_embedding

# Usage
enrollment = StreamingEnrollment(embedding_extractor=trainer, required_utterances=5)

# Start enrollment
enrollment.start_enrollment('alice')

# Collect utterances
for i in range(5):
 audio, _ = librosa.load(f'alice_utterance_{i}.wav', sr=16000)
 result = enrollment.add_utterance('alice', audio)
 print(f"Progress: {result['progress']}/{result['required']}")

# Finalize
if result['complete']:
 final_embedding = enrollment.finalize_enrollment('alice')
 print(f"Enrollment complete! Embedding shape: {final_embedding.shape}")
``

---

## Evaluation Metrics

### Performance Metrics

``python
class SpeakerRecognitionEvaluator:
 """
 Comprehensive evaluation for speaker recognition
 """
 
 def __init__(self):
 pass
 
 def compute_eer_and_det(
 self,
 genuine_scores: np.ndarray,
 impostor_scores: np.ndarray
 ) -> dict:
 """
 Compute EER and DET curve
 
 Args:
 genuine_scores: Similarity scores for same-speaker pairs
 impostor_scores: Similarity scores for different-speaker pairs
 
 Returns:
 Evaluation metrics and DET curve data
 """
 thresholds = np.linspace(-1, 1, 1000)
 
 fars = []
 frrs = []
 
 for threshold in thresholds:
 # False Accept Rate
 far = np.mean(impostor_scores >= threshold)
 
 # False Reject Rate
 frr = np.mean(genuine_scores < threshold)
 
 fars.append(far)
 frrs.append(frr)
 
 fars = np.array(fars)
 frrs = np.array(frrs)
 
 # Equal Error Rate
 eer_idx = np.argmin(np.abs(fars - frrs))
 eer = (fars[eer_idx] + frrs[eer_idx]) / 2
 eer_threshold = thresholds[eer_idx]
 
 # Detection Cost Function (DCF)
 # Weighted combination of FAR and FRR
 c_miss = 1.0
 c_fa = 1.0
 p_target = 0.01 # Prior probability of target speaker
 
 dcf = c_miss * frrs * p_target + c_fa * fars * (1 - p_target)
 min_dcf = np.min(dcf)
 
 return {
 'eer': eer,
 'eer_threshold': eer_threshold,
 'min_dcf': min_dcf,
 'det_curve': {
 'fars': fars,
 'frrs': frrs,
 'thresholds': thresholds
 }
 }
 
 def plot_det_curve(self, fars, frrs):
 """
 Plot Detection Error Tradeoff (DET) curve
 """
 import matplotlib.pyplot as plt
 
 plt.figure(figsize=(8, 6))
 plt.plot(fars * 100, frrs * 100)
 plt.xlabel('False Acceptance Rate (%)')
 plt.ylabel('False Rejection Rate (%)')
 plt.title('DET Curve')
 plt.grid(True)
 plt.xscale('log')
 plt.yscale('log')
 plt.show()
``

---

## Security Considerations

### Attack Vectors

1. **Replay Attack:** Recording and replaying legitimate user's voice
2. **Synthesis Attack:** TTS or voice cloning
3. **Impersonation:** Human mimicking target speaker
4. **Adversarial Audio:** Crafted audio to fool model

### Mitigation Strategies

``python
class SecurityEnhancedVerifier:
 """
 Speaker verification with security enhancements
 """
 
 def __init__(self, verifier, anti_spoofing_detector):
 self.verifier = verifier
 self.anti_spoofing = anti_spoofing_detector
 self.challenge_phrases = [
 "My voice is my password",
 "Today is a beautiful day",
 "Open sesame"
 ]
 
 def verify_with_liveness(
 self,
 audio,
 claimed_identity: str,
 expected_phrase: str = None
 ) -> dict:
 """
 Verify with liveness detection
 
 Steps:
 1. Anti-spoofing check
 2. Speaker verification
 3. Optional: Speech content verification
 """
 # Step 1: Anti-spoofing
 spoofing_result = self.anti_spoofing.detect_spoofing(audio)
 
 if not spoofing_result['is_genuine']:
 return {
 'verified': False,
 'reason': 'spoofing_detected',
 'spoofing_confidence': spoofing_result['confidence']
 }
 
 # Step 2: Speaker verification
 verification_result = self.verifier.verify(audio, claimed_identity)
 
 if not verification_result['is_same_speaker']:
 return {
 'verified': False,
 'reason': 'speaker_mismatch',
 'similarity': verification_result['similarity']
 }
 
 # Step 3: Optional phrase verification
 if expected_phrase:
 # Use ASR to verify phrase
 # transcription = asr_model.transcribe(audio)
 # phrase_match = transcription.lower() == expected_phrase.lower()
 phrase_match = True # Placeholder
 
 if not phrase_match:
 return {
 'verified': False,
 'reason': 'phrase_mismatch'
 }
 
 return {
 'verified': True,
 'similarity': verification_result['similarity'],
 'spoofing_confidence': spoofing_result['confidence']
 }
``

---

## Key Takeaways

✅ **Speaker embeddings** (x-vectors) map audio → fixed vector 
✅ **Verification** (1:1) vs **Identification** (1:N) 
✅ **Cosine similarity** for comparing embeddings 
✅ **EER** (Equal Error Rate) balances FAR and FRR 
✅ **FAISS** enables fast similarity search for millions of speakers 
✅ **Speaker diarization** segments audio by speaker 
✅ **Domain adaptation** critical for robustness across conditions 
✅ **Multi-modal biometrics** combine voice + face for stronger security 
✅ **Model compression** enables edge deployment 
✅ **Anti-spoofing** critical for security applications 
✅ **Streaming enrollment** builds profiles incrementally 
✅ **Production systems** need enrollment, verification, and identification APIs 
✅ **Real-world uses:** Voice assistants, call centers, security, forensics 

---

**Originally published at:** [arunbaby.com/speech-tech/0005-speaker-recognition](https://www.arunbaby.com/speech-tech/0005-speaker-recognition/)

*If you found this helpful, consider sharing it with others who might benefit.*

