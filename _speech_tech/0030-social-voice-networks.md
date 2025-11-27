---
title: "Social Voice Networks"
day: 30
collection: speech_tech
categories:
  - speech_tech
tags:
  - social
  - voice
  - recommendation
  - speaker recognition
subdomain: "Social Speech Applications"
tech_stack: [Clubhouse, Discord, x-vector, ASR, GNN]
scale: "Real-time, Millions of Users"
companies: [Clubhouse, Discord, Twitter Spaces, LinkedIn Audio]
related_dsa_day: 30
related_ml_day: 30
related_speech_day: 30
---

**"Building recommendation and moderation systems for voice-based social platforms."**

## 1. What are Social Voice Networks?

Social Voice Networks are platforms where users interact primarily through **live audio** rather than text or images.

**Examples:**
-   **Clubhouse:** Live audio rooms with speakers and listeners.
-   **Twitter Spaces:** Audio conversations linked to Twitter.
-   **Discord Voice Channels:** Real-time voice chat for gaming/communities.
-   **LinkedIn Audio:** Professional networking via voice events.

**Unique Challenges:**
1. **Ephemeral:** Content disappears (unlike text posts).
2. **Real-time Moderation:** Can't wait for human review.
3. **Speaker Identification:** Who said what?
4. **Content Recommendation:** Suggest relevant rooms/conversations.

## 2. System Architecture

```
┌──────────────────────────────────────────────┐
│         Social Voice Network Platform         │
├──────────────────────────────────────────────┤
│                                               │
│  ┌────────────┐  ┌────────────┐              │
│  │ Live Audio │  │  Speaker   │              │
│  │  Streams   │  │Recognition │              │
│  └──────┬─────┘  └──────┬─────┘              │
│         │                │                    │
│         v                v                    │
│  ┌──────────────────────────────┐            │
│  │      ASR (Speech-to-Text)    │            │
│  └──────────┬───────────────────┘            │
│             │                                 │
│             v                                 │
│  ┌──────────────────────────────┐            │
│  │   Content Moderation          │            │
│  │   (Toxicity, Misinformation)  │            │
│  └──────────┬───────────────────┘            │
│             │                                 │
│             v                                 │
│  ┌──────────────────────────────┐            │
│  │  Topic Extraction & Indexing │            │
│  └──────────┬───────────────────┘            │
│             │                                 │
│             v                                 │
│  ┌──────────────────────────────┐            │
│  │  Recommendation Engine        │            │
│  │  (User → Room matching)       │            │
│  └──────────────────────────────┘            │
│                                               │
└──────────────────────────────────────────────┘
```

## 3. Speaker Recognition (Diarization)

**Problem:** In a room with 10 speakers, attribute each utterance to the correct speaker.

### x-Vector Embeddings

**Architecture:**
```
Audio (MFCC features)
  ↓
TDNN (Time Delay Neural Network)
  ↓
Statistics Pooling (mean + std over time)
  ↓
Fully Connected Layers
  ↓
x-vector (512-dim embedding)
```

**Training:** Softmax loss over speaker IDs.
**Inference:** Extract x-vector for each segment, cluster to identify speakers.

```python
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# Load pre-trained x-vector model
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")

def extract_speaker_embedding(audio_path):
    # Load audio
    signal, fs = torchaudio.load(audio_path)
    
    # Extract x-vector
    embeddings = classifier.encode_batch(signal)
    
    return embeddings.squeeze()  # [512]

# Clustering speakers
from sklearn.cluster import AgglomerativeClustering

embeddings = [extract_speaker_embedding(segment) for segment in segments]
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
labels = clustering.fit_predict(embeddings)

# labels[i] = speaker ID for segment i
```

### Speaker Change Detection

**Problem:** Detect when a new speaker starts talking.

**Approach: Bayesian Information Criterion (BIC)**
```
For each potential change point t:
  Model 1: [0, t] and [t+1, T] as two separate Gaussians
  Model 2: [0, T] as one Gaussian
  ΔBIC = BIC(Model1) - BIC(Model2)
  
If ΔBIC > threshold: Change point detected
```

**Neural Approach:** LSTM that predicts change points.
```python
class SpeakerChangeDetector(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(input_size=40, hidden_size=256, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(512, 2)  # Binary classification: change or no change
    
    def forward(self, mfcc):
        # mfcc: [batch, time, 40]
        lstm_out, _ = self.lstm(mfcc)
        logits = self.fc(lstm_out)  # [batch, time, 2]
        return logits
```

## 4. Real-time ASR for Transcription

**Challenge:** Transcribe live audio with < 200ms latency.

**Solution: Streaming RNN-T (RNN-Transducer)**

**Architecture:**
1. **Encoder:** Processes audio chunks (e.g., 80ms frames).
2. **Prediction Network:** Language model over previously emitted tokens.
3. **Joint Network:** Combines encoder + prediction to emit tokens.

```python
class StreamingRNNT(nn.Module):
    def __init__(self):
        self.encoder = ConformerEncoder(streaming=True)
        self.prediction = nn.LSTM(input_size=vocab_size, hidden_size=512)
        self.joint = nn.Linear(512 + 512, vocab_size)
    
    def forward(self, audio_chunk, prev_token, prev_hidden):
        # Encode audio
        enc_out = self.encoder(audio_chunk)  # [1, 512]
        
        # Predict next token
        pred_out, hidden = self.prediction(prev_token, prev_hidden)  # [1, 512]
        
        # Joint
        joint_out = self.joint(torch.cat([enc_out, pred_out], dim=1))  # [1, vocab_size]
        
        # Greedy decode
        token = joint_out.argmax(dim=1)
        
        return token, hidden
```

**Latency Breakdown:**
- Audio chunk: 80ms
- Encoder: 50ms
- Prediction + Joint: 10ms
- **Total:** ~140ms (meets real-time requirement)

## 5. Content Moderation

**Challenge:** Detect toxic speech, misinformation, harassment in real-time.

### Toxicity Detection

**Pipeline:**
1. **ASR:** Audio → Text.
2. **Text Classifier:** BERT-based toxicity detector.
3. **Audio Features:** Prosody (shouting, aggressive tone).
4. **Fusion:** Combine text + audio scores.

```python
class ToxicityDetector(nn.Module):
    def __init__(self):
        self.text_encoder = BERTModel()
        self.audio_encoder = ResNet1D()  # Conv1D on mel-spectrogram
        self.fusion = nn.Linear(768 + 256, 2)  # Binary: toxic or not
    
    def forward(self, text_tokens, audio):
        text_emb = self.text_encoder(text_tokens).pooler_output  # [batch, 768]
        audio_emb = self.audio_encoder(audio).squeeze()  # [batch, 256]
        
        combined = torch.cat([text_emb, audio_emb], dim=1)
        logits = self.fusion(combined)
        
        return logits

# Inference
text = asr(audio_chunk)
logits = toxicity_detector(text, audio_chunk)
is_toxic = logits.argmax(dim=1) == 1

if is_toxic:
    # Mute speaker, alert moderators
    send_alert(speaker_id, timestamp)
```

### Misinformation Detection

**Challenge:** "This vaccine contains microchips" needs to be flagged.

**Approach:**
1. **Fact-Checking API:** Query external fact-checkers (Snopes, FactCheck.org).
2. **Claim Detection:** NER to extract claims ("vaccine contains microchips").
3. **Verification:** Compare claim against knowledge base.

```python
def detect_misinformation(transcript):
    # Extract claims using NER
    claims = ner_model.extract_claims(transcript)
    
    for claim in claims:
        # Query fact-checking APIs
        fact_check_result = fact_check_api.verify(claim)
        
        if fact_check_result.confidence > 0.8 and fact_check_result.verdict == "false":
            return True, claim
    
    return False, None
```

## 6. Topic Extraction and Tagging

**Problem:** Tag each room with topics (e.g., "Technology", "Startup Funding", "AI").

**Approach: LDA + Neural Topic Models**

### Latent Dirichlet Allocation (LDA)
```python
from sklearn.decomposition import LatentDirichletAllocation

# Collect transcripts from a room
transcripts = [asr(audio) for audio in room_audio_chunks]
combined_text = " ".join(transcripts)

# Vectorize
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform([combined_text])

# LDA
lda = LatentDirichletAllocation(n_components=10)
topics = lda.fit_transform(X)

# Top topic
top_topic_id = topics.argmax()
```

### Neural Topic Model (with BERT)
```python
class NeuralTopicModel(nn.Module):
    def __init__(self, num_topics=100):
        self.encoder = BERTModel()
        self.topic_layer = nn.Linear(768, num_topics)
    
    def forward(self, text):
        emb = self.encoder(text).pooler_output  # [batch, 768]
        topic_dist = F.softmax(self.topic_layer(emb), dim=1)  # [batch, num_topics]
        return topic_dist

# Tag room
topic_dist = neural_topic_model(room_transcript)
top_topics = topic_dist.argsort(descending=True)[:3]  # Top 3 topics
room_tags = [topic_names[t] for t in top_topics]
```

## 7. Room Recommendation (User → Room Matching)

**Challenge:** Suggest relevant rooms to users.

### Graph-based Approach

**Graph:**
- **Nodes:** Users, Rooms, Topics.
- **Edges:** 
  - User --joined--> Room
  - Room --tagged_with--> Topic
  - User --interested_in--> Topic

**Recommendation:**
1. **Random Walk:** Start from user, walk through graph.
2. **Frequency:** Rooms visited most often in walks are recommended.

```python
def personalized_pagerank(graph, user_id, alpha=0.85, num_walks=1000):
    scores = defaultdict(float)
    
    for _ in range(num_walks):
        current = user_id
        for step in range(10):
            if random.random() < (1 - alpha):
                current = user_id  # Restart
            else:
                neighbors = graph.neighbors(current)
                if neighbors:
                    current = random.choice(neighbors)
            
            if graph.node_type(current) == "Room":
                scores[current] += 1
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
```

### Collaborative Filtering

**Matrix:** Users × Rooms (1 if user joined room, 0 otherwise).
**Matrix Factorization:**
\\[
R \approx U V^T
\\]
where \\(U\\) is user embeddings, \\(V\\) is room embeddings.

```python
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_rooms, k=128):
        self.user_emb = nn.Embedding(num_users, k)
        self.room_emb = nn.Embedding(num_rooms, k)
    
    def forward(self, user_id, room_id):
        u = self.user_emb(user_id)  # [batch, k]
        v = self.room_emb(room_id)  # [batch, k]
        return (u * v).sum(dim=1)  # Dot product

# Training
loss = mse_loss(model(user_batch, room_batch), labels)
```

### Content-based Filtering

**Idea:** Recommend rooms similar to those the user joined before.

```python
# Extract room content features
room_features = {
    room_id: topic_model(room_transcript) for room_id in rooms
}

# User profile: average of joined rooms
user_profile = np.mean([room_features[r] for r in user_joined_rooms], axis=0)

# Recommend by cosine similarity
recommendations = sorted(
    [(r, cosine_similarity(user_profile, room_features[r])) for r in rooms],
    key=lambda x: x[1],
    reverse=True
)[:10]
```

## Deep Dive: Clubhouse's Recommendation Algorithm

**Clubhouse** uses a multi-stage funnel:

### Stage 1: Candidate Generation
- **Social Graph:** Rooms that user's friends are in (95% of recommendations come from social graph).
- **Topic Graph:** Rooms tagged with user's interested topics.
- **Collaborative Filtering:** "Users similar to you joined these rooms."

### Stage 2: Ranking
**Features:**
1. **User Features:** Interests, past room joins, time of day.
2. **Room Features:** Number of speakers, current topic, speaker reputation.
3. **Interaction Features:** Number of friends in the room, historical engagement.

**Model:** Gradient Boosted Trees (XGBoost).
**Target:** P(user joins and stays > 5 minutes).

### Stage 3: Diversity
Re-rank to ensure variety (not all tech rooms, not all from same friend).

**Maximal Marginal Relevance (MMR):**
\\[
\text{Score}(r) = \lambda \cdot \text{Relevance}(r) - (1 - \lambda) \cdot \max_{r' \in S} \text{Similarity}(r, r')
\\]
where \\(S\\) is the set of already selected rooms.

## Deep Dive: Discord's Voice Activity Detection (VAD)

**Challenge:** Detect when a user is speaking vs. background noise.

**Traditional VAD:** Energy threshold (volume > X dB).
**Problem:** Fails with background noise (TV, music).

**Neural VAD:**
```python
class NeuralVAD(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=2)
        self.fc = nn.Linear(128, 2)  # Speech or silence
    
    def forward(self, mfcc):
        # mfcc: [batch, time, 40]
        lstm_out, _ = self.lstm(mfcc)
        logits = self.fc(lstm_out[:, -1, :])  # Use last timestep
        return logits

# Inference
is_speaking = neural_vad(audio_chunk).argmax(dim=1) == 1
if is_speaking:
    transmit_audio_to_server()
```

**Benefit:** Reduces bandwidth by 80% (don't transmit silence).

## Deep Dive: Echo Cancellation for Voice Chat

**Problem:** User A hears User B. User B hears their own voice echoed back (loop).

**Acoustic Echo Cancellation (AEC):**
```
Reference signal: What we played (User B's voice)
Microphone signal: What we recorded (User A speaking + User B's echo)

Goal: Subtract the echo from the microphone signal
```

**Adaptive Filter (NLMS - Normalized Least Mean Squares):**
```python
def nlms_aec(reference, microphone, step_size=0.01, filter_length=512):
    h = np.zeros(filter_length)  # Adaptive filter coefficients
    output = []
    
    for n in range(filter_length, len(microphone)):
        # Reference window
        x = reference[n - filter_length:n]
        
        # Predicted echo
        echo_estimate = np.dot(h, x)
        
        # Error (remove echo)
        error = microphone[n] - echo_estimate
        output.append(error)
        
        # Update filter (NLMS)
        h += (step_size / (np.dot(x, x) + 1e-8)) * error * x
    
    return np.array(output)
```

**Modern Approach:** End-to-end neural AEC (Facebook's Demucs).

## Deep Dive: Noise Suppression (Krisp, NVIDIA RTX Voice)

**Problem:** Background noise (dogs barking, keyboard clicks, traffic).

**Solution: Deep Learning Noise Suppression**

**Architecture: U-Net on Spectrogram**
```python
class NoiseSuppressionUNet(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid()  # Mask (0 to 1)
        )
    
    def forward(self, noisy_spec):
        # noisy_spec: [batch, 1, freq, time]
        enc = self.encoder(noisy_spec)
        mask = self.decoder(enc)  # [batch, 1, freq, time]
        clean_spec = noisy_spec * mask  # Element-wise multiply
        return clean_spec
```

**Training:**
- Input: Noisy spectrogram.
- Target: Clean spectrogram.
- Loss: L1 loss on magnitude spectrogram.

**Inference:** 10-20ms latency (real-time capable).

## Deep Dive: Speaker Identification for Personalization

**Problem:** Recognize specific users by their voice (not just cluster speakers).

**Approach: Speaker Verification**
```python
class SpeakerVerifier(nn.Module):
    def __init__(self):
        self.encoder = ResNet1D()  # Extract speaker embedding
    
    def forward(self, audio_enrollment, audio_test):
        emb_enroll = self.encoder(audio_enrollment)  # [1, 512]
        emb_test = self.encoder(audio_test)  # [1, 512]
        
        # Cosine similarity
        similarity = F.cosine_similarity(emb_enroll, emb_test)
        
        return similarity  # > 0.8 → Same speaker

# Enrollment
user_emb = model.encoder(user_audio_samples)
db.store(user_id, user_emb)

# Test
test_emb = model.encoder(test_audio)
similarity = cosine_similarity(test_emb, db.get(user_id))
if similarity > 0.8:
    authenticated = True
```

**Use Case:** Automatically mute/unmute the correct participant in a meeting.

## Deep Dive: Bandwidth Optimization (Opus Codec)

**Challenge:** Stream high-quality audio with minimal bandwidth.

**Opus Codec:**
- **Bitrate:** 6-510 kbps (adaptive).
- **Latency:** 5-66

 ms.
- **Quality:** Superior to MP3 at same bitrate.

**Adaptive Bitrate:**
```python
def adjust_bitrate(network_conditions):
    if network_conditions['bandwidth'] > 100:  # kbps
        return 128  # High quality
    elif network_conditions['bandwidth'] > 50:
        return 64  # Medium quality
    else:
        return 24  # Low quality (voice still intelligible)
```

## Deep Dive: Scalability (Handling Million Concurrent Users)

**Architecture:**
```
         ┌──────────────┐
         │  Load Balancer│
         └───────┬───────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐  ┌────▼────┐  ┌────▼────┐
│Server1│  │ Server2 │  │ Server3 │
└───────┘  └─────────┘  └─────────┘
    │            │            │
    └────────────┼────────────┘
                 │
         ┌───────▼────────┐
         │   Media Server  │
         │   (Janus, Jitsi)│
         └────────────────┘
```

**Techniques:**
1. **WebRTC SFU (Selective Forwarding Unit):** Server forwards audio streams without decoding/encoding (low CPU).
2. **Regional Servers:** Route users to nearest server (reduce latency).
3. **Adaptive Quality:** Reduce bitrate under load.

**Clubhouse Scale:**
- Peak: 2M concurrent users.
- Solution: Agora.io (infrastructure provider) with auto-scaling.

## Implementation: Full Social Voice Network Backend

```python
import torch
import torch.nn as nn
from transformers import BertModel
import torchaudio

class SocialVoiceBackend:
    def __init__(self):
        self.asr_model = load_asr_model()
        self.speaker_recognition = load_xvector_model()
        self.toxicity_detector = ToxicityDetector()
        self.topic_model = NeuralTopicModel()
        self.recommender = GraphRecommender()
    
    def process_audio_chunk(self, audio_chunk, room_id, user_id):
        # 1. Speaker Recognition
        speaker_emb = self.speaker_recognition.encode(audio_chunk)
        speaker_id = self.cluster_speaker(speaker_emb, room_id)
        
        # 2. ASR
        transcript = self.asr_model.transcribe(audio_chunk)
        
        # 3. Content Moderation
        is_toxic, reason = self.toxicity_detector(transcript, audio_chunk)
        if is_toxic:
            self.mute_speaker(speaker_id, room_id, reason)
            return
        
        # 4. Update room metadata
        self.update_room_transcript(room_id, transcript)
        
        # 5. Extract topics (every 60 seconds)
        if should_update_topics(room_id):
            topics = self.topic_model(get_room_transcript(room_id))
            self.update_room_topics(room_id, topics)
    
    def recommend_rooms(self, user_id, top_k=10):
        # Get user interests
        user_profile = self.get_user_profile(user_id)
        
        # Candidate generation
        candidates = []
        
        # 1. Social graph
        friends = self.get_friends(user_id)
        candidates += self.get_rooms_with_users(friends)
        
        # 2. Topic matching
        candidates += self.get_rooms_by_topics(user_profile['interests'])
        
        # 3. Collaborative filtering
        similar_users = self.find_similar_users(user_id)
        candidates += self.get_popular_rooms_among(similar_users)
        
        # Rank candidates
        scores = self.recommender.rank(user_id, candidates)
        
        # Diversify
        diverse_rooms = self.apply_mmr(scores, lambda_param=0.7)
        
        return diverse_rooms[:top_k]

# Usage
backend = SocialVoiceBackend()

# Process incoming audio
for audio_chunk in stream:
    backend.process_audio_chunk(audio_chunk, room_id="tech_talk_123", user_id="alice")

# Recommend rooms
recommendations = backend.recommend_rooms(user_id="alice", top_k=10)
```

## Top Interview Questions

**Q1: How do you handle speaker overlap (two people speaking simultaneously)?**
*Answer:*
Use **source separation** models (e.g., Conv-TasNet, Sudo RM-RF) to separate the overlapping voices into individual tracks. Then run ASR and speaker recognition on each track separately.

**Q2: How do you ensure low latency for global users?**
*Answer:*
- Deploy servers in multiple regions (US East, US West, Europe, Asia).
- Route users to nearest server using GeoDNS.
- Use CDN for static assets.
- Optimize codec (Opus) with adaptive bitrate.

**Q3: How do you detect and prevent spam/abuse in voice rooms?**
*Answer:*
- **Real-time ASR + Toxicity Detection:** Flag toxic speech immediately.
- **Rate Limiting:** Limit number of rooms a user can create per day.
- **Reputation System:** Users with low reputation (many reports) are auto-moderated.
- **Audio Fingerprinting:** Detect and block pre-recorded spam ads.

**Q4: How do you make recommendations work for new users (cold start)?**
*Answer:*
- **Onboarding:** Ask users to select interests during signup.
- **Popular Rooms:** Show trending rooms to new users.
- **Social Graph:** If user connects social accounts, bootstrap recommendations from friends' activity.

## Key Takeaways

1. **Real-time Constraints:** ASR, speaker recognition, moderation must run in < 200ms.
2. **Speaker Diarization:** x-vector embeddings + clustering to attribute speech.
3. **Content Moderation:** Combine text (ASR output) + audio (prosody) for toxicity detection.
4. **Recommendations:** Graph-based (social graph + topic graph) outperform pure collaborative filtering.
5. **Scalability:** Use SFU architecture, regional servers, adaptive bitrate for millions of concurrent users.

## Summary

| Aspect | Insight |
|:---|:---|
| **Core Components** | ASR, Speaker Recognition, Moderation, Recommendation |
| **Key Challenges** | Real-time latency, ephemeral content, cold start |
| **Architectures** | Streaming RNN-T (ASR), x-vector (Speaker), GNN (Recommendations) |
| **Real-World** | Clubhouse, Discord, Twitter Spaces |

---

**Originally published at:** [arunbaby.com/speech-tech/0030-social-voice-networks](https://www.arunbaby.com/speech-tech/0030-social-voice-networks/)

*If you found this helpful, consider sharing it with others who might benefit.*


