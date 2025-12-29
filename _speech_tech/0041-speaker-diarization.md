---
title: "Speaker Diarization"
day: 41
related_dsa_day: 41
related_ml_day: 41
related_agents_day: 41
collection: speech_tech
categories:
 - speech-tech
tags:
 - speaker-recognition
 - clustering
 - segmentation
 - embeddings
difficulty: Hard
---

**"Who spoke when? The art of untangling voices."**

## 1. Introduction

**Speaker Diarization** is the task of partitioning an audio stream into homogeneous segments according to the speaker identity. In simple terms: "Who spoke when?"

**Input:** Audio recording (e.g., meeting, podcast, phone call).
**Output:** Timeline with speaker labels.

**Example:**
``
[00:00 - 00:15] Speaker A: "Hello, how are you?"
[00:15 - 00:30] Speaker B: "I'm doing well, thanks!"
[00:30 - 00:45] Speaker A: "Great to hear."
``

**Applications:**
* **Meeting Transcription:** Zoom, Google Meet, Microsoft Teams.
* **Call Center Analytics:** Separate agent from customer.
* **Podcast Indexing:** Identify different speakers for search.
* **Forensics:** Identify speakers in surveillance recordings.

## 2. Problem Formulation

**Challenges:**
1. **Unknown Number of Speakers:** We don't know how many people are speaking.
2. **Overlapping Speech:** Multiple people speaking simultaneously.
3. **Variable Segment Length:** Speakers may talk for 1 second or 10 minutes.
4. **Acoustic Variability:** Background noise, channel effects, emotion.

**Metrics:**
* **Diarization Error Rate (DER):** Percentage of time that is incorrectly attributed.
 * `DER = \frac{FA + MISS + CONF}{TOTAL}`
 * **FA (False Alarm):** Non-speech detected as speech.
 * **MISS:** Speech detected as non-speech.
 * **CONF (Confusion):** Speech attributed to wrong speaker.

## 3. Traditional Pipeline

The classic diarization system has 4 stages:

### 3.1. Speech Activity Detection (SAD)
* **Goal:** Remove silence and non-speech (music, noise).
* **Method:** Energy-based VAD or DNN-based VAD.

### 3.2. Segmentation
* **Goal:** Split audio into short, homogeneous segments (e.g., 1-2 seconds).
* **Method:** 
 * **Fixed-Length:** Simple, but may split mid-sentence.
 * **Change Point Detection:** Detect speaker changes using Bayesian Information Criterion (BIC) or GLR (Generalized Likelihood Ratio).

### 3.3. Embedding Extraction
* **Goal:** Convert each segment into a fixed-size vector (embedding) that represents the speaker's voice.
* **Method:** 
 * **i-vectors:** Traditional approach using GMM-UBM.
 * **x-vectors:** Deep learning approach using TDNN (Time-Delay Neural Network).
 * **d-vectors:** Trained with triplet loss.

### 3.4. Clustering
* **Goal:** Group segments by speaker.
* **Method:**
 * **Agglomerative Hierarchical Clustering (AHC):** Bottom-up clustering.
 * **Spectral Clustering:** Graph-based clustering.
 * **PLDA (Probabilistic Linear Discriminant Analysis):** Probabilistic scoring.

## 4. Deep Dive: X-Vectors

**Architecture:**
* **Input:** Mel-Frequency Cepstral Coefficients (MFCCs) or Mel-Filterbanks.
* **Layers:**
 1. **Frame-level TDNN:** 5 layers with temporal context (e.g., [-2, +2] frames).
 2. **Statistics Pooling:** Compute mean and standard deviation across time.
 3. **Segment-level Fully Connected:** 2 layers.
 4. **Output:** 512-dimensional embedding.

**Training:**
* **Dataset:** VoxCeleb (1M+ utterances, 7000+ speakers).
* **Loss:** Softmax (speaker classification) or AAM-Softmax (Additive Angular Margin).

**Inference:**
* Extract x-vector for each segment.
* Compute cosine similarity between x-vectors.
* Cluster based on similarity.

## 5. Deep Dive: Clustering Algorithms

### 5.1. Agglomerative Hierarchical Clustering (AHC)

**Algorithm:**
1. Start with each segment as its own cluster.
2. **Merge Step:** Find the two closest clusters and merge them.
3. Repeat until a stopping criterion is met (e.g., threshold on distance, or target number of clusters).

**Distance Metrics:**
* **Average Linkage:** Average distance between all pairs.
* **Complete Linkage:** Maximum distance between any pair.
* **PLDA Score:** Probabilistic score.

**Stopping Criterion:**
* **Threshold:** Stop when the minimum distance > threshold.
* **Eigengap:** Use spectral clustering to estimate the number of speakers.

### 5.2. Spectral Clustering

**Algorithm:**
1. Construct an **Affinity Matrix** `A` where `A_{ij} = \text{similarity}(x_i, x_j)`.
2. Compute the **Graph Laplacian** `L = D - A` where `D` is the degree matrix.
3. Compute the eigenvectors of `L`.
4. Use the first `k` eigenvectors as features.
5. Apply k-means clustering.

**Advantage:** Can handle non-convex clusters.

## 6. End-to-End Neural Diarization (EEND)

**Idea:** Train a single neural network to directly predict speaker labels for each frame.

**Architecture:**
* **Input:** Mel-Filterbanks (e.g., 500 frames).
* **Encoder:** Transformer or LSTM.
* **Output:** `T \times K` matrix where `T` is time frames, `K` is max number of speakers. Each entry is the probability that speaker `k` is active at time `t`.

**Training:**
* **Loss:** Binary Cross-Entropy for each speaker.
* **Permutation Invariant Training (PIT):** Since speaker labels are arbitrary, we need to find the best permutation of predicted labels to match ground truth.

**Advantage:**
* No need for separate segmentation and clustering.
* Can handle overlapping speech naturally.

**Disadvantage:**
* Requires large amounts of labeled data.
* Fixed maximum number of speakers.

## 7. Handling Overlapping Speech

Traditional diarization assumes only one speaker at a time. In reality, people interrupt each other.

**Solutions:**

**1. Multi-Label Classification:**
* Instead of assigning one speaker per frame, allow multiple speakers.
* Output: `T \times K` binary matrix.

**2. Source Separation:**
* Use a speech separation model (e.g., Conv-TasNet) to separate overlapping speakers.
* Then run diarization on each separated stream.

**3. EEND with Overlap:**
* Train EEND to predict overlapping speakers.

## 8. System Design: Real-Time Diarization for Video Conferencing

**Scenario:** Zoom meeting with 10 participants. We want to display speaker labels in real-time.

**Constraints:**
* **Latency:** < 1 second.
* **Accuracy:** DER < 10%.
* **Scalability:** Handle 100+ concurrent meetings.

**Architecture:**

**Step 1: Audio Capture**
* Each participant's audio is streamed to the server.

**Step 2: VAD**
* Run a lightweight VAD model (e.g., WebRTC VAD) to detect speech.

**Step 3: Embedding Extraction**
* Use a streaming x-vector model.
* Extract embeddings every 1 second (with 0.5s overlap).

**Step 4: Online Clustering**
* Use **Online AHC** or **Online Spectral Clustering**.
* Update clusters incrementally as new embeddings arrive.

**Step 5: Speaker Tracking**
* Maintain a "speaker profile" for each participant.
* Match new embeddings to existing profiles.

**Step 6: Display**
* Send speaker labels back to clients.
* Display "Speaker A is talking" in the UI.

**Optimization:**
* **Batching:** Process multiple meetings in parallel on GPU.
* **Caching:** Cache embeddings for known speakers.

## 9. Deep Dive: PLDA (Probabilistic Linear Discriminant Analysis)

PLDA is a probabilistic model for speaker verification and diarization.

**Model:**
`x = m + Fy + \epsilon`
* `x`: Observed embedding (x-vector).
* `m`: Global mean.
* `F`: Factor loading matrix (speaker variability).
* `y`: Latent speaker factor.
* `\epsilon`: Residual noise.

**Scoring:**
Given two embeddings `x_1` and `x_2`, compute the log-likelihood ratio:
`\text{score} = \log \frac{P(x_1, x_2 | \text{same speaker})}{P(x_1, x_2 | \text{different speakers})}`

**Use in Diarization:**
* Use PLDA score as the distance metric in AHC.

## 10. Evaluation Metrics

**Diarization Error Rate (DER):**
`DER = \frac{\text{False Alarm} + \text{Missed Speech} + \text{Speaker Confusion}}{\text{Total Speech Time}}`

**Jaccard Error Rate (JER):**
Measures overlap between predicted and ground truth speaker segments.

**Optimal Mapping:**
Since speaker labels are arbitrary (Speaker A vs Speaker 1), we need to find the optimal mapping between predicted and ground truth labels (Hungarian algorithm).

## 11. Datasets

**1. CALLHOME:**
* Telephone conversations.
* 2-7 speakers per conversation.
* Used in NIST evaluations.

**2. AMI Meeting Corpus:**
* Meeting recordings.
* 4 speakers per meeting.
* Includes video and slides.

**3. DIHARD:**
* Diverse domains (meetings, interviews, broadcasts, YouTube).
* Challenging: overlapping speech, noise.

**4. VoxConverse:**
* YouTube videos.
* Variable number of speakers.

## 12. Interview Questions

1. **Explain Speaker Diarization.** What are the main challenges?
2. **X-Vectors vs I-Vectors.** What are the differences?
3. **Clustering.** Why use AHC instead of k-means?
4. **Overlapping Speech.** How do you handle it?
5. **Real-Time Diarization.** How would you design a system for live meetings?
6. **Calculate DER.** Given ground truth and predictions, compute DER.

## 13. Common Mistakes

* **Ignoring Overlapping Speech:** Assuming only one speaker at a time leads to high confusion errors.
* **Fixed Number of Speakers:** Using k-means with a fixed `k` when the number of speakers is unknown.
* **Poor VAD:** If VAD misses speech or includes noise, diarization will fail.
* **Not Normalizing Embeddings:** Cosine similarity requires normalized embeddings.
* **Overfitting to Domain:** A model trained on telephone speech may fail on meeting recordings.

## 14. Advanced EEND Architectures

### 14.1. EEND-EDA (Encoder-Decoder Attractor)

**Problem:** Standard EEND has a fixed maximum number of speakers.

**Solution:** Use an encoder-decoder architecture with **attractors**.

**Architecture:**
1. **Encoder:** Transformer encodes the audio.
2. **Attractor Estimation:** A decoder estimates the number of speakers and their "attractors" (prototype embeddings).
3. **Speaker Assignment:** For each frame, compute similarity to each attractor. Assign to the closest attractor.

**Benefit:** Handles variable number of speakers without retraining.

### 14.2. EEND with Self-Attention

**Idea:** Use self-attention to model long-range dependencies between frames.

**Architecture:**
* **Input:** Mel-Filterbanks.
* **Encoder:** Multi-head self-attention (Transformer).
* **Output:** Speaker activity matrix.

**Training:**
* **Permutation Invariant Training (PIT):** Find the best permutation of predicted speakers to match ground truth.
* **Loss:** Binary Cross-Entropy for each speaker.

**Result:** State-of-the-art performance on CALLHOME (DER < 10%).

### 14.3. EEND with Target-Speaker Voice Activity Detection (TS-VAD)

**Idea:** Given a target speaker's enrollment audio, detect when that speaker is active.

**Use Case:** "Show me all segments where Alice spoke."

**Architecture:**
* **Input:** (1) Meeting audio, (2) Enrollment audio of Alice.
* **Encoder:** Extract embeddings for both.
* **Attention:** Cross-attention between meeting and enrollment.
* **Output:** Binary mask (Alice active or not).

## 15. Production Case Study: Zoom Diarization

**Scenario:** Zoom meeting with 10 participants. Display speaker labels in real-time.

**Challenges:**
* **Latency:** < 1 second.
* **Accuracy:** DER < 10%.
* **Scalability:** 100K+ concurrent meetings.

**Solution:**

**Step 1: Audio Capture**
* Each participant's audio is streamed to the server (separate tracks).

**Step 2: VAD**
* Run WebRTC VAD on each track.

**Step 3: Embedding Extraction**
* Use a streaming x-vector model (ResNet-based).
* Extract embeddings every 1 second.

**Step 4: Online Clustering**
* Use **Online AHC** with PLDA scoring.
* Update clusters incrementally.

**Step 5: Speaker Tracking**
* Maintain a "speaker profile" for each participant.
* Use speaker verification to match new embeddings to profiles.

**Step 6: Display**
* Send speaker labels to clients via WebSocket.

**Optimization:**
* **GPU Batching:** Process 100 meetings in parallel on a single V100.
* **Caching:** Cache embeddings for known speakers (reduces compute by 50%).

**Result:**
* **Latency:** 500ms.
* **DER:** 8%.
* **Cost:** $0.01/hour/meeting.

## 16. Multi-Modal Diarization (Audio + Video)

**Idea:** Use visual cues (lip movement, face detection) to improve diarization.

**Approach:**

**1. Face Detection:**
* Use MTCNN or RetinaFace to detect faces in each frame.

**2. Active Speaker Detection (ASD):**
* Train a model to predict if a face is speaking based on lip movement.
* **Input:** Face crop + audio.
* **Output:** Probability of speaking.

**3. Fusion:**
* Combine audio-based diarization with video-based ASD.
* **Rule:** If audio says Speaker A is active AND face detection says Person 1 is speaking, then Person 1 = Speaker A.

**Benefit:** Reduces confusion errors by 30-50% in meetings with video.

## 17. Deep Dive: Online Diarization

**Challenge:** Traditional diarization is offline (requires the entire audio). For live meetings, we need online diarization.

**Approach:**

**1. Sliding Window:**
* Process audio in chunks (e.g., 10 seconds).
* Run diarization on each chunk.

**2. Speaker Linking:**
* Link speakers across chunks using embeddings.
* **Challenge:** Speaker labels may change across chunks (Speaker A in chunk 1 = Speaker B in chunk 2).
* **Solution:** Use a global speaker tracker.

**3. Incremental Clustering:**
* Use **Online AHC** or **DBSCAN**.
* Add new segments to existing clusters or create new clusters.

## 18. Privacy & Security

**Concerns:**
* **Voice Biometrics:** Speaker embeddings can be used to identify individuals.
* **Surveillance:** Diarization enables tracking who said what.

**Solutions:**

**1. On-Device Diarization:**
* Run diarization locally (no audio sent to cloud).
* **Challenge:** Requires lightweight models.

**2. Differential Privacy:**
* Add noise to embeddings to prevent re-identification.
* **Trade-off:** Slight accuracy drop.

**3. Anonymization:**
* Replace speaker labels with pseudonyms (Speaker 1, Speaker 2).
* Don't store raw audio, only transcripts.

## 19. Deep Dive: Speaker Change Detection

**Problem:** Detect when the speaker changes (boundary detection).

**Approaches:**

**1. Bayesian Information Criterion (BIC):**
* For each potential change point `t`, compute:
 * `BIC(t) = \log P(X_{1:t}) + \log P(X_{t+1:T}) - \log P(X_{1:T})`
* If `BIC(t) > \theta`, there's a speaker change at `t`.

**2. Generalized Likelihood Ratio (GLR):**
* Similar to BIC, but uses likelihood ratio.

**3. Neural Change Detection:**
* Train a CNN to predict speaker change points.
* **Input:** Spectrogram.
* **Output:** Binary label (change or no change).

## 20. Production Monitoring

**Metrics to Track:**
* **DER:** Aggregate across all meetings. Alert if DER > 15%.
* **Latency:** P95 latency. Alert if > 2 seconds.
* **Throughput:** Meetings processed per second.
* **Error Analysis:** Which types of errors are most common? (FA, MISS, CONF)

**Dashboards:**
* **Grafana:** Real-time metrics.
* **Kibana:** Log analysis (search for "speaker change detected").

**A/B Testing:**
* Deploy new model to 5% of meetings.
* Compare DER with baseline.

## 21. Cost Analysis

**Scenario:** Diarization for 1M meetings/month (average 30 minutes each).

**Baseline (Cloud-based):**
* **Compute:** 1M meetings × 30 min = 500K hours.
* **Cost:** 500K hours × `0.10/hour (GPU) = `50K/month.

**Optimized (Batching + Caching):**
* **Batching:** Process 100 meetings in parallel. Reduces GPU hours by 10x.
* **Caching:** 40% cache hit rate. Reduces compute by 40%.
* **Cost:** 500K hours × 0.1 (batching) × 0.6 (caching) × `0.10 = `3K/month.

**Savings:** $47K/month (94% cost reduction).

## 22. Advanced Technique: Few-Shot Speaker Diarization

**Problem:** Diarization fails when speakers have very short utterances (< 1 second).

**Solution:** Use few-shot learning.

**Approach:**
1. **Meta-Learning:** Train a model on many diarization tasks.
2. **Prototypical Networks:** Learn a metric space where speakers cluster tightly.
3. **Inference:** Given a few examples of each speaker, classify new segments.

**Benefit:** Works with as few as 3 seconds of enrollment audio per speaker.

## 23. Future Trends

**1. Zero-Shot Diarization:**
* Diarize without any training data for the target domain.
* Use pre-trained models (e.g., Wav2Vec 2.0) as feature extractors.

**2. Continuous Learning:**
* Model adapts to new speakers over time.
* **Challenge:** Catastrophic forgetting.

**3. Multi-Lingual Diarization:**
* Handle meetings where speakers switch languages.
* **Challenge:** Language-specific acoustic features.

**4. Emotion-Aware Diarization:**
* Not just "who spoke" but "who spoke angrily/happily".
* **Use Case:** Call center analytics.

## 24. Benchmarking

**Dataset:** CALLHOME (2-speaker telephone conversations).

| Method | DER (%) | Latency (s) | Model Size (MB) |
|--------|---------|-------------|-----------------|
| i-vector + AHC | 12.3 | 5.0 | 50 |
| x-vector + AHC | 9.8 | 3.0 | 20 |
| x-vector + Spectral | 8.5 | 4.0 | 20 |
| EEND | 7.2 | 2.0 | 100 |
| EEND-EDA | 6.5 | 2.5 | 120 |

**Observation:** EEND achieves the best DER but requires more compute.

## 25. Conclusion

Speaker diarization is a critical component of modern speech systems. From meeting transcription to call center analytics, the ability to answer "who spoke when" unlocks powerful applications.

**Key Takeaways:**
* **Traditional Pipeline:** SAD → Segmentation → Embedding → Clustering.
* **EEND:** End-to-end neural approach. Handles overlapping speech naturally.
* **X-Vectors:** State-of-the-art embeddings for speaker recognition.
* **Clustering:** AHC with PLDA scoring is the gold standard.
* **Production:** Real-time diarization requires online clustering and caching.
* **Multi-Modal:** Combining audio and video improves accuracy.

The future of diarization is multi-modal, privacy-preserving, and adaptive. As remote work becomes the norm, the demand for accurate, real-time diarization will only grow. Mastering these techniques is essential for speech engineers.

## 26. Deep Dive: ResNet-based X-Vector Architecture

**Detailed Architecture:**
``
Input: 40-dim MFCCs (T frames)
↓
Frame-level Layers:
 - TDNN1: 512 units, context [-2, +2]
 - TDNN2: 512 units, context [-2, 0, +2]
 - TDNN3: 512 units, context [-3, 0, +3]
 - TDNN4: 512 units, context {0}
 - TDNN5: 1500 units, context {0}
↓
Statistics Pooling: [mean, std] → 3000-dim
↓
Segment-level Layers:
 - FC1: 512 units
 - FC2: 512 units (x-vector embedding)
↓
Output: 7000 units (speaker classification)
``

**Training Details:**
* **Dataset:** VoxCeleb1 + VoxCeleb2 (7000+ speakers, 1M+ utterances).
* **Augmentation:** Add noise, reverb, codec distortion.
* **Loss:** Softmax or AAM-Softmax (Additive Angular Margin).
* **Optimizer:** Adam with learning rate 0.001.
* **Batch Size:** 128 utterances.

**Inference:**
* Extract the 512-dim embedding from FC2.
* Normalize to unit length.
* Use cosine similarity for clustering.

## 27. Deep Dive: VoxCeleb Dataset

**VoxCeleb1:**
* **Speakers:** 1,251.
* **Utterances:** 153K.
* **Duration:** 352 hours.
* **Source:** YouTube celebrity interviews.

**VoxCeleb2:**
* **Speakers:** 6,112.
* **Utterances:** 1.1M.
* **Duration:** 2,442 hours.

**Challenges:**
* **In-the-Wild:** Background noise, music, laughter.
* **Multi-Speaker:** Some videos have multiple speakers.
* **Variable Length:** Utterances range from 1 second to 10 minutes.

**Preprocessing:**
* **VAD:** Remove silence using WebRTC VAD.
* **Segmentation:** Split into 2-3 second chunks.
* **Normalization:** Mean-variance normalization of MFCCs.

## 28. Production Optimization: Batch Processing

**Problem:** Processing 1M meetings sequentially takes days.

**Solution:** Batch processing on GPU.

**Algorithm:**
1. **Collect** 100 meetings.
2. **Pad** all audio to the same length (e.g., 30 minutes).
3. **Extract** MFCCs in parallel (GPU).
4. **Batch Inference:** Run x-vector model on all 100 meetings simultaneously.
5. **Clustering:** Run AHC on each meeting in parallel (CPU).

**Speedup:** 100x faster than sequential processing.

**Implementation (PyTorch):**
``python
import torch

# Batch of 100 meetings, each 30 minutes (180K frames)
mfccs = torch.randn(100, 180000, 40).cuda()

# X-vector model
model = XVectorModel().cuda()
model.eval()

# Batch inference
with torch.no_grad():
 embeddings = model(mfccs) # (100, T', 512)

# Post-process each meeting
for i in range(100):
 emb = embeddings[i] # (T', 512)
 # Run clustering
 labels = cluster(emb)
``

## 29. Advanced Evaluation: Detailed Error Analysis

**DER Breakdown:**
* **False Alarm (FA):** 2% (non-speech detected as speech).
* **Missed Speech (MISS):** 3% (speech detected as non-speech).
* **Speaker Confusion (CONF):** 5% (wrong speaker label).
* **Total DER:** 10%.

**Error Analysis:**
* **FA:** Mostly music and laughter.
 * **Fix:** Improve VAD with music detection.
* **MISS:** Mostly whispered speech.
 * **Fix:** Train VAD on whispered speech data.
* **CONF:** Mostly overlapping speech.
 * **Fix:** Use EEND to handle overlaps.

## 30. Deep Dive: Permutation Invariant Training (PIT)

**Problem:** In EEND, speaker labels are arbitrary. Ground truth might be [A, B], but prediction might be [B, A].

**Solution:** PIT finds the best permutation.

**Algorithm:**
1. **Predict:** Model outputs `P \in \mathbb{R}^{T \times K}` (K speakers).
2. **Ground Truth:** `Y \in \mathbb{R}^{T \times K}`.
3. **Enumerate Permutations:** For K=2, there are 2! = 2 permutations.
4. **Compute Loss for Each Permutation:**
 * Perm 1: `L_1 = BCE(P[:, 0], Y[:, 0]) + BCE(P[:, 1], Y[:, 1])`
 * Perm 2: `L_2 = BCE(P[:, 0], Y[:, 1]) + BCE(P[:, 1], Y[:, 0])`
5. **Choose Minimum:** `L = \min(L_1, L_2)`.

**Complexity:** O(K!) for K speakers. For K>3, use Hungarian algorithm.

## 31. Production Case Study: Call Center Diarization

**Scenario:** Analyze 10K calls/day to separate agent from customer.

**Challenges:**
* **2-Speaker:** Always agent + customer.
* **Overlapping Speech:** Frequent interruptions.
* **Background Noise:** Office noise, typing.

**Solution:**

**Step 1: Stereo Audio**
* Agent and customer are on separate channels (stereo).
* **Benefit:** No need for diarization! Just label left=agent, right=customer.

**Step 2: Mono Audio (Fallback)**
* If stereo is unavailable, use diarization.
* **Optimization:** Since K=2, use a simpler clustering algorithm (k-means with k=2).

**Step 3: Speaker Verification**
* Verify that the agent is who they claim to be (security).
* Extract x-vector from agent's speech.
* Compare with enrolled agent profile.

**Result:**
* **Latency:** 10 seconds (for a 5-minute call).
* **DER:** 5% (better than general diarization because K is known).

## 32. Advanced Technique: Self-Supervised Learning for Diarization

**Problem:** Labeled diarization data is expensive (need to manually annotate who spoke when).

**Solution:** Use self-supervised learning.

**Approach:**
1. **Pretext Task:** Train a model to predict if two segments are from the same speaker.
2. **Contrastive Learning:** Pull embeddings of the same speaker together, push different speakers apart.
3. **Fine-Tuning:** Fine-tune on a small labeled dataset.

**Example: SimCLR for Speaker Embeddings:**
``python
# Positive pair: Two segments from the same speaker
emb1 = model(segment1)
emb2 = model(segment2)

# Negative pairs: Segments from different speakers
emb3 = model(segment3)

# Contrastive loss
loss = -log(exp(sim(emb1, emb2) / tau) / (exp(sim(emb1, emb2) / tau) + exp(sim(emb1, emb3) / tau)))
``

## 33. Interview Deep Dive: Diarization vs Speaker Recognition

**Q: What's the difference between speaker diarization and speaker recognition?**

**A:**
* **Diarization:** "Who spoke when?" Unknown speakers. Clustering problem.
* **Recognition:** "Is this speaker Alice?" Known speakers. Classification problem.

**Q: Can you use the same model for both?**

**A:** Yes! X-vectors can be used for both.
* **Diarization:** Cluster x-vectors.
* **Recognition:** Compare x-vector to enrolled speaker's x-vector.

## 34. Future Trends: Transformer-based Diarization

**EEND with Conformer:**
* Replace LSTM with Conformer (Convolution + Transformer).
* **Benefit:** Better long-range dependencies.
* **Result:** DER < 5% on CALLHOME.

**Wav2Vec 2.0 for Diarization:**
* Use pre-trained Wav2Vec 2.0 as feature extractor.
* **Benefit:** No need for MFCCs. Learn features end-to-end.
* **Challenge:** Large model (300M params). Need compression for production.

## 35. Conclusion & Best Practices

**Best Practices:**
1. **Start with X-Vectors + AHC:** Proven, reliable, easy to implement.
2. **Use PLDA Scoring:** Better than cosine similarity for clustering.
3. **Handle Overlapping Speech:** Use EEND or multi-label classification.
4. **Optimize for Production:** Batch processing, caching, GPU acceleration.
5. **Monitor DER:** Track FA, MISS, CONF separately for targeted improvements.

**Diarization Checklist:**
- [ ] Implement VAD (WebRTC or DNN-based)
- [ ] Extract x-vectors (train or use pre-trained)
- [ ] Implement AHC with PLDA scoring
- [ ] Evaluate on CALLHOME (target DER < 10%)
- [ ] Handle overlapping speech (EEND or multi-label)
- [ ] Optimize for real-time (online clustering)
- [ ] Add multi-modal (video) if available
- [ ] Monitor in production (DER, latency, cost)
- [ ] Set up A/B testing
- [ ] Iterate based on error analysis

The journey from "who spoke when" to production-ready diarization involves mastering embeddings, clustering, and system design. As meetings move online and voice interfaces proliferate, diarization will become even more critical. The techniques you've learned here—from x-vectors to EEND to multi-modal fusion—will serve you well in building the next generation of speech systems.



---

**Originally published at:** [arunbaby.com/speech-tech/0041-speaker-diarization](https://www.arunbaby.com/speech-tech/0041-speaker-diarization/)

*If you found this helpful, consider sharing it with others who might benefit.*

