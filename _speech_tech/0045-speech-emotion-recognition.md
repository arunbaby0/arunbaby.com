---
title: "Speech Emotion Recognition"
day: 45
collection: speech_tech
categories:
  - speech-tech
tags:
  - emotion
  - classification
  - deep-learning
  - multimodal
difficulty: Hard
---

**"Teaching machines to hear feelings."**

## 1. Introduction

**Speech Emotion Recognition (SER)** is the task of identifying the emotional state of a speaker from their voice.

**Emotions Typically Recognized:**
*   **Basic:** Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral.
*   **Dimensional:** Valence (positive/negative), Arousal (activation level), Dominance.

**Applications:**
*   **Customer Service:** Detect frustrated callers, route to specialists.
*   **Mental Health:** Monitor emotional state over time.
*   **Human-Robot Interaction:** Empathetic responses.
*   **Gaming:** Adaptive game difficulty based on player emotion.
*   **Automotive:** Detect driver stress or drowsiness.

## 2. Challenges in SER

**1. Subjectivity:**
*   Same utterance can be perceived differently.
*   Cultural differences in emotional expression.

**2. Speaker Variability:**
*   Emotional expression varies by person.
*   Age, gender, and language effects.

**3. Context Dependency:**
*   "Really?" can be surprised, sarcastic, or angry.
*   Need context to disambiguate.

**4. Data Scarcity:**
*   Labeled emotional speech is expensive to collect.
*   Acted vs spontaneous speech differs.

**5. Class Imbalance:**
*   Neutral is often dominant.
*   Extreme emotions (rage, despair) are rare.

## 3. Acoustic Features for SER

### 3.1. Prosodic Features

**Pitch (F0):**
*   Higher pitch → excitement, anger.
*   Lower pitch → sadness, boredom.

**Energy:**
*   Higher energy → anger, happiness.
*   Lower energy → sadness.

**Speaking Rate:**
*   Faster → excitement, nervousness.
*   Slower → sadness, hesitation.

### 3.2. Spectral Features

**MFCCs:**
*   Standard speech features.
*   13-40 coefficients + deltas.

**Mel Spectrogram:**
*   Raw input for CNNs.
*   Captures timbral qualities.

**Formants:**
*   Vowel quality changes with emotion.

### 3.3. Voice Quality Features

**Jitter and Shimmer:**
*   Irregularities in pitch and amplitude.
*   Higher in stressed/emotional speech.

**Harmonic-to-Noise Ratio (HNR):**
*   Clarity of voice.
*   Lower in breathy or tense speech.

## 4. Traditional ML Approaches

### 4.1. Feature Extraction + Classifier

**Pipeline:**
1.  Extract hand-crafted features (openSMILE).
2.  Train SVM, Random Forest, or GMM.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Extract features (using openSMILE or librosa)
X_train = extract_features(train_audio)
X_test = extract_features(test_audio)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
clf = SVC(kernel='rbf', C=1.0)
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
```

### 4.2. openSMILE Features

**openSMILE** extracts thousands of features:
*   eGeMAPS: 88 features (standardized for emotion).
*   ComParE: 6373 features (comprehensive).

```python
import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

features = smile.process_file('audio.wav')
```

## 5. Deep Learning Approaches

### 5.1. CNN on Spectrograms

**Architecture:**
1.  Convert audio to mel spectrogram.
2.  Treat as image, apply 2D CNN.
3.  Global pooling + dense layers.

```python
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

### 5.2. LSTM/GRU on Sequences

**Architecture:**
1.  Extract frame-level features (MFCCs).
2.  Feed to bidirectional LSTM.
3.  Attention or pooling over time.

```python
class EmotionLSTM(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_classes=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, time, features)
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.fc(context)
```

### 5.3. Transformer-Based Models

**Using Pretrained Models:**
*   **Wav2Vec 2.0:** Self-supervised audio representations.
*   **HuBERT:** Hidden unit BERT for speech.
*   **WavLM:** Microsoft's large speech model.

```python
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class EmotionWav2Vec(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        hidden = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(hidden)
```

## 6. Datasets

### 6.1. IEMOCAP

*   12 hours of audiovisual data.
*   5 sessions, 10 actors.
*   Emotions: Angry, Happy, Sad, Neutral, Excited, Frustrated.
*   Gold standard for SER research.

### 6.2. RAVDESS

*   24 actors (12 male, 12 female).
*   7 emotions + calm.
*   Acted speech and song.

### 6.3. CREMA-D

*   7,442 clips from 91 actors.
*   6 emotions.
*   Diverse ethnic backgrounds.

### 6.4. CMU-MOSEI

*   23,453 video clips.
*   Multimodal: text, audio, video.
*   Sentiment and emotion labels.

### 6.5. EmoDB (German)

*   535 utterances.
*   10 actors, 7 emotions.
*   Classic dataset for SER.

## 7. Evaluation Metrics

**Classification Metrics:**
*   **Accuracy:** Overall correct predictions.
*   **Weighted F1:** Accounts for class imbalance.
*   **Unweighted Accuracy (UA):** Average recall across classes.
*   **Confusion Matrix:** Understand per-class performance.

**For Dimensional Emotions:**
*   **CCC (Concordance Correlation Coefficient):** Agreement measure.
*   **MSE/MAE:** For valence/arousal prediction.

## 8. System Design: Call Center Emotion Analytics

**Scenario:** Detect customer emotions during support calls.

**Requirements:**
*   Real-time analysis.
*   Handle noisy telephony audio.
*   Alert supervisors on negative emotions.

**Architecture:**
```
┌─────────────────┐
│   Phone Call    │
│   (Audio Stream)│
└────────┬────────┘
         │
┌────────▼────────┐
│  Voice Activity │
│    Detection    │
└────────┬────────┘
         │
┌────────▼────────┐
│  Speaker        │
│  Diarization    │
└────────┬────────┘
         │
┌────────▼────────┐
│  Emotion        │
│  Recognition    │
└────────┬────────┘
         │
         ├──────────────┐
         │              │
┌────────▼────────┐    ┌▼────────────────┐
│   Dashboard     │    │  Alert System   │
│   (Real-time)   │    │  (Supervisor)   │
└─────────────────┘    └─────────────────┘
```

**Implementation Details:**
*   Process in 3-second windows.
*   Apply noise reduction first.
*   Track emotion trajectory over call.
*   Trigger alert if anger/frustration persists.

## 9. Multimodal Emotion Recognition

**Combine modalities for better accuracy:**
*   **Audio:** Voice, prosody.
*   **Text:** Transcribed words, sentiment.
*   **Video:** Facial expressions, body language.

### 9.1. Early Fusion

**Concatenate features before classification:**
```python
audio_features = audio_encoder(audio)
text_features = text_encoder(text)
combined = torch.cat([audio_features, text_features], dim=1)
output = classifier(combined)
```

### 9.2. Late Fusion

**Combine predictions from each modality:**
```python
audio_pred = audio_model(audio)
text_pred = text_model(text)
combined_pred = (audio_pred + text_pred) / 2
```

### 9.3. Cross-Modal Attention

**Let modalities attend to each other:**
```python
class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
    
    def forward(self, x1, x2):
        # x1 attends to x2
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.size(-1)), dim=-1)
        return torch.bmm(attn, v)
```

## 10. Real-Time Considerations

**Latency Requirements:**
*   Call center: <500ms per segment.
*   Gaming: <100ms for responsiveness.

**Optimization Strategies:**
1.  **Streaming:** Process overlapping windows.
2.  **Model Pruning:** Reduce model size.
3.  **Quantization:** INT8 inference.
4.  **GPU Batching:** Process multiple calls together.

## 11. Interview Questions

1.  **Features for SER:** What acoustic features capture emotion?
2.  **IEMOCAP:** Describe the dataset and common practices.
3.  **Class Imbalance:** How do you handle it in SER?
4.  **Multimodal Fusion:** Early vs late vs attention fusion?
5.  **Real-Time Design:** Design an emotion detector for virtual meetings.

## 12. Common Mistakes

*   **Ignoring Speaker Effects:** Train with speaker-independent splits.
*   **Leaking Speakers:** Same speaker in train and test.
*   **Wrong Metrics:** Use weighted/unweighted accuracy for imbalanced data.
*   **Acted vs Spontaneous:** Models trained on acted data fail on real speech.
*   **Ignoring Context:** Sentence-level emotion misses conversational dynamics.

## 13. Future Trends

**1. Self-Supervised Pretraining:**
*   Wav2Vec, HuBERT for emotion.
*   Less labeled data needed.

**2. Personalized Emotion Recognition:**
*   Adapt to individual expression patterns.
*   Few-shot learning.

**3. Continuous Emotion Tracking:**
*   Not discrete labels, but continuous trajectories.
*   Valence-arousal-dominance space.

**4. Explainable SER:**
*   Which parts of audio indicate emotion.
*   Attention visualization.

## 14. Conclusion

Speech Emotion Recognition is a challenging but impactful task. It requires understanding of both speech processing and machine learning.

**Key Takeaways:**
*   **Features:** Prosody, spectral, voice quality.
*   **Models:** CNN on spectrograms, LSTM on sequences, Transformers.
*   **Data:** IEMOCAP is the gold standard.
*   **Evaluation:** Weighted F1 for imbalanced classes.
*   **Multimodal:** Combining audio + text improves accuracy.

As AI becomes more empathetic, SER will be central to human-computer interaction. Master it to build systems that truly understand their users.

## 15. Training Pipeline

### 15.1. Data Preprocessing

```python
import librosa
import numpy as np

def preprocess_audio(audio_path, target_sr=16000, max_duration=10):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Pad or truncate
    max_samples = target_sr * max_duration
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    else:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    
    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=target_sr, n_mels=80, hop_length=160
    )
    log_mel = np.log(mel + 1e-8)
    
    return log_mel
```

### 15.2. Data Loading

```python
from torch.utils.data import Dataset, DataLoader

class EmotionDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        mel = preprocess_audio(self.audio_paths[idx])
        label = self.labels[idx]
        return torch.tensor(mel).unsqueeze(0), label

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
```

### 15.3. Training Loop

```python
model = EmotionCNN(num_classes=7)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(100):
    model.train()
    for mel, labels in train_loader:
        outputs = model(mel)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for mel, labels in val_loader:
            outputs = model(mel)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch}, Val Accuracy: {100 * correct / total:.2f}%")
```

## 16. Data Augmentation

**Audio Augmentations:**
```python
import audiomentations as A

augment = A.Compose([
    A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    A.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    A.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    A.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

def augment_audio(audio, sr):
    return augment(samples=audio, sample_rate=sr)
```

**SpecAugment:**
```python
def spec_augment(mel, freq_mask=10, time_mask=20):
    # Frequency masking
    f0 = np.random.randint(0, mel.shape[0] - freq_mask)
    mel[f0:f0+freq_mask, :] = 0
    
    # Time masking
    t0 = np.random.randint(0, mel.shape[1] - time_mask)
    mel[:, t0:t0+time_mask] = 0
    
    return mel
```

## 17. Handling Class Imbalance

**Strategies:**
1.  **Weighted Loss:**
```python
class_weights = compute_class_weights(labels)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

2.  **Oversampling:**
```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X, y)
```

3.  **Focal Loss:**
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

## 18. Dimensional Emotion Recognition

**Valence-Arousal-Dominance (VAD) Model:**
*   **Valence:** Positive (happy) to Negative (sad).
*   **Arousal:** Active (excited) to Passive (calm).
*   **Dominance:** Dominant to Submissive.

**Regression Instead of Classification:**
```python
class EmotionVADRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.regressor = nn.Linear(768, 3)  # Predict V, A, D
    
    def forward(self, x):
        features = self.encoder(x).last_hidden_state.mean(dim=1)
        return self.regressor(features)

# Training with MSE loss
criterion = nn.MSELoss()
output = model(audio)
loss = criterion(output, torch.tensor([valence, arousal, dominance]))
```

**Evaluation Metric (CCC):**
```python
def concordance_correlation_coefficient(pred, target):
    mean_pred = pred.mean()
    mean_target = target.mean()
    var_pred = pred.var()
    var_target = target.var()
    covar = ((pred - mean_pred) * (target - mean_target)).mean()
    
    ccc = 2 * covar / (var_pred + var_target + (mean_pred - mean_target)**2)
    return ccc
```

## 19. Production Deployment

### 19.1. Model Export

```python
# Export to ONNX
dummy_input = torch.randn(1, 1, 80, 400)
torch.onnx.export(model, dummy_input, "emotion_model.onnx")

# Or TorchScript
scripted = torch.jit.script(model)
scripted.save("emotion_model.pt")
```

### 19.2. Inference Service

```python
from fastapi import FastAPI, UploadFile
import soundfile as sf

app = FastAPI()

@app.post("/predict")
async def predict_emotion(file: UploadFile):
    # Read audio
    audio, sr = sf.read(file.file)
    
    # Preprocess
    mel = preprocess_audio_from_array(audio, sr)
    
    # Predict
    with torch.no_grad():
        output = model(torch.tensor(mel).unsqueeze(0))
        emotion_idx = output.argmax().item()
    
    emotions = ["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"]
    return {"emotion": emotions[emotion_idx]}
```

### 19.3. Streaming Processing

```python
class StreamingEmotionDetector:
    def __init__(self, model, window_size=3.0, hop_size=1.0, sr=16000):
        self.model = model
        self.window_samples = int(window_size * sr)
        self.hop_samples = int(hop_size * sr)
        self.buffer = []
    
    def process_chunk(self, audio_chunk):
        self.buffer.extend(audio_chunk)
        
        results = []
        while len(self.buffer) >= self.window_samples:
            window = self.buffer[:self.window_samples]
            emotion = self.predict(window)
            results.append(emotion)
            self.buffer = self.buffer[self.hop_samples:]
        
        return results
    
    def predict(self, audio):
        mel = compute_mel(audio)
        with torch.no_grad():
            output = self.model(torch.tensor(mel).unsqueeze(0))
        return output.argmax().item()
```

## 20. Mastery Checklist

**Mastery Checklist:**
- [ ] Extract prosodic features (F0, energy)
- [ ] Extract spectral features (MFCC, mel spectrogram)
- [ ] Train CNN on spectrograms
- [ ] Train LSTM with attention
- [ ] Fine-tune Wav2Vec2 for emotion
- [ ] Handle class imbalance (weighted loss, oversampling)
- [ ] Implement multimodal fusion
- [ ] Evaluate with weighted F1 and UA
- [ ] Deploy real-time emotion detector
- [ ] Understand dimensional emotion models

## 21. Conclusion

Speech Emotion Recognition bridges the gap between AI and human emotional intelligence. It's a challenging task that requires:
*   **Domain Knowledge:** Understanding how emotions manifest in speech.
*   **ML Expertise:** Selecting and training appropriate models.
*   **Data Engineering:** Handling imbalanced, subjective labels.
*   **System Design:** Building real-time, production-ready systems.

**The Path Forward:**
1.  Start with IEMOCAP and a CNN baseline.
2.  Upgrade to Wav2Vec2 for better features.
3.  Add multimodal (text) for improved accuracy.
4.  Deploy with streaming for real-time applications.

As AI assistants become more prevalent, emotional intelligence will be a key differentiator. Systems that understand and respond to human emotions will create more natural, empathetic interactions. Master SER to be at the forefront of this revolution.

