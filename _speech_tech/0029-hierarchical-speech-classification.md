---
title: "Hierarchical Speech Classification"
day: 29
collection: speech_tech
categories:
  - speech_tech
tags:
  - classification
  - audio
  - speech recognition
  - hierarchical
subdomain: "Speech Understanding"
tech_stack: [Kaldi, PyTorch, wav2vec2, HuBERT]
scale: "Real-time, Thousands of Classes"
companies: [Google, Amazon Alexa, Apple Siri]
related_dsa_day: 29
related_ml_day: 29
related_speech_day: 29
---

**"From broad categories to fine-grained speech understanding."**

## 1. What is Hierarchical Speech Classification?

Hierarchical speech classification organizes audio into a **taxonomy of categories**, moving from coarse to fine-grained predictions.

**Example: Voice Command Classification**
```
Intent
├── Media Control
│   ├── Music
│   │   ├── Play Song
│   │   └── Pause Song
│   └── Video
│       ├── Play Video
│       └── Stop Video
└── Smart Home
    ├── Lights
    │   ├── Turn On
    │   └── Turn Off
    └── Thermostat
```

**Problem:** Given the audio "Hey Google, turn on the bedroom lights", classify into:
- Intent: Smart Home > Lights > Turn On
- Entity: "bedroom"

## 2. Why Hierarchical for Speech?

| Challenge | Flat Classification | Hierarchical Classification |
|:---|:---|:---|
| **Acoustic Similarity** | Confuses "Play music" and "Pause music" | Groups under "Music Control" first |
| **Scalability** | 10,000 command A single model | Modular (one model per subtree) |
| **Out-of-Domain** | No fallback | Can classify to parent if uncertain about child |
| **Interpretability** | Black box | Clear decision path |

## 3. Speech Hierarchy Types

### Type 1: Speaker Recognition
```
Speech
├── Speaker 1
├── Speaker 2
└── Unknown
    ├── Male
    └── Female
        ├── Child
        └── Adult
```

### Type 2: Language Identification
```
Audio
├── English
│   ├── US
│   ├──UK
│   └── Australia
├── Spanish
│   ├── Spain
│   └── Mexico
└── Chinese
    ├── Mandarin
    └── Cantonese
```

### Type 3: Emotion Recognition
```
Emotion
├── Positive
│   ├── Happy
│   └── Excited
├── Negative
│   ├── Angry
│   └── Sad
└── Neutral
```

### Type 4: Command Classification (Voice Assistants)
```
Domain
├── Music
│   ├── Play
│   ├── Pause
│   └── Skip
├── Navigation
│   ├── Directions
│   └── Traffic
└── Communication
    ├── Call
    └── Message
```

## 4. Hierarchical Classification Approaches

### Approach 1: Global Audio Classifier

Train a **single end-to-end model** predicting all leaf categories from raw audio.

**Architecture:**
```python
class GlobalSpeechClassifier(nn.Module):
    def __init__(self, num_classes=10000):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, audio):
        features = self.wav2vec(audio).last_hidden_state
        pooled = features.mean(dim=1)  # Mean pooling
        logits = self.classifier(pooled)
        return logits
```

**Pros:**
- Simple architecture.
- End-to-end optimization.

**Cons:**
- **Class Imbalance:** "Play music" has 1M examples, "Set thermostat to 68°F" has 100.
- **No hierarchy exploitation.**

### Approach 2: Coarse-to-Fine Pipeline

**Stage 1:** Classify into broad categories (Domain).
**Stage 2:** For each domain, classify into intents.

**Example:**
```python
# Stage 1: Domain classification
domain = domain_classifier(audio)  # Music, Navigation, Communication

# Stage 2: Intent classification
if domain == "Music":
    intent = music_intent_classifier(audio)  # Play, Pause, Skip
elif domain == "Navigation":
    intent = nav_intent_classifier(audio)   # Directions, Traffic
```

**Pros:**
- Modular: Can update one stage without touching the other.
- Balanced training (each stage sees balanced data).

**Cons:**
- **Error Propagation:** If Stage 1 is wrong, Stage 2 has no chance.
- **Latency:** Two forward passes.

### Approach 3: Multi-Task Learning (MTL)

Train a **shared encoder** with **multiple output heads** (one per level).

**Architecture:**
```python
class HierarchicalSpeechMTL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Heads for each level
        self.domain_head = nn.Linear(768, 10)   # 10 domains
        self.intent_head = nn.Linear(768, 100)  # 100 intents
        self.slot_head = nn.Linear(768, 1000)   # 1000 slots
    
    def forward(self, audio):
        features = self.encoder(audio).last_hidden_state.mean(dim=1)
        
        domain_logits = self.domain_head(features)
        intent_logits = self.intent_head(features)
        slot_logits = self.slot_head(features)
        
        return {
            'domain': domain_logits,
            'intent': intent_logits,
            'slot': slot_logits
        }
```

**Loss:**
```python
loss = α * domain_loss + β * intent_loss + γ * slot_loss
```

**Pros:**
- Shared representations learn general audio features.
- Joint optimization.

**Cons:**
- Balancing loss weights (α, β, γ) is tricky.

## 5. Handling Audio-Specific Challenges

### Challenge 1: Acoustic Variability

**Problem:** "Play music" can be said in 1000 ways (different speakers, accents, noise).

**Solution: Data Augmentation**
```python
import torchaudio

def augment_audio(waveform):
    # Time stretching
    waveform = torchaudio.functional.time_stretch(waveform, rate=random.uniform(0.9, 1.1))
    
    # Pitch shift
    waveform = torchaudio.functional.pitch_shift(waveform, sample_rate=16000, n_steps=random.randint(-2, 2))
    
    # Add noise
    noise = torch.randn_like(waveform) * 0.005
    waveform = waveform + noise
    
    return waveform
```

### Challenge 2: Imbalanced Hierarchy

**Problem:** "Play music" appears 1M times, "Set thermostat to 72°F and humidity to 50%" appears 10 times.

**Solution: Hierarchical Sampling**
- Sample uniformly across **domains** first.
- Then sample uniformly within each domain.
- Ensures rare intents get seen during training.

## Deep Dive: Conformer for Hierarchical Speech

**Conformer** (Convolution + Transformer) is the SOTA architecture for speech.

**Why Conformer?**
- **Local Features:** Convolution captures phonetic details.
- **Global Context:** Self-attention captures long-range dependencies (e.g., "turn on the **bedroom** lights" - "bedroom" modifies "lights").

**Hierarchical Conformer:**
```python
class HierarchicalConformer(nn.Module):
    def __init__(self):
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock() for _ in range(12)
        ])
        
        # Insert classification heads at different depths
        self.domain_head = nn.Linear(512, 10)  # After block 4
        self.intent_head = nn.Linear(512, 100) # After block 8
        self.slot_head = nn.Linear(512, 1000)  # After block 12
    
    def forward(self, audio):
        x = audio
        outputs = {}
        
        for i, block in enumerate(self.conformer_blocks):
            x = block(x)
            
            if i == 3:  # After block 4
                outputs['domain'] = self.domain_head(x.mean(dim=1))
            if i == 7:  # After block 8
                outputs['intent'] = self.intent_head(x.mean(dim=1))
            if i == 11:  # After block 12
                outputs['slot'] = self.slot_head(x.mean(dim=1))
        
        return outputs
```

**Intuition:**
- Early layers: Broad acoustic features → Domain classification.
- Middle layers: Phonetic patterns → Intent classification.
- Deep layers: Semantic understanding → Slot filling.

## Deep Dive: Hierarchical Attention

Use **attention mechanisms** to focus on different parts of the audio for different levels.

**Example:**
- **Domain:** Attend to the first word ("Play", "Navigate", "Call").
- **Intent:** Attend to the verb + object ("Play **music**", "Play **video**").
- **Slot:** Attend to entities ("Play music by **Taylor Swift**").

**Implementation:**
```python
class HierarchicalAttention(nn.Module):
    def __init__(self):
        self.domain_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.intent_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
    
    def forward(self, features):
        # features: [seq_len, batch, 512]
        
        # Domain: Attend to first 10% of audio
        domain_context, _ = self.domain_attention(features[:10], features, features)
        domain_logits = self.domain_head(domain_context.mean(dim=0))
        
        # Intent: Attend to middle 50% of audio
        intent_context, _ = self.intent_attention(features, features, features)
        intent_logits = self.intent_head(intent_context.mean(dim=0))
        
        return domain_logits, intent_logits
```

## Deep Dive: Speaker-Aware Hierarchical Classification

**Problem:** Different users say the same command differently.

**Solution: Speaker Embeddings**
```python
class SpeakerAwareClassifier(nn.Module):
    def __init__(self):
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.speaker_encoder = SpeakerNet()  # x-vector or d-vector
        self.fusion = nn.Linear(768 + 256, 512)
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, audio):
        audio_features = self.audio_encoder(audio).last_hidden_state.mean(dim=1)
        speaker_embedding = self.speaker_encoder(audio)
        
        combined = torch.cat([audio_features, speaker_embedding], dim=1)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        return logits
```

**Benefit:** Model learns speaker-specific patterns (e.g., User A always says "Play tunes", User B says "Play music").

## Deep Dive: Hierarchical Spoken Language Understanding (SLU)

SLU combines **Intent Classification** and **Slot Filling**.

**Example:**
- Input: "Set a timer for 5 minutes"
- Intent: `SetTimer`
- Slots: `duration=5 minutes`

**Hierarchy:**
```
Root
├── Timer Intent
│   ├── Set Timer
│   ├── Cancel Timer
│   └── Check Timer
└── Alarm Intent
    ├── Set Alarm
    └── Snooze Alarm
```

**Joint Model:**
```python
class HierarchicalSLU(nn.Module):
    def __init__(self):
        self.encoder = BERTModel()  # Or Conformer for end-to-end from audio
        self.intent_classifier = nn.Linear(768, num_intents)
        self.slot_tagger = nn.Linear(768, num_slot_tags)  # BIO tagging
    
    def forward(self, tokens):
        embeddings = self.encoder(tokens)
        
        # Intent: Use [CLS] token
        intent_logits = self.intent_classifier(embeddings[:, 0, :])
        
        # Slots: Use all tokens
        slot_logits = self.slot_tagger(embeddings)
        
        return intent_logits, slot_logits
```

## Deep Dive: Hierarchical Metrics for Speech

### Metric 1: Intent Accuracy at Each Level

```python
def hierarchical_accuracy(pred_path, true_path):
    correct_at_level = []
    for i, (pred_node, true_node) in enumerate(zip(pred_path, true_path)):
        correct_at_level.append(1 if pred_node == true_node else 0)
    return correct_at_level

# Example:
# True: [Media, Music, Play]
# Pred: [Media, Video, Play]
# Accuracy: [1.0, 0.0, 0.0]  # Got Media right, but wrong afterwards
```

### Metric 2: Partial Match Score

Give credit for getting part of the hierarchy correct.
\\[
\text{Score} = \frac{\sum_{i} w_i \cdot \mathbb{1}[\text{pred}_i == \text{true}_i]}{\sum_i w_i}
\\]
where \\(w_i\\) increases with depth (deeper levels weighted more).

## Deep Dive: Google Assistant's Hierarchical Command Classification

Google processes **billions** of voice commands daily.

**Architecture:**
1. **Hotword Detection:** "Hey Google" (on-device, low power).
2. **Audio Streaming:** Send audio to cloud.
3 **ASR:** Convert audio to text (Conformer-based RNN-T).
4. **Domain Classification:** Is this Music, Navigation, SmartHome, etc.? (BERT classifier).
5. **Intent Classification:** Within domain, what's the intent? (Domain-specific BERT).
6. **Slot Filling:** Extract entities (CRF on top of BERT).
7. **Execution:** Call the appropriate API.

**Hierarchical Optimization:**
- **Domain Model:** Trained on all 1B+ queries.
- **Intent Models:** Separate model per domain, trained only on that domain's data (more focused, higher accuracy).

**Latency Budget:**
- Hotword: < 100ms
- ASR: < 500ms
- NLU (Domain + Intent + Slot): < 200ms
- **Total:** < 800ms (target)

## Deep Dive: Alexa's Hierarchical Skill Routing

Amazon Alexa has **100,000+ skills** (third-party voice apps).

**Problem:** Route the user's command to the correct skill.

**Hierarchy:**
```
Utterance
├── Built-in Skills
│   ├── Music (Amazon Music)
│   ├── Shopping (Amazon Shopping)
│   └── SmartHome (Alexa Smart Home)
└── Third-Party Skills
    ├── Category: Games
    ├── Category: News
    └── Category: Productivity
```

**Routing Algorithm:**
1. **Explicit Invocation:** "Ask **Spotify** to play music" → Route to Spotify skill.
2. **Implicit Invocation:** "Play music" → Disambiguate:
   - Check user's default music provider.
   - If ambiguous, ask: "Would you like Amazon Music or Spotify?"

3. **Hierarchical Classification:**
   - **Level 1:** Built-in vs. Third-Party.
   - **Level 2:** If Third-Party, which category?
   - **Level 3:** Within category, which skill?

## Deep Dive: Multilingual Hierarchical Speech

**Challenge:** Support 100+ languages.

**Approach 1: Per-Language Models**
- Train separate models for each language.
- **Cons:** 100 models to maintain.

**Approach 2: Multilingual Shared Encoder**
- Train a single wav2vec2 model on data from all languages.
- Add language-specific heads.
```python
class MultilingualHierarchical(nn.Module):
    def __init__(self):
        self.shared_encoder = Wav2Vec2Model()  # Trained on 100 languages
        self.language_heads = nn.ModuleDict({
            'en': nn.Linear(768, 1000),  # English intents
            'es': nn.Linear(768, 1000),  # Spanish intents
            'zh': nn.Linear(768, 1000),  # Chinese intents
        })
    
    def forward(self, audio, language):
        features = self.shared_encoder(audio).last_hidden_state.mean(dim=1)
        logits = self.language_heads[language](features)
        return logits
```

**Benefit:** Transfer learning. Low-resource languages benefit from high-resource languages.

## Deep Dive: Confidence Calibration Across Levels

**Problem:** The model predicts:
- Domain: Music (confidence = 0.99)
- Intent: Play (confidence = 0.51)

Is the overall prediction reliable?

**Solution: Hierarchical Confidence**
\\[
C_{\text{overall}} = C_{\text{domain}} \times C_{\text{intent}} \times C_{\text{slot}}
\\]

If \\(C_{\text{overall}} < 0.7\\), ask for clarification: "Did you want to play music?"

## Deep Dive: Active Learning for Rare Intents

**Problem:** "Set thermostat to 68°F and humidity level to 45%" appears only 5 times in training data.

**Solution: Active Learning**
1. Deploy model.
2. Log all predictions with \\(C_{\text{overall}} < 0.5\\) (uncertain).
3. Human reviews and labels these uncertain examples.
4. Retrain model with new labels.

**Hierarchical Active Learning:**
- Prioritize examples where the model is uncertain at **multiple levels**.
- Example: Uncertain about both Domain and Intent → High priority for labeling.

## Deep Dive: Temporal Hierarchies (Sequential Commands)

**Problem:** "Play Taylor Swift, then set a timer for 5 minutes."

**Two Intents in One Utterance:**
1. Play Music (artist = Taylor Swift)
2. Set Timer (duration = 5 minutes)

**Approach: Segmentation + Per-Segment Classification**
```python
# Step 1: Segment audio
segments = segment_audio(audio)  # ["Play Taylor Swift", "set a timer for 5 minutes"]

# Step 2: Classify each segment
for segment in segments:
    intent, slots = hierarchical_classifier(segment)
    execute(intent, slots)
```

**Segmentation Techniques:**
- **Pause Detection:** Split on silences > 500ms.
- **Semantic Segmentation:** Use a sequence tagging model to predict segment boundaries.

## Deep Dive: Hierarchical Few-Shot Learning

**Problem:** A new intent "Book a table at a restaurant" is added. We have only 10 labeled examples.

**Solution: Prototypical Networks**
```python
def prototypical_network(support_set, query):
    # support_set: [(audio_1, label_1), (audio_2, label_2), ...]
    # Compute prototype for each class
    prototypes = {}
    for audio, label in support_set:
        features = encoder(audio)
        if label not in prototypes:
            prototypes[label] = []
        prototypes[label].append(features)
    
    for label in prototypes:
        prototypes[label] = torch.stack(prototypes[label]).mean(dim=0)
    
    # Classify query by nearest prototype
    query_features = encoder(query)
    distances = {label: cosine_distance(query_features, proto) for label, proto in prototypes.items()}
    predicted_label = min(distances, key=distances.get)
    return predicted_label
```

**Benefit:** Can add new intents with < 10 examples.

## Deep Dive: Noise Robustness in Hierarchical Speech

**Problem:** Background noise (TV, traffic) degrades classification.

**Solution: Multi-Condition Training (MCT)**
```python
def add_noise(clean_audio, noise_audio, snr_db):
    # Signal-to-Noise Ratio
    noise_power = clean_audio.norm() / ( 10 ** (snr_db / 20))
    scaled_noise = noise_audio * noise_power / noise_audio.norm()
    noisy_audio = clean_audio + scaled_noise
    return noisy_audio

# Training
for audio, label in dataset:
    noise = random.choice(noise_dataset)  # TV, traffic, babble
    snr = random.uniform(-5, 20)  # dB
    noisy_audio = add_noise(audio, noise, snr)
    loss = criterion(model(noisy_audio), label)
```

**Advanced: Denoising Front-End**
- Use a **speech enhancement** model before the classifier.
- Example: Conv-TasNet, Sudormian.

## Implementation: Full Hierarchical Speech Pipeline

```python
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class HierarchicalSpeechClassifier(nn.Module):
    def __init__(self, domain_classes=10, intent_classes=100):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Domain classifier (coarse)
        self.domain_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, domain_classes)
        )
        
        # Intent classifier (fine)
        self.intent_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, intent_classes)
        )
    
    def forward(self, audio, return_features=False):
        # audio: [batch, waveform]
        features = self.wav2vec(audio).last_hidden_state  # [batch, time, 768]
        pooled = features.mean(dim=1)  # [batch, 768]
        
        domain_logits = self.domain_head(pooled)
        intent_logits = self.intent_head(pooled)
        
        if return_features:
            return domain_logits, intent_logits, pooled
        return domain_logits, intent_logits

def hierarchical_loss(domain_logits, intent_logits, domain_target, intent_target, alpha=0.3):
    domain_loss = nn.CrossEntropyLoss()(domain_logits, domain_target)
    intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_target)
    return alpha * domain_loss + (1 - alpha) * intent_loss

# Training loop
model = HierarchicalSpeechClassifier()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(10):
    for audio, domain_label, intent_label in dataloader:
        optimizer.zero_grad()
        domain_logits, intent_logits = model(audio)
        loss = hierarchical_loss(domain_logits, intent_logits, domain_label, intent_label)
        loss.backward()
        optimizer.step()
```

## Top Interview Questions

**Q1: How do you handle code-switching (mixing languages) in hierarchical speech classification?**
*Answer:*
Use a **multilingual encoder** (e.g., wav2vec2 fine-tuned on mixed-language data). Add a **language identification head** to detect which language(s) are spoken, then route to the appropriate intent classifier.

**Q2: What if the hierarchy changes frequently (new intents added)?**
*Answer:*
Use **modular design**: separate models for each level. When a new intent is added, only retrain the intent-level model. Alternatively, use **label embeddings** (encode intent names as text) so new intents can be added without retraining.

**Q3: How do you ensure low latency for real-time voice assistants?**
*Answer:*
- **Streaming Models:** Use RNN-T or streaming Conformer that outputs predictions as audio arrives.
- **Early Exit:** If the domain classifier is very confident, skip deeper layers.
- **Edge Deployment:** Run lightweight models on-device (quantized, pruned).

**Q4: How do you evaluate hierarchical speech models?**
*Answer:*
- **Accuracy at Each Level:** Report domain accuracy, intent accuracy separately.
- **Partial Match Score:** Give credit for getting higher levels correct even if lower levels are wrong.
- **Confusion Matrices:** Per-level confusion matrices to identify systematic errors.

## Key Takeaways

1. **Hierarchy Reduces Confusion:** Grouping similar commands improves accuracy.
2. **Multi-Task Learning:** Shared encoder exploits commonalities across levels.
3. **Modular Design:** Easier to update individual levels without retraining everything.
4. **Attention Mechanisms:** Focus on different audio segments for different levels.
5. **Evaluation:** Use hierarchical metrics (accuracy per level, partial match).

## Summary

| Aspect | Insight |
|:---|:---|
| **Approaches** | Global, Coarse-to-Fine Pipeline, Multi-Task Learning |
| **Architecture** | Conformer (Convolution + Transformer) is SOTA |
| **Challenges** | Acoustic variability, imbalanced data, multilingual |
| **Real-World** | Google Assistant, Alexa use hierarchical routing |

---

**Originally published at:** [arunbaby.com/speech-tech/0029-hierarchical-speech-classification](https://www.arunbaby.com/speech-tech/0029-hierarchical-speech-classification/)

*If you found this helpful, consider sharing it with others who might benefit.*


