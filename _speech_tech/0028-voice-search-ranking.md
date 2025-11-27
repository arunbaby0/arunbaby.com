---
title: "Voice Search Ranking"
day: 28
collection: speech_tech
categories:
  - speech_tech
tags:
  - voice_search
  - asr
  - ranking
  - nlu
subdomain: "Speech Applications"
tech_stack: [Kaldi, BERT, Lucene]
scale: "Real-time, High Ambiguity"
companies: [Google Assistant, Alexa, Siri]
related_dsa_day: 28
related_ml_day: 28
related_speech_day: 28
---

**"Play Call Me Maybe". Did you mean the song, the video, or the contact named 'Maybe'?**

## 1. The Unique Challenge of Voice Search

Voice Search is harder than Text Search for three reasons:
1.  **ASR Errors:** The user said "Ice cream", but the ASR heard "I scream".
2.  **Ambiguity:** "Play Frozen" could mean the movie, the soundtrack, or a playlist.
3.  **No UI:** In a smart speaker, you can't show 10 results. You must pick **The One** (Top-1 Accuracy is critical).

## 2. High-Level Architecture

```
[Audio]
   |
   v
[ASR Engine] -> Generates N-Best List
   |            (1. "Play Call Me Maybe", 0.9)
   |            (2. "Play Call Me Baby", 0.4)
   v
[Query Understanding (NLU)] -> Intent Classification & Slot Filling
   |
   v
[Federated Search] -> Search Music, Contacts, Videos, Web
   |
   v
[Cross-Domain Ranking] -> Selects the best domain (Music vs. Video)
   |
   v
[Response Generation] -> TTS ("Playing Call Me Maybe on Spotify")
```

## 3. ASR N-Best Lists and Lattice Rescoring

We don't just take the top ASR hypothesis. We take the **N-Best List** (e.g., top 10).
We re-score these hypotheses using a **Personalized Language Model**.

**Example:**
-   User says: "Call [Name]"
-   ASR Top 1: "Call Al" (Generic LM probability is high).
-   ASR Top 2: "Call Hal" (User has a contact named 'Hal').
-   **Rescoring:** Boost "Call Hal" because 'Hal' is in the user's contact list (Personalization).

**Lattice Rescoring:**
Instead of a list, we can use the full **Word Lattice** (a graph of all possible paths).
We can search the lattice for entities (Contacts, Song Titles) using Finite State Transducers (FSTs).

## 4. Spoken Language Understanding (SLU)

Once we have the text, we need **Intent** and **Slots**.

**Query:** "Play Taylor Swift"
-   **Intent:** `PlayMusic`
-   **Slot:** `Artist = "Taylor Swift"`

**Model:** BERT-based Joint Intent/Slot model.
**Challenge:** Spoken language is messy. "Umm, play that song by, you know, Taylor."
**Solution:** Train on noisy, disfluent text.

## 5. Federated Search & Domain Ranking

Voice Assistants connect to multiple backends (Domains).
-   **Music:** Spotify, Apple Music.
-   **Video:** YouTube, Netflix.
-   **Knowledge:** Wikipedia.
-   **IoT:** Smart Lights.

**The Domain Ranking Problem:**
User says: "Frozen".
-   Music Domain Confidence: 0.8 (Soundtrack).
-   Video Domain Confidence: 0.9 (Movie).
-   **Winner:** Video.

**Calibration:**
Confidence scores from different domains are not comparable.
We train a **Domain Selector Model** (Classifier) that takes features from all domains and predicts the probability of user satisfaction.

## 6. Personalization Signals

Personalization is the strongest signal in Voice Search.
1.  **App Usage:** If the user uses Spotify 90% of the time, "Play X" implies Spotify.
2.  **Location:** "Where is the nearest Starbucks?" depends entirely on GPS.
3.  **Device State:** "Turn on the lights" implies the lights in the *same room* as the speaker.

## 7. Evaluation Metrics

-   **WER (Word Error Rate):** ASR metric.
-   **SemER (Semantic Error Rate):** Did we get the Intent/Slots right? (Even if ASR was wrong).
-   **Task Completion Rate:** Did the music actually start playing?
-   **Latency:** Voice users are impatient. Total budget < 1 second.

## Deep Dive: Weighted Finite State Transducers (WFST)

Traditional ASR decoding uses the **HCLG** graph.
\[ H \circ C \circ L \circ G \]
-   **H (HMM):** Acoustic states -> Context-dependent phones.
-   **C (Context):** Context-dependent phones -> Monophones.
-   **L (Lexicon):** Monophones -> Words.
-   **G (Grammar):** Words -> Sentences (N-gram LM).

**Lattice Generation:**
The decoder outputs a **Lattice** (a compact representation of many hypotheses).
-   Nodes: Time points.
-   Arcs: Words with acoustic and LM scores.
-   **Rescoring:** We can take this lattice and intersect it with a larger, more complex LM (e.g., a Neural LM) to find a better path.

## Deep Dive: Contextual Biasing (Class-Based LMs)

How do we recognize "Call **Arun**" if "Arun" is a rare name?
We use **Contextual Biasing**.

**Mechanism:**
1.  **Class Tagging:** In the LM, we replace names with a class tag: `Call @CONTACT_NAME`.
2.  **Runtime Injection:** When the user speaks, we fetch their contact list `["Arun", "Bob", "Charlie"]`.
3.  **FST Composition:** We dynamically build a small FST for the contact list and compose it with the main graph at the `@CONTACT_NAME` node.

**Result:**
The model effectively has a dynamic vocabulary that changes per user, per query.
This is critical for:
-   Contacts ("Call X")
-   App Names ("Open Y")
-   Song Titles ("Play Z")



## Deep Dive: Hotword Detection (Wake Word) in Depth

Before any ranking happens, the device must wake up.
**Constraint:** Must run on a DSP with < 100KB RAM and < 1mW power.

**Architectures:**
1.  **DNN:** Simple fully connected layers. (Old).
2.  **CNN:** ResNet-15 or TC-ResNet. Good accuracy, but computationally heavy.
3.  **DS-CNN (Depthwise Separable CNN):** Separates spatial and channel convolutions. 8x smaller and faster. Standard for mobile.

**Metrics:**
-   **False Reject Rate (FRR):** User says "Hey Google" and nothing happens. (Frustrating).
-   **False Accept Rate (FAR):** TV says "Hey Poodle" and device wakes up. (Creepy).
-   **Trade-off:** We tune the threshold to minimize FAR (0.1 per hour) while keeping FRR reasonable (< 5%).

## Deep Dive: End-to-End ASR Architectures

Ranking depends on the ASR N-best list. How is it generated?

### 1. Listen-Attend-Spell (LAS)
-   **Encoder:** Pyramidal LSTM.
-   **Decoder:** Attention-based LSTM.
-   **Pros:** High accuracy.
-   **Cons:** Not streaming. Must wait for end of utterance to start decoding.

### 2. RNN-T (Recurrent Neural Network Transducer)
-   **Encoder:** Audio -> Features.
-   **Prediction Network:** Text -> Features (Language Model).
-   **Joint Network:** Combines them.
-   **Pros:** Streaming. Low latency.
-   **Cons:** Hard to train (huge memory).

### 3. Conformer (Convolution + Transformer)
-   Combines local convolution (for fine-grained audio details) with global self-attention (for long-range context).
-   **Status:** Current SOTA for streaming ASR.

## Deep Dive: Language Model Fusion Strategies

How do we combine the ASR model (AM) with the external Language Model (LM)?

1.  **Shallow Fusion:**
    -   Linearly interpolate scores at inference time.
    -   Simple, flexible. Can swap LMs easily.
2.  **Deep Fusion:**
    -   Concatenate hidden states of AM and LM and feed to a gating network.
    -   Requires retraining the AM-LM interface.
3.  **Cold Fusion:**
    -   Train the AM *conditioned* on the LM.
    -   The AM learns to rely on the LM for difficult words.

## Deep Dive: Speaker Diarization

"Who spoke when?"
**Pipeline:**
1.  **Segmentation:** Split audio into homogeneous segments.
2.  **Embedding:** Extract d-vector for each segment.
3.  **Clustering:**
    -   **K-Means:** If we know the number of speakers (N=2).
    -   **Spectral Clustering:** If N is unknown.
    -   **UIS-RNN:** Unbounded Interleaved-State RNN. Fully supervised online clustering.

## Deep Dive: Voice Activity Detection (VAD)

Ranking needs to know when the user *stopped* speaking to execute the query.
**Approaches:**
1.  **Energy-Based:** If volume < Threshold for 500ms -> End. (Fails in noisy rooms).
2.  **Model-Based:** Small GMM or DNN trained on Speech vs Noise.
3.  **Semantic VAD:** "Play music" (Complete). "Play..." (Incomplete). Wait longer if incomplete.

## Deep Dive: Text-to-Speech (TTS) for Response

The ranking system chooses the *text* response. The TTS system generates the *audio*.
**Ranking Influence:**
-   If the ranker is unsure ("Did you mean A or B?"), the TTS should sound **inquisitive** (rising pitch).
-   If the ranker is confident, the TTS should sound **affirmative**.

**TTS Architecture:**
1.  **Text Analysis:** Normalize text, predict prosody.
2.  **Acoustic Model (Tacotron 2):** Text -> Mel Spectrogram.
3.  **Vocoder (WaveNet / HiFi-GAN):** Mel Spectrogram -> Waveform.
    -   **WaveNet:** Autoregressive, slow.
    -   **HiFi-GAN:** GAN-based, real-time, high fidelity.

## Deep Dive: Inverse Text Normalization (ITN)

ASR outputs: "play song number two".
The Music API expects: `song_id: 2`.
ITN converts spoken form to written form.

**Rules:**
-   "twenty twenty four" -> "2024" (Year) or "2024" (Number). Context matters.
-   "five dollars" -> "$5".
-   "doctor smith" -> "Dr. Smith".

**Technology:**
-   **FSTs (Finite State Transducers):** Hand-written grammars (Kestrel).
-   **Neural ITN:** Seq2Seq models that translate "spoken" to "written".

## Deep Dive: Confidence Calibration

The ASR model outputs a probability. "I am 90% sure this is 'Call Mom'".
But is it *really* 90% sure?
**Calibration Error:** When the model says 90%, it should be right 90% of the time.
Deep Neural Networks are notoriously **Overconfident**.

**Temperature Scaling:**
\[ P = \text{softmax}(logits / T) \]
-   If \(T > 1\), the distribution flattens (less confident).
-   We tune \(T\) on a validation set to minimize ECE (Expected Calibration Error).
-   **Why it matters:** If confidence is low, we should ask the user to repeat ("Sorry, I didn't catch that"). If we are overconfident, we execute the wrong command.

## Deep Dive: Voice Search UX Guidelines

Ranking for Voice is different from Web.
**The "Eyes-Free" Constraint.**

1.  **Brevity:** Don't read a Wikipedia article. Read the summary.
2.  **Confirmation:**
    -   High Confidence: Just do it. ("Turning on lights").
    -   Medium Confidence: Implicit confirm. ("OK, playing Taylor Swift").
    -   Low Confidence: Explicit confirm. ("Did you say Taylor Swift?").
3.  **Disambiguation:**
    -   Don't list 10 options. List 2 or 3.
    -   "I found a few contacts. Did you mean Bob Smith or Bob Jones?"

## Deep Dive: End-to-End SLU (Audio -> Intent)

Why have ASR -> Text -> NLU?
Errors cascade.
**E2E SLU:**
-   Input: Audio.
-   Output: Intent/Slots directly.
-   **Pros:** The model can use prosody (tone of voice). "Yeah right" (Sarcasm) -> Negative Sentiment. Text-only models miss this.
-   **Cons:** Requires massive Audio-Intent labeled data, which is rare.

## Deep Dive: Zero-Shot Entity Recognition

How do we handle "Play [New Song that came out 5 minutes ago]"?
The ASR model hasn't seen it. The NLU model hasn't seen it.

**Solution: Copy Mechanisms (Pointer Networks).**
-   The NLU model can decide to "Copy" a span of text from the ASR output into the `Song` slot, even if it doesn't recognize the entity.
-   We rely on the **Knowledge Graph** (KG) to validate it.
-   **KG Lookup:** Fuzzy match the copied span against the daily updated KG index.

## Deep Dive: Multilingual Voice Search

"Play Despacito" (Spanish song, English user).
**Code Switching** is a nightmare for ASR.

**Approaches:**
1.  **LID (Language ID):** Run a classifier first. "Is this English or Spanish?".
    -   **Problem:** Latency. And "Despacito" is a Spanish word in an English sentence.
2.  **Bilingual Models:** Train one model on English + Spanish data.
    -   **Problem:** The phone sets might conflict.
3.  **Transliteration:**
    -   Map Spanish phonemes to English approximations.
    -   User says "Des-pa-see-to".
    -   ASR outputs "Despacito".

## Deep Dive: Privacy and Federated Learning

Voice data is sensitive. Users don't want their bedroom conversations sent to the cloud.

**Federated Learning:**
1.  **Local Training:** The wake word model ("Hey Siri") runs locally.
2.  **Gradient Updates:** If the model fails (False Reject), the user manually activates it.
3.  **Aggregation:** The phone computes the gradient update and sends *only the gradient* (encrypted) to the cloud.
4.  **Global Model:** The cloud aggregates gradients from millions of phones and updates the global model.
5.  **No Audio Upload:** The raw audio never leaves the device.

## Deep Dive: Federated Learning for Hotword

We want to improve "Hey Google" detection without uploading false accepts (privacy).
**Protocol:**
1.  **Local Cache:** Device stores the last 10 "Hey Google" triggers that the user *cancelled* (False Accepts).
2.  **Training:** When charging + on WiFi, the device fine-tunes the Hotword model on these negative examples.
3.  **Aggregation:** Device sends weight updates to the cloud.
4.  **Result:** The global model learns that "Hey Poodle" is NOT a wake word, without ever hearing the user's voice in the cloud.

## Deep Dive: Audio Codecs (Opus vs Lyra)

Streaming raw audio (PCM) is heavy (16kHz * 16bit = 256kbps).
We need compression.
1.  **Opus:** Standard for VoIP. Good quality at 24kbps.
2.  **Lyra (Google):** Neural Audio Codec.
    -   Uses a Generative Model (WaveRNN) to reconstruct speech.
    -   **Bitrate:** 3kbps (Very low bandwidth).
    -   **Quality:** Comparable to Opus at 3kbps.
    -   **Use Case:** Voice Search on 2G networks.

## Deep Dive: Microphone Arrays & Beamforming

How do we hear a user across the room?
**Hardware:** 2-7 microphones (Linear on TV, Circular on Speaker).
**Math (Delay-and-Sum Beamforming):**
-   Sound arrives at Mic 1 at time \(t\).
-   Sound arrives at Mic 2 at time \(t + \Delta t\).
-   We shift Mic 2's signal by \(-\Delta t\) and sum them.
-   **Constructive Interference:** Signals from the target direction add up.
-   **Destructive Interference:** Noise from other directions cancels out.
-   **MVDR (Minimum Variance Distortionless Response):** Adaptive beamforming that minimizes noise power while keeping the target signal.

## Deep Dive: Emotion Recognition (Sentiment Analysis)

Voice contains signals that text doesn't: **Prosody** (Pitch, Volume, Speed).
**Use Case:**
-   User shouts "Representative!" (Anger).
-   System detects High Pitch + High Energy.
-   **Action:** Route to a senior agent immediately. Don't ask "Did you mean...".

**Model:**
-   **Input:** Mel Spectrogram.
-   **Backbone:** CNN (ResNet).
-   **Head:** Classification (Neutral, Happy, Sad, Angry).
-   **Fusion:** Combine with Text Sentiment (BERT).

## Deep Dive: Personalization Architecture

Personalization is the "Secret Sauce" of Voice Search.
If I say "Play my workout playlist", the system needs to know:
1.  Who am I? (Speaker ID).
2.  What is "my workout playlist"? (Entity Resolution).

**Architecture:**
-   **User Graph Service:** A low-latency KV store (e.g., Bigtable/Cassandra) storing user entities.
    -   Key: `UserID`
    -   Value: `{Contacts: [...], Playlists: [...], SmartDevices: [...]}`
-   **Biasing Context:**
    -   When a request comes in, we fetch the User Profile.
    -   We extract relevant entities (e.g., "Workout Mix").
    -   We inject these into the ASR (Contextual Biasing) and the Ranker (Personalization Features).

**Latency Challenge:**
Fetching 10,000 contacts takes time.
**Optimization:**
-   **Prefetching:** Fetch profile as soon as "Hey Google" is detected.
-   **Caching:** Cache active user profiles on the Edge (near the ASR server).

## Deep Dive: Multi-Device Arbitration

You are in the living room. You have a Phone, a Smart Watch, and a Smart Speaker.
You say "Hey Google". All three wake up.
Who answers?

**The Arbitration Protocol:**
1.  **Wake Word Detection:** All devices detect the wake word.
2.  **Energy Estimation:** Each device calculates the volume (energy) of the speech.
3.  **Gossip:** Devices broadcast their energy scores over the local network (WiFi/BLE).
    -   `Speaker: Energy=90`
    -   `Phone: Energy=60`
    -   `Watch: Energy=40`
4.  **Decision:** The device with the highest energy "wins" and lights up. The others go back to sleep.
5.  **Cloud Arbitration:** If devices are not on the same WiFi, the Cloud decides based on timestamp and account ID.

## Deep Dive: Privacy-Preserving ASR (On-Device)

Sending audio to the cloud is a privacy risk.
Modern devices (Pixel, iPhone) run ASR **completely on-device**.

**Technical Challenges:**
1.  **Model Size:** Cloud models are 10GB+. Mobile models must be < 100MB.
    -   **Solution:** Quantization (Int8), Pruning, Knowledge Distillation.
2.  **Battery:** Continuous listening drains battery.
    -   **Solution:** Low-power DSP for Wake Word. Main CPU only wakes up for the query.
3.  **Updates:** How to update the model?
    -   **Solution:** Federated Learning (train on device, send gradients).

**The Hybrid Approach:**
-   Run On-Device ASR for speed and privacy.
-   Run Cloud ASR for accuracy (if network is available).
-   **Ranker:** Choose the result with higher confidence.

## Deep Dive: Evaluation Frameworks (SxS)

How do we measure "Quality"? WER is not enough.
We use **Side-by-Side (SxS)** evaluation.

**Process:**
1.  Take a sample of 1000 queries.
2.  Run them through **System A** (Production) and **System B** (Experiment).
3.  Show the results to human raters.
4.  **Question:** "Which result is better?"
    -   A is much better.
    -   A is slightly better.
    -   Neutral.
    -   B is slightly better.
    -   B is much better.

**Metrics:**
-   **Wins/Losses:** "System B won 10% more queries".
-   **Satisifaction Score:** Average rating (1-5 stars).

## Deep Dive: Latency Optimization

Latency is the #1 killer of Voice UX.
**Budget:** 200ms ASR + 200ms NLU + 200ms Search + 200ms TTS = 800ms.

**Techniques:**
1.  **Streaming RPCs (gRPC):** Don't wait for the full audio. Stream chunks.
2.  **Speculative Execution:**
    -   ASR says "Play Tay...".
    -   Search starts searching for "Taylor Swift", "Taylor Lautner".
    -   ASR finishes "...lor Swift".
    -   Search is already done.
3.  **Early Media:** Start playing the music *before* the TTS says "Okay, playing...".

## Deep Dive: Internationalization (i18n)

Voice Search must work in 50+ languages.
**Challenges:**
1.  **Date/Time Formats:** "Set alarm for half past five".
    -   US: 5:30.
    -   Germany: 5:30 (halb sechs).
2.  **Addresses:**
    -   US: Number, Street, City.
    -   Japan: Prefecture, City, Ward, Block, Number.
3.  **Code Switching:**
    -   India (Hinglish): "Play *Kal Ho Naa Ho* song".
    -   The model must support mixed-language input.

## Deep Dive: The "Long Tail" Problem

Top 1000 queries (Head) are easy ("Weather", "Timer", "Music").
The "Long Tail" (Rare queries) is hard.
"Who was the second cousin of Napoleon?"

**Solution:**
-   **Knowledge Graph:** Structured data helps answer factual queries.
-   **LLM Integration:** Use Large Language Models (Gemini/GPT) to generate answers for long-tail queries where structured data fails.
-   **RAG (Retrieval Augmented Generation):** Retrieve documents -> LLM summarizes -> TTS reads summary.

"Call Mom".
-   If I say it, call *my* mom.
-   If my wife says it, call *her* mom.

**Speaker Diarization / Identification:**
-   We extract a **d-vector** (Speaker Embedding) from the audio.
-   We compare it to the enrolled voice profiles on the device.
-   **Fusion:**
    -   Input: `[Audio Embedding, User Profile Embedding]`
    -   The NLU uses this to resolve "Mom".

## Deep Dive: Spoken Language Understanding (SLU)

ASR gives text. SLU gives meaning.
**Task:** Slot Filling.
**Input:** "Play the new song by Taylor Swift"
**Output:**
-   `Intent: PlayMusic`
-   `SortOrder: Newest`
-   `Artist: Taylor Swift`

**Architecture:**
**BERT + CRF (Conditional Random Field).**
1.  **BERT:** Generates contextual embeddings for each token.
2.  **Linear Layer:** Predicts the intent (`[CLS]` token).
3.  **CRF Layer:** Predicts the slot tags (BIO format) for each token.
    -   `Play (O)`
    -   `the (O)`
    -   `new (B-Sort)`
    -   `song (O)`
    -   `by (O)`
    -   `Taylor (B-Artist)`
    -   `Swift (I-Artist)`

**CRF Importance:**
The CRF ensures valid transitions. It prevents predicting `I-Artist` without a preceding `B-Artist`.

## Deep Dive: Federated Search Logic

The "Federator" is a meta-ranker.
It sends the query to N domains and decides which one wins.

**Signals for Federation:**
1.  **Explicit Trigger:** "Ask **Spotify** to play..." -> Force Music Domain.
2.  **Entity Type:** "Play **Frozen**". 'Frozen' is in the Video Knowledge Graph and Music Knowledge Graph.
3.  **Historical P(Domain | Query):** "Frozen" usually means the movie (80%), not the soundtrack (20%).

**Arbitration:**
If confidence scores are close (Music: 0.85, Video: 0.84), the system might:
-   **Disambiguate:** Ask the user "Did you mean the movie or the soundtrack?"
-   **Multimodal Response:** Show the movie on the screen but play the song (if on a Smart Display).

## Deep Dive: Neural Beam Search

Standard Beam Search uses a simple N-gram LM.
Neural Beam Search uses a Transformer LM (GPT-style).

**Challenge:** Neural LMs are slow.
**Solution:**
1.  **First Pass:** Use a small N-gram LM to generate a lattice.
2.  **Second Pass (Rescoring):** Use the Neural LM to rescore the paths in the lattice.
3.  **Shallow Fusion:**
    \[ Score = \log P_{ASR}(y|x) + \lambda \log P_{NeuralLM}(y) \]
    Compute \(P_{NeuralLM}\) only for the top K tokens in the beam.

## Deep Dive: Handling Noise and Far-Field Audio

Smart Speakers are often 5 meters away, with TV playing in the background.
**Signal Processing Pipeline (Front-End):**
1.  **AEC (Acoustic Echo Cancellation):** Subtract the music the speaker itself is playing from the microphone input.
2.  **Beamforming:** Use the microphone array (7 mics) to focus on the direction of the user's voice.
3.  **Dereverberation:** Remove the room echo.
4.  **NS (Noise Suppression):** Remove steady-state noise (AC, Fan).

**Impact:** Without this, WER increases from 5% to 50%.

## Deep Dive: Context Carryover (Conversational AI)

User: "Who is the President of France?"
System: "Emmanuel Macron."
User: "How old is **he**?"

**Coreference Resolution:**
The system must resolve "**he**" to "**Emmanuel Macron**".
**Architecture:**
-   **Context Encoder:** Encodes the previous turn (`Query_t-1`, `Response_t-1`).
-   **Current Encoder:** Encodes `Query_t`.
-   **Fusion:** Concatenates the embeddings.
-   **NLU:** Predicts `Intent: GetAge`, `Entity: Emmanuel Macron`.

## Deep Dive: Training Data Generation (TTS Augmentation)

We need audio data for "Play [New Song]".
We don't have recordings of users saying it yet.
**Solution:** Use TTS.
1.  **Text Generation:** Generate millions of sentences: "Play X", "Listen to X".
2.  **TTS Synthesis:** Use a high-quality TTS engine to generate audio.
3.  **Audio Augmentation:** Add background noise (Cafe, Street) and Room Impulse Response (Reverb).
4.  **Training:** Train the ASR model on this synthetic data.
**Result:** The model learns to recognize the new entity before a single user has spoken it.

## Deep Dive: The Future (Multimodal LLMs)

Traditional: ASR -> Text -> LLM -> Text -> TTS.
**Latency:** High (Cascading delays).
**Loss:** Prosody is lost. (Emotion, Sarcasm).

**Future: Audio-In, Audio-Out (GPT-4o, Gemini Live).**
-   The model takes raw audio tokens as input.
-   It generates raw audio tokens as output.
-   **Benefits:**
    -   **End-to-End Latency:** < 500ms.
    -   **Emotion:** Can laugh, whisper, and sing.
    -   **Interruption:** Can handle "Barge-in" naturally.

## Deep Dive: Hardware Acceleration (TPU/NPU)

Running a Conformer model (100M params) on a phone CPU is too slow.
We use dedicated **Neural Processing Units (NPUs)** or **Edge TPUs**.

**Optimization Pipeline:**
1.  **Quantization:** Convert Float32 weights to Int8.
    -   **Post-Training Quantization (PTQ):** Easy, slight accuracy drop.
    -   **Quantization Aware Training (QAT):** Train with simulated quantization noise. Recovers accuracy.
2.  **Operator Fusion:** Merge `Conv2D + BatchNorm + ReLU` into a single kernel call. Reduces memory bandwidth.
3.  **Systolic Arrays:**
    -   TPUs use a grid of Multiplier-Accumulators (MACs).
    -   Data flows through the array like a pulse (systole).
    -   Maximizes reuse of loaded weights.

**Result:**
-   **CPU:** 200ms latency, 1W power.
-   **Edge TPU:** 10ms latency, 0.1W power.

## Evaluation: Semantic Error Rate (SemER)

WER is not enough.
-   Ref: "Play Taylor Swift"
-   Hyp: "Play Tailor Swift"
-   **WER:** 1/3 (33% error).
-   **SemER:** 0% (The NLU correctly maps 'Tailor' to the artist entity 'Taylor Swift').

**Calculation:**
\[ \text{SemER} = \frac{D + I + S}{C + D + S} \]
Where D, I, S are Deletion, Insertion, Substitution of **Slots**.
If we miss the `Artist` slot, that's an error. If we get the artist right but misspell the word 'Play', it's not a Semantic Error.

## Deep Dive: Common Failure Modes

Even with SOTA models, things go wrong.
1.  **Barge-in Failure:**
    -   User: "Hey Google, play music."
    -   System: "Okay, playing..." (Music starts loud).
    -   User: "Hey Google, STOP!"
    -   **Failure:** The AEC cannot cancel the loud music fast enough. The system doesn't hear "Stop".
2.  **Side-Speech:**
    -   User A: "Hey Google, set a timer."
    -   User B (to User A): "No, we need to leave now."
    -   **Failure:** System hears "Set a timer no we need to leave". Intent classification fails.
3.  **Trigger-Happy:**
    -   TV Commercial: "OK, Google."
    -   **Failure:** Millions of devices wake up.
    -   **Fix:** Audio Fingerprinting on the commercial audio to blacklist it in real-time.

## Top Interview Questions

**Q1: How do you handle "Homophones" in Voice Search?**
*Answer:*
"Call **Al**" vs "Call **Hal**".
1.  **Personalization:** Check contacts. If 'Hal' exists, boost it.
2.  **Entity Linking:** Check the Knowledge Graph.
3.  **Query Rewriting:** If ASR consistently outputs "Call Al", but users usually mean "Hal", add a rewrite rule `Al -> Hal` based on click logs.

**Q2: Why is Latency so critical in Voice?**
*Answer:*
In text search, results stream in. In voice, the system is silent while thinking.
Silence > 1 second feels "broken".
**Techniques:**
-   **Streaming ASR:** Transcribe while the user is speaking.
-   **Speculative Execution:** Start searching "Taylor Swift" before the user finishes saying "Play...".

**Q3: What is "Endpointing" and why is it hard?**
*Answer:*
Endpointing is deciding when the user has finished speaking.
-   **Too fast:** Cut off the user mid-sentence ("Play Call Me...").
-   **Too slow:** The system feels laggy.
-   **Solution:** Hybrid approach. Use VAD (Voice Activity Detection) + Semantic Completeness (NLU says "Intent is complete").

## Key Takeaways
## 8. Summary

| Component | Role | Key Technology |
| :--- | :--- | :--- |
| **ASR** | Audio -> Text Candidates | Conformer / RNN-T |
| **Rescoring** | Fix ASR errors using Context | Biasing / FSTs |
| **NLU** | Text -> Intent/Slots | BERT / LLMs |
| **Ranker** | Select Best Domain/Action | GBDT / Neural Ranker |

---

**Originally published at:** [arunbaby.com/speech-tech/0028-voice-search-ranking](https://www.arunbaby.com/speech-tech/0028-voice-search-ranking/)

*If you found this helpful, consider sharing it with others who might benefit.*


