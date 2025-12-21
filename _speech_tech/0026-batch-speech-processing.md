---
title: "Batch Speech Processing"
day: 26
collection: speech_tech
categories:
  - speech-tech
  - asr
  - pipelines
tags:
  - offline-asr
  - diarization
  - forced-alignment
  - whisper
  - kaldi
subdomain: "ASR Systems"
tech_stack: [Python, FFMpeg, Whisper, Pyannote]
scale: "Transcribing millions of hours"
companies: [Otter.ai, Rev.com, Descript, YouTube]
related_dsa_day: 26
related_ml_day: 26
---

**Real-time ASR is hard. Offline ASR is big.**

## The Use Case: "Transcribe This Meeting"

In Real-Time ASR (Siri), latency is king. You sacrifice accuracy for speed.
In Batch ASR (Otter.ai, YouTube Captions), **Accuracy is king**. You have the whole file. You can look ahead. You can run massive models.

**Key Differences:**
- **Lookahead:** Batch models can see the *future* context (Bidirectional RNNs/Transformers). Real-time models are causal (Unidirectional).
- **Compute:** Batch can run on a massive GPU cluster overnight.
- **Features:** Batch enables Speaker Diarization ("Who said what?") and Summarization.

## High-Level Architecture: The Batch Pipeline

```ascii
+-----------+     +------------+     +-------------+
| Raw Audio | --> | Preprocess | --> | Segmentation|
+-----------+     +------------+     +-------------+
(MP3/M4A)         (FFMpeg/VAD)       (30s Chunks)
                                           |
                                           v
+-----------+     +------------+     +-------------+
| Final Txt | <-- | Post-Proc  | <-- | Transcription|
+-----------+     +------------+     +-------------+
(SRT/JSON)        (Diarization)      (Whisper/ASR)
```

## Pipeline Architecture

A typical Batch Speech Pipeline has 4 stages:

1.  **Preprocessing (FFMpeg):**
    - Convert format (MP3/M4A -> WAV).
    - Resample (44.1kHz -> 16kHz).
    - Mixdown (Stereo -> Mono).
    - **VAD (Voice Activity Detection):** Remove silence to save compute.

2.  **Segmentation:**
    - Split long audio (1 hour) into chunks (30 seconds).
    - Why? Transformer attention scales quadratically \(O(N^2)\). You can't feed 1 hour into BERT.

3.  **Transcription (ASR):**
    - Run Whisper / Conformer on each chunk.
    - Get text + timestamps.

4.  **Post-Processing:**
    - **Diarization:** Cluster segments by speaker.
    - **Punctuation & Capitalization:** "hello world" -> "Hello world."
    - **Inverse Text Normalization (ITN):** "twenty dollars" -> "$20".

## Deep Dive: Speaker Diarization

**Goal:** Partition the audio stream into homogeneous segments according to the speaker identity.
**Output:** "Speaker A spoke from 0:00 to 0:10. Speaker B spoke from 0:10 to 0:15."

**Algorithm:**
1.  **Embedding:** Extract a vector (d-vector / x-vector) for every 1-second sliding window.
2.  **Clustering:** Use **Spectral Clustering** or **Agglomerative Hierarchical Clustering (AHC)** to group similar vectors.
3.  **Re-segmentation:** Refine boundaries using a Viterbi decode.

**Tool:** `pyannote.audio` is the industry standard library.

## Deep Dive: Forced Alignment

**Goal:** Align text to audio at the *phoneme* level.
**Input:** Audio + Transcript ("The cat sat").
**Output:** "The" (0.1s - 0.2s), "cat" (0.2s - 0.5s)...

**How?**
We know the sequence of phonemes (from the text). We just need to find the optimal path through the audio frames that matches this sequence.
This is solved using **Viterbi Alignment** (HMMs) or **CTC Segmentation**.

**Use Case:**
- **Karaoke:** Highlighting lyrics.
- **Video Editing:** "Delete this word" -> Cuts the audio frame. (Descript).

## System Design: YouTube Auto-Captions

**Scale:** 500 hours of video uploaded *every minute*.

**Architecture:**
1.  **Upload:** Video lands in Colossus (Google File System).
2.  **Trigger:** Pub/Sub message to ASR Service.
3.  **Sharding:** Video is split into 5-minute chunks.
4.  **Parallelism:** 12 chunks are processed by 12 TPUs in parallel.
5.  **Merge:** Results are stitched together.
6.  **Indexing:** Captions are indexed for Search (SEO).

## Python Example: Batch Transcription with Whisper

```python
import whisper

# Load model (Large-v2 is slow but accurate)
model = whisper.load_model("large")

# Transcribe
# beam_size=5: Better accuracy, slower
result = model.transcribe("meeting.mp3", beam_size=5)

print(result["text"])

# Segments with timestamps
for segment in result["segments"]:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    print(f"[{start:.2f} - {end:.2f}]: {text}")
```

## Deep Dive: Whisper Architecture (The New Standard)

OpenAI's Whisper changed the game in 2022.
**Architecture:** Standard Transformer Encoder-Decoder.
**Key Innovation:** Weakly Supervised Training on 680,000 hours of internet audio.

**Input:**
- Log-Mel Spectrogram (80 channels).
- 30-second windows (padded/truncated).

**Decoder Tasks (Multitasking):**
The model predicts special tokens to control behavior:
- `<|startoftranscript|>`
- `<|en|>` (Language ID)
- `<|transcribe|>` or `<|translate|>`
- `<|timestamps|>` (Predict time alignment)

**Why it wins:**
It's robust to accents, background noise, and technical jargon because it was trained on "wild" data, not just clean audiobooks (LibriSpeech).

## Deep Dive: Word Error Rate (WER)

How do we measure accuracy? **Levenshtein Distance.**
`WER = (S + D + I) / N`
- **S (Substitution):** "cat" -> "bat"
- **D (Deletion):** "the cat" -> "cat"
- **I (Insertion):** "cat" -> "the cat"
- **N:** Total words in reference.

**Example:**
Ref: "The cat sat on the mat"
Hyp: "The bat sat on mat"
- Sub: cat->bat (1)
- Del: the (1)
- WER = (1 + 1 + 0) / 6 = 33%

**Pitfall:** WER doesn't care about meaning. "I am happy" -> "I am not happy" is a small WER but a huge semantic error.

## Engineering: Inverse Text Normalization (ITN)

ASR output: "i have twenty dollars"
User wants: "I have $20."

**ITN** is the process of converting spoken-form text to written-form text.
**Techniques:**
1.  **FST (Finite State Transducers):** Rule-based grammars (Nvidia NeMo). Fast, deterministic.
2.  **LLM Rewriting:** "Rewrite this transcript to be formatted correctly." Slower, but handles complex cases ("Call 1-800-FLOWERS").

## Advanced Topic: CTC vs. Transducer vs. Attention

How does the model align Audio (T frames) to Text (L tokens)? `T >> L`.

**1. CTC (Connectionist Temporal Classification):**
- Output a probability for every frame.
- Merge repeats: `cc-aa-t` -> `cat`.
- **Pros:** Fast, parallel.
- **Cons:** Conditional independence assumption (frame `t` doesn't know about `t-1`).

**2. RNN-Transducer (RNN-T):**
- Has a "Prediction Network" (Language Model) that feeds back previous tokens.
- **Pros:** Streaming friendly, better accuracy than CTC.
- **Cons:** Hard to train (memory intensive).

**3. Attention (Encoder-Decoder):**
- The Decoder attends to the entire Encoder output.
- **Pros:** Best accuracy (Global context).
- **Cons:** Not streaming friendly (requires full audio). **Perfect for Batch.**

## Deep Dive: Voice Activity Detection (VAD)

In Batch processing, VAD is a **Cost Optimization**.
If 50% of the recording is silence, VAD saves 50% of the GPU/API cost.

**Silero VAD:**
The current state-of-the-art open-source VAD.
- **Model:** Enterprise-grade DNN.
- **Size:** < 1MB.
- **Speed:** < 1ms per chunk.

**Pipeline:**
1.  Run VAD. Get timestamps `[(0, 10), (15, 20)]`.
2.  Crop audio.
3.  Batch the speech segments.
4.  Send to Whisper.
5.  Re-align timestamps to original file.

## Appendix B: Interview Questions

1.  **Q:** "Why is Whisper better than Wav2Vec 2.0?"
    **A:** Wav2Vec 2.0 is Self-Supervised (needs fine-tuning). Whisper is Weakly Supervised (trained on labeled data). Whisper handles punctuation and casing out-of-the-box.

2.  **Q:** "How do you handle 'Hallucinations' in Whisper?"
    **A:**
    - **Beam Search:** Increase beam size.
    - **Temperature Fallback:** If log-prob is low, increase temperature.
    - **VAD:** Don't feed silence to Whisper (it tries to transcribe noise as words).

3.  **Q:** "What is the difference between Speaker Diarization and Speaker Identification?"
    **A:**
    - **Diarization:** "Who spoke when?" (Speaker A, Speaker B). No names.
    - **Identification:** "Is this Elon Musk?" (Matches against a database of voice prints).

## Deep Dive: Conformer Architecture (The "Macaron" Net)

Whisper uses Transformers. But Google uses **Conformers**.
**Idea:** Transformers are good at global context (Attention). CNNs are good at local features (Edges).
**Conformer = CNN + Transformer.**

**The Block:**
1.  **Feed Forward (Half-Step):** Like a Macaron sandwich.
2.  **Self-Attention:** Captures long-range dependencies.
3.  **Convolution Module:** Captures local patterns (phonemes).
4.  **Feed Forward (Half-Step).**
5.  **Layer Norm.**

**Why?** It converges faster and requires less data than pure Transformers for speech.

## Deep Dive: SpecAugment (Data Augmentation)

How do you prevent overfitting in ASR?
**SpecAugment:** Augment the Spectrogram, not the Audio.

**Transformations:**
1.  **Time Warping:** Stretch/squeeze parts of the spectrogram.
2.  **Frequency Masking:** Zero out a block of frequencies (Simulates a broken mic).
3.  **Time Masking:** Zero out a block of time (Simulates packet loss).

**Impact:** It forces the model not to rely on any single frequency band or time slice. It learns robust features.

## System Design: Privacy and PII Redaction

**Scenario:** You are transcribing Call Center audio. It contains Credit Card numbers.
**Requirement:** Redact PII (Personally Identifiable Information).

**Pipeline:**
1.  **ASR:** Transcribe audio to text + timestamps.
2.  **NER (Named Entity Recognition):** Run a BERT model to find `[CREDIT_CARD]`, `[PHONE_NUMBER]`.
3.  **Redaction:**
    - **Text:** Replace with `[REDACTED]`.
    - **Audio:** Beep out the segment using the timestamps.

**Challenge:** "My name is **Art**" vs "This is **Art**". NER is hard on lowercase ASR output.
**Solution:** Use a Truecasing model first.

## Engineering: GPU Inference Optimization

Batch processing is expensive. How do we make it cheaper?

**1. Dynamic Batching:**
- Don't run 1 file at a time.
- Pack 32 files into a batch.
- **Padding:** Pad all files to the length of the longest file.
- **Sorting:** Sort files by duration *before* batching to minimize padding (and wasted compute).

**2. TensorRT / ONNX Runtime:**
- Compile the PyTorch model to TensorRT.
- Fuses layers (Conv + ReLU).
- Quantization (FP16 or INT8).
- **Speedup:** 2x - 5x.

**3. Flash Attention:**
- IO-aware exact attention.
- Reduces memory usage from \(O(N^2)\) to \(O(N)\). Allows processing 1-hour files in one go.

## Advanced Topic: Multilingual ASR

**Problem:** Code-switching ("Hindi-English").
**Solution:**
1.  **Language ID:** Predict language every 5 seconds.
2.  **Multilingual Model:** Train one model on 100 languages (Whisper).
    - It learns a shared representation of "Speech".
    - It can even do **Zero-Shot Translation** (Speech in French -> Text in English).

## Case Study: Spotify Podcast Transcription

**Goal:** Transcribe 5 million podcasts for Search.
**Challenges:**
- **Music:** Podcasts have intro music. ASR tries to transcribe lyrics.
- **Overlapping Speech:** 3 people laughing.
- **Length:** Joe Rogan is 3 hours long.

**Solution:**
1.  **Music Detection:** Remove music segments.
2.  **Chunking:** Split into 30s chunks with 5s overlap.
3.  **Deduplication:** Merge the overlap regions using "Longest Common Subsequence".

## Appendix C: Advanced Interview Questions

1.  **Q:** "How does CTC Loss handle alignment?"
    **A:** It introduces a "blank" token `<eps>`. `c-a-t` aligns to `c <eps> a <eps> t`. It sums over all possible alignments that produce the target text.

2.  **Q:** "What is the difference between Online and Offline Diarization?"
    **A:**
    - **Online:** Must decide "Who is speaking?" *now*. Hard.
    - **Offline:** Can look at the whole file. Can run clustering (Spectral/AHC) on all embeddings. Much better accuracy.

3.  **Q:** "How do you optimize ASR for a specific domain (e.g., Medical)?"
    **A:**
    - **Fine-tuning:** Train the acoustic model on medical audio.
    - **Language Model Fusion:** Train a text-only LM on medical journals. Fuse it with the ASR output during decoding (Shallow Fusion).

## Deep Dive: Beam Search Decoding

The model outputs probabilities for each token: `P(token | audio)`.
**Greedy Search:** Pick the highest probability token at each step.
- Problem: It misses the global optimum. "The" (0.9) -> "cat" (0.1) might be worse than "A" (0.4) -> "dog" (0.8).

**Beam Search:**
Keep the top `K` (Beam Width) hypotheses alive at each step.
1.  Start with `[<s>]`.
2.  Expand all `K` paths by 1 token.
3.  Calculate score: `log(P(path))`.
4.  Keep top `K`.
5.  Repeat.

**Beam Width Trade-off:**
- `K=1`: Greedy (Fast, Bad).
- `K=5`: Good balance (Whisper default).
- `K=100`: Diminishing returns, very slow.

## Advanced Topic: Language Model Integration (Shallow vs. Deep Fusion)

ASR models know phonetics. Language Models (GPT) know grammar.
How do we combine them?

**1. Shallow Fusion:**
`Score = log P_ASR(y|x) + lambda * log P_LM(y)`
- We interpolate the scores during Beam Search.
- **Pros:** No retraining of ASR. Can swap LMs easily (Medical LM, Legal LM).

**2. Deep Fusion:**
- Concatenate the hidden states of ASR and LM *before* the softmax layer.
- **Pros:** Better integration.
- **Cons:** Requires retraining.

## System Design: Audio Fingerprinting (Shazam)

**Problem:** Identify the song in the background.
**Algorithm:**
1.  **Spectrogram:** Convert audio to Time-Frequency.
2.  **Peaks:** Find local maxima (constellation map).
3.  **Hashes:** Form pairs of peaks (Anchor point + Target point).
    - `Hash = (Freq1, Freq2, DeltaTime)`
4.  **Database:** Store `Hash -> (SongID, AbsoluteTime)`.
5.  **Query:** Match hashes. Find a cluster of matches with consistent time offset.

## Engineering: FFMpeg Tricks for Speech

FFMpeg is the Swiss Army Knife of Batch Processing.

**1. Normalization (Loudness):**
`ffmpeg -i input.wav -filter:a loudnorm output.wav`
Ensures consistent volume (-14 LUFS).

**2. Silence Removal:**
`ffmpeg -i input.wav -af silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-50dB output.wav`
Trims silence > 1 second.

**3. Speed Up (without changing pitch):**
`ffmpeg -i input.wav -filter:a atempo=1.5 output.wav`
Listen to podcasts at 1.5x.

## Case Study: Alexa's Wake Word Detection (Cascade)

Alexa is always listening. But sending audio to the cloud is expensive (and creepy).
**Solution: Cascade Architecture.**

1.  **Stage 1 (DSP):** Low power chip. Detects energy. (mW).
2.  **Stage 2 (Tiny NN):** On-device model. Detects "Alexa". (High Recall, Low Precision).
3.  **Stage 3 (Cloud):** Full ASR. Verifies "Alexa" and processes the command. (High Precision).

## Appendix D: The "Long-Tail" Problem

ASR works great for "Standard American English".
It fails for:
- **Accents:** Scottish, Singaporean.
- **Stuttering:** "I... I... want to go".
- **Code Switching:** "Chalo let's go".

**Solution:**
- **Data Augmentation:** Add noise, speed perturbation.
- **Transfer Learning:** Fine-tune on specific accent datasets (Common Voice).

## Conclusion

Batch processing allows us to use the "Heavy Artillery" of AI.
We can use the biggest models, look at the entire context, and perform complex post-processing.
*If you found this helpful, consider sharing it with others who might benefit.*




---

**Originally published at:** [arunbaby.com/speech-tech/0026-batch-speech-processing](https://www.arunbaby.com/speech-tech/0026-batch-speech-processing/)

*If you found this helpful, consider sharing it with others who might benefit.*

