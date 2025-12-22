---
title: "Speech Model Export"
day: 47
collection: speech_tech
categories:
  - speech-tech
tags:
  - model-export
  - onnx
  - tflite
  - deployment
  - edge-ai
difficulty: Hard
subdomain: "Model Deployment"
tech_stack: Python, ONNX, TensorFlow Lite
scale: "From cloud to edge devices"
companies: Google, Amazon, Apple, Microsoft, Nuance
related_dsa_day: 47
related_ml_day: 47
related_agents_day: 47
---

**"A speech model that only runs in your training environment isn't a product—it's a prototype."**

## 1. Introduction: The Challenge of Speech Model Deployment

You've trained a fantastic speech recognition model. It transcribes audio with impressive accuracy in your development environment. But your product vision is bigger:

- The model should run on smartphones without an internet connection
- It should work on smart speakers with limited compute
- It should process audio in real-time with sub-100ms latency
- It should be small enough to fit alongside other apps

This is where speech model export becomes critical. Export isn't just saving weights to disk—it's transforming your model into a form that can run efficiently on target hardware.

Speech models present unique export challenges compared to typical deep learning models:

**Streaming requirements:** Speech arrives continuously. The model must process audio chunk-by-chunk, not all at once.

**Real-time constraints:** Users expect instant responses. Latency matters more than for batch processing.

**Audio preprocessing:** Feature extraction (spectrograms, MFCCs) is part of the model pipeline but often handled separately.

**Large vocabulary:** CTC and attention decoders require careful export due to dynamic sequence lengths.

**Model size:** Users won't download a 1GB model for a voice assistant app.

This post explores how to take speech models from training frameworks to deployment-ready formats.

---

## 2. The Speech Model Export Pipeline

A complete speech recognition system has multiple components that each need export consideration:

### 2.1 The Typical Speech Recognition Pipeline

```
Audio Input
    ↓
Preprocessing (feature extraction)
    ↓
Acoustic Model (encoder)
    ↓
Decoder (CTC, attention, or transducer)
    ↓
Language Model (optional post-processing)
    ↓
Text Output
```

Each stage may be exported separately or together, depending on deployment requirements.

### 2.2 Feature Extraction: To Include or Not?

A key decision: should feature extraction be inside or outside the model?

**Feature extraction outside the model:**
- Spectrograms computed in separate code (librosa, torchaudio)
- Model receives features, not raw audio
- More flexible—can change features without re-exporting
- But: deployment must include feature extraction code

**Feature extraction inside the model:**
- First layers transform raw audio to features
- Model receives waveform, outputs text
- Self-contained deployment
- But: less flexibility, larger model

For production, **end-to-end models with integrated feature extraction** are increasingly preferred. They're self-contained and avoid preprocessing mismatches between training and inference.

### 2.3 Handling Streaming Audio

Batch models process complete utterances. Streaming models process audio chunks as they arrive.

**Batch model:** Receives [time × features] tensor, outputs complete transcription
**Streaming model:** Receives [chunk × features] tensor, outputs partial transcription, maintains state

Streaming requires special handling during export:

**Explicit state management:**
- Hidden states must be inputs/outputs of the model
- Each inference call takes previous state, returns new state
- The exported model becomes stateful

**Chunked attention:**
- Full self-attention over entire sequence doesn't work for streaming
- Use causal attention or chunked attention patterns
- These must be preserved during export

---

## 3. Export Format Options

### 3.1 ONNX: The Universal Format

ONNX is often the first choice for speech model export because:

- Works across frameworks (PyTorch, TensorFlow)
- Runs on ONNX Runtime (CPU, GPU, mobile)
- Extensive operator support for speech (LSTM, GRU, attention)
- Good tooling for optimization

**ONNX export considerations for speech:**

**Dynamic sequence lengths:** Audio length varies. ONNX supports dynamic axes—mark the time dimension as dynamic during export.

**Recurrent layers:** LSTMs and GRUs export well to ONNX. The recurrence is unrolled or handled by ONNX operators.

**Attention mechanisms:** Standard attention exports cleanly. Custom attention may need operator mapping.

**Complex decoding:** CTC greedy decoding exports well. Beam search with language models is more complex—often kept outside the ONNX model.

### 3.2 TensorFlow Lite: Mobile and Edge

TensorFlow Lite is designed for mobile phones, IoT devices, and edge deployment.

**Advantages:**
- Very small runtime (~1MB on Android)
- Optimized for ARM processors (phones, Raspberry Pi)
- Supports on-device GPU and NPU acceleration
- Integrated with Android ML Kit

**Speech-specific considerations:**

**Operator support:** TFLite supports most common speech operations, but some custom ops may need TFLite custom implementations.

**Quantization:** TFLite excels at quantization. INT8 speech models run 2-4x faster with minimal accuracy loss.

**Audio input:** TFLite doesn't handle audio I/O—your app captures audio and converts to tensors.

### 3.3 Core ML: Apple Ecosystem

For iOS, macOS, watchOS, and tvOS applications.

**Advantages:**
- Optimized for Apple Neural Engine
- Tight integration with iOS APIs
- On-device privacy (no cloud round-trip)
- Can use Apple's Speech framework for some features

**Speech considerations:**
- Works well for on-device transcription
- Can integrate with Apple's own speech APIs
- Model size limits for App Store (150MB OTA limit considerations)

### 3.4 TensorRT: NVIDIA GPU Inference

For server-side deployment on NVIDIA GPUs.

**Advantages:**
- Maximum throughput on NVIDIA hardware
- Automatic kernel fusion and optimization
- FP16 and INT8 with calibration

**Speech considerations:**
- Great for batch transcription services
- Less relevant for edge/mobile
- Works well with sequence models

---

## 4. Streaming Model Export

Streaming export requires special attention. The model must maintain state between inference calls.

### 4.1 The State Management Pattern

**Non-streaming model:**
```
Input: Complete audio sequence
Output: Complete transcription
State: None (stateless inference)
```

**Streaming model:**
```
Input: Audio chunk + previous state
Output: Partial transcription + new state
State: Encoder hidden states, attention cache, decoder state
```

During export, states become explicit inputs and outputs of the model graph.

### 4.2 What State Needs to Be Managed?

**For RNN-based encoders (LSTM, GRU):**
- Hidden state (h)
- Cell state (c, for LSTM)
- One per layer per direction

**For Transformer encoders with streaming:**
- Key/value cache for previous positions
- Grows with sequence length (bounded for practical use)

**For decoders:**
- Previous predictions
- Attention state
- Language model state (if integrated)

### 4.3 Chunked Processing Considerations

When processing 20ms audio chunks:
- Encoder processes chunk, updates state
- Decoder may wait for sufficient context before emitting tokens
- Trade-off between latency and accuracy

Export must preserve these chunking semantics. The chunk size becomes part of the model's interface.

---

## 5. Optimization During Export

Export is an opportunity to optimize the model for deployment.

### 5.1 Quantization for Speech Models

Speech models tolerate quantization remarkably well:

| Precision | Typical WER Impact | Size Reduction | Speed Improvement |
|-----------|-------------------|----------------|-------------------|
| FP32 (baseline) | 0% | 1x | 1x |
| FP16 | 0-0.5% | 50% | 1.5-2x (GPU) |
| INT8 | 0.5-2% | 75% | 2-4x (CPU/edge) |

**Why speech models quantize well:**
- Feature magnitudes are relatively bounded
- Small quantization errors average out over sequences
- WER (word error rate) is a string-level metric—token-level noise is tolerated

**Quantization approaches:**

**Post-training quantization:** Quantize after training using calibration data. Fast, simple, works well for speech.

**Quantization-aware training:** Simulate quantization during training. Better accuracy but requires retraining.

For most speech models, post-training quantization with proper calibration is sufficient.

### 5.2 Model Pruning

Remove unnecessary weights to reduce size and improve speed:

**Structured pruning:** Remove entire neurons, channels, or attention heads. Results in genuinely smaller model.

**Unstructured pruning:** Zero out individual weights but keep tensor shapes. Requires sparse computation support.

Speech models often have redundancy—attention heads that learn similar patterns, encoder layers that don't all contribute equally. Pruning can remove 30-50% of parameters with <1% WER increase.

### 5.3 Knowledge Distillation

Train a smaller "student" model to mimic your large "teacher":

1. Large teacher model provides "soft" targets
2. Small student learns to match teacher outputs
3. Export the small student

Common patterns for speech:
- Distill Whisper Large to a Whisper Small equivalent
- 5-10x size reduction with 5-10% relative WER increase
- Much better than training small model from scratch

---

## 6. Validating Exported Models

Export can introduce subtle changes. Validation is crucial.

### 6.1 Numerical Equivalence Testing

Compare outputs between original and exported models:

1. Run same audio through both models
2. Compare output distributions (not just final predictions)
3. Allow small numerical differences (1e-4 to 1e-6)
4. Flag large discrepancies

**Common sources of differences:**
- Floating point precision (FP32 vs FP16)
- Operator implementation differences
- Reordered operations (not perfectly associative)
- Random state (dropout should be off for inference)

### 6.2 End-to-End WER Testing

Run a full evaluation on test sets:

1. Export model
2. Run evaluation through exported model
3. Compare WER to original model

**Expected behavior:**
- FP32 export: WER should be identical
- FP16 export: WER within 0.5% relative
- INT8 export: WER within 2-5% relative

Larger discrepancies indicate export problems.

### 6.3 Latency and Throughput Testing

Measure performance on target hardware:

**Latency:** Time from audio input to text output. Critical for real-time.

**Throughput:** Samples processed per second. Critical for batch processing.

**Memory:** Peak memory usage during inference. Critical for edge devices.

Test under realistic conditions—actual device, actual audio, actual batch sizes.

---

## 7. Edge Deployment Considerations

Edge devices have unique constraints.

### 7.1 Memory Constraints

| Device Type | Typical Available Memory |
|-------------|--------------------------|
| High-end smartphone | 2-4 GB |
| Budget smartphone | 512 MB - 1 GB |
| Smart speaker | 256 MB - 1 GB |
| IoT device | 64 - 256 MB |
| Microcontroller | 256 KB - 2 MB |

Your model + runtime must fit with room for the app and audio buffers.

### 7.2 Compute Constraints

| Device Type | Compute Capability |
|-------------|-------------------|
| Phone with NPU | 5-10 TOPS |
| Phone CPU only | 0.5-2 TOPS |
| Raspberry Pi 4 | ~0.1 TOPS |
| Microcontroller | ~0.001 TOPS |

Match model FLOPS requirements to device capability.

### 7.3 Power Constraints

Edge devices run on batteries. Continuous ASR drains batteries fast.

**Mitigation strategies:**
- Keyword spotting (small model) → Full ASR (after wake word)
- VAD to avoid processing silence
- Efficient models (Conformer-S vs Conformer-L)
- Hardware accelerators (NPU more efficient than CPU)

---

## 8. Case Study: Exporting a Conformer Model

Let's trace through exporting a Conformer-based ASR model:

### 8.1 The Starting Point

- Conformer encoder (12 layers, 256 dim)
- CTC decoder
- Trained in PyTorch
- ~30M parameters, ~120MB FP32

### 8.2 Export Strategy

**Step 1: Remove training-only code**
- Disable dropout (set to eval mode)
- Remove auxiliary losses
- Remove teacher forcing (if present)

**Step 2: Handle streaming**
- Refactor encoder for chunk-based processing
- Add explicit state inputs/outputs
- Choose chunk size (e.g., 640ms with 160ms steps)

**Step 3: Export to ONNX**
- Trace model with sample input
- Mark time dimension as dynamic
- Include state tensors in input/output

**Step 4: Validate**
- Numerical comparison vs PyTorch
- WER comparison on test set
- Latency measurement

**Step 5: Optimize**
- Quantize to INT8 with calibration set
- Test WER again (should stay within tolerance)
- Measure final size and speed

### 8.3 Results

| Metric | Original | Exported ONNX FP32 | Exported ONNX INT8 |
|--------|----------|-------------------|-------------------|
| Size | 120 MB | 120 MB | 30 MB |
| WER | 5.2% | 5.2% | 5.5% |
| Latency (CPU) | 450ms | 380ms | 190ms |
| Latency (GPU) | 85ms | 70ms | N/A (FP16: 45ms) |

The exported INT8 model is 4x smaller and 2.4x faster with minimal accuracy loss.

---

## 9. Connection to Model Serialization (ML Day 47) and Tree Serialization (DSA Day 47)

Today's topics share a common theme:

| Aspect | Tree Serialization | Model Serialization | Speech Export |
|--------|-------------------|---------------------|---------------|
| Core challenge | Preserve structure | Preserve computation | Preserve streaming |
| Key insight | Mark nulls explicitly | Self-contained vs code-dependent | State management |
| Optimization | Compact encoding | Quantization/pruning | Latency vs accuracy |
| Validation | Reconstruct correctly | Numerical equivalence | WER preservation |

All three involve transforming complex, stateful structures into portable formats while preserving essential properties.

---

## 10. Key Takeaways

1. **Speech export is more than serialization.** Streaming requirements, real-time constraints, and preprocessing integration make it unique.

2. **Choose format based on target platform.** ONNX for portability, TFLite for mobile, TensorRT for GPUs, Core ML for Apple.

3. **Streaming requires explicit state.** For real-time ASR, states become model inputs/outputs. Design models with this in mind.

4. **Quantization is highly effective for speech.** INT8 models often achieve 4x size reduction with minimal WER impact.

5. **Validate thoroughly.** Export can introduce subtle bugs. Check numerical equivalence and end-to-end WER.

6. **Match model to device constraints.** Edge deployment requires balancing accuracy, size, latency, and power.

7. **Export is part of the product.** A model that can't be deployed efficiently isn't a finished model—it's a research artifact.

Speech model export bridges the gap between research and products that millions of people use every day. Whether it's a voice assistant on a phone, dictation on a laptop, or transcription on a server, efficient export makes it possible.

---

**Originally published at:** [arunbaby.com/speech-tech/0047-speech-model-export](https://www.arunbaby.com/speech-tech/0047-speech-model-export/)

*If you found this helpful, consider sharing it with others who might benefit.*
