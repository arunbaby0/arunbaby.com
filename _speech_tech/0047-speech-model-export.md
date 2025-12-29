---
title: "Speech Model Export"
day: 47
related_dsa_day: 47
related_ml_day: 47
related_agents_day: 47
collection: speech_tech
categories:
 - speech-tech
tags:
 - onnx
 - streaming
 - tflite
 - coreml
 - edge-ai
difficulty: Hard
subdomain: "Deployments"
tech_stack: PyTorch, ONNX, TFLite
scale: "Sub-100ms latency on mobile devices"
companies: Apple (Siri), Amazon (Alexa), Google (Assistant)
---

**"A model that runs in a Jupyter notebook is an experiment. A model that runs on an iPhone is a product."**

## 1. Problem Statement

Speech models are uniquely difficult to export compared to Image models.
- **Image**: Input `[1, 3, 224, 224]`. Static.
- **Speech**: Input is **Infinite** (Streaming Audio).
- **State**: Recurrent models (RNN/Transformer) maintain "memory" of the past sounds.

**The Problem**: How do you export a PyTorch model such that it can run on a Raspberry Pi or Android phone, processing audio in 30ms chunks, while maintaining the *Hidden State* correctly between chunks?

---

## 2. Fundamentals: Batch vs. Streaming

### Batch Inference (Offline)
The user uploads a 1-minute file.
We process the whole file at once.
This is easy to export. It's just a big matrix multiplication.

### Streaming Inference (Online)
The user talks live. We must show text *as they speak*.
We process audio in small "frames" (e.g., 20ms).
**Challenge**: The network needs to know what happened in the previous frame to understand the current frame.
- *Input*: `Current_Audio + Previous_State`
- *Output*: `Current_Text + New_State`

---

## 3. Architecture: The Stateful Export Pattern

When exporting a streaming speech model (like a Transducer or LSTM), we transform the model signature.

**Original (Training) Signature**:
``python
def forward(full_audio):
 return text
``

**Exported (Inference) Signature**:
``python
def forward(audio_chunk, h_state, c_state):
 # Process small chunk using history
 output, new_h, new_c = lstm_cell(audio_chunk, h_state, c_state)
 return output, new_h, new_c
``

The **Client Application** (iOS App) becomes responsible for holding the `new_h` and passing it back into the model on the next loop.

---

## 4. Model Selection for Edge

| Architecture | Exportability | Streaming Latency | Size |
|--------------|---------------|-------------------|------|
| **Conformer (Transformer)** | Hard. Attention caches are complex to export as state. | Medium | Large |
| **LSTM / RNN-T** | Easy. State is just two vectors (h, c). | Lowest | Small |
| **QuartzNet (CNN)** | Easy. "State" is just the left-padding buffer. | Low | Medium |

For ultra-low latency edge devices (Wake Word detection), **Depthwise Separable CNNs** or **LSTMs** are still preferred over heavy Transformers.

---

## 5. Implementation: Exporting a Streaming LSTM to ONNX

We must define a wrapper class that exposes the state as input/output.

``python
import torch
import torch.nn as nn

class StreamingLSTM(nn.Module):
 def __init__(self, input_dim, hidden_dim):
 super().__init__()
 self.lstm = nn.LSTMCell(input_dim, hidden_dim)
 self.fc = nn.Linear(hidden_dim, 10) # Output chars

 def forward(self, x_chunk, h_prev, c_prev):
 # x_chunk: [Batch, 1, Features] - One frame
 h_new, c_new = self.lstm(x_chunk, (h_prev, c_prev))
 output = self.fc(h_new)
 return output, h_new, c_new

# 1. Instantiate
model = StreamingLSTM(input_dim=80, hidden_dim=256)
model.eval()

# 2. Creating Dummy Inputs for Tracing
dummy_input = torch.randn(1, 80)
dummy_h = torch.zeros(1, 256)
dummy_c = torch.zeros(1, 256)

# 3. Export to ONNX
torch.onnx.export(
 model,
 (dummy_input, dummy_h, dummy_c), # Tuple of inputs
 "streaming_asr.onnx",
 input_names=["audio_chunk", "h_in", "c_in"],
 output_names=["logits", "h_out", "c_out"],
 opset_version=12
)
``

Now, the Android developer sees an ONNX function asking for `h_in`. They don't need to know what an LSTM is. They just know they need to pass `h_out` back into `h_in`.

---

## 6. Training Considerations

- **Quantization Aware Training (QAT)**:
 If deploying to INT8 (common for Phones/DSP), you should simulate quantization *during* training. Standard FP32 training followed by simple casting often destroys accuracy for Speech (dynamic range of audio is high).
 - Use `torch.quantization.prepare_qat`.

---

## 7. Production Deployment: The Runtime

On the device, we rely on specialized runtimes:
1. **TFLite (Google)**: Standard for Android. Highly optimized for ARM CPUs.
2. **CoreML (Apple)**: Uses the Neural Engine (NPU) on iPhones. Requires converting ONNX -> CoreML.
3. **SNPE (Qualcomm)**: Runs on the Hexagon DSP (Digital Signal Processor). Extremely low power (can run 24/7 for wake words).

---

## 8. Streaming Real-Time Logic (Client Side)

The client code loop looks like this (Pseudo-C++):

``cpp
// Initialize Logic
State h = zeros();
State c = zeros();

while (is_recording) {
 // 1. Read 20ms from Microphone
 AudioBuffer chunk = Microphone.read(320_samples); 
 
 // 2. Compute Mel-Spectrogram features
 Tensor features = ComputeMFCC(chunk);
 
 // 3. Run Inference
 auto result = Model.run(features, h, c);
 
 // 4. Update State for next loop
 h = result.h_out;
 c = result.c_out;
 
 // 5. Decode text
 String text = CTSDecode(result.logits);
 Display(text);
}
``

---

## 9. Quality Metrics

- **Real-Time Factor (RTF)**: `Processing_Time / Audio_Duration`.
 - RTF < 1.0 is required.
 - Target RTF: 0.1 (Process 1 sec of audio in 0.1 sec).
- **Model Size**: < 50MB for App Store download limits over cellular.
- **Power Consumption**: mW per inference.

---

## 10. Common Failure Modes

1. **State Drift**: If the floating point precision differs between Training (GPU) and Inference (DSP), the recurrent state `h` might slowly diverge over a long conversation, leading to gibberish after 5 minutes.
 - *Fix*: Periodically reset state during silence.
2. **Buffer Underrun**: The model is too slow. Audio accumulates in the buffer faster than we process it. Latency grows infinitely.
 - *Fix*: Drop frames (bad) or use a smaller model (Quantization).

---

## 11. State-of-the-Art

**Unified Streaming/Non-Streaming**.
Google's USM models can switch modes. You export one graph.
- If you feed `state=None`, it runs in batch mode (high accuracy).
- If you feed `state=Tensor`, it runs in streaming mode (low latency).

---

## 12. Key Takeaways

1. **Explicit State**: Export hidden states as explicit I/O ports in your model graph.
2. **Quantize for Edge**: Mobile CPUs need INT8 for speed and battery life.
3. **Client Loop Responsbility**: The application code "owns" the memory (state loop). The model is just a pure function transition.
4. **Buffer Management**: Handling raw audio buffers (ring buffers) is 50% of the work in Speech deployment.

---

**Originally published at:** [arunbaby.com/speech-tech/0047-speech-model-export](https://www.arunbaby.com/speech-tech/0047-speech-model-export/)

*If you found this helpful, consider sharing it with others who might benefit.*
