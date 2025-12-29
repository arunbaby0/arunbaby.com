---
title: "Model Serialization Systems"
day: 47
related_dsa_day: 47
related_speech_day: 47
related_agents_day: 47
collection: ml_system_design
categories:
 - ml-system-design
tags:
 - onnx
 - serialization
 - deployment
 - protobuf
 - system-design
difficulty: Hard
subdomain: "MLOps"
tech_stack: ONNX, TensorRT, PyTorch, Docker
scale: "Managing 100TB of model artifacts"
companies: NVIDIA, Microsoft (ONNX), Google (TFLite)
---

**"Training is Art. Serialization is Logistics. Wars are won on logistics."**

## 1. Problem Statement

You have just finished training a state-of-the-art Transformer model in PyTorch. It lives in your Python RAM as a complex graph of objects (`nn.Linear`, `nn.MultiHeadAttn`).
Now, your Production Engineering team says:
1. "We need to run this on a C++ server for low latency."
2. "We need to run this on an iPhone (CoreML)."
3. "We need to archive this exact version for 7 years for compliance."

**The Problem**: How do you save the "Soul" of the model (Architecture + Weights + Logic) into a file that is **Portable**, **Versioned**, and **Efficient**?

This is the **Model Serialization** problem. It is the bridge between Research (Python/Notebooks) and Production (C++/Mobile/Cloud).

---

## 2. Understanding the Requirements

### 2.1 What needs to be saved?
A model isn't just weights. It is:
1. **Parameters (Weights)**: Giant tensors of floats (The "Knowledge"). ~100MB to 100GB.
2. **Computation Graph (Architecture)**: The "Map" of how data flows (Matrix Mul -> Relu -> Add).
3. **Metadata**: "Input must be 224x224", "Output class 0 = Cat".

### 2.2 Functional Requirements
- **Interoperability**: Train in PyTorch -> Run in ONNX Runtime / TensorRT.
- **No "Code" Dependency**: The serialized file should be self-contained. I shouldn't need the original `model.py` file to load it.
- **Security**: Loading a model shouldn't execute arbitrary code (The Pickle problem).

---

## 3. High-Level Architecture

The golden standard for modern ML deployment is the **Interchange Format** approach (using ONNX).

``
[PyTorch Training Env] [Intermediate Rep] [Inference Env]
+-------------------+ +----------------+ +------------------+
| Model (RAM) | ---> | ONNX Protocol | ---> | ONNX Runtime |
| (Dynamic Graph) | | Buffer File | | (C++ / Go) |
+-------------------+ +----------------+ +------------------+
 ^
 |
 (Static, Optimized Graph)
``

By serializing to a standard like ONNX, we decouple the *Authoring Tool* (PyTorch) from the *Execution Engine* (NVIDIA Trident / TFLite).

---

## 4. Component Deep-Dives

### 4.1 The Two Philosophies: Pickle vs. Graph

**Approach A: Code-Based (PyTorch `torch.save`)**
- **What it does**: Uses Python's `pickle` to serialize the *state dictionary* (Map of "layer1.weight" -> Tensor).
- **Pros**: Easy, preserves Python flexibility (loops, if-statements).
- **Cons**:
 - **Dependency Hell**: To load it, you need the *exact* original python class definition source code.
 - **Insecure**: Pickle allows code execution. `torch.load('malicious.pt')` can wipe your hard drive.

**Approach B: Graph-Based (TorchScript / ONNX)**
- **What it does**: Traces the execution of the model and records the *operators* (Add, MatMul).
- **Pros**: Self-contained. Safe. Language agnostic.
- **Cons**: Rigid. Hard to serialize dynamic logic (e.g., "if tensor.sum() > 0").

### 4.2 Protocol Buffers (Protobuf)
Under the hood, ONNX uses Google's **Protobuf**.
- It's a binary format.
- It defines a schema: `Node { string op_type; repeated string input; ... }`.
- It is extremely compact and fast to parse compared to JSON.

---

## 5. Data Flow: The Tracing Process

How do we turn Python code into a Static Graph?

1. **Dummy Input**: Create a fake tensor `x = torch.randn(1, 3, 224, 224)`.
2. **Tracer**: Pass `x` through the model.
 - The Tracer records every low-level operation the tensor touches (Conv2d, ReLU).
3. **Graph Construction**: It builds a Directed Acyclic Graph (DAG) of these operations.
4. **Weight Embedding**: The constant weights are serialized as binary blobs and attached to the Graph Nodes.
5. **Optimization**: (Optional) Constant Folding. `3 + 5` is saved as `8`.
6. **Export**: Write `.onnx` file.

---

## 6. Scaling Strategies

### 6.1 Large Model Serialization (The 100GB problem)
You cannot load a 70B Llama model (140GB) into RAM just to save it.
**Sharding**:
- Save weights in chunks (`model-001-of-050.bin`).
- Zero-Copy Loading (`mmap`): Map the file directly on disk to virtual memory without copying to RAM.
- **Safetensors**: A new format by HuggingFace designed specifically for zero-copy memory mapping and removing the `pickle` vulnerability.

### 6.2 Versioning
Never overwrite `model.onnx`.
Use semantic versioning or hashing: `model_v1.0.0_sha256.onnx`.
Store in **Artifactory** or **S3** with immutable tags.

---

## 7. Implementation: Exporting to ONNX

Here is how you actually serialize a PyTorch model to ONNX.

``python
import torch
import torch.nn as nn
import torch.onnx

# 1. Define Model
class Classifier(nn.Module):
 def __init__(self):
 super().__init__()
 self.fc = nn.Linear(10, 1)
 self.relu = nn.ReLU()

 def forward(self, x):
 return self.relu(self.fc(x))

model = Classifier()
model.eval() # Important: Switch to inference mode (freezes BatchNorm)

# 2. Create Dummy Input
dummy_input = torch.randn(1, 10)

# 3. Export
torch.onnx.export(
 model, 
 dummy_input, 
 "classifier.onnx",
 export_params=True, # Store the trained parameter weights inside the model file
 opset_version=12, # ONNX version
 do_constant_folding=True, # Optimization
 input_names = ['input'], # Variable names for graph
 output_names = ['output'],
 dynamic_axes={'input' : {0 : 'batch_size'}} # Allow variable batch size
)

print("Model serialized to classifier.onnx")
``

---

## 8. Monitoring & Metrics

In your serialization pipeline (CI/CD), track:
1. **File Size**: If v2 is 2x larger than v1 unexpectedly, alert.
2. **Inference Speed**: Run benchmarks on the `onnx` file immediately after export.
3. **Numerical Precision**: Compare `output_pytorch` vs `output_onnx`.
 - If error > 1e-5, the serialization is "Lossy" (common with complex operations like LayerNorm).

---

## 9. Failure Modes

1. **Opset Mismatch**: You used a fancy new activation function `Swish` in PyTorch. The standard ONNX runtime (C++) doesn't know what `Swish` is.
 - *Fix*: Provide a custom CUDA implementation for the operator OR decompose it into `Sigmoid * x`.
2. **Dynamic Control Flow**:
 ``python
 if x.sum() > 0: return x
 else: return y
 ``
 The *Tracer* only sees path 1. It saves a graph that *always* returns x. This is a silent bug.
 - *Fix*: Use `torch.jit.script` (Compiler) instead of `trace`, or rewrite using `torch.where`.

---

## 10. Real-World Case Study: Tesla Autopilot

Tesla trains in the cloud (PyTorch clusters) but runs on the car (FSD Chip, custom Silicon).
They cannot run Python in the car (too slow, too much memory).
**Pipeline**:
1. Train Hydranet (Backbone).
2. **Compiler**: Use a custom compiler that converts the Graph into machine code specific to the FSD NPU (Neural Processing Unit).
3. **Quantization**: Convert FP32 floats to INT8 integers to save bandwidth.
4. **Flash**: Write binary to car.

---

## 11. Cost Analysis

Efficient serialization saves money.
- **Storage**: Storing huge checkpoints costs S3 money. Pruning/Quantizing before saving reduces this by 4x.
- **Network**: Transferring 100GB models to 1,000 workers is slow and expensive (egress fees). `Safetensors` allows loading just the layers needed (if doing sharded serving).

---

## 12. Key Takeaways

1. **Decouple Research from Production**: Use ONNX as the contract.
2. **Avoid Pickle**: In untrusted environments, use `Safetensors` or ONNX. It's safer and faster (zero-copy).
3. **Trace carefully**: Be aware of "If" statements in your model code. They are the enemy of static serialization.
4. **Metadata matters**: Always bundle the expected input shape and class labels with the binary.

---

**Originally published at:** [arunbaby.com/ml-system-design/0047-model-serialization](https://www.arunbaby.com/ml-system-design/0047-model-serialization/)

*If you found this helpful, consider sharing it with others who might benefit.*
