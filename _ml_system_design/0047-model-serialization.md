---
title: "Model Serialization Systems"
day: 47
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - model-serialization
  - deployment
  - onnx
  - tensorrt
  - model-export
difficulty: Hard
subdomain: "Model Deployment"
tech_stack: Python, ONNX, TensorFlow, PyTorch
scale: "Models from KB to hundreds of GB"
companies: Google, Meta, Microsoft, NVIDIA, Amazon
related_dsa_day: 47
related_speech_day: 47
related_agents_day: 47
---

**"A model that can't be saved is a model that can't be deployed."**

## 1. Introduction: Why Model Serialization Matters

You've trained a brilliant model. It achieves state-of-the-art accuracy on your test set. Your Jupyter notebook shows impressive predictions. But then someone asks: "How do we deploy this to production?"

Suddenly, you face a cascade of questions:

- How do you save the model so the training server can be shut down?
- How do you load it on a different machine with different hardware?
- How do you share it with teammates who use different frameworks?
- How do you run it on edge devices with limited resources?
- How do you version it so you can roll back if something goes wrong?

These questions all point to the same fundamental problem: **model serialization**—converting your trained model from an in-memory object to a portable, persistent format.

This might sound like a mundane technical detail, but poor serialization choices cause real production problems:

- Models that can't be loaded after library upgrades
- Deployment failures due to framework version mismatches
- Massive storage costs from inefficient formats
- Slow inference from unoptimized serialization
- Security vulnerabilities from unsafe deserialization

Understanding model serialization is essential for anyone who wants their models to leave the laboratory and enter the real world.

---

## 2. What Gets Serialized?

When you save a model, what exactly are you saving? A trained neural network consists of several components:

### 2.1 Learned Parameters

The weights and biases that the model learned during training. This is the "knowledge" of the model.

**Typical sizes:**
- Small CNN: 1-10 MB
- ResNet-50: ~100 MB
- BERT-base: ~400 MB
- GPT-3: ~350 GB (175B parameters × 2 bytes per parameter)

For large models, parameter storage dominates everything else.

### 2.2 Model Architecture

The structure of the network: how layers connect, what operations they perform, their configurations.

This includes:
- Layer types (Linear, Conv2d, Attention, etc.)
- Layer configurations (kernel size, number of units, activation functions)
- Connection topology (sequential, skip connections, branching)
- Input/output shapes

### 2.3 Optimizer State

If you want to resume training, you also need:
- Optimizer parameters (learning rate, momentum)
- Optimizer state (moving averages in Adam, etc.)
- Training step count

This can be as large as the model itself (Adam stores 2 additional values per parameter).

### 2.4 Metadata

Additional information for practical use:
- Training hyperparameters
- Dataset information
- Performance metrics
- Version information
- Preprocessing requirements

---

## 3. Serialization Formats: The Landscape

### 3.1 Framework-Native Formats

Each deep learning framework has its own serialization format:

**PyTorch (.pt, .pth)**
- Uses Python's pickle under the hood
- Can save just state_dict (weights only) or entire model
- Tight coupling with PyTorch code

**TensorFlow SavedModel**
- Directory-based format with protocol buffers
- Includes computation graph and weights
- Designed for TensorFlow Serving

**TensorFlow Checkpoints**
- Simpler format for saving/restoring during training
- Less portable than SavedModel

**Keras (.h5, .keras)**
- HDF5-based format
- Human-inspectable with HDF5 tools
- Good for archival

### 3.2 Cross-Framework Formats

Formats designed to work across different frameworks:

**ONNX (Open Neural Network Exchange)**
- Industry standard for interoperability
- Supported by PyTorch, TensorFlow, and many inference engines
- Defines a common set of operators
- Great for deployment to different runtimes

**TorchScript**
- PyTorch's ahead-of-time compilation format
- Removes Python dependency for inference
- Enables C++ deployment

### 3.3 Optimized Inference Formats

Formats designed for fast inference on specific hardware:

**TensorRT (NVIDIA)**
- Highly optimized for NVIDIA GPUs
- Applies kernel fusion, precision calibration, layer optimization
- Often 2-5x faster than framework inference

**OpenVINO (Intel)**
- Optimized for Intel CPUs and accelerators
- Model compression and quantization built-in

**Core ML (Apple)**
- Optimized for Apple devices
- Integrates with iOS/macOS APIs

**TFLite (Google)**
- Designed for mobile and edge devices
- Supports quantization for smaller, faster models

---

## 4. The Two Philosophies: Code vs. Graph

There's a fundamental tension in how to serialize models, reflected in two approaches:

### 4.1 Code-Based Serialization

**Approach:** Save weights separately; rely on code to define architecture.

**How it works:**
1. Define model class in code
2. Save only the learned parameters (state_dict)
3. To load: create model from code, then load parameters into it

**Advantages:**
- Smaller file size (no architecture duplication)
- Full flexibility (any Python code can define the model)
- Easy to modify architecture between saves

**Disadvantages:**
- Code must be available at load time
- Version mismatches between save and load can cause failures
- Can't deploy without Python environment

**Example scenario:** Your PyTorch model uses a custom attention layer you wrote. With code-based serialization, you must have that custom layer code available when loading.

### 4.2 Graph-Based Serialization

**Approach:** Save everything—weights AND architecture—in a self-contained format.

**How it works:**
1. Trace or script the model to capture computation graph
2. Serialize the complete graph with weights embedded
3. To load: the graph contains everything needed

**Advantages:**
- Self-contained (no external code needed)
- Can run without Python (C++, mobile, embedded)
- More portable across environments

**Disadvantages:**
- Limited to operations the format supports
- May not capture dynamic control flow
- Tracing can miss conditional branches

**Example scenario:** You export your model to ONNX. Anyone can load and run it without your original training code—they just need an ONNX runtime.

### 4.3 When to Use Which

| Scenario | Recommended Approach |
|----------|---------------------|
| Checkpointing during training | Code-based (state_dict) |
| Sharing with teammates using same framework | Code-based |
| Deploying to production servers | Graph-based (TorchScript, SavedModel) |
| Running on edge devices | Graph-based (TFLite, ONNX) |
| Cross-framework interoperability | Graph-based (ONNX) |
| Long-term archival | Graph-based (more stable over time) |

---

## 5. ONNX: The Interoperability Standard

ONNX deserves special attention because it's become the standard for cross-platform model deployment.

### 5.1 What ONNX Is

ONNX (Open Neural Network Exchange) is:
- A common file format for neural network models
- A standard set of operators (convolution, matmul, attention, etc.)
- An ecosystem of tools for optimization and inference

Think of ONNX as the "PDF of machine learning"—a universal format that any compatible reader can understand.

### 5.2 How ONNX Works

An ONNX file contains:
1. **Model graph**: Nodes (operations) and edges (tensors)
2. **Operator definitions**: What each node does
3. **Weights**: The learned parameters
4. **Metadata**: Inputs, outputs, model version

When you export a PyTorch model to ONNX:
1. The model is traced with sample input
2. Operations are mapped to ONNX operators
3. The graph and weights are serialized to protobuf

### 5.3 ONNX Operators

ONNX defines ~180 standard operators covering:
- Basic math (Add, Multiply, MatMul)
- Neural network layers (Conv, BatchNorm, LSTM)
- Activations (ReLU, Sigmoid, Softmax)
- Transformations (Reshape, Transpose, Gather)

If your model uses only standard operators, it exports cleanly. Custom operations require extra work.

### 5.4 ONNX Runtime

ONNX files are executed by ONNX Runtime, which:
- Runs on CPU, GPU, TPU, and more
- Applies hardware-specific optimizations
- Provides consistent API across platforms
- Supports C++, C#, Python, Java, JavaScript

This means you can train in PyTorch, export to ONNX, and deploy with ONNX Runtime on completely different infrastructure.

### 5.5 Limitations of ONNX

ONNX isn't perfect:
- **Dynamic control flow**: If/else based on tensor values may not export
- **Custom operations**: May need to implement custom ONNX ops
- **Opset versions**: Different ONNX versions support different operators
- **Precision mismatches**: Slight numerical differences from original framework

---

## 6. Optimizing Serialized Models

Serialization is also an opportunity to optimize models for deployment.

### 6.1 Quantization

Reduce numerical precision to shrink model size and speed up inference:

**FP32 → FP16 (half precision)**
- 50% size reduction
- Often no accuracy loss
- Faster on modern GPUs

**FP32 → INT8 (8-bit integers)**
- 75% size reduction
- Requires calibration to minimize accuracy loss
- Much faster on supported hardware

**Example impact:**

| Precision | Size | Inference Speed | Accuracy |
|-----------|------|-----------------|----------|
| FP32 | 100 MB | 1x | 100% |
| FP16 | 50 MB | 1.5x | 99.9% |
| INT8 | 25 MB | 2-4x | 99-99.5% |

### 6.2 Pruning

Remove unnecessary weights (near-zero values):
- Structured pruning: Remove entire neurons/channels
- Unstructured pruning: Remove individual weights

Post-pruning, re-serialize the smaller model.

### 6.3 Knowledge Distillation

Train a smaller "student" model to mimic a larger "teacher":
- Student is trained to match teacher's outputs
- Serialize the smaller student for deployment
- Significant size reduction with modest accuracy loss

### 6.4 Graph Optimization

Inference engines optimize the computation graph:

**Operator fusion:** Combine multiple operations into one
- Conv + BatchNorm + ReLU → Single fused operation
- Reduces memory transfers and kernel launches

**Constant folding:** Pre-compute constant expressions
- Operations on known values computed at export time
- Removes unnecessary runtime computation

**Dead code elimination:** Remove unused graph portions
- Paths that never affect output are removed

---

## 7. Versioning and Compatibility

Models evolve over time. Managing versions is critical.

### 7.1 The Versioning Problem

Scenario: You trained a model 6 months ago. Today you need to load it, but:
- Your PyTorch version upgraded from 1.9 to 2.0
- A custom layer's API changed
- The preprocessing code was refactored

Will the model load? Will it produce the same results?

### 7.2 Best Practices for Model Versioning

**Include version metadata:**
- Framework version
- Git commit hash of training code
- Timestamp
- Dataset version
- Key hyperparameters

**Use semantic versioning for models:**
- Major version: Architecture changes (breaking)
- Minor version: Training improvements (backward compatible)
- Patch version: Bug fixes

**Lock dependencies:**
- Store requirements.txt or environment.yaml with model
- Consider using containers (Docker) for reproducibility

**Test backward compatibility:**
- Maintain a test suite that loads old models
- Catch breaking changes before they hit production

### 7.3 Model Registries

Production systems use model registries:
- Central storage for model artifacts
- Version tracking and comparison
- Metadata and lineage
- Approval workflows for deployment

Popular options: MLflow, Weights & Biases, AWS SageMaker Model Registry, Vertex AI Model Registry

---

## 8. Security Considerations

Model serialization has security implications that are often overlooked.

### 8.1 The Pickle Problem

PyTorch uses Python's pickle for serialization. Pickle is powerful but dangerous:
- Pickle can execute arbitrary Python code during loading
- A malicious .pt file could compromise your system
- Never load pickle files from untrusted sources

**Mitigation:**
- Only load models from trusted sources
- Use `weights_only=True` when possible
- Prefer safer formats (ONNX, SafeTensors) for sharing

### 8.2 SafeTensors

Hugging Face developed SafeTensors as a secure alternative:
- Stores only tensor data, no executable code
- Cannot execute arbitrary code during loading
- Fast loading through memory mapping
- Becoming the standard for model sharing

### 8.3 Model Integrity

Ensure models haven't been tampered with:
- Compute checksums (SHA-256) of model files
- Store checksums in model registry
- Verify before loading

---

## 9. Real-World Deployment Patterns

### 9.1 Pattern 1: Training → ONNX → Multiple Runtimes

**Scenario:** Train once, deploy to multiple platforms

**Flow:**
1. Train model in PyTorch/TensorFlow
2. Export to ONNX
3. Deploy to:
   - ONNX Runtime (cloud servers)
   - TensorRT (NVIDIA GPUs)
   - Core ML (Apple devices)
   - TFLite (Android)

**Why it works:** ONNX serves as universal intermediate format.

### 9.2 Pattern 2: SavedModel → TensorFlow Serving

**Scenario:** TensorFlow-native deployment

**Flow:**
1. Train in TensorFlow/Keras
2. Save as SavedModel
3. Deploy to TensorFlow Serving
4. Serve via REST/gRPC API

**Why it works:** Tight integration between TensorFlow and Serving.

### 9.3 Pattern 3: Checkpoint → State Dict → Optimized Runtime

**Scenario:** Large language models

**Flow:**
1. Train with periodic checkpoints (state_dict + optimizer state)
2. After training: save final state_dict only
3. Convert to inference-optimized format:
   - TensorRT-LLM for NVIDIA
   - vLLM for efficient serving
   - GGML/GGUF for CPU inference

**Why it works:** Different optimization needs for training vs. inference.

---

## 10. Connection to Tree Serialization (DSA Day 47)

The parallel between tree and model serialization is striking:

| Aspect | Tree Serialization | Model Serialization |
|--------|-------------------|---------------------|
| Structure | Node connections | Layer connections |
| Values | Node values | Weight matrices |
| Format | String/JSON | Protobuf/HDF5/Pickle |
| Challenge | Preserving structure | Preserving computation |
| Portability | Same code needed? | Same framework needed? |
| Graph representation | Pre-order/level-order | Computation graph |

Both solve the same fundamental problem: converting structured, pointer-based in-memory data to flat, portable formats.

---

## 11. Key Takeaways

1. **Model serialization is deployment's foundation.** A model that can't be properly saved is a model that can't scale.

2. **Two philosophies exist:** Code-based (weights only, needs original code) vs. graph-based (self-contained, portable).

3. **ONNX is the interoperability standard.** When you need to cross frameworks or platforms, ONNX is usually the answer.

4. **Serialization is an optimization opportunity.** Quantization, pruning, and graph optimization during export can dramatically improve inference performance.

5. **Version carefully.** Models evolve, frameworks update, and production depends on reproducibility. Track versions and dependencies meticulously.

6. **Security matters.** Pickle-based formats can execute code. Use SafeTensors or verified checksums for untrusted models.

7. **Match format to use case:** Training checkpoints, production serving, cross-platform deployment, and long-term archival have different requirements.

Model serialization bridges the gap between research and production. Master it, and you can take models from notebook experiments to serving millions of users.

---

**Originally published at:** [arunbaby.com/ml-system-design/0047-model-serialization](https://www.arunbaby.com/ml-system-design/0047-model-serialization/)

*If you found this helpful, consider sharing it with others who might benefit.*
