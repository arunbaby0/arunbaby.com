---
title: "Model Compression Techniques"
day: 41
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - optimization
  - edge-ai
  - quantization
  - pruning
  - distillation
difficulty: Hard
---

**"Fitting billion-parameter models into megabytes."**

## 1. The Compression Imperative

Modern deep learning models are massive:
*   **GPT-3:** 175B parameters = 700GB (FP32).
*   **BERT-Large:** 340M parameters = 1.3GB (FP32).
*   **ResNet-50:** 25M parameters = 100MB (FP32).

**Challenges:**
*   **Deployment:** Can't fit on mobile devices (limited RAM).
*   **Inference:** Slow on CPUs without GPU acceleration.
*   **Cost:** Cloud inference costs scale with model size.

**Goal:** Reduce model size by 10-100x while maintaining 95%+ accuracy.

## 2. Taxonomy of Compression Techniques

1.  **Quantization:** Reduce numerical precision (FP32 → INT8).
2.  **Pruning:** Remove redundant weights or neurons.
3.  **Knowledge Distillation:** Train a small model to mimic a large model.
4.  **Low-Rank Factorization:** Decompose weight matrices.
5.  **Neural Architecture Search (NAS):** Design efficient architectures.
6.  **Weight Sharing:** Cluster weights and share values.

## 3. Quantization

### 3.1. Post-Training Quantization (PTQ)

**Idea:** Convert a trained FP32 model to INT8 without retraining.

**Algorithm:**
1.  **Calibration:** Run the model on a small dataset (e.g., 1000 samples).
2.  **Collect Statistics:** Record min/max values of activations for each layer.
3.  **Compute Scale and Zero-Point:**
    *   $scale = \frac{max - min}{255}$ (for INT8).
    *   $zero\_point = -\frac{min}{scale}$.
4.  **Quantize Weights:**
    *   $W_{int8} = round(\frac{W_{fp32}}{scale} + zero\_point)$.
5.  **Dequantize for Inference:**
    *   $W_{fp32} = (W_{int8} - zero\_point) \times scale$.

**Result:** 4x memory reduction, 2-4x speedup on CPUs.

**Accuracy Drop:** Typically 0.5-2% for CNNs, 1-5% for Transformers.

### 3.2. Quantization-Aware Training (QAT)

**Idea:** Simulate quantization during training so the model learns to be robust to quantization noise.

**Algorithm:**
1.  Insert **Fake Quantization** nodes after each layer.
2.  During forward pass:
    *   Quantize activations: $a_{int8} = round(\frac{a_{fp32}}{scale})$.
    *   Dequantize: $a_{fp32} = a_{int8} \times scale$.
3.  During backward pass:
    *   Use **Straight-Through Estimator (STE):** Treat the `round()` function as identity for gradients.
4.  Train for a few epochs (fine-tuning).

**Result:** 1-2% better accuracy than PTQ.

### 3.3. Dynamic vs Static Quantization

**Static Quantization:**
*   Quantize both weights and activations.
*   Requires calibration dataset.
*   **Use Case:** Inference on edge devices.

**Dynamic Quantization:**
*   Quantize only weights. Activations remain FP32.
*   No calibration needed.
*   **Use Case:** NLP models (BERT) on CPUs.

## 4. Pruning

### 4.1. Unstructured Pruning

**Idea:** Remove individual weights with small magnitude.

**Algorithm (Magnitude-Based Pruning):**
1.  Train the model to convergence.
2.  Compute the magnitude of each weight: $|W_{ij}|$.
3.  Sort weights by magnitude.
4.  Set the smallest $p\%$ of weights to zero (e.g., $p = 90$).
5.  Fine-tune the pruned model.
6.  Repeat (iterative pruning).

**Result:** 90% sparsity with <1% accuracy drop.

**Challenge:** Sparse matrices are not efficiently supported on all hardware. Need specialized libraries (e.g., NVIDIA Sparse Tensor Cores).

### 4.2. Structured Pruning

**Idea:** Remove entire channels, filters, or attention heads.

**Algorithm:**
1.  Compute the importance of each filter (e.g., L1 norm of weights).
2.  Remove the least important filters.
3.  Fine-tune.

**Result:** 50% compression with minimal accuracy drop. Works on standard hardware.

### 4.3. Lottery Ticket Hypothesis

**Discovery (Frankle & Carbin, 2019):**
A randomly initialized network contains a "winning ticket" subnetwork that, when trained in isolation, can match the full network's accuracy.

**Algorithm:**
1.  Train the full network.
2.  Prune to sparsity $p\%$.
3.  **Rewind** weights to their initial values (not random re-initialization).
4.  Train the pruned network from the initial weights.

**Implication:** We can find small, trainable networks by pruning and rewinding.

## 5. Knowledge Distillation

**Idea:** Train a small "student" model to mimic a large "teacher" model.

**Algorithm:**
1.  **Teacher:** Train a large, accurate model (e.g., BERT-Large).
2.  **Student:** Define a smaller model (e.g., BERT-Tiny, 10x smaller).
3.  **Distillation Loss:**
    *   **Soft Targets:** Use the teacher's softmax probabilities (not hard labels).
    *   $L_{distill} = KL(P_{teacher} || P_{student})$.
    *   **Temperature Scaling:** $P_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$ where $T > 1$ (e.g., $T = 3$). Higher temperature makes the distribution "softer" (less peaked), revealing more information about the teacher's uncertainty.
4.  **Combined Loss:**
    *   $L = \alpha L_{CE}(y, P_{student}) + (1 - \alpha) L_{distill}$.
5.  Train the student on the same dataset.

**Result:** Student achieves 95-98% of teacher's accuracy with 10x fewer parameters.

### 5.1. DistilBERT

**Architecture:**
*   6 layers (vs 12 in BERT-Base).
*   Hidden size 768 (same as BERT).
*   40% fewer parameters.

**Training:**
*   Distilled from BERT-Base.
*   Triple loss: Distillation + Masked LM + Cosine Embedding (hidden states).

**Result:**
*   97% of BERT-Base accuracy on GLUE.
*   60% faster inference.

## 6. Low-Rank Factorization

**Idea:** Decompose a weight matrix $W \in \mathbb{R}^{m \times n}$ into $W = U V^T$ where $U \in \mathbb{R}^{m \times r}$ and $V \in \mathbb{R}^{n \times r}$ with $r \ll \min(m, n)$.

**Benefit:** Reduce parameters from $m \times n$ to $r(m + n)$.

**Algorithm (SVD):**
1.  Compute Singular Value Decomposition: $W = U \Sigma V^T$.
2.  Keep only the top $r$ singular values.
3.  $W_{approx} = U_r \Sigma_r V_r^T$.

**Use Case:** Compressing the embedding layer in NLP models.

## 7. System Design: On-Device Inference Pipeline

**Scenario:** Deploy a BERT model on a smartphone for real-time text classification.

**Constraints:**
*   **Model Size:** < 50MB.
*   **Latency:** < 100ms per inference.
*   **Power:** < 500mW.

**Solution:**

**Step 1: Compression**
*   **Distillation:** BERT-Base (110M params) → DistilBERT (66M params).
*   **Quantization:** FP32 → INT8 (4x reduction).
*   **Final Size:** 66M params × 1 byte = 66MB → 50MB after compression.

**Step 2: Optimization**
*   **ONNX Runtime:** Convert PyTorch model to ONNX for optimized inference.
*   **Operator Fusion:** Fuse LayerNorm + GELU into a single kernel.
*   **Graph Optimization:** Remove redundant nodes.

**Step 3: Hardware Acceleration**
*   **Android:** Use NNAPI (Neural Networks API) to leverage GPU/DSP.
*   **iOS:** Use Core ML with ANE (Apple Neural Engine).

**Step 4: Caching**
*   Cache embeddings for frequently seen inputs.

## 8. Deep Dive: Mixed-Precision Quantization

Not all layers need the same precision. Sensitive layers (e.g., first/last layer) can remain FP16, while others are INT8.

**Algorithm:**
1.  **Sensitivity Analysis:** For each layer, measure accuracy drop when quantized to INT8.
2.  **Pareto Frontier:** Find the set of layer precisions that maximize accuracy for a given model size.
3.  **AutoML:** Use Neural Architecture Search to find the optimal precision for each layer.

**Example (MobileNetV2):**
*   First layer: FP16 (sensitive to quantization).
*   Middle layers: INT8.
*   Last layer: FP16 (classification head).

## 9. Deep Dive: Gradient Compression for Distributed Training

In distributed training, gradients are communicated across GPUs. Compressing gradients reduces bandwidth.

**Techniques:**

**1. Gradient Sparsification:**
*   Send only the top-k largest gradients.
*   Accumulate the rest locally.
*   **Result:** 99% compression with minimal accuracy drop.

**2. Gradient Quantization:**
*   Quantize gradients to 8-bit or even 1-bit.
*   **1-bit SGD:** Send only the sign of the gradient.

**3. Error Feedback:**
*   Track the quantization error and add it to the next gradient.
*   Ensures convergence despite lossy compression.

## 10. Case Study: TensorFlow Lite

**Goal:** Run TensorFlow models on mobile and embedded devices.

**Features:**
*   **Converter:** Converts TensorFlow/Keras models to `.tflite` format.
*   **Quantization:** Built-in PTQ and QAT.
*   **Optimizations:** Operator fusion, constant folding.
*   **Delegates:** Hardware acceleration (GPU, DSP, NPU).

**Example:**
```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('model.h5')

# Convert to TFLite with INT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Calibration
def representative_dataset():
    for data in calibration_data:
        yield [data]

converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## 11. Interview Questions

1.  **Explain Quantization.** What is the difference between PTQ and QAT?
2.  **Knowledge Distillation.** Why use soft targets instead of hard labels?
3.  **Pruning.** What is the Lottery Ticket Hypothesis?
4.  **Trade-offs.** Quantization vs Pruning vs Distillation. When to use which?
5.  **Calculate Compression Ratio.** A model has 100M parameters. After 90% pruning and INT8 quantization, what is the final size?
    *   Original: 100M × 4 bytes = 400MB.
    *   After pruning: 10M params.
    *   After quantization: 10M × 1 byte = 10MB.
    *   **Compression:** 40x.

## 12. Common Pitfalls

*   **Quantizing Batch Norm:** Batch Norm layers should be fused with Conv layers before quantization.
*   **Calibration Data:** Using too little calibration data leads to poor quantization ranges.
*   **Pruning Ratio:** Pruning too aggressively (>95%) often causes unrecoverable accuracy loss.
*   **Distillation Temperature:** Too high ($T > 10$) makes the soft targets too uniform (no information). Too low ($T = 1$) is equivalent to hard labels.

## 13. Hardware-Specific Optimizations

### 13.1. ARM NEON (Mobile CPUs)

**NEON** is ARM's SIMD (Single Instruction Multiple Data) instruction set.

**Optimization:**
*   **INT8 GEMM:** Matrix multiplication using 8-bit integers.
*   **Vectorization:** Process 16 INT8 values in parallel.
*   **Fused Operations:** Combine Conv + ReLU + Quantization in a single kernel.

**Performance Gain:** 4-8x speedup over FP32 on ARM Cortex-A76.

### 13.2. NVIDIA Tensor Cores

**Tensor Cores** are specialized hardware for matrix multiplication.

**Supported Precisions:**
*   **FP16:** 312 TFLOPS on A100.
*   **INT8:** 624 TOPS (Tera Operations Per Second).
*   **INT4:** 1248 TOPS.

**Optimization:**
*   Use **Automatic Mixed Precision (AMP)** in PyTorch.
*   Ensure matrix dimensions are multiples of 8 (for INT8) or 16 (for FP16).

### 13.3. Google TPU (Tensor Processing Unit)

**TPU v4:**
*   **Precision:** BF16 (Brain Float 16).
*   **Systolic Array:** Optimized for matrix multiplications.
*   **Memory:** 32GB HBM (High Bandwidth Memory).

**Optimization:**
*   Use **XLA (Accelerated Linear Algebra)** compiler for graph optimization.
*   Batch size should be large (>128) to saturate the TPU.

## 14. Production Case Study: BERT Compression for Search

**Scenario:** Deploy BERT for semantic search at Google scale (billions of queries/day).

**Challenges:**
*   **Latency:** Must respond in <50ms.
*   **Cost:** Running BERT-Base on every query costs millions/day.

**Solution:**

**Step 1: Distillation**
*   BERT-Base (110M params) → DistilBERT (66M params).
*   **Result:** 40% smaller, 60% faster, 97% accuracy.

**Step 2: Quantization**
*   DistilBERT FP32 → INT8.
*   **Result:** 4x smaller (66MB → 16MB), 2x faster.

**Step 3: Pruning**
*   Structured pruning: Remove 30% of attention heads.
*   **Result:** 50MB → 35MB, 1.2x faster.

**Step 4: Caching**
*   Cache embeddings for top 1M queries.
*   **Hit Rate:** 40% (reduces compute by 40%).

**Final Result:**
*   **Latency:** 15ms (vs 80ms for BERT-Base).
*   **Cost:** $100K/day (vs $1M/day).
*   **Accuracy:** 96% of BERT-Base.

## 15. Advanced Technique: Neural Architecture Search for Compression

**Problem:** Manually designing compressed architectures is tedious.

**Solution:** Use NAS to automatically find efficient architectures.

**Approaches:**

**1. Once-for-All (OFA) Networks:**
*   Train a single "super-network" that contains all possible sub-networks.
*   At deployment, extract a sub-network that fits the target device.
*   **Benefit:** No need to retrain for each device.

**2. ProxylessNAS:**
*   Search directly on the target hardware (e.g., mobile phone).
*   **Objective:** Maximize accuracy subject to latency < 50ms.

**3. EfficientNet:**
*   Use compound scaling: scale depth, width, and resolution together.
*   **Result:** EfficientNet-B0 achieves ResNet-50 accuracy with 10x fewer parameters.

## 16. Compression Benchmarks

**Model:** ResNet-50 (25M params, 100MB FP32).

| Technique | Size (MB) | Accuracy (ImageNet) | Speedup (CPU) |
|-----------|-----------|---------------------|---------------|
| Baseline (FP32) | 100 | 76.1% | 1x |
| INT8 Quantization | 25 | 75.8% | 3x |
| 50% Pruning | 50 | 75.5% | 1.5x |
| Distillation (ResNet-18) | 45 | 73.2% | 2x |
| Pruning + Quantization | 12.5 | 75.0% | 4x |
| Distillation + Quantization | 11 | 72.8% | 5x |

**Observation:** Combining techniques yields the best compression ratio.

## 17. Deep Dive: Weight Clustering

**Idea:** Cluster weights into $K$ centroids (e.g., 256). Store only the centroid index (8 bits) instead of the full weight (32 bits).

**Algorithm (K-Means):**
1.  Flatten all weights into a 1D array.
2.  Run K-Means clustering with $K = 256$.
3.  Replace each weight with its cluster centroid.
4.  Store: (1) Codebook (256 centroids), (2) Indices (8 bits per weight).

**Compression Ratio:**
*   Original: 32 bits/weight.
*   Compressed: 8 bits/weight + codebook overhead.
*   **Result:** ~4x compression.

**Accuracy:** Typically <1% drop for $K = 256$.

## 18. Deep Dive: Dynamic Neural Networks

**Idea:** Adapt the model size based on input complexity.

**Techniques:**

**1. Early Exit:**
*   Add intermediate classifiers at different layers.
*   For easy inputs, exit early (use fewer layers).
*   For hard inputs, use the full network.
*   **Example:** BranchyNet, MSDNet.

**2. Adaptive Computation Time (ACT):**
*   Each layer decides whether to continue processing or stop.
*   **Benefit:** Variable compute based on input.

**3. Slimmable Networks:**
*   Train a single network that can run at different widths (e.g., 0.25x, 0.5x, 0.75x, 1x).
*   At runtime, choose the width based on available resources.

## 19. Production Deployment: Model Serving

**Scenario:** Serve a compressed model in production.

**Architecture:**

**1. Model Repository:**
*   Store compressed models in S3/GCS.
*   Version control (v1, v2, v3).

**2. Model Server:**
*   **TensorFlow Serving:** Supports TFLite models.
*   **TorchServe:** Supports quantized PyTorch models.
*   **ONNX Runtime:** Cross-framework support.

**3. Load Balancer:**
*   Distribute requests across multiple model servers.

**4. Monitoring:**
*   Track latency (P50, P95, P99).
*   Track accuracy (A/B test compressed vs full model).
*   Track resource usage (CPU, memory).

**5. Auto-Scaling:**
*   Scale up during peak hours.
*   Scale down during off-peak to save cost.

## 20. Cost-Benefit Analysis

**Scenario:** Deploying a compressed model for image classification (1M requests/day).

**Baseline (FP32 ResNet-50):**
*   **Latency:** 100ms/request.
*   **Compute:** 1M requests × 100ms = 100K seconds/day = 28 hours/day.
*   **Cost:** 28 hours × $0.10/hour (AWS EC2 c5.xlarge) = $2.80/day = $1,022/year.

**Compressed (INT8 ResNet-50):**
*   **Latency:** 30ms/request (3x faster).
*   **Compute:** 1M requests × 30ms = 30K seconds/day = 8.3 hours/day.
*   **Cost:** 8.3 hours × $0.10/hour = $0.83/day = $303/year.

**Savings:** $719/year (70% cost reduction).

**Accuracy Trade-off:** 76.1% → 75.8% (0.3% drop).

**ROI:** If the accuracy drop is acceptable, compression is a no-brainer.

## 21. Ethical Considerations

**Bias Amplification:**
*   Compression can amplify biases in the training data.
*   **Example:** A pruned model might be less accurate on underrepresented groups.
*   **Solution:** Evaluate fairness metrics (e.g., demographic parity) after compression.

**Environmental Impact:**
*   Training large models consumes energy (carbon footprint).
*   Compression reduces inference energy, but the compression process itself (e.g., NAS) can be energy-intensive.
*   **Solution:** Use efficient compression methods (PTQ instead of NAS).

## 22. Future Trends

**1. Extreme Quantization:**
*   **Binary Neural Networks (BNNs):** 1-bit weights and activations.
*   **Ternary Neural Networks (TNNs):** Weights in {-1, 0, 1}.
*   **Challenge:** Significant accuracy drop (5-10%).

**2. Hardware-Software Co-Design:**
*   Design hardware specifically for compressed models.
*   **Example:** Google Edge TPU optimized for INT8.

**3. On-Device Learning:**
*   Fine-tune compressed models on-device using user data.
*   **Challenge:** Privacy (Federated Learning).

## 23. Conclusion

Model compression is essential for deploying deep learning at scale. The key is to combine multiple techniques (quantization + pruning + distillation) to achieve the best compression ratio while maintaining acceptable accuracy.

**Key Takeaways:**
*   **Quantization:** 4x compression with minimal accuracy drop.
*   **Pruning:** 50-90% sparsity, but requires specialized hardware for speedup.
*   **Distillation:** 10x compression, but requires retraining.
*   **Hardware Matters:** Optimize for the target device (ARM, NVIDIA, TPU).
*   **Production:** Monitor latency, accuracy, and cost.

The future of AI is edge AI. As models grow larger, compression will become even more critical. Mastering these techniques is a must for ML engineers.

## 24. Deep Dive: ONNX (Open Neural Network Exchange)

**Problem:** Models trained in PyTorch can't run on TensorFlow Serving. Models trained in TensorFlow can't run on ONNX Runtime.

**Solution:** ONNX is a universal format for neural networks.

**Workflow:**
1.  **Train** in PyTorch/TensorFlow/Keras.
2.  **Export** to ONNX format.
3.  **Optimize** using ONNX Runtime.
4.  **Deploy** on any platform (mobile, edge, cloud).

**Example (PyTorch → ONNX):**
```python
import torch
import torch.onnx

# Load PyTorch model
model = torch.load('model.pth')
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

**ONNX Runtime Optimizations:**
*   **Graph Optimization:** Fuse operators, eliminate redundant nodes.
*   **Quantization:** INT8 quantization.
*   **Execution Providers:** CPU, CUDA, TensorRT, DirectML, CoreML.

## 25. Mobile Deployment: iOS (Core ML)

**Core ML** is Apple's framework for on-device ML.

**Workflow:**
1.  Train model in PyTorch/TensorFlow.
2.  Convert to Core ML format (`.mlmodel`).
3.  Integrate into iOS app.

**Conversion (PyTorch → Core ML):**
```python
import coremltools as ct
import torch

# Load PyTorch model
model = torch.load('model.pth')
model.eval()

# Trace the model
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Convert to Core ML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input", shape=(1, 3, 224, 224))],
    convert_to="neuralnetwork"  # or "mlprogram" for newer format
)

# Save
mlmodel.save("model.mlmodel")
```

**Optimization:**
*   **Quantization:** Use `ct.compression.quantize_weights()`.
*   **Pruning:** Use `ct.compression.prune_weights()`.
*   **Neural Engine:** Optimize for Apple's ANE (Neural Engine).

## 26. Mobile Deployment: Android (TensorFlow Lite)

**TensorFlow Lite** is Google's framework for mobile/edge ML.

**Workflow:**
1.  Train model in TensorFlow/Keras.
2.  Convert to TFLite format (`.tflite`).
3.  Integrate into Android app.

**Conversion:**
```python
import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model('model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

**Optimization:**
*   **GPU Delegate:** Use GPU for inference.
*   **NNAPI Delegate:** Use Android's Neural Networks API.
*   **Hexagon Delegate:** Use Qualcomm's DSP.

## 27. Advanced Technique: Sparse Tensor Cores (NVIDIA)

**NVIDIA Ampere** (A100, RTX 3090) introduced **Sparse Tensor Cores** that accelerate 2:4 structured sparsity.

**2:4 Sparsity:** In every 4 consecutive weights, at least 2 must be zero.

**Example:**
```
Original: [0.5, 0.3, 0.1, 0.2]
2:4 Sparse: [0.5, 0.3, 0.0, 0.0]  ✓ (2 zeros out of 4)
```

**Speedup:** 2x faster than dense operations on Sparse Tensor Cores.

**Training with 2:4 Sparsity:**
```python
import torch
from torch.ao.pruning import prune_to_structured_sparsity

model = MyModel()
prune_to_structured_sparsity(model, sparsity_pattern="2:4")
# Train normally
```

## 28. Deep Dive: Huffman Coding for Weight Compression

**Idea:** Use variable-length encoding for weights. Frequent weights get shorter codes.

**Algorithm:**
1.  **Cluster** weights into $K$ centroids (e.g., 256).
2.  **Count** frequency of each centroid.
3.  **Build Huffman Tree:** Assign shorter codes to frequent centroids.
4.  **Encode:** Replace each weight with its Huffman code.

**Compression Ratio:**
*   Fixed-length (8 bits): 8 bits/weight.
*   Huffman: ~5-6 bits/weight (depending on distribution).

**Trade-off:** Decoding overhead during inference.

## 29. Production Case Study: MobileNet Deployment

**Scenario:** Deploy MobileNetV2 for image classification on a smartphone.

**Baseline (FP32):**
*   **Size:** 14MB.
*   **Latency:** 80ms (CPU).
*   **Accuracy:** 72% (ImageNet).

**Optimization Pipeline:**

**Step 1: Quantization (INT8)**
*   **Size:** 3.5MB (4x reduction).
*   **Latency:** 25ms (3x speedup).
*   **Accuracy:** 71.5% (0.5% drop).

**Step 2: Pruning (50%)**
*   **Size:** 1.75MB (2x reduction).
*   **Latency:** 15ms (1.6x speedup).
*   **Accuracy:** 70.8% (0.7% drop).

**Step 3: Knowledge Distillation (MobileNetV2 → MobileNetV3-Small)**
*   **Size:** 1.2MB.
*   **Latency:** 10ms.
*   **Accuracy:** 68% (4% drop from baseline).

**Final Result:**
*   **Size:** 1.2MB (12x compression).
*   **Latency:** 10ms (8x speedup).
*   **Accuracy:** 68% (acceptable for mobile use case).

## 30. Advanced Technique: Neural ODE Compression

**Neural ODEs** (Ordinary Differential Equations) model continuous transformations.

**Compression:**
*   Instead of storing weights for 100 layers, store the ODE parameters.
*   **Benefit:** Constant memory regardless of depth.
*   **Trade-off:** Slower inference (need to solve ODE).

**Use Case:** Compressing very deep networks (ResNet-1000).

## 31. Monitoring Compressed Models in Production

**Metrics to Track:**
1.  **Latency Distribution:** P50, P95, P99. Alert if P95 > SLA.
2.  **Accuracy Drift:** Compare predictions with a "shadow" full-precision model.
3.  **Resource Usage:** CPU, memory, battery drain (for mobile).
4.  **Error Analysis:** Which classes have the highest error rate after compression?

**A/B Testing:**
*   **Control:** Full-precision model (10% of traffic).
*   **Treatment:** Compressed model (90% of traffic).
*   **Metrics:** Latency, accuracy, user engagement.

**Rollback Strategy:**
*   If compressed model's accuracy drops > 2%, automatically rollback to full-precision.

## 32. Interview Deep Dive: Compression Trade-offs

**Q: When would you use quantization vs pruning vs distillation?**

**A:**
*   **Quantization:** When you need fast inference on CPUs/mobile. Works well for CNNs. Minimal accuracy drop.
*   **Pruning:** When you have specialized hardware (Sparse Tensor Cores) or can tolerate irregular sparsity. Best for very large models.
*   **Distillation:** When you can afford to retrain. Best for transferring knowledge from an ensemble to a single model. Works well for NLP.

**Q: How do you choose the quantization precision (INT8 vs INT4)?**

**A:**
*   **INT8:** Standard. 4x compression, <1% accuracy drop for most models.
*   **INT4:** Aggressive. 8x compression, but 2-5% accuracy drop. Use only if latency is critical and accuracy drop is acceptable.
*   **Mixed Precision:** Use INT4 for less sensitive layers, INT8 for sensitive layers.

## 33. Future Research Directions

**1. Learned Compression:**
*   Use a neural network to learn the optimal compression strategy for each layer.
*   **Example:** AutoML for compression.

**2. Post-Training Sparsification:**
*   Prune without retraining (like PTQ for quantization).
*   **Challenge:** Maintaining accuracy.

**3. Hardware-Aware Compression:**
*   Compress specifically for the target hardware (e.g., iPhone 15 Pro's A17 chip).
*   **Benefit:** Maximize performance on that specific device.

## 34. Conclusion & Best Practices

**Best Practices:**
1.  **Start with Quantization:** Easiest, fastest, minimal accuracy drop.
2.  **Combine Techniques:** Quantization + Pruning + Distillation for maximum compression.
3.  **Profile First:** Measure latency bottlenecks before optimizing.
4.  **Test on Target Device:** Don't rely on desktop benchmarks.
5.  **Monitor in Production:** Track accuracy drift and latency.

**Compression Checklist:**
- [ ] Benchmark baseline model (size, latency, accuracy)
- [ ] Apply INT8 quantization (PTQ or QAT)
- [ ] Measure accuracy drop (<2% acceptable)
- [ ] Apply structured pruning (30-50%)
- [ ] Fine-tune pruned model
- [ ] Consider distillation if retraining is feasible
- [ ] Convert to target format (ONNX, TFLite, Core ML)
- [ ] Test on target hardware
- [ ] Deploy with monitoring
- [ ] Set up A/B test
- [ ] Monitor and iterate

The art of model compression is balancing the three-way trade-off: **Size, Speed, Accuracy**. There's no one-size-fits-all solution. The best approach depends on your use case, hardware, and constraints. Mastering these techniques will make you indispensable in the era of edge AI.



---

**Originally published at:** [arunbaby.com/ml-system-design/0041-model-compression](https://www.arunbaby.com/ml-system-design/0041-model-compression/)

*If you found this helpful, consider sharing it with others who might benefit.*

