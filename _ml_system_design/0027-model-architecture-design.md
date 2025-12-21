---
title: "Model Architecture Design"
day: 27
collection: ml_system_design
categories:
  - ml-system-design
  - deep-learning
tags:
  - neural-networks
  - transformers
  - cnn
  - nas
  - architecture-search
subdomain: "Model Design"
tech_stack: [PyTorch, TensorFlow, Keras, AutoKeras]
scale: "From MobileNet to GPT-4"
companies: [Google, Meta, OpenAI, DeepMind, NVIDIA]
related_dsa_day: 27
related_speech_day: 27
related_agents_day: 27
---

**Architecture is destiny. The difference between 50% accuracy and 90% accuracy is often just a skip connection.**

## Problem Statement

You are given a dataset and a task (e.g., "Classify these 1 million images" or "Translate this text").
**How do you design the Neural Network architecture?**
Do you use a CNN? A Transformer? How deep? How wide? Which activation function? Which normalization?

This post explores the **First Principles of Model Architecture Design**. We aren't just using `resnet50(pretrained=True)`. We are learning how to build `resnet50` from scratch, and why it was built that way.

## Understanding the Requirements

Designing a model is an engineering trade-off between three variables:
1.  **Capacity (Accuracy):** Can the model learn the complex patterns in the data?
2.  **Compute (FLOPs/Latency):** Can it run fast enough on the target hardware?
3.  **Memory (Parameters):** Does it fit in VRAM?

### The "Inductive Bias"
Every architecture makes assumptions about the data:
-   **Fully Connected (MLP):** No assumptions. Every input relates to every other input. (Data inefficient).
-   **CNN:** Assumes **Locality** (pixels nearby matter) and **Translation Invariance** (a cat is a cat whether it's in the top-left or bottom-right).
-   **RNN:** Assumes **Sequentiality** (time matters).
-   **Transformer:** Assumes **Pairwise Relationships** (Attention) matter most, regardless of distance.

## High-Level Architecture: The Building Blocks

A modern Deep Learning model is like a Lego castle built from a few fundamental blocks.

```ascii
+---------------------------------------------------------------+
|                        The Model Head                         |
|             (Classifier / Regressor / Decoder)                |
+---------------------------------------------------------------+
                                ^
                                |
+---------------------------------------------------------------+
|                       The Backbone                            |
|               (Feature Extractor / Encoder)                   |
|                                                               |
|  +-------+    +-------+    +-------+    +-------+             |
|  | Block | -> | Block | -> | Block | -> | Block |             |
|  +-------+    +-------+    +-------+    +-------+             |
+---------------------------------------------------------------+
                                ^
                                |
+---------------------------------------------------------------+
|                        The Stem                               |
|              (Initial Convolution / Embedding)                |
+---------------------------------------------------------------+
```

### 1. The Stem
The entry point. It transforms raw data (pixels, text tokens) into the embedding space.
-   **Images:** Usually a `7x7 Conv, stride 2` to reduce resolution early (ResNet).
-   **Text:** A Lookup Table (`nn.Embedding`) + Positional Encodings.

### 2. The Backbone (The Body)
This is 90% of the compute. It consists of repeated **Blocks**.
-   **ResNet Block:** `Conv -> BN -> ReLU -> Conv -> BN -> Add Input`.
-   **Transformer Block:** `LayerNorm -> Attention -> Add -> LayerNorm -> MLP -> Add`.

### 3. The Head
The task-specific output.
-   **Classification:** `GlobalAveragePooling -> Linear -> Softmax`.
-   **Detection:** `Conv` layers predicting Bounding Boxes.
-   **Segmentation:** Upsampling layers to restore resolution.

## The Evolution of Architectures: A Historical Perspective

To understand *why* we use ResNets and Transformers today, we must understand the failures of the past.

### 1. The Dark Ages (Pre-2012)
Neural Networks were "Multi-Layer Perceptrons" (MLPs).
-   **Structure:** Dense Matrix Multiplications.
-   **Problem:** No translation invariance. A cat in the top-left corner required different weights than a cat in the bottom-right.
-   **Result:** Couldn't scale to images larger than 28x28 (MNIST).

### 2. The AlexNet Revolution (2012)
Alex Krizhevsky used GPUs to train a deep CNN.
-   **Key Innovation:** ReLU (instead of Sigmoid) to fix vanishing gradients. Dropout to fix overfitting.
-   **Architecture:** 5 Conv layers, 3 Dense layers.
-   **Impact:** Error rate on ImageNet dropped from 26% to 15%.

### 3. The VGG Era (2014)
"Simplicity is the ultimate sophistication."
-   **Idea:** Replace large kernels (11x11, 5x5) with stacks of 3x3 kernels.
-   **Why?** Two 3x3 layers have the same receptive field as one 5x5 layer but fewer parameters and more non-linearity.
-   **Legacy:** The "VGG Backbone" is still used in Transfer Learning.

### 4. The ResNet Breakthrough (2015)
"Deep networks are harder to train."
-   **Problem:** Adding layers made performance *worse* due to optimization difficulties (not overfitting).
-   **Solution:** Residual Connections (`x + F(x)`).
-   **Result:** We could train 100+ layer networks.

### 5. The Transformer Invasion (2017-Present)
"Attention is All You Need."
-   **Shift:** Inductive bias of "Locality" (CNNs) was replaced by "Global Correlation" (Attention).
-   **Vision Transformers (ViT):** Treat an image as a sequence of 16x16 patches.
-   **Dominance:** Transformers now rule NLP (GPT), Vision (ViT), and Speech (Conformer).

## Deep Dive: Normalization Layers

Normalization is the unsung hero of Deep Learning. It smooths the loss landscape, allowing larger learning rates.

### 1. Batch Normalization (BN)
\[ \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta \]
-   **Mechanism:** Compute mean/var across the **Batch (N)** and **Spatial (H, W)** dimensions.
-   **Training:** Uses current batch stats.
-   **Inference:** Uses running average stats.
-   **Pros:** Fuses into Convolution (free at inference).
-   **Cons:**
    -   Requires large batch size (>32).
    -   Fails in RNNs (sequence length varies).
    -   Training/Inference discrepancy causes bugs.

### 2. Layer Normalization (LN)
-   **Mechanism:** Compute mean/var across the **Channel (C)** dimension for a *single sample*.
-   **Pros:** Independent of batch size. Works great for RNNs/Transformers.
-   **Cons:** Cannot be fused. Slower inference.

### 3. Group Normalization (GN)
-   **Mechanism:** Split channels into \(G\) groups. Normalize within each group.
-   **Use Case:** Object Detection (where batch size is small, e.g., 1 or 2).
-   **Performance:** Better than BN at small batch sizes, worse at large batch sizes.

### 4. Instance Normalization (IN)
-   **Mechanism:** Normalize each channel independently.
-   **Use Case:** Style Transfer. It removes "contrast" information (style) while keeping content.

## Deep Dive: Activation Functions

The non-linearity is what gives NNs their power.

### 1. Sigmoid / Tanh
-   **Formula:** \(\sigma(x) = \frac{1}{1+e^{-x}}\)
-   **Problem:** **Vanishing Gradient.** For large \(x\), the gradient is 0. The network stops learning.
-   **Status:** Deprecated for hidden layers. Used only for output (Probability).

### 2. ReLU (Rectified Linear Unit)
-   **Formula:** \(\max(0, x)\)
-   **Pros:** Computationally free. No vanishing gradient for \(x > 0\).
-   **Cons:** **Dead ReLU.** If \(x < 0\) always, the neuron dies and never recovers.

### 3. Leaky ReLU / PReLU
-   **Formula:** \(\max(\alpha x, x)\) where \(\alpha \approx 0.01\).
-   **Fix:** Allows a small gradient to flow when \(x < 0\), reviving dead neurons.

### 4. GeLU (Gaussian Error Linear Unit)
-   **Formula:** \(x \cdot \Phi(x)\) (approx \(x \cdot \sigma(1.702x)\)).
-   **Intuition:** A smooth version of ReLU.
-   **Why?** The smoothness helps optimization in very deep Transformers (BERT, GPT).

### 5. Swish / SiLU
-   **Formula:** \(x \cdot \sigma(x)\).
-   **Origin:** Discovered by Google using Neural Architecture Search.
-   **Properties:** Non-monotonic. It dips slightly below 0 for negative values. This "self-gating" property helps information flow.

## Component Deep-Dives

### 1. Convolutions: The Workhorse of Vision
Standard Convolutions are expensive: \(O(K^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W)\).

**Optimizations:**
-   **Depthwise Separable Conv (MobileNet):**
    1.  **Depthwise:** Spatial convolution per channel.
    2.  **Pointwise:** 1x1 convolution to mix channels.
    -   Reduces parameters by ~9x.
-   **Dilated Conv (Atrous):** Increases Receptive Field without reducing resolution. Great for Segmentation.

### 2. Attention: The Global Context
Self-Attention calculates the relationship between every pair of tokens.
\[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]
-   **Pros:** Infinite Receptive Field.
-   **Cons:** \(O(N^2)\) complexity. Hard to scale to long sequences.

### 3. Normalization: The Stabilizer
Normalization ensures the activations have mean 0 and variance 1. This prevents exploding/vanishing gradients.
-   **Batch Norm (BN):** Normalize across the batch dimension.
    -   *Pros:* Fuses into Conv during inference (free!).
    -   *Cons:* Fails with small batch sizes.
-   **Layer Norm (LN):** Normalize across the channel dimension.
    -   *Pros:* Batch size independent. Standard for Transformers/RNNs.
-   **RMSNorm (Root Mean Square Norm):** Like LN but skips mean subtraction. Faster. Used in LLaMA.

### 4. Activations: The Non-Linearity
-   **ReLU:** `max(0, x)`. The classic. Fast. Dead ReLU problem.
-   **LeakyReLU:** `max(0.01x, x)`. Fixes dead ReLU.
-   **GeLU (Gaussian Error Linear Unit):** Smooth approximation of ReLU. Standard in BERT/GPT.
-   **Swish (SiLU):** `x * sigmoid(x)`. Discovered by NAS. Used in EfficientNet/LLaMA.

## Deep Dive: Vision Transformers (ViT)

The Transformer changed everything. But how do you feed an image (2D) into a model designed for text (1D)?

### 1. Patch Embedding
-   **Concept:** Break the image into fixed-size patches (e.g., 16x16 pixels).
-   **Linear Projection:** Flatten each patch (16x16x3 = 768) and map it to a vector of size \(D\).
-   **Result:** An image of 224x224 becomes a sequence of 196 tokens (14x14 patches).

### 2. The CLS Token
-   **Problem:** In BERT, we use a special `[CLS]` token to aggregate sentence-level information.
-   **ViT:** We prepend a learnable `[CLS]` token to the patch sequence.
-   **Output:** The state of the `[CLS]` token at the final layer serves as the image representation.

### 3. Positional Embeddings
-   **Problem:** Self-Attention is permutation invariant. It doesn't know that Patch 1 is next to Patch 2.
-   **Solution:** Add learnable position vectors to each patch embedding.
-   **1D vs 2D:** Surprisingly, standard 1D learnable embeddings work as well as 2D grid embeddings. The model learns the grid structure on its own.

### 4. Inductive Bias vs. Data
-   **CNNs:** Have strong inductive bias (Locality, Translation Invariance). They work well on small data.
-   **ViT:** Has weak inductive bias. It assumes nothing. It needs **massive data** (JFT-300M) to learn that "pixels nearby are related".
-   **DeiT (Data-efficient Image Transformers):** Uses Distillation to train ViTs on ImageNet without extra data.

## Deep Dive: MobileNet and Efficient Architecture

Not everyone has an A100. How do we run models on phones?

### 1. Depthwise Separable Convolutions
Standard Conv: \(K \times K \times C_{in} \times C_{out}\) parameters.
Depthwise Separable:
1.  **Depthwise:** \(K \times K \times 1 \times C_{in}\). (Spatial mixing).
2.  **Pointwise:** \(1 \times 1 \times C_{in} \times C_{out}\). (Channel mixing).
**Reduction:** \(\frac{1}{C_{out}} + \frac{1}{K^2}\). For 3x3 kernels, it's ~8-9x fewer FLOPs.

### 2. Inverted Residuals (MobileNetV2)
-   **ResNet:** Wide -> Narrow -> Wide. (Bottleneck).
-   **MobileNetV2:** Narrow -> Wide -> Narrow.
-   **Why?** We expand the low-dimensional manifold into high dimensions to apply non-linearity (ReLU), then project back.
-   **Linear Bottlenecks:** The last 1x1 projection has **No ReLU**. Why? ReLU destroys information in low dimensions.

### 3. Squeeze-and-Excitation (SE)
MobileNetV3 added SE blocks. They are cheap (parameters) but powerful.
They allow the model to say "This channel (e.g., 'fur detector') is important for this image, but that channel ('wheel detector') is not."

## Deep Dive: Neural Architecture Search (NAS)

Designing architectures by hand is tedious. Let's automate it.

### 1. Reinforcement Learning (NASNet)
-   **Agent:** An RNN controller.
-   **Action:** Generate a string describing a layer (e.g., "Conv 3x3, ReLU").
-   **Environment:** Train the child network for 5 epochs.
-   **Reward:** Validation Accuracy.
-   **Cost:** 2000 GPU-days. (Expensive!).

### 2. Evolutionary Algorithms (AmoebaNet)
-   **Population:** A set of architectures.
-   **Mutation:** Randomly change one operation (e.g., 3x3 -> 5x5).
-   **Selection:** Train and keep the best. Kill the worst.
-   **Result:** AmoebaNet matched NASNet with less compute.

### 3. Differentiable NAS (DARTS)
-   **Relaxation:** Instead of choosing *one* operation, compute a weighted sum of *all* operations.
    \[ \bar{o}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o)}{\sum \exp(\alpha_{o'})} o(x) \]
-   **Bilevel Optimization:**
    1.  Update weights \(w\) to minimize Train Loss.
    2.  Update architecture alphas \(\alpha\) to minimize Val Loss.
-   **Cost:** 4 GPU-days.

## Design Patterns in Architecture

### 1. Residual Connections (Skip Connections)
**Problem:** In deep networks (e.g., 20 layers), gradients vanish during backpropagation. The signal degrades.
**Solution:** Add the input to the output: `y = F(x) + x`.
**Why it works:** It creates a "gradient superhighway". The gradient can flow unchanged through the `+ x` path. This allowed ResNet to go from 20 layers to 152 layers.

### 2. The Bottleneck Design
**Problem:** 3x3 Convolutions on high-dimensional channels (e.g., 256) are expensive.
**Solution:**
1.  **1x1 Conv:** Reduce channels (256 -> 64).
2.  **3x3 Conv:** Process spatial features on low channels (64).
3.  **1x1 Conv:** Expand channels back (64 -> 256).
**Result:** 10x fewer parameters for the same depth.

### 3. Squeeze-and-Excitation (SE)
**Idea:** Not all channels are important. Let the network learn to weight them.
1.  **Squeeze:** Global Average Pooling to get a 1x1xC vector.
2.  **Excite:** A small MLP learns a weight for each channel (sigmoid).
3.  **Scale:** Multiply the original feature map by these weights.
**Result:** 1-2% accuracy boost for negligible compute.

## Implementation: Building a Modern ResNet Block

Let's implement a "Pre-Activation" ResNet block with Squeeze-and-Excitation in PyTorch.

```python
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Bottleneck design
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)
        
        # Shortcut handling (if dimensions change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply SE Attention
        out = self.se(out)
        
        out += residual
        out = self.relu(out)
        return out

# Test
x = torch.randn(2, 64, 32, 32)
block = ResNetBlock(64, 256)
y = block(x)
print(y.shape) # torch.Size([2, 256, 32, 32])
```

## Scaling Laws: The Physics of Deep Learning

In 2020, Kaplan et al. (OpenAI) and later Hoffmann et al. (DeepMind "Chinchilla") discovered that model performance scales as a **Power Law** with respect to:
1.  **N:** Number of Parameters.
2.  **D:** Dataset Size.
3.  **C:** Compute (FLOPs).

\[ L(N) \propto \frac{1}{N^\alpha} \]

**Key Insight:**
-   If you double the model size, you need to double the data to train it optimally.
-   Most models (like GPT-3) were **undertrained**. They were too big for the amount of data they saw.
-   **Chinchilla Optimal:** For a fixed compute budget, you should balance model size and data size equally.

## Neural Architecture Search (NAS)

Why design by hand when AI can design AI?

**1. Reinforcement Learning (RL):**
-   A "Controller" RNN generates an architecture string (e.g., "Conv 3x3 -> MaxPool").
-   Train the child model for a few epochs. Get accuracy.
-   Use accuracy as "Reward" to update the Controller.
-   *Example:* NASNet (Google).

**2. Differentiable NAS (DARTS):**
-   Define a "Supergraph" containing all possible operations (Conv3x3, Conv5x5, MaxPool) on every edge.
-   Assign a continuous weight \(\alpha\) to each operation.
-   Train the weights via Gradient Descent.
-   Prune the weak operations at the end.
-   *Pros:* Much faster than RL (GPU days vs GPU years).

## Case Study: EfficientNet (Compound Scaling)

Before EfficientNet, people scaled models randomly.
-   "Let's make it deeper!" (ResNet-152)
-   "Let's make it wider!" (WideResNet)
-   "Let's increase resolution!"

**EfficientNet Insight:**
Depth, Width, and Resolution are coupled.
-   If you make the image bigger, you need more layers (Depth) to increase the receptive field.
-   If you make it deeper, you need more channels (Width) to capture fine-grained patterns.

**Compound Scaling Method:**
Scale all three dimensions uniformly using a coefficient \(\phi\):
-   Depth: \(d = \alpha^\phi\)
-   Width: \(w = \beta^\phi\)
-   Resolution: \(r = \gamma^\phi\)
-   Constraint: \(\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2\)

This principled approach produced a family of models (B0 to B7) that dominated the ImageNet leaderboard while being 10x smaller than competitors.

## Deep Dive: Distributed Training Strategies

When your model is too big for one GPU (e.g., GPT-3 is 175B parameters, requiring 800GB VRAM), you need Distributed Training.

### 1. Data Parallelism (DDP)
-   **Scenario:** Model fits in one GPU, but Batch Size is small.
-   **Mechanism:** Replicate the model on 8 GPUs. Split the batch (e.g., 32 images -> 4 per GPU).
-   **Sync:** Gradients are averaged across GPUs using `AllReduce` (Ring Algorithm).

### 2. Model Parallelism (Tensor Parallelism)
-   **Scenario:** Model layer is too wide for one GPU.
-   **Mechanism:** Split a single Matrix Multiplication across 2 GPUs.
    -   GPU 1 computes the top half of the matrix.
    -   GPU 2 computes the bottom half.
-   **Sync:** Requires high-bandwidth interconnect (NVLink).

### 3. Pipeline Parallelism
-   **Scenario:** Model is too deep.
-   **Mechanism:** Put Layers 1-10 on GPU 1, Layers 11-20 on GPU 2.
-   **Issue:** The "Bubble". GPU 2 sits idle while GPU 1 works.
-   **Fix:** Micro-batches.

### 4. ZeRO (Zero Redundancy Optimizer)
-   **Idea:** Don't replicate the Optimizer States and Gradients on every GPU. Shard them.
-   **ZeRO-3:** Shards the Model Parameters too. Allows training trillion-parameter models.

## Failure Modes in Architecture Design

1.  **The Vanishing Gradient:**
    -   *Symptom:* Loss doesn't decrease.
    -   *Cause:* Network too deep without Residual connections. Sigmoid/Tanh activations.
    -   *Fix:* Use ResNets, ReLU, Batch Norm.

2.  **The Information Bottleneck:**
    -   *Symptom:* Poor performance on fine-grained tasks.
    -   *Cause:* Downsampling too aggressively (Stride 2) too early.
    -   *Fix:* Keep resolution high for longer. Use Dilated Convolutions.

3.  **Over-Parameterization (Overfitting):**
    -   *Symptom:* Training loss 0, Validation loss high.
    -   *Cause:* Model too big for the dataset.
    -   *Fix:* Dropout, Weight Decay, Data Augmentation, or smaller model.

## Cost Analysis: FLOPs vs. Latency

**FLOPs (Floating Point Operations)** is a theoretical metric.
**Latency (ms)** is what matters in production.

They are not always correlated!
-   **Depthwise Separable Convs** have low FLOPs but high Latency on GPUs. Why? Because they are **Memory Bound**. They have low arithmetic intensity (compute/memory ratio).
-   **Standard Convs** are highly optimized by cuDNN and Tensor Cores.

**Takeaway:** Don't just optimize FLOPs. Benchmark on the target hardware (T4, A100, Mobile CPU).

## Deep Dive: Training Loop Implementation

Designing the architecture is only half the battle. You need to train it.
Here is a standard PyTorch training loop for our ResNet.

```python
import torch.optim as optim

def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 1. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # SGD with Momentum is standard for ResNets
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss {running_loss/len(train_loader):.4f} | Acc {train_acc:.2f}%")
        
        # Validation
        validate(model, val_loader, criterion, device)
        
        scheduler.step()

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    print(f"Val Loss {val_loss/len(loader):.4f} | Val Acc {100.*correct/total:.2f}%")
```

## Top Interview Questions

**Q1: Why does ResNet work?**
*Answer:*
1.  **Gradient Flow:** The skip connection `x + F(x)` allows gradients to flow through the network without being multiplied by weight matrices at every layer. This prevents vanishing gradients.
2.  **Ensemble Hypothesis:** A ResNet can be seen as an ensemble of many shallower networks. Dropping a layer in ResNet doesn't kill performance, unlike in VGG.
3.  **Identity Mapping:** It's easier for the network to learn `F(x) = 0` (identity mapping) than to learn a specific transformation.

**Q2: When should you use Layer Norm over Batch Norm?**
*Answer:*
-   Use **Layer Norm** for RNNs and Transformers (NLP/Speech). It works well when sequence lengths vary and batch sizes are small.
-   Use **Batch Norm** for CNNs (Vision). It acts as a regularizer and speeds up convergence, but requires large fixed-size batches.

**Q3: How do you calculate the number of parameters in a Convolutional Layer?**
*Answer:*
\[ \text{Params} = (K \times K \times C_{in} + 1) \times C_{out} \]
Where \(K\) is kernel size, \(C_{in}\) is input channels, \(C_{out}\) is output channels, and \(+1\) is for the bias.

**Q4: What is the "Receptive Field" and how do you increase it?**
*Answer:*
The Receptive Field is the region of the input image that affects a specific neuron.
To increase it:
1.  Add more layers (Depth).
2.  Use larger kernels (e.g., 7x7).
3.  Use **Dilated Convolutions** (Atrous).
4.  Use Pooling / Strided Convolutions (Downsampling).

**Q5: Why do we use "He Initialization" (Kaiming Init) for ReLU networks?**
*Answer:*
Xavier (Glorot) initialization assumes linear activations. ReLU is non-linear (half the activations are zeroed).
He Initialization scales the weights by \(\sqrt{2/n}\) instead of \(\sqrt{1/n}\) to maintain the variance of activations through the layers.

## Deep Dive: Regularization Techniques

Deep networks are prone to overfitting. We need to stop them from memorizing the training data.

### 1. Dropout (Hinton et al., 2012)
-   **Mechanism:** Randomly zero out neurons during training with probability \(p\) (usually 0.5).
-   **Effect:** Prevents co-adaptation of features. Forces the network to learn redundant representations.
-   **Inference:** Scale weights by \((1-p)\) or use Inverted Dropout during training.

### 2. DropConnect (Wan et al., 2013)
-   **Mechanism:** Instead of zeroing neurons (activations), zero out the **weights**.
-   **Effect:** A generalization of Dropout.

### 3. Stochastic Depth (Huang et al., 2016)
-   **Mechanism:** Randomly drop entire **Residual Blocks** during training.
-   **Effect:** Effectively trains an ensemble of networks of different depths. Crucial for training very deep ResNets (>100 layers) and Vision Transformers.

### 4. Label Smoothing
-   **Mechanism:** Instead of targeting `[0, 1, 0]`, target `[0.1, 0.8, 0.1]`.
-   **Effect:** Prevents the model from becoming over-confident. Calibrates probabilities.

## Deep Dive: Optimizers (SGD vs. Adam)

Which optimizer should you use for your architecture?

### 1. SGD with Momentum
-   **Formula:** \(v_t = \mu v_{t-1} + g_t\); \(w_t = w_{t-1} - \eta v_t\).
-   **Best for:** **CNNs (ResNet, VGG)**.
-   **Why?** It generalizes better. It finds flatter minima.

### 2. Adam (Adaptive Moment Estimation)
-   **Formula:** Maintains per-parameter learning rates based on first and second moments of gradients.
-   **Best for:** **Transformers (BERT, GPT, ViT)** and **RNNs**.
-   **Why?** Transformers have very complex loss landscapes. SGD gets stuck. Adam navigates the curvature better.

### 3. AdamW (Adam with Weight Decay)
-   **Fix:** Standard L2 regularization in Adam is broken. AdamW decouples weight decay from the gradient update.
-   **Status:** The default optimizer for all modern LLMs and ViTs.

## Deep Dive: Hardware Efficiency (The Memory Wall)

Why is a 100M parameter model faster than a 50M parameter model sometimes?
Because of **Arithmetic Intensity**.

\[ \text{Intensity} = \frac{\text{FLOPs}}{\text{Bytes Access}} \]

-   **Compute Bound:** Layers like Conv2d (large channels) or Linear (large batch). The GPU cores are 100% utilized.
-   **Memory Bound:** Layers like Activation (ReLU), Normalization (BN), or Element-wise Add. The GPU cores are waiting for data from VRAM.

**Optimization:**
-   **Operator Fusion:** Fuse `Conv + BN + ReLU` into a single kernel. This reads data once, does 3 ops, and writes once.
-   **FlashAttention:** A hardware-aware attention algorithm that reduces HBM (High Bandwidth Memory) access by tiling the computation in SRAM (L1 Cache). It speeds up Transformers by 3-4x.

## Further Reading

1.  **ResNet:** [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)
2.  **Transformer:** [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
3.  **EfficientNet:** [EfficientNet: Rethinking Model Scaling for CNNs (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
4.  **ViT:** [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
5.  **Chinchilla:** [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)

## Key Takeaways

1.  **Inductive Bias:** Choose the architecture that fits your data structure (CNN for images, Transformer for sets/sequences).
2.  **Residuals are King:** You cannot train deep networks without skip connections.
3.  **Normalization is Queen:** Batch Norm or Layer Norm is essential for convergence.
4.  **Scale Principledly:** Use Compound Scaling (EfficientNet) or Chinchilla laws.
5.  **Don't Reinvent:** Start with a standard backbone (ResNet, ViT) and modify the Head. Only design a custom backbone if you have a very specific constraint.

---

**Originally published at:** [arunbaby.com/ml-system-design/0027-model-architecture-design](https://www.arunbaby.com/ml-system-design/0027-model-architecture-design/)

*If you found this helpful, consider sharing it with others who might benefit.*


