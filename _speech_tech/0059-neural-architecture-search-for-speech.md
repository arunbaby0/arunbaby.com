---
title: "Neural Architecture Search (NAS) for Speech"
day: 59
related_dsa_day: 59
related_ml_day: 59
related_agents_day: 59
collection: speech_tech
categories:
  - speech-tech
tags:
  - nas
  - asr
  - tts
  - automl
  - conformer
  - latency-optimization
  - mobilenet
subdomain: "Automated Speech Engineering"
tech_stack: [Optuna, PyTorch, Ray, ONNX, Kaldi]
scale: "Searching billions of possible Conformer-Transformer combinations for low-latency ASR"
companies: [Apple, Google, Meta, NVIDIA, Samsung]
difficulty: Hard
---


**"Hand-crafting speech architectures is reaching its limits. For the next generation of voice assistants, we don't build the model—we define the search space and let the computer discover the most efficient physics of sound."**

## 1. Introduction: The Mobile Challenge

In the world of Speech Tech, we are constantly fighting two warring factions: **Word Error Rate (WER)** and **Real-Time Factor (RTF)**.

-   **The Giant**: A model like OpenAI's Whisper-Large-v3 (1.5B parameters) has a low WER but requires a V100 GPU to run in reasonable time. It is impossible to deploy on an iPhone or an IoT thermostat.
-   **The Dwarf**: A tiny model (like DeepSpeech-Tiny) fits on the phone but fails miserably when the user has an accent or there is background noise.

Historically, humans tried to bridge this gap by hand-designing efficient architectures. We invented **MobileNet** (separable convolutions), **SqueezeNet** (1x1 filters), and the **Conformer** (combining CNNs and Transformers). But "Human Descent" is slow. A researcher might test 10 architectures in a month.

**Neural Architecture Search (NAS) for Speech** is the automation of this discovery. It treats a model's topology (number of heads, kernel sizes, dilation rates) not as a fixed decision, but as a **Variable in a Search Problem**.

In this deep dive, we will explore how to build a system that *discovers* the optimal speech model for a given hardware constraint (e.g., "Must run on Raspberry Pi 4 with <200ms latency"). We move from "Guessing" to "Evolution."

---

## 2. The Search Space: Building Blocks of Speech

NAS depends on the "Architecture Palette." For speech, we typically search across:

### 2.1 Convolutional Front-ends
- Kernel sizes (3x3 vs 5x5 vs 7x7).
- Dilated Convolutions: How far should the "context window" reach in a single time step?
- Grouped vs. Depthwise Separable Convolutions (to save parameters).

### 2.2 Transformer/Conformer Blocks
- Number of attention heads.
- Dimension of the Feed-Forward Network (FFN).
- Placement of the Convolutional module within the Transformer layer.

### 2.3 The Connectivity
- Should we use "Dense" connections (skip connections) or a linear stack?
- Where should we downsample the time-dimension (Striding)?

---

## 3. High-Level Architecture: The Performance-Aware Searcher

A production NAS system for speech follows a three-stage loop:

1. **Search Controller (The Agent)**: Proposes a "Candidate Model" from the search space.
2. **Training & Evaluation (The Trial)**: Trains the candidate on a subset of the dataset (e.g., Librispeech) for a few epochs.
3. **Hardware Profiler**: Measures the **Latency** on a specific target device and the **Power Consumption**.
4. **Reward Function**: Rewards candidates that have the best "WER-Latency" trade-off.

---

## 4. Implementation: Once-for-all (OFA) Search

One of the most efficient NAS strategies is the **OFA (Once-for-all)** approach. Instead of training 1,000 separate models, we train one "Super-Network" that contains all possible sub-networks.

### The Logic
1. **Train the Super-Net**: Ensure that any "Slice" of the network is still functionally valid.
2. **Architectural Sampling**: Randomly pick sub-networks during training and update their weights.
3. **The Result**: At the end of training, you have a single set of weights from which you can "extract" the best model for any hardware constraint without re-training.

```python
class SpeechSuperNet(nn.Module):
    def __init__(self, max_heads=8, max_layers=12):
        self.layers = nn.ModuleList([
            DynamicTransformerBlock(max_heads) for _ in range(max_layers)
        ])

    def forward(self, x, current_config):
        # Dynamically 'slice' the model based on current search trial
        num_layers = current_config['layers']
        for i in range(num_layers):
            x = self.layers[i](x, heads=current_config['heads'][i])
        return x
```

---

## 5. The Reward Function: The "Efficiency Frontier"

We don't just want the lowest Word Error Rate (WER). We want to solve for:
`Reward = -WER - lambda * log(Latency)`

Where `lambda` represents how much we value speed.

---

## 6. Real-time Implementation: On-Device Accuracy

When an architecture is discovered, how is it deployed?
1. **Export to ONNX/CoreML**: Convert the neural graph to a format optimized for the mobile NPU.
2. **Quantization-Aware Discovery**: Search for models that perform well with 8-bit integers.
3. **Phonetic Pruning**: Prune redundant layers for specific acoustic environments.

---

## 7. Comparative Analysis: Hand-crafted vs. NAS Models

| Metric | Hand-crafted (Conformer) | NAS-Optimized (S-NAS) |
| :--- | :--- | :--- |
| **WER (Noise)** | 5.2% | 4.8% |
| **Params** | 120M | 35M |
| **Latency (iPhone)** | 120ms | 40ms |
| **Search Time** | 3 months (Human) | 48 hours (GPU) |

---

## 8. Failure Modes in Speech NAS

1. **Invalid Topologies**: The searcher proposes a model that is too deep for target memory.
  * *Mitigation*: Implement "Soft Constraints" that reject configurations exceeding a budget.
2. **The "Hardware Gap"**: A model fast on a CPU might be slow on a DSP.
  * *Mitigation*: Perform evaluations on the **Physical Hardware**.
3. **Feature Mismatch**: The NAS finds a great model for 16kHz but the production uses 8kHz.

---

## 9. Real-World Case Study: Google’s "E-NAS" for Voice Assistant

Google used NAS to design the "New Google Assistant" models.
- **The Challenge**: The model had to understand voice locally on a phone.
- **The Result**: Discovered a "Hydra" architecture—a shared trunk with multiple heads for different tasks (ASR, Intent Detection). Reduced parameter count by 75%.

---

## 10. Key Takeaways

1. **Search is the new Engineering**: Automating the search for the "Correct digits in the grid" is the only way to achieve peak efficiency.
2. **Hardware-in-the-loop**: A speech model is only as good as its speed on the target device.
3. **NAS is not just for WER**: Optimize for battery life, memory, and privacy.
4. **Pruning starts in the Search phase**: Use AutoML principles to kill poor architectures early.

---

**Originally published at:** [arunbaby.com/speech-tech/0059-neural-architecture-search-for-speech](https://www.arunbaby.com/speech-tech/0059-neural-architecture-search-for-speech/)

*If you found this helpful, consider sharing it with others who might benefit.*
