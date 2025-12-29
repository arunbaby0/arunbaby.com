---
title: "Neural Architecture Search (NAS) for Speech"
day: 59
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
related_dsa_day: 59
related_ml_day: 59
related_agents_day: 59
---

**"Hand-crafting speech architectures is reaching its limits. For the next generation of voice assistants, we don't build the model—we define the search space and let the computer discover the most efficient physics of sound."**

## 1. Introduction: The Mobile Challenge

In the world of Speech Tech, we are constantly fighting two warring factions: **Word Error Rate (WER)** and **Real-Time Factor (RTF)**. 
- A massive model like Whisper-Large has a low WER but can never run on a phone in real-time.
- A tiny model fits on the phone but misses the nuances of accents and noise.

Historically, humans designed "Mobile-Optimized" architectures like **MobileNet** or **Cuside-Transformer** by hand. However, sound is complex. The optimal architecture for a "Quiet Home" is different from an "Industrial Warehouse."

**Neural Architecture Search (NAS) for Speech** is the automation of this discovery. It treats a model's topology (number of heads, kernel sizes, dilation rates) as a **Constraint Satisfaction Problem** (connecting to our **Sudoku Solver** DSA topic). Today, we build the pipeline for discovering speech models that are "Pareto-Optimal"—better accuracy for less compute.

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

A production NAS system for speech (like Apple’s research into efficient ASR) follows a three-stage loop:

1.  **Search Controller (The Agent)**: Proposes a "Candidate Model" from the search space.
2.  **Training & Evaluation (The Trial)**: Trains the candidate on a subset of the dataset (e.g., Librispeech-100) for a few epochs.
3.  **Hardware Profiler**: Measures the **Latency** on a specific target device (e.g., iPhone 15, Pixel 8) and the **Power Consumption**.
4.  **Reward Function**: Rewards candidates that have the best "WER-Latency" trade-off.

---

## 4. Implementation: Once-for-all (OFA) Search

One of the most efficient NAS strategies is the **OFA (Once-for-all)** approach. Instead of training 1,000 separate models, we train one "Super-Network" that contains all possible sub-networks.

### The Logic
1.  **Train the Super-Net**: Ensure that any "Slice" of the network (e.g., using only 4 heads instead of 8) is still functionally valid.
2.  **Architectural Sampling**: Randomly pick sub-networks during training and update their weights.
3.  **The Result**: At the end of training, you have a single set of weights from which you can "extract" the best model for any hardware constraint (one for a high-end server, one for a smartwatch) without re-training.

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
$Reward = -WER - \lambda \cdot \log(Latency)$

Where $\lambda$ represents how much we value speed.
- If $\lambda$ is high, the system will favor tiny, lightning-fast models.
- If $\lambda$ is low, it will favor heavy, accurate models.

---

## 6. Real-time Implementation: On-Device Accuracy

When an architecture is discovered, how is it deployed?
1.  **Export to ONNX/CoreML**: Convert the neural graph to a static format optimized for the mobile NPU (Neural Processing Unit).
2.  **Quantization-Aware Discovery**: The NAS system searches for models that perform well even when their weights are compressed from 32-bit floats to 8-bit integers.
3.  **Phonetic Pruning**: Prune layers that the NAS system identifies as redundant for specific acoustic environments.

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

1.  **Invalid Topologies**: The searcher proposes a model that is too deep to fit in the GPU's memory.
    *   *Mitigation*: Implement "Soft Constraints" in the search controller that reject configurations exceeding a resource budget (The Sudoku Link).
2.  **The "Hardware Gap"**: A model that is fast on a CPU might be slow on a DSP (Digital Signal Processor).
    *   *Mitigation*: Always perform evaluations on the **Physical Hardware**, not a simulator.
3.  **Feature Mismatch**: The NAS finds a great model for 16kHz audio, but the production system uses 8kHz.

---

## 9. Real-World Case Study: Google’s "E-NAS" for Voice Assistant

Google used NAS to design the "New Google Assistant" models.
- **The Challenge**: The model had to understand voice locally on a phone to eliminate latency.
- **The Result**: NAS discovered a "Hydra" architecture—a single shared convolutional trunk with multiple "Heads" for different tasks (ASR, Intent Detection). This reduced parameter count by 75% compared to the original design.

---

## 10. Key Takeaways

1.  **Search is the new Engineering**: (The DSA Link) Automating the search for the "Correct digits in the grid" is the only way to achieve peak efficiency.
2.  **Hardware-in-the-loop**: A speech model is only as good as its speed on the target device.
3.  **NAS is not just for WER**: Use it to optimize for battery life, memory, and even privacy.
4.  **Pruning starts in the Search phase**: (The ML Link) Use AutoML principles to kill poor architectures early.

---

**Originally published at:** [arunbaby.com/speech-tech/0059-neural-architecture-search-for-speech](https://www.arunbaby.com/speech-tech/0059-neural-architecture-search-for-speech/)

*If you found this helpful, consider sharing it with others who might benefit.*
