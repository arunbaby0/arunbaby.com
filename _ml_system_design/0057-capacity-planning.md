---
title: "ML Capacity Planning and Infrastructure Scaling"
day: 57
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - capacity-planning
  - scalability
  - gpus
  - cost-optimization
  - infra
  - throughput
  - latency
subdomain: "Infrastructure"
tech_stack: [Kubernetes, Prometheus, Grafana, AWS, GCP, NVIDIA Triton]
scale: "Scaling from 1 to 10,000+ GPUs for global inference"
companies: [OpenAI, Meta, NVIDIA, Uber, Cruise]
difficulty: Hard
related_dsa_day: 57
related_speech_day: 57
related_agents_day: 57
---

**"Capacity Planning is the art of predicting the future while paying for the present. In ML, it is the difference between a high-growth product and a bankrupt one."**

## 1. Introduction: The Billion Dollar Guessing Game

In 2023, the biggest constraint on AI progress wasn't algorithms; it was the availability of H100 GPUs. 
When you build a machine learning system, you face a terrifying math problem:
- If you buy **too few** servers, your latency spikes, your users churn, and your "Rectangle of Success" (the area of profitable operation) collapses.
- If you buy **too many** servers, your burn rate explodes, and you run out of capital before reaching product-market fit.

**Capacity Planning** is the engineering framework for managing this trade-off. It combines historical data, probabilistic modeling, and hardware performance profiles to determine the most cost-efficient infrastructure roadmap. Today, on Day 57, we explore how to project load, handle "thundering herds," and scale gracefully.

---

## 2. Higher-Level Goals: The Three Pillars

1.  **Availability**: Ensure the system stays up during peak traffic (e.g., Black Friday or a viral launch).
2.  **Performance**: Maintain the "Latency Budget" across the 99th percentile (p99).
3.  **Cost Efficiency**: Maximize the "Utilization" of expensive GPU resources. (Targeting > 70% utilization).

---

## 3. High-Level Architecture: The Capacity Lifecycle

### 3.1 Load Projection (The Input)
- **Growth Modeling**: Using historical DAU (Daily Active Users) to predict next month's traffic.
- **Seasonality**: Factoring in time-of-day and weekly cycles.

### 3.2 Benchmarking (The Profiler)
- **Baseline Throughput**: How many requests per second (RPS) can one Instance of our model handle at our target latency?
- **Resource Saturation**: At what RPS does the CPU, GPU Memory, or Network bandwidth become the bottleneck?

### 3.3 Provisioning (The Output)
- **Buffer/Headroom**: Always leave ~30% empty space for sudden spikes.
- **Auto-Scaling**: Dynamically adding/removing nodes based on real-time metrics.

---

## 4. Operational Math: Calculating "The Number"

Let's do a capacity exercise for a hypothetical **Speech ASR** system (connecting to today's **Speech Tech** theme).

### 4.1 Variables
- **Peak Traffic**: 10,000 requests per second (RPS).
- **Single Instance Performance**: 5 RPS per GPU (with a 200ms p99 latency).
- **Redundancy (N+1)**: We need at least 1 extra node to handle failure.
- **Overhead**: 20% system overhead.

### 4.2 The Calculation
$Required\_Nodes = \frac{Peak\_RPS}{Instance\_RPS} \times (1 + Overhead)$
$Required\_Nodes = \frac{10,000}{5} \times 1.2 = 2,400$ GPUs.

At \$2 per GPU hour, this system costs **\$4,800 per hour** to operate. If your model efficiency drops to 4 RPS, your costs jump by **\$1,200 per hour**. This is why **Latency Optimization** is actually a **Profit Margin Optimization**.

---

## 5. Scaling Strategy: Distributed vs. Vertical

- **Vertical Scaling**: Upgrading from an A10 GPU to an H100. Best for models that require massive VRAM.
- **Horizontal Scaling**: Adding more nodes to a Kubernetes cluster. Best for handling massive request volume.
- **The Hybrid Approach**: Use high-end GPUs for training and cost-effective, specialized inference chips (like NVIDIA L4 or AWS Inferentia) for serving.

---

## 6. Real-time Implementation: Auto-scaling and Load Balancing

When scaling, you must solve the "Connection Problem."
- **Least-Loaded Balancing**: Don't just send requests randomly. Send them to the worker with the most free GPU memory.
- **Predictive Auto-scaling**: Based on the last 30 days, we know traffic spikes at 8 AM. We start spinning up new servers at 7:50 AM, so they are "Warm" when the traffic hits.

---

## 7. Comparative Analysis: Cloud vs. On-Premise

| Metric | Cloud (AWS/GCP) | On-Premise (Owned Hardware) |
| :--- | :--- | :--- |
| **CapEx** | \$0 | Extremely High |
| **Agility** | High (Spin up in seconds) | Low (Weeks to ship/install) |
| **Visibility** | Low (Shared hardware) | High (Full control) |
| **Break-even** | Good for startups | Good for massive established products |

---

## 8. Failure Modes in Infrastructure Scaling

1.  **The "Thundering Herd"**: An upstream service recovers and sends 1 million queued requests simultaneously, crashing your new nodes before they can even warm up.
    *   *Mitigation*: Use **Rate Limiting** and **Circuit Breakers**.
2.  **Resource Fragmentation**: Small requests are scattered across different GPUs, preventing any single GPU from batching effectively.
3.  **Cloud Stockouts**: You want to scale to 1,000 GPUs, but the cloud provider simply doesn't have them available in your region.

---

## 9. Real-World Case Study: OpenAI’s Scaling Crisis

OpenAI famously had to pause "ChatGPT Plus" signups in late 2023. This was a classic **Capacity Planning Failure**. 
- The limit wasn't their code; it was the physical space in Microsoft’s data centers and the lead time on new H100 clusters.
- **Lesson**: Your product's growth is limited by the **Maximum Area** of your hardware histogram (the DSA link). If you don't pre-book the capacity, your growth hits a hard ceiling.

---

## 10. Key Takeaways

1.  **Throughput is a Profit Metric**: Making a model twice as fast is equivalent to cutting your infrastructure bill in half.
2.  **Proactive is Cheaper than Reactive**: Predictive scaling prevents customer churn during outages.
3.  **The Histogram Connection**: (The DSA Link) Capacity is about finding the "Largest Stable Area" of operation without crossing the red line of saturation.
4.  **Reliability is Scaling**: (The Agent Link) An unreliable agent is often just a slow agent running on saturated hardware.

---

**Originally published at:** [arunbaby.com/ml-system-design/0057-capacity-planning](https://www.arunbaby.com/ml-system-design/0057-capacity-planning/)

*If you found this helpful, consider sharing it with others who might benefit.*
