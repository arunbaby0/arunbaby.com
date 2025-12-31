---
title: "ML Capacity Planning and Infrastructure Scaling"
day: 57
related_dsa_day: 57
related_speech_day: 57
related_agents_day: 57
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
---


**"Capacity Planning is the art of predicting the future while paying for the present. In ML, it is the difference between a high-growth product and a bankrupt one."**

## 1. Introduction: The Billion Dollar Guessing Game


---

## 2. Higher-Level Goals: The Three Pillars

1. **Availability**: Ensure the system stays up during peak traffic.
2. **Performance**: Maintain the latency budget across the 99th percentile (p99).
3. **Cost Efficiency**: Maximize the utilization of expensive GPU resources (targeting > 70%).

---

## 3. High-Level Architecture: The Capacity Lifecycle

### 3.1 Load Projection (The Input)
- **Growth Modeling**: Using historical DAU (Daily Active Users) to predict traffic.
- **Seasonality**: Factoring in time-of-day and weekly cycles.

### 3.2 Benchmarking (The Profiler)
- **Baseline Throughput**: How many requests per second (RPS) can one instance handle at the target latency?
- **Resource Saturation**: Identifying bottlenecks at specific RPS levels.

### 3.3 Provisioning (The Output)
- **Buffer/Headroom**: Leaving ~30% capacity for sudden spikes.
- **Auto-Scaling**: Dynamically adding/removing nodes based on real-time metrics.

---

## 4. Operational Math: Calculating "The Number"

Consider a hypothetical **Speech ASR** system:

### 4.1 Variables
- **Peak Traffic**: 10,000 requests per second (RPS).
- **Single Instance Performance**: 5 RPS per GPU (with a 200ms p99 latency).
- **Overhead**: 20% system overhead.

### 4.2 The Calculation

`Required_Nodes = (Peak_RPS / Instance_RPS) * (1 + Overhead)`
`Required_Nodes = (10,000 / 5) * 1.2 = 2,400` GPUs.

At `2 per hour per GPU, this system costs **`4,800 per hour** to operate. If efficiency drops to 4 RPS, costs jump by **\1,200 per hour**. This is why **Latency Optimization** is actually a **Profit Margin Optimization**.

---

## 5. Scaling Strategy: Distributed vs. Vertical

- **Vertical Scaling**: Upgrading to more powerful GPUs. Best for models with massive memory requirements.
- **Horizontal Scaling**: Adding more nodes to a cluster. Best for handling massive request volume.
- **The Hybrid Approach**: Use high-end GPUs for training and cost-effective chips for serving.

---

## 6. Real-time Implementation: Auto-scaling and Load Balancing

- **Least-Loaded Balancing**: Directing requests to workers with the most free resources.
- **Predictive Auto-scaling**: Scaling up servers in anticipation of known traffic patterns.

---

## 7. Comparative Analysis: Cloud vs. On-Premise

| Metric | Cloud (AWS/GCP) | On-Premise (Owned Hardware) |
| :--- | :--- | :--- |
| **CapEx** | 0 | Extremely High |
| **Agility** | High | Low |
| **Visibility** | Low | High |
| **Break-even** | Good for startups | Good for established volume |

---

## 8. Failure Modes in Infrastructure Scaling

1. **The "Thundering Herd"**: Sudden traffic surges that crash new nodes before they warm up.
  * *Mitigation*: Use **Rate Limiting** and **Circuit Breakers**.
2. **Resource Fragmentation**: Inefficient request distribution preventing effective batching.
3. **Cloud Stockouts**: Regional unavailability of specific GPU types.

---

## 9. Real-World Case Study: Capacity Limits

Major AI companies have had to pause new signups due to physical constraints in data centers and lead times on hardware. growth is limited by the **Maximum Area** of your hardware histogram. If you don't pre-book capacity, growth hits a hard ceiling.

---

## 10. Key Takeaways

1. **Throughput is a Profit Metric**: Making a model twice as fast is equivalent to cutting your infrastructure bill in half.
2. **Proactive is Cheaper than Reactive**: Predictive scaling prevents customer churn.
3. **The Histogram Connection**: Capacity is about finding the "Largest Stable Area" of operation without crossing into saturation.
4. **Reliability is Scaling**: An unreliable agent is often just a slow agent running on saturated hardware.

---

**Originally published at:** [arunbaby.com/ml-system-design/0057-capacity-planning](https://www.arunbaby.com/ml-system-design/0057-capacity-planning/)

*If you found this helpful, consider sharing it with others who might benefit.*
