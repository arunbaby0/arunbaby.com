---
title: "Resource Partitioning in ML Clusters"
day: 36
collection: ml_system_design
categories:
  - ml_system_design
tags:
  - kubernetes
  - gpu
  - scheduling
  - ray
  - distributed-systems
subdomain: "Infrastructure"
tech_stack: [Kubernetes, Ray, Slurm, NVIDIA MIG]
scale: "10,000+ GPUs"
companies: [OpenAI, Meta, Google, Uber]
---

**"How to share a supercomputer without stepping on each other's toes."**

## 1. The Problem: Multi-Tenant ML Clusters

Training Large Language Models (LLMs) requires massive compute clusters (e.g., 10k H100s).
- **Users:** Research, Product, Data Science.
- **Workloads:**
    -   **Training:** Long-running (weeks), gang-scheduled, fault-intolerant.
    -   **Inference:** Latency-sensitive, high availability, bursty.
    -   **Dev/Notebooks:** Interactive, low utilization.

**Goal:** Maximize utilization while ensuring fairness and isolation.

## 2. Partitioning Strategies

### 1. Static Partitioning
- **Concept:** Dedicate physical nodes to specific teams.
    -   Team A gets Rack 1-10.
    -   Team B gets Rack 11-20.
- **Pros:** Perfect isolation. No "noisy neighbors".
- **Cons:** Low utilization. If Team A is sleeping, their GPUs sit idle.

### 2. Dynamic Partitioning (Cluster Autoscaling)
- **Concept:** A shared pool of resources. Jobs request what they need.
- **Scheduler:** Decides placement based on availability and priority.
- **Pros:** High utilization.
- **Cons:** Complex scheduling, potential for interference.

## 3. Kubernetes Resource Management

Kubernetes (K8s) is the de-facto OS for ML clusters.

### Requests vs. Limits
- **Request:** "I need at least 4 CPUs." (Used for scheduling).
- **Limit:** "Kill me if I use more than 8 CPUs." (Used for isolation/throttling).
- **Best Practice:** For Training, set `Request == Limit` (Guaranteed QoS).

### Namespaces & Quotas
- **Namespace:** Logical partition (e.g., `team-nlp`, `team-vision`).
- **ResourceQuota:** Hard limit on aggregate resource usage per namespace.
  ```yaml
  apiVersion: v1
  kind: ResourceQuota
  metadata:
    name: gpu-quota
  spec:
    hard:
      requests.nvidia.com/gpu: "100"
  ```

## 4. GPU Partitioning

GPUs are expensive. We need to slice them.

### 1. Multi-Instance GPU (MIG)
- **Hardware Support:** NVIDIA A100/H100.
- **Mechanism:** Physically partitions the GPU into up to 7 isolated instances (Compute + Memory).
- **Use Case:** Inference, small model training.
- **Isolation:** Strong (Hardware-level).

### 2. Time-Slicing (MPS)
- **Mechanism:** Multiple processes share the GPU context. CUDA kernels are interleaved.
- **Use Case:** Dev notebooks.
- **Isolation:** Weak. One process can crash the GPU or hog memory bandwidth.

### 3. Virtualization (vGPU)
- **Mechanism:** Hypervisor manages GPU access.
- **Use Case:** Cloud providers (AWS EC2 instances).

## 5. Scheduling Algorithms

How do we decide who goes next?

### 1. FIFO (First-In, First-Out)
- Simple.
- **Problem:** Head-of-line blocking. A small job waits behind a massive 1-week training job.

### 2. Dominant Resource Fairness (DRF)
- Generalization of Max-Min Fairness for multi-dimensional resources (CPU, RAM, GPU).
- **Idea:** Equalize the "dominant share" of resources across users.
- If User A uses lots of CPU and User B uses lots of RAM, DRF balances their dominant usage.

### 3. Gang Scheduling (All-or-Nothing)
- **Critical for Distributed Training.**
- If a job needs 100 GPUs, it must get **all 100 at once**.
- If it gets 99, it waits (holding the 99 hostage).
- **Coscheduling:** K8s plugins (Volcano, Kueue) ensure gang scheduling.

## 6. Deep Dive: Ray for ML Orchestration

**Ray** sits on top of K8s and provides Python-native resource management.

- **Actors:** Stateful workers (e.g., a Parameter Server).
- **Tasks:** Stateless functions.
- **Placement Groups:**
  ```python
  # Reserve a bundle of resources atomically
  pg = placement_group([{"CPU": 1, "GPU": 1}] * 10, strategy="STRICT_PACK")
  ray.get(pg.ready())
  ```
- **Strategy:**
    - `STRICT_PACK`: Put everything on the same node (low latency).
    - `SPREAD`: Distribute across nodes (fault tolerance).

## 7. Real-World Case Studies

### Case Study 1: OpenAI's Supercomputer
- **Scale:** Thousands of GPUs.
- **Challenge:** Training GPT-4.
- **Solution:** Kubernetes + Azure. Custom scheduler to handle massive gang scheduling and topology-aware placement (minimizing InfiniBand hops).

### Case Study 2: Uber Michelangelo
- **Workload:** Mixed (Training + Inference).
- **Solution:**
    - **Training:** Mesos (later K8s) with gang scheduling.
    - **Inference:** Dedicated serving cluster with HPA (Horizontal Pod Autoscaler).
    - **Quota:** "Elastic Quota" allows teams to burst into idle capacity but get preempted if the owner returns.

## 8. Summary

| Level | Technology | Strategy |
| :--- | :--- | :--- |
| **Cluster** | Kubernetes | Namespaces, Quotas |
| **Node** | Kubelet | Requests/Limits |
| **Device** | NVIDIA MIG | Hardware Partitioning |
| **Workload** | Ray / Volcano | Gang Scheduling |

## 9. Deep Dive: Kubernetes Scheduler Internals

The K8s scheduler decides which node a pod goes to. It runs in a loop:

1.  **Filtering (Predicates):** Remove nodes that don't fit.
    -   `PodFitsResources`: Does the node have enough CPU/GPU?
    -   `PodFitsHostPorts`: Is the port available?
    -   `TaintToleration`: Does the pod tolerate the node's taints (e.g., `gpu-node=true:NoSchedule`)?

2.  **Scoring (Priorities):** Rank remaining nodes.
    -   `LeastRequestedPriority`: Spread pods to balance load.
    -   `MostRequestedPriority`: Pack pods to fill nodes (bin packing, good for autoscaling).
    -   `ImageLocalityPriority`: Prefer nodes that already have the Docker image.

3.  **Binding:** Assign the pod to the highest-ranked node.

**Custom Scheduling for ML:**
Standard K8s scheduler is bad for ML because it schedules one pod at a time. It doesn't understand "I need 100 GPUs or nothing".

## 10. Deep Dive: Gang Scheduling with Volcano

**Volcano** is a batch system built on K8s.

**Architecture:**
-   **Volcano Scheduler:** Replaces default K8s scheduler.
-   **PodGroup:** A CRD that groups pods together.
    ```yaml
    apiVersion: scheduling.volcano.sh/v1beta1
    kind: PodGroup
    metadata:
      name: tensorflow-job
    spec:
      minMember: 10  # Gang size
    ```

**Logic:**
1.  Wait until `minMember` pods are pending.
2.  Check if cluster has resources for *all* of them.
3.  If yes, bind all.
4.  If no, wait (don't partial schedule). This prevents deadlocks where Job A holds 50% GPUs and Job B holds 50%, and both wait for 100%.

## 11. Deep Dive: GPU Virtualization (vGPU vs MIG)

**NVIDIA vGPU (Virtual GPU):**
-   **Software-based.** Hypervisor time-slices the GPU.
-   **Memory:** Shared (can oversubscribe).
-   **Performance:** Overhead from context switching.
-   **Use Case:** VDI (Virtual Desktop), Cloud Gaming.

**NVIDIA MIG (Multi-Instance GPU):**
-   **Hardware-based.** A100/H100 splits into partitions.
-   **Memory:** Dedicated (cannot oversubscribe).
-   **Performance:** Near-native. No context switching overhead between instances.
-   **Configuration:**
    -   `1g.5gb`: 1 Compute Slice, 5GB RAM.
    -   `3g.20gb`: 3 Compute Slices, 20GB RAM.
-   **Use Case:** Inference, lightweight training.

## 12. Deep Dive: Ray Autoscaler Logic

Ray's autoscaler is smarter than K8s HPA (Horizontal Pod Autoscaler).

**Logic:**
1.  **Resource Demand:** Ray scheduler sees a task needing `{"GPU": 1}` but no node has it.
2.  **Pending State:** The task goes into `PENDING`.
3.  **Upscaling:** Autoscaler calculates: "I need a node of type `g4dn.xlarge` to satisfy this."
4.  **Launch:** Calls cloud provider API (AWS EC2) to launch node.
5.  **Bin Packing:** Ray tries to pack tasks onto existing nodes before launching new ones to save money.

**Downscaling:**
-   If a node is idle for `idle_timeout_minutes` (default 5), terminate it.

## 13. Deep Dive: Spot Instance Management

Training on Spot Instances (preemptible) saves 70% cost but adds risk.

**Strategy:**
1.  **Checkpointing:** Save model weights to S3 every 10 minutes.
2.  **Elastic Training (TorchElastic):**
    -   If a node dies, the job pauses.
    -   Remaining nodes re-rendezvous.
    -   Training continues with fewer nodes (or waits for replacement).
3.  **Mixed Cluster:** Use On-Demand for the "Head Node" (Parameter Server / Scheduler) and Spot for "Worker Nodes".

## 14. Case Study: Meta's Training Cluster (RSC)

**Research SuperCluster (RSC):**
-   **Hardware:** 16,000 A100 GPUs.
-   **Network:** InfiniBand (dedicated high-speed network).
-   **Storage:** Pure Storage FlashBlade (high throughput).

**Partitioning:**
-   **Physical Isolation:** Experiments are physically separated to avoid "noisy neighbor" network congestion.
-   **Topology-Aware Scheduling:** The scheduler knows the physical wiring. It places communicating pods on the same switch to minimize latency.

## 15. System Design: Multi-Tenant Inference Platform

**Scenario:** Serve 1000 different models (LLMs, ResNets, BERTs) for 50 teams.

**Design:**
1.  **Ingress:** Nginx/Istio routes request to correct service.
2.  **Model Loading:**
    -   **Hot:** Top 50 models loaded in GPU memory.
    -   **Warm:** Next 200 models in CPU RAM.
    -   **Cold:** Rest in S3.
3.  **Partitioning:**
    -   **Tier 1 (High Priority):** Dedicated GPUs. No sharing.
    -   **Tier 2 (Batch):** MIG-partitioned GPUs.
    -   **Tier 3 (Dev):** Time-sliced GPUs (MPS).
4.  **Isolation:** K8s Namespaces + NetworkPolicies prevent Team A from calling Team B's model.

## 16. Summary

| Level | Technology | Strategy |
| :--- | :--- | :--- |
| **Cluster** | Kubernetes | Namespaces, Quotas |
| **Node** | Kubelet | Requests/Limits |
| **Device** | NVIDIA MIG | Hardware Partitioning |
| **Workload** | Ray / Volcano | Gang Scheduling |
| **Cost** | Spot Instances | Checkpointing, Elasticity |

---

**Originally published at:** [arunbaby.com/ml-system-design/0036-resource-partitioning](https://www.arunbaby.com/ml-system-design/0036-resource-partitioning/)
