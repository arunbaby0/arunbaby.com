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

- **Isolation:** K8s Namespaces + NetworkPolicies prevent Team A from calling Team B's model.

## 16. Deep Dive: Preemption and Priority Classes

**Problem:** High-priority job arrives, but cluster is full.

**Solution:** Preempt (kill) low-priority jobs.

**Kubernetes PriorityClass:**
```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority
value: 1000
globalDefault: false
description: "For production inference"
---
apiVersion: v1
kind: Pod
metadata:
  name: inference-pod
spec:
  priorityClassName: high-priority
  containers:
  - name: model
    image: bert-serving
```

**Preemption Logic:**
1. Scheduler sees high-priority pod can't fit.
2. Finds low-priority pods to evict.
3. Sends `SIGTERM` to those pods.
4. Waits for graceful shutdown (default 30s).
5. Schedules high-priority pod.

**Best Practice:**
- **Production Inference:** Priority 1000.
- **Training:** Priority 500.
- **Dev/Notebooks:** Priority 100.

## 17. Deep Dive: Resource Fragmentation Problem

**Scenario:** Cluster has 100 GPUs total, but they're spread across 100 nodes (1 GPU each).
A job needs 8 GPUs on the same node (for NVLink). **It can't run.**

**This is fragmentation.**

**Solutions:**

**1. Bin Packing (MostRequestedPriority):**
- Pack pods tightly to leave some nodes completely empty.
- Empty nodes can be terminated (autoscaling) or reserved for large jobs.

**2. Defragmentation (Descheduler):**
- Periodically evict pods from underutilized nodes.
- Re-schedule them to consolidate resources.

**3. Topology-Aware Scheduling:**
- Prefer nodes with multiple GPUs when scheduling multi-GPU jobs.

**Code (Custom Scheduler Plugin):**
```go
func (p *GPUTopologyPlugin) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
    nodeInfo, _ := p.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
    
    // Count available GPUs on this node
    availableGPUs := nodeInfo.Allocatable.ScalarResources["nvidia.com/gpu"]
    
    // Prefer nodes with more GPUs
    return availableGPUs * 10, nil
}
```

## 18. Advanced: NUMA-Aware Scheduling

**NUMA (Non-Uniform Memory Access):** Modern servers have multiple CPU sockets. Memory attached to Socket 0 is faster for cores on Socket 0.

**Problem:** If a pod's CPUs are on Socket 0 but its memory is on Socket 1, performance degrades (cross-socket memory access).

**Solution: Topology Manager (K8s 1.18+):**
```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: training
    resources:
      requests:
        cpu: "16"
        memory: "64Gi"
  # Topology Manager ensures CPUs and memory are on the same NUMA node
```

**Policies:**
- `none`: No alignment (default).
- `best-effort`: Try to align, but don't fail if impossible.
- `restricted`: Only allow if alignment is possible.
- `single-numa-node`: All resources must be on a single NUMA node.

## 19. Advanced: Network Topology-Aware Placement

**Problem:** Distributed training has massive inter-GPU communication (All-Reduce).

**Network Hierarchy:**
```
GPU 0 ←→ GPU 1  (NVLink: 600 GB/s)
  ↓       ↓
Node 0 ←→ Node 1  (InfiniBand: 200 GB/s)
  ↓       ↓
Rack 0 ←→ Rack 1  (Ethernet: 100 GB/s)
```

**Optimization:** Place all 8 GPUs of a job on the same node (NVLink) > same rack (IB) > different racks (Ethernet).

**Implementation (Volcano Topology Plugin):**
```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: distributed-training
spec:
  minMember: 8
  queue: default
  # Topology constraint
  affinity:
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            job: distributed-training
        topologyKey: "topology.kubernetes.io/zone"  # Same rack
```

## 20. Case Study: Google Borg (Predecessor to Kubernetes)

**Borg** is Google's internal cluster manager (2003-present).

**Key Innovations:**
1. **Allocs (Allocations):** Reserve resources for future use.
   - "I'll need 100 GPUs tomorrow at 9am."
2. **Quota Reclamation:** If Team A's quota is unused, Team B can borrow it (but gets preempted when A returns).
3. **Borgmaster:** Centralized scheduler (handles 10k+ machines).
4. **Borglet:** Agent on each machine (like Kubelet).

**Lessons for K8s:**
- **Declarative API:** "I want 10 replicas" (not "start pod 1, start pod 2...").
- **Labels/Selectors:** Flexible grouping.
- **Reconciliation Loop:** Continuously drive actual state toward desired state.

## 21. Production Monitoring and Debugging

**Key Metrics:**

1. **Cluster Utilization:**
   ```
   GPU_Utilization = (Allocated_GPUs / Total_GPUs) * 100
   ```
   - **Target:** > 80%.
   - **Alert:** If < 60% for > 1 hour, investigate.

2. **Pending Pods:**
   - Pods stuck in `Pending` state indicate scheduling failures.
   - **Reasons:** Insufficient resources, taints, affinity rules.

3. **Preemption Rate:**
   - How often are low-priority jobs killed?
   - **High rate:** Users frustrated. Consider adding capacity.

**Debugging Tools:**
```bash
# Why is my pod pending?
kubectl describe pod my-pod | grep -A 10 Events

# Common reasons:
# - "Insufficient nvidia.com/gpu"
# - "Node had taints that the pod didn't tolerate"
# - "PodGroup is not ready" (gang scheduling)

# Check node resources
kubectl describe node gpu-node-01 | grep -A 5 Allocated

# Check quota
kubectl describe resourcequota -n team-nlp
```

## 22. Common Pitfalls and How to Avoid Them

**Pitfall 1: Setting Limits Too High**
- `limits.memory: 1TB` on a 512GB node.
- **Result:** Pod gets scheduled, then OOMKilled.
- **Fix:** Set `limits` close to `requests`.

**Pitfall 2: Forgetting Gang Scheduling**
- Distributed training job requests 100 GPUs.
- K8s schedules 99, waits for 1.
- **Result:** Deadlock (99 GPUs wasted).
- **Fix:** Use Volcano/Kueue.

**Pitfall 3: Ignoring Topology**
- 8-GPU job spread across 8 nodes.
- **Result:** 10x slower (network bottleneck).
- **Fix:** Use affinity rules or topology-aware scheduler.

**Pitfall 4: No Resource Quotas**
- One team launches 1000 pods, starves everyone.
- **Fix:** Enforce `ResourceQuota` per namespace.

**Pitfall 5: Not Monitoring Fragmentation**
- Cluster is "full" but no single node has 8 GPUs.
- **Fix:** Run descheduler periodically.

## 23. Advanced: Elastic Training with TorchElastic

**Problem:** Spot instances can be reclaimed mid-training.

**TorchElastic (PyTorch 1.9+):**
- **Rendezvous:** Workers discover each other dynamically.
- **Fault Tolerance:** If a worker dies, remaining workers re-form the group.
- **Elasticity:** Can add/remove workers mid-training.

**Code:**
```python
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

@record
def train():
    dist.init_process_group(backend="nccl")
    
    # Training loop
    for epoch in range(100):
        for batch in dataloader:
            # If a worker dies here, TorchElastic handles it
            loss = model(batch)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    # Launch with torchrun (replaces torch.distributed.launch)
    # torchrun --nproc_per_node=8 --nnodes=10 train.py
    train()
```

**Benefit:** Training survives spot interruptions without manual intervention.

## 24. Further Reading

1. **"Large-scale cluster management at Google with Borg" (Verma et al., 2015):** The Borg paper.
2. **"Kubernetes Scheduling" (K8s Docs):** Official scheduler documentation.
3. **"Volcano: A Cloud Native Batch System" (Volcano Team):** Gang scheduling for ML.
4. **"Ray: A Distributed Framework for Emerging AI Applications" (Moritz et al., 2018):** Ray architecture.
5. **"NVIDIA Multi-Instance GPU User Guide":** MIG configuration and best practices.

- **NVIDIA Multi-Instance GPU User Guide:** MIG configuration and best practices.

## 24. Interview Questions for Resource Partitioning

**Q1: How would you design a scheduler for a multi-tenant GPU cluster?**
*Answer:* Use Kubernetes with custom scheduler plugins. Implement:
- **Gang scheduling** (Volcano) for distributed training
- **Priority classes** for production vs. dev workloads
- **Resource quotas** per team/namespace
- **Topology-aware placement** to minimize network latency
- **Preemption** for high-priority jobs

**Q2: What's the difference between MIG and time-slicing?**
*Answer:* 
- **MIG:** Hardware partitioning. Strong isolation, dedicated memory, no context switching overhead. Only on A100/H100.
- **Time-Slicing (MPS):** Software multiplexing. Weak isolation, shared memory, context switching overhead. Works on any GPU.

**Q3: How do you handle a job that requests 100 GPUs but only 99 are available?**
*Answer:* This is the gang scheduling problem. Solutions:
- Use **Volcano/Kueue** to ensure all-or-nothing scheduling
- Implement **backfilling**: If a small job can run without blocking the large job, schedule it
- **Preemption**: Kill low-priority jobs to free resources

**Q4: How would you optimize cost for training on spot instances?**
*Answer:*
- **Checkpointing** every 10 minutes to S3
- **TorchElastic** for fault tolerance
- **Mixed cluster**: On-demand for head node, spot for workers
- **Diversification**: Use multiple instance types to reduce interruption probability

**Q5: What metrics would you monitor for cluster health?**
*Answer:*
- **Utilization**: GPU/CPU/Memory usage (target >80%)
- **Pending pods**: Indicates scheduling bottlenecks
- **Preemption rate**: High rate = users frustrated
- **Job completion time**: Detect performance degradation
- **Network bandwidth**: Detect congestion

## 25. Cost Optimization Strategies

**1. Right-Sizing:**
- Don't request 8 GPUs if you only use 4.
- **Tool:** Profile with `nvidia-smi` to measure actual usage.

**2. Spot Instance Strategies:**
```python
# Diversify across instance types
instance_types = ['p3.8xlarge', 'p3.16xlarge', 'p4d.24xlarge']

# Bid strategy: Max price = On-Demand price
for instance_type in instance_types:
    launch_spot_instance(
        instance_type=instance_type,
        max_price=get_on_demand_price(instance_type)
    )
```

**3. Autoscaling Policies:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gpu-autoscaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70
```

**4. Idle Resource Reclamation:**
- **Descheduler**: Evict pods from underutilized nodes.
- **Cluster Autoscaler**: Terminate empty nodes after 10 minutes.

**5. Batch Job Scheduling:**
- Run low-priority batch jobs during off-peak hours (nights, weekends).
- **Savings:** 50-70% by using cheaper spot instances.

**Cost Breakdown Example:**
```
Training GPT-3 (175B params):
- Hardware: 10,000 V100s for 1 month
- On-Demand: $3/hr/GPU × 10,000 × 720 hrs = $21.6M
- Spot (70% discount): $6.5M
- Savings: $15.1M
```

## 26. Ethical Considerations

**1. Energy Consumption:**
- Training GPT-3 consumed ~1,287 MWh (equivalent to 120 US homes for a year).
- **Mitigation:** 
  - Use renewable energy data centers (Google, AWS have carbon-neutral regions)
  - Efficient architectures (MoE, sparse models)
  - Model distillation (train large, deploy small)

**2. Access Inequality:**
- Only large organizations can afford 10k GPU clusters.
- **Impact:** Concentrates AI research in Big Tech.
- **Solutions:**
  - Public compute grants (NSF, EU HPC)
  - Open-source pre-trained models (Hugging Face)
  - Federated learning (train on distributed data)

**3. Resource Hoarding:**
- One team reserves 1000 GPUs "just in case".
- **Fix:** Enforce quotas, implement "use it or lose it" policies.

**4. Bias in Allocation:**
- Prioritizing certain teams/projects over others.
- **Transparency:** Publish allocation policies, audit logs.

## 27. Advanced: Multi-Cloud Resource Partitioning

**Scenario:** Burst to AWS when on-prem cluster is full.

**Architecture:**
```
┌──────────────┐
│  On-Prem K8s │ (Primary)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Kubefed     │ (Federation)
└──────┬───────┘
       │
   ┌───┴────┐
   ▼        ▼
┌─────┐  ┌─────┐
│ AWS │  │ GCP │ (Burst)
└─────┘  └─────┘
```

**Challenges:**
1. **Data Transfer:** Moving training data to cloud is slow.
   - **Solution:** Replicate data to S3/GCS in advance.
2. **Network Latency:** Cross-cloud communication is slow.
   - **Solution:** Use cloud for independent jobs, not distributed training.
3. **Cost:** Egress fees can be expensive.
   - **Solution:** Minimize data movement, use cloud-native storage.

- **Solution:** Minimize data movement, use cloud-native storage.

## 28. Production Deployment Checklist

**Before launching a multi-tenant ML cluster:**

**Infrastructure:**
- [ ] Set up Kubernetes cluster with GPU support
- [ ] Install NVIDIA device plugin
- [ ] Configure MIG partitions (if using A100/H100)
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure logging (ELK stack or CloudWatch)

**Resource Management:**
- [ ] Define ResourceQuotas for each team/namespace
- [ ] Create PriorityClasses (production, training, dev)
- [ ] Install gang scheduler (Volcano or Kueue)
- [ ] Configure autoscaling policies
- [ ] Set up descheduler for defragmentation

**Security:**
- [ ] Enable RBAC (Role-Based Access Control)
- [ ] Configure NetworkPolicies for isolation
- [ ] Set up Pod Security Policies
- [ ] Enable audit logging
- [ ] Implement secrets management (Vault or AWS Secrets Manager)

**Cost Optimization:**
- [ ] Enable cluster autoscaler
- [ ] Configure spot instance policies
- [ ] Set up cost monitoring (Kubecost)
- [ ] Implement idle resource reclamation
- [ ] Define off-peak batch job schedules

**Disaster Recovery:**
- [ ] Set up automated backups (etcd, persistent volumes)
- [ ] Test failover procedures
- [ ] Document runbooks for common issues
- [ ] Configure alerts for critical failures
- [ ] Implement multi-region redundancy (if needed)

## 29. Further Reading

1. **"Large-scale cluster management at Google with Borg" (Verma et al., 2015):** The Borg paper.
2. **"Kubernetes Scheduling" (K8s Docs):** Official scheduler documentation.
3. **"Volcano: A Cloud Native Batch System" (Volcano Team):** Gang scheduling for ML.
4. **"Ray: A Distributed Framework for Emerging AI Applications" (Moritz et al., 2018):** Ray architecture.
5. **"NVIDIA Multi-Instance GPU User Guide":** MIG configuration and best practices.

## 30. Conclusion

Resource partitioning in ML clusters is a balancing act between **utilization** (pack jobs tightly), **fairness** (everyone gets their share), and **performance** (avoid interference). The shift from static partitioning to dynamic, topology-aware scheduling has enabled organizations to train models 10x larger on the same hardware budget. As models grow (GPT-5 will likely need 100k+ GPUs), the challenges of gang scheduling, fault tolerance, and network optimization will only intensify. The future lies in **intelligent schedulers** that understand model characteristics (memory footprint, communication patterns) and **elastic training** that adapts to resource availability in real-time.

## 31. Summary

| Level | Technology | Strategy |
| :--- | :--- | :--- |
| **Cluster** | Kubernetes | Namespaces, Quotas |
| **Node** | Kubelet | Requests/Limits |
| **Device** | NVIDIA MIG | Hardware Partitioning |
| **Workload** | Ray / Volcano | Gang Scheduling |
| **Cost** | Spot Instances | Checkpointing, Elasticity |
| **Performance** | Topology-Aware | NUMA, Network Placement |

---

**Originally published at:** [arunbaby.com/ml-system-design/0036-resource-partitioning](https://www.arunbaby.com/ml-system-design/0036-resource-partitioning/)
