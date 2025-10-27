---
title: "Resource Allocation for ML"
day: 13
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - resource-allocation
  - infrastructure
  - optimization
  - cost-management
  - capacity-planning
  - kubernetes
  - distributed-systems
subdomain: "Infrastructure"
tech_stack: [Kubernetes, Ray, Kubeflow, Prometheus, Grafana, TensorFlow, PyTorch]
scale: "1000s of models, 100K+ GPU hours/month, multi-region"
companies: [Google, Meta, Netflix, Uber, Airbnb]
related_dsa_day: 13
related_ml_day: 13
related_speech_day: 13
---

**Build production ML infrastructure that dynamically allocates resources using greedy optimization to maximize throughput and minimize costs.**

## Problem Statement

Design a **Resource Allocation System for ML** that efficiently manages compute resources (CPUs, GPUs, memory, storage) across hundreds of ML models and workflows.

### Functional Requirements

1. **Dynamic allocation:** Assign resources to training/inference jobs based on priority, deadlines, and resource availability
2. **Multi-tenancy:** Support multiple teams/projects with fair resource sharing
3. **Cost optimization:** Minimize cloud spending while meeting SLAs
4. **Auto-scaling:** Scale resources up/down based on demand
5. **Resource types:** Handle heterogeneous resources (different GPU types, CPU configurations)
6. **Queue management:** Prioritize jobs intelligently
7. **Preemption:** Allow high-priority jobs to preempt lower-priority ones
8. **Monitoring:** Track resource utilization and costs in real-time

### Non-Functional Requirements

1. **Latency:** Resource allocation decisions in <100ms
2. **Utilization:** >80% GPU utilization during peak hours
3. **Fairness:** No team monopolizes resources
4. **Availability:** 99.9% uptime
5. **Scale:** Support 1000+ concurrent jobs, 10K+ GPUs
6. **Cost efficiency:** Reduce cloud spending by 30-50% through optimization
7. **Elasticity:** Handle 10x traffic spikes

## Understanding the Requirements

This is the **infrastructure backbone** of any ML organization. Poor resource allocation leads to:

- **Wasted money:** Idle GPUs cost $1-3/hour each
- **Slow iteration:** Researchers waiting hours for resources
- **Missed deadlines:** Production models not trained on time
- **Unfairness:** Some teams starve while others over-provision

### Scale Context

At a typical large tech company:
- **Google/Meta:** 100K+ GPUs, millions of training jobs/month
- **Uber/Netflix:** 10K+ GPUs, thousands of models in production
- **Startup (Series B+):** 100-1000 GPUs, hundreds of models

**Cost implications:**
- A100 GPU: ~$3/hour on AWS/GCP
- 1000 GPUs at 50% utilization waste: $1.5M/month
- **Goal:** Increase utilization from 50% → 85% saves $1M+/month

### Key Challenges

1. **Heterogeneous resources:** V100 vs A100 vs H100, different memory sizes
2. **Variable job durations:** 5-minute inference vs 3-day training
3. **Priority conflicts:** Production inference vs experimental training
4. **Resource fragmentation:** Many small jobs prevent large jobs from running
5. **Multi-dimensional constraints:** GPU + memory + network bandwidth
6. **Cost vs performance:** Spot instances are cheap but can be preempted

### The Greedy Optimization Connection

Just like the **Container With Most Water** problem:

| Container Problem | Resource Allocation |
|-------------------|---------------------|
| Two lines (heights) | Multiple resource constraints (GPU/memory) |
| Bottleneck (shorter line) | Resource bottleneck (GPU/memory/bandwidth) |
| Maximize area | Maximize utilization × performance |
| Greedy choice: move shorter pointer | Greedy choice: allocate to bottleneck first |
| O(N) efficiency | O(N) scheduling decisions |

**Core insight:** The bottleneck resource determines system throughput, so optimize for it first.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Resource Allocation System               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐         ┌──────────────────────────────────────┐
│   Clients   │         │         Control Plane                 │
│             │         │                                       │
│ - Training  │────────▶│  ┌────────────────────────────┐     │
│ - Inference │         │  │  Scheduler                 │     │
│ - Tuning    │         │  │  - Job queue               │     │
│ - Batch     │         │  │  - Priority management     │     │
└─────────────┘         │  │  - Resource matching       │     │
                        │  └────────────┬───────────────┘     │
                        │               │                      │
                        │               ▼                      │
                        │  ┌────────────────────────────┐     │
                        │  │  Allocator (Greedy)        │     │
┌─────────────┐         │  │  - Bin packing             │     │
│  Monitoring │◀────────│  │  - Preemption logic        │     │
│             │         │  │  - Fair share calculation  │     │
│ - Prometheus│         │  └────────────┬───────────────┘     │
│ - Grafana   │         │               │                      │
│ - Alerts    │         │               ▼                      │
└─────────────┘         │  ┌────────────────────────────┐     │
                        │  │  Resource Manager          │     │
                        │  │  - Available resources     │     │
                        │  │  - Usage tracking          │     │
                        │  │  - Cost accounting         │     │
                        │  └────────────┬───────────────┘     │
                        └───────────────┼────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────┐
│                     Data Plane                                │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Cluster 1  │  │  Cluster 2  │  │  Cluster N  │         │
│  │             │  │             │  │             │         │
│  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │         │
│  │  │GPU Pod│  │  │  │GPU Pod│  │  │  │GPU Pod│  │         │
│  │  └───────┘  │  │  └───────┘  │  │  └───────┘  │         │
│  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │         │
│  │  │GPU Pod│  │  │  │CPU Pod│  │  │  │TPU Pod│  │         │
│  │  └───────┘  │  │  └───────┘  │  │  └───────┘  │         │
│  │             │  │             │  │             │         │
│  │  us-west-1  │  │  us-east-1  │  │  eu-west-1  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Scheduler:** Receives jobs, manages queue, makes assignment decisions
2. **Allocator:** Implements greedy bin-packing algorithm
3. **Resource Manager:** Tracks available resources across clusters
4. **Monitoring:** Real-time metrics and cost tracking
5. **Data Plane:** Kubernetes clusters running actual ML workloads

## Component Deep-Dives

### 1. Scheduler - Job Queue and Prioritization

The scheduler maintains a priority queue of pending jobs and makes allocation decisions.

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import heapq

class JobPriority(Enum):
    """Job priority levels."""
    CRITICAL = 0      # Production inference (P0)
    HIGH = 1          # Production training (P1)
    MEDIUM = 2        # Experimentation (P2)
    LOW = 3           # Batch processing (P3)
    PREEMPTIBLE = 4   # Can be killed anytime

@dataclass
class ResourceRequest:
    """Resources needed for a job."""
    cpus: int
    memory_gb: int
    gpus: int
    gpu_type: str  # "V100", "A100", "H100"
    disk_gb: int
    
    def __hash__(self):
        return hash((self.cpus, self.memory_gb, self.gpus, self.gpu_type))

@dataclass
class Job:
    """ML job to be scheduled."""
    job_id: str
    user_id: str
    team_id: str
    priority: JobPriority
    resources: ResourceRequest
    estimated_duration_hours: float
    deadline: Optional[datetime]
    submitted_at: datetime
    started_at: Optional[datetime] = None
    can_preempt: bool = False
    
    def __lt__(self, other):
        """
        Priority comparison for heap queue.
        
        Sorting criteria (in order):
        1. Priority level (lower is better)
        2. Deadline proximity (earlier deadline wins)
        3. Submission time (FIFO for same priority)
        """
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        
        # If both have deadlines, prioritize closer deadline
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        
        # Jobs with deadlines beat those without
        if self.deadline:
            return True
        if other.deadline:
            return False
        
        # FIFO for same priority
        return self.submitted_at < other.submitted_at
    
    @property
    def wait_time(self) -> timedelta:
        """How long has this job been waiting?"""
        if self.started_at:
            return self.started_at - self.submitted_at
        return datetime.now() - self.submitted_at
    
    @property
    def is_deadline_critical(self) -> bool:
        """Is this job close to missing its deadline?"""
        if not self.deadline:
            return False
        
        time_until_deadline = self.deadline - datetime.now()
        return time_until_deadline < timedelta(hours=self.estimated_duration_hours * 1.5)


class JobScheduler:
    """
    Priority-based job scheduler with fair share and preemption.
    
    Greedy strategy:
    1. Prioritize high-priority jobs
    2. Within same priority, consider deadlines and wait time
    3. Implement fair share to prevent starvation
    4. Preempt low-priority jobs for critical ones
    """
    
    def __init__(self, fair_share_window_hours: int = 24):
        self.pending_jobs: List[Job] = []  # Min heap by priority
        self.running_jobs: Dict[str, Job] = {}
        self.fair_share_window = timedelta(hours=fair_share_window_hours)
        self.team_usage: Dict[str, float] = {}  # GPU-hours used
        self.team_quotas: Dict[str, float] = {}  # GPU-hours quota
        
        heapq.heapify(self.pending_jobs)
    
    def submit_job(self, job: Job):
        """
        Submit a new job to the scheduler.
        
        Why heap?
        - O(log N) insertion
        - O(1) to peek highest priority
        - Automatically maintains priority order
        """
        heapq.heappush(self.pending_jobs, job)
    
    def get_next_job(self, available_resources: ResourceRequest) -> Optional[Job]:
        """
        Get the next job to schedule using greedy algorithm.
        
        Greedy strategy (like container problem):
        1. Check highest priority job
        2. If it fits, schedule it
        3. If not, check if we can preempt lower-priority jobs
        4. Consider fair share quotas
        
        Time: O(log N) - heap operations
        """
        if not self.pending_jobs:
            return None
        
        # Peek at highest priority job (don't pop yet)
        candidate = self.pending_jobs[0]
        
        # Check if resources are sufficient
        if self._can_fit(candidate.resources, available_resources):
            # Check fair share quota
            if self._check_fair_share(candidate):
                return heapq.heappop(self.pending_jobs)
        
        # High-priority job can't fit - check for preemption
        if candidate.priority in [JobPriority.CRITICAL, JobPriority.HIGH]:
            preempt_candidate = self._find_preemptible_job(candidate.resources)
            if preempt_candidate:
                # Preempt lower-priority job
                self._preempt_job(preempt_candidate)
                return heapq.heappop(self.pending_jobs)
        
        # Try to find smaller job that fits (avoid fragmentation)
        for i, job in enumerate(self.pending_jobs):
            if self._can_fit(job.resources, available_resources):
                if self._check_fair_share(job):
                    # Remove from middle of heap (O(N) worst case)
                    self.pending_jobs[i] = self.pending_jobs[-1]
                    self.pending_jobs.pop()
                    heapq.heapify(self.pending_jobs)
                    return job
        
        return None
    
    def _can_fit(self, request: ResourceRequest, available: ResourceRequest) -> bool:
        """
        Check if requested resources fit in available resources.
        
        All dimensions must fit (like multi-dimensional bin packing).
        """
        return (
            request.cpus <= available.cpus and
            request.memory_gb <= available.memory_gb and
            request.gpus <= available.gpus and
            (request.gpu_type == available.gpu_type or available.gpu_type == "any") and
            request.disk_gb <= available.disk_gb
        )
    
    def _check_fair_share(self, job: Job) -> bool:
        """
        Check if team is within their fair share quota.
        
        Prevent one team from monopolizing resources.
        """
        if job.team_id not in self.team_quotas:
            return True  # No quota set
        
        current_usage = self.team_usage.get(job.team_id, 0.0)
        quota = self.team_quotas[job.team_id]
        
        # Allow critical jobs to exceed quota
        if job.priority == JobPriority.CRITICAL:
            return True
        
        # Check if within quota (with 10% grace)
        return current_usage < quota * 1.1
    
    def _find_preemptible_job(self, needed: ResourceRequest) -> Optional[str]:
        """
        Find a running job that can be preempted to free resources.
        
        Greedy choice: preempt lowest-priority job that frees enough resources.
        """
        preemptible = [
            (job_id, job) 
            for job_id, job in self.running_jobs.items()
            if job.can_preempt and job.priority == JobPriority.PREEMPTIBLE
        ]
        
        # Sort by least resource usage (preempt smallest job if possible)
        preemptible.sort(key=lambda x: x[1].resources.gpus)
        
        # Greedy: find first job that frees enough resources
        for job_id, job in preemptible:
            if self._can_fit(needed, job.resources):
                return job_id
        
        return None
    
    def _preempt_job(self, job_id: str):
        """
        Preempt a running job.
        
        In production:
        - Send SIGTERM, wait for graceful shutdown
        - Save checkpoint if training job
        - Re-queue job with higher priority
        """
        job = self.running_jobs.pop(job_id)
        
        # Re-queue with higher priority (prevent starvation)
        job.priority = JobPriority.MEDIUM
        job.submitted_at = datetime.now()
        self.submit_job(job)
    
    def mark_started(self, job: Job):
        """Mark job as started."""
        job.started_at = datetime.now()
        self.running_jobs[job.job_id] = job
    
    def mark_completed(self, job_id: str):
        """Mark job as completed and update usage."""
        if job_id in self.running_jobs:
            job = self.running_jobs.pop(job_id)
            
            # Update team usage for fair share
            gpu_hours = job.resources.gpus * (
                (datetime.now() - job.started_at).total_seconds() / 3600
            )
            self.team_usage[job.team_id] = self.team_usage.get(job.team_id, 0) + gpu_hours
```

### 2. Allocator - Greedy Bin Packing

The allocator implements a greedy algorithm to pack jobs onto available resources.

```python
from typing import List, Dict, Tuple, Optional

@dataclass
class Node:
    """A compute node (e.g., GPU instance)."""
    node_id: str
    available: ResourceRequest
    capacity: ResourceRequest
    cost_per_hour: float
    region: str
    is_spot: bool = False  # Spot/preemptible instance
    
    @property
    def utilization(self) -> float:
        """GPU utilization percentage."""
        if self.capacity.gpus == 0:
            return 0.0
        return 1.0 - (self.available.gpus / self.capacity.gpus)


class ResourceAllocator:
    """
    Greedy resource allocator using bin packing.
    
    Similar to Container With Most Water:
    - Multiple bins (nodes) with capacities
    - Jobs to pack (containers to fill)
    - Greedy choice: pack job to minimize waste
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.allocations: Dict[str, str] = {}  # job_id -> node_id
    
    def add_node(self, node: Node):
        """Add a compute node to the pool."""
        self.nodes[node.node_id] = node
    
    def allocate(self, job: Job) -> Optional[str]:
        """
        Allocate resources for a job using greedy algorithm.
        
        Greedy strategies (in order of preference):
        1. Best fit: minimize wasted resources
        2. First fit: fill nodes to capacity (consolidation)
        3. Worst fit: spread load evenly
        
        We use BEST FIT for cost efficiency.
        
        Returns:
            node_id if allocation successful, None otherwise
        """
        best_node = None
        min_waste = float('inf')
        
        for node_id, node in self.nodes.items():
            # Check if job fits
            if not self._can_allocate(job.resources, node):
                continue
            
            # Calculate waste (remaining resources after allocation)
            waste = self._calculate_waste(job.resources, node)
            
            # Greedy choice: minimize waste (best fit)
            if waste < min_waste:
                min_waste = waste
                best_node = node_id
        
        if best_node:
            self._allocate_to_node(job, best_node)
            return best_node
        
        return None
    
    def allocate_multi_node(self, job: Job, max_nodes: int = 8) -> Optional[List[str]]:
        """
        Allocate job across multiple nodes (for distributed training).
        
        Greedy approach:
        1. Find nodes with most available resources
        2. Allocate greedily to top candidates
        3. Prefer nodes in same region (reduce network latency)
        """
        if max_nodes == 1:
            node = self.allocate(job)
            return [node] if node else None
        
        # Calculate per-node resource requirement
        gpus_per_node = (job.resources.gpus + max_nodes - 1) // max_nodes
        per_node_request = ResourceRequest(
            cpus=job.resources.cpus // max_nodes,
            memory_gb=job.resources.memory_gb // max_nodes,
            gpus=gpus_per_node,
            gpu_type=job.resources.gpu_type,
            disk_gb=job.resources.disk_gb // max_nodes
        )
        
        # Find candidate nodes
        candidates = [
            (node_id, node)
            for node_id, node in self.nodes.items()
            if self._can_allocate(per_node_request, node)
        ]
        
        if len(candidates) < max_nodes:
            return None  # Not enough nodes
        
        # Greedy: sort by region to co-locate
        region_groups = {}
        for node_id, node in candidates:
            region_groups.setdefault(node.region, []).append((node_id, node))
        
        # Pick largest region group
        best_region = max(region_groups.keys(), key=lambda r: len(region_groups[r]))
        best_nodes = region_groups[best_region][:max_nodes]
        
        # Allocate to all nodes
        allocated = []
        for node_id, node in best_nodes:
            self._allocate_to_node_partial(per_node_request, node_id)
            allocated.append(node_id)
        
        return allocated
    
    def deallocate(self, job_id: str):
        """Release resources when job completes."""
        if job_id not in self.allocations:
            return
        
        node_id = self.allocations.pop(job_id)
        # In practice, restore node.available resources
    
    def _can_allocate(self, request: ResourceRequest, node: Node) -> bool:
        """Check if request fits in node."""
        return (
            request.cpus <= node.available.cpus and
            request.memory_gb <= node.available.memory_gb and
            request.gpus <= node.available.gpus and
            request.gpu_type == node.capacity.gpu_type and
            request.disk_gb <= node.available.disk_gb
        )
    
    def _calculate_waste(self, request: ResourceRequest, node: Node) -> float:
        """
        Calculate resource waste if we allocate request to node.
        
        Waste metric: sum of fractional unused resources.
        Lower waste = better fit.
        """
        cpu_waste = (node.available.cpus - request.cpus) / node.capacity.cpus
        mem_waste = (node.available.memory_gb - request.memory_gb) / node.capacity.memory_gb
        gpu_waste = (node.available.gpus - request.gpus) / node.capacity.gpus if node.capacity.gpus > 0 else 0
        
        # Weighted sum (GPU waste matters most)
        return 0.5 * gpu_waste + 0.3 * mem_waste + 0.2 * cpu_waste
    
    def _allocate_to_node(self, job: Job, node_id: str):
        """Actually allocate job to node."""
        node = self.nodes[node_id]
        
        # Decrease available resources
        node.available.cpus -= job.resources.cpus
        node.available.memory_gb -= job.resources.memory_gb
        node.available.gpus -= job.resources.gpus
        node.available.disk_gb -= job.resources.disk_gb
        
        # Track allocation
        self.allocations[job.job_id] = node_id
    
    def _allocate_to_node_partial(self, request: ResourceRequest, node_id: str):
        """Allocate partial resources (for multi-node jobs)."""
        node = self.nodes[node_id]
        node.available.cpus -= request.cpus
        node.available.memory_gb -= request.memory_gb
        node.available.gpus -= request.gpus
        node.available.disk_gb -= request.disk_gb
    
    def get_utilization_stats(self) -> Dict:
        """Calculate cluster utilization statistics."""
        total_gpus = sum(node.capacity.gpus for node in self.nodes.values())
        used_gpus = sum(
            node.capacity.gpus - node.available.gpus 
            for node in self.nodes.values()
        )
        
        return {
            "total_gpus": total_gpus,
            "used_gpus": used_gpus,
            "utilization": used_gpus / total_gpus if total_gpus > 0 else 0,
            "num_nodes": len(self.nodes),
            "active_jobs": len(self.allocations)
        }
```

### 3. Auto-Scaler - Dynamic Resource Provisioning

```python
from typing import List
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_nodes: int
    max_nodes: int
    target_utilization: float = 0.80
    scale_up_threshold: float = 0.90
    scale_down_threshold: float = 0.50
    cooldown_minutes: int = 5
    

class AutoScaler:
    """
    Auto-scaler for dynamic resource provisioning.
    
    Greedy scaling decisions:
    - Scale up: when utilization > threshold OR queue is growing
    - Scale down: when utilization < threshold AND queue is empty
    """
    
    def __init__(self, policy: ScalingPolicy, allocator: ResourceAllocator):
        self.policy = policy
        self.allocator = allocator
        self.last_scale_action = datetime.now()
        self.pending_queue_size_history: List[int] = []
    
    def should_scale_up(self, current_utilization: float, queue_size: int) -> bool:
        """
        Decide if we should scale up.
        
        Greedy conditions:
        1. High utilization (>90%)
        2. Growing queue
        3. Not in cooldown period
        """
        # Check cooldown
        if (datetime.now() - self.last_scale_action).total_seconds() < self.policy.cooldown_minutes * 60:
            return False
        
        # Check if at max capacity
        if len(self.allocator.nodes) >= self.policy.max_nodes:
            return False
        
        # High utilization trigger
        if current_utilization > self.policy.scale_up_threshold:
            return True
        
        # Growing queue trigger
        self.pending_queue_size_history.append(queue_size)
        if len(self.pending_queue_size_history) > 5:
            self.pending_queue_size_history.pop(0)
        
        if len(self.pending_queue_size_history) >= 3:
            # Queue is growing consistently
            if all(
                self.pending_queue_size_history[i] < self.pending_queue_size_history[i+1]
                for i in range(len(self.pending_queue_size_history) - 1)
            ):
                return True
        
        return False
    
    def should_scale_down(self, current_utilization: float, queue_size: int) -> bool:
        """
        Decide if we should scale down.
        
        Conservative approach:
        - Only scale down if utilization is low AND queue is empty
        - Respect minimum nodes
        """
        # Check cooldown
        if (datetime.now() - self.last_scale_action).total_seconds() < self.policy.cooldown_minutes * 60:
            return False
        
        # Check if at min capacity
        if len(self.allocator.nodes) <= self.policy.min_nodes:
            return False
        
        # Low utilization and empty queue
        return (
            current_utilization < self.policy.scale_down_threshold and
            queue_size == 0
        )
    
    def scale_up(self, num_nodes: int = 1) -> List[str]:
        """
        Add nodes to the cluster.
        
        In practice:
        - Call cloud provider API (AWS/GCP/Azure)
        - Choose instance type based on queue composition
        - Prefer spot instances for cost savings
        """
        new_nodes = []
        for i in range(num_nodes):
            node_id = f"node-{len(self.allocator.nodes) + i}"
            
            # Create node (example: A100 instance)
            node = Node(
                node_id=node_id,
                available=ResourceRequest(
                    cpus=32,
                    memory_gb=244,
                    gpus=8,
                    gpu_type="A100",
                    disk_gb=1000
                ),
                capacity=ResourceRequest(
                    cpus=32,
                    memory_gb=244,
                    gpus=8,
                    gpu_type="A100",
                    disk_gb=1000
                ),
                cost_per_hour=24.48,  # AWS p4d.24xlarge pricing
                region="us-west-2",
                is_spot=True  # Use spot for cost savings
            )
            
            self.allocator.add_node(node)
            new_nodes.append(node_id)
        
        self.last_scale_action = datetime.now()
        return new_nodes
    
    def scale_down(self, num_nodes: int = 1) -> List[str]:
        """
        Remove nodes from the cluster.
        
        Greedy choice: remove least utilized nodes first.
        """
        # Find nodes with lowest utilization
        nodes_by_util = sorted(
            self.allocator.nodes.items(),
            key=lambda x: x[1].utilization
        )
        
        removed = []
        for node_id, node in nodes_by_util[:num_nodes]:
            if node.available.gpus == node.capacity.gpus:  # Fully idle
                del self.allocator.nodes[node_id]
                removed.append(node_id)
                # In practice: call cloud API to terminate instance
        
        if removed:
            self.last_scale_action = datetime.now()
        
        return removed
```

## Data Flow

### Job Submission to Completion

```
1. User submits job
   └─> API validates request
   └─> Scheduler adds to priority queue
   
2. Scheduler loop (every 1 second)
   └─> Get next job from queue (greedy priority)
   └─> Check resource availability
   └─> If available: allocate
   └─> If not: check preemption or wait
   
3. Allocator assigns job to node(s)
   └─> Best fit bin packing
   └─> Update node available resources
   └─> Send job to Kubernetes
   
4. Job runs on GPU pod
   └─> Training/inference executes
   └─> Metrics streamed to monitoring
   
5. Job completes
   └─> Resources released
   └─> Usage logged for billing
   └─> Fair share quotas updated
   
6. Auto-scaler (every 30 seconds)
   └─> Check utilization
   └─> Decide scale up/down
   └─> Provision/deprovision nodes
```

### Resource Allocation Decision Tree

```
Job arrives
    │
    ├─> Priority: CRITICAL?
    │   ├─> Yes: Preempt lower-priority jobs if needed
    │   └─> No: Continue
    │
    ├─> Resources available?
    │   ├─> Yes: Allocate immediately (greedy)
    │   └─> No: Continue
    │
    ├─> Team within quota?
    │   ├─> Yes: Continue
    │   └─> No: Wait or reject
    │
    ├─> Can preempt?
    │   ├─> Yes: Preempt and allocate
    │   └─> No: Add to queue
    │
    └─> Queue job by priority
```

## Scaling Strategies

### Horizontal Scaling

**Challenge:** How many nodes to add when scaling up?

```python
def calculate_scale_up_nodes(queue: List[Job], current_nodes: int) -> int:
    """
    Greedy calculation: how many nodes to add?
    
    Strategy:
    1. Calculate total resource requirements in queue
    2. Divide by node capacity
    3. Add 20% buffer
    4. Cap at max_nodes
    """
    if not queue:
        return 0
    
    # Aggregate resource needs
    total_gpus_needed = sum(job.resources.gpus for job in queue)
    total_memory_needed = sum(job.resources.memory_gb for job in queue)
    
    # Assume 8 GPUs per node (A100 instance)
    gpus_per_node = 8
    nodes_for_gpus = (total_gpus_needed + gpus_per_node - 1) // gpus_per_node
    
    # Add 20% buffer
    nodes_to_add = int(nodes_for_gpus * 1.2)
    
    return max(1, nodes_to_add)
```

### Vertical Scaling

**Challenge:** Should we use bigger instances?

| Instance Type | GPUs | Memory | Cost/hr | Use Case |
|---------------|------|--------|---------|----------|
| p3.2xlarge | 1×V100 | 61GB | $3.06 | Small jobs |
| p3.8xlarge | 4×V100 | 244GB | $12.24 | Medium jobs |
| p4d.24xlarge | 8×A100 | 1152GB | $32.77 | Large jobs |
| p5.48xlarge | 8×H100 | 2048GB | $98.32 | Frontier models |

**Greedy choice:** Match instance type to job requirements to minimize cost.

### Handling Spot Instance Interruptions

```python
class SpotInstanceManager:
    """
    Manage spot/preemptible instances for cost savings.
    
    Spot instances are 70-90% cheaper but can be interrupted.
    """
    
    def __init__(self):
        self.checkpointing_jobs: Dict[str, str] = {}  # job_id -> checkpoint_path
    
    def handle_interruption(self, node_id: str, notice_seconds: int = 120):
        """
        Handle spot instance interruption (2-minute warning).
        
        Greedy strategy:
        1. Save checkpoints for training jobs
        2. Re-queue jobs to on-demand instances
        3. Prioritize critical jobs
        """
        # Find jobs running on this node
        jobs_on_node = [
            job for job in running_jobs.values()
            if job.node_id == node_id
        ]
        
        for job in jobs_on_node:
            if job.priority in [JobPriority.CRITICAL, JobPriority.HIGH]:
                # Move to on-demand instance immediately
                self._migrate_to_on_demand(job)
            else:
                # Save checkpoint and re-queue
                self._checkpoint_and_requeue(job)
    
    def _checkpoint_and_requeue(self, job: Job):
        """Save model checkpoint and re-queue job."""
        # Trigger checkpoint save
        checkpoint_path = f"s3://checkpoints/{job.job_id}/latest"
        # ... save logic ...
        
        # Re-queue with checkpoint resume
        job.resume_from_checkpoint = checkpoint_path
        scheduler.submit_job(job)
```

## Implementation: Complete System

Here's a simplified but functional implementation:

```python
import asyncio
from typing import Dict, List
import logging

class MLResourceManager:
    """
    Complete ML resource allocation system.
    
    Integrates scheduler, allocator, and auto-scaler.
    """
    
    def __init__(self, scaling_policy: ScalingPolicy):
        self.scheduler = JobScheduler()
        self.allocator = ResourceAllocator()
        self.auto_scaler = AutoScaler(scaling_policy, self.allocator)
        self.logger = logging.getLogger(__name__)
        
        # Initialize with some nodes
        self._bootstrap_cluster()
    
    def _bootstrap_cluster(self):
        """Start with minimum nodes."""
        for i in range(self.auto_scaler.policy.min_nodes):
            node = Node(
                node_id=f"node-{i}",
                available=ResourceRequest(32, 244, 8, "A100", 1000),
                capacity=ResourceRequest(32, 244, 8, "A100", 1000),
                cost_per_hour=24.48,
                region="us-west-2"
            )
            self.allocator.add_node(node)
    
    async def run(self):
        """Main control loop."""
        self.logger.info("Starting ML Resource Manager")
        
        while True:
            try:
                # 1. Process pending jobs
                await self._schedule_jobs()
                
                # 2. Check auto-scaling
                await self._check_scaling()
                
                # 3. Update metrics
                self._update_metrics()
                
                # Sleep before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in control loop: {e}")
    
    async def _schedule_jobs(self):
        """Schedule pending jobs to available resources."""
        while True:
            # Get available resources
            stats = self.allocator.get_utilization_stats()
            
            # Get next job
            # Create aggregate ResourceRequest for available resources
            # (simplified - in practice, check per-node)
            total_available = self._get_total_available()
            
            job = self.scheduler.get_next_job(total_available)
            if not job:
                break  # No more jobs or no jobs fit
            
            # Allocate resources
            allocation = self.allocator.allocate(job)
            if allocation:
                self.scheduler.mark_started(job)
                self.logger.info(f"Scheduled job {job.job_id} to node {allocation}")
                
                # In production: submit to Kubernetes
                await self._submit_to_k8s(job, allocation)
            else:
                # Put back in queue
                self.scheduler.submit_job(job)
                break
    
    async def _check_scaling(self):
        """Check if we need to scale cluster."""
        stats = self.allocator.get_utilization_stats()
        queue_size = len(self.scheduler.pending_jobs)
        
        if self.auto_scaler.should_scale_up(stats["utilization"], queue_size):
            num_nodes = calculate_scale_up_nodes(
                self.scheduler.pending_jobs,
                stats["num_nodes"]
            )
            new_nodes = self.auto_scaler.scale_up(num_nodes)
            self.logger.info(f"Scaled up: added {len(new_nodes)} nodes")
        
        elif self.auto_scaler.should_scale_down(stats["utilization"], queue_size):
            removed = self.auto_scaler.scale_down(1)
            self.logger.info(f"Scaled down: removed {len(removed)} nodes")
    
    def _get_total_available(self) -> ResourceRequest:
        """Get aggregate available resources."""
        total_gpus = sum(node.available.gpus for node in self.allocator.nodes.values())
        # Simplified: return max available for any single node
        if not self.allocator.nodes:
            return ResourceRequest(0, 0, 0, "none", 0)
        
        max_node = max(self.allocator.nodes.values(), key=lambda n: n.available.gpus)
        return max_node.available
    
    async def _submit_to_k8s(self, job: Job, node_id: str):
        """Submit job to Kubernetes (placeholder)."""
        # In production: create Kubernetes Job/Pod
        # kubectl apply -f job.yaml
        pass
    
    def _update_metrics(self):
        """Update monitoring metrics."""
        stats = self.allocator.get_utilization_stats()
        # Send to Prometheus/Datadog
        # metrics.gauge("ml.gpu.utilization", stats["utilization"])
        # metrics.gauge("ml.queue.size", len(self.scheduler.pending_jobs))
        pass
    
    def submit_job(self, job: Job):
        """Public API to submit a job."""
        self.scheduler.submit_job(job)
        self.logger.info(f"Job {job.job_id} submitted")


# Usage example
async def main():
    # Configure scaling policy
    policy = ScalingPolicy(
        min_nodes=2,
        max_nodes=50,
        target_utilization=0.80,
        scale_up_threshold=0.90,
        scale_down_threshold=0.50,
        cooldown_minutes=5
    )
    
    # Create resource manager
    manager = MLResourceManager(policy)
    
    # Submit some jobs
    for i in range(10):
        job = Job(
            job_id=f"job-{i}",
            user_id="user1",
            team_id="ml-team",
            priority=JobPriority.MEDIUM,
            resources=ResourceRequest(
                cpus=8,
                memory_gb=61,
                gpus=1,
                gpu_type="A100",
                disk_gb=100
            ),
            estimated_duration_hours=2.0,
            deadline=None,
            submitted_at=datetime.now(),
            can_preempt=True
        )
        manager.submit_job(job)
    
    # Run manager
    await manager.run()


if __name__ == "__main__":
    asyncio.run(main())
```

## Monitoring & Metrics

### Key Metrics to Track

```python
class MetricsCollector:
    """Collect and export metrics for monitoring."""
    
    def collect_metrics(self, manager: MLResourceManager) -> Dict:
        """Collect current system metrics."""
        stats = manager.allocator.get_utilization_stats()
        
        return {
            # Resource utilization
            "gpu_utilization": stats["utilization"],
            "total_gpus": stats["total_gpus"],
            "used_gpus": stats["used_gpus"],
            "idle_gpus": stats["total_gpus"] - stats["used_gpus"],
            
            # Queue metrics
            "queue_size": len(manager.scheduler.pending_jobs),
            "running_jobs": len(manager.scheduler.running_jobs),
            
            # Performance metrics
            "avg_wait_time_minutes": self._calculate_avg_wait_time(manager),
            "p95_wait_time_minutes": self._calculate_p95_wait_time(manager),
            
            # Cost metrics
            "hourly_cost": self._calculate_hourly_cost(manager),
            "cost_per_job": self._calculate_cost_per_job(manager),
            
            # Fair share
            "teams_over_quota": self._count_teams_over_quota(manager),
        }
    
    def _calculate_avg_wait_time(self, manager: MLResourceManager) -> float:
        """Average wait time for jobs in queue."""
        if not manager.scheduler.pending_jobs:
            return 0.0
        
        total_wait = sum(
            job.wait_time.total_seconds() / 60
            for job in manager.scheduler.pending_jobs
        )
        return total_wait / len(manager.scheduler.pending_jobs)
    
    def _calculate_hourly_cost(self, manager: MLResourceManager) -> float:
        """Current hourly cost of running cluster."""
        return sum(
            node.cost_per_hour
            for node in manager.allocator.nodes.values()
        )
```

### Alerts to Configure

1. **High utilization:** GPU utilization > 95% for 10+ minutes
2. **Low utilization:** GPU utilization < 30% for 30+ minutes (wasting money)
3. **Long queue:** >50 jobs waiting for >30 minutes
4. **Failed jobs:** >10% job failure rate
5. **Cost spike:** Hourly cost increases >50% from baseline
6. **Quota exceeded:** Team uses >120% of quota
7. **Preemptions:** >10 preemptions per hour (indicates resource pressure)

## Failure Modes

### 1. Resource Fragmentation

**Problem:** Many small free slots but can't fit large jobs.

```python
def defragment_cluster(allocator: ResourceAllocator) -> int:
    """
    Defragment cluster by migrating jobs to consolidate resources.
    
    Greedy approach:
    1. Find fragmented nodes (partial utilization)
    2. Migrate small jobs to create large free nodes
    3. Prioritize migration of preemptible jobs
    """
    # Find nodes with <50% utilization
    fragmented = [
        (node_id, node)
        for node_id, node in allocator.nodes.items()
        if 0 < node.utilization < 0.5
    ]
    
    migrations = 0
    for node_id, node in fragmented:
        # Try to migrate jobs off this node
        # ... implementation ...
        migrations += 1
    
    return migrations
```

**Solution:**
- Periodic defragmentation
- Bin packing improvements (first-fit decreasing)
- Reserved nodes for large jobs

### 2. Priority Inversion

**Problem:** Low-priority job holds resources needed by high-priority job.

**Solution:**
- Preemption (implemented above)
- Priority aging (gradually increase priority of waiting jobs)
- Resource reservations for critical jobs

### 3. Spot Instance Interruptions

**Problem:** Spot instances terminated mid-job.

**Solution:**
- Checkpointing (every N minutes)
- Fallback to on-demand for critical jobs
- Distribute jobs across spot and on-demand mix

### 4. Quota Gaming

**Problem:** Teams submit fake jobs to consume quota before reset.

**Solution:**
- Rolling quotas (not daily reset)
- Job validation (reject suspiciously short jobs)
- Charge-back system with real money

### 5. Deadline Missed

**Problem:** Job with deadline doesn't get scheduled in time.

**Solution:**
- Deadline-aware scheduling (EDF - Earliest Deadline First)
- Reserve capacity for deadline-critical jobs
- Alert teams early if deadline at risk

## Real-World Case Study: Meta's Resource Allocation

### Meta's Approach

Meta runs one of the world's largest ML infrastructure:
- **100K+ GPUs** across multiple data centers
- **Millions of ML jobs** per month
- **Hundreds of teams** competing for resources

**Their solution:**

1. **Twine:** Resource allocation system
   - Priority-based scheduling with fair share
   - Dynamic bin packing across heterogeneous GPUs
   - Supports preemption for critical jobs

2. **Fair Share Model:**
   - Each team gets base quota (proportional to headcount)
   - Can burst above quota if resources available
   - Long-running over-quota usage results in throttling

3. **Cost Attribution:**
   - Every GPU-hour tracked and charged to team budget
   - Creates incentive to optimize job efficiency
   - Teams can trade quota allocations

4. **Auto-scaling:**
   - Scales down underutilized clusters during off-hours
   - Scales up aggressively during model release crunch times
   - Predictive scaling based on historical patterns

**Results:**
- **85%+ GPU utilization** (up from 60%)
- **40% cost reduction** through spot instances and optimization
- **<5 minute wait time** for p95 of jobs
- **$10M+ annual savings**

### Key Lessons

1. **Greedy works:** Simple greedy bin packing beats complex optimizations
2. **Fair share essential:** Prevents monopolization
3. **Cost visibility drives efficiency:** When teams see costs, they optimize
4. **Preemption is necessary:** For handling urgent production issues
5. **Heterogeneous resources are hard:** V100 vs A100 vs H100 requires smart matching

## Cost Analysis

### Cost Breakdown

For a typical mid-size ML team (1000 GPUs):

| Component | Monthly Cost | Optimization |
|-----------|--------------|--------------|
| Compute (on-demand A100) | $1.8M | Use spot (-70%) → $540K |
| Storage (model checkpoints) | $50K | Lifecycle policies → $30K |
| Network (multi-region) | $20K | Co-location → $15K |
| Orchestration overhead | $10K | - |
| **Total** | **$1.88M** | **$595K** |
| **Savings** | | **68% reduction** |

### Optimization Strategies

1. **Spot/Preemptible Instances:**
   - 70-90% cheaper than on-demand
   - Risk: can be interrupted (2-min warning)
   - Use for: training jobs with checkpointing

2. **Right-sizing:**
   - Match instance type to job requirements
   - Don't use 8-GPU instance for 1-GPU job
   - Savings: 30-40%

3. **Off-peak Training:**
   - Schedule large training jobs during off-hours
   - Take advantage of lower spot prices
   - Savings: 20-30%

4. **Model Optimization:**
   - Quantization, pruning, distillation reduce compute needs
   - Faster training → less GPU time
   - Savings: 50%+ for inference

5. **Batch Processing:**
   - Batch multiple inference requests
   - Increase GPU utilization from 30% → 85%
   - Savings: 60%+

### ROI Calculation

**Investment in resource allocation system:**
- Engineering: 3 engineers × 6 months = $300K
- Infrastructure: $50K/year

**Returns (1000 GPU cluster):**
- Before: 50% utilization, all on-demand = $1.8M/month
- After: 85% utilization, 70% spot = $595K/month
- **Savings: $1.2M/month = $14.4M/year**

**ROI:** 48x in first year!

## Key Takeaways

✅ **Resource allocation is a greedy optimization problem** - like Container With Most Water, allocate to bottlenecks first

✅ **Multi-dimensional bin packing** is the core algorithm for job placement

✅ **Priority queues with fair share** prevent starvation and monopolization

✅ **Auto-scaling based on utilization + queue length** maintains efficiency

✅ **Preemption is necessary** for handling critical production jobs

✅ **Spot instances + checkpointing** save 70%+ on costs

✅ **Monitoring and cost visibility** drive team optimization behaviors

✅ **Defragmentation prevents resource fragmentation** waste

✅ **Real-world systems use greedy algorithms** because they're fast and effective

✅ **Similar principles apply** to container optimization (DSA) and speech compute allocation

### Connection to Thematic Link: Greedy Optimization and Resource Management

All three topics share the same core insight:

**DSA (Container With Most Water):**
- Greedy choice: move pointer at bottleneck (shorter line)
- Maximize area under constraints

**ML System Design (Resource Allocation):**
- Greedy choice: allocate to highest-priority job that fits
- Maximize utilization under budget constraints

**Speech Tech (Compute Allocation):**
- Greedy choice: allocate compute to slowest pipeline stage
- Maximize throughput under latency constraints

The **bottleneck principle** is universal: optimize the limiting factor first.

---

**Originally published at:** [arunbaby.com/ml-system-design/0013-resource-allocation-for-ml](https://www.arunbaby.com/ml-system-design/0013-resource-allocation-for-ml/)

*If you found this helpful, consider sharing it with others who might benefit.*

