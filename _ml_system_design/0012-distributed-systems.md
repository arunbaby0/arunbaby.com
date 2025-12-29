---
title: "Distributed ML Systems"
day: 12
related_dsa_day: 12
related_speech_day: 12
related_agents_day: 12
collection: ml_system_design
categories:
 - ml-system-design
tags:
 - distributed-systems
 - scalability
 - fault-tolerance
 - consistency
 - load-balancing
 - microservices
subdomain: Infrastructure
tech_stack: [Python, Kubernetes, Docker, gRPC, Redis, Kafka, etcd, Consul]
scale: "Multi-region, petabyte-scale, millions of requests/sec"
companies: [Google, Meta, Amazon, Netflix, Uber, Airbnb, Twitter]
---

**Design distributed ML systems that scale to billions of predictions: Master replication, sharding, consensus, and fault tolerance for production ML.**

## Problem Statement

Design a **distributed machine learning system** that can:
1. Handle **billions of predictions per day** across multiple regions
2. Train models on **terabytes of data** across multiple machines
3. Serve models with **low latency** (<100ms) and **high availability** (99.99%)
4. Handle **failures gracefully** without data loss or service disruption
5. Scale **horizontally** by adding more machines

### Why Distributed Systems?

**The fundamental constraint**: A single machine can't handle:

**Data:**
- Training data: 10TB+ (won't fit in RAM)
- Model size: 100GB+ (large language models, embeddings)
- Inference load: 100,000 requests/sec (CPU melts 🔥)

**Computation:**
- Training time: Days/weeks on single GPU
- Inference: Can't serve millions of users from one server

**Geography:**
- Users worldwide: Tokyo, London, New York, São Paulo
- Latency: Can't serve Tokyo users from Virginia (150ms+ RTT)

**Reliability:**
- Single machine fails → Entire service down ❌
- Need redundancy and fault tolerance

### Real-World Scale Examples

| Company | Scale | Challenge |
|---------|-------|-----------|
| **Google Search** | 8.5B searches/day | Distributed indexing + serving |
| **Netflix** | 200M users, 1B hours/day | Personalization at scale |
| **Uber** | 19M trips/day | Real-time matching + prediction |
| **Meta** | 3B users | Social graph + recommendation |

**Common pattern**: All use distributed ML systems!

---

## Understanding Distributed Systems Fundamentals

### What Makes Systems "Distributed"?

**Definition**: Multiple computers working together as one system.

**Simple analogy**: Restaurant kitchen
- **Single machine**: One chef makes everything (slow, bottleneck)
- **Distributed**: Multiple chefs, each specializing (fast, parallel)

But coordination is hard:
- How do chefs know what to cook?
- What if a chef is sick?
- How to avoid making duplicate orders?

These are **distributed systems problems**!

### The CAP Theorem

**CAP Theorem states**: You can only have 2 of 3:

1. **Consistency (C)**: All nodes see same data at same time
2. **Availability (A)**: System always responds (even if some nodes down)
3. **Partition Tolerance (P)**: System works despite network failures

**In practice**: Network partitions happen, so you must have P.
**Real choice**: Consistency (CP) vs Availability (AP)

**Example scenarios:**

``
Scenario: Network split between US and EU data centers

CP System (Choose Consistency):
- Reject writes until partition healed
- Data stays consistent
- But users in EU can't use system! ❌

AP System (Choose Availability):
- Accept writes in both regions
- Users happy! ✓
- But data may conflict later (eventual consistency)
``

**For ML systems:**
- **Training**: CP (want consistent data)
- **Serving**: AP (availability critical for user experience)

### Key Concepts for Junior Engineers

**1. Horizontal vs Vertical Scaling**

``
Vertical Scaling (Scale UP):
 1 machine → Bigger machine
 4 CPU → 64 CPU
 16GB RAM → 512GB RAM
 
 Pros: Simple, no code changes
 Cons: Expensive, limited (can't buy infinite RAM), single point of failure

Horizontal Scaling (Scale OUT):
 1 machine → 10 machines
 
 Pros: Cheaper, unlimited, fault-tolerant
 Cons: Complex (distributed systems problems!)
``

**ML systems need horizontal scaling** because:
- Data too big for one machine
- Training too slow on one machine
- Serving load too high for one machine

**2. Replication vs Sharding**

**Replication**: Same data on multiple machines
``
Machine 1: [A, B, C, D]
Machine 2: [A, B, C, D] ← Same data!
Machine 3: [A, B, C, D]

Use case: High availability, load distribution
Example: Model weights replicated to 100 servers
``

**Sharding**: Different data on each machine
``
Machine 1: [A, B]
Machine 2: [C, D] ← Different data!
Machine 3: [E, F]

Use case: Data too big for one machine
Example: Training data split across 10 machines
``

**3. Synchronous vs Asynchronous**

**Synchronous**: Wait for response before continuing
``python
result = call_other_service() # Block here
process(result) # Wait until call returns
``
- **Pros**: Simple, consistent
- **Cons**: Slow (latency adds up)

**Asynchronous**: Don't wait, continue immediately
``python
future = call_other_service_async() # Don't block
do_other_work() # Continue immediately
result = future.get() # Get result when needed
``
- **Pros**: Fast, better resource usage
- **Cons**: Complex, harder to debug

---

## Architecture Patterns

### Pattern 1: Master-Worker (for Training)

**Use case**: Distributed model training

``
┌─────────────────────────────────────────────┐
│ MASTER NODE │
│ • Coordinates workers │
│ • Aggregates gradients │
│ • Updates global model │
└────────┬──────────┬──────────┬──────────────┘
 │ │ │
 ┌────▼────┐ ┌──▼─────┐ ┌──▼─────┐
 │Worker 1 │ │Worker 2│ │Worker 3│
 │ GPU 1 │ │ GPU 2 │ │ GPU 3 │
 │Batch 1 │ │Batch 2 │ │Batch 3 │
 └─────────┘ └────────┘ └────────┘
``

**How it works:**

1. Master distributes data batches to workers
2. Each worker computes gradients on its batch
3. Workers send gradients back to master
4. Master averages gradients, updates model
5. Master broadcasts updated model to workers
6. Repeat

**Python implementation:**

``python
class MasterNode:
 """
 Master node for distributed training
 
 Coordinates multiple worker nodes
 """
 
 def __init__(self, model, workers):
 self.model = model
 self.workers = workers
 self.global_step = 0
 
 def train_step(self, data_batches):
 """
 One distributed training step
 
 1. Send model to workers
 2. Workers compute gradients
 3. Aggregate gradients
 4. Update model
 """
 # Distribute work to workers
 futures = []
 for worker, batch in zip(self.workers, data_batches):
 # Send model and data to worker
 future = worker.compute_gradients_async(
 self.model.state_dict(),
 batch
 )
 futures.append(future)
 
 # Wait for all workers (synchronous)
 gradients = [future.get() for future in futures]
 
 # Aggregate gradients (averaging)
 avg_gradients = self._average_gradients(gradients)
 
 # Update model
 self.model.update(avg_gradients)
 self.global_step += 1
 
 return self.model
 
 def _average_gradients(self, gradients_list):
 """Average gradients from all workers"""
 avg_grads = {}
 
 for param_name in gradients_list[0].keys():
 # Average this parameter's gradients
 param_grads = [g[param_name] for g in gradients_list]
 avg_grads[param_name] = sum(param_grads) / len(param_grads)
 
 return avg_grads

class WorkerNode:
 """
 Worker node that computes gradients
 """
 
 def __init__(self, worker_id, device='cuda'):
 self.worker_id = worker_id
 self.device = device
 
 def compute_gradients_async(self, model_state, batch):
 """
 Compute gradients on local batch
 
 Returns: Future that will contain gradients
 """
 import concurrent.futures
 
 executor = concurrent.futures.ThreadPoolExecutor()
 future = executor.submit(
 self._compute_gradients,
 model_state,
 batch
 )
 
 return future
 
 def _compute_gradients(self, model_state, batch):
 """Actually compute gradients"""
 import torch
 
 # Load model
 model = load_model()
 model.load_state_dict(model_state)
 model.to(self.device)
 
 # Forward + backward
 loss = model(batch)
 loss.backward()
 
 # Extract gradients
 gradients = {
 name: param.grad.cpu()
 for name, param in model.named_parameters()
 }
 
 return gradients
``

**Challenges:**

1. **Straggler problem**: Slowest worker delays everyone
 - **Solution**: Asynchronous updates, backup tasks
 
2. **Communication overhead**: Sending gradients is expensive
 - **Solution**: Gradient compression, local updates
 
3. **Fault tolerance**: What if worker crashes?
 - **Solution**: Checkpoint frequently, redistribute work

### Pattern 2: Load Balancer + Replicas (for Serving)

**Use case**: Serving ML predictions at scale

``
 ┌──────────────┐
 Requests ──→ │Load Balancer │
 │ (Round Robin)│
 └──────┬───────┘
 │
 ┌────────────────┼────────────────┐
 ▼ ▼ ▼
 ┌─────────┐ ┌─────────┐ ┌─────────┐
 │ Replica 1│ │Replica 2│ │Replica 3│
 │ Model │ │ Model │ │ Model │
 │+ Cache │ │+ Cache │ │+ Cache │
 └─────────┘ └─────────┘ └─────────┘
``

**Benefits:**

- **High availability**: If one replica dies, others handle load
- **Load distribution**: 10K req/sec across 10 replicas = 1K each
- **Zero-downtime deploys**: Update replicas one at a time

**Implementation:**

``python
class LoadBalancer:
 """
 Simple round-robin load balancer
 
 Distributes requests across healthy replicas
 """
 
 def __init__(self, replicas):
 self.replicas = replicas
 self.current_index = 0
 self.health_checker = HealthChecker(replicas)
 self.health_checker.start()
 
 def route_request(self, request):
 """
 Route request to healthy replica
 
 Uses round-robin for simplicity
 """
 # Get healthy replicas
 healthy = self.health_checker.get_healthy_replicas()
 
 if not healthy:
 raise Exception("No healthy replicas available!")
 
 # Round-robin selection
 replica = healthy[self.current_index % len(healthy)]
 self.current_index += 1
 
 # Forward request
 try:
 response = replica.predict(request)
 return response
 except Exception as e:
 # Retry with different replica
 return self._retry_request(request, exclude=[replica])
 
 def _retry_request(self, request, exclude=None):
 """Retry failed request on different replica"""
 exclude = exclude or []
 healthy = [
 r for r in self.health_checker.get_healthy_replicas()
 if r not in exclude
 ]
 
 if not healthy:
 raise Exception("All replicas failed")
 
 return healthy[0].predict(request)

class HealthChecker:
 """
 Continuously monitor replica health
 
 Marks unhealthy replicas so LB doesn't route to them
 """
 
 def __init__(self, replicas, check_interval=10):
 self.replicas = replicas
 self.check_interval = check_interval
 self.health_status = {r: True for r in replicas}
 self.running = False
 
 def start(self):
 """Start health checking in background"""
 import threading
 
 self.running = True
 self.thread = threading.Thread(
 target=self._health_check_loop,
 daemon=True
 )
 self.thread.start()
 
 def _health_check_loop(self):
 """Continuously check replica health"""
 import time
 
 while self.running:
 for replica in self.replicas:
 is_healthy = replica.health_check()
 self.health_status[replica] = is_healthy
 
 if not is_healthy:
 print(f"⚠️ Replica {replica.id} unhealthy!")
 
 time.sleep(self.check_interval)
 
 def get_healthy_replicas(self):
 """Get list of currently healthy replicas"""
 return [
 replica for replica in self.replicas
 if self.health_status[replica]
 ]
``

### Pattern 3: Pub-Sub for Async Communication

**Use case**: Model updates, feature updates, async tasks

``
 ┌───────────────┐
 │ Message Bus │
 │ (Kafka) │
 └───────┬───────┘
 │
 ┌───────────────┼───────────────┐
 ▼ ▼ ▼
 ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
 │ Subscriber 1 │ │ Subscriber 2 │ │ Subscriber 3 │
 │ Update model │ │ Update cache │ │ Log metrics │
 └──────────────┘ └──────────────┘ └──────────────┘
``

**When to use:**

- Model deployment: Notify all servers to reload model
- Feature updates: Broadcast new feature values
- Logging: Send metrics/logs asynchronously
- Training triggers: Data arrives → trigger training job

**Implementation:**

``python
class PubSubSystem:
 """
 Publish-Subscribe system for async communication
 
 Publishers send messages, subscribers receive them
 """
 
 def __init__(self):
 self.subscribers = {} # topic -> [subscribers]
 
 def subscribe(self, topic, callback):
 """
 Subscribe to a topic
 
 Args:
 topic: Topic name (e.g., 'model.updated')
 callback: Function to call when message received
 """
 if topic not in self.subscribers:
 self.subscribers[topic] = []
 
 self.subscribers[topic].append(callback)
 print(f"✓ Subscribed to {topic}")
 
 def publish(self, topic, message):
 """
 Publish message to topic
 
 All subscribers will receive it asynchronously
 """
 if topic not in self.subscribers:
 return
 
 for callback in self.subscribers[topic]:
 # Call asynchronously (non-blocking)
 import threading
 thread = threading.Thread(
 target=callback,
 args=(message,)
 )
 thread.start()
 
 print(f"📢 Published to {topic}: {message}")

# Example usage
pubsub = PubSubSystem()

# Subscriber 1: Model server that reloads on updates
def reload_model(message):
 print(f"🔄 Reloading model: {message['model_version']}")
 # Load new model...

pubsub.subscribe('model.updated', reload_model)

# Subscriber 2: Cache that invalidates on updates
def invalidate_cache(message):
 print(f"🗑️ Invalidating cache for: {message['model_version']}")
 # Clear cache...

pubsub.subscribe('model.updated', invalidate_cache)

# Publisher: Training job publishes when done
def training_complete(model_path, version):
 pubsub.publish('model.updated', {
 'model_path': model_path,
 'model_version': version,
 'timestamp': time.time()
 })

# Trigger
training_complete('s3://models/v123', 'v123')
# Both subscribers receive message asynchronously!
``

---

## Handling Failures

**Key principle**: In distributed systems, failures are **normal**, not exceptional!

### Types of Failures

1. **Machine failure**: Server crashes
2. **Network partition**: Network splits, can't communicate
3. **Slow nodes**: "Stragglers" delay entire system
4. **Corrupted data**: Silent data corruption
5. **Cascading failures**: One failure triggers others

### Fault Tolerance Strategies

**1. Replication (Multiple Copies)**

``python
class ReplicatedStorage:
 """
 Store data on multiple nodes
 
 If one fails, others have copy
 """
 
 def __init__(self, nodes, replication_factor=3):
 self.nodes = nodes
 self.replication_factor = replication_factor
 
 def write(self, key, value):
 """
 Write to multiple nodes
 
 Succeeds if majority succeed (quorum)
 """
 # Pick nodes to write to
 target_nodes = self._pick_nodes(key, self.replication_factor)
 
 # Write to all (parallel)
 import concurrent.futures
 with concurrent.futures.ThreadPoolExecutor() as executor:
 futures = [
 executor.submit(node.write, key, value)
 for node in target_nodes
 ]
 
 # Wait for majority
 successes = sum(1 for f in futures if f.result())
 
 # Require majority for success (quorum)
 quorum = (self.replication_factor // 2) + 1
 
 if successes >= quorum:
 return True
 else:
 raise Exception(f"Write failed: only {successes}/{self.replication_factor} succeeded")
 
 def read(self, key):
 """
 Read from multiple nodes, return most recent
 
 Handles node failures gracefully
 """
 target_nodes = self._pick_nodes(key, self.replication_factor)
 
 # Read from all
 values = []
 for node in target_nodes:
 try:
 value = node.read(key)
 values.append(value)
 except Exception:
 # Node failed, skip it
 continue
 
 if not values:
 raise Exception("All replicas failed!")
 
 # Return most recent (highest version)
 return max(values, key=lambda v: v['version'])
``

**2. Checkpointing (Save Progress)**

``python
class CheckpointedTraining:
 """
 Save training progress periodically
 
 If crash, resume from last checkpoint
 """
 
 def __init__(self, model, checkpoint_dir, checkpoint_every=1000):
 self.model = model
 self.checkpoint_dir = checkpoint_dir
 self.checkpoint_every = checkpoint_every
 self.global_step = 0
 
 def train(self, data_loader):
 """Train with checkpointing"""
 # Try to resume from checkpoint
 self.global_step = self._load_checkpoint()
 
 for batch in data_loader:
 # Skip batches we've already processed
 if self.global_step < batch.id:
 continue
 
 # Training step
 loss = self.model.train_step(batch)
 self.global_step += 1
 
 # Checkpoint periodically
 if self.global_step % self.checkpoint_every == 0:
 self._save_checkpoint()
 print(f"✓ Checkpoint saved at step {self.global_step}")
 
 def _save_checkpoint(self):
 """Save model + training state"""
 import torch
 
 checkpoint = {
 'model_state': self.model.state_dict(),
 'global_step': self.global_step,
 'timestamp': time.time()
 }
 
 path = f"{self.checkpoint_dir}/ckpt-{self.global_step}.pt"
 torch.save(checkpoint, path)
 
 def _load_checkpoint(self):
 """Load latest checkpoint if exists"""
 import glob
 import torch
 
 checkpoints = glob.glob(f"{self.checkpoint_dir}/ckpt-*.pt")
 
 if not checkpoints:
 return 0
 
 # Load latest
 latest = max(checkpoints, key=lambda p: int(p.split('-')[1].split('.')[0]))
 checkpoint = torch.load(latest)
 
 self.model.load_state_dict(checkpoint['model_state'])
 print(f"✓ Resumed from step {checkpoint['global_step']}")
 
 return checkpoint['global_step']
``

**3. Circuit Breaker (Prevent Cascading Failures)**

``python
class CircuitBreaker:
 """
 Prevent cascading failures
 
 If service keeps failing, stop calling it (open circuit)
 Give it time to recover, then try again
 """
 
 def __init__(self, failure_threshold=5, timeout=60):
 self.failure_threshold = failure_threshold
 self.timeout = timeout
 self.failures = 0
 self.state = 'closed' # closed, open, half_open
 self.last_failure_time = 0
 
 def call(self, func, *args, **kwargs):
 """
 Call function with circuit breaker protection
 """
 import time
 # Check if circuit is open
 if self.state == 'open':
 # Check if timeout passed
 if time.time() - self.last_failure_time > self.timeout:
 self.state = 'half_open'
 print("🔄 Circuit half-open, trying again...")
 else:
 raise Exception("Circuit breaker OPEN - service unavailable")
 
 # Try the call
 try:
 result = func(*args, **kwargs)
 
 # Success! Reset if we were half-open
 if self.state == 'half_open':
 self.state = 'closed'
 self.failures = 0
 print("✓ Circuit closed - service recovered")
 
 return result
 
 except Exception as e:
 # Failure
 self.failures += 1
 self.last_failure_time = time.time()
 
 # Open circuit if too many failures
 if self.failures >= self.failure_threshold:
 self.state = 'open'
 print(f"⚠️ Circuit breaker OPEN after {self.failures} failures")
 
 raise e

# Example usage
circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)

def call_unreliable_service(data):
 """This service sometimes fails"""
 import random
 if random.random() < 0.5:
 raise Exception("Service failed!")
 return "Success"

# Try calling with circuit breaker
for i in range(10):
 try:
 result = circuit_breaker.call(call_unreliable_service, "data")
 print(f"Request {i}: {result}")
 except Exception as e:
 print(f"Request {i}: {e}")
 
 time.sleep(1)
``

---

## Consistency Models

### Strong Consistency

**Guarantee**: All reads see the most recent write

``python
class StronglyConsistentStore:
 """
 Every read returns the latest write
 
 Achieved by: Single master, synchronous replication
 """
 
 def __init__(self):
 self.master = {} # Single source of truth
 self.replicas = [{}, {}] # Read replicas
 self.version = 0
 
 def write(self, key, value):
 """
 Write to master, then synchronously replicate
 
 Slow but consistent!
 """
 # Update version
 self.version += 1
 
 # Write to master
 self.master[key] = {'value': value, 'version': self.version}
 
 # Synchronously replicate to all replicas
 for replica in self.replicas:
 replica[key] = {'value': value, 'version': self.version}
 
 # Only return after all replicas updated
 print(f"✓ Write {key}={value} replicated to all")
 
 def read(self, key):
 """
 Read from master (always latest)
 """
 return self.master.get(key, {}).get('value')
``

**Pros**: Simple to reason about
**Cons**: Slow (sync replication), single point of failure

### Eventual Consistency

**Guarantee**: Reads **eventually** see the latest write (but not immediately)

``python
class EventuallyConsistentStore:
 """
 Reads may see stale data temporarily
 
 Achieved by: Asynchronous replication
 """
 
 def __init__(self):
 self.replicas = [{}, {}, {}]
 self.version = 0
 
 def write(self, key, value):
 """
 Write to one replica, asynchronously propagate
 
 Fast but eventually consistent
 """
 self.version += 1
 
 # Write to first replica immediately
 self.replicas[0][key] = {'value': value, 'version': self.version}
 
 # Asynchronously replicate to others
 import threading
 for replica in self.replicas[1:]:
 thread = threading.Thread(
 target=self._async_replicate,
 args=(replica, key, value, self.version)
 )
 thread.start()
 
 # Return immediately (don't wait for replication)
 return "OK"
 
 def _async_replicate(self, replica, key, value, version):
 """Replicate asynchronously"""
 import time
 time.sleep(0.1) # Simulate network delay
 replica[key] = {'value': value, 'version': version}
 
 def read(self, key):
 """
 Read from random replica
 
 May return stale data if replication not complete!
 """
 import random
 replica = random.choice(self.replicas)
 return replica.get(key, {}).get('value')
``

**Pros**: Fast, highly available
**Cons**: Can read stale data temporarily

**For ML systems:**
- **Model weights**: Eventual consistency OK (small staleness acceptable)
- **Feature store**: Strong consistency for critical features
- **Predictions**: No consistency needed (stateless)

---

## Consensus Algorithms

**Problem**: How do multiple nodes agree on a value when some might fail?

**Example**: Leader election - which node should be the master?

### Understanding the Challenge

``
Scenario: 3 nodes need to elect a leader

Node A thinks: "I should be leader!"
Node B thinks: "No, I should be leader!"
Node C crashes before voting

Challenge:
- Network delays mean messages arrive out of order
- Nodes might fail mid-process
- Must guarantee exactly ONE leader elected
``

**This is the consensus problem!**

### Raft Algorithm (Simplified)

**Raft** is easier to understand than Paxos, achieving the same goal.

**Key concepts:**

1. **States**: Each node is in one of three states:
 - **Follower**: Accepts commands from leader
 - **Candidate**: Trying to become leader
 - **Leader**: Sends commands to followers

2. **Terms**: Time divided into terms (like presidencies)
 - Each term has at most one leader
 - Term number increases after each election

3. **Election process:**

``python
class RaftNode:
 """
 Simplified Raft consensus node
 
 Real implementation is more complex!
 """
 
 def __init__(self, node_id, peers):
 self.node_id = node_id
 self.peers = peers
 self.state = 'follower'
 self.current_term = 0
 self.voted_for = None
 import random, time
 self.election_timeout = random.uniform(150, 300) # ms
 self.last_heartbeat = time.time()
 
 def start_election(self):
 """
 Become candidate and request votes
 
 Called when election timeout expires without hearing from leader
 """
 # Increment term
 self.current_term += 1
 self.state = 'candidate'
 self.voted_for = self.node_id # Vote for self
 
 print(f"Node {self.node_id}: Starting election for term {self.current_term}")
 
 # Request votes from all peers
 votes_received = 1 # Self vote
 
 for peer in self.peers:
 if self._request_vote(peer):
 votes_received += 1
 
 # Check if won election (majority)
 majority = (len(self.peers) + 1) // 2 + 1
 
 if votes_received >= majority:
 self._become_leader()
 else:
 # Lost election, revert to follower
 self.state = 'follower'
 
 def _request_vote(self, peer):
 """
 Request vote from peer
 
 Peer grants vote if:
 - Haven't voted in this term yet
 - Candidate's log is at least as up-to-date
 """
 request = {
 'term': self.current_term,
 'candidate_id': self.node_id
 }
 
 response = peer.handle_vote_request(request)
 
 return response.get('vote_granted', False)
 
 def _become_leader(self):
 """
 Become leader for this term
 
 Start sending heartbeats to maintain leadership
 """
 self.state = 'leader'
 print(f"Node {self.node_id}: Became leader for term {self.current_term}")
 
 # Send heartbeats to all followers
 self._send_heartbeats()
 
 def _send_heartbeats(self):
 """
 Send periodic heartbeats to prevent new elections
 
 Leader must send heartbeats < election_timeout
 """
 import time
 while self.state == 'leader':
 for peer in self.peers:
 peer.receive_heartbeat({
 'term': self.current_term,
 'leader_id': self.node_id
 })
 
 time.sleep(0.05) # 50ms heartbeat interval
 
 def receive_heartbeat(self, message):
 """
 Receive heartbeat from leader
 
 Reset election timeout
 """
 # Check term
 if message['term'] >= self.current_term:
 self.current_term = message['term']
 self.state = 'follower'
 self.last_heartbeat = time.time()
 # Reset election timeout
 
 return {'success': True}
 
 def handle_vote_request(self, request):
 """
 Handle vote request from candidate
 
 Grant vote if haven't voted in this term yet
 """
 # Check term
 if request['term'] < self.current_term:
 return {'vote_granted': False}
 
 # Check if already voted
 if self.voted_for is None or self.voted_for == request['candidate_id']:
 self.voted_for = request['candidate_id']
 self.current_term = request['term']
 return {'vote_granted': True}
 
 return {'vote_granted': False}
``

**Why this works:**

1. **Split votes**: If multiple candidates, may get no majority → retry
2. **Random timeouts**: Reduces likelihood of split votes
3. **Term numbers**: Ensures old messages ignored
4. **Majority requirement**: Ensures at most one leader per term

**Use in ML systems:**

- **Distributed training**: Elect master node
- **Model serving**: Elect coordinator for A/B test assignments
- **Feature store**: Elect primary for writes

---

## Data Partitioning Strategies

**Problem**: Training data is 10TB. Can't fit on one machine!

**Solution**: Partition (shard) across multiple machines.

### Strategy 1: Range Partitioning

**Idea**: Split data by key ranges

``
User IDs: 0 - 1,000,000

Partition 1: Users 0 - 250,000
Partition 2: Users 250,001 - 500,000
Partition 3: Users 500,001 - 750,000
Partition 4: Users 750,001 - 1,000,000
``

**Pros**: Simple, range queries efficient
**Cons**: Hotspots if data skewed

**Example:**

``python
class RangePartitioner:
 """
 Partition data by key ranges
 """
 
 def __init__(self, partitions):
 self.partitions = partitions # [(0, 250000, node1), (250001, 500000, node2), ...]
 
 def get_partition(self, key):
 """
 Find which partition handles this key
 """
 for start, end, node in self.partitions:
 if start <= key <= end:
 return node
 
 raise ValueError(f"Key {key} not in any partition")
 
 def write(self, key, value):
 """Write to appropriate partition"""
 node = self.get_partition(key)
 node.write(key, value)
 
 def read(self, key):
 """Read from appropriate partition"""
 node = self.get_partition(key)
 return node.read(key)

# Usage
partitioner = RangePartitioner([
 (0, 250000, node1),
 (250001, 500000, node2),
 (500001, 750000, node3),
 (750001, 1000000, node4)
])

# Write user data
partitioner.write(user_id=123456, value={'name': 'Alice', ...})

# Read user data
user_data = partitioner.read(user_id=123456)
``

**Hotspot problem:**

``
If most users have IDs 0-100,000:
 Partition 1: Overloaded! 📈
 Partition 2-4: Idle 💤
 
Unbalanced load!
``

### Strategy 2: Hash Partitioning

**Idea**: Hash key, use hash to determine partition

``
key → hash(key) → partition

Example:
user_id = 123456
hash(123456) = 42
partition = 42 % 4 = 2
→ Send to Partition 2
``

**Pros**: Even distribution (no hotspots)
**Cons**: Range queries impossible

``python
class HashPartitioner:
 """
 Partition data by hash of key
 """
 
 def __init__(self, nodes):
 self.nodes = nodes
 self.num_nodes = len(nodes)
 
 def get_partition(self, key):
 """
 Hash key to determine partition
 """
 # Hash key
 hash_value = hash(key)
 
 # Modulo to get partition index
 partition_idx = hash_value % self.num_nodes
 
 return self.nodes[partition_idx]
 
 def write(self, key, value):
 node = self.get_partition(key)
 node.write(key, value)
 
 def read(self, key):
 node = self.get_partition(key)
 return node.read(key)

# Usage
partitioner = HashPartitioner([node1, node2, node3, node4])

# Even distribution!
partitioner.write(1, 'data1') # node2
partitioner.write(2, 'data2') # node4
partitioner.write(3, 'data3') # node1
partitioner.write(123456, 'data') # node2
``

**Problem with adding/removing nodes:**

``
With 4 nodes: hash(key) % 4 = 2 → node2
Add node5 (now 5 nodes): hash(key) % 5 = 4 → node5

All keys need remapping! 😱
Expensive!
``

### Strategy 3: Consistent Hashing

**Idea**: Minimize remapping when adding/removing nodes

**How it works:**

1. Hash both keys and nodes to same space (e.g., 0-360°)
2. Place nodes on circle
3. Key goes to next node clockwise

``
Circle (0-360°):
 0°
 |
 Node B (45°)
 |
 Node C (120°)
 |
 Node D (200°)
 |
 Node A (290°)
 |
 360° (= 0°)

Key x hashes to 100° → Goes to Node C (next clockwise at 120°)
Key y hashes to 250° → Goes to Node A (next clockwise at 290°)

Add Node E at 160°:
- Only keys between 120° and 160° move from C to E
- All other keys unchanged!
``

``python
import bisect

class ConsistentHashRing:
 """
 Consistent hashing for minimal remapping
 """
 
 def __init__(self, nodes, virtual_nodes=150):
 self.virtual_nodes = virtual_nodes
 self.ring = []
 self.node_map = {}
 
 for node in nodes:
 self._add_node(node)
 
 def _add_node(self, node):
 """
 Add node to ring with multiple virtual nodes
 
 Virtual nodes for better distribution
 """
 for i in range(self.virtual_nodes):
 # Hash node + replica number
 virtual_key = f"{node.id}-{i}"
 hash_value = hash(virtual_key) % (2**32)
 
 # Insert into sorted ring
 bisect.insort(self.ring, hash_value)
 self.node_map[hash_value] = node
 
 def get_node(self, key):
 """
 Find node for key
 
 O(log N) lookup using binary search
 """
 # Hash key
 hash_value = hash(key) % (2**32)
 
 # Find next node clockwise
 idx = bisect.bisect_right(self.ring, hash_value)
 
 if idx == len(self.ring):
 idx = 0 # Wrap around
 
 ring_position = self.ring[idx]
 return self.node_map[ring_position]
 
 def add_node(self, node):
 """
 Add new node
 
 Only ~1/N keys need remapping!
 """
 self._add_node(node)
 print(f"Added {node.id}, only ~{100/len(self.ring)*self.virtual_nodes:.1f}% keys remapped")
 
 def remove_node(self, node):
 """Remove node from ring"""
 for i in range(self.virtual_nodes):
 virtual_key = f"{node.id}-{i}"
 hash_value = hash(virtual_key) % (2**32)
 
 idx = self.ring.index(hash_value)
 del self.ring[idx]
 del self.node_map[hash_value]

# Usage
ring = ConsistentHashRing([node1, node2, node3, node4])

# Keys distributed evenly
key1_node = ring.get_node('user_123')
key2_node = ring.get_node('user_456')

# Add node - minimal disruption!
ring.add_node(node5)
``

**Use in ML:**

- **Feature store**: Partition features by entity ID
- **Training data**: Distribute examples across workers
- **Model serving**: Distribute prediction requests

---

## Real-World Case Study: Netflix Recommendation System

### Architecture

``
┌───────────────────────────────────────────────────────────┐
│ Global Load Balancer │
│ (GeoDNS) │
└───────────────┬───────────────────────────────────────────┘
 │
 ┌───────────┼───────────┐
 ▼ ▼ ▼
┌────────┐ ┌────────┐ ┌────────┐
│ US │ │ EU │ │ APAC │ ← Regional clusters
│ Region │ │ Region │ │ Region │
└────┬───┘ └────┬───┘ └────┬───┘
 │ │ │
 ▼ ▼ ▼
┌─────────────────────────────┐
│ Cassandra (User Profiles) │ ← Distributed database
│ Replicated across regions │
└─────────────────────────────┘
 │
 ▼
┌─────────────────────────────┐
│ Recommendation Service │ ← 1000s of instances
│ (Load balanced) │
└──────┬──────────────────────┘
 │
 ┌───┴────┐
 ▼ ▼
┌─────┐ ┌─────┐
│Cache│ │Model│ ← Redis cache + Model replicas
│Redis│ │Serve│
└─────┘ └─────┘
``

### Key Distributed Systems Principles Used

1. **Geographic distribution**: Users routed to nearest region (low latency)
2. **Replication**: User data replicated across 3 regions (high availability)
3. **Caching**: Hot recommendations cached (reduce compute)
4. **Load balancing**: Requests distributed across 1000s of servers
5. **Eventual consistency**: Viewing history can be slightly stale
6. **Partitioning**: Users partitioned by user_id (horizontal scaling)

### Numbers

- **200M+ users**
- **1B+ recommendation requests/day**
- **3 regions** (US, EU, APAC)
- **1000s of servers** per region
- **< 100ms** p99 latency for recommendations

**How they handle failure:**

- **Region failure**: Route traffic to other regions
- **Server failure**: Load balancer removes from pool
- **Cache miss**: Fall back to model inference
- **Database failure**: Serve stale data from replica

---

## Key Takeaways

✅ **Horizontal scaling** - Add machines, not bigger machines 
✅ **Replication** - Multiple copies for availability 
✅ **Sharding** - Split data for scalability 
✅ **Load balancing** - Distribute requests evenly 
✅ **Fault tolerance** - Design for failure, not perfection 
✅ **Async communication** - Pub-sub for decoupling 
✅ **Consistency trade-offs** - CP vs AP based on use case 

**Core principles:**
1. Failures are normal - design for them
2. Network is unreliable - use retries, timeouts
3. Consistency costs performance - choose wisely
4. Monitoring is essential - you can't fix what you can't see

---

**Originally published at:** [arunbaby.com/ml-system-design/0012-distributed-systems](https://www.arunbaby.com/ml-system-design/0012-distributed-systems/)

*If you found this helpful, consider sharing it with others who might benefit.*

