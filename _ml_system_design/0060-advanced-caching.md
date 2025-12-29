---
title: "Advanced Caching for ML Systems"
day: 60
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - caching
  - infrastructure
  - redis
  - distributed-systems
  - latent-variables
  - consistency
  - feature-store
subdomain: "Infrastructure"
tech_stack: [Redis, Memcached, Python, AWS ElastiCache, Aerospike]
scale: "Handling 10M+ cache lookups per second with sub-1ms p99 latency"
companies: [Facebook, Netflix, Google, Pinterest, Spotify]
difficulty: Hard
related_dsa_day: 60
related_speech_day: 60
related_agents_day: 60
---

**"In the world of high-scale machine learning, the fastest inference is the one you never had to compute. Caching is not just about saving time; it's about making the impossible latency targets possible."**

## 1. Introduction: The Inference Bottleneck

As machine learning models grow from simple Decision Trees to 70B parameter Transformers, the "Cost per Inference" has become a defining business metric. 
- A single LLM generate call can cost cents.
- A real-time ranking call for 10,000 items can take hundreds of milliseconds.
- Storing pre-computed features for 1 billion users requires Petabytes of storage.

**Advanced Caching** is the strategy of using a tiered memory hierarchy to "cheat" the latency-throughput-cost trade-off. It is the art of memorizing the results of expensive computations. Today, on Day 60, we look at the architecture of the world's most sophisticated ML caching systems, connecting it to the theme of **Complex Persistent State and Efficient Retrieval**.

---

## 2. The Functional Requirements of an ML Cache

A production-grade ML cache must handle more than just simple string lookups:
1.  **Embedding Storage**: Storing high-dimensional vectors (e.g., 512-dim floats) representing users or items.
2.  **Partial Computation Caching**: In a Transformer, the "K-V Cache" stores internal model states to speed up token generation.
3.  **Semantic Caching**: Using vector similarity to find "near-matches" for user queries (e.g., "What's the weather?" matching "Show me the weather").
4.  **Feature Hydration**: Taking a list of 1,000 `item_ids` and "hydrating" them with their current price, stock, and popularity in < 10ms.

---

## 3. High-Level Architecture: The Multi-Tier Hierarchy

Modern ML systems (like those at Netflix or Facebook) use a three-layer caching strategy:

### Layer 1: In-Process Cache (L1)
- **Tech**: Local RAM, LRU/LFU in Python/C++ memory.
- **Latency**: 0.01ms - 0.1ms.
- **Role**: Store the "Hot 10,000" features. If a user is currently browsing, their profile stays in L1.

### Layer 2: Distributed Memory Cache (L2)
- **Tech**: Redis, Memcached, Amazon ElastiCache.
- **Latency**: 1ms - 5ms.
- **Role**: The primary shared store. If Instance A computes a feature, Instance B can access it immediately.

### Layer 3: Solid-State Cache / Near-line (L3)
- **Tech**: Aerospike, DynamoDB (DAX), ScyllaDB.
- **Latency**: 10ms - 50ms.
- **Role**: Support for "Warm" data. This stores millions of user embeddings that are accessed frequently but don't fit in RAM.

---

## 4. Implementation: The Consistent Hashing Problem

When you scale the cache to hundreds of Redis nodes, you cannot simply use `hash(key) % num_nodes`. Why? Because if one node fails, the denominator changes, and **every single key in your cache moves**, leading to a "Cache Stampede" that crashes your database.

### The Solution: Consistent Hashing
We place nodes and keys on a logical "Ring" (the unit circle).
1.  Map nodes to positions on the ring using their IP.
2.  Map keys to positions using their hash.
3.  A key belongs to the **first node found clockwise** from its position.
4.  If a node fails, only the keys belonging to that specific node move to the next one. The rest of the cache stays stable.

```python
import hashlib

class ConsistentHash:
    def __init__(self, nodes, replicas=3):
        self.ring = {}
        self.sorted_keys = []
        for node in nodes:
            for i in range(replicas):
                # Create 'virtual nodes' to balance the load
                h = int(hashlib.md5(f"{node}:{i}".encode()).hexdigest(), 16)
                self.ring[h] = node
                self.sorted_keys.append(h)
        self.sorted_keys.sort()

    def get_node(self, key):
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        # Find the first node clockwise
        import bisect
        idx = bisect.bisect(self.sorted_keys, h)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]
```

---

## 5. Cache Strategy: LFU vs. LRU in ML

In the **DSA post** today, we explored the **LFU (Least Frequently Used)** algorithm. Why is it important for ML?
- **The "Heavy Hitter" Problem**: In a social network, 0.1% of users generate 90% of the traffic. 
- **LRU Flaw**: If a burst of random bot traffic arrives, a standard LRU cache will "evict" the embeddings of your most valuable, frequent users.
- **LFU Strength**: LFU recognizes frequency over time. It keeps these high-value "Heavy Hitters" in the cache regardless of recent noise.

**Senior Design Tip**: Use LFU for **Static Feature Metadata** (categories, item attributes) and LRU for **User Session State** (temporary intent).

---

## 6. Advanced Concept: Semantic Caching for Generative AI

Traditional caches require an EXACT string match. Generative AI requires **Semantic Match**.
1.  **Ingress**: User asks "How do I reset my password?".
2.  **Embedding**: Convert the string to a vector using a small model (e.g., `all-MiniLM-L6-v2`).
3.  **Vector Search**: Query a cache of previous questions (e.g., in Faiss/RedisVL).
4.  **Threshold**: If a previous question exists with > 0.98 cosine similarity, return its cached answer.
5.  **Impact**: This can reduce LLM API costs by 30-50% for customer support bots.

---

## 7. Failure Modes: The "Thundering Herd"

When a popular cache key (e.g., "Trending_Items") expires, 1,000 concurrent requests find a "Cache Miss" and all hit the database simultaneously to recompute it.
- **Solution A: Mutex Locking (Single-Flight)**: Only a single request is allowed to recompute the key; others wait.
- **Solution B: Probabilistic Expiry**: Instead of a hard TTL of 3600s, use a random value between 3500 and 3700. This staggers the recomputations.
- **Solution C: Soft TTL**: Return the "Stale" value while a background thread updates the cache.

---

## 8. Real-world Case Study: Netflix "EVCache"

Netflix developed EVCache (based on Memcached) to handle the extreme scale of global streaming.
- **Scale**: Trillions of requests per day.
- **Strategy**: **Cross-Region Replication**. If a cache is updated in the "US-East" data center, it is asynchronously replicated to "EU-West" and "Asia-Pacific". 
- **The Benefit**: When a user travels from NY to London, their profile (L3) and recent history (L2) are already "warm" in the local European cache.

---

## 9. Monitoring and Cache Observability

How do you know your cache is healthy?
- **Cache Hit Ratio (CHR)**: Items Found / Total Lookups. Ideally > 80%.
- **Eviction Rate**: If this is high, your cache is too small or your TTL is too low.
- **Request Latency (p99)**: The network "tail" of distributed caching.

---

## 10. Key Takeaways

1.  **Caching is the First Line of Defense**: Protect your expensive GPU inference and Database layers with multiple tiers of memory.
2.  **Consistency vs. Latency**: In ML, it is often better to serve a stale prediction (eventual consistency) than to wait 200ms for a fresh one.
3.  **Context Matters**: (The DSA Link) Choose your eviction policy (LRU vs. LFU) based on the "intent" of the data. 
4.  **Semantic Search is the Future**: Move beyond keys to vector similarity for Generative AI caching.

---

**Originally published at:** [arunbaby.com/ml-system-design/0060-advanced-caching](https://www.arunbaby.com/ml-system-design/0060-advanced-caching/)

*If you found this helpful, consider sharing it with others who might benefit.*
