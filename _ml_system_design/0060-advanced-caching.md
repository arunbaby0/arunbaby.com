---
title: "Advanced Caching for ML Systems"
day: 60
related_dsa_day: 60
related_speech_day: 60
related_agents_day: 60
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
---


**"In the world of high-scale machine learning, the fastest inference is the one you never had to compute. Caching is not just about saving time; it's about making the impossible latency targets possible."**

## 1. Introduction: The Inference Bottleneck

As machine learning models grow from simple Decision Trees to 70B parameter Transformers, the "Cost per Inference" has become a defining business metric.
-   A single LLM generate call can cost cents.
-   A real-time ranking call for 10,000 items can take hundreds of milliseconds.
-   Storing pre-computed features for 1 billion users often requires Petabytes of storage.

If you are designing the Recommendation System for TikTok or the Search Ranking for Amazon, you cannot afford to call the heavy Re-ranking Model for every single user interaction. You must use **Advanced Caching Strategies**.

This post moves beyond the basics of "put it in Redis." We explore the **Multi-Tiered Cache Architecture**, **Semantic Caching for LLMs**, **Feature Store Hydration**, and how to solve the dreaded **Thundering Herd** problem in distributed systems.

---

## 2. Functional Requirements of an ML Cache

ML Caching is distinct from standard Web Caching (e.g., caching HTML pages).
1.  **High Throughput, Low Latency**: We need to fetch 1,000 features for a candidate batch in < 5ms.
2.  **Approximate Correctness**: Unlike a Bank Ledger, it's okay if a user's "Interest Vector" is 5 minutes stale.
3.  **High Dimensionality**: We aren't just caching strings; we are caching Vectors (Embeddings) and Tensors.
4.  **Semantic Retrieval**: Use Vector Similarity to find "near-matches" for user queries, allowing us to reuse LLM generations for similar prompts.

---

## 3. The Architecture: The Latency Hierarchy

Modern ML systems (like those at Netflix or Facebook) use a strictly tiered memory hierarchy.

### 3.1 Tier 1: The Local (In-Process) Cache
-   **Technology**: Python `dict`, C++ `std::unordered_map`.
-   **Latency**: Nanoseconds (Reference access) to Microseconds.
-   **Capacity**: Small (1GB - 10GB).
-   **Strategy**: **Hot-Keys Only**. We use algorithms like **TinyLFU** to greedily admit only the top 1% most popular items (e.g., The "Global Bias" features or "Justin Bieber" embedding).
-   **Gotcha**: Cache Coherency. If Node A updates the cache, Node B doesn't know. We accept this inconsistency.

### 3.2 Tier 2: The Distributed (Remote) Cache
-   **Technology**: Redis (Cluster Mode), Memcached, Amazon ElastiCache.
-   **Latency**: 1ms - 5ms (Network RTT).
-   **Capacity**: Medium to Large (Terabytes).
-   **Strategy**: **Sharded & Replica**. We shard data across 100+ nodes using **Consistent Hashing**.

### 3.3 Tier 3: The Persistent (SSD) Cache
-   **Technology**: Aerospike, DynamoDB (DAX), NVMe Pools.
-   **Latency**: 5ms - 20ms.
-   **Capacity**: Petabytes.
-   **Role**: **Feature Store History**. When you need to backfill data or fetch user history from 30 days ago.

### 3.4 Summary Table
| Tier | Technology | Latency | Capacity | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **L1** | Python dict, C++ map | < 0.1ms | 1-10GB | Hot keys, Global features |
| **L2** | Redis, Memcached | 1-5ms | 1-10TB | User features, Session data |
| **L3** | Aerospike, DAX | 5-20ms | Petabytes | Historical features, Cold users |

## 4. Deep Dive: Consistent Hashing for Distributed Caching

When you scale Redis to 100 nodes, efficient key distribution is critical.
Naive hashing `hash(key) % N` is broken. If one node dies, `N` changes to `N-1`, and **100% of keys are remapped**. This causes a "Cache Miss Storm" that can take down your database.

### 4.1 The Solution: Virtual Nodes on a Ring
We map both Nodes and Keys to a 32-bit integer ring.
-   A Key belongs to the first Node found moving clockwise on the ring.
-   **Virtual Nodes**: To balance load, one physical server is hashed to 1,000 points on the ring.
-   **Result**: If a node dies, only `1/N` of keys need to be moved to the neighbor. `(N-1)/N` keys stay put.

```python
import hashlib
import bisect

class ConsistentHash:
    def __init__(self, nodes=None, replicas=100):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        if nodes:
            for node in nodes:
                self.add_node(node)

    def add_node(self, node):
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            bisect.insort(self.sorted_keys, key)

    def remove_node(self, node):
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            del self.ring[key]
            self.sorted_keys.remove(key)

    def get_node(self, key):
        if not self.ring: return None
        h = self._hash(key)
        # Binary Search for the next node on the ring
        idx = bisect.bisect(self.sorted_keys, h)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
```

### 4.2 Replication for Read Scalability
Each shard can have read replicas.
-   **Primary**: Handles writes.
-   **Replicas (2-3)**: Handle reads.
-   **Trade-off**: Replication Lag. A write to Primary might not be visible on Replica for 10ms. For ML features, this is usually acceptable.

---

## 5. Embedding Caching: The ML-Specific Challenge

Standard caching assumes values are small (< 1KB). ML embeddings are large (512-dim float32 = 2KB) and there are billions of them.

### 5.1 Storage Optimization
-   **Quantization**: Store embeddings as `int8` instead of `float32`. Reduces storage by 4x with < 1% accuracy loss.
-   **Dimension Reduction**: Use PCA or learned projections to reduce 512-dim to 128-dim. 4x storage reduction.

### 5.2 Batch Fetching
When ranking 1000 items, we need 1000 embeddings.
-   **Naive**: 1000 sequential `GET` calls = 1000ms.
-   **Pipeline**: `MGET` (multi-get) = 5ms.
```python
# Bad: Sequential
embeddings = [redis.get(f"item:{id}") for id in item_ids]

# Good: Batch Pipeline
pipe = redis.pipeline()
for id in item_ids:
    pipe.get(f"item:{id}")
embeddings = pipe.execute()
```

### 5.3 Precomputation vs. On-Demand
-   **User Embeddings**: Change slowly (daily). Precompute and cache with 24h TTL.
-   **Item Embeddings**: Change on edit. Cache on first access, invalidate on update.
-   **Query Embeddings**: Ephemeral. Compute on-the-fly, cache for 5 minutes for repeat queries.

---

## 5. The "Thundering Herd" Problem & Solutions

Scenario: A celebrity tweets. Millions of users request their profile.
The cache key `user_profile:123` expires at 12:00:00.
At 12:00:01, 10,000 concurrent requests hit the cache -> Miss -> Hit the Database.
**The Database Dies.**

### 5.1 Solution 1: Request Coalescing (SingleFlight)
Only ONE request per key is allowed to compute. All other requests for that key **wait** (block) until the first one returns.
-   Implemented in Go (`singleflight`), Nginx (`proxy_cache_lock`).

### 5.2 Solution 2: Probabilistic Early Expiration (PER)
Instead of a hard TTL, we fetch based on a probability.
-   `gap = time.now() - lease_start`
-   `probability = gap / TTL`
-   If `random() < probability`, we recompute *before* expiry.
-   This spreads the refresh load over time.

---

## 6. Semantic Caching for LLMs (GPT-4 / Llama 3)

LLM queries are expensive and slow.
-   Q1: "Who is the President of France?"
-   Q2: "Who is the current leader of France?"
Traditional cache says **MISS** (Strings don't match).
**Semantic Cache** says **HIT**.

### 6.1 The Logic
1.  **Embed Query**: `v1 = BERT("Who is the President of France?")`
2.  **Vector Search**: Search Milvus/Faiss for nearest neighbor `v2`.
3.  **Distance Threshold**: If `Cosine_Similarity(v1, v2) > 0.95`, return the cached answer associated with `v2`.

### 6.2 Architecture
-   **Store**: Vectors in Faiss, Metadata in Redis.
-   **Tradeoff**: 5ms Latency (Embedding + Search) vs 2000ms Latency (LLM Generation).
-   **Cost Savings**: Massive. 30-40% of queries in production chat apps are semantically identical.

---

## 7. Feature Caching Strategies

In Recommender Systems, we need to fetch user features (`[age, gender, clicks_7d]`) and item features (`[category, price, ctr]`).

### 7.1 Row-Oriented vs Column-Oriented Caching
-   **Row Cache (Redis Strings)**: `key=user:123` -> `val=protobuf_blob`.
    -   Good for fetching single entities.
    -   **Serialization Formats**: Protobuf (fast, compact), JSON (readable, slower), MessagePack (good balance).
-   **Column Cache (Redis Lists/Bitmaps)**: `key=clicked_item:55` -> `val=[user1, user2, user3]`.
    -   Good for "Find all users who clicked X."
    -   **Use Case**: Co-occurrence matrices for collaborative filtering.

### 7.2 Feature Versioning
ML features evolve. You might change `ctr_7d` calculation from "clicks/views" to "clicks/(views + 100)".
-   **Problem**: Old cached values are now stale/incorrect.
-   **Solution**: Include version in key: `user:123:v2`.
-   **Deployment**: On deploy, warm cache with new version keys. Old keys expire naturally.

### 7.3 TTL Strategies by Feature Type
| Feature Type | TTL | Rationale |
| :--- | :--- | :--- |
| **Static** (age, gender) | 24h | Changes rarely |
| **Slowly Changing** (preferences) | 1h | Updates daily |
| **Real-Time** (clicks_5min) | 60s | Must be fresh |
| **Computed** (ML score) | 5min | Expensive to compute |

### 7.2 The Cache Pushing Pattern (Write-Around vs Write-Through)
-   **Write-Through**: The application writes to DB and Cache simultaneously. Safe but slow.
-   **Cache-Aside (Lazy Loading)**: Read DB on miss. Preferred for most features.
-   **Push-Based (Stream Hydration)**:
    -   User clicks item.
    -   Kafka Event triggers Flink Job.
    -   Flink computes `clicks_7d`.
    -   Flink **Pushes** update directly to Redis.
    -   The Inference service serves strictly from Redis (never computes).

---

## 8. Failure Modes

### 8.1 Hot Keys
One key (e.g., "Justin Bieber") gets more traffic than a single Redis shard can handle.
-   **Fix**: **Local Cache Replica**. Detect hot keys and replicate them to L1 (Application Memory).
-   **Fix**: **Key Splitting**. Store `bieber` as `bieber:1`, `bieber:2`. Randomly read one.

### 8.2 Network Bandwidth Saturation
Fetching 1000 items * 2KB embedding = 2MB payload per request.
At 1000 QPS, that is **2 GB/s**. This saturates a 10Gbps NIC.
-   **Fix**: Compression (Zstd, LZ4).
-   **Fix**: Quantization (Store float32 embeddings as int8).

---

## 9. Real-World Look: Facebook TAO

Facebook's "The Association Object" (TAO) cache handles billions of reads/sec.
-   **Graph-Aware**: It caches edges (`User -> Likes -> Page`) not just keys.
-   **Eventual Consistency**: It allows followers to be stale for seconds.
-   **Lease Mechanism**: Preventing thundering herds using "Leases" (tokens) given to clients.

---

## 10. Cache Invalidation: The Two Hardest Problems

Phil Karlton famously said: "There are only two hard things in Computer Science: cache invalidation and naming things."

### 10.1 Time-Based Invalidation (TTL)
The simplest strategy. Set `EXPIRE key 3600` (1 hour).
-   **Pros**: Simple. No coordination needed.
-   **Cons**: User might see stale data for up to 1 hour. Hot features like "Live Stock Price" cannot tolerate this.

### 10.2 Event-Based Invalidation (CDC)
Use Change Data Capture (Debezium, Maxwell) to listen to database changes.
-   User updates their profile in PostgreSQL.
-   Debezium captures the WAL event.
-   Kafka delivers `{user_id: 123, type: UPDATE}` to our Invalidation Consumer.
-   Consumer calls `DEL user:123` on Redis.
-   Next read fetches fresh data from DB and populates cache.

**Implementation Pattern**:
```python
from kafka import KafkaConsumer
import redis

consumer = KafkaConsumer('db.users', bootstrap_servers=['kafka:9092'])
r = redis.Redis()

for message in consumer:
    event = json.loads(message.value)
    user_id = event['after']['id']
    
    # Invalidate the cache for this user
    r.delete(f"user:{user_id}")
    r.delete(f"user_embedding:{user_id}")
    
    print(f"Invalidated cache for user {user_id}")
```

### 10.3 The Dual-Write Anti-Pattern
Never do this in production:
```python
# WRONG: Race Condition!
db.update(user)
cache.delete(user_key)
```
If `cache.delete` fails (network hiccup), your cache is forever stale.

**Safer Pattern**:
```python
# RIGHT: Transactional Outbox
db.update(user)
db.insert_into_outbox(event)

# Separate worker polls outbox
for event in db.poll_outbox():
    cache.delete(event.key)
    db.delete_from_outbox(event.id)
```

---

## 11. Monitoring and Observability

You cannot manage what you cannot measure. A production cache requires rigorous observability.

### 11.1 Key Metrics to Track
| Metric | Target | Alarm Threshold |
| :--- | :--- | :--- |
| **Cache Hit Ratio (CHR)** | > 90% | < 80% |
| **p99 Latency (Redis)** | < 5ms | > 15ms |
| **Memory Utilization** | < 80% | > 90% |
| **Eviction Rate** | Low | Sudden Spike |
| **Connection Pool Exhaustion** | 0 | > 0 |

### 11.2 Instrumenting Your Cache Client
Wrap your Redis calls with metrics.

```python
import time
from prometheus_client import Histogram, Counter

CACHE_LATENCY = Histogram('cache_latency_seconds', 'Cache operation latency', ['op', 'status'])
CACHE_HITS = Counter('cache_hits_total', 'Number of cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Number of cache misses')

class InstrumentedCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        
    def get(self, key):
        start = time.time()
        value = self.redis.get(key)
        duration = time.time() - start
        
        if value is None:
            CACHE_MISSES.inc()
            CACHE_LATENCY.labels(op='get', status='miss').observe(duration)
        else:
            CACHE_HITS.inc()
            CACHE_LATENCY.labels(op='get', status='hit').observe(duration)
            
        return value
```

### 11.3 Dashboards You Need
1.  **Hit Ratio over Time**: Detects model drift or changed usage patterns.
2.  **Latency Heatmap**: Identifies slow shards or network issues.
3.  **Memory Fragmentation**: Redis can use 2x memory due to fragmentation. Track `mem_fragmentation_ratio`.

---

## 12. KV Cache for Transformers (LLM Inference)

This is a specialized cache that lives *inside* the model inference, not in the infrastructure.

### 12.1 The Problem
In Transformers (GPT-4, Llama 3), generating 100 tokens requires computing Attention N times.
Each Attention computation re-uses the same Key and Value vectors for the *previous* tokens.
Without caching, you recompute these KV pairs at every step. This is O(N^2) for N tokens.

### 12.2 The KV Cache Solution
We store the Key and Value matrices in GPU VRAM.
-   Token 1: Compute K1, V1. Store.
-   Token 2: Load K1, V1. Compute K2, V2. Store K1, K2, V1, V2.
-   Token 50: Load K1..K49, V1..V49. Compute K50, V50. Store.

**VRAM Cost**: For Llama 70B, the KV Cache for 4096 tokens can consume **40GB** of VRAM. This is often the bottleneck, not the model weights.

### 12.3 Optimization: Paged Attention (vLLM)
vLLM (from UC Berkeley) treats KV Cache like OS Virtual Memory.
-   **Pages**: Divide the cache into fixed-size "Pages" (e.g., 256 tokens per page).
-   **On-Demand Allocation**: Only allocate pages when needed.
-   **Sharing**: If two prompts share a prefix, they can share the KV pages for that prefix.
-   **Result**: 24x higher throughput than naive HuggingFace `generate()`.

---

## 13. Production Case Study: Pinterest's Feature Store Caching

Pinterest serves 500M+ users. Their recommender system needs to fetch ~1000 features per request.

### 13.1 The Architecture
-   **L1 (Application)**: 5GB LRU cache per pod. Holds global features (e.g., "is_weekend").
-   **L2 (Redis Cluster)**: 500 nodes, 10TB capacity. Holds user features.
-   **L3 (Rockstore/SSTable)**: Hourly snapshots of the feature store for cold users.

### 13.2 Key Optimizations
1.  **Feature Packing**: Instead of 1000 `GET` calls, they pack all features for a user into a single Protobuf blob. 1 `GET` per user.
2.  **Speculative Warming**: When a user logs in, a background job pre-fetches their features into L2 before the first recommendation request.
3.  **Tiered Eviction**: Global features have TTL=infinity. User features have TTL=7 days. Ephemeral features (session data) have TTL=10 minutes.

### 13.3 Results
-   **p50 Latency**: 2ms (L1 hit).
-   **p99 Latency**: 8ms (L2 hit).
-   **Cache Hit Ratio**: 98.5%.
-   **Cost Savings**: 90% reduction in Feature Store SSD reads.

---

## 14. Advanced Topic: Negative Caching

What happens when you query for a user that doesn't exist?
-   Request: `GET user:99999999`
-   Redis: MISS
-   Database: `SELECT * FROM users WHERE id=99999999` -> Empty.
-   **Problem**: Next request repeats this expensive DB query.

### 14.1 The Solution: Cache the Null
Store a special sentinel value:
```python
value = db.get_user(user_id)
if value is None:
    cache.set(f"user:{user_id}", "NULL", ttl=300)  # Cache the non-existence
else:
    cache.set(f"user:{user_id}", serialize(value), ttl=3600)
```
On read:
```python
cached = cache.get(f"user:{user_id}")
if cached == "NULL":
    return None  # Don't hit DB
```

### 14.2 Danger: Botnet Attacks
If an attacker queries millions of random non-existent IDs, you fill your cache with "NULL" entries, evicting real data.
-   **Fix**: Use a **Bloom Filter** in front of the cache. Only check DB if the Bloom Filter says "Maybe Exists."

---

## 15. Cache Warming Strategies

A cold cache is a dangerous cache. The first 5 minutes after deployment can see massive latency spikes.

### 15.1 Pre-Warming on Deploy
Before routing traffic to a new pod:
1.  **Query Log Replay**: Replay the last hour of cache access logs against the new pod.
2.  **Feature Snapshot Load**: Load a snapshot of the top 10,000 most popular features directly into L1.
3.  **Health Check with Warm Threshold**: Don't mark the pod "Ready" until Cache Hit Ratio > 50%.

### 15.2 Shadow Traffic
Before cutting over to a new cache cluster:
-   **Fork Traffic**: Send a copy of all production requests to the new cluster (without returning results to users).
-   **Goal**: Pre-populate the cache with real access patterns.
-   **Duration**: Run for 10 minutes before cutting over.

### 15.3 Scheduled Warming for Time-Bound Features
Features like `day_of_week` or `is_holiday` are known in advance.
-   **CRON Job**: At midnight, compute and push all time-based features for the next 24 hours.
-   **Result**: Zero cold-start latency for the first recommendation of the day.

---

## 16. Production Best Practices

Based on experience operating ML caches at scale:

### 16.1 Separate Caches by SLA
Don't mix real-time serving cache with batch processing cache.
-   **Serving Cache**: Low TTL, High Memory Priority, Strict SLA.
-   **Training Cache**: High TTL, Lower Priority, Best Effort.
-   **Risk**: A batch job could evict serving data.

### 16.2 Use Client-Side Connection Pooling
Every Redis call opens a TCP connection. Opening connections is slow (3ms handshake).
-   Use a **Connection Pool** (e.g., `redis-py` with `ConnectionPool`).
-   Set `max_connections` based on your QPS and concurrency.

### 16.3 Monitor Memory Fragmentation
Redis uses jemalloc. Heavy SET/DEL cycles lead to fragmentation.
-   **Metric**: `mem_fragmentation_ratio` > 1.5 is a warning sign.
-   **Fix**: Schedule periodic `MEMORY DOCTOR` and consider `activedefrag yes`.

### 16.4 Implement Circuit Breakers
If Redis is down, don't hammer it with retries.
-   Use a **Circuit Breaker** (Hystrix, resilience4j, pybreaker).
-   When open, serve from L1 cache only or return a default/fallback prediction.

---

## 17. Key Takeaways

1.  **Caching is the First Line of Defense**: Protect your expensive GPU inference layers with a multi-tier hierarchy.
2.  **Consistency vs. Latency**: In ML, it is often better to serve a stale prediction (eventual consistency) than to wait.
3.  **Context Matters**: Choose your eviction policy based on the "intent" of the data. LFU for popular items, LRU for temporal access.
4.  **Semantic Search is the Future**: Move beyond exact-match keys to vector similarity for LLM caching.
5.  **Invalidation is Hard**: Use CDC/Event-based invalidation, not dual-writes.
6.  **Monitor Everything**: Cache Hit Ratio, Latency p99, Eviction Rate are your core metrics.

**Originally published at:** [arunbaby.com/ml-system-design/0060-advanced-caching](https://www.arunbaby.com/ml-system-design/0060-advanced-caching/)

*If you found this helpful, consider sharing it with others who might benefit.*
