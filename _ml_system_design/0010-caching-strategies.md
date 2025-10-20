---
title: "Caching Strategies for ML Systems"
day: 10
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - caching
  - performance
  - distributed-systems
  - redis
  - memcached
subdomain: Infrastructure
tech_stack: [Python, Redis, Memcached, DynamoDB, CDN, Nginx]
scale: "Low latency, high throughput"
companies: [Google, Meta, Netflix, Uber, Amazon, Twitter]
related_dsa_day: 10
related_speech_day: 10
---

**Design efficient caching layers for ML systems to reduce latency, save compute costs, and improve user experience at scale.**

## Introduction

**Caching** temporarily stores computed results to serve future requests faster. In ML systems, caching is critical for:

**Why caching matters:**
- **Latency reduction:** ms instead of seconds for predictions
- **Cost savings:** Avoid expensive model inference
- **Scalability:** Handle more requests with same resources
- **Availability:** Serve cached results if model service is down

**Common caching scenarios in ML:**
- Model predictions (feature → prediction)
- Feature computations (raw data → engineered features)
- Embeddings (entity → vector representation)
- Model artifacts (model weights, config)
- Training data (preprocessed datasets)

---

## Cache Hierarchy

```
┌────────────────────────────────────────────────────┐
│                   Client/Browser                    │
│              (Local Storage, Cookies)               │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│                   CDN Cache                         │
│            (CloudFlare, Akamai, CloudFront)        │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│                 Application Cache                   │
│              (Redis, Memcached, Local)             │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│                   ML Model Service                  │
│               (TensorFlow Serving, etc.)           │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│                   Database                          │
│            (PostgreSQL, MongoDB, etc.)             │
└────────────────────────────────────────────────────┘
```

---

## Cache Eviction Policies

### LRU (Least Recently Used)

**Most common for ML systems**

```python
from collections import OrderedDict

class LRUCache:
    """
    LRU Cache implementation
    
    Evicts least recently used items when capacity is reached
    """
    
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        """
        Get value and mark as recently used
        
        Time: O(1)
        """
        if key not in self.cache:
            return None
        
        # Move to end (most recent)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        """
        Put key-value pair
        
        Time: O(1)
        """
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        
        # Evict if over capacity
        if len(self.cache) > self.capacity:
            # Remove first item (least recently used)
            self.cache.popitem(last=False)
    
    def stats(self):
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'capacity': self.capacity,
            'utilization': len(self.cache) / self.capacity
        }

# Usage
cache = LRUCache(capacity=1000)

# Cache predictions
def get_prediction_cached(features, model):
    cache_key = hash(tuple(features))
    
    # Check cache
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Compute prediction
    prediction = model.predict([features])[0]
    
    # Cache result
    cache.put(cache_key, prediction)
    
    return prediction
```

### LFU (Least Frequently Used)

**Good for skewed access patterns**

```python
from collections import defaultdict
import heapq

class LFUCache:
    """
    LFU Cache - evicts least frequently used items
    
    Better for "hot" items that are accessed repeatedly
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> (value, frequency)
        self.freq_map = defaultdict(set)  # frequency -> set of keys
        self.min_freq = 0
        self.access_count = 0
    
    def get(self, key):
        """Get value and increment frequency"""
        if key not in self.cache:
            return None
        
        value, freq = self.cache[key]
        
        # Update frequency
        self.freq_map[freq].remove(key)
        if not self.freq_map[freq] and freq == self.min_freq:
            self.min_freq += 1
        
        new_freq = freq + 1
        self.freq_map[new_freq].add(key)
        self.cache[key] = (value, new_freq)
        
        return value
    
    def put(self, key, value):
        """Put key-value pair"""
        if self.capacity == 0:
            return
        
        if key in self.cache:
            # Update existing key
            _, freq = self.cache[key]
            self.cache[key] = (value, freq)
            self.get(key)  # Update frequency
            return
        
        # Evict if at capacity
        if len(self.cache) >= self.capacity:
            # Remove item with minimum frequency
            evict_key = next(iter(self.freq_map[self.min_freq]))
            self.freq_map[self.min_freq].remove(evict_key)
            del self.cache[evict_key]
        
        # Add new key
        self.cache[key] = (value, 1)
        self.freq_map[1].add(key)
        self.min_freq = 1
    
    def get_top_k(self, k: int):
        """Get top k most frequently accessed items"""
        items = [(freq, key) for key, (val, freq) in self.cache.items()]
        return heapq.nlargest(k, items)

# Usage for embeddings (frequently accessed)
embedding_cache = LFUCache(capacity=10000)

def get_embedding_cached(entity_id, embedding_model):
    cached_emb = embedding_cache.get(entity_id)
    if cached_emb is not None:
        return cached_emb
    
    embedding = embedding_model.encode(entity_id)
    embedding_cache.put(entity_id, embedding)
    
    return embedding
```

### TTL (Time-To-Live) Cache

**Good for time-sensitive data**

```python
import time

class TTLCache:
    """
    TTL Cache - items expire after specified time
    
    Perfect for:
    - User sessions
    - Real-time features (stock prices, weather)
    - Model predictions that become stale
    """
    
    def __init__(self, default_ttl_seconds=3600):
        self.cache = {}  # key -> (value, expiration_time)
        self.default_ttl = default_ttl_seconds
    
    def get(self, key):
        """Get value if not expired"""
        if key not in self.cache:
            return None
        
        value, expiration = self.cache[key]
        
        # Check expiration
        if time.time() > expiration:
            del self.cache[key]
            return None
        
        return value
    
    def put(self, key, value, ttl=None):
        """Put key-value pair with TTL"""
        if ttl is None:
            ttl = self.default_ttl
        
        expiration = time.time() + ttl
        self.cache[key] = (value, expiration)
    
    def cleanup(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            k for k, (v, exp) in self.cache.items()
            if current_time > exp
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)

# Usage for time-sensitive predictions
prediction_cache = TTLCache(default_ttl_seconds=300)  # 5 minutes

def predict_stock_price(symbol, model):
    """Predictions expire quickly for real-time data"""
    cached = prediction_cache.get(symbol)
    if cached is not None:
        return cached
    
    prediction = model.predict(symbol)
    prediction_cache.put(symbol, prediction, ttl=60)  # 1 minute TTL
    
    return prediction
```

---

## Distributed Caching

### Redis-Based Cache

```python
import redis
import json
import pickle
import hashlib

class RedisMLCache:
    """
    Redis-based cache for ML predictions
    
    Features:
    - Distributed across multiple servers
    - Persistence
    - TTL support
    - Pub/sub for cache invalidation
    """
    
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False
        )
        
        self.hits = 0
        self.misses = 0
    
    def _serialize(self, obj):
        """Serialize Python object"""
        return pickle.dumps(obj)
    
    def _deserialize(self, data):
        """Deserialize to Python object"""
        if data is None:
            return None
        return pickle.loads(data)
    
    def _make_key(self, prefix, *args):
        """Generate cache key"""
        # Hash arguments for consistent key
        key_str = f"{prefix}:{':'.join(map(str, args))}"
        return key_str
    
    def get_prediction(self, model_id, features):
        """
        Get cached prediction
        
        Args:
            model_id: Model identifier
            features: Feature vector (hashable)
        
        Returns:
            Cached prediction or None
        """
        # Create cache key
        feature_hash = hashlib.md5(
            str(features).encode()
        ).hexdigest()
        key = self._make_key('prediction', model_id, feature_hash)
        
        # Get from Redis
        cached = self.redis_client.get(key)
        
        if cached is not None:
            self.hits += 1
            return self._deserialize(cached)
        
        self.misses += 1
        return None
    
    def set_prediction(self, model_id, features, prediction, ttl=3600):
        """Cache prediction with TTL"""
        feature_hash = hashlib.md5(
            str(features).encode()
        ).hexdigest()
        key = self._make_key('prediction', model_id, feature_hash)
        
        # Serialize and store
        value = self._serialize(prediction)
        self.redis_client.setex(key, ttl, value)
    
    def get_embedding(self, entity_id):
        """Get cached embedding"""
        key = self._make_key('embedding', entity_id)
        cached = self.redis_client.get(key)
        
        if cached:
            self.hits += 1
            # Embeddings stored as JSON arrays
            return json.loads(cached)
        
        self.misses += 1
        return None
    
    def set_embedding(self, entity_id, embedding, ttl=None):
        """Cache embedding"""
        key = self._make_key('embedding', entity_id)
        value = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)
        
        if ttl:
            self.redis_client.setex(key, ttl, value)
        else:
            self.redis_client.set(key, value)
    
    def invalidate_model(self, model_id):
        """Invalidate all predictions for a model (SCAN + DEL)"""
        pattern = self._make_key('prediction', model_id, '*')
        cursor = 0
        total_deleted = 0
        
        while True:
            cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=1000)
            if keys:
                total_deleted += self.redis_client.delete(*keys)
            if cursor == 0:
                break
        
        return total_deleted
    
    def get_stats(self):
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_keys': self.redis_client.dbsize()
        }

# Usage
cache = RedisMLCache(host='localhost', port=6379)

def predict_with_cache(features, model, model_id):
    """Predict with Redis caching"""
    # Check cache
    cached = cache.get_prediction(model_id, features)
    if cached is not None:
        return cached
    
    # Compute prediction
    prediction = model.predict([features])[0]
    
    # Cache result
    cache.set_prediction(model_id, features, prediction, ttl=3600)
    
    return prediction

# Check cache performance
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

### Multi-Level Cache

```python
class MultiLevelCache:
    """
    Multi-level caching with L1 (local) and L2 (Redis)
    
    Pattern:
    1. Check L1 (in-memory, fastest)
    2. If miss, check L2 (Redis, shared)
    3. If miss, compute and populate both levels
    """
    
    def __init__(self, l1_capacity=1000, redis_host='localhost'):
        # L1: Local LRU cache
        self.l1 = LRUCache(capacity=l1_capacity)
        
        # L2: Redis cache
        self.l2 = RedisMLCache(host=redis_host)
        
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
    
    def get(self, key):
        """Get value from multi-level cache"""
        # Try L1
        value = self.l1.get(key)
        if value is not None:
            self.l1_hits += 1
            return value
        
        # Try L2
        value = self.l2.redis_client.get(key)
        if value is not None:
            self.l2_hits += 1
            
            # Populate L1
            value = self.l2._deserialize(value)
            self.l1.put(key, value)
            
            return value
        
        # Miss
        self.misses += 1
        return None
    
    def put(self, key, value, ttl=3600):
        """Put value in both cache levels"""
        # Store in L1
        self.l1.put(key, value)
        
        # Store in L2
        self.l2.redis_client.setex(
            key,
            ttl,
            self.l2._serialize(value)
        )
    
    def get_stats(self):
        """Get multi-level cache statistics"""
        total = self.l1_hits + self.l2_hits + self.misses
        
        return {
            'l1_hits': self.l1_hits,
            'l2_hits': self.l2_hits,
            'misses': self.misses,
            'total_requests': total,
            'l1_hit_rate': self.l1_hits / total if total > 0 else 0,
            'l2_hit_rate': self.l2_hits / total if total > 0 else 0,
            'overall_hit_rate': (self.l1_hits + self.l2_hits) / total if total > 0 else 0
        }

# Usage
ml_cache = MultiLevelCache(l1_capacity=1000, redis_host='localhost')

def get_user_embedding(user_id, embedding_model):
    """Get user embedding with multi-level caching"""
    key = f"user_emb:{user_id}"
    
    # Try cache
    embedding = ml_cache.get(key)
    if embedding is not None:
        return embedding
    
    # Compute
    embedding = embedding_model.encode(user_id)
    
    # Cache
    ml_cache.put(key, embedding, ttl=3600)
    
    return embedding
```

---

## Cache Warming Strategies

### Proactive Cache Warming

```python
import threading
import time
from queue import Queue

class CacheWarmer:
    """
    Proactively warm cache before requests arrive
    
    Strategies:
    1. Popular items (based on historical data)
    2. Scheduled warmup (daily, hourly)
    3. Predictive warmup (ML-based)
    """
    
    def __init__(self, cache, compute_fn):
        self.cache = cache
        self.compute_fn = compute_fn
        
        self.warmup_queue = Queue()
        self.is_running = False
    
    def warm_popular_items(self, items, priority='high'):
        """Warm cache with popular items"""
        print(f"Warming {len(items)} popular items...")
        
        for item in items:
            key, args = item
            
            # Check if already cached
            if self.cache.get(key) is not None:
                continue
            
            # Compute and cache
            try:
                result = self.compute_fn(*args)
                self.cache.put(key, result)
            except Exception as e:
                print(f"Error warming {key}: {e}")
    
    def warm_on_schedule(self, items, interval_seconds=3600):
        """Periodically warm cache"""
        def warmup_worker():
            while self.is_running:
                self.warm_popular_items(items)
                time.sleep(interval_seconds)
        
        self.is_running = True
        worker = threading.Thread(target=warmup_worker, daemon=True)
        worker.start()
    
    def stop(self):
        """Stop scheduled warmup"""
        self.is_running = False

# Usage
def compute_recommendation(user_id, model):
    """Expensive recommendation computation"""
    return model.recommend(user_id, n=10)

cache = LRUCache(capacity=10000)
warmer = CacheWarmer(cache, compute_recommendation)

# Warm cache with top 1000 users
popular_users = get_top_1000_active_users()
items = [
    (f"rec:{user_id}", (user_id, recommendation_model))
    for user_id in popular_users
]

warmer.warm_popular_items(items)

# Or schedule periodic warmup
warmer.warm_on_schedule(items, interval_seconds=3600)
```

---

## Cache Invalidation

### Push-Based Invalidation

```python
import redis

class CacheInvalidator:
    """
    Cache invalidation using Redis Pub/Sub
    
    Pattern:
    - When model updates, publish invalidation message
    - All cache instances subscribe and clear relevant entries
    """
    
    def __init__(self, redis_host='localhost'):
        self.redis_pub = redis.Redis(host=redis_host)
        self.redis_sub = redis.Redis(host=redis_host)
        
        self.cache = {}
        self.invalidation_count = 0
    
    def subscribe_to_invalidations(self, channel='cache:invalidate'):
        """Subscribe to invalidation messages"""
        pubsub = self.redis_sub.pubsub()
        pubsub.subscribe(channel)
        
        def listen():
            for message in pubsub.listen():
                if message['type'] == 'message':
                    self._handle_invalidation(message['data'])
        
        # Start listener thread
        listener = threading.Thread(target=listen, daemon=True)
        listener.start()
    
    def _handle_invalidation(self, message):
        """Handle invalidation message"""
        # Message format: "model_id:v2"
        invalidation_key = message.decode('utf-8')
        
        # Remove matching cache entries
        keys_to_remove = [
            k for k in self.cache.keys()
            if k.startswith(invalidation_key)
        ]
        
        for key in keys_to_remove:
            del self.cache[key]
        
        self.invalidation_count += len(keys_to_remove)
        print(f"Invalidated {len(keys_to_remove)} cache entries")
    
    def invalidate_model(self, model_id):
        """Publish invalidation message"""
        message = f"{model_id}:v"
        self.redis_pub.publish('cache:invalidate', message)

# Usage
invalidator = CacheInvalidator()
invalidator.subscribe_to_invalidations()

# When model is updated
def update_model(model_id, new_model):
    """Update model and invalidate cache"""
    # Deploy new model
    deploy_model(new_model)
    
    # Invalidate all predictions for this model
    invalidator.invalidate_model(model_id)
```

---

## Feature Store Caching

```python
class FeatureStoreCache:
    """
    Caching layer for feature store
    
    Features:
    - Cache precomputed features
    - Batch feature retrieval
    - Freshness guarantees
    """
    
    def __init__(self, redis_client, ttl=3600):
        self.redis = redis_client
        self.ttl = ttl
    
    def get_features(self, entity_ids, feature_names):
        """
        Get features for multiple entities (batch)
        
        Args:
            entity_ids: List of entity IDs
            feature_names: List of feature names
        
        Returns:
            Dict of entity_id -> feature_dict
        """
        results = {}
        cache_misses = []
        
        # Try cache first
        for entity_id in entity_ids:
            cache_key = f"features:{entity_id}"
            cached = self.redis.get(cache_key)
            
            if cached:
                # Parse cached features
                features = json.loads(cached)
                
                # Filter to requested features
                filtered = {
                    fname: features[fname]
                    for fname in feature_names
                    if fname in features
                }
                
                if len(filtered) == len(feature_names):
                    results[entity_id] = filtered
                else:
                    cache_misses.append(entity_id)
            else:
                cache_misses.append(entity_id)
        
        # Compute missing features
        if cache_misses:
            computed = self._compute_features(cache_misses, feature_names)
            
            # Cache computed features
            for entity_id, features in computed.items():
                self._cache_features(entity_id, features)
                results[entity_id] = features
        
        return results
    
    def _compute_features(self, entity_ids, feature_names):
        """Compute features from feature store"""
        # Call actual feature store
        return compute_features_batch(entity_ids, feature_names)
    
    def _cache_features(self, entity_id, features):
        """Cache features for entity"""
        cache_key = f"features:{entity_id}"
        self.redis.setex(
            cache_key,
            self.ttl,
            json.dumps(features)
        )
    
    def invalidate_entity(self, entity_id):
        """Invalidate features for entity"""
        cache_key = f"features:{entity_id}"
        self.redis.delete(cache_key)

# Usage
feature_cache = FeatureStoreCache(redis_client, ttl=300)

# Get features for batch of users
user_ids = [123, 456, 789]
feature_names = ['age', 'location', 'purchase_count']

features = feature_cache.get_features(user_ids, feature_names)
```

---

## Connection to Linked Lists (Day 10 DSA)

Cache implementations heavily use linked list concepts:

```python
class DoublyLinkedNode:
    """Node for doubly-linked list (used in LRU)"""
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class ProductionLRUCache:
    """
    Production LRU cache using doubly-linked list
    
    Connection to Day 10 DSA:
    - Uses linked list for maintaining order
    - Pointer manipulation similar to reversal
    - O(1) operations through careful pointer management
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        
        # Dummy head and tail
        self.head = DoublyLinkedNode(0, 0)
        self.tail = DoublyLinkedNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove node from list"""
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_head(self, node):
        """Move node to head (most recently used)"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self):
        """Remove least recently used (tail.prev)"""
        res = self.tail.prev
        self._remove_node(res)
        return res
    
    def get(self, key):
        """Get value"""
        node = self.cache.get(key)
        if not node:
            return -1
        
        self._move_to_head(node)
        return node.value
    
    def put(self, key, value):
        """Put key-value"""
        node = self.cache.get(key)
        
        if node:
            node.value = value
            self._move_to_head(node)
        else:
            new_node = DoublyLinkedNode(key, value)
            self.cache[key] = new_node
            self._add_node(new_node)
            
            if len(self.cache) > self.capacity:
                tail = self._pop_tail()
                del self.cache[tail.key]
```

---

## Understanding Cache Performance

### Cache Hit Rate Analysis

```python
class CachePerformanceAnalyzer:
    """
    Analyze and optimize cache performance
    
    Key metrics:
    - Hit rate: % of requests served from cache
    - Miss rate: % of requests requiring computation
    - Latency reduction: Time saved by caching
    - Memory efficiency: Cache size vs hit rate
    """
    
    def __init__(self):
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.hit_latencies = []
        self.miss_latencies = []
    
    def record_hit(self, latency_ms):
        """Record cache hit"""
        self.cache_hits += 1
        self.total_requests += 1
        self.hit_latencies.append(latency_ms)
    
    def record_miss(self, latency_ms):
        """Record cache miss"""
        self.cache_misses += 1
        self.total_requests += 1
        self.miss_latencies.append(latency_ms)
    
    def get_metrics(self):
        """Calculate performance metrics"""
        if self.total_requests == 0:
            return {}
        
        hit_rate = self.cache_hits / self.total_requests
        miss_rate = self.cache_misses / self.total_requests
        
        avg_hit_latency = (
            sum(self.hit_latencies) / len(self.hit_latencies)
            if self.hit_latencies else 0
        )
        
        avg_miss_latency = (
            sum(self.miss_latencies) / len(self.miss_latencies)
            if self.miss_latencies else 0
        )
        
        # Calculate latency reduction
        avg_latency_with_cache = (
            hit_rate * avg_hit_latency + miss_rate * avg_miss_latency
        )
        
        latency_reduction = (
            (avg_miss_latency - avg_latency_with_cache) / avg_miss_latency
            if avg_miss_latency > 0 else 0
        )
        
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'miss_rate': miss_rate,
            'avg_hit_latency_ms': avg_hit_latency,
            'avg_miss_latency_ms': avg_miss_latency,
            'avg_overall_latency_ms': avg_latency_with_cache,
            'latency_reduction_pct': latency_reduction * 100
        }
    
    def print_report(self):
        """Print performance report"""
        metrics = self.get_metrics()
        
        print("\n" + "="*60)
        print("CACHE PERFORMANCE REPORT")
        print("="*60)
        print(f"Total Requests:        {metrics['total_requests']:,}")
        print(f"Cache Hits:            {metrics['cache_hits']:,}")
        print(f"Cache Misses:          {metrics['cache_misses']:,}")
        print(f"Hit Rate:              {metrics['hit_rate']:.2%}")
        print(f"Miss Rate:             {metrics['miss_rate']:.2%}")
        print(f"\nLatency Analysis:")
        print(f"  Cache Hit:           {metrics['avg_hit_latency_ms']:.2f} ms")
        print(f"  Cache Miss:          {metrics['avg_miss_latency_ms']:.2f} ms")
        print(f"  Overall Average:     {metrics['avg_overall_latency_ms']:.2f} ms")
        print(f"  Latency Reduction:   {metrics['latency_reduction_pct']:.1f}%")
        print("="*60)

# Usage example
analyzer = CachePerformanceAnalyzer()

# Simulate requests
import random
import time

cache = LRUCache(capacity=100)

for i in range(1000):
    key = f"key_{random.randint(1, 150)}"
    
    # Check cache
    start = time.perf_counter()
    value = cache.get(key)
    
    if value is not None:
        # Cache hit (fast)
        latency = (time.perf_counter() - start) * 1000
        analyzer.record_hit(latency)
    else:
        # Cache miss (slow - simulate computation)
        time.sleep(0.001)  # 1ms computation
        latency = (time.perf_counter() - start) * 1000
        analyzer.record_miss(latency)
        
        # Store in cache
        cache.put(key, f"value_{key}")

analyzer.print_report()
```

### Cache Size Optimization

```python
class CacheSizeOptimizer:
    """
    Find optimal cache size for given workload
    
    Trade-off: Larger cache = higher hit rate but more memory
    """
    
    def __init__(self, workload):
        """
        Args:
            workload: List of access patterns (keys)
        """
        self.workload = workload
    
    def find_optimal_size(self, max_size=10000, step=100):
        """
        Test different cache sizes
        
        Returns optimal size based on diminishing returns
        """
        results = []
        
        print("Testing cache sizes...")
        print(f"{'Size':<10} {'Hit Rate':<12} {'Marginal Gain':<15}")
        print("-" * 40)
        
        prev_hit_rate = 0
        
        for size in range(step, max_size + 1, step):
            hit_rate = self._simulate_cache(size)
            marginal_gain = hit_rate - prev_hit_rate
            
            results.append({
                'size': size,
                'hit_rate': hit_rate,
                'marginal_gain': marginal_gain
            })
            
            print(f"{size:<10} {hit_rate:<12.2%} {marginal_gain:<15.4%}")
            
            prev_hit_rate = hit_rate
            
            # Stop if marginal gain is too small
            if marginal_gain < 0.001:  # 0.1% gain
                print(f"\nDiminishing returns detected at size {size}")
                break
        
        return results
    
    def _simulate_cache(self, size):
        """Simulate cache with given size"""
        cache = LRUCache(capacity=size)
        hits = 0
        
        for key in self.workload:
            if cache.get(key) is not None:
                hits += 1
            else:
                cache.put(key, True)
        
        return hits / len(self.workload)

# Generate workload (Zipf distribution - realistic for many applications)
import numpy as np

def generate_zipf_workload(n_items=1000, n_requests=10000, alpha=1.5):
    """
    Generate Zipf-distributed workload
    
    Zipf law: Some items are accessed much more frequently
    (80/20 rule, power law distribution)
    """
    # Zipf distribution
    probabilities = np.array([1.0 / (i ** alpha) for i in range(1, n_items + 1)])
    probabilities /= probabilities.sum()
    
    # Generate requests
    workload = np.random.choice(
        [f"key_{i}" for i in range(n_items)],
        size=n_requests,
        p=probabilities
    )
    
    return workload.tolist()

# Find optimal cache size
workload = generate_zipf_workload(n_items=1000, n_requests=10000)
optimizer = CacheSizeOptimizer(workload)
results = optimizer.find_optimal_size(max_size=500, step=50)

# Plot results
import matplotlib.pyplot as plt

sizes = [r['size'] for r in results]
hit_rates = [r['hit_rate'] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(sizes, hit_rates, marker='o')
plt.xlabel('Cache Size')
plt.ylabel('Hit Rate')
plt.title('Cache Size vs Hit Rate')
plt.grid(True)
plt.savefig('cache_size_optimization.png')
```

---

## Advanced Caching Patterns

### Write-Through vs Write-Back Cache

```python
class WriteThroughCache:
    """
    Write-through cache: Write to cache and database simultaneously
    
    Pros:
    - Data consistency
    - Simple to implement
    
    Cons:
    - Slower writes
    - Every write hits database
    """
    
    def __init__(self, cache, database):
        self.cache = cache
        self.database = database
    
    def get(self, key):
        """Read with cache"""
        # Try cache first
        value = self.cache.get(key)
        if value is not None:
            return value
        
        # Cache miss: read from database
        value = self.database.get(key)
        if value is not None:
            self.cache.put(key, value)
        
        return value
    
    def put(self, key, value):
        """Write to both cache and database"""
        # Write to database first
        self.database.put(key, value)
        
        # Then update cache
        self.cache.put(key, value)

class WriteBackCache:
    """
    Write-back cache: Write to cache only, flush to database later
    
    Pros:
    - Fast writes
    - Batching possible
    
    Cons:
    - Risk of data loss
    - More complex
    """
    
    def __init__(self, cache, database, flush_interval=5):
        self.cache = cache
        self.database = database
        self.flush_interval = flush_interval
        
        self.dirty_keys = set()
        self.last_flush = time.time()
    
    def get(self, key):
        """Read with cache"""
        value = self.cache.get(key)
        if value is not None:
            return value
        
        value = self.database.get(key)
        if value is not None:
            self.cache.put(key, value)
        
        return value
    
    def put(self, key, value):
        """Write to cache only"""
        self.cache.put(key, value)
        self.dirty_keys.add(key)
        
        # Check if we need to flush
        if time.time() - self.last_flush > self.flush_interval:
            self.flush()
    
    def flush(self):
        """Flush dirty keys to database"""
        if not self.dirty_keys:
            return
        
        print(f"Flushing {len(self.dirty_keys)} dirty keys...")
        
        for key in self.dirty_keys:
            value = self.cache.get(key)
            if value is not None:
                self.database.put(key, value)
        
        self.dirty_keys.clear()
        self.last_flush = time.time()

# Example database simulation
class SimpleDatabase:
    def __init__(self):
        self.data = {}
        self.read_count = 0
        self.write_count = 0
    
    def get(self, key):
        self.read_count += 1
        time.sleep(0.001)  # Simulate latency
        return self.data.get(key)
    
    def put(self, key, value):
        self.write_count += 1
        time.sleep(0.001)  # Simulate latency
        self.data[key] = value

# Compare write-through vs write-back
db1 = SimpleDatabase()
cache1 = LRUCache(capacity=100)
write_through = WriteThroughCache(cache1, db1)

db2 = SimpleDatabase()
cache2 = LRUCache(capacity=100)
write_back = WriteBackCache(cache2, db2)

# Benchmark writes
import time

# Write-through
start = time.time()
for i in range(100):
    write_through.put(f"key_{i}", f"value_{i}")
wt_time = time.time() - start

# Write-back
start = time.time()
for i in range(100):
    write_back.put(f"key_{i}", f"value_{i}")
write_back.flush()  # Final flush
wb_time = time.time() - start

print(f"Write-through: {wt_time:.3f}s, DB writes: {db1.write_count}")
print(f"Write-back:    {wb_time:.3f}s, DB writes: {db2.write_count}")
```

### Cache Aside Pattern

```python
class CacheAsidePattern:
    """
    Cache-aside (lazy loading): Application manages cache
    
    Most common pattern for ML systems
    
    Flow:
    1. Check cache
    2. If miss, query database
    3. Store in cache
    4. Return result
    """
    
    def __init__(self, cache, database):
        self.cache = cache
        self.database = database
        
        self.stats = {
            'reads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'writes': 0
        }
    
    def get(self, key):
        """
        Get with cache-aside pattern
        
        Application is responsible for loading cache
        """
        self.stats['reads'] += 1
        
        # Try cache first
        value = self.cache.get(key)
        if value is not None:
            self.stats['cache_hits'] += 1
            return value
        
        # Cache miss: load from database
        self.stats['cache_misses'] += 1
        value = self.database.get(key)
        
        if value is not None:
            # Populate cache for next time
            self.cache.put(key, value)
        
        return value
    
    def put(self, key, value):
        """
        Write to database, invalidate cache
        
        Simple approach: Just write to DB and remove from cache
        Next read will repopulate
        """
        self.stats['writes'] += 1
        
        # Write to database
        self.database.put(key, value)
        
        # Invalidate cache entry
        # (Could also update cache here - depends on use case)
        if key in self.cache.cache:
            del self.cache.cache[key]
    
    def get_stats(self):
        """Get cache statistics"""
        hit_rate = (
            self.stats['cache_hits'] / self.stats['reads']
            if self.stats['reads'] > 0 else 0
        )
        
        return {
            **self.stats,
            'hit_rate': hit_rate
        }

# Usage for ML predictions
class MLPredictionService:
    """
    ML prediction service with cache-aside pattern
    """
    
    def __init__(self, model, cache_capacity=1000):
        self.model = model
        self.cache = LRUCache(capacity=cache_capacity)
        
        # Fake database for persisted predictions
        self.prediction_db = {}
        
        self.pattern = CacheAsidePattern(
            self.cache,
            self.prediction_db
        )
    
    def predict(self, features):
        """
        Predict with caching
        
        Args:
            features: Feature vector (tuple for hashability)
        
        Returns:
            Prediction
        """
        # Create cache key from features
        cache_key = hash(features)
        
        # Try cache-aside pattern
        cached_prediction = self.pattern.get(cache_key)
        if cached_prediction is not None:
            return cached_prediction
        
        # Compute prediction (expensive)
        prediction = self.model.predict([features])[0]
        
        # Store in database and cache
        self.pattern.put(cache_key, prediction)
        
        return prediction
    
    def get_cache_stats(self):
        """Get caching statistics"""
        return self.pattern.get_stats()

# Example usage
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train simple model
X_train = np.random.randn(100, 5)
y_train = (X_train.sum(axis=1) > 0).astype(int)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

# Create prediction service
service = MLPredictionService(model, cache_capacity=100)

# Make predictions (some repeated)
for _ in range(1000):
    # Generate features (with some repetition)
    features = tuple(np.random.randint(0, 10, size=5))
    prediction = service.predict(features)

print("Cache statistics:")
print(service.get_cache_stats())
```

---

## Cache Stampede Prevention

### Problem: Thundering Herd

```python
class CacheStampedeProtection:
    """
    Prevent cache stampede (thundering herd)
    
    Problem:
    - Cache entry expires
    - Many requests try to regenerate simultaneously
    - Database/model gets overwhelmed
    
    Solution:
    - Use locking to ensure only one request regenerates
    - Others wait for that request to complete
    """
    
    def __init__(self, cache, compute_fn):
        self.cache = cache
        self.compute_fn = compute_fn
        
        # Lock for each key
        self.locks = {}
        self.master_lock = threading.Lock()
    
    def get(self, key):
        """
        Get with stampede protection
        
        Uses double-check locking pattern
        """
        # First check: Try cache (no lock)
        value = self.cache.get(key)
        if value is not None:
            return value
        
        # Get or create lock for this key
        with self.master_lock:
            if key not in self.locks:
                self.locks[key] = threading.Lock()
            key_lock = self.locks[key]
        
        # Acquire key-specific lock
        with key_lock:
            # Second check: Try cache again (another thread might have filled it)
            value = self.cache.get(key)
            if value is not None:
                return value
            
            # Compute value (only one thread does this)
            print(f"Computing value for {key} (thread: {threading.current_thread().name})")
            value = self.compute_fn(key)
            
            # Store in cache
            self.cache.put(key, value)
            
            return value

# Demo: Simulate stampede
import threading
import time

def expensive_computation(key):
    """Simulate expensive computation"""
    time.sleep(0.1)  # 100ms
    return f"computed_value_for_{key}"

cache = LRUCache(capacity=100)
protector = CacheStampedeProtection(cache, expensive_computation)

# Simulate stampede: 10 threads requesting same key
def make_request(key, results, index):
    start = time.time()
    result = protector.get(key)
    duration = time.time() - start
    results[index] = duration

results = [0] * 10
threads = []

# Clear cache to force computation
cache = LRUCache(capacity=100)
protector.cache = cache

print("Simulating cache stampede for key 'popular_item'...")
start_time = time.time()

for i in range(10):
    t = threading.Thread(
        target=make_request,
        args=('popular_item', results, i),
        name=f"Thread-{i}"
    )
    threads.append(t)
    t.start()

for t in threads:
    t.join()

total_time = time.time() - start_time

print(f"\nTotal time: {total_time:.3f}s")
print(f"Average request time: {sum(results)/len(results):.3f}s")
print(f"Max request time: {max(results):.3f}s")
print(f"Min request time: {min(results):.3f}s")
print("\nWith protection, only one thread computed (others waited)")
```

### Probabilistic Early Expiration

```python
class ProbabilisticCache:
    """
    Probabilistic early expiration to prevent stampede
    
    Idea: Refresh cache before expiration with increasing probability
    This spreads out refresh operations
    """
    
    def __init__(self, cache, compute_fn, ttl=60, beta=1.0):
        """
        Args:
            ttl: Time to live in seconds
            beta: Controls early expiration probability
        """
        self.cache = cache
        self.compute_fn = compute_fn
        self.ttl = ttl
        self.beta = beta
        
        # Track insertion times
        self.insertion_times = {}
    
    def get(self, key):
        """
        Get with probabilistic early expiration
        
        Formula: Should refresh if:
        current_time - stored_time * beta * log(random) >= ttl
        """
        # Check cache
        value = self.cache.get(key)
        
        if value is not None and key in self.insertion_times:
            # Calculate age
            age = time.time() - self.insertion_times[key]
            
            # Probabilistic early expiration
            import random
            import math
            
            # XFetch algorithm
            delta = self.ttl - age
            if delta * self.beta * math.log(random.random()) < 0:
                # Refresh early
                print(f"Early refresh for {key} (age: {age:.1f}s)")
                value = self._refresh(key)
            
            return value
        
        # Cache miss or expired
        return self._refresh(key)
    
    def _refresh(self, key):
        """Refresh cache entry"""
        value = self.compute_fn(key)
        self.cache.put(key, value)
        self.insertion_times[key] = time.time()
        return value

# Demo
def compute_value(key):
    time.sleep(0.01)
    return f"value_{key}_{time.time()}"

pcache = ProbabilisticCache(
    LRUCache(capacity=100),
    compute_value,
    ttl=5,  # 5 second TTL
    beta=1.0
)

# Access same key multiple times
for i in range(20):
    value = pcache.get('test_key')
    time.sleep(0.3)  # 300ms between requests
```

---

## Distributed Cache Challenges

### Cache Consistency

```python
class DistributedCacheCoordinator:
    """
    Coordinate cache across multiple instances
    
    Challenges:
    1. Keeping caches in sync
    2. Handling partial failures
    3. Eventual consistency
    """
    
    def __init__(self, redis_client, instance_id):
        self.redis = redis_client
        self.instance_id = instance_id
        
        # Local L1 cache
        self.local_cache = LRUCache(capacity=1000)
        
        # Subscribe to invalidation messages
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe('cache:invalidate')
        
        # Start listener thread
        self.listener_thread = threading.Thread(
            target=self._listen_for_invalidations,
            daemon=True
        )
        self.listener_thread.start()
    
    def get(self, key):
        """
        Get from multi-level cache
        
        L1 (local) -> L2 (Redis) -> Compute
        """
        # Try local cache
        value = self.local_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis
        value = self.redis.get(key)
        if value is not None:
            value = pickle.loads(value)
            # Populate local cache
            self.local_cache.put(key, value)
            return value
        
        return None
    
    def put(self, key, value, ttl=3600):
        """
        Put in both levels and notify others
        """
        # Store in local cache
        self.local_cache.put(key, value)
        
        # Store in Redis
        self.redis.setex(key, ttl, pickle.dumps(value))
        
        # Notify other instances to invalidate their L1
        self.redis.publish(
            'cache:invalidate',
            json.dumps({
                'key': key,
                'source_instance': self.instance_id
            })
        )
    
    def _listen_for_invalidations(self):
        """Listen for invalidation messages"""
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                
                # Don't invalidate if we sent the message
                if data['source_instance'] != self.instance_id:
                    key = data['key']
                    
                    # Invalidate local cache
                    if key in self.local_cache.cache:
                        del self.local_cache.cache[key]
                        print(f"Invalidated {key} from local cache")

# Usage across multiple instances
# Instance 1
coordinator1 = DistributedCacheCoordinator(redis_client, instance_id='instance1')

# Instance 2
coordinator2 = DistributedCacheCoordinator(redis_client, instance_id='instance2')

# Instance 1 writes
coordinator1.put('shared_key', 'value_from_instance1')

# Instance 2 reads (will get from Redis)
value = coordinator2.get('shared_key')
```

---

## Key Takeaways

✅ **Multiple eviction policies** - LRU, LFU, TTL for different use cases  
✅ **Distributed caching** - Redis for shared cache across services  
✅ **Multi-level caching** - L1 (local) + L2 (distributed) for optimal performance  
✅ **Cache warming** - Proactive population of hot items  
✅ **Invalidation strategies** - Push-based and pull-based  
✅ **Linked list connection** - Understanding pointer manipulation helps with cache implementation  
✅ **Monitor cache metrics** - Hit rate, latency, memory usage  

---

**Originally published at:** [arunbaby.com/ml-system-design/0010-caching-strategies](https://www.arunbaby.com/ml-system-design/0010-caching-strategies/)

*If you found this helpful, consider sharing it with others who might benefit.*


