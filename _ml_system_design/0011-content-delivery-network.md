---
title: "Content Delivery Networks (CDN)"
day: 11
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - cdn
  - edge-computing
  - caching
  - load-balancing
  - model-serving
  - global-distribution
  - latency-optimization
subdomain: Infrastructure
tech_stack: [Redis, Python, ONNX, FastAPI, GeoDNS, Docker, Kubernetes]
scale: "1M+ requests/sec globally, 50+ regions"
companies: [Cloudflare, Akamai, AWS CloudFront, Google Cloud CDN, Azure CDN, Fastly]
related_dsa_day: 11
related_speech_day: 11
related_agents_day: 11
---

**Design a global CDN for ML systems: Edge caching reduces latency from 500ms to 50ms. Critical for real-time predictions worldwide.**

## Problem Statement

Design a **Content Delivery Network (CDN)** for serving:
1. **ML model inference** (predictions at the edge)
2. **Static assets** (model weights, configs, embeddings)
3. **API responses** (cached predictions, feature data)

### Why Do We Need a CDN?

**The Core Problem: Distance Creates Latency**

Imagine you're a user in Tokyo trying to access a website hosted in Virginia, USA:

```
User (Tokyo) ──────── 10,000 km ──────── Server (Virginia)
                  ~150ms round-trip time
```

**The physics problem:**
- Light travels at 300,000 km/s
- Signal in fiber: ~200,000 km/s
- Tokyo ↔ Virginia: ~10,000 km
- **Theoretical minimum:** 50ms
- **Reality with routing:** 150-200ms

**What if we could serve from Tokyo instead?**

```
User (Tokyo) ── 10 km ── Edge Server (Tokyo)
                ~1-2ms!
```

That's a **75-100x improvement** just from being geographically closer!

### Real-World Impact on ML Systems

**Scenario: Real-time recommendation system**

| Architecture | Latency | User Experience |
|--------------|---------|-----------------|
| **Without CDN**: Request → US → Model Inference → Response | 200ms+ | Noticeable delay, users leave |
| **With CDN**: Request → Local Edge → Cached/Local Inference → Response | 20-50ms | Feels instant ✓ |

**The business impact:**
- Every 100ms of latency = 1% drop in sales (Amazon study)
- For ML systems: Users won't wait for slow predictions
- CDN makes your ML system feel instant globally

### What CDN Does for You

**1. Geographic Distribution**
Cache content at multiple locations worldwide (edge servers)

**2. Intelligent Caching**
Store frequently accessed content close to users

**3. Smart Routing**
Direct users to the best edge server (closest + healthy + low load)

**4. Fault Tolerance**
If one edge fails, route to another

**5. Bandwidth Savings**
Serve from edge → Less traffic to origin → Lower costs

### Requirements

**Functional:**
- Serve content from geographically distributed edge locations
- Cache popular content close to users
- Route requests to nearest/best edge server
- Handle cache invalidation and updates
- Support both static and dynamic content

**Non-Functional:**
- **Latency:** < 50ms p99 for edge hits (vs 200-500ms from origin)
- **Availability:** 99.99% uptime (4 minutes downtime/month)
- **Scalability:** Handle 1M+ requests/second globally
- **Cache hit rate:** > 80% for static content (fewer origin requests)
- **Global coverage:** Presence in 50+ regions

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       USER REQUESTS                          │
│   🌍 Asia    🌍 Europe    🌍 Americas    🌍 Africa          │
└───────┬────────────┬────────────┬──────────────┬────────────┘
        │            │            │              │
        ↓            ↓            ↓              ↓
┌────────────────────────────────────────────────────────────┐
│                   DNS / GLOBAL LOAD BALANCER               │
│  • GeoDNS routing                                          │
│  • Health checks                                           │
│  • Latency-based routing                                   │
└────────┬────────────┬────────────┬──────────────┬──────────┘
         │            │            │              │
    ┌────▼────┐  ┌───▼─────┐ ┌───▼─────┐   ┌───▼─────┐
    │ Edge    │  │ Edge    │ │ Edge    │   │ Edge    │
    │ Tokyo   │  │ London  │ │ N.Virginia│  │ Mumbai  │
    │         │  │         │ │         │   │         │
    │ L1 Cache│  │ L1 Cache│ │ L1 Cache│   │ L1 Cache│
    │ (Redis) │  │ (Redis) │ │ (Redis) │   │ (Redis) │
    │         │  │         │ │         │   │         │
    │ ML Model│  │ ML Model│ │ ML Model│   │ ML Model│
    └────┬────┘  └────┬────┘ └────┬────┘   └────┬────┘
         │            │            │              │
         └────────────┴────────────┴──────────────┘
                      │
              ┌───────▼────────┐
              │ ORIGIN SERVERS │
              │                │
              │ • Master models│
              │ • Databases    │
              │ • Feature store│
              │ • Object store │
              └────────────────┘
```

---

## Core Components

### 1. Edge Servers

**Purpose:** Serve content from locations close to users

Before we dive into code, let's understand the concept:

**What is an Edge Server?**

Think of edge servers like local convenience stores:
- **Origin Server** = Central warehouse (far away, has everything)
- **Edge Server** = Local store (nearby, has popular items)

When you need milk:
- Without edge: Drive to warehouse (30 min)
- With edge: Walk to local store (2 min)

**Multi-Level Cache Strategy**

Edge servers use multiple cache layers:

```
Request → L1 Cache (Redis, in-memory)  ← Fastest, smallest
          ↓ Miss
          L2 Cache (Disk, local SSD)    ← Fast, medium
          ↓ Miss
          Origin Server (Database)      ← Slow, largest
```

**Why multiple levels?**

1. **L1 (Redis)**: Hot data, 50-100ms access, expensive ($100/GB/month)
2. **L2 (Disk)**: Warm data, 5-10ms access, cheap ($10/GB/month)
3. **Origin**: Cold data, 100-500ms access, cheapest ($0.02/GB/month)

**Trade-off**: Speed vs Cost vs Capacity

| Cache | Speed | Cost | Capacity | Use Case |
|-------|-------|------|----------|----------|
| L1 (Redis) | 1ms | High | Small (10GB) | Prediction results, hot features |
| L2 (Disk) | 10ms | Medium | Medium (100GB) | Model weights, embeddings |
| Origin | 200ms | Low | Large (TB+) | Full dataset, historical data |

```python
class EdgeServer:
    """
    CDN edge server
    
    Components:
    - L1 cache (Redis): Hot content
    - L2 cache (local disk): Warm content
    - ML model: For edge inference
    - Origin client: Fetch misses from origin
    """
    
    def __init__(self, region, origin_url):
        self.region = region
        self.origin_url = origin_url
        
        # Multi-level cache
        import redis
        import pickle
        self.l1_cache = redis.Redis(host='localhost', port=6379)
        
        # Minimal DiskCache stub for illustration
        class DiskCache:
            def __init__(self, size_gb=100):
                self.store = {}
            def get(self, key):
                return self.store.get(key)
            def set(self, key, value):
                self.store[key] = value
            def delete(self, key):
                self.store.pop(key, None)
            def delete_pattern(self, pattern):
                # naive pattern matcher
                import fnmatch
                keys = [k for k in self.store.keys() if fnmatch.fnmatch(k, pattern)]
                for k in keys:
                    self.store.pop(k, None)
        self.l2_cache = DiskCache(size_gb=100)
        
        # ML model for edge inference
        def load_model(path):
            return object()
        self.model = load_model('model.onnx')
        
        # Metrics
        self.metrics = EdgeMetrics()
    
    async def handle_request(self, request):
        """
        Handle incoming request
        
        Flow:
        1. Check L1 cache (Redis)
        2. Check L2 cache (disk)
        3. Fetch from origin
        4. Update caches
        """
        import time, json, pickle
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Try L1 cache
        response = await self._check_l1_cache(cache_key)
        if response:
            self.metrics.record_hit('l1', time.time() - start_time)
            return response
        
        # Try L2 cache
        response = await self._check_l2_cache(cache_key)
        if response:
            # Promote to L1
            await self._store_l1_cache(cache_key, response)
            self.metrics.record_hit('l2', time.time() - start_time)
            return response
        
        # Cache miss: fetch from origin
        response = await self._fetch_from_origin(request)
        
        # Update caches
        await self._store_l1_cache(cache_key, response)
        await self._store_l2_cache(cache_key, response)
        
        self.metrics.record_miss(time.time() - start_time)
        
        return response
    
    async def _check_l1_cache(self, key):
        """Check L1 (Redis) cache"""
        try:
            data = self.l1_cache.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            print(f"L1 cache error: {e}")
        
        return None
    
    async def _store_l1_cache(self, key, value, ttl=300):
        """Store in L1 cache with TTL"""
        try:
            self.l1_cache.setex(
                key,
                ttl,
                pickle.dumps(value)
            )
        except Exception as e:
            print(f"L1 cache store error: {e}")
    
    async def _check_l2_cache(self, key):
        """Check L2 (disk) cache"""
        return self.l2_cache.get(key)
    
    async def _store_l2_cache(self, key, value):
        """Store in L2 cache"""
        self.l2_cache.set(key, value)
    
    async def _fetch_from_origin(self, request):
        """Fetch from origin server"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.origin_url}{request.path}",
                json=request.data
            ) as response:
                return await response.json()
    
    def _generate_cache_key(self, request):
        """Generate cache key from request"""
        import hashlib
        
        # Include path and normalized data
        key_data = f"{request.path}:{json.dumps(request.data, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

class EdgeMetrics:
    """Track edge server metrics"""
    
    def __init__(self):
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
        
        self.l1_latencies = []
        self.l2_latencies = []
        self.miss_latencies = []
    
    def record_hit(self, level, latency):
        if level == 'l1':
            self.l1_hits += 1
            self.l1_latencies.append(latency)
        elif level == 'l2':
            self.l2_hits += 1
            self.l2_latencies.append(latency)
    
    def record_miss(self, latency):
        self.misses += 1
        self.miss_latencies.append(latency)
    
    def get_stats(self):
        total = self.l1_hits + self.l2_hits + self.misses
        
        return {
            'l1_hit_rate': self.l1_hits / total if total > 0 else 0,
            'l2_hit_rate': self.l2_hits / total if total > 0 else 0,
            'miss_rate': self.misses / total if total > 0 else 0,
            'avg_l1_latency_ms': np.mean(self.l1_latencies) * 1000 if self.l1_latencies else 0,
            'avg_l2_latency_ms': np.mean(self.l2_latencies) * 1000 if self.l2_latencies else 0,
            'avg_miss_latency_ms': np.mean(self.miss_latencies) * 1000 if self.miss_latencies else 0,
        }

# Example usage
edge = EdgeServer(region='us-east-1', origin_url='https://api.example.com')

# Simulate requests
async def simulate_requests():
    for i in range(100):
        request = Request(
            path='/predict',
            data={'features': [1, 2, 3, 4, 5]}
        )
        
        response = await edge.handle_request(request)
        print(f"Request {i}: {response}")
    
    # Print metrics
    stats = edge.metrics.get_stats()
    print("\nEdge Server Metrics:")
    for key, value in stats.items():
        if 'rate' in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value:.2f}")

# Run
import asyncio
asyncio.run(simulate_requests())
```

### 2. Global Load Balancer / GeoDNS

**Purpose:** Route requests to optimal edge server

```python
class GlobalLoadBalancer:
    """
    Route requests to best edge server
    
    Routing strategies:
    1. Geographic proximity
    2. Server load
    3. Health status
    4. Network latency
    """
    
    def __init__(self):
        self.edge_servers = self._discover_edge_servers()
        self.health_checker = HealthChecker(self.edge_servers)
        
        # Start health checking
        self.health_checker.start()
    
    def route_request(self, client_ip, request):
        """
        Route request to best edge server
        
        Args:
            client_ip: Client IP address
            request: Request object
        
        Returns:
            Best edge server
        """
        # Get client location
        client_location = self._geolocate_ip(client_ip)
        
        # Get healthy edge servers
        healthy_servers = self.health_checker.get_healthy_servers()
        
        if not healthy_servers:
            raise Exception("No healthy edge servers available")
        
        # Score each server
        scores = []
        
        for server in healthy_servers:
            score = self._score_server(
                server,
                client_location,
                request
            )
            scores.append((server, score))
        
        # Sort by score (higher is better)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return best server
        return scores[0][0]
    
    def _score_server(self, server, client_location, request):
        """
        Score server for given request
        
        Factors:
        - Geographic distance (weight: 0.5)
        - Server load (weight: 0.3)
        - Cache hit rate (weight: 0.2)
        """
        # Geographic proximity
        distance = self._calculate_distance(
            client_location,
            server.location
        )
        distance_score = 1.0 / (1.0 + distance / 1000)  # Normalize
        
        # Server load
        load = server.get_current_load()
        load_score = 1.0 - min(load, 1.0)
        
        # Cache hit rate
        hit_rate = server.metrics.get_stats()['l1_hit_rate']
        
        # Weighted score
        score = (
            0.5 * distance_score +
            0.3 * load_score +
            0.2 * hit_rate
        )
        
        return score
    
    def _geolocate_ip(self, ip):
        """
        Get geographic location from IP
        
        Uses MaxMind GeoIP or similar
        """
        import geoip2.database
        
        reader = geoip2.database.Reader('GeoLite2-City.mmdb')
        response = reader.city(ip)
        
        return {
            'lat': response.location.latitude,
            'lon': response.location.longitude,
            'city': response.city.name,
            'country': response.country.name
        }
    
    def _calculate_distance(self, loc1, loc2):
        """
        Calculate distance between two locations (km)
        
        Uses Haversine formula
        """
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in km
        
        lat1, lon1 = radians(loc1['lat']), radians(loc1['lon'])
        lat2, lon2 = radians(loc2['lat']), radians(loc2['lon'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        distance = R * c
        
        return distance
    
    def _discover_edge_servers(self):
        """Discover available edge servers"""
        # In production, this would query service registry
        return [
            EdgeServerInfo('us-east-1', 'https://edge-us-east-1.example.com', {'lat': 39.0, 'lon': -77.5}),
            EdgeServerInfo('eu-west-1', 'https://edge-eu-west-1.example.com', {'lat': 53.3, 'lon': -6.3}),
            EdgeServerInfo('ap-northeast-1', 'https://edge-ap-northeast-1.example.com', {'lat': 35.7, 'lon': 139.7}),
        ]

class HealthChecker:
    """
    Monitor health of edge servers
    
    Checks:
    - HTTP health endpoint
    - Response time
    - Error rate
    """
    
    def __init__(self, servers, check_interval=10):
        self.servers = servers
        self.check_interval = check_interval
        
        self.health_status = {server.region: True for server in servers}
        self.last_check = {server.region: 0 for server in servers}
        
        self.running = False
    
    def start(self):
        """Start health checking in background"""
        self.running = True
        
        import threading
        self.thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop health checking"""
        self.running = False
    
    def _health_check_loop(self):
        """Health check loop"""
        while self.running:
            for server in self.servers:
                healthy = self._check_server_health(server)
                self.health_status[server.region] = healthy
                self.last_check[server.region] = time.time()
            
            import time
            time.sleep(self.check_interval)
    
    def _check_server_health(self, server):
        """Check if server is healthy"""
        try:
            import requests
            
            response = requests.get(
                f"{server.url}/health",
                timeout=5
            )
            
            if response.status_code == 200:
                # Check response time
                if response.elapsed.total_seconds() < 1.0:
                    return True
            
            return False
        
        except Exception as e:
            print(f"Health check failed for {server.region}: {e}")
            return False
    
    def get_healthy_servers(self):
        """Get list of healthy servers"""
        return [
            server for server in self.servers
            if self.health_status[server.region]
        ]

class EdgeServerInfo:
    """Edge server information"""
    def __init__(self, region, url, location):
        self.region = region
        self.url = url
        self.location = location
        self.metrics = EdgeMetrics()
    
    def get_current_load(self):
        """Get current server load (0-1)"""
        # In production, query server metrics
        return 0.5  # Placeholder

# Example usage
glb = GlobalLoadBalancer()

# Route request
client_ip = '8.8.8.8'  # Google DNS (US)
request = Request(path='/predict', data={})

best_server = glb.route_request(client_ip, request)
print(f"Routing to: {best_server.region}")
```

### 3. Cache Invalidation System

**Purpose:** Propagate updates across edge servers

```python
class CacheInvalidationSystem:
    """
    Propagate cache invalidations to edge servers
    
    Methods:
    1. Push-based: Immediate invalidation
    2. Pull-based: Periodic refresh
    3. TTL-based: Automatic expiration
    """
    
    def __init__(self, edge_servers):
        self.edge_servers = edge_servers
        
        # Message queue for invalidations
        self.invalidation_queue = redis.Redis(host='localhost', port=6379)
        
        # Pub/sub for real-time propagation
        self.pubsub = self.invalidation_queue.pubsub()
        self.pubsub.subscribe('cache:invalidate')
    
    def invalidate(self, keys, pattern=False):
        """
        Invalidate cache keys across all edge servers
        
        Args:
            keys: List of keys to invalidate
            pattern: If True, treat keys as patterns
        """
        message = {
            'keys': keys,
            'pattern': pattern,
            'timestamp': time.time()
        }
        
        # Publish to all edge servers
        self.invalidation_queue.publish(
            'cache:invalidate',
            json.dumps(message)
        )
        
        print(f"Invalidated {len(keys)} keys across edge network")
    
    def invalidate_prefix(self, prefix):
        """
        Invalidate all keys with given prefix
        
        Example: invalidate_prefix('user:123:*')
        """
        self.invalidate([prefix], pattern=True)
    
    def invalidate_model_update(self, model_id):
        """
        Invalidate caches after model update
        
        Invalidates:
        - Model predictions
        - Model metadata
        - Related embeddings
        """
        patterns = [
            f"model:{model_id}:*",
            f"prediction:{model_id}:*",
            f"embedding:{model_id}:*"
        ]
        
        self.invalidate(patterns, pattern=True)
        
        print(f"Invalidated caches for model {model_id}")

class EdgeInvalidationListener:
    """
    Listen for invalidation messages on edge server
    """
    
    def __init__(self, edge_server):
        self.edge_server = edge_server
        
        # Subscribe to invalidations
        self.pubsub = redis.Redis(host='localhost', port=6379).pubsub()
        self.pubsub.subscribe('cache:invalidate')
        
        self.running = False
    
    def start(self):
        """Start listening for invalidations"""
        self.running = True
        
        import threading
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop listening"""
        self.running = False
    
    def _listen_loop(self):
        """Listen for invalidation messages"""
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                self._handle_invalidation(data)
    
    def _handle_invalidation(self, data):
        """Handle invalidation message"""
        keys = data['keys']
        pattern = data['pattern']
        
        if pattern:
            # Invalidate by pattern
            for key_pattern in keys:
                self._invalidate_pattern(key_pattern)
        else:
            # Invalidate specific keys
            for key in keys:
                self._invalidate_key(key)
    
    def _invalidate_key(self, key):
        """Invalidate specific key"""
        # Remove from L1 cache
        self.edge_server.l1_cache.delete(key)
        
        # Remove from L2 cache
        self.edge_server.l2_cache.delete(key)
        
        print(f"Invalidated key: {key}")
    
    def _invalidate_pattern(self, pattern):
        """Invalidate keys matching pattern"""
        # Scan L1 cache
        for key in self.edge_server.l1_cache.scan_iter(match=pattern):
            self.edge_server.l1_cache.delete(key)
        
        # Scan L2 cache
        self.edge_server.l2_cache.delete_pattern(pattern)
        
        print(f"Invalidated pattern: {pattern}")

# Example usage
edge_servers = [
    EdgeServer('us-east-1', 'https://origin.example.com'),
    EdgeServer('eu-west-1', 'https://origin.example.com'),
]

invalidation_system = CacheInvalidationSystem(edge_servers)

# Start listeners on each edge
for edge in edge_servers:
    listener = EdgeInvalidationListener(edge)
    listener.start()

# Trigger invalidation
invalidation_system.invalidate_model_update('model_v2')
```

---

## ML Model Serving at Edge

### Edge Inference

```python
class EdgeMLServer:
    """
    Serve ML models at edge for low-latency inference
    
    Benefits:
    - Reduced latency (no round trip to origin)
    - Reduced bandwidth
    - Better privacy (data doesn't leave region)
    """
    
    def __init__(self, model_path):
        # Load ONNX model for edge inference
        import onnxruntime as ort
        
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Cache for predictions
        self.prediction_cache = LRUCache(capacity=10000)
    
    def predict(self, features):
        """
        Predict with caching
        
        Args:
            features: Input features (must be hashable)
        
        Returns:
            Prediction
        """
        # Generate cache key
        cache_key = hash(features)
        
        # Check cache
        cached_prediction = self.prediction_cache.get(cache_key)
        if cached_prediction != -1:
            return cached_prediction
        
        # Run inference
        features_array = np.array([features], dtype=np.float32)
        
        prediction = self.session.run(
            [self.output_name],
            {self.input_name: features_array}
        )[0][0]
        
        # Cache result
        self.prediction_cache.put(cache_key, prediction)
        
        return prediction
    
    async def batch_predict(self, features_list):
        """
        Batch prediction for efficiency
        
        Separates cache hits from misses
        """
        predictions = {}
        cache_misses = []
        cache_miss_indices = []
        
        # Check cache
        for i, features in enumerate(features_list):
            cache_key = hash(features)
            cached = self.prediction_cache.get(cache_key)
            
            if cached != -1:
                predictions[i] = cached
            else:
                cache_misses.append(features)
                cache_miss_indices.append(i)
        
        # Batch inference for cache misses
        if cache_misses:
            features_array = np.array(cache_misses, dtype=np.float32)
            
            batch_predictions = self.session.run(
                [self.output_name],
                {self.input_name: features_array}
            )[0]
            
            # Store in cache and results
            for i, pred in zip(cache_miss_indices, batch_predictions):
                cache_key = hash(features_list[i])
                self.prediction_cache.put(cache_key, pred)
                predictions[i] = pred
        
        # Return in original order
        return [predictions[i] for i in range(len(features_list))]

# Example: Edge API server with ML inference
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# Load model at startup
edge_ml_server = EdgeMLServer('model.onnx')

@app.post("/predict")
async def predict(request: dict):
    """
    Edge prediction endpoint
    
    Returns cached or computed prediction
    """
    features = tuple(request['features'])
    
    try:
        prediction = edge_ml_server.predict(features)
        
        return {
            'prediction': float(prediction),
            'cached': edge_ml_server.prediction_cache.get(hash(features)) != -1,
            'edge_region': 'us-east-1'
        }
    except Exception as e:
        return {'error': str(e)}, 500

@app.post("/batch_predict")
async def batch_predict(request: dict):
    """
    Batch prediction endpoint
    """
    features_list = [tuple(f) for f in request['features']]
    
    predictions = await edge_ml_server.batch_predict(features_list)
    
    return {
        'predictions': [float(p) for p in predictions],
        'count': len(predictions),
        'edge_region': 'us-east-1'
    }

# Run edge server
# uvicorn.run(app, host='0.0.0.0', port=8000)
```

### Model Distribution to Edge

```python
class ModelDistributionSystem:
    """
    Distribute ML models to edge servers
    
    Challenges:
    - Large model sizes (GB)
    - Many edge locations
    - Version management
    - Atomic updates
    """
    
    def __init__(self, s3_bucket, edge_servers):
        self.s3_bucket = s3_bucket
        self.edge_servers = edge_servers
        
        # Track model versions at each edge
        self.edge_versions = {
            server.region: None
            for server in edge_servers
        }
    
    def distribute_model(self, model_path, version):
        """
        Distribute model to all edge servers
        
        Steps:
        1. Upload to S3
        2. Notify edge servers
        3. Edge servers download
        4. Edge servers validate
        5. Edge servers activate
        """
        print(f"Distributing model {version} to {len(self.edge_servers)} edge servers...")
        
        # Step 1: Upload to S3
        s3_key = f"models/{version}/model.onnx"
        self._upload_to_s3(model_path, s3_key)
        
        # Step 2: Notify edge servers
        results = []
        
        for server in self.edge_servers:
            result = self._distribute_to_edge(server, s3_key, version)
            results.append((server.region, result))
        
        # Check results
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        
        print(f"\nDistribution complete:")
        print(f"  Successful: {len(successful)}/{len(self.edge_servers)}")
        print(f"  Failed: {len(failed)}")
        
        if failed:
            print(f"  Failed regions: {[r[0] for r in failed]}")
        
        return len(failed) == 0
    
    def _upload_to_s3(self, local_path, s3_key):
        """Upload model to S3"""
        import boto3
        
        s3 = boto3.client('s3')
        
        print(f"Uploading {local_path} to s3://{self.s3_bucket}/{s3_key}")
        
        s3.upload_file(
            local_path,
            self.s3_bucket,
            s3_key,
            ExtraArgs={'ServerSideEncryption': 'AES256'}
        )
    
    def _distribute_to_edge(self, server, s3_key, version):
        """
        Notify edge server to download model
        
        Edge server will:
        1. Download from S3
        2. Validate checksum
        3. Load model
        4. Run health checks
        5. Activate (atomic swap)
        """
        try:
            import requests
            
            response = requests.post(
                f"{server.url}/admin/update_model",
                json={
                    's3_bucket': self.s3_bucket,
                    's3_key': s3_key,
                    'version': version
                },
                timeout=300  # 5 minutes for large models
            )
            
            if response.status_code == 200:
                self.edge_versions[server.region] = version
                print(f"  ✓ {server.region}: Updated to {version}")
                return True
            else:
                print(f"  ✗ {server.region}: Failed - {response.text}")
                return False
        
        except Exception as e:
            print(f"  ✗ {server.region}: Error - {e}")
            return False
    
    def rollback_model(self, target_version):
        """
        Rollback to previous model version
        
        Useful if new model has issues
        """
        print(f"Rolling back to version {target_version}...")
        
        s3_key = f"models/{target_version}/model.onnx"
        
        return self.distribute_model(f"/tmp/model_{target_version}.onnx", target_version)
    
    def get_version_status(self):
        """Get model versions deployed at each edge"""
        return self.edge_versions

# Example usage
edge_servers = [
    EdgeServerInfo('us-east-1', 'https://edge-us-east-1.example.com', {}),
    EdgeServerInfo('eu-west-1', 'https://edge-eu-west-1.example.com', {}),
    EdgeServerInfo('ap-northeast-1', 'https://edge-ap-northeast-1.example.com', {}),
]

distributor = ModelDistributionSystem(
    s3_bucket='my-ml-models',
    edge_servers=edge_servers
)

# Distribute new model
success = distributor.distribute_model('model_v3.onnx', 'v3')

if success:
    print("\nModel distribution successful!")
    print("Current versions:")
    for region, version in distributor.get_version_status().items():
        print(f"  {region}: {version}")
else:
    print("\nModel distribution failed, rolling back...")
    distributor.rollback_model('v2')
```

---

## Monitoring & Observability

### CDN Metrics Dashboard

```python
class CDNMetricsDashboard:
    """
    Aggregate and visualize CDN metrics
    
    Key metrics:
    - Cache hit rate
    - Latency (p50, p95, p99)
    - Bandwidth usage
    - Error rate
    - Request rate
    """
    
    def __init__(self, edge_servers):
        self.edge_servers = edge_servers
        
        # Time series database for metrics
        from prometheus_client import Counter, Histogram, Gauge
        
        self.request_count = Counter(
            'cdn_requests_total',
            'Total CDN requests',
            ['region', 'status']
        )
        
        self.latency = Histogram(
            'cdn_request_latency_seconds',
            'CDN request latency',
            ['region', 'cache_level']
        )
        
        self.cache_hit_rate = Gauge(
            'cdn_cache_hit_rate',
            'Cache hit rate',
            ['region', 'cache_level']
        )
    
    def collect_metrics(self):
        """
        Collect metrics from all edge servers
        
        Returns aggregated view
        """
        global_metrics = {
            'total_requests': 0,
            'total_cache_hits': 0,
            'regions': {}
        }
        
        for server in self.edge_servers:
            stats = server.metrics.get_stats()
            
            total_requests = (
                server.metrics.l1_hits +
                server.metrics.l2_hits +
                server.metrics.misses
            )
            
            total_cache_hits = server.metrics.l1_hits + server.metrics.l2_hits
            
            global_metrics['total_requests'] += total_requests
            global_metrics['total_cache_hits'] += total_cache_hits
            
            global_metrics['regions'][server.region] = {
                'requests': total_requests,
                'cache_hits': total_cache_hits,
                'stats': stats
            }
        
        # Calculate global hit rate
        if global_metrics['total_requests'] > 0:
            global_metrics['cache_hit_rate'] = (
                global_metrics['total_cache_hits'] / global_metrics['total_requests']
            )
        else:
            global_metrics['cache_hit_rate'] = 0
        
        return global_metrics
    
    def print_dashboard(self):
        """Print metrics dashboard"""
        metrics = self.collect_metrics()
        
        print("\n" + "="*70)
        print("CDN METRICS DASHBOARD")
        print("="*70)
        
        print(f"\nGlobal Metrics:")
        print(f"  Total Requests:     {metrics['total_requests']:,}")
        print(f"  Cache Hit Rate:     {metrics['cache_hit_rate']:.2%}")
        
        print(f"\nRegional Breakdown:")
        
        for region, data in metrics['regions'].items():
            print(f"\n  {region}:")
            print(f"    Requests:         {data['requests']:,}")
            print(f"    L1 Hit Rate:      {data['stats']['l1_hit_rate']:.2%}")
            print(f"    L2 Hit Rate:      {data['stats']['l2_hit_rate']:.2%}")
            print(f"    Miss Rate:        {data['stats']['miss_rate']:.2%}")
            print(f"    Avg L1 Latency:   {data['stats']['avg_l1_latency_ms']:.2f}ms")
            print(f"    Avg Miss Latency: {data['stats']['avg_miss_latency_ms']:.2f}ms")
        
        print("="*70)
    
    def plot_latency_distribution(self):
        """Plot latency distribution by region"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(self.edge_servers), 1, figsize=(12, 4 * len(self.edge_servers)))
        
        for i, server in enumerate(self.edge_servers):
            ax = axes[i] if len(self.edge_servers) > 1 else axes
            
            # Get latencies
            l1_latencies = np.array(server.metrics.l1_latencies) * 1000  # ms
            l2_latencies = np.array(server.metrics.l2_latencies) * 1000
            miss_latencies = np.array(server.metrics.miss_latencies) * 1000
            
            # Plot histograms
            ax.hist(l1_latencies, bins=50, alpha=0.5, label='L1 Cache', color='green')
            ax.hist(l2_latencies, bins=50, alpha=0.5, label='L2 Cache', color='blue')
            ax.hist(miss_latencies, bins=50, alpha=0.5, label='Origin', color='red')
            
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Latency Distribution - {server.region}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cdn_latency_distribution.png')
        plt.close()
        
        print("Latency distribution plot saved to cdn_latency_distribution.png")

# Example usage
edge_servers = [
    # ... initialize edge servers
]

dashboard = CDNMetricsDashboard(edge_servers)

# Collect and display metrics
dashboard.print_dashboard()

# Plot latency distribution
dashboard.plot_latency_distribution()
```

---

## Cost Optimization

### Tiered Caching Strategy

```python
class TieredCachingStrategy:
    """
    Optimize costs with tiered caching
    
    Tiers:
    1. Hot (L1 - Redis): Most accessed, expensive, fast
    2. Warm (L2 - Local disk): Frequently accessed, cheap, medium speed
    3. Cold (S3): Rarely accessed, cheapest, slow
    
    Move items between tiers based on access patterns
    """
    
    def __init__(self):
        self.l1_cost_per_gb_per_month = 100  # Redis
        self.l2_cost_per_gb_per_month = 10   # SSD
        self.l3_cost_per_gb_per_month = 0.02  # S3
        
        self.l1_size_gb = 10
        self.l2_size_gb = 100
        self.l3_size_gb = 1000
    
    def calculate_monthly_cost(self):
        """Calculate monthly storage cost"""
        l1_cost = self.l1_size_gb * self.l1_cost_per_gb_per_month
        l2_cost = self.l2_size_gb * self.l2_cost_per_gb_per_month
        l3_cost = self.l3_size_gb * self.l3_cost_per_gb_per_month
        
        total_cost = l1_cost + l2_cost + l3_cost
        
        return {
            'l1_cost': l1_cost,
            'l2_cost': l2_cost,
            'l3_cost': l3_cost,
            'total_cost': total_cost
        }
    
    def optimize_tier_sizes(self, access_patterns):
        """
        Optimize tier sizes based on access patterns
        
        Goal: Minimize cost while maintaining hit rate
        """
        # Analyze access frequency
        access_freq = {}
        
        for item_id, accesses in access_patterns.items():
            access_freq[item_id] = len(accesses)
        
        # Sort by frequency
        sorted_items = sorted(
            access_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Allocate to tiers
        l1_items = sorted_items[:100]  # Top 100
        l2_items = sorted_items[100:1000]  # Next 900
        l3_items = sorted_items[1000:]  # Rest
        
        print(f"Tier allocation:")
        print(f"  L1 (Hot):  {len(l1_items)} items")
        print(f"  L2 (Warm): {len(l2_items)} items")
        print(f"  L3 (Cold): {len(l3_items)} items")
        
        # Calculate expected hit rate
        total_accesses = sum(access_freq.values())
        l1_accesses = sum(freq for _, freq in l1_items)
        l2_accesses = sum(freq for _, freq in l2_items)
        
        l1_hit_rate = l1_accesses / total_accesses
        l2_hit_rate = l2_accesses / total_accesses
        
        print(f"\nExpected hit rates:")
        print(f"  L1: {l1_hit_rate:.2%}")
        print(f"  L2: {l2_hit_rate:.2%}")
        print(f"  Combined (L1+L2): {(l1_hit_rate + l2_hit_rate):.2%}")

# Example
strategy = TieredCachingStrategy()

# Calculate costs
costs = strategy.calculate_monthly_cost()
print("Monthly CDN storage costs:")
for key, value in costs.items():
    print(f"  {key}: ${value:.2f}")

# Simulate access patterns
access_patterns = {
    f"item_{i}": [time.time() - random.random() * 86400 for _ in range(random.randint(1, 100))]
    for i in range(10000)
}

# Optimize
strategy.optimize_tier_sizes(access_patterns)
```

---

## Key Takeaways

✅ **Edge caching** - Serve content close to users for low latency  
✅ **Multi-level cache** - L1 (Redis), L2 (disk), origin (database)  
✅ **Smart routing** - GeoDNS + latency-based + load-based  
✅ **Cache invalidation** - Pub/sub for real-time propagation  
✅ **Edge ML serving** - Deploy models to edge for fast inference  
✅ **Cost optimization** - Tiered storage based on access patterns  

**Key Metrics:**
- Cache hit rate: > 80%
- P99 latency: < 50ms for cache hits
- Origin latency: 200-500ms
- Bandwidth savings: 70-90%

---

**Originally published at:** [arunbaby.com/ml-system-design/0011-content-delivery-network](https://www.arunbaby.com/ml-system-design/0011-content-delivery-network/)

*If you found this helpful, consider sharing it with others who might benefit.*

