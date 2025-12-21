---
title: "Recommendation System: Candidate Retrieval"
day: 1
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - recommendation-systems
  - retrieval
  - embeddings
domain: Recommendations
scale: "100M+ users, 10M+ items"
key_components: [embeddings, ann-search, caching]
companies: [Google, Meta, Netflix, YouTube]
related_dsa_day: 1
related_speech_day: 1
related_agents_day: 1
---

**How do you narrow down 10 million items to 1000 candidates in under 50ms? The art of fast retrieval at scale.**

## Introduction

Every day, you interact with recommendation systems dozens of times: YouTube suggests videos, Netflix recommends shows, Amazon suggests products, Spotify curates playlists, and Instagram fills your feed. Behind each recommendation is a sophisticated system that must:

- Search through **millions** of items in **milliseconds**
- Personalize results for **hundreds of millions** of users
- Balance relevance, diversity, and freshness
- Handle new users and new content gracefully
- Scale horizontally to serve billions of requests per day

The naive approach computing scores for all items for each user is mathematically impossible at scale. If we have 100M users and 10M items, that's 1 quadrillion (10^15) combinations to score. Even at 1 billion computations per second, this would take **11+ days per request**.

This post focuses on the **candidate generation** (or retrieval) stage: how we efficiently narrow millions of items down to hundreds of candidates that might interest a user. This is the first and most critical stage of any recommendation system, as it determines the maximum possible quality of recommendations while constraining latency and cost.

**What you'll learn:**
- Why most recommendation systems use a funnel architecture
- How embedding-based retrieval enables personalization at scale
- Approximate nearest neighbor (ANN) search algorithms
- Multiple retrieval strategies and how to combine them
- Caching patterns for sub-50ms latency
- Cold start problem solutions
- Real production architectures from YouTube, Pinterest, and Spotify

---

## Problem Definition

Design the **candidate generation stage** of a recommendation system that:

### Functional Requirements

1. **Personalized Retrieval**
   - Different candidates for each user based on their preferences
   - Not just "popular items for everyone"
   - Must capture user's interests, behavior patterns, and context

2. **Multiple Retrieval Strategies**
   - Collaborative filtering (users with similar taste)
   - Content-based filtering (items similar to what user liked)
   - Trending/popular items (what's hot right now)
   - Social signals (what friends are engaging with)

3. **Diversity**
   - Avoid filter bubbles (all items too similar)
   - Show variety of content types, topics, creators
   - Enable exploration (help users discover new interests)

4. **Freshness**
   - New items should appear within minutes of publication
   - System should adapt to changing user interests
   - Handle trending topics and viral content

5. **Cold Start Handling**
   - New users with no history
   - New items with no engagement data
   - Graceful degradation when data is sparse

### Non-Functional Requirements

1. **Latency**
   - p50 < 20ms (median request)
   - p95 < 40ms (95th percentile)
   - p99 < 50ms (99th percentile)
   - Why so strict? Candidate generation is just one stage; ranking, re-ranking, and other processing add more latency

2. **Throughput**
   - 100M daily active users
   - Assume 100 requests per user per day (feed refreshes, scrolls)
   - 10 billion requests per day
   - ~115k QPS average, ~500k QPS peak

3. **Scale**
   - 100M+ active users
   - 10M+ active items (videos, posts, products)
   - Billions of historical interactions
   - Petabytes of training data

4. **Availability**
   - 99.9% uptime (43 minutes downtime per month)
   - Graceful degradation when components fail
   - No single points of failure

5. **Cost Efficiency**
   - Minimize compute costs (GPU/CPU)
   - Optimize storage (embeddings, features)
   - Reduce data transfer (network bandwidth)

### Out of Scope (Clarify These)

- Ranking stage (scoring the 1000 candidates to get top 20)
- Re-ranking and diversity post-processing
- A/B testing infrastructure
- Training pipeline and data collection
- Content moderation and safety
- Business logic (e.g., promoted content, ads)

---

## High-Level Architecture

The recommendation system follows a **funnel architecture**:

```
10M Items
    ↓ Candidate Generation (This Post)
1000 Candidates
    ↓ Ranking (Lightweight Model)
100 Candidates
    ↓ Re-ranking (Heavy Model + Business Logic)
20 Final Results
```

**Why a funnel?**
- **Cannot score all items:** 10M items × 50ms per item = 5.8 days per request
- **Quality vs. Speed tradeoff:** Fast approximate methods first, expensive accurate methods last
- **Resource optimization:** Apply expensive computations only to promising candidates

Our focus: **10M → 1000 in < 50ms**

### Component Architecture

```
User Request
  ├─ user_id: 12345
  ├─ context: {device: mobile, time: evening, location: US-CA}
  └─ num_candidates: 1000
    ↓
┌─────────────────────────────────────────────┐
│         Feature Lookup (5ms)                 │
│  • User Embedding (Redis)                   │
│  • User Profile (Cassandra)                 │
│  • Recent Activity (Redis Stream)           │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│    Retrieval Strategies (Parallel, 30ms)    │
│  ┌────────────────┐  ┌──────────────────┐  │
│  │ Collaborative  │  │  Content-Based   │  │
│  │  Filtering     │  │    Filtering     │  │
│  │  (ANN Search)  │  │  (Tag Matching)  │  │
│  │   400 items    │  │    300 items     │  │
│  └────────────────┘  └──────────────────┘  │
│  ┌────────────────┐  ┌──────────────────┐  │
│  │   Trending     │  │     Social       │  │
│  │   (Sorted)     │  │  (Friends' Feed) │  │
│  │   200 items    │  │    100 items     │  │
│  └────────────────┘  └──────────────────┘  │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│    Merge & Deduplicate (5ms)                │
│  • Combine all sources                      │
│  • Remove duplicates                        │
│  • Basic filtering (already seen, blocked)  │
└──────────────┬──────────────────────────────┘
               ↓
     Return ~1000 candidates
```

**Latency Budget (50ms total):**
```
Feature lookup:        5ms
Retrieval (parallel): 30ms
Merge/dedup:          5ms
Network overhead:     10ms
Total:               50ms ✓
```

---

## Core Component 1: User and Item Embeddings

### What are Embeddings?

**Embeddings** are dense vector representations that capture semantic meaning in a continuous space.

**Example:**
```python
# User embedding (128 dimensions)
user_12345 = [0.23, -0.45, 0.67, ..., 0.12]  # 128 numbers

# Item embeddings
item_5678 = [0.19, -0.41, 0.72, ..., 0.15]   # Similar to user!
item_9999 = [-0.78, 0.92, -0.34, ..., -0.88]  # Very different

# Similarity = dot product
similarity = sum(u * i for u, i in zip(user_12345, item_5678))
# High similarity → good recommendation!
```

**Why embeddings work:**
- **Semantic similarity:** Similar users/items have similar vectors
- **Efficient computation:** Dot product is fast (O(d) for d dimensions)
- **Learned representations:** Neural networks learn meaningful patterns
- **Dense vs. sparse:** 128 floats vs. millions of categorical features

### Two-Tower Architecture

The most common architecture for retrieval is the **two-tower model**:

```
User Features              Item Features
  ├─ Demographics           ├─ Title/Description
  ├─ Historical Behavior    ├─ Category/Tags
  ├─ Recent Activity        ├─ Creator Info
  └─ Context               └─ Metadata
      ↓                         ↓
  ┌─────────┐             ┌─────────┐
  │  User   │             │  Item   │
  │  Tower  │             │  Tower  │
  │  (NN)   │             │  (NN)   │
  └────┬────┘             └────┬────┘
       │                       │
       └───────────┬───────────┘
                   ↓
            Dot Product
                   ↓
           Similarity Score
```

**Implementation:**

```python
import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, user_feature_dim=100, item_feature_dim=80, embedding_dim=128):
        super().__init__()
        
        # User tower: transform user features to embedding
        self.user_tower = nn.Sequential(
            nn.Linear(user_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
        
        # Item tower: transform item features to embedding
        self.item_tower = nn.Sequential(
            nn.Linear(item_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
        
        # L2 normalization layer
        self.normalize = lambda x: x / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
    
    def forward(self, user_features, item_features):
        # Generate embeddings
        user_emb = self.user_tower(user_features)  # (batch, 128)
        item_emb = self.item_tower(item_features)  # (batch, 128)
        
        # Normalize to unit vectors (cosine similarity = dot product)
        user_emb = self.normalize(user_emb)
        item_emb = self.normalize(item_emb)
        
        # Compute similarity (dot product)
        score = (user_emb * item_emb).sum(dim=1)  # (batch,)
        
        return score, user_emb, item_emb
    
    def get_user_embedding(self, user_features):
        """Get just the user embedding (for serving)"""
        with torch.no_grad():
            user_emb = self.user_tower(user_features)
            user_emb = self.normalize(user_emb)
        return user_emb
    
    def get_item_embedding(self, item_features):
        """Get just the item embedding (for indexing)"""
        with torch.no_grad():
            item_emb = self.item_tower(item_features)
            item_emb = self.normalize(item_emb)
        return item_emb
```

### Training the Model

**Training Data:**
- **Positive examples:** (user, item) pairs where user engaged with item (click, watch, purchase)
- **Negative examples:** (user, item) pairs where user didn't engage

**Loss Function:**

```python
def contrastive_loss(positive_scores, negative_scores, margin=0.5):
    """
    Encourage positive pairs to have high scores,
    negative pairs to have low scores
    """
    # Positive examples should have score > 0
    positive_loss = torch.relu(margin - positive_scores).mean()
    
    # Negative examples should have score < 0
    negative_loss = torch.relu(margin + negative_scores).mean()
    
    return positive_loss + negative_loss


def triplet_loss(anchor_emb, positive_emb, negative_emb, margin=0.5):
    """
    Distance to positive should be less than distance to negative
    """
    pos_distance = torch.norm(anchor_emb - positive_emb, dim=1)
    neg_distance = torch.norm(anchor_emb - negative_emb, dim=1)
    
    loss = torch.relu(pos_distance - neg_distance + margin)
    return loss.mean()


def batch_softmax_loss(user_emb, item_emb_positive, item_emb_negatives):
    """
    Treat as multi-class classification: which item did user engage with?
    
    user_emb: (batch, dim)
    item_emb_positive: (batch, dim)
    item_emb_negatives: (batch, num_negatives, dim)
    """
    # Positive score
    pos_score = (user_emb * item_emb_positive).sum(dim=1)  # (batch,)
    
    # Negative scores
    # user_emb: (batch, 1, dim), item_emb_negatives: (batch, num_neg, dim)
    neg_scores = torch.bmm(
        item_emb_negatives, 
        user_emb.unsqueeze(-1)
    ).squeeze(-1)  # (batch, num_neg)
    
    # Concatenate: first column is positive, rest are negatives
    all_scores = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)  # (batch, 1+num_neg)
    
    # Target: index 0 (positive item)
    targets = torch.zeros(all_scores.size(0), dtype=torch.long, device=all_scores.device)
    
    # Cross-entropy loss
    loss = nn.CrossEntropyLoss()(all_scores, targets)
    return loss
```

**Training Loop:**

```python
def train_two_tower_model(model, train_loader, num_epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Unpack batch
            user_features = batch['user_features']
            positive_item_features = batch['positive_item_features']
            negative_item_features = batch['negative_item_features']  # (batch, num_neg, dim)
            
            # Forward pass
            _, user_emb, pos_item_emb = model(user_features, positive_item_features)
            
            # Get negative embeddings
            batch_size, num_negatives, feature_dim = negative_item_features.shape
            neg_item_features_flat = negative_item_features.view(-1, feature_dim)
            neg_item_emb_flat = model.get_item_embedding(neg_item_features_flat)
            neg_item_emb = neg_item_emb_flat.view(batch_size, num_negatives, -1)
            
            # Compute loss
            loss = batch_softmax_loss(user_emb, pos_item_emb, neg_item_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model
```

**Negative Sampling Strategies:**

1. **Random Negatives:** Sample random items user didn't interact with
   - Pro: Simple, covers broad space
   - Con: Often too easy (user clearly not interested)

2. **Hard Negatives:** Sample items user almost engaged with (scrolled past, clicked but didn't purchase)
   - Pro: More informative, improves model discrimination
   - Con: Harder to obtain, may need separate model to identify

3. **Batch Negatives:** Use positive items from other users in batch as negatives
   - Pro: No additional sampling needed, efficient
   - Con: Not truly negative (another user liked it)

4. **Mixed Strategy:** Combine all three
   ```python
   negatives = []
   negatives.extend(sample_random(user, k=10))
   negatives.extend(sample_hard(user, k=5))
   negatives.extend(batch_negatives(batch, exclude=user))
   ```

### Why Two-Tower Works

**Key advantage:** User and item embeddings are **decoupled**.

```
Traditional approach:
  user × item → score
  Problem: Need to compute for all 10M items online

Two-tower approach:
  user → user_embedding (online, 1ms)
  item → item_embedding (offline, precompute for all items)
  Retrieval: Find items with embeddings similar to user_embedding (ANN, 20ms)
```

**Precomputation:**
```python
# Offline: Compute all item embeddings once
all_item_embeddings = {}
for item in all_items:
    item_features = get_item_features(item.id)
    item_emb = model.get_item_embedding(item_features)
    all_item_embeddings[item.id] = item_emb

# Online: Just compute user embedding and search
user_features = get_user_features(user_id)
user_emb = model.get_user_embedding(user_features)
similar_item_ids = ann_search(user_emb, all_item_embeddings, k=400)
```

---

## Core Component 2: Approximate Nearest Neighbor (ANN) Search

### The Problem

Given a user embedding, find the top-k items with most similar embeddings.

**Naive approach (exact search):**
```python
def exact_nearest_neighbors(query, all_embeddings, k=1000):
    similarities = []
    for item_id, item_emb in all_embeddings.items():
        similarity = dot_product(query, item_emb)
        similarities.append((item_id, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]
```

**Problem:** O(n) where n = 10M items
- 10M dot products × 128 dimensions = 1.28B operations
- At 1B ops/sec: 1.28 seconds per query
- Way too slow for 50ms latency target!

### Approximate Nearest Neighbor (ANN)

**Trade accuracy for speed:** Find items that are *approximately* nearest, not *exactly* nearest.

**Typical tradeoff:**
- Exact search: 100% recall, 1000ms latency
- ANN search: 95% recall, 20ms latency

**Key algorithms:**
1. **HNSW** (Hierarchical Navigable Small World) - Best overall
2. **ScaNN** (Google) - Excellent for large scale
3. **FAISS** (Facebook) - Multiple algorithms, well-optimized
4. **Annoy** (Spotify) - Simple, good for smaller datasets

### HNSW (Hierarchical Navigable Small World)

**Core idea:** Build a multi-layer graph where:
- Top layers: Long-range connections (coarse search)
- Bottom layers: Short-range connections (fine search)

**Visualization:**
```
Layer 2: •─────────────•        (Sparse, long jumps)

Layer 1: •──•──•────•──•──•     (Medium density)

Layer 0: •─•─•─•─•─•─•─•─•─•    (Dense, precise)
```

**Search algorithm:**
1. Start at top layer
2. Greedily move to closest neighbor
3. When can't improve, descend to lower layer
4. Repeat until bottom layer
5. Return k nearest neighbors

**Implementation with FAISS:**

```python
import faiss
import numpy as np

class HNSWIndex:
    def __init__(self, dimension=128, M=32, ef_construction=200):
        """
        Args:
            dimension: Embedding dimension
            M: Number of bi-directional links per layer (higher = more accurate, more memory)
            ef_construction: Size of dynamic candidate list during construction (higher = better quality, slower build)
        """
        self.dimension = dimension
        self.index = faiss.IndexHNSWFlat(dimension, M)
        self.index.hnsw.efConstruction = ef_construction
        self.item_ids = []
    
    def add(self, item_ids, embeddings):
        """
        Add items to index
        
        Args:
            item_ids: List of item IDs
            embeddings: numpy array of shape (n, dimension)
        """
        # FAISS requires float32
        embeddings = embeddings.astype('float32')
        
        # Add to index
        self.index.add(embeddings)
        self.item_ids.extend(item_ids)
        
        print(f"Index now contains {self.index.ntotal} items")
    
    def search(self, query_embedding, k=1000, ef_search=100):
        """
        Search for k nearest neighbors
        
        Args:
            query_embedding: numpy array of shape (dimension,) or (1, dimension)
            k: Number of neighbors to return
            ef_search: Size of dynamic candidate list during search (higher = more accurate, slower)
        
        Returns:
            item_ids: List of k item IDs
            distances: List of k distances
        """
        # Set search parameter
        self.index.hnsw.efSearch = ef_search
        
        # Reshape query
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Map indices to item IDs
        item_ids = [self.item_ids[idx] for idx in indices[0]]
        
        return item_ids, distances[0]
    
    def save(self, filepath):
        """Save index to disk"""
        faiss.write_index(self.index, filepath)
    
    def load(self, filepath):
        """Load index from disk"""
        self.index = faiss.read_index(filepath)

# Usage
index = HNSWIndex(dimension=128, M=32, ef_construction=200)

# Build index offline
item_embeddings = get_all_item_embeddings()  # Shape: (10M, 128)
item_ids = list(range(10_000_000))
index.add(item_ids, item_embeddings)
index.save("item_index.faiss")

# Search online
user_embedding = get_user_embedding(user_id)  # Shape: (128,)
candidate_ids, distances = index.search(user_embedding, k=400, ef_search=100)
# ~20ms for 10M items!
```

### Parameter Tuning

**Build-time parameters (offline):**

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| M | Connections per node | 16-64 (32 is good default) |
| ef_construction | Build quality | 200-400 for production |

**Search-time parameters (online):**

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| ef_search | Search quality | 1.5-2× k for good recall |

**Tuning process:**
```python
def tune_ann_parameters(index, queries, ground_truth, k=1000):
    """
    Find optimal ef_search that balances recall and latency
    """
    results = []
    
    for ef_search in [50, 100, 200, 400, 800]:
        start_time = time.time()
        recalls = []
        
        for query, truth in zip(queries, ground_truth):
            results_ids, _ = index.search(query, k=k, ef_search=ef_search)
            results_set = set(results_ids)
            truth_set = set(truth)
            recall = len(results_set & truth_set) / len(truth_set)
            recalls.append(recall)
        
        avg_recall = np.mean(recalls)
        latency = (time.time() - start_time) / len(queries) * 1000  # ms
        
        results.append({
            'ef_search': ef_search,
            'recall': avg_recall,
            'latency_ms': latency
        })
        
        print(f"ef_search={ef_search}: recall={avg_recall:.3f}, latency={latency:.1f}ms")
    
    return results

# Example output:
# ef_search=50:  recall=0.850, latency=12.3ms
# ef_search=100: recall=0.920, latency=18.7ms  ← Good balance
# ef_search=200: recall=0.960, latency=31.2ms
# ef_search=400: recall=0.985, latency=54.8ms  ← Diminishing returns
```

**Production choice:** ef_search=100 gives 92% recall @ 20ms

### Alternative: Product Quantization

For even larger scale, use **product quantization** to compress embeddings:

```python
# Reduce memory footprint: 128 floats (512 bytes) → 64 bytes
# 10M items: 5GB → 640MB

index = faiss.IndexIVFPQ(
    faiss.IndexFlatL2(dimension),
    dimension,
    nlist=1000,      # Number of clusters
    M=64,            # Number of subquantizers
    nbits=8          # Bits per subquantizer
)

# Train quantizer
index.train(training_embeddings)

# Add items
index.add(item_embeddings)

# Search (slightly less accurate, much more memory-efficient)
distances, indices = index.search(query, k=400)
```

---

## Core Component 3: Multiple Retrieval Strategies

Relying on a single retrieval method limits quality. **Diversify sources:**

### Strategy 1: Collaborative Filtering (40% of candidates)

**Idea:** "Users who liked X also liked Y"

```python
def collaborative_filtering_retrieval(user_id, k=400):
    # Get user embedding
    user_emb = get_user_embedding(user_id)
    
    # ANN search in item embedding space
    candidate_ids = ann_index.search(user_emb, k=k)
    
    return candidate_ids
```

**Pros:**
- Captures implicit patterns
- Discovers non-obvious connections
- Scales well with data

**Cons:**
- Cold start for new users/items
- Popularity bias (recommends popular items disproportionately)

### Strategy 2: Content-Based Filtering (30% of candidates)

**Idea:** Recommend items similar to what user liked before

```python
def content_based_retrieval(user_id, k=300):
    # Get user's liked items
    liked_items = get_user_history(user_id, limit=50)
    
    # For each liked item, find similar items
    candidates = set()
    for item_id in liked_items:
        # Find items with similar tags, categories, creators
        similar = find_similar_content(item_id, k=10)
        candidates.update(similar)
        
        if len(candidates) >= k:
            break
    
    return list(candidates)[:k]

def find_similar_content(item_id, k=10):
    item = get_item(item_id)
    
    # Match by tags
    similar_by_tags = query_database(
        f"SELECT item_id FROM items WHERE tags && {item.tags} ORDER BY similarity DESC LIMIT {k}"
    )
    
    return similar_by_tags
```

**Pros:**
- Explainable ("because you liked X")
- Works for new users with stated preferences
- No popularity bias

**Cons:**
- Limited discovery (filter bubble)
- Requires good item metadata
- May over-specialize

### Strategy 3: Trending (20% of candidates)

**Idea:** What's popular right now

```python
def trending_retrieval(k=200, time_window_hours=24):
    # Redis sorted set by engagement score
    trending_items = redis.zrevrange(
        f"trending:{time_window_hours}h",
        start=0,
        end=k-1,
        withscores=True
    )
    
    return [item_id for item_id, score in trending_items]

def update_trending_scores():
    """Background job runs every 5 minutes"""
    now = time.time()
    window = 24 * 3600  # 24 hours
    
    for item_id, engagement_data in recent_engagements():
        # Weighted by recency and engagement type
        score = (
            engagement_data['views'] * 1.0 +
            engagement_data['clicks'] * 2.0 +
            engagement_data['likes'] * 3.0 +
            engagement_data['shares'] * 5.0
        ) * math.exp(-(now - engagement_data['timestamp']) / (6 * 3600))  # Decay over 6 hours
        
        redis.zadd(f"trending:24h", {item_id: score})
```

**Pros:**
- Discovers viral content
- No cold start
- High CTR (users like trending items)

**Cons:**
- Same for all users (not personalized)
- Can amplify low-quality viral content
- Rich-get-richer effect

### Strategy 4: Social (10% of candidates)

**Idea:** What are my friends engaging with

```python
def social_retrieval(user_id, k=100):
    # Get user's friends
    friends = get_friends(user_id, limit=100)
    
    # Get their recent activity
    recent_engagements = {}
    for friend_id in friends:
        activities = get_recent_activities(friend_id, hours=24, limit=10)
        for activity in activities:
            item_id = activity['item_id']
            recent_engagements[item_id] = recent_engagements.get(item_id, 0) + 1
    
    # Sort by frequency
    sorted_items = sorted(
        recent_engagements.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [item_id for item_id, count in sorted_items[:k]]
```

**Pros:**
- Highly relevant (social proof)
- Encourages engagement/sharing
- Natural diversity

**Cons:**
- Requires social graph
- Privacy concerns
- Cold start for users with few friends

### Merging Strategies

```python
def retrieve_candidates(user_id, total_k=1000):
    # Run all strategies in parallel
    with ThreadPoolExecutor() as executor:
        cf_future = executor.submit(collaborative_filtering_retrieval, user_id, k=400)
        cb_future = executor.submit(content_based_retrieval, user_id, k=300)
        tr_future = executor.submit(trending_retrieval, k=200)
        sc_future = executor.submit(social_retrieval, user_id, k=100)
        
        # Wait for all to complete
        cf_candidates = cf_future.result()
        cb_candidates = cb_future.result()
        tr_candidates = tr_future.result()
        sc_candidates = sc_future.result()
    
    # Merge and deduplicate
    all_candidates = []
    seen = set()
    
    for candidate in cf_candidates + cb_candidates + tr_candidates + sc_candidates:
        if candidate not in seen:
            all_candidates.append(candidate)
            seen.add(candidate)
        
        if len(all_candidates) >= total_k:
            break
    
    return all_candidates
```

**Weighting sources:**
Instead of fixed counts, use probability-based sampling:

```python
def weighted_merge(sources, weights, total_k=1000):
    """
    sources: {
        'cf': [item1, item2, ...],
        'cb': [item3, item4, ...],
        ...
    }
    weights: {'cf': 0.4, 'cb': 0.3, 'tr': 0.2, 'sc': 0.1}
    """
    merged = []
    seen = set()
    
    # For each position, sample a source based on weights
    for _ in range(total_k * 2):  # Oversample to account for duplicates
        # Sample source
        source = np.random.choice(
            list(weights.keys()),
            p=list(weights.values())
        )
        
        # Pop next item from that source
        if sources[source]:
            item = sources[source].pop(0)
            if item not in seen:
                merged.append(item)
                seen.add(item)
        
        if len(merged) >= total_k:
            break
    
    return merged
```

---

## Core Component 4: Caching Strategy

To achieve < 50ms latency, aggressive caching is essential.

### Three-Level Cache Architecture

```
Request
  ↓
L1: Candidate Cache (Redis, TTL=5min)
  ├─ Hit → Return cached candidates (5ms)
  └─ Miss ↓
L2: User Embedding Cache (Redis, TTL=1hour)
  ├─ Hit → Skip embedding computation (3ms saved)
  └─ Miss ↓
L3: Precomputed Candidates (Redis, TTL=10min, top 10% users only)
  ├─ Hit → Return precomputed (2ms)
  └─ Miss → Full computation (40ms)
```

### Implementation

```python
class CandidateCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        
        # TTLs
        self.candidate_ttl = 300  # 5 minutes
        self.embedding_ttl = 3600  # 1 hour
        self.precomputed_ttl = 600  # 10 minutes
    
    def get_candidates(self, user_id, k=1000):
        """
        Try L1 → L2 → L3 → Compute
        """
        # L1: Candidate cache
        cache_key = f"candidates:{user_id}:{k}"
        cached = self.redis.get(cache_key)
        if cached:
            print("[L1 HIT] Returning cached candidates")
            return json.loads(cached)
        
        # L2: Embedding cache
        emb_key = f"user_emb:{user_id}"
        user_emb_cached = self.redis.get(emb_key)
        
        if user_emb_cached:
            print("[L2 HIT] Using cached embedding")
            user_emb = np.frombuffer(user_emb_cached, dtype=np.float32)
        else:
            print("[L2 MISS] Computing embedding")
            user_features = get_user_features(user_id)
            user_emb = compute_user_embedding(user_features)
            # Cache embedding
            self.redis.setex(emb_key, self.embedding_ttl, user_emb.tobytes())
        
        # Retrieve candidates
        candidates = retrieve_candidates_with_embedding(user_emb, k)
        
        # Cache candidates
        self.redis.setex(cache_key, self.candidate_ttl, json.dumps(candidates))
        
        return candidates
    
    def precompute_for_active_users(self, user_ids):
        """
        Background job: precompute candidates for top 10% active users
        Runs every 10 minutes
        """
        for user_id in user_ids:
            candidates = self.get_candidates(user_id)
            
            precomp_key = f"precomputed:{user_id}"
            self.redis.setex(
                precomp_key,
                self.precomputed_ttl,
                json.dumps(candidates)
            )
        
        print(f"Precomputed candidates for {len(user_ids)} active users")
```

### Cache Warming Strategy

```python
def identify_active_users(lookback_hours=24):
    """
    Find top 10% active users for precomputation
    """
    # Query analytics database
    query = f"""
    SELECT user_id, COUNT(*) as activity_count
    FROM user_activities
    WHERE timestamp > NOW() - INTERVAL '{lookback_hours}' HOUR
    GROUP BY user_id
    ORDER BY activity_count DESC
    LIMIT {int(total_users * 0.1)}
    """
    
    active_users = execute_query(query)
    return [row['user_id'] for row in active_users]

def warm_cache_scheduler():
    """
    Runs every 10 minutes
    """
    while True:
        active_users = identify_active_users()
        cache.precompute_for_active_users(active_users)
        
        time.sleep(600)  # 10 minutes
```

### Cache Invalidation

**Problem:** When should we invalidate cached candidates?

**Triggers:**
1. **User action:** User engages with item → invalidate their candidates
2. **Time-based:** Fixed TTL (5 minutes)
3. **New item published:** Invalidate trending cache
4. **Model update:** Invalidate all embeddings and candidates

```python
def on_user_engagement(user_id, item_id, action):
    """
    Called when user clicks/likes/shares item
    """
    # Invalidate candidate cache (stale now)
    # Redis DEL does not support globs; use SCAN + DEL for safety
    cursor = 0
    pattern = f"candidates:{user_id}:*"
    while True:
        cursor, keys = redis.scan(cursor=cursor, match=pattern, count=1000)
        if keys:
            redis.delete(*keys)
        if cursor == 0:
            break
    
    # Don't invalidate embedding cache (more stable)
    # Will naturally expire after 1 hour
    
    # Log event for retraining
    log_engagement_event(user_id, item_id, action)
```

### Cache Hit Rate Monitoring

```python
class CacheMetrics:
    def __init__(self):
        self.hits = {'L1': 0, 'L2': 0, 'L3': 0}
        self.misses = {'L1': 0, 'L2': 0, 'L3': 0}
    
    def record_hit(self, level):
        self.hits[level] += 1
    
    def record_miss(self, level):
        self.misses[level] += 1
    
    def get_stats(self):
        stats = {}
        for level in ['L1', 'L2', 'L3']:
            total = self.hits[level] + self.misses[level]
            hit_rate = self.hits[level] / total if total > 0 else 0
            stats[level] = {
                'hit_rate': hit_rate,
                'hits': self.hits[level],
                'misses': self.misses[level]
            }
        return stats

# Expected hit rates:
# L1 (candidates): 60-70% (users refresh feed multiple times)
# L2 (embeddings): 80-90% (embeddings stable for ~1 hour)
# L3 (precomputed): 10-15% (only for top 10% users)
```

---

## Handling Cold Start

### New User Problem

**Challenge:** User with no history → no personalization signals

**Solution Hierarchy:**

**Level 1: Onboarding Survey**
```python
def handle_new_user_onboarding(user_id, selected_interests):
    """
    User selects 3-5 interests during signup
    """
    # Map interests to item tags
    interest_tags = map_interests_to_tags(selected_interests)
    
    # Find items matching these tags
    candidates = query_items_by_tags(interest_tags, k=1000)
    
    # Cache for fast retrieval
    redis.setex(f"new_user_candidates:{user_id}", 3600, json.dumps(candidates))
    
    return candidates
```

**Level 2: Demographic-based Defaults**
```python
def get_demographic_defaults(user_id):
    user = get_user_profile(user_id)
    
    # Lookup popular items for this demographic
    cache_key = f"popular_items:{user.age_group}:{user.location}:{user.language}"
    
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Query most popular items for similar users
    popular = query_popular_items(
        age_group=user.age_group,
        location=user.location,
        language=user.language,
        k=1000
    )
    
    redis.setex(cache_key, 3600, json.dumps(popular))
    return popular
```

**Level 3: Explore-Heavy Mix**
```python
def new_user_retrieval(user_id):
    """
    For new users, use more exploration
    """
    # 50% popular items (safe choices)
    popular = get_popular_items(k=500)
    
    # 30% based on stated interests
    interests = get_user_interests(user_id)
    interest_based = get_items_by_interests(interests, k=300)
    
    # 20% random exploration
    random_items = sample_random_items(k=200)
    
    return merge_and_shuffle(popular, interest_based, random_items)
```

**Rapid Learning:**
```python
def update_new_user_preferences(user_id, engagement):
    """
    Weight early engagements heavily to quickly build profile
    """
    engagement_count = get_engagement_count(user_id)
    
    if engagement_count < 10:
        # First 10 engagements: 5x weight
        weight = 5.0
    elif engagement_count < 50:
        # Next 40 engagements: 2x weight
        weight = 2.0
    else:
        # Normal weight
        weight = 1.0
    
    update_user_profile(user_id, engagement, weight=weight)
```

### New Item Problem

**Challenge:** Item with no engagement history → no collaborative signal

**Solution 1: Content-Based Features**
```python
def get_new_item_candidates_for_users(item_id):
    """
    Find users who might like this new item based on content
    """
    item = get_item(item_id)
    
    # Extract content features
    tags = item.tags
    category = item.category
    creator = item.creator_id
    
    # Find users interested in these features
    candidate_users = []
    
    # Users who liked similar tags
    candidate_users.extend(
        get_users_by_tag_preferences(tags, k=10000)
    )
    
    # Users who follow this creator
    candidate_users.extend(
        get_creator_followers(creator)
    )
    
    return list(set(candidate_users))
```

**Solution 2: Small-Scale Exploration**
```python
def bootstrap_new_item(item_id):
    """
    Show new item to small random sample to gather initial signals
    """
    # Sample 1% of users randomly
    sample_size = int(total_users * 0.01)
    sampled_users = random.sample(all_users, sample_size)
    
    # Add this item to their candidate pools with high position
    for user_id in sampled_users:
        inject_item_into_candidates(user_id, item_id, position=50)
    
    # Monitor for 1 hour
    # If engagement rate > threshold, continue showing
    # If engagement rate < threshold, reduce exposure
```

**Solution 3: Multi-Armed Bandit**
```python
class ThompsonSamplingBandit:
    """
    Balance exploration (new items) vs exploitation (proven items)
    """
    def __init__(self):
        self.successes = {}  # item_id -> success count
        self.failures = {}   # item_id -> failure count
    
    def select_item(self, candidate_items, k=20):
        """
        Sample items based on estimated CTR with uncertainty
        """
        selected = []
        
        for item_id in candidate_items:
            alpha = self.successes.get(item_id, 1)  # Prior: 1 success
            beta = self.failures.get(item_id, 1)     # Prior: 1 failure
            
            # Sample from Beta distribution
            theta = np.random.beta(alpha, beta)
            
            selected.append((item_id, theta))
        
        # Sort by sampled theta and return top k
        selected.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in selected[:k]]
    
    def update(self, item_id, success):
        """
        Update counts after showing item
        """
        if success:
            self.successes[item_id] = self.successes.get(item_id, 0) + 1
        else:
            self.failures[item_id] = self.failures.get(item_id, 0) + 1
```

---

## Real-World Examples

### YouTube Recommendations

**Architecture (circa 2016):**
- Two-stage: Candidate generation → Ranking
- Candidate generation: Deep neural network with collaborative filtering
- Features: Watch history, search history, demographics
- 800k candidates → Hundreds for ranking
- Uses TensorFlow for training

**Key innovations:**
- "Example age" feature (prefer fresh content)
- Normalized watch time (account for video length)
- Asymmetric co-watch (A→B doesn't mean B→A)

### Pinterest (PinSage)

**Architecture:**
- Graph neural network (GNN) on Pin-Board graph
- 3 billion nodes, 18 billion edges
- Random walk sampling for neighborhoods
- Two-tower model: Pin embeddings, User embeddings
- Production deployment on GPUs

**Key innovations:**
- Importance pooling (weight neighbors by importance)
- Hard negative sampling (visually similar but topically different)
- Multi-task learning (save, click, hide)

### Spotify Recommendations

**Architecture:**
- Collaborative filtering (matrix factorization)
- Content-based (audio features via CNNs)
- Natural language processing (playlist names, song metadata)
- Reinforcement learning (sequential recommendations)

**Key innovations:**
- Audio embedding from raw waveforms
- Contextual bandits for playlist curation
- Session-based recommendations

---

## Monitoring and Evaluation

### Online Metrics

**User Engagement:**
- Click-through rate (CTR)
- Watch time / Dwell time
- Like / Share rate
- Session length
- Return rate (DAU / MAU)

**Diversity Metrics:**
- Intra-list diversity (avg pairwise distance)
- Coverage (% of catalog recommended)
- Concentration (Gini coefficient)

**System Metrics:**
- Candidate generation latency (p50, p95, p99)
- Cache hit rates (L1, L2, L3)
- ANN recall@k
- QPS per server

### Offline Metrics

**Retrieval Quality:**
```python
def evaluate_retrieval(model, test_set):
    """
    Evaluate on held-out test set
    """
    recalls = []
    precisions = []
    
    for user_id, ground_truth_items in test_set:
        # Generate candidates
        candidates = retrieve_candidates(user_id, k=1000)
        
        # Recall: What % of ground truth items were retrieved?
        recall = len(set(candidates) & set(ground_truth_items)) / len(ground_truth_items)
        recalls.append(recall)
        
        # Precision: What % of candidates are relevant?
        precision = len(set(candidates) & set(ground_truth_items)) / len(candidates)
        precisions.append(precision)
    
    print(f"Recall@1000: {np.mean(recalls):.3f}")
    print(f"Precision@1000: {np.mean(precisions):.3f}")
```

**Target: Recall@1000 > 0.90** (retrieve 90% of items user would engage with)

### A/B Testing

```python
class ABExperiment:
    def __init__(self, name, control_config, treatment_config, traffic_split=0.05):
        self.name = name
        self.control = control_config
        self.treatment = treatment_config
        self.traffic_split = traffic_split
    
    def assign_variant(self, user_id):
        """
        Consistent hashing for stable assignment
        """
        hash_val = hashlib.md5(f"{user_id}:{self.name}".encode()).hexdigest()
        hash_int = int(hash_val, 16)
        
        if (hash_int % 100) < (self.traffic_split * 100):
            return 'treatment'
        return 'control'
    
    def get_config(self, user_id):
        variant = self.assign_variant(user_id)
        return self.treatment if variant == 'treatment' else self.control

# Example: Test new retrieval mix
experiment = ABExperiment(
    name="retrieval_mix_v2",
    control_config={'cf': 0.4, 'cb': 0.3, 'tr': 0.2, 'sc': 0.1},
    treatment_config={'cf': 0.5, 'cb': 0.2, 'tr': 0.2, 'sc': 0.1},  # More CF, less CB
    traffic_split=0.05  # 5% treatment, 95% control
)

# Usage
config = experiment.get_config(user_id)
candidates = retrieve_with_mix(user_id, weights=config)

# Measure:
# - CTR improvement: +2.3% ✓
# - Diversity: -1.2% (acceptable)
# - Latency: No change
# Decision: Ship to 100%
```

---

## Key Takeaways

✅ **Funnel architecture** (millions → thousands → dozens) is essential for scale  
✅ **Two-tower models** decouple user/item embeddings for efficient retrieval  
✅ **ANN search** (HNSW, ScaNN) provides 95%+ recall @ 20ms vs 1000ms exact search  
✅ **Multiple retrieval strategies** (CF, content, trending, social) improve diversity  
✅ **Aggressive caching** (3-level) achieves sub-50ms latency  
✅ **Cold start** requires explicit strategies (onboarding, demographics, exploration)  
✅ **Monitoring** both online metrics (CTR, diversity) and offline metrics (recall@k)

---

## Further Reading

**Papers:**
- [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/pub45530/)
- [PinSage: Graph Convolutional Neural Networks](https://arxiv.org/abs/1806.01973)
- [HNSW: Efficient and Robust Approximate Nearest Neighbor Search](https://arxiv.org/abs/1603.09320)

**Libraries:**
- [Faiss (Facebook)](https://github.com/facebookresearch/faiss)
- [ScaNN (Google)](https://github.com/google-research/google-research/tree/master/scann)
- [Annoy (Spotify)](https://github.com/spotify/annoy)

**Books:**
- *Recommender Systems Handbook* (Ricci et al.)
- *Practical Recommender Systems* (Kim Falk)

**Courses:**
- [Stanford CS246: Mining Massive Datasets](http://web.stanford.edu/class/cs246/)
- [RecSys Conference Tutorials](https://recsys.acm.org/)

---

## Conclusion

Recommendation systems are one of the most impactful applications of machine learning, directly affecting user experience for billions of people daily. The candidate generation stage is where the magic begins efficiently narrowing millions of possibilities to a manageable set of high-quality candidates.

The key insights:
1. **Embeddings** capture semantic similarity in continuous space
2. **ANN search** makes similarity search practical at scale
3. **Diversity** in retrieval strategies prevents filter bubbles
4. **Caching** is not optional it's essential for latency
5. **Cold start** requires thoughtful product and engineering solutions

As you build recommendation systems, remember: the best system balances multiple objectives (relevance, diversity, freshness, serendipity) while maintaining the strict latency and cost constraints of production environments.

Now go build something that helps users discover content they'll love! 🚀

---

**Originally published at:** [arunbaby.com/ml-system-design/0001-recommendation-system](https://www.arunbaby.com/ml-system-design/0001-recommendation-system/)

*If you found this helpful, consider sharing it with others who might benefit.*
