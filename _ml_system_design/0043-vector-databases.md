---
title: "Vector Databases"
day: 43
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - embeddings
  - similarity-search
  - infrastructure
  - neural-search
difficulty: Hard
---

**"The infrastructure for semantic search and AI-native applications."**

## 1. Introduction: The Rise of Embeddings

Modern ML represents everything as vectors (embeddings):
*   **Text:** BERT, GPT → 768-4096 dim vectors.
*   **Images:** CLIP, ResNet → 512-2048 dim vectors.
*   **Audio:** Wav2Vec → 768 dim vectors.
*   **Users/Items:** Collaborative filtering → 64-256 dim vectors.

**Challenge:** How do we efficiently store and search billions of these vectors?

**Solution:** Vector Databases.

## 2. What is a Vector Database?

A **vector database** is a specialized database designed for:
1.  **Storing** high-dimensional vectors with metadata.
2.  **Indexing** vectors for fast similarity search.
3.  **Querying** to find the k most similar vectors (k-NN).

**Key Operations:**
*   `insert(id, vector, metadata)`
*   `search(query_vector, k)` → top-k similar vectors
*   `filter(metadata_condition)` → filter before/after search
*   `delete(id)`
*   `update(id, new_vector)`

## 3. Why Not Traditional Databases?

**SQL/NoSQL Databases:**
*   Optimized for exact match queries (WHERE id = 123).
*   B-tree/Hash indexes don't work for high-dimensional similarity.

**Problem:** Finding k nearest neighbors in 768 dimensions is computationally expensive.

**Brute Force:** Compare query to every vector. $O(N \cdot d)$ per query. Too slow for billions of vectors.

**Solution:** Approximate Nearest Neighbor (ANN) algorithms.

## 4. Approximate Nearest Neighbor (ANN) Algorithms

### 4.1. Locality-Sensitive Hashing (LSH)

**Idea:** Hash similar vectors to the same bucket.

**Algorithm:**
1.  Define random hyperplanes.
2.  Hash: For each hyperplane, record 0/1 based on which side the vector falls.
3.  Similar vectors have the same hash with high probability.

**Pros:** Simple, works for any distance metric.
**Cons:** Requires many hash tables for high recall.

### 4.2. Inverted File Index (IVF)

**Idea:** Cluster vectors, search only relevant clusters.

**Algorithm:**
1.  **Training:** Cluster vectors (k-means) into C clusters.
2.  **Indexing:** Assign each vector to its nearest cluster.
3.  **Search:** Find the q nearest clusters to the query, search only those.

**Parameters:**
*   **nlist:** Number of clusters (C).
*   **nprobe:** Number of clusters to search (q).

**Pros:** Tunable accuracy/speed trade-off.
**Cons:** Cluster centroids must fit in memory.

### 4.3. Hierarchical Navigable Small World (HNSW)

**Idea:** Build a graph where similar vectors are connected.

**Algorithm:**
1.  **Construction:** Insert vectors one by one. Connect to nearest neighbors at multiple layers.
2.  **Search:** Start at a random node, greedily walk to the nearest neighbor. Repeat at each layer.

**Parameters:**
*   **M:** Max connections per node.
*   **efConstruction:** Search depth during construction.
*   **efSearch:** Search depth during query.

**Pros:** State-of-the-art recall/speed. Works well in high dimensions.
**Cons:** Memory-intensive (stores graph structure).

### 4.4. Product Quantization (PQ)

**Idea:** Compress vectors by quantizing subvectors.

**Algorithm:**
1.  Split vector into M subvectors.
2.  Cluster each subvector space into K centroids.
3.  Represent each vector as M centroid indices (8 bits each).

**Compression:**
*   Original: 768 floats × 4 bytes = 3KB.
*   PQ (M=96, K=256): 96 × 1 byte = 96 bytes. **30x compression**.

**Pros:** Massive memory savings.
**Cons:** Some accuracy loss.

## 5. Popular Vector Databases

### 5.1. Pinecone (Managed)

**Features:**
*   Fully managed, serverless.
*   Scales to billions of vectors.
*   Metadata filtering.
*   Hybrid search (vector + keyword).

**Use Case:** Production semantic search, recommendation systems.

### 5.2. Milvus (Open Source)

**Features:**
*   GPU acceleration.
*   Multiple index types (IVF, HNSW, PQ).
*   Distributed architecture.
*   Cloud offering (Zilliz).

**Use Case:** Large-scale similarity search, research.

### 5.3. Weaviate (Open Source)

**Features:**
*   Built-in vectorization (integrates with OpenAI, Cohere).
*   GraphQL API.
*   Hybrid search.
*   Modules for different ML models.

**Use Case:** AI-native applications, semantic search.

### 5.4. Qdrant (Open Source)

**Features:**
*   Rust-based (high performance).
*   Rich filtering.
*   Payload storage.
*   On-disk indexes.

**Use Case:** Low-latency search, edge deployment.

### 5.5. Chroma (Open Source)

**Features:**
*   Lightweight, developer-friendly.
*   In-memory or persistent.
*   LangChain integration.

**Use Case:** Prototyping, RAG applications.

### 5.6. pgvector (PostgreSQL Extension)

**Features:**
*   Adds vector support to PostgreSQL.
*   IVFFlat and HNSW indexes.
*   Familiar SQL interface.

**Use Case:** Adding vector search to existing PostgreSQL apps.

## 6. System Architecture

```
                    ┌─────────────────────┐
                    │   ML Model          │
                    │  (Embedding Gen)    │
                    └─────────┬───────────┘
                              │ vectors
                    ┌─────────▼───────────┐
                    │   Vector Database   │
                    │   (Pinecone/Milvus) │
                    └─────────┬───────────┘
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
┌────────▼────────┐                      ┌────────▼────────┐
│   Index         │                      │   Storage       │
│   (HNSW/IVF)    │                      │   (Vectors +    │
│                 │                      │    Metadata)    │
└─────────────────┘                      └─────────────────┘
```

## 7. Building a Semantic Search System

**Scenario:** Build a semantic search for a knowledge base (10M documents).

**Step 1: Embed Documents**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)  # (10M, 384)
```

**Step 2: Index in Vector DB**
```python
import pinecone

pinecone.init(api_key="...", environment="...")
index = pinecone.Index("knowledge-base")

# Upsert in batches
for i in range(0, len(documents), 100):
    batch = [
        (str(j), embeddings[j].tolist(), {"text": documents[j]})
        for j in range(i, min(i + 100, len(documents)))
    ]
    index.upsert(vectors=batch)
```

**Step 3: Query**
```python
query = "What is machine learning?"
query_embedding = model.encode([query])[0]

results = index.query(
    vector=query_embedding.tolist(),
    top_k=5,
    include_metadata=True
)

for match in results['matches']:
    print(f"Score: {match['score']}, Text: {match['metadata']['text']}")
```

## 8. Hybrid Search: Vector + Keyword

**Problem:** Pure vector search may miss keyword-specific matches.

**Example:**
*   Query: "Python Flask tutorial"
*   Vector search returns: Generic web development tutorials.
*   Keyword search returns: Exact "Flask" matches.

**Solution:** Combine both.

**Approach (Reciprocal Rank Fusion):**
1.  Get top-k from vector search.
2.  Get top-k from keyword search (BM25).
3.  Combine rankings: $\text{score} = \sum \frac{1}{k + \text{rank}}$.

**Implementation (Weaviate):**
```graphql
{
  Get {
    Document(
      hybrid: {
        query: "Python Flask tutorial"
        alpha: 0.5  # 0 = keyword only, 1 = vector only
      }
    ) {
      text
    }
  }
}
```

## 9. Filtering: Pre-filter vs Post-filter

**Pre-filter:**
*   Filter vectors by metadata, then search.
*   **Pros:** Faster (smaller search space).
*   **Cons:** May miss relevant vectors if filter is too strict.

**Post-filter:**
*   Search all vectors, then filter results.
*   **Pros:** Higher recall.
*   **Cons:** Slower (search entire index).

**Best Practice:** Use pre-filter when filter is selective, post-filter otherwise.

## 10. Scalability Considerations

### 10.1. Sharding

**Horizontal Scaling:** Distribute vectors across multiple shards.

**Strategies:**
*   **Hash-based:** Hash(id) % num_shards.
*   **Range-based:** Vectors with similar metadata go to the same shard.

### 10.2. Replication

**High Availability:** Replicate each shard.

**Read Scaling:** Distribute reads across replicas.

### 10.3. Tiered Storage

**Hot/Cold Storage:**
*   **Hot:** Frequently accessed vectors in memory/SSD.
*   **Cold:** Rarely accessed vectors on disk/object storage.

## 11. Production Case Study: Pinterest Visual Search

**Problem:** Given an image, find similar pins.

**Architecture:**
*   **Embedding Model:** CNN (ResNet-based) → 2048-dim vectors.
*   **Index:** Billion+ vectors, sharded across multiple machines.
*   **Storage:** Vectors in memory, metadata in MySQL.

**Optimizations:**
*   **PQ Compression:** 32x reduction in memory.
*   **GPU Acceleration:** Batch queries on GPU.
*   **Caching:** LRU cache for popular queries.

**Result:**
*   **Latency:** <50ms for 1B vectors.
*   **Recall:** >95% at top-10.

## 12. Production Case Study: Spotify Recommendations

**Problem:** Recommend songs based on user's listening history.

**Architecture:**
*   **User Embedding:** Average of recently played song embeddings.
*   **Song Embedding:** Learn from listening patterns (like/skip).
*   **Index:** 100M song embeddings in HNSW index.

**Query:**
*   Get user embedding.
*   Find k nearest songs.
*   Filter by user preferences (genre, language).

## 13. Interview Questions

1.  **What is a Vector Database?** Explain ANN algorithms.
2.  **HNSW vs IVF:** When would you use each?
3.  **Hybrid Search:** How do you combine vector and keyword search?
4.  **Scaling:** How do you scale a vector database to billions of vectors?
5.  **Design:** Design a semantic search for Stack Overflow.

## 14. Common Pitfalls

*   **Ignoring Dimensionality:** High dimensions (>1000) are slow. Use dimensionality reduction.
*   **No Metadata Filtering:** Leads to irrelevant results.
*   **Poor Embedding Model:** Garbage in, garbage out.
*   **Ignoring Freshness:** Stale embeddings for dynamic content.
*   **Over-relying on Recall Metrics:** Production needs latency too.

## 15. Deep Dive: Distance Metrics

**Euclidean Distance (L2):**
$$d(a, b) = \sqrt{\sum_i (a_i - b_i)^2}$$

**Cosine Similarity:**
$$\cos(a, b) = \frac{a \cdot b}{||a|| \cdot ||b||}$$

**Inner Product (Dot Product):**
$$\text{IP}(a, b) = a \cdot b$$

**Which to Use:**
*   **Normalized Vectors (unit length):** Cosine = Inner Product.
*   **Non-normalized:** Euclidean or Inner Product depending on use case.
*   **Recommendations:** Often Inner Product (higher score = more similar).

## 16. Conclusion

Vector databases are the backbone of AI-native applications. From semantic search to recommendations to RAG, the ability to efficiently search high-dimensional spaces is critical.

**Key Takeaways:**
*   **ANN Algorithms:** HNSW, IVF, PQ enable fast approximate search.
*   **Choose Wisely:** Managed (Pinecone) for simplicity, open-source (Milvus, Weaviate) for control.
*   **Hybrid Search:** Combine vector and keyword for best results.
*   **Scale:** Shard, replicate, compress.

As LLMs become more prevalent, vector databases will become as essential as relational databases are today. Master them to build the next generation of AI applications.

## 17. Deep Dive: Retrieval-Augmented Generation (RAG)

**RAG** combines vector databases with LLMs for grounded generation.

**Architecture:**
1.  **Embed Query:** Convert user query to vector.
2.  **Retrieve:** Find k most relevant documents from vector DB.
3.  **Generate:** Pass query + retrieved docs to LLM for answer.

**Example:**
```python
# Step 1: Embed query
query = "What is the capital of France?"
query_embedding = model.encode(query)

# Step 2: Retrieve from vector DB
results = index.query(vector=query_embedding, top_k=3)
context = "\n".join([r['metadata']['text'] for r in results['matches']])

# Step 3: Generate with LLM
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
answer = llm.generate(prompt)
```

**Benefits:**
*   **Grounded:** Answers based on facts, not hallucinations.
*   **Up-to-date:** Vector DB can be updated without retraining LLM.
*   **Scalable:** Works with billions of documents.

## 18. Benchmarking Vector Databases

**Benchmark: ANN-Benchmarks**

**Metrics:**
*   **Recall@k:** % of true nearest neighbors retrieved.
*   **QPS (Queries Per Second):** Throughput.
*   **Memory:** Index size.

**Results (1M vectors, 128 dims):**
| Algorithm | Recall@10 | QPS | Memory |
|-----------|-----------|-----|--------|
| HNSW | 99% | 5000 | 500MB |
| IVF-PQ | 95% | 8000 | 150MB |
| LSH | 85% | 10000 | 200MB |

**Observation:** HNSW has best recall, IVF-PQ best memory efficiency.

## 19. Implementation Guide: Building a Simple Vector Index

**Step 1: Define Data Structures**
```python
import numpy as np
from sklearn.cluster import KMeans

class IVFIndex:
    def __init__(self, n_clusters=100):
        self.n_clusters = n_clusters
        self.centroids = None
        self.inverted_lists = None
    
    def train(self, vectors):
        # K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(vectors)
        self.centroids = kmeans.cluster_centers_
        
        # Build inverted lists
        self.inverted_lists = [[] for _ in range(self.n_clusters)]
        labels = kmeans.predict(vectors)
        for i, label in enumerate(labels):
            self.inverted_lists[label].append((i, vectors[i]))
    
    def search(self, query, k=10, n_probe=10):
        # Find nearest clusters
        distances = np.linalg.norm(self.centroids - query, axis=1)
        nearest_clusters = np.argsort(distances)[:n_probe]
        
        # Search within clusters
        candidates = []
        for cluster in nearest_clusters:
            for idx, vector in self.inverted_lists[cluster]:
                dist = np.linalg.norm(vector - query)
                candidates.append((dist, idx))
        
        # Return top-k
        candidates.sort()
        return [idx for _, idx in candidates[:k]]
```

**Step 2: Test**
```python
# Generate random vectors
vectors = np.random.randn(10000, 128)
index = IVFIndex(n_clusters=100)
index.train(vectors)

# Query
query = np.random.randn(128)
results = index.search(query, k=10)
print(f"Top 10 results: {results}")
```

## 20. Advanced: Quantization for Memory Efficiency

**Scalar Quantization:**
*   Convert FP32 to INT8.
*   4x memory reduction.
*   Slight accuracy loss.

**Product Quantization (PQ):**
*   Split vector into M subvectors.
*   Quantize each subvector to K centroids.
*   Store M × log(K) bits per vector.

**Example (768-dim vector):**
*   PQ with M=96, K=256: 96 bytes (vs 3072 bytes for FP32).
*   **32x compression**.

## 21. Advanced: Multi-Vector Representations

**Problem:** A document may have multiple aspects (title, body, images).

**Solution:** Store multiple vectors per document.

**Approaches:**
1.  **Concatenation:** Combine into one long vector.
2.  **Multi-Vector Index:** Separate index for each type, aggregate results.
3.  **ColBERT:** Represent document as matrix (one vector per token).

**Use Case:** Multi-modal search (text + image).

## 22. Monitoring Vector DBs in Production

**Metrics:**
*   **Query Latency:** P50, P95, P99.
*   **Recall:** Periodic evaluation on held-out queries.
*   **Index Size:** Monitor growth.
*   **QPS:** Queries per second.

**Alerts:**
*   Latency spike (P99 > 500ms).
*   Recall drop (below threshold).
*   Index corruption (failed queries).

## 23. Cost Analysis

**Scenario:** 10M documents, 768-dim embeddings.

**Storage:**
*   Raw: 10M × 768 × 4 bytes = 30GB.
*   Compressed (PQ): 10M × 96 bytes = 1GB.

**Compute:**
| Provider | Storage | QPS | Cost/Month |
|----------|---------|-----|------------|
| Pinecone (p1) | 30GB | 1000 | $70 |
| Self-hosted (Milvus on AWS) | 30GB | 1000 | $50 |
| Self-hosted (Qdrant on GCP) | 30GB | 1000 | $60 |

## 24. Interview Deep Dive

**Q: Design a semantic search for e-commerce (10M products).**

**A:**
1.  **Embedding Model:** CLIP (text + images).
2.  **Vector DB:** Milvus with HNSW index.
3.  **Metadata:** Category, price, stock.
4.  **Hybrid Search:** Vector + filters (price < $100, category = "shoes").
5.  **Serving:** Cache popular queries, batch similar queries.

**Q: How do you handle real-time updates?**

**A:**
*   Use a vector DB that supports online updates (Pinecone, Milvus).
*   For batch updates, rebuild index periodically.

## 25. Future Trends

**1. Native LLM Integration:**
*   Vector DBs with built-in LLM calls.
*   Example: Weaviate's generative modules.

**2. Graph + Vector:**
*   Combine knowledge graphs with vector search.
*   Example: Neo4j + vector index.

**3. On-Device Vector Search:**
*   Run on mobile/edge devices.
*   Smaller models, efficient indexes.

**4. Multi-Modal Search:**
*   Text + image + audio in one query.
*   Unified embedding space.

## 26. Mastery Checklist

**Mastery Checklist:**
- [ ] Explain ANN algorithms (HNSW, IVF, PQ)
- [ ] Compare vector database options
- [ ] Implement a simple IVF index
- [ ] Build a semantic search system
- [ ] Implement hybrid search (vector + keyword)
- [ ] Understand pre-filter vs post-filter
- [ ] Design for scale (sharding, replication)
- [ ] Monitor production vector DBs
- [ ] Implement RAG with vector DB
- [ ] Cost analysis for different providers

## 27. Conclusion

Vector databases are the foundation of AI-native applications. They enable:
*   **Semantic Search:** Find by meaning, not just keywords.
*   **Recommendations:** Find similar items.
*   **RAG:** Ground LLM responses in facts.

The choice between HNSW, IVF, and PQ depends on your accuracy, speed, and memory requirements. Managed solutions like Pinecone offer simplicity, while open-source options like Milvus and Weaviate offer flexibility.

As embeddings become ubiquitous, vector databases will become as fundamental as SQL databases. Mastering them is essential for any ML engineer building production AI systems.

