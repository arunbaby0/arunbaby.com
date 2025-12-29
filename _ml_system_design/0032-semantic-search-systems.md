---
title: "Semantic Search Systems"
day: 32
related_dsa_day: 32
related_speech_day: 32
related_agents_day: 32
collection: ml_system_design
categories:
 - ml_system_design
tags:
 - search
 - nlp
 - embeddings
 - vector-db
subdomain: "Information Retrieval"
tech_stack: [Elasticsearch, Milvus, Pinecone, BERT, Faiss]
scale: "Billions of documents, <100ms latency"
companies: [Google, Amazon, Spotify, Notion]
---

**"Moving beyond keywords to understand the *meaning* of a query."**

## 1. The Problem: Keyword Search is Not Enough

Traditional search engines (like Lucene/Elasticsearch) rely on **Lexical Search** (BM25/TF-IDF). They match exact words or their stems.

**Failure Case:**
- Query: "How to fix a flat tire"
- Document: "Guide to repairing a punctured wheel"
- **Result:** No match! (No shared words: fix/repair, flat/punctured, tire/wheel).

**Semantic Search** solves this by mapping queries and documents to a **vector space** where similar meanings are close together. It captures:
- **Synonyms:** "car" ≈ "automobile"
- **Polysemy:** "bank" (river) vs "bank" (money)
- **Context:** "Apple" (fruit) vs "Apple" (company)

## 2. Dense Retrieval (Bi-Encoders)

**Core Idea:** Use a Deep Learning model (Transformer) to convert text into a fixed-size vector (embedding).

**Architecture:**
``
Query "fix flat tire" → [BERT] → Vector Q (768-dim)
 ↓
 Similarity (Dot Product)
 ↑
Doc "repair wheel" → [BERT] → Vector D (768-dim)
``

**Bi-Encoder (Siamese Network):**
- Two identical BERT models (sharing weights).
- Process Query and Document independently.
- **Fast Retrieval:** Pre-compute all document vectors. At query time, only encode the query.

``python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. Indexing (Offline)
docs = ["Guide to repairing a punctured wheel", "Best pizza in NY"]
doc_embeddings = model.encode(docs) # [2, 384]

# 2. Search (Online)
query = "How to fix a flat tire"
query_embedding = model.encode(query) # [1, 384]

# 3. Similarity
import numpy as np
scores = np.dot(doc_embeddings, query_embedding)
# scores[0] will be high, scores[1] low
``

## 3. Vector Databases & ANN Search

**Problem:** Calculating dot product against 1 Billion vectors is too slow (O(N)).
- 1B vectors `\times` 768 dims `\times` 4 bytes `\approx` 3 TB RAM.
- Brute force scan takes seconds/minutes.

**Solution:** Approximate Nearest Neighbor (ANN) Search.
- Trade-off: slightly lower recall (99% instead of 100%) for massive speedup (O(\log N)).

### HNSW (Hierarchical Navigable Small World)
**Mechanism:**
- Builds a multi-layer graph (skip list structure).
- **Layer 0:** All nodes (vectors). High connectivity.
- **Layer 1:** Subset of nodes.
- **Layer N:** Very few nodes (entry points).
- **Search:** Start at top layer, greedily move to the neighbor closest to the query. When local minimum reached, drop to lower layer.

**Pros:**
- Extremely fast (sub-millisecond).
- High recall.
- Supports incremental updates (unlike IVF).

**Cons:**
- High memory usage (stores graph edges).

### Faiss (Facebook AI Similarity Search)
**IVF (Inverted File):**
- Cluster vectors into `K` Voronoi cells (centroids).
- Assign each vector to the nearest centroid.
- **Search:** Find the closest centroid to the query, then scan only vectors in that cell (and neighbors).

**PQ (Product Quantization):**
- Compress vectors to save RAM.
- Split 768-dim vector into 8 sub-vectors of 96 dims.
- Quantize each sub-vector to 1 byte (256 centroids).
- **Result:** 768 floats (3 KB) `\to` 8 bytes. 300x compression!

``python
import faiss

d = 384 # Dimension
nlist = 100 # Number of clusters (Voronoi cells)
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Train (find centroids)
index.train(doc_embeddings)

# Add vectors
index.add(doc_embeddings)

# Search
D, I = index.search(query_embedding, k=5) # Return top 5
``

## 4. Cross-Encoders (Re-Ranking)

**Problem:** Bi-Encoders compress a whole document into one vector. Information is lost ("bottleneck").
**Solution:** Cross-Encoders process Query and Document **together**.

``
[CLS] Query [SEP] Document [SEP] → [BERT] → [Linear] → Score
``

**Mechanism:**
- The self-attention mechanism attends to every word in the query against every word in the document.
- Captures subtle interactions (negation, exact phrasing).

**Pros:**
- Much higher accuracy/NDCG.

**Cons:**
- **Slow:** Must run BERT for every (Query, Doc) pair. Cannot pre-compute.
- O(N) inference at query time.

**Production Pattern: Retrieve & Re-Rank**
1. **Retriever (Bi-Encoder/BM25):** Get top 100 candidates (Fast, High Recall).
2. **Re-Ranker (Cross-Encoder):** Score top 100 candidates (Accurate, High Precision).
3. **Return:** Top 10.

``python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Re-rank candidates
candidates = [("How to fix a flat tire", "Guide to repairing a punctured wheel"), ...]
scores = cross_encoder.predict(candidates)
``

## 5. Hybrid Search (Best of Both Worlds)

**Problem:**
- **Semantic Search Fails:** Exact matches (Part Number "XJ-900", Error Code "0x4F"), Out-of-Vocabulary words.
- **Keyword Search Fails:** Synonyms, typos, conceptual queries.

**Solution: Hybrid Search**
Combine Dense Vector Score + Sparse Keyword Score (BM25).

**Linear Combination:**
\\[
\text{Score} = \alpha \cdot \text{VectorScore} + (1 - \alpha) \cdot \text{BM25Score}
\\]
*Challenge:* Vector scores (cosine: 0-1) and BM25 scores (0-infinity) are hard to normalize.

**Reciprocal Rank Fusion (RRF):**
Instead of scores, use rank positions. Robust and parameter-free.
\\[
\text{RRF}(d) = \sum_{r \in \text{Rankers}} \frac{1}{k + \text{rank}_r(d)}
\\]
where `k` is a constant (usually 60).

**Example:**
- Doc A: Rank 1 in Vector, Rank 10 in BM25. Score = `1/61 + 1/70`.
- Doc B: Rank 5 in Vector, Rank 5 in BM25. Score = `1/65 + 1/65`.

## 6. Deep Dive: Training Embedding Models

How do we train a model to put "car" and "auto" close together?

**Contrastive Loss (InfoNCE):**
- **Input:** A batch of `(Query, PositiveDoc)` pairs.
- **Goal:** Maximize similarity of positive pairs, minimize similarity with *all other docs in the batch* (In-batch Negatives).

\\[
\mathcal{L} = -\log \frac{e^{\text{sim}(q, p) / \tau}}{\sum_{n \in \text{Batch}} e^{\text{sim}(q, n) / \tau}}
\\]

**Hard Negative Mining:**
- Random negatives (easy) aren't enough. The model needs to distinguish "Python programming" from "Python snake".
- **Strategy:** Use BM25 to find top docs for a query. If a top doc is NOT the ground truth, it's a **Hard Negative**.
- Training on hard negatives boosts performance significantly.

**Matryoshka Representation Learning (MRL):**
- Train embeddings such that the first `k` dimensions alone are good.
- Loss = `\sum_{k \in \{64, 128, 256, 768\}} \text{Loss}_k`.
- **Benefit:** Use 64 dims for fast initial filter (12x faster), 768 dims for re-ranking.

## 7. Deep Dive: Handling Long Documents

BERT has a 512-token limit. Real-world docs are longer.

**Strategies:**
1. **Truncation:** Just take the first 512 tokens. (Works surprisingly well as titles/abstracts contain most info).
2. **Chunking:** Split doc into 200-word chunks with 50-word overlap.
 - Index each chunk as a separate vector.
 - **Retrieval:** Return the parent document of the matching chunk.
3. **Pooling:**
 - **Max-Pooling:** Doc score = Max(Chunk scores). Good for finding specific passages.
 - **Mean-Pooling:** Doc score = Average(Chunk scores). Good for overall topic match.

## 8. Deep Dive: Multilingual Semantic Search

**Problem:** Query in English, Document in French.

**Solution:** Multilingual Models (e.g., LaBSE, mE5).
- Trained on parallel corpora (Translation pairs).
- Align vector spaces across 100+ languages.
- "Cat" (En) and "Chat" (Fr) map to the same vector point.

**Use Case:** Global e-commerce search (User searches "zapatos", finds "shoes").

## 9. Deep Dive: Domain Adaptation (GPL)

**Problem:** Pre-trained models (on MS MARCO) fail on specialized domains (Medical, Legal).
**Solution:** Generative Pseudo Labeling (GPL).

1. **Generate Queries:** Use T5 to generate synthetic queries for your domain documents.
2. **Mine Negatives:** Use BM25 to find hard negatives for these queries.
3. **Label:** Use a Cross-Encoder to score (Query, Doc) pairs (Teacher).
4. **Train:** Train the Bi-Encoder (Student) to mimic the Cross-Encoder scores.

**Result:** Adapts to your domain without human labels!

## 10. Deep Dive: Evaluation Metrics Implementation

How do we measure success?

**NDCG (Normalized Discounted Cumulative Gain):**
Measures the quality of ranking. Highly relevant items should be at the top.

``python
import numpy as np

def dcg_at_k(r, k):
 r = np.asfarray(r)[:k]
 if r.size:
 return np.sum(r / np.log2(np.arange(2, r.size + 2)))
 return 0.

def ndcg_at_k(r, k):
 dcg_max = dcg_at_k(sorted(r, reverse=True), k)
 if not dcg_max:
 return 0.
 return dcg_at_k(r, k) / dcg_max

# Example: Relevance scores of retrieved items
# 3 = Highly Relevant, 2 = Relevant, 1 = Somewhat, 0 = Irrelevant
relevance = [3, 2, 3, 0, 1, 2]
print(f"NDCG@5: {ndcg_at_k(relevance, 5)}")
``

**MRR (Mean Reciprocal Rank):**
Focuses on the *first* relevant item.
\\[
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
\\]
If the first relevant item is at rank 1, MRR=1. If at rank 2, MRR=0.5.

## 11. Deep Dive: Production Architecture for 100M QPS

Scaling Semantic Search is hard because ANN search is CPU/RAM intensive.

**Architecture:**
1. **Query Service:** Stateless API. Encodes query using ONNX Runtime (faster than PyTorch).
2. **Caching Layer:** Redis/Memcached. Caches (Query Vector -> Result IDs).
 - *Semantic Caching:* If query B is very close to query A (cosine > 0.99), return cached result of A.
3. **Sharding:**
 - **Horizontal Sharding:** Split 1B vectors into 10 shards of 100M.
 - Query all 10 shards in parallel (Scatter-Gather).
 - Merge results.
4. **Replication:** Replicate each shard 3x for high availability and throughput.
5. **Quantization:** Use int8 quantization for the embedding model to speed up inference.

## Implementation: End-to-End Semantic Search API

``python
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()

# Load Model
# 'all-MiniLM-L6-v2' is a small, fast model good for English
model = SentenceTransformer('all-MiniLM-L6-v2')

# Mock Database
documents = [
 {"id": 1, "text": "The quick brown fox jumps over the lazy dog."},
 {"id": 2, "text": "A fast auburn canine leaps over a sleepy hound."},
 {"id": 3, "text": "Python is a programming language."},
 {"id": 4, "text": "Pythons are large constricting snakes."},
]

# Build Index
# Normalize embeddings for Cosine Similarity (Dot Product on normalized vectors)
embeddings = model.encode([d['text'] for d in documents])
faiss.normalize_L2(embeddings)

# IndexFlatIP = Exact Inner Product search
index = faiss.IndexFlatIP(384)
index.add(embeddings)

@app.get("/search")
def search(q: str, k: int = 2):
 # Encode Query
 q_emb = model.encode([q])
 q_emb = q_emb.reshape(1, -1)
 faiss.normalize_L2(q_emb)
 
 # Search
 scores, indices = index.search(q_emb, k)
 
 results = []
 for score, idx in zip(scores[0], indices[0]):
 if idx == -1: continue
 results.append({
 "id": documents[idx]["id"],
 "text": documents[idx]["text"],
 "score": float(score)
 })
 
 return results

# Example Usage:
# /search?q=coding -> Returns "Python is a programming language"
# /search?q=reptile -> Returns "Pythons are large constricting snakes"
``

## Top Interview Questions

**Q1: How do you handle updates (insert/delete) in a Vector DB?**
*Answer:*
HNSW graphs are hard to update dynamically.
- **Real-time:** Use a mutable index (like in Milvus/Pinecone) which uses a buffer for new inserts (LSM-tree style). Periodically merge/rebuild the main index.
- **Deletes:** Mark as "deleted" in a bitmask/bitmap. Filter out these IDs during search results post-processing.

**Q2: How do you evaluate Semantic Search?**
*Answer:*
- **NDCG@10 (Normalized Discounted Cumulative Gain):** Measures ranking quality.
- **MRR (Mean Reciprocal Rank):** How high is the first relevant result?
- **Recall@K:** % of relevant docs found in top K.
- **Datasets:** MS MARCO, BEIR benchmark.

**Q3: When should you NOT use Semantic Search?**
*Answer:*
- **Exact Match:** Searching for specific IDs, error codes, or names (e.g., "User 1234").
- **Low Latency:** If < 10ms is required, BM25/Inverted Index is faster.
- **Interpretability:** Vector scores are opaque; BM25 explains "why" (term frequency).

**Q4: How to scale to 1 Billion vectors?**
*Answer:*
- **Compression:** Product Quantization (PQ) to reduce RAM (e.g., 4KB → 64 bytes per vector).
- **Sharding:** Split index across multiple nodes (horizontal scaling).
- **GPU Acceleration:** Faiss on GPU is 10x faster than CPU.

**Q5: What is the "Curse of Dimensionality" in vector search?**
*Answer:*
As dimensions increase, the distance between the nearest and farthest points becomes negligible. However, for text embeddings (768-dim), this is usually manageable. The bigger issue is the computational cost of distance calculation.

## Key Takeaways

1. **Bi-Encoders** enable fast retrieval by pre-computing document vectors.
2. **Cross-Encoders** provide high accuracy re-ranking but are computationally expensive.
3. **Vector DBs** (HNSW, IVF-PQ) are essential for scaling to millions/billions of documents.
4. **Hybrid Search** (Dense + Sparse) is the robust industry standard to handle both semantic and exact match queries.
5. **Hard Negatives** are crucial for training effective embedding models.

## Summary

| Aspect | Insight |
|:---|:---|
| **Core Idea** | Embed text into vector space to capture meaning |
| **Architecture** | Retrieve (Bi-Encoder) → Re-Rank (Cross-Encoder) |
| **Indexing** | HNSW (Graph), IVF-PQ (Clustering + Compression) |
| **Challenges** | Exact match, long docs, scale, domain adaptation |

---

**Originally published at:** [arunbaby.com/ml-system-design/0032-semantic-search-systems](https://www.arunbaby.com/ml-system-design/0032-semantic-search-systems/)

*If you found this helpful, consider sharing it with others who might benefit.*
