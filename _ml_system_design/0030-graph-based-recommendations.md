---
title: "Graph-based Recommendation Systems"
day: 30
collection: ml_system_design
categories:
  - ml_system_design
tags:
  - graphs
  - recommendations
  - gnn
  - link prediction
subdomain: "Recommendations"
tech_stack: [PyTorch Geometric, DGL, Neo4j, NetworkX]
scale: "Billions of edges, Millions of nodes"
companies: [Pinterest, LinkedIn, Facebook, Twitter]
related_dsa_day: 30
related_ml_day: 30
related_speech_day: 30
---

**"Leveraging the connection structure to predict what users will love."**

## 1. Why Graph-based Recommendations?

Traditional recommender systems use **user-item matrices**. Graph-based systems model the entire **interaction network**.

**Example: Social Media**
```
Users: Alice, Bob, Charlie
Items: Post1, Post2, Post3

Graph:
Alice --likes--> Post1 <--likes-- Bob
  |                |
  +--follows--> Charlie
                    |
                 created
                    |
                  Post2
```

**Advantages of Graphs:**
1. **Richer Context:** Capture multi-hop relationships (friend-of-friend recommendations).
2. **Heterogeneous:** Mix users, items, tags, locations in one graph.
3. **Explainability:** "We recommend this post because your friend Bob liked it."
4. **Cold Start:** New users can benefit from their social connections.

## 2. Graph Representation

### Homogeneous Graph
All nodes and edges are the same type.
**Example:** Friendship network (all nodes are users, all edges are "friends").

### Heterogeneous Graph
Multiple node/edge types.
**Example:** E-commerce
- Nodes: Users, Products, Brands, Categories
- Edges: User --bought--> Product, Product --belongs_to--> Category

### Bipartite Graph
Two types of nodes with edges only between different types.
**Example:** User-Item interactions.

**Adjacency Matrix:**
\\[
A_{ij} = 
\begin{cases}
1 & \text{if user } i \text{ interacted with item } j \\
0 & \text{otherwise}
\end{cases}
\\]

## 3. Traditional Graph-Based Approaches

### Approach 1: Collaborative Filtering on Graphs

**Idea:** If users A and B both liked items X and Y, recommend to A what B liked but A hasn't seen.

**Graph Random Walk:**
1. Start at user node.
2. Walk to liked items.
3. Walk to other users who liked those items.
4. Walk to items those users liked.
5. Recommend items with highest visit frequency.

```python
def personalized_pagerank(graph, user_node, damping=0.85, iterations=100):
    scores = {node: 0 for node in graph.nodes}
    scores[user_node] = 1.0
    
    for _ in range(iterations):
        new_scores = {node: 0 for node in graph.nodes}
        for node in graph.nodes:
            for neighbor in graph.neighbors(node):
                new_scores[neighbor] += damping * scores[node] / len(list(graph.neighbors(node)))
            new_scores[node] += (1 - damping) if node == user_node else 0
        scores = new_scores
    
    return scores

# Recommend top-K items with highest scores
```

**Time Complexity:** \\(O(I \cdot E)\\) where I is iterations and E is number of edges.

### Approach 2: Node2Vec

**Idea:** Learn node embeddings by treating random walks as "sentences" and applying Skip-Gram (Word2Vec).

**Algorithm:**
1. Generate random walks starting from each node.
2. Treat walks as sentences: `[UserA, Item1, UserB, Item3, ...]`.
3. Train Skip-Gram to predict context nodes given target node.

```python
from node2vec import Node2Vec

# Generate walks
walks = Node2Vec(graph, dimensions=128, walk_length=80, num_walks=10, workers=4)

# Train Skip-Gram
model = walks.fit(window=10, min_count=1, batch_words=4)

# Get embeddings
user_embedding = model.wv['UserA']
item_embedding = model.wv['Item1']

# Recommend by cosine similarity
recommended_items = model.wv.most_similar('UserA', topn=10)
```

**Pros:**
- Simple and effective.
- Works on any graph.

**Cons:**
- Doesn't use node features (only structure).
- Expensive for large graphs (millions of walks).

## 4. Graph Neural Networks (GNNs)

**Core Idea:** Aggregate information from neighbors to update node representations.

### Message Passing Framework

**General Form:**
\\[
h_v^{(k+1)} = \text{UPDATE}\\left(h_v^{(k)}, \text{AGGREGATE}(\{h_u^{(k)} : u \in \mathcal{N}(v)\})\\right)
\\]

- \\(h_v^{(k)}\\): Representation of node \\(v\\) at layer \\(k\\).
- \\(\mathcal{N}(v)\\): Neighbors of \\(v\\).

**After K layers:** Node \\(v\\) has aggregated information from \\(K\\)-hop neighbors.

### Graph Convolutional Network (GCN)

**Update Rule:**
\\[
H^{(k+1)} = \sigma\\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(k)} W^{(k)}\\right)
\\]

- \\(\tilde{A} = A + I\\) (adjacency matrix + self-loops).
- \\(\tilde{D}\\): Degree matrix of \\(\tilde{A}\\).
- \\(W^{(k)}\\): Learnable weight matrix.

**Implementation:**
```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.conv1 = GCNConv(embedding_dim, 256)
        self.conv2 = GCNConv(256, 128)
    
    def forward(self, edge_index):
        # edge_index: [2, num_edges] (source and target nodes)
        
        # Initialize embeddings
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        # Message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        return x  # [num_nodes, 128]

# Predict interaction score
user_emb = embeddings[user_id]
item_emb = embeddings[item_id]
score = torch.dot(user_emb, item_emb)
```

### GraphSAGE (Sampling and Aggregation)

**Problem:** GCN needs the full adjacency matrix (doesn't scale to billions of edges).

**Solution:** Sample a fixed number of neighbors.

**Algorithm:**
```python
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, hidden_dim)  # Concat self + aggregated
    
    def forward(self, x, edge_index, num_samples=10):
        # x: [num_nodes, in_dim]
        # edge_index: [2, num_edges]
        
        aggregated = []
        for node in range(x.size(0)):
            # Sample neighbors
            neighbors = edge_index[1][edge_index[0] == node]
            if len(neighbors) > num_samples:
                neighbors = neighbors[torch.randperm(len(neighbors))[:num_samples]]
            
            # Aggregate (mean pooling)
            neighbor_embs = x[neighbors]
            agg = neighbor_embs.mean(dim=0)
            aggregated.append(agg)
        
        aggregated = torch.stack(aggregated)
        
        # Concat self + aggregated
        combined = torch.cat([x, aggregated], dim=1)
        output = F.relu(self.linear(combined))
        
        return output
```

**Benefit:** \\(O(K \cdot S \cdot D)\\) where K = layers, S = samples per node, D = embedding dim (independent of graph size!).

## 5. Pinterest's PinSage

**PinSage** is the largest-scale GNN in production (3B nodes, 18B edges).

**Key Innovations:**
1. **Importance-based Sampling:** Sample neighbors with highest visit frequency (from random walks).
2. **Hard Negative Mining:** For each positive interaction, sample negative items similar to the positive (harder to distinguish).
3. **Multi-GPU Training:** Distribute graph across GPUs.
4. **MapReduce Inference:** Precompute embeddings offline using Spark.

**Architecture:**
```
Input: Pin features (image, text) + Graph structure
  |
  v
3 layers of GraphSAGE (neighbor sampling)
  |
  v
Pin Embedding (256-dim)
  |
  v
Cosine Similarity → Recommendations
```

**Results:**
- **Offline:** +40% recall@100 over baseline.
- **Online:** +20% engagement (repins, clicks).

## 6. LinkedIn's Skills Graph

LinkedIn models **Users, Jobs, Skills, Companies** as a heterogeneous graph.

**Example Query:**
"Recommend jobs for a user with skills in Python, ML."

**Solution: Meta-Path-Based Random Walk**
- Meta-path: User --has_skill--> Skill <--requires-- Job
- Walk: `[UserA, Python, Job1, ML, UserB, Scala, Job2]`
- Recommend jobs that appear frequently in walks starting from UserA.

**Heterogeneous GNN:**
```python
class HeteroGNN(nn.Module):
    def __init__(self):
        self.user_conv = GCNConv(128, 256)
        self.job_conv = GCNConv(128, 256)
        self.skill_conv = GCNConv(128, 256)
    
    def forward(self, user_features, job_features, skill_features, edges):
        # edges: {('user', 'has_skill', 'skill'): edge_index, ...}
        
        user_emb = self.user_conv(user_features, edges[('user', 'has_skill', 'skill')])
        job_emb = self.job_conv(job_features, edges[('job', 'requires', 'skill')])
        skill_emb = self.skill_conv(skill_features, edges[('user', 'has_skill', 'skill')])
        
        return user_emb, job_emb, skill_emb
```

## Deep Dive: Training at Scale (Billion-Edge Graphs)

**Challenge:** Graph doesn't fit in GPU memory.

### Solution 1: Mini-Batch Training with Neighbor Sampling

**Cluster-GCN:**
1. Partition the graph into clusters (Louvain algorithm).
2. Sample a batch of clusters.
3. Train GNN on subgraph induced by those clusters.

**Benefit:** Each mini-batch is a small, densely connected subgraph.

### Solution 2: Distributed Training

**DistDGL (Distributed Deep Graph Library):**
- **Graph Store:** Distributed across multiple machines (sharded by node ID).
- **Sampling:** Each worker samples locally and fetches remote neighbors via RPC.
- **Aggregation:** Use MPI All-Reduce to aggregate gradients.

**Scalability:** Trains on graphs with 100B+ edges (Alibaba's product graph).

## Deep Dive: Cold Start with Side Information

**Problem:** New user has no interactions.

**Solution: Use Content Features**
```python
class HybridGNN(nn.Module):
    def __init__(self):
        self.text_encoder = BERTModel()  # Encode user bio, item description
        self.image_encoder = ResNet()    # Encode user profile pic, item image
        self.gnn = GraphSAGE()
    
    def forward(self, text, image, edge_index):
        text_emb = self.text_encoder(text)
        image_emb = self.image_encoder(image)
        
        # Initial embedding = concat(text, image)
        x_init = torch.cat([text_emb, image_emb], dim=1)
        
        # Message passing
        x_final = self.gnn(x_init, edge_index)
        
        return x_final
```

**For new users:** Use \\(x_{\text{init}}\\) directly (no graph info yet).

## Deep Dive: Temporal Graphs (Dynamic Recommendations)

**Problem:** User preferences change over time.

**Solution: Temporal GNN**
```python
class TemporalGNN(nn.Module):
    def __init__(self):
        self.gru = nn.GRU(input_size=128, hidden_size=256)
        self.gnn = GraphSAGE()
    
    def forward(self, snapshots):
        # snapshots: List of (features, edge_index) at different timestamps
        
        h_t = None
        for features, edge_index in snapshots:
            x = self.gnn(features, edge_index)
            x, h_t = self.gru(x.unsqueeze(0), h_t)
        
        return x  # Final embedding incorporates temporal dynamics
```

**Use Case:** Reddit recommending trending posts (graph changes every minute).

## Deep Dive: Knowledge Graph Embeddings (TransE, DistMult)

**Knowledge Graph:** Entities and Relations.
**Example:**
```
(Python, is_a, Programming Language)
(TensorFlow, used_for, Deep Learning)
(Alice, knows, Python)
```

**TransE:**
Embed entities and relations in the same space.
\\[
h + r \approx t
\\]
where \\(h\\) = head entity, \\(r\\) = relation, \\(t\\) = tail entity.

**Loss:**
\\[
\mathcal{L} = \sum_{(h, r, t) \in \mathcal{T}} \max(0, \gamma + d(h + r, t) - d(h' + r, t'))
\\]
where \\((h', r, t')\\) is a negative sample.

**Application:** Amazon's product knowledge graph for recommendations.

## Deep Dive: Graph Augmentation for Robustness

**Problem:** Sparse graphs lead to poor embeddings.

**Solutions:**
1. **Edge Dropout:** Randomly remove edges during training (forces model to not rely on single edges).
2. **Node Mixup:** Interpolate between node features: \\(x_{\text{mix}} = \lambda x_i + (1 - \lambda) x_j\\).
3. **Virtual Nodes:** Add a global node connected to all nodes (helps with long-range dependencies).

```python
def graph_augmentation(edge_index, drop_rate=0.1):
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > drop_rate
    return edge_index[:, mask]
```

## Deep Dive: Explainability with GNN (GNNExplainer)

**Problem:** Why did the model recommend Item X to User Y?

**GNNExplainer:** Find the minimal subgraph that most influences the prediction.

**Algorithm:**
1. Given a node \\(v\\) and prediction \\(y\\), find a subgraph \\(G_S\\).
2. Maximize \\(MI(Y, G_S) = H(Y) - H(Y | G = G_S)\\) (mutual information).
3. Optimize via gradient descent with edge mask.

**Output:** "We recommended this movie because you liked these 3 similar movies (highlighted subgraph)."

## Deep Dive: Negative Sampling Strategies

**Problem:** For each positive interaction (User --likes--> Item), we need negatives.

**Strategies:**
1. **Random:** Sample random items (easy negatives, model learns quickly but not well).
2. **Popularity-based:** Sample popular items (harder, but can bias toward popular items).
3. **Hard Negatives:** Sample items similar to the positive item (e.g., using k-NN on item embeddings).

**Dynamic Hard Negative Mining:**
```python
# During training
positive_items = batch['items']
positive_embs = item_embeddings[positive_items]

# Find K nearest items in embedding space
hard_negatives = faiss_index.search(positive_embs, K)

loss = bpr_loss(user_emb, positive_embs, item_embeddings[hard_negatives])
```

## Deep Dive: Fairness in Graph-based Recommendations

**Problem:** Graph structure can encode bias (e.g., popular items get more exposure).

**Metrics:**
- **Exposure Fairness:** Items with equal quality should get equal exposure.
- **Demographic Parity:** Recommendations should be similar across demographic groups.

**Debiasing:**
1. **Re-weighting:** Upweight interactions with underrepresented items.
2. **Adversarial Training:** Train a discriminator to predict user demographics from embeddings. Maximize recommendation loss, minimize discriminator accuracy.

```python
class FairGNN(nn.Module):
    def forward(self, x, edge_index):
        emb = self.gnn(x, edge_index)
        
        # Recommendation loss
        rec_loss = self.recommendation_loss(emb)
        
        # Fairness loss (fool the discriminator)
        demographics_pred = self.discriminator(emb)
        fair_loss = -self.discriminator_loss(demographics_pred, true_demographics)
        
        total_loss = rec_loss + lambda * fair_loss
        return total_loss
```

## Deep Dive: Graph-based Bandits (Exploration vs. Exploitation)

**Problem:** Should we recommend popular items (exploitation) or explore new items?

**LinUCB with Graphs:**
\\[
\text{Score}(item) = \theta^T x_{item} + \alpha \sqrt{x_{item}^T A^{-1} x_{item}}
\\]
where the second term is the uncertainty (exploration bonus).

**Graph Extension:**
Use GNN to compute \\(x_{\text{item}}\\) (includes neighborhood information).

## Implementation: Full GNN Recommender

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

class GraphRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.num_users = num_users
        
        # Initial embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # GNN layers
        self.conv1 = SAGEConv(embedding_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embedding_dim)
    
    def forward(self, edge_index):
        # Concat user and item embeddings
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        # Message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x
    
    def predict(self, user_emb, item_emb):
        # Dot product
        return (user_emb * item_emb).sum(dim=1)

# Training
def train(model, data, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(data.edge_index)
        
        # BPR Loss (Bayesian Personalized Ranking)
        user_embs = embeddings[data.pos_edges[0]]
        pos_item_embs = embeddings[data.pos_edges[1]]
        neg_item_embs = embeddings[data.neg_edges]
        
        pos_scores = model.predict(user_embs, pos_item_embs)
        neg_scores = model.predict(user_embs, neg_item_embs)
        
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Inference
@torch.no_grad()
def recommend(model, user_id, top_k=10):
    model.eval()
    embeddings = model(data.edge_index)
    
    user_emb = embeddings[user_id]
    item_embs = embeddings[model.num_users:]  # All items
    
    scores = model.predict(user_emb.unsqueeze(0), item_embs)
    top_items = scores.argsort(descending=True)[:top_k]
    
    return top_items
```

## Top Interview Questions

**Q1: How do you handle graphs that don't fit in memory?**
*Answer:*
Use **neighbor sampling** (GraphSAGE) to limit the number of neighbors aggregated. Use **distributed training** (DistDGL) to shard the graph across machines. For inference, precompute embeddings offline.

**Q2: GNNs vs. Matrix Factorization: when to use which?**
*Answer:*
- **Matrix Factorization:** Simpler, faster, works well if you only care about direct user-item interactions.
- **GNNs:** Better when you have rich graph structure (social connections, item similarities, multi-hop relationships).

**Q3: How do you evaluate graph-based recommenders?**
*Answer:*
- **Offline:** Recall@K, NDCG@K, Hit Rate.
- **Online:** A/B test (CTR, engagement).
- **Graph-specific:** Coverage (% of items recommended), Diversity (how different are recommended items).

**Q4: How do you handle new users/items (cold start)?**
*Answer:*
Use **content features** (text, images) in addition to graph structure. For new items with no interactions, compute initial embedding from content. As interactions occur, refine embedding via GNN.

## Key Takeaways

1. **Graphs Capture Structure:** Use connections (social, similarity) for better recommendations.
2. **GNNs are SOTA:** Message passing aggregates multi-hop information.
3. **Scalability Challenges:** Use sampling (GraphSAGE) and distributed training (DistDGL).
4. **Real-World Systems:** Pinterest (PinSAGE), LinkedIn (Skills Graph), Alibaba (Product Graph).
5. **Hybrid Approaches:** Combine graph structure + content features for cold start robustness.

## Summary

| Aspect | Insight |
|:---|:---|
| **Core Idea** | Aggregate neighbor information to learn node embeddings |
| **Key Architectures** | GCN, GraphSAGE, GAT, PinSage |
| **Challenges** | Scalability, cold start, fairness |
| **Applications** | Social media, e-commerce, job recommendations |

---

**Originally published at:** [arunbaby.com/ml-system-design/0030-graph-based-recommendations](https://www.arunbaby.com/ml-system-design/0030-graph-based-recommendations/)

*If you found this helpful, consider sharing it with others who might benefit.*


