---
title: "Knowledge Graph Systems"
day: 34
collection: ml_system_design
categories:
  - ml_system_design
tags:
  - knowledge-graph
  - graph-neural-networks
  - nlp
  - database
subdomain: "Data Systems"
tech_stack: [Neo4j, RDF, SPARQL, GraphQL, PyTorch Geometric]
scale: "Billions of entities, Trillions of edges"
companies: [Google, LinkedIn, Pinterest, Airbnb]
---

**"Structuring the world's information into connected entities and relationships."**

## 1. What is a Knowledge Graph (KG)?

A Knowledge Graph is a structured representation of facts, consisting of **entities** (nodes) and **relationships** (edges).

- **Entities:** Real-world objects (e.g., "Barack Obama", "Hawaii", "President").
- **Relationships:** Connections between them (e.g., "born_in", "role").
- **Fact:** `(Barack Obama, born_in, Hawaii)`

**Why use KGs?**
- **Search:** "Who is the wife of the 44th president?" requires traversing relationships.
- **Recommendations:** "Users who liked 'Inception' also liked movies directed by 'Christopher Nolan'".
- **Q&A Systems:** Providing direct answers instead of just blue links.

## 2. Data Model: RDF vs. Labeled Property Graph

There are two main ways to model KGs:

### 1. Resource Description Framework (RDF)
- **Standard:** W3C standard for semantic web.
- **Structure:** Triples `(Subject, Predicate, Object)`.
- **Example:**
  - `(DaVinci, painted, MonaLisa)`
  - `(MonaLisa, located_in, Louvre)`
- **Query Language:** SPARQL.
- **Pros:** Great for interoperability, public datasets (DBpedia, Wikidata).
- **Cons:** Verbose, hard to attach properties to edges (requires reification).

### 2. Labeled Property Graph (LPG)
- **Structure:** Nodes and edges have internal key-value properties.
- **Example:**
  - Node: `Person {name: "DaVinci", born: 1452}`
  - Edge: `PAINTED {year: 1503}`
  - Node: `Artwork {title: "Mona Lisa"}`
- **Query Language:** Cypher (Neo4j), Gremlin.
- **Pros:** Intuitive, efficient for traversal, flexible schema.
- **Cons:** Less standardized than RDF.

**Industry Choice:** Most tech companies (LinkedIn, Airbnb, Uber) use **LPG** for internal applications due to performance and flexibility.

## 3. Knowledge Graph Construction Pipeline

Building a KG from unstructured text (web pages, documents) is a massive NLP challenge.

### Step 1: Named Entity Recognition (NER)
Identify entities in text.
- **Input:** "Elon Musk founded SpaceX in 2002."
- **Output:** `[Elon Musk] (PERSON)`, `[SpaceX] (ORG)`, `[2002] (DATE)`.
- **Model:** BERT-NER, BiLSTM-CRF.

### Step 2: Entity Linking (Resolution)
Map the extracted mention to a unique ID in the KG.
- **Challenge:** "Michael Jordan" -> Basketball player or ML researcher?
- **Solution:** Contextual embeddings. Compare the context of the mention with the description of the candidate entities.

### Step 3: Relation Extraction (RE)
Identify the relationship between entities.
- **Input:** "Elon Musk" and "SpaceX".
- **Context:** "...founded..."
- **Output:** `founded_by(SpaceX, Elon Musk)`.
- **Model:** Relation Classification heads on BERT.

### Step 4: Knowledge Fusion
Merge facts from multiple sources.
- Source A: `(Obama, born, Hawaii)`
- Source B: `(Obama, birth_place, Honolulu)`
- **Resolution:** "Honolulu" is part of "Hawaii". Merge or create hierarchy.

## 4. Storage Architecture

How do we store billions of nodes and trillions of edges?

### 1. Native Graph Databases (Neo4j, Amazon Neptune)
- **Storage:** Index-free adjacency. Each node physically stores pointers to its neighbors.
- **Pros:** $O(1)$ traversal per hop. Fast for deep queries.
- **Cons:** Hard to shard (graph partitioning is NP-hard).

### 2. Relational Backends (Facebook TAO, LinkedIn Liquid)
- **Storage:** MySQL/PostgreSQL sharded by ID.
- **Schema:** `(id, type, data)` and `(id1, type, id2, time)`.
- **Pros:** Extremely scalable, leverages existing DB infra.
- **Cons:** Multi-hop queries require multiple DB lookups (higher latency).

### 3. Distributed Key-Value Stores (Google Knowledge Graph)
- **Storage:** BigTable / HBase.
- **Key:** Subject ID.
- **Value:** List of (Predicate, Object).
- **Pros:** Massive write throughput.

## 5. Knowledge Graph Inference

We don't just store facts; we infer **new** facts.

### 1. Link Prediction (Knowledge Graph Completion)
Predict missing edges.
- **Query:** `(Tom Hanks, acted_in, ?)`
- **Task:** Rank all movies by probability.

### 2. Knowledge Graph Embeddings (KGE)
Map entities and relations to vector space.
- **TransE:** $h + r \approx t$. The translation of head $h$ by relation $r$ should land near tail $t$.
- **RotatE:** Models relations as rotations in complex space (handles symmetry/antisymmetry).
- **DistMult:** Uses bilinear product.

### 3. Graph Neural Networks (GNNs)
- **GraphSAGE / GAT:** Aggregate information from neighbors to generate node embeddings.
- **Use Case:** Node classification (is this account a bot?), Link prediction (friend recommendation).

## 6. Real-World Case Studies

### Case Study 1: Google Knowledge Graph
- **Scale:** 500B+ facts.
- **Use:** "Things, not strings." Powers the info box on the right side of search results.
- **Innovation:** Massive scale entity disambiguation using search logs.

### Case Study 2: LinkedIn Economic Graph
- **Entities:** Members, Companies, Skills, Jobs, Schools.
- **Edges:** `employed_by`, `has_skill`, `alumni_of`.
- **Use:** "People You May Know", Job Recommendations, Skill Gap Analysis.
- **Tech:** "Liquid" (Graph DB built on top of relational sharding).

### Case Study 3: Pinterest Taste Graph
- **Entities:** Users, Pins (Images), Boards.
- **Edges:** `saved_to`, `clicked_on`.
- **Model:** **PinSage** (GNN).
- **Innovation:** Random-walk based sampling to train GNNs on billions of nodes.

## 7. Deep Dive: Graph RAG (Retrieval Augmented Generation)

LLMs hallucinate. KGs provide ground truth.

**Architecture:**
1. **User Query:** "What drugs interact with Aspirin?"
2. **KG Lookup:** Query KG for `(Aspirin, interacts_with, ?)`.
3. **Context Retrieval:** Get subgraph: `(Aspirin, interacts_with, Warfarin)`, `(Aspirin, interacts_with, Ibuprofen)`.
4. **Prompt Augmentation:** "Context: Aspirin interacts with Warfarin and Ibuprofen. Question: What drugs interact with Aspirin?"
5. **LLM Generation:** "Aspirin interacts with Warfarin and Ibuprofen..."

**Pros:** Factual accuracy, explainability (can cite the KG edge).

## 8. Deep Dive: Scaling Graph Databases

**Sharding Problem:**
Cutting a graph cuts edges. Queries that traverse cuts are slow (network calls).

**Solution 1: Hash Partitioning**
- `ShardID = hash(NodeID) % N`.
- **Pros:** Even distribution.
- **Cons:** Random cuts. A 3-hop query might hit 4 shards.

**Solution 2: METIS (Graph Partitioning)**
- Minimize edge cuts. Keep communities on the same shard.
- **Pros:** Faster local traversals.
- **Cons:** Hard to maintain as graph changes dynamically.

**Solution 3: Replication (Facebook TAO)**
- Cache the entire "social graph" in RAM across thousands of memcache nodes.
- Read-heavy workload optimization.

## 9. System Design Interview: Design a KG for Movies

**Requirements:**
- Store Movies, Actors, Directors.
- Query: "Movies directed by Nolan starring DiCaprio".
- Scale: 1M movies, 10M people.

**Schema:**
- Nodes: `Movie`, `Person`.
- Edges: `DIRECTED`, `ACTED_IN`.

**Storage:**
- **Neo4j** (Single instance fits in RAM). 11M nodes is small.
- If 10B nodes -> **JanusGraph** on Cassandra.

**API:**
- GraphQL is perfect for hierarchical graph queries.
```graphql
query {
  director(name: "Christopher Nolan") {
    movies {
      title
      actors(name: "Leonardo DiCaprio") {
        name
      }
    }
  }
}
```

## 10. Top Interview Questions

**Q1: How do you handle entity resolution at scale?**
*Answer:* Blocking (LSH) to find candidates, then pairwise classification (XGBoost/BERT) to verify.

**Q2: TransE vs GNNs?**
*Answer:* TransE is shallow (lookup table). GNNs are deep (aggregate features). GNNs generalize to unseen nodes (inductive), TransE is transductive.

**Q3: How to update a KG in real-time?**
*Answer:* Lambda architecture. Batch pipeline re-builds the high-quality graph nightly. Streaming pipeline adds temporary edges from Kafka events.

## 11. Summary

| Component | Technology |
| :--- | :--- |
| **Data Model** | Labeled Property Graph (LPG) |
| **Storage** | Neo4j, JanusGraph, Amazon Neptune |
| **Query** | Cypher, Gremlin, GraphQL |
| **Inference** | GraphSAGE, TransE |
| **Use Cases** | Search, RecSys, Fraud Detection |
| **Scale** | Billions of nodes, Trillions of edges |

## 12. Deep Dive: Graph Neural Networks (GNNs)

Traditional embeddings (TransE) are "shallow" — they learn a unique vector for every node. They cannot generalize to new nodes without retraining. **GNNs** solve this.

**GraphSAGE (Graph Sample and Aggregate):**
- **Idea:** Generate node embeddings by sampling and aggregating features from a node's local neighborhood.
- **Inductive:** Can generate embeddings for unseen nodes if we know their features and neighbors.

**Algorithm:**
1.  **Sample:** For each node, sample a fixed number of neighbors (e.g., 10).
2.  **Aggregate:** Combine neighbor embeddings (Mean, LSTM, or Max Pooling).
3.  **Update:** Concatenate self-embedding with aggregated neighbor embedding and pass through a Neural Network.
4.  **Repeat:** Do this for $K$ layers (hops).

**Code Snippet (PyTorch Geometric):**
```python
import torch
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x: Node feature matrix
        # edge_index: Graph connectivity
        
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        return x
```

**Graph Attention Networks (GAT):**
- **Idea:** Not all neighbors are equal. Learn **attention weights** to prioritize important neighbors.
- **Mechanism:** Compute attention coefficient $\alpha_{ij}$ for edge $i \to j$.
- **Benefit:** Better performance on noisy graphs.

## 13. Deep Dive: Knowledge Graph Embeddings (KGE) Math

Let's look at the math behind **TransE** and **RotatE**.

**TransE (Translating Embeddings):**
- **Score Function:** $f(h, r, t) = -||h + r - t||$
- **Objective:** Minimize margin-based ranking loss.
  $$L = \sum_{(h,r,t) \in S} \sum_{(h',r,t') \in S'} [\gamma + f(h, r, t) - f(h', r, t')]_+$$
- **Limitation:** Cannot model 1-to-N relations (e.g., `Teacher -> Student`). If $h+r \approx t_1$ and $h+r \approx t_2$, then $t_1 \approx t_2$, forcing all students to be identical.

**RotatE (Rotation Embeddings):**
- **Idea:** Map entities to the complex plane $\mathbb{C}$.
- **Relation:** Rotation in complex space. $t = h \circ r$, where $|r_i| = 1$.
- **Capability:** Can model:
  - **Symmetry:** $r \circ r = 1$ (e.g., `spouse`).
  - **Antisymmetry:** $r \circ r \neq 1$ (e.g., `parent`).
  - **Inversion:** $r_2 = r_1^{-1}$ (e.g., `hypernym` vs `hyponym`).

## 14. Deep Dive: Entity Linking at Scale

How do you link "MJ" to "Michael Jordan" (Basketball) vs "Michael Jackson" (Singer) when you have 100M entities?

**Two-Stage Pipeline:**

**Stage 1: Blocking / Candidate Generation (Recall)**
- **Goal:** Retrieve top-K (e.g., 100) candidates quickly.
- **Technique:**
  - **Inverted Index:** Map surface forms ("MJ", "Mike") to Entity IDs.
  - **Dense Retrieval:** Encode mention context and entity description into vectors. Use FAISS to find nearest neighbors.

**Stage 2: Re-Ranking (Precision)**
- **Goal:** Select the best match from candidates.
- **Model:** Cross-Encoder (BERT).
  - Input: `[CLS] Mention Context [SEP] Entity Description [SEP]`
  - Output: Probability of match.
- **Features:**
  - **String Similarity:** Edit distance.
  - **Prior Probability:** $P(Entity)$. "Michael Jordan" usually means the basketball player.
  - **Coherence:** Does this entity fit with other entities in the document? (e.g., "Chicago Bulls" is nearby).

## 15. Deep Dive: Graph RAG Implementation

**Retrieval Augmented Generation** with Graphs is powerful for multi-hop reasoning.

**Scenario:** "Who is the CEO of the company that acquired GitHub?"

**Vector RAG Failure:**
- Vector search might find docs about "GitHub acquisition" and "Microsoft CEO".
- But it might miss the connection if they are in separate documents.

**Graph RAG Success:**
1.  **Entity Linking:** Extract "GitHub". Link to `GitHub (Company)`.
2.  **Graph Traversal (1-hop):** `GitHub -[ACQUIRED_BY]-> Microsoft`.
3.  **Graph Traversal (2-hop):** `Microsoft -[CEO_IS]-> Satya Nadella`.
4.  **Context Construction:** "GitHub was acquired by Microsoft. Microsoft's CEO is Satya Nadella."
5.  **LLM Answer:** "Satya Nadella."

**Implementation Steps:**
1.  **Ingest:** Parse documents into triples `(Subject, Predicate, Object)`.
2.  **Index:** Store triples in Neo4j.
3.  **Query:**
    - Use LLM to generate Cypher query.
    - `MATCH (c:Company {name: "GitHub"})-[:ACQUIRED_BY]->(parent)-[:CEO]->(ceo) RETURN ceo.name`
4.  **Generate:** Pass result to LLM for natural language response.

## 16. Deep Dive: Temporal Knowledge Graphs

Facts change over time.
- `(Obama, role, President)` is true only for `[2009, 2017]`.

**Modeling Time:**
1.  **Reification:** Turn the edge into a node.
    - `(Obama) -> [Term] -> (President)`
    - `[Term]` has property `start: 2009`, `end: 2017`.
2.  **Quadruples:** Store `(Subject, Predicate, Object, Timestamp)`.
3.  **Temporal Embeddings:** $f(h, r, t, \tau)$. The embedding evolves over time.

## 17. Deep Dive: Quality Assurance in KGs

Garbage In, Garbage Out. How to ensure KG quality?

**1. Schema Constraints (SHACL):**
- Define rules: `Person` can only `marry` another `Person`.
- `BirthDate` must be a valid date.

**2. Consistency Checking:**
- Logic rules: `born_in(X, Y) AND located_in(Y, Z) -> born_in(X, Z)`.
- If KG says `born_in(Obama, Kenya)` but also `born_in(Obama, Hawaii)` and `Hawaii != Kenya`, flag contradiction.

**3. Human-in-the-Loop:**
- High-confidence facts -> Auto-merge.
- Low-confidence facts -> Send to human annotators (crowdsourcing).

## 18. Deep Dive: Federated Knowledge Graphs

Enterprises often have data silos.
- **HR Graph:** Employees, Roles.
- **Sales Graph:** Customers, Deals.
- **Product Graph:** SKUs, Specs.

**Challenge:** Query across silos. "Which sales rep sold Product X to Customer Y?"

**Solution: Data Fabric / Virtual Graph**
- Leave data where it is (SQL, NoSQL, APIs).
- Create a **Virtual Semantic Layer** on top.
- Map local schemas to a global ontology.
- Query Federation engine (e.g., Starogard) decomposes SPARQL/GraphQL query into sub-queries for each backend.

## 19. System Design: Real-Time Fraud Detection with KG

**Problem:** Detect credit card fraud.
**Insight:** Fraudsters often share attributes (same phone, same IP, same device) forming "rings".

**Design:**
1.  **Ingestion:** Kafka stream of transactions.
2.  **Graph Update:** Add node `Transaction`. Link to `User`, `Device`, `IP`.
3.  **Feature Extraction (Real-time):**
    - Count connected components size.
    - Cycle detection (User A -> Card B -> User C -> Card A).
    - PageRank (guilt by association).
4.  **Inference:** Pass graph features to XGBoost model.
5.  **Latency:** < 200ms.
    - Use in-memory graph (RedisGraph or Neo4j Causal Cluster).

## 20. Advanced: Neuro-Symbolic AI

Combining the learning capability of Neural Networks with the reasoning of Symbolic Logic (KGs).

**Concept:**
- **Neural:** Good at perception (images, text).
- **Symbolic:** Good at reasoning, math, consistency.

**Application:**
- **Visual Question Answering (VQA):**
  - Image: "A red cube on a blue cylinder."
  - Neural: Detect objects (Cube, Cylinder) and attributes (Red, Blue).
  - Symbolic: Build scene graph. Query `on(Cube, Cylinder)`.

## 21. Summary

| Component | Technology |
| :--- | :--- |
| **Data Model** | Labeled Property Graph (LPG) |
| **Storage** | Neo4j, JanusGraph, Amazon Neptune |
| **Query** | Cypher, Gremlin, GraphQL |
| **Inference** | GraphSAGE, TransE |
| **Use Cases** | Search, RecSys, Fraud Detection |
| **Scale** | Billions of nodes, Trillions of edges |

## 22. Deep Dive: Graph Databases vs. Relational Databases

When should you use a Graph DB over Postgres?

**Relational (SQL):**
- **Data Model:** Tables, Rows, Foreign Keys.
- **Join:** Computed at query time. $O(N \log N)$ or $O(N^2)$.
- **Use Case:** Structured data, transactions, aggregations.
- **Query:** "Find all users who bought item X." (1 Join).

**Graph (Neo4j):**
- **Data Model:** Nodes, Edges.
- **Join:** Pre-computed (edges are pointers). $O(1)$ per hop.
- **Use Case:** Highly connected data, pathfinding.
- **Query:** "Find all users who bought item X, and their friends who bought item Y." (Multi-hop).

**Benchmark:**
For a 5-hop query on a social network:
- **SQL:** 10+ seconds (5 joins).
- **Graph:** < 100ms (pointer traversal).

## 23. Deep Dive: Ontology Design

An **Ontology** is the schema of your Knowledge Graph.

**Components:**
1.  **Classes:** `Person`, `Company`, `City`.
2.  **Properties:** `name` (string), `age` (int).
3.  **Relationships:** `WORKS_AT` (Person -> Company).
4.  **Inheritance:** `Employee` is a subclass of `Person`.

**Design Patterns:**
- **Reification:** Don't just link `Actor -> Movie`. Link `Actor -> Role -> Movie` to store "character name".
- **Hierarchy:** Use `subClassOf` sparingly. Too deep hierarchies make inference slow.

**Example (OWL/Turtle):**
```turtle
:Person a owl:Class .
:Employee a owl:Class ;
    rdfs:subClassOf :Person .
:worksAt a owl:ObjectProperty ;
    rdfs:domain :Employee ;
    rdfs:range :Company .
```

## 24. Deep Dive: Reasoning Engines

Reasoning allows inferring implicit facts.

**Types of Reasoning:**
1.  **RDFS Reasoning:**
    - Rule: `Employee subClassOf Person`.
    - Fact: `John is Employee`.
    - Inference: `John is Person`.
2.  **Transitive Reasoning:**
    - Rule: `partOf` is transitive.
    - Fact: `Finger partOf Hand`, `Hand partOf Arm`.
    - Inference: `Finger partOf Arm`.
3.  **Inverse Reasoning:**
    - Rule: `parentOf` inverseOf `childOf`.
    - Fact: `A parentOf B`.
    - Inference: `B childOf A`.

**Tools:**
- **Jena Inference Engine** (Java).
- **GraphDB** (Ontotext).

## 25. Deep Dive: Graph Visualization Tools

Visualizing 1B nodes is impossible. We need tools to explore subgraphs.

**Tools:**
1.  **Gephi:** Desktop tool. Good for static analysis of medium graphs (100k nodes).
2.  **Cytoscape:** Bio-informatics focus. Good for protein interaction networks.
3.  **Neo4j Bloom:** Interactive exploration. "Show me the shortest path between X and Y."
4.  **KeyLines / ReGraph:** JavaScript libraries for building web-based graph visualizers.

**Visualization Techniques:**
- **Force-Directed Layout:** Simulates physics (nodes repel, edges attract).
- **Community Detection coloring:** Color nodes by Louvain community.
- **Ego-Network:** Only show node X and its immediate neighbors.

## 26. Code: Loading Data into Neo4j

How to ingest data programmatically.

```python
from neo4j import GraphDatabase

class KnowledgeGraphLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def add_person(self, name, age):
        with self.driver.session() as session:
            session.run(
                "MERGE (p:Person {name: $name}) SET p.age = $age",
                name=name, age=age
            )

    def add_friendship(self, name1, name2):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (a:Person {name: $name1})
                MATCH (b:Person {name: $name2})
                MERGE (a)-[:FRIEND]->(b)
                """,
                name1=name1, name2=name2
            )

# Usage
loader = KnowledgeGraphLoader("bolt://localhost:7687", "neo4j", "password")
loader.add_person("Alice", 30)
loader.add_person("Bob", 32)
loader.add_friendship("Alice", "Bob")
loader.close()
```

## 27. Summary

| Component | Technology |
| :--- | :--- |
| **Data Model** | Labeled Property Graph (LPG) |
| **Storage** | Neo4j, JanusGraph, Amazon Neptune |
| **Query** | Cypher, Gremlin, GraphQL |
| **Inference** | GraphSAGE, TransE |
| **Use Cases** | Search, RecSys, Fraud Detection |
| **Scale** | Billions of nodes, Trillions of edges |

## 28. Deep Dive: Graph Partitioning Algorithms

To scale to billions of nodes, we must shard the graph.

**Problem:** Minimizing "edge cuts" (edges that span across shards) to reduce network latency.

**Algorithm 1: METIS (Multilevel Graph Partitioning)**
- **Phase 1 (Coarsening):** Collapse adjacent nodes into super-nodes to create a smaller graph.
- **Phase 2 (Partitioning):** Partition the coarse graph.
- **Phase 3 (Uncoarsening):** Project the partition back to the original graph and refine.
- **Pros:** High quality partitions.
- **Cons:** Slow, requires global graph view (offline).

**Algorithm 2: Fennel (Streaming Partitioning)**
- Assign nodes to shards as they arrive in a stream.
- **Heuristic:** Place node $v$ in shard $i$ that maximizes:
  $$Score(v, i) = |N(v) \cap S_i| - \alpha (|S_i|)^{\gamma}$$
  - Term 1: Attraction (place where neighbors are).
  - Term 2: Repulsion (load balancing).
- **Pros:** Fast, scalable, works for dynamic graphs.

## 29. Deep Dive: Graph Query Optimization

Just like SQL optimizers, Graph DBs need to plan queries.

**Query:** `MATCH (p:Person)-[:LIVES_IN]->(c:City {name: 'London'})-[:HAS_RESTAURANT]->(r:Restaurant)`

**Execution Plans:**
1.  **Scan Person:** Find all people, check if they live in London... (Bad, 1B people).
2.  **Index Scan City:** Find 'London' (1 node). Traverse out to `Person` (8M nodes). Traverse out to `Restaurant` (20k nodes).
3.  **Bi-directional:** Start at 'London', traverse both ways.

**Cost-Based Optimizer:**
- Uses statistics (node counts, degree distribution).
- "City" has cardinality 10,000. "Person" has 1B.
- Start with the most selective filter (`name='London'`).

## 30. Deep Dive: Graph Analytics Algorithms

Beyond simple queries, we run global algorithms.

**1. PageRank:**
- Measure node importance.
- **Use Case:** Search ranking, finding influential Twitter users.
- **Update Rule:** $PR(u) = (1-d) + d \sum_{v \in N_{in}(u)} \frac{PR(v)}{OutDegree(v)}$.

**2. Louvain Modularity (Community Detection):**
- Detect clusters of densely connected nodes.
- **Use Case:** Fraud rings, topic detection.

**3. Betweenness Centrality:**
- Number of shortest paths passing through a node.
- **Use Case:** Identifying bottlenecks in a supply chain or network router.

## 31. Deep Dive: Hardware Acceleration for Graphs

CPUs are bad at graph processing (random memory access = cache misses).

**Graphcore IPU (Intelligence Processing Unit):**
- **Architecture:** Massive MIMD (Multiple Instruction, Multiple Data).
- **Memory:** In-processor memory (SRAM) instead of HBM.
- **Benefit:** 10x-100x speedup for GNN training and random walks.

**Cerebras Wafer-Scale Engine:**
- A single chip the size of a wafer.
- Holds the entire graph in SRAM.
- Zero latency communication between cores.

## 32. Summary

| Component | Technology |
| :--- | :--- |
| **Data Model** | Labeled Property Graph (LPG) |
| **Storage** | Neo4j, JanusGraph, Amazon Neptune |
| **Query** | Cypher, Gremlin, GraphQL |
| **Inference** | GraphSAGE, TransE |
| **Use Cases** | Search, RecSys, Fraud Detection |
| **Scale** | Billions of nodes, Trillions of edges |

---

**Originally published at:** [arunbaby.com/ml-system-design/0034-knowledge-graph-systems](https://www.arunbaby.com/ml-system-design/0034-knowledge-graph-systems/)
