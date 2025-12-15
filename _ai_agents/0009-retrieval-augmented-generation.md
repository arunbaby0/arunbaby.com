---
title: "Retrieval-Augmented Generation (RAG)"
day: 9
collection: ai_agents
categories:
  - ai-agents
tags:
  - rag
  - vector-db
  - embeddings
  - hnsw
  - ragas
  - knowledge-retrieval
difficulty: Medium-Easy
---

**"Giving the Brain a Library: The Foundation of Knowledge-Intensive Agents."**

## 1. Introduction: The Hallucination Problem

LLMs are geniuses with amnesia. They are "Frozen-in-Time" encyclopedias. A base model like GPT-4 knows everything about the world up to its training cutoff (e.g., Oct 2023). It knows **nothing** about:
1.  **Recent Events:** The winner of the Super Bowl yesterday.
2.  **Private Data:** Your company's internal wiki, your emails, or your proprietary codebase.
3.  **Specific Domains:** The nuances of your specific 200-page HR policy.

When you ask an agent about these topics, it faces a dilemma: admit ignorance (which it is trained to avoid) or hallucinate (make up a plausible-sounding answer). For an autonomous agent executing actions, hallucination is fatal. You cannot have an agent acting on a hallucinated stock price or deleting a file it "thinks" is unused.

**Retrieval-Augmented Generation (RAG)** is the architectural solution. Instead of relying on the model's *internal* weights (Parametric Memory), we allow it to look up information in an *external* database (Non-Parametric Memory) before answering.

For Agents, RAG is not just a feature; it is a **Core Organ**. It is the bridge between reasoning and reality.

---

## 2. The Mechanics of RAG

RAG is fundamentally a data pipeline followed by an inference step. It consists of two distinct phases: **Indexing** (Offline) and **Retrieval** (Runtime).

### 2.1 Phase 1: Indexing (The ETL Pipeline)
How do we turn a PDF into numbers that a machine can "understand"?

1.  **Loading:** Ingest data. Text, PDFs, Markdown, HTML, Notion pages, Slack dumps.
2.  **Splitting (Chunking):** LLMs have context limits. We can't feed a 500-page book in one go. We split it into smaller "Chunks" (e.g., 500 tokens).
    *   *The Pivot:* Chunking is an art. If you chunk too small, you lose context ("It costs $5" - what is *it*?). If you chunk too big, you confuse the retrieval.
    *   *Strategy:* **Recursive Character Splitting** (split by paragraphs, then sentences) is the standard.
3.  **Embedding:** We pass each chunk through an **Embedding Model** (e.g., OpenAI `text-embedding-3-small`, `bge-m3`).
    *   *Input:* "The sky is blue."
    *   *Output:* `[0.12, -0.45, 0.88, ...]` (A vector of 1536 floating point numbers).
    *   *Concept:* This vector represents the **semantic meaning** of the text in a high-dimensional latent space.
4.  **Storing:** Save the Vectors + Original Text + Metadata in a **Vector Database** (Pinecone, Weaviate, Chroma).

### 2.2 Phase 2: Retrieval (The Agent Loop)
When the user asks a question:

1.  **Query Embedding:** User asks "What is our vacation policy?". We embed this query into a vector using the *same model* as indexing.
2.  **Semantic Search (k-NN):** We query the Vector DB: "Find the top 3 chunks geometrically closest to this query vector."
    *   *Result:* Chunk A ("Vacation Policy..."), Chunk B ("Leave types...").
3.  **Context Injection:** We construct a prompt:
    ```text
    System: Answer the user using ONLY the context below. Do not use outside knowledge.
    Context:
    - Vacation Policy: 20 days per year...
    - Leave types: Sick, Annual...
    
    User: What is our vacation policy?
    ```
4.  **Generation:** The LLM reads the context and answers "You get 20 days pear year."

---

## 3. Deep Dive: Vector Database Internals

How does a database find the "nearest neighbor" among 100 million vectors in 10 milliseconds? It doesn't compare them all (that would be $O(N)$). It uses **Approximate Nearest Neighbors (ANN)** algorithms.

### 3.1 HNSW (Hierarchical Navigable Small Worlds)
This is the industry standard algorithm (used by Pinecone, Weaviate).
*   **Concept:** It builds a multi-layer graph.
    *   *Top Layer:* Like an express highway. Long jumps across the vector space.
    *   *Bottom Layer:* Local roads. Dense connections between close neighbors.
*   **Search:** The query starts at the top, zooms to the general neighborhood of the data, and drills down to the local connections.
*   **Tradeoff:** Fast search, but high memory usage (Requires storing the graph structure).

### 3.2 IVF (Inverted File Index)
Used by FAISS.
*   **Concept:** Cluster the vectors into 1000 "Voronoi Cells" (centroids).
*   **Search:** First, find which cell the query belongs to. Then scan only the vectors in that cell.
*   **Tradeoff:** Very efficient, but recall can drop (if the answer is just on the border of another cell).

---

## 4. Naive RAG vs. Advanced RAG

The standard pipeline is called "Naive RAG." It works for simple demos but fails in production.

### 4.1 Failure Modes
*   **Keyword Misses:** "Vacation" vs "Time Off." Vectors handle typical synonyms, but specific jargon (Product IDs like `XJ-900` vs `XJ900`) often fails semantic search.
*   **Loss of Context:** Retrieving a chunk that says "He agreed to the terms" is useless if the chunk doesn't say *who* "He" is.
*   **Distraction:** Retrieving 10 chunks, 9 of which are irrelevant, can confuse the LLM ("Lost in the Middle").

### 4.2 Advanced Patterns
1.  **Hybrid Search (Alpha):** Combine Vector Search (Semantic) with Keyword Search (BM25/Splade).
    *   *Formula:* $Score = \alpha \cdot VectorScore + (1-\alpha) \cdot KeywordScore$.
    *   *Why:* Ensures exact matches for IDs and Names work while retaining semantic understanding.
2.  **Re-Ranking (The Cross-Encoder):**
    *   Retrieve top 50 results (fast & loose).
    *   Pass them through a powerful "Re-Ranker Model" (e.g., Cohere Rerank) that compares the Query and Document interacting token-by-token.
    *   Take the top 5.
    *   *Impact:* Increases accuracy by 20-30%.
3.  **Parent-Child Indexing:**
    *   *Index:* Small chunks (sentences) for precise search.
    *   *Retrieve:* The Parent chunk (full paragraph) for context.
    *   *Why:* "Small to search, Big to read."

---

## 5. RAG Evaluation (RAGAS)

How do you look at your RAG system and say "It's good"? Feeling good isn't engineering.
We use the **RAGAS** (RAG Assessment) framework, which defines metrics calculated by an LLM-as-a-Judge.

1.  **Faithfulness:** Does the answer come *solely* from the retrieved context? (Low hallucination).
2.  **Answer Relevance:** Does the answer actually address the user query?
3.  **Context Precision:** Did we retrieve the *right* chunks? (Signal-to-noise ratio).
4.  **Context Recall:** Did we retrieve *all* the necessary info?

---

## 6. Making RAG a Tool

For an agent, RAG is just another **Tool**.
We don't hard-code the retrieval. We give the agent a tool call.

### The Agentic RAG Pattern
*   **Tool:** `search_knowledge_base(query: str, filters: dict)`
*   **Description:** "Use this tool to look up technical documentation. You can filter by year or tag."

### The "Self-Querying" Retriever
An advanced pattern where the LLM writes a Structured Database Query.
*   *User:* "Show me emails from Alice last week."
*   *LLM:* `search(query="from Alice", filter={"date": "> 2023-10-01"})`
*   *Runtime:* Applies a Metadata Filter on the Vector DB. This is vastly superior to pure semantic search for structured questions.

---

## 7. Pseudo-Code: Semantic Search Logic

Understanding the inner loop of a vector search engine (conceptual).

```python
def semantic_search(query, k=5):
    # 1. Embed Query
    query_vector = model.embed(query)
    
    # 2. Score All Documents (Naive Calculation)
    scores = []
    for doc in database:
        # Cosine Similarity = (A . B) / (|A| * |B|)
        score = dot_product(query_vector, doc.vector)
        scores.append((score, doc))
        
    # 3. Sort (Rank)
    scores.sort(reverse=True)
    
    # 4. Filter (Optional Hybrid Step)
    # Check for keyword matches if needed
    
    # 5. Return Top K
    return scores[:k]
```

---

## 8. Summary

RAG turns an LLM from a Dreamer into a Librarian.
*   **Vectors** encode meaning.
*   **Vector DBs** enable fast search (HNSW).
*   **Agents** use retrieval as a tool to ground their answers in fact.

However, RAG is only as good as the incoming data. If you feed it messy PDFs, you get messy vectors.
Next, we must look at the input side of this pipeline: **Document Processing**. How do we actually read messy PDFs, Tables, and Charts?
