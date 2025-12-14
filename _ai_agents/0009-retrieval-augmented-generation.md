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
  - knowledge-retrieval
difficulty: Medium-Easy
---

**"Giving the Brain a Library: The Foundation of Knowledge-Intensive Agents."**

## 1. Introduction: The Hallucination Problem

LLMs are "Frozen-in-Time" encyclopedias. GPT-4 knows everything about the world up to its training cutoff (e.g., 2023). It knows nothing about:
1.  **Recent Events:** The winner of the Super Bowl yesterday.
2.  **Private Data:** Your company's internal wiki, your emails, or your proprietary codebase.

When you ask an agent about these topics, it faces a dilemma: admit ignorance (rare) or make something up (common). This is **Hallucination**.

**Retrieval-Augmented Generation (RAG)** is the architecture that solves this. Instead of relying on the model's *internal* weights (Parametric Memory), we allow it to look up information in an *external* database (Non-Parametric Memory) before answering.

For Agents, RAG is not just a feature; it is a **Core Organ**. It is the bridge between reasoning and reality.

---

## 2. The Mechanics of RAG

RAG is a pipeline. It consists of two phases: **Indexing** (Offline) and **Retrieval** (Runtime).

### 2.1 Phase 1: Indexing (The ETL Pipeline)
How do we turn a PDF into numbers?

1.  **Loading:** Ingest data. Text, PDFs, Markdown, HTML.
2.  **Splitting (Chunking):** LLMs have context limits. We can't feed a 500-page book in one go. We split it into smaller "Chunks" (e.g., 500 tokens).
    *   *Challenge:* Don't split in the middle of a sentence. Keep semantic meaning intact (e.g., keep a question and its answer in the same chunk).
3.  **Embedding:** We pass each chunk through an **Embedding Model** (e.g., OpenAI `text-embedding-3-small`).
    *   *Input:* "The sky is blue."
    *   *Output:* `[0.12, -0.45, 0.88, ...]` (A vector of 1536 floating point numbers).
    *   *Concept:* This vector represents the **semantic meaning** of the text. "Dog" and "Puppy" will have vectors that are close together (high Cosine Similarity).
4.  **Storing:** Save the Vectors + Original Text in a **Vector Database** (Pinecone, Weaviate, Chroma).

### 2.2 Phase 2: Retrieval (The Agent Loop)
When the user asks a question:

1.  **Query Embedding:** User asks "What is our vacation policy?". We embed this query into a vector.
2.  **Semantic Search:** We query the Vector DB: "Find the top 3 chunks closest to this query vector."
    *   *Result:* Chunk A ("Vacation Policy..."), Chunk B ("Leave types...").
3.  **Context Injection:** We construct a prompt:
    ```text
    System: Answer the user using ONLY the context below.
    Context:
    - Vacation Policy: 20 days per year...
    - Leave types: Sick, Annual...
    
    User: What is our vacation policy?
    ```
4.  **Generation:** The LLM reads the context and answers "You get 20 days pear year."

---

## 3. Naive RAG vs. Advanced RAG

The pipeline above is "Naive RAG." It works for simple demos but fails in production. Why?
*   **Keyword Misses:** "Vacation" vs "Time Off." Vectors handle synonyms well, but specific jargon (Product IDs) fails.
*   **Loss of Context:** A chunk saying "It costs $5" is useless if you don't know *what* "It" refers to (which was in the previous chunk).

### 3.1 Advanced Techniques
1.  **Hybrid Search:** Combine Vector Search (Semantic) with Keyword Search (BM25).
    *   *Why:* Ensure exact matches for IDs and Names work.
2.  **Re-Ranking (The Cross-Encoder):**
    *   Retrieve top 50 results (fast & loose).
    *   Pass them through a powerful "Re-Ranker Model" (slow & precise) to sort them by accurate relevance.
    *   Take the top 5.
3.  **Parent-Child Indexing:**
    *   *Index:* Small chunks (sentences) for precise search.
    *   *Retrieve:* The Parent chunk (full paragraph) for context.

---

## 4. Making RAG a Tool

For an agent, RAG is just another **Tool**.
We don't hard-code the retrieval. We give the agent a tool called `search_knowledge_base(query)`.

### The Agent's Decision Process
1.  User: "Hi." -> Agent: "Hi." (No retrieval needed).
2.  User: "How do I reset my password?" -> Agent Thought: "This is a technical question. I should check the docs." -> Agent Action: `search_knowledge_base("password reset")`.

### Router Engines
Sometimes you have multiple databases (Sales DB, HR DB, Codebase).
You create a **Router Query Engine**.
*   Agent analyzes query.
*   Routes to `HR_Index` or `Sales_Index`.

---

## 5. Code: A Minimal Vector Search

Using `scikit-learn` for simplicity (in memory).

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Mock Database
documents = [
    "Apple is a fruit.",
    "Apple is a tech company.",
    "Bananas are yellow."
]
# Mock Embeddings (Usually from API)
embeddings = np.array([
    [1, 0, 0],   # Fruit
    [0, 1, 0],   # Tech
    [0.9, 0, 0]  # Banana (close to fruit)
])

def search(query_vec):
    # 2. Similarity
    similarities = cosine_similarity([query_vec], embeddings)
    
    # 3. Top-K
    top_k_idx = np.argmax(similarities)
    return documents[top_k_idx]

# Query: "iPhone" -> Vector [0, 0.9, 0]
print(search([0, 0.9, 0.1])) 
# Output: "Apple is a tech company."
```

---

## 6. Summary

RAG turns an LLM from a Dreamer into a Librarian.
*   **Vectors** encode meaning.
*   **Vector DBs** enable fast search.
*   **Agents** use retrieval as a tool to ground their answers in fact.

In the next post, we will look at the input side of this pipeline: **Document Processing**. How do we actually read messy PDFs, Tables, and Charts?
