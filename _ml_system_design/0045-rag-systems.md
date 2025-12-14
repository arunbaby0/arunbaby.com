---
title: "RAG Systems"
day: 45
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - llm
  - retrieval
  - embeddings
  - knowledge
difficulty: Hard
---

**"Grounding LLMs in facts, not hallucinations."**

## 1. Introduction: The Hallucination Problem

**LLMs have limitations:**
*   **Knowledge Cutoff:** Training data has a date limit.
*   **Hallucinations:** Confidently generate false information.
*   **No Private Data:** Can't access your organization's knowledge.

**Solution:** **Retrieval-Augmented Generation (RAG)**—retrieve relevant documents and use them to ground the LLM's response.

## 2. What is RAG?

**RAG = Retrieval + Generation**

1.  **Retrieval:** Find relevant documents from a knowledge base.
2.  **Augmentation:** Add retrieved documents to the LLM prompt.
3.  **Generation:** LLM generates a response based on the context.

**Formula:**
$$P(\text{answer} | \text{query}) = P(\text{answer} | \text{query}, \text{retrieved\_docs})$$

## 3. RAG Architecture

```
                    ┌─────────────────────┐
                    │   User Query        │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Query Embedding   │
                    │   (e.g., OpenAI)    │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Vector Database   │
                    │   (Pinecone, etc.)  │
                    └─────────┬───────────┘
                              │ Top-K docs
                    ┌─────────▼───────────┐
                    │   Prompt Builder    │
                    │   Query + Context   │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   LLM Generation    │
                    │   (GPT-4, Claude)   │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Response          │
                    └─────────────────────┘
```

## 4. Building a RAG System

### Step 1: Document Ingestion

**Collect and preprocess documents:**
```python
from langchain.document_loaders import PDFLoader, WebLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = PDFLoader("knowledge_base.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
```

### Step 2: Embedding and Indexing

**Convert chunks to vectors and store:**
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# Create embeddings
embeddings = OpenAIEmbeddings()

# Store in vector database
vectorstore = Pinecone.from_documents(
    chunks,
    embeddings,
    index_name="knowledge-base"
)
```

### Step 3: Retrieval

**Find relevant chunks for a query:**
```python
def retrieve(query, k=5):
    # Embed query
    query_embedding = embeddings.embed_query(query)
    
    # Search vector database
    results = vectorstore.similarity_search(query, k=k)
    
    return results
```

### Step 4: Generation

**Build prompt and generate response:**
```python
def generate_response(query, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = f"""Answer the question based on the context below.
    
Context:
{context}

Question: {query}

Answer:"""
    
    response = llm.generate(prompt)
    return response
```

## 5. Chunking Strategies

**The quality of chunks significantly impacts RAG performance.**

### 5.1. Fixed-Size Chunking

**Split by character count:**
*   Simple, predictable.
*   May split mid-sentence.

### 5.2. Recursive Chunking

**Split by hierarchy: paragraphs → sentences → words:**
*   Preserves semantic units.
*   Variable chunk sizes.

### 5.3. Semantic Chunking

**Split based on topic changes:**
*   Use sentence embeddings.
*   Group semantically similar sentences.

### 5.4. Agentic Chunking

**Use LLM to determine optimal splits:**
*   "Where would you split this document?"
*   High quality but expensive.

**Best Practice:**
*   Chunk size: 200-500 tokens.
*   Overlap: 10-20% for context preservation.

## 6. Retrieval Strategies

### 6.1. Dense Retrieval

**Use embedding similarity:**
*   Embed query and documents.
*   K-NN search.
*   Good for semantic matching.

### 6.2. Sparse Retrieval (BM25)

**Traditional keyword matching:**
*   TF-IDF scoring.
*   Good for exact matches.

### 6.3. Hybrid Retrieval

**Combine dense and sparse:**
```python
def hybrid_search(query, k=5, alpha=0.5):
    # Dense search
    dense_results = dense_retriever.search(query, k=k)
    
    # Sparse search
    sparse_results = bm25_retriever.search(query, k=k)
    
    # Combine with reciprocal rank fusion
    combined = reciprocal_rank_fusion(dense_results, sparse_results, alpha)
    
    return combined[:k]
```

### 6.4. Reranking

**Use a more powerful model to rerank top candidates:**
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, documents, top_k=5):
    pairs = [(query, doc.content) for doc in documents]
    scores = reranker.predict(pairs)
    
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]
```

## 7. Query Processing

### 7.1. Query Expansion

**Expand query with related terms:**
```python
def expand_query(query):
    prompt = f"Generate 3 related search queries for: {query}"
    expanded = llm.generate(prompt)
    return [query] + expanded
```

### 7.2. Query Decomposition

**Break complex queries into subqueries:**
```python
def decompose_query(query):
    prompt = f"""Break this question into simpler sub-questions:
    Question: {query}
    Sub-questions:"""
    
    subqueries = llm.generate(prompt)
    return parse_subqueries(subqueries)
```

### 7.3. HyDE (Hypothetical Document Embeddings)

**Generate a hypothetical answer, then search for similar documents:**
```python
def hyde_search(query, k=5):
    # Generate hypothetical answer
    hypothetical = llm.generate(f"Answer this question: {query}")
    
    # Search using hypothetical answer
    results = vectorstore.similarity_search(hypothetical, k=k)
    return results
```

## 8. Advanced RAG Patterns

### 8.1. Multi-Hop RAG

**Answer questions requiring multiple retrieval steps:**
1.  Answer subquestion 1 using retrieval.
2.  Use answer to formulate subquestion 2.
3.  Retrieve and answer subquestion 2.
4.  Combine for final answer.

### 8.2. Self-RAG

**LLM decides when to retrieve:**
```python
def self_rag(query):
    # Ask if retrieval is needed
    needs_retrieval = llm.generate(
        f"Does this question need external knowledge? {query}"
    )
    
    if "yes" in needs_retrieval.lower():
        docs = retrieve(query)
        return generate_with_context(query, docs)
    else:
        return llm.generate(query)
```

### 8.3. Corrective RAG

**Verify and correct retrieved information:**
```python
def corrective_rag(query):
    docs = retrieve(query)
    
    # Verify relevance
    relevant_docs = []
    for doc in docs:
        relevance = llm.generate(
            f"Is this relevant to '{query}'? Document: {doc.content}"
        )
        if "yes" in relevance.lower():
            relevant_docs.append(doc)
    
    if not relevant_docs:
        # Fall back to web search
        relevant_docs = web_search(query)
    
    return generate_with_context(query, relevant_docs)
```

## 9. Evaluation Metrics

### 9.1. Retrieval Metrics

*   **Recall@K:** % of relevant docs in top K.
*   **MRR (Mean Reciprocal Rank):** Position of first relevant doc.
*   **NDCG:** Normalized discounted cumulative gain.

### 9.2. Generation Metrics

*   **Faithfulness:** Does the answer match the context?
*   **Relevance:** Does the answer address the query?
*   **Completeness:** Does the answer cover all aspects?

### 9.3. End-to-End Metrics

*   **Answer Accuracy:** Correctness of the final answer.
*   **Latency:** Time from query to response.
*   **User Satisfaction:** Ratings, thumbs up/down.

## 10. Production Considerations

### 10.1. Caching

**Cache frequent queries:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_retrieve(query_hash):
    return retrieve(query)
```

**Semantic caching:** Match similar queries, not just exact.

### 10.2. Streaming

**Stream tokens as they're generated:**
```python
async def stream_rag_response(query):
    docs = await retrieve(query)
    prompt = build_prompt(query, docs)
    
    async for token in llm.stream(prompt):
        yield token
```

### 10.3. Citation

**Include sources in the response:**
```python
prompt = f"""Answer based on the sources. Cite using [1], [2], etc.

Sources:
[1] {doc1.content}
[2] {doc2.content}

Question: {query}
Answer with citations:"""
```

### 10.4. Guardrails

**Prevent harmful outputs:**
*   Content filtering.
*   Fact-checking against sources.
*   Refuse to answer if context is insufficient.

## 11. System Design: Enterprise RAG

**Scenario:** Build a RAG system for internal company documentation.

**Requirements:**
*   Index 100K+ documents.
*   Support multiple file types (PDF, DOCX, HTML).
*   Access control (users see only permitted docs).
*   Real-time updates.

**Architecture:**
```
┌───────────────────────────────────────────────────────┐
│                    API Gateway                        │
│                   (Auth, Rate Limit)                  │
└───────────────────────────┬───────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────┐
│                   RAG Service                         │
│   ┌─────────────┐  ┌─────────────┐  ┌──────────────┐ │
│   │   Query     │  │  Retrieval  │  │  Generation  │ │
│   │  Processor  │→ │   Engine    │→ │    Engine    │ │
│   └─────────────┘  └─────────────┘  └──────────────┘ │
└───────────────────────────┬───────────────────────────┘
                            │
         ┌──────────────────┴──────────────────┐
         │                                      │
┌────────▼────────┐                   ┌────────▼────────┐
│  Vector DB      │                   │  LLM Service    │
│  (Pinecone)     │                   │  (OpenAI/Azure) │
└─────────────────┘                   └─────────────────┘
         │
┌────────▼────────┐
│  Document Store │
│  (S3/GCS)       │
└─────────────────┘
```

## 12. Common Pitfalls

*   **Poor Chunking:** Splitting mid-sentence loses context.
*   **Wrong Embedding Model:** Use domain-specific if available.
*   **Ignoring Metadata:** Filter by date, author, category.
*   **Too Few/Many Retrieved Docs:** 3-5 is usually optimal.
*   **Prompt Stuffing:** Context too long overwhelms LLM.
*   **No Fallback:** Handle "I don't know" gracefully.

## 13. Interview Questions

1.  **What is RAG?** Explain the architecture.
2.  **Dense vs Sparse Retrieval:** When to use each?
3.  **Chunking Strategies:** How do you choose chunk size?
4.  **Evaluation:** How do you measure RAG quality?
5.  **Design:** Build a Q&A system for a legal document database.

## 14. Future Trends

**1. Agentic RAG:**
*   LLM decides when/what to retrieve.
*   Multi-step reasoning with tool use.

**2. Multimodal RAG:**
*   Retrieve images, tables, diagrams.
*   Vision-language models for understanding.

**3. Real-Time RAG:**
*   Index streaming data (news, social media).
*   Sub-second updates to knowledge base.

**4. Personalized RAG:**
*   User-specific retrieval preferences.
*   Learn from interaction history.

## 15. Conclusion

RAG is the bridge between LLMs and real-world knowledge. It solves hallucinations, enables access to private data, and keeps information current.

**Key Takeaways:**
*   **Chunking:** Quality of splits impacts everything.
*   **Retrieval:** Hybrid (dense + sparse) often best.
*   **Reranking:** Improves precision significantly.
*   **Evaluation:** Measure retrieval and generation separately.
*   **Production:** Cache, stream, cite sources.

As LLMs become central to knowledge work, RAG will be the standard pattern for grounding them in facts. Master it to build trustworthy AI systems.

## 16. Advanced: Graph RAG

**Limitation of Vector RAG:** Flat retrieval misses relationships.

**Graph RAG Approach:**
1.  Build a knowledge graph from documents.
2.  Use graph traversal to find related entities.
3.  Combine with vector search.

**Implementation:**
```python
# Build graph during indexing
def build_knowledge_graph(chunks):
    graph = nx.Graph()
    
    for chunk in chunks:
        # Extract entities
        entities = extract_entities(chunk.content)
        
        # Add nodes
        for entity in entities:
            graph.add_node(entity, type=entity.type)
        
        # Add edges between co-occurring entities
        for e1, e2 in combinations(entities, 2):
            graph.add_edge(e1, e2, source=chunk.id)
    
    return graph

# Query with graph context
def graph_rag_query(query, graph, vectorstore, k=5):
    # Vector search
    vector_results = vectorstore.similarity_search(query, k=k)
    
    # Extract entities from query
    query_entities = extract_entities(query)
    
    # Find related entities via graph
    related_entities = set()
    for entity in query_entities:
        if entity in graph:
            neighbors = list(graph.neighbors(entity))[:3]
            related_entities.update(neighbors)
    
    # Retrieve chunks mentioning related entities
    graph_results = get_chunks_for_entities(related_entities)
    
    # Combine results
    all_results = list(set(vector_results + graph_results))
    return all_results
```

**Benefits:**
*   Better handling of relational queries.
*   Explainable retrieval paths.

## 17. Testing RAG Systems

### 17.1. Unit Tests

**Retrieval Tests:**
```python
def test_retrieval_returns_relevant_docs():
    query = "What is machine learning?"
    docs = retrieve(query, k=5)
    
    assert len(docs) == 5
    assert any("machine learning" in doc.content.lower() for doc in docs)

def test_chunking_preserves_sentences():
    text = "First sentence. Second sentence."
    chunks = chunk_text(text, chunk_size=50)
    
    for chunk in chunks:
        # Each chunk should have complete sentences
        assert not chunk.startswith(" ")
        assert chunk.endswith(".")
```

### 17.2. Integration Tests

```python
def test_end_to_end_rag():
    query = "What are the benefits of RAG?"
    
    # Should return a coherent answer with citations
    response = rag_system.query(query)
    
    assert len(response.answer) > 100
    assert len(response.sources) > 0
    assert "retrieval" in response.answer.lower()
```

### 17.3. Evaluation Datasets

**Create a golden dataset:**
*   Questions with known answers.
*   Expected source documents.
*   Use for regression testing.

## 18. Cost Analysis

**Cost Components:**
1.  **Embedding:** $0.0001 per 1K tokens (OpenAI).
2.  **Vector Storage:** $0.025 per GB/month (Pinecone).
3.  **LLM Generation:** $0.01-0.10 per 1K tokens.

**Example (100K documents, 1M queries/month):**
```
Embedding (one-time): 100K × 500 tokens × $0.0001/1K = $5
Storage (monthly): 100K × 1KB = 100MB = $0.003
Generation: 1M × 1K tokens × $0.03/1K = $30,000

Total monthly: ~$30,000 (dominated by generation)
```

**Optimization:**
*   Cache frequent queries.
*   Use smaller models for simple questions.
*   Batch embedding generation.

## 19. LangChain Implementation

**Complete RAG Pipeline:**
```python
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = DirectoryLoader('data/', glob='**/*.pdf')
documents = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name="rag-demo")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Query
result = qa_chain({"query": "What is RAG?"})
print(f"Answer: {result['result']}")
print(f"Sources: {[doc.metadata for doc in result['source_documents']]}")
```

## 20. LlamaIndex Implementation

**Alternative Framework:**
```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI

# Load documents
documents = SimpleDirectoryReader('data/').load_data()

# Create index
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(),
)

# Query
query_engine = index.as_query_engine(
    llm=OpenAI(model="gpt-3.5-turbo"),
    similarity_top_k=5
)

response = query_engine.query("What is RAG?")
print(response)
```

## 21. Handling Updates

**Challenge:** Documents change over time.

**Strategies:**
1.  **Full Reindex:** Rebuild entire index (simple but slow).
2.  **Incremental Updates:** Add/update/delete individual documents.
3.  **Versioning:** Keep multiple versions, timestamp queries.

**Implementation:**
```python
class RAGIndex:
    def __init__(self):
        self.vectorstore = Pinecone(index_name="documents")
    
    def add_document(self, doc):
        chunks = self.chunk(doc)
        embeddings = self.embed(chunks)
        self.vectorstore.upsert(embeddings, doc.id)
    
    def update_document(self, doc):
        self.delete_document(doc.id)
        self.add_document(doc)
    
    def delete_document(self, doc_id):
        self.vectorstore.delete(filter={"doc_id": doc_id})
```

## 22. Mastery Checklist

**Mastery Checklist:**
- [ ] Explain RAG architecture
- [ ] Implement chunking strategies
- [ ] Build a vector index (Pinecone, Chroma)
- [ ] Implement hybrid retrieval (dense + BM25)
- [ ] Add reranking with cross-encoder
- [ ] Implement query expansion/decomposition
- [ ] Add citation to responses
- [ ] Handle document updates
- [ ] Measure retrieval and generation quality
- [ ] Deploy with caching and streaming

## 23. Conclusion

RAG is the most important pattern for production LLM applications. It transforms LLMs from unreliable knowledge sources into grounded, trustworthy systems.

**The RAG Stack:**
1.  **Documents:** Your knowledge base.
2.  **Chunking:** Transform into searchable units.
3.  **Embedding:** Vector representation.
4.  **Vector Store:** Efficient similarity search.
5.  **Retrieval:** Find relevant context.
6.  **Generation:** LLM produces grounded answer.

Every step matters. Poor chunking cascades to poor retrieval. Poor retrieval cascades to poor answers. Master each component to build world-class RAG systems.

**The future is RAG + Agents:** Systems that not only retrieve but reason, plan, and take action based on retrieved knowledge. Start with RAG fundamentals, then explore the agentic frontier.

