---
title: "Memory Architectures"
day: 5
collection: ai_agents
categories:
  - ai-agents
tags:
  - memory
  - long-term-memory
  - vector-db
  - generative-agents
  - memgpt
difficulty: Easy
related_dsa_day: 5
related_ml_day: 5
related_speech_day: 5
---

**"The difference between a Chatbot and a Partner is Memory."**

## 1. Introduction: The Amnesia Problem

LLMs are fundamentally **stateless**. If you send a request to standard GPT-4, it processes it and immediately "forgets" it. The only state that exists is the **Context Window** you physically pass in with every call.

For a chatbot session, this is manageable. You pass the chat history `[(User, "Hi"), (Bot, "Hi")]` in every turn.
But for an **Autonomous Agent** running for days, handling thousands of steps, this breaks down.
1. **Cost:** Passing 128k tokens per step is prohibitively expensive.
2. **Capacity:** Even 1M tokens isn't enough for a whole lifetime of experiences.
3. **Focus:** "Context Stuffing" reduces intelligence. If you feed the model too much noise (irrelevant history), it hallucinates or gets distracted.

To build true agency, we need to engineer a **Memory Hierarchy** similar to the human brain, managing the flow of information from short-term perception to long-term storage.

---

## 2. The Cognitive Memory Hierarchy

Just like computers have layers (Registers -> RAM -> Hard Drive), agents utilize different tiers of memory to balance speed, cost, and capacity.

### 2.1 Sensory Memory (The Input)
The raw user prompt and System Instructions.
* *Capacity:* Limited by the immediate Context Window logic.
* *Mechanism:* Direct Prompt Injection.
* *Role:* Immediate perception of the "Now."

### 2.2 Short-Term / Working Memory (Context)
This tracks the *current* conversation or active task. It holds the "Scratchpad" of thoughts.
* **The Challenge:** The "Context Stuffing" problem.
* **Strategy 1: Sliding Window.** Keep last $N$ turns. Simple, but forgets the beginning of the plan.
* **Strategy 2: Summarization.** As the window fills, call the LLM to summarize the oldest 10 turns into a paragraph ("I successfully downloaded the data and cleaned it"). Inject this summary and drop the raw logs.
* **Strategy 3: Entity Extraction.** Extract specific variables ("User Name: Alice", "Goal: Fix Bug") and store them in a state dictionary, reducing the text needed to purely essential facts.

### 2.3 Long-Term Memory (Episodic)
Storage that persists *across* sessions. "What did we do last Tuesday?"
* **Implementation:** **Vector Databases (RAG)** (Pinecone, Milvus).
* **Mechanism:**
 1. Event happens ("User reset password").
 2. Embed text -> Vector.
 3. Store metadata `{"date": "2023-10-01", "type": "auth"}`.
 4. Retrieve when relevant.

### 2.4 Semantic Memory (World Knowledge)
Facts about the world, distinct from events. "The user uses Python 3.10."
* **Implementation:** **Knowledge Graphs** (Neo4j) or Structured SQL.
* **Why:** Vectors are fuzzy. If you ask "Who is Alice's manager?", a vector search might return "Alice manages Bob" (similar words, wrong relationship). A Graph query `(Alice)-[:REPORTS_TO]->(Manager)` is precise.

---

## 3. The "Generative Agents" Architecture (The Gold Standard)

In 2023, Stanford/Google researchers published *"Generative Agents: Interactive Simulacra of Human Behavior"*. They created "Smallville," a simulation where 25 AI agents lived in a town. They remembered relationships, planned parties, and gossiped.
Their memory architecture is the blueprint for **Human-Like Memory**.

### 3.1 The Memory Stream
Every observation is a distinct object in a time-ordered list.
* `[09:00] Observed Alice drinking coffee.`
* `[09:05] Observed Bob walking in.`
* `[09:06] Alice said "Hi Bob".`

### 3.2 The Retrieval Function
How do we decide what to remember? They used a weighted score of 3 factors to retrieve top memories for a given query:
$$ Score = \alpha \cdot \text{Recency} + \beta \cdot \text{Importance} + \gamma \cdot \text{Relevance} $$

1. **Recency:** Exponential decay. I care about what happened 5 minutes ago more than 5 years ago.
 * $Score = 0.99^{\text{decay\_hours}}$
2. **Importance:** A static score separating "Noise" from "Signal". "Ate toast" (1/10). "House on fire" (10/10).
 * *Implementation:* Ask the LLM to rate the importance of every new memory on ingestion.
3. **Relevance:** The standard Cosine Similarity to the module's current query.

### 3.3 The Reflection Tree (Synthesizing Wisdom)
If you only store raw logs, the agent never "learns." It just remembers details.
* *Process:* Periodically (e.g., every 100 observations), the agent takes a batch of memories and asks: *"What does this mean?"*
* *Input:* "Alice drank coffee Mon", "Alice drank coffee Tue", "Alice drank coffee Wed".
* *Insight:* "Alice is addicted to coffee."
* *Action:* Save this **Insight** as a new memory node.
* *Result:* Future queries retrieve the Insight, not the 50 raw logs. This mocks human generalization.

---

## 4. MemGPT: LLM as an Operating System

**MemGPT** (Memory-GPT) proposes a different analogy.
* **LLM Context = RAM.** (Fast, expensive, volatile).
* **Vector DB = Hard Drive.** (Slow, cheap, persistent).
* **Paging:** The OS (Prompt logic) manages "Virtual Memory." It swaps data in and out of the Context Window based on need.
* **Mechanism:** The LLM itself decides what to "write to disk" (save to DB) and what to "read from disk" (search DB) via function calls.
* **Impact:** Enables agents to run "forever" (infinite context) by actively managing their own memory slots, just `malloc` and `free`.

---

## 5. Procedural Memory: The Skill Library (Voyager)

How does an agent "learn to code"?
* **Naive Agent:** Writes code -> Fails -> Rewrites -> Succeeds -> Forgets.
* **Voyager Agent (Minecraft):** Writes code -> Fails -> Rewrites -> Succeeds -> **Saves Function to Disk**.
* **Skill Retrieval:** Next time the goal is "Mine Diamond", it checks its `skills/` folder. "Ah, I have a `mine_block` function."
* **Result:** The agent gets faster and more capable over time, building a library of tools it wrote itself. This is akin to "Muscle Memory."

---

## 6. Code: Implementing Generative Retrieval (Conceptual)

The formula for the Stanford retrieval function.

```python
def retrieve_memories(query_vector, memory_stream, alpha=1, beta=1, gamma=1):
 """
 Ranks memories by Recency, Importance, and Relevance.
 """
 scored_memories = []

 current_time = now()

 for memory in memory_stream:
 # 1. Relevance: Cosine Similarity
 relevance = cosine_similarity(query_vector, memory.vector)

 # 2. Recency: Exponential Decay
 time_diff = current_time - memory.timestamp
 recency = 0.99 ** time_diff.hours

 # 3. Importance: Pre-calculated static score (1-10)
 importance = memory.importance_score / 10.0

 # Combined Score
 total_score = (alpha * recency) + (beta * importance) + (gamma * relevance)

 scored_memories.append((total_score, memory))

 # Python's sort is stable
 scored_memories.sort(key=lambda x: x[0], reverse=True)

 return [m[1] for m in scored_memories[:3]]
```

---

## 7. Summary

Memory is the bedrock of identity.
* **Short-term memory** allows for coherent conversation.
* **Long-term memory** allows for personalization.
* **Reflection** allows for learning.

Without robust memory architectures, AI agents will forever be "Goldfish"—brilliant in the moment, but incapable of growth. In the next section, we move from components to architectures, looking at the major frameworks like LangChain and AutoGen.


---

**Originally published at:** [arunbaby.com/ai-agents/0005-memory-architectures](https://www.arunbaby.com/ai-agents/0005-memory-architectures/)

*If you found this helpful, consider sharing it with others who might benefit.*

