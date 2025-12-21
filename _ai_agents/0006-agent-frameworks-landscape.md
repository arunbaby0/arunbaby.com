---
title: "Agent Frameworks Landscape"
day: 6
collection: ai_agents
categories:
  - ai-agents
tags:
  - frameworks
  - langchain
  - langgraph
  - autogen
  - crewai
  - llamaindex
difficulty: Easy
related_dsa_day: 6
related_ml_day: 6
related_speech_day: 6
---

**"To Framework or Not to Framework? Navigating the Agent Ecosystem."**

## 1. Introduction: The Wild West of Agentic AI

In 2023, building an agent was a simple affair: you wrote a `while True` loop in Python, appended strings to a list called `messages`, and called the OpenAI API. It was raw, messy, and understandable.

Fast forward to today, and we are drowning in frameworks. **LangChain, LangGraph, AutoGen, CrewAI, LlamaIndex, Semantic Kernel, MetaGPT...** the ecosystem has exploded. Every week, a new library promises to be the "Rails for Agents."

For a developer, this **Paralysis of Choice** is dangerous. Choosing the wrong framework can lock you into a rigid architecture, force you to learn obscure abstractions that wrap simple API calls, and essentially become "Technical Debt as a Service." Conversely, refusing to use frameworks can leave you re-implementing basic utilities (like PDF parsing or retry logic) for weeks.

In this comprehensive landscape analysis, we will map the Agent Framework Ecosystem. We won't just list features; we will analyze the **Philosophy** behind each framework, their abstraction costs, their "Opinionatedness," and ultimately, when you should (or shouldn't) use them.

---

## 2. The Abstraction Layers

To really understand the landscape, we must visualize the layers of abstraction. Not all "frameworks" are solving the same problem. Some are utilities; others are full-blown operating systems.

### 2.1 Level 0: Raw Code (The Control Freak)
* **Tech:** Python `requests`, `openai` SDK (`pip install openai`).
* **Philosophy:** "I want to see the prompt. I want to control the bytes."
* **Pros:**
 * **Infinite Flexibility:** You are limited only by Python and the API.
 * **Zero Overhead:** No latency from wrapper libraries.
 * **Debuggability:** When it breaks, you know exactly where. There is no "magic" happening behind the scenes.
* **Cons:**
 * **Boilerplate:** You reinvent the wheel constanty. You have to write your own recursive retry logic, your own JSON parser, your own token counter.
 * **Maintenance:** As APIs change (e.g., OpenAI moving from `Function Calling` to `Tools`), you have to refactor everything manually.
* **Verdict:** Best for production engineers building high-performance, strictly defined tools where reliability is paramount.

### 2.2 Level 1: The Utilities (The Standard Library)
* **Tech:** **LangChain Core**, **LlamaIndex (Core)**.
* **Philosophy:** "Give me tools for the boring stuff, but let me write the logic."
* **Capabilities:**
 * **Loaders:** "Read this PDF/notion/slack."
 * **Splitters:** "Chunk this text into 500-token logical blocks."
 * **Vector Connectors:** "Talk to Pinecone/Chroma."
* **Pros:** Saves massive time on commodity tasks (ETL).
* **Cons:** The "Leaky Abstraction" problem. Sometime the specific way LangChain chunks text isn't what you want, and overriding it is harder than writing it yourself.

### 2.3 Level 2: The Orchestrators (The Graphs)
* **Tech:** **LangGraph**, **LlamaIndex Workflows**.
* **Philosophy:** "Agents are State Machines. Let me define the nodes (functions) and edges (logic)."
* **Mechanism:** These frameworks force you to define a **Directed Acyclic Graph (DAG)** or, more commonly for agents, a **Cyclic Graph**.
* **Pros:**
 * **Structure:** Enforces discipline. You can't just have spaghetti code; you must define state transitions.
 * **Persistence:** They often come with "Checkpointers" that save the state of the graph to a database after every step. This allows for "Time Travel" debugging.
* **Cons:** High cognitive load. You have to think in graphs.

### 2.4 Level 3: The Multi-Agent Platforms (The Swarms)
* **Tech:** **AutoGen (Microsoft)**, **CrewAI**.
* **Philosophy:** "Agents are people. Let them talk to each other."
* **Mechanism:** You define "Personas" and "Conversation Policies." The framework handles the message passing.
* **Pros:**
 * **Emergent Behavior:** You can get complex results with very little code. "Here is a Coder Agent and a Reviewer Agent. Goal: Fix this bug."
* **Cons:**
 * **Non-Determinism:** It's hard to control exactly what happens. Agents might chat endlessly.
 * **Cost:** "Chatter" consumes tokens.

---

## 3. Deep Dive: The Big Players

Let's dissect the specific frameworks dominating the market in 2025.

### 3.1 LangGraph (The Industry Standard)
LangChain realized that its original "Chain" abstraction (Sequence A -> Sequence B) was too rigid for agents, which loop and branch. They pivoted to **LangGraph**.

* **Core Concept:** **Stateful Graph**.
 * There is a shared `State` object (a Python dictionary/TypedDict).
 * **Nodes** are Python functions that take the State, modify it, and return an update.
 * **Edges** define where to go next (e.g., `conditional_edge(check_output)`).
* **The "Human-in-the-Loop" Feature:** Because LangGraph saves the state after every node execution (using a Checkpointer), you can pause execution.
 * *Scenario:* Agent reaches "Execute Code" node. Graph pauses. Human Admin gets a ping. Human approves. Graph resumes.
 * This is critical for enterprise safety.
* **Code Snippet (Conceptual):**
 ```python
 # Define State
 class State(TypedDict):
 messages: list

 # Define Graph
 workflow = StateGraph(State)
 workflow.add_node("agent", call_llm)
 workflow.add_node("tool", run_tool)

 # Define Logic
 workflow.set_entry_point("agent")
 workflow.add_conditional_edges("agent", should_continue)
 workflow.add_edge("tool", "agent")

 app = workflow.compile()
 ```

### 3.2 AutoGen (The Conversationalist)
Developed by Microsoft Research, AutoGen takes a different approach. It treats everything as a "Agent" that can send/receive messages.

* **Core Concept:** **UserProxy and Assistant**.
 * **AssistantAgent:** The LLM. It suggests plans and code.
 * **UserProxyAgent:** A proxy for the human (or a system execution environment). It can **execute code** locally or in Docker.
* **The Magic:** AutoGen excels at **Code Generation**.
 * *Step 1:* Assistant writes Python code to plot a chart.
 * *Step 2:* UserProxy detects the code block, executes it (automatically!), and returns the result (or the error trace) to the Assistant.
 * *Step 3:* Assistant fixes the error.
* **Use Case:** Data Science. "Here is a csv. Analyze it." AutoGen will write pandas code, run it, fix errors, and generate the final plot, all with 0 human intervention.

### 3.3 CrewAI (The Role-Player)
CrewAI is built on top of LangChain but simplifies the API into a "Team" metaphor.

* **Core Concept:** **Processes**.
 * **Agents:** Defined with `Role`, `Goal`, `Backstory`. (e.g., "You are a veteran journalist.")
 * **Tasks:** Specific units of work assigned to agents.
 * **Process:** How they work together.
 * *Sequential:* A -> B -> C.
 * *Hierarchical:* A Manager Agent assigns tasks to A and B, reviews work, and delegates.
* **Why people love it:** It is incredibly readable. The code looks like an org chart.
* **Critique:** It can be slow. The "Manager" LLM adds latency and cost as it orchestrates everything.

### 3.4 LlamaIndex (The Data Expert)
LlamaIndex started as "GPT Index," a tool to connect LLMs to your data. It has evolved into a full agent framework.

* **Core Concept:** **RAG-First Agents**.
 * While other frameworks focus on generic tools, LlamaIndex focuses on **Query Engines**.
 * **Workflow:** Their new event-driven workflow engine allows you to build agents that are triggered by data events (e.g., "New file added to folder").
 * **Context Management:** LlamaIndex has the best algorithms for **Token Packing** and **Chunking Optimization**.
* **Use Case:** If your agent's primary job is reading 500 PDF contracts and answering questions about them, LlamaIndex is the superior choice.

---

## 4. Architectural Patterns: Graph vs. Conversation

When choosing a framework, you are choosing an architecture.

### 4.1 The Graph Pattern (LangGraph)
* **Structure:** Explicit State Machine.
* **Control:** High. You explicitly define every transition. "If A succeeds, go to B. If A fails, go to C."
* **Reliability:** High. It behaves predictably.
* **Development Speed:** Slower. You have to define the graph structure.

### 4.2 The Conversational Pattern (AutoGen)
* **Structure:** Free-form Chat.
* **Control:** Low. The LLM decides who speaks next based on the conversation history.
* **Reliability:** Lower. Agents might get stuck in "Politeness loops" ("Thank you!" "No, thank you!") or fail to hand off tasks correctly.
* **Development Speed:** Fast. Just define agents and say "Chat."

---

## 5. Decision Matrix: Which one should you choose?

Here is a guide for the perplexed engineer in 2025.

| Scenario | Recommendation | Why? |
| :--- | :--- | :--- |
| **Simple "Chat with PDF"** | **LlamaIndex** | Best data connectors and chunking logic. RAG is their bread and butter. |
| **Production Enterprise SaaS** | **LangGraph** | You need strict state management, "Human-in-the-loop" approval, and unit testing. You can't afford non-determinism. |
| **Experimental Data Analysis** | **AutoGen** | Best code execution sandbox. It writes and runs code better than anything else. |
| **Creative Content / Marketing** | **CrewAI** | Role-playing abstraction is perfect for creative tasks where "style" matters multiple agents (Writer, Editor) improve quality. |
| **High-Performance Micro-Agent** | **Raw Python / OpenAI SDK** | Don't pay the latency tax of a framework. If looking for a "Router," just write the `if` statements. |

---

## 6. The "No-Framework" Manifesto

Many senior AI engineers advocate for avoiding frameworks entirely, especially in the beginning. This is often called the "Hamal" approach (after Hamel Husain's famous critique).

* **The Argument:** Agent frameworks add layers of **"Prompt Magic"**. They often inject hidden system prompts ("You are a helpful agent...") that you can't see or change easily. This interferes with your ability to prompt engineer specifically for your use case.
* **The Debugging Nightmare:** When a LangChain agent fails, the stack trace goes through 15 layers of abstraction. Debugging `raw_python.py` is trivial; debugging `AgentExecutor.run()` is hell.
* **The Strategy:**
 1. Start with raw Python. Build your `chat_loop`. Handle your `tool_call`.
 2. Write your own `Tool` class (it's 10 lines of code).
 3. Only adopt a framework (like LangGraph) when you find yourself reinventing state persistence and graph execution for the 3rd time.

---

## 7. Summary

The framework landscape is consolidating around two poles:
1. **Graphs (LangGraph):** For engineers who want control, state, and reliability.
2. **Swarms (AutoGen/CrewAI):** For researchers who want capability, emergence, and conversation.

There is no "Best Framework." There is only the right level of abstraction for your problem.
* Building a **Banking Bot**? Use LangGraph (Control).
* Building a **Stock Research Bot**? Use AutoGen (Code Execution).
* Building a **Story Writer**? Use CrewAI (Creativity).

With the landscape understood, the next step is to **Build Your First Agent** from scratch, using raw Python to see exactly what these frameworks hide from you.


---

**Originally published at:** [arunbaby.com/ai-agents/0006-agent-frameworks-landscape](https://www.arunbaby.com/ai-agents/0006-agent-frameworks-landscape/)

*If you found this helpful, consider sharing it with others who might benefit.*

