---
title: "Agent Workflow Patterns"
day: 8
collection: ai_agents
categories:
 - ai-agents
tags:
 - architecture
 - patterns
 - reflection
 - planning
 - andrew-ng
 - multi-agent
difficulty: Medium-Easy
related_dsa_day: 8
related_ml_day: 8
related_speech_day: 8
---

**"Better workflows beat better models." — Dr. Andrew Ng**

## 1. Introduction: The Workflow Revolution

For a long time, the obsession in AI was "Bigger Models." To get better answers, everyone thought we needed GPT-5, GPT-6, and GPT-7. The assumption was that intelligence scales primarily with parameter count.

However, in 2024, a quiet revolution occurred. Research highlighted by Andrew Ng and teams from Stanford/Princeton showed that **Agentic Workflows**—wrapping a smaller model (like GPT-3.5 or Llama-3) in a smart cognitive loop—could outperform a larger model (GPT-4) running zero-shot.

The paradigm has shifted. We are no longer just "Prompting" a model to guess the right answer immediately; we are "Flow Engineering" a system to reason its way to the answer.
In this post, we will dissect the four canonical design patterns for agentic systems: **Reflection, Tool Use, Planning, and Multi-Agent Collaboration.**

---

## 2. Pattern 1: Reflection (The Editor Loop)

Humans rarely write perfect code or perfect essays on the first try. We write a draft, look at it, spot errors, and rewrite.
Standard naive LLM usage asks for perfection in one shot. This is unnatural. **Reflection** mimics the human edit loop.

### 2.1 The Architecture
1. **Actor:** Generates the initial output (e.g., a Python function).
2. **Critic:** Prompts the model to review the output *without* rewriting it yet. "Look for bugs, logic errors, or style issues."
3. **Reviser:** Takes the *Original Output* and the *Critique* and produces the *Final Output*.

### 2.2 Case Study: Coding (HumanEval)
On the HumanEval benchmark (Python coding problems):
* **GPT-4 Zero-Shot:** ~67% accuracy.
* **GPT-4 + Reflexion:** ~88% accuracy.
* *Mechanism:* The model knows how to catch its own mistakes if you ask it to look specifically for mistakes. It cannot do "Generation" and "Verification" in the same forward pass because generation is autoregressive (left-to-right). It hasn't seen the end of the line before it writes the start. Reflection provides a second pass.

### 2.3 Pseudo-Code Implementation
``python
def reflection_loop(goal):
 # Pass 1: Generation
 draft = llm.generate(goal)

 # Pass 2: Verification
 critique = llm.generate(f"Critique this code for bugs: {draft}")

 # Pass 3: Correction
 if "No bugs found" in critique:
 return draft
 else:
 final = llm.generate(f"Fix the code based on critique: {draft} \n Feedback: {critique}")
 return final
``

---

## 3. Pattern 2: Tool Use (The ReAct Loop)

This is the pattern where the LLM stops being a "Know-it-all" and becomes a "Seek-it-all."

### 3.1 The Architecture
* **Thought:** The model reasons about what is missing.
* **Action:** The model calls an external function.
* **Observation:** The system returns data.
* **Repeat.**

### 3.2 Key Insight: "Cognitive Offloading"
Tool use isn't just about accessing data; it's about offloading computation.
* *Task:* "What is 329 * 482?"
* *LLM:* "158,578" (Hallucination - doing math in weights is hard).
* *Agent:* Calls `calculator(329, 482)`. Returns `158578` (Fact).
By offloading deterministic tasks to deterministic tools, we save the model's "Token Budget" for reasoning rather than computation.

---

## 4. Pattern 3: Planning (Reasoning then Acting)

For complex tasks ("Build a videogame" or "Research the impact of AI on healthcare"), diving straight into execution (ReAct) leads to chaos. The agent gets lost in the details of the first step and forgets the macro-goal.
**Planning** separates **Strategy** from **Execution**.

### 4.1 The Architecture (Plan-and-Solve)
1. **Planner Agent:**
 * *Input:* "Build a Snake game."
 * *Output:* A structured list.
 1. Setup Pygame window.
 2. Create Snake class.
 3. Create Food class.
 4. Handle collision logic.
2. **Executor Agent:**
 * Takes the Plan.
 * Executes Step 1.
 * Executes Step 2.
 * ...
3. **Replanner (Optional):**
 * If Step 3 fails ("Pygame library not found"), the Replanner updates the plan ("Install pygame first") dynamically.

### 4.2 Dynamic vs. Static Planning
* **Static:** Generate 5 steps. Do 5 steps blindly. (Fragile).
* **Dynamic:** Generate 5 steps. Do Step 1. Look at the result. Re-generate the remaining 4 steps based on the new world state. (Robust).

---

## 5. Pattern 4: Multi-Agent Collaboration

Why ask one brain to do everything? A single "Generalist" system prompt ("You are helpful") often fails at niche tasks.
Specialization reduces token confusion.

### 5.1 Hierarchical (The Boss & Workers)
* **Manager Agent:** Breaks down the user goal and delegates.
* **Worker A (Researcher):** Only has `search_tool`.
* **Worker B (Writer):** Only has `text_editor`.
* *Flow:* User -> Manager -> Researcher -> Manager -> Writer -> Manager -> User.
* *Benefit:* Encapsulation. The Writer doesn't need to see the messy search logs. It just sees the clean notes. This keeps the context window clean.

### 5.2 Joint Collaboration (The Debate)
* **Persona A:** "You are an optimistic feature designer."
* **Persona B:** "You are a cynical security engineer."
* *Task:* "Design a login system."
* *Flow:* A proposes. B critiques ("That's insecure"). A revises. B critiques.
* *Benefit:* Reducing hallucinations. The "Cynic" acts as a filter, cancelling out the "Optimist's" tendency to invent features that don't exist.

### 5.3 Sequential Handoffs (The Assembly Line)
* **Agent A:** Scrapes data. Output: JSON. ->
* **Agent B:** Analyzes JSON. Output: Summary. ->
* **Agent C:** Writes Email. Output: Text.
* *Benefit:* Extremely distinct contexts. Agent C doesn't even know Agent A exists.

---

## 6. The "Fifth Pattern": Evaluation (LLM-as-a-Judge)

There is a hidden workflow that powers all of these: **Automated Evaluation**.
How do you know if your Reflection loop is working? You need a Judge.

### The Architecture
1. **Input:** User Query + Agent Response.
2. **Judge Prompt:** "On a scale of 1-5, how accurate is this response? output JSON."
3. **Model:** GPT-4.

This allows you to grid-search your prompts and workflows. "Does adding a 'Critique' step improve the 'Judge Score'?"

---

## 7. Choosing Your Pattern

This decision matrix guides your architecture.

| Complexity | Pattern | Example |
| :--- | :--- | :--- |
| **Simple** | **Zero-Shot** | "Tell me a joke." "Classify this email." |
| **Medium** | **Reflection** | "Write a poem, then improve it." "Write code." |
| **Hard** | **ReAct (Tools)** | "What is the stock price of Apple?" "Query the DB." |
| **Very Hard** | **Planning** | "Write a Snake game in Python." "Research a new topic." |
| **Complex** | **Multi-Agent** | "Run a marketing campaign involved research, writing, and posting." |

---

## 8. Summary: Flow Engineering

The job of the AI Engineer is shifting. We are no longer just "Prompting." We are designing **Cognitive Architectures**.
* **Reflection** makes agents reliable.
* **Planning** makes agents capable of long horizons.
* **Multi-Agent** makes agents specialized.

To build agents that master specific domains, we must first master **RAG and Document Intelligence**, which allows agents to read and retrieve external knowledge.


---

**Originally published at:** [arunbaby.com/ai-agents/0008-agent-workflow-patterns](https://www.arunbaby.com/ai-agents/0008-agent-workflow-patterns/)

*If you found this helpful, consider sharing it with others who might benefit.*

