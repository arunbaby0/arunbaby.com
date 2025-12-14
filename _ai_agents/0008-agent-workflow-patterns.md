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
---

**"Better workflows beat better models." — Dr. Andrew Ng**

## 1. Introduction: The Workflow Revolution

For a long time, the obsession in AI was "Bigger Models." To get better answers, everyone thought we needed GPT-5, GPT-6, and GPT-7. The assumption was that intelligence scales with parameter count.

However, in 2024, a quiet revolution occurred. Research highlighted by Andrew Ng and teams from Stanford/Princeton showed that **Agentic Workflows**—wrapping a smaller model (like GPT-3.5 or Llama-3) in a smart loop—could outperform a larger model (GPT-4) running zero-shot.

The paradigm has shifted. We are no longer just "Prompting" a model; we are "Flow Engineering" a system.
In this post, we will dissect the four canonical design patterns for agentic systems: **Reflection, Tool Use, Planning, and Multi-Agent Collaboration.**

---

## 2. Pattern 1: Reflection (The Editor Loop)

Humans rarely write perfect code or perfect essays on the first try. We write a draft, look at it, spot errors, and rewrite.
Standard LLM usage asks for perfection in one shot. This is unnatural. **Reflection** mimics the human edit loop.

### 2.1 The Architecture
1.  **Actor:** Generates the initial output.
2.  **Critic:** Promps the model to review the output. "Look for bugs, logic errors, or style issues."
3.  **Reviser:** Takes the *Original Output* and the *Critique* and generates *Final Output*.

### 2.2 Case Study: Coding (HumanEval)
On the HumanEval benchmark (Python coding problems):
*   **GPT-4 Zero-Shot:** ~67% accuracy.
*   **GPT-4 + Reflexion:** ~88% accuracy.
*   *Why?* The model knows how to catch its own mistakes if you ask it to looking specifically for mistakes. It cannot do "Genration" and "Verification" in the same forward pass.

### 2.3 Implementation Logic
```python
def reflection_loop(goal):
    # 1. Draft
    draft = llm.generate(goal)
    
    # 2. Critique
    critique = llm.generate(f"Critique this code for bugs: {draft}")
    
    # 3. Decision
    if "No bugs found" in critique:
        return draft
        
    # 4. Revise
    final = llm.generate(f"Fix the code based on critique: {draft} \n Feedback: {critique}")
    return final
```

---

## 3. Pattern 2: Tool Use (The ReAct Loop)

We have covered this extensively in previous posts. This is the pattern where the LLM stops being a "Know-it-all" and becomes a "Seek-it-all."

### 3.1 The Architecture
*   **Thought:** The model reasons about what is missing.
*   **Action:** The model calls an external function.
*   **Observation:** The system returns data.
*   **Repeat.**

### 3.2 Key Insight: "Cognitive Offloading"
Tool use isn't just about accessing data; it's about offloading computation.
*   *Task:* "What is 329 * 482?"
*   *LLM:* "158,578" (Hallucination - doing math in weights is hard).
*   *Agent:* Calls `calculator(329, 482)`. Returns `158578` (Fact).

---

## 4. Pattern 3: Planning (Reasoning then Acting)

For complex tasks ("Build a videogame" or "Research the impact of AI on healthcare"), diving straight into execution (ReAct) leads to chaos. The agent gets lost in the details of the first step and forgets the macro-goal.
**Planning** separates **Strategy** from **Execution**.

### 4.1 The Architecture (Plan-and-Solve)
1.  **Planner Agent:**
    *   *Input:* "Build a Snake game."
    *   *Output:* A structured list.
        1.  Setup Pygame window.
        2.  Create Snake class.
        3.  Create Food class.
        4.  Handle collision logic.
2.  **Executor Agent:**
    *   Takes the Plan.
    *   Executes Step 1.
    *   Executes Step 2.
    *   ...
3.  **Replanner (Optional):**
    *   If Step 3 fails ("Pygame library not found"), the Replanner updates the plan ("Install pygame first").

### 4.2 Dynamic vs. Static Planning
*   **Static:** Generate 5 steps. Do 5 steps. (Fragile).
*   **Dynamic:** Generate 5 steps. Do Step 1. Look at the result. Re-generate remaining 4 steps based on new world state. (Robust).

---

## 5. Pattern 4: Multi-Agent Collaboration

Why ask one brain to do everything? A single "Generalist" system prompt ("You are helpful") often fails at niche tasks.
Specialization reduces token confusion.

### 5.1 Hierarchical (The Boss & Workers)
*   **Manager Agent:** Breaks down the user goal and delegates.
*   **Worker A (Researcher):** Only has `search_tool`.
*   **Worker B (Writer):** Only has `text_editor`.
*   *Flow:* User -> Manager -> Researcher -> Manager -> Writer -> Manager -> User.
*   *Benefit:* Encapsulation. The Writer doesn't need to see the messy search logs. It just sees the clean notes.

### 5.2 Joint Collaboration (The Debate)
*   **Persona A:** "You are an optimistic feature designer."
*   **Persona B:** "You are a cynical security engineer."
*   *Task:* "Design a login system."
*   *Flow:* A proposes. B critiques ("That's insecure"). A revises. B critiques.
*   *Benefit:* Reducing hallucinations. The "Cynic" acts as a filter.

### 5.3 Sequential Handoffs (The Assembly Line)
*   **Agent A:** Scrapes data. Output: JSON. ->
*   **Agent B:** Analyzes JSON. Output: Summary. ->
*   **Agent C:** Writes Email. Output: Text.
*   *Benefit:* Extremely distinct contexts. Agent C doesn't even know Agent A exists.

---

## 6. The "Fifth Pattern": Evaluation (LLM-as-a-Judge)

There is a hidden workflow that powers all of these: **Automated Evaluation**.
How do you know if your Reflection loop is working? You need a Judge.

### The Architecture
1.  **Input:** User Query + Agent Response.
2.  **Judge Prompt:** "On a scale of 1-5, how accurate is this response? output JSON."
3.  **Model:** GPT-4.

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
*   **Reflection** makes agents reliable.
*   **Planning** makes agents capable of long horizons.
*   **Multi-Agent** makes agents specialized.

In the next section of the curriculum (Days 9-28), we enter the **Intermediate** phase. We will stop talking about generic agents and start building agents that master specific domains, starting with the most important domain of all: **RAG and Document Intelligence**.
