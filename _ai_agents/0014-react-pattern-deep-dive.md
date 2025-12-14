---
title: "The ReAct Pattern Deep Dive"
day: 14
collection: ai_agents
categories:
  - ai-agents
tags:
  - react
  - reasoning
  - architecture
  - loop
  - agents
difficulty: Medium-Easy
---

**"Reason + Act: The Loop that Changed Everything."**

## 1. Introduction: The Seminal Moment

If there is one paper that defines the modern era of AI Agents, it is *"ReAct: Synergizing Reasoning and Acting in Language Models"* (Yao et al., ICLR 2023).

Before ReAct, we had two disconnected worlds:
1.  **Chain of Thought (CoT):** Models that could reason ("I think therefore I am") but couldn't touch the world. They lived in an internal hallway of hallucinations.
2.  **Action Loops:** Simple scripts that called tools but didn't think. ("If input contains 'weather', call API.") They were brittle.

ReAct merged them. It proposed a specific interleaving format: **Thought → Action → Observation**.
This loop grounded the model's high-level reasoning in low-level reality. In this deep dive, we will dissect the ReAct pattern, its failure modes, and why it remains the default architecture for 90% of agents in production.

---

## 2. The Anatomy of the Loop

ReAct is not a specific library; it's a **Prompting Strategy** combined with a **Python Loop**.

### 2.1 The Prompt Structure
The System Prompt must enforce a rigid structure.

```text
You are a helpful assistant.
To solve a problem, you interleave Thought, Action, and Observation.
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [search, calculate]
Observation: the result of the action
... (this Thought/Action/Observation acts can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```

### 2.2 The Execution Trace
Let's watch a ReAct agent solve: *"Who is the wife of the actor who played Matrix's Neo?"*

1.  **Thought 1:** "I need to find out who played Neo in The Matrix. Then I need to find his wife."
2.  **Action 1:** `search("Neo actor Matrix")`
3.  **Observation 1:** `Keanu Reeves played Neo in ...`
4.  **Thought 2:** "Okay, the actor is Keanu Reeves. Now I need to find his wife. I will search for Keanu Reeves wife."
5.  **Action 2:** `search("Keanu Reeves wife")`
6.  **Observation 2:** `Keanu Reeves has never married...`
7.  **Thought 3:** "He has no wife. The answer is 'No one'."
8.  **Final Answer:** "Keanu Reeves is unmarried."

**Grounding:** Notice how Thought 2 depends entirely on Observation 1. If Observation 1 had said "Will Smith", Thought 2 would have searched for "Will Smith wife". The agent is reacting to the world.

---

## 3. Implementation Details: The Agent Engine

To run ReAct, you need a Python `while` loop that acts as the **Runtime Environment**.

### 3.1 The Stop Sequence
This is the most critical technical detail.
When the model generates `Action: search("Neo")`, it naturally wants to keep generating. It might hallucinate `Observation: Neo is played by...`.
We must stop it.
*   **Stop Token:** We configure the LLM API to stop generation at the newline `\nObservation:`.
*   **Implication:** The model hands control back to Python. Python executes the tool, gets the real observation, appends it to the string, and calls the LLM again.

### 3.2 Parsing Robustness
LLMs are messy. Sometimes they write `Action: search` (missing brackets). Sometimes they write `Action: Let's search...`
*   **The Parser:** Your regex needs to be forgiving.
*   **Self-Correction:** If the parser fails, do *not* crash. Feed the error back: `Observation: Invalid Syntax. Please use Action: tool(arg).` The model will see the error in memory and correct itself in the next turn.

---

## 4. Failure Modes: Why ReAct Breaks

ReAct is powerful, but it's not AGI. It has specific pathologies.

### 4.1 The Infinite Loop
*   *Thought:* "I need to check the weather."
*   *Action:* `weather("London")`
*   *Observation:* `API Error: Timeout`.
*   *Thought:* "I need to check the weather."
*   *Action:* `weather("London")`
*   *Result:* Agent runs forever until bank account drains.
*   **Fix:** Implementation of `max_steps` (e.g., 10) and `unique_action_check` (prevent repeating the exact same tool call 3 times).

### 4.2 Hallucinated Success
*   *Action:* `save_file("report.txt")`
*   *Observation:* (Empty. The tool returned nothing on success).
*   *Thought:* "I failed. I will try again."
*   *Result:* Agent saves the file 5 times because it expected a text confirmation.
*   **Fix:** Tools must always be verbose. Return `"Success: File saved."` instead of `Nonetype`.

### 4.3 Context Overflow
The prompt grows with every step.
*   Step 1: 500 tokens.
*   Step 10: 8,000 tokens.
*   Step 20: Crash (Context Limit).
*   **Fix:** Memory Management (Day 5 & 12). Summarize the middle steps: *"Steps 1-5 involved searching for the actor."*

---

## 5. Evolution: ReAct + Reflexion

Standard ReAct is greedy. It doesn't look back.
**Reflexion** adds a "Critic" step.
*   *Thought:* ...
*   *Action:* ...
*   *Observation:* `Error`.
*   *Reflection:* "Why did I fail? I used the wrong argument. I should use the right argument."
*   *New Thought:* ...

This simple addition (asking the model "Why?") creates **Self-Healing Agents**.

---

## 6. Code: A Robust ReAct Loop

Conceptual Python logic for the engine.

```python
MAX_STEPS = 10

def run_react_agent(question):
    history = f"Question: {question}\n"
    
    for i in range(MAX_STEPS):
        # 1. Think and Act
        # Stop at "Observation:" to let Python take over
        response = llm.generate(history, stop=["Observation:"])
        
        history += response
        
        # 2. Parse Action
        action_match = regex.parse(response)
        if not action_match:
            # Check if finalized
            if "Final Answer:" in response:
                return extract_answer(response)
            else:
                history += "\nObservation: Invalid Format. Use Action: tool()\n"
                continue
                
        tool_name, tool_arg = action_match
        print(f"Executing {tool_name} with {tool_arg}")
        
        # 3. Execute
        try:
            result = tools[tool_name](tool_arg)
        except Exception as e:
            result = f"Error: {e}"
            
        # 4. Observe
        observation = f"\nObservation: {result}\n"
        history += observation
        
    return "Error: Max steps reached without answer."
```

---

## 7. Summary

The ReAct pattern is the "Hello World" of Agent Engineering, but it is also the "Operating System" of advanced agents.
*   **Thought** provides reasoning space.
*   **Action** provides capability.
*   **Observation** provides grounding.

By mastering the nuances of this loop—stopping sequences, parsing, and error feedback—you move from "Prompt Engineering" to "Agent Runtime Engineering."

In the next post, we will look at how to scale this up to problems that require complex strategy using **Planning and Decomposition**.
