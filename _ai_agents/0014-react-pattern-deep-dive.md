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
  - mrkl
  - toolformer
difficulty: Medium-Easy
---

**"Reason + Act: The Loop that Changed Everything."**

## 1. Introduction: The Seminal Moment

If there is one paper that defines the modern era of AI Agents, it is *"ReAct: Synergizing Reasoning and Acting in Language Models"* (Yao et al., ICLR 2023).

Before ReAct, the AI landscape was divided into two disconnected worlds:
1.  **Chain of Thought (CoT):** Models that could reason beautifully ("I think therefore I am") but couldn't touch the world. They lived in an internal hallway of hallucinations. If you asked "What is the price of Bitcoin?", they would hallucinate a price from 2021.
2.  **Action Loops (MRKL):** Simple scripts that called tools but didn't think clearly. ("If input contains 'price', call `get_price()`"). They were brittle and could not handle ambiguity.

ReAct merged them. It proposed a specific **Prompting Strategy** that interleave **Reasoning Traces** with **Action Execution**.
*   **Thought:** The model thinks about the plan.
*   **Action:** The model emits a command.
*   **Observation:** The environment returns data.
*   **Repeat.**

This loop grounded the model's high-level reasoning in low-level reality. In this deep dive, we will dissect the ReAct pattern, its history from MRKL to Toolformer, its failure modes, and why it remains the default architecture for 90% of agents (LangChain, LlamaIndex) in production today.

---

## 2. The Anatomy of the Loop

ReAct is not a software library; it is a protocol between the **Controller (LLM)** and the **Runtime (Python)**.

### 2.1 The Prompt Structure
The System Prompt must enforce a rigid structure. The model is told to never output just an answer, but to output a thought process.

```text
You are a helpful assistant.
To solve a problem, you interleave Thought, Action, and Observation.
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [search, calculate]
Observation: the result of the action
... (this Thought/Action/Observation cycle can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```

### 2.2 The Execution Trace
Let's watch a ReAct agent solve a multi-hop reasoning question: *"Who is the wife of the actor who played Matrix's Neo?"*

1.  **Turn 1 (LLM Generation):**
    *   **Thought:** "I need to find out who played Neo in The Matrix. Then I need to find his wife."
    *   **Action:** `search("Neo actor Matrix")`
    *   *(The model stops here because of a Stop Sequence)*.
2.  **Turn 1 (Runtime Execution):**
    *   Python executes `search`.
    *   Result: `Keanu Reeves played Neo in the Matrix series...`
    *   Python appends `Observation: Keanu Reeves played Neo...` to the prompt.
3.  **Turn 2 (LLM Generation):**
    *   **Thought:** "Okay, the actor is Keanu Reeves. Now I need to find his wife. I will search for Keanu Reeves wife."
    *   **Action:** `search("Keanu Reeves wife")`
4.  **Turn 2 (Runtime Execution):**
    *   Result: `Keanu Reeves has never married. He had a partner Jennifer Syme...`
    *   Python appends `Observation: Keanu Reeves has never married...`.
5.  **Turn 3 (LLM Generation):**
    *   **Thought:** "He has no wife. The answer is 'No one'."
    *   **Final Answer:** "Keanu Reeves is unmarried."

**Grounding:** Notice how Thought 2 depends entirely on Observation 1. If Observation 1 had said "Will Smith", Thought 2 would have searched for "Will Smith wife". The agent is reacting to the world.

---

## 3. Implementation Details: The Agent Engine

To run ReAct, you need a critical piece of infrastructure: **The Orchestration Loop**.

### 3.1 The Stop Sequence
This is the most critical technical detail that beginners miss.
When the model generates `Action: search("Neo")`, it naturally wants to keep generating. It "knows" what follows an Action is an Observation. It will likely hallucinate: `Observation: Neo is played by Tom Cruise.`

We must prevents this.
*   **Mechanism:** We configure the LLM API (OpenAI/Anthropic) with `stop=["\nObservation:"]`.
*   **Result:** As soon as the model types the newline before "Observation:", the API cuts the connection.
*   **Handoff:** The partial string is returned to Python. Python executes the tool, gets the *real* observation, appends it to the string, and calls the LLM again.

### 3.2 Parsing Robustness
LLMs are messy. Sometimes they write `Action: search(Neo)`. Sometimes `Action: search("Neo")`. Sometimes `Action: [search, Neo]`.
*   **The Parser:** Your regex needs to be forgiving.
*   **Self-Correction:** If the parser fails, do *not* crash the agent. Feed the error back to the model.
    *   `Observation: Invalid Syntax. You wrote 'search(Neo)' but I expected 'search("Neo")'. Please try again.`
    *   The model, seeing the error in its memory, will virtually always correct itself in the next turn.

---

## 4. Failure Modes: Why ReAct Breaks

ReAct is powerful, but it is not AGI. It has specific pathologies that you must guard against.

### 4.1 The Infinite Loop
*   *Thought:* "I need to check the weather."
*   *Action:* `weather("London")`
*   *Observation:* `API Error: Timeout`.
*   *Thought:* "I need to check the weather." (Models are stubborn).
*   *Action:* `weather("London")`
*   *Result:* Agent runs forever until bank account drains.
*   **Fix:**
    1.  **Max Steps:** Hard limit of 10 loops.
    2.  **Unique Action Check:** If the agent emits the exact same action 3 times in a row, kill it or inject a System Message: *"You are repeating yourself. Try something else."*

### 4.2 Hallucinated Success
*   *Action:* `save_file("report.txt")`
*   *Observation:* (Empty. The tool returned `None` on success).
*   *Thought:* "I failed because I got no output. I will try again."
*   *Result:* Agent saves the file 5 times.
*   **Fix:** Tools must always be **Verbose**. Return `"Success: File report.txt saved successfully."` instead of `None`. The LLM needs the "pat on the back" to know it can move to the next step.

### 4.3 Context Overflow
The prompt grows with every step.
*   Step 1: 500 tokens.
*   Step 10: 8,000 tokens.
*   Step 20: Crash (Context Limit).
*   **Fix:** **Thought Summarization.**
    *   When $N > 10$, take the first 5 steps and summarize them: *"Steps 1-5 involved searching for the actor and confirming his identity."*
    *   Replace the raw log with the summary.

---

## 5. Evolution: ReAct + Reflexion vs Toolformer

### 5.1 Reflexion (Self-Correction)
Standard ReAct is greedy. It doesn't look back.
**Reflexion** adds a "Critic" step.
*   *Thought:* ...
*   *Action:* ...
*   *Observation:* `Error`.
*   *Reflection:* "Why did I fail? I used the wrong argument. I should use the right argument."
*   *New Thought:* ...
This simple addition (asking the model "Why?") creates **Self-Healing Agents**.

### 5.2 Toolformer (Fine-Tuning)
ReAct relies on Prompt Engineering. **Toolformer** (Meta) relies on Fine-Tuning.
*   The model is trained on millions of examples where API calls are embedded in the text.
*   `The capital of France is [API: Wiki("France")] Paris.`
*   **Pros:** Much cheaper/faster than ReAct. No massive system prompt needed.
*   **Cons:** Hard to add new tools without re-training. ReAct allows you to add a tool just by editing the prompt.

---

## 6. Code: A Robust ReAct Engine

A conceptual Python implementation of the `AgentExecutor`.

```python
import re

MAX_STEPS = 10

def run_react_agent(question, llm, tools):
    # 1. Initialize Prompt
    history = f"""System: Use Thought/Action/Observation.\nQuestion: {question}\n"""
    
    for i in range(MAX_STEPS):
        # 2. Generate Thought & Action
        # Stop at "Observation:" to let Python take over
        response = llm.generate(history, stop=["Observation:"])
        
        # Append the thought to history
        history += response
        
        # 3. Check for Final Answer
        if "Final Answer:" in response:
            return response.split("Final Answer:")[1].strip()
            
        # 4. Parse Action
        # Regex to find: Action: tool_name("arg")
        action_match = re.search(r"Action: (\w+)\((.*)\)", response)
        
        if not action_match:
            # Feedback Loop for Parser Error
            history += "\nObservation: Parse Error. Please use format Action: tool(arg)\n"
            continue
                
        tool_name = action_match.group(1)
        tool_arg = action_match.group(2).strip('"') # Clean quotes
        print(f"[Step {i}] Executing {tool_name} with {tool_arg}")
        
        # 5. Execute Tool
        if tool_name in tools:
            try:
                result = tools[tool_name](tool_arg)
            except Exception as e:
                result = f"Tool Error: {str(e)}"
        else:
            result = f"Error: Tool '{tool_name}' not found."
            
        # 6. Append Observation
        observation = f"\nObservation: {result}\n"
        history += observation
        
    return "Error: Max steps reached without final answer."
```

---

## 7. Summary

The ReAct pattern works because it respects the **Correspondence Theory of Truth**: ideas (Thoughts) must check against reality (Observations).

*   **Thought** provides the reasoning space (CoT).
*   **Action** provides the capability.
*   **Observation** provides the grounding.

By mastering the nuances of this loop—stopping sequences, parsing robustness, and error feedback—you move from "Prompt Engineering" to "Agent Runtime Engineering," building systems that can autonomously navigate the real world.

While ReAct loops handle immediate tasks, larger problems require comprehensive **Planning and Decomposition** strategies.
