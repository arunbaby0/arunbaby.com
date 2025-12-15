---
title: "Multi-Step Reasoning"
day: 13
collection: ai_agents
categories:
  - ai-agents
tags:
  - reasoning
  - chain-of-thought
  - tot
  - self-consistency
  - o1
  - pal
difficulty: Medium-Easy
---

**"Thinking Fast and Slow: How to make LLMs stop guessing and start solving."**

## 1. Introduction: The System 1 Trap

Standard LLMs are **probability engines**, not logic engines. When you ask *"What is 15 * 27?"*, the model doesn't "calculate" it using an ALU (Arithmetic Logic Unit). It predicts the most likely next token based on its training distribution.
*   If it has seen `15 * 27 = 405` a thousand times, it outputs `405` (Recitation).
*   If it hasn't, it might output `415` because the digits "feel" right (Hallucination).

This is **System 1 Thinking** (Intuitive, Fast, Error-Prone, Pattern-Matching), as described by Daniel Kahneman.
For creative writing, System 1 is great. For Agents, it is a disaster.
Agents need to execute code, manage budgets, and delete files. We cannot afford "intuitive guesses." We need **System 2 Thinking** (Logical, Slow, Deliberate, Algorithm-Following).

In this post, we will explore the Prompt Engineering and Architectural patterns used to force LLMs into System 2 mode: **Chain of Thought (CoT), Self-Consistency, Tree of Thoughts (ToT), and Program-Aided Language Models (PAL).** Finally, we will demystify OpenAI's new **o1 (Reasoning Tokens)** paradigm.

---

## 2. Chain of Thought (CoT): The "Let's Think" Revolution

In 2022, a paper from Google titled *"Chain of Thought Prompting Elicits Reasoning in Large Language Models"* changed the trajectory of AI. The discovery was surprisingly simple: **If you ask the model to explain its work significantly before giving the answer, its accuracy roughly triples on math/logic tasks.**

### 2.1 The Mechanic: Computational Space
Why does this work? It's not magic; it's **Computational Space**.
*   **Without CoT:** The model must transform Input -> Answer (`Input: 15*27`, `Output: 405`) in a single forward pass per output digit. It has almost zero depth to "carry the one" or store intermediate variables.
*   **With CoT:** The model outputs `10 * 27 = 270`. `5 * 27 = 135`. `270 + 135...`.
    *   By forcing it to generate these text tokens, we are giving the transformer layers more **hops** to process the state.
    *   We are effectively letting it write to a "Scratchpad" (the Context Window) which it can then **attend to** for the final calculation.
    *   **Time = Compute.** Generating more tokens takes more time, allowing for more computation.

### 2.2 Zero-Shot vs. Few-Shot CoT
*   **Zero-Shot CoT:** Just appending the magic phrase: *"Let's think step by step."* (Kojima et al., 2022).
    *   *Result:* Massive boost for large models (GPT-4), minimal boost for small models (Llama-8B).
*   **Few-Shot CoT:** Providing 3 examples of Questions + Step-by-Step reasoning in the prompt.
    *   *Result:* Superior performance. It enforces the **Structure** of the reasoning. It prevents the model from being "lazy" and just jumping to the answer.

### 2.3 Limits of CoT
CoT is linear. It assumes the problem can be solved in a straight line from A -> B -> C.
If the model makes a mistake at Step B (e.g., `5 * 27 = 125`), the error propagates to Step C. The model rarely "Backtracks" in a standard CoT because it is autoregressive (always moving forward).

---

## 3. Self-Consistency: Democracy for Tokens

CoT isn't perfect. Sometimes the reasoning chain goes off the rails due to the probabilistic nature (Temperature > 0).
**Self-Consistency** (Wang et al., 2022) is an ensemble technique that exploits this randomness.

### 3.1 The Algorithm
1.  Take the prompt (with CoT).
2.  Run the model **5 to 10 times** in parallel (at high Temperature, e.g., 0.7).
    *   *Path A:* "Reasoning... 5+6=11."
    *   *Path B:* "Reasoning... 5+6=11."
    *   *Path C:* "Reasoning... 5+6=10." (Hallucination)
    *   *Path D:* "Reasoning... 5+6=11."
    *   *Path E:* "Reasoning... 5+6=12."
3.  **Vote (Marginalization):** We look only at the **Final Answers**.
    *   "11": 3 votes.
    *   "10": 1 vote.
    *   "12": 1 vote.
4.  **Result:** Output "11".

### 3.2 Why it works
Reasoning paths are diverse, but correct answers are usually unique. There are infinite ways to be wrong (hallucinate random numbers), but usually only one way to be right. By sampling multiple paths, the "Signal" (Truth) interferes constructively, while the "Noise" (Hallucinations) cancels out.
*   **Trade-off:** It costs **N** times more tokens and money.

---

## 4. Tree of Thoughts (ToT): Searching the Space

For planning tasks (e.g., "Write a novel," "Solve a 24-puzzle," "Debug a complex codebase"), a single linear chain isn't enough. You need to explore options, backtrack, and prune dead ends.
**Tree of Thoughts** (Yao et al., 2023) applies classical search algorithms (BFS/DFS) to LLM thoughts.

### 4.1 The Architecture
1.  **Decomposition:** Break the problem into steps.
2.  **Generation:** At Step 1, generate 3 possible next moves (Thoughts).
3.  **Evaluation (The Critic):** Use a "Judge" prompt to score each move (Sure/Likely/Impossible).
4.  **Search:**
    *   If "Impossible", **Prune** the branch.
    *   If "Sure", keep generating from there.
    *   If "Likely", keep as a backup.
    *   This usually requires a **Controller Script** (Python) to manage the state tree.

### 4.2 Application in Agents: The "Devin" Pattern
This is how autonomous coding agents work conceptually.
*   *Task:* Fix bug in `main.py`.
*   *Thought 1:* "Maybe it's a typo." -> Action: `cat main.py`. -> Observation: No typo. -> **Backtrack.**
*   *Thought 2:* "Maybe it's a logic error." -> Action: `test_logic.py`. -> Observation: Fails. -> **Proceed.**

ToT turns the "Stream of Consciousness" into a "Graph of Consciousness."

---

## 5. Program-Aided Language Models (PAL)

For math and logic, LLMs are bad calculators. Even with CoT, they fail `3284 * 1293` often.
**PAL** (Gao et al., 2022) says: *"Don't ask the LLM to calculate. Ask it to write a Python program to calculate."*

*   **Prompt:** "Roger has 5 balls..."
*   **PAL Output:**
    ```python
    balls_initial = 5
    cans = 2
    balls_per_can = 3
    total = balls_initial + (cans * balls_per_can)
    print(total)
    ```
*   **Execution:** The runtime executes the Python code.
*   **Result:** Exact precision.
*   **Agent Takeaway:** Always offload deterministic logic (Math, Date handling, String manipulation) to **Tools/Code**, never rely on the LLM's weights.

---

## 6. The Future: Reasoning Tokens (o1)

In late 2024, OpenAI released **o1**. It introduced the concept of **Hidden Reasoning Chains**.

### 6.1 Training for Thought
Instead of the user prompting "Let's think step by step," the model is trained via **Reinforcement Learning** to generating its own CoT.
*   It generates thousands of "Thought Tokens" that are **Hidden** from the user.
*   It backtracks, corrects itself ("Wait, that's wrong"), and tries new angles.
*   Only when it is confident does it emit the visible "Answer Tokens."

### 6.2 The Paradigm Shift
*   **Inference-Time Compute:** We used to think inference was constant cost ($O(N)$). Now, we treat inference like search. We can spend more time (and money) to get a better answer.
*   **The "Thinking" Placeholder:** While o1 is processing, it shows "Thinking...". This is actually the model generating hidden tokens.
*   **Implication:** We are moving from "Prompt Engineering Reason" to "Buying Reason." We pay for the compute time of the model's reflection.

---

## 7. Code: Implementing Self-Consistency

A robust Python pattern to perform voting on reasoning chains.

```python
from collections import Counter
import re
import openai

def generate_self_consistency_answer(prompt, n=5):
    """
    Generates N responses and returns the most common numeric answer.
    """
    answers = []
    reasoning_traces = []
    
    # Run N parallel requests (or one request with n=5 if supported)
    # High temperature is key for diversity
    responses = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt + "\nLet's think step by step."}],
        n=n,
        temperature=0.7 
    )
    
    for choice in responses.choices:
        text = choice.message.content
        reasoning_traces.append(text)
        
        # Simple extraction logic (in production, use structured output)
        # Find the last number in the text
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if numbers:
            answers.append(numbers[-1]) 
            
    # Voting
    if not answers:
        return None, reasoning_traces
        
    vote_counts = Counter(answers)
    most_common, count = vote_counts.most_common(1)[0]
    confidence = count / n
    
    return {
        "final_answer": most_common,
        "confidence": confidence,
        "votes": dict(vote_counts),
        "traces": reasoning_traces
    }

# Usage
prompt = "If I have 3 apples and buy 2 dozen more, then eat 5, how many do I have?"
result = generate_self_consistency_answer(prompt)

print(f"Consensus Answer: {result['final_answer']}")
print(f"Confidence: {result['confidence']*100}%")
# Output: 22 (Confidence 100%)
# Logic: 3 + 24 - 5 = 22.
```

---

## 8. Summary

Multi-Step Reasoning turns an LLM from a Text Predictor into a Logic Engine.

1.  **CoT:** The baseline. Always force the model to "show its work."
2.  **Self-Consistency:** The reliability layer. Run it 5 times and vote.
3.  **ToT:** The search layer. For hard problems, explore the tree.
4.  **PAL:** The cheat code. Use Python for math.

For an autonomous agent, **never** accept the first token as truth for a critical decision. Force the model to think, verify, and vote before acting.

Reasoning is efficient, but it becomes actionable when combined with tools in the **ReAct Pattern**.
