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
difficulty: Medium-Easy
---

**"Thinking Fast and Slow: How to make LLMs stop guessing."**

## 1. Introduction: The System 1 Trap

Standard LLMs are **probability engines**, not logic engines. When you ask * "What is 15 * 27?"*, the model doesn't "calculate" it. It predicts the most likely next token based on training data. If it has seen `15 * 27 = 405` a thousand times, it outputs `405`. If it hasn't, it might output `415` because the digits "feel" right.

This is **System 1 Thinking** (Intuitive, Fast, Error-Prone).

For Agents, this is a disaster. Agents execute code, manage money, and delete files. We cannot afford "intuitive guesses." We need **System 2 Thinking** (Logical, Slow, Deliberate).

In this post, we will explore the techniques to force LLMs into System 2 mode: **Chain of Thought (CoT), Self-Consistency, and Tree of Thoughts (ToT).** We will also look at the future of **"Reasoning Tokens"** (introduced by OpenAI's o1 model), which separates the "Thinking Process" from the "Final Answer."

---

## 2. Chain of Thought (CoT): The "Let's Think" Revolution

In 2022, a paper from Google titled *"Chain of Thought Prompting Elicits Reasoning in Large Language Models"* changed everything. The discovery was simple: **If you ask the model to explain its work, it gets the answer right.**

### 2.1 The Mechanic
*   **Standard Prompt:**
    *   Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many does he have?
    *   A: 11.
*   **CoT Prompt:**
    *   Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many does he have? **Let's think step by step.**
    *   A: Roger started with 5 balls. 2 cans * 3 balls/can = 6 balls. 5 + 6 = 11. The answer is 11.

### 2.2 Why it works
It's not magic; it's **Computational Space**.
An LLM has limited computation per token. It cannot solve a 3-step logic puzzle in a single forward pass used to generate the token "11".
By forcing it to generate the text "2 cans * 3 balls...", we are giving the transformer layers more **hops** to process the intermediate state. We are effectively letting it write to a "Scratchpad" (the Context Window) which it can then attend to for the final calc.

### 2.3 Zero-Shot vs. Few-Shot CoT
*   **Zero-Shot:** Just appending "Let's think step by step." (Works well on big models).
*   **Few-Shot:** Providing 3 examples of Questions + Step-by-Step reasoning. (Works much better, enforces the *structure* of the reasoning).

---

## 3. Self-Consistency: Democracy for Tokens

CoT isn't perfect. Sometimes the reasoning chain goes off the rails.
**Self-Consistency** is an ensemble technique.

### 3.1 The Algorithm
1.  Take the prompt (with CoT).
2.  Run the model **5 times** (at high Temperature, e.g., 0.7).
    *   *Path A:* "5+6=11."
    *   *Path B:* "5+6=11."
    *   *Path C:* "5+6=10." (Halucination)
    *   *Path D:* "5+6=11."
    *   *Path E:* "5+6=12."
3.  **Vote:** The answer "11" appears 3 times. "10" appears once. "12" appears once.
4.  **Result:** Output "11".

### 3.2 Why it works
Reasoning paths are diverse, but correct answers are unique. There are many ways to be wrong, but usually only one way to be right. By sampling multiple paths, the "Signal" (Truth) interferes constructively, while the "Noise" (Hallucinations) cancels out.
*   **Trade-off:** It costs 5x more tokens and latency.

---

## 4. Tree of Thoughts (ToT): Searching the Space

For planning tasks (e.g., "Write a novel" or "Solve a 24-puzzle"), a single linear chain isn't enough. You need to explore options, backtrack, and prune dead ends.

### 4.1 The Algorithm
1.  **Decomposition:** Break the problem into steps.
2.  **Generation:** At Step 1, generate 3 possible next moves.
3.  **Evaluation:** Use a "Judge" prompt to score each move (Good/Bad/Maybe).
4.  **Search:**
    *   If "Bad", prune the branch.
    *   If "Good", keep generating from there.
    *   (Uses BFS or DFS search algorithms).

### 4.2 Application in Agents
This is how autonomous coding agents (like Devin) work conceptually.
*   *Task:* Fix bug.
*   *Thought 1:* "Maybe it's a typo." -> Check file. -> (No typo). -> **Backtrack.**
*   *Thought 2:* "Maybe it's a logic error." -> Check function. -> (Found it). -> **Proceed.**

ToT requires an **Agent Loop**, not just a single prompt.

---

## 5. The Future: Reasoning Tokens (o1)

In late 2024, OpenAI released **o1**. It introduced the concept of **Hidden Reasoning Chains**.
Instead of the user prompting "Let's think step by step," the model performs a massive CoT internally before emitting the first user-visible token.

*   **Training:** The model is trained via Reinforcement Learning to "Think longer" for hard problems.
*   **Behavior:** It outputs "Thinking..." (placeholder) while it generates thousands of internal reasoning tokens that verify its own logic.
*   **Implication:** We are moving from "Prompt Engineering Reason" to "Buying Reason." We pay for the compute time of the model's reflection.

---

## 6. Code: Implementing Self-Consistency

A Python function to perform voting on reasoning chains.

```python
from collections import Counter
import re

def generate_cot__answer(prompt, n=5):
    """
    Generates N responses and returns the most common numeric answer.
    """
    answers = []
    
    for _ in range(n):
        # High temperature for diversity
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt + "\nLet's think step by step."}],
            temperature=0.7 
        )
        text = response.choices[0].message.content
        
        # Simple regex to find the last number (Simulated "Answer extraction")
        # In prod, ask the model to output JSON {"reasoning": "...", "final_answer": "11"}
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if numbers:
            answers.append(numbers[-1]) # Assume last number is the answer
            
    # Voting
    vote_counts = Counter(answers)
    most_common, count = vote_counts.most_common(1)[0]
    
    return most_common, vote_counts

# Usage
prompt = "If I have 3 apples and buy 2 dozen more, then eat 5, how many do I have?"
final_ans, details = generate_cot__answer(prompt)
print(f"Consensus Answer: {final_ans}")
print(f"Votes: {details}")
```

---

## 7. Summary

Multi-Step Reasoning turns an LLM from a Text Predictor into a Logic Engine.
*   **CoT** gives it time to think.
*   **Self-Consistency** filters out luck.
*   **Tree of Thoughts** allows it to explore.

For an agent, **never** accept the first token as truth for a critical decision. Force the model to show its work.

In the next post, we will revisit the **ReAct Pattern**, combining reasoning with tools in a rigid loop.
