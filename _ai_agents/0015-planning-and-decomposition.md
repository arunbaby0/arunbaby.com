---
title: "Planning and Decomposition"
day: 15
collection: ai_agents
categories:
  - ai-agents
tags:
  - planning
  - plan-and-solve
  - hierarchical-planning
  - dag
  - reasoning
difficulty: Medium-Easy
---

**"If you fail to plan, you are planning to fail (and burn tokens)."**

## 1. Introduction: The Greedy Agent Problem

The ReAct pattern (Day 14) is powerful, but it suffers from a fundamental flaw: it is **myopic** (short-sighted).
ReAct looks at the current state, decides the *immediate* next step, acts, and repeats.
*   *Analogy:* ReAct is like driving by looking only 5 feet in front of the car. You might avoid the pothole, but you might also miss the turnoff for the highway and drive in circles.
*   *Symptom:* For long-horizon tasks ("Write a full-stack app"), ReAct agents get "distracted." They spend 50 steps perfecting the CSS of the login button and forget to write the backend API.

To solve complex problems, we need **Planning**. The agent must separate "Strategy" (What to do) from "Execution" (Doing it).

---

## 2. Planning Architectures

There are three main paradigms for giving agents foresight.

### 2.1 Plan-and-Solve (Linear Planning)
The simplest upgrade over ReAct.
*   **Step 1 (Planner):** Ask the LLM to generate a static checklist.
    *   *Prompt:* "Create a step-by-step plan to build a Snake game."
    *   *Output:*
        1. Write `snake.py` class.
        2. Write `food.py` class.
        3. Write main game loop.
        4. Add unit tests.
*   **Step 2 (Executor):** Data-drive the execution. The agent iterates through the list index by index.
*   **Pros:** Keeps the agent focused. Global context is maintained.
*   **Cons:** **Brittle.** If Step 1 reveals that `snake.py` needs a library that doesn't exist, the plan breaks. It lacks adaptability.

### 2.2 Dynamic Replanning (Adaptive)
A more robust approach. The Plan is not a static text; it's a **State Object**.
*   **State:** `Plan: [Task A (Done), Task B (Todo), Task C (Todo)]`
*   **The Loop:**
    1.  Pick first "Todo" task (Task B).
    2.  Execute it.
    3.  **Observation:** "Task B failed because database is locked."
    4.  **Replan:** The Planner Agent looks at the failure and the remaining plan.
    5.  **New Plan:** `[Task A (Done), Task B.1 (Unlock DB), Task B.2 (Retry Task B), Task C (Todo)]`.
*   *Impact:* The agent can pivot while maintaining the long-term goal.

### 2.3 Hierarchical Planning (The Manager-Worker)
For massive tasks, a linear list is too long (Step 1 to Step 100). The context window overflows.
We use a Tree Structure.
*   **Level 0 (CEO):** "Goal: Launch Website." -> Decomposes into: "Frontend", "Backend", "Deploy".
*   **Level 1 (VP of Frontend):** "Goal: Frontend." -> Decomposes into: "HTML", "CSS", "JS".
*   **Level 2 (Worker):** "Goal: HTML." -> Writes the code.
*   *Mechanism:* Each "Manager" agent spawns a sub-agent, waits for the result, and reports up the chain.
*   *Tool Check:* The "CEO" uses a `delegate_to_backend(task)` tool.

---

## 3. Decomposition Strategies

How do we break a problem down?

### 3.1 Temporal Decomposition
"Do A, then B, then C."
*   *Input:* "Tell me the percentage growth of Tesla stock compared to Ford in 2023."
*   *Plan:*
    1.  Get Tesla 2023 start/end price.
    2.  Get Ford 2023 start/end price.
    3.  Calculate growth % for both.
    4.  Compare.

### 3.2 Logical Decomposition (Least-to-Most)
Break a hard question into easy sub-questions.
*   *Input:* "Is the current President of the US older than the iPhone?"
*   *Sub-Q 1:* "Who is the current President?" -> Biden.
*   *Sub-Q 2:* "What is Biden's age?" -> 81.
*   *Sub-Q 3:* "When was iPhone released?" -> 2007.
*   *Sub-Q 4:* "Age of iPhone?" -> 17 years.
*   *Comparison:* 81 > 17. Yes.

### 3.3 DAG Planning (Dependency Graphs)
Sometimes tasks can be parallel.
*   *Graph:*
    *   Task A (Get Stock Data)
    *   Task B (Get News Data)
    *   Task C (Synthesize Report - Depends on A & B)
*   *Execution:* Run A and B in parallel (using `asyncio`). When both finish, unlock C.
*   *Frameworks:* **LangGraph** excels at this DAG structure.

---

## 4. The Planner Prompts

The "Planner" is just an LLM. Its prompt needs to be specific.
*   *Bad Prompt:* "Make a plan."
*   *Good Prompt:* "You are a Technical Planner. Break this objective down into atomic, executable steps. Each step must be achievable by a single function call. Output a JSON list of objects: `{'id': 1, 'description': '...', 'dependencies': []}`."

**Output Schema Enforcment** is critical. If the planner outputs prose ("First, you typically..."), the executor cannot parse it. You need structured JSON.

---

## 5. Code: A Dynamic Planner Class

Conceptual logic for a Replanning Agent.

```python
class PlannerAgent:
    def __init__(self, goal):
        self.goal = goal
        self.plan = [] # List of strings
        self.completed_steps = []
        
    def generate_initial_plan(self):
        # Call LLM to get list
        self.plan = llm.generate_json(f"Goal: {self.goal}. Create plan.")
        print(f"Initial Plan: {self.plan}")

    def execute_next(self):
        if not self.plan:
            return "Done"
            
        current_step = self.plan[0]
        print(f"Executing: {current_step}")
        
        # Hands off to Executor (ReAct Agent)
        result, success = Executor.run(current_step)
        
        if success:
            self.completed_steps.append(current_step)
            self.plan.pop(0) # Remove from Todo
        else:
            print("Step failed. Replanning...")
            self.replan(failed_step=current_step, error=result)
            
    def replan(self, failed_step, error):
        context = f"""
        Goal: {self.goal}
        Completed: {self.completed_steps}
        Failed Step: {failed_step}
        Error: {error}
        Remaining Plan: {self.plan}
        
        INSTRUCTION: Fix the plan. Break the failed step down or try an alternative.
        """
        self.plan = llm.generate_json(context)
        print(f"New Plan: {self.plan}")

# Usage
agent = PlannerAgent("Write a Weather App")
agent.generate_initial_plan()
while agent.plan:
    agent.execute_next()
```

---

## 6. Summary

Planning turns an Agent from a "Task Doer" into a "Project Manager."
*   **Plan-and-Solve** avoids distraction.
*   **Replanning** handles reality.
*   **Hierarchies** handle scale.

An agent without a plan is a leaf in the wind. An agent with a plan is a guided missile.

This concludes the "Foundations" and "Intermediate Concepts" of our curriculum. We have covered Models, Prompts, Tools, Memory, RAG, and Planning.
In the next section (Days 16-28), we will move into **Deep Domain Specialization**: Real-time Pipelines, Voice Agents, and Vision Agents.
