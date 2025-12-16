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
  - csp
difficulty: Medium-Easy
related_dsa_day: 15
related_ml_day: 15
related_speech_day: 15
---

**"If you fail to plan, you are planning to fail (and burn tokens)."**

## 1. Introduction: The Greedy Agent Problem

The ReAct pattern is powerful, but it suffers from a fundamental flaw: it is **myopic** (short-sighted).
ReAct looks at the current state, decides the *immediate* next step, acts, and repeats.
*   *Analogy:* ReAct is like driving by looking only 5 feet in front of the car. You might avoid the pothole, but you might also miss the turnoff for the highway and drive in circles.
*   *Symptom:* For long-horizon tasks ("Write a full-stack app"), ReAct agents get "distracted." They spend 50 steps perfecting the CSS of the login button and forget to write the backend API.
*   *Cost:* Without a plan, agents often backtrack, repeating expensive RAG searches or tool calls.

To solve complex problems, we need **Planning**. The agent must separate "Strategy" (What to do globaly) from "Execution" (How to do the local task).
In this post, we will explore the Plan-and-Solve patterns, Hierarchical Architectures, and the use of DAGs (Directed Acyclic Graphs) for parallel agent execution.

---

## 2. Planning Architectures

There are three main paradigms for giving agents foresight.

### 2.1 Plan-and-Solve (Linear Planning)
The simplest upgrade over ReAct. It breaks the "Think-Act" loop into two phases.

*   **Phase 1 (The Architect):** Ask the LLM to generate a static checklist.
    *   *Prompt:* "You are a Software Architect. Create a step-by-step plan to build a Snake game in Python. Do not write code yet."
    *   *Output:*
        1. Set up `pygame` environment.
        2. Create `Snake` class with movement logic.
        3. Create `Food` class with random placement.
        4. Implement collision detection.
        5. Write the game loop.
*   **Phase 2 (The Builder):** Data-drive the execution. The agent iterates through the list index by index.
    *   *Prompt:* "Current Task: Set up `pygame`. Execute this."
*   **Pros:** Keeps the agent focused. Global context is maintained.
*   **Cons:** **Brittle.** If Step 1 reveals that `pygame` is not installed on the system, the plan breaks. It lacks adaptability concepts.

### 2.2 Dynamic Replanning (Adaptive Plan-and-Solve)
A more robust approach. The Plan is not a static text; it's a **Mutable State Object**.
*   **State:** `Plan: [Task A (Done), Task B (Todo), Task C (Todo)]`
*   **The Loop:**
    1.  Pick first "Todo" task (Task B).
    2.  Execute it (using ReAct).
    3.  **Observation:** "Task B failed because database is locked."
    4.  **Replan Trigger:** The Executor reports failure to the Planner.
    5.  **Replan:** The Planner Agent looks at the failure and the global goal.
    6.  **New Plan:** `[Task A (Done), Task B.1 (Unlock DB), Task B.2 (Retry Task B), Task C (Todo)]`.
*   *Impact:* The agent can pivot while maintaining the long-term goal. This mimics how humans handle unexpected obstacles.

### 2.3 Hierarchical Planning (The Manager-Worker)
For massive tasks, a linear list is too long (Step 1 to Step 100). The context window overflows.
We use a **Recursive Tree Structure**.

*   **Level 0 (CEO Agent):** "Goal: Launch Website." -> Decomposes into: "Frontend", "Backend", "Deploy".
*   **Level 1 (VP of Frontend):** "Goal: Frontend." -> Decomposes into: "HTML", "CSS", "JS".
*   **Level 2 (Worker):** "Goal: HTML." -> Writes the code.
*   **Mechanism:**
    *   The "CEO" does not have tools to write code. It has a tool called `delegate_to_backend(task)`.
    *   When called, a *new* agent spawns with a fresh context window.
    *   When the sub-agent finishes, it returns a **Summary** ("Backend created at /api").
    *   The CEO sees only the summary, keeping its context clean.
*   **Frameworks:** This is the core logic of **CrewAI** and **LangGraph**.

---

## 3. Decomposition Strategies

How do we break a problem down? The prompt strategy matters.

### 3.1 Temporal Decomposition
"Do A, then B, then C." Use this for procedural tasks.
*   *Input:* "Tell me the percentage growth of Tesla stock compared to Ford in 2023."
*   *Plan:*
    1.  Get Tesla 2023 start/end price.
    2.  Get Ford 2023 start/end price.
    3.  Calculate growth % for both.
    4.  Compare.

### 3.2 Logical Decomposition (Least-to-Most)
Break a hard question into easy sub-questions. Use this for reasoning/riddles.
*   *Input:* "Is the current President of the US older than the iPhone?"
*   *Sub-Q 1:* "Who is the current President?" -> Biden.
*   *Sub-Q 2:* "What is Biden's age?" -> 81.
*   *Sub-Q 3:* "When was iPhone released?" -> 2007.
*   *Sub-Q 4:* "Age of iPhone?" -> 17 years.
*   *Comparison:* 81 > 17. Yes.

### 3.3 DAG Planning (Dependency Graphs)
Sometimes tasks describe a graph, not a list.
*   *Graph:*
    *   Task A (Get Stock Data)
    *   Task B (Get News Data)
    *   Task C (Synthesize Report - Depends on A & B)
*   *Execution:*
    *   The engine detects that A and B have no dependencies.
    *   It spins up two threads/agents to run A and B **Parallel**.
    *   It waits (joins) for both to complete.
    *   It unlocks C.
*   *Performance:* This reduces wall-clock latency significantly for I/O bound agents.

---

## 4. Constraint Satisfaction Problems (CSP)

Advanced planners don't just list steps; they solve constraints.
*   *Task:* "Book a flight to Paris under $500 leaving on a Friday."
*   *Constraints:* `price < 500`, `dest = Paris`, `day = Friday`.
*   *Planner Logic:*
    1.  Search flights.
    2.  Filter results against constraints.
    3.  If `results == 0`, **Relax Constraint**.
    4.  "No flights under $500. Searching under $600."
*   This feedback loop is critical for travel/booking agents.

---

## 5. Code: A Dynamic Planner Class

A conceptual implementation of a "Replanning Agent" that is self-healing.

```python
import json

class PlannerAgent:
    def __init__(self, goal, llm, executor):
        self.goal = goal
        self.llm = llm
        self.executor = executor
        self.plan = [] # List of structs {id, desc, status}
        self.completed_steps = []
        
    def generate_initial_plan(self):
        # Enforce JSON Schema for the plan
        context = f"Goal: {self.goal}. Create a detailed step-by-step plan."
        response = self.llm.generate_json(context, schema=PLAN_SCHEMA)
        self.plan = response['steps']
        print(f"Initial Plan: {[s['desc'] for s in self.plan]}")

    def execute_next(self):
        if not self.plan:
            return "Done"
            
        current_step = self.plan[0]
        print(f"Executing Step: {current_step['desc']}")
        
        # 1. Handoff to Executor (The ReAct Agent)
        # We give the Executor the *Global Goal* context too
        result, success = self.executor.run(
            task=current_step['desc'], 
            context={"global_goal": self.goal}
        )
        
        # 2. Check Success
        if success:
            self.completed_steps.append({"step": current_step, "result": result})
            self.plan.pop(0) # Remove from Todo
        else:
            print("Step failed. Triggering Replanning...")
            self.replan(failed_step=current_step, error=result)
            
    def replan(self, failed_step, error):
        """
        The Core Logic: Ask the Planner to fix the mess.
        """
        context = f"""
        Goal: {self.goal}
        Completed Steps: {self.completed_steps}
        
        The Current Step FAILED.
        Step: {failed_step['desc']}
        Error: {error}
        
        Remaining Plan: {self.plan}
        
        INSTRUCTION: Update the plan. You can:
        1. Break the failed step into smaller steps.
        2. Propose a different way to achieve the goal.
        3. Skip the step if it's optional.
        """
        response = self.llm.generate_json(context, schema=PLAN_SCHEMA)
        self.plan = response['steps'] # Overwrite the queue
        print(f"New Plan: {[s['desc'] for s in self.plan]}")

# Usage Simulation
# agent = PlannerAgent("Write a Weather App")
# agent.generate_initial_plan()
# while agent.plan:
#    agent.execute_next()
```

---

## 6. Summary

Planning turns an Agent from a "Task Doer" into a "Project Manager."

*   **Plan-and-Solve** avoids distraction by separating concerns.
*   **Replanning** handles the messy reality of API failures.
*   **Hierarchies** (Manager-Worker) allow agents to match the scale of human organizations.
*   **DAGs** allow for high-performance parallelism.

An agent without a plan is a leaf in the wind. An agent with a plan is a guided missile.

This concludes the "Foundations" and "Intermediate Concepts" of our curriculum. We have covered Models, Prompts, Tools, Memory, RAG, and Planning.
With planning in place, we move into **Deep Domain Specialization**: Real-time Pipelines, Voice Agents, and Vision Agents.
