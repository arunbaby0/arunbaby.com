---
title: "Hierarchical Planning"
day: 40
collection: ai_agents
categories:
  - ai-agents
tags:
  - hierarchical-planning
  - planning
  - decomposition
  - execution
  - verification
  - budgets
  - state
difficulty: Medium-Hard
related_dsa_day: 40
related_ml_day: 40
related_speech_day: 40
---

**"Make agents reliable at large tasks: plan at multiple levels, execute in small verified steps, and stop when budgets say so."**

## 1. The problem hierarchical planning solves

Simple “single-shot” prompting breaks down when tasks are:

- multi-step (10+ actions)
- ambiguous (requires clarification)
- tool-heavy (search, code execution, APIs)
- long-running (state must persist)

Without structure, agents fail in predictable ways:

- they jump into details too early
- they forget constraints
- they loop (“try again”) instead of changing strategy
- they burn budgets without progress

**Hierarchical planning** is the pattern of making the agent plan at multiple resolutions:

- **high-level goal decomposition** (what are the big chunks?)
- **mid-level step planning** (what’s the sequence of actions?)
- **low-level execution** (tool calls and micro-decisions)

The hierarchy is what keeps the agent oriented when tasks are larger than a single context window or a single reasoning pass.

---

## 2. The mental model: strategy vs. tactics vs. actions

Think of planning as three layers:

1. **Strategy (why):** what approach are we taking and why?
2. **Tactics (what/when):** what steps will we do and in what order?
3. **Actions (how):** the concrete tool calls and operations

If your agent tries to do strategy, tactics, and actions all at once, it tends to become inconsistent. Hierarchical planning separates these concerns so each layer can be validated and updated.

---

## 2.5 A practical rule: “One level up decides, one level down executes”

Hierarchies can get messy if every level tries to do everything. A clean division of responsibility is:

- **Level \(N\)**: decides *what* to do next (selects subgoals, allocates budget)
- **Level \(N+1\)**: decides *how* to do it (concrete steps, tool calls)

Example:

- Strategy layer decides: “We’ll solve this by collecting evidence first, then drafting, then critiquing.”
- Tactics layer decides: “For evidence, run 3 searches and fetch up to 5 sources.”
- Action layer decides: “Call `search(q1)`, then `fetch(url1)`…”

This prevents a common failure mode where the action layer “changes strategy mid-run” and the whole system loses coherence.

---

## 3. A reference architecture: H-Plan + H-Execute + H-Verify

Here’s a robust architecture for hierarchical planning:

```text
Objective + Constraints
   |
   v
High-Level Planner (H-Plan)
   |
   v
Task Graph / Outline (big chunks)
   |
   v
Step Planner (per chunk)
   |
   v
Executor (tools + actions)
   |
   v
Verifier (checks + gates)
   |
   v
State Update + Stop Check
```

### Key design choice: commit plans to state

Don’t treat “the plan” as text in the model’s head. Store it as structured state:

- tasks with IDs
- dependencies
- status
- evidence requirements
- budgets

This makes planning inspectable and debuggable.

---

## 3.5 Plan schemas: the simplest structure that works

If you can only add one structured artifact to your agent system, make it a plan schema. Even a minimal schema helps:

```json
{
  "goal": "…",
  "constraints": ["…"],
  "tasks": [
    {
      "id": "T1",
      "title": "Collect evidence",
      "status": "pending",
      "done_definition": "Have ≥2 sources for key claims",
      "budget": {"max_steps": 6, "max_tool_calls": 6},
      "depends_on": []
    }
  ]
}
```

Why include `done_definition` and `budget`?

- `done_definition` stops the agent from “finishing early” without evidence.
- `budget` stops the agent from “never finishing” due to endless refinement.

**Further reading (optional):** for schema-first reliability patterns, see [Structured Output Patterns](/ai-agents/0035-structured-output-patterns/).

---

## 4. Plan representations: list, tree, and DAG

### 4.1 Linear plan (list)

Best for:

- small tasks
- low parallelism
- simple execution environments

Example:

1. Gather requirements
2. Fetch sources
3. Extract facts
4. Write answer

### 4.2 Tree plan (hierarchical outline)

Best for:

- writing documents
- building multi-section outputs
- tasks that naturally decompose into “chapters”

Example:

- Section A: requirements
  - A1: functional
  - A2: non-functional
- Section B: design
  - B1: components
  - B2: data flow

### 4.3 DAG plan (dependencies + parallelism)

Best for:

- tool calls that can be parallelized
- workflows with shared prerequisites

Example:

- fetch docs (A)
- fetch logs (B)
- parse docs (depends on A)
- parse logs (depends on B)
- synthesize (depends on both)

If your system supports parallel tool execution, DAG plans reduce latency and reduce the “agent waits doing nothing” problem.

**Further reading (optional):** if you want strict plan schemas, see [Structured Output Patterns](/ai-agents/0035-structured-output-patterns/).

---

## 5. The key technique: plan refinement (coarse → fine)

Hierarchical planning works best when the agent refines plans progressively:

1. **Coarse plan:** identify 3–7 big tasks
2. **Refine one chunk:** expand only the next chunk into concrete steps
3. **Execute:** run that chunk with verification
4. **Update:** revise plan if reality differs

This avoids the most common planning failure:

> The agent writes a beautiful 30-step plan, then step 2 fails, and the rest becomes irrelevant.

Progressive refinement keeps the plan aligned with reality.

---

## 5.5 Replanning triggers: when the plan must change

Good planning systems replan when reality changes. Define explicit triggers like:

### 5.5.1 Tool failure trigger
If the same tool call fails twice (timeout, 403, schema mismatch), replan instead of retrying forever.

### 5.5.2 Evidence gap trigger
If a chunk’s done-definition requires evidence but the agent cannot obtain it within budget, replan:

- broaden query
- change source strategy
- ask user for constraints

### 5.5.3 Budget pressure trigger
If remaining budget is low and key tasks are incomplete, replan for a smaller scope:

- deliver partial result
- mark what’s missing and why

This is what makes hierarchical planning robust: it explicitly handles “plans becoming stale.”

---

## 6. Planning with constraints: make constraints first-class

Constraints are where most agent plans fail. Typical constraints:

- “no unsafe writes”
- “cite sources”
- “use a specific format”
- “budget limit”

Implementation rule:

**Constraints must be stored in state and checked by the verifier at every step.**

If constraints are only in the initial prompt, they drift over long runs.

---

## 6.5 Constraint propagation: enforce constraints at every level

Constraints must be applied at:

1. **High-level plan:** tasks should not violate constraints (no “delete database” tasks if writes are forbidden).
2. **Step plan:** tools and steps must remain within allowlists.
3. **Execution:** tool calls must be validated in code.

Practical technique:

- Store constraints once in state.
- Every planner/verifier prompt includes the same `constraints` field.
- The verifier’s output must explicitly say which constraint each step satisfies (or violates).

This turns constraints from “text” into “guardrails.”

---

## 7. Stop conditions and budgets (planning without budgets is wishful thinking)

Hierarchical planning needs stop conditions, otherwise agents keep refining forever.

Useful budgets:

- max total steps
- max tool calls
- max retries per step
- max tokens
- max time

Useful stop states:

- SUCCESS (requirements met)
- NEEDS_INPUT (ambiguous / missing data)
- NEEDS_APPROVAL (high-risk action)
- FAILURE (cannot proceed within constraints/budgets)

This is how you turn “autonomous” into “predictable.”

---

## 7.5 Budget allocation across the hierarchy

If you give one global budget (e.g., max 20 steps) without allocation, agents tend to overspend on early phases (usually searching or over-planning).

Instead, allocate budgets by chunk:

- 20% requirements and clarification
- 40% evidence gathering / tool work
- 25% drafting or execution
- 15% critique + finalization

These are not universal numbers—pick what matches your workload—but the idea is:

**Spending should reflect value.** Evidence gathering and execution often deserve more budget than prose.

---

## 8. Verification gates: how you prevent plans from becoming hallucinations

At each layer, add gates:

### 8.1 Gate the high-level plan
Check:

- does it cover all requirements?
- is it realistic within budgets?
- does it separate read vs write steps?

### 8.2 Gate the step plan
Check:

- are tool calls necessary?
- are tool calls safe (allowlists)?
- are dependencies satisfied?

### 8.3 Gate execution
Check:

- tool outputs are valid (schemas)
- evidence exists (quotes/tests)
- progress is real (not repeating)

**Further reading (optional):** see [Self-Reflection and Critique](/ai-agents/0039-self-reflection-and-critique/) for how to structure the critic role.

---

## 8.5 Verification checklists per plan layer (copy/paste)

### 8.5.1 High-level plan checklist
- Does every task contribute to the goal?
- Are constraints reflected (no forbidden actions)?
- Are done-definitions measurable?
- Are budgets realistic?

### 8.5.2 Step plan checklist
- Does each step have a clear expected output?
- Are tool calls necessary and safe (allowlists)?
- Are dependencies satisfied?
- Is there an obvious repetition risk?

### 8.5.3 Execution checklist
- Did the tool output validate (schema + semantics)?
- Did we record evidence (quotes, test output, logs)?
- Did we update state (facts, failures, budgets)?
- Did we hit stop conditions correctly?

These checklists can be used by a verifier model or by deterministic validators.

---

## 9. Common failure modes (and fixes)

### 9.1 Over-planning
Symptom: agent writes huge plans and never executes.

Fix: force refinement only for the next chunk; enforce a planning budget (e.g., max 200 tokens for plan).

### 9.2 Under-planning
Symptom: agent executes immediately and gets lost.

Fix: require at least a coarse plan before the first tool call.

### 9.3 Plan brittleness
Symptom: step 2 fails and the plan collapses.

Fix: progressive refinement + replanning triggers (“if tool fails twice, replan”).

### 9.4 Infinite loops
Symptom: repeating similar steps with no new evidence.

Fix: repetition detectors + negative memory + circuit breakers.

**Further reading (optional):** see [Error Handling and Recovery](/ai-agents/0033-error-handling-recovery/).

---

## 9.5 Planning + state: persistence for long-running tasks

Hierarchical planning is much more useful when tasks run long enough that you can’t keep everything in context.

Practical pattern:

- After each chunk completes (or fails), checkpoint:
  - plan status
  - evidence list
  - budgets spent
  - failures

This gives you:

- resumability (“pick up from where we stopped”)
- auditability (“what did the agent do and why?”)
- cost control (you can stop mid-run safely)

**Further reading (optional):** see [State Management and Checkpoints](/ai-agents/0032-state-management-checkpoints/) for checkpointing mechanics and schema versioning.

---

## 10. Implementation sketch (pseudocode)

```python
def hierarchical_agent(objective: str, constraints: list[str], state: dict) -> dict:
    # 1) Ensure we have a high-level plan
    if not state.get("high_plan"):
        state["high_plan"] = llm.make_high_plan(objective, constraints)
        state["high_plan"] = validate_plan_schema(state["high_plan"])

    # 2) Pick next chunk
    chunk = select_next_chunk(state["high_plan"])
    if chunk is None:
        return {"status": "SUCCESS", "state": state}

    # 3) Refine chunk into steps (bounded)
    steps = llm.refine_chunk(objective, constraints, chunk, state)
    steps = validate_step_schema(steps)

    # 4) Execute steps with verification
    for step in steps:
        if budget_exceeded(state):
            return {"status": "FAILURE", "reason": "BUDGET_EXCEEDED", "state": state}

        decision = verifier.precheck(step, constraints, state)
        if decision["status"] == "BLOCK":
            return {"status": "NEEDS_INPUT", "question": decision["question"], "state": state}

        result = tools.run(step["tool"], step["args"])
        state = update_state(state, step, result)

        if repetition_detected(state):
            return {"status": "FAILURE", "reason": "REPETITION_DETECTED", "state": state}

    # 5) Mark chunk done and loop
    mark_chunk_complete(state["high_plan"], chunk)
    return hierarchical_agent(objective, constraints, state)
```

The key idea: you only refine and execute a small piece at a time, and you validate each layer.

---

## 10.5 Practical tip: write plans in terms of artifacts, not vague actions

Plans become actionable when each step produces an artifact:

- “Requirements doc” (bullet list)
- “Evidence table” (claims + citations)
- “Patch/diff” (for code changes)
- “Test report” (pass/fail + logs)

If steps do not produce artifacts, it’s hard to verify progress and easy for the agent to drift into “thinking.”

This is especially important for long tasks: artifacts allow you to resume and audit without re-running everything.

---

## 11. Case study: building a “research + write” agent that doesn’t spiral

Goal: produce a short report with citations.

Hierarchical plan:

- Chunk 1: define scope and queries
- Chunk 2: collect sources and extract evidence
- Chunk 3: write draft using evidence
- Chunk 4: critique for unsupported claims
- Chunk 5: finalize and format

Where the hierarchy helps:

- browsing is bounded to chunk 2 (no endless searching)
- writing doesn’t start until evidence exists
- critique happens after writing, and revisions are bounded

**Further reading (optional):** [Web Browsing Agents](/ai-agents/0036-web-browsing-agents/) and [Self-Reflection and Critique](/ai-agents/0039-self-reflection-and-critique/).

---

## 11.5 Case study: “Autonomous fix-it agent” (bounded remediation)

Goal: help fix a failing CI build safely.

High-level plan (hierarchical):

- Chunk 1: gather context (logs, failing tests)
- Chunk 2: hypothesize causes
- Chunk 3: propose minimal patch
- Chunk 4: run tests in a sandbox
- Chunk 5: produce final diff for review

Key safety gate:

- the agent must never apply changes directly; it produces a patch and evidence (test output).

Where hierarchical planning helps:

- it stops the agent from “randomly editing files” without a hypothesis
- it forces small patches and verification after each patch
- it enforces budgets (“if 2 patch attempts fail, stop and ask for human input”)

**Further reading (optional):** if the agent runs code/tests, see [Code Execution Agents](/ai-agents/0037-code-execution-agents/).

---

## 11.6 A concrete evaluation approach for hierarchical planners

If you can’t measure planning quality, you’ll end up optimizing for “plans that look nice” instead of “plans that finish.”

A practical evaluation harness:

### 11.6.1 Step-budget success rate
Run a set of tasks with a fixed step budget and measure:

- success rate
- average steps used
- p95 steps used

Good hierarchical planners should improve success rate *and* reduce wasted steps.

### 11.6.2 Plan stability
Measure how often the agent replans:

- replans per task
- replans triggered by tool failure vs. triggered by new evidence

Replanning is not bad; **thrashing** is bad. If the agent replans every step, it’s not really planning—it's improvising.

### 11.6.3 Evidence coverage
For tasks that require evidence (citations, test outputs), measure:

- fraction of key claims with evidence
- fraction of outputs passing verification gates

This aligns planning with real-world correctness.

**Further reading (optional):** for instrumentation patterns, see [Observability and Tracing](/ai-agents/0034-observability-tracing/).

---

## 12. Summary & Junior Engineer Roadmap

Hierarchical planning turns large tasks into manageable, verifiable loops:

1. **Plan at multiple levels:** strategy → tactics → actions.
2. **Refine progressively:** coarse plan first, details only when needed.
3. **Store plans in state:** plans are data, not vibes.
4. **Verify at every layer:** gates prevent plan hallucinations.
5. **Use budgets + stop conditions:** autonomy must be bounded.

### Mini-project (recommended)

Build a small hierarchical agent:

- It always writes a coarse plan (max 7 tasks).
- It refines only the next task into 3–6 steps.
- It runs tools with strict validation.
- It stops with a clear reason when budgets are exceeded.

If you can make this predictable, you’re ready for bigger autonomous systems.

### Common rookie mistakes (avoid these)

1. **Planning in prose only:** if the plan isn’t structured data, you can’t validate it or execute it reliably.
2. **No definition of done:** “finish the task” is not a done-definition. Define measurable completion criteria.
3. **No replanning triggers:** without triggers, agents either never replan (and get stuck) or replan constantly (thrash).
4. **No budgets per chunk:** one global budget usually gets spent on early tasks; allocate budgets by phase.
5. **No verification gates:** if you don’t gate each layer, the agent can “complete” a plan that never met requirements.

### Further reading (optional)

- For multi-step system architecture patterns: [Autonomous Agent Architectures](/ai-agents/0038-autonomous-agent-architectures/)
- For critique rubrics and revision loops: [Self-Reflection and Critique](/ai-agents/0039-self-reflection-and-critique/)




---

**Originally published at:** [arunbaby.com/ai-agents/0040-hierarchical-planning](https://www.arunbaby.com/ai-agents/0040-hierarchical-planning/)

*If you found this helpful, consider sharing it with others who might benefit.*

