---
title: "Autonomous Agent Architectures"
day: 38
collection: ai_agents
categories:
  - ai-agents
tags:
  - autonomous-agents
  - orchestration
  - planning
  - tools
  - safety
  - memory
  - supervision
difficulty: Medium-Hard
related_dsa_day: 38
related_ml_day: 38
related_speech_day: 38
---

**"Architecture beats prompting: build autonomous agents with clear state, strict tool boundaries, and measurable stop conditions."**

## 1. What “autonomous” actually means (and what it doesn’t)

An **autonomous agent** is a system that can take a high-level goal and make a sequence of decisions—often across multiple steps and tools—without a human telling it exactly what to do next.

Autonomy is not one feature; it’s a bundle:

- **Goal-driven behavior:** it keeps the objective in mind across steps
- **Tool use:** it calls external tools (search, code execution, databases, APIs)
- **Statefulness:** it maintains memory about what it tried and what happened
- **Adaptation:** it changes strategy when a path fails
- **Stopping:** it knows when to stop (done, failed, needs human input)

What autonomy is *not*:

- “Run forever”
- “Do everything automatically”
- “No guardrails”

In production, the best autonomous agents are **bounded**: they’re highly capable inside a restricted sandbox, and they degrade gracefully when the task exceeds their safe envelope.

---

## 2. The central idea: autonomy is a control loop

Most autonomous agents can be modeled as a loop:

```text
Observe -> Decide -> Act -> Observe -> ...
```

In practice, the loop looks like this:

1. **Observe:** collect inputs (user request + state + tool outputs)
2. **Decide:** select the next action (plan, tool call, ask user, stop)
3. **Act:** run the action (tool execution or internal reasoning)
4. **Update state:** record what happened
5. **Stop condition:** check if done / failed / needs escalation

This “agent loop” is why architecture matters: you’re building a tiny decision-making system, not just generating text.

---

## 3. Reference architecture: Planner–Executor–Verifier (PEV)

One of the most reliable architectures is to separate the agent into roles:

```text
User Request
   |
   v
Planner  -> produces a plan + constraints + stop conditions
   |
   v
Executor -> runs tools / performs actions per plan
   |
   v
Verifier -> checks correctness, safety, and completeness
   |
   v
Finalizer -> summarizes outcome + evidence + next actions (if any)
```

### Why this works

- The **Planner** is optimized for strategy and decomposition.
- The **Executor** is optimized for doing, with minimal reasoning.
- The **Verifier** is optimized for skepticism.

This reduces “single-model overconfidence.” Many agent failures are “the model confidently chose a bad step and then rationalized it.” Verification breaks that pattern.

---

## 4. State model: your agent needs more than chat history

A production-grade agent usually needs structured state like:

- **Objective:** what success means (including constraints)
- **Plan:** a list of steps and dependencies
- **Progress:** what’s done, what’s blocked
- **Facts:** verified information (with provenance)
- **Tool cache:** avoid repeating identical tool calls
- **Failures:** what didn’t work (negative memory)
- **Budget:** tokens, time, tool calls, money

A simple state shape (conceptual) is:

```json
{
  "objective": "...",
  "constraints": ["..."],
  "plan": [{"id": 1, "task": "...", "status": "pending"}],
  "facts": [{"key": "...", "value": "...", "source": "..."}],
  "attempts": [{"step_id": 2, "action": "...", "result": "failed", "why": "..."}],
  "budget": {"max_steps": 10, "spent_steps": 3}
}
```

**Junior engineer tip:** your orchestration code should treat this state as the source of truth. The LLM should *read* it and propose updates, but your code should validate and persist it.

---

## 4.5 Memory architecture inside autonomy: short-term, long-term, and “facts”

Autonomous agents often fail because they treat “memory” as one big blob. In practice, you want to separate three layers:

### 4.5.1 Short-term context (what the model can “see” right now)
This is the recent conversation plus a small state snapshot. It’s limited by context length and expensive to resend.

**Engineering implication:** don’t append forever. Use summaries and a sliding window, and keep the prompt focused on the current step.

### 4.5.2 Long-term memory (what the system can retrieve when needed)
This is a store of documents, notes, and prior outcomes. The key property is that it is **retrieval-based**, not “always in prompt.”

**Engineering implication:** retrieve only what’s relevant for the current step. If you retrieve too much, the agent gets distracted and loses the goal.

### 4.5.3 Facts (what you consider verified)
Facts are not just “things the model said.” Facts should be:

- backed by evidence (citations, tool outputs, test results)
- stored as structured key/value
- ideally with a timestamp and provenance (where did this come from?)

**Engineering implication:** treat your verifier as the gatekeeper for facts. Executors can propose facts; verifiers decide what becomes a fact.

If you separate these layers, your agent becomes calmer: it stops re-litigating old decisions and stops hallucinating “facts” that aren’t grounded.

---

## 5. Planning strategies: from linear plans to DAGs

### 5.1 Linear plans (good starting point)

A linear plan is a list:

1. Do A
2. Then do B
3. Then do C

This is easy to implement and easy to debug.

### 5.2 DAG plans (when parallelism matters)

If tasks can run in parallel (independent tools), a DAG plan is better:

- Node: a step
- Edge: dependency

Example:

- Step 1: fetch docs
- Step 2: fetch logs
- Step 3: parse docs (depends on 1)
- Step 4: parse logs (depends on 2)
- Step 5: synthesize (depends on 3 and 4)

If you have a tool execution environment that supports parallelism, DAG planning can drastically reduce latency.

**Further reading (optional):** for structured plans and strict schemas, see [Structured Output Patterns](/ai-agents/0035-structured-output-patterns/).

---

## 6. Tool boundaries: autonomy without tool discipline is chaos

Tools are where agents touch the real world. So tools need strict boundaries:

### 6.1 Principle of least privilege

Give the agent the smallest toolset required for the task.

Example:

- Research agent: `search`, `fetch`, `extract`
- Execution agent: `run_code` (sandboxed)
- Writer agent: no tools, only formatting

### 6.2 Separate read tools from write tools

Many safety incidents come from mixing reads and writes:

- agent reads untrusted content
- agent immediately uses write tools based on that content

A common pattern is:

- browsing phase: read-only tools
- action phase: write tools only after explicit validation + (optionally) human approval

### 6.3 Tool contracts and validation

Every tool call should be validated:

- schema validation (types)
- semantic validation (allowed domains, allowed file paths, max sizes)
- budget validation (rate limits, cost caps)

This is not optional; it’s how you prevent “autonomous” from turning into “uncontrolled.”

---

## 6.5 The “Supervisor + Workers” variant (when tasks are complex)

The Planner–Executor–Verifier split works well inside a single “agent.” For bigger tasks, you often get better reliability by introducing a **Supervisor** that delegates to **workers** (specialists).

```text
Supervisor (goal + state owner)
   |
   +--> Research Worker (read-only tools)
   |
   +--> Execution Worker (sandboxed code)
   |
   +--> Writer Worker (formatting, docs)
   |
   +--> Verifier Worker (checks claims/tests)
```

### Why this helps

- The supervisor maintains one canonical state (plan, budgets, stop conditions).
- Workers are smaller and easier to keep “on role.”
- You can enforce tool boundaries per worker (least privilege).

### The orchestration contract

To avoid chaos, each worker should receive:

- a **task** (one sentence)
- required **inputs** (documents, files, structured state fields)
- expected **output schema**
- strict **budget** (max tool calls, max tokens)

And each worker should return:

- `result` (structured)
- `evidence` (quotes, logs, test output)
- `confidence`
- `next_questions` (optional)

This looks boring, but boring is good. It gives you systems behavior instead of “vibes.”

---

## 7. Stop conditions: the most underrated part of autonomy

A good autonomous agent stops for the right reason. Common stop states:

- **SUCCESS:** objective met
- **FAILURE:** objective cannot be met under constraints
- **NEEDS_INPUT:** missing info, ambiguous request
- **NEEDS_APPROVAL:** high-risk action requires human confirmation

### 7.1 Concrete success checks

Success should be measurable. Instead of “I think I’m done,” define checks like:

- output contains required fields
- tests pass (if code produced)
- citations support claims (if research task)
- API response indicates success

### 7.2 Budget-based stopping

Stop when budgets are exceeded:

- max tool calls
- max steps
- max token spend
- max wall time

These budgets keep systems predictable and cost-controlled.

---

## 7.5 A practical “done” checklist (copy/paste)

Many agents fail because they don’t know what “done” means. A simple checklist helps:

1. **Requirements met:** did we satisfy all constraints (format, scope, safety)?
2. **Evidence present:** if we made claims, do we have citations/quotes or test output?
3. **No open TODOs:** are there unresolved plan steps marked “required”?
4. **No red flags:** did a verifier flag anything as unsafe or unsupported?
5. **Stop reason recorded:** success, failure, needs input, needs approval.

As a junior engineer, treat this like an API response contract: “we only return SUCCESS if the checklist passes.”

---

## 8. Failure recovery: safe autonomy means safe retries

Autonomous agents fail constantly: bad sources, flaky APIs, wrong assumptions.

Good recovery patterns:

### 8.1 Retry with evidence
On failure, feed the exact error back to the planner/executor and ask for a minimal fix.

### 8.2 Strategy pivoting
If one approach fails twice, change strategy:

- try a different tool
- narrow the task scope
- ask for user clarification

### 8.3 Escalation
If a task is high risk or repeatedly failing, stop and escalate:

- request human review
- return partial progress + what’s missing

**Further reading (optional):** see [Error Handling and Recovery](/ai-agents/0033-error-handling-recovery/) for a deeper taxonomy of failures and circuit breaker patterns.

---

## 8.5 Avoiding the “agent spiral”: negative memory + repetition detectors

A common failure mode is the **agent spiral**:

- it tries a tool call
- it fails
- it retries the same call with minor variations
- it burns budget without making progress

Two very effective mitigations:

### 8.5.1 Negative memory (“what not to do again”)
Store structured “failures” in state:

- action attempted (tool name + args hash)
- error class (timeout, 403, parsing error)
- short reason (“blocked domain”, “rate limit”, “schema mismatch”)

Then instruct the planner: “Do not propose actions whose hash is in `failures` unless you change strategy.”

### 8.5.2 Repetition detectors
Hard-stop when:

- the same tool call is repeated 3 times
- the same plan step is re-entered without new evidence

When tripped, the agent must either:

- ask the user a question
- switch strategy
- escalate

This is how you prevent infinite loops and unpredictable costs.

---

## 9. Observability: autonomy without traces is un-debuggable

When agents run multi-step loops, you need visibility:

- what did the agent try?
- why did it choose that tool?
- where did time/money go?
- where did it get stuck?

At minimum, log:

- decision per step
- tool calls + arguments (redacted)
- tool results (summarized, referenced for large outputs)
- token usage and latency
- stop reason

**Further reading (optional):** see [Observability and Tracing](/ai-agents/0034-observability-tracing/) for span-level tracing and evaluation hooks.

---

## 9.5 Measuring autonomy: metrics that matter

If you want autonomous agents to improve over time, you need metrics that represent “agent health,” not just “model quality.”

Good practical metrics:

- **Success rate by intent:** % of tasks completed successfully (per task type)
- **Steps-to-success:** median and p95 number of steps
- **Tool error rate:** % of tool calls failing (per tool)
- **Retry rate:** average retries per step
- **Budget burn:** tokens/cost per success (and p95)
- **Escalation rate:** how often tasks require approval or human help
- **Verifier disagreement rate:** how often verifier rejects executor outputs

The interesting part is not the average; it’s the **outliers**. Autonomy tends to fail in long tails: weird inputs, flaky dependencies, ambiguous goals. Your tracing should help you find those quickly.

---

## 10. Safety patterns for autonomy (real-world guardrails)

### 10.1 Allowlists for high-risk operations
For file, database, or infra operations:

- allowlist directories, tables, endpoints
- block wildcards
- require explicit scope

### 10.2 “Dry run” mode
Before applying changes, produce a diff or plan and verify it:

- generate patch
- run tests
- review diff
- only then apply

### 10.3 Human-in-the-loop gates
Use approval gates for:

- destructive actions (delete, overwrite)
- actions with money impact (billing)
- actions with privacy impact (PII)

### 10.4 Prompt injection boundaries
Autonomous agents often browse. Treat external content as untrusted data and never allow it to rewrite instructions.

**Further reading (optional):** see [Web Browsing Agents](/ai-agents/0036-web-browsing-agents/) for a concrete “tainted input” browsing pipeline.

---

## 10.6 Governance: treat prompts, tools, and policies as “code”

When autonomy goes wrong in production, you want the same thing you want in normal software: version control and change management.

Practical governance patterns:

- **Prompt/version pinning:** store prompts in git and record prompt versions in traces.
- **Tool registry:** a single source of truth for tool schemas, safety rules, and allowlists.
- **Policy tests:** unit tests for “dangerous behavior” (no network in sandboxes, no writes without approval).
- **Canary releases:** ship prompt/policy updates to a small percentage of traffic and watch your key metrics.

This is how you avoid: “we changed one line in a prompt and now the agent behaves completely differently.”

---

## 10.5 Implementation sketch: a minimal autonomous loop (pseudocode)

Below is a conceptual skeleton. The key is that the orchestrator (your code) owns the loop, budgets, and validation.

```python
def run_agent(objective: str, state: dict) -> dict:
    for step_idx in range(state["budget"]["max_steps"]):
        decision = llm.decide({
            "objective": objective,
            "state": state,
            "policy": {
                "no_unvalidated_writes": True,
                "max_tool_calls_per_step": 1
            }
        })

        if decision["type"] == "ASK_USER":
            return {"status": "NEEDS_INPUT", "question": decision["question"], "state": state}

        if decision["type"] == "STOP":
            return {"status": decision["status"], "state": state, "summary": decision.get("summary")}

        if decision["type"] == "TOOL_CALL":
            tool_name, args = decision["tool"], decision["args"]

            validate_tool_call(tool_name, args, state)  # allowlists, budgets, schemas
            result = tools.run(tool_name, args)
            state = update_state_with_result(state, decision, result)

            if is_repetition(state):
                return {"status": "FAILURE", "reason": "REPETITION_DETECTED", "state": state}

    return {"status": "FAILURE", "reason": "STEP_BUDGET_EXCEEDED", "state": state}
```

This is intentionally strict. Autonomy in production is about **controlled freedom**.

If you want the LLM to generate a plan, do it as a structured object (and validate it), then execute it in this loop.

---

## 11. Case study: an “Autonomous DevOps Helper” (bounded autonomy)

Goal: assist with incident response in a safe way.

Safe design:

1. **Observe:** fetch logs/metrics (read-only tools)
2. **Diagnose:** propose hypotheses + tests
3. **Execute tests:** run safe queries only
4. **Recommend:** suggest changes (no write tools)
5. **Escalate:** if change required, produce a patch and request approval

What this avoids:

- the agent restarting services repeatedly
- applying risky configuration changes without review
- “fixing” the incident by masking symptoms

The best autonomy here is “autonomous diagnosis + assisted remediation.”

---

## 11.5 Case study: “Autonomous research assistant” with citations (bounded browsing)

Goal: produce a short report with citations on a topic.

Safe architecture:

1. **Planner:** generate 3–5 queries and a report outline (budgeted).
2. **Research worker:** browse the web using read-only tools and extract evidence.
3. **Verifier:** enforce “quote supports claim” and require ≥2 sources for key claims.
4. **Writer:** draft the report using only verified claims + citations.
5. **Stop:** success only if citations exist and verifier passes.

What this avoids:

- hallucinated citations
- sources that are irrelevant or untrusted
- endless browsing spirals

This is a great “starter autonomy” domain because you can measure correctness: do links support claims?

---

## 12. Summary & Junior Engineer Roadmap

Autonomous agent architectures are systems engineering problems:

1. **Model the loop:** observe → decide → act → update → stop.
2. **Store real state:** plans, facts, failures, budgets—not just chat history.
3. **Separate roles:** planner/executor/verifier prevents overconfident mistakes.
4. **Enforce tool boundaries:** validate every call; isolate read vs write.
5. **Define stop conditions:** success checks + budgets + escalation.
6. **Trace everything:** autonomy without observability becomes unmaintainable.

### Mini-project (recommended)
Build a tiny agent loop with explicit state:

- State stored as JSON.
- Max 8 steps.
- One read tool (search/fetch) and one safe compute tool (sandboxed execution).
- A verifier that rejects unsupported claims or missing test evidence.

If you can make that predictable, you’re already doing “autonomous architecture” the right way.

### Common rookie mistakes (avoid these)

1. **Letting the model own state:** if the model is the only place state “exists,” you can’t validate or recover cleanly. Keep state in your code and store it durably.
2. **No stop condition:** agents that “keep thinking” are just infinite loops with a nicer UI. Always enforce budgets and a stop reason.
3. **Mixing untrusted inputs with instructions:** browsing output should never be treated as policy. Keep a strict separation between data and instructions.
4. **Too many tools at once:** a large toolset increases the chance of wrong tool selection. Start with a minimal toolbox and grow intentionally.

### Further reading (optional)

- If your autonomous agent executes code, see [Code Execution Agents](/ai-agents/0037-code-execution-agents/) for sandboxing patterns.
- If your agent depends on web sources, see [Web Browsing Agents](/ai-agents/0036-web-browsing-agents/) for safe retrieval and citation pipelines.


