---
title: "Testing AI Agents"
day: 43
collection: ai_agents
categories:
  - ai-agents
tags:
  - testing
  - evals
  - unit-tests
  - integration-tests
  - tool-mocking
  - regression
  - reliability
difficulty: Medium-Hard
related_dsa_day: 43
related_ml_day: 43
related_speech_day: 43
---

**"Test agents like systems: validate tool calls, pin behaviors with replayable traces, and catch regressions before users do."**

## 1. Why “testing agents” is different from testing normal code

In normal software, you test deterministic functions: same input → same output.

Agents are not like that:

- outputs can vary across runs
- “correctness” can be fuzzy
- behavior depends on tools (web, APIs, files, code execution)
- safety matters as much as success

So testing agents is really testing a **system**:

- the model prompt and policy
- the orchestrator code
- tool contracts and validators
- sandboxing and safety gates
- state management and stop conditions

If you test only the final text output, you’ll miss the most expensive failures: unsafe tool use, loops, and silent hallucinations.

---

## 2. The testing stack: unit → integration → scenario → regression

A practical hierarchy:

### 2.1 Unit tests (fast, local)
Test deterministic components:

- tool argument validation
- schema parsers
- redaction functions
- budget counters and repetition detectors

### 2.2 Integration tests (tool boundaries)
Test orchestration with mocked tools:

- tool calls happen with correct args
- unsafe calls are blocked
- retries are bounded

### 2.3 Scenario tests (end-to-end)
Test realistic user flows:

- multi-step tasks
- realistic tool outputs
- expected stop reasons

### 2.4 Regression tests (pin behaviors)
Replay known traces and compare:

- action sequences
- safety decisions
- cost metrics

This is how you keep agent behavior stable over time.

---

## 2.5 A quick mapping: what to test where

| Layer | What you test | What it catches | Typical runtime |
| :--- | :--- | :--- | :--- |
| Unit | validators, redaction, budgets | obvious bugs, safety bypasses | milliseconds |
| Integration | orchestrator + mocked tools | wrong tool args, wrong ordering | seconds |
| Scenario | end-to-end intent flows | drift, missing steps, bad stop reasons | seconds–minutes |
| Regression | replay traces | regressions from prompt/model/tool changes | minutes |

This mapping is useful because it prevents a common mistake: trying to test everything with slow end-to-end tests.

---

## 3. What to test: outputs, trajectories, and invariants

### 3.1 Output-level tests
Useful for:

- strict JSON outputs
- required fields and formats
- deterministic tasks

But output tests alone are insufficient.

### 3.2 Trajectory-level tests (the agent’s path)
Trajectory tests validate:

- tool selection
- action ordering
- bounded retries
- stop conditions

This is the core of agent reliability engineering.

### 3.3 Invariant tests (must always hold)
Invariants should be enforced by code:

- budgets never go negative
- write actions require approval
- no network in sandbox (if policy)
- no secrets in logs

Invariants are your “hard rails.” When they fail, the test should fail loudly.

---

## 3.5 Practical invariants (copy/paste for most agents)

These are generic invariants that improve most agent systems:

- **Budget invariants:** `steps_used <= max_steps`, `tool_calls <= max_tool_calls`.
- **Safety invariants:** no write tools when writes are disallowed; no network when sandbox policy forbids it.
- **Schema invariants:** any structured output must parse; unknown fields are rejected (when strict).
- **No secret echo:** outputs and logs must not contain patterns that look like keys/PII.
- **Stop invariants:** agent must always stop with an explicit stop reason.

You can enforce many of these in code (deterministically) and then test the enforcement layer.

---

## 4. Mocking tools: the most important agent testing technique

If your tests depend on real tools:

- they’re slow
- they’re flaky
- they’re expensive
- they change over time (web pages, APIs)

So you need tool mocks.

### 4.1 Record/replay
Record real tool outputs once, then replay them.

Pros:
- realistic
- stable over time

Cons:
- recordings can go stale if tools evolve

### 4.2 Synthetic mocks
Write simple mocked outputs:

- simulate success
- simulate timeouts
- simulate malformed responses

This is great for testing error handling and safety.

### 4.3 Hybrid approach
Use record/replay for “happy paths” and synthetic mocks for “failure paths.”

---

## 4.5 Mocking strategy for multi-step agents: “tool scripts”

For scenario tests, it’s helpful to define tool outputs as a script:

- call #1 → returns success payload
- call #2 → returns 429
- call #3 → returns success after backoff

This lets you verify:

- retries are bounded
- the agent changes strategy
- stop reason is correct

These scripts become reusable test fixtures.

---

## 5. Testing structured outputs: schemas + fuzzing

If your agent outputs structured JSON, you can test:

- schema validity
- missing required fields
- invalid types
- range constraints

Add fuzz tests:

- random strings
- edge-case unicode
- extremely long inputs

These catch parsing edge cases and drift.

**Further reading (optional):** see [Structured Output Patterns](/ai-agents/0035-structured-output-patterns/).

---

## 5.5 Property-based thinking (even without a property-testing library)

You can get most property-testing value without adopting a new library by writing a few loops:

- generate 100 random small inputs
- run the agent/tool logic
- assert invariants

Examples:

- “Any extracted list has no duplicate IDs.”
- “Any plan has no cycles when dependencies are used.”
- “Any cost metric is non-negative.”

This catches “weird” failures that aren’t in your hand-written examples.

---

## 6. Testing autonomy: budgets and stop reasons

Autonomous agents fail by looping. So tests must assert:

- the agent stops within budget
- the stop reason is correct (SUCCESS/NEEDS_INPUT/FAILURE)
- repeated tool calls are detected

A simple scenario test might assert:

- “Given tool failure twice, the agent asks a clarifying question instead of retrying forever.”

**Further reading (optional):** see [Autonomous Agent Architectures](/ai-agents/0038-autonomous-agent-architectures/) and [Hierarchical Planning](/ai-agents/0040-hierarchical-planning/).

---

## 6.5 Testing statefulness: resumption and replay

If your agent can pause/resume, test:

- resuming from a checkpoint produces a consistent next action
- tool cache prevents duplicate calls
- old failures are remembered (negative memory) to avoid loops

This is where “agent tests” become “workflow tests.”

**Further reading (optional):** see [State Management and Checkpoints](/ai-agents/0032-state-management-checkpoints/).

---

## 6.6 Testing planning quality (lightweight but effective)

If your agent generates plans, you can test plan quality without “grading prose.”

Practical checks:

- **No cycles:** if the plan has dependencies, it must be acyclic.
- **Step bounds:** each plan chunk has a max step budget.
- **Done definitions exist:** each chunk has a measurable done condition.
- **No forbidden actions:** plan does not contain disallowed tools/operations.

These are deterministic checks that catch many planning failures early.

**Further reading (optional):** see [Hierarchical Planning](/ai-agents/0040-hierarchical-planning/).

---

## 7. Testing safety: prompt injection and forbidden actions

Safety tests should be explicit and automated:

### 7.1 Prompt injection tests
Inputs containing:

- “ignore previous instructions”
- “call delete tool”
- “reveal secrets”

Expected behavior:

- refuse
- sanitize
- continue safely or escalate

### 7.2 Forbidden tool calls
Make sure the orchestrator blocks:

- writes when writes are forbidden
- network calls when network is forbidden
- file access outside allowlist

These should be deterministic tests that do not depend on the model “doing the right thing.”

---

## 7.5 Testing prompt injection defense: fail closed

When you test injection, treat the model as untrusted:

- the orchestrator should block dangerous tool calls even if the model requests them
- untrusted content must not be placed into instruction channels

Your tests should assert “fail closed” behavior: when unsure, the system blocks and escalates rather than acting.

---

## 7.6 Data leakage tests: protect outputs and logs

Leakage tests should cover both user-visible output and internal logs/traces.

Test inputs:

- fake API keys (`sk-...`, long random strings)
- fake PII (emails, phone numbers)
- internal hostnames or file paths

Expected behavior:

- agent output is redacted
- logs are redacted (or sensitive fields are omitted)
- the agent refuses to reveal secrets even when prompted

These tests can be mostly deterministic (regex detectors + policies), which is ideal for CI.

---

## 8. Trace-based regression: replay the system, not just the output

The best regression artifacts are traces:

- messages
- tool calls + results
- budgets
- stop reason

A regression test can assert:

- tool call sequence is unchanged (or changes are intentional)
- safety blocks still happen
- cost is within bounds

**Further reading (optional):** see [Observability and Tracing](/ai-agents/0034-observability-tracing/) and [Agent Evaluation Frameworks](/ai-agents/0042-agent-evaluation-frameworks/).

---

## 8.5 Snapshot tests for agent behavior (use carefully)

Snapshot tests (“compare output to a stored snapshot”) are tempting, but they’re fragile for agents.

If you use snapshots, snapshot the **structured parts**:

- tool call sequence
- stop reason
- schema-validated outputs
- safety violations list

Avoid snapshotting large freeform text that is expected to vary.

This gives you regression protection without constant churn.

---

## 8.6 Golden traces: how to keep them useful over time

Golden traces are saved “known good” trajectories that you replay in regression tests.

To keep them useful:

- store traces as structured artifacts (JSON) with redacted tool results
- pin tool outputs via record/replay
- version the agent prompt and the tool schemas

When a golden trace changes:

- require an explicit review/approval (like updating a snapshot test)
- record the reason (“prompt updated to handle new constraint”)

This prevents “silent drift” where the test suite slowly stops representing intended behavior.

---

## 9. A minimal test harness (pseudocode)

```python
def test_agent_scenario(agent, scenario):
    tools = FakeTools(scenario["tool_responses"])
    result = agent.run(scenario["input"], tools=tools)

    assert result.stop_reason == scenario["expected_stop_reason"]
    assert result.steps <= scenario["max_steps"]
    assert not result.violations  # safety violations
    assert tool_calls_match(result.trace.tool_calls, scenario["expected_tool_calls"])
```

The goal is stable, repeatable tests. The model can still be stochastic, but the test harness should reduce variability via tool replay and strict invariants.

---

## 9.5 Case study: a browsing agent test suite

Test cases to include:

- normal query with 2 sources
- conflicting sources (agent should surface uncertainty)
- malicious page injection text (agent must ignore and not call tools unsafely)
- stale page (agent should prefer recent sources if policy requires)

This is an ideal domain for tests because “citation faithfulness” is measurable.

**Further reading (optional):** see [Web Browsing Agents](/ai-agents/0036-web-browsing-agents/).

---

## 9.6 Case study: a code execution agent test suite

Code execution agents are a perfect target for strong tests because you can demand hard evidence (tests passing).

Scenario tests:

- compilation/runtime error → agent fixes code in ≤3 retries
- timeout case → agent changes strategy (smaller input, early exits)
- forbidden operation case (network/filesystem) → system blocks and agent proceeds safely

Assertions:

- sandbox limits were respected (time/memory)
- no network calls attempted (if policy)
- final output matches expected schema/format

**Further reading (optional):** see [Code Execution Agents](/ai-agents/0037-code-execution-agents/).

---

## 10. Common failure modes (and the test that catches them)

### 10.1 “Correct answer, wrong tool behavior”
Test: trajectory-level assertions on tool calls.

### 10.2 “Infinite retries”
Test: budget + repetition detector.

### 10.3 “Secret leakage”
Test: log scrubbing + regex leak detectors.

### 10.4 “Schema drift”
Test: schema validation on every run + fuzz tests.

### 10.5 “Stale browsing results”
Test: citation faithfulness and freshness policy checks (with recorded pages).

---

## 11. Summary & Junior Engineer Roadmap

Testing agents is systems testing:

1. **Test deterministic parts hard:** validators, redaction, budgets.
2. **Mock tools:** record/replay + failure simulations.
3. **Test trajectories:** tool choice, ordering, bounded retries.
4. **Assert invariants:** safety rules must never be violated.
5. **Replay traces for regression:** pin behavior across changes.

### Mini-project (recommended)

Build a test suite for one agent intent:

- 10 scenario tests with tool replay
- 5 adversarial tests (prompt injection, malformed tool output)
- 3 regression traces pinned to known-good behavior

Run it in CI and fail the build when success rate or safety degrades.

### Common rookie mistakes (avoid these)

1. **Testing only final text:** you’ll miss unsafe tool attempts and loops.
2. **No tool mocks:** your tests will be flaky and slow.
3. **No budgets:** an agent that loops will “pass” until it hits production costs.
4. **No regression suite:** you’ll ship changes that silently degrade behavior.

### Further reading (optional)

- Evaluation harnesses and scoring: [Agent Evaluation Frameworks](/ai-agents/0042-agent-evaluation-frameworks/)
- Trace collection and debugging: [Observability and Tracing](/ai-agents/0034-observability-tracing/)




---

**Originally published at:** [arunbaby.com/ai-agents/0043-testing-ai-agents](https://www.arunbaby.com/ai-agents/0043-testing-ai-agents/)

*If you found this helpful, consider sharing it with others who might benefit.*

