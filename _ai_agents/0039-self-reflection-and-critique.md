---
title: "Self-Reflection and Critique"
day: 39
collection: ai_agents
categories:
  - ai-agents
tags:
  - self-reflection
  - critique
  - verification
  - debiasing
  - evals
  - reliability
  - multi-pass
difficulty: Medium-Hard
related_dsa_day: 39
related_ml_day: 39
related_speech_day: 39
---

**"Make agents less overconfident: separate drafting from critique, force evidence, and turn failures into actionable feedback."**

## 1. Why “self-reflection” matters (and what it really is)

Most production failures are not “the model can’t write English.” They’re:

- wrong tool choice
- missing an edge case
- citing weak evidence as fact
- continuing confidently despite contradictory signals

**Self-reflection** is an engineering pattern that makes an agent pause and evaluate its own output before it commits to it.

In practice, self-reflection is:

- a second pass with a different objective (criticize, verify, test, find risks)
- a structured checklist (“did we satisfy constraints?”)
- sometimes a different model or role (“skeptical auditor”)

What it is *not*:

- infinite deliberation
- a vague “think harder” prompt
- a replacement for tests, schemas, or tool validation

Self-reflection is most valuable when it’s **operationalized**: you can measure it and you can enforce it.

---

## 2. The core architecture: Draft → Critique → Revise → Freeze

A reliable pattern is a controlled multi-pass pipeline:

```text
Draft (fast, constructive)
  |
  v
Critique (skeptical, rule-based)
  |
  v
Revise (apply fixes only)
  |
  v
Freeze (lock answer, emit evidence)
```

### Why this works

- The drafter is allowed to be creative and expansive.
- The critic is allowed to be harsh and “annoying.”
- The reviser is allowed to change only what the critic flags.

This separation prevents a common failure mode: the agent “rationalizes” its own answer instead of correcting it.

---

## 3. Reflection targets: what should the critic look for?

Good critiques are specific and actionable. Here are the top categories to check.

### 3.1 Requirement compliance
- Did we follow the requested format?
- Did we answer the question asked (not a nearby question)?
- Did we respect constraints (no unsafe actions, no speculation)?

### 3.2 Evidence and grounding
- Are claims supported by citations, quotes, or tool outputs?
- Are there any claims that are “vibe-based” (confident but unsupported)?

### 3.3 Logical correctness
- Are there contradictions?
- Are there missing steps?
- Are edge cases handled?

### 3.4 Risk and safety
- Is any step unsafe or irreversible?
- Does the plan require approvals?
- Are we accidentally leaking sensitive data?

### 3.5 Cost and efficiency
- Did we use too many tools?
- Did we repeat work?
- Could we stop earlier?

---

## 4. The most important trick: make critique structured

If critique is freeform, you’ll get comments like “looks good.” That’s useless.

A structured critique should produce fields like:

- `verdict`: PASS / FAIL
- `issues`: list of issues with severity and location
- `fixes`: concrete patch suggestions
- `questions`: clarification questions if needed

Example (conceptual):

```json
{
  "verdict": "FAIL",
  "issues": [
    {"severity": "high", "issue": "Claim lacks evidence", "location": "Section 2"},
    {"severity": "medium", "issue": "Edge case missing for empty input", "location": "Algorithm"}
  ],
  "fixes": [
    {"action": "add", "details": "Add test case for empty input"},
    {"action": "replace", "details": "Remove claim or add citation"}
  ]
}
```

**Further reading (optional):** if your agent uses strict schemas, see [Structured Output Patterns](/ai-agents/0035-structured-output-patterns/).

---

## 4.5 A ready-to-use critique rubric (copy/paste)

If you’re building a critic, it’s tempting to start with a vague prompt like “review this.” That usually produces low-signal feedback.

Instead, use a rubric that forces the critic to check concrete properties.

### 4.5.1 Evidence rubric (grounding)
- **FAIL (high)** if any “hard claim” has no evidence.
  - Hard claim = numbers, policies, “X supports Y,” “best practice is…”
- **FAIL (medium)** if evidence exists but is weak:
  - citation doesn’t contain the claim
  - quote is too vague or unrelated
- **PASS** only if every hard claim has supporting evidence or is clearly marked as uncertain.

### 4.5.2 Safety rubric (risk)
- **FAIL (high)** if any step performs a write/destructive action without validation or approval gates.
- **FAIL (high)** if the answer includes secrets, credentials, or internal paths.
- **FAIL (medium)** if it suggests risky operations without warning or alternatives.

### 4.5.3 Completeness rubric (requirements)
- **FAIL (high)** if any explicit user requirement is missing (format, fields, constraints).
- **FAIL (medium)** if the answer is correct but incomplete (missing edge cases, missing steps).

### 4.5.4 Efficiency rubric (cost)
- **WARN** if the plan uses unnecessary tools or repeats calls.
- **WARN** if the response could be shorter or more direct without losing correctness.

The key is that the critic must produce **specific issues** and **specific fixes**, not “looks good.”

---

## 5. Reflection without leakage: keep “thoughts” private

In many systems, you do not want to expose internal critique to the user because:

- it can be confusing (“why are you arguing with yourself?”)
- it can leak sensitive system instructions

A good design is:

- keep critique and revision internal
- expose only the final answer (plus evidence)

This keeps the UX clean while still benefiting from the reliability gains.

---

## 5.5 Reflection outputs: what to show the user (and what not to)

Even if critique is internal, users often benefit from a small amount of transparency:

- **Show:** final answer, citations/test output, and a short “assumptions” list.
- **Hide:** internal system rules, raw tool arguments that might include secrets, and long internal critique transcripts.

A nice compromise is to expose a short “Quality notes” section:

- “Citations verified for each claim.”
- “Assumptions: …”
- “Limits: …”

That builds trust without overwhelming the user.

---

## 6. Where reflection helps most (high leverage use cases)

### 6.1 Tool calls and structured outputs
Reflection can validate:

- tool arguments match schema
- missing required fields
- semantic constraints (allowed domains, safe ranges)

### 6.2 Web browsing and citations
Reflection can enforce:

- “quote supports claim”
- minimum number of sources for key claims
- freshness heuristics

**Further reading (optional):** see [Web Browsing Agents](/ai-agents/0036-web-browsing-agents/) for safe browsing pipelines.

### 6.3 Code generation and debugging
Reflection can enforce:

- tests pass
- outputs match question
- no unsafe operations (network, filesystem writes) in generated code

**Further reading (optional):** see [Code Execution Agents](/ai-agents/0037-code-execution-agents/) for sandbox and test loops.

---

## 6.5 Reflection for tool calls: validate semantics, not just JSON

Many teams stop at “the JSON parsed.” That’s necessary, but not sufficient. A good critic validates semantics:

- **Domain allowlists:** is the URL/tool target allowed?
- **Path allowlists:** is the file path inside an approved directory?
- **Risk classification:** is this action read-only or write?
- **Budget compliance:** does it exceed tool call limits or cost limits?

Example: a tool call can be valid JSON and still be dangerous:

```json
{ "action": "delete", "path": "/" }
```

Parsing success does not imply safety. Critique is the place to enforce “safe defaults.”

---

## 7. A practical “critic prompt” (role + rubric)

The best critics are narrow and uncompromising.

Critic persona:

- skeptical
- concise
- allergic to unsupported claims

Rubric example:

1. Flag any claim without evidence.
2. Flag any missing constraint.
3. Flag any ambiguous step.
4. Flag any unsafe action.
5. Return PASS only if no high-severity issues remain.

**Junior engineer tip:** make the critic’s job easier by giving it the same structured state the agent uses (objective, constraints, evidence list).

---

## 7.5 A “critic contract” that prevents vague feedback

If you want consistent critique quality, treat critique itself as an API.

Minimum contract:

- Every issue must include:
  - **severity**: high / medium / low
  - **category**: evidence / correctness / safety / format / efficiency
  - **location**: where the issue appears (section name, field name, step id)
  - **why it matters**: one sentence
  - **fix**: an actionable fix (add/remove/replace with concrete text)

If the critic cannot produce a concrete fix, it should output a **question** instead (what info is missing?).

This contract is how you avoid the common “critic says it’s wrong but can’t explain how to fix it” failure mode.

---

## 8. The “self-critique fallacy”: why a single model can still miss issues

Self-reflection is not magic. If the same model drafted and critiques, it can:

- repeat the same bias
- overlook the same edge case
- defend its initial reasoning

Mitigations:

### 8.1 Role separation
Use very different prompts. The critic should be pessimistic and strict.

### 8.2 Model diversity (optional)
Use a smaller/cheaper model as a critic for “format + schema” checks and a stronger model for deep reasoning— or use two different providers for high-stakes tasks.

### 8.3 External checks
Whenever possible, prefer deterministic checks:

- JSON schema validation
- unit tests
- citation matching scripts

Reflection should complement these, not replace them.

---

## 8.5 Calibrating the critic: avoid “too strict” and “too lenient”

Critics fail in two opposite ways:

### 8.5.1 Critic is too lenient (false negatives)
Symptoms:

- everything passes
- obvious missing evidence isn’t flagged

Fixes:

- tighten rubric (define “hard claims” explicitly)
- require evidence fields (quote + URL) for each claim
- add an “abstain” rule: if evidence is missing, FAIL with a clear request

### 8.5.2 Critic is too strict (false positives)
Symptoms:

- it blocks harmless outputs
- it demands citations for trivial statements

Fixes:

- introduce severity levels and allow WARNs
- define what needs citations (numbers, policies, factual claims) vs. what doesn’t (definitions, simple clarifications)
- allow the critic to pass with WARNs when safety is not at risk

### 8.5.3 Use “confidence budgets”
Force the critic to express confidence:

- If the critic is uncertain, it should ask a clarification question or request more evidence.

This is how you avoid a critic that behaves like a random gate.

---

## 9. Cost control: reflection budgets and stop rules

Reflection increases token usage. You need budgets:

- max critique passes (often 1 is enough)
- max total retries (e.g., 2–3)
- “stop if no progress” rule (don’t loop)

One practical approach:

- If the critic returns FAIL twice with the same high-severity issue, escalate or ask the user for clarification.

This keeps systems from turning into expensive loops.

---

## 9.5 Reflection + retries: the “progress or stop” rule

The most expensive failure mode is looping between revise and critique without actually improving.

Add a simple rule:

- Track a **progress score** per attempt (e.g., number of high-severity issues remaining).
- If the score doesn’t improve between attempts, stop and escalate.

Example policy:

- Attempt 1: 3 high issues
- Attempt 2: 3 high issues (no change) → stop and ask for input / escalate

This turns runaway loops into bounded behavior.

---

## 10. Observability: measure whether reflection actually helps

If you add reflection and your success rate doesn’t improve, something is wrong.

Track:

- critique PASS/FAIL rate
- top reasons for FAIL (missing evidence, schema errors, edge cases)
- post-revision improvement (did a FAIL become a PASS?)
- cost impact per successful task

This turns reflection into an engineering lever, not a philosophical feature.

**Further reading (optional):** see [Observability and Tracing](/ai-agents/0034-observability-tracing/) for trace structure and eval hooks.

---

## 10.5 What to log from critique (without leaking sensitive content)

Log just enough to debug and improve:

- `verdict`
- issue categories (evidence/safety/format/logic)
- severity counts (how many high/medium/low)
- which checks failed (e.g., “missing citation”, “schema mismatch”)
- whether revision fixed the issue

Avoid logging raw user secrets or full tool outputs. Treat critique logs like production logs: useful, minimal, redacted.

---

## 11. Implementation sketch: a minimal reflection loop (pseudocode)

```python
def draft_then_critique(objective: str, constraints: list[str], context: dict) -> dict:
    draft = llm.generate({"objective": objective, "constraints": constraints, "context": context})

    critique = llm.criticize({
        "objective": objective,
        "constraints": constraints,
        "draft": draft,
        "rubric": ["evidence", "correctness", "safety", "format"]
    })

    if critique["verdict"] == "PASS":
        return {"final": draft, "critique": critique}

    revised = llm.revise({
        "draft": draft,
        "critique": critique,
        "rule": "Fix only issues listed. Do not add new claims without evidence."
    })

    # Optional: run critic again with a small budget
    critique2 = llm.criticize({"objective": objective, "constraints": constraints, "draft": revised})
    if critique2["verdict"] == "PASS":
        return {"final": revised, "critique": critique2}

    return {"status": "NEEDS_INPUT", "final": revised, "critique": critique2}
```

Key design choice: the revise step is constrained to apply fixes, not rewrite everything.

---

## 11.5 Reflection + deterministic validators (recommended hybrid)

The strongest setup is:

1. **Deterministic validators first** (schema validation, allowlists, unit tests)
2. **LLM critic second** (semantic issues, completeness, clarity)

Why this ordering?

- deterministic validators are cheap and reliable
- they prevent the critic from wasting tokens on obvious failures

If a validator fails, the critic should focus on “how to fix the validator error,” not re-reviewing everything.

---

## 12. Case study: “Report agent” that must cite sources

Goal: produce a report that is accurate and well-cited.

Failure mode without critique:

- agent writes a nice report
- citations are missing or weak
- a few claims are unsupported

With critique:

1. Draft produces the report.
2. Critic checks every paragraph:
   - “Does this paragraph include citations?”
   - “Does the citation support the paragraph’s claim?”
3. Reviser adds citations or removes claims.
4. Finalizer outputs the report with an evidence list.

This is one of the highest ROI uses of reflection because it is measurable: citations either support claims or they don’t.

---

## 12.5 Case study: “Code agent” that must pass tests before answering

Goal: produce a correct answer for an algorithmic problem (or a data transform) by generating and running code.

Without critique:

- the agent outputs code that “looks right”
- edge cases fail (empty input, duplicates, off-by-one)
- the agent still answers confidently

With critique + tests:

1. Drafter writes a small solution + a small set of tests (including edge cases).
2. Runner executes tests in a sandbox and returns the output.
3. Critic checks:
   - tests actually cover edge cases
   - test output indicates pass
   - the printed result matches the user’s asked output format
4. Reviser only changes code/tests where failures are shown.

This pattern gives you a concrete success signal (“tests passed”) and turns correctness from “confidence” into “evidence.”

---

## 12.6 Case study: “Tool-using agent” that must not perform unsafe writes

Goal: let an agent use tools to get work done, without accidentally executing risky actions.

Common unsafe pattern:

- agent reads untrusted input
- agent proposes a write action immediately (delete, update, deploy)
- there is no gate that forces validation or approval

A safer architecture:

1. **Draft:** propose a plan and classify each step as read-only or write.
2. **Critique:** enforce policy:
   - “No writes unless explicitly allowed”
   - “Writes require validation + approval gate”
   - “All tool calls must pass allowlists”
3. **Revise:** split the output into:
   - a safe, read-only execution phase
   - a proposed change set (diff / patch) that requires approval
4. **Freeze:** return the safe phase result plus the proposed change set and the risks.

What you gain:

- the agent can still move fast (read-only work is autonomous)
- high-risk operations become reviewable artifacts (diffs, patches)
- the critic becomes a consistent policy enforcer

This is the pattern that turns “agents with credit cards” into “agents with purchase approvals.”

---

## 13. Summary & Junior Engineer Roadmap

Self-reflection is a reliability multiplier when it’s engineered, not improvised:

1. **Split roles:** drafting and critique have different goals.
2. **Structure the critique:** PASS/FAIL + issues + fixes.
3. **Prefer deterministic checks:** schemas, tests, and scripts first.
4. **Budget it:** one critique pass is often enough; avoid loops.
5. **Measure impact:** success rate, error categories, and cost.

If you internalize one principle, make it this:

**Reflection should change behavior, not just generate more text.** If the critic can’t point to a concrete issue and a concrete fix, you’re paying extra tokens for noise.

### Mini-project (recommended)

Build a tiny “draft + critic” system:

- Draft writes a structured JSON result.
- Critic validates schema + flags unsupported fields.
- Reviser fixes only flagged issues.
- Log the before/after and measure how often critique prevents an error.

### Further reading (optional)

- For debugging and measuring critique loops: [Observability and Tracing](/ai-agents/0034-observability-tracing/)
- For safer multi-step systems: [Error Handling and Recovery](/ai-agents/0033-error-handling-recovery/)
- For browser-grounded evidence: [Web Browsing Agents](/ai-agents/0036-web-browsing-agents/)




---

**Originally published at:** [arunbaby.com/ai-agents/0039-self-reflection-and-critique](https://www.arunbaby.com/ai-agents/0039-self-reflection-and-critique/)

*If you found this helpful, consider sharing it with others who might benefit.*

