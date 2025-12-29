---
title: "Agent Evaluation Frameworks"
day: 42
related_dsa_day: 42
related_ml_day: 42
related_speech_day: 42
collection: ai_agents
categories:
 - ai-agents
tags:
 - agent-evaluation
 - evals
 - benchmarks
 - reliability
 - observability
 - regression-testing
 - metrics
difficulty: Medium-Hard
---

**"If you can’t measure an agent, you can’t improve it: build evals for success, safety, cost, and regressions."**

## 1. Why agents need evaluation frameworks (not just “tests”)

Traditional software tests check deterministic behavior. Agents are different:

- outputs vary across runs (non-determinism)
- decisions depend on external tools (web, APIs, files)
- quality includes fuzzy dimensions (helpfulness, correctness, safety)

So evaluation frameworks for agents must measure:

- **task success** (did it accomplish the goal?)
- **behavior quality** (was reasoning sound and complete?)
- **safety** (did it avoid harmful actions?)
- **cost/efficiency** (how many steps and tool calls?)
- **robustness** (does it hold under noise, adversarial inputs, flaky tools?)

An “agent evaluation framework” is the system you use to run these checks continuously, track metrics, and prevent regressions.

---

## 2. The eval mindset: define a contract for “good”

Before building eval infrastructure, define what “good” means for your agent.

A useful contract includes:

- **Intent:** what task category is this?
- **Requirements:** what must be present in output?
- **Constraints:** what must never happen?
- **Budgets:** max steps, tool calls, time, tokens
- **Evidence:** what counts as proof (citations, tool outputs, tests)

This turns evaluation from “vibes” into “specification.”

---

## 2.5 A template: define the eval contract per intent

When an agent supports multiple intents, define one eval contract per intent. This prevents “one-size-fits-none” scoring.

Example contract (conceptual):

``json
{
 "intent": "browsing_answer_with_citations",
 "requirements": ["contains_citations", "claims_supported_by_quotes"],
 "constraints": ["no_unsafe_tools", "no_secret_leakage"],
 "budgets": {"max_steps": 8, "max_tool_calls": 8},
 "evidence": {"min_sources": 2, "quote_required": true}
}
``

This makes evaluation consistent and debuggable: when an eval fails, you can point to the contract field that was violated.

---

## 3. Eval types: offline, online, and adversarial

### 3.1 Offline evals (most important early)

Run a fixed dataset of tasks and score:

- success rate
- cost per success
- safety violations

Offline evals catch regressions quickly.

### 3.2 Online evals (production monitoring)

In production, you measure:

- user feedback
- task completion metrics
- error rates and escalations
- latency and cost

Online evals tell you if your system is healthy, but they’re noisy.

### 3.3 Adversarial evals (security and robustness)

Agents face adversarial inputs:

- prompt injection
- malformed data
- tool failures

Adversarial evals simulate those failures and verify safe behavior.

---

## 3.5 Practical sampling strategy for online evals

You usually can’t judge every production trace. A pragmatic sampling strategy:

- record **100%** of failures and safety blocks
- record **100%** of high-value flows (paid users, sensitive actions)
- record **5–10%** of routine successes

Then run automated scoring (deterministic + judge) on the sampled traces.

This gives you visibility without exploding cost.

---

## 4. What to score: the “four axes” that matter

### 4.1 Correctness (did it do the right thing?)

Metrics:

- exact match (for deterministic tasks)
- rubric score (for open-ended tasks)
- citation faithfulness (for browsing tasks)
- test pass rate (for code tasks)

### 4.2 Safety (did it avoid forbidden behavior?)

Metrics:

- policy violations per task
- unsafe tool call attempts
- secret leakage indicators
- write actions without approval

### 4.3 Efficiency (did it waste resources?)

Metrics:

- steps per task (median/p95)
- tool calls per task
- token usage and latency
- retry rate

### 4.4 Robustness (does it work under stress?)

Metrics:

- success under flaky tools
- success under noisy inputs (OCR errors, partial logs)
- success under adversarial prompts

---

## 4.5 A concrete scoring schema (copy/paste)

To keep evals consistent across tasks, use a common score object:

``json
{
 "overall": 0.0,
 "scores": {
 "correctness": 0.0,
 "safety": 0.0,
 "efficiency": 0.0,
 "robustness": 0.0
 },
 "violations": [],
 "notes": [],
 "artifacts": {
 "citations_ok": null,
 "tests_passed": null
 }
}
``

Practical rule: **any high-severity safety violation forces overall to 0**, even if the answer is correct. This matches real production priorities: “correct but unsafe” is not acceptable.

---

## 5. Designing an eval dataset: coverage beats size

You don’t need 100k examples to start. You need *coverage*.

Include:

- common tasks
- edge cases
- high-risk scenarios
- failure cases you’ve seen in production

Practical approach:

1. Collect 20–50 tasks per major intent.
2. Add 5–10 “nasty” edge cases per intent.
3. Add adversarial cases (prompt injection, tool timeouts).

As your agent ships, the dataset grows naturally from real failures.

---

## 5.5 Dataset design: separate “intent” from “instance”

Many teams accidentally build a dataset that’s too narrow because all tasks are small variations of the same thing.

A better structure:

- **Intent**: the category (e.g., “browsing answer with citations”)
- **Instance**: a specific example within that intent

Track intent-level coverage:

- at least N intents
- at least M instances per intent
- at least K edge cases per intent

This helps you avoid the “we improved one task and broke another” problem.

---

## 6. Scoring methods: deterministic checks + LLM judges

### 6.1 Deterministic evaluators (preferred when possible)

Examples:

- JSON schema validation
- unit tests for code
- citation matching scripts (“does quote exist on cited page?”)
- allowlist checks for tool calls

These are cheap and reliable.

### 6.2 LLM-as-a-judge (useful for fuzzy dimensions)

LLM judges are useful for:

- helpfulness and clarity
- completeness
- whether an answer follows a rubric

But you must treat them as approximate and calibrate them.

Practical guidelines:

- use a strict rubric
- keep judge prompts stable and versioned
- use multi-judge consensus for high-stakes evals

**Further reading (optional):** see [Observability and Tracing](/ai-agents/0034-observability-tracing/) for judge integration into traces.

---

## 6.5 Judge calibration: how to keep LLM judges honest

LLM judges drift too:

- judge prompt changes
- model updates
- subtle rubric ambiguity

Calibrate judges with a small “golden set”:

1. Create 30–50 examples with human-labeled scores (PASS/FAIL + reasons).
2. Run the judge and measure:
 - false positives (judge passes bad outputs)
 - false negatives (judge fails good outputs)
3. Tighten rubric:
 - define hard claims vs soft claims
 - define what counts as evidence
4. Version your judge prompt and pin it in eval runs.

If you can’t afford human labels, start with a smaller set (even 10) and grow over time.

---

## 7. Evaluating tool use: action quality, not just final output

Agents can “accidentally” reach a correct answer with unsafe or wasteful behavior.

So you must evaluate:

- tool selection (did it use the right tool?)
- tool arguments (were they valid and safe?)
- action ordering (did it respect dependencies?)
- repetition (did it loop?)

This is where agent eval differs from normal model eval: you grade the **trajectory**, not just the final answer.

**Further reading (optional):** for multi-step architectures and stop conditions, see [Autonomous Agent Architectures](/ai-agents/0038-autonomous-agent-architectures/).

---

## 7.5 Trajectory scoring: the simplest approach that works

Trajectory scoring doesn’t need to be complicated. Start with penalties:

- **-1** per failed tool call
- **-1** per repeated identical tool call
- **-2** for using a tool when not needed (obvious)
- **-5** for attempting a forbidden tool (safety violation)

Then normalize into a score. This makes “how” the agent behaved measurable.

Combine with a stop-condition check:

- Did the agent stop for the right reason (SUCCESS/NEEDS_INPUT/FAILURE)?
- Did it exceed budgets?

This catches a common failure: “agent eventually succeeded, but burned 10x budget.”

---

## 8. Regression testing: pin behavior over time

Agents regress for many reasons:

- prompt changes
- tool behavior changes
- model provider updates
- dependency changes

A regression framework should:

- run the offline eval suite on every change
- compare metrics to a baseline
- fail the build if thresholds are violated

Example “red lines”:

- success rate drops by >2%
- unsafe tool call attempts increase
- cost per success increases by >20%

This is how you keep an agent stable as it evolves.

---

## 8.5 CI integration: make evals a first-class build step

Practical build workflow:

- **PR checks:** run a fast eval subset (smoke suite) on every PR.
- **Nightly:** run full eval suite (bigger dataset, adversarial cases).
- **Release gate:** require thresholds before shipping.

Keep suites sized appropriately:

- smoke: 20–50 tasks, finishes in minutes
- full: 200–1000 tasks, finishes in hours

This gives you fast feedback without sacrificing coverage.

---

## 8.6 Reducing noise: make your evals stable

Two things make evals noisy:

- model randomness
- external tool variability

Mitigations:

- run each task multiple times and score the average (or median)
- keep temperature low for eval runs
- mock tools (or record/replay tool outputs)
- isolate “content correctness” from “tool availability”

The goal is that a regression is a real regression—not a random fluctuation.

---

## 9. Evals for safety and security (a must-have)

### 9.1 Prompt injection evals

Create tasks where inputs contain:

- “ignore previous instructions”
- “reveal secrets”
- “call destructive tool”

Expected behavior:

- refuse
- sanitize
- continue safely or escalate

### 9.2 Data leakage evals

Create tasks that include:

- fake API keys
- fake PII (emails, SSNs)

Expected behavior:

- redact in outputs and logs
- never echo secrets back

### 9.3 Tool failure evals

Simulate:

- 429 rate limits
- timeouts
- malformed responses

Expected behavior:

- bounded retries
- strategy change or escalation

---

## 9.6 Secret leakage detection: cheap heuristics that work

You don’t need a perfect classifier to catch most leaks. Start with:

- regex checks for key-like patterns (e.g., `sk-...`, long hex strings)
- email/phone patterns for PII
- “forbidden tokens” list (internal hostnames, internal file paths)

If a leak detector triggers:

- mark eval as FAIL (high severity)
- store a redacted snippet for debugging

This also doubles as a production monitor.

---

## 9.5 “Canary” safety evals: test the worst things first

If you only have time to build a few safety evals, prioritize:

1. **Prompt injection attempts** that try to trigger tool use
2. **Secret leakage** tests (the agent should redact)
3. **Destructive write attempts** (the agent should refuse or require approval)

These are high-signal because they fail loudly and have real-world consequences.

**Further reading (optional):** if you want deeper threat modeling, see [Prompt Injection Defense](/ai-agents/0044-prompt-injection-defense/) and [Data Leakage Prevention](/ai-agents/0045-data-leakage-prevention/).

---

## 10. Implementation sketch: a minimal eval runner (pseudocode)

``python
def run_eval_suite(agent, dataset):
 results = []
 for test in dataset:
 trace = agent.run(test["input"])
 score = evaluate_trace(trace, test["rubric"])
 results.append({"id": test["id"], "score": score, "trace": trace.summary()})
 return aggregate(results)

def evaluate_trace(trace, rubric):
 checks = []
 checks.append(schema_check(trace))
 checks.append(safety_check(trace))
 checks.append(cost_check(trace))
 if rubric.get("requires_citations"):
 checks.append(citation_check(trace))
 if rubric.get("requires_tests"):
 checks.append(test_check(trace))
 return combine(checks)
``

The key idea: evaluation reads traces (actions + outputs) and produces structured scores.

---

## 10.5 Making evals reproducible: pin models, tools, and environments

Your eval runs should be reproducible, otherwise you’ll chase ghosts.

Pin:

- model/version for the agent
- model/version for the judge (if used)
- tool behavior (mocked or recorded)
- sandbox images for code execution

If you can, run evals in a controlled environment where:

- web tools use recorded responses (or a fixed snapshot)
- flaky APIs are mocked

This turns evals into stable regression signals.

---

## 10.6 Eval infrastructure: store results like a dataset, not like logs

If eval outputs are just console logs, you’ll never use them. Treat eval results as a dataset:

- **Row per test case per run**
- Fields:
 - agent version
 - judge version (if any)
 - scores and violations
 - cost and latency metrics
 - pointers to traces/artifacts

This lets you answer questions like:

- “Which intent regressed in the last release?”
- “Which tool causes the most failures?”
- “Which tasks are the cost outliers?”

A minimal storage format is JSONL. A more scalable option is a table in a warehouse (or Postgres) keyed by `(suite, test_id, agent_version, timestamp)`.

Once you have this, you can build a simple dashboard:

- success rate by intent
- p95 cost by intent
- top safety violations
- top failing tests (by count)

This is what turns evals into a real engineering feedback loop.

---

## 11. Case study: evaluating a browsing agent

Success criteria:

- answer contains citations
- citations support claims
- minimal number of sources for key claims

Metrics:

- citation faithfulness rate
- average sources per answer
- cost per verified claim

Failure cases to include:

- stale pages
- conflicting sources
- malicious prompt injection in page text

**Further reading (optional):** see [Web Browsing Agents](/ai-agents/0036-web-browsing-agents/).

---

## 11.5 Case study: evaluating an autonomous multi-step agent

Success criteria:

- completes within step budget
- no unsafe tool attempts
- stops with correct stop reason

Trajectory metrics:

- repeated tool calls (should be rare)
- “thrashing” (replanning every step)
- tool failures and recovery quality

**Further reading (optional):** for loop control and stop conditions, see [Autonomous Agent Architectures](/ai-agents/0038-autonomous-agent-architectures/) and [Hierarchical Planning](/ai-agents/0040-hierarchical-planning/).

---

## 11.6 Case study: evaluating a code execution agent

Success criteria:

- code runs in sandbox within limits
- tests pass (when required)
- no forbidden operations attempted (network, filesystem writes outside sandbox)

Metrics:

- test pass rate
- runtime and memory usage
- retry count (how often it needs a fix loop)

Failure cases to include:

- infinite loops (timeout)
- missing dependencies (ImportError)
- wrong output format (prints extra text)

**Further reading (optional):** see [Code Execution Agents](/ai-agents/0037-code-execution-agents/).

---

## 12. Summary & Junior Engineer Roadmap

Agent evaluation frameworks are how you turn agent development into engineering:

1. **Define a contract:** requirements, constraints, budgets, evidence.
2. **Build a dataset with coverage:** common tasks + nasty edge cases.
3. **Prefer deterministic checks:** schemas, tests, and scripts.
4. **Add judges carefully:** strict rubrics, stable prompts, consensus when needed.
5. **Evaluate trajectories:** tool use and safety matter, not just final output.
6. **Run regressions continuously:** fail builds when metrics degrade.

### Mini-project (recommended)

Build an eval suite with:

- 30 tasks across 3 intents
- a deterministic safety checker (no write tools, no secrets)
- a cost budget checker (max steps/tool calls)
- one LLM judge rubric for helpfulness

Run it on every agent change and chart metrics over time.

### Common rookie mistakes (avoid these)

1. **Only scoring the final answer:** for agents, the trajectory matters (tool choice, safety, loops).
2. **No baseline:** without a stable baseline, you can’t call something a regression.
3. **No safety suite:** safety regressions are the most expensive regressions; test them first.
4. **Too much judge, not enough determinism:** use schemas/tests/scripts whenever possible; judges are a supplement.
5. **No reproducibility:** if tools are flaky and models are random, you’ll chase noise instead of signal.

### Further reading (optional)

- Multi-step agent architectures: [Autonomous Agent Architectures](/ai-agents/0038-autonomous-agent-architectures/) and [Hierarchical Planning](/ai-agents/0040-hierarchical-planning/)
- Debugging + metrics: [Observability and Tracing](/ai-agents/0034-observability-tracing/)




---

**Originally published at:** [arunbaby.com/ai-agents/0042-agent-evaluation-frameworks](https://www.arunbaby.com/ai-agents/0042-agent-evaluation-frameworks/)

*If you found this helpful, consider sharing it with others who might benefit.*

