---
title: "Scaling Multi-Agent Systems"
day: 54
related_dsa_day: 54
related_ml_day: 54
related_speech_day: 54
collection: ai_agents
categories:
 - ai-agents
tags:
 - multi-agent
 - scaling
 - orchestration
 - coordination
 - reliability
 - safety
difficulty: Hard
subdomain: "Multi-Agent Architecture"
tech_stack:
 - Python
 - Message Queues
 - Redis
 - OpenTelemetry
scale: "Hundreds of agents, thousands of tasks, bounded cost"
companies:
 - Microsoft
 - Google
 - OpenAI
 - Anthropic
---

**"A single agent is a demo. Scaling agents is distributed systems with language models in the loop."**

## 1. Introduction

Multi-agent systems show up when:
- one agent is too slow (you need parallelism)
- one agent is too general (you need specialization)
- tasks naturally decompose (research, code, evaluate, deploy)

But scaling agents is hard because you inherit distributed systems problems:
- coordination overhead
- shared state consistency
- retries, deduplication, partial failures
- cost explosion (tokens + tools)

Thematic link today is **pattern matching and state machines**:
- you route work to the right agent (pattern matching over task types)
- you coordinate via protocols (explicit state machines)

### 1.1 What “scale” really means for agents
Scaling is not only “more QPS”. For multi-agent systems, “scale” shows up as:
- **more parallelism** (many concurrent tasks per job)
- **more heterogeneity** (different tools, permissions, and capabilities)
- **more state** (artifacts, memory, intermediate decisions)
- **more failure surface** (more tool calls, more partial failures)

If you only optimize for throughput, you often ship a system that is fast but untrustworthy.

### 1.2 A concrete target: “hundreds of agents” without chaos
A realistic “hard mode” target might look like:
- hundreds of workers (specialized agents) in a pool
- thousands of tasks/day (or more) across many tenants
- strict budgets per job (tokens, tool calls, dollars)
- predictable behavior (no infinite loops; retries are safe)

That target forces you to treat agents like distributed systems components:
typed messages, durable queues, idempotency, state stores, and observability.

---

## 2. Core Concepts

### 2.1 Roles and specialization

Common roles:
- **Planner**: breaks goals into subtasks and dependencies
- **Worker**: executes a subtask (research/coding)
- **Critic/Reviewer**: checks correctness and safety
- **Executor**: runs tools in a constrained environment

Specialization improves quality but increases coordination cost.

### 2.2 Communication primitives

- **Point-to-point**: direct requests (low overhead, less scalable routing)
- **Pub/Sub topics**: tasks published to a topic (scales, needs governance)
- **Broadcast**: expensive; use rarely (or with strict budgets)

### 2.3 Shared memory and consistency

Multi-agent systems need shared state:
- a task graph
- artifacts (docs/code)
- decisions and constraints

Consistency problems:
- two agents edit the same artifact concurrently
- one agent reads stale state and makes wrong decisions

Mitigation patterns:
- single writer per artifact (ownership locks)
- optimistic concurrency control (version checks)
- append-only logs + conflict resolution

### 2.4 Cost is the first-class constraint

Scaling multi-agent often fails due to cost:
- parallel agents multiply token usage
- tool calls become the dominant latency and cost

So the system must enforce budgets:
- max agents per job
- max tool calls per task
- max tokens per task and per job

### 2.5 Work representation: tasks, artifacts, and invariants
If you want multi-agent behavior to be predictable, you need a shared work model.
At minimum:
- **task**: unit of work with inputs, outputs, and dependencies
- **artifact**: durable output (doc, code diff, dataset, decision record)
- **invariants**: constraints that must always hold (budgets, allowed tools, required approvals)

This is the agent equivalent of schema design in data systems:
if the schema is vague, the system becomes vague.

### 2.6 Protocols and state machines (how you prevent “agent soup”)
Multi-agent coordination works best when you define explicit protocols:
- planner produces a DAG
- workers produce artifacts with version IDs
- critic produces approvals/rejections with reason codes
- executor runs tool calls with idempotency keys

You can model each task as a state machine:
`PENDING → RUNNING → (SUCCEEDED | FAILED | CANCELLED)`
with bounded retries and explicit transitions.

This is the same engineering mindset as pattern matching engines:
compiled states + bounded execution beats “free-form loops”.

### 2.7 Capability registries and routing (pattern matching over tasks)
As the number of agents grows, routing becomes a first-class problem:
- which agent can do this task?
- which agent is allowed to do this task (permissions)?
- which agent is best for this task (quality/latency/cost)?

A practical design uses:
- capability tags (research, code, review, execute)
- tool allowlists per role
- routing rules that match on task metadata (kind, tenant, sensitivity)

This is literally pattern matching: route tasks by matching metadata patterns to agent capabilities.

---

## 3. Architecture Patterns

### 3.1 Orchestrator + worker pool

``
 +-------------------+
 | Orchestrator |
 | (planner+router) |
 +----+---------+----+
 | |
 v v
 +-----------+ +-----------+
 | Worker A | | Worker B |
 | (research)| | (coding) |
 +-----------+ +-----------+
 |
 v
 +-----------+
 | Critic |
 +-----------+
``

The orchestrator assigns tasks, enforces budgets, merges results, and triggers retries.

### 3.2 DAG execution (dependency graph)

Represent work as a DAG:
- independent nodes run in parallel
- join nodes merge results
- failures trigger retries or replanning

This mirrors pipeline orchestration and avoids “ad-hoc loops”.

### 3.3 Debate / critique loops (bounded)

Pattern:
- two workers propose solutions
- critic selects or synthesizes

This improves quality, but must be bounded:
- max debate rounds
- max tokens per round
- stop early if confidence high

### 3.4 Hierarchical orchestration (team lead + specialists)
A common production topology is hierarchical:

``
 +--------------------+
 | Lead/Planner Agent |
 +---------+----------+
 |
 +------------+------------+
 | |
 v v
 +-----------+ +-----------+
 | Specialist| | Specialist|
 | (data) | | (infra) |
 +-----+-----+ +-----+-----+
 | |
 +-----------+-------------+
 |
 v
 +-------------+
 | Critic/QA |
 +-------------+
``

This keeps planning centralized (reduces coordination overhead) while still allowing parallel execution.

### 3.5 “Blackboard” / shared workspace pattern
In the blackboard pattern:
- agents write intermediate facts and artifacts to a shared workspace
- agents subscribe to updates and pick up tasks opportunistically

It’s flexible, but needs strong guardrails:
- ownership and locking (avoid concurrent edits)
- provenance and timestamps (avoid stale facts)
- budgets (avoid infinite “helpful” loops)

### 3.6 Map-reduce style agent workflows
For large research/summarization tasks:
- map: many workers process shards (documents, tickets)
- reduce: a synthesizer merges results
- verify: a critic checks consistency and citations

This looks like distributed data processing, because it is: you’re parallelizing “cognition” the same way you parallelize ETL.

### 3.7 Multi-tenant scaling (blast radius control)
In real deployments, “multi-agent” usually also means “multi-tenant”:
- different teams/customers with different tools and policies
- different risk tolerances

Patterns:
- isolate state stores per tenant (or strict ACLs)
- per-tenant canaries (don’t ramp globally)
- per-tenant budgets and quotas (cost control)

---

## 4. Implementation Approaches

### 4.1 Task routing by capability tags

Tag tasks:
- `kind=research`, `kind=code`, `kind=review`, `kind=execute`

Route them to workers with matching capabilities.
This is “pattern matching” over task metadata.

### 4.2 Durable queues and idempotent tasks

At scale, you need durability:
- tasks in a queue (Kafka/SQS/RabbitMQ)
- workers ack tasks when done
- retries happen automatically

Idempotency is mandatory:
- retries should not create duplicate side effects
- use idempotency keys for write tools

### 4.3 State store as the source of truth

Store:
- task states (pending/running/done/failed)
- artifacts and versions
- budgets consumed

Avoid storing “truth” only in chat logs.

### 4.4 Budget enforcement (tokens, tools, time, dollars)
Budgets must be enforced by the orchestrator/executor, not by prompts.
Practical budgets:
- max_steps per task
- max_tool_calls per task
- max_tokens per task and per job
- max wall-clock time per task (timeouts)

Budget outcomes should be explicit:
- if a task hits budget: return a partial result + reason code (`budget_exhausted`)
- escalate to a human or request user input instead of looping

### 4.5 Scheduling and backpressure
When you scale up, scheduling becomes the difference between “fast” and “meltdown”.
You need:
- priority queues (urgent tasks vs background)
- per-tenant rate limits
- backpressure when tools are overloaded (429s/timeouts)

A healthy system prefers “degrade gracefully” over “retry storm”.

### 4.6 Artifact versioning and merge protocols
Artifacts are shared state. So treat them like code:
- version IDs
- diffs
- single-writer locks or PR-based merges

If two agents can change the same file concurrently without a protocol, you’ll get nondeterministic outcomes and long debugging sessions.

### 4.7 Security boundaries (tool sandboxes)
Multi-agent systems multiply tool access. Strong patterns:
- run tools in sandboxes (per-tenant isolation)
- least-privilege tokens per role
- audit logs for every tool call
- allowlists and policy checks outside the model

This is the same safety logic as in data validation and pattern matching: untrusted inputs must go through deterministic gates.

---

## 5. Code Examples (Toy DAG Runner)

``python
from dataclasses import dataclass
from typing import List, Dict, Set


@dataclass
class Task:
 id: str
 kind: str
 deps: List[str]
 payload: str


def runnable(tasks: List[Task], done: Set[str]) -> List[Task]:
 return [t for t in tasks if t.id not in done and all(d in done for d in t.deps)]


def execute_plan(tasks: List[Task]) -> List[str]:
 """
 Toy sequential executor that respects dependencies.
 Production version would run runnable tasks in parallel with budgets and retries.
 """
 done: Set[str] = set()
 order: List[str] = []

 while len(done) < len(tasks):
 ready = runnable(tasks, done)
 if not ready:
 raise ValueError("Cycle or missing dependency detected")
 for t in ready:
 # placeholder "execute"
 done.add(t.id)
 order.append(t.id)
 return order
``

This highlights the core: dependency management is a first-class part of scaling.

### 5.1 Adding budgets and idempotency (minimal production skeleton)
Below is a conceptual sketch of what “budgeted execution” looks like.
The important part is not the exact code; it’s the **explicit accounting** and **explicit failure reasons**.

``python
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Budget:
 max_steps: int
 max_tool_calls: int
 steps: int = 0
 tool_calls: int = 0


class IdempotencyStore:
 def __init__(self) -> None:
 self._seen: Dict[str, str] = {}

 def check_or_set(self, key: str, value: str) -> Optional[str]:
 """
 Returns existing value if key exists, else sets it and returns None.
 """
 if key in self._seen:
 return self._seen[key]
 self._seen[key] = value
 return None


def enforce_budget(b: Budget, is_tool_call: bool) -> None:
 b.steps += 1
 if is_tool_call:
 b.tool_calls += 1
 if b.steps > b.max_steps:
 raise RuntimeError("budget_exhausted:steps")
 if b.tool_calls > b.max_tool_calls:
 raise RuntimeError("budget_exhausted:tool_calls")
``

In a real system:
- budgets are per task and per job (nested budgets)
- idempotency keys are stored durably (Redis/DB), not in-memory
- budget exhaustion is handled gracefully (partial output + escalation), not as a crash

---

## 6. Production Considerations

### 6.1 Reliability: retries, dedupe, and partial failure

Failures are normal:
- tool timeouts
- worker crashes
- network partitions

Mitigations:
- idempotent tasks
- retry budgets
- dedupe keys for side-effectful actions
- compensating actions (undo) where possible

### 6.1.1 Partial failure is the default, not the exception
In distributed tools, “some things succeeded” is common:
- tool executed the action but the response timed out
- a write succeeded but the ack was dropped

If your agent retries blindly, it amplifies incidents.
Production patterns:
- idempotency keys for any write tool
- store tool call outcomes in the state store (so retries can be conditional)
- “two-phase commit” for risky actions (draft → confirm)

### 6.2 Observability

You need traces per:
- job
- task
- agent
- tool call

Track:
- tokens, steps, tool calls
- task queue latency
- failure reasons and retry counts

### 6.2.1 Tracing that works: stable IDs and structured events
To debug multi-agent systems you need to reconstruct causality:
- job_id → task_id → step_id → tool_call_id

If you log only free-form text, you can’t answer:
- “which task caused the deployment?”
- “why did we retry 12 times?”

Use structured events and propagate IDs across:
- orchestration logs
- tool executors
- state store updates

### 6.3 Safety and governance

Multi-agent multiplies action surface:
- more tool calls
- more parallel actions

Safety patterns:
- per-role tool allowlists
- policy engine that can block actions
- HITL for high-risk steps

### 6.3.1 Safe rollouts (shadow → canary → ramp)
Agents regress in surprising ways because:
- prompts and tool schemas evolve
- retrieval corpora evolve
- user distributions differ at scale

So ship changes like you ship infra:
- shadow mode comparisons (no real actions)
- canary cohorts (small traffic, tight budgets)
- ramp with rollback triggers (policy violations, error rate, cost spikes)

### 6.4 Concurrency control for artifacts

If two agents can edit the same file:
- use locks (single writer)
- or use merge protocols (PR-based workflow)
- always keep version history (diffs) for auditability

### 6.5 Evaluation and continuous testing
Multi-agent systems need more than “unit tests”.
High-signal evals include:
- golden workflows (end-to-end tasks)
- adversarial tests (prompt injection, malicious tool outputs)
- tool failure simulation (timeouts, 429s, partial writes)
- long-horizon consistency tests (does it loop? does it contradict itself?)

This is the agent version of CI: every production incident should become a new eval case.

### 6.6 Cost controls (the reality check)
The failure mode at scale is usually cost:
- parallelism multiplies tokens
- tool calls dominate tail latency

Practical cost levers:
- cap parallelism (max active workers per job)
- cache expensive retrieval/tool results
- prefer smaller models for low-risk subtasks
- stop early when confidence is high (don’t “keep thinking”)

### 6.7 State store schema (what you persist so the system is debuggable)
If you want reliability, persist state explicitly. A practical minimal schema:

- **Job**
 - `job_id`, `tenant_id`, `created_at`
 - `goal` (sanitized), `bundle_version`
 - budgets: `max_tokens`, `max_tool_calls`, `max_wall_time_s`
 - status: `RUNNING | SUCCEEDED | FAILED | CANCELLED`

- **Task**
 - `task_id`, `job_id`, `kind`, `deps`
 - assigned agent role, attempt count, timestamps
 - status transitions with reason codes (`tool_timeout`, `budget_exhausted`, `policy_denied`)

- **Artifact**
 - `artifact_id`, `task_id`
 - versioning metadata (hash, diff, parent_version)
 - ownership lock state (single writer) or PR state (open/merged)

- **Tool call**
 - `tool_call_id`, `task_id`
 - tool name, args hash, latency, outcome
 - **idempotency_key** for any write action

Why this matters:
- it lets you answer “what happened?” without scraping free-form chat logs
- it enables safe retries (idempotency)
- it enables postmortems and regression tests

### 6.8 Incident response patterns (agents need SRE discipline)
When (not if) something goes wrong, you need fast, deterministic mitigations:

- **Kill switches**
 - disable high-risk tools instantly (email/send/deploy/delete)
 - force “draft-only” mode (no side effects)

- **Degraded modes**
 - reduce parallelism
 - reduce model size / shorten context
 - disable retrieval sources that are injecting bad content

- **Rollback**
 - roll back the agent bundle (prompt + tools + policies + routing)
 - roll back tool schema versions if the executor is rejecting calls

- **Audit workflow**
 - list all tool calls in the incident window
 - verify idempotency (no duplicate side effects)

The critical mindset shift:
> treat agent behavior regressions like production incidents, not like “prompt tuning”.

---

## 7. Common Pitfalls

1. **Unbounded parallelism**: cost blowups and noisy results.
2. **No shared source of truth**: agents disagree and drift.
3. **No idempotency**: retries create duplicate side effects.
4. **No budgets**: loops and tool spam.
5. **No evaluation**: quality regressions discovered by users.

### 7.1 Pitfall: “coordination tax” overwhelms the benefits
Teams often assume “more agents = faster”.
But coordination has overhead:
- more messages
- more merging
- more inconsistent partial results

If you don’t model work as a DAG and define merge semantics, you get:
- duplicated work
- contradictory artifacts
- expensive debate loops

### 7.2 Pitfall: concurrency without merge protocols
If two agents can edit the same artifact without a protocol:
- results become nondeterministic
- “random” regressions appear

The safest default is often:
- single writer + reviewer
- or PR-based merges with a critic gate

---

## 8. Best Practices

1. Model work as a DAG with explicit dependencies.
2. Keep a durable state store and version artifacts.
3. Enforce budgets at every layer (job/task/tool).
4. Use strong observability and rollback/kill switches.
5. Prefer structured state and deterministic control planes over “prompt magic”.

### 8.1 A practical “GA checklist” for multi-agent systems
Before you allow agents to take real actions at scale:

- **State and execution**
 - durable task store (not chat logs)
 - explicit task state machine with bounded retries
 - idempotency keys for write tools

- **Safety**
 - per-role tool allowlists and RBAC
 - policy engine outside the model
 - HITL for high-risk actions

- **Observability**
 - job/task/tool traces with stable IDs
 - cost metrics (tokens/tool calls per job)
 - dashboards by tenant/segment

- **Evaluation**
 - golden workflows
 - adversarial/prompt-injection tests
 - tool failure simulation

If you can’t confidently say “we can roll back in minutes”, you’re not ready for high-autonomy actions.

---

## 9. Real-World Examples

- Research systems: parallel web browsing + synthesis.
- Code review systems: coder + reviewer + executor.
- Enterprise workflows: planner + policy checker + executor with approvals.

### 9.1 Example: code-change workflow (the “agent CI” pattern)
A common multi-agent workflow for code changes:
- planner: proposes a change plan and file list
- coder: implements changes
- tester/executor: runs checks in a sandbox
- reviewer/critic: reviews diffs and verifies constraints

The key is that each step has:
- explicit inputs/outputs (artifacts)
- explicit budgets
- explicit rollback (revert PR / revert bundle)

### 9.2 Example: enterprise ticket triage (multi-tenant + RBAC)
For enterprise support:
- tasks are routed by type (billing, technical, compliance)
- tools differ by role (support can read; only certain roles can write)
- auditability is mandatory

This is where multi-agent becomes governance-heavy: without RBAC and audit logs, you will fail compliance reviews.

---

## 10. Future Directions

- learned routing (auto-select best agent for a task)
- typed protocols and state machines
- continuous evaluation pipelines for multi-agent behaviors

### 10.1 Typed workflows and “agent bundles”
A high-leverage direction is bundling:
- prompt templates
- tool schemas
- policy rules
- routing configuration
into a single versioned “agent bundle”.

This makes rollbacks and canaries practical: you promote bundles through environments the same way you promote model versions.

### 10.2 Better schedulers: cost-aware and reliability-aware
Schedulers will likely incorporate:
- predicted token/tool cost per task
- tool health signals (avoid routing to degraded tools)
- tenant-level quotas and priorities

At scale, scheduling is where you turn “agent intelligence” into “operable systems”.

---

## 11. Key Takeaways

1. Scaling agents is distributed systems engineering.
2. Coordination, state, and budgets dominate cost and reliability.
3. Use DAGs, durable state, idempotency, and observability to make systems operable.

### 11.1 Connections to other topics (shared theme)
The shared theme is **pattern matching and state machines**:
- “Wildcard Matching” is DP over a matching state machine; multi-agent orchestration is also a state machine (task states + transitions).
- “Pattern Matching in ML” emphasizes compilation, budgets, and observability; multi-agent systems need the same control-plane discipline.
- “Acoustic Pattern Matching” uses coarse→fine pipelines (retrieve → verify); multi-agent systems often use the same structure (plan → execute → verify).

If you build your agents around explicit state, explicit transitions, and explicit budgets, you get predictability — and predictability is what lets you scale safely.

---

**Originally published at:** [arunbaby.com/ai-agents/0054-scaling-multi-agent-systems](https://www.arunbaby.com/ai-agents/0054-scaling-multi-agent-systems/)

*If you found this helpful, consider sharing it with others who might benefit.*

