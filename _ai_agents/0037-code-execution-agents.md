---
title: "Code Execution Agents"
day: 37
collection: ai_agents
categories:
  - ai-agents
tags:
  - code-execution
  - sandboxing
  - security
  - containers
  - evals
  - tool-use
  - deterministic
difficulty: Medium-Hard
related_dsa_day: 37
related_ml_day: 37
related_speech_day: 37
---

**"Let agents run code safely: sandbox execution, cap damage, and verify outputs like a production system."**

## 1. What is a code execution agent?

A **code execution agent** is an AI system that doesn’t just *write* code—it can also **run** code to:

- compute answers (math, statistics, data transforms)
- validate hypotheses (“does this regex match?”)
- generate artifacts (plots, reports, generated files)
- debug programs (reproduce bugs, run tests, narrow down failures)

If you’ve ever asked a model to “calculate the result” or “simulate this,” you’ve probably seen the core limitation: a model can be confident and still be wrong. Execution closes that loop:

> Instead of “I think the answer is…”, the agent can say “I ran the code, here are the results.”

But execution also introduces real risk:

- untrusted code can exfiltrate secrets
- code can delete files or fork-bomb your machine
- a bug can burn money (infinite loops, huge memory allocations)

So the main engineering goal is:

**Make execution a tool that is useful and predictable—without letting it become a security incident.**

---

## 2. The core architecture: Plan → Generate → Execute → Verify

The most reliable pattern is a loop with explicit stages:

```text
User Request
  |
  v
Planner (LLM)  -> decides if execution is needed
  |
  v
Coder (LLM)    -> writes minimal code to answer the question
  |
  v
Sandbox Runner -> runs code in a restricted environment
  |
  v
Verifier       -> checks outputs, edge cases, and correctness signals
  |
  v
Answer Writer  -> returns result + evidence (outputs, logs)
```

### Why split roles?

- The **Coder** should optimize for correctness and minimalism.
- The **Runner** should be dumb and deterministic: “run code, return stdout/stderr/artifacts.”
- The **Verifier** should be skeptical: “does this output actually answer the question?”

This keeps the system debuggable. If something goes wrong, you know whether it was a generation problem, a sandbox problem, or a verification problem.

---

## 3. Threat model: assume the generated code is hostile

Treat model-generated code as **untrusted**. Even if the user is benign, the model can produce unsafe code accidentally.

You should assume the code might try to:

- read secrets from environment variables
- read files on disk (`~/.ssh`, config files, credentials)
- open network connections (exfiltration, DDoS)
- fork processes, spawn threads, or consume resources
- exploit the runtime via known vulnerabilities

**Junior engineer rule:** if you allow arbitrary Python execution on your production machine, you are already compromised—you just haven’t noticed yet.

---

## 4. Sandboxing options (from “good enough” to “serious”)

There are multiple levels of sandboxing. Your choice depends on risk and scale.

### 4.1 Process sandbox (local subprocess)

**What it is:** run code as a subprocess on the same machine (e.g., `python -c ...`).

**Pros:**
- simplest to implement
- fastest iteration for prototypes

**Cons:**
- very hard to secure properly
- shares OS kernel, file system access, and often network access

Use this only for local experiments and never for untrusted multi-user systems.

### 4.2 Container sandbox (recommended baseline)

**What it is:** run code in a container with:

- a minimal filesystem
- no mounted host secrets
- strict CPU/memory limits
- optional: no network

**Pros:**
- good isolation for most product workloads
- strong operational tooling (images, logs, limits)

**Cons:**
- still shares the kernel (containers are not full VMs)
- needs careful configuration (capabilities, mounts)

### 4.3 VM / microVM sandbox (highest isolation)

**What it is:** run code inside a lightweight VM (e.g., microVM) or a full VM.

**Pros:**
- strong isolation boundary
- safer for hostile code and high-stakes environments

**Cons:**
- more complex and slower than containers
- more expensive to operate at scale

If you’re building a platform where many users can run code, VM-style isolation is often worth it.

---

## 4.5 A concrete baseline sandbox profile (copy this)

If you want a safe baseline without overthinking it, start with a “no surprises” profile like this:

- **CPU:** 1 vCPU
- **Memory:** 256–512MB
- **Timeout:** 2–5 seconds wall clock
- **Disk:** 50–200MB writable space
- **Network:** disabled
- **Filesystem:** read-only root, a single writable `/tmp`
- **User:** non-root
- **Capabilities:** drop all Linux capabilities

Then add privileges *only* when your product requires them. Most “calculation” and “data cleanup” tasks work fine under this profile.

---

## 5. The execution contract: define exactly what the sandbox does

Your sandbox should behave like a predictable service with a strict API:

**Inputs**
- `language` (python/js/…)
- `code`
- optional `stdin`
- optional `files` (small inputs, or references to stored artifacts)
- `limits` (timeout, memory, cpu)

**Outputs**
- `stdout`
- `stderr`
- `exit_code`
- `artifacts` (files produced by the run, with size caps)
- `metrics` (runtime ms, max memory, cpu time)

This contract prevents chaos. The agent can’t “kind of run something” in a half-broken state; it either runs within the contract or fails with a clear error.

---

## 6. Safety controls that matter in production

### 6.1 Timeouts (stop infinite loops)

Every run must have a hard timeout (wall clock). A second useful limit is CPU time.

- Wall clock timeout catches I/O hangs and deadlocks.
- CPU time catches compute-heavy loops.

### 6.2 Memory limits (stop accidental OOM)

Memory limits protect the machine and also prevent “accidental denial of service” where one run eats all RAM.

### 6.3 Disk limits (stop dumping huge files)

Agents can generate large outputs by mistake (e.g., writing a 5GB file). Put limits on:

- writable disk space inside the sandbox
- artifact size returned to the orchestrator

### 6.4 No network by default

For most code execution tasks (math, parsing, data transforms), you don’t need network. Disabling network:

- prevents exfiltration
- prevents the agent from downloading untrusted dependencies at runtime
- makes runs deterministic

If you must allow network, allowlist domains and log every request.

### 6.5 Read-only filesystem + controlled mounts

Mount only what you need:

- provide input files explicitly (or via a blob store reference)
- keep the rest read-only
- never mount host credential directories

### 6.6 Dependency policy

The biggest footgun is “pip install anything the agent wants.”

Better options:

- prebuild images with a fixed set of safe dependencies
- use a curated internal package mirror
- block installs entirely for the default sandbox

This makes runs reproducible and avoids supply-chain attacks.

---

## 6.7 Secrets handling: keep the sandbox “empty”

The most common real-world failure is accidental secret exposure. Engineers often run sandboxes in the same environment as the orchestrator, where environment variables contain API keys.

Safe defaults:

- **No inherited environment:** start the sandbox with an empty environment.
- **Explicit allowlist:** if the code needs a variable, provide it explicitly and minimally.
- **Output scrubbing:** before storing logs, redact anything that matches key-like patterns.

If you can avoid putting secrets in the sandbox entirely, do it. It’s dramatically easier than trying to “secure” untrusted code that can read process state.

---

## 7. Determinism and reproducibility (debugging becomes possible)

In production, you want to be able to replay:

- the exact code
- the exact inputs
- the exact environment

Practical techniques:

- pin dependency versions in the sandbox image
- record the image digest used per run
- store code + inputs in a run record (trace)

If the system is not reproducible, debugging becomes guesswork.

---

## 7.5 Data access patterns: keep inputs explicit and bounded

Execution agents often fail when inputs are “implicit”:

- “Use the data from my database”
- “Load the file from my drive”
- “Just fetch the dataset from the internet”

In a safe system, the sandbox should only see what you explicitly pass in. Practical patterns:

- **Small inputs inline:** pass small JSON/CSV as a file artifact.
- **Large inputs by reference:** download data outside the sandbox, scan it, then mount a read-only file into the sandbox.
- **Row/byte caps:** cap input sizes to prevent accidental huge loads.

This keeps runs predictable and prevents the agent from “discovering” resources it shouldn’t have access to.

---

## 8. Verification: don’t trust stdout blindly

Execution reduces hallucinations, but it doesn’t eliminate them. The agent can still:

- run the wrong code
- misinterpret outputs
- choose the wrong dataset
- answer a different question than the user asked

So you want verification steps such as:

### 8.1 Output-to-question alignment

Ask: “Does this output actually answer the user’s question?”

Example failure:
- user asks for “top 5”
- code prints “top 10”

### 8.2 Sanity checks and invariants

If the task has invariants, validate them:

- probabilities sum to ~1
- counts are non-negative
- shapes match expected dimensions

### 8.3 Cross-check with an alternate method (when cheap)

For some tasks, you can verify with a second approach:

- brute force for small inputs
- a reference library call
- a known formula check

This is the same mindset as testing: trust, but verify.

---

## 8.5 Testing strategy: make the agent prove correctness

If you let the agent execute code, you can ask it to generate **tests** and use the sandbox to run them. This is a powerful reliability upgrade because the agent must commit to concrete examples.

Two practical test styles:

### 8.5.1 Example-based tests
- Provide 5–10 small inputs with expected outputs.
- Include edge cases: empty input, large values, negative numbers, Unicode strings.
- Add “boring” cases: already-sorted input, duplicate rows, missing values.

### 8.5.2 Property-style checks (lightweight)
Even without a full property-testing library, you can assert invariants:

- sorting output is non-decreasing
- output length equals input length (unless you’re filtering)
- probabilities sum to ~1
- parsing doesn’t drop rows silently (track counts)

The goal is not to mathematically prove correctness; it’s to catch the most common wrong-code failures quickly and cheaply.

---

## 9. Error handling: turn failures into useful feedback loops

Execution failures are common. The key is to make them *actionable* for the agent:

### 9.1 Capture structured errors

Return error information as structured data:

- syntax errors (line/column)
- runtime errors (exception type, message)
- timeout vs OOM vs forbidden syscalls

### 9.2 Retry with constraints

A good retry prompt includes:

- the error message
- the failing code snippet
- explicit instruction: “fix only the minimal needed lines”

### 9.3 Circuit breakers

Never let an agent loop indefinitely trying to fix code. Put a cap:

- max retries per task (e.g., 3)
- max total execution time budget

If it still fails, return a clean failure with logs so a human can take over.

---

## 9.5 Common execution failures (and what they usually mean)

In production, failures cluster into predictable buckets:

- **SyntaxError / TypeError:** invalid code or mismatched types. Usually fixed by a targeted retry prompt that includes the exact stack trace line.
- **ImportError:** dependency not available. Either add the dependency to the sandbox image or push the agent toward a dependency-free solution.
- **Timeout:** algorithm too slow or stuck. Prompt the model to reduce complexity, add early exits, and print progress only when necessary.
- **MemoryError / OOM kill:** code loaded too much data or created huge arrays. Ask for streaming / chunked processing.
- **Permission denied / forbidden syscall:** the code tried to do something disallowed (network, filesystem). This often means your constraints weren’t explicit enough, or the agent is attempting a risky approach.

Seeing these patterns quickly is how you mature from “it fails sometimes” to “we know why it fails and how to fix it.”

---

## 10. Observability: what to log for code execution agents

For each run, log:

- code hash
- sandbox image digest
- resource limits (cpu/mem/time)
- runtime metrics
- stdout/stderr (with redaction policies)
- artifacts metadata (names, sizes)

These logs are not “nice to have.” They are how you debug:

- why something timed out
- why a result changed after a deploy
- why the model started producing slow code

---

## 10.5 Cost control: execution budget + token budget

Execution isn’t free:

- sandbox compute costs money (CPU time)
- logs and artifacts cost storage
- retries cost tokens

Practical controls:

- **Execution budget:** max total runtime per request (e.g., 10 seconds across all retries).
- **Artifact budget:** max artifact bytes returned (e.g., 5MB).
- **Token budget:** max model tokens across generation + retries.

If you enforce budgets, your system stays predictable even under worst-case prompts and edge cases.

---

## 10.6 Result caching: don’t re-run what you already ran

If your agent iterates (“try → fail → fix → retry”), it’s easy to re-run the same code with the same inputs multiple times. That wastes compute and increases latency.

A simple caching strategy:

- compute a hash of `(language, code, input_files_hash, stdin)`
- store `{stdout, stderr, exit_code, artifacts_metadata}` under that hash for a short TTL
- if the agent tries the same run again, return the cached result

This is especially useful when the verifier asks the coder to “explain again” but the code itself didn’t change.

---

## 11. Practical patterns that make execution agents reliable

### 11.1 “Minimal program” rule
Ask the model to write the smallest possible program:

- no extra dependencies
- no unnecessary printing
- no complex frameworks

This reduces the surface area for failures.

### 11.2 “Pure function” interface
Encourage code shaped like:

```python
def solve(input_data):\n    ...\n    return output\n```

Then your runner can pass inputs and capture outputs consistently.

### 11.3 “Explain then execute” (but keep it short)
Have the agent explain *briefly* what it intends to do before executing. This improves alignment and makes debugging easier if the code is wrong.

### 11.4 Artifact-first outputs (when relevant)
If the goal is a plot, a report, or a file, treat that as the primary output and validate:

- file exists
- file size reasonable
- file format correct

---

## 11.5 Multi-language execution: choose the smallest hammer

You don’t need to support every language to get value.

A pragmatic approach:

- Start with **Python** for parsing, math, data cleanup, and quick scripts.
- Add **JavaScript** only if you need browser-adjacent logic or your team prefers the ecosystem for JSON-heavy transforms.
- Avoid compile-heavy languages early; toolchains add complexity and increase failure modes.

If you do support multiple languages, keep the contract the same (`stdout/stderr/exit_code/artifacts/metrics`) so the agent doesn’t need a different mental model per runtime.

---

## 11.6 Hardening checklist (quick security review before you ship)

Before you expose code execution to real users, run through a short checklist:

1. **Network default deny:** Confirm the sandbox cannot reach the internet unless you explicitly allow it.
2. **No secret inheritance:** Confirm the sandbox environment is empty by default (no host env passthrough).
3. **No host mounts:** Confirm you are not mounting host directories containing credentials, configs, or logs.
4. **Non-root execution:** Confirm the process runs as a non-root user.
5. **Resource limits enforced:** Confirm timeouts and memory limits actually kill runs (don’t rely on “polite” signals).
6. **Artifact caps:** Confirm you cap returned artifact size and redact logs before storage.
7. **Dependency control:** Confirm installs are pinned or blocked; no dynamic `pip install` from arbitrary indexes.

This checklist is intentionally boring. Boring is good. It’s how you prevent the two worst outcomes: data leaks and runaway costs.

---

## 12. Case study: “Data cleanup agent” for CSV normalization

Imagine a user asks:

> “Normalize this CSV: trim whitespace, standardize dates, and remove duplicates.”

Execution is ideal here because:

- transformations are deterministic
- results can be validated (row counts, schema)
- the agent can produce the cleaned file artifact

Reliable approach:

1. Extract requirements into a checklist
2. Generate a minimal script
3. Run it in a sandbox with the CSV mounted read-only
4. Validate:
   - duplicates reduced (or not) with counts
   - date parse success rate
   - schema stability
5. Return:
   - cleaned CSV artifact
   - summary of changes
   - warnings for rows that failed parsing

---

## 12.5 Case study: “Algorithmic helper” for tricky edge cases

Execution agents shine when the user’s task is algorithmic and easy to get subtly wrong by pure reasoning.

Example requests:

- “Given these intervals, merge overlaps and return the result.”
- “Compute a similarity score across these strings and sort by score.”
- “Simulate this queue/stack logic and return the final state.”

A reliable pattern is:

1. **Generate a reference implementation** (simple and clear).
2. **Generate a few tests** (including edge cases).
3. **Run** and compare expected vs. actual.
4. **Only then** produce the final answer.

Even if the code is small, running it catches the classic mistakes: off-by-one errors, wrong sorting keys, and failure to handle empty input.

---

## 13. Summary & Junior Engineer Roadmap

Execution agents are powerful because they replace “guessing” with “running.”

To make them production-grade:

1. **Sandbox hard:** containers or VMs, no network by default, strict limits.
2. **Define a contract:** inputs/outputs/limits as a clear API.
3. **Verify outputs:** align results to the question, check invariants, cross-check when cheap.
4. **Control retries:** structured errors + bounded retry loops.
5. **Log everything important:** code hash, image digest, limits, metrics, stderr.

If you internalize one principle, make it this:

**Generated code should be treated like user input.** You wouldn’t run random code pasted from the internet on your production machine. Treat the model’s code the same way.

### Mini-project (recommended)
Build a tiny execution service:

- A `run_python(code, files, timeout_ms)` endpoint.
- No network, 2s timeout, 256MB memory.
- Return `stdout/stderr/exit_code`.

Then connect an agent that can:

- write a function
- run it on test inputs
- fix errors up to 3 retries
- produce a final answer with the actual output

### Common rookie mistake (avoid this)
Don’t let the agent “execute” by calling external services directly from generated code. Keep side effects in **separate, audited tools**. Generated code should ideally be pure computation on explicit inputs, inside a sandbox, producing explicit outputs.

**Further reading (optional):** If you want the broader system view (roles, state, recovery, tracing), see [Role-Based Agent Design](/ai-agents/0031-role-based-agent-design/), [State Management and Checkpoints](/ai-agents/0032-state-management-checkpoints/), [Error Handling and Recovery](/ai-agents/0033-error-handling-recovery/), and [Observability and Tracing](/ai-agents/0034-observability-tracing/).




---

**Originally published at:** [arunbaby.com/ai-agents/0037-code-execution-agents](https://www.arunbaby.com/ai-agents/0037-code-execution-agents/)

*If you found this helpful, consider sharing it with others who might benefit.*

