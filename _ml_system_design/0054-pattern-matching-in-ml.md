---
title: "Pattern Matching in ML"
day: 54
related_dsa_day: 54
related_speech_day: 54
related_agents_day: 54
collection: ml_system_design
categories:
 - ml-system-design
tags:
 - pattern-matching
 - rules
 - weak-supervision
 - data-quality
 - feature-engineering
 - mlops
difficulty: Hard
subdomain: "Data Processing"
tech_stack:
 - Python
 - Spark
 - RE2
 - Great Expectations
scale: "10B events/day, low-latency rule evaluation"
companies:
 - Google
 - Meta
 - Netflix
 - Amazon
---

**"Most ML pipelines are quietly powered by pattern matching—rules, validators, and weak labels before the model ever trains."**

## 1. Problem Statement

“Pattern matching” shows up everywhere in ML systems:
- data cleaning (regex, wildcard rules)
- entity extraction (emails, phone numbers)
- feature generation (URL parsing, query templates)
- weak supervision (labeling functions)
- safety filters (PII detection)

The challenge is designing a pattern matching system that is:
- correct (matches what you intend)
- fast (billions of events)
- safe (no regex DoS, bounded runtime)
- maintainable (versioned rules, test suites)
- observable (why did this rule match?)

### 1.1 Why this matters (and why it’s “ML system design”, not just regex)
Pattern matching is frequently the *first* correctness gate in an ML pipeline:
- it decides what data is “valid” enough to train on
- it decides what gets labeled (weak supervision)
- it decides what gets filtered (PII/safety/compliance)

When pattern matching is wrong, downstream model quality issues are often **misdiagnosed**:
- a spike in training loss might be “model drift” or might be “a new rule started deleting 30% of positives”
- a drop in CTR might be “ranking regression” or might be “query parser template rule changed and broke features”

This is the same systems lesson you see in speech and agents:
> Most “model regressions” are actually pipeline regressions.

### 1.2 A concrete example: one rule change, many downstream effects
Suppose you have a URL template rule used for:
- feature generation (extract `campaign_id`)
- spam filtering (block suspicious redirect patterns)
- weak labels (mark “likely fraud” if URL matches a known template)

A small rule update can:
- change feature distributions (training drift)
- change class balance (label drift)
- change serving behavior (filtering drift)

So “pattern matching” needs the same rigor as model deployment:
versioning, evaluation, canaries, and observability.

### 1.3 Thematic link: pattern matching as “routing + state machines”
Across systems, pattern matching is fundamentally about:
- **routing**: which handler/rule/agent should see this input?
- **state machines**: once a pattern is compiled, matching is transitions over states

That’s why the same mental model shows up in wildcard DP, acoustic alignment (DTW), and multi-agent orchestration: different domains, same engineering primitive.
Once you see matching as “compiled states + bounded execution”, you can design systems that are fast, safe, and debuggable at real scale.

---

## 2. Understanding the Requirements

### 2.1 Functional requirements
- Support multiple pattern types:
 - literals and wildcards
 - regex
 - dictionary/allowlist rules
 - structured templates (e.g., URL patterns)
- Provide match outputs:
 - boolean match
 - captured groups (when needed)
 - match reason codes (rule IDs)
- Provide governance:
 - rule versioning
 - rollout and rollback
 - ownership and approvals

### 2.2 Non-functional requirements
- Low latency evaluation
- Safe execution (avoid catastrophic backtracking)
- High throughput (batch + streaming)
- Debuggability (explanations)

### 2.3 Scale and latency targets (make this concrete)
Different use cases imply different budgets:

- **Streaming inference features**
 - p95 rule evaluation: single-digit milliseconds
 - strict tail latency: avoid worst-case patterns

- **Log processing / batch ETL**
 - throughput dominates (TB/day)
 - latency per record can be higher, but total CPU cost matters

- **Safety / PII filters**
 - false negatives are expensive (compliance risk)
 - explainability and audit logs matter as much as latency

At 10B events/day, you’re roughly in the regime where:
- you must avoid per-event heavyweight parsing
- you must precompile and vectorize
- you must treat the matcher as shared infrastructure (multi-tenant)

### 2.4 Output contract (what a “match” returns)
In production you want more than a boolean:
- **rule_id** (stable identifier)
- **rule_version** (for rollback and audits)
- **match_span** or captured groups (if used for feature extraction)
- **severity** (warn/block/tag)
- **explanation** (why it matched; which clause)

This is what makes downstream debugging possible: “which rule caused this row to be filtered?” is a first-class question.

---

## 3. High-Level Architecture

``
 +-------------------+
 | Rule Authoring |
 | (Git, UI, PRs) |
 +---------+---------+
 |
 v
 +-------------------+
 | Rule Compiler |
 | (regex → automata |
 | wildcard → DP) |
 +---------+---------+
 |
 v
 +-------------------+ +-------------------+
 | Rule Registry | ---> | Runtime Evaluator |
 | (versions, ACLs) | | (stream + batch) |
 +---------+---------+ +---------+---------+
 | |
 v v
 +-------------------+ +-------------------+
 | Observability | | Downstream |
 | (matches, stats) | | (labels, filters) |
 +-------------------+ +-------------------+
``

### 3.1 Control plane vs data plane (the operability split)
Pattern matching systems typically split into:

- **Control plane**
 - rule authoring UX + approvals
 - compilation and validation (linting, safety checks)
 - versioning, rollout, rollback
 - ownership and audit logs

- **Data plane**
 - low-latency evaluator (streaming and/or serving)
 - batch evaluator (Spark/Flink jobs)
 - caching and rate limiting

This split is important because it lets you:
- make the data plane boring and fast
- keep the “human workflows” (approval, review) in the control plane

### 3.2 Rule lifecycle (how rules move through the system)
Practical lifecycle:
1. author rule (Git PR or UI)
2. run lint + safety checks (ReDoS checks, complexity budgets)
3. run offline evaluation (golden tests + sampled production logs)
4. compile + publish to registry
5. shadow mode / canary (compare match rates, latency)
6. ramp to full traffic
7. monitor, roll back if metrics regress

If your system lacks steps 2–6, you will end up treating incidents as “we should be more careful next time” instead of encoding the learning in the platform.

---

## 4. Component Deep-Dives

### 4.1 Pattern types and trade-offs

| Pattern type | Pros | Cons | Best for |
|-------------|------|------|----------|
| Wildcards | simple, fast | limited expressiveness | filename-like rules |
| Regex | expressive | risk of ReDoS, complex | extraction, complex rules |
| Dictionaries | precise | maintenance, coverage | entity lists, allowlists |
| Templates | structured | requires parsers | URLs, logs, structured fields |

### 4.1.1 Rule language design (the hidden hard part)
The fastest way to create an unmaintainable matcher is to let rules become “random regex strings”.
A more production-friendly approach is to define a small rule language with:
- stable IDs and metadata (owner, severity, rollout)
- explicit pattern types (wildcard vs regex vs template)
- explicit semantics (full match vs substring match)
- explicit actions (tag vs block vs extract)

This makes it possible to:
- review rules meaningfully (diff shows semantic change, not just regex edits)
- test rules systematically
- explain matches (“rule X matched because clause Y fired”)

### 4.1.2 Compilation strategies (how you get speed without losing correctness)
You almost never want to “interpret” patterns per record.
You want to compile patterns into runtime artifacts:

- **Wildcards**
 - compile to automata where possible
 - or use a safe matching algorithm (DP or greedy) with bounded runtime

- **Multiple literal patterns**
 - compile into a trie / Aho-Corasick automaton (multi-pattern substring matching)

- **Regex**
 - compile into NFA/DFA-like machines (or use a guaranteed-linear engine like RE2)

This is the same conceptual move as the DSA “Wildcard Matching” problem:
matching is a state machine; compilation turns “text patterns” into “states + transitions”.

### 4.2.1 Wildcards: DP vs greedy and why engines care
There are multiple correct ways to match wildcards:
- DP is straightforward and safe, but can be \(O(MN)\) per evaluation.
- greedy two-pointer solutions can be very fast in practice, but require careful correctness reasoning.

In ML pipelines, the key is not “which algorithm is cutest”; it’s:
- can you **bound runtime**?
- can you **precompile** and reuse?
- can you **explain** why it matched?

### 4.2 Safety: avoiding regex DoS

Never run untrusted “backtracking regex” in production.
Prefer:
- RE2 (linear-time regex engine)
- precompilation
- timeouts/budgets

#### 4.2.1 Safety checklist for pattern systems
“Safety” is broader than ReDoS:
- **Resource safety**: bounded CPU/memory per record; no catastrophic cases.
- **Semantic safety**: rules don’t over-match and silently delete training data.
- **Change safety**: rule updates are staged and reversible.

Practical guardrails:
- disallow or heavily restrict advanced backtracking features (catastrophic constructs)
- cap pattern length and number of alternations
- cap total rules evaluated per record (candidate pruning)
- run rule diffs through offline replay tests before rollout

#### 4.2.2 Budgets as a first-class design primitive
Budgets are the simplest, most reliable safety mechanism:
- max rules evaluated per event
- max total match time per event (soft budget → degrade, hard budget → fail closed/open)
- max number of matches emitted (avoid cardinality explosions downstream)

These “bounded execution” ideas are exactly what makes multi-agent systems operable too: without budgets, rare cases dominate cost and reliability.

### 4.3 Observability and explanations

Pattern systems need:
- rule IDs in outputs
- match counts per rule
- top matched rules by segment
- sampling of matched examples (privacy-safe)

#### 4.3.1 Explanations: what a good “why did this match?” looks like
If you want engineers to trust the system, explanations must be:
- deterministic (not “sometimes”)
- attributable (rule_id + clause_id)
- segment-aware (“this is only happening on android+bluetooth+region=IN”)

Common explanation artifacts:
- “matched because regex group X captured Y”
- “matched because wildcard prefix matched path segment Z”
- “matched because dictionary rule list_id=foo contained token bar”

#### 4.3.2 The cardinality trap in observability
It’s easy to build an observability system that DDoSes itself:
- logging raw matched strings
- high-cardinality labels (user_id, raw URL)

Production pattern:
- log **rule IDs** and **hashed/capped** examples
- maintain a small “debug cohort” pipeline for deeper samples

### 4.4 Governance patterns that scale beyond one team
Governance sounds bureaucratic until you run this at scale.
Without governance, rule systems fail in predictable ways:
- rules get loosened to stop pages, then become useless
- rules get tightened by one team and break another team’s pipeline
- nobody knows who owns “high impact” rules

Practical governance patterns:
- **ownership metadata** on every rule (team, on-call)
- **approval tiers**: low-risk tag rules auto-merge; block rules require review
- **severity levels**: warn vs block vs extract, with different rollout gates
- **audit logs**: who changed what, when, and why (especially for PII rules)

This keeps the system usable for many teams without turning it into a centralized bottleneck.

### 4.5 Testing: treat rules like code (because they are)
Rules are executable logic.
So they need the same testing stack you expect for production code:

- **Golden tests**
 - small curated examples with expected match outputs
 - should include edge cases and “near misses”

- **Property-based tests**
 - fuzz around boundaries (e.g., URL templates, Unicode, whitespace)
 - catch unexpected over-matching

- **Replay tests**
 - run against sampled production logs (privacy-safe)
 - diff match rates and top rule matches by segment

### 4.6 Explainability UX (what teams actually want)
Engineers don’t want to read raw regex.
They want:
- “which rule fired?”
- “what part matched?”
- “did this change after a deploy?”

If your UI can answer those in seconds, adoption goes up and the “disable the validator” instinct goes down during incidents.

---

## 5. Data Flow

### 5.1 Streaming
- compile rules to a runtime format
- evaluate per event
- emit match flags and reason codes

### 5.2 Batch
- apply rules on partitions
- store aggregated match statistics
- use results for labeling/validation

### 5.3 Replay testing: the bridge between control plane and data plane
Before shipping a rule change, a high-leverage workflow is **log replay**:
- sample production events (privacy-safe)
- run old rules and new rules side-by-side
- diff match rates by rule and by segment

This surfaces the two most common failures early:
- the rule matches far more than intended (blasts recall)
- the rule matches far less than intended (misses critical cases)

### 5.4 Shadow mode and canaries (the safe rollout pattern)
For streaming/serving use cases:
- run the new rule set in shadow mode (compute matches, don’t act)
- compare metrics (match rate, latency)
- only then enable actions (block/tag) behind a canary rollout

This is the same safe deployment discipline as agent rollouts:
shadow → canary → ramp → rollback.

---

## 6. Implementation (Core Logic in Python)

Here’s an example of wildcard matching used inside a data pipeline (conceptual):

``python
def wildcard_match(s: str, p: str) -> bool:
 # Same DP idea as the DSA wildcard problem; simplified for illustration.
 m, n = len(s), len(p)
 dp = [False] * (n + 1)
 dp[0] = True
 for j in range(1, n + 1):
 dp[j] = dp[j - 1] if p[j - 1] == "*" else False

 for i in range(1, m + 1):
 prev_diag = dp[0]
 dp[0] = False
 for j in range(1, n + 1):
 tmp = dp[j]
 if p[j - 1] == "*":
 dp[j] = dp[j] or dp[j - 1]
 else:
 dp[j] = prev_diag and (p[j - 1] == "?" or p[j - 1] == s[i - 1])
 prev_diag = tmp
 return dp[n]
``

In production, you’d compile wildcard patterns once and run them many times, rather than DP per event.

### 6.1 What “compile once” means (in a pipeline)
A practical implementation usually looks like:
- load rule bundles (a versioned artifact) at service startup
- compile patterns into runtime objects (RE2 compiled regex, preprocessed wildcard tokens, tries)
- store compiled objects in memory with fast dispatch

The key is that “matching” should be a tight loop over bytes/characters, not a loop that parses patterns repeatedly.

### 6.2 Rule bundles and reason codes (minimal data model)
You want a stable rule schema so downstream systems can depend on it:
- `rule_id` (stable)
- `version` (immutable)
- `pattern_type` (wildcard/regex/dict/template)
- `pattern` or `payload`
- `action` (tag/block/extract)
- `owner` and `severity`

Even a simple JSON schema here pays off because it enables:
- validation at publish time
- reproducibility (“what rules were active when this model trained?”)
- auditability for compliance filters

---

## 7. Scaling Strategies

- Precompile patterns
- Group rules by “first token” or prefix to reduce candidates
- Use automata-based engines for repeated matching
- Cache matches for frequent strings (hot keys)

### 7.1 Candidate pruning is the first optimization
If you evaluate 10k rules against every event, you’ve already lost.
The main scaling trick is reducing the candidate set:
- route rules by event type or schema (only evaluate relevant rules)
- route by prefix/domain (URL host, path prefix)
- route by first token or anchor (if pattern has an anchored prefix)

### 7.2 Use the right algorithm for the right pattern family
Some patterns have specialized fast engines:
- multi-substring matching → trie/Aho-Corasick
- dictionary membership → hash sets with normalization
- URL templates → parsers with typed fields (avoid regex where possible)
- regex → RE2 or compiled automata (avoid backtracking)

### 7.3 Vectorization and batch evaluation (especially in Spark)
For batch pipelines, you get speed from:
- columnar processing (evaluate on columns, not per-row Python loops)
- pushing matching into native code (JVM/Scala/UDF avoidance)
- precomputing derived fields (e.g., URL host/path) once, then matching on those

Anti-pattern:
- running Python regex in a per-row UDF across TB/day.

### 7.4 Caching: hot keys and repeated patterns
Many real streams have heavy hitters:
- same domains/paths
- same user agents
- same query templates

You can cache:
- match results for normalized strings
- partial parsing (URL decomposition)

But cache carefully:
- cap memory
- include rule_bundle_version in the cache key
- watch for adversarial hot keys (attackers can force cache churn)

### 7.5 Multi-tenant isolation
If multiple teams share the matcher:
- enforce quotas (max rules per tenant, max CPU time per tenant)
- enforce approval flows for “expensive” patterns
- isolate indexes/dictionaries per tenant where privacy requires it

This is where “system design” becomes real: you’re not optimizing one pipeline; you’re operating a shared platform.

### 7.6 Runtime engineering: where does the evaluator live?
At scale, your evaluator usually ends up in one of:
- a JVM service (easy integration with Kafka/Flink/Spark, good operational tooling)
- a C++/Rust library embedded in services (fast and safe, but higher integration cost)
- a native engine exposed as a service (shared infra, but network and rollout complexity)

The key is to avoid “slow path” surprises:
- a Python UDF is almost always the wrong place for regex at TB/day
- repeated parsing of patterns at runtime is a silent cost multiplier

### 7.7 “Safe by construction” engines beat after-the-fact timeouts
Timeouts help, but the strongest safety posture is to choose engines with predictable performance:
- RE2-style regex avoids catastrophic backtracking by design
- automata/trie-based matchers have bounded behavior on typical workloads

If you must support “expressive” patterns:
- restrict the allowed syntax
- force compilation in the control plane
- require replay testing before rollout

---

## 8. Monitoring & Metrics

- match rate per rule
- rule latency
- rule churn (how often rules change)
- false positive / false negative feedback loop (human labels)

### 8.1 The “three dashboards” that prevent most incidents
If you build only three dashboards, make them:

1) **Match volume over time**
- total matches
- matches by top rules
- matches by segment (region/app version)

2) **Latency and budget health**
- p50/p95/p99 evaluation latency
- budget overruns (timeouts, fallbacks)
- CPU per event (or per partition)

3) **Change correlation**
- overlay deployments (rule bundle version changes)
- show diffs in match rates after change events

### 8.2 Feedback loops: turning “wrong rules” into improvements
Rules inevitably have false positives/negatives.
If you want the system to improve, design a feedback loop:
- allow downstream teams to label matches as correct/incorrect
- store those judgments with rule_id + version
- use them to drive:
 - rule updates (narrow/broaden)
 - training data for learned verifiers
 - regression tests (golden cases)

This is why the system needs stable IDs and versioning: feedback is meaningless if rules can’t be referenced precisely.

---

## 9. Failure Modes

- overly broad rules (match everything)
- overly strict rules (miss matches)
- conflicting rules (two labels)
- regex ReDoS and performance regressions

Mitigation:
- test suites, canary rollouts, ownership

### 9.1 Failure mode: semantic drift (rules encode yesterday’s world)
Even if a rule is “correct”, it can become wrong when the world changes:
- new URL formats
- new product surfaces
- new locales/languages

Mitigation:
- monitor match rates by segment
- require periodic reviews for high-impact rules
- keep “unknown/other” buckets instead of forcing hard blocks

### 9.2 Failure mode: rule conflicts and inconsistent actions
Conflicts happen when:
- two rules assign different labels to the same event
- one rule blocks, another tags, and downstream teams disagree

Mitigation:
- define precedence (severity tiers)
- define deterministic conflict resolution (most specific wins, or highest severity wins)
- make conflicts visible (metrics: conflict_rate)

### 9.3 Failure mode: regressions from “innocent” performance optimizations
Common example:
- caching introduced without including `rule_bundle_version` in the key
- after a rule update, cache returns stale match results

Mitigation:
- versioned caches
- shadow-mode comparisons after optimizations
- canary deployments for evaluator code changes (not only rule changes)

### 9.4 Runbook: what to do when match rate spikes
1. Identify which rules changed (or which segments spiked).
2. Compare old vs new bundle in replay testing.
3. If unclear, roll back bundle (fast mitigation).
4. Add a regression test for the incident pattern.

This is “stop the line” for data pipelines: rule systems are safety-critical infrastructure.

---

## 10. Case Study

Weak supervision pipelines often use pattern matching:
- label clicks as “spam” if URL matches suspicious patterns
- label product categories if title matches templates

Pattern matching is the “fast prior” that bootstraps ML.

### 10.1 Case study: PII detection as a pattern system
Many ML orgs build PII detectors that run:
- on training data (to avoid ingesting sensitive data)
- on logs (to avoid storing sensitive payloads)
- on model outputs (to avoid leaking)

Pattern matching is often the first layer:
- email/phone patterns
- credit card patterns (with checksum validation)
- IDs and secret formats

Production nuance:
- you don’t want to log raw matches (privacy risk)
- you need deterministic reason codes for audits
- you need “safe execution” so adversarial inputs can’t DoS the filter

This ties directly to the speech track’s privacy constraints: observability must not become a data leak path.

### 10.2 Case study: weak labels for classification (labeling functions)
A common weak supervision design:
- author labeling functions as pattern rules (regex/template/dictionary)
- combine them with a label model or heuristic precedence rules
- train a downstream model on the weak labels

The biggest failure mode is silent label drift:
- a rule change shifts class balance
- your model “improves” offline but degrades online

That’s why rule bundle versioning should be stored alongside training datasets:
> a model is not reproducible without the exact rule version that produced its labels.

---

## 11. Cost Analysis

Costs:
- CPU for evaluation
- governance overhead for rule review

Savings:
- fewer incidents due to bad data
- faster iteration in labeling and filters

### 11.1 Cost drivers (what actually gets expensive)
At high scale, costs come from:
- per-event parsing (e.g., URL parsing in a tight loop)
- expensive regex (or too many regexes)
- high cardinality in outputs (downstream storage/metrics costs)
- index rebuilds / replays (control plane compute)

One good mental model:
> treat “rules evaluated per event” like “features computed per request” in serving — it is your cost multiplier.

### 11.2 Cost levers (how teams keep this affordable)
- prune candidates aggressively (route rules to relevant events)
- prefer structured templates and dictionaries over complex regex
- compile once, run in native engines (RE2, automata)
- batch evaluation in columnar engines for offline pipelines

### 11.3 The ROI argument (why orgs build a platform)
Teams start with scattered scripts and notebook checks.
At org scale, a platform wins because it provides:
- consistent semantics
- consistent governance and auditability
- consistent observability and RCA

That consistency reduces repeat incidents — and incident avoidance is usually the biggest cost saver.

---

## 12. Key Takeaways

1. Pattern matching is a core ML systems primitive.
2. Safe execution and governance matter as much as correctness.
3. Compile and observe rules like production code.

### 12.1 Connections across tracks (the shared theme)
The shared theme is **pattern matching and state machines**:
- “Wildcard Matching” demonstrates matching as DP over states; production matchers are state machines with budgets.
- “Acoustic Pattern Matching” uses the same coarse→fine design: retrieve candidates, then verify with a more expensive alignment scorer.
- “Scaling Multi-Agent Systems” reinforces the control-plane lesson: budgets, rollouts, observability, and rollback are what make complex systems operable.

If you can explain your matching system as:
> semantics + compilation + budgets + observability + safe rollout,
you’re thinking like an ML systems engineer (not just “someone who knows regex”).

### 12.2 Interview framing (how to answer under time pressure)
If you’re asked to “design pattern matching for ML pipelines”, a crisp structure is:

1. **Start with semantics**
 - what patterns exist (wildcard/regex/dict/template)?
 - full match vs substring match?
 - what outputs are needed (bool vs spans vs captures)?

2. **Split control plane vs data plane**
 - authoring + approvals + compilation + versioning
 - fast evaluator for streaming/batch

3. **Make safety explicit**
 - RE2 / bounded engines
 - budgets + canaries + rollback

4. **Make observability explicit**
 - rule IDs, match rates, segment dashboards, replay diffs

This stands out because it’s operational and practical: it sounds like someone who has paged on a rule regression before.

---

**Originally published at:** [arunbaby.com/ml-system-design/0054-pattern-matching-in-ml](https://www.arunbaby.com/ml-system-design/0054-pattern-matching-in-ml/)

*If you found this helpful, consider sharing it with others who might benefit.*

