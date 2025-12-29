---
title: "Data Validation"
day: 53
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - data-validation
  - data-quality
  - mlops
  - schema
  - monitoring
  - drift
difficulty: Hard
subdomain: "Data Quality & Governance"
tech_stack: Python, Great Expectations, TFDV, Spark, Kafka
scale: "100TB/day, multi-tenant, low-latency gates"
companies: Google, Meta, Netflix, Uber
related_dsa_day: 53
related_speech_day: 53
related_agents_day: 53
---

**"Most ML failures aren’t model bugs—they’re invalid data quietly passing through."**

## 1. Problem Statement

Production ML systems consume data from many upstream sources:
- event streams (clicks, purchases, searches)
- batch ETL tables (user profiles, catalogs)
- logs and metrics (latency, errors)
- third-party feeds

Data changes constantly:
- schemas evolve
- units change (ms → s)
- new categories appear
- null rates drift
- pipelines partially fail

**Data validation** is the system that ensures:
- data conforms to expectations (schema, ranges, types)
- anomalies are detected early (before training/serving)
- quality regressions are gated (stop the line)
- teams can debug quickly with clear evidence

The shared theme today is **data validation and edge case handling**:
like the “First Missing Positive” algorithm, you define the valid domain and ignore/flag everything outside it.

---

## 2. Understanding the Requirements

### 2.1 Functional requirements

- Validate **schema**: required columns, types, nested structures.
- Validate **constraints**: ranges, enums, monotonicity, uniqueness.
- Validate **distribution**: drift, skew, outliers, rare category explosions.
- Validate **freshness/completeness**: partitions present, record counts, latency.
- Produce **actions**: pass, warn, quarantine, block, rollback.
- Produce **artifacts**: validation reports, dashboards, alerts, lineage links.

### 2.2 Non-functional requirements

- **Scale**: TB/day, high-cardinality features, many tenants.
- **Low latency gates**: must run before training jobs / feature publishing.
- **Reliability**: validators must not become the bottleneck.
- **Explainability**: “what failed and why” must be human-friendly.
- **Governance**: who owns expectations? who can change them? audit trails.
- **Safety**: avoid both false negatives (bad data slips) and false positives (blocks everything).

---

## 3. High-Level Architecture

```
            +------------------------+
            | Upstream Sources       |
            | (streams + batch)      |
            +-----------+------------+
                        |
                        v
              +-------------------+
              | Ingestion Layer   |
              | (Kafka/S3/HDFS)   |
              +---------+---------+
                        |
           +------------+-------------+
           |                          |
           v                          v
 +-------------------+       +-------------------+
 | Validation (Fast) |       | Validation (Deep) |
 | schema + counts   |       | dist + drift      |
 +---------+---------+       +---------+---------+
           |                          |
           +------------+-------------+
                        |
                        v
              +-------------------+
              | Policy Engine     |
              | (pass/warn/block) |
              +---------+---------+
                        |
           +------------+-------------+
           |                          |
           v                          v
 +-------------------+       +-------------------+
 | Downstream:       |       | Quarantine Store  |
 | feature store,    |       | + RCA reports     |
 | training, serving |       +-------------------+
 +-------------------+
```

Key idea:
- **Fast checks** run on every partition quickly (schema, counts, null rates).
- **Deep checks** run on sampled or aggregated data (distribution drift, correlations).
- A **policy engine** decides what to do based on severity and ownership.

---

## 4. Component Deep-Dives

### 4.1 Expectations: what do we validate?

Expectations come in tiers:

**Tier 0: schema and contracts**
- required columns exist
- types are correct (int/float/string)
- nested schema matches (JSON/Avro/Proto)

**Tier 1: basic quality**
- null rates under thresholds
- ranges sane (age in [0, 120])
- enum values known (country in ISO list)

**Tier 2: distribution and drift**
- feature distributions stable (histogram distance)
- category proportions stable
- embedding stats stable (mean/variance)

**Tier 3: semantic / business constraints**
- revenue >= 0
- timestamps monotonic per entity
- joins don’t explode (fanout constraints)

The key lesson:
> You can’t validate everything deeply at all times; choose expectations that catch real failures.

### 4.2 Where validation runs (batch vs streaming)

- **Batch validation**
  - validates partitions (daily tables)
  - good for training data and offline features
- **Streaming validation**
  - validates events in near real time (schema, rate, missing fields)
  - good for online features and real-time inference pipelines

Most orgs end up with both.

### 4.2.1 Validation in the feature store (the “last mile”)

Even if batch pipelines are perfect, the **feature store** is where validation failures become user-facing.
Common feature-store validations:
- **freshness**: feature timestamps within TTL
- **availability**: fraction of requests with missing features
- **default rate**: how often a feature is defaulted due to missing/invalid values
- **range guards**: clamp or null-out impossible values

A strong pattern:
> Treat validation as a continuous contract between producers and consumers.

If a producer changes a feature distribution, the feature store should detect it and route it as an incident or a planned evolution (via schema/expectation update).

### 4.2.2 Validation during model serving (online “data contracts”)

Online model serving has different constraints than training:
- you can’t block traffic (you must respond)
- you must be robust to missing or corrupted features

So serving validation usually results in:
- **fallback behavior** (defaults, last-known-good features, simpler model)
- **logging and alerting**
- **traffic shaping** (route to safe model variant)

A useful mental model:
> Training validation prevents bad learning. Serving validation prevents bad decisions.

### 4.3 Policy engine: what happens when validation fails?

Possible actions:
- **pass**: everything ok
- **warn**: log + notify owner, allow pipeline
- **block**: prevent publish/train/deploy
- **quarantine**: store bad partition for RCA
- **rollback**: revert to last known good snapshot

This turns validation from “metrics” into “control”.

### 4.3.1 Quarantine design (how to store “bad data” safely)

When validation fails, teams often want to inspect the bad partition.
But you need to design quarantine carefully:
- quarantine should preserve raw inputs (for RCA) **and** derived summaries (for fast diagnosis)
- quarantine access may involve sensitive data (PII), so it needs RBAC and audit logs
- quarantine data should have TTLs to avoid indefinite retention

Common quarantine artifacts:
- a small sample of failing records (where policy allows)
- aggregated histograms and null maps
- top failing fields and rules
- the producing pipeline version and change log references

This is the “RCA-first” principle: every failure should come with enough evidence to fix it quickly.

### 4.4 Ownership and change management

Expectations must be owned.
Otherwise teams either:
- never update thresholds (false positives forever), or
- loosen everything to stop pages (false negatives forever)

Good patterns:
- expectation files in Git, PR-reviewed
- a “break glass” path with audit logs
- a change log correlation panel for failures

### 4.4.1 A practical ownership model (who owns expectations?)

Expectations should usually be owned by:
- **producer teams** for schema and basic constraints (Tier 0/1)
- **consumer teams** (ML teams) for distribution and semantic expectations (Tier 2/3)

Why split ownership:
- producers understand correctness of raw fields
- consumers understand what the model relies on

In practice, high-performing orgs establish:
- a shared “data contract” repo
- an on-call rotation for critical datasets
- a lightweight review process for expectation changes (avoid bottlenecks)

---

## 5. Data Flow (Training vs Serving)

### 5.1 Training data validation

Typical flow:
1. ETL writes a daily partition
2. validation runs (schema + counts + nulls)
3. deep drift checks compare against recent partitions
4. policy decides pass/warn/block
5. training starts only if gates pass

### 5.2 Serving data validation (online)

Online inference uses:
- streaming features
- feature store reads

Validations include:
- schema validation at ingestion (drop invalid events)
- online feature freshness (TTL checks)
- feature value range checks (clamp or default)

In serving, “blocking” is rarely acceptable. Instead you:
- fall back to defaults
- degrade gracefully
- alert for investigation

---

## 6. Implementation (Core Logic in Python)

Below are minimal examples of validation primitives. In production you’d use Great Expectations/TFDV, but the logic is universal.

### 6.0 A Great Expectations / TFDV mental mapping

If you’re familiar with common libraries:

- **Great Expectations**
  - “expectation suite” = a set of rules for a dataset
  - “checkpoint” = validation run that produces a report
  - good fit for: batch tables, SQL/Spark pipelines

- **TFDV (TensorFlow Data Validation)**
  - “schema” + “statistics” = baseline + constraints
  - drift/skew detection = compare stats between datasets/slices
  - good fit for: TF pipelines and feature-heavy datasets

Even if you don’t use these libraries, the primitives are the same:
- schema checks
- range checks
- distribution checks
- policy decisions

### 6.1 Schema validation (dict-based)

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ValidationError:
    field: str
    message: str


def validate_schema(record: Dict[str, Any], required: Dict[str, type]) -> List[ValidationError]:
    errors: List[ValidationError] = []
    for field, t in required.items():
        if field not in record:
            errors.append(ValidationError(field, "missing"))
            continue
        if record[field] is None:
            errors.append(ValidationError(field, "null"))
            continue
        if not isinstance(record[field], t):
            errors.append(ValidationError(field, f"type_mismatch: expected {t.__name__}"))
    return errors
```

### 6.2 Range and enum checks

```python
def validate_range(x: float, lo: float, hi: float) -> bool:
    return lo <= x <= hi


def validate_enum(x: str, allowed: set[str]) -> bool:
    return x in allowed
```

### 6.3 Drift check on histograms (L1 distance)

```python
import numpy as np


def l1_hist_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    h1 = h1 / (np.sum(h1) + 1e-9)
    h2 = h2 / (np.sum(h2) + 1e-9)
    return float(np.sum(np.abs(h1 - h2)))
```

This is a lightweight baseline:
- cheap
- interpretable
- works well for many numeric features when you bucket them

### 6.4 Data completeness and freshness checks (table-level)

Most catastrophic failures are “missing data” masquerading as “valid zeros”.
Two simple but high-signal checks:
- expected record count bounds
- partition freshness bounds

In practice:
- training data: block if partition missing or count drops sharply
- serving: alert and fall back to last-known-good

### 6.5 A minimal policy engine (pass/warn/block)

```python
from dataclasses import dataclass
from typing import List


@dataclass
class PolicyDecision:
    status: str  # "pass" | "warn" | "block"
    reasons: List[str]


def decide_policy(errors: List[str], warn_only: set[str]) -> PolicyDecision:
    """
    Very simple policy:
    - if any error is not in warn_only -> block
    - else if any errors exist -> warn
    - else pass
    """
    if not errors:
        return PolicyDecision("pass", [])

    hard = [e for e in errors if e not in warn_only]
    if hard:
        return PolicyDecision("block", hard)
    return PolicyDecision("warn", errors)
```

In real systems, policies are per-dataset and per-severity tier.
But this skeleton captures the core idea: validation results must be translated into actions.

---

## 7. Scaling Strategies

### 7.1 Sampling vs full scans

Deep validation on 100TB/day can’t scan everything.
Common approach:
- run Tier 0/1 validations on full partitions (cheap aggregates)
- run Tier 2/3 on samples or sketches

### 7.2 Sketches for distributions

Use streaming-friendly summaries:
- Count-Min Sketch for frequencies
- HyperLogLog for cardinality
- t-digest / KLL for quantiles

This enables:
- drift detection without storing raw data
- low-latency dashboards

### 7.3 Cardinality governance

High-cardinality features can DOS your validation and storage.
Guardrails:
- label allowlists/denylists
- bucketing strategies (device model buckets)
- top-k heavy hitter tracking

---

## 8. Monitoring & Metrics

Monitor validators like production services:
- validation latency and throughput
- percent partitions passing/warning/blocking
- top failing expectations
- “expectation churn” (how often thresholds change)
- quarantine volume

Also monitor downstream impact:
- training job failure rates due to gates
- serving fallback rates

### 8.1 RCA-first reporting (what a good failure report contains)

A validation failure should come with a “triage packet”:
- **What failed**: rule name, field, threshold
- **How bad**: current value vs baseline, severity tier
- **Where**: top segments affected (if applicable)
- **When**: start time, detection time, duration
- **Why likely**: correlated change events (deploys, schema changes)
- **Next**: owner, runbook, rollback instructions, quarantine link

This is the difference between:
- “we saw drift” (noise)
- and “we know exactly what to fix” (actionable)

### 8.2 Distribution drift at scale: histograms vs sketches vs embeddings

Drift detection can be done with different representations:

- **Histograms**
  - easy to interpret
  - good for numeric features with stable ranges
  - costs grow with number of bins × number of features

- **Sketches**
  - compact
  - streaming-friendly
  - great for cardinality and heavy hitters
  - less interpretable than histograms but still useful

- **Embedding statistics**
  - for high-dimensional features (text embeddings, image embeddings)
  - track mean/variance, PCA projections, or cluster assignments
  - useful for catching “silent changes” upstream (new encoder version)

Practical platform approach:
- use cheap, interpretable drift checks for “core” features
- use embedding stats and sketches for high-dimensional/high-cardinality features

### 8.3 When validation should page (avoid alert fatigue)

Validation alerts can easily become noisy. A robust policy:
- **page** only when user impact is likely (serving) or training integrity is at risk (training gates)
- **notify** for moderate drift or new category growth
- **log-only** for exploratory detectors

This aligns with reality: teams have limited attention, and “too many alerts” is a failure mode of the system.

---

## 9. Failure Modes (and Mitigations)

### 9.1 Overly strict checks (false positives)
Mitigation:
- severity tiers
- gradual rollout of new expectations (shadow mode)
- auto-suggest threshold updates with human review

### 9.2 Too loose checks (false negatives)
Mitigation:
- incident postmortems feed new expectations
- add drift checks for silent failures

### 9.3 Unit changes and schema drift
Mitigation:
- schema registry
- explicit unit metadata
- canary validations on new producer versions

### 9.4 PII and privacy failures (validation can leak!)

Validation systems often process raw records, which may contain PII.
A common “oops” is that validation logs accidentally store:
- raw emails/phone numbers
- full JSON payloads
- user IDs and sensitive attributes

Mitigations:
- treat validation reports as sensitive artifacts
- redact or hash PII in logs
- store only aggregated statistics by default
- gate sample extraction behind explicit access controls and audit logs

This is particularly important if your validation system is used across many teams (multi-tenant).

### 9.5 The “validator outage” problem

If validators are required gates, they become critical infrastructure.
Failure modes:
- validation service down blocks training
- schema registry outage blocks ingestion
- state store lag causes false failures

Mitigations:
- redundancy and caching (serve last-known schema)
- graceful degradation (warn-only mode temporarily)
- explicit “break glass” workflows with audit logs

In other words: the data validator must be operated like an SRE-owned service.

---

## 10. Real-World Case Study

A classic incident:
- upstream changes `duration_ms` to `duration_s`
- models train fine but behave incorrectly

Validation catches this via:
- range checks (duration now too small)
- distribution shift (histogram moves)
- change-log correlation (producer version bump)

The value is not the detection alone; it’s the fast RCA path.

### 10.1 Another case study: silent label leakage and “too good to be true”

A different class of data validation failure is **leakage**:
- a feature accidentally contains future information
- a join accidentally includes the label itself
- an ID leaks the target (e.g., “refund_status” used to predict refunds)

Symptoms:
- offline metrics jump dramatically
- training loss collapses unexpectedly fast
- online performance does not improve (or gets worse)

Validation signals to add:
- correlation checks between features and labels (suspiciously high correlation)
- “future timestamp” checks (feature timestamps after label timestamps)
- join key audits (prevent joining on future tables)

This is why validation is not only about “schema correctness”; it’s also about **semantic correctness**.

### 10.2 A case study: category explosion (cardinality drift)

Example:
- upstream starts logging raw URLs as a feature
- cardinality explodes from 1k to 10M

Impact:
- feature store storage blows up
- model quality degrades (rare categories, sparse embeddings)
- validators and dashboards become slow

Validation mitigations:
- enforce cardinality budgets
- bucket rare categories into “OTHER”
- allowlist canonical categories (domain restriction)

This is the same domain restriction pattern as the DSA problem: decide what values are “valid” for downstream use.

---

## 11. Cost Analysis

Cost drivers:
- compute for deep validations
- storage for quarantine and reports
- operational overhead (ownership, governance)

Cost levers:
- do cheap checks everywhere, deep checks selectively
- use sketches and aggregates
- reuse shared platform components

### 11.1 Why data validation becomes a platform (org-scale evolution)

Teams often start with:
- ad-hoc SQL queries (“null rate looks high”)
- notebook checks
- one-off scripts

At org scale, this fails because:
- every team re-implements the same checks differently
- thresholds are undocumented and drift over time
- incidents repeat because postmortems don’t translate into enforceable gates
- no one owns cross-team datasets

So validation becomes a platform:
- shared schema registry and contracts
- shared expectation suites and policy engine
- shared RCA UI and quarantine flows
- shared governance (ownership, auditability, change logs)

This is the same scaling story as anomaly detection: you build a platform because consistency and operability matter more than clever algorithms.

### 11.2 Training vs serving cost trade-offs

Training validation:
- can afford heavier scans (batch, offline)
- benefits from deep distribution checks (prevent poisoning)

Serving validation:
- must be low latency and non-blocking
- focuses on cheap guards + safe fallback behavior

If you treat both as “the same pipeline”, you’ll either:
- slow down serving (bad)
- or under-validate training (also bad)

### 11.3 What to do when you can’t block (graceful degradation)

In many real-time systems, you can’t block data entirely.
Patterns:
- default missing features
- clamp out-of-range features
- route to a “safe” model variant that uses fewer features
- route requests to “shadow” logging to capture evidence without impacting users

This is effectively “error handling for data”.
It’s an underappreciated part of ML system reliability.

---

## 12. Key Takeaways

1. **Define the valid domain**: schema + ranges + constraints.
2. **Use policy gates**: validation without action is observability, not safety.
3. **Make it operable**: ownership, change logs, RCA-first reports.

### 12.1 Appendix: a minimal “expectations checklist” for new datasets

When onboarding a dataset, start with:

- **Schema**
  - required fields present
  - types correct
  - nullability documented

- **Completeness**
  - expected partition cadence (hourly/daily)
  - record count bounds
  - freshness/latency bounds

- **Range and enums**
  - numeric ranges (guard impossible values)
  - known enums (country/device/app_version) with a safe “unknown” bucket

- **Distribution**
  - basic histograms for critical features
  - category cardinality limits (avoid explosion)

- **Policy**
  - warn vs block definitions
  - ownership routing (who gets paged)
  - links to runbooks and change logs

The point is not perfection; it’s catching high-impact failures early.

### 12.2 Appendix: how Day 53 connects across tracks

The same pattern appears in all four tracks:
- DSA: restrict to `[1..n]` and detect the missing element
- ML: restrict to valid schema/range domains and detect missing/invalid partitions
- Speech: restrict to valid audio formats and detect corrupt audio
- Agents: restrict to valid tool schemas/policies and prevent unsafe actions

It’s the same engineering move: define the domain, encode invariants, and treat edge cases explicitly.

### 12.3 Appendix: a minimal “triage packet” template for validators

If you build a validation UI or Slack alert, include:
- dataset name + partition
- failing rule + expected vs observed
- severity + policy decision (warn/block)
- top affected segments (if applicable)
- linked change log events (producer version, schema changes)
- owner and runbook link

If you provide this consistently, you will see two outcomes:
- faster MTTR
- fewer “mystery regressions” blamed on the model

### 12.4 Appendix: quick interview framing

If asked “how would you design data validation for ML?”:
- Start with Tier 0/1 checks everywhere (schema, counts, nulls, ranges).
- Add Tier 2 drift checks for critical features (histograms/sketches).
- Add a policy engine (warn/block) and quarantine for RCA.
- Make it operable: ownership, audit logs, rollout/shadow mode for new rules.

This is a strong answer because it emphasizes operability and governance, not just metrics.

### 12.5 Appendix: “stop the line” vs “degrade gracefully” (a crisp rule)

A simple rule that works well in practice:

- **Stop the line** (block) when:
  - training data integrity is compromised (missing partition, unit mismatch, label leakage)
  - publishing features would poison many downstream models

- **Degrade gracefully** (fallback) when:
  - serving needs to respond (online inference)
  - you can default/clamp safely and alert the owner

This prevents the most common confusion:
teams either block everything (causing outages) or allow everything (causing silent correctness failures).

### 12.6 Appendix: dataset lineage and “blast radius” estimation

When a dataset fails validation, the first question is:
> “Who is impacted?”

That requires lineage:
- which downstream tables/features are derived from this dataset?
- which models consume those features?
- which products and teams are behind those models?

Lineage enables:
- targeted paging (don’t page everyone)
- smarter policies (block only the impacted publish path)
- faster mitigation (roll back a single feature, not the whole pipeline)

Practical implementation:
- maintain a DAG of datasets → features → models
- attach ownership metadata to each node
- include lineage in every validation alert (“blast radius: Models A,B; Products X,Y”)

This is also where validation connects to incident management:
lineage is what turns “data is bad” into “here is what to do now”.

### 12.7 Appendix: validation maturity model

A useful way to think about progress:

- **Level 0: ad-hoc**
  - manual SQL checks, notebooks
  - failures discovered after model regressions

- **Level 1: schema checks**
  - schema registry and required fields
  - basic null/range checks

- **Level 2: gated publishing**
  - warn/block policies
  - quarantine and RCA packets

- **Level 3: drift and segment checks**
  - histograms, sketches, segment baselines
  - change-log correlation

- **Level 4: closed-loop reliability**
  - automated mitigations (rollback, safe fallback models)
  - continuous evaluation of validation rules (shadow mode for new rules)

Most real orgs are between levels 1 and 3. Level 4 is where validation becomes a competitive advantage.

### 12.8 Appendix: join validation (the silent correctness killer)

Many ML features come from joins:
- user table JOIN events table
- item catalog JOIN impressions
- labels JOIN feature snapshots

Joins fail silently in two ways:

1. **Fanout explosions**
   - a one-to-one join becomes one-to-many
   - record counts inflate
   - models learn duplicated labels or skewed weights

2. **Join drop / sparsity**
   - key mismatch causes many null joins
   - features default unexpectedly
   - online/offline skew increases

Validation checks for joins:
- expected row count ratio bounds (post-join / pre-join)
- null join rate bounds (percent missing after join)
- key uniqueness checks (primary keys truly unique)
- “top join keys” diagnostics (are a few keys causing fanout?)

This is high leverage because many ML regressions ultimately trace back to “the join changed”.

### 12.9 Appendix: skew vs drift (offline vs online mismatch)

Teams often mix these concepts:
- **drift**: distribution changes over time (today vs last week)
- **skew**: distribution differs between training and serving (offline vs online)

You can have:
- low drift but high skew (offline pipeline differs from online)
- high drift but low skew (both pipelines drift together)

Validation should handle both:
- drift checks: time-based comparisons
- skew checks: offline features vs online features on the same entities

Skew is especially important for:
- feature stores
- real-time personalization
- feedback-loop systems

### 12.10 Appendix: a concrete expectation suite example

For a dataset `user_events_daily`, a practical expectation suite might include:

- **Schema**
  - required: `user_id`, `event_time`, `event_type`, `device`, `country`
  - types: `user_id` string, `event_time` timestamp

- **Completeness**
  - record_count within [0.8x, 1.2x] of 7-day median
  - partition latency < 2 hours

- **Null/range**
  - `country` null rate < 0.1%
  - `event_time` within partition date +/- 24h
  - `age` in [0, 120] if present

- **Enums**
  - `event_type` in allowlist (with unknown bucket)
  - `device` in known taxonomy buckets

- **Distribution**
  - `event_type` proportion drift L1 distance < threshold
  - heavy hitter checks on top event types and countries

- **Join contracts**
  - join with `user_profile` yields < 1% missing profiles

The key is not to validate everything; validate what prevents expensive incidents.

### 12.11 Appendix: streaming validation patterns (fast gates)

For Kafka-like streams, validation is usually:
- schema validation at ingestion (drop/route invalid events)
- rate checks (sudden drop = pipeline outage)
- null-field checks for critical fields
- sampling-based distribution checks (cheap histograms)

Streaming validation should be designed to:
- be non-blocking (don’t stall producers)
- route bad events to a dead-letter queue
- emit counters and traces for RCA

This is exactly how mature infra treats “invalid messages” in distributed systems.

---

**Originally published at:** [arunbaby.com/ml-system-design/0053-data-validation](https://www.arunbaby.com/ml-system-design/0053-data-validation/)

*If you found this helpful, consider sharing it with others who might benefit.*

