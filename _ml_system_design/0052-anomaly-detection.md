---
title: "Anomaly Detection"
day: 52
related_dsa_day: 52
related_speech_day: 52
related_agents_day: 52
collection: ml_system_design
categories:
 - ml-system-design
tags:
 - anomaly-detection
 - monitoring
 - time-series
 - streaming
 - alerting
 - mlops
difficulty: Hard
subdomain: "Reliability & Monitoring"
tech_stack: Python, Kafka, Flink, Prometheus, PyOD
scale: "1M events/sec, multi-tenant, low false positives"
companies: Netflix, Uber, Google, Amazon
---

**"Anomaly detection is trapping rain water for metrics: find the boundaries of ‘normal’ and measure what overflows."**

## 1. Problem Statement

In production ML and data systems, anomalies are the early warning signals of:
- data quality regressions
- pipeline failures
- model performance drift
- fraud and abuse
- infrastructure incidents

The goal is to design an **anomaly detection system** that:
- ingests high-volume metrics/events
- detects meaningful anomalies quickly
- keeps false positives low (alert fatigue is real)
- supports investigation and root cause analysis
- is robust to seasonality and shifting baselines

Today’s shared theme is **pattern recognition**:
in DSA, we detect the “boundary pattern” that determines trapped water.
In anomaly detection, we detect deviations from the learned “normal pattern”.

---

## 2. Understanding the Requirements

### 2.1 Functional requirements

- **Inputs**
 - time-series metrics (QPS, latency, error rates)
 - categorical counts (events by user type, country)
 - distributions (histograms, sketches)
 - model signals (loss, accuracy proxies, embedding stats)
- **Outputs**
 - anomaly alerts with severity
 - explanation context (top contributing dimensions)
 - links to dashboards/runbooks
 - ability to suppress/acknowledge

### 2.2 Non-functional requirements

- **Low latency**: detect within minutes (or seconds for fraud/infra).
- **Scalability**: multi-tenant at 1M events/sec.
- **Precision**: too many alerts kill trust.
- **Recall**: missing important anomalies is costly.
- **Robustness**: handle missing data, late arrivals, backfills.
- **Interpretability**: support “why did this alert fire?”.
- **Governance**: alert ownership, runbooks, on-call routing.

---

## 3. High-Level Architecture

### 3.1 Diagram

``
 Producers (services / models / pipelines)
 |
 v
 +-------------+ +-------------------+
 | Kafka/PubSub| -----> | Stream Processor |
 +-------------+ | (Flink/Spark) |
 | +---------+---------+
 | |
 | v
 | +-------------------+
 | | Feature Builder |
 | | (windows, joins) |
 | +---------+---------+
 | |
 v v
 +-------------------+ +-------------------+
 | TS Store | | Detector Service |
 | (Prom/TSDB) | | rules + ML |
 +---------+---------+ +---------+---------+
 | |
 v v
 +-------------------+ +-------------------+
 | Dashboards | | Alert Router |
 | (Grafana) | | (Pager/Slack) |
 +-------------------+ +---------+---------+
 |
 v
 +-------------------+
 | Case Mgmt / RCA |
 | (tickets, notes) |
 +-------------------+
``

### 3.2 Core idea

You generally run a **two-layer detector**:
- **fast rules** for clear anomalies (error rate > X, no events for Y minutes)
- **statistical/ML detectors** for subtle drift (seasonality-aware, multivariate)

Rules are cheap and interpretable.
ML handles complex patterns, but must be deployed carefully.

---

## 4. Component Deep-Dives

### 4.1 Data ingestion and schema discipline

The most common failure is not “bad model”, it’s bad data:
- missing metrics due to instrumentation bugs
- unit changes (ms → s)
- counter resets
- duplicate events

So your system needs:
- schema registry (event contracts)
- validation (ranges, types, monotonicity for counters)
- backfill handling (late arrivals)

### 4.2 Feature engineering for anomaly detection

Features depend on signal type:

**Time series**
- moving averages, EWMA
- seasonal decomposition (daily/weekly)
- derivative and acceleration (change rates)

**Categorical slices**
- per-dimension rates (country, device)
- top-k contributors (heavy hitters)
- ratio metrics (errors / requests)

**Distributions**
- quantiles (p50/p95/p99)
- histogram distance metrics (KL divergence, Wasserstein)

This is the “pattern recognition” layer: translate raw signals into something that makes deviations obvious.

### 4.3 Detectors: rules, statistics, ML

Common detector families:

- **Rules**
 - simple thresholds, rate-of-change, “no data” checks
 - best for: clear SLO violations, missing pipelines

- **Statistical**
 - z-score with robust estimators (median/MAD)
 - EWMA control charts
 - seasonal baselines (Holt-Winters)
 - best for: stable metrics with known periodicity

- **ML / Unsupervised**
 - Isolation Forest, One-Class SVM
 - autoencoders
 - clustering distance to centroids
 - best for: high-dimensional signals, unknown patterns

The important production truth:
> You don’t “pick one model”. You build a system that can run multiple detectors and fuse them.

### 4.4 Alerting and routing

Anomaly detection is useless without action.
You need:
- deduplication (same alert grouped)
- suppression (maintenance windows)
- routing by ownership (service/team)
- escalation policies and runbooks

### 4.5 Explanation (why did this fire?)

For slice-based anomalies, a powerful technique is “top contributors”:
- overall error rate rose
- 80% of delta is from `country=IN` and `device=android_low_end`

This reduces MTTR dramatically.

### 4.6 Thresholding is a product decision (not just statistics)

A detector outputs a score. You still need to decide:
- when do we page a human?
- when do we log-only?
- when do we auto-mitigate?

A practical severity ladder:
- **S0 (log-only)**: weak signal, used for trend analysis
- **S1 (notify)**: Slack alert, no pager
- **S2 (page)**: likely user impact or SLO breach
- **S3 (auto-mitigate + page)**: safety-critical systems

How to set thresholds:
- start with conservative paging thresholds (avoid alert fatigue)
- use historical replay to estimate expected alert volume
- adjust per metric and per service criticality

The uncomfortable truth:
> “Correct” thresholds depend on team capacity, business impact, and tolerance for noise.

### 4.7 Root cause analysis (RCA) requires drill-down, not just detection

Detection answers: “something is wrong”.
RCA answers: “what is wrong, where, and why”.

A common production pattern:
1. detect anomaly on an aggregate metric (e.g., error_rate)
2. compute deltas by dimension to find top contributors:
 - region, cluster, device, app_version, request_type
3. present ranked suspects with evidence

This is essentially “explainability via decomposition”.
It often beats fancy ML models in reducing MTTR.

### 4.8 Handling seasonality and holidays (the hard baseline problem)

Many metrics have:
- daily cycles (traffic peaks)
- weekly cycles (weekday vs weekend)
- event-driven spikes (launches, holidays)

If you ignore seasonality, you’ll page every day at 9am.

Baseline strategies (in increasing complexity):
- compare to “same time yesterday / last week”
- rolling window median and MAD
- Holt-Winters / Prophet-like seasonal models
- learned baselines per segment

For production, the best choice is often:
- simple seasonal comparisons + robust statistics
- plus explicit suppression windows for known events (launch windows)

---

## 5. Data Flow (Streaming + Batch)

### 5.1 Streaming path
- ingest events into Kafka
- compute rolling windows (1m, 5m, 1h)
- compute features (rates, deltas)
- run rule-based detectors for immediate signals
- run lightweight statistical detectors (EWMA)
- emit alerts

### 5.2 Batch path
- nightly backfills, seasonality model fitting
- compute baselines per metric and per segment
- retrain unsupervised models if used
- store parameters in a registry

This hybrid is important because:
- streaming is low-latency but limited context
- batch has rich history but higher latency

---

## 6. Implementation (Core Logic in Python)

The goal here is to show “detector primitives” that can be composed. In production you’d wrap these in streaming jobs and persistent state.

### 6.1 Robust z-score (median + MAD)

``python
import numpy as np


def robust_zscore(x: np.ndarray) -> np.ndarray:
 """
 Robust z-score using median and MAD (median absolute deviation).
 Less sensitive to outliers than mean/std.
 """
 med = np.median(x)
 mad = np.median(np.abs(x - med)) + 1e-9
 return 0.6745 * (x - med) / mad


def detect_anomaly_robust_z(x_window: np.ndarray, threshold: float = 4.0) -> bool:
 z = robust_zscore(x_window)
 return abs(z[-1]) >= threshold
``

### 6.2 EWMA control chart

``python
def ewma(series: np.ndarray, alpha: float = 0.2) -> np.ndarray:
 out = np.zeros_like(series, dtype=float)
 out[0] = series[0]
 for i in range(1, len(series)):
 out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
 return out


def detect_ewma_shift(series: np.ndarray, alpha: float = 0.2, k: float = 3.0) -> bool:
 """
 Simple EWMA shift detector: compare last value to EWMA +/- k * std.
 (In production, use rolling std and seasonality-aware baselines.)
 """
 smoothed = ewma(series, alpha=alpha)
 mu = np.mean(smoothed[:-1])
 sigma = np.std(smoothed[:-1]) + 1e-9
 return abs(smoothed[-1] - mu) > k * sigma
``

### 6.3 Fusing detectors (rule + stats)

``python
from dataclasses import dataclass


@dataclass
class DetectionResult:
 is_anomaly: bool
 reason: str


def fused_detector(series: np.ndarray) -> DetectionResult:
 # Rule: “no data” or impossible values
 if np.isnan(series[-1]):
 return DetectionResult(True, "missing_data")
 if series[-1] < 0:
 return DetectionResult(True, "negative_value")

 # Statistical: robust z-score
 if detect_anomaly_robust_z(series, threshold=4.0):
 return DetectionResult(True, "robust_zscore")

 # Statistical: EWMA shift
 if detect_ewma_shift(series, alpha=0.2, k=3.0):
 return DetectionResult(True, "ewma_shift")

 return DetectionResult(False, "normal")
``

This illustrates a production principle:
**anomaly detection is a system of small components**, not a single model.

### 6.4 Multivariate anomaly detection (when one metric isn’t enough)

Many real anomalies only show up when you look at multiple signals together:
- latency rises while QPS is stable (infra saturation)
- error rate rises only for one endpoint (partial outage)
- embedding stats shift while overall accuracy proxy is stable (silent drift)

Multivariate approaches:
- Isolation Forest / One-Class SVM on feature vectors
- reconstruction error from autoencoders
- clustering distance to normal centroids

Here’s a small Isolation Forest example (conceptual; production needs careful feature scaling and segment baselines):

``python
from sklearn.ensemble import IsolationForest
import numpy as np


def fit_isolation_forest(X: np.ndarray) -> IsolationForest:
 """
 X shape: [n_samples, n_features]
 Features could include: latency_p95, error_rate, qps, cpu, mem, etc.
 """
 model = IsolationForest(
 n_estimators=200,
 contamination="auto",
 random_state=0,
 )
 model.fit(X)
 return model


def detect_multivariate(model: IsolationForest, x_latest: np.ndarray, score_threshold: float) -> bool:
 # sklearn: higher score = more normal; lower = more anomalous
 score = float(model.score_samples(x_latest.reshape(1, -1))[0])
 return score < score_threshold
``

Operational note:
- multivariate models are powerful but harder to explain
- pair them with decomposition-style explanations (top contributor slices) for RCA
- gate paging on both “model score” and “impact metrics” (e.g., SLO risk)

---

## 7. Scaling Strategies

### 7.1 Cardinality management (the silent killer)

If you alert on `metric x country x device x app_version`, cardinality explodes.
Strategies:
- alert only on top-K segments (heavy hitters)
- hierarchical detection:
 - detect anomaly globally
 - then drill down into segments to find top contributors
- sketching (Count-Min Sketch for counts, t-digest/KLL for quantiles)

### 7.2 State management in streaming

Detectors need history:
- rolling windows
- seasonal baselines
- suppression state

Store state in:
- stream processor state (RocksDB in Flink)
- or external state stores (Redis, Cassandra)

### 7.3 Multi-tenancy

Different teams want different sensitivity.
You need:
- per-metric configs
- per-tenant budgets (alert quotas)
- isolation (one tenant’s high-cardinality metrics shouldn’t DOS the system)

### 7.4 Change-log correlation (bridging detection to action)

The fastest way to reduce MTTR is to correlate anomalies with “what changed”:
- deploys
- feature flags
- config pushes
- schema changes
- infrastructure events (autoscaling, failovers)

A pragmatic design:
- ingest change events into the same stream as metrics
- attach a “recent changes” panel to every alert
- compute correlation candidates (“this alert fired within 10 minutes of deploy X in region Y”)

This does not require fancy causality to be useful.
In practice, showing the top 3 recent changes near the anomaly often saves hours.

### 7.5 Collective anomalies (the ones point detectors miss)

Not all anomalies are single spikes. Common anomaly types:
- **point anomaly**: one bad point
- **contextual anomaly**: normal value at the wrong time (seasonality)
- **collective anomaly**: a sustained subtle shift (e.g., p95 latency +5% for 6 hours)

Collective anomalies are the hardest in production because:
- they are “small enough” to evade thresholds
- but “long enough” to cause real user impact

Detectors for collective anomalies:
- EWMA / CUSUM-style change detection
- burn-rate based alerting (SLO-focused)
- rolling baseline comparisons (same time last week)

---

## 8. Monitoring & Metrics (for the anomaly system itself)

You need to monitor your monitor:
- alert volume per tenant
- false positive rate proxies (how many alerts are auto-acked)
- time-to-ack, time-to-resolve
- detector latency and backlog
- “missing data” alert rates (often instrumentation failures)

And you need a feedback loop:
when humans label an alert as “noise”, it should improve the system.

### 8.1 Evaluating anomaly detection (without perfect labels)

Unlike classification, anomaly detection often lacks ground truth.
Practical evaluation strategies:

- **Historical replay**
 - replay a week/month of metrics
 - compare alerts to known incidents and change logs

- **Synthetic injections**
 - inject controlled anomalies (drop QPS, spike errors, shift distributions)
 - ensure detectors catch them with acceptable delay

- **Proxy labels**
 - incident tickets, rollbacks, on-call notes
 - not perfect, but useful for measuring recall on “real pain”

- **Human-in-the-loop tuning**
 - collect feedback (noise vs real)
 - tune thresholds and suppression policies

You typically optimize for:
- high precision for paging alerts
- high recall for “notify” alerts (lower severity)

### 8.2 “Impact gating” (avoid paging on harmless anomalies)

A classic production trick:
don’t page just because a detector score is high.
Page when:
- anomaly score is high **and**
- impact metrics indicate user risk

Examples:
- error rate anomaly + SLO burn rate increasing
- latency anomaly + p99 above a hard threshold

This reduces alert fatigue while keeping critical incidents caught.

---

## 9. Failure Modes (and Mitigations)

### 9.1 Alert fatigue
Mitigation:
- dedupe + grouping
- severity levels
- alert budgets and rate limits
- better routing + suppression windows

### 9.2 Seasonality false positives
Mitigation:
- baseline models (daily/weekly)
- compare to same time last week
- use robust estimators

### 9.3 Missing data vs true zero
Mitigation:
- explicit “heartbeat” signals
- separate pipelines for “metric missing” vs “value is zero”

### 9.4 Feedback loops and gaming
If teams learn to “silence alerts”, your system loses trust.
Mitigation:
- governance: who can suppress, for how long
- audit logs
- periodic review of suppressed alerts

### 9.5 High-cardinality explosions

A common production outage pattern:
- someone adds a new label/dimension (e.g., `user_id`)
- cardinality explodes
- storage and stream processing costs spike
- detectors become meaningless (each series has too few points)

Mitigations:
- enforce label allowlists/denylists
- use hierarchical detection (alert on aggregate first, then drill down)
- cap top-k dimensions and use heavy-hitter sketches
- charge back cost to teams for uncontrolled cardinality (governance matters)

### 9.6 “Anomaly” isn’t always “bad” (launches look like incidents)

Product launches and marketing events create legitimate shifts.
If the system treats every spike as an incident, it loses credibility.

Mitigations:
- change-log correlation: if a known launch happened, adjust baseline
- planned maintenance windows (suppression)
- “expected anomaly” annotations (Grafana-style)

This is another reason anomaly detection is a system: you need metadata and governance, not just math.

---

## 10. Real-World Case Study

### Netflix/Streaming quality
Netflix-like systems track:
- playback start failures
- buffering ratio
- CDN health

Anomalies matter because:
- small regressions affect millions of users quickly
- seasonality exists (prime time)

Common pattern:
- global anomaly triggers
- drill-down identifies region/device/CDN nodes as top contributors
- routing goes to the owning team (CDN vs player vs backend)

### 10.1 A second case study: data pipeline anomaly in ML training

Consider a feature pipeline that writes training data daily.
One day, an upstream service changes a field from “milliseconds” to “seconds”.

What happens:
- models still train (no crash)
- loss might even look “okay”
- but downstream predictions degrade subtly

Anomaly detection can catch this early with:
- **schema validation**: units and ranges
- **distribution shift**: feature histograms vs baseline
- **training metrics drift**: gradient norms, embedding stats

The key lesson:
the most expensive ML failures are often “silent correctness failures”.
You need anomaly detection not just for infra metrics, but for data and model health signals.

### 10.2 An “RCA-first” UI pattern

When an alert fires, show:
- the metric that triggered
- the baseline comparison (last week, seasonal)
- the top contributing dimensions (country/device/version)
- links to relevant dashboards and runbooks
- recent change log correlation (deploys, config changes, schema changes)

This turns anomaly detection from “pager noise” into a decision system.

---

## 11. Cost Analysis

Cost drivers:
- stream processing state (memory/disk)
- high-cardinality storage
- alert routing and case management integration

Cost levers:
- limit cardinality and use sketches
- run heavy ML detectors only when cheaper detectors suspect an anomaly
- compress histories and store only what’s needed for RCA windows

---

## 12. Key Takeaways

1. **Anomaly detection is a system, not a model**: ingestion, features, detectors, alerting, RCA.
2. **Boundaries and invariants matter**: define “normal” and detect boundary violations, like two-pointer reasoning.
3. **Operational success = low false positives + good explanations**: otherwise no one trusts alerts.

### 12.1 Appendix: detector selection cheat sheet

Use this as a quick mapping from problem → detector:

| Signal type | Typical anomaly | Best first detector | Notes |
|------------|------------------|---------------------|------|
| Counters (QPS, errors) | drop to zero, spike | rules + burn-rate | also detect missing vs true zero |
| Latency percentiles | sustained shift | EWMA / baseline compare | seasonality is common |
| High-dimensional telemetry | weird combinations | Isolation Forest / AE | require careful explanation |
| Categorical slices | one segment dominates | hierarchical drill-down | “top contributors” is key |
| Data features (ML) | distribution shift | histogram distance + rules | unit changes are common |

Rule of thumb:
- page on rules + high-impact metrics
- use ML scores as supporting evidence or for “notify” tier alerts

### 12.2 Appendix: an alerting playbook (how to keep trust)

If your anomaly system is noisy, it dies. A practical playbook:

- **Start narrow**
 - detect a small set of critical metrics
 - keep paging volume low

- **Add explainability**
 - top contributors
 - baseline comparisons
 - recent change log correlation

- **Use severity tiers**
 - paging only for high-confidence/high-impact signals
 - Slack-only for exploratory detectors

- **Build feedback**
 - allow “noise” labeling
 - review suppressed alerts periodically

This is the same “pattern recognition” lesson as the DSA problem:
don’t react to every fluctuation; define the boundary of “actionable abnormal”.

### 12.3 Appendix: baseline modeling options (from simplest to strongest)

When teams say “anomaly detection”, they often mean “baseline modeling”.
Here’s a pragmatic ladder:

- **Static thresholds**
 - best for: hard limits (error_rate must be < 1%)
 - worst for: seasonal metrics

- **Rolling robust baseline (median/MAD)**
 - best for: noisy but mostly stationary signals
 - robust to spikes and outliers

- **Seasonal baseline (same time last week)**
 - best for: strong weekly patterns
 - easy to explain and cheap to run

- **Holt-Winters / ETS**
 - best for: smooth seasonality + trend
 - good interpretability

- **Learned baselines (per segment)**
 - best for: complex multi-segment products
 - requires strong governance to avoid accidental bias

Production tip:
start with seasonal comparisons + robust statistics.
Only move to heavier models when simpler baselines fail.

### 12.4 Appendix: a minimal “config contract” for detectors

To run this system safely at scale, you need a configuration contract:
- metric name
- aggregation window (1m/5m/1h)
- baseline method
- sensitivity / threshold
- severity tier (log/notify/page)
- routing owner (team/on-call)
- suppression windows (maintenance)

This turns anomaly detection into an operable platform:
teams can onboard new signals without changing code, and you can audit changes.

### 12.5 Appendix: anomaly vs drift vs outlier (how teams get confused)

These terms are often used interchangeably, but they’re different:

- **Outlier**: a single unusual point relative to its neighbors.
 - Example: one batch has a huge spike in null values.

- **Anomaly**: a point or segment that violates an expected pattern.
 - Example: error rate spike at 2am that is not part of normal seasonality.

- **Drift**: the underlying distribution changes over time.
 - Example: feature distribution slowly shifts after a product change.

Operationally:
- outliers are often “data quality” issues
- anomalies are “incident” candidates
- drift is “model health” and requires different response (retraining, feature changes)

The best platforms support all three, but keep the response paths distinct:
- anomalies page humans
- drift triggers investigation and model iteration
- outliers often trigger data validation and pipeline fixes

### 12.6 Appendix: an on-call runbook for anomaly alerts

When an alert pages a human, the system should also provide a runbook-style checklist. A practical sequence:

1. **Confirm impact**
 - are user-facing SLOs burning?
 - is this confined to a segment (region/device) or global?

2. **Check data validity**
 - missing data vs real zeros
 - instrumentation changes (new labels, new units)

3. **Correlate with recent changes**
 - deploys, feature flags, config changes
 - schema changes in pipelines

4. **Drill down**
 - top contributors by dimension
 - compare against historical baselines (same time last week)

5. **Mitigate**
 - rollback suspect deploys
 - fail over traffic if it’s infra-related
 - suppress alerts only with an explicit maintenance annotation

This checklist turns “anomaly detection” into operational reality: consistent response beats clever scoring.

### 12.7 Appendix: testing detectors safely before production

Before enabling paging:
- run in shadow mode (log-only) for 1–2 weeks
- replay historical incidents and ensure alerts would have fired
- inject synthetic anomalies (spikes, drops, sustained shifts)
- check alert volume budgets per team

If you do this, you avoid the most common failure:
shipping a detector that immediately pages everyone and loses trust forever.

### 12.8 Appendix: cost and scaling intuition (why platforms exist)

Teams often start anomaly detection as ad-hoc scripts per service.
At small scale, that works. At org scale, it collapses due to:
- duplicated logic (everyone re-implements EWMA differently)
- inconsistent thresholds and alert policies
- uncontrolled cardinality costs
- lack of auditability (“who changed this threshold?”)

That’s why anomaly detection becomes a platform:
- shared ingestion and schema validation
- shared baseline modeling primitives
- shared routing/suppression/governance
- shared RCA UI patterns (top contributors, change-log correlation)

If you can explain this evolution clearly in interviews, it signals strong “systems taste”.

### 12.9 Appendix: multi-tenancy and fairness (an under-discussed risk)

Anomaly systems can create “fairness” issues at the org level:
- teams with noisier metrics consume disproportionate paging attention
- teams with low traffic get ignored because their signals are sparse
- some regions/devices get under-monitored because baselines are not segment-aware

Mitigations:
- per-tenant alert budgets (rate limits and quotas)
- segment-aware baselines for critical dimensions (region/device/app version)
- “coverage” metrics: which segments have enough data to monitor reliably
- explicit ownership: every alert route must map to a team/runbook

This is the same lesson as other large platforms: governance is part of the technical design.

### 12.10 Appendix: a minimal anomaly “triage packet” (what every alert should carry)

To make alerts actionable, every alert should include a small, standardized packet:

- **What**: metric name, current value, baseline value, anomaly score
- **When**: start time, duration, detection time
- **Where**: top affected segments (region/device/version) with contribution deltas
- **Impact**: SLO burn rate / user-impact proxy
- **Why likely**: top recent change events correlated (deploy/flag/config/schema)
- **Next**: runbook link + owning team + mitigation suggestions (rollback, failover)

This single design choice reduces MTTR more than most detector tweaks, because it turns “math output” into “operational context”.

---

**Originally published at:** [arunbaby.com/ml-system-design/0052-anomaly-detection](https://www.arunbaby.com/ml-system-design/0052-anomaly-detection/)

*If you found this helpful, consider sharing it with others who might benefit.*

