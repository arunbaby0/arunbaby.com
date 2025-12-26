---
title: "Federated Learning"
day: 51
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - federated-learning
  - distributed-training
  - privacy
  - secure-aggregation
  - on-device-ml
  - mlops
difficulty: Hard
subdomain: "Distributed Training"
tech_stack: PyTorch, TensorFlow Federated, gRPC, Kubernetes
scale: "10M+ devices, intermittent connectivity, privacy constraints"
companies: Google, Apple, Meta, Samsung
related_dsa_day: 51
related_ml_day: 51
related_speech_day: 51
related_agents_day: 51
---

**"If data can’t move, move the model—and design the system so the server never sees what matters."**

## 1. Problem Statement

Many of the most valuable datasets live on user devices:
- typed text (next-word prediction)
- voice snippets (wake word, personalization)
- photos, app usage patterns, sensor streams

But collecting raw data centrally is often unacceptable due to:
- privacy expectations
- legal constraints (GDPR/CCPA, children’s data)
- trust and product risk
- bandwidth and battery cost

**Federated Learning (FL)** trains a global model by sending the model to devices, training locally, and sending only updates back.

**Goal**: Design a production-grade federated learning system that:
- improves model quality
- preserves privacy
- scales to millions of devices
- handles unreliable clients and skewed data
- is observable, debuggable, and safe

Thematic link for today: **binary search and distributed algorithms**.
Federated learning is a distributed optimization algorithm under strict constraints; the engineering challenge is making that algorithm reliable at scale.

---

## 2. Understanding the Requirements

### 2.1 Functional requirements

- Train a global model over many client datasets without centralizing raw data.
- Periodically produce a new model version for serving (on-device or server-side).
- Support multiple model types:
  - logistic regression / small neural nets
  - large embeddings
  - speech models (acoustic / personalization heads)
- Support experimentation: A/B testing between FL variants (FedAvg vs FedProx, compression choices, DP).

### 2.2 Non-functional requirements (the real difficulty)

- **Privacy**: server should not learn individual client examples (or even individual client updates, ideally).
- **Scalability**: millions of eligible devices; only a fraction participate per round.
- **Reliability**: client dropouts are normal. Devices sleep, go offline, change networks.
- **Heterogeneity**: devices differ in CPU/GPU/NPU, memory, battery, OS version.
- **Data skew (non-IID)**: one user types “football”, another types “biology”. Local distributions differ drastically.
- **Security**: malicious clients can poison updates or try to infer other clients.
- **Observability**: you need metrics without raw data; debugging is harder than centralized training.
- **Cost**: bandwidth, device compute, server aggregation capacity.

---

## 3. High-Level Architecture

At a high level, an FL system runs repeated **rounds**:
1. select clients
2. broadcast model + training plan
3. local training on-device
4. upload updates
5. secure aggregation
6. server optimization step
7. evaluate and promote model

### 3.1 Architecture diagram (control + data plane)

```
                    +----------------------------------+
                    |        FL Orchestrator           |
                    | (round scheduler + policy engine)|
                    +-------------------+--------------+
                                        |
                     (model + plan)     |    (eligibility + sampling)
                                        v
  +----------------------+     +----------------------+     +----------------------+
  | Model Registry       |     | Client Selector      |     | Privacy / Security   |
  | (versions, metadata) |     | (sampling, quotas)   |     | (DP params, SA keys) |
  +----------+-----------+     +----------+-----------+     +----------+-----------+
             |                                |                          |
             |                                |                          |
             v                                v                          v
    +-------------------+           +-------------------+        +-------------------+
    | Broadcast Service |---------> |  Devices (clients)|------> | Secure Aggregator |
    | (CDN / gRPC)      |  model+   | local train,      | updates| (no raw updates)  |
    +-------------------+  plan     | clip/noise,       |        +---------+---------+
                                     | upload partials   |                  |
                                     +---------+---------+                  |
                                               |                            |
                                               v                            v
                                       +-------------------+      +-------------------+
                                       | Telemetry         |      | Server Optimizer  |
                                       | (metrics only)    |      | (FedAvg/FedOpt)   |
                                       +-------------------+      +---------+---------+
                                                                         |
                                                                         v
                                                                 +-------------------+
                                                                 | Eval + Promotion  |
                                                                 | (holdout + guard) |
                                                                 +-------------------+
```

### 3.2 Key responsibilities

- **Orchestrator**: defines “who participates, when, and with what constraints.”
- **Client runtime**: runs on-device training safely, respecting battery/network policies.
- **Secure aggregation**: ensures server only sees aggregated updates, not individual updates.
- **Model registry**: versioning, metadata, reproducibility, rollback.
- **Evaluation**: must work without raw client data (or with privacy-safe evaluation).

---

## 4. Component Deep-Dives

### 4.1 Client selection (sampling) and eligibility

In production, you rarely train on “all devices”.
You define an **eligibility policy**, for example:
- device on Wi-Fi
- charging
- idle
- minimum battery %
- opted-in (consent)
- locale / language segment match
- model version compatibility (runtime can execute)

Then you sample eligible devices:
- uniform random sampling can over-represent highly active regions
- you may need stratified sampling (by geography, device class, language)

**Why this matters**: the sampling policy is your “data loader” in FL. If it’s biased, your model is biased.

### 4.2 Client training plan (what executes on device)

A “round” is typically packaged as:
- base model weights
- optimizer configuration (learning rate, epochs, batch size)
- data selection rules (which local records are included)
- privacy rules (clipping norm, DP noise, secure aggregation protocol version)

This is similar to shipping a mini training job to untrusted compute.
So you need:
- strict sandboxing
- deterministic kernels where possible
- a way to revoke/break-glass if a plan misbehaves (battery drain incidents are real)

### 4.3 Aggregation + optimization

The canonical algorithm is **FedAvg**:
1. each client trains locally from \(w_t\) to \(w_t^k\)
2. server averages client deltas weighted by number of examples

\[
w_{t+1} = \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j}\; w_t^k
\]

Variants:
- **FedAvgM** (server momentum)
- **FedAdam / FedYogi** (server-side adaptive optimizers)
- **FedProx** (adds proximal term to stabilize with non-IID data)

Choosing the right optimizer is a systems decision:
- adaptive optimizers can converge faster but may be more sensitive to client drift
- non-IID data often benefits from proximal terms or personalization layers

### 4.4 Secure aggregation (SA)

Even model updates can leak private information (membership inference, gradient inversion).
Secure aggregation aims to ensure:
- server learns only the sum/average of updates across many clients
- individual updates remain hidden

Typical properties:
- **threshold** \(T\): aggregation succeeds only if at least \(T\) clients complete
- **dropout tolerance**: protocol must handle client dropouts without revealing individuals

Operationally:
- SA adds protocol complexity and latency
- SA affects observability (you can’t inspect individual updates)

### 4.5 Differential privacy (DP) and clipping

DP is often layered on top of FL:
- each client clips its update to a maximum norm \(C\)
- adds noise (local DP or central DP depending on threat model)

Why clipping matters:
- without clipping, a single user with extreme data can dominate the aggregate
- clipping bounds sensitivity, enabling DP guarantees

There’s a “binary search” motif here:
in practice you often **tune the clipping norm** by searching for a value that preserves utility while controlling leakage and outliers—this becomes an iterative “search over constraints”, not unlike how we search for a partition in the median problem.

### 4.6 Secure aggregation protocol details (what actually happens)

Secure aggregation sounds like a single box (“the secure aggregator”), but it’s usually a multi-step protocol. A simplified view:

1. **Key agreement / setup**
   - Clients establish pairwise secrets (or receive public keys) for masking.

2. **Masking**
   - Each client creates a random mask vector \(r_k\).
   - Client sends *shares* of masks such that masks can be reconstructed only if enough clients finish.
   - The client uploads masked update: \(u_k + r_k\).

3. **Dropout handling**
   - Some clients drop out mid-round.
   - The protocol reconstructs masks for missing clients (or cancels their masks) so the final sum unmasks correctly.

4. **Aggregation**
   - Server ends with \(\sum_k u_k\) (or weighted sum), but never sees any individual \(u_k\).

Design implications:
- You need a **threshold** (e.g., 1k clients) to reduce privacy leakage and to make SA feasible.
- You need a **protocol state store** (often a database) to track which clients completed which phase.
- SA can become your biggest source of operational failures (timeouts, key exchange bugs, state corruption).

### 4.7 Robust aggregation (Byzantine resilience)

Even with privacy, you still need robustness. Some clients may be buggy or malicious.
Robust aggregators attempt to reduce the impact of outliers:

- **Trimmed mean**: drop top/bottom \(p\%\) per coordinate.
- **Coordinate-wise median**: median per coordinate (very robust, but can be noisy in high dimensions).
- **Norm bounding**: reject aggregates if norm spikes beyond historical ranges.

Trade-off table:

| Aggregator | Pros | Cons | When to use |
|-----------|------|------|-------------|
| FedAvg | simple, fast | sensitive to outliers | trusted clients, small cohorts |
| Trimmed mean | robust to outliers | loses signal if trimming too aggressive | noisy client populations |
| Median | strong robustness | can slow convergence | high poisoning risk |

### 4.8 DP accounting (epsilon budgets) in practice

DP is not “add noise once and forget”.
You need a privacy accountant to track privacy loss across rounds.

Operationally this means:
- per model / per cohort DP budget (e.g., “this model can spend \(\epsilon \le 3\) over 30 days”)
- round-level parameters: clip norm, noise multiplier, cohort size
- stopping criteria: “stop training when budget is exhausted”

The practical system design question:
> Where does the DP accounting live, and how do you enforce it so teams can’t silently ship a ‘privacy-unsafe’ training plan?

Good answer:
- DP accounting is part of the orchestrator policy engine
- training plans are validated before launch
- model registry stores DP metadata (noise, clip, epsilon consumed)

---

## 5. Data Flow (One Round End-to-End)

### 5.1 Step-by-step

1. **Round start**
   - Orchestrator decides: model `v123`, plan `p77`, cohort: “English-US”, target: 50k clients.

2. **Client selection**
   - Selector samples from eligible clients, respecting quotas.

3. **Broadcast**
   - Devices fetch:
     - model weights (CDN)
     - training plan (gRPC / config service)

4. **Local training**
   - Device trains for a small number of steps/epochs.
   - Applies clipping and optional noise.

5. **Upload**
   - Device uploads masked update shares for SA (or direct update if SA disabled).

6. **Secure aggregation**
   - Aggregator reconstructs only the sum/average update once threshold reached.

7. **Server update**
   - Server optimizer applies aggregated update to produce new model candidate `v124-candidate`.

8. **Evaluation**
   - Evaluate on:
     - server-side public/curated datasets
     - privacy-safe on-device eval (clients compute metrics and aggregate)

9. **Promotion**
   - If guardrails pass (quality + privacy + regressions), promote to `v124`.
   - Otherwise rollback, adjust plan, or reduce cohort.

### 5.2 What makes FL “different”
In centralized training, you can log:
- per-example losses
- per-batch gradients
- exact failure samples

In FL, you mostly see:
- aggregated metrics
- participation rates
- update norms distributions (sometimes)

This is why your evaluation design matters as much as your optimizer.

---

## 6. Scaling Strategies

### 6.1 Participation scaling: bandwidth and concurrency

Assume:
- model size: 20 MB
- 50k clients per round

Naively broadcasting 1 TB per round is expensive.
So systems use:
- **delta updates** (send weight diffs between versions)
- **compression** (quantization, sparsification)
- **CDN caching** for base weights

On upload:
- each client might send 100 KB–5 MB depending on compression
- server must handle bursty uploads (thundering herd after a “charging + Wi-Fi” window)

### 6.2 Straggler and dropout handling

Dropouts are normal. Designs:
- set a strict round deadline; accept partial participation
- over-sample clients anticipating dropouts
- SA must tolerate dropout while preserving privacy

### 6.3 Dealing with non-IID data

Non-IID is the default.
Techniques:
- **FedProx**: reduces client drift
- **personalization layers**: shared backbone + user-specific head
- **clustered FL**: group clients with similar distributions, train multiple models

This is where FL resembles distributed algorithms beyond “just average”:
you’re effectively solving a global objective with local constraints and partial information.

### 6.4 Communication efficiency

Common approaches:
- **8-bit / 4-bit quantization** of updates
- **top-k sparsification** (send largest gradient components)
- **sketching** (CountSketch-style)
- **periodic averaging** (clients do more local steps, fewer rounds)

Trade-off:
- more local steps reduces comms but increases drift on non-IID data

---

## 7. Implementation (Core Logic in Python)

This is a simplified simulation of FedAvg. In production, the core is the same conceptually, but wrapped in client runtimes, orchestration, and secure aggregation.

```python
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class ClientUpdate:
    n_examples: int
    delta: np.ndarray  # flattened weights delta


def clip_update(delta: np.ndarray, clip_norm: float) -> np.ndarray:
    """L2 clip to bound sensitivity and reduce outliers."""
    norm = np.linalg.norm(delta)
    if norm <= clip_norm or norm == 0.0:
        return delta
    return delta * (clip_norm / norm)


def client_train_step(w: np.ndarray, data: Tuple[np.ndarray, np.ndarray]) -> ClientUpdate:
    """
    Toy local training:
    - pretend we run a couple of SGD steps and return a model delta
    """
    x, y = data
    # Fake gradient: (this is placeholder logic for illustration)
    grad = x.T @ (x @ w - y) / max(1, len(y))
    lr = 0.1
    w_new = w - lr * grad
    delta = w_new - w
    return ClientUpdate(n_examples=len(y), delta=delta)


def fedavg_aggregate(updates: List[ClientUpdate]) -> np.ndarray:
    """Weighted average of deltas."""
    total = sum(u.n_examples for u in updates)
    if total == 0:
        raise ValueError("No examples aggregated")
    return sum((u.n_examples / total) * u.delta for u in updates)


def federated_round(w: np.ndarray, client_datas: List[Tuple[np.ndarray, np.ndarray]], clip_norm: float) -> np.ndarray:
    """One round: local train -> clip -> aggregate."""
    updates = []
    for data in client_datas:
        upd = client_train_step(w, data)
        upd.delta = clip_update(upd.delta, clip_norm)
        updates.append(upd)
    agg_delta = fedavg_aggregate(updates)
    return w + agg_delta
```

What’s missing (intentionally) compared to real FL:
- secure aggregation (so the server never sees `upd.delta`)
- device eligibility and scheduling
- privacy noise (DP)
- anti-poisoning defenses
- robust aggregation (median/trimmed mean)

Those missing pieces are the “system design interview”.

---

## 8. Monitoring & Metrics

You need metrics in four buckets:

### 8.1 Participation health
- eligible devices / selected devices / completed devices
- dropout rate by stage (download, train, upload)
- round latency distribution
- SA threshold success rate

### 8.2 Model update health (privacy-aware)
- aggregate update norm
- per-layer norm summaries (aggregated)
- fraction of updates clipped (clients can report “clipped yes/no” as a bit; aggregate counts)
- divergence signals (server loss spikes)

### 8.3 Quality metrics
- on-device eval accuracy (aggregated)
- server-side eval (public dataset) accuracy
- segment metrics (locale, device class)

### 8.4 Cost metrics
- bytes downloaded/uploaded per successful client
- device compute time per round
- server CPU time per aggregation

Key challenge:
you must learn to debug using **distributions and aggregates**, not individual samples.

### 8.5 Debugging playbook (what you do when quality drops)

When a centralized model regresses, you can often:
- inspect failed examples
- run slice analysis on raw data
- re-train with a cleaned dataset

In FL, the playbook is different. A practical incident response flow:

1. **Check participation first**
   - Did eligible devices drop (OS update, policy bug, backend outage)?
   - Did completion rate change (network conditions, app crash)?

2. **Check update health**
   - Did aggregate update norm spike or collapse?
   - Did clipping rate jump (indicating outliers or plan misconfiguration)?

3. **Check segment shifts**
   - Did the sampling distribution change (more low-end devices, different geos)?
   - Are regressions concentrated in one locale/device segment?

4. **Check plan drift**
   - Did someone change local epochs / learning rate / batch size?
   - Did the DP policy change (more noise, smaller cohorts)?

5. **Roll back safely**
   - FL systems should support rollback to last-known-good model + plan.
   - Never “debug in production” by shipping risky plans to all devices.

This is why model registry metadata and orchestration policies are first-class engineering.

### 8.6 Privacy-safe evaluation patterns

To measure quality without collecting raw client data:
- run a local eval on device (loss/accuracy on a held-out local slice)
- report only aggregated metrics (SA and/or DP)

Common design:
- the orchestrator ships an “eval-only plan” (no training)
- devices compute metrics and upload only counts/sums
- you can then compare cohorts without seeing a single example

This becomes a reusable system primitive ("federated eval") that product teams can use.

### 8.7 Fleet health metrics (client runtime as a distributed system)

Treat the client runtime like a large distributed compute cluster.
You need SRE-style metrics:
- crash rate during training (by device model/OS)
- battery impact distributions
- CPU time / memory usage
- network bytes and failure codes

The “training plan” is effectively code you deploy to that cluster.
So you need:
- canarying
- staged rollout
- rapid disable switches

### 8.8 Auditability and compliance

If FL is used for privacy reasons, you must be able to prove:
- which models were trained with which privacy parameters
- which cohorts were eligible/participating
- what DP budget was consumed over time

Operationally:
- store DP metadata in the model registry
- sign training plans (policy enforcement)
- create an audit trail of promotion decisions

---

## 9. Failure Modes (and Mitigations)

### 9.1 Client poisoning
Attack: a malicious client sends a crafted update to cause targeted behavior.

Mitigations:
- robust aggregation (trimmed mean, coordinate-wise median)
- anomaly detection on update norms (client-side + aggregate)
- limit per-client contribution via clipping
- attestations / trusted execution where possible

### 9.2 Privacy leakage via updates
Attack: gradient inversion or membership inference.

Mitigations:
- secure aggregation + minimum cohort size
- clipping + DP noise
- restrict model capacity for sensitive tasks

### 9.3 Non-IID drift and catastrophic regressions
Example: model overfits heavy users in one locale.

Mitigations:
- stratified sampling
- segment-based guardrails
- personalization layers rather than fully global updates

### 9.4 Operational instability
- SA failures due to too many dropouts
- app version fragmentation breaks client runtime

Mitigations:
- over-sample, shorten round deadlines
- strict compatibility checks
- staged rollouts for new plan versions

### 9.5 Data poisoning without “malice” (bugs look like attacks)

Not all bad updates are adversarial. Many incidents are plain bugs:
- a client preprocessing change flips labels
- a corrupted local cache creates nonsense examples
- a new OS version changes tokenizer behavior

In centralized training you might detect this via offline data validation.
In FL you must rely on:
- plan versioning (so you can pinpoint which plan produced the regression)
- participation segmentation (so you can isolate which app/OS version is problematic)
- aggregate anomaly detection (norm spikes, sudden loss changes)

This is one reason why FL orchestration feels like operating a distributed system:
the client fleet is not homogeneous, and “bad actors” often appear accidentally.

### 9.6 Silent sampling bias (the most common real-world failure)

Even if your training algorithm is perfect, your sampling can drift:
- you accidentally prefer devices that are always on Wi‑Fi and charging (a specific demographic)
- you exclude low-end phones due to memory limits (hurts global fairness)
- you oversample one geography because of timezone scheduling

Mitigations:
- explicit stratified quotas (by geo, device class, app version)
- sampling audits (“who participated last week vs target distribution?”)
- fairness guardrails (segment metrics must not regress)

If you want to impress in interviews, say:
> In FL, the client selector is your data loader. Sampling bias is a training bug.

---

## 10. Real-World Case Study

### 10.1 Google Gboard next-word prediction
One of the most cited FL deployments.
The model trains on-device on typed text without uploading raw keystrokes.

Key lessons:
- device scheduling is as important as the optimizer
- evaluation is hard without raw data
- privacy constraints force different debugging techniques

### 10.2 Apple on-device personalization
Apple has publicly discussed privacy-preserving analytics and on-device ML patterns.
Even when not strictly FL, the constraints are similar:
- data stays on device
- server learns via aggregates

### 10.3 A speech-flavored case study (personalized wake words / accents)

Speech is a great “stress test” for FL:
- client data is extremely sensitive
- models are large
- non-IID is strong (accent, environment, microphone)

A realistic production approach:
- keep the base wake word detector global and stable
- personalize a small threshold or adapter layer per device
- aggregate only privacy-protected signals across devices to improve the global model slowly

Engineering lesson:
you often split learning into:
- **fast local personalization** (small, device-specific)
- **slow global improvements** (federated, heavily gated)

This reduces risk while still benefiting from collective learning.

---

## 11. Cost Analysis

Cost drivers:
- **bandwidth** (download + upload)
- **server aggregation** (compute + storage for protocol states)
- **engineering complexity** (privacy/security requirements increase operational overhead)

Practical levers:
- compress models and updates
- reduce round frequency (but monitor for quality stagnation)
- select only clients likely to complete (charging + Wi-Fi windows)

### 11.1 A back-of-the-envelope cost model (so you can reason in interviews)

Assume a mid-size on-device model:
- model weights: 20 MB (download)
- update payload (compressed): 200 KB (upload)
- 50k participating clients per day (across many rounds)

Daily bandwidth:
- downloads: \(50{,}000 \times 20\text{MB} = 1{,}000{,}000\text{MB} \approx 1\text{TB}\)
- uploads: \(50{,}000 \times 200\text{KB} = 10{,}000{,}000\text{KB} \approx 10\text{GB}\)

Even if CDNs make downloads “cheap”, 1 TB/day is not free.
This is why production FL systems invest in:
- delta updates between model versions
- caching via CDN
- smaller adapter-based updates rather than full model updates

### 11.2 The “client compute cost” you don’t pay (but your users do)

Server-side training costs show up on your cloud bill.
Federated training moves a large part of compute to devices:
- CPU/NPU cycles
- battery usage
- thermal constraints

If training is too aggressive, you get:
- user complaints (“phone gets hot”)
- OS throttling
- drops in completion rate (which also hurts model quality)

So cost is not just “dollars”; it’s:
- product risk
- engagement risk
- fleet health risk

### 11.3 ROI framing

In many teams, FL is justified when:
- privacy constraints block centralized data collection
- personalization materially improves retention/engagement
- the incremental engineering effort pays off via model quality and trust

In interviews, a crisp way to say it:
> FL is expensive to build, but sometimes it’s the only path to high-quality models under privacy constraints.

---

## 12. Key Takeaways

1. **Federated learning is distributed optimization under constraints**: unreliable clients, non-IID data, privacy/security.
2. **System design dominates**: client eligibility, secure aggregation, privacy, observability, and rollout safety are the hard parts.
3. **Boundary-first thinking scales**: like the median partition trick, FL often succeeds by optimizing what you exchange (bounded, aggregated signals) rather than moving raw data.

### 12.1 A concise “design review checklist” for FL systems

If you’re reviewing a federated learning design, ask:

- **Client policy**
  - How do we decide eligibility (Wi‑Fi, charging, opt-in)?
  - What is the target sampling distribution (and how do we audit it)?

- **Privacy**
  - Is secure aggregation enabled for sensitive models?
  - Do we clip updates? Where is the DP budget accounted and enforced?
  - What is the minimum cohort size per segment?

- **Robustness**
  - What aggregator do we use (FedAvg vs robust variants)?
  - How do we handle dropouts and stragglers?
  - What anti-poisoning mitigations exist?

- **MLOps**
  - Are plans versioned and validated before execution?
  - Can we canary/roll back both model and plan quickly?
  - What metrics exist without inspecting raw client data?

If these questions have crisp answers, you’re operating FL like a real distributed system instead of a research prototype.

### 12.2 Deployment note: FL still needs a serving story

Federated learning updates a model, but product impact comes from **serving**:
- on-device inference (TFLite/CoreML)
- server-side inference (if the model isn’t privacy-sensitive at query time)

Practical serving considerations:
- keep runtime compatibility (model format + ops supported on older devices)
- stage rollouts with kill-switches (bad models are user-facing incidents)
- monitor on-device latency and battery impact (quality gains can be negated by UX regressions)

In other words: FL is not just training infrastructure; it’s an end-to-end system from “round” to “release”.

---

**Originally published at:** [arunbaby.com/ml-system-design/0051-federated-learning](https://www.arunbaby.com/ml-system-design/0051-federated-learning/)

*If you found this helpful, consider sharing it with others who might benefit.*

