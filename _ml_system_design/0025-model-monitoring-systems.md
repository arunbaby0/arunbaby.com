---
title: "Model Monitoring Systems"
day: 25
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - monitoring
  - drift-detection
  - production-ml
  - observability
  - devops
subdomain: "MLOps"
tech_stack: [Prometheus, Grafana, EvidentlyAI, Arize]
scale: "Monitoring 1000+ models in production"
companies: [Uber, Netflix, Datadog, Arize AI]
related_dsa_day: 25
related_ml_day: 25
related_speech_day: 25
---

**The silent killer of ML models is not a bug in the code, but a change in the world.**

## The Problem: Silent Failure

In traditional software, if you deploy a bug, the code crashes (Stack Trace, 500 Error). You know immediately.
In Machine Learning, if the world changes but your model stays the same, the model **fails silently**. It continues to output predictions, but they are wrong.

**Example:**
- **Training (2019):** Predict house prices. Feature: "Number of Bedrooms".
- **Production (2020):** COVID hits. People want "Home Offices".
- **Result:** The model still predicts high prices for small downtown apartments, but the market has shifted to suburbs. The model's error rate spikes, but no alarm bells ring.

This phenomenon is called **Drift**.

## Types of Drift

### 1. Data Drift (Covariate Shift)
The distribution of the input data `P(X)` changes.
- **Example:** You trained on English tweets. Suddenly, users start tweeting in Spanish.
- **Detection:** Compare the histogram of inputs today vs. training data.

### 2. Concept Drift (Prior Probability Shift)
The relationship between inputs and outputs `P(Y|X)` changes.
- **Example:** "Spam" definition changes. "Work from home" was spam in 2018, but legitimate in 2020.
- **Detection:** Requires **Ground Truth** (labels). If you don't have immediate labels (e.g., credit default takes months), this is hard.

### 3. Label Drift
The distribution of the target variable `P(Y)` changes.
- **Example:** In a fraud model, usually 1% is fraud. Suddenly, 20% is fraud (attack underway).

## High-Level Architecture: The Monitoring Stack

```ascii
+-----------+     +------------+     +-------------+
| Live App  | --> | Log Stream | --> | Drift Calc  |
+-----------+     +------------+     +-------------+
(FastAPI/Go)      (Kafka/S3)         (Airflow/Spark)
                                           |
                                           v
+-----------+     +------------+     +-------------+
| Alerting  | <-- | Dashboard  | <-- | Metrics DB  |
+-----------+     +------------+     +-------------+
(PagerDuty)       (Grafana)          (Prometheus)
```

How do we build a system to catch this?

### 1. The Data Layer (Logging)
Every prediction request and response must be logged.
- **Payload:** `{"input": [0.5, 2.1, ...], "output": 0.9, "model_version": "v1.2", "timestamp": 12345}`
- **Store:** Kafka -> Data Lake (S3/BigQuery).

### 2. The Calculation Layer (Drift Detection)
A batch job (Airflow) or stream processor (Flink) calculates statistics.
- **Statistical Tests:**
    - **KS-Test (Kolmogorov-Smirnov):** For continuous features. Measures max distance between two CDFs.
    - **Chi-Square Test:** For categorical features.
    - **PSI (Population Stability Index):** Industry standard for financial models.
    - **KL Divergence:** Measures information loss.

### 3. The Visualization Layer (Dashboards)
- **Tools:** Grafana, Arize AI, Evidently AI.
- **Alerts:** PagerDuty if `Drift Score > Threshold`.

## Deep Dive: Population Stability Index (PSI)

PSI is the gold standard metric.
`PSI = Sum( (Actual% - Expected%) * ln(Actual% / Expected%) )`

**Interpretation:**
- `PSI < 0.1`: No significant drift.
- `0.1 < PSI < 0.2`: Moderate drift. Investigate.
- `PSI > 0.2`: Significant drift. **Retrain immediately.**

**Implementation in Python:**

```python
import numpy as np

def calculate_psi(expected, actual, buckets=10):
    # 1. Define buckets (breakpoints) based on Expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    
    # 2. Calculate frequencies
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # 3. Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # 4. Calculate PSI
    psi_values = (actual_percents - expected_percents) * \
                 np.log(actual_percents / expected_percents)
    
    return np.sum(psi_values)

# Example
train_data = np.random.normal(0, 1, 1000)
prod_data = np.random.normal(0.5, 1, 1000) # Mean shifted
print(f"PSI: {calculate_psi(train_data, prod_data):.4f}")
```

## Production Engineering: Async vs. Sync Monitoring

**Synchronous (Blocking):**
- Calculate drift *during* the request.
- **Pros:** Immediate blocking of bad data.
- **Cons:** Adds latency. Expensive computation.
- **Use Case:** Fraud detection (block the transaction).

**Asynchronous (Non-Blocking):**
- Log data, calculate drift every hour/day.
- **Pros:** Zero latency impact.
- **Cons:** Delayed reaction.
- **Use Case:** Recommendation systems, Ad ranking.

## Advanced Monitoring: Embedding Drift

Statistical tests like KS-Test work great for tabular data (Age, Income).
But how do you monitor **Embeddings** (Vectors of size 768)?
You can't run KS-Test on 768 dimensions independently.

**Solution: Dimensionality Reduction.**
1.  **UMAP / t-SNE:** Project the 768-dim vectors to 2D.
2.  **Visual Inspection:** Plot the Training Data (Blue) and Production Data (Red).
3.  **Drift:** If the Red points form a cluster where there are no Blue points, you have **Out-of-Distribution (OOD)** data.

**Automated Metric:**
- **M-Statistic (Maximum Mean Discrepancy):** Measures the distance between two distributions in high-dimensional space using a kernel function.
- **Cosine Similarity:** Calculate the average cosine similarity between production embeddings and the "centroid" of training embeddings.

## Advanced Monitoring: Feature Attribution Drift (SHAP)

Sometimes the data looks fine, and the predictions look fine, but the **reasoning** has changed.
**Example:**
- **Train:** Model relies on "Income" to predict "Loan".
- **Prod:** Model starts relying on "Zip Code" (because Income became noisy).

**Detection:**
1.  Calculate **SHAP values** for a sample of production requests.
2.  Rank features by importance.
3.  Compare with training feature importance.
4.  **Alert:** If the rank order changes significantly (Kendall's Tau correlation).

## Ethics: Bias and Fairness Monitoring

Drift isn't just about accuracy; it's about **Fairness**.
If your model starts rejecting more women than men, you need to know *immediately*.

**Metrics to Monitor:**
1.  **Demographic Parity:** `P(Predicted=1 | Male) == P(Predicted=1 | Female)`.
2.  **Equal Opportunity:** `TPR(Male) == TPR(Female)`.
3.  **Disparate Impact:** `Ratio of acceptance rates > 0.8`.

**Implementation:**
You need "Protected Attributes" (Race, Gender) in your logs.
*Warning:* Often you are legally *not allowed* to store these.
*Workaround:* Use a trusted third-party auditor or aggregate metrics anonymously.

## Implementation: Building a Monitoring Service

Let's build a real-time monitor using **FastAPI** and **Prometheus**.

```python
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app
import numpy as np

app = FastAPI()

# Metrics
PREDICTION_COUNTER = Counter("model_predictions_total", "Total predictions")
INPUT_VALUE_HIST = Histogram("input_feature_value", "Distribution of input feature")
DRIFT_GAUGE = Gauge("model_drift_score", "Current PSI score")

# Reference Distribution (loaded from file)
REF_DIST = np.load("reference_dist.npy")
CURRENT_WINDOW = []

@app.post("/predict")
def predict(features: list):
    # 1. Log Metric
    PREDICTION_COUNTER.inc()
    INPUT_VALUE_HIST.observe(features[0]) # Monitor 1st feature
    
    # 2. Add to Window for Drift Calc
    CURRENT_WINDOW.append(features[0])
    if len(CURRENT_WINDOW) > 1000:
        score = calculate_psi(REF_DIST, CURRENT_WINDOW)
        DRIFT_GAUGE.set(score)
        CURRENT_WINDOW.clear()
        
    # 3. Predict
    return {"prediction": 0.9}

# Expose /metrics endpoint for Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

**Architecture:**
1.  **FastAPI:** Serves the model.
2.  **Prometheus Client:** Aggregates metrics in memory.
3.  **Prometheus Server:** Scrapes `/metrics` every 15s.
4.  **Grafana:** Visualizes the histograms.

## Production Engineering: Alerting Strategies

**The Boy Who Cried Wolf:** If you alert on every minor drift, engineers will ignore PagerDuty.

**Best Practices:**
1.  **Dynamic Thresholds:** Don't use `PSI > 0.2`. Use `PSI > MovingAverage(PSI) + 3*StdDev`.
2.  **Multi-Window Alerts:** Alert only if drift persists for 3 hours.
3.  **Severity Levels:**
    - **P1 (Wake up):** Model returns `NaN` or 500s.
    - **P2 (Next Morning):** PSI > 0.2.
    - **P3 (Weekly Review):** Feature importance shift.

## Data Quality: Great Expectations

Before checking for drift, check for **Data Quality**.
**Great Expectations (GX)** is a library for unit-testing data.

**Tests:**
- `expect_column_values_to_be_not_null("user_id")`
- `expect_column_values_to_be_between("age", 0, 120)`
- `expect_column_kl_divergence_to_be_less_than("income", 0.1)`

**Pipeline:**
`ETL -> GX Validation -> Training -> GX Validation -> Serving`

## Advanced Technique: Outlier Detection

Drift detects "Aggregate" shifts. But what if individual requests are weird?
**Example:** A user inputs `Age = 200`. The mean `Age` might still be 35, so Drift doesn't trigger. But the model will fail for that user.

**Solution: Isolation Forest.**
An unsupervised algorithm that isolates anomalies.
1.  Randomly select a feature.
2.  Randomly select a split value.
3.  Repeat.
4.  Anomalies are isolated quickly (short path length). Normal points require many splits (long path length).

**Implementation:**
```python
from sklearn.ensemble import IsolationForest

# Train on "Normal" data
clf = IsolationForest(random_state=0).fit(X_train)

# Predict on new data
# -1 = Outlier, 1 = Normal
preds = clf.predict(X_new)
```

## Deployment Strategies: A/B Testing vs. Shadow Mode

How do you deploy a new model safely?

### 1. Shadow Mode (Dark Launch)
- Deploy `Model B` alongside `Model A`.
- Route traffic to both.
- Return `Model A`'s prediction to the user.
- Log `Model B`'s prediction asynchronously.
- **Compare:** Does `Model B` crash? Is the distribution similar?

### 2. Canary Deployment
- Route 1% of traffic to `Model B`.
- Monitor error rates.
- Gradually ramp up: 10% -> 50% -> 100%.

### 3. A/B Testing (Online Experiment)
- Route 50% to A, 50% to B.
- Measure **Business Metrics** (Click-Through Rate, Conversion).
- **Goal:** Prove `Model B` is better, not just "working".

### 4. Interleaving (Ranking)
- Mix results from A and B in the same list.
- See which one the user clicks.
- **Pros:** Removes bias (users don't know which model produced which item).

## Incident Response Playbook

**Scenario:** PagerDuty fires at 3 AM. "Fraud Model Drift > 0.5".

**Step 1: Triage**
- Is the service down? (500 errors).
- Is it a data pipeline failure? (Check Airflow).
- Is it a real world shift? (Black Friday sale started).

**Step 2: Mitigation**
- **Rollback:** Revert to the previous model version immediately.
- **Fallback:** Switch to a heuristic (Rule-based engine).

**Step 3: Investigation**
- **Feature Attribution:** Which feature caused the drift?
- **Segment Analysis:** Is it only iOS users? Only users in Canada?

**Step 4: Resolution**
- **Retrain:** If the world changed, retrain on the new data.
- **Fix Pipeline:** If a feature broke (e.g., `Age` became `NaN`), fix the SQL query.

## Deep Dive: The "Feedback Loop" Problem

If you retrain on your own model's predictions, you create a **Feedback Loop**.
**Example:**
1.  Model recommends "Action Movies".
2.  User clicks "Action Movies" (because that's all they see).
3.  Model learns "User loves Action Movies".
4.  Model recommends *only* "Action Movies".

**Solution: Exploration.**
- **Epsilon-Greedy:** 5% of the time, show random recommendations.
- **Bandits:** Use Contextual Bandits to balance Exploration vs. Exploitation.
- **Positional Bias Correction:** Users click the top item because it's at the top. Log the *position* and use it as a feature (or debias the loss function).

## Appendix C: Tools Landscape

| Tool | Type | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Prometheus/Grafana** | Infrastructure | Free, Standard | Hard to do complex stats (KS-Test) |
| **Evidently AI** | Library | Great Reports, Open Source | Not a service (you host it) |
| **Arize AI** | SaaS | Full features, Embedding support | Expensive |
| **Fiddler** | SaaS | Explainability focus | Expensive |
| **WhyLabs** | SaaS | Privacy-preserving (logs sketches) | Requires `whylogs` integration |

## Advanced Topic: Monitoring Large Language Models (LLMs)

Monitoring tabular models (XGBoost) is solved. Monitoring LLMs (GPT-4) is the Wild West.

**New Metrics:**
1.  **Hallucination Rate:** Does the model invent facts?
    - *Detection:* Use a second LLM (Judge) to verify the answer against a Knowledge Base (RAG).
2.  **Prompt Injection:** Is the user trying to jailbreak the model?
    - *Detection:* Regex filters, Perplexity scores, or a specialized BERT classifier.
3.  **Token Usage:** Cost per request.
4.  **Toxicity:** Is the output offensive?

**The "LLM-as-a-Judge" Pattern:**
Use GPT-4 to monitor GPT-3.5.
`Score = GPT4.evaluate(User_Prompt, GPT3.5_Response)`
This is expensive but effective.

## Deep Dive: Feedback Loops in Recommender Systems

The "Feedback Loop" is a mathematical trap.
Let `P(click | item)` be the true probability.
The model estimates `P_hat(click | item)`.
The system shows items where `P_hat` is high.
The user clicks *only* what is shown.
The training data becomes biased towards high `P_hat` items.

**Mathematical Formulation:**
`Data_Train ~ P(User, Item) * P(Shown | User, Item)`
If `P(Shown)` depends on the model, the data is **Not IID** (Independent and Identically Distributed).

**Correction: Inverse Propensity Weighting (IPW).**
Weight each sample by `1 / P(Shown)`.
If an item had a 1% chance of being shown but was clicked, it's a *very* strong signal. Upweight it by 100x.

## System Design: Feature Store Integration

Drift often happens because the **Offline** features (in Data Warehouse) don't match the **Online** features (in Redis).
**Example:**
- Offline: `avg_clicks_7d` calculated at midnight.
- Online: `avg_clicks_7d` calculated in real-time.
- **Drift:** The online value is "fresher" and thus different.

**Solution: Feature Store (Feast / Tecton).**
A single definition of the feature logic.
`feature_view = avg_clicks.window(7 days)`
The Feature Store ensures that Offline (Training) and Online (Serving) values are consistent (Point-in-Time Correctness).

## FinOps: Cost Monitoring

ML is expensive. You need to monitor the **Cost per Prediction**.

**Metrics:**
1.  **GPU Utilization:** If it's 20%, you are wasting money. Scale down.
2.  **Spot Instance Availability:** If Spot price > On-Demand, switch.
3.  **Model Size vs. Value:** Does the 100B parameter model generate 10x more revenue than the 10B model? Usually no.

**Auto-Scaling Policies:**
- Scale based on **Queue Depth** (Lag), not CPU.
- If `Lag > 100ms`, add replicas.

## Regulatory Compliance: GDPR and "Right to Explanation"

**GDPR Article 22:** Users have the right not to be subject to a decision based solely on automated processing.
**Implication:** You must be able to **explain** why a user was rejected for a loan.

**Monitoring for Compliance:**
1.  **Audit Logs:** Immutable log of every decision + model version + input data.
2.  **Reproducibility:** Can you replay the request from 3 years ago and get the exact same result? (Requires versioning Code + Data + Model + Environment).

## Advanced Topic: Monitoring Vector Databases (RAG)

In RAG (Retrieval Augmented Generation), the "Model" is the LLM + Vector DB.
If the Vector DB returns bad context, the LLM hallucinates.

**What to Monitor:**
1.  **Recall@K:** Are we retrieving the right documents? (Requires Ground Truth).
2.  **Index Drift:** Does the HNSW graph structure degrade over time?
3.  **Embedding Distribution:** Use PCA to visualize if new documents are clustering in a new region (Topic Drift).
4.  **Latency:** Vector search can be slow. Monitor p99 latency.

## Deep Dive: Online Learning (Continual Learning)

Most models are "Static" (trained once).
**Online Learning** models update weights with every sample.
**Example:** Ad Click Prediction (FTRL - Follow The Regularized Leader).

**Monitoring Challenges:**
- **Catastrophic Forgetting:** The model learns the new trend but forgets the old one.
- **Feedback Loops:** Instant feedback amplifies bias.
- **Stability:** One bad batch of data can destroy the model weights.

**Safety:**
- **Holdout Set:** Keep a fixed "Golden Set" of data. Evaluate the model on it every minute. If accuracy drops, rollback.

If you found this helpful, consider sharing it with others who might benefit.


## Privacy: Differential Privacy in Monitoring

You want to know the distribution of "Income", but you can't log individual incomes.
**Solution: Local Differential Privacy (LDP).**
Add noise to the data *on the device* before sending it to the server.
`Value = True_Value + Laplace_Noise`

**RAPPOR (Google):**
Randomized Aggregatable Privacy-Preserving Ordinal Response.
Allows monitoring frequency of strings (e.g., URLs) without knowing who visited them.

## Engineering: Grafana Dashboard as Code

Don't click around in the UI. Version control your dashboards.
**Jsonnet** is the standard for generating Grafana JSON.

```jsonnet
local grafana = import 'grafonnet/grafana.libsonnet';
local dashboard = grafana.dashboard;
local graph = grafana.graphPanel;
local prometheus = grafana.prometheus;

dashboard.new(
  'Model Health',
  schemaVersion=16,
)
.addPanel(
  graph.new(
    'Prediction Drift (PSI)',
    datasource='Prometheus',
  )
  .addTarget(
    prometheus.target(
      'model_drift_score',
    )
  )
)
```

## Appendix E: The Human-in-the-Loop

Sometimes, the monitor shouldn't just alert; it should **escalate**.
**Active Learning:**
1.  Model is unsure (Confidence < 0.6).
2.  Route request to a Human Reviewer.
3.  Human labels it.
4.  Add to training set.
5.  Retrain.

This turns "Low Confidence" from a failure into a feature.

## Appendix F: Interview Questions

1.  **Q:** "How do you monitor a model that runs on a mobile device (Edge)?"
    **A:** You can't send all data to the cloud (bandwidth). Use **Federated Monitoring**. Calculate histograms locally on the device, send only the histogram (small JSON) to the server for aggregation.

2.  **Q:** "What is the difference between Model Performance and Business Performance?"
    **A:**
    - **Model:** AUC, Accuracy, LogLoss.
    - **Business:** Revenue, Churn, CTR.
    - *Crucial:* Good Model Performance != Good Business Performance. (e.g., A clickbait model has high CTR but increases Churn).

3.  **Q:** "How do you distinguish between Data Drift and a Bug?"
    **A:**
    - **Bug:** Sudden step-change in metrics (usually after a deployment).
    - **Drift:** Gradual trend over days/weeks.
    - *Check:* Did we deploy code yesterday? Did the upstream data schema change?

## Conclusion

Monitoring is the difference between a "Science Project" and a "Product".
A model is a living organism. It eats data. If the food spoils, the organism gets sick. Your job is to be the doctor.
