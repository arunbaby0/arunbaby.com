---
title: "DAG Pipeline Orchestration"
day: 49
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - airflow
  - kubeflow
  - mlops
  - dag
  - orchestration
difficulty: Hard
subdomain: "MLOps Architecture"
tech_stack: Apache Airflow, Kubeflow, Argo Workflows
scale: "Orchestrating 10,000 tasks daily"
companies: Airbnb (Airflow), Google (Kubeflow), Lyft (Flyte)
related_dsa_day: 49
related_ml_day: 49
related_speech_day: 49
related_agents_day: 49
---

**"Cron is not an orchestrator. A script is not a pipeline."**

## 1. Problem Statement

Standard software runs via `Request -> Response`.
ML software runs as a **Pipeline**.
-   Step 1: Ingest Data (Wait for Hive partition).
-   Step 2: Clean Data (Requires Step 1).
-   Step 3: Distributed Training (Requires Step 2, Requires GPU cluster).
-   Step 4: Validation (Requires Step 3).
-   Step 5: Deploy.

**The Problem**: How do you manage this dependency graph reliably, handling retries, backfills, and distributed execution, without writing spaghetti bash scripts?

---

## 2. Understanding the Requirements

### 2.1 Why `cron` fails
`0 2 * * * train_model.sh`.
1.  **Dependency Hell**: What if the Data Ingest job (scheduled at 1:00 AM) is delayed? `train_model.sh` runs on empty data.
2.  **No State**: If Step 3 fails, `cron` doesn't know. It won't retry just Step 3. You have to re-run the whole thing.
3.  **No Scaling**: `cron` runs on one machine. We need to run Step 3 on a Kubernetes GPU cluster.

### 2.2 The Solution: The DAG
We represent the workflow as a **Directed Acyclic Graph (DAG)**.
-   **Directed**: Data flows `A -> B`.
-   **Acyclic**: B cannot feed back into A (infinite loop).
-   **Graph**: Nodes are tasks, Edges are data/control dependencies.

---

## 3. High-Level Architecture

An MLOps Orchestrator (like Airflow or Kubeflow Pipelines) consists of:

```
[Scheduler (The Brain)]
   |  (Polls DAGs, Checks Time, Checks Dependencies)
   |
   +---> [Queue (Redis/RabbitMQ)]
            |
            v
   [Workers (The Brawn)]
      | Worker 1: [Task A: Data Prep]
      | Worker 2: [Task B: Training (GPU)]
      | Worker 3: [Task C: Evaluation]
```

### Key Components
1.  **Metastore (Postgres)**: "Task B failed at 2:03 PM".
2.  **Executor**: "Launch this task on Kubernetes" vs "Launch this task in a local process".

---

## 4. Component Deep-Dives

### 4.1 Airflow (The Scheduler King)
-   **Python as Config**: You define DAGs in Python code.
-   **Operators**: `PostgresOperator`, `PythonOperator`, `BashOperator`.
-   **Sensors**: Special tasks that "wait" for something (e.g., `S3KeySensor` waits for a file to appear).
-   **Best For**: Data Engineering, Scheduled jobs.

### 4.2 Kubeflow Pipelines (The ML Specialist)
-   **Container Native**: Every task is a Docker container.
-   **Artifact Tracking**: Automatically logs "Task A produced `dataset.csv` (hash: xyz)".
-   **Best For**: Deep Learning workflows where reproducibility (Docker) is critical.

---

## 5. Data Flow: The Backfill

A unique feature of Orchestrators is **Backfilling**.
Imagine you change your Feature Engineering logic today (Day 50).
You want to re-train the model on data from Day 1 to Day 49 using the *new* logic.

With `cron`, this is a nightmare.
With Airflow:
1.  Clear the state of "FeatureEng Task" for `dates=[Day 1...Day 49]`.
2.  The Scheduler sees these tasks as "Null state".
3.  Since they have no dependencies blocking them (historical data exists), it schedules them all in parallel (up to `max_active_runs`).

This turns "Historical Reprocessing" into a one-click operation.

---

## 6. Scaling Strategies

### 6.1 The Kubernetes Executor
Instead of having a fixed pool of workers (which sit idle or get overwhelmed), use the **KubernetesExecutor**.
-   Scheduler: "Task A needs to run."
-   K8s: Spin up a Pod just for Task A.
-   Task A: Runs.
-   K8s: Kill Pod.
**Pros**: Infinite scale. Zero cost when idle. Perfect isolation (Task A uses PyTorch 1.9, Task B uses PyTorch 2.0).

### 6.2 Caching (Memoization)
If Task A takes 5 hours to generate `clean_data.csv`, and Task B fails...
When we retry, we *don't* want to re-run Task A.
Kubeflow natively checks inputs. If `Hash(Inputs, Code)` matches a previous run, it reuses the cached output.

---

## 7. Implementation: defining a DAG (Airflow)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.s3 import S3KeySensor
from datetime import datetime

def train_model(**context):
    # ML Logic here
    print("Training...")

with DAG("ml_pipeline_v1", 
         start_date=datetime(2023, 1, 1), 
         schedule_interval="@daily",
         catchup=False) as dag:

    # 1. Wait for data to arrive in S3
    wait_for_data = S3KeySensor(
        task_id="wait_for_input",
        bucket_key="s3://data/incoming/{{ ds }}/data.csv"
    )

    # 2. Train logic
    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    # 3. Dependency Definition
    wait_for_data >> train
```

---

## 8. Monitoring & Metrics

1.  **DAG Parse Time**: If your Python DAG file is slow to parse (e.g., connects to DB at top level), the Scheduler hangs. Anti-pattern!
2.  **Task Latency**: Track `Input_Size` vs `Duration`. If data grew 2x but time grew 10x, you have a non-linear scaling bug.

---

## 9. Failure Modes

1.  **Thundering Herd**: You backfill 1 year of data. 365 jobs launch instantly. Your Database crashes.
    -   *Fix*: Set `concurrency=10` at the DAG level.
2.  **Deadlock**: Task A waits for Task B. Task B waits for connection slot held by Task A.
3.  **Zombies**: The Scheduler thinks Task A is running. Task A's pod died OOM. The state stays "Running" forever.
    -   *Fix*: Frequent "Heartbeat" checks by the Scheduler.

---

## 10. Real-World Case Study: Lyft Flyte

Lyft built **Flyte** because Airflow wasn't strict enough about data types.
Flyte enforces **Type Safety** between tasks.
-   Task A output: `dataframe_schema[age: int]`
-   Task B input: `dataframe_schema[age: String]`
-   Flyte catches this mismatch at *Compile Link Time*, preventing runtime failures 5 hours into the pipeline.

---

## 11. Cost Analysis

-   **Spot Instances**: Since ML pipelines are fault-tolerant (auto-retry), use Spot Instances for the Heavy Workers. Savings: 70%.
-   **Idle Clusters**: Using K8s Executor prevents paying for idle workers at night.

---

## 12. Key Takeaways

1.  **Dependency Management**: This is the core value prop. "Run B only after A succeeds."
2.  **Idempotency**: Every task must be idempotent. If run twice, it should produce the same result (or overwrite safely).
3.  **Infrastructure as Code**: The Pipeline *is* code. Version control your DAGs.
4.  **Static vs Dynamic**: Airflow is Static (DAG defined upfront). Prefect/Metaflow allow Dynamic (DAG defined at runtime).

---

**Originally published at:** [arunbaby.com/ml-system-design/0049-dag-pipeline-orchestration](https://www.arunbaby.com/ml-system-design/0049-dag-pipeline-orchestration/)

*If you found this helpful, consider sharing it with others who might benefit.*
