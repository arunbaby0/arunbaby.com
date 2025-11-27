---
title: "Batch Processing Pipelines"
day: 26
collection: ml_system_design
categories:
  - ml-system-design
  - data-engineering
tags:
  - airflow
  - spark
  - mapreduce
  - etl
  - data-lake
subdomain: "Data Engineering"
tech_stack: [Apache Airflow, Apache Spark, Hadoop, AWS Glue]
scale: "Petabytes of data"
companies: [Airbnb, Netflix, Uber, Databricks]
related_dsa_day: 26
related_ml_day: 26
related_speech_day: 26
---

**Not everything needs to be real-time. Sometimes, "tomorrow morning" is fast enough.**

## The Case for Batch

In the age of Real-Time Streaming (Kafka, Flink), Batch Processing feels archaic.
But 90% of ML workloads are still Batch.
- **Training:** You train on a *dataset* (Batch), not a stream.
- **Reporting:** "Daily Active Users" is a batch metric.
- **Backfilling:** Re-computing history requires batch.

**Batch vs. Stream:**
- **Batch:** High Throughput, High Latency. (Process 1TB in 1 hour).
- **Stream:** Low Throughput, Low Latency. (Process 1 event in 10ms).

## Architecture: The Modern Data Stack

1.  **Ingestion:** Fivetran / Airbyte. Pull data from Postgres/Salesforce into the Warehouse.
2.  **Storage:** Data Lake (S3/GCS) or Data Warehouse (Snowflake/BigQuery).
3.  **Transformation:** dbt (SQL) or Spark (Python).
4.  **Orchestration:** Airflow / Dagster / Prefect.

## High-Level Architecture: The Modern Data Stack

```ascii
+-----------+     +------------+     +-------------+     +-------------+
|  Sources  | --> | Ingestion  | --> |  Data Lake  | --> |  Warehouse  |
+-----------+     +------------+     +-------------+     +-------------+
(Postgres)        (Fivetran)         (S3 / GCS)          (Snowflake)
                                                               |
                                                               v
+-----------+     +------------+     +-------------+     +-------------+
| Dashboard | <-- | Serving    | <-- | Transform   | <-- | Orchestrator|
+-----------+     +------------+     +-------------+     +-------------+
(Tableau)         (Redis/API)        (dbt / Spark)       (Airflow)
```

## Deep Dive: Apache Airflow (Orchestration)

Airflow allows you to define pipelines as code (Python).
**DAG (Directed Acyclic Graph):** A collection of tasks with dependencies.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def extract():
    print("Extracting data from S3...")

def transform():
    print("Running Spark Job...")

def load():
    print("Loading into Feature Store...")

with DAG("daily_training_pipeline", start_date=datetime(2023, 1, 1)) as dag:
    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="transform", python_callable=transform)
    t3 = PythonOperator(task_id="load", python_callable=load)

    t1 >> t2 >> t3 # Define dependencies
```

**Key Concepts:**
- **Scheduler:** Monitors time and triggers DAGs.
- **Executor:** Runs the tasks (Local, Celery, Kubernetes).
- **Backfill:** Rerunning the DAG for past dates (e.g., "Run for all of 2022").

## Deep Dive: Apache Spark (Processing)

When data doesn't fit in RAM (Pandas), you need Spark.
Spark is a **Distributed Computing Engine**.

**RDD (Resilient Distributed Dataset):**
- **Distributed:** Data is split into partitions across nodes.
- **Resilient:** If a node fails, Spark rebuilds the partition using the lineage graph.
- **Lazy Evaluation:** Transformations (`map`, `filter`) are not executed until an Action (`count`, `save`) is called.

**PySpark Example:**

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FeatureEng").getOrCreate()

# Read 1TB of logs
df = spark.read.json("s3://bucket/logs/*.json")

# Group By User and Count Clicks
user_features = df.groupBy("user_id").count()

# Write to Parquet
user_features.write.parquet("s3://bucket/features/")
```

## System Design: Retraining Pipeline

**Scenario:** Design a system to retrain the Recommendation Model every week.

**Components:**
1.  **Trigger:** Airflow DAG runs every Sunday at 00:00.
2.  **Data Prep:** Spark job reads last 30 days of clicks from Data Lake. Joins with User Table. Saves `training_data.parquet`.
3.  **Validation:** Great Expectations checks for nulls/outliers.
4.  **Training:** SageMaker Training Job launches a GPU instance. Loads parquet. Trains XGBoost. Saves `model.tar.gz`.
5.  **Evaluation:** Load model. Predict on Holdout Set. If `AUC > 0.8`, tag as `Production`.
6.  **Deployment:** Update the SageMaker Endpoint to point to the new model artifact.

## Engineering: The "Small Files" Problem

A common mistake in Data Lakes: Saving millions of tiny files (1KB each).
**Why is it bad?**
- **S3 API Costs:** You pay per PUT/GET request.
- **Spark Slowness:** Listing millions of files takes forever. Opening a file has overhead.

**Solution:** Compaction.
Run a nightly job to merge small files into larger files (128MB - 1GB).
`df.repartition(10).write...`

## Deep Dive: Lambda vs. Kappa Architecture

How do you combine Batch (Accuracy) and Stream (Speed)?

**1. Lambda Architecture (The Old Way):**
- **Speed Layer:** Kafka + Flink. Provides low-latency, approximate results.
- **Batch Layer:** Hadoop/Spark. Provides high-latency, accurate results.
- **Serving Layer:** Merges the two.
- **Pros:** Robust. If Stream fails, Batch fixes it.
- **Cons:** **Maintenance Nightmare.** You write logic twice (Java for Flink, Python for Spark).

**2. Kappa Architecture (The New Way):**
- **Everything is a Stream.**
- Batch is just a stream of bounded data.
- Use **Flink** for both.
- **Pros:** Single codebase.
- **Cons:** Reprocessing history is harder (requires replaying the Kafka topic).

## Engineering: File Formats (Parquet vs. Avro vs. ORC)

CSV/JSON are terrible for Big Data (Slow parsing, no schema, large size).

**1. Parquet (Columnar):**
- **Best for:** Analytics (OLAP). "Select average(age) from users".
- **Why:** It only reads the "age" column from disk. Skips the rest.
- **Compression:** Snappy/Gzip. Very efficient.

**2. Avro (Row-based):**
- **Best for:** Write-heavy workloads, Kafka messages.
- **Why:** Schema evolution is first-class. Good for appending data.

**3. ORC (Optimized Row Columnar):**
- **Best for:** Hive/Presto. Similar to Parquet but optimized for the Hadoop ecosystem.

## Deep Dive: Partitioning and Bucketing

How do you make `SELECT * FROM logs WHERE date = '2023-01-01'` fast?

**1. Partitioning:**
- Organize folders by key.
- `s3://bucket/logs/date=2023-01-01/part-001.parquet`
- Spark automatically "prunes" partitions. It only scans the relevant folder.
- **Warning:** Don't partition by high-cardinality columns (UserID). You'll get millions of tiny folders.

**2. Bucketing:**
- Hash the key to `N` buckets.
- `hash(user_id) % 100`.
- Useful for **Joins**. If two tables are bucketed by `user_id`, the join is a "Sort-Merge Join" (no shuffle needed).

## System Design: Handling Late Data

In Batch, "Daily" doesn't mean "Midnight to Midnight".
Data arrives late (mobile device offline).

**Strategies:**
1.  **Watermark:** Wait for X hours (e.g., process "Yesterday" at 2 AM today).
2.  **Lookback:** When processing "Today", also re-process "Yesterday" to catch late arrivals.
3.  **Delta Lake / Hudi:** These "Lakehouse" formats allow **Upserts**. You can update yesterday's partition without rewriting the whole table.

## Deep Dive: Idempotency

**Definition:** `f(f(x)) = f(x)`. Running the job twice produces the same result.
**Why:** Airflow *will* retry your job if it fails.

**Anti-Pattern:**
`INSERT INTO table SELECT ...`
(Running twice duplicates data).

**Pattern:**
`INSERT OVERWRITE table PARTITION (date='2023-01-01') SELECT ...`
(Running twice overwrites the partition).

## Engineering: Data Quality with Great Expectations

**The Nightmare:**
The upstream team changes `age` from "Years" (Int) to "Birthdate" (String).
Your Spark job crashes at 3 AM.

**Solution: Circuit Breakers.**
Use **Great Expectations** to validate data *before* processing.

```python
import great_expectations as ge

df = ge.read_parquet("s3://bucket/input/")

# Define expectations
df.expect_column_values_to_be_not_null("user_id")
df.expect_column_values_to_be_between("age", 0, 120)
df.expect_table_row_count_to_be_between(10000, 100000)

# Validate
results = df.validate()
if not results["success"]:
    raise ValueError("Data Quality Check Failed!")
```

## Appendix B: Workflow Orchestration Wars

**Airflow:**
- **Pros:** Industry standard, huge community, Python.
- **Cons:** Scheduling latency, complex setup, "The Scheduler Loop".

**Prefect:**
- **Pros:** "Negative Engineering" (handles retries/failures elegantly), hybrid execution model.
- **Cons:** Smaller ecosystem.

**Dagster:**
- **Pros:** Data-aware (knows about assets, not just tasks), strong typing.
- **Cons:** Steep learning curve.

## Appendix C: Interview Questions

1.  **Q:** "What is the difference between Transformation and Action in Spark?"
    **A:** Transformations (Map, Filter) are lazy (build the DAG). Actions (Count, Collect) trigger execution.

2.  **Q:** "How do you handle Skewed Data in a Join?"
    **A:**
    - **Salting:** Add a random number (salt) to the skewed key to split it.
    - **Broadcast Join:** If one table is small, broadcast it to all nodes.

3.  **Q:** "Explain the difference between Data Warehouse and Data Lake."
    **A:**
    - **Warehouse (Snowflake):** Structured, Schema-on-Write, SQL, Expensive.
    - **Lake (S3):** Unstructured/Semi-structured, Schema-on-Read, Files, Cheap.
    - **Lakehouse:** Best of both (ACID on S3).

## Deep Dive: Spark Catalyst Optimizer

Why is Spark SQL faster than raw RDDs? **Catalyst.**
It's an extensible query optimizer.

**Phases:**
1.  **Analysis:** Resolve column names (`SELECT name FROM users`).
2.  **Logical Optimization:**
    - **Predicate Pushdown:** Move `FILTER` before `JOIN`.
    - **Column Pruning:** Read only used columns.
3.  **Physical Planning:** Choose join strategy (Broadcast vs. Sort-Merge).
4.  **Code Generation:** Generate Java bytecode on the fly (Whole-Stage Code Gen).

## Deep Dive: The Shuffle (Timsort)

The bottleneck of any distributed system is the **Shuffle**.
Moving data from Mapper to Reducer over the network.

**Sort-Based Shuffle:**
Spark sorts data on the Mapper side before sending it.
It uses **Timsort** (Hybrid of Merge Sort and Insertion Sort).
- **Complexity:** \(O(N \log N)\).
- **Memory:** Efficient. Spills to disk if RAM is full.

**Tuning:**
`spark.sql.shuffle.partitions` (Default 200).
- Too low: OOM (Out of Memory).
- Too high: Too many small files/tasks.

## System Design: Data Lineage (OpenLineage)

**Problem:** "The revenue number is wrong. Where did it come from?"
**Solution:** Data Lineage.

**OpenLineage:** Standard spec for lineage.
- **Job:** "Daily Revenue ETL".
- **Input:** `s3://bucket/orders`.
- **Output:** `s3://bucket/revenue`.

**Marquez:** An Open Source lineage server.
It visualizes the graph: `Postgres -> Spark -> S3 -> Snowflake -> Tableau`.
If the dashboard breaks, you trace it back to the source.

## Engineering: CI/CD for Data Pipelines

Software Engineers have CI/CD. Data Engineers usually test in production. **Don't.**

**Pipeline:**
1.  **Unit Test:** Test individual Python functions (PyTest).
2.  **Integration Test:** Run the DAG on a small sample dataset (Dockerized Airflow).
3.  **Staging:** Deploy to Staging environment. Run on full data (or 10%).
4.  **Production:** Deploy.

**Tools:**
- **DataOps:** The philosophy of applying DevOps to Data.
- **dbt test:** Validates SQL logic.

## FinOps: Autoscaling Strategies

Batch jobs are bursty. You need 100 nodes for 1 hour, then 0.

**1. Cluster Autoscaling:**
- If pending tasks > 0, add nodes.
- If CPU utilization < 50%, remove nodes.

**2. Spot Instances:**
- Use AWS Spot Instances (90% cheaper).
- **Risk:** AWS can reclaim them with 2-minute warning.
- **Mitigation:** Spark is fault-tolerant. If a node dies, the driver reschedules the task on another node.

## Case Study: Netflix Data Mesh

Netflix moved from a Monolithic Data Lake to a **Data Mesh**.
**Principles:**
1.  **Domain-Oriented Ownership:** The "Content Team" owns the "Content Data Product".
2.  **Data as a Product:** Data must have SLAs, Documentation, and Quality Checks.
3.  **Self-Serve Infrastructure:** Platform team provides the tools (Spark/Airflow), Domain teams build the pipelines.
4.  **Federated Governance:** Global standards (GDPR), local implementation.

## Appendix D: The "Small Files" Problem (Revisited)

**Compaction Strategies:**
1.  **Coalesce:** `df.coalesce(10)`. Moves data to fewer partitions. No shuffle.
2.  **Repartition:** `df.repartition(10)`. Full shuffle. Balances data perfectly.
3.  **Bin-Packing:** Combine small files into a single task during reading (`spark.sql.files.maxPartitionBytes`).

## Appendix E: Advanced Interview Questions

1.  **Q:** "What is the difference between `repartition()` and `coalesce()`?"
    **A:** `repartition` does a full shuffle (network I/O). `coalesce` just merges local partitions (no shuffle). Use `coalesce` to reduce file count.

2.  **Q:** "How do you handle a Hot Key in a Join?"
    **A:** If "Justin Bieber" has 100M clicks, the node processing him will OOM.
    - **Solution:** Salt the key. `key = key + random(1, 100)`. Explode the other table 100 times.

3.  **Q:** "What is a Broadcast Variable?"
    **A:** A read-only variable cached on every machine. Used to send a small lookup table (Country Codes) to all workers to avoid a Shuffle Join.

## Deep Dive: Bloom Filters (Probabilistic Data Structures)

**Problem:** You have 1 Billion URLs. You want to check if a new URL is already in the set.
**Naive:** Store all URLs in a HashSet. (Requires 100GB RAM).
**Solution:** Bloom Filter. (Requires 1GB RAM).

**Mechanism:**
1.  Bit array of size `M`.
2.  `K` hash functions.
3.  **Add(item):** Hash item `K` times. Set bits at those indices to 1.
4.  **Check(item):** Hash item `K` times. If all bits are 1, return "Maybe Present". If any bit is 0, return "Definitely Not Present".

**False Positives:** Possible.
**False Negatives:** Impossible.

## Deep Dive: HyperLogLog (Cardinality Estimation)

**Problem:** Count unique visitors (DAU) for Facebook (2 Billion users).
**Naive:** `SELECT COUNT(DISTINCT user_id)`. Requires storing all IDs. Slow.
**Solution:** HyperLogLog (HLL).

**Mechanism:**
1.  Hash the user ID.
2.  Count the number of leading zeros in the binary hash.
3.  If you see a hash with 10 leading zeros, you probably saw `2^10` items.
4.  Average this across many buckets (Harmonic Mean).

**Accuracy:** 99% accuracy using only 1.5KB of memory.

## Deep Dive: Count-Min Sketch (Frequency Estimation)

**Problem:** Find the "Top 10" most popular songs.
**Solution:** Count-Min Sketch.

**Mechanism:**
1.  2D Array `[Depth][Width]`.
2.  **Add(item):** Hash item `Depth` times. Increment the counter at `[d][hash(item)]`.
3.  **Query(item):** Return `min(counters)` for that item.

**Why Min?** Because collisions only *increase* the count. The minimum is the closest to the truth.

## Engineering: Handling PII with Hashing/Salting

**Requirement:** GDPR "Right to be Forgotten".
**Problem:** If you delete a user from the DB, their ID is still in the logs/backups.

**Solution: Crypto-Shredding.**
1.  Don't store `user_id`. Store `HMAC(user_id, key)`.
2.  Store the `key` in a separate Key Management Service (KMS).
3.  To "forget" a user, delete their `key`.
4.  Now the logs contain garbage that can never be decrypted.

## Case Study: Uber's Michelangelo (Batch Training)

Uber built an internal ML-as-a-Service platform.
**Workflow:**
1.  **Feature Store:** Hive tables containing pre-computed features (`avg_ride_cost_7d`).
2.  **Selection:** Data Scientist selects features in UI.
3.  **Join:** Spark job joins features with labels (Point-in-Time correct).
4.  **Train:** Distributed XGBoost / Horovod (Deep Learning).
5.  **Model Store:** Versioned artifact saved to S3.
6.  **Serving:** Model deployed to a Docker container.

## Appendix F: The "Thundering Herd" Problem

**Scenario:** 10,000 Airflow tasks are scheduled for 00:00.
**Result:** The Scheduler crashes. The Database CPU spikes to 100%.

**Solution:**
1.  **Jitter:** Add a random delay (0-60s) to the start time.
2.  **Pools:** Limit concurrency. `pool='heavy_sql', slots=10`.
3.  **Sensor Deferral:** Use `SmartSensor` (Async) instead of blocking threads.

## Conclusion

Batch processing is the workhorse of ML.
While Real-Time is sexy, Batch is reliable, replayable, and easy to debug.
*If you found this helpful, consider sharing it with others who might benefit.*


