---
title: "Experiment Tracking Systems"
day: 19
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - experiment-tracking
  - mlops
  - metadata
  - versioning
  - reproducibility
  - infrastructure
subdomain: "MLOps & Infrastructure"
tech_stack: [MLflow, Weights & Biases, Neptune, TensorBoard, PostgreSQL, S3, Git, Docker]
scale: "10K+ experiments, 1M+ runs, PB-scale artifacts"
companies: [Google, Meta, OpenAI, Microsoft, Amazon, Uber, Airbnb]
related_dsa_day: 19
related_speech_day: 19
related_agents_day: 19
---

**Design robust experiment tracking systems that enable systematic exploration, reproducibility, and collaboration across large ML teams.**

## Problem Statement

Design an **Experiment Tracking System** for ML teams that:

1. **Tracks all experiment metadata**: hyperparameters, metrics, code versions, data versions, artifacts
2. **Supports large scale**: Thousands of experiments, millions of runs, petabyte-scale model artifacts
3. **Enables comparison and visualization**: Compare runs, plot learning curves, analyze hyperparameter impact
4. **Ensures reproducibility**: Any experiment can be re-run from tracked metadata
5. **Integrates with training pipelines**: Minimal code changes, automatic logging
6. **Supports collaboration**: Share experiments, notebook integration, API access

### Functional Requirements

1. **Experiment lifecycle management:**
   - Create experiments and runs
   - Log parameters, metrics, tags, notes
   - Upload artifacts (models, plots, datasets)
   - Track code versions (Git commit, diff)
   - Track data versions (dataset hashes, splits)
   - Link parent/child runs (hyperparameter sweeps, ensemble members)

2. **Query and search:**
   - Filter by parameters, metrics, tags
   - Full-text search over notes and descriptions
   - Query by date, user, project

3. **Visualization and comparison:**
   - Learning curves (metric vs step/epoch)
   - Hyperparameter sweeps (parallel coordinates, scatter)
   - Compare multiple runs side-by-side
   - Export to notebooks (Jupyter, Colab)

4. **Artifact management:**
   - Store and version models, checkpoints, plots
   - Efficient storage for large artifacts (deduplication, compression)
   - Support for streaming logs (real-time metrics)

5. **Reproducibility:**
   - Capture full environment (packages, hardware, Docker image)
   - Re-run experiments from tracked metadata
   - Audit trail for compliance

6. **Integration:**
   - Python SDK (PyTorch, TensorFlow, JAX)
   - CLI for automation
   - REST API for custom clients
   - Webhook/notification support

### Non-Functional Requirements

1. **Scalability:** Support 10K+ concurrent experiments, 1M+ total runs
2. **Performance:** Log metrics with <10ms latency, query results in <1s
3. **Reliability:** 99.9% uptime, no data loss
4. **Security:** Role-based access control, encryption at rest and in transit
5. **Cost efficiency:** Optimize storage costs for artifacts (tiered storage, compression)

## Understanding the Requirements

### Why Experiment Tracking Matters

Without systematic tracking, ML teams face:

- **Lost experiments:** "Which hyperparameters gave us 92% accuracy last month?"
- **Wasted compute:** Re-running experiments accidentally
- **Non-reproducibility:** "It worked on my laptop, but we can't reproduce it"
- **Collaboration friction:** Hard to share and compare results

A good experiment tracking system is the **foundation of MLOps**—it enables:

- Systematic exploration of model/data/hyperparameter spaces
- Clear audit trails for model governance
- Faster iteration through better visibility

### The Systematic Iteration Connection

Just like **Spiral Matrix** systematically traverses a 2D structure layer-by-layer:

- **Experiment tracking** systematically explores multi-dimensional spaces:
  - Hyperparameters × architectures × data configurations × training schedules
- Both require **clear state management**:
  - Spiral: track boundaries (top, bottom, left, right)
  - Experiments: track runs (completed, running, failed), checkpoints, metrics
- Both enable **resumability**:
  - Spiral: can pause and resume traversal
  - Experiments: can restart from checkpoints, resume sweeps

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Experiment Tracking System                      │
└─────────────────────────────────────────────────────────────────┘

                            Client Layer
        ┌────────────────────────────────────────────┐
        │  Python SDK  │  CLI  │  Web UI  │  API    │
        └─────────────────────┬──────────────────────┘
                              │
                       API Gateway
                ┌──────────────┴──────────────┐
                │  - Auth & rate limiting      │
                │  - Request routing           │
                │  - Logging & monitoring      │
                └──────────────┬──────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
      ┌───────▼────────┐ ┌────▼─────┐ ┌───────▼────────┐
      │  Metadata      │ │  Metrics │ │  Artifact      │
      │  Service       │ │  Service │ │  Service       │
      │                │ │          │ │                │
      │ - Experiments  │ │ - Logs   │ │ - Models       │
      │ - Runs         │ │ - Curves │ │ - Plots        │
      │ - Parameters   │ │ - Scalars│ │ - Datasets     │
      │ - Tags         │ │ - Hists  │ │ - Checkpoints  │
      └───────┬────────┘ └────┬─────┘ └───────┬────────┘
              │               │                │
      ┌───────▼────────┐ ┌────▼─────┐ ┌───────▼────────┐
      │  Postgres /    │ │  TimeSeries│ │  Object Store │
      │  MySQL         │ │  DB        │ │  (S3/GCS)     │
      │                │ │  (InfluxDB)│ │                │
      │ - Structured   │ │ - Metrics  │ │ - Large files │
      │   metadata     │ │ - Fast     │ │ - Versioned   │
      └────────────────┘ │   queries  │ │ - Deduped     │
                         └────────────┘ └────────────────┘
```

### Key Components

1. **Metadata Service:**
   - Stores experiment and run metadata (params, tags, code versions, user, etc.)
   - Relational DB for structured queries
   - Indexes on common query patterns (user, project, date, tags)

2. **Metrics Service:**
   - High-throughput metric logging (train loss, val accuracy, etc.)
   - Time-series database (InfluxDB, Prometheus, or custom)
   - Support for streaming metrics (real-time plots)

3. **Artifact Service:**
   - Stores large files (models, checkpoints, plots, datasets)
   - Object storage (S3, GCS, Azure Blob)
   - Deduplication (hash-based), compression, tiered storage

4. **API Gateway:**
   - Authentication & authorization (OAuth, API keys)
   - Rate limiting (per user/project)
   - Request routing and load balancing

5. **Web UI:**
   - Dashboards for experiments, runs, metrics
   - Comparison tools (side-by-side, parallel coordinates)
   - Notebook integration (export to Jupyter)

## Component Deep-Dive

### 1. Metadata Schema

**Experiments** group related runs (e.g., "ResNet ablation study").

**Runs** are individual training jobs with:

- Unique run ID
- Parameters (hyperparameters, model config)
- Metrics (logged scalars, step-indexed)
- Tags (labels for filtering)
- Code version (Git commit, diff)
- Data version (dataset hash, split config)
- Environment (Python packages, Docker image, hardware)
- Artifacts (model files, plots, logs)
- Status (running, completed, failed)
- Timestamps (start, end)

**Schema example (simplified):**

```sql
CREATE TABLE experiments (
    experiment_id UUID PRIMARY KEY,
    name VARCHAR(255),
    description TEXT,
    created_at TIMESTAMP,
    user_id VARCHAR(255),
    project_id UUID
);

CREATE TABLE runs (
    run_id UUID PRIMARY KEY,
    experiment_id UUID REFERENCES experiments(experiment_id),
    name VARCHAR(255),
    status VARCHAR(50),  -- running, completed, failed
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    user_id VARCHAR(255),
    git_commit VARCHAR(40),
    git_diff TEXT,
    docker_image VARCHAR(255),
    environment JSONB,  -- packages, hardware
    notes TEXT
);

CREATE TABLE run_params (
    run_id UUID REFERENCES runs(run_id),
    key VARCHAR(255),
    value TEXT,  -- JSON serialized
    PRIMARY KEY (run_id, key)
);

CREATE TABLE run_metrics (
    run_id UUID REFERENCES runs(run_id),
    key VARCHAR(255),  -- e.g., 'train_loss', 'val_accuracy'
    step INT,
    value FLOAT,
    timestamp TIMESTAMP,
    PRIMARY KEY (run_id, key, step)
);

CREATE TABLE run_tags (
    run_id UUID REFERENCES runs(run_id),
    key VARCHAR(255),
    value VARCHAR(255),
    PRIMARY KEY (run_id, key)
);

CREATE TABLE run_artifacts (
    artifact_id UUID PRIMARY KEY,
    run_id UUID REFERENCES runs(run_id),
    path VARCHAR(1024),  -- e.g., 'model.pt', 'plots/loss.png'
    size_bytes BIGINT,
    content_hash VARCHAR(64),  -- SHA-256
    storage_uri TEXT,  -- S3 URI
    created_at TIMESTAMP
);
```

### 2. Python SDK (Client Interface)

```python
import experiment_tracker as et

# Initialize client
client = et.Client(api_url="https://tracking.example.com", api_key="...")

# Create experiment
experiment = client.create_experiment(
    name="ResNet50 ImageNet Ablation",
    description="Testing different optimizers and learning rates"
)

# Start a run
run = experiment.start_run(
    name="run_adam_lr0.001",
    tags={"optimizer": "adam", "dataset": "imagenet"}
)

# Log parameters
run.log_params({
    "model": "resnet50",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 90
})

# Training loop
for epoch in range(90):
    train_loss = train_one_epoch(model, optimizer, train_loader)
    val_acc = validate(model, val_loader)
    
    # Log metrics
    run.log_metrics({
        "train_loss": train_loss,
        "val_accuracy": val_acc
    }, step=epoch)

# Save model
run.log_artifact("model.pt", local_path="./checkpoints/model_epoch90.pt")

# Mark run as complete
run.finish()
```

### 3. Metric Logging & Streaming

For real-time metric visualization:

- Clients send metrics via WebSocket or HTTP streaming
- Metrics Service buffers and batches writes to time-series DB
- Web UI subscribes to metric streams for live plots

```python
# Streaming metrics example
def train_with_streaming_metrics(model, run):
    for step, batch in enumerate(train_loader):
        loss = train_step(model, batch)
        
        # Log every N steps for live tracking
        if step % 10 == 0:
            run.log_metric("train_loss", loss, step=step)
            # Internally: buffered, batched, sent asynchronously
```

### 4. Artifact Storage & Deduplication

**Challenge:** Models can be GBs–TBs. Storing every checkpoint is expensive.

**Solution:**

- **Content-based deduplication:**
  - Hash each artifact (SHA-256).
  - If hash exists, create metadata entry but don't re-upload.
- **Tiered storage:**
  - Hot: Recent artifacts on fast storage (SSD, S3 standard).
  - Cold: Old artifacts on cheaper storage (S3 Glacier, tape).
- **Compression:**
  - Compress models before upload (gzip, zstd).

```python
import hashlib
import gzip

def upload_artifact(run_id: str, path: str, local_file: str):
    # Compute hash
    with open(local_file, 'rb') as f:
        content = f.read()
        content_hash = hashlib.sha256(content).hexdigest()
    
    # Check if artifact with this hash exists
    existing = artifact_service.get_by_hash(content_hash)
    if existing:
        # Create metadata entry pointing to existing storage
        artifact_service.link_artifact(run_id, path, existing.storage_uri, content_hash)
        return
    
    # Compress and upload
    compressed = gzip.compress(content)
    storage_uri = object_store.upload(f"{run_id}/{path}.gz", compressed)
    
    # Create metadata entry
    artifact_service.create_artifact(
        run_id=run_id,
        path=path,
        size_bytes=len(content),
        content_hash=content_hash,
        storage_uri=storage_uri
    )
```

### 5. Query & Search

**Common queries:**

- "Show all runs with learning_rate > 0.01 and val_accuracy > 0.9"
- "Find best run in experiment X by val_accuracy"
- "Show runs created in last 7 days by user Y"

**Implementation:**

```python
# Query API example
runs = client.search_runs(
    experiment_ids=["exp123"],
    filter_string="params.learning_rate > 0.01 AND metrics.val_accuracy > 0.9",
    order_by=["metrics.val_accuracy DESC"],
    max_results=10
)

for run in runs:
    print(f"Run {run.run_id}: LR={run.params['learning_rate']}, Acc={run.metrics['val_accuracy']}")
```

**Optimization:**

- Index on common filter fields (user_id, experiment_id, tags, status, timestamps)
- Cache popular queries (top runs, recent runs)
- Use read replicas for heavy read workloads

## Scaling Strategies

### 1. Sharding Experiments/Runs

For very large deployments:

- Shard metadata DB by experiment_id or user_id
- Each shard handles a subset of experiments
- API Gateway routes requests to correct shard

### 2. Metric Buffering & Batching

High-throughput training jobs can log metrics at high frequency (100s–1000s/sec):

- Client buffers metrics locally
- Batches and sends every N seconds or M metrics
- Server-side ingestion queue (Kafka, SQS) for further buffering

### 3. Artifact Caching

Frequently accessed artifacts (latest models, popular checkpoints):

- Cache in CDN (CloudFront, Fastly)
- Local cache on training nodes (NVMe SSD)
- Lazy loading: download only when accessed

### 4. Distributed Artifact Storage

For petabyte-scale artifact storage:

- Use distributed object stores (S3, GCS, Ceph)
- Implement multipart upload for large files
- Use pre-signed URLs for direct client-to-storage uploads (bypass API server)

## Monitoring & Observability

### Key Metrics

**System metrics:**
- Request latency (p50, p95, p99)
- Throughput (requests/sec, metrics logged/sec, artifacts uploaded/sec)
- Error rates (4xx, 5xx)
- Storage usage (DB size, object store size)

**User metrics:**
- Active experiments/runs
- Average metrics logged per run
- Average artifact size
- Query response times

**Dashboards:**
- Real-time experiment dashboard (running/completed/failed runs)
- System health dashboard (latency, error rates, resource usage)
- Cost dashboard (storage costs, compute costs)

## Failure Modes & Mitigations

| Failure Mode | Impact | Mitigation |
|-------------|--------|------------|
| **Metadata DB down** | Can't create/query experiments | Read replicas, automatic failover, local caching |
| **Object store unavailable** | Can't upload/download artifacts | Retry with exponential backoff, fallback to local storage |
| **Metric ingestion backlog** | Delayed metric visibility | Buffering, rate limiting, auto-scaling ingest workers |
| **Lost run metadata** | Experiment not reproducible | Periodic backups, transaction logs, write-ahead logs |
| **Concurrent write conflicts** | Metrics/artifacts overwritten | Optimistic locking, append-only logs |
| **API rate limit hit** | Client blocked | Exponential backoff, client-side buffering, increase limits |

## Real-World Case Study: Large-Scale ML Team

**Scenario:**
- 100+ ML engineers and researchers
- 50K+ experiments, 1M+ runs
- 10 PB of artifacts (models, datasets, checkpoints)
- Multi-cloud (AWS, GCP)

**Architecture:**
- Metadata: PostgreSQL with read replicas, sharded by experiment_id
- Metrics: InfluxDB cluster, 100K metrics/sec write throughput
- Artifacts: S3 + GCS with cross-region replication
- API: Kubernetes cluster with auto-scaling (10–100 pods)
- Web UI: React SPA, served via CDN

**Key optimizations:**
- Pre-signed URLs for large artifact uploads (direct to S3/GCS)
- Client-side metric buffering (log every 10 steps, batch send)
- Artifact deduplication (saved ~30% storage cost)
- Tiered storage (hot: S3 Standard, cold: S3 Glacier, ~50% cost reduction)

**Outcomes:**
- 99.95% uptime
- Median query latency: 120ms
- p99 metric log latency: 8ms
- $200K/year savings from deduplication and tiered storage

## Cost Analysis

### Example: Medium-Sized Team

**Assumptions:**
- 10 researchers
- 100 experiments/month, 1000 runs/month
- Average run: 10 GB artifacts, 10K metrics
- Retention: 2 years

| Component | Cost/Month |
|-----------|-----------|
| Metadata DB (PostgreSQL RDS, db.r5.large) | $300 |
| Metrics DB (InfluxDB Cloud) | $500 |
| Object storage (S3, 10 TB) | $230 |
| API compute (Kubernetes, 5 nodes) | $750 |
| Data transfer | $100 |
| **Total** | **$1,880** |

**Optimization levers:**
- Deduplication: -20–30% storage cost
- Tiered storage: -30–50% storage cost (move old artifacts to Glacier)
- Reserved instances: -30% compute cost
- Compression: -50% storage and transfer cost

## Advanced Topics

### 1. Hyperparameter Sweep Integration

Integrate with hyperparameter tuning libraries (Optuna, Ray Tune):

```python
import optuna
import experiment_tracker as et

def objective(trial):
    run = experiment.start_run(name=f"trial_{trial.number}")
    
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    run.log_params({"learning_rate": lr, "batch_size": batch_size})
    
    # Train and log metrics
    val_acc = train_and_evaluate(lr, batch_size, run)
    
    run.finish()
    return val_acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

### 2. Model Registry Integration

Link experiment tracking with model registry:

- Best run → promoted to staging → production
- Track lineage: model → run → experiment → dataset

### 3. Data Versioning

Track data versions alongside experiments:

- Dataset hash (content-based)
- Data pipeline version (Git commit)
- Train/val/test split configs

```python
run.log_dataset(
    name="imagenet_v2",
    hash="sha256:abc123...",
    split_config={"train": 0.8, "val": 0.1, "test": 0.1}
)
```

### 4. Compliance & Audit Trails

For regulated industries (healthcare, finance):

- Immutable experiment logs
- Audit trail for all changes (who, what, when)
- Data lineage tracking
- Access control and encryption

## Practical Debugging & Operations Checklist

### For Platform Engineers

- **Monitor ingestion lag:** Metrics should appear in UI within seconds of logging.
- **Set up alerts:** DB disk space >80%, API error rate >1%, artifact upload failures.
- **Test disaster recovery:** Can you restore from backups? Time to recover?
- **Load test:** Can the system handle 10x current load?

### For ML Engineers

- **Always log hyperparameters:** Even "fixed" ones—you'll want to compare later.
- **Use tags liberally:** Makes filtering/searching much easier.
- **Log environment:** Git commit, Docker image, package versions—critical for reproducibility.
- **Log artifacts incrementally:** Don't wait until end of training to upload checkpoints.
- **Use run names:** Descriptive names make comparison easier (`resnet50_adam_lr0.001` vs `run_42`).

## Key Takeaways

✅ **Experiment tracking is foundational for MLOps**—enables reproducibility, collaboration, and systematic exploration.

✅ **Scale requires separation of concerns**: metadata, metrics, artifacts each have different storage/query needs.

✅ **Deduplication and tiered storage** are critical for cost efficiency at scale.

✅ **Client-side buffering** avoids overwhelming the backend with high-frequency metric logging.

✅ **Systematic iteration through experiment spaces** mirrors structured traversal patterns (like Spiral Matrix).

✅ **Integration with existing tools** (Git, Docker, hyperparameter tuning) is key for adoption.

✅ **Observability and cost monitoring** are as important as core functionality.

### Connection to Thematic Link: Systematic Iteration and State Tracking

All three Day 19 topics converge on **systematic, stateful exploration**:

**DSA (Spiral Matrix):**
- Layer-by-layer traversal with boundary tracking
- Explicit state management (top, bottom, left, right)
- Resume/pause friendly

**ML System Design (Experiment Tracking Systems):**
- Systematic exploration of hyperparameter/architecture spaces
- Track state of experiments (running, completed, failed)
- Resume from checkpoints, recover from failures

**Speech Tech (Speech Experiment Management):**
- Organize speech model experiments across multiple dimensions
- Track model versions, data versions, training configs
- Enable reproducibility and comparison

The **unifying pattern**: structured iteration through complex spaces, with clear state persistence and recoverability.

---

**Originally published at:** [arunbaby.com/ml-system-design/0019-experiment-tracking-systems](https://www.arunbaby.com/ml-system-design/0019-experiment-tracking-systems/)

*If you found this helpful, consider sharing it with others who might benefit.*






