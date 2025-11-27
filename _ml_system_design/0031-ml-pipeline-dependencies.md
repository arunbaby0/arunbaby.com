---
title: "ML Pipeline Dependencies & Orchestration"
day: 31
collection: ml_system_design
categories:
  - ml_system_design
tags:
  - pipelines
  - orchestration
  - airflow
  - dependencies
subdomain: "MLOps"
tech_stack: [Airflow, Kubeflow, Prefect, Dagster, Metaflow]
scale: "Thousands of tasks, Petabytes of data"
companies: [Netflix, Uber, Airbnb, Spotify]
related_dsa_day: 31
related_ml_day: 31
related_speech_day: 31
---

**"Managing complex ML workflows with thousands of interdependent tasks."**

## 1. Why ML Pipeline Orchestration Matters

Machine Learning workflows are complex DAGs (Directed Acyclic Graphs) with dependencies:

```
Data Ingestion → Feature Engineering → Training → Evaluation → Deployment
       ↓                ↓                   ↓
   Validation     Feature Store      Model Registry
```

**Challenges:**
1. **Dependencies:** Training depends on feature engineering completing successfully.
2. **Scheduling:** Run daily at 2 AM, or trigger when new data arrives.
3. **Retry Logic:** If a task fails due to transient error, retry 3 times.
4. **Monitoring:** Alert if any step takes > 2 hours.
5. **Backfilling:** Re-run pipeline for historical dates.

## 2. Apache Airflow - The Industry Standard

**Airflow** models workflows as **DAGs** (Directed Acyclic Graphs) of tasks.

### Example: Training Pipeline DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['alerts@company.com']
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Daily model training',
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False
)

def extract_data(**context):
    # Pull data from warehouse
    data = query_warehouse(context['ds'])  # ds = execution date
    save_to_gcs(data, f"raw_data_{context['ds']}.parquet")

def feature_engineering(**context):
    data = load_from_gcs(f"raw_data_{context['ds']}.parquet")
    features = compute_features(data)
    save_to_gcs(features, f"features_{context['ds']}.parquet")

def train_model(**context):
    features = load_from_gcs(f"features_{context['ds']}.parquet")
    model = train(features)
    save_model(model, f"model_{context['ds']}.pkl")

def evaluate_model(**context):
    model = load_model(f"model_{context['ds']}.pkl")
    metrics = evaluate(model, test_data)
    
    if metrics['auc'] < 0.75:
        raise ValueError("Model quality below threshold!")
    
    log_metrics(metrics)

def deploy_model(**context):
    model_path = f"model_{context['ds']}.pkl"
    deploy_to_production(model_path)

# Define task dependencies
extract = PythonOperator(task_id='extract_data', python_callable=extract_data, dag=dag)
feature_eng = PythonOperator(task_id='feature_engineering', python_callable=feature_engineering, dag=dag)
train = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)
evaluate = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model, dag=dag)
deploy = PythonOperator(task_id='deploy_model', python_callable=deploy_model, dag=dag)

# Dependencies (topological order)
extract >> feature_eng >> train >> evaluate >> deploy
```

**Key Features:**
- **Scheduling:** Cron-like syntax (`schedule_interval`).
- **Backfilling:** Re-run for past dates with `airflow backfill`.
- **Retries:** Automatic retry with exponential backoff.
- **Monitoring:** Web UI shows task status, logs, duration.

## 3. Kubeflow Pipelines (Kubernetes-Native)

**Kubeflow** runs ML pipelines on Kubernetes.

**Benefits:**
- **Scalability:** Auto-scale workers on K8s.
- **Reproducibility:** Each task runs in a Docker container.
- **GPU Support:** Easily request GPU resources.

### Example: Training Pipeline

```python
import kfp
from kfp import dsl

@dsl.component
def data_ingestion(output_path: str):
    import pandas as pd
    # Fetch data
    df = pd.read_sql("SELECT * FROM users", conn)
    df.to_parquet(output_path)

@dsl.component
def feature_engineering(input_path: str, output_path: str):
    import pandas as pd
    df = pd.read_parquet(input_path)
    # Engineer features
    df['age_squared'] = df['age'] ** 2
    df.to_parquet(output_path)

@dsl.component(base_image='tensorflow/tensorflow:latest-gpu')
def train_model(input_path: str, model_output: str):
    import tensorflow as tf
    data = tf.data.Dataset.from_tensor_slices(...)
    model = tf.keras.models.Sequential([...])
    model.fit(data, epochs=10)
    model.save(model_output)

@dsl.pipeline(name='ML Training Pipeline')
def ml_pipeline():
    data_task = data_ingestion(output_path='/data/raw.parquet')
    feature_task = feature_engineering(
        input_path=data_task.output,
        output_path='/data/features.parquet'
    )
    train_task = train_model(
        input_path=feature_task.output,
        model_output='/models/model.h5'
    ).set_gpu_limit(1)  # Request 1 GPU

# Compile and run
kfp.compiler.Compiler().compile(ml_pipeline, 'pipeline.yaml')
client = kfp.Client()
client.create_run_from_pipeline_func(ml_pipeline)
```

**Advantages:**
- **Containerization:** Each step is isolated.
- **Resource Management:** Request specific CPU/GPU/memory.
- **Artifact Tracking:** Automatic versioning of data, models.

## 4. Dependency Patterns in ML Pipelines

### Pattern 1: Linear Pipeline
```
A → B → C → D
```
**Example:** Data Ingestion → Preprocessing → Training → Deployment.

### Pattern 2: Fan-Out (Parallel Tasks)
```
       → B1 →
A →    → B2 →   → D
       → B3 →
```
**Example:** Train 3 models in parallel, then ensemble them.

```python
# Airflow
feature_eng >> [train_model_1, train_model_2, train_model_3] >> ensemble >> deploy
```

### Pattern 3: Fan-In (Join)
```
A1 →
A2 →   → C
A3 →
```
**Example:** Process data from 3 sources, then merge.

### Pattern 4: Diamond (Complex Dependencies)
```
     → B →
A →         → D
     → C →
```
**Example:** Extract features from images (B) and text (C), then combine for training (D).

## 5. Handling Failures and Retries

### Transient vs. Persistent Failures

**Transient:** Network timeout, database busy.
- **Solution:** Retry with exponential backoff.

**Persistent:** Bug in code, missing data.
- **Solution:** Alert humans, don't retry forever.

### Retry Strategy

```python
# Airflow
default_args = {
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(hours=1)
}

# Prefect
@flow(retries=3, retry_delay_seconds=300)
def my_flow():
    pass
```

### Idempotency

**Critical:** Tasks must be idempotent (running twice = running once).

**Bad (not idempotent):**
```python
def bad_task():
    db.execute("INSERT INTO results VALUES (...)")  # Duplicate inserts!
```

**Good (idempotent):**
```python
def good_task(**context):
    date = context['ds']
    db.execute(f"DELETE FROM results WHERE date = '{date}'")
    db.execute(f"INSERT INTO results VALUES (...)")
```

## 6. Dynamic DAG Generation

**Problem:** You have 100 models to train daily. Don't want to write 100 tasks manually.

**Solution:** Programmatically generate DAGs.

```python
# Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator

models = ['model_A', 'model_B', 'model_C', ...]

dag = DAG('dynamic_training', ...)

for model_name in models:
    task = PythonOperator(
        task_id=f'train_{model_name}',
        python_callable=train_model,
        op_kwargs={'model_name': model_name},
        dag=dag
    )
```

## Deep Dive: Netflix's ML Pipeline at Scale

**Scale:**
- **1000s of models** (one per region, per content type).
- **Petabytes of data.**
- **Hourly re-training** for some models.

**Architecture:**
1. **Metaflow (Netflix's open-source tool):**
   - Simplifies Airflow/Kubernetes complexity.
   - Auto-versioning of data/model artifacts.
2. **S3 for Data Lake.**
3. **Spark for Feature Engineering.**
4. **TensorFlow/PyTorch for Training.**
5. **Titus (Netflix's container platform) for Execution.**

**Example Metaflow Pipeline:**
```python
from metaflow import FlowSpec, step, batch, resources

class RecommendationFlow(FlowSpec):
    
    @step
    def start(self):
        self.data = load_from_s3('s3://netflix-data/user_views.parquet')
        self.next(self.feature_engineering)
    
    @batch(cpu=16, memory=64000)  # 16 CPUs, 64GB RAM
    @step
    def feature_engineering(self):
        self.features = compute_features(self.data)
        self.next(self.train)
    
    @batch(gpu=4, memory=128000)  # 4 GPUs
    @step
    def train(self):
        self.model = train_model(self.features)
        self.next(self.evaluate)
    
    @step
    def evaluate(self):
        metrics = evaluate(self.model)
        if metrics['precision@10'] > 0.8:
            self.next(self.deploy)
        else:
            self.next(self.end)  # Don't deploy bad models
    
    @step
    def deploy(self):
        deploy_to_production(self.model)
        self.next(self.end)
    
    @step
    def end(self):
        print("Pipeline complete!")

if __name__ == '__main__':
    RecommendationFlow()
```

## Deep Dive: Uber's Michelangelo Platform

**Michelangelo** is Uber's end-to-end ML platform.

**Pipeline Components:**
1. **Data Sources:** Ride data, driver data, geospatial data.
2. **Feature Store:** Pre-computed features (e.g., "average rides per hour in this area").
3. **Training:** Distributed training on Spark/Horovod.
4. **Model Serving:** Deploy to Uber's serving infrastructure.
5. **Monitoring:** Track prediction accuracy, latency.

**Scheduling:**
- **Batch:** Daily models for surge pricing.
- **Streaming:** Real-time fraud detection.

**Dependency Management:**
- Use **Airflow** for orchestration.
- Feature engineering depends on multiple data sources joining correctly.

## Deep Dive: Feature Store Integration

**Problem:** Feature computation is expensive. Don't recompute for every model.

**Solution: Feature Store (Feast, Tecton)**

**Architecture:**
```
Batch Pipeline (Airflow) → Compute Features → Feature Store (Online + Offline)
                                                    ↓
                                            Training & Serving
```

**Integration with Airflow:**
```python
@task
def compute_features(**context):
    date = context['ds']
    raw_data = load_data(date)
    features = transform(raw_data)
    
    # Write to Feature Store
    feast_client.push_features(features, timestamp=date)

@task
def train_model(**context):
    date = context['ds']
    # Read from Feature Store
    features = feast_client.get_historical_features(
        entity_rows={'user_id': [1, 2, 3, ...]},
        feature_refs=['user_profile:age', 'user_profile:country']
    )
    model = train(features)
```

## Deep Dive: Pipeline Testing and Validation

**Unit Tests:** Test individual tasks.
```python
def test_feature_engineering():
    input_data = pd.DataFrame({'age': [25, 30, 35]})
    output = feature_engineering(input_data)
    assert 'age_squared' in output.columns
    assert output['age_squared'].tolist() == [625, 900, 1225]
```

**Integration Tests:** Test full pipeline on sample data.
```python
def test_pipeline_end_to_end():
    # Run pipeline with small test dataset
    result = run_pipeline(test_data)
    assert result['model_quality'] > 0.7
```

**Data Validation:** Check data quality before training.
```python
from great_expectations import DataContext

@task
def validate_data(**context):
    data = load_data(context['ds'])
    
    # Expectations
    assert data['age'].between(0, 120).all(), "Invalid ages!"
    assert data['revenue'].notnull().all(), "Missing revenue!"
    
    # Use Great Expectations for complex validation
    context = DataContext()
    results = context.run_checkpoint('data_quality_checkpoint', batch_request=...)
    
    if not results.success:
        raise ValueError("Data quality check failed!")
```

## Deep Dive: Backfilling Historical Data

**Problem:** You fixed a bug in feature engineering. Need to re-run for the past 90 days.

**Airflow Backfill:**
```bash
airflow dags backfill \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    ml_training_pipeline
```

**Challenges:**
- **Resource Limits:** Running 90 days worth of jobs simultaneously can overwhelm infrastructure.
- **Solution:** Use `max_active_runs` parameter to limit parallelism.

```python
dag = DAG(
    'ml_pipeline',
    max_active_runs=5,  # Run max 5 dates in parallel
    ...
)
```

## Deep Dive: Monitoring and Alerting

**Metrics to Track:**
1. **Task Duration:** Alert if task takes > 2x usual time.
2. **Task Failure Rate:** Alert if > 5% failure rate.
3. **Data Volume:** Alert if input data is 50% smaller than yesterday (possible upstream issue).
4. **Model Quality:** Alert if AUC drops below threshold.

**Airflow Integration with Prometheus/Grafana:**
```python
from airflow.providers.prometheus.operators.push_gateway import PushGatewayOperator

@task
def push_metrics(**context):
    metrics = {
        'model_auc': 0.85,
        'training_duration': 3600,  # seconds
        'data_rows': 1000000
    }
    
    # Push to Prometheus
    push_gateway = PushGatewayOperator(
        task_id='push_metrics',
        metrics=metrics,
        gateway_url='http://pushgateway:9091'
    )
```

## Deep Dive: Cost Optimization

**Problem:** Training 1000 models on GPUs is expensive.

**Strategies:**
1. **Spot Instances:** Use AWS Spot / GCP Preemptible VMs (70% cheaper).
2. **Auto-Scaling:** Scale down workers when idle.
3. **Resource Right-Sizing:** Don't request more CPU/RAM than needed.
4. **Caching:** Cache intermediate results (features).

**Airflow on Spot Instances:**
```python
from airflow.providers.amazon.aws.operators.ecs import ECSOperator

train_task = ECSOperator(
    task_id='train_model',
    task_definition='ml-training',
    cluster='ml-cluster',
    overrides={
        'containerOverrides': [{
            'name': 'training-container',
            'environment': [...],
        }],
        'cpu': '4096',  # 4 vCPUs
        'memory': '16384',  # 16GB
        'taskRoleArn': 'arn:aws:iam::...',
    },
    launch_type='FARGATE_SPOT',  # Use Spot instances
    dag=dag
)
```

## Implementation: Full ML Pipeline Orchestration

{% raw %}
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import pandas as pd

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['ml-alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'complete_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline',
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
)

def data_quality_check(**context):
    """Validate data quality before processing"""
    date = context['ds']
    data = load_data(date)
    
    # Checks
    assert len(data) > 10000, "Insufficient data!"
    assert data['label'].notnull().all(), "Missing labels!"
    
    # Log stats
    context['ti'].xcom_push(key='data_size', value=len(data))

def extract_features(**context):
    """Feature engineering"""
    date = context['ds']
    data = load_data(date)
    features = engineer_features(data)
    features.to_parquet(f'/data/features_{date}.parquet')

def train_models(**context):
    """Train multiple models in parallel"""
    date = context['ds']
    features = pd.read_parquet(f'/data/features_{date}.parquet')
    
    models = {}
    for model_type in ['xgboost', 'random_forest', 'logistic_regression']:
        model = train_model(features, model_type)
        models[model_type] = model
        save_model(model, f'/models/{model_type}_{date}.pkl')
    
    context['ti'].xcom_push(key='model_paths', value=models.keys())

def evaluate_and_select_best(**context):
    """Evaluate models and select the best"""
    date = context['ds']
    model_types = context['ti'].xcom_pull(task_ids='train_models', key='model_paths')
    
    best_model = None
    best_auc = 0
    
    for model_type in model_types:
        model = load_model(f'/models/{model_type}_{date}.pkl')
        auc = evaluate(model, test_data)
        
        if auc > best_auc:
            best_auc = auc
            best_model = model_type
    
    # Quality gate
    if best_auc < 0.75:
        raise ValueError(f"Best model AUC ({best_auc}) below threshold!")
    
    context['ti'].xcom_push(key='best_model', value=best_model)
    context['ti'].xcom_push(key='best_auc', value=best_auc)

def deploy_best_model(**context):
    """Deploy the best model to production"""
    best_model = context['ti'].xcom_pull(task_ids='evaluate_and_select_best', key='best_model')
    date = context['ds']
    
    model_path = f'/models/{best_model}_{date}.pkl'
    deploy_to_production(model_path)
    
    print(f"Deployed {best_model} to production!")

# Define tasks
quality_check = Python Operator(task_id='data_quality_check', python_callable=data_quality_check, dag=dag)
feature_eng = PythonOperator(task_id='extract_features', python_callable=extract_features, dag=dag)
train = PythonOperator(task_id='train_models', python_callable=train_models, dag=dag)
evaluate = PythonOperator(task_id='evaluate_and_select_best', python_callable=evaluate_and_select_best, dag=dag)
deploy = PythonOperator(task_id='deploy_best_model', python_callable=deploy_best_model, dag=dag)

success_email = EmailOperator(
    task_id='success_email',
    to='ml-team@company.com',
    subject='ML Pipeline Success - {{ ds }}',
    html_content='Pipeline completed successfully. Best AUC: {{ ti.xcom_pull(task_ids=\'evaluate_and_select_best\', key=\'best_auc\') }}',
    dag=dag
)

# Dependencies
quality_check >> feature_eng >> train >> evaluate >> deploy >> success_email
```
{% endraw %}

## Top Interview Questions

**Q1: How do you handle a task that depends on multiple upstream tasks?**
*Answer:*
Use list syntax in Airflow: `[task1, task2, task3] >> downstream_task`. All must complete successfully before downstream runs.

**Q2: What's the difference between `schedule_interval` and `start_date`?**
*Answer:*
`start_date` is when the DAG becomes active. `schedule_interval` determines how often it runs. First run happens at `start_date + schedule_interval`.

**Q3: How do you pass data between tasks?**
*Answer:*
- **Small data:** XCom (Airflow's inter-task communication).
- **Large data:** Write to shared storage (S3, GCS) and pass the path via XCom.

**Q4: How do you handle long-running tasks (> 24 hours)?**
*Answer:*
Use sensors or split into smaller tasks. Or use `execution_timeout` parameter to fail gracefully if task hangs.

## Key Takeaways

1. **DAG Structure:** ML pipelines are DAGs with dependencies modeled via topological sort.
2. **Airflow is Standard:** Most companies use Apache Airflow for orchestration.
3. **Idempotency is Critical:** Tasks must be safe to re-run.
4. **Monitoring:** Track task duration, failure rate, data quality.
5. **Cost Optimization:** Use spot instances, auto-scaling, caching.

## Summary

| Aspect | Insight |
|:---|:---|
| **Core Problem** | Orchestrate complex ML workflows with dependencies |
| **Best Tools** | Airflow (general), Kubeflow (K8s), Metaflow (Netflix) |
| **Key Patterns** | Linear, Fan-Out, Fan-In, Diamond |
| **Critical Features** | Scheduling, retries, monitoring, backfilling |

---

**Originally published at:** [arunbaby.com/ml-system-design/0031-ml-pipeline-dependencies](https://www.arunbaby.com/ml-system-design/0031-ml-pipeline-dependencies/)

*If you found this helpful, consider sharing it with others who might benefit.*


