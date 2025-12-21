---
title: "DAG-based Pipeline Orchestration"
day: 49
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - dag
  - pipeline-orchestration
  - workflow
  - airflow
  - mlops
  - dependency-management
difficulty: Hard
subdomain: "MLOps"
tech_stack: Python, Airflow, Prefect, Dagster
scale: "1000s of tasks, complex dependencies"
companies: Airbnb, Spotify, Uber, Netflix
related_dsa_day: 49
related_speech_day: 49
related_agents_day: 49
---

**"Every ML pipeline is a DAG—order matters when data flows."**

## 1. Problem Statement

Design a **DAG-based pipeline orchestration system** that:
1. Defines tasks with dependencies
2. Executes tasks in correct order
3. Handles failures and retries
4. Supports parallel execution
5. Provides monitoring and observability

## 2. Core Concepts

### Why DAG?

```
ML Pipeline:
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Ingest  │───►│ Process  │───►│  Train   │
└──────────┘    └──────────┘    └────┬─────┘
                                     │
                ┌──────────┐         │
                │ Validate │◄────────┘
                └────┬─────┘
                     │
                ┌────▼─────┐
                │  Deploy  │
                └──────────┘

DAG ensures:
- No circular dependencies
- Clear execution order
- Parallel where possible
```

## 3. Task and DAG Definition

```python
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Any, Set
from enum import Enum
from datetime import datetime
import asyncio

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """A single unit of work in the pipeline."""
    name: str
    func: Callable
    dependencies: List[str] = field(default_factory=list)
    retries: int = 3
    retry_delay: int = 60
    timeout: int = 3600
    
    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = None
    start_time: datetime = None
    end_time: datetime = None
    
    def __hash__(self):
        return hash(self.name)


class DAG:
    """Directed Acyclic Graph of tasks."""
    
    def __init__(self, name: str):
        self.name = name
        self.tasks: Dict[str, Task] = {}
        self._validated = False
    
    def add_task(self, task: Task):
        """Add a task to the DAG."""
        self.tasks[task.name] = task
        self._validated = False
    
    def task(self, dependencies: List[str] = None, **kwargs):
        """Decorator to create tasks."""
        def decorator(func):
            task = Task(
                name=func.__name__,
                func=func,
                dependencies=dependencies or [],
                **kwargs
            )
            self.add_task(task)
            return func
        return decorator
    
    def validate(self) -> bool:
        """
        Validate DAG has no cycles.
        
        Uses DFS with coloring (exactly like Course Schedule!).
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {name: WHITE for name in self.tasks}
        
        def has_cycle(task_name):
            if colors[task_name] == GRAY:
                return True  # Cycle!
            if colors[task_name] == BLACK:
                return False
            
            colors[task_name] = GRAY
            
            for dep in self.tasks[task_name].dependencies:
                if dep not in self.tasks:
                    raise ValueError(f"Unknown dependency: {dep}")
                if has_cycle(dep):
                    return True
            
            colors[task_name] = BLACK
            return False
        
        for task_name in self.tasks:
            if has_cycle(task_name):
                return False
        
        self._validated = True
        return True
    
    def get_execution_order(self) -> List[str]:
        """
        Get topological order for execution.
        
        Uses Kahn's algorithm.
        """
        if not self._validated:
            self.validate()
        
        # Calculate in-degrees
        in_degree = {name: 0 for name in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                # dep must complete before task
                pass
        
        # Count how many tasks depend on each task
        dependents = {name: [] for name in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                dependents[dep].append(task.name)
                in_degree[task.name] += 1
        
        # Start with tasks having no dependencies
        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return order
    
    def get_ready_tasks(self, completed: Set[str]) -> List[str]:
        """Get tasks ready to run (all deps completed)."""
        ready = []
        for name, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                if all(dep in completed for dep in task.dependencies):
                    ready.append(name)
        return ready
```

## 4. Executor

```python
class DAGExecutor:
    """Execute DAG with parallel task support."""
    
    def __init__(self, dag: DAG, max_parallel: int = 4):
        self.dag = dag
        self.max_parallel = max_parallel
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
    
    async def run(self) -> bool:
        """
        Execute the DAG.
        
        Returns True if all tasks succeeded.
        """
        if not self.dag.validate():
            raise ValueError("DAG has cycles!")
        
        while True:
            # Get tasks ready to run
            ready = self.dag.get_ready_tasks(self.completed)
            
            if not ready:
                # Check if we're done or stuck
                if len(self.completed) + len(self.failed) == len(self.dag.tasks):
                    break
                if self.failed:
                    # Some tasks failed, can't proceed
                    break
            
            # Run ready tasks in parallel (up to limit)
            batch = ready[:self.max_parallel]
            results = await asyncio.gather(
                *[self._run_task(name) for name in batch],
                return_exceptions=True
            )
            
            for name, result in zip(batch, results):
                if isinstance(result, Exception):
                    self.failed.add(name)
                else:
                    self.completed.add(name)
        
        return len(self.failed) == 0
    
    async def _run_task(self, name: str):
        """Run a single task with retries."""
        task = self.dag.tasks[name]
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        
        for attempt in range(task.retries):
            try:
                # Get inputs from dependencies
                inputs = {
                    dep: self.dag.tasks[dep].result
                    for dep in task.dependencies
                }
                
                # Run with timeout
                task.result = await asyncio.wait_for(
                    self._execute(task, inputs),
                    timeout=task.timeout
                )
                
                task.status = TaskStatus.SUCCESS
                task.end_time = datetime.now()
                return task.result
                
            except Exception as e:
                task.error = str(e)
                if attempt < task.retries - 1:
                    await asyncio.sleep(task.retry_delay)
        
        task.status = TaskStatus.FAILED
        task.end_time = datetime.now()
        raise Exception(f"Task {name} failed after {task.retries} attempts")
    
    async def _execute(self, task: Task, inputs: Dict):
        """Execute task function."""
        if asyncio.iscoroutinefunction(task.func):
            return await task.func(**inputs)
        else:
            return task.func(**inputs)
```

## 5. ML Pipeline Example

```python
# Define ML training pipeline
pipeline = DAG("ml_training")

@pipeline.task()
def load_data():
    """Load raw data from storage."""
    return {"data": "raw_data", "rows": 10000}

@pipeline.task(dependencies=["load_data"])
def preprocess(load_data):
    """Clean and transform data."""
    return {"data": "processed", "rows": load_data["rows"]}

@pipeline.task(dependencies=["preprocess"])
def split_data(preprocess):
    """Split into train/val/test."""
    return {
        "train": "train_data",
        "val": "val_data",
        "test": "test_data"
    }

@pipeline.task(dependencies=["split_data"])
def train_model(split_data):
    """Train the model."""
    return {"model": "trained_model", "metrics": {"accuracy": 0.95}}

@pipeline.task(dependencies=["train_model", "split_data"])
def evaluate(train_model, split_data):
    """Evaluate on test set."""
    return {"test_accuracy": 0.93}

@pipeline.task(dependencies=["evaluate"])
def deploy(evaluate):
    """Deploy if metrics pass threshold."""
    if evaluate["test_accuracy"] > 0.9:
        return {"status": "deployed"}
    return {"status": "rejected"}


# Run pipeline
async def main():
    executor = DAGExecutor(pipeline, max_parallel=2)
    success = await executor.run()
    
    for name, task in pipeline.tasks.items():
        print(f"{name}: {task.status.value}")
        if task.result:
            print(f"  Result: {task.result}")

asyncio.run(main())
```

## 6. Scheduling and Triggers

```python
from datetime import timedelta

class ScheduledDAG(DAG):
    """DAG with scheduling support."""
    
    def __init__(
        self,
        name: str,
        schedule: str = None,  # Cron expression
        start_date: datetime = None,
        catchup: bool = False
    ):
        super().__init__(name)
        self.schedule = schedule
        self.start_date = start_date
        self.catchup = catchup
    
    def should_run(self, execution_date: datetime) -> bool:
        """Check if DAG should run for given date."""
        # Parse cron and check
        pass


class Scheduler:
    """Schedule and trigger DAG runs."""
    
    def __init__(self):
        self.dags: Dict[str, ScheduledDAG] = {}
        self.running: Dict[str, DAGExecutor] = {}
    
    def register(self, dag: ScheduledDAG):
        self.dags[dag.name] = dag
    
    async def tick(self):
        """Check for DAGs to run."""
        now = datetime.now()
        
        for name, dag in self.dags.items():
            if name not in self.running:
                if dag.should_run(now):
                    executor = DAGExecutor(dag)
                    self.running[name] = executor
                    asyncio.create_task(self._run_dag(name, executor))
    
    async def _run_dag(self, name: str, executor: DAGExecutor):
        try:
            await executor.run()
        finally:
            del self.running[name]
```

## 7. Monitoring and Observability

```python
from dataclasses import asdict
import json

class DAGMonitor:
    """Monitor DAG execution."""
    
    def __init__(self, dag: DAG):
        self.dag = dag
        self.events = []
    
    def log_event(self, event_type: str, task_name: str, data: dict):
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "task": task_name,
            "data": data
        })
    
    def get_status(self) -> dict:
        """Get current DAG status."""
        return {
            "dag_name": self.dag.name,
            "tasks": {
                name: {
                    "status": task.status.value,
                    "start_time": task.start_time.isoformat() if task.start_time else None,
                    "end_time": task.end_time.isoformat() if task.end_time else None,
                    "error": task.error
                }
                for name, task in self.dag.tasks.items()
            },
            "progress": f"{len([t for t in self.dag.tasks.values() if t.status == TaskStatus.SUCCESS])}/{len(self.dag.tasks)}"
        }
    
    def get_gantt_data(self) -> list:
        """Get data for Gantt chart visualization."""
        return [
            {
                "task": name,
                "start": task.start_time.timestamp() if task.start_time else 0,
                "end": task.end_time.timestamp() if task.end_time else 0,
                "status": task.status.value
            }
            for name, task in self.dag.tasks.items()
        ]
```

## 8. Connection to Course Schedule

DAG orchestration is Course Schedule in production:

| Course Schedule | Pipeline Orchestration |
|----------------|----------------------|
| Course | Task |
| Prerequisite | Dependency |
| Can finish? | Is DAG valid? |
| Topological order | Execution order |
| Cycle detection | Deadlock prevention |

The algorithms are identical—same problem, different domain.

## 9. Key Takeaways

1. **DAG = No cycles** = deterministic execution order
2. **Topological sort** determines task order
3. **Parallel execution** for independent tasks
4. **Retry logic** handles transient failures
5. **Monitoring** essential for debugging pipelines

---

**Originally published at:** [arunbaby.com/ml-system-design/0049-dag-pipeline-orchestration](https://www.arunbaby.com/ml-system-design/0049-dag-pipeline-orchestration/)

*If you found this helpful, consider sharing it with others who might benefit.*
