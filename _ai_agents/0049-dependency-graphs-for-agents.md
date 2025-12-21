---
title: "Dependency Graphs for Agent Tasks"
day: 49
collection: ai_agents
categories:
  - ai-agents
tags:
  - dependency-graphs
  - task-orchestration
  - dag
  - parallel-execution
  - planning
difficulty: Hard
subdomain: "Agent Architecture"
tech_stack: Python, LangGraph, asyncio
scale: "Complex multi-step workflows"
companies: OpenAI, Anthropic, LangChain
related_dsa_day: 49
related_ml_day: 49
related_speech_day: 49
---

**"Complex tasks have structure—dependency graphs make it explicit."**

## 1. Introduction

When agents tackle complex tasks, subtasks often have dependencies: "Research X before writing about X", "Collect data before analyzing". Modeling these as a DAG enables parallel execution and clear ordering.

### Why Dependency Graphs?

```
Task: "Write a research report on AI safety"

Naive: Sequential execution
1. Search papers (30s)
2. Read paper 1 (20s)
3. Read paper 2 (20s)
4. Read paper 3 (20s)
5. Synthesize (30s)
6. Write draft (40s)
7. Format (10s)
Total: 170s

With dependency graph:
1. Search papers (30s)
   ├── Read paper 1 (20s) ─┐
   ├── Read paper 2 (20s) ─┼── Synthesize (30s) → Write (40s) → Format (10s)
   └── Read paper 3 (20s) ─┘
Total: 30 + 20 + 30 + 40 + 10 = 130s (23% faster)
```

## 2. Task Graph Definition

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional
from enum import Enum
import asyncio

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentTask:
    """A task in the agent's execution graph."""
    id: str
    description: str
    execute: Callable  # async function
    dependencies: List[str] = field(default_factory=list)
    
    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    
    # Execution hints
    priority: int = 0
    timeout: int = 60
    retries: int = 2


class TaskGraph:
    """DAG of agent tasks with dependency tracking."""
    
    def __init__(self):
        self.tasks: Dict[str, AgentTask] = {}
    
    def add_task(self, task: AgentTask):
        self.tasks[task.id] = task
    
    def validate(self) -> bool:
        """Check for cycles using DFS coloring."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {tid: WHITE for tid in self.tasks}
        
        def has_cycle(task_id):
            if colors[task_id] == GRAY:
                return True
            if colors[task_id] == BLACK:
                return False
            
            colors[task_id] = GRAY
            for dep in self.tasks[task_id].dependencies:
                if dep not in self.tasks:
                    raise ValueError(f"Unknown dependency: {dep}")
                if has_cycle(dep):
                    return True
            colors[task_id] = BLACK
            return False
        
        return not any(has_cycle(tid) for tid in self.tasks)
    
    def get_ready_tasks(self, completed: set) -> List[str]:
        """Get tasks whose dependencies are all complete."""
        ready = []
        for tid, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                if all(dep in completed for dep in task.dependencies):
                    ready.append(tid)
        return ready
    
    def get_execution_order(self) -> List[str]:
        """Topological sort."""
        in_degree = {tid: len(t.dependencies) for tid, t in self.tasks.items()}
        dependents = {tid: [] for tid in self.tasks}
        
        for tid, task in self.tasks.items():
            for dep in task.dependencies:
                dependents[dep].append(tid)
        
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            # Sort by priority
            queue.sort(key=lambda t: -self.tasks[t].priority)
            current = queue.pop(0)
            order.append(current)
            
            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return order
```

## 3. Task Graph Executor

```python
class TaskGraphExecutor:
    """Execute task graph with parallel execution."""
    
    def __init__(self, graph: TaskGraph, max_parallel: int = 4):
        self.graph = graph
        self.max_parallel = max_parallel
        self.completed: set = set()
        self.failed: set = set()
    
    async def run(self) -> Dict[str, Any]:
        """Execute all tasks respecting dependencies."""
        if not self.graph.validate():
            raise ValueError("Task graph has cycles!")
        
        while True:
            ready = self.graph.get_ready_tasks(self.completed)
            
            if not ready:
                if len(self.completed) + len(self.failed) == len(self.graph.tasks):
                    break
                if self.failed:
                    break
                await asyncio.sleep(0.1)
                continue
            
            # Execute ready tasks in parallel
            batch = ready[:self.max_parallel]
            results = await asyncio.gather(
                *[self._execute_task(tid) for tid in batch],
                return_exceptions=True
            )
            
            for tid, result in zip(batch, results):
                if isinstance(result, Exception):
                    self.failed.add(tid)
                    self.graph.tasks[tid].status = TaskStatus.FAILED
                    self.graph.tasks[tid].error = str(result)
                else:
                    self.completed.add(tid)
        
        return {
            tid: task.result
            for tid, task in self.graph.tasks.items()
            if task.status == TaskStatus.SUCCESS
        }
    
    async def _execute_task(self, task_id: str):
        """Execute single task with retries."""
        task = self.graph.tasks[task_id]
        task.status = TaskStatus.RUNNING
        
        # Gather dependency results
        dep_results = {
            dep: self.graph.tasks[dep].result
            for dep in task.dependencies
        }
        
        for attempt in range(task.retries):
            try:
                task.result = await asyncio.wait_for(
                    task.execute(dep_results),
                    timeout=task.timeout
                )
                task.status = TaskStatus.SUCCESS
                return task.result
            except Exception as e:
                if attempt == task.retries - 1:
                    raise
                await asyncio.sleep(1)
```

## 4. LLM-Driven Task Decomposition

```python
class TaskPlanner:
    """Use LLM to decompose tasks into dependency graph."""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def plan(self, goal: str) -> TaskGraph:
        """Decompose goal into task graph."""
        prompt = f"""Decompose this goal into subtasks with dependencies.

Goal: {goal}

Output JSON:
{{
  "tasks": [
    {{"id": "task_1", "description": "...", "dependencies": []}},
    {{"id": "task_2", "description": "...", "dependencies": ["task_1"]}}
  ]
}}

Rules:
- Each task should be atomic and achievable
- Dependencies must reference earlier task IDs
- No circular dependencies
"""
        
        response = await self.llm.generate(prompt)
        plan = self._parse_response(response)
        
        return self._build_graph(plan)
    
    def _parse_response(self, response: str) -> dict:
        import json
        return json.loads(response)
    
    def _build_graph(self, plan: dict) -> TaskGraph:
        graph = TaskGraph()
        
        for task_data in plan["tasks"]:
            task = AgentTask(
                id=task_data["id"],
                description=task_data["description"],
                execute=self._create_executor(task_data["description"]),
                dependencies=task_data.get("dependencies", [])
            )
            graph.add_task(task)
        
        return graph
    
    def _create_executor(self, description: str) -> Callable:
        """Create async executor for task."""
        async def execute(deps: Dict[str, Any]) -> Any:
            context = "\n".join(
                f"Result of {k}: {v}" for k, v in deps.items()
            )
            
            prompt = f"""Execute this task:
Task: {description}

Context from previous tasks:
{context if context else 'None'}

Provide the result."""
            
            return await self.llm.generate(prompt)
        
        return execute
```

## 5. Example: Research Report Agent

```python
async def research_report_example():
    """Build and execute research report task graph."""
    
    graph = TaskGraph()
    
    # Define tasks
    async def search_papers(deps):
        # Simulate paper search
        return ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
    
    async def read_paper(deps, paper_id):
        papers = deps["search"]
        return f"Summary of {papers[paper_id]}"
    
    async def synthesize(deps):
        summaries = [deps[f"read_{i}"] for i in range(3)]
        return f"Synthesis: {', '.join(summaries)}"
    
    async def write_draft(deps):
        synthesis = deps["synthesize"]
        return f"Draft based on: {synthesis}"
    
    async def format_report(deps):
        draft = deps["write"]
        return f"Formatted: {draft}"
    
    # Build graph
    graph.add_task(AgentTask(
        id="search",
        description="Search for relevant papers",
        execute=search_papers
    ))
    
    for i in range(3):
        graph.add_task(AgentTask(
            id=f"read_{i}",
            description=f"Read paper {i}",
            execute=lambda deps, i=i: read_paper(deps, i),
            dependencies=["search"]
        ))
    
    graph.add_task(AgentTask(
        id="synthesize",
        description="Synthesize findings",
        execute=synthesize,
        dependencies=["read_0", "read_1", "read_2"]
    ))
    
    graph.add_task(AgentTask(
        id="write",
        description="Write draft",
        execute=write_draft,
        dependencies=["synthesize"]
    ))
    
    graph.add_task(AgentTask(
        id="format",
        description="Format report",
        execute=format_report,
        dependencies=["write"]
    ))
    
    # Execute
    executor = TaskGraphExecutor(graph, max_parallel=3)
    results = await executor.run()
    
    return results["format"]


# Run
result = asyncio.run(research_report_example())
print(result)
```

## 6. Dynamic Graph Modification

```python
class DynamicTaskGraph(TaskGraph):
    """Support runtime graph modifications."""
    
    async def add_task_at_runtime(
        self,
        task: AgentTask,
        executor: TaskGraphExecutor
    ):
        """Add task during execution."""
        # Validate dependencies exist
        for dep in task.dependencies:
            if dep not in self.tasks:
                raise ValueError(f"Dependency {dep} not found")
        
        # Check dependencies are complete
        for dep in task.dependencies:
            if self.tasks[dep].status != TaskStatus.SUCCESS:
                raise ValueError(f"Dependency {dep} not complete")
        
        self.tasks[task.id] = task
        # Executor will pick it up in next iteration
    
    def remove_task(self, task_id: str):
        """Remove task if not started."""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        if task.status != TaskStatus.PENDING:
            raise ValueError("Cannot remove started task")
        
        # Check no other tasks depend on it
        for other in self.tasks.values():
            if task_id in other.dependencies:
                raise ValueError(f"Task {other.id} depends on {task_id}")
        
        del self.tasks[task_id]
```

## 7. Visualization

```python
def visualize_graph(graph: TaskGraph) -> str:
    """Generate Mermaid diagram."""
    lines = ["graph TD"]
    
    for tid, task in graph.tasks.items():
        # Node with status color
        status_color = {
            TaskStatus.PENDING: "white",
            TaskStatus.RUNNING: "yellow",
            TaskStatus.SUCCESS: "green",
            TaskStatus.FAILED: "red"
        }[task.status]
        
        lines.append(f"    {tid}[{task.description}]")
        
        for dep in task.dependencies:
            lines.append(f"    {dep} --> {tid}")
    
    return "\n".join(lines)
```

## 8. Connection to Course Schedule

Agent task graphs are Course Schedule in action:

| Course Schedule | Agent Tasks |
|----------------|-------------|
| Course | Task |
| Prerequisite | Dependency |
| Can finish? | Can complete goal? |
| Topological order | Execution order |
| Parallel courses | Parallel tasks |

## 9. Key Takeaways

1. **Model complex tasks as DAGs** for clarity
2. **Topological sort** ensures correct order
3. **Parallel execution** speeds up independent tasks
4. **LLM planning** can decompose goals automatically
5. **Dynamic modification** handles evolving requirements

---

**Originally published at:** [arunbaby.com/ai-agents/0049-dependency-graphs-for-agents](https://www.arunbaby.com/ai-agents/0049-dependency-graphs-for-agents/)

*If you found this helpful, consider sharing it with others who might benefit.*
