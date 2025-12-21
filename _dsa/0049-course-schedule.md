---
title: "Course Schedule (Topological Sort)"
day: 49
collection: dsa
categories:
  - dsa
tags:
  - graphs
  - topological-sort
  - dfs
  - bfs
  - cycle-detection
  - dag
difficulty: Medium
subdomain: "Graphs"
tech_stack: Python
scale: "O(V + E) time and space"
companies: Google, Meta, Amazon, Microsoft, Uber
related_ml_day: 49
related_speech_day: 49
related_agents_day: 49
---

**"To take course B, first complete course A—dependency ordering at its core."**

## 1. Problem Statement

There are `numCourses` courses labeled `0` to `numCourses-1`. You're given an array `prerequisites` where `prerequisites[i] = [a, b]` indicates you must take course `b` before course `a`.

Return `true` if you can finish all courses, otherwise return `false`.

**Example 1:**
```
numCourses = 2
prerequisites = [[1, 0]]
Output: true
Explanation: Take course 0, then course 1.
```

**Example 2:**
```
numCourses = 2
prerequisites = [[1, 0], [0, 1]]
Output: false
Explanation: Circular dependency—impossible!
```

## 2. Understanding the Problem

This is **cycle detection in a directed graph**. If there's a cycle, some courses can't be completed (they depend on each other circularly).

```
Course Schedule I: Can you finish? (cycle detection)
Course Schedule II: In what order? (topological sort)
```

### Graph Representation

```
prerequisites = [[1,0], [2,0], [3,1], [3,2]]

Graph:
0 → 1 → 3
 ↘ 2 ↗

Topological order: 0, 1, 2, 3 (or 0, 2, 1, 3)
```

## 3. Approach 1: DFS with Cycle Detection

```python
from typing import List
from collections import defaultdict

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Detect if graph has a cycle using DFS.
        
        State tracking:
        - 0: unvisited
        - 1: visiting (in current DFS path)
        - 2: visited (completed)
        
        Cycle exists if we reach a node that's currently being visited.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for course, prereq in prerequisites:
            graph[prereq].append(course)
        
        # 0: unvisited, 1: visiting, 2: visited
        state = [0] * numCourses
        
        def has_cycle(course):
            if state[course] == 1:
                return True  # Back edge = cycle
            if state[course] == 2:
                return False  # Already processed
            
            state[course] = 1  # Mark as visiting
            
            for next_course in graph[course]:
                if has_cycle(next_course):
                    return True
            
            state[course] = 2  # Mark as visited
            return False
        
        # Check all courses (graph may be disconnected)
        for course in range(numCourses):
            if has_cycle(course):
                return False
        
        return True
```

## 4. Approach 2: Kahn's Algorithm (BFS)

```python
from collections import deque

class Solution:
    def canFinish_bfs(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Kahn's algorithm: BFS-based topological sort.
        
        1. Count in-degrees for each node
        2. Start with nodes having in-degree 0
        3. Process each node, reduce neighbors' in-degrees
        4. If all nodes processed, no cycle
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build graph and in-degree count
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        # Start with courses having no prerequisites
        queue = deque()
        for course in range(numCourses):
            if in_degree[course] == 0:
                queue.append(course)
        
        completed = 0
        
        while queue:
            course = queue.popleft()
            completed += 1
            
            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        # If we completed all courses, no cycle
        return completed == numCourses
```

## 5. Course Schedule II (Full Topological Sort)

```python
def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Return a valid course order, or [] if impossible.
    """
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([c for c in range(numCourses) if in_degree[c] == 0])
    order = []
    
    while queue:
        course = queue.popleft()
        order.append(course)
        
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return order if len(order) == numCourses else []
```

## 6. Why Two Approaches?

| Aspect | DFS | Kahn's (BFS) |
|--------|-----|--------------|
| Cycle detection | State coloring | Count mismatch |
| Order output | Reverse post-order | Direct |
| Stack usage | Recursion stack | Queue |
| Implementation | Slightly complex | Simpler |

Both are O(V + E), choose based on preference or constraints.

## 7. Visualizing the DFS States

```
Course graph: 0 → 1 → 2
                ↘ 3

DFS from 0:
- Visit 0: state[0] = 1 (visiting)
  - Visit 1: state[1] = 1
    - Visit 2: state[2] = 1
    - No neighbors, state[2] = 2 (done)
  - state[1] = 2
  - Visit 3: state[3] = 1
  - No neighbors, state[3] = 2
- state[0] = 2

All nodes visited, no back edges → No cycle!

With cycle: 0 → 1 → 2 → 0
DFS from 0:
- Visit 0: state[0] = 1
  - Visit 1: state[1] = 1
    - Visit 2: state[2] = 1
      - Visit 0: state[0] = 1 (already visiting!)
      → Cycle detected!
```

## 8. Common Variations

### 8.1 Parallel Course Completion

```python
def minimumSemesters(n: int, relations: List[List[int]]) -> int:
    """
    Minimum semesters to complete all courses.
    (Take max parallel courses each semester)
    """
    graph = defaultdict(list)
    in_degree = [0] * (n + 1)
    
    for prereq, course in relations:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([c for c in range(1, n+1) if in_degree[c] == 0])
    semesters = 0
    completed = 0
    
    while queue:
        semesters += 1
        # Take all available courses this semester
        for _ in range(len(queue)):
            course = queue.popleft()
            completed += 1
            
            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
    
    return semesters if completed == n else -1
```

### 8.2 All Topological Orders

```python
def allTopologicalOrders(n: int, edges: List[List[int]]) -> List[List[int]]:
    """
    Generate all valid topological orderings.
    (Backtracking)
    """
    graph = defaultdict(list)
    in_degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    result = []
    current = []
    visited = [False] * n
    
    def backtrack():
        if len(current) == n:
            result.append(current[:])
            return
        
        for node in range(n):
            if not visited[node] and in_degree[node] == 0:
                # Choose
                visited[node] = True
                current.append(node)
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                
                backtrack()
                
                # Unchoose
                visited[node] = False
                current.pop()
                for neighbor in graph[node]:
                    in_degree[neighbor] += 1
    
    backtrack()
    return result
```

## 9. Testing

```python
def test_course_schedule():
    s = Solution()
    
    # Basic - can finish
    assert s.canFinish(2, [[1, 0]]) == True
    
    # Cycle
    assert s.canFinish(2, [[1, 0], [0, 1]]) == False
    
    # Longer chain
    assert s.canFinish(4, [[1,0], [2,1], [3,2]]) == True
    
    # Disconnected - all independent  
    assert s.canFinish(3, []) == True
    
    # Self-loop
    assert s.canFinish(1, [[0, 0]]) == False
    
    print("All tests passed!")
```

## 10. Connection to DAG Pipeline Orchestration

Course Schedule is the algorithmic core of **pipeline orchestration**:

| Course Schedule | ML Pipeline |
|----------------|-------------|
| Course | Task/Step |
| Prerequisite | Dependency |
| Cycle = Impossible | Deadlock |
| Topological order | Execution order |

When you run an ML pipeline, the orchestrator performs exactly this algorithm to determine task execution order.

---

**Originally published at:** [arunbaby.com/dsa/0049-course-schedule](https://www.arunbaby.com/dsa/0049-course-schedule/)

*If you found this helpful, consider sharing it with others who might benefit.*
