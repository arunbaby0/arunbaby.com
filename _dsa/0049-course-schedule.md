---
title: "Course Schedule (Topological Sort)"
day: 49
collection: dsa
categories:
  - dsa
tags:
  - graph
  - topological-sort
  - bfs
  - dfs
  - cycle-detection
difficulty: Medium
subdomain: "Graph Algorithms"
tech_stack: Python
scale: "O(V + E)"
companies: Amazon, Google, Uber, Coursera
related_ml_day: 49
related_speech_day: 49
related_agents_day: 49
---

**"Putting on socks before shoes is a dependency. Topological sort is just the math of getting dressed."**

## 1. Introduction: The Prerequisite Problem

Imagine you are a university student planning your major.
- You want to take **Advanced AI (Course A)**.
- But A requires **Machine Learning (Course B)**.
- And B requires **Calculus (Course C)**.
- And C requires **A**.

Wait. If C requires A, and A requires C... you can never graduate! This is a **Cycle**.

The **Course Schedule** problem asks two fundamental questions:
1. **Feasibility**: Given a set of courses and prerequisites, is it even *possible* to finish all of them? (i.e., Is there a cycle?)
2. **Ordering**: If it is possible, in what order should I take them?

This is not just about college. This is how:
- `pip install` resolves package dependencies.
- `make` determines which files to compile first.
- `Task scheduling` systems decide which jobs to run.

---

## 2. Modeling the Problem: The Graph

We can model this as a **Directed Graph**:
- **Nodes (V)**: Courses (0, 1, 2...).
- **Edges (E)**: Prerequisites. An edge `B -> A` means "Take B before A".

### Example 1 (Linear)
`[1, 0]` means "To take 1, you must first take 0".
Graph: `0 -> 1`
Order: `0, 1`. Valid!

### Example 2 (Branching)
`[1, 0], [2, 0], [3, 1], [3, 2]`
Graph:
```
      0
     / \
    1   2
     \ /
      3
```
Order: `0, 1, 2, 3` or `0, 2, 1, 3`. Both are valid!

### Example 3 (Cycle)
`[1, 0], [0, 1]`
Graph: `0 -> 1 -> 0`
Order: Impossible.

---

## 3. The Algorithm: Kahn's Algorithm (BFS)

There are two main ways to solve this: DFS and BFS (Kahn's Algorithm). Kahn's is often more intuitive for "ordering" problems.

### 3.1 The Concept: In-Degrees
The **In-Degree** of a node is the number of incoming edges (prerequisites).
- If In-Degree is `0`: This course has **no prerequisites**. You can take it right now!

### 3.2 The Steps

1. **Calculate In-Degrees**: Count prerequisites for every course.
2. **Find Starters**: Put all courses with `In-Degree == 0` into a Queue.
3. **Process Queue**:
   - Pop a course `curr` (Take the course).
   - Add it to our `sorted_order` list.
   - **"Remove" the course**: Go to all its neighbors (courses that required `curr`). Decrease their In-Degree by 1.
   - **Check for new Starters**: If a neighbor's In-Degree becomes 0, add it to the Queue!
4. **Conclusion**:
   - If `len(sorted_order) == num_courses`, we succeeded!
   - If `len(sorted_order) < num_courses`, there is a cycle (some courses never reached degree 0).

---

## 4. Visual Walkthrough

Dependency: `[[1,0], [2,0], [3,1], [3,2]]` (Courses: 0, 1, 2, 3)

**Step 1: Init**
- In-Degrees: `{0: 0, 1: 1, 2: 1, 3: 2}`
- Adjacency List: `{0: [1, 2], 1: [3], 2: [3], 3: []}`
- Queue: `[0]` (Only 0 has in-degree 0)

**Step 2: Processing**
- **Pop 0**. Order: `[0]`.
  - Neighbors of 0 are `1` and `2`.
  - Decrement 1: In-Degree becomes `0`. **Add 1 to Queue**.
  - Decrement 2: In-Degree becomes `0`. **Add 2 to Queue**.
- Queue: `[1, 2]`

- **Pop 1**. Order: `[0, 1]`.
  - Neighbors of 1 is `3`.
  - Decrement 3: In-Degree becomes `1`. (Not 0 yet!)
- Queue: `[2]`

- **Pop 2**. Order: `[0, 1, 2]`.
  - Neighbor of 2 is `3`.
  - Decrement 3: In-Degree becomes `0`. **Add 3 to Queue**.
- Queue: `[3]`

- **Pop 3**. Order: `[0, 1, 2, 3]`.
  - No neighbors.
- Queue: `[]`. Done.

Count is 4. Total courses 4. **Success!**

---

## 5. Implementation (Python)

```python
from collections import deque

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 1. Build Graph and Indegree array
        # graph[A] contains list of courses that depend on A
        graph = {i: [] for i in range(numCourses)}
        indegree = [0] * numCourses
        
        for dest, src in prerequisites:
            graph[src].append(dest)
            indegree[dest] += 1
            
        # 2. Init Queue with courses having no prereqs
        queue = deque()
        for i in range(numCourses):
            if indegree[i] == 0:
                queue.append(i)
        
        # 3. Process BFS
        taken_courses = 0
        while queue:
            course = queue.popleft()
            taken_courses += 1
            
            # "Complete" this course, notifying dependents
            for neighbor in graph[course]:
                indegree[neighbor] -= 1
                # If all prereqs for neighbor are done, it's ready
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 4. Check results
        return taken_courses == numCourses
```

---

## 6. Time & Space Complexity

- **Time**: `O(V + E)`
  - `V` (Vertices): Number of courses.
  - `E` (Edges): Number of dependencies.
  - We look at every node once and every dependency once.
  
- **Space**: `O(V + E)`
  - We store the graph (Adjacency list) and the Indegree array.

---

## 7. Extensions: Topological Sort Applications

Understanding this algorithm unlocks solutions to many other problems:

1. **Build Systems**: Determining compilation order (Makefile targets).
2. **Spreadsheet Evaluation**: Order of cell formula calculation.
3. **Data Pipelines**: ETL jobs (Extract before Transform before Load).
4. **React `useEffect`**: Determining the execution order of side effects.

The concept is universal: **Dependencies define a Directed Acyclic Graph (DAG)**. If it's not acyclic (has a cycle), the system deadlocks. If it is acyclic, there is a valid linear ordering.

---

**Originally published at:** [arunbaby.com/dsa/0049-course-schedule](https://www.arunbaby.com/dsa/0049-course-schedule)

*If you found this helpful, consider sharing it with others who might benefit.*
