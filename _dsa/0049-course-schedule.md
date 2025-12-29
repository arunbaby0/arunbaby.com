---
title: "Course Schedule (Topological Sort)"
day: 49
related_ml_day: 49
related_speech_day: 49
related_agents_day: 49
collection: dsa
categories:
 - dsa
tags:
 - graph
 - topological-sort
 - bfs
 - kahn-algorithm
 - medium
difficulty: Medium
subdomain: "Graph Algorithms"
tech_stack: Python
scale: "Scheduling 100k jobs with dependencies"
companies: Amazon, Google, Uber, Netflix
---

**"You can't build the roof before you pour the foundation."**

## 1. Problem Statement

This is the canonical "Dependency Resolution" problem.
There are `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [a, b]` indicates that you **must** take course `b` first if you want to take course `a`.

Return `true` if you can finish all courses. Otherwise, return `false`.

**Example 1:**
- Input: `numCourses = 2`, `prerequisites = [[1,0]]`
- Output: `true`
- Explanation: To take course 1 you should have finished course 0. So it is possible.

**Example 2:**
- Input: `numCourses = 2`, `prerequisites = [[1,0],[0,1]]`
- Output: `false`
- Explanation: To take 1 you need 0. To take 0 you need 1. Impossible (Cycle).

---

## 2. Understanding the Problem

### 2.1 Directed Acyclic Graphs (DAGs)
This problem models a **Directed Graph**.
- Nodes = Courses.
- Edges = Dependencies (`b -> a` implies `b` creates `a`).

The question "Can I finish?" is synonymous with: **"Does this graph contain a Cycle?"**
If there is a cycle (`A -> B -> A`), you can never start A or B.
If there is no cycle (a DAG), there always exists a valid linear ordering, called a **Topological Sort**.

### 2.2 The "Indegree" Intuition
How do you know which course to take first?
Simple: Take the one with **zero prerequisites**.
- If Course 0 has no prerequisites (Indegree = 0), take it.
- Once taken, Course 0 is "done". Now, any course that *only* needed Course 0 might now have its prerequisites satisfied.
- We effectively "remove" Course 0 and its outgoing edges from the graph.
- Repeat until empty.

---

## 3. Approach 1: DFS (Cycle Detection)

We can process each node with DFS.
- States: `Unvisited` (0), `Visiting` (1), `Visited` (2).
- If we encounter a node marked `Visiting`, we found a back-edge -> **Cycle**.

**Pros**: Easy to write.
**Cons**: Harder to generate the actual sort order (requires reversing the post-order). Large recursion depth.

---

## 4. Approach 2: Kahn's Algorithm (BFS) -- The Standard

This approach directly simulates the "peeling onion" strategy.

### Algorithm
1. **Build Graph**: Adjacency list + Indegree array.
 - `Adj[u] = [v1, v2]`
 - `Indegree[v1]++`, `Indegree[v2]++`
2. **Initialize Queue**: Add all nodes with `Indegree == 0` to Queue.
3. **Process**:
 - While Queue is not empty:
 - Pop `u`. Add to list of `taken_courses`.
 - For each child `v` of `u`:
 - `Indegree[v]--` (We satisfied one prerequisite).
 - If `Indegree[v] == 0`, push `v` to Queue.
4. **Check**: If `count(taken_courses) == numCourses`, return True. Else, False (Cycle detected, some nodes never reached Indegree 0).

---

## 5. Implementation: Kahn's Algorithm

``python
from collections import deque

class Solution:
 def canFinish(self, numCourses: int, prerequisites: list[list[int]]) -> bool:
 # 1. Initialize Graph
 adj = [[] for _ in range(numCourses)]
 indegree = [0] * numCourses
 
 # 2. Build Graph (Time: O(E), Space: O(E))
 for dest, src in prerequisites:
 # src -> dest (take src first)
 adj[src].append(dest)
 indegree[dest] += 1
 
 # 3. Add seeds (Time: O(V))
 queue = deque()
 for i in range(numCourses):
 if indegree[i] == 0:
 queue.append(i)
 
 processed_count = 0
 
 # 4. BFS Traversal (Time: O(V + E))
 while queue:
 node = queue.popleft()
 processed_count += 1
 
 for neighbor in adj[node]:
 indegree[neighbor] -= 1
 if indegree[neighbor] == 0:
 queue.append(neighbor)
 
 # 5. Final check
 return processed_count == numCourses

# Example Usage
sol = Solution()
print(sol.canFinish(2, [[1,0]])) # True
print(sol.canFinish(2, [[1,0], [0,1]])) # False
``

---

## 6. Testing Strategy

### Test Case 1: Disconnected Components
`0 -> 1` and `2 -> 3`.
- Queue starts with `{0, 2}`.
- Pop 0 -> Add 1 to Queue.
- Pop 2 -> Add 3 to Queue.
- All processed. **Valid**.

### Test Case 2: Deadlock (Cycle)
`0 -> 1 -> 0`.
- Indegree: `[1, 1]`.
- Queue starts Empty.
- Processed Count = 0.
- Returns **False**.

---

## 7. Complexity Analysis

- **Time**: O(V + E).
 - `V`: Initializing lists.
 - `E`: Building graph edges.
 - BFS loop visits every node (`V`) and every edge (`E`) exactly once.
- **Space**: O(V + E).
 - To store the Adjacency List.

This is asymptotically optimal.

---

## 8. Production Considerations

This algorithm is the backbone of **Build Systems** (Make, Maven, Bazel).
- **Parallel Dependencies**:
 The Queue length represents the level of **parallelism**.
 If Queue has 5 items, it means 5 tasks are ready to run *simultaneously* on 5 worker threads.
 This is how `make -j8` works!

---

## 9. Connections to ML Systems

This directly maps to **DAG Pipeline Orchestration** (ML System Design).
- **Airflow / Kubeflow**: These tools literally execute Kahn's Algorithm.
 - Nodes = ETL Tasks (Load Data, Train Model).
 - Edges = Data dependencies.
 - The Scheduler puts tasks with `Indegree 0` into the worker pool.

---

## 10. Interview Strategy

1. **Identify Graph**: Start by saying "This is a dependency problem, which can be modeled as a Graph."
2. **Define Edge Direction**: Be careful! "Prerequisite `[a, b]` means `b -> a`". Getting this backward ruins the code.
3. **Mention Cycle**: Explicitly mention "If there's a cycle, we can't finish."
4. **Compare DFS vs BFS**: "I prefer Kahn's (BFS) because it's iterative (no stack overflow) and easy to extend to parallel execution logic."

---

## 11. Key Takeaways

1. **Indegree is Key**: It represents "Unsatisfied Constraints".
2. **Topological Sort order is not unique**: If queue has `[A, B]`, `AB` and `BA` are both valid sorts.
3. **Cycle Detection**: If `processed < total`, the remaining nodes are locked in a cycle.

---

**Originally published at:** [arunbaby.com/dsa/0049-course-schedule](https://www.arunbaby.com/dsa/0049-course-schedule/)

*If you found this helpful, consider sharing it with others who might benefit.*
