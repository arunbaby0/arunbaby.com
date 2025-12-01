---
title: "Course Schedule (Topological Sort)"
day: 31
collection: dsa
categories:
  - dsa
tags:
  - graph
  - topological sort
  - dfs
  - cycle detection
difficulty: Medium
related_ml_day: 31
related_speech_day: 31
---

**"Can you finish all courses given their prerequisites?"**

## 1. Problem Statement

There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [ai, bi]` indicates that you must take course `bi` first if you want to take course `ai`.

Return `true` if you can finish all courses. Otherwise, return `false`.

**Example 1:**
```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: Take course 0, then course 1.
```

**Example 2:**
```
Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: Circular dependency (cycle).
```

**This is a cycle detection problem in a directed graph!**

## 2. DFS Solution (Cycle Detection)

**Intuition:**
- Build a directed graph: `course_a -> course_b` means "a depends on b".
- Use DFS to detect cycles.
- If there's a cycle, courses can't be completed.

**States during DFS:**
- **0 (White):** Unvisited.
- **1 (Gray):** Visiting (currently in DFS stack).
- **2 (Black):** Visited (DFS completed).

**Cycle Detection:** If we encounter a **Gray** node during DFS, there's a cycle.

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # Build adjacency list
        graph = defaultdict(list)
        for course, prereq in prerequisites:
            graph[course].append(prereq)
        
        # States: 0 = unvisited, 1 = visiting, 2 = visited
        state = [0] * numCourses
        
        def has_cycle(course):
            if state[course] == 1:  # Currently visiting → cycle!
                return True
            if state[course] == 2:  # Already processed
                return False
            
            # Mark as visiting
            state[course] = 1
            
            # Visit all prerequisites
            for prereq in graph[course]:
                if has_cycle(prereq):
                    return True
            
            # Mark as visited
            state[course] = 2
            return False
        
        # Check all courses
        for course in range(numCourses):
            if has_cycle(course):
                return False
        
        return True
```

**Time Complexity:** \\(O(V + E)\\) where V = courses, E = prerequisites.
**Space Complexity:** \\(O(V + E)\\) for graph + \\(O(V)\\) for recursion stack.

## 3. BFS Solution (Kahn's Algorithm - Topological Sort)

**Intuition:**
- Count in-degree (number of prerequisites) for each course.
- Process courses with in-degree 0 (no prerequisites).
- Remove processed courses and update in-degrees.
- If all courses are processed, return true.

```python
from collections import deque, defaultdict

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # Build graph and in-degree count
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)  # prereq -> course
            in_degree[course] += 1
        
        # Queue of courses with no prerequisites
        queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
        processed = 0
        
        while queue:
            course = queue.popleft()
            processed += 1
            
            # Remove this course and update dependent courses
            for dependent in graph[course]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return processed == numCourses
```

**Time Complexity:** \\(O(V + E)\\).
**Space Complexity:** \\(O(V + E)\\).

## 4. Course Schedule II (Return the Order)

**Problem:** Return a valid course order. If impossible, return `[]`.

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
        order = []
        
        while queue:
            course = queue.popleft()
            order.append(course)
            
            for dependent in graph[course]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return order if len(order) == numCourses else []
```

## Deep Dive: Topological Sort - Why It Works

**Topological Ordering:** A linear ordering of vertices such that for every directed edge \\(u \to v\\), \\(u\\) comes before \\(v\\).

**Theorem:** A topological ordering exists **if and only if** the graph is a **DAG** (Directed Acyclic Graph).

**Proof:**
- **⇒ (If topological ordering exists, then no cycles):**
  - Suppose there's a cycle \\(v_1 \to v_2 \to \ldots \to v_k \to v_1\\).
  - In a topological ordering, \\(v_1\\) must come before \\(v_2\\), \\(v_2\\) before \\(v_3\\), ..., \\(v_k\\) before \\(v_1\\).
  - This implies \\(v_1\\) comes before \\(v_1\\) → **Contradiction!**
  
- **⇐ (If no cycles, then topological ordering exists):**
  - In a DAG, there exists at least one vertex with in-degree 0 (no incoming edges).
  - Remove this vertex and repeat. This gives a topological ordering.

## Deep Dive: Kahn's Algorithm Correctness

**Algorithm:**
1. Find all vertices with in-degree 0.
2. Remove them (add to result) and update neighbors' in-degrees.
3. Repeat until no more vertices with in-degree 0.

**Why it works:**
- Vertices with in-degree 0 have no dependencies → safe to process.
- Removing a vertex is equivalent to marking it as "done".
- If the graph has a cycle, there will always be vertices with in-degree > 0 (can't find a starting point).

**Invariant:** At each step, processed vertices form a valid prefix of a topological ordering.

## Deep Dive: DFS-based Topological Sort

**Idea:** Use DFS and add vertices to the result in **post-order** (after visiting all descendants).

```python
def topological_sort_dfs(graph, num_vertices):
    visited = [False] * num_vertices
    stack = []
    
    def dfs(v):
        visited[v] = True
        for neighbor in graph[v]:
            if not visited[neighbor]:
                dfs(neighbor)
        stack.append(v)  # Add after visiting all descendants
    
    for v in range(num_vertices):
        if not visited[v]:
            dfs(v)
    
    return stack[::-1]  # Reverse to get topological order
```

**Why reverse?**
- DFS post-order gives vertices in **decreasing finish time**.
- In a topological ordering, vertices with higher finish time should come first.

## Deep Dive: Minimum Number of Semesters

**Problem:** What's the minimum number of semesters needed to complete all courses?

**Solution:** This is finding the **longest path** in the DAG.

```python
def minimumSemesters(numCourses, prerequisites):
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([(i, 1) for i in range(numCourses) if in_degree[i] == 0])  # (course, semester)
    processed = 0
    max_semester = 0
    
    while queue:
        course, semester = queue.popleft()
        processed += 1
        max_semester = max(max_semester, semester)
        
        for dependent in graph[course]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append((dependent, semester + 1))
    
    return max_semester if processed == numCourses else -1
```

**Time Complexity:** \\(O(V + E)\\).

## Deep Dive: Parallel Course Scheduling (Load Balancing)

**Problem:** You have \\(K\\) workers. What's the minimum time to complete all courses?

**Approach: DP on DAG**
```python
def minTimeToFinish(numCourses, prerequisites, time, K):
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # dp[course] = earliest time this course can start
    dp = [0] * numCourses
    
    # Topological sort with time tracking
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    
    while queue:
        # Process K courses in parallel
        current_batch = []
        for _ in range(min(K, len(queue))):
            if queue:
                current_batch.append(queue.popleft())
        
        for course in current_batch:
            for dependent in graph[course]:
                dp[dependent] = max(dp[dependent], dp[course] + time[course])
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
    
    return max(dp[i] + time[i] for i in range(numCourses))
```

## Deep Dive: All Possible Topological Orderings

**Problem:** Print all valid course orderings.

**Approach: Backtracking**
```python
def allTopologicalSorts(graph, num_vertices):
    in_degree = [0] * num_vertices
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    
    result = []
    
    def backtrack(current_order, remaining_in_degree):
        # Find all vertices with in-degree 0
        available = [v for v in range(num_vertices) if remaining_in_degree[v] == 0 and v not in current_order]
        
        if not available:
            if len(current_order) == num_vertices:
                result.append(current_order[:])
            return
        
        for v in available:
            # Choose v
            current_order.append(v)
            
            # Update in-degrees
            new_in_degree = remaining_in_degree[:]
            new_in_degree[v] = -1  # Mark as used
            for neighbor in graph[v]:
                new_in_degree[neighbor] -= 1
            
            # Recurse
            backtrack(current_order, new_in_degree)
            
            # Unchoose
            current_order.pop()
    
    backtrack([], in_degree)
    return result
```

**Time Complexity:** \\(O(V! \cdot E)\\) in worst case (exponential).

## Deep Dive: Lexicographically Smallest Topological Sort

**Problem:** Among all valid orderings, return the lexicographically smallest.

**Solution:** Use a min-heap instead of queue in Kahn's algorithm.

```python
import heapq

def lexicographicallySmallestOrder(numCourses, prerequisites):
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Min-heap instead of queue
    heap = [i for i in range(numCourses) if in_degree[i] == 0]
    heapq.heapify(heap)
    
    order = []
    while heap:
        course = heapq.heappop(heap)
        order.append(course)
        
        for dependent in graph[course]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                heapq.heappush(heap, dependent)
    
    return order if len(order) == numCourses else []
```

**Time Complexity:** \\(O((V + E) \log V)\\) due to heap operations.

## Deep Dive: Detecting Strongly Connected Components (Kosaraju's Algorithm)

**Strongly Connected Component (SCC):** A maximal subgraph where every vertex is reachable from every other vertex.

**In the context of courses:** Courses in the same SCC form a circular dependency (impossible to complete).

**Kosaraju's Algorithm:**
1. Perform DFS on original graph, record finish times.
2. Reverse the graph.
3. Perform DFS on reversed graph in decreasing finish time order.
4. Each DFS tree in step 3 is an SCC.

```python
def findSCC(graph, num_vertices):
    # Step 1: DFS to get finish times
    visited = [False] * num_vertices
    stack = []
    
    def dfs1(v):
        visited[v] = True
        for neighbor in graph[v]:
            if not visited[neighbor]:
                dfs1(neighbor)
        stack.append(v)
    
    for v in range(num_vertices):
        if not visited[v]:
            dfs1(v)
    
    # Step 2: Reverse graph
    reversed_graph = defaultdict(list)
    for u in graph:
        for v in graph[u]:
            reversed_graph[v].append(u)
    
    # Step 3: DFS on reversed graph
    visited = [False] * num_vertices
    sccs = []
    
    def dfs2(v, component):
        visited[v] = True
        component.append(v)
        for neighbor in reversed_graph[v]:
            if not visited[neighbor]:
                dfs2(neighbor, component)
    
    while stack:
        v = stack.pop()
        if not visited[v]:
            component = []
            dfs2(v, component)
            sccs.append(component)
    
    return sccs
```

**Application:** If any SCC has size > 1, there's a cycle.

## Deep Dive: Critical Path Method (CPM)

**Problem:** In project management, find the longest path (critical path) which determines project duration.

**Example:**
- Task A takes 3 days.
- Task B takes 5 days and depends on A.
- Task C takes 2 days and depends on A.
- Task D takes 4 days and depends on B and C.

**Critical Path:** A → B → D (3 + 5 + 4 = 12 days).

```python
def criticalPath(tasks, dependencies):
    graph = defaultdict(list)
    in_degree = [0] * len(tasks)
    
    for task, dependency in dependencies:
        graph[dependency].append(task)
        in_degree[task] += 1
    
    # Earliest start time
    earliest = [0] * len(tasks)
    queue = deque([i for i in range(len(tasks)) if in_degree[i] == 0])
    
    while queue:
        task = queue.popleft()
        for dependent in graph[task]:
            earliest[dependent] = max(earliest[dependent], earliest[task] + tasks[task])
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    
    # Latest start time (backward pass)
    latest = [max(earliest)] * len(tasks)
    in_degree = [len(graph[i]) for i in range(len(tasks))]
    queue = deque([i for i in range(len(tasks)) if in_degree[i] == 0])  # Tasks with no dependents
    
    while queue:
        task = queue.popleft()
        for predecessor in reversed_graph[task]:
            latest[predecessor] = min(latest[predecessor], latest[task] - tasks[predecessor])
            in_degree[predecessor] -= 1
            if in_degree[predecessor] == 0:
                queue.append(predecessor)
    
    # Critical tasks have earliest == latest (no slack)
    critical_tasks = [i for i in range(len(tasks)) if earliest[i] == latest[i]]
    
    return max(earliest) + max(tasks), critical_tasks
```

## Comparison Table

| Approach | Time | Space | Best Use Case |
|:---|:---|:---|:---|
| **DFS (Cycle Detection)** | \\(O(V + E)\\) | \\(O(V)\\) | Simple cycle detection |
| **BFS (Kahn's)** | \\(O(V + E)\\) | \\(O(V)\\) | Topological ordering |
| **DFS (Post-order)** | \\(O(V + E)\\) | \\(O(V)\\) | Topological ordering |
| **Min-Heap Kahn's** | \\(O((V+E)\log V)\\) | \\(O(V)\\) | Lexicographically smallest order |
| **All Orderings** | \\(O(V! \cdot E)\\) | \\(O(V^2)\\) | Enumerate all valid orderings |

## Implementation in Other Languages

**C++:**
```cpp
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> graph(numCourses);
        vector<int> indegree(numCourses, 0);
        
        for (auto& pre : prerequisites) {
            graph[pre[1]].push_back(pre[0]);
            indegree[pre[0]]++;
        }
        
        queue<int> q;
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) q.push(i);
        }
        
        int count = 0;
        while (!q.empty()) {
            int course = q.front(); q.pop();
            count++;
            
            for (int next : graph[course]) {
                if (--indegree[next] == 0) {
                    q.push(next);
                }
            }
        }
        
        return count == numCourses;
    }
};
```

**Java:**
```java
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }
        
        int[] indegree = new int[numCourses];
        for (int[] pre : prerequisites) {
            graph.get(pre[1]).add(pre[0]);
            indegree[pre[0]]++;
        }
        
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) queue.offer(i);
        }
        
        int count = 0;
        while (!queue.isEmpty()) {
            int course = queue.poll();
            count++;
            
            for (int next : graph.get(course)) {
                if (--indegree[next] == 0) {
                    queue.offer(next);
                }
            }
        }
        
        return count == numCourses;
    }
}
```

## Top Interview Questions

**Q1: What's the difference between DFS and BFS for topological sort?**
*Answer:*
Both have \\(O(V + E)\\) time complexity. DFS uses post-order traversal and requires reversing the result. BFS (Kahn's algorithm) is more intuitive and naturally produces the ordering. Choose Kahn's for simplicity.

**Q2: Can there be multiple valid topological orderings?**
*Answer:*
Yes! For example, given courses [0, 1, 2] with prerequisites [[2, 0], [2, 1]], both [0, 1, 2] and [1, 0, 2] are valid orderings (0 and 1 can be taken in any order before 2).

**Q3: How do you handle multiple disconnected components in the graph?**
*Answer:*
Both DFS and BFS approaches naturally handle this. In DFS, we iterate through all vertices. In BFS, we start with all vertices having in-degree 0, which handles all components.

**Q4: What if prerequisites have duplicates?**
*Answer:*
Use a set to avoid duplicate edges: `graph[prereq].append(course)` only if `course` not already in `graph[prereq]`. Or, accept duplicates as they don't affect correctness, just efficiency slightly.

## Key Takeaways

1. **Cycle Detection = Impossibility:** If the dependency graph has a cycle, courses cannot be completed.
2. **Two Approaches:** DFS (3-color marking) and BFS (Kahn's algorithm) both work in \\(O(V + E)\\) time.
3. **Topological Sort:** Linear ordering of vertices respecting edge directions (only exists for DAGs).
4. **Applications:** Build systems (Make, Gradle), dependency resolution (npm, pip), job scheduling, spreadsheet calculations.
5. **Critical Path:** Finding the longest path in a DAG determines project completion time.

## Summary

| Aspect | Insight |
|:---|:---|
| **Core Problem** | Detect cycles in a directed graph |
| **Best Solution** | BFS with Kahn's algorithm (intuitive) |
| **Key Insight** | Process nodes with in-degree 0 first |
| **Applications** | Build systems, package managers, project planning |

---

**Originally published at:** [arunbaby.com/dsa/0031-course-schedule](https://www.arunbaby.com/dsa/0031-course-schedule/)

*If you found this helpful, consider sharing it with others who might benefit.*


