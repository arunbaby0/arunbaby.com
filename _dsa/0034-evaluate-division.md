---
title: "Evaluate Division (Graph/Union-Find)"
day: 34
related_ml_day: 34
related_speech_day: 34
related_agents_day: 34
collection: dsa
categories:
 - dsa
tags:
 - graph
 - dfs
 - bfs
 - union-find
 - math
difficulty: Medium
---

**"Modeling algebraic equations as graph path problems."**

## 1. Problem Statement

You are given an array of variable pairs `equations` and an array of real numbers `values`, where `equations[i] = [Ai, Bi]` and `values[i]` represent the equation `A_i / B_i = values[i]`.

Each `A_i` or `B_i` is a string that represents a single variable.

You are also given some `queries`, where `queries[j] = [Cj, Dj]` represents the `j`-th query where you must find the answer for `C_j / D_j = ?`.

Return the answers to all queries. If a single answer cannot be determined, return `-1.0`.

**Example:**
``
Input: 
equations = [["a","b"],["b","c"]]
values = [2.0, 3.0]
queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]

Output: [6.00000, 0.50000, -1.00000, 1.00000, -1.00000]

Explanation: 
Given: a / b = 2.0, b / c = 3.0
queries are: 
1. a / c = ? 
 a / c = (a / b) * (b / c) = 2.0 * 3.0 = 6.0
2. b / a = ? 
 b / a = 1 / (a / b) = 1 / 2.0 = 0.5
3. a / e = ? 
 'e' is undefined => -1.0
4. a / a = ? 
 a / a = 1.0
5. x / x = ? 
 'x' is undefined => -1.0
``

**Constraints:**
- `1 <= equations.length <= 20`
- `equations[i].length == 2`
- `1 <= queries.length <= 20`
- The input is always valid. You may assume that evaluating the queries will not result in division by zero and that there is no contradiction.

## 2. Graph Modeling Insight

This problem can be modeled as a **Directed Weighted Graph**.

- **Nodes:** Variables (e.g., "a", "b", "c").
- **Edges:** Relationships derived from equations.
 - If `a / b = 2.0`, there is an edge `a -> b` with weight `2.0`.
 - Implicitly, there is also an edge `b -> a` with weight `1 / 2.0 = 0.5`.

**Query Interpretation:**
- Finding `a / c` is equivalent to finding a **path** from node `a` to node `c`.
- The result is the **product of edge weights** along the path.
 - Path: `a -> b -> c`
 - Weight: `weight(a->b) * weight(b->c) = 2.0 * 3.0 = 6.0`.

## 3. Approach 1: DFS / BFS (Path Finding)

Since the constraints are small (`N <= 20`), a simple graph traversal (DFS or BFS) for each query is efficient enough.

**Algorithm:**
1. **Build the Graph:** Use an adjacency list. `graph[u] = [(v, weight), ...]`.
2. **Process Queries:** For each query `(start, end)`:
 - If `start` or `end` is not in the graph, return `-1.0`.
 - If `start == end`, return `1.0`.
 - Perform DFS/BFS to find a path from `start` to `end`.
 - Maintain a `visited` set to avoid cycles.
 - Accumulate the product of weights along the path.

**DFS Implementation (Python):**

``python
from collections import defaultdict

class Solution:
 def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
 # 1. Build the graph
 graph = defaultdict(dict)
 for (a, b), val in zip(equations, values):
 graph[a][b] = val
 graph[b][a] = 1.0 / val
 
 def dfs(curr, target, visited):
 if curr == target:
 return 1.0
 
 visited.add(curr)
 
 for neighbor, weight in graph[curr].items():
 if neighbor not in visited:
 res = dfs(neighbor, target, visited)
 if res != -1.0:
 return weight * res
 
 return -1.0

 # 2. Process queries
 results = []
 for c, d in queries:
 if c not in graph or d not in graph:
 results.append(-1.0)
 elif c == d:
 results.append(1.0)
 else:
 results.append(dfs(c, d, set()))
 
 return results
``

**Complexity Analysis:**
- **Time:** `O(Q \cdot (N + E))`, where `Q` is queries, `N` is variables, `E` is equations. In worst case, we traverse the whole graph for each query.
- **Space:** O(N + E) to store the graph.

## 4. Approach 2: Union-Find (Disjoint Set Union)

For larger datasets or online queries, **Union-Find** is more efficient. We can group connected variables into components.

**Key Idea:**
- In a connected component, all variables can be expressed relative to a common **root**.
- If `a` and `b` are in the same component with root `r`:
 - We store `a / r` and `b / r`.
 - Then `a / b = (a / r) / (b / r)`.

**Data Structure:**
- `parent[x]`: The parent of node `x`.
- `weight[x]`: The value of `x / parent[x]`.

**Path Compression with Weights:**
When we call `find(x)`, we recursively find the root. As we collapse the path, we update `weight[x]` to point directly to the root.

- If `x -> y` (weight `w_1`) and `y -> root` (weight `w_2`):
- New edge `x -> root` will have weight `w_1 \times w_2`.

**Union Operation:**
Given `a / b = val`:
- Find root of `a` (`rootA`) and root of `b` (`rootB`).
- If `rootA \neq rootB`, merge them.
- We want to set `parent[rootA] = rootB`.
- We need to find `weight[rootA] = rootA / rootB`.
- We know:
 - `a / b = val`
 - `a / rootA = weight[a]`
 - `b / rootB = weight[b]`
- Derivation:
 - `rootA / rootB = (rootA / a) * (a / b) * (b / rootB)`
 - `rootA / rootB = (1 / weight[a]) * val * weight[b]`

**Implementation:**

``python
class Solution:
 def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
 parent = {}
 weight = {} # weight[x] = x / parent[x]
 
 def find(x):
 if x not in parent:
 parent[x] = x
 weight[x] = 1.0
 
 if parent[x] != x:
 orig_parent = parent[x]
 root = find(orig_parent)
 # Update weight: x/root = (x/orig_parent) * (orig_parent/root)
 weight[x] = weight[x] * weight[orig_parent]
 parent[x] = root
 
 return parent[x]
 
 def union(a, b, val):
 rootA = find(a)
 rootB = find(b)
 
 if rootA != rootB:
 # Merge rootA into rootB
 parent[rootA] = rootB
 # weight[rootA] = rootA / rootB
 # val = a / b
 # weight[a] = a / rootA
 # weight[b] = b / rootB
 # rootA / rootB = (rootA / a) * (a / b) * (b / rootB)
 # = (1/weight[a]) * val * weight[b]
 weight[rootA] = (val * weight[b]) / weight[a]
 
 # 1. Build Union-Find Structure
 for (a, b), val in zip(equations, values):
 if a not in parent:
 parent[a] = a
 weight[a] = 1.0
 if b not in parent:
 parent[b] = b
 weight[b] = 1.0
 union(a, b, val)
 
 # 2. Process Queries
 results = []
 for c, d in queries:
 if c not in parent or d not in parent:
 results.append(-1.0)
 continue
 
 rootC = find(c)
 rootD = find(d)
 
 if rootC != rootD:
 results.append(-1.0)
 else:
 # c / d = (c / root) / (d / root)
 results.append(weight[c] / weight[d])
 
 return results
``

**Complexity Analysis:**
- **Time:** `O((N + Q) \cdot \alpha(N))`, where `\alpha` is the Inverse Ackermann function (nearly constant). This is much faster than DFS for many queries.
- **Space:** O(N) to store parent and weight maps.

## 5. Approach 3: Floyd-Warshall (All-Pairs Shortest Path)

If the number of variables is small and the graph is dense, we can precompute all possible divisions using Floyd-Warshall.

**Algorithm:**
1. Initialize a 2D matrix `dist` where `dist[a][b] = val` if `a/b = val`.
2. Iterate through all intermediate nodes `k`.
3. Update `dist[i][j]` if a path exists through `k`: `dist[i][j] = dist[i][k] * dist[k][j]`.

``python
class Solution:
 def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
 # Map variables to indices 0..N-1
 vars = set()
 for a, b in equations:
 vars.add(a)
 vars.add(b)
 
 var_map = {v: i for i, v in enumerate(vars)}
 n = len(vars)
 
 # Initialize matrix with -1.0 (unknown)
 graph = [[-1.0] * n for _ in range(n)]
 for i in range(n):
 graph[i][i] = 1.0
 
 for (a, b), val in zip(equations, values):
 u, v = var_map[a], var_map[b]
 graph[u][v] = val
 graph[v][u] = 1.0 / val
 
 # Floyd-Warshall
 for k in range(n):
 for i in range(n):
 for j in range(n):
 if graph[i][k] != -1.0 and graph[k][j] != -1.0:
 graph[i][j] = graph[i][k] * graph[k][j]
 
 # Queries
 res = []
 for c, d in queries:
 if c not in var_map or d not in var_map:
 res.append(-1.0)
 else:
 res.append(graph[var_map[c]][var_map[d]])
 return res
``

**Complexity:** O(N^3). Good for `N \le 100`, but bad for large `N`.

## 6. Deep Dive: Handling Contradictions

What if the input contains `a / b = 2.0` and `b / a = 3.0`? Or `a / b = 2.0`, `b / c = 2.0`, `a / c = 5.0`?

- **DFS/BFS:** Might find multiple paths with different products.
- **Union-Find:** When calling `union(a, b, val)`, if `a` and `b` are already in the same set, we can check consistency.
 - Existing relation: `a / b = weight[a] / weight[b]`.
 - New relation: `val`.
 - If `abs((weight[a] / weight[b]) - val) > epsilon`, we have a contradiction.

**Real-world implication:** In currency exchange systems, this indicates an **arbitrage opportunity** (or a data error).

## 7. Deep Dive: Currency Arbitrage Application

This problem is isomorphic to finding arbitrage in currency exchange markets.
- Nodes: Currencies (USD, EUR, JPY).
- Edges: Exchange rates.
- Cycle: `USD -> EUR -> JPY -> USD`.
- If the product of weights along a cycle is `> 1.0`, you can make infinite money (theoretically).

**Algorithm for Arbitrage Detection:**
- Use **Bellman-Ford** or **SPFA** (Shortest Path Faster Algorithm).
- Since we are dealing with products, we can convert to sums using logarithms:
 - `\log(a / b) = \log(a) - \log(b)`.
 - `w_{new} = -\log(w_{old})`.
 - Finding a path with product `> 1` becomes finding a negative weight cycle in the log-graph.

## 8. Deep Dive: Dynamic Updates

What if equations are added dynamically?
- **DFS/BFS:** Slow, need to re-traverse for every query.
- **Floyd-Warshall:** Can update in O(N^2) for each new edge.
- **Union-Find:** Best for incremental updates. `union` is nearly O(1).

## 9. Deep Dive: Precision Issues

Floating point arithmetic can accumulate errors.
- `a / b = 3.0`, `b / c = 1.0 / 3.0`.
- `a / c` should be `1.0`.
- In float, it might be `0.99999999`.

**Mitigation:**
- Use a small epsilon (`1e-9`) for comparisons.
- Or simply return the calculated float as is (problem statement usually allows small error).

## 10. Real-World Applications

1. **Unit Conversion:**
 - Input: `1 m = 100 cm`, `1 km = 1000 m`, `1 in = 2.54 cm`.
 - Query: `1 km = ? in`.
 - The graph finds the conversion chain.

2. **Currency Exchange:**
 - Calculating cross-rates between illiquid currency pairs via a liquid intermediary (e.g., USD).

3. **Chemical Stoichiometry:**
 - Balancing chemical equations or converting between moles of different reactants.

4. **Social Network Influence:**
 - If influence is multiplicative (probability of infection), this models propagation paths.

## 11. Edge Cases

1. **Disconnected Graph:** Query `a / e` where `e` is in a different component. Return `-1.0`.
2. **Unknown Variable:** Query `x / x` where `x` was never seen. Return `-1.0`.
3. **Zero Division:** Constraints say `values[i] > 0.0`, so no division by zero.
4. **Self Loop:** `a / a` should always be `1.0` if `a` exists.

## 12. Interview Tips

- **Clarify Input:** Are values always positive? (Yes). Are there contradictions? (No, usually).
- **Choose the Right Tool:**
 - Small `N`, many queries? **Floyd-Warshall**.
 - Large `N`, sparse graph? **DFS/BFS**.
 - Dynamic updates or massive queries? **Union-Find**.
- **Space Complexity:** Mention that Union-Find is very space-efficient.

## 13. Summary

| Approach | Time Complexity | Space Complexity | Best For |
| :--- | :--- | :--- | :--- |
| **DFS / BFS** | `O(Q \cdot (N+E))` | O(N+E) | Sparse graphs, few queries |
| **Union-Find** | `O((N+Q) \cdot \alpha(N))` | O(N) | Many queries, dynamic updates |
| **Floyd-Warshall** | O(N^3 + Q) | O(N^2) | Dense graphs, small `N` |

## 14. Deep Dive: Bi-Directional BFS for Faster Queries

For very large graphs where `N` is large (e.g., 100,000 nodes), a standard BFS might visit too many nodes. **Bi-Directional BFS** can significantly reduce the search space.

**Concept:**
- Start BFS from `source` (forward) and `target` (backward) simultaneously.
- When the two searches meet at a node `meet_node`, we have found a path.
- Path weight = `weight(source -> meet_node) * weight(meet_node -> target)`.
- Note: `weight(meet_node -> target)` is `1 / weight(target -> meet_node)`.

**Implementation:**

``python
from collections import deque

def calcEquationBiDir(graph, start, end):
 if start not in graph or end not in graph:
 return -1.0
 if start == end:
 return 1.0
 
 # Queue stores (node, current_product)
 q_fwd = deque([(start, 1.0)])
 q_bwd = deque([(end, 1.0)])
 
 visited_fwd = {start: 1.0}
 visited_bwd = {end: 1.0}
 
 while q_fwd and q_bwd:
 # Expand forward
 if len(q_fwd) <= len(q_bwd):
 curr, prod = q_fwd.popleft()
 if curr in visited_bwd:
 return prod * (1.0 / visited_bwd[curr])
 
 for neighbor, weight in graph[curr].items():
 if neighbor not in visited_fwd:
 visited_fwd[neighbor] = prod * weight
 q_fwd.append((neighbor, prod * weight))
 else:
 # Expand backward
 curr, prod = q_bwd.popleft()
 if curr in visited_fwd:
 return visited_fwd[curr] * (1.0 / prod)
 
 for neighbor, weight in graph[curr].items():
 if neighbor not in visited_bwd:
 visited_bwd[neighbor] = prod * weight
 q_bwd.append((neighbor, prod * weight))
 
 return -1.0
``

**Why it works:**
- Standard BFS search space: `b^d` (branching factor `b`, depth `d`).
- Bi-Directional BFS search space: `2 \cdot b^{d/2}`.
- For `b=10, d=6`: `10^6` vs `2 \cdot 10^3 = 2000`. Massive speedup!

## 15. Deep Dive: Offline Queries with Union-Find

If we have a massive batch of queries and the graph is static, we can optimize using **Offline Processing**.

**Idea:**
- Sort queries or process them in a way that minimizes overhead.
- With Union-Find, we can answer queries in nearly O(1) time after building the structure.
- If we have dynamic updates intermixed with queries, we can use **Time-Travel Union-Find** (persistent data structure) or process updates and queries chronologically.

**Batch Processing:**
1. Read all equations.
2. Build Union-Find.
3. Read all queries.
4. Answer using `weight[c] / weight[d]`.

This is already what Approach 2 does, but "Offline" implies we have all data upfront.

## 16. Deep Dive: Detecting Contradictions in Detail

As mentioned, contradictions are critical in real-world data (e.g., sensor data fusion, financial data).

**Algorithm for Consistency Checking:**
1. Maintain a `Union-Find` structure.
2. For every new equation `a / b = val`:
 - If `a` and `b` are already connected:
 - Calculate existing ratio `existing_val = weight[a] / weight[b]`.
 - If `abs(existing_val - val) > 1e-9`: **Contradiction Found!**
 - Return `False` or raise Error.
 - Else:
 - `union(a, b, val)`.

**Code Example:**

``python
class ConsistencyChecker:
 def __init__(self):
 self.parent = {}
 self.weight = {}

 def find(self, x):
 if x not in self.parent:
 self.parent[x] = x
 self.weight[x] = 1.0
 if self.parent[x] != x:
 orig_parent = self.parent[x]
 root = self.find(orig_parent)
 self.weight[x] *= self.weight[orig_parent]
 self.parent[x] = root
 return self.parent[x]

 def add_equation(self, a, b, val):
 rootA = self.find(a)
 rootB = self.find(b)
 
 if rootA != rootB:
 # Merge
 self.parent[rootA] = rootB
 self.weight[rootA] = (val * self.weight[b]) / self.weight[a]
 return True
 else:
 # Check consistency
 # a / b should be val
 # We have a / b = (a/root) / (b/root) = weight[a] / weight[b]
 existing_val = self.weight[a] / self.weight[b]
 if abs(existing_val - val) > 1e-5:
 print(f"Contradiction: {a}/{b} new={val}, old={existing_val}")
 return False
 return True
``

## 17. Deep Dive: Graph Simplification (Transitive Reduction)

In some applications, we want to store the **minimum** number of equations needed to derive all others. This is the opposite of transitive closure.

**Problem:** Given `a->b`, `b->c`, `a->c`, remove `a->c` because it's redundant.

**Algorithm:**
1. For every edge `u -> v`:
 - Temporarily remove `u -> v`.
 - Run BFS/DFS to see if `v` is still reachable from `u`.
 - If yes, `u -> v` is redundant. Permanently remove it.
 - If no, put it back.

**Note:** This is expensive (`O(E \cdot (N+E))`). Only do this if read-heavy and storage-constrained.

## 18. LeetCode Variations

**1. Similar to: 399. Evaluate Division**
- This is the exact problem.

**2. 230. Kth Smallest Element in a BST**
- Not graph, but involves traversing structures.

**3. 133. Clone Graph**
- Graph traversal basics.

**4. 684. Redundant Connection**
- Union-Find application to detect cycles.

**5. 1135. Connecting Cities With Minimum Cost**
- MST (Minimum Spanning Tree) using Kruskal's (Union-Find) or Prim's.

## 19. Performance Benchmarking

Let's compare DFS vs. Union-Find on a large dataset.

**Scenario:**
- `N = 10,000` variables.
- `E = 10,000` equations (sparse).
- `Q = 10,000` queries.

**DFS:**
- Each query takes O(N).
- Total: `10,000 \times 10,000 = 10^8` operations.
- Time: ~10-20 seconds (Python).

**Union-Find:**
- Build: `O(E \cdot \alpha(N)) \approx 10,000`.
- Query: `O(Q \cdot \alpha(N)) \approx 10,000`.
- Total: `2 \cdot 10^4` operations.
- Time: < 0.1 seconds.

**Conclusion:** Union-Find is **orders of magnitude faster** for dense query workloads.

## 20. Advanced: Handling Log-Probabilities

In probability graphs (e.g., Bayesian Networks), edges represent conditional probabilities `P(B|A)`.
- Path `A \to B \to C` implies `P(C|A) = P(C|B) \cdot P(B|A)`.
- This is exactly Evaluate Division.
- To avoid underflow with small probabilities, use **Log-Probabilities**.
 - `\log(P(C|A)) = \log(P(C|B)) + \log(P(B|A))`.
 - Multiplication becomes Addition.
 - Division becomes Subtraction.
 - Shortest Path algorithms (Dijkstra) work naturally on sums.

## 21. Advanced: Multi-Source BFS

If we want to find `a / x` for *all* `x` reachable from `a`.
- Run BFS starting from `a`.
- `dist[start] = 1.0`.
- For neighbor `v` of `u`: `dist[v] = dist[u] * weight(u->v)`.
- This gives the ratio relative to `a` for the entire connected component.

## 22. Production Considerations

1. **Precision:** Use `decimal.Decimal` in Python or `BigDecimal` in Java for financial applications to avoid floating point drift.
2. **Concurrency:** If the graph is updated by multiple threads, use **Read-Write Locks**. Queries (Reads) can run in parallel; Updates (Writes) need exclusive access.
3. **Caching:** Cache query results `(a, b) -> val`. Invalidate cache if `a` or `b` or any node on the path is updated. (Hard to track dependencies, usually better to just re-query Union-Find).

## 23. Summary of Graph Algorithms for Division

| Algorithm | Use Case | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **DFS** | Simple, One-off | Easy to code | Slow for many queries |
| **BFS** | Shortest Path | Finds path with min hops (less error) | Memory intensive |
| **Union-Find** | Batch Queries | Extremely fast, handles dynamic updates | Harder to implement |
| **Floyd-Warshall** | Dense Graph | All-pairs precomputed | O(N^3) slow build |
| **Bi-Dir BFS** | Large Graph | Faster than BFS | Complex state management |

| **Bi-Dir BFS** | Large Graph | Faster than BFS | Complex state management |

## 24. Deep Dive: Iterative DFS Implementation

Recursive DFS can hit recursion limits (default 1000 in Python) for deep graphs. An iterative approach using a stack is safer for production.

**Implementation:**

``python
def calcEquationIterative(graph, start, end):
 if start not in graph or end not in graph:
 return -1.0
 if start == end:
 return 1.0
 
 stack = [(start, 1.0)]
 visited = {start}
 
 while stack:
 curr, prod = stack.pop()
 
 if curr == end:
 return prod
 
 for neighbor, weight in graph[curr].items():
 if neighbor not in visited:
 visited.add(neighbor)
 stack.append((neighbor, prod * weight))
 
 return -1.0
``

**Trade-off:** Iterative DFS is slightly harder to read but robust against StackOverflowErrors.

## 25. Deep Dive: Optimizing with Strongly Connected Components (SCC)

If the graph has cycles (e.g., currency exchange), we can condense Strongly Connected Components into single "super-nodes" to speed up queries.

**Tarjan's Algorithm:**
1. Find all SCCs.
2. If `a` and `b` are in the same SCC, there is definitely a path (and potentially a cycle).
3. If we only care about reachability (not values), this reduces the graph size significantly.
4. For Evaluate Division, cycles must have product 1.0. If not, it's a contradiction.

**Algorithm:**
- Run Tarjan's to find SCCs.
- Verify all cycles in SCCs have product 1.0.
- Build a DAG of SCCs.
- Query becomes: Path in DAG + Path within SCC.

## 26. Deep Dive: Unit Testing Strategies

How do we test this graph logic?

**Test Case 1: Simple Chain**
- `a/b=2`, `b/c=3` -> `a/c=6`.

**Test Case 2: Inverse**
- `a/b=2` -> `b/a=0.5`.

**Test Case 3: Disconnected**
- `a/b=2`, `c/d=3` -> `a/c=-1`.

**Test Case 4: Cycle (Consistent)**
- `a/b=2`, `b/c=3`, `c/a=1/6`.

**Test Case 5: Cycle (Inconsistent)**
- `a/b=2`, `b/c=3`, `c/a=2` (Product 12 != 1).

**Python Unittest:**

``python
import unittest

class TestEvaluateDivision(unittest.TestCase):
 def test_simple_chain(self):
 eq = [["a","b"], ["b","c"]]
 val = [2.0, 3.0]
 q = [["a","c"]]
 sol = Solution()
 res = sol.calcEquation(eq, val, q)
 self.assertAlmostEqual(res[0], 6.0)

 def test_disconnected(self):
 eq = [["a","b"], ["c","d"]]
 val = [2.0, 3.0]
 q = [["a","c"]]
 sol = Solution()
 res = sol.calcEquation(eq, val, q)
 self.assertEqual(res[0], -1.0)
``

## 27. Deep Dive: Error Handling and Logging

In a production system (e.g., a currency conversion microservice), "return -1.0" is not enough.

**Requirements:**
1. **Structured Logging:** Log *why* it failed. "Node 'JPY' not found" vs "No path from 'USD' to 'BTC'".
2. **Metrics:** Track `cache_hit_rate` (if using Union-Find or caching), `query_latency`, `error_rate`.
3. **Alerting:** If `inconsistent_data_error` spikes, alert the data team.

**Example Log:**
``json
{
 "level": "WARN",
 "event": "conversion_failed",
 "source": "USD",
 "target": "BTC",
 "reason": "disconnected_component",
 "timestamp": "2023-10-27T10:00:00Z"
}
``

## 28. Code: Full Union-Find Implementation with Path Compression

``python
class UnionFind:
 def __init__(self):
 self.parent = {}
 self.weight = {}

 def add(self, x):
 if x not in self.parent:
 self.parent[x] = x
 self.weight[x] = 1.0

 def find(self, x):
 if x not in self.parent:
 return None, None
 if self.parent[x] != x:
 orig_parent = self.parent[x]
 root, root_weight = self.find(orig_parent)
 self.weight[x] *= root_weight
 self.parent[x] = root
 return self.parent[x], self.weight[x]

 def union(self, a, b, val):
 self.add(a)
 self.add(b)
 rootA, wA = self.find(a)
 rootB, wB = self.find(b)
 if rootA != rootB:
 self.parent[rootA] = rootB
 self.weight[rootA] = (val * wB) / wA

 def query(self, a, b):
 if a not in self.parent or b not in self.parent:
 return -1.0
 rootA, wA = self.find(a)
 rootB, wB = self.find(b)
 if rootA != rootB:
 return -1.0
 return wA / wB
``

## 29. System Design: Building a Currency Conversion API

**Scenario:** Design a microservice that handles 10,000 QPS (queries per second) for currency conversions.

**Requirements:**
1. **Low Latency:** < 10ms p99.
2. **High Availability:** 99.99% uptime.
3. **Dynamic Updates:** Exchange rates update every minute.
4. **Arbitrage Detection:** Alert if cycles exist with product != 1.0.

**Architecture:**

``
┌─────────────┐
│ Client │
└──────┬──────┘
 │
 ▼
┌─────────────────────────────────┐
│ API Gateway (Rate Limiting) │
└──────────────┬──────────────────┘
 │
 ▼
┌──────────────────────────────────┐
│ Conversion Service (Stateless) │
│ - Load Balancer (3 instances) │
└──────────────┬───────────────────┘
 │
 ┌──────┴──────┐
 ▼ ▼
┌──────────────┐ ┌──────────────┐
│ Redis Cache │ │ Union-Find │
│ (Hot Pairs) │ │ In-Memory │
└──────────────┘ └──────┬───────┘
 │
 ▼
 ┌──────────────┐
 │ PostgreSQL │
 │ (Equations) │
 └──────────────┘
``

**Implementation Details:**

**1. Data Model (PostgreSQL):**
``sql
CREATE TABLE exchange_rates (
 id SERIAL PRIMARY KEY,
 from_currency VARCHAR(3),
 to_currency VARCHAR(3),
 rate DECIMAL(18, 8),
 timestamp TIMESTAMP DEFAULT NOW(),
 UNIQUE(from_currency, to_currency)
);

CREATE INDEX idx_currencies ON exchange_rates(from_currency, to_currency);
``

**2. In-Memory Union-Find:**
``python
class CurrencyConverter:
 def __init__(self):
 self.uf = UnionFind()
 self.last_update = 0
 self.lock = threading.RLock()
 
 def reload_from_db(self):
 with self.lock:
 # Fetch all rates
 rates = db.query("SELECT from_currency, to_currency, rate FROM exchange_rates")
 
 # Rebuild Union-Find
 self.uf = UnionFind()
 for from_curr, to_curr, rate in rates:
 self.uf.union(from_curr, to_curr, rate)
 
 self.last_update = time.time()
 
 def convert(self, from_curr, to_curr, amount):
 # Check cache first
 cache_key = f"{from_curr}:{to_curr}"
 cached_rate = redis.get(cache_key)
 
 if cached_rate:
 return amount * float(cached_rate)
 
 # Query Union-Find
 with self.lock:
 rate = self.uf.query(from_curr, to_curr)
 
 if rate == -1.0:
 raise ValueError(f"No conversion path from {from_curr} to {to_curr}")
 
 # Cache for 60 seconds
 redis.setex(cache_key, 60, rate)
 
 return amount * rate
``

**3. Background Worker (Rate Updates):**
``python
import schedule

def update_rates():
 # Fetch from external API (e.g., fixer.io)
 new_rates = fetch_external_rates()
 
 # Update DB
 for from_curr, to_curr, rate in new_rates:
 db.execute("""
 INSERT INTO exchange_rates (from_currency, to_currency, rate)
 VALUES (%s, %s, %s)
 ON CONFLICT (from_currency, to_currency) 
 DO UPDATE SET rate = EXCLUDED.rate, timestamp = NOW()
 """, (from_curr, to_curr, rate))
 
 # Reload in-memory structure
 converter.reload_from_db()
 
 # Detect arbitrage
 detect_arbitrage()

schedule.every(1).minutes.do(update_rates)
``

**4. Arbitrage Detection:**
``python
def detect_arbitrage():
 # Find all cycles
 # For each cycle, check if product != 1.0
 # This is expensive, run asynchronously
 
 currencies = get_all_currencies()
 graph = build_graph()
 
 for start in currencies:
 # DFS to find cycles
 visited = set()
 path = []
 product = 1.0
 
 def dfs(curr, prod):
 if curr in visited:
 if curr == start and abs(prod - 1.0) > 0.01:
 alert(f"Arbitrage detected: {path}, product={prod}")
 return
 
 visited.add(curr)
 path.append(curr)
 
 for neighbor, weight in graph[curr].items():
 dfs(neighbor, prod * weight)
 
 path.pop()
 visited.remove(curr)
 
 dfs(start, 1.0)
``

**5. Monitoring & Metrics:**
``python
from prometheus_client import Counter, Histogram

conversion_requests = Counter('conversion_requests_total', 'Total conversion requests')
conversion_latency = Histogram('conversion_latency_seconds', 'Conversion latency')
cache_hits = Counter('cache_hits_total', 'Cache hits')
cache_misses = Counter('cache_misses_total', 'Cache misses')

@conversion_latency.time()
def convert_with_metrics(from_curr, to_curr, amount):
 conversion_requests.inc()
 
 if redis.exists(f"{from_curr}:{to_curr}"):
 cache_hits.inc()
 else:
 cache_misses.inc()
 
 return converter.convert(from_curr, to_curr, amount)
``

**Performance:**
- **Cold Query:** ~5ms (Union-Find lookup + DB roundtrip).
- **Warm Query:** ~0.5ms (Redis cache hit).
- **Throughput:** 10,000 QPS easily handled with 3 instances.

## 30. Final Thoughts

Evaluate Division is a classic example of how a math problem can be transformed into a graph problem. The choice of algorithm (DFS vs. Union-Find) depends heavily on the constraints (number of queries vs. updates). Mastering Union-Find with path compression and weight tracking is a powerful tool for your algorithmic toolkit.



---

**Originally published at:** [arunbaby.com/dsa/0034-evaluate-division](https://www.arunbaby.com/dsa/0034-evaluate-division/)

*If you found this helpful, consider sharing it with others who might benefit.*

