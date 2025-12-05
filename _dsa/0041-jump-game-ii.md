---
title: "Jump Game II"
day: 41
collection: dsa
categories:
  - dsa
tags:
  - greedy
  - dynamic-programming
  - array
  - bfs
difficulty: Medium
---

**"Finding the optimal path through a sequence of choices."**

## 1. Problem Statement

You are given a **0-indexed** array of integers `nums` of length `n`. You are initially positioned at `nums[0]`.

Each element `nums[i]` represents the **maximum** length of a forward jump from index `i`. In other words, if you are at `nums[i]`, you can jump to any `nums[i + j]` where:
- `0 <= j <= nums[i]`
- `i + j < n`

Return the **minimum number of jumps** to reach `nums[n - 1]`. The test cases are generated such that you can reach `nums[n - 1]`.

**Example 1:**
```
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

**Example 2:**
```
Input: nums = [2,3,0,1,4]
Output: 2
```

**Constraints:**
*   `1 <= nums.length <= 10^4`
*   `0 <= nums[i] <= 1000`
*   It's guaranteed that you can reach `nums[n-1]`.

## 2. Intuition

This is a classic **Greedy** problem disguised as a graph traversal.

**Key Insight:** At each position, we want to jump to the position that allows us to reach the farthest in the next jump. This is a **local greedy choice** that leads to a **global optimal solution**.

Think of it as BFS where each "level" represents the positions reachable with `k` jumps.

## 3. Approach 1: Greedy (Optimal)

**Idea:** We maintain the farthest position we can reach with the current number of jumps. When we exhaust the current range, we increment the jump count.

**Algorithm:**
1.  Initialize `jumps = 0`, `current_end = 0` (end of current jump range), `farthest = 0` (farthest we can reach).
2.  Iterate through the array (except the last element, since we're already there if we reach it).
3.  For each position `i`:
    *   Update `farthest = max(farthest, i + nums[i])`.
    *   If `i == current_end` (we've exhausted the current jump range):
        *   Increment `jumps`.
        *   Set `current_end = farthest` (start a new jump range).
4.  Return `jumps`.

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        jumps = 0
        current_end = 0
        farthest = 0
        
        for i in range(n - 1):  # Don't need to check the last element
            farthest = max(farthest, i + nums[i])
            
            if i == current_end:
                jumps += 1
                current_end = farthest
                
                # Early exit if we can already reach the end
                if current_end >= n - 1:
                    break
                    
        return jumps
```

**Complexity:**
*   **Time:** $O(N)$. Single pass through the array.
*   **Space:** $O(1)$. Constant extra space.

## 4. Approach 2: BFS (Level-Order Traversal)

We can model this as a graph where each index is a node, and there's an edge from `i` to `j` if `j <= i + nums[i]`.

**Algorithm:**
1.  Use BFS. Each level represents positions reachable with `k` jumps.
2.  For each level, find the farthest position reachable.
3.  If the farthest position reaches or exceeds `n-1`, return the level count.

```python
from collections import deque

class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        queue = deque([0])
        visited = {0}
        jumps = 0
        
        while queue:
            size = len(queue)
            jumps += 1
            
            for _ in range(size):
                i = queue.popleft()
                
                # Try all possible jumps from position i
                for j in range(i + 1, min(i + nums[i] + 1, n)):
                    if j == n - 1:
                        return jumps
                    if j not in visited:
                        visited.add(j)
                        queue.append(j)
                        
        return jumps
```

**Complexity:**
*   **Time:** $O(N^2)$ in worst case (e.g., `[1,1,1,1,1]`).
*   **Space:** $O(N)$ for the queue and visited set.

## 5. Approach 3: Dynamic Programming

Define `dp[i]` = minimum jumps to reach index `i`.

**Transition:**
For each position `i`, we can jump to any position `j` where `i < j <= i + nums[i]`.
`dp[j] = min(dp[j], dp[i] + 1)`.

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [float('inf')] * n
        dp[0] = 0
        
        for i in range(n):
            for j in range(i + 1, min(i + nums[i] + 1, n)):
                dp[j] = min(dp[j], dp[i] + 1)
                
        return dp[n - 1]
```

**Complexity:**
*   **Time:** $O(N \times M)$ where $M$ is the average jump length. Worst case $O(N^2)$.
*   **Space:** $O(N)$ for the DP array.

## 6. Deep Dive: Why Greedy Works (Proof)

**Claim:** The greedy algorithm always finds the minimum number of jumps.

**Proof by Exchange Argument:**
Suppose the greedy algorithm produces a solution with `k` jumps: $0 \to i_1 \to i_2 \to \dots \to i_k = n-1$.
Suppose there exists an optimal solution with fewer jumps: $0 \to j_1 \to j_2 \to \dots \to j_m = n-1$ where $m < k$.

Consider the first position where the two solutions differ. Let's say the greedy chooses $i_1$ and the optimal chooses $j_1$.
By the greedy choice, $i_1$ is the farthest position reachable from 0. Therefore, $j_1 \leq i_1$.

Now, from $j_1$, the optimal solution reaches $j_2$. But since $j_1 \leq i_1$, and the greedy algorithm considers all positions reachable from $i_1$, it must be that the greedy can also reach $j_2$ (or farther) in the next jump.

By induction, we can show that the greedy solution reaches at least as far as the optimal solution at each step. Since both reach $n-1$, and the greedy makes the farthest jump at each step, it cannot make more jumps than the optimal.

**Contradiction.** Therefore, the greedy algorithm is optimal.

## 7. Detailed Walkthrough

Let's trace `nums = [2, 3, 1, 1, 4]`.

**Initial State:**
*   `jumps = 0`, `current_end = 0`, `farthest = 0`.

**Iteration:**
1.  `i = 0`: `farthest = max(0, 0 + 2) = 2`. `i == current_end`, so `jumps = 1`, `current_end = 2`.
2.  `i = 1`: `farthest = max(2, 1 + 3) = 4`. `i < current_end`, so no jump yet.
3.  `i = 2`: `farthest = max(4, 2 + 1) = 4`. `i == current_end`, so `jumps = 2`, `current_end = 4`.
4.  `i = 3`: We've reached `n - 1 = 4` with `current_end = 4`, so we stop.

**Result:** `jumps = 2`.

**Path:** $0 \to 1 \to 4$ (Jump to index 1, then jump to index 4).

## 8. Variant: Jump Game I (Can Reach?)

**Problem:** Given `nums`, return `true` if you can reach the last index.

**Greedy Solution:**
```python
def canJump(nums):
    farthest = 0
    for i in range(len(nums)):
        if i > farthest:
            return False  # Can't reach position i
        farthest = max(farthest, i + nums[i])
        if farthest >= len(nums) - 1:
            return True
    return True
```

## 9. Variant: Jump Game III (Reach Zero)

**Problem:** Given `arr` and `start`, you can jump to `start + arr[start]` or `start - arr[start]`. Return `true` if you can reach any index with value 0.

**BFS Solution:**
```python
from collections import deque

def canReach(arr, start):
    n = len(arr)
    queue = deque([start])
    visited = {start}
    
    while queue:
        i = queue.popleft()
        if arr[i] == 0:
            return True
        
        for next_i in [i + arr[i], i - arr[i]]:
            if 0 <= next_i < n and next_i not in visited:
                visited.add(next_i)
                queue.append(next_i)
                
    return False
```

## 10. System Design: Pathfinding in Games

Jump Game II is essentially a simplified version of pathfinding algorithms used in game AI.

**Real-World Application: Platformer Games**
*   **Problem:** Find the shortest sequence of jumps for a character to reach a goal.
*   **Constraints:** Jump height, gravity, obstacles.
*   **Algorithm:** A* search with heuristic = Manhattan distance to goal.

**Optimization:**
*   **Precompute Reachability Graph:** For static levels, precompute which platforms are reachable from each platform.
*   **Caching:** Cache optimal paths for frequently visited platform pairs.

## 11. Deep Dive: Jump Game with Costs

**Problem:** Each jump from `i` to `j` has a cost `cost[i][j]`. Find the minimum cost to reach the end.

**Algorithm:** Dijkstra's Algorithm.
```python
import heapq

def minCostJump(nums, cost):
    n = len(nums)
    dist = [float('inf')] * n
    dist[0] = 0
    heap = [(0, 0)]  # (cost, index)
    
    while heap:
        d, i = heapq.heappop(heap)
        if i == n - 1:
            return d
        if d > dist[i]:
            continue
            
        for j in range(i + 1, min(i + nums[i] + 1, n)):
            new_cost = d + cost[i][j]
            if new_cost < dist[j]:
                dist[j] = new_cost
                heapq.heappush(heap, (new_cost, j))
                
    return dist[n - 1]
```

## 12. Interview Questions

1.  **Jump Game II (Classic):** Solve in O(N) time and O(1) space.
2.  **Jump Game I:** Can you reach the last index?
3.  **Jump Game III:** Can you reach any index with value 0?
4.  **Jump Game IV:** Given `arr`, you can jump to `i+1`, `i-1`, or any `j` where `arr[j] == arr[i]`. Find minimum jumps to reach the last index.
5.  **Frog Jump:** A frog can jump `k-1`, `k`, or `k+1` units. Can it cross the river?
6.  **Minimum Jumps with Cost:** Each jump has a cost. Find the minimum cost path.

## 13. Common Mistakes

*   **Off-by-One:** Iterating through the entire array including the last element (unnecessary).
*   **Not Handling Single Element:** `nums = [0]` should return `0` jumps.
*   **Greedy Choice:** Thinking we should always jump to the farthest position immediately (wrong! We should jump to the position that allows the farthest next jump).
*   **BFS Optimization:** Not using the "level-by-level" optimization, leading to $O(N^2)$ instead of $O(N)$.

## 14. Performance Comparison

```python
import time
import random

def benchmark():
    sizes = [100, 1000, 10000]
    
    for size in sizes:
        nums = [random.randint(1, 10) for _ in range(size)]
        
        # Greedy
        start = time.time()
        # greedy_solution(nums)
        greedy_time = time.time() - start
        
        # DP
        start = time.time()
        # dp_solution(nums)
        dp_time = time.time() - start
        
        print(f"Size {size}: Greedy={greedy_time:.6f}s, DP={dp_time:.6f}s")

# Expected: Greedy is 10-100x faster than DP for large inputs.
```

## 15. Deep Dive: Jump Game IV (BFS with HashMap)

**Problem:** Given an array `arr`, you can jump from index `i` to:
*   `i + 1`
*   `i - 1`
*   Any index `j` where `arr[j] == arr[i]` and `i != j`

Find the minimum number of jumps to reach the last index.

**Example:**
```
Input: arr = [100,-23,-23,404,100,23,23,23,3,404]
Output: 3
Explanation: 0 -> 4 -> 3 -> 9
```

**Algorithm (BFS with Optimization):**
```python
from collections import deque, defaultdict

def minJumps(arr):
    n = len(arr)
    if n == 1:
        return 0
    
    # Build value -> indices mapping
    graph = defaultdict(list)
    for i, val in enumerate(arr):
        graph[val].append(i)
    
    queue = deque([0])
    visited = {0}
    steps = 0
    
    while queue:
        size = len(queue)
        steps += 1
        
        for _ in range(size):
            i = queue.popleft()
            
            # Try all three types of jumps
            # 1. i + 1
            if i + 1 < n and i + 1 not in visited:
                if i + 1 == n - 1:
                    return steps
                visited.add(i + 1)
                queue.append(i + 1)
            
            # 2. i - 1
            if i - 1 >= 0 and i - 1 not in visited:
                visited.add(i - 1)
                queue.append(i - 1)
            
            # 3. Same value jumps
            for j in graph[arr[i]]:
                if j not in visited:
                    if j == n - 1:
                        return steps
                    visited.add(j)
                    queue.append(j)
            
            # CRITICAL: Clear the list to avoid revisiting
            graph[arr[i]].clear()
    
    return steps
```

**Optimization:** After visiting all indices with value `arr[i]`, we clear the list. This prevents revisiting the same value group multiple times.

**Complexity:**
*   **Time:** $O(N)$. Each index is visited at most once.
*   **Space:** $O(N)$ for the graph and visited set.

## 16. Deep Dive: Frog Jump (DP with Set)

**Problem:** A frog is crossing a river by jumping on stones. The frog can jump `k - 1`, `k`, or `k + 1` units where `k` is the last jump distance. Can the frog cross?

**Example:**
```
Input: stones = [0,1,3,5,6,8,12,17]
Output: true
Explanation: 0 -> 1 (1 unit) -> 3 (2 units) -> 5 (2 units) -> 6 (1 unit) -> 8 (2 units) -> 12 (4 units) -> 17 (5 units)
```

**Algorithm (DP with HashMap):**
```python
def canCross(stones):
    if stones[1] != 1:
        return False  # First jump must be 1 unit
    
    stone_set = set(stones)
    dp = {}  # (position, last_jump) -> bool
    
    def dfs(pos, k):
        if pos == stones[-1]:
            return True
        if (pos, k) in dp:
            return dp[(pos, k)]
        
        for next_k in [k - 1, k, k + 1]:
            if next_k > 0:
                next_pos = pos + next_k
                if next_pos in stone_set:
                    if dfs(next_pos, next_k):
                        dp[(pos, k)] = True
                        return True
        
        dp[(pos, k)] = False
        return False
    
    return dfs(1, 1)
```

**Complexity:**
*   **Time:** $O(N^2)$. At most $N$ positions and $N$ possible jump distances.
*   **Space:** $O(N^2)$ for memoization.

## 17. Production Application: Route Optimization

**Scenario:** Delivery truck routing (Amazon, UPS).

**Problem:** Given a list of delivery locations and the maximum distance the truck can travel from each location, find the minimum number of "hops" (refueling stops) to deliver all packages.

**Mapping to Jump Game:**
*   `nums[i]` = maximum distance from location `i`.
*   Goal: Reach the last location with minimum refueling stops.

**Extensions:**
*   **Time Windows:** Each location has a delivery time window.
*   **Capacity Constraints:** Truck has limited capacity.
*   **Multiple Vehicles:** Coordinate multiple trucks.

**Algorithm:** Jump Game II + Constraint Satisfaction.

## 18. Production Application: Network Packet Routing

**Scenario:** Data packet routing in a network.

**Problem:** A packet needs to travel from source to destination. Each router can forward the packet to routers within a certain "hop distance". Find the minimum number of hops.

**Mapping:**
*   Nodes = routers.
*   `nums[i]` = maximum hop distance from router `i`.
*   Goal: Minimum hops from source to destination.

**Real-World Constraints:**
*   **Congestion:** Some routers are overloaded (higher cost).
*   **Latency:** Each hop has a latency.
*   **Reliability:** Some links may fail.

**Algorithm:** Dijkstra's Algorithm with dynamic weights.

## 19. Advanced Variant: Jump Game with Obstacles

**Problem:** Some positions are obstacles (cannot land on them). Find the minimum jumps.

**Algorithm (Modified BFS):**
```python
def jumpWithObstacles(nums, obstacles):
    n = len(nums)
    obstacle_set = set(obstacles)
    
    if 0 in obstacle_set or n - 1 in obstacle_set:
        return -1  # Can't start or can't finish
    
    queue = deque([0])
    visited = {0}
    jumps = 0
    
    while queue:
        size = len(queue)
        jumps += 1
        
        for _ in range(size):
            i = queue.popleft()
            
            for j in range(i + 1, min(i + nums[i] + 1, n)):
                if j in obstacle_set:
                    continue  # Skip obstacles
                if j == n - 1:
                    return jumps
                if j not in visited:
                    visited.add(j)
                    queue.append(j)
    
    return -1  # Can't reach
```

## 20. Mathematical Analysis: Expected Jumps

**Question:** If `nums[i]` is uniformly random in `[1, k]`, what is the expected number of jumps for an array of length `n`?

**Analysis:**
*   Average jump length: $\frac{k+1}{2}$.
*   Expected number of jumps: $\approx \frac{n}{\frac{k+1}{2}} = \frac{2n}{k+1}$.

**Example:** $n = 100$, $k = 10$.
*   Expected jumps: $\frac{200}{11} \approx 18$.

## 21. Parallel Algorithm: Jump Game on GPU

**Problem:** Solve Jump Game II for millions of arrays in parallel (batch processing).

**Algorithm (CUDA):**
1.  **Kernel:** Each thread processes one array.
2.  **Shared Memory:** Store the array in shared memory for fast access.
3.  **Reduction:** Use parallel reduction to find the farthest reachable position.

**Pseudocode:**
```cuda
__global__ void jumpGameKernel(int* arrays, int* results, int n, int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    int* nums = arrays + tid * n;
    int jumps = 0, current_end = 0, farthest = 0;
    
    for (int i = 0; i < n - 1; i++) {
        farthest = max(farthest, i + nums[i]);
        if (i == current_end) {
            jumps++;
            current_end = farthest;
        }
    }
    
    results[tid] = jumps;
}
```

## 22. Interview Deep Dive: Jump Game V

**Problem:** Given `arr` and `d`, you can jump at most `d` indices away. You can only jump to indices with smaller values. Find the maximum number of indices you can visit.

**Example:**
```
Input: arr = [6,4,14,6,8,13,9,7,10,6,12], d = 2
Output: 4
Explanation: 6 -> 4 -> 8 -> 6 (indices 0 -> 1 -> 4 -> 3)
```

**Algorithm (DP with Sorting):**
```python
def maxJumps(arr, d):
    n = len(arr)
    dp = [1] * n  # dp[i] = max visits starting from i
    
    # Sort indices by value (process smaller values first)
    indices = sorted(range(n), key=lambda i: arr[i])
    
    for i in indices:
        # Try jumping left
        for j in range(i - 1, max(-1, i - d - 1), -1):
            if arr[j] >= arr[i]:
                break  # Can't jump to taller
            dp[i] = max(dp[i], dp[j] + 1)
        
        # Try jumping right
        for j in range(i + 1, min(n, i + d + 1)):
            if arr[j] >= arr[i]:
                break
            dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)
```

**Complexity:** $O(N \log N + N \cdot d)$.

## 23. Conclusion

Jump Game II is a beautiful problem that teaches us the power of greedy algorithms. The key insight—that we can make locally optimal choices to achieve a globally optimal solution—is a recurring theme in algorithm design.

**Key Takeaways:**
*   **Greedy > DP:** For this problem, greedy is simpler and faster.
*   **BFS Perspective:** Thinking in terms of "levels" helps visualize the solution.
*   **Proof Techniques:** Exchange arguments are powerful for proving greedy correctness.
*   **Real-World Applications:** Routing, pathfinding, resource allocation.

The variants (Jump Game I, III, IV, V, Frog Jump) test your ability to adapt the core algorithm to different constraints. Mastering these variations prepares you for a wide range of interview questions.

