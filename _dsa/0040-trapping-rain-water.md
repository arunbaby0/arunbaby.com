---
title: "Trapping Rain Water"
day: 40
related_ml_day: 40
related_speech_day: 40
related_agents_day: 40
collection: dsa
categories:
 - dsa
tags:
 - array
 - two-pointers
 - dynamic-programming
 - stack
 - monotonic-stack
difficulty: Hard
---

**"Calculating capacity in a fragmented landscape."**

## 1. Problem Statement

Given `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining.

**Example 1:**
``
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
``

**Example 2:**
``
Input: height = [4,2,0,3,2,5]
Output: 9
``

**Constraints:**
* `n == height.length`
* `1 <= n <= 2 * 10^4`
* `0 <= height[i] <= 10^5`

## 2. Intuition

The core idea is to understand what determines the amount of water stored at a specific index `i`.
Water can be trapped at index `i` if there are higher bars on both the left and right sides.
The amount of water at index `i` is determined by the shorter of the two tallest bars surrounding it, minus the height of the bar at `i` itself.

Formula:
`Water[i] = min(max_left[i], max_right[i]) - height[i]`
(If the result is negative, it means the bar is higher than the water level, so `Water[i] = 0`).

## 3. Approach 1: Brute Force

For each element, we iterate left to find the maximum height, and iterate right to find the maximum height.

``python
class Solution:
 def trap(self, height: List[int]) -> int:
 n = len(height)
 ans = 0
 for i in range(n):
 max_left = 0
 max_right = 0
 
 # Search left
 for j in range(i, -1, -1):
 max_left = max(max_left, height[j])
 
 # Search right
 for j in range(i, n):
 max_right = max(max_right, height[j])
 
 ans += min(max_left, max_right) - height[i]
 return ans
``

**Complexity:**
* **Time:** O(N^2). For each element, we scan the array.
* **Space:** O(1).

## 4. Approach 2: Dynamic Programming (Pre-computation)

We can optimize the brute force approach by pre-computing the `max_left` and `max_right` for every index.

1. Create an array `left_max` of size `n`. `left_max[i]` will store the maximum height from index `0` to `i`.
2. Create an array `right_max` of size `n`. `right_max[i]` will store the maximum height from index `i` to `n-1`.
3. Iterate and compute the answer.

``python
class Solution:
 def trap(self, height: List[int]) -> int:
 if not height:
 return 0
 
 n = len(height)
 left_max = [0] * n
 right_max = [0] * n
 
 left_max[0] = height[0]
 for i in range(1, n):
 left_max[i] = max(height[i], left_max[i-1])
 
 right_max[n-1] = height[n-1]
 for i in range(n-2, -1, -1):
 right_max[i] = max(height[i], right_max[i+1])
 
 ans = 0
 for i in range(n):
 ans += min(left_max[i], right_max[i]) - height[i]
 
 return ans
``

**Complexity:**
* **Time:** O(N). Three passes (left, right, calculate).
* **Space:** O(N) for the two auxiliary arrays.

## 5. Approach 3: Two Pointers (Space Optimization)

Notice that for `left_max[i]`, we only need the maximum to the left. In the DP approach, we calculate all `left_max` and `right_max` first.
However, if we use two pointers, `left` and `right`, we can maintain `left_max` and `right_max` on the fly.

The key insight:
If `left_max < right_max`, then the water level at the `left` pointer is determined by `left_max`. It doesn't matter how tall the `right_max` actually is, as long as it's taller than `left_max`, the bottleneck is `left_max`.

``python
class Solution:
 def trap(self, height: List[int]) -> int:
 if not height:
 return 0
 
 left, right = 0, len(height) - 1
 left_max, right_max = 0, 0
 ans = 0
 
 while left < right:
 if height[left] < height[right]:
 if height[left] >= left_max:
 left_max = height[left]
 else:
 ans += left_max - height[left]
 left += 1
 else:
 if height[right] >= right_max:
 right_max = height[right]
 else:
 ans += right_max - height[right]
 right -= 1
 return ans
``

**Complexity:**
* **Time:** O(N). Single pass.
* **Space:** O(1).

## 6. Approach 4: Monotonic Stack

We can use a stack to keep track of the bars that are bounded by longer bars and hence, may store water.
We keep the stack decreasing. When we see a bar taller than the top of the stack, it means we found a right boundary for the bars in the stack.

``python
class Solution:
 def trap(self, height: List[int]) -> int:
 stack = [] # Stores indices
 ans = 0
 current = 0
 
 while current < len(height):
 while stack and height[current] > height[stack[-1]]:
 top = stack.pop()
 if not stack:
 break
 
 distance = current - stack[-1] - 1
 bounded_height = min(height[current], height[stack[-1]]) - height[top]
 ans += distance * bounded_height
 
 stack.append(current)
 current += 1
 
 return ans
``

**Complexity:**
* **Time:** O(N). Each element is pushed and popped at most once.
* **Space:** O(N) for the stack.

## 7. Deep Dive: Trapping Rain Water II (3D)

What if the input is a 2D grid `heightMap[m][n]`?
Now water can spill in 4 directions.
The boundary is no longer just left/right, but the contour surrounding a cell.

**Approach: Priority Queue (Min-Heap)**
1. Add all boundary cells to a Min-Heap. These form the initial "wall".
2. Mark boundary cells as visited.
3. While heap is not empty:
 * Pop the cell with the minimum height (`h`). This is the lowest point in the current wall. Water cannot be higher than this point without spilling.
 * Check its 4 neighbors.
 * If a neighbor is unvisited:
 * If neighbor's height < `h`, it traps `h - neighbor_height` water. Push `h` (the water level) to heap.
 * If neighbor's height >= `h`, it becomes a new part of the wall. Push `neighbor_height` to heap.
 * Mark neighbor as visited.

``python
import heapq

class Solution:
 def trapRainWater(self, heightMap: List[List[int]]) -> int:
 if not heightMap or not heightMap[0]:
 return 0
 
 m, n = len(heightMap), len(heightMap[0])
 visited = [[False] * n for _ in range(m)]
 heap = []
 
 # Add border cells
 for i in range(m):
 for j in range(n):
 if i == 0 or i == m - 1 or j == 0 or j == n - 1:
 heapq.heappush(heap, (heightMap[i][j], i, j))
 visited[i][j] = True
 
 ans = 0
 directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
 
 while heap:
 h, x, y = heapq.heappop(heap)
 
 for dx, dy in directions:
 nx, ny = x + dx, y + dy
 if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny]:
 visited[nx][ny] = True
 ans += max(0, h - heightMap[nx][ny])
 # The new boundary height is max(current_boundary, neighbor_height)
 heapq.heappush(heap, (max(h, heightMap[nx][ny]), nx, ny))
 
 return ans
``

## 8. System Design: Flood Prediction & Capacity Planning

While "Trapping Rain Water" is an algorithmic puzzle, it maps directly to **Hydrological Modeling** and **Resource Capacity Planning**.

### 8.1. Hydrological Modeling (GIS)
In Geographic Information Systems (GIS), Digital Elevation Models (DEM) are used to simulate flooding.
* **Sink Filling:** The algorithm we used for 3D trapping is essentially a "sink filling" algorithm used to remove depressions in DEMs before hydrological analysis.
* **Flow Direction:** Determining where water flows (steepest descent).
* **Accumulation:** Calculating how much water drains through a specific cell.

### 8.2. Resource Capacity Planning (The "Leaky Bucket")
Imagine a system where requests (water) arrive and are processed (drained) or buffered (trapped).
* **Height:** Represents the processing capacity or buffer limit at a specific node.
* **Trapped Water:** Represents the backlog or queue depth.
* **Bottleneck:** The "max_left" and "max_right" represent the constraints of upstream and downstream dependencies.

**Scenario:** A microservices chain.
Service A -> Service B -> Service C.
If Service B has low throughput (low height) but A and C have high throughput, requests might pile up (trap) at B if there isn't proper backpressure.
However, the analogy is slightly inverse here; usually, a "low" bar in the problem means capacity to hold water, whereas in systems, a "low" bar usually means low capacity.
A better analogy: **Resource Pooling**.
* You have a cluster of servers with varying capacities (heights).
* You want to "fill" the cluster with jobs.
* The total capacity is determined by the "peaks" (high capacity nodes) that can handle the load distribution.

## 9. Deep Dive: Container With Most Water

A related but distinct problem.
**Problem:** Given `height` array, find two lines that together with the x-axis form a container, such that the container contains the most water.
**Difference:**
* **Trapping Rain Water:** Calculates total volume over the *entire* terrain. Considers local dips.
* **Container With Most Water:** Calculates the *single largest rectangle* formed by two outer lines. Ignores the bars in between (assumes they don't exist or we are choosing the outer walls of a tank).

**Algorithm (Two Pointers):**
1. Start with `left = 0`, `right = n - 1`.
2. Area = `min(height[left], height[right]) * (right - left)`.
3. Move the shorter pointer inward. Why? Moving the taller pointer can only decrease the width without increasing the height (limited by the shorter one). Moving the shorter one *might* find a taller line.

``python
class Solution:
 def maxArea(self, height: List[int]) -> int:
 left, right = 0, len(height) - 1
 max_area = 0
 
 while left < right:
 width = right - left
 h = min(height[left], height[right])
 max_area = max(max_area, width * h)
 
 if height[left] < height[right]:
 left += 1
 else:
 right -= 1
 
 return max_area
``

## 10. Deep Dive: Pour Water

This is a simulation variant often asked in interviews (e.g., Airbnb).
**Problem:** Given an elevation map `heights`, and `V` units of water falling at index `K`.
**Rules:**
1. Water falls at `K`.
2. It tries to move **Left** if the neighbor is lower. It continues moving left until it can't (either hits a wall or a flat surface).
3. If it can't move left, it tries to move **Right** similarly.
4. If it can't move left or right, it stays at the current position (increments height).
5. Repeat for `V` drops.

**Algorithm:**
Simulate drop by drop.
For each drop:
1. Start at `curr = K`.
2. **Scan Left:** Find the lowest point to the left.
 * While `heights[curr - 1] <= heights[curr]`, move left.
 * If we find a dip (strictly smaller), record it.
3. **Scan Right:** If no drop position found on left, scan right.
4. **Update:** Increment `heights[best_index]`.

``python
class Solution:
 def pourWater(self, heights: List[int], volume: int, k: int) -> List[int]:
 for _ in range(volume):
 curr = k
 
 # Try to move left
 while curr > 0 and heights[curr - 1] <= heights[curr]:
 curr -= 1
 
 # If we went left but are now climbing back up, we need to find the "valley"
 # The simple while loop above goes to the edge of the plateau.
 # We need to be careful: Water flows to the *lowest* point.
 
 # Correct Simulation Logic:
 # 1. Find lowest point on left.
 l = k
 best_l = k
 while l > 0 and heights[l - 1] <= heights[l]:
 l -= 1
 if heights[l] < heights[best_l]:
 best_l = l
 
 if best_l != k:
 heights[best_l] += 1
 continue
 
 # 2. If not left, find lowest point on right.
 r = k
 best_r = k
 while r < len(heights) - 1 and heights[r + 1] <= heights[r]:
 r += 1
 if heights[r] < heights[best_r]:
 best_r = r
 
 if best_r != k:
 heights[best_r] += 1
 continue
 
 # 3. Stay at K
 heights[k] += 1
 
 return heights
``

## 11. Deep Dive: Largest Rectangle in Histogram

This problem is the "inverse" of Trapping Rain Water in terms of Monotonic Stack usage.
**Problem:** Given heights of bars, find the largest rectangle that can be formed within the histogram.
**Example:** `[2, 1, 5, 6, 2, 3]` -> Max Area = 10 (bars 5 and 6, min height 5 * width 2).

**Algorithm (Monotonic Increasing Stack):**
* We want to find, for each bar `i`, the first smaller bar to the left (`L`) and first smaller bar to the right (`R`).
* The width of the rectangle with height `heights[i]` is `R - L - 1`.
* We maintain a stack of indices with *increasing* heights.
* When we see a bar `h[i]` smaller than `h[stack.top()]`, it means `i` is the **Right Boundary** for the bar at `stack.top()`.
* The **Left Boundary** is the new `stack.top()` (after popping).

``python
class Solution:
 def largestRectangleArea(self, heights: List[int]) -> int:
 # Append 0 to flush the stack at the end
 heights.append(0)
 stack = [-1]
 max_area = 0
 
 for i, h in enumerate(heights):
 while stack[-1] != -1 and h < heights[stack[-1]]:
 height = heights[stack.pop()]
 width = i - stack[-1] - 1
 max_area = max(max_area, height * width)
 stack.append(i)
 
 heights.pop() # Restore array
 return max_area
``

**Connection to Trapping Rain Water:**
* **Trapping Rain Water:** Stack stores *decreasing* heights. We calculate area when we find a *taller* bar (forming a basin).
* **Largest Rectangle:** Stack stores *increasing* heights. We calculate area when we find a *shorter* bar (limiting the rectangle's extension).

## 12. Detailed Walkthrough: Two Pointers

Let's trace `height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]`.
`left = 0`, `right = 11`. `left_max = 0`, `right_max = 0`. `ans = 0`.

1. `h[0] (0) < h[11] (1)`.
 * `h[0] >= left_max (0)`? Yes. `left_max = 0`.
 * `left` moves to 1.
2. `h[1] (1) <= h[11] (1)`.
 * `h[1] >= left_max (0)`? Yes. `left_max = 1`.
 * `left` moves to 2.
3. `h[2] (0) < h[11] (1)`.
 * `h[2] >= left_max (1)`? No.
 * `ans += 1 - 0 = 1`.
 * `left` moves to 3.
4. `h[3] (2) > h[11] (1)`. (Switch to Right side logic)
 * `h[11] >= right_max (0)`? Yes. `right_max = 1`.
 * `right` moves to 10.
5. `h[3] (2) <= h[10] (2)`.
 * `h[3] >= left_max (1)`? Yes. `left_max = 2`.
 * `left` moves to 4.
6. `h[4] (1) < h[10] (2)`.
 * `h[4] >= left_max (2)`? No.
 * `ans += 2 - 1 = 1`. (Total: 2)
 * `left` moves to 5.
... and so on.

**Why it works:**
At step 3, `left_max` is 1. We don't know the true `right_max` for index 2. We only know that `h[11]` is 1, so the true `right_max` is *at least* 1.
Since `left_max (1) <= right_max (>=1)`, the water level is determined by `left_max`.
The condition `if height[left] < height[right]` essentially guarantees that `left_max < right_max` (roughly speaking, or at least that the left side is the bottleneck).

## 13. System Design: Rate Limiting (Token Bucket)

The "Trapping Rain Water" concept of filling and draining maps to **Token Bucket** and **Leaky Bucket** algorithms.

**Leaky Bucket:**
* Water (requests) enters the bucket at an irregular rate.
* Water leaks from the bottom at a constant rate.
* If the bucket overflows, water is lost (requests dropped).
* **Analogy:** `height` is the bucket capacity. The trapped water is the current queue.

**Token Bucket:**
* Tokens are added to the bucket at a constant rate.
* The bucket has a max capacity (`height`).
* When a request comes, it must consume a token.
* Allows for "bursts" of traffic up to the bucket size.

**Implementation in Redis (Lua Script):**
``lua
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local fill_time = capacity / rate
local ttl = math.floor(fill_time * 2)

local last_tokens = tonumber(redis.call("get", key))
if last_tokens == nil then
 last_tokens = capacity
end

local last_refreshed = tonumber(redis.call("get", key .. ":ts"))
if last_refreshed == nil then
 last_refreshed = 0
end

local delta = math.max(0, now - last_refreshed)
local filled_tokens = math.min(capacity, last_tokens + (delta * rate))
local allowed = filled_tokens >= requested
local new_tokens = filled_tokens

if allowed then
 new_tokens = filled_tokens - requested
end

redis.call("setex", key, ttl, new_tokens)
redis.call("setex", key .. ":ts", ttl, now)

return allowed
``

## 15. Deep Dive: Segment Tree Approach (Range Max Query)

While O(N) is optimal, understanding the **Segment Tree** solution is valuable for variations where updates happen.
**Scenario:** What if the elevation map changes dynamically? `update(index, new_height)`.
We need to query `max_left` and `max_right` in O(\log N) time.

**Structure:**
* Build a Segment Tree where each node stores the `max` of its range.
* `query(L, R)` returns `\max(height[L \dots R])`.
* `max_left[i] = query(0, i)`.
* `max_right[i] = query(i, n-1)`.

**Complexity:**
* Build: O(N).
* Query: O(\log N).
* Update: O(\log N).
* Total for static array: O(N \log N).

``python
class SegmentTree:
 def __init__(self, data):
 self.n = len(data)
 self.tree = [0] * (4 * self.n)
 self._build(data, 0, 0, self.n - 1)
 
 def _build(self, data, node, start, end):
 if start == end:
 self.tree[node] = data[start]
 else:
 mid = (start + end) // 2
 self._build(data, 2 * node + 1, start, mid)
 self._build(data, 2 * node + 2, mid + 1, end)
 self.tree[node] = max(self.tree[2 * node + 1], self.tree[2 * node + 2])
 
 def query(self, L, R):
 return self._query(0, 0, self.n - 1, L, R)
 
 def _query(self, node, start, end, L, R):
 if R < start or end < L:
 return 0
 if L <= start and end <= R:
 return self.tree[node]
 mid = (start + end) // 2
 p1 = self._query(2 * node + 1, start, mid, L, R)
 p2 = self._query(2 * node + 2, mid + 1, end, L, R)
 return max(p1, p2)

# Usage in Trap Rain Water
# ans += min(st.query(0, i), st.query(i, n-1)) - height[i]
``

## 16. Deep Dive: Parallel Algorithm (Prefix Sums / Scan)

How do we solve this on a GPU with 10,000 cores? We can't use Two Pointers (sequential).
We use **Parallel Prefix Scan**.

**Algorithm:**
1. **Parallel Max-Scan (Left):** Compute `left_max` array.
 * Operation: `max`.
 * Input: `height`.
 * Output: `left_max`.
 * Complexity: O(\log N) depth, O(N) work using Hillis-Steele or Blelloch scan.
2. **Parallel Max-Scan (Right):** Compute `right_max` array (reverse scan).
3. **Parallel Map:** Compute `min(left_max[i], right_max[i]) - height[i]` for all `i` in parallel.
4. **Parallel Reduce (Sum):** Sum the results.

**CUDA Kernel Logic (Simplified):**
``cpp
__global__ void compute_water(int* height, int* left_max, int* right_max, int* water, int n) {
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < n) {
 water[i] = min(left_max[i], right_max[i]) - height[i];
 }
}
``
This demonstrates how algorithmic choices change based on hardware (CPU vs GPU).

## 17. Mathematical Proof of Two Pointers Correctness

**Theorem:** The Two Pointers algorithm correctly computes `min(max_left[i], max_right[i])` for all `i`.

**Proof by Invariant:**
Invariant: At any step, `left_max` is the true maximum of `height[0...left]` and `right_max` is the true maximum of `height[right...n-1]`.

**Case 1:** `height[left] < height[right]`.
* We know `left_max` is the max of the left prefix.
* We know `right_max` is the max of the right suffix.
* Since `height[right]` exists and is greater than `height[left]`, and `right_max >= height[right]`, it implies `right_max > left_max`.
* Therefore, `min(true_max_left, true_max_right)` for the element at `left` MUST be `left_max`.
 * `true_max_left` is exactly `left_max` (by definition).
 * `true_max_right` is *at least* `height[right]`, which is greater than `height[left]` (and potentially `left_max`).
 * Actually, strictly speaking, we know `true_max_right >= right_max`.
 * If `left_max < right_max`, then `left_max < true_max_right`.
 * So `min(left_max, true_max_right) = left_max`.
* Thus, we can safely calculate water for `left` using only `left_max`.

**Case 2:** `height[left] >= height[right]`.
* Symmetric argument. `right_max` is the bottleneck.

**Conclusion:** The algorithm never underestimates or overestimates the water level because it always processes the side with the *smaller* known maximum, ensuring that the *other* side is guaranteed to be large enough to hold the water.

## 18. Comprehensive Performance Benchmarking

Let's simulate a large-scale benchmark.

``python
import time
import random
import sys

# Increase recursion depth for deep recursion tests
sys.setrecursionlimit(20000)

def benchmark_suite():
 sizes = [1000, 10000, 100000, 1000000]
 results = {}
 
 for size in sizes:
 print(f"Benchmarking size: {size}")
 height = [random.randint(0, 10000) for _ in range(size)]
 
 # 1. Brute Force (Skip for large sizes)
 if size <= 10000:
 start = time.time()
 # brute_force(height)
 end = time.time()
 results[f"Brute Force {size}"] = end - start
 
 # 2. DP
 start = time.time()
 # dp_solution(height)
 end = time.time()
 results[f"DP {size}"] = end - start
 
 # 3. Two Pointers
 start = time.time()
 # two_pointers(height)
 end = time.time()
 results[f"Two Pointers {size}"] = end - start
 
 # 4. Stack
 start = time.time()
 # stack_solution(height)
 end = time.time()
 results[f"Stack {size}"] = end - start

 print("\nResults (Seconds):")
 for k, v in results.items():
 print(f"{k}: {v:.6f}")

# Expected Output Trend:
# Brute Force: Quadratic explosion.
# DP: Fast, but 3 passes + memory allocation overhead.
# Stack: Fast, but push/pop overhead.
# Two Pointers: Fastest. Single pass, cache friendly, no extra memory.
``

## 19. Interview Questions (Advanced)

1. **Trapping Rain Water (Classic):** Solve in O(N) time and O(1) space.
2. **Trapping Rain Water II (2D):** Solve using a Heap. What is the time complexity? (`O(MN \log(MN))`).
3. **Pour Water:** Given an elevation map and `V` units of water falling at index `K`, simulate where the water lands. (Simulates gravity: water drops, moves left if possible, else right, else stays).
4. **Largest Rectangle in Histogram:** Related stack problem.
5. **Product of Array Except Self:** Similar "Left/Right" array pattern.

## 11. Common Mistakes

* **Corner Cases:** Empty array, array with < 3 elements (cannot trap water).
* **DP Space:** Forgetting that DP takes O(N) space and not optimizing to Two Pointers if asked.
* **Stack Logic:** Confusing when to push/pop in the monotonic stack approach. Remember: we pop when we find a *taller* bar (right boundary).
* **3D Boundary:** In the 2D problem, forgetting to add *all* boundary cells to the heap initially.

## 12. Performance Benchmarking

Let's compare the Python implementations.

``python
import time
import random

def benchmark():
 size = 1000000
 height = [random.randint(0, 1000) for _ in range(size)]
 
 # Two Pointers
 start = time.time()
 # ... (impl)
 # end = time.time()
 # Two pointers is generally the fastest due to cache locality and single pass logic.
``

(Detailed benchmarking code would go here).


---

**Originally published at:** [arunbaby.com/dsa/0040-trapping-rain-water](https://www.arunbaby.com/dsa/0040-trapping-rain-water/)

*If you found this helpful, consider sharing it with others who might benefit.*

