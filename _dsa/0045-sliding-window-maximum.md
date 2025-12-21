---
title: "Sliding Window Maximum"
day: 45
collection: dsa
categories:
  - dsa
tags:
  - sliding-window
  - deque
  - monotonic-deque
  - array
difficulty: Hard
---

**"Finding the king of every window."**

## 1. Problem Statement

You are given an array of integers `nums`, and there is a sliding window of size `k` which moves from the very left to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position, return the max of each window.

**Example:**
```
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation:
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

**Constraints:**
*   `1 <= nums.length <= 10^5`
*   `-10^4 <= nums[i] <= 10^4`
*   `1 <= k <= nums.length`

## 2. Intuition

**Naive Approach:** For each window, scan all `k` elements to find max. $O(n \cdot k)$.

**Key Insight:** We don't need to consider elements that are:
1.  **Out of the window:** Elements more than `k` positions behind.
2.  **Dominated:** If `nums[j] >= nums[i]` and `j > i`, then `nums[i]` can never be the max.

**Solution:** Use a **monotonic deque** that stores indices of useful elements (in decreasing order of their values).

## 3. Approach 1: Brute Force

```python
def maxSlidingWindow(nums, k):
    n = len(nums)
    result = []
    
    for i in range(n - k + 1):
        window_max = max(nums[i:i+k])
        result.append(window_max)
    
    return result
```

**Complexity:**
*   **Time:** $O(n \cdot k)$.
*   **Space:** $O(1)$ (excluding output).

## 4. Approach 2: Monotonic Deque (Optimal)

**Algorithm:**
1.  Maintain a deque of indices in decreasing order of values.
2.  For each element:
    *   Remove indices outside the current window (from front).
    *   Remove indices of elements smaller than current (from back).
    *   Add current index (to back).
    *   If window is complete, front of deque is the max.

```python
from collections import deque

def maxSlidingWindow(nums, k):
    dq = deque()  # Stores indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements (they'll never be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        # Add current index
        dq.append(i)
        
        # Add to result if window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

**Complexity:**
*   **Time:** $O(n)$. Each element is added and removed at most once.
*   **Space:** $O(k)$ for the deque.

## 5. Detailed Walkthrough

**Example:** `nums = [1,3,-1,-3,5,3,6,7]`, `k = 3`

| i | nums[i] | Deque (indices) | Deque (values) | Window Max |
|---|---------|-----------------|----------------|------------|
| 0 | 1 | [0] | [1] | - |
| 1 | 3 | [1] | [3] | - |
| 2 | -1 | [1, 2] | [3, -1] | 3 |
| 3 | -3 | [1, 2, 3] | [3, -1, -3] | 3 |
| 4 | 5 | [4] | [5] | 5 |
| 5 | 3 | [4, 5] | [5, 3] | 5 |
| 6 | 6 | [6] | [6] | 6 |
| 7 | 7 | [7] | [7] | 7 |

**Observations:**
*   At i=1: 3 > 1, so we remove index 0.
*   At i=4: 5 > -3, -1, 3, so we remove all and start fresh.
*   The front of the deque always has the maximum.

## 6. Why Monotonic Deque Works

**Invariant:** The deque is always:
1.  **Sorted by index:** Indices are in increasing order.
2.  **Sorted by value:** Values are in decreasing order.

**Key Property:** If we see a larger element, all smaller preceding elements become irrelevant—they can never be the max for any future window.

**Amortized Analysis:** Each element is pushed once and popped at most once. Total operations = $O(2n) = O(n)$.

## 7. Approach 3: Segment Tree

**Build a segment tree for range maximum queries.**

```python
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.build(arr, 0, 0, self.n - 1)
    
    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self.build(arr, 2*node+1, start, mid)
            self.build(arr, 2*node+2, mid+1, end)
            self.tree[node] = max(self.tree[2*node+1], self.tree[2*node+2])
    
    def query(self, node, start, end, l, r):
        if r < start or l > end:
            return float('-inf')
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        return max(
            self.query(2*node+1, start, mid, l, r),
            self.query(2*node+2, mid+1, end, l, r)
        )

def maxSlidingWindow(nums, k):
    st = SegmentTree(nums)
    result = []
    for i in range(len(nums) - k + 1):
        result.append(st.query(0, 0, len(nums)-1, i, i+k-1))
    return result
```

**Complexity:**
*   **Time:** $O(n \log n)$ (build) + $O((n-k+1) \log n)$ (queries).
*   **Space:** $O(n)$ for the tree.

**When to use:** When you need both range max and point updates.

## 8. Approach 4: Sparse Table (Static)

For static arrays, sparse table gives $O(1)$ range max queries.

**Build:** $O(n \log n)$.
**Query:** $O(1)$.

**Trade-off:** No updates possible.

## 9. Extension: Sliding Window Minimum

Same algorithm, just reverse the comparison:
*   Remove elements **larger** than current (instead of smaller).

```python
def minSlidingWindow(nums, k):
    dq = deque()
    result = []
    
    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        while dq and nums[dq[-1]] > nums[i]:  # Changed comparison
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

## 10. Related Problem: Shortest Subarray with Sum at Least K

**Problem (LeetCode 862):** Find the shortest subarray with sum ≥ K (can have negative numbers).

**Approach:** Use monotonic deque on prefix sums.

**Key Insight:** For prefix sum array `P`:
*   We want shortest `j - i` where `P[j] - P[i] >= K`.
*   Deque stores indices in increasing order of `P[i]`.

```python
from collections import deque

def shortestSubarray(nums, k):
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + nums[i]
    
    dq = deque()
    result = float('inf')
    
    for i in range(n + 1):
        # Check if we can form a valid subarray
        while dq and prefix[i] - prefix[dq[0]] >= k:
            result = min(result, i - dq.popleft())
        
        # Maintain monotonicity
        while dq and prefix[dq[-1]] >= prefix[i]:
            dq.pop()
        
        dq.append(i)
    
    return result if result != float('inf') else -1
```

## 11. System Design: Real-Time Analytics

**Scenario:** Compute the maximum value in the last 5 minutes from a stream of events.

**Architecture:**
1.  Events arrive as (timestamp, value).
2.  Maintain a deque of (timestamp, value).
3.  On each event:
    *   Remove events older than 5 minutes.
    *   Remove smaller values (they'll never be max).
    *   Add new event.
4.  Query: Return the front of the deque.

**Use Cases:**
*   Max stock price in last N minutes.
*   Peak CPU usage in a time window.
*   Highest bid in an auction window.

## 12. Interview Questions

1.  **Sliding Window Maximum (Classic):** Solve with monotonic deque.
2.  **Sliding Window Minimum:** Variant with min instead of max.
3.  **Shortest Subarray with Sum ≥ K:** Prefix sums + deque.
4.  **Jump Game VI:** DP + monotonic deque.
5.  **Constrained Subsequence Sum:** DP + deque optimization.

## 13. Common Mistakes

*   **Off-by-One:** Window starts at index `k-1`, not `k`.
*   **Using Values Instead of Indices:** Store indices to check window boundaries.
*   **Wrong Removal Order:** Remove from front for window bounds, from back for monotonicity.
*   **Forgetting to Add Current Element:** Always append to deque.
*   **Edge Cases:** k = 1, k = n.

## 14. Variant: Count Elements Greater Than X in Each Window

**Problem:** For each window, count elements > X.

**Approach:**
1.  Preprocess: Create array where `a[i] = 1 if nums[i] > X else 0`.
2.  Use sliding window sum (prefix sum or two pointers).

## 15. Variant: Median of Sliding Window

**Problem:** Find the median of each window.

**Approach:**
*   Two heaps (like streaming median).
*   SortedList for O(log k) insertions/deletions.

```python
from sortedcontainers import SortedList

def medianSlidingWindow(nums, k):
    window = SortedList()
    result = []
    
    for i, num in enumerate(nums):
        window.add(num)
        
        if len(window) > k:
            window.remove(nums[i - k])
        
        if len(window) == k:
            if k % 2 == 1:
                result.append(window[k // 2])
            else:
                result.append((window[k // 2 - 1] + window[k // 2]) / 2)
    
    return result
```

## 16. Performance Comparison

| Approach | Time | Space | Use Case |
|----------|------|-------|----------|
| Brute Force | $O(n \cdot k)$ | $O(1)$ | Small k |
| Monotonic Deque | $O(n)$ | $O(k)$ | General (optimal) |
| Segment Tree | $O(n \log n)$ | $O(n)$ | Range queries + updates |
| Sparse Table | $O(n \log n)$ build, $O(1)$ query | $O(n \log n)$ | Static arrays |

## 17. Deep Dive: Deque Implementation

**Python `collections.deque`:**
*   Doubly-linked list implementation.
*   O(1) append/pop from both ends.
*   Thread-safe for single operations.

**Java `ArrayDeque`:**
*   Circular array implementation.
*   O(1) amortized for all operations.
*   More cache-friendly than linked list.

## 18. Application: Stock Price Analysis

**Scenario:** Find the maximum stock price in each trading window.

**Requirements:**
*   Support add new price.
*   Support query max in last k prices.
*   Support remove old prices.

**Implementation:**
```python
class StockAnalyzer:
    def __init__(self, window_size):
        self.k = window_size
        self.prices = deque()
        self.max_deque = deque()
    
    def add_price(self, price):
        # Remove old prices
        while len(self.prices) >= self.k:
            old = self.prices.popleft()
            if self.max_deque and self.max_deque[0] == old:
                self.max_deque.popleft()
        
        # Maintain monotonicity
        while self.max_deque and self.max_deque[-1] < price:
            self.max_deque.pop()
        
        self.prices.append(price)
        self.max_deque.append(price)
    
    def get_max(self):
        return self.max_deque[0] if self.max_deque else None
```

## 19. Conclusion

Sliding Window Maximum is a classic problem that teaches the power of monotonic data structures. The key insight is that we only need to track "useful" elements—those that could potentially be the maximum.

**Key Takeaways:**
*   **Monotonic Deque:** O(n) solution, O(k) space.
*   **Invariant:** Deque is sorted by both index and value.
*   **Amortized Analysis:** Each element pushed/popped once.
*   **Extensions:** Min, sum ≥ K, median.

This pattern appears frequently in interview problems. Master it, and you'll recognize it in many disguises.

## 20. Mastery Checklist

**Mastery Checklist:**
- [ ] Implement brute force O(nk) solution
- [ ] Implement monotonic deque O(n) solution
- [ ] Explain why elements are removed
- [ ] Solve sliding window minimum variant
- [ ] Solve Shortest Subarray with Sum ≥ K
- [ ] Implement with segment tree
- [ ] Handle edge cases (k=1, k=n)
- [ ] Apply to real-time analytics

## 21. Advanced Application: Jump Game VI

**Problem (LeetCode 1696):** Given array `nums` and integer `k`, start at index 0. From index `i`, you can jump to any index in `[i+1, min(i+k, n-1)]`. Score = sum of values at visited indices. Find maximum score to reach the last index.

**DP Recurrence:**
$$dp[i] = nums[i] + \max_{j=\max(0,i-k)}^{i-1} dp[j]$$

**Naive:** $O(n \cdot k)$.

**Optimized with Monotonic Deque:** $O(n)$.

```python
from collections import deque

def maxResult(nums, k):
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    
    dq = deque([0])  # Stores indices with max dp values
    
    for i in range(1, n):
        # Remove out-of-window indices
        while dq and dq[0] < i - k:
            dq.popleft()
        
        # dp[i] = nums[i] + max dp in window
        dp[i] = nums[i] + dp[dq[0]]
        
        # Maintain monotonicity (max deque)
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        
        dq.append(i)
    
    return dp[-1]
```

## 22. Advanced Application: Constrained Subsequence Sum

**Problem (LeetCode 1425):** Find maximum sum of a subsequence where consecutive elements are at most `k` indices apart.

**Same pattern:** DP + monotonic deque.

```python
def constrainedSubsetSum(nums, k):
    n = len(nums)
    dp = nums.copy()
    dq = deque()
    
    for i in range(n):
        if dq:
            dp[i] = max(dp[i], nums[i] + dp[dq[0]])
        
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        
        if dp[i] > 0:  # Only add positive contributions
            dq.append(i)
        
        while dq and dq[0] <= i - k:
            dq.popleft()
    
    return max(dp)
```

## 23. Pattern Recognition

**When to use Monotonic Deque:**
1.  **Sliding window problems** with max/min queries.
2.  **DP optimization** where state transition involves max/min over a range.
3.  **Range queries** with fixed window size.

**Key Questions:**
*   Do I need max/min of a sliding range?
*   Is the range size fixed or bounded?
*   Can I remove elements that will never be useful?

## 24. Implementation Variants

### 24.1. Using Tuple (Value, Index)

```python
def maxSlidingWindow(nums, k):
    dq = deque()  # (value, index)
    result = []
    
    for i, num in enumerate(nums):
        while dq and dq[0][1] <= i - k:
            dq.popleft()
        
        while dq and dq[-1][0] < num:
            dq.pop()
        
        dq.append((num, i))
        
        if i >= k - 1:
            result.append(dq[0][0])
    
    return result
```

### 24.2. Max-Heap with Lazy Deletion

```python
import heapq

def maxSlidingWindow(nums, k):
    heap = []  # Max-heap: (-value, index)
    result = []
    
    for i, num in enumerate(nums):
        heapq.heappush(heap, (-num, i))
        
        # Remove out-of-window elements
        while heap[0][1] <= i - k:
            heapq.heappop(heap)
        
        if i >= k - 1:
            result.append(-heap[0][0])
    
    return result
```

**Complexity:** $O(n \log n)$ due to lazy deletion.

## 25. Testing Strategy

**Test Cases:**
1.  **Basic:** `[1,3,-1,-3,5,3,6,7], k=3` → `[3,3,5,5,6,7]`.
2.  **All Same:** `[5,5,5,5], k=2` → `[5,5,5]`.
3.  **Increasing:** `[1,2,3,4,5], k=3` → `[3,4,5]`.
4.  **Decreasing:** `[5,4,3,2,1], k=3` → `[5,4,3]`.
5.  **k=1:** `[1,2,3], k=1` → `[1,2,3]`.
6.  **k=n:** `[1,2,3], k=3` → `[3]`.
7.  **Negative Numbers:** `[-1,-3,-2,-4], k=2` → `[-1,-2,-2]`.

## 26. Complexity Deep Dive

**Time Complexity Proof:**

Each element:
*   Enters the deque once: $O(n)$ total.
*   Exits the deque at most once: $O(n)$ total.

Therefore, total operations = $O(2n) = O(n)$.

**Space Complexity:**

Deque stores at most `k` elements (window size).

Space = $O(k)$.

## 27. Related LeetCode Problems

*   **LC 239:** Sliding Window Maximum (this problem).
*   **LC 862:** Shortest Subarray with Sum at Least K.
*   **LC 1696:** Jump Game VI.
*   **LC 1425:** Constrained Subsequence Sum.
*   **LC 480:** Sliding Window Median.
*   **LC 1438:** Longest Continuous Subarray with Abs Diff ≤ Limit.

## 28. Conclusion

Sliding Window Maximum is a gateway to understanding monotonic data structures. The deque-based solution is elegant and efficient, demonstrating how maintaining invariants can lead to optimal algorithms.

**Key Insights:**
1.  **Monotonic Invariant:** Elements in decreasing order of value.
2.  **Amortized O(n):** Each element pushed/popped once.
3.  **Template:** The pattern applies to many DP optimizations.
4.  **Real-World:** Stock analysis, sensor monitoring, network analytics.

Once you internalize this pattern, you'll see opportunities to apply it everywhere. It's one of the most versatile techniques in the algorithmic toolbox.

**Practice Challenge:** Implement sliding window maximum without using any library data structures (raw arrays only). This deepens understanding of the underlying mechanics.



---

**Originally published at:** [arunbaby.com/dsa/0045-sliding-window-maximum](https://www.arunbaby.com/dsa/0045-sliding-window-maximum/)

*If you found this helpful, consider sharing it with others who might benefit.*

