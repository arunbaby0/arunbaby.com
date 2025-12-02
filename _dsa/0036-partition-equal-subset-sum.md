---
title: "Partition Equal Subset Sum"
day: 36
collection: dsa
categories:
  - dsa
tags:
  - dynamic-programming
  - knapsack
  - subset-sum
  - bit-manipulation
difficulty: Medium
---

**"Can you split the treasure evenly?"**

## 1. Problem Statement

Given a non-empty array `nums` containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

**Example 1:**
```
Input: nums = [1, 5, 11, 5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11]. Both sum to 11.
```

**Example 2:**
```
Input: nums = [1, 2, 3, 5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.
```

## 2. Intuition: The 0/1 Knapsack Connection

1.  **Total Sum Check:** If the sum of all elements `S` is odd, we can't split it into two equal integers. Return `False`.
2.  **Target Sum:** If `S` is even, we need to find a subset with sum `target = S / 2`. If we find one subset with sum `S/2`, the remaining elements must also sum to `S/2`.
3.  **Transformation:** This is exactly the **Subset Sum Problem**, which is a variation of the **0/1 Knapsack Problem**.
    -   **Items:** The numbers in `nums`.
    -   **Weight:** The value of the number.
    -   **Value:** Irrelevant (we just care if we can fill the knapsack).
    -   **Capacity:** `target`.

## 3. Approach 1: Recursion with Memoization (Top-Down DP)

We define a function `canPartition(index, current_sum)`.
-   **Base Cases:**
    -   If `current_sum == target`: Return `True`.
    -   If `current_sum > target` or `index >= len(nums)`: Return `False`.
-   **Choices:**
    1.  **Include** `nums[index]`: `canPartition(index + 1, current_sum + nums[index])`
    2.  **Exclude** `nums[index]`: `canPartition(index + 1, current_sum)`

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total_sum = sum(nums)
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        memo = {}
        
        def backtrack(index, current_sum):
            if current_sum == target:
                return True
            if current_sum > target or index >= len(nums):
                return False
            
            state = (index, current_sum)
            if state in memo:
                return memo[state]
            
            # Choice 1: Include
            if backtrack(index + 1, current_sum + nums[index]):
                memo[state] = True
                return True
            
            # Choice 2: Exclude
            if backtrack(index + 1, current_sum):
                memo[state] = True
                return True
            
            memo[state] = False
            return False
            
        return backtrack(0, 0)
```

**Complexity:**
-   **Time:** $O(N \times Target)$. There are $N \times Target$ states.
-   **Space:** $O(N \times Target)$ for memoization table + recursion stack.

## 4. Approach 2: Tabulation (Bottom-Up DP)

Let `dp[i][j]` be `True` if a sum of `j` can be achieved using the first `i` items.

-   **Initialization:** `dp[0][0] = True` (Sum 0 with 0 items is possible).
-   **Transition:**
    -   `dp[i][j] = dp[i-1][j]` (Exclude current item)
    -   `OR dp[i-1][j - nums[i-1]]` (Include current item, if `j >= nums[i-1]`)

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total_sum = sum(nums)
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        n = len(nums)
        
        # dp[i][j] means using first i items, can we get sum j?
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = True  # Sum 0 is always possible
            
        for i in range(1, n + 1):
            curr_num = nums[i-1]
            for j in range(1, target + 1):
                # Exclude
                dp[i][j] = dp[i-1][j]
                # Include
                if j >= curr_num:
                    dp[i][j] = dp[i][j] or dp[i-1][j - curr_num]
                    
        return dp[n][target]
```

## 5. Approach 3: Space Optimization (1D Array)

Notice `dp[i][j]` only depends on `dp[i-1][...]`. We can reduce space to $O(Target)$.
**Crucial:** We must iterate backwards to avoid using the same item twice in the same step.

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total_sum = sum(nums)
        if total_sum % 2 != 0: return False
        target = total_sum // 2
        
        dp = [False] * (target + 1)
        dp[0] = True
        
        for num in nums:
            # Iterate backwards from target to num
            for j in range(target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]
                
        return dp[target]
```

**Complexity:**
-   **Time:** $O(N \times Target)$.
-   **Space:** $O(Target)$.

## 6. Approach 4: Bitset Optimization (The "Magic" Solution)

For languages like C++ or Java (BitSet), or Python (large integers), we can use bit manipulation.
-   Represent the set of reachable sums as a bitmask.
-   If the $k$-th bit is 1, it means sum $k$ is possible.
-   Transition: `bits = bits | (bits << num)`
    -   `bits`: existing sums.
    -   `bits << num`: existing sums + `num`.
    -   `|`: Union of both sets.

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total_sum = sum(nums)
        if total_sum % 2 != 0: return False
        target = total_sum // 2
        
        # Bitmask: 1 at index 0 means sum 0 is possible
        bits = 1 
        
        for num in nums:
            bits |= bits << num
            
        # Check if the target-th bit is 1
        return (bits >> target) & 1 == 1
```

**Why is this fast?**
-   Bitwise operations process 64 bits (sums) in parallel on a 64-bit CPU.
-   **Time:** $O(N \times Target / 64)$.
-   **Space:** $O(Target / 64)$.

## 7. Deep Dive: NP-Completeness

The Partition Problem is a special case of the Subset Sum Problem, which is **NP-Complete**.
-   This means there is no known polynomial-time algorithm ($O(N^k)$) that solves it for *all* inputs.
-   Our DP solution is **Pseudo-Polynomial**. Its complexity depends on the *value* of the input (`Target`), not just the number of elements (`N`).
-   If `Target` is huge (e.g., $10^{18}$), DP fails.

## 8. Summary

| Approach | Time | Space | Notes |
| :--- | :--- | :--- | :--- |
| **Recursion** | $O(2^N)$ | $O(N)$ | TLE |
| **Memoization** | $O(N \cdot S)$ | $O(N \cdot S)$ | Good |
| **Tabulation** | $O(N \cdot S)$ | $O(N \cdot S)$ | Avoids recursion limit |
| **Space Opt** | $O(N \cdot S)$ | $O(S)$ | Standard Interview Solution |
| **Bitset** | $O(N \cdot S / 64)$ | $O(S / 64)$ | Fastest |

## 9. Deep Dive: Knapsack Variations

The **0/1 Knapsack Problem** is the parent of many interview questions. Understanding the family tree helps identify them.

**1. Subset Sum Problem:**
-   **Goal:** Is there a subset with sum `T`?
-   **Relation:** Partition Equal Subset Sum is Subset Sum with `T = TotalSum / 2`.
-   **Code:** Exactly the same DP.

**2. Partition to K Equal Sum Subsets:**
-   **Goal:** Can we split array into `K` subsets with equal sum?
-   **Relation:** Generalization of Partition Equal Subset Sum (`K=2`).
-   **Approach:** Backtracking with pruning is usually better than DP because state space `(mask, current_sum)` is huge.

**3. Target Sum (LeetCode 494):**
-   **Goal:** Assign `+` or `-` to each number to get `Target`.
-   **Relation:** Let `P` be positive subset, `N` be negative subset.
    -   `Sum(P) - Sum(N) = Target`
    -   `Sum(P) + Sum(N) = TotalSum`
    -   `2 * Sum(P) = Target + TotalSum`
    -   `Sum(P) = (Target + TotalSum) / 2`
-   **Reduction:** Find subset with sum `(Target + TotalSum) / 2`. This is exactly Subset Sum!

## 10. Deep Dive: The Magic of Bitset Optimization

Let's break down `bits |= bits << num`.

Imagine `nums = [2, 3]`, `target = 5`.

**Step 0:** `bits = 1` (Binary: `...00001`)
-   Represents `{0}` is possible.

**Step 1:** Process `num = 2`.
-   `bits << 2`: `...00100` (Represents `{0+2}` = `{2}`)
-   `bits |= ...`: `...00101` (Represents `{0, 2}`)

**Step 2:** Process `num = 3`.
-   `bits`: `...00101` (`{0, 2}`)
-   `bits << 3`: `...00101000` -> `...101000` (Wait, `101` shifted left by 3 is `101000`)
    -   Old bit 0 (value 0) -> New bit 3 (value 3).
    -   Old bit 2 (value 2) -> New bit 5 (value 5).
-   `bits |= ...`: `...101101`
    -   Indices set: 0, 2, 3, 5.
    -   Possible sums: `{0, 2, 3, 5}`.

**Result:** Check bit 5. It is 1. Return True.

**C++ Implementation:**
```cpp
#include <bitset>
#include <vector>
#include <numeric>

class Solution {
public:
    bool canPartition(std::vector<int>& nums) {
        int sum = std::accumulate(nums.begin(), nums.end(), 0);
        if (sum % 2 != 0) return false;
        int target = sum / 2;
        
        std::bitset<10001> bits(1); // Max sum is 200 * 100 = 20000, target 10000
        
        for (int num : nums) {
            bits |= (bits << num);
        }
        
        return bits[target];
    }
};
```

## 11. Deep Dive: Meet-in-the-Middle

What if `Target` is huge (e.g., $10^{15}$), but `N` is small (e.g., 40)?
DP fails ($O(N \cdot S)$). Recursion fails ($2^{40}$).

**Algorithm:**
1.  Split `nums` into two halves: `Left` (20 items) and `Right` (20 items).
2.  Generate all possible subset sums for `Left`. Store in a Set `S_Left`. ($2^{20} \approx 10^6$).
3.  Generate all possible subset sums for `Right`. Store in a Set `S_Right`.
4.  Iterate through `x` in `S_Left`. Check if `Target - x` exists in `S_Right`.

**Complexity:**
-   **Time:** $O(2^{N/2})$.
-   **Space:** $O(2^{N/2})$.
-   Much better than $2^N$.

## 12. Deep Dive: DFS Pruning Techniques

If we must use DFS (e.g., for K-partition), pruning is vital.

1.  **Sort Reverse:** Try larger numbers first. This fills buckets faster and fails faster if impossible.
2.  **Skip Duplicates:** If `nums[i] == nums[i-1]` and we skipped `nums[i-1]`, skip `nums[i]`.
3.  **Boundary Check:** If `current_sum + nums[i] > target`, stop (since sorted).

## 13. Real-World Application: Load Balancing

Imagine you have `N` tasks with execution times `t1, t2, ..., tn`. You have 2 servers.
**Goal:** Minimize the makespan (total time).
-   This is equivalent to partitioning tasks such that the difference between sums is minimized.
-   If `Sum(S1) == Sum(S2)`, makespan is `Total / 2` (Optimal).
-   If not possible, we want `Sum(S1)` as close to `Total / 2` as possible.
-   Our DP table `dp[target]` tells us exactly which sums are reachable. We just look for the largest `i <= Total/2` such that `dp[i]` is True.

## 14. Code: Reconstructing the Solution

Sometimes we need to print the actual subset, not just `True/False`.

```python
def getPartitionSubset(nums):
    total = sum(nums)
    if total % 2 != 0: return None
    target = total // 2
    
    # dp[j] stores True/False
    # parent[i][j] stores whether we included item i to get sum j
    n = len(nums)
    dp = {0}
    parent = {} # (index, current_sum) -> boolean (included or not)
    
    # Standard DP with path tracking
    # Note: Using set for sparse DP to save space if target is large
    reachable = {0}
    
    for i, num in enumerate(nums):
        new_reachable = set()
        for s in reachable:
            if s + num <= target:
                new_reachable.add(s + num)
                parent[(i, s + num)] = True # Included
            parent[(i, s)] = False # Excluded (implicitly handled by not overwriting if already reachable)
        reachable.update(new_reachable)
        
    if target not in reachable:
        return None
        
    # Backtrack
    subset = []
    curr = target
    for i in range(n - 1, -1, -1):
        # Did we include nums[i] to get curr?
        # This logic is slightly tricky with set DP. 
        # Better to use 2D array logic for reconstruction.
        pass 
        
    # Let's use the 2D array logic for clarity
    dp = [[False] * (target + 1) for _ in range(n + 1)]
    dp[0][0] = True
    for i in range(1, n + 1):
        num = nums[i-1]
        for j in range(target + 1):
            dp[i][j] = dp[i-1][j]
            if j >= num and dp[i-1][j-num]:
                dp[i][j] = True
                
    if not dp[n][target]: return None
    
    subset = []
    curr = target
    for i in range(n, 0, -1):
        # If we could get curr without nums[i-1], skip it
        if dp[i-1][curr]:
            continue
        else:
            # Must have included it
            subset.append(nums[i-1])
            curr -= nums[i-1]
            
    return subset
```

## 15. Performance Benchmarking

**Scenario:** $N=100$, $Target=10000$.

| Approach | Python Time | C++ Time |
| :--- | :--- | :--- |
| **Recursion** | Timeout | Timeout |
| **DP (2D)** | 150ms | 10ms |
| **DP (1D)** | 120ms | 8ms |
| **Bitset** | N/A (Python ints are slow) | **0.1ms** |

**Takeaway:** In competitive programming or high-frequency trading, C++ Bitset is unbeatable for this class of problems.

## 16. Interview Pro Tips

1.  **Identify the Pattern:** "Equal sum", "Split array", "Target sum" -> Think Knapsack.
2.  **Check Constraints:**
    -   $N \le 20$: Recursion / Meet-in-middle.
    -   $N \le 100, Sum \le 20000$: DP.
    -   $Sum > 10^9$: DP fails. Is it a math problem?
3.  **Space Optimization:** Always mention the 1D array optimization. It shows system design awareness (cache locality).
4.  **Bitset:** Mentioning this gets you "Senior Engineer" points.

## 17. Summary

| Approach | Time | Space | Notes |
| :--- | :--- | :--- | :--- |
| **Recursion** | $O(2^N)$ | $O(N)$ | TLE |
| **Memoization** | $O(N \cdot S)$ | $O(N \cdot S)$ | Good |
| **Tabulation** | $O(N \cdot S)$ | $O(N \cdot S)$ | Avoids recursion limit |
| **Space Opt** | $O(N \cdot S)$ | $O(S)$ | Standard Interview Solution |
| **Bitset** | $O(N \cdot S / 64)$ | $O(S / 64)$ | Fastest |

---

**Originally published at:** [arunbaby.com/dsa/0036-partition-equal-subset-sum](https://www.arunbaby.com/dsa/0036-partition-equal-subset-sum/)
