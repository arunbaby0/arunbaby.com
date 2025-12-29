---
title: "Climbing Stairs"
day: 6
related_ml_day: 6
related_speech_day: 6
related_agents_day: 6
collection: dsa
categories:
 - dsa
tags:
 - dynamic-programming
 - recursion
 - fibonacci
topic: Dynamic Programming
difficulty: Easy
companies: [Google, Meta, Amazon, Microsoft, Adobe]
leetcode_link: "https://leetcode.com/problems/climbing-stairs/"
time_complexity: "O(n)"
space_complexity: "O(1)"
---

**The Fibonacci problem in disguise, teaching the fundamental transition from recursion to dynamic programming to space optimization.**

## Problem

You are climbing a staircase. It takes `n` steps to reach the top.

Each time you can either climb **1 or 2 steps**. In how many distinct ways can you climb to the top?

**Example 1:**
``
Input: n = 2
Output: 2
Explanation: Two ways:
1. 1 step + 1 step
2. 2 steps
``

**Example 2:**
``
Input: n = 3
Output: 3
Explanation: Three ways:
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
``

**Constraints:**
- `1 <= n <= 45`

---

## Intuition

**Key Insight:** To reach step `n`, you must have come from either step `n-1` (then climb 1 step) or step `n-2` (then climb 2 steps).

**Recurrence relation:**
``
ways(n) = ways(n-1) + ways(n-2)
``

This is the **Fibonacci sequence**!

**Why?**
- `ways(1) = 1` (one way: single step)
- `ways(2) = 2` (two ways: 1+1 or 2)
- `ways(3) = ways(2) + ways(1) = 2 + 1 = 3`
- `ways(4) = ways(3) + ways(2) = 3 + 2 = 5`
- ...

---

## Approach 1: Recursion (Not Optimal)

Direct recursive implementation.

### Implementation

``python
def climbStairs(n: int) -> int:
 """
 Recursive solution
 
 Time: O(2^n) - exponential!
 Space: O(n) - recursion stack
 """
 # Base cases
 if n <= 2:
 return n
 
 # Recursive case
 return climbStairs(n - 1) + climbStairs(n - 2)

# Example
print(climbStairs(5)) # 8
``

**Why this is bad:**
``
climbStairs(5)
├── climbStairs(4)
│ ├── climbStairs(3)
│ │ ├── climbStairs(2) ✓
│ │ └── climbStairs(1) ✓
│ └── climbStairs(2) ✓ (recomputed!)
└── climbStairs(3) (entire subtree recomputed!)
 ├── climbStairs(2) ✓
 └── climbStairs(1) ✓

Massive redundant computation!
``

**Time Complexity:** O(2^n) - each call spawns two more calls 
**Space Complexity:** O(n) - maximum recursion depth

---

## Approach 2: Recursion with Memoization

Cache results to avoid recomputation.

### Implementation

``python
def climbStairs(n: int) -> int:
 """
 Recursion with memoization (top-down DP)
 
 Time: O(n) - each subproblem solved once
 Space: O(n) - memoization cache + recursion stack
 """
 memo = {}
 
 def helper(n):
 # Base cases
 if n <= 2:
 return n
 
 # Check memo
 if n in memo:
 return memo[n]
 
 # Compute and cache
 memo[n] = helper(n - 1) + helper(n - 2)
 return memo[n]
 
 return helper(n)

# Example
print(climbStairs(10)) # 89
``

**Time Complexity:** O(n) - each value computed once 
**Space Complexity:** O(n) - memo dictionary + recursion stack

---

## Approach 3: Dynamic Programming (Bottom-Up)

Build solution iteratively from base cases.

### Implementation

``python
def climbStairs(n: int) -> int:
 """
 Bottom-up dynamic programming
 
 Time: O(n)
 Space: O(n)
 """
 if n <= 2:
 return n
 
 # DP table
 dp = [0] * (n + 1)
 
 # Base cases
 dp[1] = 1
 dp[2] = 2
 
 # Fill table
 for i in range(3, n + 1):
 dp[i] = dp[i - 1] + dp[i - 2]
 
 return dp[n]

# Example
print(climbStairs(5)) # 8
``

### Walkthrough

``
n = 5

Initial: dp = [0, 1, 2, 0, 0, 0]
 0 1 2 3 4 5

i = 3:
 dp[3] = dp[2] + dp[1] = 2 + 1 = 3
 dp = [0, 1, 2, 3, 0, 0]

i = 4:
 dp[4] = dp[3] + dp[2] = 3 + 2 = 5
 dp = [0, 1, 2, 3, 5, 0]

i = 5:
 dp[5] = dp[4] + dp[3] = 5 + 3 = 8
 dp = [0, 1, 2, 3, 5, 8]

Answer: dp[5] = 8
``

**Time Complexity:** O(n) 
**Space Complexity:** O(n)

---

## Approach 4: Space Optimized (Optimal)

Since we only need previous two values, use two variables.

### Implementation

``python
def climbStairs(n: int) -> int:
 """
 Space-optimized DP
 
 Time: O(n)
 Space: O(1) - only two variables!
 """
 if n <= 2:
 return n
 
 # Only need previous two values
 prev2 = 1 # ways(1)
 prev1 = 2 # ways(2)
 
 for i in range(3, n + 1):
 current = prev1 + prev2
 prev2 = prev1
 prev1 = current
 
 return prev1

# Example
print(climbStairs(10)) # 89
``

**Time Complexity:** O(n) 
**Space Complexity:** O(1) - optimal!

---

## Approach 5: Fibonacci Formula (Constant Time)

Use Binet's formula for Fibonacci numbers.

### Implementation

``python
import math

def climbStairs(n: int) -> int:
 """
 Mathematical formula (Binet's formula)
 
 Time: O(1) - constant time!
 Space: O(1)
 
 Note: May have floating point precision issues for large n
 """
 sqrt5 = math.sqrt(5)
 phi = (1 + sqrt5) / 2 # Golden ratio
 psi = (1 - sqrt5) / 2
 
 # Binet's formula (adjusted for stairs indexing)
 result = (phi ** (n + 1) - psi ** (n + 1)) / sqrt5
 
 return int(round(result))

# Example
print(climbStairs(10)) # 89
``

**Time Complexity:** O(1) - direct calculation 
**Space Complexity:** O(1)

**Caveat:** Floating point arithmetic may cause precision issues for very large n.

---

## Approach 6: Matrix Exponentiation (Logarithmic Time)

For very large n, we can use matrix exponentiation to achieve O(log n) time.

### Mathematical Foundation

The Fibonacci recurrence can be expressed as matrix multiplication:

``
[F(n+1)] [1 1] [F(n) ]
[F(n) ] = [1 0] × [F(n-1)]
``

Therefore:

``
[F(n+1)] [1 1]^n [F(1)]
[F(n) ] = [1 0] × [F(0)]
``

We can compute the matrix power in O(log n) time using **exponentiation by squaring**.

### Implementation

``python
import numpy as np

def climbStairsMatrix(n: int) -> int:
 """
 Matrix exponentiation approach
 
 Time: O(log n)
 Space: O(1)
 
 Best for very large n where even O(n) is too slow
 """
 if n <= 2:
 return n
 
 def matrix_multiply(A, B):
 """Multiply two 2x2 matrices"""
 return [
 [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
 [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
 ]
 
 def matrix_power(M, n):
 """Compute M^n using exponentiation by squaring"""
 if n == 1:
 return M
 
 if n % 2 == 0:
 half = matrix_power(M, n // 2)
 return matrix_multiply(half, half)
 else:
 return matrix_multiply(M, matrix_power(M, n - 1))
 
 # Base matrix
 base = [[1, 1], [1, 0]]
 
 # Compute base^n
 result = matrix_power(base, n)
 
 # Result is in result[0][0] (adjusted for our indexing)
 return result[0][0]

# Example
print(climbStairsMatrix(10)) # 89
print(climbStairsMatrix(50)) # 20365011074
``

**Time Complexity:** O(log n) - halving problem size at each step 
**Space Complexity:** O(log n) - recursion stack (can be optimized to O(1) iteratively)

### When to Use Matrix Exponentiation

**Advantages:**
- Fastest asymptotic time complexity
- Works for extremely large n (where n iterations would be too slow)

**Disadvantages:**
- More complex to implement
- Overkill for typical interview constraints (n ≤ 45)
- Risk of integer overflow for very large results

---

## Performance Comparison

Let's benchmark all approaches:

``python
import time

def climbStairsBottomUp(n):
 if n <= 2:
 return n
 dp = [0] * (n + 1)
 dp[1], dp[2] = 1, 2
 for i in range(3, n + 1):
 dp[i] = dp[i-1] + dp[i-2]
 return dp[n]

def climbStairsSpaceOptimized(n):
 if n <= 2:
 return n
 prev2, prev1 = 1, 2
 for _ in range(3, n + 1):
 prev2, prev1 = prev1, prev1 + prev2
 return prev1

def benchmark(func, n, iterations=1000):
 """Benchmark function execution time"""
 start = time.perf_counter()
 for _ in range(iterations):
 func(n)
 end = time.perf_counter()
 return (end - start) / iterations * 1000 # ms

# Test different approaches
n = 30

approaches = {
 'Space Optimized O(n)': climbStairsSpaceOptimized,
 'Bottom-up O(n)': climbStairsBottomUp,
 'Matrix O(log n)': climbStairsMatrix,
 'Binet O(1)': climbStairs # Using Binet's formula
}

print(f"Benchmarking for n={n} (1000 iterations each):\n")
for name, func in approaches.items():
 time_ms = benchmark(func, n)
 print(f"{name:30s}: {time_ms:.4f} ms")
``

**Key Insights:**

1. **For n ≤ 50:** Binet's formula or space-optimized DP is fastest
2. **For interviews:** Space-optimized DP is best (simple + optimal)
3. **For very large n:** Matrix exponentiation avoids iteration but has constant overhead
4. **Recursive with memo:** Never the best choice (overhead of recursion + dictionary lookups)

---

## Common Mistakes & Edge Cases

### Mistake 1: Off-by-One Errors

``python
# WRONG: Incorrect base case
def climbStairsWrong(n: int) -> int:
 if n == 0:
 return 0 # Wrong! Should be 1 (one way to do nothing)
 if n == 1:
 return 1
 # ...

# CORRECT
def climbStairsCorrect(n: int) -> int:
 if n <= 2:
 return n
 # ...
``

### Mistake 2: Not Handling Edge Cases

``python
# WRONG: Doesn't handle n=0
def climbStairsWrong(n: int) -> int:
 prev2, prev1 = 1, 2
 for i in range(3, n + 1): # Breaks if n < 3
 current = prev1 + prev2
 prev2, prev1 = prev1, current
 return prev1

# CORRECT
def climbStairsCorrect(n: int) -> int:
 if n <= 2:
 return n # Handle small n explicitly
 
 prev2, prev1 = 1, 2
 for i in range(3, n + 1):
 current = prev1 + prev2
 prev2, prev1 = prev1, current
 return prev1
``

### Mistake 3: Integer Overflow

For very large n (e.g., n=100), the result exceeds typical integer limits in some languages.

``python
# Python handles big integers automatically
print(climbStairs(100)) # 573147844013817084101

# In Java/C++, you'd need BigInteger or modular arithmetic
# Often interview problems ask for result % MOD
def climbStairsMod(n: int, MOD: int = 10**9 + 7) -> int:
 """Return result modulo MOD to prevent overflow"""
 if n <= 2:
 return n
 
 prev2, prev1 = 1, 2
 for i in range(3, n + 1):
 current = (prev1 + prev2) % MOD
 prev2, prev1 = prev1, current
 
 return prev1

print(climbStairsMod(100)) # 687995182
``

### Mistake 4: Modifying Input in Memoization

``python
# WRONG: Global memo persists across test cases
memo = {}

def climbStairsWrong(n: int) -> int:
 if n <= 2:
 return n
 if n in memo:
 return memo[n]
 memo[n] = climbStairsWrong(n-1) + climbStairsWrong(n-2)
 return memo[n]

# First call: correct
print(climbStairsWrong(5)) # 8

# Second call: uses stale memo
print(climbStairsWrong(3)) # 3, but used cached values from n=5 call

# CORRECT: Memo as local variable or function argument
def climbStairsCorrect(n: int) -> int:
 memo = {} # Fresh memo for each call
 
 def helper(n):
 if n <= 2:
 return n
 if n in memo:
 return memo[n]
 memo[n] = helper(n-1) + helper(n-2)
 return memo[n]
 
 return helper(n)
``

### Edge Case: n = 0

``python
# Problem statement says 1 <= n <= 45, but defensive coding:
def climbStairsSafe(n: int) -> int:
 if n < 0:
 raise ValueError("n must be non-negative")
 if n == 0:
 return 1 # One way to not climb (stay at ground)
 if n <= 2:
 return n
 
 prev2, prev1 = 1, 2
 for i in range(3, n + 1):
 current = prev1 + prev2
 prev2, prev1 = prev1, current
 
 return prev1
``

---

## Production Engineering Considerations

### 1. Caching for Repeated Queries

In a production system handling many queries:

``python
from functools import lru_cache

class StairClimber:
 """
 Production-ready stair climbing calculator
 
 Use case: API endpoint that computes climbing ways for various n
 """
 
 @lru_cache(maxsize=128)
 def compute(self, n: int) -> int:
 """
 Compute with caching for repeated queries
 
 LRU cache stores recent results
 """
 if n <= 2:
 return n
 
 prev2, prev1 = 1, 2
 for i in range(3, n + 1):
 current = prev1 + prev2
 prev2, prev1 = prev1, current
 
 return prev1
 
 def get_cache_info(self):
 """Get cache statistics"""
 return self.compute.cache_info()

# Usage
climber = StairClimber()

# First calls compute
print(climber.compute(10)) # Computes
print(climber.compute(10)) # Cache hit
print(climber.compute(15)) # Computes

print(climber.get_cache_info())
# CacheInfo(hits=1, misses=2, maxsize=128, currsize=2)
``

### 2. Precomputation for Low Latency

If you need ultra-low latency and n has known upper bound:

``python
class PrecomputedStairs:
 """
 Precompute all results up to MAX_N
 
 Use case: Latency-critical systems (e.g., real-time game logic)
 """
 
 MAX_N = 100
 
 def __init__(self):
 """Precompute all values at initialization"""
 self._precompute()
 
 def _precompute(self):
 """Compute all values from 1 to MAX_N"""
 self.cache = [0] * (self.MAX_N + 1)
 self.cache[1] = 1
 if self.MAX_N >= 2:
 self.cache[2] = 2
 
 for i in range(3, self.MAX_N + 1):
 self.cache[i] = self.cache[i-1] + self.cache[i-2]
 
 def compute(self, n: int) -> int:
 """O(1) lookup"""
 if n > self.MAX_N:
 raise ValueError(f"n must be <= {self.MAX_N}")
 return self.cache[n]

# Usage
stairs = PrecomputedStairs() # Precompute on init

# All queries are O(1)
print(stairs.compute(50)) # Instant lookup
print(stairs.compute(100)) # Instant lookup
``

### 3. Handling Large-Scale Distributed Systems

``python
class DistributedStairComputer:
 """
 Handle climbing stairs in distributed system
 
 Use case: Distributed computing cluster
 """
 
 def compute_range(self, start: int, end: int) -> dict[int, int]:
 """
 Compute multiple values efficiently
 
 Instead of computing each independently, compute iteratively
 and return all values in range
 """
 if start < 1 or end < start:
 raise ValueError("Invalid range")
 
 results = {}
 
 # Bootstrap
 if start == 1:
 results[1] = 1
 prev2, prev1 = 1, 2
 current_n = 2
 elif start == 2:
 results[2] = 2
 prev2, prev1 = 1, 2
 current_n = 2
 else:
 # Compute up to start
 prev2, prev1 = 1, 2
 for i in range(3, start):
 current = prev1 + prev2
 prev2, prev1 = prev1, current
 current_n = start - 1
 
 # Compute range
 for n in range(max(start, current_n), end + 1):
 if n == 1:
 results[1] = 1
 elif n == 2:
 results[2] = 2
 else:
 current = prev1 + prev2
 results[n] = current
 prev2, prev1 = prev1, current
 
 return results

# Usage
computer = DistributedStairComputer()

# Compute batch of values (e.g., for multiple users)
batch_results = computer.compute_range(10, 20)
print(batch_results)
# {10: 89, 11: 144, 12: 233, ..., 20: 10946}
``

---

## Deep Dive: Why Dynamic Programming?

### The Optimal Substructure Property

**Definition:** A problem has *optimal substructure* if the optimal solution can be constructed from optimal solutions of its subproblems.

**For climbing stairs:**
``
Optimal way to reach step n = 
 Optimal way to reach step (n-1) + take 1 step
 OR
 Optimal way to reach step (n-2) + take 2 steps
``

This property is **necessary** for DP to work.

### The Overlapping Subproblems Property

**Definition:** The problem can be broken down into subproblems which are reused multiple times.

**For climbing stairs:**
``
climbStairs(5) needs:
 - climbStairs(4) and climbStairs(3)

climbStairs(4) needs:
 - climbStairs(3) and climbStairs(2)

Notice: climbStairs(3) is computed TWICE!
``

### Why Not Greedy?

**Greedy approach:** Always take the largest possible step (2 steps).

``python
def climbStairsGreedy(n: int) -> int:
 """
 WRONG: Greedy doesn't work here
 
 This would always take 2-steps when possible
 """
 ways = 0
 while n > 0:
 if n >= 2:
 n -= 2 # Take 2 steps
 else:
 n -= 1 # Take 1 step
 ways += 1
 return ways

print(climbStairsGreedy(5)) # Wrong answer!
``

**Why greedy fails:** We're counting *number of ways*, not finding *optimal path*. Greedy works for optimization problems with greedy choice property, not counting problems.

---

## Variations

### Variation 1: Can Climb 1, 2, or 3 Steps

``python
def climbStairsThreeSteps(n: int) -> int:
 """
 Can climb 1, 2, or 3 steps at a time
 
 Recurrence: ways(n) = ways(n-1) + ways(n-2) + ways(n-3)
 
 Time: O(n)
 Space: O(1)
 """
 if n <= 2:
 return n
 if n == 3:
 return 4 # 1+1+1, 1+2, 2+1, 3
 
 # Track previous three values
 prev3 = 1 # ways(1)
 prev2 = 2 # ways(2)
 prev1 = 4 # ways(3)
 
 for i in range(4, n + 1):
 current = prev1 + prev2 + prev3
 prev3 = prev2
 prev2 = prev1
 prev1 = current
 
 return prev1

# Example
print(climbStairsThreeSteps(4)) # 7
# 1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 2+2, 1+3, 3+1
``

### Variation 2: Variable Step Sizes

``python
def climbStairsVariableSteps(n: int, steps: list[int]) -> int:
 """
 Can climb any step size in 'steps' list
 
 Example: steps = [1, 2, 5]
 
 Time: O(n * k) where k = len(steps)
 Space: O(n)
 """
 if n == 0:
 return 1
 if n < 0:
 return 0
 
 # DP table
 dp = [0] * (n + 1)
 dp[0] = 1 # One way to stay at ground (do nothing)
 
 # For each position
 for i in range(1, n + 1):
 # Try each step size
 for step in steps:
 if i - step >= 0:
 dp[i] += dp[i - step]
 
 return dp[n]

# Example
print(climbStairsVariableSteps(5, [1, 2, 5]))
# Can reach 5 using: 1+1+1+1+1, 1+1+1+2, 1+1+2+1, 1+2+1+1, 2+1+1+1, 1+2+2, 2+1+2, 2+2+1, 5
``

### Variation 3: Minimum Cost Climbing Stairs

``python
def minCostClimbingStairs(cost: list[int]) -> int:
 """
 LeetCode 746: Min Cost Climbing Stairs
 
 Each step has a cost. Find minimum cost to reach top.
 Can start from step 0 or step 1.
 
 Time: O(n)
 Space: O(1)
 """
 n = len(cost)
 
 if n <= 1:
 return 0
 
 # Track min cost to reach previous two steps
 prev2 = cost[0]
 prev1 = cost[1]
 
 for i in range(2, n):
 current = cost[i] + min(prev1, prev2)
 prev2 = prev1
 prev1 = current
 
 # Can finish from either last or second-last step
 return min(prev1, prev2)

# Example
cost = [10, 15, 20]
print(minCostClimbingStairs(cost)) # 15
# Start at index 1, pay 15, step to top
``

### Variation 4: Count Paths with Constraints

``python
def climbStairsWithConstraint(n: int, max_consecutive_ones: int = 2) -> int:
 """
 Count ways to climb stairs with constraint on consecutive 1-steps
 
 Example: max_consecutive_ones = 2 means can't take more than 2
 consecutive single steps
 
 Time: O(n)
 Space: O(n)
 """
 # dp[i][j] = ways to reach step i with j consecutive 1-steps at end
 dp = [[0] * (max_consecutive_ones + 1) for _ in range(n + 1)]
 dp[0][0] = 1
 
 for i in range(n):
 for j in range(max_consecutive_ones + 1):
 if dp[i][j] == 0:
 continue
 
 # Take 2 steps (resets consecutive count)
 if i + 2 <= n:
 dp[i + 2][0] += dp[i][j]
 
 # Take 1 step (increment consecutive count)
 if i + 1 <= n and j + 1 <= max_consecutive_ones:
 dp[i + 1][j + 1] += dp[i][j]
 
 return sum(dp[n])

# Example
print(climbStairsWithConstraint(5, max_consecutive_ones=2))
``

---

## Connection to ML Systems

### Model Training Iteration Strategy

``python
class TrainingScheduler:
 """
 Determine number of training strategies given constraints
 
 Similar to stairs: at each epoch, choose next action
 """
 
 def count_training_paths(
 self,
 total_epochs: int,
 actions: list[str] = ['continue', 'adjust_lr', 'early_stop']
 ) -> int:
 """
 Count possible training paths
 
 At each epoch, can take different actions (like step sizes)
 """
 # Similar to variable step climbing stairs
 # Each action advances training by different amounts
 
 action_advances = {
 'continue': 1, # Continue one epoch
 'adjust_lr': 1, # Adjust and continue
 'early_stop': total_epochs # Jump to end
 }
 
 # Count paths using DP (similar to climbing stairs)
 dp = [0] * (total_epochs + 1)
 dp[0] = 1
 
 for epoch in range(total_epochs):
 for action in actions:
 advance = action_advances.get(action, 1)
 next_epoch = min(epoch + advance, total_epochs)
 dp[next_epoch] += dp[epoch]
 
 return dp[total_epochs]

# Usage
scheduler = TrainingScheduler()
paths = scheduler.count_training_paths(total_epochs=5)
print(f"Possible training strategies: {paths}")
``

### Feature Selection Combinations

``python
class FeatureSelectionCounter:
 """
 Count ways to select features with constraints
 
 Similar pattern to climbing stairs
 """
 
 def count_feature_subsets(
 self,
 num_features: int,
 max_features_per_selection: int = 2
 ) -> int:
 """
 Count ways to select features where each step selects 1-k features
 
 Similar to climbing stairs with variable step sizes
 """
 # dp[i] = ways to select i features
 dp = [0] * (num_features + 1)
 dp[0] = 1 # Empty selection
 
 for i in range(1, num_features + 1):
 # Can select 1, 2, ..., max_features_per_selection at once
 for k in range(1, min(i, max_features_per_selection) + 1):
 dp[i] += dp[i - k]
 
 return dp[num_features]

# Usage
counter = FeatureSelectionCounter()
ways = counter.count_feature_subsets(num_features=10, max_features_per_selection=3)
print(f"Ways to build feature set: {ways}")
``

### Pipeline Stage Combinations

``python
class MLPipelineCounter:
 """
 Count valid ML pipeline configurations
 
 Each stage can have different options (like step sizes)
 """
 
 def count_pipeline_configs(
 self,
 stages: list[dict]
 ) -> int:
 """
 Count possible pipeline configurations
 
 Args:
 stages: List of stage definitions
 e.g., [{'name': 'preprocessing', 'options': 3},
 {'name': 'feature_eng', 'options': 2}]
 
 Returns:
 Total number of valid pipelines
 """
 if not stages:
 return 1
 
 # Multiplicative principle (not exactly stairs, but similar counting)
 total = 1
 for stage in stages:
 total *= stage.get('options', 1)
 
 return total
 
 def count_sequential_pipelines(
 self,
 total_stages: int,
 stage_options: list[int]
 ) -> int:
 """
 Count ways to build pipeline where each step uses 1-k stages
 
 More directly analogous to climbing stairs
 """
 # dp[i] = ways to build pipeline with i stages
 dp = [0] * (total_stages + 1)
 dp[0] = 1
 
 for i in range(1, total_stages + 1):
 for num_stages in stage_options:
 if i - num_stages >= 0:
 dp[i] += dp[i - num_stages]
 
 return dp[total_stages]

# Usage
pipeline_counter = MLPipelineCounter()

# Count sequential pipeline configurations
# Can add 1, 2, or 3 stages at a time
configs = pipeline_counter.count_sequential_pipelines(
 total_stages=5,
 stage_options=[1, 2, 3]
)
print(f"Sequential pipeline configurations: {configs}")
``

---

## Testing

### Comprehensive Test Suite

``python
import unittest

class TestClimbStairs(unittest.TestCase):
 
 def test_base_cases(self):
 """Test base cases"""
 self.assertEqual(climbStairs(1), 1)
 self.assertEqual(climbStairs(2), 2)
 
 def test_small_values(self):
 """Test small n"""
 self.assertEqual(climbStairs(3), 3)
 self.assertEqual(climbStairs(4), 5)
 self.assertEqual(climbStairs(5), 8)
 
 def test_fibonacci_sequence(self):
 """Verify it follows Fibonacci"""
 # F(1)=1, F(2)=2, F(3)=3, F(4)=5, F(5)=8, ...
 expected = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
 for i, exp in enumerate(expected, 1):
 self.assertEqual(climbStairs(i), exp)
 
 def test_large_value(self):
 """Test larger n"""
 # n=10 should give 89 (11th Fibonacci number)
 self.assertEqual(climbStairs(10), 89)
 
 # n=20 should give 10946
 self.assertEqual(climbStairs(20), 10946)
 
 def test_all_approaches_agree(self):
 """All approaches should give same answer"""
 for n in range(1, 15):
 memo = climbStairsBottomUp(n)
 space_opt = climbStairsSpaceOptimized(n)
 self.assertEqual(memo, space_opt, f"Mismatch at n={n}")

def climbStairsBottomUp(n):
 """Helper for testing"""
 if n <= 2:
 return n
 dp = [0] * (n + 1)
 dp[1], dp[2] = 1, 2
 for i in range(3, n + 1):
 dp[i] = dp[i-1] + dp[i-2]
 return dp[n]

def climbStairsSpaceOptimized(n):
 """Helper for testing"""
 if n <= 2:
 return n
 prev2, prev1 = 1, 2
 for i in range(3, n + 1):
 current = prev1 + prev2
 prev2, prev1 = prev1, current
 return prev1

if __name__ == '__main__':
 unittest.main()
``

---

## Interview Tips

### Recognizing the Pattern

**When you see:**
- "Count number of ways"
- "Reach position n"
- "Each step has limited options"
- "Previous decisions affect current options"

**Think:** Dynamic Programming (likely Fibonacci-like)

### Interview Strategy: How to Approach This Problem

**Step 1: Clarify the Problem (1-2 minutes)**

Ask clarifying questions:
- "Can I confirm: we can take either 1 or 2 steps at a time?"
- "Are we counting distinct ways, not the minimum number of steps?"
- "Is n guaranteed to be positive?"
- "What's the maximum value of n I should handle?"

**Step 2: Walkthrough Examples (2-3 minutes)**

``
n = 1: [1] → 1 way
n = 2: [1,1], [2] → 2 ways
n = 3: [1,1,1], [1,2], [2,1] → 3 ways
n = 4: [1,1,1,1], [1,1,2], [1,2,1], [2,1,1], [2,2] → 5 ways

Pattern: Each n is sum of previous two → Fibonacci!
``

**Step 3: Propose Brute Force (1 minute)**

"The naive approach is recursion: to reach step n, we can come from n-1 or n-2. But this has exponential time complexity due to repeated subproblems."

**Step 4: Optimize with DP (3-5 minutes)**

"We can optimize using dynamic programming. Since we're recomputing the same subproblems, we can either:
1. Use memoization (top-down)
2. Use tabulation (bottom-up)

I'll go with bottom-up since it's simpler and avoids recursion overhead."

**Step 5: Further Optimize Space (1-2 minutes)**

"Since we only need the previous two values, we can optimize from O(n) space to O(1) using two variables."

**Step 6: Code + Test (5-7 minutes)**

Write the space-optimized solution and test with examples.

**Step 7: Discuss Edge Cases & Complexity (1-2 minutes)**

- Edge cases: n=1, n=2
- Time: O(n)
- Space: O(1)

**Total: ~15-20 minutes**

---

### Common Follow-ups

**Q1: What if we want to print all possible paths?**

``python
def climbStairsAllPaths(n: int) -> list[list[int]]:
 """
 Return all distinct paths to reach top
 
 Time: O(2^n) - exponential number of paths
 Space: O(2^n) - storing all paths
 """
 def backtrack(remaining, path, all_paths):
 if remaining == 0:
 all_paths.append(path[:])
 return
 
 if remaining < 0:
 return
 
 # Try 1 step
 path.append(1)
 backtrack(remaining - 1, path, all_paths)
 path.pop()
 
 # Try 2 steps
 path.append(2)
 backtrack(remaining - 2, path, all_paths)
 path.pop()
 
 all_paths = []
 backtrack(n, [], all_paths)
 return all_paths

# Example
paths = climbStairsAllPaths(4)
print(f"All paths to climb 4 stairs:")
for path in paths:
 print(path)
# Output:
# [1, 1, 1, 1]
# [1, 1, 2]
# [1, 2, 1]
# [2, 1, 1]
# [2, 2]
``

**Q2: What if steps have weights and we want minimum weight path?**

See "Minimum Cost Climbing Stairs" variation above.

**Q3: What's the space complexity of the recursive solution with memoization?**

O(n) for both memoization cache and recursion stack.

---

## Key Takeaways

✅ **Fibonacci pattern** - Recognize when problem reduces to Fibonacci 
✅ **DP progression** - Recursion → Memoization → Bottom-up → Space-optimized 
✅ **Space optimization** - Only need last k values for k-way recurrence 
✅ **Counting problems** - DP naturally solves "count number of ways" 
✅ **Recurrence relations** - Key to DP is finding the recurrence 
✅ **ML applications** - Similar counting patterns in training strategies, feature selection 
✅ **Variations** - Variable step sizes, constraints, costs all use same DP template 

---

## Related Problems

Practice these to master the pattern:
- **[Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/)** - Add cost dimension
- **[House Robber](https://leetcode.com/problems/house-robber/)** - Similar DP pattern with constraints
- **[Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)** - Direct Fibonacci
- **[N-th Tribonacci Number](https://leetcode.com/problems/n-th-tribonacci-number/)** - Three-way recurrence
- **[Decode Ways](https://leetcode.com/problems/decode-ways/)** - Similar counting pattern

---

**Originally published at:** [arunbaby.com/dsa/0006-climbing-stairs](https://www.arunbaby.com/dsa/0006-climbing-stairs/)

*If you found this helpful, consider sharing it with others who might benefit.*

