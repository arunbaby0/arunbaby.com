---
title: "Unique Paths"
day: 21
related_ml_day: 21
related_speech_day: 21
related_agents_day: 21
collection: dsa
categories:
 - dsa
tags:
 - dynamic-programming
 - combinatorics
 - grid
 - path-counting
 - memoization
 - medium
subdomain: "Dynamic Programming"
tech_stack: [Python]
scale: "O(M×N) time, O(M×N) space (optimizable to O(N))"
companies: [Google, Meta, Amazon, Microsoft, Apple, Bloomberg]
---

**Master grid path counting with dynamic programming—the same optimization technique used in neural architecture search and speech model design.**

## Problem Statement

There is a robot on an `m x n` grid. The robot is initially located at the **top-left corner** (i.e., `grid[0][0]`). The robot tries to move to the **bottom-right corner** (i.e., `grid[m-1][n-1]`). The robot can only move either **down** or **right** at any point in time.

Given the two integers `m` and `n`, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

### Examples

**Example 1:**

``
Input: m = 3, n = 7
Output: 28

Visualization (3×7 grid):
Start → → → → → → 
 ↓ → → → → → ↓
 ↓ → → → → → → Goal
``

**Example 2:**

``
Input: m = 3, n = 2
Output: 3

Visualization:
Start → Goal
 ↓ ↓
 ↓ → → Goal

Paths:
1. Right → Down → Down
2. Down → Right → Down 
3. Down → Down → Right
``

**Example 3:**

``
Input: m = 1, n = 1
Output: 1
Explanation: Already at goal.
``

### Constraints

- `1 <= m, n <= 100`

## Understanding the Problem

This is a **classic path-counting problem** that teaches:

1. **Dynamic programming** - breaking down into smaller subproblems
2. **Combinatorial mathematics** - counting paths systematically
3. **State space optimization** - reducing memory usage
4. **Grid navigation** - foundation for many pathfinding algorithms

### Key Insight

To reach position `(i, j)`, we must have come from either:
- Position `(i-1, j)` (from above), OR
- Position `(i, j-1)` (from left)

Therefore:
``
paths(i, j) = paths(i-1, j) + paths(i, j-1)
``

This is the **recurrence relation** that enables dynamic programming.

### Why This Problem Matters

1. **DP fundamentals:** Learn bottom-up and top-down DP
2. **Combinatorics:** Connection to binomial coefficients
3. **Path optimization:** Foundation for finding shortest/best paths
4. **Real-world applications:**
 - Neural architecture search (paths through search space)
 - Route planning (number of ways to reach destination)
 - Resource allocation (paths through decision tree)
 - Pipeline optimization (paths through processing stages)

### The Path Optimization Connection

| Unique Paths | Neural Architecture Search | Speech Architecture Search |
|--------------|---------------------------|---------------------------|
| Count paths in grid | Count architectures in search space | Count model configs |
| DP to optimize | DP/RL to find best architecture | DP to find best speech model |
| m×n grid | Layer×operation search space | Encoder×decoder configs |
| Exponential paths | Exponential architectures | Exponential combinations |
| DP reduces to polynomial | Search reduces complexity | Search finds optimal |

All three use **dynamic programming and path optimization** to navigate exponentially large search spaces.

## Approach 1: Brute Force Recursion

### Intuition

Recursively explore all paths from `(0,0)` to `(m-1, n-1)`.

### Implementation

``python
def uniquePaths_bruteforce(m: int, n: int) -> int:
 """
 Brute force recursion: explore all paths.
 
 Time: O(2^(m+n)) - exponential branching
 Space: O(m+n) - recursion depth
 
 Why this approach?
 - Shows the recursive structure
 - Demonstrates exponential complexity
 - Motivates need for DP
 
 Problem:
 - Extremely slow for moderate inputs
 - Redundant subproblem solutions
 """
 def count_paths(row: int, col: int) -> int:
 """Count paths from (row, col) to (m-1, n-1)."""
 # Base cases
 if row == m - 1 and col == n - 1:
 return 1 # Reached goal
 
 if row >= m or col >= n:
 return 0 # Out of bounds
 
 # Recursive case: go down or right
 paths_down = count_paths(row + 1, col)
 paths_right = count_paths(row, col + 1)
 
 return paths_down + paths_right
 
 return count_paths(0, 0)


# Test
print(uniquePaths_bruteforce(3, 7)) # 28
print(uniquePaths_bruteforce(3, 2)) # 3
``

### Analysis

**Time Complexity: O(2^(m+n))**
- Each cell has 2 choices (down, right)
- Path length is m + n - 2 moves
- Exponential branching

**Space Complexity: O(m+n)**
- Recursion depth

**For m=n=20:** Over 1 billion recursive calls! Too slow.

## Approach 2: Dynamic Programming (Top-Down with Memoization)

### Intuition

Cache results of subproblems to avoid recomputation.

### Implementation

``python
def uniquePaths_memo(m: int, n: int) -> int:
 """
 DP with memoization (top-down).
 
 Time: O(m×n) - each subproblem solved once
 Space: O(m×n) - memoization cache + recursion stack
 
 Optimization:
 - Cache prevents redundant calculations
 - Transforms exponential to polynomial
 """
 memo = {}
 
 def count_paths(row: int, col: int) -> int:
 # Check cache
 if (row, col) in memo:
 return memo[(row, col)]
 
 # Base cases
 if row == m - 1 and col == n - 1:
 return 1
 
 if row >= m or col >= n:
 return 0
 
 # Recursive case with memoization
 paths = count_paths(row + 1, col) + count_paths(row, col + 1)
 memo[(row, col)] = paths
 
 return paths
 
 return count_paths(0, 0)
``

## Approach 3: Dynamic Programming (Bottom-Up) - Optimal

### Intuition

Build the solution from the base case upward. For each cell, the number of paths is the sum of paths from the cell above and the cell to the left.

### Implementation

``python
def uniquePaths(m: int, n: int) -> int:
 """
 DP bottom-up (optimal for clarity).
 
 Time: O(m×n)
 Space: O(m×n)
 
 Algorithm:
 1. Create dp table where dp[i][j] = paths to reach (i,j)
 2. Initialize first row and column to 1 (only one way)
 3. Fill table using recurrence: dp[i][j] = dp[i-1][j] + dp[i][j-1]
 4. Return dp[m-1][n-1]
 """
 # Create DP table
 dp = [[0] * n for _ in range(m)]
 
 # Base cases: first row and first column
 for i in range(m):
 dp[i][0] = 1 # Only one way: all down
 
 for j in range(n):
 dp[0][j] = 1 # Only one way: all right
 
 # Fill table using recurrence relation
 for i in range(1, m):
 for j in range(1, n):
 dp[i][j] = dp[i-1][j] + dp[i][j-1]
 
 return dp[m-1][n-1]
``

### Step-by-Step Visualization

**Input:** m=3, n=3

``
Build DP table:

Initialize first row and column to 1:
1 1 1
1 ? ?
1 ? ?

Fill cell (1,1):
 dp[1][1] = dp[0][1] + dp[1][0] = 1 + 1 = 2

1 1 1
1 2 ?
1 ? ?

Fill cell (1,2):
 dp[1][2] = dp[0][2] + dp[1][1] = 1 + 2 = 3

1 1 1
1 2 3
1 ? ?

Fill cell (2,1):
 dp[2][1] = dp[1][1] + dp[2][0] = 2 + 1 = 3

1 1 1
1 2 3
1 3 ?

Fill cell (2,2):
 dp[2][2] = dp[1][2] + dp[2][1] = 3 + 3 = 6

1 1 1
1 2 3
1 3 6

Answer: dp[2][2] = 6
``

## Approach 4: Space-Optimized DP

### Intuition

We only need the previous row to compute the current row. Reduce space from O(m×n) to O(n).

### Implementation

``python
def uniquePaths_optimized(m: int, n: int) -> int:
 """
 Space-optimized DP.
 
 Time: O(m×n)
 Space: O(n) - only one row at a time
 
 Optimization:
 - Only store current row
 - Update in-place using previous values
 """
 # Single row (represents current row being computed)
 dp = [1] * n # First row is all 1s
 
 # Process each row
 for i in range(1, m):
 for j in range(1, n):
 # dp[j] currently holds value from row i-1 (above)
 # dp[j-1] holds value from current row (left)
 dp[j] = dp[j] + dp[j-1]
 
 return dp[n-1]
``

### Explanation

``
For m=3, n=3:

Initial (row 0): dp = [1, 1, 1]

Process row 1:
 j=1: dp[1] = dp[1] + dp[0] = 1 + 1 = 2
 j=2: dp[2] = dp[2] + dp[1] = 1 + 2 = 3
 Result: dp = [1, 2, 3]

Process row 2:
 j=1: dp[1] = dp[1] + dp[0] = 2 + 1 = 3
 j=2: dp[2] = dp[2] + dp[1] = 3 + 3 = 6
 Result: dp = [1, 3, 6]

Answer: dp[2] = 6
``

## Approach 5: Mathematical (Combinatorics)

### Intuition

To go from `(0,0)` to `(m-1, n-1)`:
- We need exactly `m-1` down moves
- We need exactly `n-1` right moves
- Total moves: `(m-1) + (n-1) = m+n-2`

The number of unique paths is the number of ways to choose which `m-1` positions (out of `m+n-2` total moves) are "down" moves.

This is a **binomial coefficient**:

\[
\text{paths} = \binom{m+n-2}{m-1} = \frac{(m+n-2)!}{(m-1)! \times (n-1)!}
\]

### Implementation

``python
def uniquePaths_math(m: int, n: int) -> int:
 """
 Combinatorial solution.
 
 Time: O(m+n) - computing binomial coefficient
 Space: O(1)
 
 Formula: C(m+n-2, m-1) where C is binomial coefficient
 """
 from math import comb
 
 return comb(m + n - 2, m - 1)


# Alternative: compute without library
def uniquePaths_math_manual(m: int, n: int) -> int:
 """Compute binomial coefficient manually to avoid overflow."""
 # We need to compute C(m+n-2, min(m-1, n-1))
 # Use smaller k to reduce computation
 
 total_moves = m + n - 2
 down_moves = m - 1
 right_moves = n - 1
 
 # Use smaller of the two
 k = min(down_moves, right_moves)
 
 # Compute C(total_moves, k) = total_moves! / (k! * (total_moves-k)!)
 # Optimize: multiply and divide incrementally to avoid overflow
 
 result = 1
 for i in range(k):
 result = result * (total_moves - i) // (i + 1)
 
 return result
``

## Implementation: Production-Grade Solution

``python
from typing import List, Optional
import logging
from functools import lru_cache

class UniquePathsSolver:
 """
 Production-ready unique paths solver.
 
 Features:
 - Multiple algorithms
 - Input validation
 - Performance metrics
 - Path reconstruction
 """
 
 def __init__(self, algorithm: str = "dp_optimized"):
 """
 Initialize solver.
 
 Args:
 algorithm: "bruteforce", "memo", "dp", "dp_optimized", "math"
 """
 self.algorithm = algorithm
 self.logger = logging.getLogger(__name__)
 self.subproblems_solved = 0
 
 def count_paths(self, m: int, n: int) -> int:
 """
 Count unique paths from (0,0) to (m-1, n-1).
 
 Args:
 m: Number of rows
 n: Number of columns
 
 Returns:
 Number of unique paths
 
 Raises:
 ValueError: If inputs are invalid
 """
 # Validate
 if not isinstance(m, int) or not isinstance(n, int):
 raise ValueError("m and n must be integers")
 
 if m < 1 or n < 1:
 raise ValueError("m and n must be >= 1")
 
 if m > 100 or n > 100:
 raise ValueError("m and n must be <= 100")
 
 # Reset metrics
 self.subproblems_solved = 0
 
 # Choose algorithm
 if self.algorithm == "bruteforce":
 result = self._bruteforce(m, n)
 elif self.algorithm == "memo":
 result = self._memoization(m, n)
 elif self.algorithm == "dp":
 result = self._dp_bottomup(m, n)
 elif self.algorithm == "dp_optimized":
 result = self._dp_optimized(m, n)
 elif self.algorithm == "math":
 result = self._mathematical(m, n)
 else:
 raise ValueError(f"Unknown algorithm: {self.algorithm}")
 
 self.logger.info(
 f"Grid {m}×{n}: {result} paths, "
 f"Algorithm: {self.algorithm}, "
 f"Subproblems: {self.subproblems_solved}"
 )
 
 return result
 
 def _bruteforce(self, m: int, n: int) -> int:
 """Brute force recursion."""
 def count(row: int, col: int) -> int:
 self.subproblems_solved += 1
 
 if row == m - 1 and col == n - 1:
 return 1
 if row >= m or col >= n:
 return 0
 
 return count(row + 1, col) + count(row, col + 1)
 
 return count(0, 0)
 
 def _memoization(self, m: int, n: int) -> int:
 """Top-down DP with memoization."""
 memo = {}
 
 def count(row: int, col: int) -> int:
 if (row, col) in memo:
 return memo[(row, col)]
 
 self.subproblems_solved += 1
 
 if row == m - 1 and col == n - 1:
 return 1
 if row >= m or col >= n:
 return 0
 
 paths = count(row + 1, col) + count(row, col + 1)
 memo[(row, col)] = paths
 return paths
 
 return count(0, 0)
 
 def _dp_bottomup(self, m: int, n: int) -> int:
 """Bottom-up DP."""
 dp = [[0] * n for _ in range(m)]
 
 # Initialize
 for i in range(m):
 dp[i][0] = 1
 for j in range(n):
 dp[0][j] = 1
 
 # Fill table
 for i in range(1, m):
 for j in range(1, n):
 dp[i][j] = dp[i-1][j] + dp[i][j-1]
 self.subproblems_solved += 1
 
 return dp[m-1][n-1]
 
 def _dp_optimized(self, m: int, n: int) -> int:
 """Space-optimized DP."""
 dp = [1] * n
 
 for i in range(1, m):
 for j in range(1, n):
 dp[j] = dp[j] + dp[j-1]
 self.subproblems_solved += 1
 
 return dp[n-1]
 
 def _mathematical(self, m: int, n: int) -> int:
 """Combinatorial solution."""
 from math import comb
 return comb(m + n - 2, m - 1)
 
 def reconstruct_path(self, m: int, n: int, path_index: int = 0) -> List[tuple]:
 """
 Reconstruct the k-th path (lexicographically).
 
 Args:
 m, n: Grid dimensions
 path_index: Which path to return (0-indexed)
 
 Returns:
 List of (row, col) positions
 """
 if path_index >= self.count_paths(m, n):
 raise ValueError("Path index out of range")
 
 path = [(0, 0)]
 row, col = 0, 0
 
 # Build DP table for path reconstruction
 dp = [[0] * n for _ in range(m)]
 for i in range(m):
 dp[i][0] = 1
 for j in range(n):
 dp[0][j] = 1
 
 for i in range(1, m):
 for j in range(1, n):
 dp[i][j] = dp[i-1][j] + dp[i][j-1]
 
 # Reconstruct path
 remaining = path_index
 
 while row < m - 1 or col < n - 1:
 if row == m - 1:
 # Must go right
 col += 1
 elif col == n - 1:
 # Must go down
 row += 1
 else:
 # Choose based on path index
 paths_if_go_down = dp[row + 1][col] if row + 1 < m else 0
 
 if remaining < paths_if_go_down:
 # This path goes down
 row += 1
 else:
 # This path goes right
 remaining -= paths_if_go_down
 col += 1
 
 path.append((row, col))
 
 return path


# Example usage
if __name__ == "__main__":
 logging.basicConfig(level=logging.INFO)
 
 solver = UniquePathsSolver(algorithm="dp_optimized")
 
 test_cases = [(3, 7), (3, 2), (1, 1), (7, 3)]
 
 for m, n in test_cases:
 result = solver.count_paths(m, n)
 print(f"\nGrid {m}×{n}: {result} paths")
 print(f"Stats: {solver.subproblems_solved} subproblems solved")
 
 # Reconstruct first few paths
 if result <= 10:
 for i in range(result):
 path = solver.reconstruct_path(m, n, i)
 print(f" Path {i}: {path}")
``

## Testing

### Comprehensive Test Suite

``python
import pytest

class TestUniquePaths:
 """Comprehensive test suite."""
 
 @pytest.fixture
 def solver(self):
 return UniquePathsSolver(algorithm="dp_optimized")
 
 def test_basic_examples(self, solver):
 """Test provided examples."""
 assert solver.count_paths(3, 7) == 28
 assert solver.count_paths(3, 2) == 3
 assert solver.count_paths(1, 1) == 1
 
 def test_edge_cases(self, solver):
 """Test edge cases."""
 # 1×n and m×1 grids
 assert solver.count_paths(1, 10) == 1
 assert solver.count_paths(10, 1) == 1
 
 # 2×2 grid
 assert solver.count_paths(2, 2) == 2
 
 # Larger grids
 assert solver.count_paths(5, 5) == 70
 
 def test_symmetry(self, solver):
 """Test that paths(m,n) = paths(n,m)."""
 assert solver.count_paths(3, 7) == solver.count_paths(7, 3)
 assert solver.count_paths(4, 6) == solver.count_paths(6, 4)
 
 def test_algorithm_equivalence(self):
 """Test that all algorithms give same result."""
 test_cases = [(3, 7), (3, 2), (1, 1), (5, 5)]
 
 for m, n in test_cases:
 results = []
 for algo in ["memo", "dp", "dp_optimized", "math"]:
 solver = UniquePathsSolver(algorithm=algo)
 results.append(solver.count_paths(m, n))
 
 # All should be equal
 assert len(set(results)) == 1
 
 def test_invalid_input(self, solver):
 """Test input validation."""
 with pytest.raises(ValueError):
 solver.count_paths(0, 5)
 
 with pytest.raises(ValueError):
 solver.count_paths(5, 0)
 
 with pytest.raises(ValueError):
 solver.count_paths(101, 5)
 
 def test_path_reconstruction(self, solver):
 """Test path reconstruction."""
 paths = [solver.reconstruct_path(3, 2, i) for i in range(3)]
 
 # All paths should start at (0,0) and end at (2,1)
 for path in paths:
 assert path[0] == (0, 0)
 assert path[-1] == (2, 1)
 
 # Verify each step is valid (down or right)
 for i in range(len(path) - 1):
 curr = path[i]
 next_pos = path[i+1]
 
 # Must be exactly one step down or right
 assert (next_pos == (curr[0]+1, curr[1]) or 
 next_pos == (curr[0], curr[1]+1))


# Run tests
if __name__ == "__main__":
 pytest.main([__file__, "-v"])
``

## Complexity Analysis

### Time Complexity

| Approach | Time | Explanation |
|----------|------|-------------|
| Brute Force | O(2^(m+n)) | Exponential branching |
| Memo (Top-down) | O(m×n) | Each cell computed once |
| DP (Bottom-up) | O(m×n) | Fill m×n table |
| DP (Optimized) | O(m×n) | Same iterations, less space |
| Mathematical | O(m+n) | Binomial coefficient |

### Space Complexity

| Approach | Space | Explanation |
|----------|-------|-------------|
| Brute Force | O(m+n) | Recursion stack |
| Memo (Top-down) | O(m×n) | Cache + stack |
| DP (Bottom-up) | O(m×n) | DP table |
| DP (Optimized) | O(n) | Single row |
| Mathematical | O(1) | No extra space |

**Recommended:** Use **DP optimized** for interviews (good balance of clarity and efficiency).

## Production Considerations

### 1. Large Grids

For very large grids that don't fit in memory:

``python
def uniquePaths_streaming(m: int, n: int):
 """
 Stream computation for massive grids.
 
 Compute one row at a time, write to disk if needed.
 """
 prev_row = [1] * n
 
 for i in range(1, m):
 curr_row = [1] # First column always 1
 
 for j in range(1, n):
 curr_row.append(curr_row[j-1] + prev_row[j])
 
 prev_row = curr_row
 
 return prev_row[n-1]
``

### 2. Path Enumeration

Sometimes we need to list all paths, not just count them:

``python
def enumerate_all_paths(m: int, n: int) -> List[List[tuple]]:
 """
 Generate all unique paths.
 
 Warning: Exponential number of paths!
 Only practical for small m, n.
 """
 paths = []
 
 def backtrack(row: int, col: int, current_path: List[tuple]):
 if row == m - 1 and col == n - 1:
 paths.append(current_path[:])
 return
 
 if row < m - 1:
 current_path.append((row + 1, col))
 backtrack(row + 1, col, current_path)
 current_path.pop()
 
 if col < n - 1:
 current_path.append((row, col + 1))
 backtrack(row, col + 1, current_path)
 current_path.pop()
 
 backtrack(0, 0, [(0, 0)])
 return paths
``

### 3. Obstacles (Unique Paths II)

If some cells have obstacles:

``python
def uniquePathsWithObstacles(obstacleGrid: List[List[int]]) -> int:
 """
 Count paths with obstacles.
 
 obstacleGrid[i][j] = 1 means obstacle, 0 means free.
 """
 m, n = len(obstacleGrid), len(obstacleGrid[0])
 
 # If start or end is blocked
 if obstacleGrid[0][0] == 1 or obstacleGrid[m-1][n-1] == 1:
 return 0
 
 dp = [[0] * n for _ in range(m)]
 dp[0][0] = 1
 
 # Fill first column
 for i in range(1, m):
 if obstacleGrid[i][0] == 0:
 dp[i][0] = dp[i-1][0]
 
 # Fill first row
 for j in range(1, n):
 if obstacleGrid[0][j] == 0:
 dp[0][j] = dp[0][j-1]
 
 # Fill rest
 for i in range(1, m):
 for j in range(1, n):
 if obstacleGrid[i][j] == 0:
 dp[i][j] = dp[i-1][j] + dp[i][j-1]
 
 return dp[m-1][n-1]
``

## Connections to ML Systems

The **path optimization and DP** pattern from this problem directly applies to neural architecture search:

### 1. Neural Architecture Search (NAS)

**Similarity to Unique Paths:**
- **Grid:** Search space of architectures
- **Paths:** Different architecture configurations
- **Goal:** Find optimal architecture (best accuracy)
- **DP:** Optimize search through the space

``python
class NASSearchSpace:
 """
 Neural architecture search space as a grid.
 
 Similar to unique paths:
 - Each 'cell' is a layer configuration
 - 'Paths' are full model architectures
 - DP to count/enumerate architectures efficiently
 """
 
 def __init__(self, num_layers: int, ops_per_layer: int):
 self.num_layers = num_layers
 self.ops_per_layer = ops_per_layer
 # Each layer can choose from ops_per_layer operations
 
 def count_architectures(self) -> int:
 """
 Count total possible architectures.
 
 If each layer has k choices and we have n layers:
 Total = k^n
 
 But with constraints (path dependencies), use DP.
 """
 # Simple case: independent layers
 return self.ops_per_layer ** self.num_layers
 
 def search_with_dp(self, validation_data):
 """
 Use DP to search architecture space efficiently.
 
 Similar to unique paths DP:
 - Build table of best architectures
 - Use previously computed results
 """
 # dp[layer][op] = best accuracy achievable up to this layer with this op
 dp = {}
 
 # Base case: first layer
 for op in range(self.ops_per_layer):
 arch = [op]
 acc = evaluate_partial_arch(arch, validation_data)
 dp[(0, op)] = (acc, arch)
 
 # Fill table
 for layer in range(1, self.num_layers):
 for op in range(self.ops_per_layer):
 best_acc = 0
 best_arch = []
 
 # Try all previous operations
 for prev_op in range(self.ops_per_layer):
 prev_acc, prev_arch = dp[(layer-1, prev_op)]
 new_arch = prev_arch + [op]
 new_acc = evaluate_partial_arch(new_arch, validation_data)
 
 if new_acc > best_acc:
 best_acc = new_acc
 best_arch = new_arch
 
 dp[(layer, op)] = (best_acc, best_arch)
 
 # Find best final architecture
 best_final = max(
 [dp[(self.num_layers-1, op)] for op in range(self.ops_per_layer)],
 key=lambda x: x[0]
 )
 
 return best_final[1] # Return architecture
``

### 2. Grid Search vs Smart Search

Traditional grid search is like brute force path enumeration:
- Try all combinations (exponential)
- Slow for large search spaces

DP-based search is like optimized path counting:
- Reuse subproblem solutions
- Prune unpromising branches
- Polynomial complexity

### Key Parallels

| Unique Paths | Neural Architecture Search |
|--------------|---------------------------|
| m×n grid | Layer×operation search space |
| Count all paths | Count all architectures |
| DP recurrence | DP search optimization |
| O(m×n) time | O(layers×ops) time |
| Path reconstruction | Architecture reconstruction |

## Interview Strategy

### How to Approach

**1. Clarify (1 min)**
``
- Can only move down/right? (Yes)
- Grid always valid (m, n >= 1)? (Yes)
- Any obstacles? (No, unless follow-up)
``

**2. Explain Intuition (2 min)**
``
"To reach any cell (i,j), I must have come from (i-1,j) or (i,j-1).
So paths to (i,j) = paths to (i-1,j) + paths to (i,j-1).
This is a DP problem with clear subproblem structure."
``

**3. Discuss Approaches (2 min)**
``
1. Recursion: O(2^(m+n)), too slow
2. DP with memo: O(m×n) time and space
3. DP bottom-up: O(m×n) time and space, cleaner
4. DP optimized: O(m×n) time, O(n) space
5. Math: O(m+n) time, O(1) space

I'll implement DP bottom-up for clarity, then optimize space.
``

**4. Code (8-10 min)**
- Start with 2D DP
- Optimize to 1D if time permits

**5. Test (3 min)**
- Walk through 3×2 example
- Test edge case: 1×1

**6. Complexity (2 min)**
- Time: O(m×n)
- Space: O(n) optimized, or O(m×n) for 2D

### Common Mistakes

1. **Wrong initialization:**
 - First row/column should be 1, not 0

2. **Off-by-one in loops:**
 - Start from index 1, not 0 (after initialization)

3. **Incorrect recurrence:**
 - Must be `dp[i][j] = dp[i-1][j] + dp[i][j-1]`, not multiply

4. **Not optimizing space:**
 - Mention space optimization even if you implement 2D version

### Follow-up Questions

**Q1: With obstacles?**

See `uniquePathsWithObstacles` above.

**Q2: Minimum path sum?**

Different problem - need to minimize cost, not count paths. Use similar DP but track min sum.

**Q3: Enumerate all paths?**

Exponential in number of paths, use backtracking (see `enumerate_all_paths` above).

## Additional Practice & Variants

### 1. Unique Paths II (With Obstacles)

Implemented above. Key difference: check for obstacles before using cell in recurrence.

### 2. Minimum Path Sum (LeetCode 64)

**Problem:** Find path with minimum sum of numbers.

``python
def minPathSum(grid: List[List[int]]) -> int:
 """
 Find path with minimum sum.
 
 Similar DP recurrence but use min instead of sum:
 dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
 """
 m, n = len(grid), len(grid[0])
 dp = [[0] * n for _ in range(m)]
 
 dp[0][0] = grid[0][0]
 
 # First column
 for i in range(1, m):
 dp[i][0] = dp[i-1][0] + grid[i][0]
 
 # First row
 for j in range(1, n):
 dp[0][j] = dp[0][j-1] + grid[0][j]
 
 # Fill table
 for i in range(1, m):
 for j in range(1, n):
 dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
 
 return dp[m-1][n-1]
``

### 3. Unique Paths III (All Paths with Constraints)

**Problem:** Walk over every non-obstacle square exactly once.

Uses backtracking (not DP) since we need to track visited cells.

## Key Takeaways

✅ **Dynamic programming transforms exponential to polynomial** - from O(2^(m+n)) to O(m×n)

✅ **Recurrence relation** is key: `dp[i][j] = dp[i-1][j] + dp[i][j-1]`

✅ **Space optimization** reduces O(m×n) to O(n) by using rolling array

✅ **Mathematical insight** gives O(1) space solution via combinatorics

✅ **DP pattern applies broadly** - grid paths, architecture search, optimization problems

✅ **Path reconstruction** shows DP table encodes all solution information

✅ **Testing edge cases** - 1×1, 1×n, m×1, obstacles, large grids

✅ **Same DP pattern** in neural architecture search - count/optimize paths through search space

✅ **Bottom-up vs top-down** - both work, bottom-up often cleaner for grid DP

✅ **Production extensions** - obstacles, costs, path enumeration, large grids

### Mental Model

Think of this problem as:
- **Grid Paths:** DP to count paths efficiently
- **Architecture Search:** DP to search model space
- **Speech Model Search:** DP to find optimal configurations

All use the pattern: **break into subproblems → solve small cases → build up solution → optimize with memoization/tables**

### Connection to Thematic Link: Dynamic Programming and Path Optimization

All three topics use **DP for path optimization in exponential search spaces**:

**DSA (Unique Paths):**
- DP to count paths in m×n grid
- Recurrence: paths(i,j) = paths(i-1,j) + paths(i,j-1)
- Reduces exponential to O(m×n)

**ML System Design (Neural Architecture Search):**
- DP/RL to search architecture space
- Build optimal networks from smaller components
- Reduce exponential search to manageable complexity

**Speech Tech (Speech Architecture Search):**
- DP to explore encoder/decoder configurations
- Build speech models from optimal sub-architectures
- Systematic search through design space

The **unifying principle**: use dynamic programming to navigate exponentially large search spaces by breaking problems into subproblems and building optimal solutions from optimal sub-solutions.

---

**Originally published at:** [arunbaby.com/dsa/0021-unique-paths](https://www.arunbaby.com/dsa/0021-unique-paths/)

*If you found this helpful, consider sharing it with others who might benefit.*



