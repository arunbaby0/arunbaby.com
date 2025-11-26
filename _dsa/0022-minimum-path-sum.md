---
title: "Minimum Path Sum"
day: 22
collection: dsa
categories:
  - dsa
tags:
  - dynamic-programming
  - matrix
  - grid
  - optimization
  - medium
subdomain: "Grid Dynamic Programming"
tech_stack: [Python, C++, Java]
scale: "O(M×N) time, O(1) space"
companies: [Google, Amazon, Apple, Bloomberg, Goldman Sachs]
related_dsa_day: 22
related_ml_day: 22
related_speech_day: 22
---

**The classic grid optimization problem that bridges the gap between simple recursion and 2D Dynamic Programming.**

## Problem

Given a `m x n` grid filled with non-negative numbers, find a path from the **top-left** cell to the **bottom-right** cell which minimizes the sum of all numbers along its path.

**Constraints:**
- You can only move either **down** or **right** at any point in time.
- `m` and `n` are the dimensions of the grid.
- The numbers in the grid are non-negative.

**Example 1:**
```
Input: grid = [
  [1, 3, 1],
  [1, 5, 1],
  [4, 2, 1]
]
Output: 7
```

**Explanation:**
The path is `1 → 3 → 1 → 1 → 1`.
Sum: `1 + 3 + 1 + 1 + 1 = 7`.

Let's visualize the grid and the path:
```
[1] -> [3] -> [1]
               |
              [1] -> [1]
```
Wait, looking at the grid:
(0,0)=1 -> (0,1)=3 -> (0,2)=1 -> (1,2)=1 -> (2,2)=1.
Total = 7.

Is there any other path?
- Down, Down, Right, Right: `1 -> 1 -> 4 -> 2 -> 1` = 9.
- Down, Right, Down, Right: `1 -> 1 -> 5 -> 2 -> 1` = 10.
- Right, Down, Down, Right: `1 -> 3 -> 5 -> 2 -> 1` = 12.

Clearly, 7 is the minimum.

## Thematic Connection: Path Finding and Cost Minimization

Before we dive into the code, let's pause and appreciate why this problem matters.

In **Machine Learning System Design**, we often face choices that can be modeled as a directed graph or a grid. For example, when designing a data pipeline, we might have multiple stages (preprocessing, training, evaluation). Each stage has a "cost" (time, money, compute resources). Finding the most efficient pipeline configuration is mathematically similar to finding the minimum path sum.

In **Speech Technology**, the Viterbi algorithm used in Hidden Markov Models (HMMs) for speech recognition is essentially finding the most likely (minimum cost) sequence of hidden states that generate the observed audio. The "grid" there is formed by time steps on one axis and possible states (phonemes) on the other.

So, mastering this grid optimization logic gives you the mental framework to solve complex system optimization problems later.

## Mathematical Foundation

To truly understand Dynamic Programming, we need to understand two key properties: **Optimal Substructure** and **Overlapping Subproblems**.

### 1. Optimal Substructure
A problem has optimal substructure if an optimal solution can be constructed from optimal solutions of its subproblems.

**Proof:**
Let `Cost(i, j)` be the minimum cost to reach cell `(i, j)`.
To reach `(i, j)`, we must have come from either `(i-1, j)` (top) or `(i, j-1)` (left).
Suppose the path coming from the top is the optimal one. Then, the path from `(0, 0)` to `(i-1, j)` *must* also be the minimum cost path to reach `(i-1, j)`.
**Why?** Proof by Contradiction:
Assume there exists a path to `(i-1, j)` with a lower cost than our current "optimal" path. Then we could simply take that lower-cost path to `(i-1, j)` and then move down to `(i, j)`, resulting in a total cost strictly less than our supposed optimal cost. This contradicts the assumption that we had the optimal path.
Therefore, `Cost(i, j) = grid[i][j] + min(Cost(i-1, j), Cost(i, j-1))`.

### 2. Overlapping Subproblems
A problem has overlapping subproblems if the recursive algorithm visits the same subproblems repeatedly.

In our grid, to calculate `Cost(i, j)`, we need `Cost(i-1, j)` and `Cost(i, j-1)`.
To calculate `Cost(i-1, j+1)`, we need `Cost(i-2, j+1)` and `Cost(i-1, j)`.
Notice that `Cost(i-1, j)` is needed by both `(i, j)` and `(i-1, j+1)`.
In a large grid, this overlap happens exponentially often. This is why simple recursion fails and why we need DP.

## Approach 1: Brute Force Recursion

Let's start with the most intuitive approach. If we are at any cell `(i, j)`, what are our choices?
1. Move **Right** to `(i, j+1)`
2. Move **Down** to `(i+1, j)`

We want to choose the move that eventually leads to the smallest total sum. This sounds like a recursive definition!

Let `minPath(i, j)` be the minimum cost to reach the bottom-right corner `(m-1, n-1)` starting from cell `(i, j)`.

The cost of the current cell `grid[i][j]` is always added. Then we need to add the minimum of the rest of the path.
`minPath(i, j) = grid[i][j] + min(minPath(i, j+1), minPath(i+1, j))`

### Base Cases
1. **Destination Reached:** If we are at `(m-1, n-1)`, the cost is just `grid[m-1][n-1]`. We have nowhere else to go.
2. **Out of Bounds:** If `i >= m` or `j >= n`, this is an invalid path. We should return "Infinity" so that the `min()` function never chooses this path.

### Python Implementation (Recursive)

```python
import math

def minPathSum_recursive(grid):
    m, n = len(grid), len(grid[0])
    
    def calculate(i, j):
        # Base Case: Reached bottom-right
        if i == m - 1 and j == n - 1:
            return grid[i][j]
        
        # Base Case: Out of bounds
        if i >= m or j >= n:
            return math.inf
        
        # Recursive Step
        move_right = calculate(i, j + 1)
        move_down = calculate(i + 1, j)
        
        return grid[i][j] + min(move_right, move_down)

    return calculate(0, 0)
```

### Complexity Analysis
- **Time Complexity:** \(O(2^{m+n})\). At each step, we branch into two possibilities. The depth of the recursion is \(m+n\). This is exponential and extremely slow for large grids.
- **Space Complexity:** \(O(m+n)\) for the recursion stack.

### Why is it slow?
Let's trace the calls for a simple 2x2 grid.
```
(0,0)
  |-- (0,1)
  |     |-- (0,2) [Out]
  |     |-- (1,1) [Target]
  |
  |-- (1,0)
        |-- (1,1) [Target]
        |-- (2,0) [Out]
```
Notice that `(1,1)` is reached from `(0,1)` AND from `(1,0)`. In a larger grid, the number of overlapping subproblems explodes. We are re-calculating the minimum path for the same cells over and over again.

## Approach 2: Recursion with Memoization (Top-Down DP)

To fix the overlapping subproblems, we can store the result of `calculate(i, j)` in a cache (memoization table). If we encounter the same `(i, j)` again, we just return the stored value.

### Python Implementation (Memoization)

```python
def minPathSum_memo(grid):
    m, n = len(grid), len(grid[0])
    memo = {}
    
    def calculate(i, j):
        # Check cache first
        if (i, j) in memo:
            return memo[(i, j)]
        
        if i == m - 1 and j == n - 1:
            return grid[i][j]
        
        if i >= m or j >= n:
            return float('inf')
        
        res = grid[i][j] + min(calculate(i, j + 1), calculate(i + 1, j))
        
        # Store in cache
        memo[(i, j)] = res
        return res

    return calculate(0, 0)
```

### Complexity Analysis
- **Time Complexity:** \(O(m \times n)\). There are \(m \times n\) unique states (cells). Each state is computed once.
- **Space Complexity:** \(O(m \times n)\) for the memoization table + \(O(m + n)\) for recursion stack.

This is much better! But we can do even better by removing the recursion overhead.

## Approach 3: Iterative Dynamic Programming (Bottom-Up)

Recursion starts from the top-left and asks "what's the cost to the end?". Iterative DP usually starts from the end (or the beginning) and builds the solution up.

Let's flip the definition. Let `dp[i][j]` be the minimum path sum **to reach** cell `(i, j)` **from** the top-left `(0, 0)`.

**Recurrence Relation:**
To reach `(i, j)`, we must have come from either:
1. Top: `(i-1, j)`
2. Left: `(i, j-1)`

So, `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`.

**Boundary Conditions:**
- `dp[0][0] = grid[0][0]`
- First Row `(i=0)`: We can only come from the left. `dp[0][j] = dp[0][j-1] + grid[0][j]`.
- First Column `(j=0)`: We can only come from above. `dp[i][0] = dp[i-1][0] + grid[i][0]`.

### Visualization of the DP Table

Input:
```
1 3 1
1 5 1
4 2 1
```

**Step 1: Initialize (0,0)**
```
1 . .
. . .
. . .
```

**Step 2: Fill First Row**
`dp[0][1] = 1 + 3 = 4`
`dp[0][2] = 4 + 1 = 5`
```
1 4 5
. . .
. . .
```

**Step 3: Fill First Column**
`dp[1][0] = 1 + 1 = 2`
`dp[2][0] = 2 + 4 = 6`
```
1 4 5
2 . .
6 . .
```

**Step 4: Fill Inner Cells**
`dp[1][1] = grid[1][1] + min(dp[0][1], dp[1][0])`
`dp[1][1] = 5 + min(4, 2) = 5 + 2 = 7`

`dp[1][2] = grid[1][2] + min(dp[0][2], dp[1][1])`
`dp[1][2] = 1 + min(5, 7) = 1 + 5 = 6`

`dp[2][1] = grid[2][1] + min(dp[1][1], dp[2][0])`
`dp[2][1] = 2 + min(7, 6) = 2 + 6 = 8`

`dp[2][2] = grid[2][2] + min(dp[1][2], dp[2][1])`
`dp[2][2] = 1 + min(6, 8) = 1 + 6 = 7`

Final DP Table:
```
1 4 5
2 7 6
6 8 7
```
The answer is `dp[2][2] = 7`.

### Python Implementation (Iterative)

```python
def minPathSum_iterative(grid):
    if not grid:
        return 0
    
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                dp[i][j] = grid[i][j]
            elif i == 0:
                # First row, can only come from left
                dp[i][j] = dp[i][j-1] + grid[i][j]
            elif j == 0:
                # First column, can only come from top
                dp[i][j] = dp[i-1][j] + grid[i][j]
            else:
                # Inner cell, choose min of top or left
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
                
    return dp[m-1][n-1]
```

### Complexity Analysis
- **Time Complexity:** \(O(m \times n)\). We iterate through the grid once.
- **Space Complexity:** \(O(m \times n)\) for the `dp` table.

## Approach 4: Space Optimization (In-Place)

Do we really need a separate `dp` table? Look at the recurrence again:
`dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`

We are just adding values to the original grid values. If the interviewer allows modifying the input, we can store the DP values directly in `grid`.

```python
def minPathSum_inplace(grid):
    m, n = len(grid), len(grid[0])
    
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            elif i == 0:
                grid[i][j] += grid[i][j-1]
            elif j == 0:
                grid[i][j] += grid[i-1][0]
            else:
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
                
    return grid[m-1][n-1]
```

- **Space Complexity:** \(O(1)\).

## Approach 5: Space Optimization (1D Array)

What if we cannot modify the input (e.g., the grid is read-only or shared)? Do we still need `O(m*n)` space?

Notice that to calculate row `i`, we only need the values from row `i` (which we are currently computing) and row `i-1` (the previous row). We don't need row `i-2` or anything before that.

So, we can just keep two rows: `prev_row` and `curr_row`.
Actually, we can do even better. We can use a single 1D array!

Let `dp[j]` represent the minimum path sum to reach the cell at column `j` in the *current* row we are processing.

When we are at `grid[i][j]`:
- `dp[j]` currently holds the value for `grid[i-1][j]` (value from above).
- `dp[j-1]` holds the value for `grid[i][j-1]` (value from left, which we just updated).

So: `dp[j] = grid[i][j] + min(dp[j], dp[j-1])`.

```python
def minPathSum_1d(grid):
    m, n = len(grid), len(grid[0])
    dp = [0] * n
    
    # Initialize first value
    dp[0] = grid[0][0]
    
    # Initialize first row
    for j in range(1, n):
        dp[j] = dp[j-1] + grid[0][j]
        
    for i in range(1, m):
        # Handle first column of current row
        dp[0] += grid[i][0]
        
        for j in range(1, n):
            # dp[j] is value from top
            # dp[j-1] is value from left
            dp[j] = grid[i][j] + min(dp[j], dp[j-1])
            
    return dp[n-1]
```

- **Space Complexity:** \(O(n)\). This is the most optimal space complexity without modifying input.

## Multi-Language Implementations

As a junior engineer, you might be working in a codebase that uses Java or C++. It's important to be fluent in multiple syntaxes.

### C++ Implementation

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        if (grid.empty()) return 0;
        int m = grid.size();
        int n = grid[0].size();
        
        // We will use the in-place approach for C++
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 && j == 0) continue;
                
                if (i == 0) {
                    grid[i][j] += grid[i][j-1];
                } else if (j == 0) {
                    grid[i][j] += grid[i-1][j];
                } else {
                    grid[i][j] += min(grid[i-1][j], grid[i][j-1]);
                }
            }
        }
        return grid[m-1][n-1];
    }
};
```

### Java Implementation

```java
class Solution {
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        int m = grid.length;
        int n = grid[0].length;
        
        // Using 1D array approach for Java
        int[] dp = new int[n];
        
        dp[0] = grid[0][0];
        
        // Initialize first row
        for (int j = 1; j < n; j++) {
            dp[j] = dp[j-1] + grid[0][j];
        }
        
        for (int i = 1; i < m; i++) {
            // First column of current row
            dp[0] += grid[i][0];
            
            for (int j = 1; j < n; j++) {
                dp[j] = Math.min(dp[j], dp[j-1]) + grid[i][j];
            }
        }
        
        return dp[n-1];
    }
}
```

## Interview Simulation: The Dialogue

To help you prepare, let's simulate how a real interview conversation might go.

**Interviewer:** "Okay, here's the problem. You have a grid of numbers. You start at top-left, end at bottom-right. You can only go down or right. Find the path with the minimum sum."

**Candidate (You):** "Understood. Just to clarify, are the numbers always positive?"

**Interviewer:** "Yes, non-negative integers."

**Candidate:** "And if the grid is empty, should I return 0?"

**Interviewer:** "Yes, assume valid input generally, but 0 for empty is fine."

**Candidate:** "Great. My first thought is that this looks like a shortest path problem. Since we can only move down and right, there are no cycles. This suggests a Dynamic Programming approach because the decision at any cell depends on the optimal decisions made for previous cells. Specifically, to reach cell `(i, j)` with minimal cost, I must have come from either the cell above it or the cell to the left of it."

**Interviewer:** "That sounds correct. Can you define the recurrence relation?"

**Candidate:** "Sure. Let `dp[i][j]` be the min path sum to reach `(i, j)`. Then `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`. The base case is `dp[0][0] = grid[0][0]`."

**Interviewer:** "Good. What about the first row and first column?"

**Candidate:** "Ah, yes. For the first row, we can only come from the left, so `dp[0][j] = dp[0][j-1] + grid[0][j]`. Similarly for the first column, we can only come from above, so `dp[i][0] = dp[i-1][0] + grid[i][0]`."

**Interviewer:** "Excellent. Go ahead and code it."

*(Candidate writes the code...)*

**Interviewer:** "Looks good. What is the space complexity?"

**Candidate:** "I used a 2D array, so it's `O(m*n)`. But looking at the recurrence, I only need the previous row to calculate the current row. So I could optimize this to `O(n)` space using a 1D array."

**Interviewer:** "Can you do it in `O(1)` space?"

**Candidate:** "If I am allowed to modify the input grid, I can store the cumulative sums directly in the grid cells. That would be `O(1)` auxiliary space."

**Interviewer:** "Perfect. Let's assume the input is read-only. How would you handle a case where the grid is extremely wide but not very tall (e.g., 10 rows, 1,000,000 columns)?"

**Candidate:** "That's a great edge case. If `n` is much larger than `m`, my `O(n)` space solution would use a lot of memory. I should check which dimension is smaller. If `m < n`, I can treat columns as rows and iterate column by column, using an array of size `m` instead. So the space complexity would be `O(min(m, n))`."

**Interviewer:** "Very impressive. One last question: What if we could move diagonally?"

**Candidate:** "If we can move diagonally (down-right), then I would just add a third term to the min function: `min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])`. The rest of the logic stays the same."

## Related Problems: The Grid DP Family

Once you master Minimum Path Sum, you can solve a whole family of problems. Let's look at 5 of them in detail.

### 1. Unique Paths (LeetCode 62)
- **Problem:** Count the number of unique paths from top-left to bottom-right.
- **Difference:** Instead of `min()`, we use `sum()`.
- **Recurrence:** `dp[i][j] = dp[i-1][j] + dp[i][j-1]`.
- **Code Snippet:**
```python
def uniquePaths(m, n):
    dp = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]
    return dp[n-1]
```

### 2. Unique Paths II (LeetCode 63)
- **Problem:** Same as above, but with obstacles.
- **Difference:** If `grid[i][j] == obstacle`, then `dp[i][j] = 0`.
- **Key Insight:** An obstacle blocks all flow through that cell.
- **Code Snippet:**
```python
def uniquePathsWithObstacles(obstacleGrid):
    if obstacleGrid[0][0] == 1: return 0
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [0] * n
    dp[0] = 1
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]
    return dp[n-1]
```

### 3. Dungeon Game (LeetCode 174)
- **Problem:** Start with health `H`. Some cells decrease health (monsters), some increase it (potions). Find min initial health to survive.
- **Difference:** This is a "reverse" DP. You start from the bottom-right and work your way back to the top-left.
- **Recurrence:** `dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - grid[i][j])`.
- **Why Reverse?** Because the decision at `(i, j)` depends on the *future* requirement at `(i+1, j)` or `(i, j+1)`. If we went forward, we wouldn't know if a future potion would save us.

### 4. Cherry Pickup (LeetCode 741)
- **Problem:** Go from top-left to bottom-right, then back to top-left, collecting maximum cherries.
- **Difference:** This requires two simultaneous paths. The state becomes `dp[r1][c1][r2]`. It's a Hard problem, but built on the same principles.
- **Complexity:** `O(N^3)`.

### 5. Maximum Path Sum
- **Problem:** Find the path with the maximum sum.
- **Difference:** Just change `min` to `max`.
- **Application:** Finding the most profitable route for a salesperson.

## Debugging Guide: Common Mistakes

Even experienced engineers make mistakes with DP. Here are the most common ones and how to spot them.

### 1. The "Off-by-One" Error
- **Symptom:** `IndexError: list index out of range`.
- **Cause:** Iterating up to `m` instead of `m-1`, or accessing `dp[i-1]` when `i=0`.
- **Fix:** Always handle the first row and first column separately, or pad the DP table with an extra row/column of "Infinity" values.

### 2. The "Greedy" Trap
- **Symptom:** Wrong answer on complex test cases.
- **Cause:** Thinking "I should just pick the smaller number at each step".
- **Example:**
```
1 100 1
1   1 1
```
Greedy would go Right (1 -> 100) because 100 is... wait, Greedy minimizes.
Example:
```
1 2 5
9 1 1
```
Greedy at (0,0): Right (2) vs Down (9). Picks Right.
Path: 1 -> 2 -> 5. Sum = 8.
Optimal: Down (9) -> Right (1) -> Right (1). Sum = 12. (Wait, this is max path).
For Min Path:
```
1 9 9
5 1 1
```
Greedy at (0,0): Down (5) vs Right (9). Picks Down.
Path: 1 -> 5 -> 1 -> 1. Sum = 8.
Optimal: 1 -> 9 -> 9... wait.
Actually, Greedy fails because it doesn't look ahead.
Correct Example:
```
1 5 100
1 1 1
```
Greedy at (0,0): Right (5) vs Down (1). Picks Down.
Path: 1 -> 1 -> 1 -> 1. Sum = 4.
Path 2: 1 -> 5 -> 100. Sum = 106.
Here Greedy worked.
Counter-Example for Greedy Min Path:
```
1 10 10
1  1  1
```
Greedy at (0,0): Right (10) vs Down (1). Picks Down.
Path: 1 -> 1 -> 1 -> 1. Sum = 4.
Path 2: 1 -> 10 -> 10. Sum = 21.
Greedy works often on simple grids, but fails when a "locally bad" move leads to a "globally good" path.
Imagine:
```
1  100 1
10 100 1
```
Start (0,0).
Right: 100. Down: 10.
Greedy picks Down.
Path: 1 -> 10 -> 100 -> 1 = 112.
Optimal: 1 -> 100 -> 1 -> 1... wait.
Let's construct a proper counter-example.
```
1 1 10
5 1 1
```
Start (0,0).
Right (1) vs Down (5). Greedy picks Right.
Path: 1 -> 1 -> 10 -> 1 (forced). Sum = 13.
Optimal: 1 -> 5 -> 1 -> 1. Sum = 8.
**Fix:** Never use Greedy on grids unless you prove the "Greedy Choice Property". Use DP.

### 3. The "Uninitialized DP Table"
- **Symptom:** Random huge numbers or zeros.
- **Cause:** Forgetting to set the base case `dp[0][0]`.
- **Fix:** Always initialize the start point before the loops.

## Real-World Application: Seam Carving

One fascinating application of this algorithm is **Seam Carving** for content-aware image resizing.
- **Goal:** Resize an image (reduce width) without distorting important objects.
- **Method:** Find a "seam" (a path of pixels from top to bottom) that has the least "energy" (least importance/detail) and remove it.
- **Algorithm:**
    1. Calculate "energy" of each pixel (e.g., gradient magnitude).
    2. Use the Minimum Path Sum algorithm (allowing diagonal moves) to find the vertical seam with the lowest total energy.
    3. Remove that seam.
    4. Repeat.

This is exactly the same logic! You are finding a path through a grid of pixels to minimize the sum of energies.

## Conclusion

The Minimum Path Sum problem is a cornerstone of Dynamic Programming. It teaches you:
1.  **State Definition:** How to represent the problem at step `(i, j)`.
2.  **Transition:** How to move from previous states to the current state.
3.  **Optimization:** How to reduce space from `O(m*n)` to `O(n)`.

Mastering this gives you the tools to solve harder variations like "Dungeon Game", "Unique Paths II", and "Cherry Pickup".

**Practice Problems:**
- [Unique Paths](https://leetcode.com/problems/unique-paths/)
- [Unique Paths II](https://leetcode.com/problems/unique-paths-ii/)
- [Dungeon Game](https://leetcode.com/problems/dungeon-game/)
- [Cherry Pickup](https://leetcode.com/problems/cherry-pickup/)

Happy Coding!

---

**Originally published at:** [arunbaby.com/dsa/0022-minimum-path-sum](https://www.arunbaby.com/dsa/0022-minimum-path-sum/)

*If you found this helpful, consider sharing it with others who might benefit.*

<div style="opacity: 0.6; font-size: 0.8em; margin-top: 2em;">
  Created with LLM assistance
</div>
