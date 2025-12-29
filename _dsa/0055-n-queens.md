---
title: "N-Queens"
day: 55
collection: dsa
categories:
  - dsa
tags:
  - backtracking
  - recursion
  - constraint-satisfaction
  - hard
  - bit-manipulation
difficulty: Hard
subdomain: "Backtracking"
tech_stack: Python
scale: "O(N!) complexity, optimized with symmetry and bitmasks"
companies: Google, Amazon, Meta, Microsoft
related_ml_day: 55
related_speech_day: 55
related_agents_day: 55
---

**"The N-Queens problem is the 'Hello World' of constraint satisfaction—it teaches us how to prune the search tree before it consumes our CPU."**

## 1. Problem Statement

The **N-Queens** puzzle is the problem of placing `n` chess queens on an `n × n` chessboard such that no two queens attack each other.

A queen can attack another if they are in the same:
1. **Row**
2. **Column**
3. **Diagonal** (both directions)

Given an integer `n`, return all distinct solutions to the N-Queens puzzle. You may return the answer in any order. Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' indicate a queen and an empty space, respectively.

### 1.1 Constraints
- `1 <= n <= 9` (Standard LeetCode)
- In real-world optimization scenarios, `n` can be much larger, requiring advanced heuristics.

---

## 2. Understanding the Problem

### 2.1 The Combinatorial Explosion
The N-Queens problem is a classic example of a **Combinatorial Search** problem.
- On a 4x4 board, there are $\binom{16}{4} = 1,820$ ways to place 4 queens.
- On an 8x8 board, there are $\binom{64}{8} = 4,426,165,368$ ways.
- On a 16x16 board, the number is approximately $3.6 \times 10^{20}$.

If we used a naive brute-force approach (checking every possible combination), the sun would die before we finished $n=20$. We need a way to **prune** the search space.

### 2.2 Constraint Satisfaction
N-Queens is a perfect example of a **Constraint Satisfaction Problem (CSP)**. In a CSP, we have:
- **Variables**: The positions of the $N$ queens.
- **Domains**: The possible squares on the board.
- **Constraints**: No two queens in the same row, column, or diagonal.

By placing queens one row at a time, we implicitly satisfy the "one queen per row" constraint. This reduces our variables to "which column is the queen in for Row $i$?".

### 2.3 Thematic Link: Constraint Satisfaction and Search
Today's shared theme across all four tracks is **Constraint Satisfaction and Search**:
- **DSA**: We are using backtracking to solve a discrete constraint problem.
- **ML System Design**: AutoML systems must search the space of hyperparameters and architectures under constraints (latency, memory, accuracy).
- **Speech Tech**: Neural Architecture Search (NAS) for Speech is a specialized version of AutoML, finding the optimal acoustic model architecture.
- **Agents**: Agent orchestration involves scheduling and coordinating multiple agents to solve a complex task under resource and dependency constraints.

---

## 3. Approach 1: Simple Backtracking (The Foundation)

### 3.1 The Logic
1. Start at row 0.
2. For each column in the current row:
   - Check if placing a queen is **safe**.
   - If safe, place the queen and move to the next row (recursion).
   - If not safe, or if the recursive call returns (backtrack), remove the queen and try the next column.
3. If we reach row $n$, we've found a valid solution.

### 3.2 What is "Safe"?
To check if `(row, col)` is safe, we check:
1. Column `col`.
2. Upper-left diagonal: `(row-1, col-1), (row-2, col-2)...`
3. Upper-right diagonal: `(row-1, col+1), (row-2, col+2)...`

We don't need to check lower squares because we haven't placed queens there yet.

---

## 4. Approach 2: Optimized Backtracking (Using Sets)

### 4.1 The Bottleneck
In Approach 1, the "isSafe" check takes $O(N)$ time. Since we call this millions of times, we can optimize it to $O(1)$ by using boolean arrays or sets.

### 4.2 Mapping Diagonals
The trick to $O(1)$ diagonal checks is identifying the invariant for each diagonal:
- **Same Column**: `col` is constant.
- **Main Diagonal (Top-Left to Bottom-Right)**: The value `row - col` is constant for all squares on the same diagonal.
- **Anti-Diagonal (Top-Right to Bottom-Left)**: The value `row + col` is constant for all squares on the same diagonal.

By maintaining three sets (`cols`, `main_diagonals`, `anti_diagonals`), we can check safety in $O(1)$.

---

## 5. Approach 3: Bitmask Optimization (The System Engineer's Way)

This is the fastest version of backtracking. Instead of sets or arrays, we use bits in an integer to represent which columns and diagonals are occupied.

### 5.1 How the Bitmask Works
- `cols_mask`: Bit $i$ is 1 if column $i$ is taken.
- `main_diag_mask`: Represents "shadows" cast by queens on the main diagonals.
- `anti_diag_mask`: Represents "shadows" cast by queens on the anti-diagonals.

When moving from `row` to `row + 1`:
- `cols_mask` stays the same.
- `main_diag_mask` shifts **right** (as we go down, the diagonal moves to higher column indices relative to the next row).
- `anti_diag_mask` shifts **left**.

This is why compilers and high-performance engines love N-Queens; it's a pure bit-manipulation dance.

---

## 6. Implementation (Optimized Backtracking)

```python
from typing import List

class Solution:
    """
    N-Queens Solution using local state sets for O(1) safety checks.
    Time Complexity: O(N!) - Though heavily pruned.
    Space Complexity: O(N^2) for the result storage, O(N) for recursion.
    """
    def solveNQueens(self, n: int) -> List[List[str]]:
        results = []
        # State tracking sets
        cols = [False] * n
        main_diag = [False] * (2 * n - 1) # row - col + (n-1) to keep index positive
        anti_diag = [False] * (2 * n - 1) # row + col
        
        # Initial board
        board = [["."] * n for _ in range(n)]
        
        def backtrack(row: int):
            if row == n:
                # Found a valid configuration
                results.append(["".join(r) for r in board])
                return
            
            for col in range(n):
                # Diagonal index transforms
                m_idx = row - col + (n - 1)
                a_idx = row + col
                
                # Pruning: Is it safe?
                if not (cols[col] or main_diag[m_idx] or anti_diag[a_idx]):
                    # Place Queen
                    board[row][col] = "Q"
                    cols[col] = main_diag[m_idx] = anti_diag[a_idx] = True
                    
                    # Search deeper
                    backtrack(row + 1)
                    
                    # Backtrack (Undo)
                    board[row][col] = "."
                    cols[col] = main_diag[m_idx] = anti_diag[a_idx] = False
                    
        backtrack(0)
        return results

# High Performance Bit-Manipulation Version (for counting solutions)
class SolutionBitmask:
    def totalNQueens(self, n: int) -> int:
        self.count = 0
        self.full_mask = (1 << n) - 1
        
        def solve(cols, main, anti):
            if cols == self.full_mask:
                self.count += 1
                return
            
            # Bits that are currently occupied
            occupied = cols | main | anti
            # Bits that are free (within the first n bits)
            free = self.full_mask & (~occupied)
            
            while free:
                # Pick the lowest set bit (position for a queen)
                curr_bit = free & -free
                # Remove this bit from free slots
                free ^= curr_bit
                
                # Recurse:
                # - New main: (old_main | bit) >> 1
                # - New anti: (old_anti | bit) << 1
                solve(cols | curr_bit, 
                      (main | curr_bit) >> 1, 
                      (anti | curr_bit) << 1)
        
        solve(0, 0, 0)
        return self.count
```

---

### 6.2 The Mathematical Intuition Behind the Bitmask
The bitmask approach isn't just a "coding trick"—it's an implementation of **Parallel State Search**.
When we do `free = self.full_mask & (~(cols | main | anti))`, we are performing a hardware-level logical `AND` and `NOT`. In a single CPU cycle, we are checking the "safeness" of all $N$ columns simultaneously.

The shifts:
- `(main | curr_bit) >> 1`: This represents the diagonal line moving "down and right." As you move to the next row, a queen at column $c$ "threatens" column $c+1$ in the next row.
- `(anti | curr_bit) << 1`: This represents the diagonal moving "down and left." A queen at column $c$ "threatens" column $c-1$ in the row below.

By using integers, we treat the CPU's register as a specialized "Chess Constraint Co-processor."

---

## 7. Advanced Optimization: Symmetry and Group Theory

The N-Queens problem exhibits **Rotational and Reflectional Symmetry**. The square board is a member of the **Dihedral Group $D_4$**, which has 8 symmetries (0°, 90°, 180°, 270° rotations, and 4 reflection axes).

### 7.1 Symmetry Reduction
If you find a solution on one side of the board, you can automatically derive 7 others (unless the solution itself is symmetric).
In production solvers:
- We only search for the first queen in columns $[0, \lfloor N/2 \rfloor]$.
- If $N$ is odd and we place the first queen in the exact middle column, we only need to search the second queen in columns $[0, \lfloor N/2 \rfloor]$ for that branch.
- This effectively provides a **2x to 8x speedup** depending on how aggressive the symmetry breaking is.

### 7.2 The "All-Solutions" vs "Unique-Solutions" Distinction
In interview settings, make sure to ask: "Do you want all 92 solutions for N=8, or only the 12 unique solutions (accounting for symmetry)?"
Generating only unique solutions requires checking each found solution against its rotations/reflections using a **Canonical Form** (usually the lexicographically smallest representation).

---

## 8. Complexity Analysis & Benchmarking

### 8.1 Theoretical Bounds
- **Time Complexity**: $O(N!)$. While the pruning is efficient, the upper bound remains factorial.
- **Space Complexity**: $O(N)$ for the recursion stack. This is the "Gold Standard" for combinatorial search.

### 8.2 Real-world Benchmarks
On a modern CPU (e.g., Apple M2 or Intel i9), a well-optimized bitmask solver can find:
- **N=8**: < 1ms
- **N=12**: ~10ms
- **N=15**: ~1 second
- **N=18**: ~2 minutes
- **N=27**: This is the current "Frontier" of human computation. Finding all solutions for $N=27$ took over a year of distributed computing in 2016.

---

## 9. Common Bugs and Edge Cases

1. **Diagonal Indexing Off-by-One**: In the set-based approach, mapping `row - col` to an array index requires adding an offset like `(n-1)` to avoid negative numbers.
2. **Global State Pollution**: If you don't "undo" your changes to the `cols` or `diag` sets correctly during backtracking, future branches will be incorrectly pruned. Always ensure your "Place" and "Remove" logic are perfectly symmetrical.
3. **The N=2/N=3 Trap**: $N=2$ and $N=3$ have **zero** solutions. Your code must return an empty list, not crash.
4. **Deep Recursion**: For $N > 15$ in Python, you will hit the recursion limit. Use an iterative approach with an explicit stack for large $N$.

---

## 10. Social and Historical Context: The "Gauss" Connection

The N-Queens problem isn't just for LeetCode. It was first proposed by Max Bezzel in 1848, but it became famous when the legendary mathematician **Carl Friedrich Gauss** began working on it.
- Gauss initially thought there were only 76 solutions for $N=8$.
- A blind mathematician named Franz Nauck eventually proved there were 92.
This problem has been used for over 150 years as a benchmark for human logic, and now, for silicon performance. It represents the transition from "Manual Calculation" to "Algorithmic Automation."

---

## 11. Production Engineering Considerations

In a real-world system (e.g., a chip-routing engine or an AWS resource scheduler):

### 11.1 Iterative Deepening
Instead of a full DFS, we might use **Iterative Deepening Search (IDS)** if we are looking for the *shallowest* solution in a more complex state space.

### 11.2 The "Fail-Fast" Principle
The moment a constraint is violated, the code returns. This "Fail-Fast" behavior is critical in distributed systems. If a microservice detects an invalid state, it should error out quickly rather than consuming more resources (CPU/Memory).

### 11.3 Memory Alignment
In C/C++ implementations, keeping the `diag` and `cols` arrays aligned to **Cache Line** boundaries (typically 64 bytes) can provide a 5-10% performance boost by reducing the number of memory fetches.

---

## 12. Connections to ML Systems

### 12.1 Constraint Satisfaction in AutoML
(Existing content here builds on this...)
AutoML systems (like Google Vizier) treat hyperparameter tuning as a search problem.
- **Search Space**: All possible combinations of LR, Batch Size, etc.
- **Pruning**: If a configuration starts showing a "Diverging Loss" in the first 5 epochs, the system "Backtracks" (kills the trial) and moves to a different region of the space.
- This is exactly like N-Queens realizing that placing a queen at `(2,3)` makes it impossible to finish the board and immediately jumping to `(2,4)`.

---

## 13. Interview Strategy: The "Senior" Level Playbook

1. **Clarification**: Ask about $N$ limits. If $N$ is small (up to 12), simple backtracking is fine. If $N$ is large, talk about **Heuristics** (like the Min-Conflicts heuristic).
2. **Visual Communication**: Use a small grid to explain `r-c` and `r+c`. It proves you aren't just memorizing code; you understand the 2D arithmetic.
3. **Walkthrough of Failure**: Choose a case that **doesn't** work (e.g., trying to place 2 queens on a 3x3 board) to show how the backtrack triggers. This is actually more informative than showing a successful run.
4. **Mention Optimization**: Even if you don't write the bitmask version, mentioning it as the "ultimate optimization" shows you know where the ceiling of performance is.

---

## 14. Key Takeaways

1. **Backtracking is Trial with Memory**: It isn't just random guessing; it's a systematic exploration that records why it failed.
2. **Constraint Encoding is High Art**: Whether you use Sets, Bitmasks, or 1D arrays, how you "represent" the board determines the speed of the "isSafe" check.
3. **From Queens to Agents**: Every autonomous system you see today—from a self-driving car to a multi-agent coder—is fundamentally solving a sequence of constraint satisfaction problems.

### 11.2 The "Backjumping" Optimization
In standard backtracking, when we hit a failure, we go back to the *immediate previous* level. **Backjumping** is a smarter version that identifies the *source* of the conflict and jumps back multiple levels to the row that actually caused the problem. 
- For example, if row 8 is impossible to fill because of a choice made in row 2, why waste time trying every column in rows 7, 6, 5, 4, and 3? 
- Backjumping uses a "Conflict Set" for each row. If we fail, we jump to the most recent row in the current row's conflict set.

### 11.3 Multi-threaded Search Considerations
When parallelizing N-Queens, we use the **Work Stealing** pattern.
- The search space is divided into sub-tasks (e.g., "All solutions where Queen 0 is at Col 0").
- If one thread finishes its sub-tasks early, it "steals" a branch from another busy thread's stack.
- This ensures balanced load across all CPU cores, which is vital for large $N$ ($N > 20$).

---

## 12. Visualization: The "Life of a Search"

Let's look at the solutions for $N=4$ (the smallest $N$ with a solution).

**The 12 Unique Placements for N=4 (Only 2 are valid):**

```text
Solution 1:          Solution 2:
. Q . .              . . Q .
. . . Q              Q . . .
Q . . .              . . . Q
. . Q .              . Q . .
```

Notice that Solution 2 is just the **Reflection** of Solution 1. This is the symmetry we discussed earlier. If we find Solution 1, we get Solution 2 for free.

---

## 13. Heuristics: The Min-Conflicts Algorithm

For very large $N$ (e.g., $N=1,000,000$), backtracking is impossible. We use **Local Search** heuristics instead.

### 13.1 How Min-Conflicts Works
1. Place all $N$ queens randomly on the board (one per row).
2. Count the total conflicts.
3. While conflicts > 0:
   - Pick a queen that has a conflict.
   - Move it to the column in its row that minimizes the number of conflicts.
4. Repeat.

Surprisingly, this can solve the $1,000,000$-queens problem in a matter of seconds! It doesn't guarantee finding *all* solutions, but it's incredibly fast at finding *one* solution.

---

## 14. Deep Dive: Constraint Satisfaction Problems (CSP)

N-Queens is the textbook definition of a CSP. To formalize it:
- **Variables ($V$):** $\{Q_1, Q_2, \dots, Q_n\}$, where $Q_i$ is the column of the queen in row $i$.
- **Domains ($D$):** $\{1, 2, \dots, n\}$ for each variable.
- **Constraints ($C$):** 
  - $Q_i \neq Q_j$ (Cols must be different)
  - $|Q_i - Q_j| \neq |i - j|$ (Diagonals must be different)

### 14.1 Arc Consistency (AC-3)
In modern solvers, we use **Arc Consistency** to reduce domains. Before we even place a queen in Row 2, we "look ahead" and remove all columns from Row 3's domain that would be attacked by every possible choice in Row 2.
This is called **Forward Checking**. It's the same principle used in **AutoML** (ML track) to prune hyperparameter regions that will never satisfy the accuracy threshold.

---

## 15. Common Senior Engineer Interview Questions

**Q: Can you solve N-Queens iteratively?**
**A:** Yes, by simulating the recursion stack with a list of indices. The state would keep track of `[current_row, current_col_attempt]`. This is often required in low-level systems (like embedded C) where stack space is strictly capped at a few kilobytes.

**Q: How do you optimize for memory when $N$ is massive?**
**A:** Instead of an $N \times N$ board, use an array of size $N$ where `arr[i] = j` means a queen is at `(i, j)`. This reduces space from $O(N^2)$ to $O(N)$. For the "Is Safe" check, use bitsets or the $r-c/r+c$ arrays we discussed.

**Q: How does N-Queens relate to the "Knight's Tour" or "Sudoku"?**
**A:** They are all **Exact Cover** problems. Sudoku is a 2D constraint problem on 9x9 grids. Knight's Tour is a Hamiltonian Path problem on a graph. All can be solved using **Knuth’s Algorithm X (Dancing Links)**, which is the most efficient general-purpose solver for these types of puzzles.

---

## 16. Key Takeaways

1. **Backtracking is BFS with Pruning:** It explores the depth of the tree, but "chops off" branches that it knows are dead ends.
2. **Symmetry is a Free Lunch:** Exploiting rotations and reflections can cut your search time in half (or more).
3. **From Puzzles to Production:** The same logic that solves N-Queens is used in the Linux kernel to schedule tasks, in AWS to pack instances into servers, and in AI Agents to plan a sequence of steps.

---

## 17. N-Queens as a Graph Problem: The Conflict Graph

For the mathematically inclined, N-Queens can be modeled as an **Independent Set** problem on a specific graph called the **Queen's Graph**.
- **Nodes**: Every square $(i, j)$ on the $N \times N$ board is a node. There are $N^2$ nodes.
- **Edges**: Two nodes are connected if they are in the same row, column, or diagonal.
- **Goal**: Find an independent set of size $N$ in this graph. (An independent set is a set of vertices where no two vertices are adjacent).

Finding the maximum independent set is generally an NP-Hard problem, but because the Queen's Graph has a very specific structure, we can use the specialized backtracking algorithms we've discussed. This perspective is useful when connecting N-Queens to **Network Topology** and **Frequency Assignment** in telecommunications.

---

**Originally published at:** [arunbaby.com/dsa/0055-n-queens](https://www.arunbaby.com/dsa/0055-n-queens/)

*If you found this helpful, consider sharing it with others who might benefit.*
