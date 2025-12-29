---
title: "Sudoku Solver"
day: 59
related_ml_day: 59
related_speech_day: 59
related_agents_day: 59
collection: dsa
categories:
  - dsa
tags:
  - backtracking
  - recursion
  - constraint-satisfaction
  - hard
  - search-algorithms
  - bitmasking
difficulty: Hard
subdomain: "Backtracking"
tech_stack: Python
scale: "9x9 grid, solving in <10ms with bitmasking optimizations"
companies: [Google, Apple, Microsoft, Amazon, Meta, Airbnb]
---

**"Sudoku Solver is the quintessential backtracking problem—it represents the transition from simple recursion to a multi-constraint search problem where every choice prunes a massive branch of the state space."**

## 1. Introduction: The Complexity of the Grid

Sudoku is more than just a pastime in the Sunday newspaper. In computer science, it is a canonical example of a **Constraint Satisfaction Problem (CSP)**. 

At a glance, a 9x9 grid seems small. However, if you were to try filling an empty board with every possible digit combination, you would be dealing with `9^{81}` possibilities—a number larger than the estimated number of atoms in the observable universe. 

Solving Sudoku efficiently is about **Pruned Search**. It is about making a move, checking if it satisfies a set of local and global constraints, and immediately "backtracking" if it leads to a dead end. We explore the algorithms that power modern solvers, from basic recursion to bitmasking and the legendary "Dancing Links."

---

## 2. The Problem Statement

Write a program to solve a Sudoku puzzle by filling the empty cells (represented by `.`).
A sudoku solution must satisfy all of the following rules:
1. Each of the digits `1-9` must occur exactly once in each row.
2. Each of the digits `1-9` must occur exactly once in each column.
3. Each of the digits `1-9` must occur exactly once in each of the 9 `3x3` sub-boxes of the grid.

**Constraints:**
- The input board is guaranteed to have exactly one solution.
- The board size is fixed at 9x9.

---

## 3. Thematic Link: Constraint Satisfaction and Search

We focus on **Constraint Satisfaction and Search**:
- **DSA**: We use backtracking to solve a rule-based grid.
- **ML System Design**: AutoML systems search through the space of hyperparameters while satisfying hardware and accuracy constraints.
- **Speech Tech**: Neural Architecture Search (NAS) for speech is a search for the best Conformer topology under latency constraints.
- **Agents**: Benchmarking agents involves measuring how efficiently an agent searches through a task space to find a valid solution.

---

## 4. Approach 1: Basic Backtracking (The Foundation)

Backtracking is a depth-first search (DFS) through the state space.

### 4.1 The Recursive Algorithm
1. **Find the next empty cell**: Ideally, we pick the first `.` we encounter.
2. **Iterate through digits 1-9**:
  - For each digit, check if it is **Safe** to place.
  - If safe, place it and move to the next empty cell (recursive call).
3. **Check the result**:
  - If the recursion returns `True`, we found the solution.
  - If it returns `False`, **Backtrack**: reset the cell to `.` and try the next digit.
4. **Base Case**: If no more empty cells exist, the puzzle is solved.

### 4.2 Code implementation

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        self.solve(board)

    def solve(self, board):
        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    for digit in "123456789":
                        if self.is_safe(board, r, c, digit):
                            board[r][c] = digit
                            if self.solve(board):
                                return True
                            board[r][c] = "." # Backtrack
                    return False
        return True

    def is_safe(self, board, r, c, digit):
        for i in range(9):
            # Row check
            if board[r][i] == digit: return False
            # Column check
            if board[i][c] == digit: return False
            # 3x3 Box check
            if board[3*(r//3) + i//3][3*(c//3) + i%3] == digit: return False
        return True
```

---

## 5. Approach 2: Optimized Backtracking with Bitmasks

The `is_safe` function in the previous approach is called hundreds of thousands of times. We can optimize this to O(1) by using **Bitmasks**.

### 5.1 The Logic
We maintain three arrays of integers (bitmasks):
- `rows[9]`: `rows[i]` is a bitmask where the `k`-th bit is 1 if digit `k` is present in row `i`.
- `cols[9]`: Same for columns.
- `boxes[9]`: Same for the 9 sub-boxes.

### 5.2 Optimized Code

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        rows = [0] * 9
        cols = [0] * 9
        boxes = [0] * 9
        to_fill = []

        # 1. Map initial state
        for r in range(9):
            for c in range(9):
                if board[r][c] != ".":
                    digit = int(board[r][c])
                    mask = 1 << digit
                    rows[r] |= mask
                    cols[c] |= mask
                    boxes[(r//3)*3 + (c//3)] |= mask
                else:
                    to_fill.append((r, c))

        def backtrack(idx):
            if idx == len(to_fill):
                return True
            
            r, c = to_fill[idx]
            box_idx = (r//3)*3 + (c//3)
            
            used = rows[r] | cols[c] | boxes[box_idx]
            
            for digit in range(1, 10):
                if not (used & (1 << digit)):
                    mask = 1 << digit
                    board[r][c] = str(digit)
                    rows[r] |= mask
                    cols[c] |= mask
                    boxes[box_idx] |= mask
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Backtrack (unset bits)
                    rows[r] ^= mask
                    cols[c] ^= mask
                    boxes[box_idx] ^= mask
            return False

        backtrack(0)
```

---

## 6. Implementation Deep Dive: Coordinate Mapping

Scaling a Sudoku solver often fails due to incorrect box index mapping. 
- A 9x9 grid has 9 boxes, indexed 0-8 starting from the top-left.
- To map `(r, c)` to `box_id`:
  `box_id = (r // 3) * 3 + (c // 3)`
- **Advanced Tip**: In a `N \times N` Sudoku where `N` is a perfect square, the formula generalizes to:
  `sub_size = int(sqrt(N))`
  `box_id = (r // sub_size) * sub_size + (c // sub_size)`

---

## 7. Theoretical Maximum: Knuth's Dancing Links (DLX)

If you are asked about the "Fastest possible Sudoku solver", the answer is **Algorithm X** using **Dancing Links**.

### 7.1 Exact Cover
Sudoku can be modeled as an **Exact Cover Problem**. We have 324 constraints and 729 choices. The goal is to select 81 choices such that every constraint is satisfied exactly once.

### 7.2 The DLL Magic
Donald Knuth's "Dancing Links" uses a circular doubly-linked list where a node can be removed and "re-inserted" with zero memory allocation. This is the ultimate optimization for the "Backtrack" step.

---

## 8. Complexity Analysis

| Metric | Complexity | Rationale |
| :--- | :--- | :--- |
| **Time** | O(9^k) | `k` is the number of empty cells. |
| **Space** | O(k) | The recursion stack goes as deep as the empty cells. |

For a 9x9 board, even a naive solver rarely takes more than 100ms. With bitmasking, it drops to < 5ms.

---

## 9. Common Pitfalls and Memory Management

1. **Bitmask Indexing**: Start your bits from 1 to match Sudoku digits (1-9).
2. **The "Unique Solution" Trap**: Your code should be robust enough to return `False` if no solution is found.
3. **Mutating the Board**: In Python, when you modify `board[r][c]`, you are changing the actual list object. Always ensure you reset it.

---

## 10. Connections to Other Topics

### 10.1 Connection to ML (Neural Sudoku Solvers)
Recent research into **SatNet** attempts to teach neural networks to solve Sudoku by embedding a "Constraint Optimization" layer into the architecture.

### 10.2 Connection to AI Agents (Benchmarking)
A Sudoku solver is essentially a "Single-Purpose Agent." We measure its "Trajectory Efficiency"—how few guesses did it make? A superior agent picks cells with the **Fewest Remaining Possibilities (Heuristic)**.

---

## 11. Interview Strategy: The "Constraint First" Mindset

1. **Define Safety**: Explain the three rules clearly before writing code.
2. **Backtracking Visualization**: Draw a small grid and show how trying a digit forces decisions.
3. **Mention Optimizations**: Mention bitmasking and "Minimum Remaining Values" (MRV) heuristic.
4. **Edge Cases**: Mention what happens if the input board is already invalid.

---

## 12. Key Takeaways

1. **Constraint Propagation is Power**: The more constraints you check upfront, the smaller your search tree.
2. **Backtracking is DFS on State**: Every call represents a state in the puzzle's life.
3. **Bitmasking for Speed**: O(1) constraint checking is the difference between a toy and a production-grade solver.

---

**Originally published at:** [arunbaby.com/dsa/0059-sudoku-solver](https://www.arunbaby.com/dsa/0059-sudoku-solver/)

*If you found this helpful, consider sharing it with others who might benefit.*
