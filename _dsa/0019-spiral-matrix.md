---
title: "Spiral Matrix"
day: 19
related_ml_day: 19
related_speech_day: 19
related_agents_day: 19
collection: dsa
categories:
 - dsa
tags:
 - matrix
 - array
 - simulation
 - iteration
 - traversal
 - medium
subdomain: "Matrix & 2D Array Algorithms"
tech_stack: [Python]
scale: "O(M×N) time, O(1) extra space"
companies: [Google, Meta, Amazon, Microsoft, Apple, Bloomberg]
---

**Master systematic matrix traversal—the same pattern used for tracking experiments, processing logs, and managing state in ML systems.**

## Problem Statement

Given an `m x n` matrix, return all elements of the matrix in **spiral order**.

### Examples

**Example 1:**

``
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]

Visualization:
1 → 2 → 3
 ↓
4 → 5 6
↑ ↓
7 ← 8 ← 9
``

**Example 2:**

``
Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]

Visualization:
1 → 2 → 3 → 4
 ↓
5 → 6 → 7 8
↑ ↓
9 ← 10 ← 11 ← 12
``

**Example 3:**

``
Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
Output: [1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10]
``

### Constraints

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 10`
- `-100 <= matrix[i][j] <= 100`

## Understanding the Problem

This problem is about **controlled iteration through a 2D structure** with shrinking boundaries. The spiral pattern requires:

1. Moving in **four directions** (right → down → left → up).
2. **Shrinking the boundary** after each complete pass.
3. **Stopping** when all elements are visited.

### Why This Problem Matters

This is not just about matrix traversal—it teaches:

- **Systematic state tracking**: Managing boundaries, directions, and iteration progress.
- **Layer-based processing**: Common in image processing, tensor slicing, and multi-level caching.
- **Stateful iteration patterns**: Used in experiment tracking, checkpoint management, and log processing.

### Real-World Analogs

| Domain | Spiral-like Pattern |
|--------|---------------------|
| **Experiment Tracking** | Iterating through hyperparameter grids in a structured order |
| **Data Processing** | Processing nested batches/shards with state |
| **Speech Models** | Hierarchical attention over time-frequency matrices |
| **Image Processing** | Layer-wise convolution, pooling in CNNs |

The key skill: **managing iteration state while respecting dynamic boundaries.**

## Approach 1: Layer-by-Layer Peeling (Optimal)

### Intuition

Think of the matrix as **concentric layers** (like an onion). We peel off the outermost layer in spiral order, then move inward to the next layer, and repeat until all elements are visited.

For each layer, we traverse:
1. **Top row** (left to right)
2. **Right column** (top to bottom, excluding corners already visited)
3. **Bottom row** (right to left, if there's a bottom row)
4. **Left column** (bottom to top, excluding corners already visited)

### Implementation

``python
from typing import List

def spiralOrder(matrix: List[List[int]]) -> List[int]:
 """
 Traverse matrix in spiral order using boundary shrinking.
 
 Time: O(M × N) - visit each element once
 Space: O(1) extra - only output list (required)
 
 Strategy:
 - Maintain four boundaries: top, bottom, left, right
 - Traverse each side of the current layer
 - Shrink boundaries after each side
 - Stop when boundaries cross
 """
 if not matrix or not matrix[0]:
 return []
 
 result = []
 m, n = len(matrix), len(matrix[0])
 
 # Initialize boundaries
 top, bottom = 0, m - 1
 left, right = 0, n - 1
 
 while top <= bottom and left <= right:
 # Traverse top row (left to right)
 for col in range(left, right + 1):
 result.append(matrix[top][col])
 top += 1 # Shrink top boundary
 
 # Traverse right column (top to bottom)
 for row in range(top, bottom + 1):
 result.append(matrix[row][right])
 right -= 1 # Shrink right boundary
 
 # Traverse bottom row (right to left), if it exists
 if top <= bottom:
 for col in range(right, left - 1, -1):
 result.append(matrix[bottom][col])
 bottom -= 1 # Shrink bottom boundary
 
 # Traverse left column (bottom to top), if it exists
 if left <= right:
 for row in range(bottom, top - 1, -1):
 result.append(matrix[row][left])
 left += 1 # Shrink left boundary
 
 return result
``

### Walkthrough Example

**Input:** `matrix = [[1,2,3],[4,5,6],[7,8,9]]`

``
Initial state:
top=0, bottom=2, left=0, right=2

Layer 1:
- Top row (row 0, col 0→2): [1, 2, 3]
 top → 1
- Right column (col 2, row 1→2): [6, 9]
 right → 1
- Bottom row (row 2, col 1→0): [8, 7]
 bottom → 1
- Left column (col 0, row 1→1): [4]
 left → 1

Layer 2:
- Top row (row 1, col 1→1): [5]
 top → 2
 (top > bottom, stop)

Result: [1, 2, 3, 6, 9, 8, 7, 4, 5]
``

### Key Details

1. **Boundary checks before bottom and left traversals:**
 - After traversing top and right, boundaries may have crossed.
 - We need `if top <= bottom` before traversing bottom row.
 - We need `if left <= right` before traversing left column.

2. **Edge cases handled:**
 - Single row: Only top traversal executes.
 - Single column: Top and right traversals execute, left/bottom skipped.
 - 1×1 matrix: Only top traversal for one element.

## Approach 2: Direction Vectors (Alternative)

### Intuition

Use **direction vectors** and track visited cells explicitly. Change direction when hitting a boundary or visited cell.

``python
def spiralOrder_directional(matrix: List[List[int]]) -> List[int]:
 """
 Use direction vectors: right, down, left, up.
 
 Time: O(M × N)
 Space: O(M × N) for visited set
 """
 if not matrix or not matrix[0]:
 return []
 
 m, n = len(matrix), len(matrix[0])
 result = []
 visited = set()
 
 # Direction vectors: right, down, left, up
 directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
 dir_idx = 0
 
 row, col = 0, 0
 
 for _ in range(m * n):
 result.append(matrix[row][col])
 visited.add((row, col))
 
 # Try to continue in current direction
 dr, dc = directions[dir_idx]
 next_row, next_col = row + dr, col + dc
 
 # Check if we need to change direction
 if (next_row < 0 or next_row >= m or 
 next_col < 0 or next_col >= n or 
 (next_row, next_col) in visited):
 # Change direction (turn right)
 dir_idx = (dir_idx + 1) % 4
 dr, dc = directions[dir_idx]
 next_row, next_col = row + dr, col + dc
 
 row, col = next_row, next_col
 
 return result
``

### Comparison

| Approach | Time | Extra Space | Clarity | Edge Case Handling |
|----------|------|-------------|---------|-------------------|
| Layer-by-layer | O(M×N) | O(1) | High | Explicit checks |
| Direction vectors | O(M×N) | O(M×N) | Medium | Implicit via visited set |

**Recommendation:** Use **layer-by-layer** for interviews (O(1) space, cleaner logic).

## Implementation: Tests and Edge Cases

``python
def test_spiral_order():
 """Comprehensive test cases for spiral matrix."""
 tests = [
 # (matrix, expected)
 ([[1,2,3],[4,5,6],[7,8,9]], [1,2,3,6,9,8,7,4,5]),
 ([[1,2,3,4],[5,6,7,8],[9,10,11,12]], [1,2,3,4,8,12,11,10,9,5,6,7]),
 ([[1]], [1]),
 ([[1,2,3]], [1,2,3]),
 ([[1],[2],[3]], [1,2,3]),
 ([[1,2],[3,4]], [1,2,4,3]),
 ([[1,2,3],[4,5,6]], [1,2,3,6,5,4]),
 ]
 
 for i, (matrix, expected) in enumerate(tests, 1):
 result = spiralOrder(matrix)
 assert result == expected, f"Test {i} failed: got {result}, expected {expected}"
 
 print("All spiral order tests passed.")


if __name__ == "__main__":
 test_spiral_order()
``

## Complexity Analysis

Let:
- \(M\) = number of rows
- \(N\) = number of columns

### Time Complexity

- We visit each cell exactly **once**.
- Total cells: \(M \times N\).
- Time: \(\boxed{O(M \times N)}\).

### Space Complexity

- **Layer-by-layer approach:**
 - Only a few integer variables for boundaries.
 - Output list: \(O(M \times N)\) (required, not counted as extra).
 - Extra space: \(\boxed{O(1)}\).

- **Direction vectors approach:**
 - Visited set: \(O(M \times N)\).
 - Extra space: \(\boxed{O(M \times N)}\).

## Production Considerations

### 1. Streaming & Large Matrices

For very large matrices that don't fit in memory:

- Process in **chunks** (tiles).
- Track state (current layer, position) across chunks.
- This is common in:
 - Image processing pipelines (satellite imagery),
 - Tensor sharding in distributed training,
 - Log file processing (nested batches).

### 2. Generalization: Other Traversal Patterns

Spiral traversal is one of many structured iteration patterns:

| Pattern | Use Case |
|---------|----------|
| Row-major | Standard tensor layout (C-style) |
| Column-major | Fortran-style, some numerical libraries |
| Diagonal | DP on 2D grids, sequence alignment |
| Spiral | Systematic exploration, experiment grids |
| Block-wise | Tiled matrix multiplication, cache optimization |

### 3. Stateful Iteration in Experiment Tracking

In ML experiment tracking systems (like MLflow, Weights & Biases), you often iterate through:

- Hyperparameter grids,
- Checkpoints across training runs,
- Multi-dimensional logs (time × metric × run).

The **spiral-like pattern** of systematic, boundary-aware iteration is directly applicable:

``python
# Pseudo-code: Iterate through experiment grid
for layer in range(num_layers):
 for config in get_configs_at_layer(layer):
 experiment_id = launch_experiment(config)
 track_state(experiment_id, layer, config)
``

### 4. Error Handling & Validation

In production, always:

- Validate matrix dimensions (non-empty, rectangular).
- Handle edge cases (1×1, single row/column).
- Log traversal state for debugging:
 - Current boundaries,
 - Number of elements processed,
 - Expected vs actual element count.

## Connections to ML Systems

The thematic link foris **systematic iteration and state tracking**, which is central to:

### 1. Experiment Tracking Systems

**Experiment Tracking Systems** (like MLflow, Weights & Biases, Neptune) organize runs in a multi-dimensional space:

- Hyperparameters × metrics × time × model versions.
- Systematic traversal (like spiral order) ensures:
 - No duplicate runs,
 - Efficient exploration of the search space,
 - Clear state persistence across crashes/restarts.

### 2. Speech Experiment Management

In speech research:

- You iterate over:
 - Model architectures,
 - Training data configurations,
 - Augmentation policies,
 - Inference hyperparameters (beam width, LM weight).
- **Stateful iteration** helps:
 - Track which configurations have been tried,
 - Resume experiments from checkpoints,
 - Visualize progress in multi-dimensional grids.

### 3. Checkpoint Management

During training, you save checkpoints periodically:

- Organized by: epoch × step × metric.
- When resuming, you need to:
 - Find the latest valid checkpoint,
 - Restore optimizer state,
 - Continue from the correct training step.

This is a form of **stateful iteration** through nested structures, akin to spiral traversal through layers.

## Interview Strategy

### How to Approach This Problem

**1. Clarify (1–2 minutes)**

- Can the matrix be empty?
- Is it always rectangular (all rows same length)?
- Do we need to handle non-integer values?

**2. Explain the Intuition (2–3 minutes)**

> "I'll treat the matrix as concentric layers. For each layer, I'll traverse the four sides (top, right, bottom, left) and shrink the boundaries inward. I'll stop when the boundaries cross."

**3. Discuss Edge Cases (1–2 minutes)**

- Single row or column.
- 1×1 matrix.
- Rectangular matrices (m ≠ n).

**4. Implement (5–10 minutes)**

- Use four boundary variables: `top`, `bottom`, `left`, `right`.
- After each side traversal, update the boundary.
- Add checks before bottom and left traversals.

**5. Test & Analyze (3–5 minutes)**

- Walk through example (3×3 matrix).
- Mention time: O(M×N), space: O(1) extra.
- Discuss alternative (direction vectors) and trade-offs.

### Common Pitfalls

1. **Forgetting boundary checks:**
 - After traversing top and right, the boundary may have collapsed.
 - Always check `if top <= bottom` before bottom row.
 - Always check `if left <= right` before left column.

2. **Off-by-one errors:**
 - Ensure correct range boundaries (`range(left, right + 1)` vs `range(left, right)`).

3. **Not handling single row/column:**
 - Test explicitly with `[[1,2,3]]` and `[[1],[2],[3]]`.

### Follow-up Questions

1. **Spiral Matrix II (generate spiral):**
 - Given `n`, generate an `n × n` spiral matrix filled with 1 to n².
 - Use the same boundary-shrinking logic, but write instead of read.

2. **Diagonal traversal:**
 - Traverse matrix diagonally (e.g., for DP problems).

3. **3D spiral:**
 - Extend to 3D matrices (tensors).

## Additional Practice & Variants

### 1. Spiral Matrix II (LeetCode 59)

**Problem:** Given `n`, generate an `n × n` matrix filled with elements from 1 to n² in spiral order.

``python
def generateMatrix(n: int) -> List[List[int]]:
 """Generate n×n matrix in spiral order."""
 matrix = [[0] * n for _ in range(n)]
 top, bottom, left, right = 0, n - 1, 0, n - 1
 num = 1
 
 while top <= bottom and left <= right:
 for col in range(left, right + 1):
 matrix[top][col] = num
 num += 1
 top += 1
 
 for row in range(top, bottom + 1):
 matrix[row][right] = num
 num += 1
 right -= 1
 
 if top <= bottom:
 for col in range(right, left - 1, -1):
 matrix[bottom][col] = num
 num += 1
 bottom -= 1
 
 if left <= right:
 for row in range(bottom, top - 1, -1):
 matrix[row][left] = num
 num += 1
 left += 1
 
 return matrix
``

### 2. Print Matrix in Diagonal Order (LeetCode 498)

**Problem:** Given an `m × n` matrix, return all elements in diagonal order.

``python
def findDiagonalOrder(matrix: List[List[int]]) -> List[int]:
 """
 Traverse diagonals alternating up-right and down-left.
 
 Key: Group by diagonal index (r + c).
 Even diagonals go bottom-left to top-right.
 Odd diagonals go top-right to bottom-left.
 """
 if not matrix or not matrix[0]:
 return []
 
 m, n = len(matrix), len(matrix[0])
 result = []
 
 # Group cells by diagonal (r + c)
 diagonals = {}
 for r in range(m):
 for c in range(n):
 diag_idx = r + c
 if diag_idx not in diagonals:
 diagonals[diag_idx] = []
 diagonals[diag_idx].append(matrix[r][c])
 
 # Traverse diagonals in order, alternating direction
 for diag_idx in range(m + n - 1):
 if diag_idx % 2 == 0:
 # Even: reverse (go up-right)
 result.extend(reversed(diagonals[diag_idx]))
 else:
 # Odd: normal (go down-left)
 result.extend(diagonals[diag_idx])
 
 return result
``

### 3. Spiral Matrix III (LeetCode 885)

**Problem:** Start at `(r0, c0)` and walk in a spiral. Return coordinates in order visited (including out-of-bounds steps).

This extends the spiral pattern to handle:

- Starting at an arbitrary position,
- Walking beyond matrix boundaries,
- Recording only valid coordinates.

## Key Takeaways

✅ **Spiral traversal is systematic layer-by-layer peeling** with boundary shrinking.

✅ **Time O(M×N), Space O(1) extra**—optimal for this problem.

✅ **Boundary checks are critical** after each side to avoid over-iteration.

✅ **Stateful iteration patterns** like this map directly to experiment tracking, checkpoint management, and hierarchical data processing in ML systems.

✅ **Edge cases** (single row/column, 1×1 matrix) are easy to miss—test explicitly.

✅ **Direction vectors approach** is an alternative but uses O(M×N) extra space.

✅ **Generalization:** This pattern extends to 3D tensors, nested experiment grids, and structured log processing.

### Connection to Thematic Link: Systematic Iteration and State Tracking

All three topics share the core pattern of **systematic, stateful iteration**:

**DSA (Spiral Matrix):**
- Traverse 2D structure layer-by-layer with shrinking boundaries.
- Track state (current boundaries, position).

**ML System Design (Experiment Tracking Systems):**
- Systematically iterate through hyperparameter grids.
- Track experiment state (configs tried, results, checkpoints).

**Speech Tech (Speech Experiment Management):**
- Organize speech model experiments across data/architecture/hyperparameter dimensions.
- Resume from checkpoints, track multi-run state.

The **unifying principle**: manage iteration progress through complex, nested structures while maintaining clear, recoverable state.

---

**Originally published at:** [arunbaby.com/dsa/0019-spiral-matrix](https://www.arunbaby.com/dsa/0019-spiral-matrix/)

*If you found this helpful, consider sharing it with others who might benefit.*






