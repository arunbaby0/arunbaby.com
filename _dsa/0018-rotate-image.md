---
title: "Rotate Image"
day: 18
related_ml_day: 18
related_speech_day: 18
related_agents_day: 18
collection: dsa
categories:
 - dsa
tags:
 - matrix
 - array
 - in-place
 - 2d-transform
 - simulation
 - medium
subdomain: "Matrix & 2D Array Algorithms"
tech_stack: [Python]
scale: "O(N²) time, O(1) extra space"
companies: [Google, Meta, Amazon, Microsoft, Apple]
---

**Master in-place matrix rotation—the same 2D transformation pattern that powers image and spectrogram augmentations in modern ML systems.**

## Problem Statement

You are given an `n x n` 2D matrix representing an image. Rotate the image by **90 degrees clockwise**, **in-place**.

In other words:

- Input: `matrix[i][j]` is the pixel at row `i`, column `j`.
- Output: after rotation, `matrix[i][j]` should hold the pixel that was at its new corresponding position.
- You **must not allocate another `n x n` matrix**; modify `matrix` itself.

### Examples

**Example 1**

``python
matrix = [
 [1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]
]
``

After rotation:

``python
[
 [7, 4, 1],
 [8, 5, 2],
 [9, 6, 3]
]
``

**Example 2**

``python
matrix = [
 [ 5, 1, 9, 11],
 [ 2, 4, 8, 10],
 [13, 3, 6, 7],
 [15, 14, 12, 16]
]
``

After rotation:

``python
[
 [15, 13, 2, 5],
 [14, 3, 4, 1],
 [12, 6, 8, 9],
 [16, 7, 10, 11]
]
``

### Constraints

- `n == len(matrix) == len(matrix[0])` (the matrix is square)
- `1 <= n <= 1000` (online judges often use smaller limits, but your algorithm should scale)
- `-10⁴ <= matrix[i][j] <= 10⁴`

## Understanding the Problem

Imagine the matrix as a grid of coordinates \((r, c)\) where:

- `r` is the row index (`0` at the top),
- `c` is the column index (`0` at the left).

For a clockwise 90° rotation:

\[
(r, c) \longrightarrow (c,\ n - 1 - r)
\]

Visually for a 3×3 matrix:

``text
Original indices: After 90° CW:

(0,0) (0,1) (0,2) (2,0) (1,0) (0,0)
(1,0) (1,1) (1,2) → (2,1) (1,1) (0,1)
(2,0) (2,1) (2,2) (2,2) (1,2) (0,2)
``

Rotating is just **moving values to new coordinates**. The difficulty comes from:

- Doing it **in-place** (no extra `n x n` buffer),
- Avoiding accidental overwrites,
- Handling all elements exactly once.

### Why This Problem Matters

Beyond the coding challenge, this pattern appears all over:

- **Vision data augmentation:** rotating input images by multiples of 90°.
- **Tensor layout transforms:** e.g., changing from `[H, W, C]` to `[W, H, C]` or performing block rotations.
- **Spectrogram manipulations:** operating on time–frequency matrices in speech pipelines.
- **GPU kernels:** where you must transform indices without extra memory.

Understanding how to rotate in-place is basically learning to **think in 2D index space and reason about what “in-place” means for a structured object**.

## Approach 1: Extra Matrix (Not In-Place, For Intuition)

### Intuition

The simplest way to rotate is to allocate a new matrix and write:

``python
rotated[c][n - 1 - r] = matrix[r][c]
``

Then copy `rotated` back into `matrix`. This is not allowed by the problem’s in-place constraint but is useful to:

- Verify your understanding of the index mapping,
- Generate expected outputs for testing your in-place implementation.

### Implementation

``python
from typing import List

def rotate_with_copy(matrix: List[List[int]]) -> None:
 \"\"\"Rotate 90° clockwise using an extra matrix (NOT in-place).

 Time: O(n^2)
 Space: O(n^2) extra
 \"\"\"
 n = len(matrix)
 rotated = [[0] * n for _ in range(n)]

 for r in range(n):
 for c in range(n):
 rotated[c][n - 1 - r] = matrix[r][c]

 # Copy back into original matrix
 for r in range(n):
 for c in range(n):
 matrix[r][c] = rotated[r][c]
``

### Why We Need More

This violates the “no extra matrix” requirement:

- Extra space is O(n²) instead of O(1).
- In a real system working with huge images or feature maps, duplicating entire tensors can be too expensive.

We want an algorithm that:

- Has the **same time complexity**,
- But uses **only constant extra space** (a few scalars/temporaries).

## Approach 2: Transpose + Reverse (Elegant In-Place)

### Key Idea

A 90° clockwise rotation can be decomposed into two simpler operations:

1. **Transpose the matrix** (swap `matrix[r][c]` with `matrix[c][r]`).
2. **Reverse each row**.

For example, starting from:

``text
1 2 3
4 5 6
7 8 9
``

Transpose across the main diagonal:

``text
1 4 7
2 5 8
3 6 9
``

Then reverse each row:

``text
7 4 1
8 5 2
9 6 3
``

Which is the desired rotated result.

### Why This Works (Coordinate Proof)

Transpose does:

\[
(r, c) \rightarrow (c, r)
\]

Reversing each row (index `c` along the row) does:

\[
(c, r) \rightarrow (c,\ n - 1 - r)
\]

Composing them:

\[
(r, c) \xrightarrow{\text{transpose}} (c, r) \xrightarrow{\text{reverse rows}} (c, n-1-r)
\]

Which matches the rotation formula.

### Implementation

``python
from typing import List

def rotate(matrix: List[List[int]]) -> None:
 \"\"\"Rotate the matrix 90 degrees clockwise in-place.

 Uses transpose + row reversal.

 Time: O(n^2)
 Space: O(1) extra
 \"\"\"
 n = len(matrix)

 # 1. Transpose in-place
 for r in range(n):
 # only swap above the diagonal to avoid double-swapping
 for c in range(r + 1, n):
 matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]

 # 2. Reverse each row
 for r in range(n):
 matrix[r].reverse()
``

### Step-by-Step Example

For `matrix = [[1,2,3],[4,5,6],[7,8,9]]`:

1. **Transpose**
 - swap `(0,1)` with `(1,0)` → `[[1,4,3],[2,5,6],[7,8,9]]`
 - swap `(0,2)` with `(2,0)` → `[[1,4,7],[2,5,6],[3,8,9]]`
 - swap `(1,2)` with `(2,1)` → `[[1,4,7],[2,5,8],[3,6,9]]`

2. **Reverse each row**
 - `[1,4,7]` → `[7,4,1]`
 - `[2,5,8]` → `[8,5,2]`
 - `[3,6,9]` → `[9,6,3]`

Result:

``python
[
 [7,4,1],
 [8,5,2],
 [9,6,3]
]
``

### Edge Cases

1. **n = 1**
 - Matrix is `[[x]]`; rotation does nothing. Our code handles this naturally.
2. **n = 2**
 - Example:
 ``python
 [[1,2],
 [3,4]]
 ``
 After transpose: `[[1,3],[2,4]]`; after row reverse: `[[3,1],[4,2]]`.
3. **Non-square matrix**
 - The problem definition assumes square; if you needed to support `m x n` matrices, transpose+reverse alone wouldn’t be enough—this is a different problem (rotate image is defined for square).

## Approach 3: Layer-by-Layer 4-Way Swaps

### Intuition

Instead of doing global operations (transpose & reverse), we can rotate the matrix in-place by **processing one “ring” (layer) at a time**.

For a 4×4 matrix:

``text
Layer 0 (outer ring): indices where min(r, c) = 0
Layer 1 (inner ring): indices where min(r, c) = 1
``

Within each layer, we perform 4-way swaps:

``text
top → right
left → top
bottom → left
right → bottom
``

More concretely, for a cell at `(row, col)`:

- Its right partner is `(col, n-1-row)`,
- Its bottom partner is `(n-1-row, n-1-col)`,
- Its left partner is `(n-1-col, row)`.

We can pick the “top” of each quadruple, save it, then rotate the other 3 values, and finally put the saved top into the right position.

### Implementation

``python
from typing import List

def rotate_layers(matrix: List[List[int]]) -> None:
 \"\"\"Rotate matrix 90° clockwise in-place using layer-by-layer 4-way swaps.

 Time: O(n^2)
 Space: O(1)
 \"\"\"\n n = len(matrix)
 # Process layers from outermost to innermost
 for layer in range(n // 2):
 first = layer
 last = n - 1 - layer

 for i in range(first, last):
 offset = i - first

 # Save top
 top = matrix[first][i]

 # left -> top
 matrix[first][i] = matrix[last - offset][first]

 # bottom -> left
 matrix[last - offset][first] = matrix[last][last - offset]

 # right -> bottom
 matrix[last][last - offset] = matrix[i][last]

 # top -> right
 matrix[i][last] = top
``

This approach is closer to how you might implement a low-level kernel or a special-case transform on a hardware accelerator.

### Comparing Approaches

| Approach | Time | Space | Notes |
|--------------------------|-------|-------|--------------------------------------------|
| Extra matrix | O(n²) | O(n²) | Easiest to understand, not in-place |
| Transpose + reverse | O(n²) | O(1) | Clean, idiomatic, recommended in interviews|
| Layer-by-layer 4-way | O(n²) | O(1) | More index-heavy but very instructive |

## Implementation: Utilities and Tests

``python
from typing import List

def rotate_in_place(matrix: List[List[int]]) -> None:
 \"\"\"Wrapper for the chosen production implementation."""
 # Choose one: rotate (transpose+reverse) or rotate_layers
 rotate(matrix)


def test_rotate():
 tests = [
 (
 [[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]],
 [[7, 4, 1],
 [8, 5, 2],
 [9, 6, 3]],
 ),
 (
 [[5, 1, 9, 11],
 [2, 4, 8, 10],
 [13, 3, 6, 7],
 [15, 14, 12, 16]],
 [[15, 13, 2, 5],
 [14, 3, 4, 1],
 [12, 6, 8, 9],
 [16, 7, 10, 11]],
 ),
 (
 [[1]],
 [[1]],
 ),
 (
 [[1, 2],
 [3, 4]],
 [[3, 1],
 [4, 2]],
 ),
 ]

 for i, (matrix, expected) in enumerate(tests, 1):
 rotate_in_place(matrix)
 assert matrix == expected, f\"Test {i} failed: {matrix} != {expected}\"

 print(\"All rotate tests passed.\")


if __name__ == \"__main__\":
 test_rotate()
``

You can strengthen the tests by adding:

- Random matrices and comparing against the `rotate_with_copy` implementation.
- Larger sizes (e.g., `10x10`, `50x50`) to catch off-by-one errors.

## Complexity Analysis

Let \(n\) be the dimension of the square matrix.

### Time Complexity

- Every approach touches each element a constant number of times.
- Transpose: about \(n(n-1)/2\) swaps.
- Reverse rows: \(n \times (n/2)\) swaps.
- Layer-based: each element is moved exactly once within a 4-cycle.
- So total time is:

\[
T(n) = \Theta(n^2)
\]

### Space Complexity

- We are only using a constant number of scalar temporaries.
- No extra `n x n` matrix is allocated in the in-place solutions.
- Therefore:

\[
S(n) = O(1)\ \text{(extra space)}
\]

## Production Considerations

Even if you never write your own rotation kernel in a production codebase, the
ideas here show up in many ML systems:

### 1. Image Data Augmentation

- Many training pipelines (e.g., for vision models) apply random rotations:
 - 90°, 180°, 270° (cheap, can be implemented as index remaps),
 - Arbitrary-angle rotations (involving interpolation).
- Libraries like torchvision, PIL, OpenCV implement these transforms, but inside
 they rely on exactly this kind of index mapping or on more advanced geometric
 transforms.
- In performance-critical settings (e.g., training on millions of high-res
 images), understanding how to do these operations with minimal copies and
 cache-friendly memory access matters.

### 2. Tensor Layout Transforms

Frameworks and kernels often need to transform tensor layouts, for example:

- `[batch, height, width, channels]` (NHWC) ↔ `[batch, channels, height, width]` (NCHW).
- Blocking and tiling for better cache and vectorization.

These are not literal rotations, but they are **permutation of axes and indices**:

- You still have to compute a mapping from source `(i, j, k, ...)` to destination `(i', j', k', ...)`.
- Thinking clearly about index math (like we did for rotation) is key to avoiding
 subtle off-by-one and alignment bugs.

### 3. Spectrogram Manipulation in Speech

ASR and other speech models often operate on time–frequency matrices:

- Time masking and frequency masking (SpecAugment),
- Time warping,
- Per-frequency scaling or normalization.

These are 2D operations on matrices with the same flavor as rotation:

- Masking a frequency band is just zeroing out rows in a given index range.
- Time warping is a more complex mapping from input time indices to output time indices.

If you are comfortable with simple 2D transforms like rotation, it’s much easier
to read and design these spectrogram-level augmentations.

## Interview Strategy

### How to Present Your Solution

When walking through this problem in an interview:

1. **Clarify constraints**
 - Confirm the matrix is square.
 - Confirm that the rotation is exactly 90° clockwise (not arbitrary angle).
 - Confirm that in-place is required (no extra `n x n` matrix).

2. **Start with the naive copy-based approach**
 - Show you understand the mapping `(r, c) → (c, n-1-r)`.
 - Explicitly state the extra-space cost and why it violates the requirement.

3. **Propose the in-place strategy**
 - Option 1: transpose + reverse rows.
 - Option 2: layer-by-layer 4-way swap.
 - Explain why you’re choosing one (simplicity vs demonstration of pointer/index mastery).

4. **Code with care**
 - Emphasize avoiding double-swaps in transpose (`c` starts at `r+1`).
 - Use descriptive variable names (`first`, `last`, `offset`) in layer-based solution.

5. **Discuss complexity & edge cases**
 - Time O(n²), space O(1).
 - Mention `n=1`, `n=2`, and why they work without special handling.

6. **Connect to real systems**
 - Briefly mention that the same patterns are used in image augmentation and tensor layout transforms.

### Common Pitfalls to Avoid

- **Wrong index math** in layer-based approach—especially mixing up `row` vs `col`.
- **Double-swapping** during transpose (if you loop over all `(r, c)` instead of half the matrix).
- **Allocating extra matrices** and ignoring the in-place requirement.

## Additional Practice & Variants

To really lock in this pattern, try the following related problems:

1. **Rotate 90° counter-clockwise**
 - Derive the mapping:
 \[
 (r, c) \rightarrow (n - 1 - c,\ r)
 \]
 - Implement in-place using transpose + column reversal, or adjust the layer-based swaps.

2. **Rotate 180° in-place**
 - Two 90° rotations, or directly:
 \[
 (r, c) \rightarrow (n - 1 - r,\ n - 1 - c)
 \]
 - Implement a single pass that swaps `(r, c)` with `(n-1-r, n-1-c)` for half the matrix.

3. **Rotate arbitrary k times**
 - For k (possibly negative), compute `k_mod = ((k % 4) + 4) % 4`.
 - Apply `rotate` `k_mod` times (0 = no-op, 1 = 90°, 2 = 180°, 3 = 270°).

4. **Rotate sub-matrices**
 - Given a large matrix and many queries `(r1, c1, r2, c2)`, rotate only the sub-square.
 - Apply the same logic but offset indices by `(r1, c1)`.

5. **In-place transpose without using a temporary**
 - Practice writing a standalone in-place transpose for a square matrix.
 - This appears directly in many ML kernels and is a good building block for more complex transforms.

These variants help you move from “I solved one LeetCode problem” to “I can
systematically reason about in-place 2D array transforms,” which is exactly the
kind of skill that shows up in senior interviews and real-world ML engineering.

## Key Takeaways

✅ A 90° rotation is just a **deterministic mapping of coordinates**—getting the index math right is the core.

✅ You can implement rotation in-place using **transpose + row reversal** or more manual **layer-based 4-way swaps**.

✅ The runtime is \(\Theta(n²)\) and extra space is \(O(1)\), which scales well even for large matrices.

✅ Many ML and systems tasks boil down to **2D/ND tensor transformations** that use the same reasoning.

✅ Practicing these transforms pays off directly in writing and debugging performance-critical data and model pipelines.

---

**Originally published at:** [arunbaby.com/dsa/0018-rotate-image](https://www.arunbaby.com/dsa/0018-rotate-image/)

*If you found this helpful, consider sharing it with others who might benefit.*








