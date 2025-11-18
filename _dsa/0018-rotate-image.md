---
title: "Rotate Image"
day: 18
collection: dsa
categories:
  - dsa
tags:
  - matrix
  - in-place
  - rotation
  - 2d-array
  - simulation
  - medium
subdomain: "Matrix & In-place Algorithms"
tech_stack: [Python]
scale: "O(N²) time, O(1) extra space"
companies: [Google, Meta, Amazon, Microsoft, Apple, Bloomberg]
related_dsa_day: 18
related_ml_day: 18
related_speech_day: 18
---

**Learn to rotate a matrix in-place—the same pattern used for data augmentation, geometric transforms, and tensor manipulations in deep learning systems.**

## Problem Statement

You are given an `n x n` 2D matrix representing an image. Rotate the image by **90 degrees** (clockwise).

You must rotate the image **in-place**, which means:

- You **cannot** allocate another `n x n` matrix for the result.
- You must modify the input matrix directly.

### Examples

**Example 1:**

```python
Input:
matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]

Output:
[
  [7, 4, 1],
  [8, 5, 2],
  [9, 6, 3]
]
```

**Example 2:**

```python
Input:
matrix = [
  [ 5,  1,  9, 11],
  [ 2,  4,  8, 10],
  [13,  3,  6,  7],
  [15, 14, 12, 16]
]

Output:
[
  [15, 13,  2,  5],
  [14,  3,  4,  1],
  [12,  6,  8,  9],
  [16,  7, 10, 11]
]
```

### Constraints

- \(1 \le n \le 20\) (in typical interview settings; online judge may allow larger)
- Matrix is square: `len(matrix) == len(matrix[0])`
- Values can be any integers

## Understanding the Problem

This is a **classic in-place matrix manipulation problem**. It teaches:

1. How to think about **2D index transformations**.
2. How to perform **in-place operations** without extra memory.
3. How to decompose global transforms into **local operations** (layers, cycles).
4. How matrix operations map to **data transformations** in ML systems.

### Visual Intuition

A 90° clockwise rotation turns:

```text
Row 0 → Column n-1
Row 1 → Column n-2
...
Row i → Column n-1-i
```

For a position \((r, c)\) in the original matrix:

- After rotation, it moves to: \((c, n-1-r)\)

Example for a 3x3 matrix:

```text
Original indices:          After 90° CW:

(0,0) (0,1) (0,2)          (2,0) (1,0) (0,0)
(1,0) (1,1) (1,2)   →      (2,1) (1,1) (0,1)
(2,0) (2,1) (2,2)          (2,2) (1,2) (0,2)
```

### Why In-Place?

In many real systems:

- Memory is limited (e.g., on-device ML, edge devices).
- Copying large image tensors is expensive.
- In-place transforms reduce **memory bandwidth** and **cache pressure**.

This is exactly the kind of transformation you see in:
- Data augmentation pipelines (image rotations),
- Tensor layout transforms for optimized inference/training,
- GPU kernels that avoid extra allocations.

## Approach 1: Extra Matrix (Simple but Not In-Place)

### Intuition

Use an extra `n x n` matrix:

- For each cell `(r, c)` in original matrix, write it to `(c, n-1-r)` in new matrix.
- Then copy new matrix back to original.

### Implementation

```python
from typing import List

def rotate_copy(matrix: List[List[int]]) -> None:
    \"\"\"Rotate matrix 90° clockwise using extra memory.

    Time:  O(n²)
    Space: O(n²) extra

    Not allowed by the problem constraints (requires extra matrix),
    but useful to verify correctness and build intuition.
    \"\"\"
    n = len(matrix)
    rotated = [[0] * n for _ in range(n)]

    for r in range(n):
        for c in range(n):
            rotated[c][n - 1 - r] = matrix[r][c]

    # Copy back
    for r in range(n):
        for c in range(n):
            matrix[r][c] = rotated[r][c]
```

### Why It's Not Enough

- Uses extra `O(n²)` space.
- Violates the explicit in-place requirement.
- However, it's great for:
  - Testing your in-place solution.
  - Understanding the mapping `(r, c) → (c, n-1-r)`.

We need an **in-place** method.

## Approach 2: Transpose + Reverse Rows (Elegant In-Place)

### Key Idea

A 90° clockwise rotation can be decomposed into:

1. **Transpose** the matrix: reflect across main diagonal.
2. **Reverse each row**.

Example:

```text
Original:
1 2 3
4 5 6
7 8 9

Transpose:
1 4 7
2 5 8
3 6 9

Reverse each row:
7 4 1
8 5 2
9 6 3
```

This matches the expected rotated matrix.

### Why It Works

Transpose swaps coordinates: \((r, c) → (c, r)\)

Then, reversing each row transforms:

```text
(c, r) → (c, n-1-r)
```

Combining both:

```text
(r, c)  --transpose-->  (c, r)
       --reverse-->     (c, n-1-r)
```

Which is exactly the 90° rotation mapping.

### Implementation

```python
from typing import List

def rotate(matrix: List[List[int]]) -> None:
    \"\"\"Rotate matrix 90° clockwise in-place using transpose + reverse.

    Time:  O(n²)
    Space: O(1) extra
    \"\"\"
    n = len(matrix)

    # 1. Transpose in-place: swap matrix[r][c] with matrix[c][r]
    for r in range(n):
        for c in range(r + 1, n):
            matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]

    # 2. Reverse each row
    for r in range(n):
        matrix[r].reverse()
```

### Step-by-Step Example

For `matrix = [[1,2,3],[4,5,6],[7,8,9]]`:

1. **Transpose:**

```text
[1,2,3]       [1,4,7]
[4,5,6]   →   [2,5,8]
[7,8,9]       [3,6,9]
```

2. **Reverse each row:**

```text
[1,4,7]   →   [7,4,1]
[2,5,8]   →   [8,5,2]
[3,6,9]   →   [9,6,3]
```

Result: `[[7,4,1],[8,5,2],[9,6,3]]`.

## Approach 3: Layer-by-Layer 4-Way Swaps

### Intuition

Think of the matrix as composed of **concentric layers (rings)**:

```text
For 4x4 matrix:
Layer 0: outer ring
Layer 1: inner 2x2 ring
```

For each layer, we rotate elements in groups of 4:

```text
top    → right
left   → top
bottom → left
right  → bottom
```

We can manually cycle four positions:

```python
top    = matrix[row][col]
right  = matrix[col][n-1-row]
bottom = matrix[n-1-row][n-1-col]
left   = matrix[n-1-col][row]
```

### Implementation

```python
from typing import List

def rotate_layers(matrix: List[List[int]]) -> None:
    \"\"\"Rotate matrix in-place by processing layers and 4-way swaps.\"\"\"\n    n = len(matrix)
    # Number of layers is n//2
    for layer in range(n // 2):
        first = layer
        last = n - 1 - layer
\n        for i in range(first, last):
            offset = i - first\n\n            # Save top\n            top = matrix[first][i]\n\n            # left -> top\n            matrix[first][i] = matrix[last - offset][first]\n\n            # bottom -> left\n            matrix[last - offset][first] = matrix[last][last - offset]\n\n            # right -> bottom\n            matrix[last][last - offset] = matrix[i][last]\n\n            # top -> right\n            matrix[i][last] = top\n```

### When to Use Which Approach?

- **Transpose + reverse:**
  - Simpler, fewer indices to get wrong.
  - Very common and idiomatic in interviews.
- **Layer-by-layer:**
  - Good for demonstrating deep index manipulation skills.
  - Natural extension when you need more complex per-layer logic.

Both are **O(n²)** time and **O(1)** extra space.

## Implementation: Tests and Utilities

```python
from typing import List

def rotate_in_place(matrix: List[List[int]]) -> None:
    \"\"\"Wrapper for the chosen implementation (transpose + reverse).\"\"\"\n    rotate(matrix)


def test_rotate():
    tests = [
        (
            [[1,2,3],[4,5,6],[7,8,9]],
            [[7,4,1],[8,5,2],[9,6,3]],
        ),
        (
            [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]],
            [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]],
        ),
        (
            [[1]],
            [[1]],
        ),
        (
            [[1,2],[3,4]],
            [[3,1],[4,2]],
        ),
    ]

    for i, (matrix, expected) in enumerate(tests, 1):
        rotate_in_place(matrix)
        assert matrix == expected, f\"Test {i} failed: {matrix} != {expected}\"

    print(\"All rotate tests passed.\")


if __name__ == \"__main__\":
    test_rotate()
```

## Complexity Analysis

Let \(n\) be the number of rows/columns (matrix is \(n \times n\)):

### Time Complexity

- We touch each element **a constant number of times**:
  - Transpose: \(\frac{n(n-1)}{2} \approx O(n²)\) swaps.
  - Reverse each row: \(n\) rows × \(n/2\) swaps = \(O(n²)\).
  - Or, in layer-by-layer, each element moves exactly once.
- Total: \(\boxed{O(n²)}\).

### Space Complexity

- We only use a few temporary variables (for swapping).
- No extra matrix allocated.
- Total extra space: \(\boxed{O(1)}\).

## Production Considerations

Although this is a DSA problem, the **matrix rotation pattern** maps directly to
real-world systems:

- **Image data augmentation:**
  - Random rotations (90°, 180°, 270°) for robustness.
  - Often implemented via libraries (e.g., PIL, OpenCV, torchvision), but the
    underlying logic is the same mapping of indices.

- **Tensor layout transforms:**
  - NCHW → NHWC conversions,
  - Channel-last vs channel-first memory layouts,
  - Strided views that avoid copies (important for GPU performance).

- **Block-based transforms:**
  - Splitting images into patches (e.g., Vision Transformers),
  - Reassembling after per-block operations.

In performance-critical code (e.g., CUDA kernels), carefully mapping indices
and minimizing memory writes/reads is exactly what this problem trains you to do.

## Connections to ML Systems

The thematic link for this day is **matrix operations and data transformations**.
Rotating a 2D image is a prototypical example of a **spatial transform**.

### 1. Data Augmentation Pipelines

In ML pipelines, we often apply random transforms:

- Rotations (90° increments or arbitrary angles),
- Flips (horizontal/vertical),
- Crops, scales, and perspective warps.

These transforms are all **coordinate mappings**:

```text
output[x', y'] = input[f(x', y')]
```

Understanding the simple case (90° rotation) makes it easier to reason about
more complex geometric transforms used in computer vision and even spectrogram
augmentations in speech.

### 2. Spectrogram Manipulation in Speech

Audio models often consume **time-frequency matrices** (spectrograms). Rotations
are less common, but operations like:

- Time masking,
- Frequency masking,
- Time warping,

are all implemented by manipulating slices along the axes. The mental model of
\"treat this 2D array as a geometric object and transform its indices\" is the
same one you use here.

### 3. Distributed Tensor Sharding

In large-scale ML training, tensors are sharded across devices:

- Splitting along batch dimension,
- Splitting along feature/channel dimensions,
- Sometimes even 2D sharding of weight matrices.

The underlying operations often boil down to:

- Slicing submatrices,
- Reassembling them after all-reduce or all-gather,
- Maintaining consistent index mappings across devices.

Being fluent with small 2D index transformations (like Rotate Image) helps you
avoid subtle bugs when working with large sharded tensors.

## Interview Strategy

### How to Communicate Your Approach

In an interview, structure your explanation like this:

1. **Clarify constraints:**
   - Square matrix?
   - In-place required?
   - 90° clockwise only, or arbitrary rotations?

2. **Outline the naive solution (extra matrix):**
   - Talk through `(r, c) → (c, n-1-r)`.
   - Acknowledge extra `O(n²)` memory and why it's not allowed.

3. **Present the in-place solution:**
   - Option A: transpose + reverse.
   - Option B: layer-by-layer 4-way swaps.
   - Pick one and explain why it’s simpler/safer to implement.

4. **Discuss complexity:**
   - Time: `O(n²)`, Space: `O(1)` extra.

5. **(Bonus) Mention applications:**
   - Data augmentation, tensor transforms, geometric reasoning.

### Common Pitfalls

- Forgetting that transpose should only swap above the diagonal
  (`for c in range(r+1, n)`), otherwise you double-swap.
- Mixing up row/column indices when computing the 4-way swaps.
- Accidentally using an extra `n x n` matrix (violates in-place requirement).

## Key Takeaways

✅ **90° rotation is just a coordinate transform**: \((r, c) → (c, n-1-r)\).

✅ **Transpose + reverse rows** gives a clean, in-place implementation.

✅ **Layer-by-layer 4-way swaps** are a more manual but instructive alternative.

✅ **Time complexity is O(n²)** and **extra space is O(1)**.

✅ **Matrix transforms** like this appear in data augmentation, tensor layout
changes, and distributed tensor sharding.

✅ **Thinking in terms of index mappings** is a valuable skill for both algorithm
interviews and real-world ML systems work.

---

**Originally published at:** [arunbaby.com/dsa/0018-rotate-image](https://www.arunbaby.com/dsa/0018-rotate-image/)

*If you found this helpful, consider sharing it with others who might benefit.*

---
title: "Rotate Image"
day: 18
collection: dsa
categories:
  - dsa
tags:
  - matrix
  - array
  - in-place
  - simulation
  - 2d-transform
  - medium
subdomain: "Matrix & 2D Array Algorithms"
tech_stack: [Python]
scale: "O(N²) time, O(1) extra space"
companies: [Google, Meta, Amazon, Microsoft, Apple]
related_dsa_day: 18
related_ml_day: 18
related_speech_day: 18
---

**Master in-place matrix rotation—the same pattern used in data augmentation pipelines and log-mel transformations for speech.**

## Problem Statement

You are given an `n x n` 2D matrix representing an image. Rotate the image by **90 degrees (clockwise)**.

You must rotate the matrix **in-place**, which means you have to modify the input matrix directly. **Do not allocate another 2D matrix for the rotation.**

### Examples

**Example 1:**

```text
Input:
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
]

Output:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
```

**Example 2:**

```text
Input:
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
]

Output:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]
```

### Constraints

- `n == matrix.length`
- `n == matrix[i].length`
- `1 <= n <= 20`
- `-1000 <= matrix[i][j] <= 1000`

## Understanding the Problem

We have a square matrix (image) and need to rotate it **90° clockwise** in-place.

Visually:

```text
Original (3x3):
1 2 3
4 5 6
7 8 9

Rotated 90° clockwise:
7 4 1
8 5 2
9 6 3
```

Think of coordinates:
- Original indices: `(row, col)`
- After rotation: `(row, col) -> (col, n - 1 - row)`

This is a **2D coordinate transform**—exactly the kind of operation we do in:
- Image augmentations (rotate, flip),
- Log-mel spectrogram manipulations,
- Matrix-based data transformations.

### Why In-Place Matters

Naively, you could allocate a new `n x n` matrix and write:

```python
rotated[col][n - 1 - row] = matrix[row][col]
```

But the problem explicitly asks for **in-place**:

- **Space constraint**: O(1) extra memory
- Focus on **in-place algorithms** and index manipulation

This is common in:
- Memory-constrained systems,
- High-performance pipelines,
- Large tensors in ML systems.

## Approach 1: Extra Matrix (Brute Force)

### Intuition

1. Allocate a new `n x n` matrix `rot`.
2. For each `(i, j)` in original:
   - Set `rot[j][n - 1 - i] = matrix[i][j]`.
3. Copy `rot` back into `matrix`.

### Implementation

```python
from typing import List

def rotate_bruteforce(matrix: List[List[int]]) -> None:
    \"\"\"Rotate matrix 90° clockwise using extra matrix.

    Time:  O(N²)
    Space: O(N²) extra
    \"\"\"
    n = len(matrix)
    # Create rotated matrix
    rot = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            # Map (i, j) -> (j, n - 1 - i)
            rot[j][n - 1 - i] = matrix[i][j]

    # Copy back
    for i in range(n):
        for j in range(n):
            matrix[i][j] = rot[i][j]
```

### Limitations

- Violates space constraint (allocates another `n x n` matrix).
- Interviewers typically want **O(1) extra space**.

But this approach helps build intuition about the coordinate transform.

## Approach 2: Transpose + Reverse Rows (Optimal)

### Key Insight

A 90° clockwise rotation can be factored into two simpler operations:

1. **Transpose** the matrix (swap rows and columns).
2. **Reverse each row**.

Let's see it on a 3x3 example:

```text
Original:
1 2 3
4 5 6
7 8 9

Transpose (swap matrix[i][j] with matrix[j][i]):
1 4 7
2 5 8
3 6 9

Reverse each row:
7 4 1
8 5 2
9 6 3
```

Exactly what we want!

### Why This Works

The transpose operation maps:
- `(i, j) -> (j, i)`

After transpose, reversing row `i` maps:
- `(i, j) -> (i, n - 1 - j)`

Compose these transformations:

```text
Original index: (i, j)
After transpose: (j, i)
After row reverse: (j, n - 1 - i)
```

Which is exactly the coordinate for a 90° clockwise rotation.

### Implementation

```python
from typing import List

def rotate(matrix: List[List[int]]) -> None:
    \"\"\"Rotate matrix 90° clockwise in-place.

    Uses transpose + reverse rows.

    Time:  O(N²)
    Space: O(1) extra
    \"\"\"
    n = len(matrix)

    # 1. Transpose in-place
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # 2. Reverse each row
    for i in range(n):
        matrix[i].reverse()
```

### Step-by-Step Example (4x4)

```text
Original:
 5  1  9 11
 2  4  8 10
13  3  6  7
15 14 12 16

Transpose:
 5  2 13 15
 1  4  3 14
 9  8  6 12
11 10  7 16

Reverse each row:
15 13  2  5
14  3  4  1
12  6  8  9
16  7 10 11
```

Exactly matches the example output.

## Approach 3: Layer-by-Layer Rotation (Direct Swaps)

### Intuition

We can think of the matrix as a series of **concentric layers** (rings) and rotate each layer independently.

For a 4x4:

```text
Layer 0 (outer):
 [0,0] [0,1] [0,2] [0,3]
 [1,0]             [1,3]
 [2,0]             [2,3]
 [3,0] [3,1] [3,2] [3,3]

Layer 1 (inner):
 [1,1] [1,2]
 [2,1] [2,2]
```

For each layer, we rotate elements in groups of 4:

```text
top    -> right
left   -> top
bottom -> left
right  -> bottom
```

### Implementation

```python
def rotate_layer_by_layer(matrix: List[List[int]]) -> None:
    \"\"\"Rotate matrix 90° clockwise in-place by rotating layers.\"\"\"\n    n = len(matrix)
    left, right = 0, n - 1

    while left < right:
        top, bottom = left, right

        # Rotate the current layer
        for i in range(right - left):
            # Save top-left
            top_left = matrix[top][left + i]

            # Move bottom-left -> top-left
            matrix[top][left + i] = matrix[bottom - i][left]

            # Move bottom-right -> bottom-left
            matrix[bottom - i][left] = matrix[bottom][right - i]

            # Move top-right -> bottom-right
            matrix[bottom][right - i] = matrix[top + i][right]

            # Move saved top-left -> top-right
            matrix[top + i][right] = top_left

        left += 1
        right -= 1
```

This approach is more explicit and useful when:
- You need more control over which elements move where,
- You want to rotate by 90°, 180°, or 270° with different patterns.

But for interviews, **transpose + reverse** is simpler and less bug-prone.

## Implementation Notes & Edge Cases

1. **Empty matrix / 1x1 matrix:**
   - `n = 0` or `n = 1` → rotation is a no-op.
2. **Negative / large values:**
   - Values don't matter; we only move them.
3. **In-place safety:**
   - Ensure you don't override values before they've been used (layer approach).

## Testing

```python
import copy

def test_rotate():
    tests = [
        (
            [
                [1,2,3],
                [4,5,6],
                [7,8,9]
            ],
            [
                [7,4,1],
                [8,5,2],
                [9,6,3]
            ]
        ),
        (
            [
                [ 5, 1, 9,11],
                [ 2, 4, 8,10],
                [13, 3, 6, 7],
                [15,14,12,16]
            ],
            [
                [15,13, 2, 5],
                [14, 3, 4, 1],
                [12, 6, 8, 9],
                [16, 7,10,11]
            ]
        ),
        (
            [[1]],
            [[1]],
        ),
        (
            [
                [1,2],
                [3,4]
            ],
            [
                [3,1],
                [4,2]
            ]
        )
    ]

    for i, (matrix, expected) in enumerate(tests, 1):
        m1 = copy.deepcopy(matrix)
        rotate(m1)
        assert m1 == expected, f\"Test {i} failed: {m1} != {expected}\"

    print(\"All tests passed for rotate().\")\n\n\nif __name__ == \"__main__\":\n    test_rotate()\n```

## Complexity Analysis

Let \(n\) be the dimension of the matrix (`n x n`).

### Time Complexity

- We visit each element a constant number of times.
- Total elements: \(n^2\).
- Time complexity: \(O(n^2)\).

### Space Complexity

- We only use a few scalar variables (indices, temporary storage).
- No extra `n x n` matrix is allocated.
- Space complexity: \(O(1)\) extra.

## Production Considerations

Although this is a DSA problem, the pattern shows up in real systems:

### 1. Image Data Augmentation

- Rotate input images by multiples of 90° for data augmentation.
- In practice, libraries (OpenCV, PIL, torchvision) handle this efficiently.
- But under the hood, they perform **coordinate transforms** similar to this:

```python
def rotate_90(image):
    # Using NumPy for brevity
    return np.rot90(image, k=-1)  # 90° clockwise
```

Understanding the index math helps when:
- You implement custom augmentations (e.g., rotate log-mel spectrograms),
- You debug misaligned data (e.g., labels not matching rotated images).

### 2. Matrix-Based Transforms in ML

- Many ML ops are linear algebra transformations on matrices/tensors:
  - Transpose, reshape, permute,
  - Rotations and flips for data augmentation,
  - Time-frequency transforms in speech.
- This problem forces you to reason about **index mappings**:
  - From `(i, j)` to `(j, n - 1 - i)`,
  - Or more complex patterns in real models.

### 3. Cache-Friendly Implementations

- Transpose + reverse rows can be more cache-friendly than arbitrary swaps.
- For very large matrices/tensors, memory access patterns matter:
  - Contiguous access vs random access,
  - Minimizing cache misses.

## Interview Strategy

### How to Approach This Problem

**1. Clarify (1–2 minutes)**

- Matrix is always square? (Yes, here.)
- Rotation direction? (90° clockwise.)
- In-place requirement? (Yes → no extra `n x n`.)

**2. Start with a simple but suboptimal solution**

> \"First, I’d solve it by allocating a new matrix and writing 
> `rot[j][n-1-i] = matrix[i][j]`. That’s O(n²) time and O(n²) space. 
> Then I’ll optimize to do it in-place.\"

**3. Present transpose + reverse idea**

Explain the math and walk through a 3x3 example.

**4. Implement carefully**

- Watch indices in the transpose loop (`for j in range(i+1, n)`).
- Use built-in `reverse()` for clarity and safety.

**5. Discuss layer-by-layer as an alternative**

- Shows deeper understanding.
- But emphasize that transpose + reverse is usually preferred.

### Common Pitfalls

1. **Transposing incorrectly:**
   - Using `for j in range(n)` instead of `range(i+1, n)` → double-swap and revert.
2. **Index off-by-one errors** in layer-based rotation.
3. **Not handling small matrices** (e.g., `n = 1`).

### Follow-up Questions

1. **Rotate 90° counter-clockwise or 180°:**
   - 90° counter-clockwise:
     - Transpose, then reverse **columns** instead of rows.
   - 180°:
     - Reverse rows, then reverse columns,
     - Or apply 90° rotation twice.

2. **Non-square matrices?**
   - Out of scope for this problem,
   - But you can still perform generalized rotate operations using a new matrix.

3. **Apply to tensors (e.g., `H x W x C` images)?**
   - Either rotate channels independently,
   - Or treat `H x W` as the rotation plane and keep `C` fixed.

## Key Takeaways

✅ **Matrix rotation is just a coordinate transform**—from `(i, j)` to `(j, n - 1 - i)`.

✅ **Transpose + reverse rows** gives an elegant, in-place O(n²) solution.

✅ **Layer-by-layer rotation** is a more explicit but error-prone alternative.

✅ **Index math mastery** pays off in both interviews and production ML code.

✅ **The same pattern appears** in image augmentation, spectrogram manipulation, and other data transformations.

---

**Originally published at:** [arunbaby.com/dsa/0018-rotate-image](https://www.arunbaby.com/dsa/0018-rotate-image/)

*If you found this helpful, consider sharing it with others who might benefit.*


