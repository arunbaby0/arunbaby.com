---
title: "Largest Rectangle in Histogram"
day: 44
related_ml_day: 44
related_speech_day: 44
related_agents_day: 44
collection: dsa
categories:
 - dsa
tags:
 - stack
 - monotonic-stack
 - array
 - divide-and-conquer
difficulty: Hard
---

**"Finding the maximum hidden in the valleys and peaks."**

## 1. Problem Statement

Given an array of integers `heights` representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

**Example 1:**
``
Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The largest rectangle has an area of 10 (heights[2] and heights[3]).
``

**Visual:**
``
 ┌───┐
 │ │
 ┌───┤ │
 │ │ │ ┌───┐
 │ │ ├───┤ │
 ├───┤ │ │ │
 │ 2 │ 1 │ 5 │ 6 │ 2 │ 3 │
``

**Example 2:**
``
Input: heights = [2,4]
Output: 4
``

**Constraints:**
* `1 <= heights.length <= 10^5`
* `0 <= heights[i] <= 10^4`

## 2. Intuition

For each bar, we want to find the largest rectangle that includes it as the limiting height. This means finding how far we can extend left and right before hitting a shorter bar.

**Key Insight:** The limiting bars on left and right are the first bars shorter than the current bar. This is exactly what a **monotonic stack** efficiently computes.

## 3. Approach 1: Brute Force

**Algorithm:**
1. For each bar `i`, find the first shorter bar on the left (`left[i]`).
2. For each bar `i`, find the first shorter bar on the right (`right[i]`).
3. Area = height[i] × (right[i] - left[i] - 1).
4. Return the maximum area.

``python
def largestRectangleArea(heights):
 n = len(heights)
 max_area = 0
 
 for i in range(n):
 # Find left boundary
 left = i - 1
 while left >= 0 and heights[left] >= heights[i]:
 left -= 1
 
 # Find right boundary
 right = i + 1
 while right < n and heights[right] >= heights[i]:
 right += 1
 
 area = heights[i] * (right - left - 1)
 max_area = max(max_area, area)
 
 return max_area
``

**Complexity:**
* **Time:** O(N^2) in worst case (e.g., sorted array).
* **Space:** O(1).

## 4. Approach 2: Monotonic Stack (Optimal)

**Idea:** Maintain a stack of indices where corresponding heights are in increasing order. When we encounter a bar shorter than the stack top, we pop and compute areas.

**Algorithm:**
1. Iterate through bars from left to right.
2. If current bar is taller than stack top, push index.
3. If current bar is shorter, pop and compute area.
4. After iteration, pop remaining elements and compute areas.

``python
def largestRectangleArea(heights):
 stack = [] # Stack of indices
 max_area = 0
 n = len(heights)
 
 for i in range(n + 1):
 # Use 0 as sentinel for index n
 h = heights[i] if i < n else 0
 
 while stack and heights[stack[-1]] > h:
 height = heights[stack.pop()]
 width = i if not stack else i - stack[-1] - 1
 max_area = max(max_area, height * width)
 
 stack.append(i)
 
 return max_area
``

**Complexity:**
* **Time:** O(N). Each index is pushed and popped at most once.
* **Space:** O(N) for the stack.

## 5. Detailed Walkthrough

**Example:** `heights = [2, 1, 5, 6, 2, 3]`

**Processing:**
| i | h | Stack | Action | Area |
|---|---|-------|--------|------|
| 0 | 2 | [0] | Push 0 | - |
| 1 | 1 | [0] | Pop 0, area = 2×1 = 2 | 2 |
| 1 | 1 | [1] | Push 1 | - |
| 2 | 5 | [1,2] | Push 2 | - |
| 3 | 6 | [1,2,3] | Push 3 | - |
| 4 | 2 | [1,2,3] | Pop 3, area = 6×1 = 6 | 6 |
| 4 | 2 | [1,2] | Pop 2, area = 5×2 = 10 | **10** |
| 4 | 2 | [1,4] | Push 4 | - |
| 5 | 3 | [1,4,5] | Push 5 | - |
| 6 | 0 | [1,4,5] | Pop 5, area = 3×1 = 3 | 10 |
| 6 | 0 | [1,4] | Pop 4, area = 2×3 = 6 | 10 |
| 6 | 0 | [1] | Pop 1, area = 1×6 = 6 | 10 |

**Result:** 10

## 6. Why Monotonic Stack Works

**Invariant:** The stack is always increasing.

**When we pop index `p`:**
* Current index `i` is the first index where `heights[i] < heights[p]`.
* Stack top (after pop) or -1 is the first index where `heights[stack[-1]] < heights[p]`.
* Width = `i - stack[-1] - 1` (or `i` if stack is empty).

**Each bar determines a rectangle:**
* Height = bar's height.
* Width = distance between first shorter bar on left and right.

## 7. Approach 3: Divide and Conquer

**Algorithm:**
1. Find the minimum bar in the range.
2. The largest rectangle either:
 * Uses the minimum bar (spans entire range).
 * Is entirely in the left half.
 * Is entirely in the right half.
3. Recursively compute.

``python
def largestRectangleArea(heights):
 def helper(left, right):
 if left > right:
 return 0
 
 min_idx = left
 for i in range(left, right + 1):
 if heights[i] < heights[min_idx]:
 min_idx = i
 
 # Area using min bar
 area_with_min = heights[min_idx] * (right - left + 1)
 
 # Recurse on left and right
 area_left = helper(left, min_idx - 1)
 area_right = helper(min_idx + 1, right)
 
 return max(area_with_min, area_left, area_right)
 
 return helper(0, len(heights) - 1)
``

**Complexity:**
* **Time:** O(N \log N) average, O(N^2) worst case.
* **Space:** O(\log N) for recursion.
* **Optimization:** Use segment tree for min queries → O(N \log N) worst case.

## 8. Approach 4: Precompute Left and Right Boundaries

**Algorithm:**
1. Precompute `left[i]`: Index of first shorter bar on left.
2. Precompute `right[i]`: Index of first shorter bar on right.
3. Area = `heights[i] × (right[i] - left[i] - 1)`.

``python
def largestRectangleArea(heights):
 n = len(heights)
 left = [-1] * n
 right = [n] * n
 stack = []
 
 # Compute left boundaries
 for i in range(n):
 while stack and heights[stack[-1]] >= heights[i]:
 stack.pop()
 left[i] = stack[-1] if stack else -1
 stack.append(i)
 
 stack = []
 
 # Compute right boundaries
 for i in range(n - 1, -1, -1):
 while stack and heights[stack[-1]] >= heights[i]:
 stack.pop()
 right[i] = stack[-1] if stack else n
 stack.append(i)
 
 # Compute max area
 max_area = 0
 for i in range(n):
 area = heights[i] * (right[i] - left[i] - 1)
 max_area = max(max_area, area)
 
 return max_area
``

**Complexity:**
* **Time:** O(N).
* **Space:** O(N) for left, right arrays and stack.

## 9. Extension: Maximal Rectangle in Binary Matrix

**Problem (LeetCode 85):** Given a binary matrix, find the largest rectangle containing only 1s.

**Approach:**
1. For each row, compute the histogram of consecutive 1s above.
2. Apply Largest Rectangle in Histogram to each row.

``python
def maximalRectangle(matrix):
 if not matrix or not matrix[0]:
 return 0
 
 rows, cols = len(matrix), len(matrix[0])
 heights = [0] * cols
 max_area = 0
 
 for row in matrix:
 for j in range(cols):
 if row[j] == '1':
 heights[j] += 1
 else:
 heights[j] = 0
 
 max_area = max(max_area, largestRectangleArea(heights))
 
 return max_area
``

**Complexity:** O(R \times C) where R = rows, C = columns.

## 10. System Design: Image Processing

**Scenario:** Find the largest uniform region in an image.

**Application:** Logo detection, OCR bounding boxes, defect detection.

**Algorithm:**
1. Convert image to binary (foreground/background).
2. Apply maximal rectangle algorithm.
3. The largest rectangle is the detected region.

## 11. Deep Dive: Segment Tree for Min Query

**Problem with Divide and Conquer:** Finding minimum in each subarray is O(N) per level.

**Solution:** Build a segment tree for range minimum queries.

**Segment Tree Operations:**
* **Build:** O(N).
* **Query (min in range):** O(\log N).

**Result:** Divide and conquer with O(N \log N) guaranteed.

``python
class SegmentTree:
 def __init__(self, arr):
 self.n = len(arr)
 self.tree = [0] * (4 * self.n)
 self.arr = arr
 self.build(0, 0, self.n - 1)
 
 def build(self, node, start, end):
 if start == end:
 self.tree[node] = start
 else:
 mid = (start + end) // 2
 self.build(2*node+1, start, mid)
 self.build(2*node+2, mid+1, end)
 left_idx = self.tree[2*node+1]
 right_idx = self.tree[2*node+2]
 self.tree[node] = left_idx if self.arr[left_idx] <= self.arr[right_idx] else right_idx
 
 def query(self, node, start, end, l, r):
 if r < start or l > end:
 return -1
 if l <= start and end <= r:
 return self.tree[node]
 
 mid = (start + end) // 2
 left_min = self.query(2*node+1, start, mid, l, r)
 right_min = self.query(2*node+2, mid+1, end, l, r)
 
 if left_min == -1:
 return right_min
 if right_min == -1:
 return left_min
 return left_min if self.arr[left_min] <= self.arr[right_min] else right_min
``

## 12. Interview Questions

1. **Largest Rectangle in Histogram (Classic):** Solve with monotonic stack.
2. **Maximal Rectangle in Binary Matrix:** Extension using histogram.
3. **Trapping Rain Water:** Similar monotonic stack pattern.
4. **Stock Span Problem:** Another monotonic stack application.
5. **Next Greater Element:** Fundamental monotonic stack problem.

## 13. Common Mistakes

* **Off-by-One Errors:** Width calculation is `i - stack[-1] - 1`, not `i - stack[-1]`.
* **Empty Stack:** Handle case when stack is empty after popping.
* **Forgetting Sentinel:** Add 0 at the end to force popping all elements.
* **Wrong Stack Order:** Stack should be increasing for this problem.
* **Integer Overflow:** For very large heights and widths.

## 14. Variant: Largest Rectangle with At Most K Distinct Heights

**Problem:** Find the largest rectangle where all bars have at most K distinct heights.

**Approach:**
1. Sliding window to find valid segments.
2. For each valid segment, apply histogram algorithm.

## 15. Performance Comparison

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Brute Force | O(N^2) | O(1) | Simple but slow |
| Monotonic Stack | O(N) | O(N) | Optimal |
| Divide & Conquer | O(N \log N) | O(\log N) | Good when N is small |
| D&C + Segment Tree | O(N \log N) | O(N) | Guaranteed O(N log N) |

## 16. Deep Dive: Online Histogram

**Problem:** Bars arrive one by one. Maintain the largest rectangle.

**Approach:**
1. Maintain a monotonic stack.
2. On each new bar, process like the standard algorithm.
3. Track global max area.

**Complexity:** O(1) amortized per bar.

## 17. Mathematical Insight

**Observation:** For a rectangle of height `h` and width `w`, the area is `h \times w`. We're maximizing this product.

**Trade-off:**
* Taller bars have larger heights but smaller widths.
* Shorter bars have smaller heights but larger widths.

**Optimal:** Often at the "knee" of the trade-off curve.

## 18. Testing Strategy

**Test Cases:**
1. **Single Bar:** `[5]` → 5.
2. **All Same Heights:** `[3,3,3,3]` → 12.
3. **Increasing:** `[1,2,3,4,5]` → 9.
4. **Decreasing:** `[5,4,3,2,1]` → 9.
5. **Valley:** `[5,1,5]` → 5.
6. **Peak:** `[1,5,1]` → 5.
7. **All Zeros:** `[0,0,0]` → 0.
8. **Large Array:** n = 100000 for performance.

## 19. Related Problems

* **Trapping Rain Water** (LeetCode 42)
* **Maximal Rectangle** (LeetCode 85)
* **Maximal Square** (LeetCode 221)
* **Container With Most Water** (LeetCode 11)
* **Next Greater Element** (LeetCode 496)

## 20. Conclusion

Largest Rectangle in Histogram is a classic problem that teaches the power of monotonic stacks. The key insight is that each bar defines a potential rectangle, and we just need to efficiently find its boundaries.

**Key Takeaways:**
* **Monotonic Stack:** The optimal solution uses this data structure.
* **O(N) Time:** Each element is pushed and popped at most once.
* **Extension:** The algorithm extends to 2D (maximal rectangle in matrix).
* **Pattern:** Monotonic stack is useful whenever we need "next smaller/greater" elements.

Mastering this problem opens the door to many related stack problems. The pattern of using a stack to track boundaries is one of the most powerful techniques in competitive programming.

## 21. Mastery Checklist

**Mastery Checklist:**
- [ ] Implement brute force O(N²) solution
- [ ] Implement monotonic stack O(N) solution
- [ ] Explain why the stack invariant works
- [ ] Extend to maximal rectangle in binary matrix
- [ ] Handle edge cases (empty, single element)
- [ ] Implement divide-and-conquer with segment tree
- [ ] Solve related problems (Trapping Rain Water)
- [ ] Analyze time complexity rigorously

## 22. Deep Dive: Maximal Square

**Problem:** Find the largest square containing only 1s in a binary matrix.

**Approach (DP):**
* `dp[i][j]` = side length of largest square ending at (i, j).
* If `matrix[i][j] == 1`: `dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1`.

``python
def maximalSquare(matrix):
 if not matrix:
 return 0
 
 rows, cols = len(matrix), len(matrix[0])
 dp = [[0] * cols for _ in range(rows)]
 max_side = 0
 
 for i in range(rows):
 for j in range(cols):
 if matrix[i][j] == '1':
 if i == 0 or j == 0:
 dp[i][j] = 1
 else:
 dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
 max_side = max(max_side, dp[i][j])
 
 return max_side * max_side
``

**Complexity:** O(R \times C).

**Comparison:**
* Maximal Rectangle: Stack-based, considers varying widths.
* Maximal Square: DP, constrained to equal sides.

## 23. Stack Variants: Next Smaller Element

**Problem:** For each element, find the first smaller element to the right.

**Algorithm (Monotonic Stack):**
``python
def nextSmallerElement(arr):
 n = len(arr)
 result = [-1] * n
 stack = []
 
 for i in range(n):
 while stack and arr[stack[-1]] > arr[i]:
 result[stack.pop()] = arr[i]
 stack.append(i)
 
 return result
``

**Use Cases:**
* Stock Span Problem.
* Daily Temperatures.
* Largest Rectangle in Histogram.

## 24. Stack Variants: Previous Smaller Element

**Problem:** For each element, find the first smaller element to the left.

``python
def previousSmallerElement(arr):
 n = len(arr)
 result = [-1] * n
 stack = []
 
 for i in range(n):
 while stack and arr[stack[-1]] >= arr[i]:
 stack.pop()
 result[i] = stack[-1] if stack else -1
 stack.append(i)
 
 return result
``

## 25. Competitive Programming: Optimizations

**1. Avoid Index Out of Bounds:**
* Add sentinels (0 at the end of heights array).
* Simplifies loop termination.

**2. Single-Pass Optimization:**
* Combine left boundary computation with area calculation.
* Reduces constant factor.

**3. Memory Optimization:**
* Instead of precomputing left/right arrays, compute on-the-fly.
* Reduces space from O(N) to O(N) for stack only.

## 26. Application: Skyline Problem

**Problem:** Given building heights and positions, compute the skyline.

**Connection:** Both use stack-based reasoning to track boundaries.

**Algorithm:**
1. Convert buildings to events (start, end).
2. Process events in order.
3. Use max-heap to track current maximum height.
4. Output changes in maximum height.

## 27. Application: Maximum Area Rectangle in Terrain

**Problem:** Given terrain elevations, find the largest flat rectangle.

**Algorithm:**
1. For each row, compute histogram (consecutive cells at same elevation).
2. Apply Largest Rectangle in Histogram.
3. Track maximum across all elevations.

## 28. Interview Deep Dive

**Q: Walk through the algorithm step by step.**

**A:**
1. Initialize empty stack and max_area = 0.
2. Iterate through bars (with sentinel 0 at the end).
3. For each bar:
 * While stack has bars taller than current bar:
 * Pop the bar.
 * Height = popped bar's height.
 * Width = current index - stack top - 1 (or current index if stack is empty).
 * Compute area, update max_area.
 * Push current index to stack.
4. Return max_area.

**Q: Why does this work?**

**A:** When we pop a bar, we know:
* The current bar is the first shorter bar on the right.
* The new stack top is the first shorter bar on the left.
* So we have both boundaries, and can compute the width.

## 29. Space-Optimized Solution

**Observation:** We can compute areas while iterating, without storing left/right arrays.

``python
def largestRectangleArea_optimized(heights):
 heights.append(0) # Sentinel
 stack = [-1] # Sentinel for left boundary
 max_area = 0
 
 for i, h in enumerate(heights):
 while stack[-1] != -1 and heights[stack[-1]] > h:
 height = heights[stack.pop()]
 width = i - stack[-1] - 1
 max_area = max(max_area, height * width)
 stack.append(i)
 
 heights.pop() # Restore original
 return max_area
``

**Space:** O(N) for stack, no additional arrays.

## 30. Proof of Correctness

**Claim:** The algorithm correctly computes the largest rectangle.

**Proof:**
1. **Completeness:** Every bar is considered as the limiting height of some rectangle.
 * When bar `i` is popped, we compute the rectangle with height[i] as the limiting height.
2. **Correctness of Width:**
 * Right boundary: Current index `j` (first bar shorter than height[i]).
 * Left boundary: New stack top (first bar shorter than height[i] on the left).
 * Width = `j - left_boundary - 1`.
3. **Maximum Area:** We compute the area for every possible limiting height and track the maximum.

## 31. Conclusion

The Largest Rectangle in Histogram is a masterclass in monotonic stack usage. The key insights are:

1. **Each bar defines a potential rectangle** with that bar as the limiting height.
2. **Boundaries are found efficiently** using the monotonic stack property.
3. **O(N) time** because each bar is pushed and popped exactly once.

This pattern extends to many problems:
* Trapping Rain Water.
* Maximal Rectangle in Binary Matrix.
* Next Greater/Smaller Element.
* Stock Span Problem.

Once you internalize this pattern, a whole class of problems becomes tractable. The monotonic stack is one of the most powerful techniques in the algorithmic toolbox.

**Practice:** Solve the related problems to cement your understanding. The pattern will become second nature!



---

**Originally published at:** [arunbaby.com/dsa/0044-largest-rectangle-histogram](https://www.arunbaby.com/dsa/0044-largest-rectangle-histogram/)

*If you found this helpful, consider sharing it with others who might benefit.*

