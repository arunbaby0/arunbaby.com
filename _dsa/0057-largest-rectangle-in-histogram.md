---
title: "Largest Rectangle in Histogram"
day: 57
related_ml_day: 57
related_speech_day: 57
related_agents_day: 57
collection: dsa
categories:
 - dsa
tags:
 - stack
 - monotonic-stack
 - array
 - hard
 - greedy
difficulty: Hard
subdomain: "Stacks & Queues"
tech_stack: Python
scale: "O(N) time for arrays up to 10^5 elements"
companies: [Google, Amazon, Meta, Microsoft, Netflix, Bloomberg]
---

**"Largest Rectangle in Histogram is the masterclass of the Monotonic Stack. It requires maintaining a sorted state of indices to solve a local minimum problem in linear time."**

## 1. Introduction: The Geometry of Data

Imagine you have a series of buildings of different heights forming a skyline. You want to find the largest rectangular billboard that can be placed against these buildings. This isn't just a geometric puzzle; it is a fundamental problem in information retrieval, image processing (finding features in histograms), and even manufacturing (optimizing raw material cuts).

The brute-force approach—checking every possible pair of buildings as boundaries—results in an O(N^2) complexity. For a high-resolution histogram with millions of bins, this is too slow. To solve it in O(N), we must move from "Blind Search" to "Intelligent Tracking." Today we master the **Monotonic Stack**, a data structure that helps us find the nearest smaller value in a single pass.

---

## 2. The Problem Statement

Given an array of integers `heights` representing a histogram's bar height where the width of each bar is 1, find the area of the largest rectangle in the histogram.

**Example 1:**
- Input: `heights = [2, 1, 5, 6, 2, 3]`
- Output: `10`
- Explanation: The largest rectangle is formed by bars `5` and `6` with height `5` and width `2`.

---

## 3. Thematic Link: Capacity Planning and Scaling

Today we focus on **Infrastructure Scaling**:
- **DSA**: We find the maximum "Capacity" (area) possible within a constrained "Architecture" (histogram).
- **ML System Design**: Capacity planning determines the number of GPUs needed based on throughput "Histograms" of user traffic.
- **Speech Tech**: Scaling speech infrastructure requires optimizing the "Maximum Load" a server can handle before audio quality degrades.
- **Agents**: Agent Reliability Engineering (ARE) ensures that an agent swarm's "Rectangle of Success" isn't collapsed by a single weak sub-agent.

---

## 4. Approach 1: Brute Force (The Foundation)

Before the stack, we must understand the "Pivot" logic.
For every bar `i`, assume it is the **shortest bar** in our rectangle. How far can we expand left and right?
1. Expand `left` until we hit a bar shorter than `heights[i]`.
2. Expand `right` until we hit a bar shorter than `heights[i]`.
3. `Width = right - left - 1`.
4. `Area = heights[i] * Width`.

The problem is the expansion takes O(N) for each bar, totaling O(N^2). We need a way to pre-calculate these boundaries or find them on the fly.

---

## 5. Approach 2: Monotonic Stack (The O(N) Solution)

A **Monotonic Increasing Stack** stores indices such that the heights corresponding to these indices are always increasing.

### 5.1 The Algorithm
1. Iterate through `heights`.
2. While the current `heights[i]` is **shorter** than the height at the `stack.top()`:
 - We have found the "Right Boundary" for the bar at the top of the stack.
 - Pop the bar from the stack. It is our "Pivot" (the shortest bar).
 - Its "Left Boundary" is the index now at the new `stack.top()`.
 - `Area = height_of_pivot * (i - left_boundary - 1)`.
3. Push the current index `i` onto the stack.

### 5.2 Implementation details

``python
class Solution:
 def largestRectangleArea(self, heights: List[int]) -> int:
 # We add a 0 at the end to force the stack to clear at the last step
 heights.append(0)
 stack = [-1] # Dummy index to handle the left boundary of the first bar
 max_area = 0

 for i in range(len(heights)):
 while len(stack) > 1 and heights[i] < heights[stack[-1]]:
 # heights[i] is the first bar to the right strictly shorter than top
 h = heights[stack.pop()]
 # The index at stack[-1] is the first bar to the left strictly shorter
 w = i - stack[-1] - 1
 max_area = max(max_area, h * w)
 stack.append(i)
 
 # Optional: Restore heights if needed
 heights.pop()
 return max_area
``

---

## 6. Implementation Deep Dive: One Pass logic

Why does this work in one pass?
The monotonic stack maintains a "Promise": *Everything in the stack is waiting for its right boundary.*
- When we see a taller bar, the promise continues. 
- When we see a shorter bar, the promise for the previous bar is "Fulfilled"—we now know exactly how wide its rectangle can be.

This is a **Greedy** approach because we process bars as soon as their boundaries are known, and we never have to look back more than once.

---

## 7. Comparative Analysis: Stack vs. Divide & Conquer

| Metric | Monotonic Stack | Divide & Conquer (Segment Tree) |
| :--- | :--- | :--- |
| **Time Complexity** | O(N) | O(N \log N) |
| **Space Complexity**| O(N) | O(N) |
| **Simplicity** | High (Once understood) | Low (Template-heavy) |
| **Best For** | Standard Histograms | Dynamic/Updating Histograms |

---

## 8. Real-world Applications: Digital Signal Processing

In our **Speech Tech** post today, we discuss **Scaling**. In Signal Processing:
- **Peak Identification**: Histograms of audio energy levels are used to detect noise floors. Finding the "Largest Rectangle" can help identify sustained audio bursts (like a siren or a constant hum) that need to be filtered out.
- **Image Thresholding**: In computer vision (OCR), we use histograms of pixel intensity to find the "Optimal Cut" for separating text from background.

---

## 9. Interview Strategy: The "Zero Padding" Trick

1. **Explain the Exhaustion**: Mention that without the `0` at the end, the stack might keep some bars (like `[1, 2, 3]`) forever. The `0` acts as a "Sentinel" that forces all bars to be popped and processed.
2. **Width Calculation**: Be very precise about `i - stack[-1] - 1`. Draw it on a board. If `stack[-1]` is `-1` and `i` is `1`, width is `1 - (-1) - 1 = 1`, which is correct for the first bar.
3. **Trace an Example**: Use `[2, 1, 2]`. Show how '2' is pushed, then '1' causes '2' to be popped and processed, then '1' is pushed.

---

## 10. Common Pitfalls

1. **Index vs Value**: Store the **index** in the stack, but compare the **value** at that index. Storing the value alone makes width calculation impossible.
2. **Strictly Monotonic vs Monotonic**: If heights are equal (`[2, 2, 2]`), you can either pop or keep. Popping is usually cleaner as it handles the area calculation incrementally, but the formula `i - stack[-1] - 1` correctly handles "Plateaus" even if you don't pop until a strictly smaller bar is seen.

---

## 11. Key Takeaways

1. **Maintain Sorted Property**: Stacks aren't just for "Last In First Out"; they are for "Order Preservation."
2. **Right Boundary is the Trigger**: The logic only executes when a constraint is violated (a shorter bar is found).
3. **Linear Complexity is the Goal**: (The ML Link) In infrastructure scaling, O(N^2) processes are the "Killers" of reliability. Mastering O(N) algorithms is a prerequisite for high-scale engineering.

---

**Originally published at:** [arunbaby.com/dsa/0057-largest-rectangle-in-histogram](https://www.arunbaby.com/dsa/0057-largest-rectangle-in-histogram/)

*If you found this helpful, consider sharing it with others who might benefit.*
