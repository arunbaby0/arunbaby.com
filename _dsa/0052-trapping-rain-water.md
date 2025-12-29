---
title: "Trapping Rain Water"
day: 52
collection: dsa
categories:
  - dsa
tags:
  - two-pointers
  - monotonic-stack
  - arrays
  - hard
  - pattern-recognition
  - prefix-max
difficulty: Hard
subdomain: "Two Pointers & Invariants"
tech_stack: Python
scale: "O(N) time, O(1) space (two pointers)"
companies: Google, Meta, Amazon, Microsoft
related_ml_day: 52
related_speech_day: 52
related_agents_day: 52
---

**"Water doesn’t care about every bar—only the highest walls to the left and right."**

## 1. Problem Statement

You are given an array `height` where `height[i]` represents the height of a bar at position `i`. Imagine rain falls, and the bars form a histogram. Return how many units of water can be trapped after raining.

**Example 1**
- Input: `height = [0,1,0,2,1,0,1,3,2,1,2,1]`
- Output: `6`

**Example 2**
- Input: `height = [4,2,0,3,2,5]`
- Output: `9`

**Key constraints**
- `0 <= height[i]`
- We want an efficient solution: ideally **O(N)**.

---

## 2. Understanding the Problem

### 2.1 The local rule (water at one index)

At index `i`, the water level is limited by the highest wall to the left and right:

\[
water[i] = \max\left(0,\ \min(\maxLeft[i],\maxRight[i]) - height[i]\right)
\]

Where:
- `maxLeft[i]` = maximum height in `height[0..i]`
- `maxRight[i]` = maximum height in `height[i..n-1]`

This formula is the whole problem. Every solution is just a different way to compute `maxLeft` and `maxRight` efficiently.

### 2.2 Why this is a “pattern recognition” question

The trap: you might think you need to simulate water “filling” across the bars. You don’t.
Water at position `i` is determined by two boundaries (left/right maxima), not by global simulation.

This “boundary thinking” shows up everywhere:
- in anomaly detection you don’t inspect every raw event; you look for boundary violations of normal patterns
- in long-context agents you can’t keep everything; you keep boundaries (summaries, constraints, invariants) that preserve correctness

### 2.3 Visual intuition (why valleys matter, not slopes)

A useful mental picture:
- water collects only where there exists a higher wall on both sides
- the amount depends on the *lower* of the two walls

```
heights:  3       3
          |       |
          |   1   |
          | _ _ _ |
index:    0 1 2 3 4
```

At index 2:
- left max = 3, right max = 3
- water = 3 - 1 = 2

The important lesson: you can compute the final trapped water purely from boundaries.

### 2.4 Why the minimum of two maxima is inevitable

If left boundary is 10 and right boundary is 3, water leaks at 3.
So the left wall being 10 doesn’t help beyond 3.
That’s why the formula uses:
\[
\min(\maxLeft[i],\maxRight[i])
\]

This exact “lower boundary dominates” idea appears in:
- “Container With Most Water” (area limited by the shorter line)
- “Trapping Rain Water” (level limited by the shorter wall)

---

## 3. Approach 1: Brute Force (Scan Left and Right for Each Index)

### 3.1 Idea
For each `i`:
- scan left to find `left_max`
- scan right to find `right_max`
- add `min(left_max, right_max) - height[i]` if positive

### 3.2 Complexity
- **Time**: \(O(N^2)\) (because each index does two scans)
- **Space**: \(O(1)\)

### 3.3 When it’s still useful
It’s a great way to derive the formula and validate understanding.
But it’s too slow for large `N`.

### 3.4 Brute force implementation (for completeness)

```python
from typing import List


class SolutionBrute:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        water = 0
        for i in range(n):
            left_max = 0
            for l in range(i, -1, -1):
                left_max = max(left_max, height[l])
            right_max = 0
            for r in range(i, n):
                right_max = max(right_max, height[r])
            water += max(0, min(left_max, right_max) - height[i])
        return water
```

This is slow but “obviously correct”, which makes it valuable for validating optimized approaches.

---

## 4. Approach 2: Precompute Prefix/Suffix Max (DP)

### 4.1 Idea
Build:
- `left_max[i] = max(height[0..i])`
- `right_max[i] = max(height[i..n-1])`

Then compute trapped water using the formula from section 2.

### 4.2 Complexity
- **Time**: \(O(N)\)
- **Space**: \(O(N)\) for the two arrays

### 4.3 Implementation (clear and interview-safe)

```python
from typing import List


class SolutionDP:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        if n == 0:
            return 0

        left_max = [0] * n
        right_max = [0] * n

        left_max[0] = height[0]
        for i in range(1, n):
            left_max[i] = max(left_max[i - 1], height[i])

        right_max[n - 1] = height[n - 1]
        for i in range(n - 2, -1, -1):
            right_max[i] = max(right_max[i + 1], height[i])

        water = 0
        for i in range(n):
            water += max(0, min(left_max[i], right_max[i]) - height[i])
        return water
```

This solution is easy to reason about and is often acceptable if memory is not a concern.

### 4.4 Space optimization note

You can reduce memory by computing one side first (e.g., `right_max`), then scanning while maintaining the other side (`left_max`) on the fly.
In interviews, the more meaningful memory optimization is usually the next approach: **two pointers** with truly O(1) extra space.

---

## 5. Approach 3: Optimal Two Pointers (O(1) Space)

### 5.1 Core insight

We don’t actually need the full `left_max[]` and `right_max[]` arrays.
As we move inward with two pointers (`l`, `r`), we maintain:
- `left_max = max(height[0..l])`
- `right_max = max(height[r..n-1])`

Then:
- If `left_max <= right_max`, water at `l` is fully determined by `left_max` (because the right side has a boundary at least as tall).
  - Add `left_max - height[l]`, move `l`.
- Else, water at `r` is determined by `right_max`.
  - Add `right_max - height[r]`, move `r`.

This is the crucial invariant:
> The smaller boundary decides the water level for the pointer on that side.

### 5.2 Why it works (invariant explanation)

Suppose `left_max <= right_max`.
For index `l`, the right boundary is **at least** `right_max`, and since `right_max >= left_max`, the limiting height is `left_max`.
So `water[l] = left_max - height[l]` (if positive), and it’s safe to finalize `l` without knowing future right details.

Symmetrically for the right side.

### 5.3 Alternative formulation (compare boundary maxima directly)

Some implementations use:
- if `left_max <= right_max`, process left
- else process right

This is equivalent in spirit, and sometimes feels cleaner because it matches the proof statement.
Pseudo-logic:
- update `left_max` and `right_max` as you move pointers
- whichever side has smaller max is the bottleneck, finalize that side

Why it’s equivalent:
- the algorithm’s correctness hinges on “smaller boundary decides”
- whether you compare the current bars or the running maxima, you’re still choosing the side that is guaranteed to be bounded by its own max

### 5.4 Common off-by-one / tie-handling guidance

Two-pointer algorithms often fail due to tie handling.
Practical rules:
- it’s safe to treat ties (`height[l] == height[r]`) as “process left” (or consistently process one side)
- update the boundary max **before** adding trapped water on that index

If you are consistent, both tie strategies work; inconsistency causes subtle bugs.

### 5.5 Complexity
- **Time**: \(O(N)\)
- **Space**: \(O(1)\)

### 5.6 Step-by-step dry run (build confidence)

Example: `height = [4,2,0,3,2,5]`

Start:
- `l=0 (4)`, `r=5 (5)`
- `left_max=0`, `right_max=0`, `water=0`

Since `height[l] <= height[r]`, finalize left:
- update `left_max=4`, move `l=1`

`l=1 (2)`:
- add `4-2=2` → `water=2`, move `l=2`

`l=2 (0)`:
- add `4-0=4` → `water=6`, move `l=3`

`l=3 (3)`:
- add `4-3=1` → `water=7`, move `l=4`

`l=4 (2)`:
- add `4-2=2` → `water=9`, move `l=5`, stop.

This trace is a great interview sanity check because it shows the algorithm is “finalizing one side” rather than guessing.

### 5.7 Implementation (recommended)

```python
from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        if n == 0:
            return 0

        l, r = 0, n - 1
        left_max, right_max = 0, 0
        water = 0

        while l < r:
            if height[l] <= height[r]:
                # Left side is the limiting boundary (or tied).
                if height[l] >= left_max:
                    left_max = height[l]
                else:
                    water += left_max - height[l]
                l += 1
            else:
                # Right side is the limiting boundary.
                if height[r] >= right_max:
                    right_max = height[r]
                else:
                    water += right_max - height[r]
                r -= 1

        return water
```

Note: some implementations compare `left_max` vs `right_max`; others compare `height[l]` vs `height[r]`. Both can be correct if invariants are maintained consistently.

---

## 6. Approach 4 (Optional): Monotonic Stack

### 6.1 Idea
Use a stack to keep indices of bars in decreasing height order.
When you find a bar taller than the stack top, you can “close a container” and compute trapped water for the valley.

This approach is also \(O(N)\) but uses \(O(N)\) space.
It’s useful when you want to compute water in terms of “basins”.

### 6.1.1 How the stack computes water (width × bounded height)

When the current bar at `i` is higher than the bar at the stack top, you’ve found a right boundary that can “close” a basin.

Terminology in the code:
- `bottom`: the valley bar index we just popped
- `left`: the new stack top after popping (`left` boundary)
- `i`: the current index (right boundary)

Then:
- **width** of trapped region is `i - left - 1`
  - everything between `left` and `i` is inside the basin
- **bounded height** is `min(height[left], height[i]) - height[bottom]`
  - the water level is limited by the shorter boundary
  - subtract the valley floor (`bottom`) height

So trapped water for that basin segment:
\[
water += width \times bounded\_height
\]

Why it’s still O(N):
- every index is pushed once
- every index is popped at most once

### 6.1.2 Quick dry run (stack intuition)

Example: `[2,0,2]`
- i=0, stack=[0]
- i=1 (0), stack=[0,1]
- i=2 (2), height[1] < 2 → pop bottom=1
  - left=0, width=2-0-1=1
  - bounded_height=min(2,2)-0=2
  - water += 1*2 = 2

This is the same answer as two pointers, just computed via “closing basins” rather than “finalizing sides”.

### 6.2 Implementation (stack)

```python
from typing import List


class SolutionStack:
    def trap(self, height: List[int]) -> int:
        stack = []
        water = 0

        for i, h in enumerate(height):
            while stack and height[stack[-1]] < h:
                bottom = stack.pop()
                if not stack:
                    break
                left = stack[-1]
                width = i - left - 1
                bounded_height = min(height[left], h) - height[bottom]
                water += width * bounded_height
            stack.append(i)

        return water
```

### 6.3 When to prefer stack vs two pointers

- **Two pointers**: simplest invariant, minimal memory; best default.
- **Stack**: basin-based view; very reusable for histogram-style problems.

---

## 7. Testing (Edge Cases Included)

### 7.1 Basic tests
- `[]` → `0`
- `[0]` → `0`
- `[1,0,1]` → `1`
- `[2,0,2]` → `2`

### 7.2 Monotonic arrays
- `[0,1,2,3]` → `0` (no right boundary for valleys)
- `[3,2,1,0]` → `0` (no left boundary)

### 7.3 Plateaus and duplicates
- `[3,3,3]` → `0`
- `[3,0,0,3]` → `6`

### 7.4 Provided examples
- `[0,1,0,2,1,0,1,3,2,1,2,1]` → `6`
- `[4,2,0,3,2,5]` → `9`

### 7.5 Differential testing (production)
Compare:
- `Solution().trap(height)` (two pointers)
- `SolutionDP().trap(height)` (prefix/suffix)
on randomized arrays.

### 7.6 Quick randomized differential test

```python
import random


def randomized_test(trials: int = 200) -> None:
    fast = Solution()
    slow = SolutionDP()
    for _ in range(trials):
        n = random.randint(0, 50)
        arr = [random.randint(0, 10) for _ in range(n)]
        if fast.trap(arr) != slow.trap(arr):
            raise AssertionError((arr, fast.trap(arr), slow.trap(arr)))


if __name__ == "__main__":
    randomized_test()
    print("OK")
```

---

## 8. Complexity Analysis

Let \(N = len(height)\).

- **Brute force**: \(O(N^2)\) time, \(O(1)\) space
- **Prefix/suffix DP**: \(O(N)\) time, \(O(N)\) space
- **Two pointers**: \(O(N)\) time, \(O(1)\) space
- **Monotonic stack**: \(O(N)\) time, \(O(N)\) space

---

## 9. Production Considerations

### 9.1 What this pattern teaches

The two-pointer solution is about maintaining a small set of invariants while shrinking the search space:
- left boundary maximum
- right boundary maximum
- and a rule that lets you finalize one side each step

That “finalize safely with partial information” mindset is exactly what you want in high-scale systems:
- streaming computations
- anomaly detection thresholds (finalize alerts based on safe bounds)
- long-context agents (finalize summaries and constraints, keep bounded state)

### 9.2 Numerical stability and data types
If heights are large, use a type that won’t overflow in languages like Java/C++ (use 64-bit).
In Python you’re safe.

### 9.3 Visual debugging tip
If you’re debugging, plot the array and track `(l, r, left_max, right_max)` each step.
Most bugs are “wrong invariant update”.

### 9.4 A proof-style intuition (why “finalize one side” is safe)

The two-pointer algorithm feels almost magical until you internalize the proof idea:

- When `height[l] <= height[r]`, there exists a right boundary at least as tall as `height[l]` (the bar at `r` itself).
- But the *actual* limiting boundary for `l` is the maximum on the left, `left_max`, compared against the best available boundary on the right.
- If we move left-to-right maintaining `left_max`, and we know the right side has “enough wall” to not be the bottleneck (because `height[r]` is already >= `height[l]`), then any future right walls can only be higher; they cannot reduce the amount of water `l` can hold.

So water at `l` can be computed immediately as:
- `left_max - height[l]` if positive

Symmetrically, when `height[r] < height[l]`, finalize the right side with `right_max`.

This is a common proof pattern:
> If you can show the unknown future cannot reduce your computed value, you are safe to finalize now.

### 9.5 Follow-up: 2D trapping rain water (what changes)

Interviewers sometimes ask about the 2D variant (“Trapping Rain Water II”).
The core change:
- boundaries are not just left/right; they are in 4 directions
- the right tool becomes a **min-heap (priority queue)** seeded with boundary cells
- you expand inward like Dijkstra:
  - the current boundary height determines how much water a neighbor can hold

The shared theme is still boundary invariants:
- in 1D you maintain two boundary maxima
- in 2D you maintain the minimum boundary around the explored region via a heap

Mentioning this shows you understand the deeper principle, not just the 1D trick.

### 9.6 Engineering checklist (if you were reviewing this code)

If this logic is in a production code path (analytics, simulation, or interview platform), review:

- **Inputs**
  - empty list returns 0
  - non-negative heights (if negatives allowed, define behavior explicitly)

- **Invariants**
  - `left_max` is updated before accumulating water at `l`
  - `right_max` is updated before accumulating water at `r`
  - pointer movement is consistent on ties

- **Complexity**
  - one pass only (no nested loops accidentally introduced)
  - constant extra memory

- **Correctness sanity**
  - water is never negative
  - output fits in 64-bit for worst-case constraints in typed languages

This is a good practice for any invariant-heavy algorithm: write down the invariants you rely on and ensure the code matches them.

### 9.7 Relationship to “monotonic stack” problems

It’s worth recognizing that “Trapping Rain Water” sits in a family:
- **Largest Rectangle in Histogram**: stack to find nearest smaller bars
- **Next Greater Element**: stack for boundary discovery
- **Trapping Rain Water**: stack for basin closure

Learning these together is efficient because the same stack invariants reappear.

---

## 10. Connections to ML Systems

Today’s shared theme is **pattern recognition**:
- Here, the “pattern” is the boundary rule that determines water.
- In anomaly detection, the “pattern” is normal behavior and deviations.
- In speech anomaly detection, the “pattern” is stable acoustics; anomalies are dropouts, clipping, or distribution shifts.
- In long-context agent strategies, the “pattern” is which information remains relevant over long horizons (summaries + constraints) and which can be compressed.

If you can solve “Trapping Rain Water” cleanly, it’s usually a sign you’re good at:
- identifying invariants
- avoiding unnecessary simulation
- reducing a problem to a small boundary state

---

## 11. Interview Strategy

1. Start from the per-index formula with `maxLeft` and `maxRight`.
2. Offer the DP precompute solution quickly.
3. Then derive why two pointers can finalize one side at a time (the key invariant).
4. Mention monotonic stack as an alternative \(O(N)\) approach.

Common mistakes:
- mixing conditions (`height[l] <= height[r]` vs `left_max <= right_max`) without consistent updates
- forgetting to update `left_max/right_max` before computing trapped water
- off-by-one in stack width calculations

### 11.1 The “explain like I’m debugging” move

If the interviewer pushes on correctness, don’t hand-wave.
Use the boundary argument:
- “When I move the left pointer, I’m asserting the right boundary is not the bottleneck.”
- “So the only thing that matters is the best left boundary so far (`left_max`).”
- “That’s why I can finalize water at this position and never revisit it.”

This style of explanation is often what differentiates a correct solution from an excellent interview performance.

### 11.2 Common follow-ups

- **“Can you do it in one pass without extra arrays?”**
  - Yes: the two-pointer approach you implemented.
- **“How would you extend this to 2D?”**
  - Use a min-heap seeded with boundary cells; expand inward (Dijkstra-like).
- **“What’s the relationship to monotonic stacks?”**
  - Both find boundaries efficiently; the stack explicitly finds “basins”.

---

## 12. Key Takeaways

1. **Water at i is bounded by left/right maxima**: the formula is everything.
2. **Two pointers work because the smaller boundary decides**: finalize one side safely.
3. **This is invariant-based thinking**: compress a global phenomenon into a tiny state machine.

### 12.1 One-sentence invariant

At every step:
- maintain `left_max` and `right_max` as boundary maxima
- compute water on the side with the smaller boundary
- move that pointer inward

That’s the whole algorithm.

### 12.2 Appendix: a compact cheat sheet (what to remember)

If you want a quick “mental card” for this problem:

- **Core formula**
  - \(water[i] = \max(0, \min(maxLeft[i], maxRight[i]) - height[i])\)

- **DP arrays**
  - compute `left_max[]` prefix maxima
  - compute `right_max[]` suffix maxima
  - single pass to sum water

- **Two pointers**
  - maintain `left_max`, `right_max`
  - move the pointer on the side that is guaranteed to be bounded
  - finalize water for that pointer immediately

- **Monotonic stack**
  - maintain decreasing stack of indices
  - when you see a higher bar, pop “bottom” and compute:
    - width = `i - left - 1`
    - bounded_height = `min(height[left], height[i]) - height[bottom]`

### 12.3 Appendix: practice variants that reuse the same ideas

These problems reuse the same “boundary/invariant” thinking:
- “Container With Most Water” (two pointers, lower boundary dominates)
- “Largest Rectangle in Histogram” (monotonic stack, boundary discovery)
- “Trapping Rain Water II” (min-heap boundary expansion in 2D)

If you learn them as a group, you build a reusable toolkit:
- boundary maxima
- monotonic stacks
- priority-queue boundary expansion

### 12.4 Appendix: what to say if asked “why not simulate?”

A crisp answer:
> Simulation is expensive and unnecessary. The final water level is determined by boundaries; once you know the boundaries, you can compute the result directly.

That’s the same mindset that scales in systems engineering:
- compute summaries and invariants
- avoid materializing everything
- finalize results when future information can’t change them

### 12.5 Appendix: sketch of the 2D solution (what you’d say in an interview)

For the 2D variant (“Trapping Rain Water II”), left/right maxima don’t apply.
Instead, you treat the outer boundary as the initial “wall” and expand inward:

1. Push all boundary cells into a **min-heap** keyed by height.
2. Mark them visited.
3. Pop the smallest boundary cell.
4. For each neighbor not visited:
   - if neighbor height is lower, it holds `boundary_height - neighbor_height` water
   - push neighbor with effective height `max(neighbor_height, boundary_height)`
5. Repeat until all cells are visited.

Why it works:
- you always expand from the currently lowest boundary, so you never “miss” a leak path
- the heap stores the best known boundary height for the frontier

Complexity:
- \(O(MN \log(MN))\) time for an \(M \times N\) grid
- \(O(MN)\) space for visited + heap

### 12.6 Appendix: choosing between two pointers and stack (how to decide)

If you’re writing production code:
- choose **two pointers** for simplicity and minimal memory
- choose **stack** if you also need explicit basin boundaries for downstream logic

In interviews:
- two pointers is usually the best default answer
- stack is a strong “alternative” mention that shows breadth

### 12.7 Appendix: rapid-fire interview Q&A

- **Q: Why is the time O(N) for the two-pointer method?**  
  **A:** Each pointer moves inward at most `N` times total; no index is revisited.

- **Q: Why do we only need the smaller boundary?**  
  **A:** Water level is capped by the minimum of left and right boundaries; if one side is already the bottleneck, the other side’s exact future cannot reduce the computed water.

- **Q: What happens with flat plateaus?**  
  **A:** Plateaus are fine. The algorithm treats equal heights consistently; choose a tie rule and stick to it.

- **Q: Can heights be negative?**  
  **A:** The classic problem assumes non-negative. If negatives are allowed, you must define semantics (negative bars imply “below ground”), and the formula still works, but test expectations must be clarified.

- **Q: When would you prefer DP arrays?**  
  **A:** When code clarity matters more than memory (small `N`, or memory is cheap), or when you want explicit `left_max/right_max` arrays for debugging/visualization.

### 12.8 Appendix: the one mistake to avoid

The most common bug in two pointers is updating the max after adding water (wrong order).
Always:
- update the boundary max first
- then accumulate `max - height[i]` if positive

If you follow that ordering, the algorithm becomes hard to break.

### 12.9 Appendix: quick complexity intuition (why O(N) is optimal)

In the comparison model, you must at least look at each bar once to know whether it contributes to trapped water.
So \(\Omega(N)\) is a natural lower bound.

Two pointers (and stack) reach this lower bound:
- each index is processed a constant number of times
- no nested scanning

This is another reason the problem is popular:
it rewards the ability to transform a naive \(O(N^2)\) process (“scan left and right for every i”) into an optimal linear pass by recognizing the right invariant.

### 12.10 Appendix: “explain it to production” in one paragraph

If someone asks why this algorithm is safe in production:
- It processes each input element once (linear time), so it scales to large arrays.
- It uses constant extra memory, so it behaves well under memory pressure.
- The correctness is driven by explicit invariants (`left_max`, `right_max` and “smaller boundary decides”), which are easy to test and reason about.
- It’s also easy to validate via differential testing against the DP baseline on randomized inputs, which catches subtle bugs quickly.

This is exactly the kind of algorithm you want in real systems: predictable, bounded, and testable.

### 12.11 Appendix: micro-optimizations (rarely needed, but good to know)

In Python, this is already fast enough for typical constraints. In lower-level languages or extremely hot loops:
- use integer types with enough headroom (64-bit for the accumulator)
- avoid repeated min/max computations inside tight loops where possible
- prefer simple branching over complex conditionals for branch predictability

Most of the time, the “optimization” that matters is algorithmic:
choosing O(N) two pointers over O(N^2) scanning.

### 12.12 Appendix: one more way to sanity-check results

A simple correctness sanity check:
- If you reverse the array, the trapped water should be the same.

Reason:
the physics and boundary logic are symmetric under reversal.

So in testing, you can assert:
- `trap(height) == trap(list(reversed(height)))`

This isn’t a proof, but it’s a strong “cheap invariant” that catches some implementation bugs.

---

**Originally published at:** [arunbaby.com/dsa/0052-trapping-rain-water](https://www.arunbaby.com/dsa/0052-trapping-rain-water/)

*If you found this helpful, consider sharing it with others who might benefit.*

