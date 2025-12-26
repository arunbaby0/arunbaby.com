---
title: "Median of Two Sorted Arrays"
day: 51
collection: dsa
categories:
  - dsa
tags:
  - binary-search
  - divide-and-conquer
  - arrays
  - partition
  - hard
  - interview-classic
difficulty: Hard
subdomain: "Binary Search on Answer"
tech_stack: Python
scale: "O(log(min(m,n))) time, O(1) space"
companies: Google, Meta, Amazon, Microsoft
related_dsa_day: 51
related_ml_day: 51
related_speech_day: 51
related_agents_day: 51
---
**"Stop thinking ‘merge’. Think ‘partition’—the median is just the boundary between two halves."**

## 1. Problem Statement

You are given two sorted arrays `nums1` and `nums2` (possibly different lengths). Return the **median** of the combined multiset of numbers.

**Median definition**:
- If total count \(N\) is odd, median is the middle element.
- If \(N\) is even, median is the average of the two middle elements.

**Example 1**
- Input: `nums1 = [1,3]`, `nums2 = [2]`
- Combined: `[1,2,3]`
- Output: `2.0`

**Example 2**
- Input: `nums1 = [1,2]`, `nums2 = [3,4]`
- Combined: `[1,2,3,4]`
- Output: `(2+3)/2 = 2.5`

**Constraints (typical interview)**
- Arrays are individually sorted (non-decreasing).
- Lengths can be imbalanced (e.g., `m=1`, `n=10^6`).
- You’re expected to do better than merging both arrays.

---

## 2. Understanding the Problem

### 2.1 Why this is hard (and why it’s famous)

Merging two sorted arrays is easy: two pointers, \(O(m+n)\).
But the “legendary” follow-up is: can you do it in **logarithmic time**?

That requirement forces a mental shift:
- Don’t build the combined array.
- Don’t scan all items.
- Instead, **search for a boundary condition** that defines the median.

This is the same mental model you use in:
- **Distributed systems**: you don’t pull all logs to one box; you query boundaries/aggregations.
- **Federated learning**: you don’t centralize raw data; you coordinate updates to reach a global optimum.
- **Privacy-preserving speech**: you can’t centralize raw audio, so you must compute what you need without seeing everything.
- **Knowledge graphs for agents**: you can’t traverse the entire graph; you rely on indexes and targeted queries.

### 2.2 The median as a partition

Imagine combining both arrays into one sorted list of length \(N = m+n\).
Let:
- \(L\) be the left half
- \(R\) be the right half

We want:
- \(|L| = \left\lfloor\frac{N+1}{2}\right\rfloor\) (left has the extra item when odd)
- Every element in \(L\) \(\le\) every element in \(R\)

If we can find such a partition, the median is trivial:
- If \(N\) is odd: median is `max(L)`
- If \(N\) is even: median is `(max(L) + min(R)) / 2`

So the task becomes:
> Find a partition across two arrays such that left sizes match and ordering holds.

---

## 3. Approach 1: Brute Force (Merge Then Median)

### 3.1 Idea
Merge `nums1` and `nums2` into a new sorted array, then compute the median.

### 3.2 Complexity
- **Time**: \(O(m+n)\)
- **Space**: \(O(m+n)\) if you build the merged array

### 3.3 Why it’s not enough
If `n=10^6`, you just did a million operations for a single median. Interviews want you to recognize that median depends on a **boundary**, not on every value.

---

## 4. Approach 2: Two Pointers Without Full Merge (Streaming Median)

### 4.1 Idea
Use two pointers to “merge” but stop once you reach the median position(s). You only need:
- the \(k\)-th element (and maybe \(k-1\)-th)
where \(k = \lfloor N/2 \rfloor\).

### 4.2 Complexity
- **Time**: \(O(m+n)\) worst case, but typically \(O(N/2)\)
- **Space**: \(O(1)\)

### 4.3 When this is acceptable
This is a strong solution if:
- constraints are small
- or you’re in production and the constant factors matter more than asymptotics

But it still fails the “log time” requirement.

### 4.4 Implementation (constant space, early stop)

This is a great “interview warm-up” implementation because it’s simple and correct.
It also becomes the baseline for differential testing against the optimal method.

```python
from typing import List


def median_two_sorted_stream(nums1: List[int], nums2: List[int]) -> float:
    """
    Merge-like scan until the median position. O(m+n) time in worst case, O(1) space.
    """
    m, n = len(nums1), len(nums2)
    total = m + n
    mid = total // 2

    i = j = 0
    prev = curr = 0

    # We only need to advance 'mid' steps.
    for _ in range(mid + 1):
        prev = curr
        if i < m and (j >= n or nums1[i] <= nums2[j]):
            curr = nums1[i]
            i += 1
        else:
            curr = nums2[j]
            j += 1

    if total % 2 == 1:
        return float(curr)
    return (prev + curr) / 2.0
```

Edge case check:
- if `nums1` is empty, we just walk `nums2`
- if duplicates exist, the `<=` tie-break maintains correctness

---

## 5. Approach 3: Optimal (Binary Search on the Partition)

### 5.1 Key insight
Instead of searching the median value, we search **where the cut happens** in the smaller array.

Let:
- `A` = the smaller array (length `m`)
- `B` = the larger array (length `n`)

We choose an index `i` as the cut in `A`:
- left part is `A[0:i]`
- right part is `A[i:]`

The cut in `B` must then be `j` such that left sizes match:

\[
i + j = \left\lfloor\frac{m+n+1}{2}\right\rfloor
\]

So:
\[
j = \left\lfloor\frac{m+n+1}{2}\right\rfloor - i
\]

Now define boundary values (with sentinels for empty sides):
- `A_left_max = A[i-1]` if `i>0` else `-∞`
- `A_right_min = A[i]` if `i<m` else `+∞`
- `B_left_max = B[j-1]` if `j>0` else `-∞`
- `B_right_min = B[j]` if `j<n` else `+∞`

The partition is valid if:
- `A_left_max <= B_right_min`
- `B_left_max <= A_right_min`

If invalid, we adjust `i` using binary search:
- If `A_left_max > B_right_min`, we cut too far right in `A` → move left (`hi = i-1`)
- Else if `B_left_max > A_right_min`, we cut too far left in `A` → move right (`lo = i+1`)

This is classic **binary search on a monotonic condition**: once you move the partition, the violations move predictably.

### 5.2 Why binary search works (monotonicity)
As `i` increases:
- `A_left_max` increases (or stays)
- `A_right_min` increases (or stays)
- `j` decreases, so `B_left_max` tends to decrease, `B_right_min` tends to decrease

So the inequality flips in a single direction—perfect for binary search.

### 5.3 Complexity
- **Time**: \(O(\log(\min(m,n)))\)
- **Space**: \(O(1)\)

This is the “correct” interview answer.

### 5.4 A concrete picture (the “four numbers” you compare)

The entire algorithm can be explained with **four boundary values**:

```
A: [ ... A[i-1] | A[i] ... ]
           ^        ^
           |        |
     A_left_max   A_right_min

B: [ ... B[j-1] | B[j] ... ]
           ^        ^
           |        |
     B_left_max   B_right_min
```

We want the left partition to contain exactly half of the total elements (rounded up):

\[
i + j = \left\lfloor\frac{m+n+1}{2}\right\rfloor
\]

And we want **all left elements** \(\le\) **all right elements**.
Because each side is already sorted, we only need to check the boundary:

- Left side max is `max(A_left_max, B_left_max)`
- Right side min is `min(A_right_min, B_right_min)`

So the partition is valid if and only if:

- `A_left_max <= B_right_min`
- `B_left_max <= A_right_min`

Once you say this out loud, the solution becomes “search for `i`”.

### 5.5 Step-by-step dry run (so it feels inevitable)

Example:
- `A = [1, 3]`
- `B = [2, 4, 5]`

Total \(N=5\), left size \(=(5+1)//2 = 3\).

Binary search on `i` in `A` (`m=2`):

**Try `i=1`**
- `j = 3 - 1 = 2`
- `A_left_max = A[0]=1`
- `A_right_min = A[1]=3`
- `B_left_max = B[1]=4`
- `B_right_min = B[2]=5`

Check:
- `A_left_max <= B_right_min` → `1 <= 5` ✓
- `B_left_max <= A_right_min` → `4 <= 3` ✗

Second inequality fails ⇒ we took **too few** from `A` (right side of A is too small).
Move right: `lo = i+1`.

**Try `i=2`**
- `j = 3 - 2 = 1`
- `A_left_max = A[1]=3`
- `A_right_min = +inf` (because i==m)
- `B_left_max = B[0]=2`
- `B_right_min = B[1]=4`

Check:
- `3 <= 4` ✓
- `2 <= +inf` ✓

Valid partition.
Total is odd ⇒ median is `max(A_left_max, B_left_max)=max(3,2)=3`.
Combined array is `[1,2,3,4,5]` ⇒ correct.

### 5.6 Proof sketch (why the inequalities are sufficient)

To prove correctness, we need to show:
1. If the inequalities hold, then every element on the left is \(\le\) every element on the right.
2. If the inequalities do not hold, we can move `i` in the correct direction.

**(1) Sufficiency**
- In `A`, everything left of `i` is \(\le A_left_max\).
- In `B`, everything left of `j` is \(\le B_left_max\).
So every left element is \(\le \max(A_left_max, B_left_max)\).

Similarly:
- In `A`, everything right of `i` is \(\ge A_right_min\).
- In `B`, everything right of `j` is \(\ge B_right_min\).
So every right element is \(\ge \min(A_right_min, B_right_min)\).

If `A_left_max <= B_right_min` and `B_left_max <= A_right_min`, then:
\[
\max(A_left_max, B_left_max) \le \min(A_right_min, B_right_min)
\]
which implies the left max is \(\le\) right min, therefore all left \(\le\) all right.

**(2) Direction of movement**
- If `A_left_max > B_right_min`, we took too many from `A`; moving `i` left decreases `A_left_max` and increases `B_right_min`.
- Else (the remaining case), `B_left_max > A_right_min`, we took too few from `A`; moving `i` right increases `A_right_min` and decreases `B_left_max`.

That monotonic behavior makes binary search valid.

### 5.7 An alternative “k-th element” perspective (useful in interviews)

Another way to think about the median is “find the \(k\)-th smallest element”, where:
- \(k = \left\lfloor \frac{N+1}{2} \right\rfloor\) for odd (the median)
- and also the \((k+1)\)-th for even

There’s a classic divide-and-conquer method that discards \(k/2\) elements per step (like binary search over ranks).
Even if you don’t code it, mentioning it shows depth:
- it generalizes to other selection problems
- it reinforces the “don’t merge” principle

The partition method you implemented is effectively a highly optimized, constant-space specialization of selection on two sorted arrays.

---

## 6. Implementation (Fully Commented Python)

```python
from typing import List


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        """
        Find the median of two sorted arrays in O(log(min(m, n))) time and O(1) space.

        Idea:
        - Binary search the partition index i in the smaller array A.
        - Derive j in the larger array B so that left side has half the elements.
        - Check partition validity via boundary comparisons.
        """

        # Ensure nums1 is the smaller array to minimize binary search space.
        A, B = nums1, nums2
        if len(A) > len(B):
            A, B = B, A

        m, n = len(A), len(B)

        # Left side size: (m+n+1)//2 ensures left gets the extra when total is odd.
        left_size = (m + n + 1) // 2

        lo, hi = 0, m  # i ranges over [0..m] inclusive boundaries

        NEG_INF = float("-inf")
        POS_INF = float("inf")

        while lo <= hi:
            i = (lo + hi) // 2
            j = left_size - i

            # Collect boundary values with sentinels for empty partitions.
            A_left_max = A[i - 1] if i > 0 else NEG_INF
            A_right_min = A[i] if i < m else POS_INF

            B_left_max = B[j - 1] if j > 0 else NEG_INF
            B_right_min = B[j] if j < n else POS_INF

            # Check if partition is valid.
            if A_left_max <= B_right_min and B_left_max <= A_right_min:
                # We found the correct partition.
                if (m + n) % 2 == 1:
                    # Odd total: median is the max of left side.
                    return float(max(A_left_max, B_left_max))

                # Even total: median is average of the two middle values.
                left_max = max(A_left_max, B_left_max)
                right_min = min(A_right_min, B_right_min)
                return (left_max + right_min) / 2.0

            # Partition is invalid; adjust binary search.
            if A_left_max > B_right_min:
                # We took too many from A; move left.
                hi = i - 1
            else:
                # We took too few from A; move right.
                lo = i + 1

        # If input arrays are sorted, we should never reach here.
        raise ValueError("Invalid input: arrays are not sorted or other invariant broken.")
```

---

## 7. Testing (Edge Cases Included)

### 7.1 Minimal cases
- `A=[]`, `B=[1]` → median `1`
- `A=[]`, `B=[1,2]` → median `1.5`

### 7.2 Highly imbalanced lengths
- `A=[1000000]`, `B=[1,2,3,4,5,6,7,8,9]`
  - Median should be `5` (combined length 10, middle two are 5 and 6 → 5.5?) Actually combined sorted is `[1..9, 1000000]`, median is `(5+6)/2=5.5`.

### 7.3 Duplicates
- `A=[1,1,1]`, `B=[1,1]` → median `1`

### 7.4 Negative numbers
- `A=[-5,-3,-1]`, `B=[-2]` → combined `[-5,-3,-2,-1]` → median `(-3 + -2)/2 = -2.5`

### 7.5 Property test mindset (production)
A good invariant-based test:
- For random sorted arrays, compare the optimal method against a slow merge baseline.

### 7.6 A compact test harness (baseline vs optimal)

In interviews you rarely write a full test harness, but in real codebases you should.
The most valuable pattern here is **differential testing**:
compare the “obviously correct” method against the “clever” method.

```python
import random


def median_merge_baseline(a: list[int], b: list[int]) -> float:
    merged = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            merged.append(a[i]); i += 1
        else:
            merged.append(b[j]); j += 1
    merged.extend(a[i:])
    merged.extend(b[j:])

    n = len(merged)
    if n % 2 == 1:
        return float(merged[n // 2])
    return (merged[n // 2 - 1] + merged[n // 2]) / 2.0


def randomized_test(trials: int = 200) -> None:
    sol = Solution()
    for _ in range(trials):
        m = random.randint(0, 20)
        n = random.randint(0, 20)
        a = sorted(random.randint(-50, 50) for _ in range(m))
        b = sorted(random.randint(-50, 50) for _ in range(n))
        if m + n == 0:
            continue
        expected = median_merge_baseline(a, b)
        got = sol.findMedianSortedArrays(a, b)
        assert abs(expected - got) < 1e-9, (a, b, expected, got)


if __name__ == "__main__":
    randomized_test()
    print("OK")
```

Why this matters:
- it catches off-by-one errors instantly
- it future-proofs refactors (porting to another language)

---

## 8. Complexity Analysis (Detailed)

Let \(m = |A|\), \(n = |B|\), and assume \(m \le n\).

- **Binary search iterations**: \(O(\log m)\)
- **Work per iteration**: constant (a handful of index checks/comparisons)

So:
- **Time**: \(O(\log m)\)
- **Space**: \(O(1)\)

---

## 9. Production Considerations

### 9.1 Numeric types
In production, numbers may be floats, timestamps, or decimals:
- Watch out for floating-point precision in the even-case average.
- If values can exceed 32-bit range, use 64-bit ints (Python is safe).

### 9.2 Data location matters
If the two arrays live on different machines (or one is on disk), an \(O(m+n)\) merge can become:
- network-heavy
- I/O-heavy

The partition-based approach hints at a pattern:
> Move computation to boundaries; minimize data movement.

This is exactly why federated and privacy-preserving designs exist.

### 9.3 Generalizing: kth element queries
Many real systems need percentiles (p90, p95) rather than medians.
This algorithm can be extended to compute the \(k\)-th element in two sorted arrays using a similar boundary search.

### 9.4 External-memory and “sorted runs” (why merging is expensive in the real world)

In production data systems, your “arrays” are often:
- sorted files on disk (SSTables)
- sorted partitions across machines
- time-ordered event logs (already sorted by timestamp)

In those settings, a full merge is not just \(O(m+n)\) CPU:
- it is **I/O** (disk reads)
- it is **network** (cross-machine transfer)
- it is **latency** (you wait for stragglers)

This is why analytic engines invest heavily in:
- indexes
- metadata (min/max per block)
- partition pruning

The partition-based median algorithm is a toy version of the same idea:
> Touch only what you need. Use order + boundaries to avoid moving everything.

### 9.5 Approximate percentiles (when exact is too expensive)

At very large scale, you often don’t compute the exact median.
You compute an approximation with guarantees:
- **t-digest**
- **KLL sketches**
- histogram-based quantiles

These algorithms are explicitly designed for distributed systems:
- each worker computes a sketch locally
- sketches merge efficiently (small payload)
- global quantiles are approximated from merged sketches

If you interview for data-intensive roles, bringing this up shows “systems taste”:
- exact algorithms are great, but approximations are often the production answer

### 9.6 Streaming and sliding windows

If values arrive over time (a stream), and you want a rolling median:
- you can’t keep arrays fully sorted cheaply
- common solution: two heaps (max-heap for left, min-heap for right)

That’s a different problem than “two sorted arrays”, but it reinforces the same invariant:
- keep a balanced partition
- keep all left \(\le\) all right

This problem teaches you to reason with that invariant precisely.

### 9.7 Testing in production: differential testing and invariants

For critical code paths, a strong strategy is **differential testing**:
- compare the optimal partition method against a slow merge baseline
- generate randomized sorted inputs (including duplicates, negatives, empties)

Additionally, assert invariants at runtime in debug builds:
- input arrays sorted
- computed partition satisfies inequalities
- returned median is within min/max bounds of inputs

These checks catch regressions early, especially when porting the algorithm between languages.

---

## 10. Connections to ML Systems

The theme today is **binary search and distributed algorithms**:
- **Federated Learning**: clients compute local statistics, then a server aggregates—global behavior emerges from boundary interactions, not centralized raw data.
- **Privacy-preserving Speech**: you can’t centralize raw audio; you must design algorithms that preserve correctness while restricting access.
- **Knowledge Graphs for Agents**: agents need fast retrieval and reasoning without scanning the entire graph; indexing and targeted search are the equivalent of “partitioning”.

Concretely, this problem teaches a transferable skill:
> When you see “two sorted streams”, don’t merge unless you have to—search for the boundary that answers the query.

---

## 11. Interview Strategy

1. **Start with merge**: show the baseline quickly (it builds confidence).
2. **Call out the constraint**: “If you need log time, we can’t merge.”
3. **Explain partition clearly**: draw the four boundary values and the two inequalities.
4. **Force the swap**: always binary search the smaller array.
5. **Use sentinels**: `-inf` and `+inf` to avoid messy edge-case branches.

Common mistakes:
- Off-by-one on left size when total is odd (use `(m+n+1)//2`).
- Forgetting that partition index ranges include `0` and `m`.
- Computing `j` outside `[0..n]` (fixed by ensuring `A` is smaller).

### 11.1 What to say while you code (signal you understand the invariants)

As you implement, narrate the invariants:
- “I’m searching `i` in the smaller array so the search space is `log m`.”
- “`j` is derived so left has exactly half the elements.”
- “I’ll compute four boundary values and only compare boundaries.”
- “If `A_left_max` is too big, shift left; otherwise shift right.”

This matters because interviewers grade **reasoning**, not just final code.

### 11.2 Follow-up questions you should be ready for

Interviewers often ask variants to see if you truly own the technique:

- **“What if I ask for the 90th percentile instead of the median?”**
  - Answer: median is a special case of the \(k\)-th smallest selection problem.
  - For a single query on two sorted arrays, you can compute the \(k\)-th element similarly.
  - At massive distributed scale, you’ll often use approximate quantiles (sketches).

- **“What if arrays contain floats or timestamps?”**
  - Answer: algorithm is order-based, works as long as comparisons are consistent.
  - For even length, the “average” step may need domain-specific behavior (e.g., midpoint timestamp).

- **“What if the arrays are descending?”**
  - Answer: either reverse comparisons or normalize by reversing inputs.
  - The invariant is “left max <= right min” under the chosen ordering.

- **“Can you do it without using infinities?”**
  - Answer: yes, but sentinels simplify edge conditions (and reduce bugs).

### 11.3 How to debug when it fails (fast)

If you get a wrong answer, print the four boundary values and check the inequalities:
- `i`, `j`
- `A_left_max`, `A_right_min`
- `B_left_max`, `B_right_min`

Most bugs are:
- wrong `left_size` formula
- wrong handling when `i==0` or `i==m` (or similarly for `j`)

### 11.4 Why this problem is a great “systems signal”

Even though it’s a DSA question, it tests a production instinct:
> Don’t move or materialize data you don’t need.

That instinct is directly useful in:
- distributed query engines (pushdown predicates and limit scans)
- privacy-sensitive systems (compute aggregates locally)
- agent retrieval systems (retrieve the smallest evidence set that answers the question)

---

## 12. Key Takeaways

1. **Median = partition boundary**, not a merge artifact.
2. **Binary search is a tool for monotonic conditions**, not just “find a number”.
3. **Boundary-first thinking scales**: it’s the same instinct you need for distributed and privacy-constrained systems.

### 12.1 Mental model summary (the one you want to internalize)

If you can remember one sentence, remember this:
> We’re not searching values; we’re searching the cut that makes left and right “balanced and ordered”.

Everything else follows:
- The “balanced” condition gives you `j = left_size - i`.
- The “ordered” condition is two inequalities on four boundary values.
- The median is computed from the boundary once the inequalities hold.

### 12.2 A second optimal method (divide-and-conquer k-th element)

Sometimes an interviewer will ask:
“Can you find the \(k\)-th smallest element of two sorted arrays?”

There’s a classic selection algorithm:
- compare \(k/2\)-th elements in each array
- discard \(k/2\) elements from one array each step
- recurse with reduced \(k\)

This is essentially “binary search over ranks”.
It also runs in \(O(\log k)\), which is \(O(\log(m+n))\).

You don’t need to implement it here (the partition method is cleaner),
but it’s useful to know as an alternative for:
- teaching
- reasoning about general percentiles
- interviews where they steer toward a recursive selection approach

High-level pseudocode:

```python
def kth(a, b, k):  # 1-indexed k
    if not a: return b[k-1]
    if not b: return a[k-1]
    if k == 1: return min(a[0], b[0])

    i = min(len(a), k//2)
    j = min(len(b), k//2)

    if a[i-1] <= b[j-1]:
        # discard first i elements of a
        return kth(a[i:], b, k - i)
    else:
        # discard first j elements of b
        return kth(a, b[j:], k - j)
```

Why it works:
- if `a[i-1] <= b[j-1]`, then `a[:i]` cannot contain the k-th element
  (there are at least `i + j - 1 >= k - 1` elements <= `b[j-1]`),
  so we can safely discard it.

This is the same “discard half the search space” principle that powers binary search.

### 12.3 Practical note: stable behavior across languages

If you implement this in C++/Java:
- avoid integer overflow when computing averages (use long / double carefully)
- sentinel usage differs (use `INT_MIN/INT_MAX` or explicit branches)

In Python, `float("inf")` is convenient, but remember:
- `inf` is only used for comparisons here, not returned as a median
- if your arrays could contain actual infinities (rare), you’d need explicit handling

### 12.4 If you’re short on time: the 30-second explanation

If you need to summarize fast:
- Pick the smaller array `A`.
- Binary search a cut `i` in `A`.
- Compute `j` so left side has half the total.
- Check two inequalities using boundary values.
- When inequalities hold, compute median from `max(left)` and `min(right)`.

That’s the entire solution.

### 12.5 One more invariant (useful for sanity checks)

After you compute a candidate median, a quick sanity check is:
- the median must lie between the minimum and maximum of the combined arrays
- more specifically, it must be:
  - \(\ge\) the maximum of the left boundary values
  - \(\le\) the minimum of the right boundary values

In code terms (once partition is valid):
- `left_max = max(A_left_max, B_left_max)`
- `right_min = min(A_right_min, B_right_min)`
- for odd: `median == left_max`
- for even: `left_max <= median <= right_min`

This is also a nice explanation when someone asks “why do we take max(left) and min(right)?”

---

**Originally published at:** [arunbaby.com/dsa/0051-median-of-two-sorted-arrays](https://www.arunbaby.com/dsa/0051-median-of-two-sorted-arrays/)

*If you found this helpful, consider sharing it with others who might benefit.*

