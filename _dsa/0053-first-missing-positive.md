---
title: "First Missing Positive"
day: 53
collection: dsa
categories:
  - dsa
tags:
  - arrays
  - in-place
  - index-mapping
  - hard
  - edge-cases
  - pattern-recognition
difficulty: Hard
subdomain: "Array Manipulation"
tech_stack: Python
scale: "O(N) time, O(1) extra space"
companies: Google, Meta, Amazon, Microsoft
related_dsa_day: 53
related_ml_day: 53
related_speech_day: 53
related_agents_day: 53
---

**"The missing number is hiding in plain sight—use the array itself as the hash table."**

## 1. Problem Statement

Given an unsorted integer array `nums`, find the smallest missing positive integer.

Constraints (classic version):
- You must run in **O(N)** time.
- You must use **O(1)** extra space.

**Example 1**
- Input: `nums = [1,2,0]`
- Output: `3`

**Example 2**
- Input: `nums = [3,4,-1,1]`
- Output: `2`

**Example 3**
- Input: `nums = [7,8,9,11,12]`
- Output: `1`

---

## 2. Understanding the Problem

### 2.1 What range matters?

If the array length is `n`, the answer is always in `[1, n+1]`.

Why?
- In the best case, the array contains `1..n` → missing is `n+1`.
- If any value in `1..n` is missing → answer is within that range.
- Values `<= 0` and `> n` cannot affect the smallest missing positive.

This range bound is the key to getting O(1) extra space:
we only care about presence/absence of `1..n`.

### 2.3 Why negative numbers and zeros are noise

The phrase “first missing **positive**” is not cosmetic—it lets us ignore large parts of the input.

- If `nums[i] <= 0`, it can never be the smallest missing positive (because `1` is the smallest positive).
- If `nums[i] > n`, it can’t affect the answer either, because if `1..n` are all present then the answer is `n+1`.

So the only “signal” values are in `[1..n]`.
This is the same habit you want in production data validation:
define the valid domain early, and treat everything outside it as invalid/irrelevant.

### 2.4 Two equivalent optimal strategies (both O(N) time, O(1) extra space)

There are two classic approaches:

1. **Cyclic placement** (swap `v` into index `v-1`)
2. **In-place marking** (use sign flips to mark presence)

Both are correct. Cyclic placement is often more intuitive (“put things where they belong”).
Marking is often more compact once you’re comfortable with it.

### 2.2 Theme: data validation and edge case handling

This problem is algorithmic “data validation”:
- filter out irrelevant values
- normalize to a known domain
- encode presence efficiently

That maps directly to production systems:
- ML data validation: enforce schema/ranges before training
- speech quality validation: detect clipping, sample rate mismatch, dropouts
- agent deployment patterns: guardrails and invariants to prevent bad rollouts

---

## 3. Approach 1: Brute Force (Check 1,2,3,...)

### 3.1 Idea
Start from `x=1` and check if `x` exists in the array. Increment until you find a missing one.

### 3.2 Complexity
- **Time**: \(O(N^2)\) if you scan the array for each x
- **Space**: \(O(1)\)

Too slow for large `N`, but useful for intuition.

---

## 4. Approach 2: Use a Hash Set

### 4.1 Idea
Put all numbers into a set and then check `1..n+1`.

### 4.2 Complexity
- **Time**: \(O(N)\)
- **Space**: \(O(N)\)

This is a great production solution when memory is allowed, but it violates the O(1) space constraint.

### 4.3 Implementation (set baseline)

This is also the perfect baseline for differential testing.

```python
from typing import List


class SolutionSet:
    def firstMissingPositive(self, nums: List[int]) -> int:
        s = set(x for x in nums if x > 0)
        n = len(nums)
        for x in range(1, n + 2):
            if x not in s:
                return x
        return n + 1
```

---

## 5. Approach 3: Optimal In-Place Index Mapping (Cyclic Placement)

### 5.1 Core insight

We want to place each value `v` in its “correct” index:
- value `1` should be at index `0`
- value `2` should be at index `1`
- ...
- value `n` should be at index `n-1`

So for a value `v` in `[1..n]`, its target index is `v-1`.

We can rearrange in-place by swapping until:
- the current number is either out of range, or already in the correct place, or a duplicate blocks progress.

This is essentially “use the array as a hash table” where the hash is `v -> v-1`.

### 5.2 Algorithm steps

1. For each index `i`:
   - while `nums[i]` is in `[1..n]` and `nums[i]` is not in its correct place:
     - swap `nums[i]` with `nums[nums[i]-1]`

2. After placement, scan from `i=0`:
   - the first index where `nums[i] != i+1` → missing is `i+1`
   - if all match, missing is `n+1`

### 5.3 Why it’s O(N)

Even though there’s a while loop, each swap places at least one value closer to its final position.
Each element can be swapped only a constant number of times before it reaches its correct spot or is ruled irrelevant.

So total swaps across the whole array is O(N).

### 5.4 Dry run (so the swapping loop feels safe)

Example: `nums = [3, 4, -1, 1]`, `n = 4`

Mapping rule:
- value `v` belongs at index `v-1`

Start:
- i=0, v=3 → swap with index 2 → `[-1, 4, 3, 1]`
- i=0, v=-1 out of range → i=1
- i=1, v=4 → swap with index 3 → `[-1, 1, 3, 4]`
- i=1, v=1 → swap with index 0 → `[1, -1, 3, 4]`
- i=1, v=-1 out of range → i=2
- i=2, v=3 already correct → i=3
- i=3, v=4 already correct → done

Final scan:
- index 0 has 1 ✓
- index 1 expects 2 but has -1 → answer is 2

This also shows the duplicate guard: we never swap if it would be a no-op (`nums[i] == nums[correct_idx]`).

### 5.5 Another dry run (duplicates are the trap)

Example: `nums = [1, 1]`, `n = 2`
- i=0, v=1 already correct → i=1
- i=1, v=1, correct_idx=0
  - since `nums[i] == nums[correct_idx]`, we do NOT swap
  - i increments and loop ends

Final scan:
- index 0 has 1 ✓
- index 1 expects 2 but has 1 → answer is 2

If you forget the duplicate guard, this case can infinite-loop.

---

## 6. Implementation (Fully Commented Python)

```python
from typing import List


class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)

        # 1) Place each value v (1..n) into index v-1.
        i = 0
        while i < n:
            v = nums[i]
            correct_idx = v - 1

            # We only care about v in [1..n].
            # Also avoid infinite loops on duplicates (if nums[i] == nums[correct_idx], swapping does nothing).
            if 1 <= v <= n and nums[i] != nums[correct_idx]:
                nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
            else:
                i += 1

        # 2) First place where index doesn't match value indicates missing.
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1

        return n + 1
```

---

## 7. Alternative Optimal Approach: In-Place Marking (Sign Trick)

This is another widely-accepted optimal approach that uses the array as a presence map.

### 7.1 Key idea

1. If `1` is missing, answer is `1`.
2. Normalize irrelevant values (`<=0` or `>n`) to `1`.
3. For each value `v` in `[1..n]`, mark presence by making `nums[v-1]` negative.
4. The first index with a positive value indicates the missing number.

### 7.2 Implementation (sign marking)

```python
from typing import List


class SolutionMarking:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 1

        has_one = any(x == 1 for x in nums)
        if not has_one:
            return 1

        # Normalize to [1..n] domain (use 1 as sentinel).
        for i in range(n):
            if nums[i] <= 0 or nums[i] > n:
                nums[i] = 1

        # Mark presence.
        for i in range(n):
            v = abs(nums[i])
            idx = v - 1
            nums[idx] = -abs(nums[idx])

        # First positive index is missing.
        for i in range(n):
            if nums[i] > 0:
                return i + 1
        return n + 1
```

### 7.3 Marking vs placement (how to choose)

- **Placement**: intuitive mapping; good when you want to “normalize” the array into `[1..n]`.
- **Marking**: compact; good when you want fewer swaps and can reason about sign flips.

Both require careful edge case handling (duplicates and bounds).

---

## 8. Testing (Edge Cases Included)

### 8.1 Simple cases
- `[1,2,0]` → `3`
- `[3,4,-1,1]` → `2`
- `[7,8,9,11,12]` → `1`

### 8.2 Duplicates
- `[1,1]` → `2`
- `[2,2,2]` → `1`

### 8.3 Already perfect
- `[1,2,3]` → `4`

### 8.4 Randomized differential testing
Compare the in-place method against a set-based baseline on random arrays.

---

## 9. Complexity Analysis

Let \(N = len(nums)\).

- **Time**: \(O(N)\)
  - total swaps are linear
  - final scan is linear
- **Space**: \(O(1)\) extra space
  - sorting happens in-place

### 9.1 Why the while-loop is still linear (amortized analysis)

People often worry that “a while loop inside a loop” implies \(O(N^2)\).
Here’s the key amortized argument:

- Every time we perform a swap, we move some value `v` closer to (or into) its final correct index `v-1`.
- Once a value is placed correctly, it will not be moved again (because future swaps check equality and stop on duplicates).

So the total number of swaps across the whole algorithm is bounded by O(N), and therefore total work is O(N).

This is a common interview pattern:
> Many “nested loops” are still linear if you can prove each inner iteration consumes a one-time resource.

---

## 10. Production Considerations

### 9.1 Mutation and safety
This algorithm mutates the input array. In production code:
- document mutation clearly
- copy input if immutability is required

### 9.2 Input validation parallels
This algorithm has an explicit “validation boundary”:
- ignore values `<= 0` and `> n`

That is the same instinct as production data validation:
- reject or quarantine invalid records
- map valid records into a canonical domain
- enforce invariants early

### 9.3 Observability
If this runs in a pipeline, emit counters:
- number of invalid values filtered
- number of duplicates in `[1..n]`
These signals can reveal upstream data corruption.

### 10.1 Differential testing (the production-quality safety net)

For invariant-heavy algorithms, differential testing is extremely effective:
- implement a slow but obviously correct baseline (set-based)
- generate random inputs (including negatives and duplicates)
- assert both methods match

Why this matters:
- most bugs here are edge-case bugs (duplicates, out-of-range values)
- randomized tests catch these quickly

### 10.2 A simple baseline for differential tests

Baseline approach:
- put all positive numbers in a set
- scan from 1 upward to find first missing

This uses O(N) memory, but it’s perfect as a reference implementation.

### 10.3 Why in-place mapping is a general-purpose technique

This “map value → index” trick appears in many systems contexts:
- bucketization and partitioning
- memory and storage layout (bins)
- canonicalization pipelines

The deeper skill is: when the domain is bounded (`1..n`), use the structure itself to store state.

### 10.4 When to prefer the set solution in production

In many real codebases, the simplest correct solution wins:
- it’s easier to review
- it’s less likely to have edge-case bugs
- it avoids mutating inputs (which can cause subtle shared-state issues)

So if memory is not a constraint, the set solution is often preferable.

The in-place solutions are still valuable when:
- memory is constrained (mobile/edge)
- the function is on a hot path and allocations matter
- you explicitly want O(1) extra space (interview constraints)

### 10.5 Mutation safety checklist

If you ship the in-place solution:
- document that input is mutated
- avoid passing shared arrays
- consider copying when correctness safety outweighs memory cost

---

## 11. Connections to ML Systems

Today’s shared theme is **data validation and edge case handling**:
- DSA: canonicalize values into `[1..n]` and detect what’s missing.
- ML system design: validate schema, ranges, and distribution before training.
- Speech tech: validate audio quality (clipping, sample rate, dropouts) before decoding/training.
- Agents: deployment patterns rely on guardrails and invariants to prevent bad behavior from propagating.

The transferable lesson:
> If you can define the valid domain tightly, you can turn “messy inputs” into a structured, reliable computation.

---

## 12. Interview Strategy

1. State the range bound `[1..n+1]` early.
2. Give the hash set solution as a stepping stone.
3. Then propose in-place placement with swaps, emphasizing the duplicate guard.
4. Walk through one example to show the loop doesn’t get stuck.

Common pitfalls:
- infinite loops on duplicates (must check `nums[i] != nums[correct_idx]`)
- forgetting to ignore out-of-range numbers
- off-by-one in index mapping (`v` goes to `v-1`)

### 12.1 Walkthrough question: “Why can’t the answer be > n+1?”

This is a common interviewer probe.
Answer:
- Among `n` slots, at most `n` distinct positive integers in `[1..n]` can be present.
- If all `1..n` are present, then `n+1` is the first missing positive.
- If any in `[1..n]` is missing, the answer is within that range.

So `n+1` is an absolute upper bound.
Once you state this bound clearly, the O(1)-space trick (use the array as storage) becomes natural.

### 12.2 How to explain termination (avoid the “while loop fear”)

For cyclic placement:
- every successful swap places some value into its correct index
- a value in `[1..n]` can only be placed correctly once
- duplicates stop swapping due to the `nums[i] != nums[correct_idx]` guard

So total swaps are O(N), and the while loop cannot run forever.

---

## 13. Key Takeaways

1. **The answer is in `[1..n+1]`**: that’s the reason O(1) space is possible.
2. **Index mapping turns the array into a hash table**: `v -> v-1`.
3. **Validation boundaries simplify everything**: ignore invalid domain values early.

### 13.1 Appendix: “why O(1) extra space is possible” in one paragraph

Because the answer is guaranteed to be in `[1..n+1]`, we only need to track presence of values `1..n`.
Instead of allocating a separate boolean array or hash set, we reuse the input array as storage:
- cyclic placement stores “presence” by placing values at their canonical indices
- marking stores “presence” by flipping signs at canonical indices

This is the general pattern:
> bounded domain → encode state inside the structure.

### 13.2 Appendix: common variants and how to answer

- **If asked for the missing number in `[0..n]`**
  - use XOR or sum formula (different problem)
- **If asked for first missing non-negative**
  - shift the domain to include 0, adjust mapping accordingly
- **If asked to not mutate input**
  - use the set solution (O(N) space) or copy then run in-place

The important thing is to narrate the domain and the mapping clearly.

### 13.3 Appendix: “array as a hash table” is a general pattern

This problem is a canonical example of a broader technique:
- when you have a bounded domain, you can map values to indices and store state in-place

You’ll see the same idea in:
- duplicate detection problems (marking seen indices)
- finding missing numbers in `[1..n]`
- in-place bucketization

The transferable skill is not the exact swaps—it’s recognizing when the domain bound lets you avoid extra memory.

---

**Originally published at:** [arunbaby.com/dsa/0053-first-missing-positive](https://www.arunbaby.com/dsa/0053-first-missing-positive/)

*If you found this helpful, consider sharing it with others who might benefit.*

