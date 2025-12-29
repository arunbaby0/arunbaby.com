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

### 1.1 Constraints (The Hard Version)
To satisfy the requirements of high-performance system design, you must adhere to the following constraints:
- **Time Complexity**: You must run in **O(N)** time. The algorithm should scale linearly with the input size.
- **Space Complexity**: You must use **O(1)** extra space. You cannot allocate new data structures like sets or additional arrays proportional to the input size.

### 1.2 Examples for Intuition
**Example 1**
- **Input**: `nums = [1, 2, 0]`
- **Output**: `3`
- **Reasoning**: 1 and 2 are present. The next positive integer is 3.

**Example 2**
- **Input**: `nums = [3, 4, -1, 1]`
- **Output**: `2`
- **Reasoning**: 1 is present, but 2 is missing.

**Example 3**
- **Input**: `nums = [7, 8, 9, 11, 12]`
- **Output**: `1`
- **Reasoning**: 1 is the smallest positive integer, and it is missing.

---

## 2. Understanding the Problem: Deep Dive

### 2.1 The Mathematical Bound: Why [1, n+1]?
The most critical insight for this problem is the search space. Many candidates start by looking for any positive integer from 1 to infinity. However, with an array of length $n$, the smallest missing positive integer **must** fall within the range $[1, n+1]$.

**The "Pigeonhole Principle" Proof**:
Imagine you have $n$ slots in your array.
- If you fill those $n$ slots with the numbers $1, 2, 3, \dots, n$, then the first missing positive is $n+1$.
- If you replace any of those numbers (say, replace $2$ with $100$ or $-5$), then a number in the range $[1, n]$ becomes missing.
- Therefore, the answer can never be larger than $n+1$.

This realization transforms the problem from an infinite search into a bounded search. Once the search is bounded to the size of the input array, the possibility of using the array itself as a storage mechanism emerges.

### 2.2 Why This is a "Hard" Problem
If we relax the constraints, the problem becomes trivial:
1. **HashSet Solution**: Add all numbers to a set, then iterate from 1 to $n+1$. This is $O(N)$ time and $O(N)$ space.
2. **Sorting Solution**: Sort the array and scan for gaps. This is $O(N \log N)$ time and $O(1)$ space.

The "Hard" designation comes from the requirement to achieve **both** $O(N)$ time and $O(1)$ space. This requires a specific class of algorithms known as **In-Place Index Mapping**.

### 2.3 Thematic Links to Systems Engineering
This problem isn't just a brain teaser; it represents a core pattern in low-level systems engineering:
- **Data Integrity**: Filtering out "noise" (negatives, zeros, large numbers) is the first step in any robust data pipeline.
- **Memory Efficiency**: In high-throughput systems (like network packet processors), you cannot afford to allocate memory for every packet. You must process the data "in-flight" within the buffer it arrived in.
- **Normalization**: Mapping a messy, wide-range input into a canonical, narrow-range domain is a fundamental pre-processing step in both ML and Speech Tech.

---

## 3. Approach 1: Cyclic Placement (The Intuitive Swap)

The "Cyclic Placement" algorithm is based on the idea of a "permuted array." In a perfect world, our array of size $n$ would contain the value $i+1$ at index $i$. Our goal is to rearrange the array to match this ideal state as much as possible.

### 3.1 The Core Logic
We iterate through the array starting from the first index. For each element, we ask: "Does this value belong here?"
- If the value $v$ is in the range $[1, n]$ AND it is not at its correct index ($v-1$), we swap it with the element currently at $v-1$.
- We continue swapping the "new" element at our current index until it is either out of range or correctly placed.

### 3.2 Step-by-Step Visualization
Let's trace `nums = [3, 4, -1, 1]` where $n=4$.

1. **Index 0 (Value 3)**:
   - Does 3 belong at index 0? No, it belongs at index 2 ($3-1$).
   - Swap `nums[0]` (3) with `nums[2]` (-1).
   - Array becomes: `[-1, 4, 3, 1]`
   - Now, does `nums[0]` (-1) belong here? No, it's out of range. Move to index 1.

2. **Index 1 (Value 4)**:
   - Does 4 belong at index 1? No, it belongs at index 3 ($4-1$).
   - Swap `nums[1]` (4) with `nums[3]` (1).
   - Array becomes: `[-1, 1, 3, 4]`
   - Now, does `nums[1]` (1) belong at index 1? No, it belongs at index 0 ($1-1$).
   - Swap `nums[1]` (1) with `nums[0]` (-1).
   - Array becomes: `[1, -1, 3, 4]`
   - Now, does `nums[1]` (-1) belong here? No, out of range. Move to index 2.

3. **Index 2 (Value 3)**:
   - Does 3 belong here? Yes (index 2 contains 3). Move to index 3.

4. **Index 3 (Value 4)**:
   - Does 4 belong here? Yes. End of loop.

**Final Scan**:
- Index 0: Has 1 (Correct)
- Index 1: Has -1 (Wait! Expected 2. Found the "hole"!)
- **Result: 2**

### 3.3 Complexity Analysis of Cyclic Placement
- **Time**: $O(N)$. Although there's a `while` loop inside a `for` loop, each swap operation places at least one number in its correct final position. Since there are only $n$ positions, there can be at most $n$ successful swaps across the entire execution. This is known as **Amortized Analysis**.
- **Space**: $O(1)$. We only used a few pointer variables.

---

## 4. Approach 2: Sign Marking (The Compact Trick)

Sign Marking is a more subtle technique often used in bit manipulation and low-level C programming. It relies on the fact that we can store one bit of extra information (the sign bit) without changing the absolute value of the number (mostly).

### 4.1 The 4-Step Process
1. **Sanity Check**: Does 1 even exist in the array? If not, the answer is 1.
2. **Normalization**: Replace all numbers $\le 0$ or $> n$ with the value 1. Now the array only contains values in $[1, n]$.
3. **Presence Encoding**: For each value $v$ in the array, find the index $(abs(v) - 1)$ and make the value at that index **negative**.
4. **The Search**: Scan the array. The first index $i$ that contains a **positive** value means that the number $i+1$ was never seen in the array.

### 4.2 Why This Works
The sign bit acts as a boolean "seen" flag. Because we normalized everything to be positive in step 2, we can safely use the negative sign to mean "this index has been visited."

---

## 5. Implementation Deep Dive

Let's look at the Python implementation for the Cyclic Placement approach, as it's the most common in interviews.

```python
from typing import List

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Phase 1: In-place Swapping (Cyclic Placement)
        i = 0
        while i < n:
            # The value we are looking at
            val = nums[i]
            # The index where this value 'should' be
            target_idx = val - 1
            
            # Condition for swapping:
            # 1. The value must be within the valid range [1, n]
            # 2. The value is not already at its correct index
            # 3. The value we are swapping WITH is not the same (prevents infinite loops on duplicates)
            if 1 <= val <= n and nums[i] != nums[target_idx]:
                # Perform the swap
                nums[i], nums[target_idx] = nums[target_idx], nums[i]
                # Note: We DO NOT increment i here, because the new nums[i] 
                # might also need to be swapped.
            else:
                i += 1
                
        # Phase 2: Linear Scan for the first 'hole'
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        
        # If no hole found, the answer is n + 1
        return n + 1
```

### 5.1 The "Duplicate Guard" Explained
Interviewer: "What happens if the input is `[1, 1]`?"
Me: 
- At `i=0`, `val=1`, `target_idx=0`. Since `nums[0] == nums[0]`, the `if` condition fails. `i` becomes 1.
- At `i=1`, `val=1`, `target_idx=0`. Now, `nums[1]` (1) is equal to `nums[0]` (1). If we swapped, we would just be putting a 1 where a 1 already is. We would never make progress and would loop forever.
- The condition `nums[i] != nums[target_idx]` is the "Duplicate Guard" that ensures terminate in the presence of repeated values.

---

## 6. Production Considerations: Beyond the Algorithm

In a real production environment, choosing an $O(1)$ space algorithm is an engineering trade-off.

### 6.1 The Cost of Mutation
In-place algorithms **mutate** their inputs. In a multi-threaded system, this is a dangerous side effect.
- **Race Conditions**: If another part of your system is reading the `nums` array while `firstMissingPositive` is swapping elements, the reading thread will see an inconsistent and "corrupted" state of the data.
- **Functional Purity**: Many modern architectural patterns (like Redux or functional programming) forbid mutation for exactly this reason. It makes debugging much harder.

**Engineering Protocol**: Always document that a function is "in-place" and mutates its arguments. If the caller requires the original data to remain intact, they must pass a copy, effectively turning the space complexity back into $O(N)$.

### 6.2 Cache Locality vs. Allocation
Why do we care about $O(1)$ if $O(N)$ memory is cheap?
- **Small Objects and GC**: In languages like Java or Python, allocating $10^6$ small objects or a large HashSet triggers a "stop-the-world" garbage collection event eventually.
- **Cache Hits**: Modern CPUs are much faster than RAM. An in-place array algorithm stays in the CPU's **L1 and L2 cache**. A HashSet involves hashing and traversing buckets, which often results in **Cache Misses**, slowing the algorithm down by a factor of 10 or more despite having the same $O(N)$ time complexity.

### 6.3 Zero-Copy Principles
If you are building a high-frequency trading platform or a real-time speech processing system, you use **Zero-Copy** buffers. The data is written by the network card directly into a shared buffer. The processing algorithm (like ours) runs directly on that buffer. This avoids the massive performance overhead of copying data between the kernel and user space.

---

## 7. Comparative Benchmarking (Theoretical)

| Input Size (N) | HashSet (Space/Time) | In-Place (Space/Time) | Advantage |
|---|---|---|---|
| 1,000 | 40 KB / 0.1ms | 0 KB / 0.05ms | Negligible |
| 1,000,000 | 40 MB / 100ms | 0 KB / 40ms | Memory Saved |
| 100,000,000 | 4 GB / 10s | 0 KB / 4s | **Critical** (prevents crash) |

On a machine with 2GB of RAM, the HashSet solution will crash (Out of Memory) for $N=10^8$, but the In-Place solution will complete successfully because it consumes no additional memory.

---

## 8. Follow-up: Missing Positives in a Stream

What if the numbers are coming from a live sensor or a web socket?
- **Scenario**: You are tracking the "Active Session IDs" of millions of concurrent users.
- **Problem**: You can't store them all in an array to sort or swap.
- **Solution**: Use a **BitMap** or a **Bloom Filter**.
  - A BitMap uses 1 bit per number. For $10^6$ IDs, you only need ~125 KB of memory to track everything precisely.
  - This is a common pattern in **Database Indexing** and **Operating System Memory Management** (tracking free frames in RAM).

---

## 9. Connections to ML Systems

### 9.1 Data Validation Guardrails
In any ML production pipeline, "Data Quality" is the primary barrier to reliability. Before feeding data into a model:
- You must validate that the indices and IDs fall within expected ranges.
- You must detect "Gaps" in the data that might indicate a sensor failure or a network dropout.
This algorithm is the most efficient way to detect the first "missing link" in a sequence of events.

### 9.2 Feature Hashing vs. Index Mapping
In ML feature engineering, we often use **Feature Hashing (the hashing trick)** to map high-cardinality features into a fixed-size vector. This algorithm uses the opposite principle: **Index Mapping**. Instead of a hash function, we use the value itself as its own address. This is the ultimate form of "Perfect Hashing" where collisions are impossible because each value has exactly one home.

### 9.3 Speech Track Link: Jitter Buffers
In Speech Technology (VoIP, live transcription), audio packets arrive out of order or go missing. A **Jitter Buffer** reorders packets based on their sequence numbers. If sequence numbers are missing, the system must decide whether to wait or to perform **Packet Loss Concealment (PLC)**. Finding the "first missing sequence number" is exactly the First Missing Positive problem.

---

## 10. Interview Strategy: How to Ace the "Hard"

1. **Clarify Constraints Early**: Confirm "Am I allowed to mutate the input?" and "Is it guaranteed to fit in memory?". This shows you understand the trade-offs of in-place algorithms.
2. **Start with the Bound**: Narrating the [1, n+1] range bound is the "aha!" moment that proves you have the right intuition.
3. **The Duplicate Trap**: Explicitly mention that you are handling duplicates. It's the #1 reason an "optimal" solution fails a hidden test case.
4. **Complexity Proof**: Don't just say $O(N)$. Explain **why** the nested `while` loop is $O(N)$ (amortized analysis).

---

## 11. Key Takeaways

1. **Bounded Domain → Perfect Storage**: If you know the range of your data, you can use the structure itself as metadata storage.
2. **Amortized Efficiency**: Nested loops do not always mean $O(N^2)$. If each operation does permanent, non-repeating work, the algorithm remains linear.
3. **Systems Mindset**: $O(1)$ space isn't just a gimmick; it's about cache locality, GC pressure, and the principles of Zero-Copy engineering.

---

## 12. Advanced Scaling: The SIMD and Branch Prediction Edge

In high-performance C++ or Rust implementations of this algorithm, we can leverage **SIMD (Single Instruction, Multiple Data)** to accelerate the normalization and final scan phases.
- **Normalization**: Instead of checking each number one by one to see if it's in `[1, n]`, you can load 8 or 16 integers into a SIMD register and compare them all in a single CPU cycle.
- **Branch Prediction**: The `if 1 <= val <= n` check can be tricky for the CPU's branch predictor if the data is highly randomized. To avoid "Branch Misprediction" penalties, senior engineers often use **Bit-Twiddling** to create a branchless mask. This keeps the CPU's instruction pipeline full and flowing at maximum speed.

## 13. Distributed Data Validation: The MapReduce Approach

What if your data is measured in Petabytes and spread across a thousand machines (e.g., in a Hadoop or Spark cluster)?
- **Step 1 (Map)**: Each worker processes its local chunk and emits a "Local BitSet" of $n$ bits. A bit is 1 if the number was seen.
- **Step 2 (Shuffle)**: The BitSets are aggregated.
- **Step 3 (Reduce)**: The reducer performs a bitwise `OR` on all BitSets. The first 0-bit in the final BitSet is the missing positive.
This allows you to find the missing positive for billions of IDs in seconds by parallelizing the computation across a cluster.

## 14. History: The Evolution of In-Place Algorithms

The study of in-place algorithms dates back to the early days of computing (1950s and 60s) when a computer might only have 4KB of RAM.
- **Quicksort**: Developed by Tony Hoare in 1959, it's the most famous in-place algorithm.
- **In-Place Mergesort**: A notoriously difficult variation that avoids the extra work of standard merge.
- **Heapsort**: Another $O(N \log N)$ in-place sorting algorithm.

Our Index Mapping technique is part of this long lineage of "Memory-Conscious Design." It echoes the era when programmers had to be extremely clever about every single byte, a skill that is becoming relevant again in the world of IoT and Edge Computing.

## 15. Advanced Follow-up: Finding ALL Missing Positives

What if the question was: "Find ALL missing numbers in the range `[1, n]`"?
- The cyclic placement algorithm works exactly the same!
- After the swap phase, instead of returning at the first mismatch, you collect all indices `i` where `nums[i] != i + 1` and return the list `[i+1 for each mismatching i]`.
- This is the foundation of **Missing Value Imputation** in data science, where you need to identify and fill gaps in your datasets before training a model.

## 16. The Future: Probabilistic Missing Positive Detection

In the future of "Big Data," we might not even need exact answers.
- **HyperLogLog**: Can estimate the number of *distinct* elements in a stream using almost zero memory.
- **Quotient Filters**: A more space-efficient alternative to Bloom Filters that allows for "merging" and "deleting" items.
These probabilistic structures allow us to monitor the "health" of massive data streams without ever storing the numbers themselves.

---

## 17. Real-world Engineering Case Study: Linux Kernel Memory Management

The principles behind the "First Missing Positive" algorithm are remarkably similar to how the Linux Kernel manages its physical memory pages.
- **The Problem**: The kernel needs to find a free "Page Frame" (a chunk of RAM) to assign to a process.
- **The Solution (Bitmaps)**: The kernel maintains a `free_area` structure which is essentially a BitMap. Each bit represents a page.
- **The Search**: To find a free page, the kernel performs a "find first zero bit" operation. This is conceptually the same as finding the "First Missing Positive" in a sequence of occupied page indices.
- **Optimization**: By using specialized CPU instructions like `CLZ` (Count Leading Zeros) or `BSR` (Bit Scan Reverse), the kernel can perform this search in just a few nanoseconds.

This illustrates a broader truth: **The most fundamental problems in computer science are often solved using the simplest, most memory-efficient indexing tricks.**

## 18. Coding Best Practices for In-Place Algorithms

When you decide to implement an in-place algorithm in a production codebase, follow these safety guidelines:
1. **Implicit vs. Explicit Mutation**: If your language supports it, mark the parameter as `mutable`. In Python, you can't do this, so you MUST add a docstring: `WARNING: This function mutates the 'nums' list in-place.`
2. **Defensive Copying**: If you are not 100% sure that the caller expects mutation, perform a shallow copy at the start: `nums = nums[:]`. While this violates $O(1)$ space, it prevents catastrophic side effects in complex systems.
3. **In-Place as an Internal Detail**: Often, the best pattern is to have a "wrapper" function that is pure (not mutating), which then calls a private "in-place" helper to do the heavy lifting. This gives you the performance of in-place with the safety of immutability.

---

**Originally published at:** [arunbaby.com/dsa/0053-first-missing-positive](https://www.arunbaby.com/dsa/0053-first-missing-positive/)

*If you found this helpful, consider sharing it with others who might benefit.*
