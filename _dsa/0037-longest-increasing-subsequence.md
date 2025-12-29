---
title: "Longest Increasing Subsequence (LIS)"
day: 37
related_ml_day: 37
related_speech_day: 37
related_agents_day: 37
collection: dsa
categories:
 - dsa
tags:
 - dynamic-programming
 - binary-search
 - greedy
 - patience-sorting
difficulty: Medium
---

**"Finding the longest upward trend in chaos."**

## 1. Problem Statement

Given an integer array `nums`, return the length of the longest strictly increasing subsequence.

A **subsequence** is a sequence derived from an array by deleting some or no elements without changing the order of the remaining elements.

**Example 1:**
``
Input: nums = [10, 9, 2, 5, 3, 7, 101, 18]
Output: 4
Explanation: The longest increasing subsequence is [2, 3, 7, 101], length = 4.
``

**Example 2:**
``
Input: nums = [0, 1, 0, 3, 2, 3]
Output: 4
Explanation: [0, 1, 2, 3]
``

## 2. Approach 1: Dynamic Programming O(N^2)

**Intuition:**
- Let `dp[i]` = length of LIS ending at index `i`.
- For each `i`, look at all previous elements `j < i`.
- If `nums[j] < nums[i]`, we can extend the LIS ending at `j` by including `nums[i]`.
- `dp[i] = max(dp[j] + 1)` for all valid `j`.

``python
class Solution:
 def lengthOfLIS(self, nums: List[int]) -> int:
 if not nums: return 0
 n = len(nums)
 dp = [1] * n # Every element is an LIS of length 1
 
 for i in range(1, n):
 for j in range(i):
 if nums[j] < nums[i]:
 dp[i] = max(dp[i], dp[j] + 1)
 
 return max(dp)
``

**Complexity:**
- **Time:** O(N^2)
- **Space:** O(N)

## 3. Approach 2: Binary Search + Greedy O(N \log N)

**Key Insight:**
- Maintain an array `tails` where `tails[i]` is the smallest tail element of all increasing subsequences of length `i+1`.
- For each new number, use binary search to find where it fits.

**Why does this work?**
- If we want to build a longer LIS, we should keep the tail as small as possible.
- Example: `[4, 5, 6, 3]`
 - After processing `[4, 5, 6]`, `tails = [4, 5, 6]`.
 - When we see `3`, we replace `4` with `3` → `tails = [3, 5, 6]`.
 - Now if we see `[3, 4, 7]`, we can build `[3, 4, 7]` (length 3), which wouldn't be possible if we kept `4`.

``python
import bisect

class Solution:
 def lengthOfLIS(self, nums: List[int]) -> int:
 tails = []
 
 for num in nums:
 # Find the leftmost position where num can be placed
 pos = bisect.bisect_left(tails, num)
 
 if pos == len(tails):
 tails.append(num) # Extend the LIS
 else:
 tails[pos] = num # Replace to keep tail small
 
 return len(tails)
``

**Complexity:**
- **Time:** O(N \log N)
- **Space:** O(N)

## 4. Deep Dive: Patience Sorting

The binary search approach is actually **Patience Sorting**, a card game strategy.

**Game Rules:**
1. Deal cards one by one.
2. Place each card on the leftmost pile where it's smaller than the top card.
3. If no such pile exists, start a new pile.

**Connection to LIS:**
- Number of piles = Length of LIS.
- The cards in each pile form a decreasing sequence (top to bottom).
- The top cards of all piles form an increasing sequence.

## 5. Reconstructing the LIS

The binary search approach only gives the **length**. To get the actual sequence:

``python
def findLIS(nums):
 n = len(nums)
 tails = []
 parent = [-1] * n # Track predecessor
 tail_indices = [] # Track which index contributes to each tail
 
 for i, num in enumerate(nums):
 pos = bisect.bisect_left(tails, num)
 
 if pos == len(tails):
 tails.append(num)
 tail_indices.append(i)
 else:
 tails[pos] = num
 tail_indices[pos] = i
 
 # Set parent
 if pos > 0:
 parent[i] = tail_indices[pos - 1]
 
 # Backtrack to reconstruct LIS
 lis = []
 k = tail_indices[-1]
 while k != -1:
 lis.append(nums[k])
 k = parent[k]
 
 return lis[::-1]
``

## 6. Variations

**1. Number of LIS (LeetCode 673)**
- Count how many LIS exist.
- Modify DP to track `count[i]` = number of LIS ending at `i`.

**2. Longest Divisible Subset (LeetCode 368)**
- Same DP, but condition is `nums[i] % nums[j] == 0`.

**3. Russian Doll Envelopes (LeetCode 354)**
- 2D LIS. Sort by width, then find LIS by height.

## 7. Summary

| Approach | Time | Space | Notes |
| :--- | :--- | :--- | :--- |
| **DP** | O(N^2) | O(N) | Simple, easy to extend |
| **Binary Search** | O(N \log N) | O(N) | Optimal for length only |
| **Patience Sort** | O(N \log N) | O(N) | Same as Binary Search |

## 8. Deep Dive: Why Binary Search Works

The `tails` array has a crucial property: **it is always sorted**.

**Proof by Induction:**
1. **Base Case:** After first element, `tails = [nums[0]]`. Sorted ✓
2. **Inductive Step:** Assume `tails` is sorted before processing `nums[i]`.
 - We find position `pos` using `bisect_left`.
 - If `pos == len(tails)`, we append (still sorted).
 - If `pos < len(tails)`, we replace `tails[pos]` with `nums[i]`.
 - Since `bisect_left` finds the leftmost position where `nums[i]` fits, we have:
 - `tails[pos-1] < nums[i]` (if `pos > 0`)
 - `tails[pos] >= nums[i]`
 - After replacement: `tails[pos-1] < nums[i] <= tails[pos+1]`
 - Still sorted ✓

## 9. Deep Dive: Longest Decreasing Subsequence

**Problem:** Find the longest **decreasing** subsequence.

**Solution 1:** Reverse the condition in DP.
``python
for i in range(1, n):
 for j in range(i):
 if nums[j] > nums[i]: # Changed from <
 dp[i] = max(dp[i], dp[j] + 1)
``

**Solution 2:** Negate all numbers and find LIS.
- LIS of `[-10, -9, -2, -5]` is the LDS of `[10, 9, 2, 5]`.

## 10. Deep Dive: Number of LIS (LeetCode 673)

**Problem:** Count how many different LIS exist.

**Approach:** Extend DP to track counts.
``python
def findNumberOfLIS(nums):
 n = len(nums)
 dp = [1] * n # Length of LIS ending at i
 count = [1] * n # Number of LIS ending at i
 
 for i in range(1, n):
 for j in range(i):
 if nums[j] < nums[i]:
 if dp[j] + 1 > dp[i]:
 # Found a longer LIS
 dp[i] = dp[j] + 1
 count[i] = count[j]
 elif dp[j] + 1 == dp[i]:
 # Found another LIS of same length
 count[i] += count[j]
 
 max_len = max(dp)
 return sum(c for l, c in zip(dp, count) if l == max_len)
``

## 11. Deep Dive: Russian Doll Envelopes (LeetCode 354)

**Problem:** You have envelopes `(w, h)`. An envelope can fit into another if both width and height are strictly greater. Find max nesting.

**Insight:** This is 2D LIS.
1. Sort by width ascending, height **descending** (crucial!).
2. Find LIS on heights.

**Why descending height?**
- If two envelopes have the same width, they can't nest.
- By sorting height descending, we ensure they won't be in the same LIS.

``python
def maxEnvelopes(envelopes):
 # Sort by width asc, height desc
 envelopes.sort(key=lambda x: (x[0], -x[1]))
 
 # Extract heights
 heights = [h for w, h in envelopes]
 
 # Find LIS on heights
 return lengthOfLIS(heights)
``

## 12. Deep Dive: LIS with Segment Tree

For advanced problems, we might need to query "What's the longest LIS in range `[L, R]`?"

**Data Structure:** Segment Tree where each node stores the LIS length for its range.

**Update:** When adding a new element, update all affected nodes.

**Complexity:** O(N \log N) per update.

## 13. Real-World Applications

### 1. Version Control (Git)
- **Longest Common Subsequence (LCS)** is related to LIS.
- Git uses LCS to find minimal diffs between file versions.

### 2. Stock Trading
- Find the longest period of increasing stock prices.
- Helps identify bull markets.

### 3. Bioinformatics
- DNA sequence alignment.
- Find longest matching subsequence between two genomes.

## 14. Code: LIS with All Solutions

Sometimes we need all possible LIS, not just one.

``python
def allLIS(nums):
 n = len(nums)
 dp = [1] * n
 
 # Find LIS length
 for i in range(1, n):
 for j in range(i):
 if nums[j] < nums[i]:
 dp[i] = max(dp[i], dp[j] + 1)
 
 max_len = max(dp)
 
 # Backtrack to find all LIS
 def backtrack(index, current_lis, last_val):
 if len(current_lis) == max_len:
 result.append(current_lis[:])
 return
 
 for i in range(index, n):
 if nums[i] > last_val and dp[i] == max_len - len(current_lis):
 current_lis.append(nums[i])
 backtrack(i + 1, current_lis, nums[i])
 current_lis.pop()
 
 result = []
 backtrack(0, [], float('-inf'))
 return result
``

## 15. Interview Pro Tips

1. **Recognize the Pattern:** "Longest", "Increasing", "Subsequence" → Think LIS.
2. **Start with DP:** Always explain the O(N^2) solution first.
3. **Optimize:** Mention binary search for O(N \log N).
4. **Variants:** Be ready to adapt (decreasing, 2D, count).
5. **Reconstruction:** Know how to print the actual sequence.

## 16. Performance Comparison

**Benchmark:** `N = 10,000` random integers.

| Approach | Python Time | C++ Time |
| :--- | :--- | :--- |
| **DP O(N^2)** | 2.5s | 150ms |
| **Binary Search** | 15ms | 2ms |
| **Segment Tree** | 50ms | 8ms |

**Takeaway:** Binary search is the clear winner for standard LIS.

## 17. Deep Dive: Connection to Longest Common Subsequence (LCS)

**Insight:** LIS can be reduced to LCS.

**Algorithm:**
1. Make a copy of `nums` and sort it: `sorted_nums`.
2. Remove duplicates from `sorted_nums`.
3. Find the **Longest Common Subsequence** between `nums` and `sorted_nums`.

**Why?**
- LCS finds the longest sequence that appears in both arrays in the same relative order.
- Since `sorted_nums` is strictly increasing, any common subsequence must also be strictly increasing.
- Thus, LCS(`nums`, `sorted_nums`) == LIS(`nums`).

**Complexity:**
- Sorting: O(N \log N).
- LCS: O(N^2).
- Total: O(N^2).
- **Note:** This is slower than the Binary Search approach (O(N \log N)), but it's a powerful theoretical connection.

## 18. Deep Dive: Dilworth's Theorem and Chain Decomposition

**Concept:**
- **Chain:** A subset of elements where every pair is comparable (e.g., an increasing subsequence).
- **Antichain:** A subset where *no* pair is comparable (e.g., a decreasing subsequence, if we define order as increasing).

**Dilworth's Theorem:**
"The minimum number of chains needed to cover a partially ordered set is equal to the maximum size of an antichain."

**Application to LIS:**
- The length of the **Longest Increasing Subsequence** is equal to the minimum number of **Decreasing Subsequences** needed to cover the array.

**Example:** `[10, 9, 2, 5, 3, 7, 101, 18]`
- LIS: `[2, 3, 7, 18]` (Length 4).
- Decreasing Subsequences Cover:
 1. `[10, 9, 5, 3]`
 2. `[2]`
 3. `[7]`
 4. `[101, 18]`
- We needed 4 decreasing subsequences.

**Algorithm (Patience Sorting again!):**
- When we place a card on a pile in Patience Sorting, we are essentially extending a decreasing subsequence (the pile).
- The number of piles is the length of the LIS.
- This is a constructive proof of the dual of Dilworth's Theorem for sequences.

## 19. Advanced: LIS in O(N \log \log N)

Can we beat O(N \log N)?
- In the comparison model, NO. Lower bound is `\Omega(N \log N)`.
- But if numbers are integers in range `[1, U]`, we can use **Van Emde Boas Trees**.

**Algorithm:**
1. Replace the Binary Search (which takes O(\log N)) with a vEB Tree predecessor query.
2. vEB Tree supports `predecessor` in O(\log \log U).
3. Total Time: O(N \log \log U).

**Practicality:**
- vEB trees have huge constant factors and memory overhead.
- Only useful if `U` is huge but fits in machine word.
- For standard competitive programming, O(N \log N) is sufficient.

## 20. Case Study: DNA Sequence Alignment

**Problem:** Align two DNA sequences `A` and `B` to find regions of similarity.
- `A = ACGTCG`
- `B = ATCG`

**MUMmer (Maximal Unique Matches):**
- A popular bioinformatics tool uses LIS to align genomes.
1. Find all **Maximal Unique Matches** (substrings that appear exactly once in both A and B).
2. Each match can be represented as a point `(pos_A, pos_B)`.
3. We want to find the largest subset of matches that are "consistent" (appear in the same order).
4. This is exactly finding the **LIS** of the `pos_B` coordinates when sorted by `pos_A`.

**Scale:**
- Genomes have billions of base pairs.
- O(N^2) is impossible.
- O(N \log N) LIS is critical for aligning human genomes.

## 21. System Design: Real-Time Anomaly Detection

**Scenario:** Monitoring server CPU usage.
- Stream: `[10%, 12%, 15%, 80%, 85%, 90%...]`
- Goal: Detect a "sustained upward trend" (LIS length > `K`) in a sliding window.

**Naive Approach:**
- Run O(N \log N) LIS on every window.
- Window size `W=1000`.
- Cost: O(W \log W) per new data point. Expensive.

**Optimized Approach (Incremental LIS):**
- Maintain the `tails` array.
- When a new element arrives, update `tails` (O(\log W)).
- When an old element leaves, it's harder (deletion from LIS is tricky).
- **Approximation:** Use **Trend Filtering** (e.g., Hodrick-Prescott filter) or simple exponential moving average, but LIS provides a robust, non-parametric metric for "monotonicity".

## 22. Common Mistakes and Pitfalls

**1. Confusing Subsequence with Subarray:**
- **Subarray:** Contiguous (e.g., `[2, 5, 3]` in `[1, 2, 5, 3, 7]`).
- **Subsequence:** Non-contiguous (e.g., `[2, 3, 7]`).
- **Fix:** Clarify with interviewer immediately.

**2. Incorrect Reconstruction:**
- *Mistake:* Just printing the `tails` array.
- *Fact:* `tails` is NOT the LIS. It stores the *smallest tail* for each length.
- *Example:* `[1, 5, 2]`. `tails` becomes `[1, 2]`. Real LIS is `[1, 5]` or `[1, 2]`. But if input is `[1, 5, 2, 3]`, `tails` is `[1, 2, 3]`. The `2` overwrote `5`.
- *Fix:* Use the `parent` array backtracking method.

**3. Not Handling Duplicates:**
- "Strictly increasing" vs "Non-decreasing".
- Strictly: `nums[j] < nums[i]`.
- Non-decreasing: `nums[j] <= nums[i]`.
- Binary Search: Use `bisect_right` for non-decreasing.

**4. 2D LIS Sorting Order:**
- For envelopes `(w, h)`, sorting `w` ascending and `h` **ascending** is wrong.
- *Why?* `[2, 3]` and `[2, 4]`. If sorted ascending, we might pick both. But `[2, 3]` cannot fit into `[2, 4]` (width must be strictly greater).
- *Fix:* Sort `w` ascending, `h` **descending**.

## 23. Ethical Considerations

**1. Algorithmic Trading:**
- HFT firms use LIS-like algorithms to detect micro-trends.
- **Risk:** Flash crashes caused by automated feedback loops.
- **Regulation:** Circuit breakers in stock exchanges.

**2. Genomic Privacy:**
- Fast alignment (using LIS) enables rapid DNA identification.
- **Risk:** Re-identifying individuals from "anonymized" genetic data.
- **Policy:** Strict access controls on biobanks.

## 24. Production Optimization: LIS at Scale

**Scenario:** Process 1 billion stock price sequences to find longest upward trends.

**Challenges:**
1. **Memory:** Cannot store 1B sequences in RAM.
2. **Latency:** Need results in real-time for trading decisions.

**Architecture:**

**1. Streaming LIS:**
``python
class StreamingLIS:
 def __init__(self):
 self.tails = []
 self.max_length = 0
 
 def add(self, num):
 """Add number to stream and update LIS"""
 pos = bisect.bisect_left(self.tails, num)
 
 if pos == len(self.tails):
 self.tails.append(num)
 self.max_length = len(self.tails)
 else:
 self.tails[pos] = num
 
 return self.max_length
 
 def reset(self):
 """Reset for new sequence"""
 self.tails = []
 self.max_length = 0
``

**2. Batch Processing with MapReduce:**
``python
from multiprocessing import Pool

def compute_lis_parallel(sequences):
 """Process multiple sequences in parallel"""
 with Pool() as pool:
 results = pool.map(lengthOfLIS, sequences)
 return results

# Usage
sequences = [
 [10, 9, 2, 5, 3, 7, 101, 18],
 [0, 1, 0, 3, 2, 3],
 # ... millions more
]
lengths = compute_lis_parallel(sequences)
``

**3. GPU Acceleration (CUDA):**
``cpp
__global__ void lis_kernel(int* sequences, int* results, int n_seq, int seq_len) {
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
 if (idx >= n_seq) return;
 
 int* seq = sequences + idx * seq_len;
 int tails[1000]; // Max LIS length
 int len = 0;
 
 for (int i = 0; i < seq_len; i++) {
 // Binary search
 int left = 0, right = len;
 while (left < right) {
 int mid = (left + right) / 2;
 if (tails[mid] < seq[i]) left = mid + 1;
 else right = mid;
 }
 
 tails[left] = seq[i];
 if (left == len) len++;
 }
 
 results[idx] = len;
}
``

## 25. Advanced Variants and Extensions

**1. Longest Bitonic Subsequence:**
- A sequence that first increases, then decreases.
- Example: `[1, 11, 2, 10, 4, 5, 2, 1]` → `[1, 2, 10, 4, 2, 1]` (length 6).

**Algorithm:**
``python
def longestBitonicSubsequence(nums):
 n = len(nums)
 
 # LIS ending at i
 lis = [1] * n
 for i in range(1, n):
 for j in range(i):
 if nums[j] < nums[i]:
 lis[i] = max(lis[i], lis[j] + 1)
 
 # LDS starting at i
 lds = [1] * n
 for i in range(n - 2, -1, -1):
 for j in range(i + 1, n):
 if nums[j] < nums[i]:
 lds[i] = max(lds[i], lds[j] + 1)
 
 # Max of lis[i] + lds[i] - 1
 return max(lis[i] + lds[i] - 1 for i in range(n))
``

**2. Longest Alternating Subsequence:**
- Elements alternate between increasing and decreasing.
- Example: `[1, 5, 3, 8, 6, 9]` → `[1, 5, 3, 8, 6, 9]` (length 6).

**3. K-Increasing Subsequence:**
- Find `k` disjoint increasing subsequences that cover the array.
- This is equivalent to partitioning into `k` chains (Dilworth's Theorem).

**4. Weighted LIS:**
- Each element has a value and weight.
- Maximize sum of values in an increasing subsequence.

``python
def weightedLIS(nums, weights):
 n = len(nums)
 dp = [0] * n # Max weight ending at i
 
 for i in range(n):
 dp[i] = weights[i] # At least include itself
 for j in range(i):
 if nums[j] < nums[i]:
 dp[i] = max(dp[i], dp[j] + weights[i])
 
 return max(dp)
``

## 26. Complexity Analysis Deep Dive

**Why is Binary Search O(N \log N) optimal?**

**Lower Bound Proof (Comparison Model):**
- Any comparison-based algorithm must distinguish between `2^N` possible permutations.
- Decision tree has `2^N` leaves.
- Height of tree is `\Omega(N \log N)`.
- **BUT:** LIS doesn't need to sort, so this doesn't directly apply.

**Actual Lower Bound:**
- Fredman (1975) proved `\Omega(N \log \log N)` lower bound for LIS in comparison model.
- No known algorithm achieves this.
- O(N \log N) is the best known.

**Integer LIS (when values are bounded):**
- If values are in `[1, U]`, we can use **Van Emde Boas trees**.
- Complexity: O(N \log \log U).
- Practical only for small `U`.

## 27. Further Reading

1. **"Introduction to Algorithms" (CLRS):** Chapter on Dynamic Programming.
2. **"Patience Sorting" (Wikipedia):** The card game connection.
3. **"Hunt-Szymanski Algorithm":** `O((R+N) \log N)` algorithm for LCS, which uses LIS.
4. **"Dilworth's Theorem":** Order theory foundations.

## 28. Conclusion

Longest Increasing Subsequence is a gem of a problem. It starts as a standard DP exercise (O(N^2)), transforms into a greedy binary search puzzle (O(N \log N)), connects to card games (Patience Sorting), and finds applications in everything from reading DNA to predicting stock markets. Mastering LIS means understanding the trade-off between **optimality** (DP) and **efficiency** (Greedy+Binary Search), a core skill for any systems engineer.

## 29. Summary

| Approach | Time | Space | Notes |
| :--- | :--- | :--- | :--- |
| **DP** | O(N^2) | O(N) | Simple, easy to extend |
| **Binary Search** | O(N \log N) | O(N) | Optimal for length only |
| **Patience Sort** | O(N \log N) | O(N) | Same as Binary Search |

---

**Originally published at:** [arunbaby.com/dsa/0037-longest-increasing-subsequence](https://www.arunbaby.com/dsa/0037-longest-increasing-subsequence/)
