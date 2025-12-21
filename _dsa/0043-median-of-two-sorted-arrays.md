---
title: "Median of Two Sorted Arrays"
day: 43
collection: dsa
categories:
  - dsa
tags:
  - binary-search
  - array
  - divide-and-conquer
difficulty: Hard
---

**"Finding the middle ground between two ordered worlds."**

## 1. Problem Statement

Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return the **median** of the two sorted arrays.

The overall run time complexity should be $O(\log(m+n))$.

**Example 1:**
```
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.
```

**Example 2:**
```
Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.
```

**Constraints:**
*   `nums1.length == m`
*   `nums2.length == n`
*   `0 <= m <= 1000`
*   `0 <= n <= 1000`
*   `1 <= m + n <= 2000`
*   `-10^6 <= nums1[i], nums2[i] <= 10^6`

## 2. Understanding the Median

**Definition:**
*   For an odd-length array: the middle element.
*   For an even-length array: the average of the two middle elements.

**Key Insight:** The median divides the combined array into two equal halves:
*   Left half: All elements ≤ median.
*   Right half: All elements ≥ median.

## 3. Approach 1: Merge and Find (Brute Force)

**Algorithm:**
1.  Merge both arrays into one sorted array.
2.  Find the median.

```python
def findMedianSortedArrays(nums1, nums2):
    merged = sorted(nums1 + nums2)
    n = len(merged)
    
    if n % 2 == 1:
        return merged[n // 2]
    else:
        return (merged[n // 2 - 1] + merged[n // 2]) / 2
```

**Complexity:**
*   **Time:** $O((m+n) \log(m+n))$ for sorting, or $O(m+n)$ if using merge sort's merge step.
*   **Space:** $O(m+n)$ for the merged array.

**Issue:** Doesn't meet the $O(\log(m+n))$ requirement.

## 4. Approach 2: Two-Pointer Merge (No Full Merge)

**Algorithm:**
1.  Use two pointers to "virtually" merge.
2.  Stop when we reach the median position.

```python
def findMedianSortedArrays(nums1, nums2):
    m, n = len(nums1), len(nums2)
    total = m + n
    
    # We need element at (total-1)//2 and total//2 for median
    target1 = (total - 1) // 2
    target2 = total // 2
    
    i = j = 0
    val1 = val2 = 0
    
    for k in range(target2 + 1):
        if i < m and (j >= n or nums1[i] <= nums2[j]):
            val = nums1[i]
            i += 1
        else:
            val = nums2[j]
            j += 1
        
        if k == target1:
            val1 = val
        if k == target2:
            val2 = val
    
    return (val1 + val2) / 2
```

**Complexity:**
*   **Time:** $O(m+n)$.
*   **Space:** $O(1)$.

**Still doesn't meet** the $O(\log(m+n))$ requirement.

## 5. Approach 3: Binary Search (Optimal)

**Key Insight:** We need to partition both arrays such that:
1.  Left partition has `(m + n + 1) // 2` elements.
2.  All elements in left partition ≤ all elements in right partition.

**Algorithm:**
1.  Binary search on the smaller array.
2.  For each partition of the smaller array, compute the corresponding partition of the larger array.
3.  Check if the partition is valid.

```python
def findMedianSortedArrays(nums1, nums2):
    # Ensure nums1 is the smaller array
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    total = m + n
    half = (total + 1) // 2
    
    lo, hi = 0, m
    
    while lo <= hi:
        i = (lo + hi) // 2  # Partition index for nums1
        j = half - i         # Partition index for nums2
        
        # Edge handling with -inf and +inf
        nums1_left = nums1[i - 1] if i > 0 else float('-inf')
        nums1_right = nums1[i] if i < m else float('inf')
        nums2_left = nums2[j - 1] if j > 0 else float('-inf')
        nums2_right = nums2[j] if j < n else float('inf')
        
        # Check if partition is valid
        if nums1_left <= nums2_right and nums2_left <= nums1_right:
            # Valid partition found
            if total % 2 == 1:
                return max(nums1_left, nums2_left)
            else:
                return (max(nums1_left, nums2_left) + min(nums1_right, nums2_right)) / 2
        elif nums1_left > nums2_right:
            hi = i - 1  # Too many elements from nums1
        else:
            lo = i + 1  # Too few elements from nums1
    
    return 0  # Should never reach here
```

**Complexity:**
*   **Time:** $O(\log(\min(m, n)))$.
*   **Space:** $O(1)$.

## 6. Detailed Walkthrough

**Example:** `nums1 = [1, 3, 8, 9, 15]`, `nums2 = [7, 11, 18, 19, 21, 25]`

**Setup:**
*   m = 5, n = 6, total = 11.
*   half = (11 + 1) // 2 = 6.
*   Need 6 elements in left partition.

**Binary Search:**

**Iteration 1:** lo=0, hi=5, i=2, j=6-2=4
*   nums1_left = nums1[1] = 3
*   nums1_right = nums1[2] = 8
*   nums2_left = nums2[3] = 19
*   nums2_right = nums2[4] = 21
*   Check: 3 ≤ 21 ✓, 19 ≤ 8 ✗
*   19 > 8, so we need more from nums1. lo = 3.

**Iteration 2:** lo=3, hi=5, i=4, j=6-4=2
*   nums1_left = nums1[3] = 9
*   nums1_right = nums1[4] = 15
*   nums2_left = nums2[1] = 11
*   nums2_right = nums2[2] = 18
*   Check: 9 ≤ 18 ✓, 11 ≤ 15 ✓
*   Valid partition!
*   Odd total: median = max(9, 11) = 11.

**Merged Array (for verification):** [1, 3, 7, 8, 9, **11**, 15, 18, 19, 21, 25]. Median = 11. ✓

## 7. Why Binary Search Works

**Invariant:** If we take `i` elements from nums1, we must take `half - i` elements from nums2.

**Validity Condition:**
*   `nums1[i-1] ≤ nums2[j]`: Largest in nums1's left ≤ smallest in nums2's right.
*   `nums2[j-1] ≤ nums1[i]`: Largest in nums2's left ≤ smallest in nums1's right.

**Binary Search Logic:**
*   If `nums1[i-1] > nums2[j]`: We took too many from nums1. Decrease i.
*   If `nums2[j-1] > nums1[i]`: We took too few from nums1. Increase i.

## 8. Edge Cases

1.  **One Array is Empty:**
    *   `nums1 = []`, `nums2 = [2, 3]` → Median of nums2 = 2.5.
    *   Handle with `-inf` and `+inf` sentinels.

2.  **Arrays of Different Sizes:**
    *   Always binary search on the smaller array for efficiency.

3.  **All Elements in One Array are Smaller:**
    *   `nums1 = [1, 2]`, `nums2 = [3, 4, 5, 6]`.
    *   Partition: All of nums1 in left, some of nums2 in left.

4.  **Single Element Arrays:**
    *   `nums1 = [1]`, `nums2 = [2]` → Median = 1.5.

## 9. System Design: Distributed Median Finding

**Problem:** Find the median of data distributed across k servers.

**Naive Approach:** Collect all data, sort, find median. $O(N \log N)$.

**Efficient Approach (Sampling-Based):**
1.  **Sample:** Each server sends a random sample.
2.  **Estimate:** Find approximate median from samples.
3.  **Count:** Each server counts elements < median, > median.
4.  **Refine:** Narrow the range containing the true median.

**Algorithm (Binary Search on Value):**
1.  Binary search on the median value (not indices).
2.  For each candidate value, count how many elements are ≤ it.
3.  If count = (total + 1) / 2, we found the median.

## 10. Deep Dive: Kth Element in Two Sorted Arrays

**Generalization:** Find the k-th smallest element in two sorted arrays.

**Algorithm (Binary Search):**
```python
def findKthElement(nums1, nums2, k):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    lo, hi = max(0, k - n), min(k, m)
    
    while lo <= hi:
        i = (lo + hi) // 2
        j = k - i
        
        nums1_left = nums1[i - 1] if i > 0 else float('-inf')
        nums1_right = nums1[i] if i < m else float('inf')
        nums2_left = nums2[j - 1] if j > 0 else float('-inf')
        nums2_right = nums2[j] if j < n else float('inf')
        
        if nums1_left <= nums2_right and nums2_left <= nums1_right:
            return max(nums1_left, nums2_left)
        elif nums1_left > nums2_right:
            hi = i - 1
        else:
            lo = i + 1
    
    return -1
```

**Use Case:** Median is just k = (m + n + 1) // 2.

## 11. Production Application: Real-Time Percentile Computation

**Scenario:** Compute the 99th percentile latency from logs stored on multiple servers.

**Algorithm:**
1.  **Histogram Binning:** Each server maintains a histogram of latencies.
2.  **Merge Histograms:** Sum histograms across servers.
3.  **Find Percentile:** Walk through the merged histogram.

**Optimization:**
*   Use exponential binning (log-scale) for accuracy across wide ranges.
*   Stream histograms instead of raw data (reduces network traffic).

## 12. Interview Questions

1.  **Median of Two Sorted Arrays (Classic):** Solve with binary search.
2.  **Kth Smallest Element:** Generalize the median solution.
3.  **Median of Data Stream:** Design with two heaps.
4.  **Distributed Median:** Explain the sampling + counting approach.
5.  **Running Median:** Use balanced BST or two heaps.

## 13. Common Mistakes

*   **Not Handling Empty Arrays:** Leads to index out of bounds.
*   **Off-by-One in Partition:** Careful with i-1, j-1.
*   **Binary Searching the Wrong Array:** Always search the smaller one.
*   **Integer Overflow:** (a + b) / 2 can overflow. Use a + (b - a) / 2.
*   **Forgetting Even/Odd Cases:** Median calculation differs.

## 14. Variant: Median of K Sorted Arrays

**Problem:** Given k sorted arrays, find the overall median.

**Approach 1: Sequential Merge**
*   Merge pairs until one array remains.
*   Find median.
*   **Time:** $O(N \log k)$ where N is total elements.

**Approach 2: Binary Search on Value**
*   Binary search on the median value.
*   For each candidate, count elements ≤ it in all arrays using binary search.
*   **Time:** $O(k \log V \log N)$ where V is the value range.

## 15. Mathematical Proof of Correctness

**Theorem:** The binary search algorithm correctly finds the median.

**Proof:**
1.  **Partition Property:** The left partition has exactly `(m + n + 1) // 2` elements.
2.  **Validity:** If `nums1[i-1] ≤ nums2[j]` and `nums2[j-1] ≤ nums1[i]`, then all left elements ≤ all right elements.
3.  **Binary Search Correctness:** 
    *   If `nums1[i-1] > nums2[j]`, we need fewer elements from nums1 (decrease i).
    *   If `nums2[j-1] > nums1[i]`, we need more elements from nums1 (increase i).
4.  **Termination:** Binary search terminates when a valid partition is found.
5.  **Median:** The median is max(left) or (max(left) + min(right)) / 2.

## 16. Deep Dive: Weighted Median

**Problem:** Given elements with weights, find the median where weight sums to 50% on each side.

**Algorithm:**
1.  Sort elements by value.
2.  Accumulate weights.
3.  Find where cumulative weight = total_weight / 2.

**Use Case:** Weighted voting, robust averaging.

## 17. Performance Benchmarking

```python
import time
import random

def benchmark():
    sizes = [(100, 100), (1000, 1000), (10000, 10000)]
    
    for m, n in sizes:
        nums1 = sorted(random.sample(range(1000000), m))
        nums2 = sorted(random.sample(range(1000000), n))
        
        # Binary Search
        start = time.time()
        for _ in range(1000):
            findMedianSortedArrays(nums1, nums2)
        bs_time = time.time() - start
        
        # Merge
        start = time.time()
        for _ in range(1000):
            findMedian_merge(nums1, nums2)
        merge_time = time.time() - start
        
        print(f"m={m}, n={n}: BS={bs_time:.4f}s, Merge={merge_time:.4f}s")

# Expected: Binary search is significantly faster for large arrays
```

## 18. Alternative: Divide and Conquer

**Algorithm:**
1.  Compare medians of both arrays.
2.  Discard the half that cannot contain the overall median.
3.  Recur on the remaining halves.

```python
def findMedian_DC(nums1, nums2):
    def kth(a, a_start, a_end, b, b_start, b_end, k):
        if a_start > a_end:
            return b[b_start + k - 1]
        if b_start > b_end:
            return a[a_start + k - 1]
        if k == 1:
            return min(a[a_start], b[b_start])
        
        mid_a = float('inf') if a_start + k // 2 - 1 > a_end else a[a_start + k // 2 - 1]
        mid_b = float('inf') if b_start + k // 2 - 1 > b_end else b[b_start + k // 2 - 1]
        
        if mid_a < mid_b:
            return kth(a, a_start + k // 2, a_end, b, b_start, b_end, k - k // 2)
        else:
            return kth(a, a_start, a_end, b, b_start + k // 2, b_end, k - k // 2)
    
    m, n = len(nums1), len(nums2)
    total = m + n
    
    if total % 2 == 1:
        return kth(nums1, 0, m - 1, nums2, 0, n - 1, total // 2 + 1)
    else:
        left = kth(nums1, 0, m - 1, nums2, 0, n - 1, total // 2)
        right = kth(nums1, 0, m - 1, nums2, 0, n - 1, total // 2 + 1)
        return (left + right) / 2
```

**Complexity:** $O(\log(m + n))$.

## 19. Conclusion

The Median of Two Sorted Arrays problem is a classic example of how binary search can achieve logarithmic time complexity in non-obvious ways.

**Key Takeaways:**
*   **Binary Search on Partition:** Instead of searching for a value, search for a valid partition.
*   **Invariant Maintenance:** Ensure left and right partitions satisfy the ordering property.
*   **Edge Handling:** Use sentinels (-inf, +inf) for boundary cases.
*   **Generalization:** The same technique works for finding the k-th element.

This problem demonstrates that mastering binary search goes beyond simple "find the target"—it's about identifying the right search space.

## 20. Related Problems

*   **Median of Data Stream** (LeetCode 295)
*   **Kth Smallest Element in a Sorted Matrix** (LeetCode 378)
*   **Find K-th Smallest Pair Distance** (LeetCode 719)
*   **Split Array Largest Sum** (LeetCode 410)

Practice these to solidify your understanding of binary search on answer patterns!

## 21. Deep Dive: Median in a Stream (Two Heaps)

**Problem:** Find the median as elements are added one by one.

**Algorithm:**
1.  Maintain two heaps:
    *   **Max-Heap (left):** Smaller half.
    *   **Min-Heap (right):** Larger half.
2.  Balance: |left| = |right| or |left| = |right| + 1.
3.  Median = top(left) or (top(left) + top(right)) / 2.

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.left = []   # Max-heap (negate values)
        self.right = []  # Min-heap
    
    def addNum(self, num):
        # Add to left (max-heap)
        heapq.heappush(self.left, -num)
        
        # Balance: Move largest from left to right
        heapq.heappush(self.right, -heapq.heappop(self.left))
        
        # Ensure left has at least as many elements
        if len(self.left) < len(self.right):
            heapq.heappush(self.left, -heapq.heappop(self.right))
    
    def findMedian(self):
        if len(self.left) > len(self.right):
            return -self.left[0]
        return (-self.left[0] + self.right[0]) / 2
```

**Complexity:**
*   **Add:** $O(\log n)$.
*   **Find Median:** $O(1)$.

## 22. Deep Dive: Median in a Matrix (Binary Search on Value)

**Problem:** Find the median of a row-wise sorted matrix.

**Algorithm:**
1.  Binary search on the value (not index).
2.  For each candidate value, count elements ≤ it in each row.
3.  If count = (rows × cols + 1) / 2, we found the median.

```python
def matrixMedian(matrix):
    rows, cols = len(matrix), len(matrix[0])
    
    lo = min(row[0] for row in matrix)
    hi = max(row[-1] for row in matrix)
    
    target = (rows * cols + 1) // 2
    
    while lo < hi:
        mid = (lo + hi) // 2
        
        # Count elements <= mid in all rows
        count = sum(bisect.bisect_right(row, mid) for row in matrix)
        
        if count < target:
            lo = mid + 1
        else:
            hi = mid
    
    return lo
```

**Complexity:** $O(R \cdot \log C \cdot \log(\max - \min))$.

## 23. Variant: Sliding Window Median

**Problem:** Find the median of each window of size k.

**Approach:**
1.  Use two heaps (like streaming median).
2.  Use lazy deletion to handle elements leaving the window.

**Challenge:** Efficiently removing elements from heaps.

```python
from sortedcontainers import SortedList

def medianSlidingWindow(nums, k):
    window = SortedList()
    result = []
    
    for i, num in enumerate(nums):
        window.add(num)
        
        if len(window) > k:
            window.remove(nums[i - k])
        
        if len(window) == k:
            if k % 2 == 1:
                result.append(window[k // 2])
            else:
                result.append((window[k // 2 - 1] + window[k // 2]) / 2)
    
    return result
```

**Complexity:** $O(n \cdot \log k)$ with balanced BST.

## 24. Application: A/B Testing Statistics

**Scenario:** Compare median latency between control and treatment groups.

**Challenge:** Latencies are stored on multiple servers.

**Approach:**
1.  Each server computes a histogram of latencies.
2.  Merge histograms.
3.  Find median from merged histogram.

**Benefit:** Avoids transferring raw data.

## 25. Application: Database Query Optimization

**Scenario:** SQL query `SELECT MEDIAN(column) FROM table`.

**Implementation:**
1.  If table is sorted: Direct access to middle element.
2.  If unsorted: Use quickselect ($O(n)$ average) or sort ($O(n \log n)$).
3.  For approximate median: Use sampling or t-digest.

## 26. Deep Dive: Approximating Median with T-Digest

**T-Digest** is a data structure for approximate percentile computation.

**Algorithm:**
1.  Maintain a set of centroids (mean, count).
2.  Centroids near the median are small (high precision).
3.  Centroids at the extremes are large (low precision).

**Operations:**
*   **Add:** Merge new element into nearest centroid.
*   **Query:** Walk through centroids to find percentile.

**Use Case:** Real-time percentile computation at scale (Netflix, Elasticsearch).

## 27. Testing Strategy

**Test Cases:**
1.  **Both Arrays Empty:** Handle gracefully.
2.  **One Array Empty:** Return median of the other.
3.  **Same Size Arrays:** `[1,2]`, `[3,4]` → 2.5.
4.  **Different Size Arrays:** `[1,3]`, `[2]` → 2.
5.  **All Elements in One Array Smaller:** `[1,2]`, `[3,4,5,6]`.
6.  **Duplicates:** `[1,1,1]`, `[1,1,1]` → 1.
7.  **Large Arrays:** Performance testing.

**Edge Cases:**
*   Single element in each array.
*   Negative numbers.
*   Very large values.

## 28. Interview Tips

**Step-by-Step Approach:**
1.  **Clarify:** Sizes, sorted order, allowed time complexity.
2.  **Brute Force:** Merge and find median ($O(m+n)$).
3.  **Optimize:** Binary search on partition ($O(\log \min(m,n))$).
4.  **Walk Through:** Use a specific example.
5.  **Edge Cases:** Empty arrays, single elements.
6.  **Code:** Write clean, bug-free code.
7.  **Complexity:** Analyze time and space.

## 29. Common Follow-up Questions

**Q: What if arrays are not sorted?**
**A:** Sort first ($O(n \log n)$), or use quickselect.

**Q: What if we can't afford $O(\log(m+n))$?**
**A:** Two-pointer merge is $O(m+n)$ and may be acceptable.

**Q: What if arrays are on different machines?**
**A:** Use the sampling/counting approach for distributed median.

**Q: What if we need the k-th element instead of median?**
**A:** Same algorithm, just change the target partition size.

## 30. Mastery Checklist

**Mastery Checklist:**
- [ ] Implement brute force (merge and find)
- [ ] Implement two-pointer (virtual merge)
- [ ] Implement binary search on partition
- [ ] Handle edge cases (empty arrays, single elements)
- [ ] Generalize to k-th element
- [ ] Understand the divide-and-conquer approach
- [ ] Explain why binary search works (partition invariant)
- [ ] Implement streaming median (two heaps)
- [ ] Implement sliding window median
- [ ] Understand distributed median algorithms

## 31. Conclusion

The Median of Two Sorted Arrays is one of the most elegant problems in computer science. It combines:
*   **Binary Search:** A powerful algorithmic paradigm.
*   **Partition Logic:** Dividing arrays into meaningful halves.
*   **Edge Case Handling:** Using sentinels for boundaries.

Mastering this problem opens the door to many advanced topics: distributed computing, real-time statistics, and database optimization. It's a testament to how mathematical insight can lead to efficient algorithms.

**The journey from $O(m+n)$ to $O(\log \min(m,n))$ teaches us that the right perspective can transform a problem.**



---

**Originally published at:** [arunbaby.com/dsa/0043-median-of-two-sorted-arrays](https://www.arunbaby.com/dsa/0043-median-of-two-sorted-arrays/)

*If you found this helpful, consider sharing it with others who might benefit.*

