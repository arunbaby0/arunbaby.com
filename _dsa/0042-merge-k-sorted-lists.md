---
title: "Merge K Sorted Lists"
day: 42
collection: dsa
categories:
  - dsa
tags:
  - linked-list
  - heap
  - divide-and-conquer
  - priority-queue
difficulty: Hard
---

**"Combining order from chaos, one element at a time."**

## 1. Problem Statement

You are given an array of `k` linked-lists `lists`, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

**Example 1:**
```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6
```

**Example 2:**
```
Input: lists = []
Output: []
```

**Example 3:**
```
Input: lists = [[]]
Output: []
```

**Constraints:**
*   `k == lists.length`
*   `0 <= k <= 10^4`
*   `0 <= lists[i].length <= 500`
*   `-10^4 <= lists[i][j] <= 10^4`
*   `lists[i]` is sorted in ascending order.
*   The sum of `lists[i].length` will not exceed `10^4`.

## 2. Intuition

This problem tests your understanding of several key concepts:
1.  **Heap (Priority Queue):** Efficiently find the minimum among k elements.
2.  **Divide and Conquer:** Recursively merge pairs of lists.
3.  **Linked List Manipulation:** Pointer management.

The key insight is that at any point, we need to pick the smallest element among the heads of all k lists. A min-heap gives us O(log k) access to the minimum.

## 3. Approach 1: Brute Force (Collect All, Sort, Rebuild)

**Algorithm:**
1.  Traverse all lists and collect all values into an array.
2.  Sort the array.
3.  Create a new linked list from the sorted array.

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        nodes = []
        
        # Collect all values
        for lst in lists:
            while lst:
                nodes.append(lst.val)
                lst = lst.next
        
        # Sort
        nodes.sort()
        
        # Rebuild linked list
        dummy = ListNode(0)
        curr = dummy
        for val in nodes:
            curr.next = ListNode(val)
            curr = curr.next
        
        return dummy.next
```

**Complexity:**
*   **Time:** $O(N \log N)$ where $N$ is total number of nodes.
*   **Space:** $O(N)$ to store all values.

## 4. Approach 2: Compare One by One

**Algorithm:**
1.  Compare the head of each list.
2.  Pick the minimum.
3.  Move the pointer of the selected list forward.
4.  Repeat until all lists are exhausted.

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        dummy = ListNode(0)
        curr = dummy
        
        while True:
            min_idx = -1
            min_val = float('inf')
            
            # Find the minimum head
            for i, lst in enumerate(lists):
                if lst and lst.val < min_val:
                    min_val = lst.val
                    min_idx = i
            
            if min_idx == -1:
                break
            
            # Add to result
            curr.next = lists[min_idx]
            curr = curr.next
            lists[min_idx] = lists[min_idx].next
        
        return dummy.next
```

**Complexity:**
*   **Time:** $O(N \cdot k)$ where $k$ is the number of lists. For each node, we scan k lists.
*   **Space:** $O(1)$ (excluding output).

## 5. Approach 3: Min-Heap (Priority Queue) - Optimal

**Algorithm:**
1.  Push the head of each list into a min-heap.
2.  Pop the minimum node from the heap.
3.  Add it to the result.
4.  Push the next node from that list into the heap.
5.  Repeat until the heap is empty.

```python
import heapq

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # Handle edge case
        if not lists:
            return None
        
        # Min-heap: (value, index, node)
        # Index is used as a tiebreaker to avoid comparing ListNode objects
        heap = []
        
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst.val, i, lst))
        
        dummy = ListNode(0)
        curr = dummy
        
        while heap:
            val, idx, node = heapq.heappop(heap)
            curr.next = node
            curr = curr.next
            
            if node.next:
                heapq.heappush(heap, (node.next.val, idx, node.next))
        
        return dummy.next
```

**Complexity:**
*   **Time:** $O(N \log k)$. Each of N nodes is pushed and popped once. Each operation is $O(\log k)$.
*   **Space:** $O(k)$ for the heap.

## 6. Approach 4: Divide and Conquer

**Algorithm:**
1.  Pair up lists and merge each pair.
2.  Repeat until only one list remains.

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None
        
        while len(lists) > 1:
            merged = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if i + 1 < len(lists) else None
                merged.append(self.mergeTwoLists(l1, l2))
            lists = merged
        
        return lists[0]
    
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        curr = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        
        curr.next = l1 if l1 else l2
        return dummy.next
```

**Complexity:**
*   **Time:** $O(N \log k)$. We merge k lists in $\log k$ rounds. Each round processes N nodes.
*   **Space:** $O(\log k)$ for recursion stack (if implemented recursively) or $O(1)$ (iterative).

## 7. Deep Dive: Why Min-Heap is Optimal

**Analysis:**
*   **k lists**, each of average length $\frac{N}{k}$.
*   At any time, the heap has at most $k$ elements.
*   Each node is pushed once: $O(\log k)$.
*   Each node is popped once: $O(\log k)$.
*   **Total:** $O(N \log k)$.

**Comparison:**
| Approach | Time | Space |
|----------|------|-------|
| Brute Force (Sort All) | $O(N \log N)$ | $O(N)$ |
| Compare One by One | $O(N \cdot k)$ | $O(1)$ |
| Min-Heap | $O(N \log k)$ | $O(k)$ |
| Divide and Conquer | $O(N \log k)$ | $O(\log k)$ |

**When $k$ is small (e.g., $k = 10$):** All approaches are similar.
**When $k$ is large (e.g., $k = 10000$):** Min-Heap and Divide & Conquer are much faster.

## 8. Detailed Walkthrough

**Example:** `lists = [[1,4,5],[1,3,4],[2,6]]`

**Min-Heap Approach:**

**Initial Heap:** [(1, 0, node1), (1, 1, node2), (2, 2, node3)]

**Iteration 1:**
*   Pop (1, 0, node1). Add to result: 1.
*   Push (4, 0, node1.next).
*   Heap: [(1, 1, node2), (2, 2, node3), (4, 0, node4)]

**Iteration 2:**
*   Pop (1, 1, node2). Add to result: 1.
*   Push (3, 1, node2.next).
*   Heap: [(2, 2, node3), (4, 0, node4), (3, 1, node5)]

**Iteration 3:**
*   Pop (2, 2, node3). Add to result: 2.
*   Push (6, 2, node3.next).
*   Heap: [(3, 1, node5), (4, 0, node4), (6, 2, node6)]

**... and so on**

**Final Result:** 1 → 1 → 2 → 3 → 4 → 4 → 5 → 6

## 9. System Design: External Merge Sort

Merge K Sorted Lists is the core of **External Merge Sort**, used when data doesn't fit in memory.

**Scenario:** Sort 100GB of data with 4GB RAM.

**Algorithm:**
1.  **Split:** Divide 100GB into 25 chunks of 4GB each.
2.  **Sort:** Load each chunk into memory, sort using quicksort, write back.
3.  **Merge:** Use a min-heap to merge 25 sorted chunks.

**Optimization:**
*   **Multi-way Merge:** Instead of 2-way merge, use k-way merge with a heap.
*   **Buffer Size:** Read/write in large buffers (e.g., 64KB) to reduce I/O.
*   **Parallel Merge:** Merge multiple pairs in parallel.

## 10. Deep Dive: K-Way Merge in Databases

**Use Case:** Merge results from k shards in a distributed database.

**Example (Cassandra):**
1.  Query sent to k nodes (shards).
2.  Each shard returns sorted results.
3.  Coordinator merges results using k-way merge.

**Optimization:**
*   **Async I/O:** Fetch from shards in parallel.
*   **Streaming:** Start merging as soon as first results arrive.
*   **Limit:** If only top 10 results are needed, stop early.

## 11. Variant: Merge K Sorted Arrays

**Problem:** Same as Merge K Sorted Lists, but with arrays instead of linked lists.

**Difference:** With arrays, we use indices instead of pointers.

```python
import heapq

def mergeKSortedArrays(arrays):
    heap = []
    
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))
    
    result = []
    
    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))
    
    return result
```

## 12. Variant: Merge K Sorted Streams

**Problem:** Each "list" is an infinite stream (e.g., from a socket).

**Challenge:** Can't store all elements. Need to process in a streaming fashion.

**Solution:**
1.  Maintain a heap of size k.
2.  Pop the minimum.
3.  Output it (or process it).
4.  Read the next element from that stream and push to heap.

**Use Case:** Merging log streams from k servers in real-time.

## 13. Production Application: Time-Series Databases

**Scenario:** InfluxDB merging time-series data from k sensors.

**Query:** "Get all temperature readings from 100 sensors, sorted by timestamp."

**Algorithm:**
1.  Each sensor has its own sorted time-series chunk.
2.  Use k-way merge to combine chunks.
3.  Apply filters (e.g., timestamp > X) during merge.

**Optimization:**
*   **Bloom Filters:** Skip chunks that don't match the filter.
*   **Column Storage:** Read only the timestamp column for merging.

## 14. Interview Questions

1.  **Merge K Sorted Lists (Classic):** Solve with a min-heap.
2.  **Find the Kth Smallest Element in K Sorted Lists:** Stop after popping k elements.
3.  **Merge K Sorted Arrays with O(1) Extra Space:** Possible? (No, need at least O(k) for heap.)
4.  **External Merge Sort:** Explain and implement.
5.  **Merge Intervals from K Streams:** Each stream gives intervals, merge overlapping ones.

## 15. Common Mistakes

*   **Comparing ListNode Objects:** Python's heapq compares tuples. If values are equal, it compares the second element. Use an index as a tiebreaker.
*   **Empty Lists:** Handle `lists = []` or `lists = [[]]`.
*   **Null Pointer Exception:** Always check if a node exists before accessing `node.val`.
*   **Off-by-One in Divide and Conquer:** When pairing lists, handle odd-length arrays.

## 16. Performance Benchmarking

```python
import time
import random
import heapq

def benchmark():
    k_values = [10, 100, 1000]
    n_per_list = 1000
    
    for k in k_values:
        # Generate k sorted lists
        lists = [sorted([random.randint(1, 100000) for _ in range(n_per_list)]) for _ in range(k)]
        
        # Heap approach
        start = time.time()
        heap_result = mergeKSortedArrays(lists)
        heap_time = time.time() - start
        
        print(f"k={k}: Heap={heap_time:.4f}s")

# Expected: Time increases logarithmically with k
```

## 17. Deep Dive: Custom Heap with Lazy Deletion

**Problem:** In some variants, we need to remove elements from the heap (not just the min).

**Solution:** Lazy Deletion.

**Algorithm:**
1.  Mark elements as "deleted" instead of removing them.
2.  When popping, skip deleted elements.
3.  Periodically rebuild the heap to remove deleted elements.

```python
class LazyHeap:
    def __init__(self):
        self.heap = []
        self.deleted = set()
    
    def push(self, item):
        heapq.heappush(self.heap, item)
    
    def pop(self):
        while self.heap:
            item = heapq.heappop(self.heap)
            if item not in self.deleted:
                return item
        return None
    
    def delete(self, item):
        self.deleted.add(item)
```

## 18. Advanced: Parallel K-Way Merge

**Problem:** Merge k sorted lists using multiple threads.

**Approach:**
1.  **Divide:** Assign pairs of lists to different threads.
2.  **Merge:** Each thread merges its pair.
3.  **Reduce:** Recursively merge the results.

**Parallelism:** log(k) rounds, each round can be parallelized.

**Implementation (Threading):**
```python
from concurrent.futures import ThreadPoolExecutor

def parallelMergeKLists(lists, num_threads=4):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        while len(lists) > 1:
            pairs = [(lists[i], lists[i+1] if i+1 < len(lists) else None) 
                     for i in range(0, len(lists), 2)]
            futures = [executor.submit(mergeTwoLists, l1, l2) for l1, l2 in pairs]
            lists = [f.result() for f in futures]
    
    return lists[0] if lists else None
```

## 19. Mathematical Analysis

**Recurrence Relation (Divide and Conquer):**
*   $T(k) = 2T(k/2) + O(N)$
*   By Master Theorem: $T(k) = O(N \log k)$.

**Lower Bound:**
*   We must look at each of N elements at least once: $\Omega(N)$.
*   We must compare elements from k lists: Information-theoretic lower bound $\Omega(\log k!)$ comparisons to determine order.
*   Combined: $\Omega(N \log k)$ is tight.

## 20. Conclusion

Merge K Sorted Lists is a foundational problem that appears in many real-world systems:
*   **External Merge Sort:** Sorting data larger than memory.
*   **Distributed Databases:** Merging results from multiple shards.
*   **Log Aggregation:** Combining logs from multiple servers.
*   **Time-Series Databases:** Merging sensor data.

**Key Takeaways:**
*   **Min-Heap:** $O(N \log k)$ time, $O(k)$ space.
*   **Divide and Conquer:** Same complexity, different approach.
*   **Brute Force:** $O(N \log N)$, acceptable when k is close to N.
*   **Real-World:** Used in databases, distributed systems, and big data processing.

Mastering this problem demonstrates proficiency in heaps, linked lists, and divide & conquer—essential skills for any software engineer.

## 21. Related Problems

*   **Merge Two Sorted Lists** (LeetCode 21)
*   **Kth Smallest Element in a Sorted Matrix** (LeetCode 378)
*   **Find K Pairs with Smallest Sums** (LeetCode 373)
*   **Smallest Range Covering Elements from K Lists** (LeetCode 632)
*   **Ugly Number II** (LeetCode 264)

Practice these to solidify your understanding of k-way merge patterns!

## 22. Advanced Variant: Smallest Range Covering Elements from K Lists

**Problem (LeetCode 632):** Given k sorted lists, find the smallest range [a, b] such that at least one element from each list is in the range.

**Example:**
```
Input: nums = [[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]
Output: [20,24]
```

**Algorithm (Min-Heap + Sliding Window):**
1.  Initialize heap with the first element of each list.
2.  Track the current max.
3.  Pop the min. Range = [min, max].
4.  Push the next element from the same list.
5.  Update max if needed.
6.  Repeat until one list is exhausted.

```python
import heapq

def smallestRange(nums):
    heap = []
    current_max = float('-inf')
    
    for i, lst in enumerate(nums):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
            current_max = max(current_max, lst[0])
    
    best_range = [float('-inf'), float('inf')]
    
    while heap:
        current_min, list_idx, elem_idx = heapq.heappop(heap)
        
        if current_max - current_min < best_range[1] - best_range[0]:
            best_range = [current_min, current_max]
        
        if elem_idx + 1 < len(nums[list_idx]):
            next_val = nums[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
            current_max = max(current_max, next_val)
        else:
            break  # One list exhausted
    
    return best_range
```

**Complexity:** $O(N \log k)$ where N is total elements.

## 23. Iterator Pattern: Merge K Sorted Iterators

**Problem:** Instead of lists, you have k iterators. Merge them lazily.

**Use Case:** Streaming data from k sources.

**Implementation:**
```python
class MergeKIterator:
    def __init__(self, iterators):
        self.heap = []
        self.iterators = iterators
        
        for i, it in enumerate(iterators):
            try:
                val = next(it)
                heapq.heappush(self.heap, (val, i))
            except StopIteration:
                pass
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.heap:
            raise StopIteration
        
        val, idx = heapq.heappop(self.heap)
        
        try:
            next_val = next(self.iterators[idx])
            heapq.heappush(self.heap, (next_val, idx))
        except StopIteration:
            pass
        
        return val

# Usage
it1 = iter([1, 4, 7])
it2 = iter([2, 5, 8])
it3 = iter([3, 6, 9])

merged = MergeKIterator([it1, it2, it3])
for val in merged:
    print(val)  # 1, 2, 3, 4, 5, 6, 7, 8, 9
```

## 24. MapReduce: K-Way Merge in Distributed Systems

**Scenario:** Merge sorted outputs from k mappers in a reducer.

**MapReduce Workflow:**
1.  **Map Phase:** Each mapper processes a partition and outputs sorted key-value pairs.
2.  **Shuffle Phase:** Keys are partitioned to reducers.
3.  **Reduce Phase:** Each reducer receives k sorted streams (one from each mapper). Merge using k-way merge.

**Implementation (Pseudo-code):**
```python
def reducer(key, iterators):
    # iterators: k sorted iterators of values for this key
    merged = MergeKIterator(iterators)
    for value in merged:
        emit(key, value)
```

## 25. Deep Dive: Merge with Custom Comparator

**Problem:** Merge k lists where elements are complex objects (not just numbers).

**Example:** Merge k lists of log entries, sorted by timestamp.

```python
import heapq
from dataclasses import dataclass, field

@dataclass(order=True)
class LogEntry:
    timestamp: float
    message: str = field(compare=False)
    source: str = field(compare=False)

def mergeLogStreams(streams):
    heap = []
    
    for i, stream in enumerate(streams):
        if stream:
            entry = stream[0]
            heapq.heappush(heap, (entry, i, 0))
    
    result = []
    
    while heap:
        entry, stream_idx, elem_idx = heapq.heappop(heap)
        result.append(entry)
        
        if elem_idx + 1 < len(streams[stream_idx]):
            next_entry = streams[stream_idx][elem_idx + 1]
            heapq.heappush(heap, (next_entry, stream_idx, elem_idx + 1))
    
    return result
```

## 26. Space Optimization: In-Place Merge

**Problem:** Can we merge k sorted arrays in-place (O(1) extra space)?

**Answer:** Not efficiently. The best we can do is:
*   **O(k) space** for the heap.
*   **O(N)** space for the output (unavoidable if we need to return a new structure).

**Special Case:** If merging into a pre-allocated array, we can avoid allocating a new array.

## 27. Testing and Debugging

**Test Cases:**
1.  **Empty Input:** `lists = []` → Output: `[]`
2.  **Single List:** `lists = [[1,2,3]]` → Output: `[1,2,3]`
3.  **All Empty Lists:** `lists = [[], [], []]` → Output: `[]`
4.  **Unequal Lengths:** `lists = [[1], [2,3,4], [5,6]]`
5.  **Duplicates:** `lists = [[1,1,1], [1,1], [1]]`
6.  **Negative Numbers:** `lists = [[-3,-2,-1], [-5,0,5]]`
7.  **Large K:** k = 10000 lists with 1 element each.

**Debugging Tips:**
*   Print the heap after each iteration.
*   Verify that the heap property is maintained.
*   Check for off-by-one errors in index handling.

## 28. Interview Strategy

**Step-by-Step Approach:**
1.  **Clarify:** Ask about constraints (k, N, duplicates, etc.).
2.  **Brute Force:** Mention the O(N log N) sorting approach.
3.  **Optimize:** Introduce the heap approach (O(N log k)).
4.  **Edge Cases:** Handle empty lists, null nodes.
5.  **Code:** Write clean, bug-free code.
6.  **Complexity:** Analyze time and space.
7.  **Follow-Up:** Be ready for variants (arrays, iterators, K-th element).

## 29. Code Template: Universal K-Way Merge

```python
import heapq

def k_way_merge(sources, key_func=lambda x: x, next_func=None):
    """
    Universal k-way merge.
    
    Args:
        sources: List of sorted sources (lists, iterators, etc.)
        key_func: Function to extract the key for comparison
        next_func: Function to get the next element from a source
    """
    heap = []
    
    for i, source in enumerate(sources):
        if source:
            elem = source[0] if isinstance(source, list) else next(source, None)
            if elem is not None:
                heapq.heappush(heap, (key_func(elem), i, elem))
    
    result = []
    indices = [0] * len(sources)
    
    while heap:
        _, src_idx, elem = heapq.heappop(heap)
        result.append(elem)
        
        indices[src_idx] += 1
        if indices[src_idx] < len(sources[src_idx]):
            next_elem = sources[src_idx][indices[src_idx]]
            heapq.heappush(heap, (key_func(next_elem), src_idx, next_elem))
    
    return result
```

## 30. Real-World Case Study: Elasticsearch

**Elasticsearch** uses k-way merge internally:
1.  **Index Segments:** Each segment is a sorted inverted index.
2.  **Search:** Query each segment, get sorted results.
3.  **Merge:** K-way merge results from all segments.
4.  **Segment Merging:** Background process merges small segments into larger ones.

**Optimization:**
*   **Lucene:** Uses a special priority queue optimized for merge operations.
*   **Skip Lists:** Skip irrelevant documents during merge.

## 31. Conclusion & Mastery Checklist

**Mastery Checklist:**
- [ ] Implement Merge K Sorted Lists with min-heap
- [ ] Implement with divide and conquer
- [ ] Handle linked lists vs arrays vs iterators
- [ ] Solve "Smallest Range Covering Elements from K Lists"
- [ ] Solve "Kth Smallest Element in a Sorted Matrix"
- [ ] Understand external merge sort
- [ ] Analyze time complexity (prove O(N log k))
- [ ] Handle edge cases (empty lists, single list, etc.)
- [ ] Implement with custom comparator
- [ ] Parallelize the merge

The k-way merge pattern is one of the most versatile algorithmic patterns. Once you master it, you'll see it everywhere—databases, distributed systems, log processing, and more. It's a testament to how a simple idea (use a heap to find the min efficiently) can solve a wide range of problems elegantly.



---

**Originally published at:** [arunbaby.com/dsa/0042-merge-k-sorted-lists](https://www.arunbaby.com/dsa/0042-merge-k-sorted-lists/)

*If you found this helpful, consider sharing it with others who might benefit.*

