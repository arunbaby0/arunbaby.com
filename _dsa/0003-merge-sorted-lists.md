---
title: "Merge Two Sorted Lists"
day: 3
collection: dsa
categories:
  - dsa
tags:
  - linked-lists
  - merge
topic: Linked Lists
difficulty: Easy
companies: [Google, Meta, Amazon, Microsoft]
leetcode_link: "https://leetcode.com/problems/merge-two-sorted-lists/"
time_complexity: "O(n+m)"
space_complexity: "O(1)"
related_ml_day: 3
related_speech_day: 3
---

**The pointer manipulation pattern that powers merge sort, data pipeline merging, and multi-source stream processing.**

## Problem

Merge two sorted linked lists into one sorted list.

**Example:**
```
List 1: 1 → 2 → 4
List 2: 1 → 3 → 4
Output: 1 → 1 → 2 → 3 → 4 → 4
```

**Constraints:**
- `0 <= list length <= 50`
- `-100 <= Node.val <= 100`
- Both lists sorted in non-decreasing order

---

## Intuition

When you have two sorted lists, you can build the merged result by repeatedly choosing the smaller of the two current heads. This is the foundation of **merge sort** and appears everywhere in systems that combine sorted streams.

**Key insight:** Since both lists are already sorted, we never need to look ahead, we always know the next element is one of the two current heads.

---

## Approach 1: Iterative Two Pointers (Optimal)

### Implementation

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    """
    Merge two sorted linked lists in-place
    
    Args:
        l1: Head of first sorted list
        l2: Head of second sorted list
    
    Returns:
        Head of merged sorted list
    
    Time: O(n + m) where n, m are list lengths
    Space: O(1) - only uses constant extra space
    """
    # Dummy node simplifies edge case handling
    dummy = ListNode(0)
    curr = dummy
    
    # While both lists have nodes
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    # Attach remaining nodes (at most one list has remaining nodes)
    curr.next = l1 if l1 else l2
    
    return dummy.next
```

### Detailed Walkthrough

```
Initial:
l1: 1 → 3 → 5 → None
l2: 2 → 4 → 6 → None
dummy: 0 → None
curr: dummy

Step 1: Compare 1 vs 2
  1 ≤ 2, so attach l1
  curr.next = l1 (1)
  l1 = l1.next (3)
  curr = curr.next (1)
  
  State:
  dummy: 0 → 1 → None
  curr: 1
  l1: 3 → 5 → None
  l2: 2 → 4 → 6 → None

Step 2: Compare 3 vs 2
  3 > 2, so attach l2
  curr.next = l2 (2)
  l2 = l2.next (4)
  curr = curr.next (2)
  
  State:
  dummy: 0 → 1 → 2 → None
  curr: 2
  l1: 3 → 5 → None
  l2: 4 → 6 → None

Step 3: Compare 3 vs 4
  3 ≤ 4, so attach l1
  curr.next = l1 (3)
  l1 = l1.next (5)
  curr = curr.next (3)
  
  State:
  dummy: 0 → 1 → 2 → 3 → None
  curr: 3
  l1: 5 → None
  l2: 4 → 6 → None

Step 4: Compare 5 vs 4
  5 > 4, so attach l2
  curr.next = l2 (4)
  l2 = l2.next (6)
  curr = curr.next (4)
  
  State:
  dummy: 0 → 1 → 2 → 3 → 4 → None
  curr: 4
  l1: 5 → None
  l2: 6 → None

Step 5: Compare 5 vs 6
  5 ≤ 6, so attach l1
  curr.next = l1 (5)
  l1 = l1.next (None)
  curr = curr.next (5)
  
  State:
  dummy: 0 → 1 → 2 → 3 → 4 → 5 → None
  curr: 5
  l1: None
  l2: 6 → None

Step 6: l1 is None
  Attach remaining l2
  curr.next = l2 (6)
  
Final:
  dummy: 0 → 1 → 2 → 3 → 4 → 5 → 6 → None
  Return: dummy.next = 1 → 2 → 3 → 4 → 5 → 6 → None
```

### Why This Works

1. **Sorted property preserved:** We always pick the smaller element, maintaining sorted order
2. **No nodes lost:** Every node from both lists appears exactly once in the result
3. **In-place:** We reuse existing nodes, only changing `next` pointers
4. **Single pass:** Visit each node exactly once

### Complexity Analysis

**Time Complexity: O(n + m)**
- Visit each node in both lists exactly once
- If list1 has n nodes and list2 has m nodes, total operations = n + m

**Space Complexity: O(1)**
- Only use constant extra space (dummy, curr, temporary pointers)
- Don't allocate new nodes
- Recursive stack not used

**Comparison to array merging:**
| Aspect | Linked List | Array |
|--------|-------------|-------|
| Time | O(n + m) | O(n + m) |
| Space | O(1) in-place | O(n + m) new array |
| Cache locality | Poor (pointer chasing) | Excellent (contiguous) |
| Random access | O(n) | O(1) |

---

## Approach 2: Recursive (Cleaner, More Stack)

### Implementation

```python
def mergeTwoListsRecursive(l1: ListNode, l2: ListNode) -> ListNode:
    """
    Recursive merge of two sorted lists
    
    Time: O(n + m)
    Space: O(n + m) for call stack
    """
    # Base cases
    if not l1:
        return l2
    if not l2:
        return l1
    
    # Recursive case: pick smaller head
    if l1.val <= l2.val:
        l1.next = mergeTwoListsRecursive(l1.next, l2)
        return l1
    else:
        l2.next = mergeTwoListsRecursive(l1, l2.next)
        return l2
```

### Recursion Tree

```
mergeTwoLists([1,3,5], [2,4,6])
│
├─ 1 ≤ 2 → return 1, recurse on ([3,5], [2,4,6])
│  │
│  ├─ 3 > 2 → return 2, recurse on ([3,5], [4,6])
│  │  │
│  │  ├─ 3 ≤ 4 → return 3, recurse on ([5], [4,6])
│  │  │  │
│  │  │  ├─ 5 > 4 → return 4, recurse on ([5], [6])
│  │  │  │  │
│  │  │  │  ├─ 5 ≤ 6 → return 5, recurse on ([], [6])
│  │  │  │  │  │
│  │  │  │  │  └─ l1 empty → return [6]
│  │  │  │  │
│  │  │  │  └─ 5 → 6
│  │  │  │
│  │  │  └─ 4 → 5 → 6
│  │  │
│  │  └─ 3 → 4 → 5 → 6
│  │
│  └─ 2 → 3 → 4 → 5 → 6
│
└─ 1 → 2 → 3 → 4 → 5 → 6
```

### Pros & Cons

**Pros:**
- ✅ Cleaner, more readable code
- ✅ Natural expression of divide-and-conquer
- ✅ Easier to prove correctness

**Cons:**
- ❌ O(n + m) stack space
- ❌ Stack overflow risk for very long lists (n + m > ~10,000)
- ❌ Function call overhead (~10-20% slower)

**When to use:**
- Interviews (cleaner to write/explain)
- Short to medium lists
- When stack space is acceptable

**When not to use:**
- Production code with unbounded input
- Memory-constrained environments
- Very long lists

---

## Understanding the Dummy Node Pattern

The **dummy node** is a powerful technique that eliminates special-case handling.

### Without Dummy Node

```python
def mergeWithoutDummy(l1, l2):
    # Special case: one or both empty
    if not l1:
        return l2
    if not l2:
        return l1
    
    # Need to determine head first
    if l1.val <= l2.val:
        head = l1
        l1 = l1.next
    else:
        head = l2
        l2 = l2.next
    
    curr = head
    
    # Now standard merge
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    curr.next = l1 if l1 else l2
    
    return head
```

**Problems:**
- Extra edge case handling
- Head determination duplicates merge logic
- More error-prone

### With Dummy Node

```python
def mergeWithDummy(l1, l2):
    dummy = ListNode(0)  # Placeholder
    curr = dummy
    
    # Uniform handling - no special cases!
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    curr.next = l1 if l1 else l2
    
    return dummy.next  # Skip dummy
```

**Benefits:**
- ✅ No special case for head
- ✅ Uniform loop logic
- ✅ Cleaner, less error-prone
- ✅ Common pattern in linked list problems

**Cost:** One extra node allocation (negligible)

**This pattern appears in:**
- Remove duplicates from sorted list
- Partition list
- Add two numbers (linked lists)
- Reverse linked list II

---

## Pointer Manipulation Deep Dive

### Understanding Pointer Movement

In linked list problems, pointer manipulation is key. Let's visualize what happens at the memory level.

```python
# Initial state (memory addresses shown)
l1 @ 0x1000: [val=1, next=0x1001]
l1 @ 0x1001: [val=3, next=0x1002]
l1 @ 0x1002: [val=5, next=None]

l2 @ 0x2000: [val=2, next=0x2001]
l2 @ 0x2001: [val=4, next=0x2002]
l2 @ 0x2002: [val=6, next=None]

# During merge
dummy @ 0x3000: [val=0, next=None]
curr = dummy  # curr points to 0x3000

# Step 1: 1 ≤ 2
curr.next = l1  # 0x3000.next = 0x1000
  Now: dummy @ 0x3000: [val=0, next=0x1000]

l1 = l1.next    # l1 = 0x1001
curr = curr.next  # curr = 0x1000

# Step 2: 3 > 2
curr.next = l2  # 0x1000.next = 0x2000
  Now: 0x1000 (node with val=1) points to 0x2000 (node with val=2)

l2 = l2.next    # l2 = 0x2001
curr = curr.next  # curr = 0x2000

# This continues, rewiring pointers without moving data
```

**Key insight:** We're **rewiring pointers**, not copying data. Each node stays at its original memory location; only the `next` pointers change.

### Memory Efficiency

```python
# Creating new nodes (NOT what we do)
def mergeByCopying(l1, l2):
    result = []
    while l1 and l2:
        if l1.val <= l2.val:
            result.append(ListNode(l1.val))  # New allocation!
            l1 = l1.next
        else:
            result.append(ListNode(l2.val))  # New allocation!
            l2 = l2.next
    # This uses O(n + m) extra space

# Rewiring pointers (what we actually do)
def mergeByRewiring(l1, l2):
    dummy = ListNode(0)  # Only 1 extra node
    curr = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1  # Pointer assignment, no allocation
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    # This uses O(1) extra space
```

**Benefit:** In-place merging is **memory-efficient** and **fast** (no allocation overhead).

---

## Advanced Variations

### Variation 1: Merge in Descending Order

```python
def mergeTwoListsDescending(l1: ListNode, l2: ListNode) -> ListNode:
    """
    Merge two ascending lists into a descending list
    
    Approach: Merge normally, then reverse
    """
    # Merge ascending
    merged = mergeTwoLists(l1, l2)
    
    # Reverse
    return reverseList(merged)


def reverseList(head: ListNode) -> ListNode:
    prev = None
    curr = head
    
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    
    return prev
```

**Alternative:** Build descending directly

```python
def mergeTwoListsDescendingDirect(l1: ListNode, l2: ListNode) -> ListNode:
    """
    Build descending list directly using head insertion
    """
    result = None  # No dummy needed for head insertion
    
    # Merge into a list, inserting at head each time
    while l1 and l2:
        if l1.val <= l2.val:
            next_node = l1.next
            l1.next = result
            result = l1
            l1 = next_node
        else:
            next_node = l2.next
            l2.next = result
            result = l2
            l2 = next_node
    
    # Attach remaining
    remaining = l1 if l1 else l2
    while remaining:
        next_node = remaining.next
        remaining.next = result
        result = remaining
        remaining = next_node
    
    return result
```

### Variation 2: Merge with Deduplication

```python
def mergeTwoListsNoDuplicates(l1: ListNode, l2: ListNode) -> ListNode:
    """
    Merge and remove duplicates
    
    Example:
      [1, 2, 4] + [1, 3, 4] → [1, 2, 3, 4]  (not [1,1,2,3,4,4])
    """
    dummy = ListNode(0)
    curr = dummy
    prev_val = None
    
    while l1 and l2:
        # Pick smaller value
        if l1.val <= l2.val:
            val = l1.val
            l1 = l1.next
        else:
            val = l2.val
            l2 = l2.next
        
        # Only add if different from previous
        if val != prev_val:
            curr.next = ListNode(val)
            curr = curr.next
            prev_val = val
    
    # Process remaining (still checking for duplicates)
    remaining = l1 if l1 else l2
    while remaining:
        if remaining.val != prev_val:
            curr.next = ListNode(remaining.val)
            curr = curr.next
            prev_val = remaining.val
        remaining = remaining.next
    
    return dummy.next
```

### Variation 3: Merge with Custom Comparator

```python
def mergeTwoListsCustom(l1: ListNode, l2: ListNode, compare_fn):
    """
    Merge using custom comparison function
    
    Example comparators:
      - lambda a, b: a.val <= b.val  (standard)
      - lambda a, b: a.val >= b.val  (descending)
      - lambda a, b: abs(a.val) <= abs(b.val)  (by absolute value)
    """
    dummy = ListNode(0)
    curr = dummy
    
    while l1 and l2:
        if compare_fn(l1, l2):
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    curr.next = l1 if l1 else l2
    
    return dummy.next


# Usage
merged = mergeTwoListsCustom(l1, l2, lambda a, b: a.val <= b.val)
merged_abs = mergeTwoListsCustom(l1, l2, lambda a, b: abs(a.val) <= abs(b.val))
```

---

## Why Dummy Node Helps

**Without dummy:**
```python
def merge(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    
    # Need to determine head
    if l1.val <= l2.val:
        head = l1
        l1 = l1.next
    else:
        head = l2
        l2 = l2.next
    
    curr = head
    # ... rest of merge
```

**With dummy:**
```python
def merge(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    # ... merge logic
    return dummy.next  # Clean!
```

**Dummy eliminates special-case handling for the first node.**

---

## Variations

### Merge K Sorted Lists
```python
def mergeKLists(lists: List[ListNode]) -> ListNode:
    if not lists:
        return None
    
    # Divide and conquer: merge pairs recursively
    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i+1] if i+1 < len(lists) else None
            merged.append(mergeTwoLists(l1, l2))
        lists = merged
    
    return lists[0]
```

**Complexity:** O(N log k) where N = total nodes, k = number of lists

### Merge with Priority Queue
```python
import heapq

def mergeKListsPQ(lists: List[ListNode]) -> ListNode:
    heap = []  # (value, unique_id, node)
    
    # Add first node from each list
    for i, node in enumerate(lists):
        if node:
            # Use a unique counter to avoid comparing ListNode on ties
            heapq.heappush(heap, (node.val, i, node))
    
    dummy = ListNode(0)
    curr = dummy
    
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return dummy.next
```

**Cleaner for k lists, O(N log k) time.**

---

## Edge Cases

```python
# Both empty
l1 = None, l2 = None → None

# One empty
l1 = None, l2 = [1,2] → [1,2]

# Different lengths
l1 = [1], l2 = [2,3,4,5] → [1,2,3,4,5]

# All from one list first
l1 = [1,2,3], l2 = [4,5,6] → [1,2,3,4,5,6]

# Interleaved
l1 = [1,3,5], l2 = [2,4,6] → [1,2,3,4,5,6]
```

---

## Connection to ML Systems & Data Pipelines

The merge pattern is fundamental to production ML systems. Let's see real-world applications.

### 1. Merging Data from Distributed Shards

When data is partitioned across shards, you often need to merge sorted streams.

```python
from dataclasses import dataclass
from typing import List, Iterator
import heapq

@dataclass
class TrainingExample:
    timestamp: int
    user_id: str
    features: dict
    label: int

class DistributedDataMerger:
    """
    Merge training data from multiple sharded databases
    
    Use case: Distributed training data collection
    - Each shard sorted by timestamp
    - Need globally sorted stream for training
    """
    
    def merge_two_shards(
        self, 
        shard1: Iterator[TrainingExample], 
        shard2: Iterator[TrainingExample]
    ) -> Iterator[TrainingExample]:
        """
        Merge two sorted iterators of training examples
        
        Pattern: Exact same as merge two sorted lists!
        """
        try:
            ex1 = next(shard1)
        except StopIteration:
            ex1 = None
        
        try:
            ex2 = next(shard2)
        except StopIteration:
            ex2 = None
        
        while ex1 and ex2:
            if ex1.timestamp <= ex2.timestamp:
                yield ex1
                try:
                    ex1 = next(shard1)
                except StopIteration:
                    ex1 = None
            else:
                yield ex2
                try:
                    ex2 = next(shard2)
                except StopIteration:
                    ex2 = None
        
        # Yield remaining
        remaining = ex1 if ex1 else ex2
        if remaining:
            yield remaining
            iterator = shard1 if ex1 else shard2
            yield from iterator
    
    def merge_k_shards(self, shards: List[Iterator[TrainingExample]]) -> Iterator[TrainingExample]:
        """
        Merge K shards using priority queue
        
        Complexity: O(N log K) where N = total examples, K = num shards
        """
        # Min-heap: (timestamp, shard_id, example)
        heap = []
        
        # Initialize with first example from each shard
        for shard_id, shard in enumerate(shards):
            try:
                example = next(shard)
                heapq.heappush(heap, (example.timestamp, shard_id, example))
            except StopIteration:
                pass
        
        # Merge
        while heap:
            timestamp, shard_id, example = heapq.heappop(heap)
            yield example
            
            # Get next from same shard
            try:
                next_example = next(shards[shard_id])
                heapq.heappush(heap, (next_example.timestamp, shard_id, next_example))
            except StopIteration:
                pass

# Usage
merger = DistributedDataMerger()
shard1 = get_shard_data(shard_id=0)  # Sorted by timestamp
shard2 = get_shard_data(shard_id=1)  # Sorted by timestamp
merged = merger.merge_two_shards(shard1, shard2)

for example in merged:
    train_model(example)
```

### 2. Feature Store Merging

Combining features from multiple feature stores, sorted by user_id or timestamp.

```python
class FeatureStoreMerger:
    """
    Merge features from multiple feature stores
    
    Real-world scenario:
    - User features from User Service (sorted by user_id)
    - Item features from Item Service (sorted by item_id)
    - Interaction features from Events Service (sorted by timestamp)
    
    Need to join/merge for training
    """
    
    def merge_user_features(self, store_a_features, store_b_features):
        """
        Merge two feature stores, both sorted by user_id
        
        Example:
          Store A: user demographics
          Store B: user behavioral features
          
        Output: Combined feature vector per user
        """
    merged = []
    i, j = 0, 0
    
        while i < len(store_a_features) and j < len(store_b_features):
            feat_a = store_a_features[i]
            feat_b = store_b_features[j]
            
            if feat_a.user_id == feat_b.user_id:
                # Same user - combine features
                merged.append({
                    'user_id': feat_a.user_id,
                    **feat_a.features,
                    **feat_b.features
                })
                i += 1
                j += 1
            elif feat_a.user_id < feat_b.user_id:
                # User only in store A
                merged.append({
                    'user_id': feat_a.user_id,
                    **feat_a.features
                })
            i += 1
        else:
                # User only in store B
                merged.append({
                    'user_id': feat_b.user_id,
                    **feat_b.features
                })
                j += 1
        
        # Append remaining (preserve unified schema)
        while i < len(store_a_features):
            feat_a = store_a_features[i]
            merged.append({
                'user_id': feat_a.user_id,
                **feat_a.features
            })
            i += 1
        
        while j < len(store_b_features):
            feat_b = store_b_features[j]
            merged.append({
                'user_id': feat_b.user_id,
                **feat_b.features
            })
            j += 1
    
    return merged
```

### 3. Model Ensemble Prediction Merging

Combining predictions from multiple models, sorted by confidence or score.

```python
from typing import List, Tuple

@dataclass
class Prediction:
    sample_id: str
    class_id: int
    confidence: float
    model_name: str

class EnsemblePredictionMerger:
    """
    Merge and combine predictions from ensemble of models
    """
    
    def merge_top_k_predictions(
        self, 
        model1_preds: List[Prediction],
        model2_preds: List[Prediction],
        k: int = 10
    ) -> List[Prediction]:
        """
        Merge predictions from two models, taking top K by confidence
        
        Use case: Ensemble serving
        - Model1 specializes in common cases
        - Model2 specializes in edge cases
        - Merge their top predictions
        
        Assumes: Both lists sorted by confidence (descending)
        """
    merged = []
    i, j = 0, 0
    
        while len(merged) < k and (i < len(model1_preds) or j < len(model2_preds)):
            if i >= len(model1_preds):
                merged.append(model2_preds[j])
                j += 1
            elif j >= len(model2_preds):
                merged.append(model1_preds[i])
                i += 1
            else:
                # Both have predictions - pick higher confidence
        if model1_preds[i].confidence >= model2_preds[j].confidence:
            merged.append(model1_preds[i])
            i += 1
        else:
            merged.append(model2_preds[j])
            j += 1
    
        return merged[:k]
    
    def merge_with_vote(
        self,
        model1_preds: List[Prediction],
        model2_preds: List[Prediction]
    ) -> List[Prediction]:
        """
        Merge by voting: if both models agree, boost confidence
        """
        merged = []
        i, j = 0, 0
        
        while i < len(model1_preds) and j < len(model2_preds):
            pred1 = model1_preds[i]
            pred2 = model2_preds[j]
            
            if pred1.sample_id == pred2.sample_id:
                if pred1.class_id == pred2.class_id:
                    # Agreement - boost confidence
                    merged.append(Prediction(
                        sample_id=pred1.sample_id,
                        class_id=pred1.class_id,
                        confidence=(pred1.confidence + pred2.confidence) / 2 * 1.2,  # Boost
                        model_name="ensemble"
                    ))
                else:
                    # Disagreement - use higher confidence
                    merged.append(pred1 if pred1.confidence >= pred2.confidence else pred2)
                i += 1
                j += 1
            elif pred1.sample_id < pred2.sample_id:
                merged.append(pred1)
                i += 1
            else:
                merged.append(pred2)
                j += 1
        
        return merged
```

### 4. Streaming Data Pipeline

Merge real-time event streams sorted by timestamp.

```python
import time
from queue import Queue
from threading import Thread

class StreamMerger:
    """
    Merge multiple real-time streams (e.g., Kafka topics)
    
    Real-world use case:
    - User click stream from web
    - User action stream from mobile app
    - Merge into unified event stream for ML feature extraction
    """
    
    def __init__(self):
        self.output_queue = Queue()
    
    def merge_streams_realtime(self, stream1: Queue, stream2: Queue):
        """
        Merge two real-time streams
        
        Complexity: Each event processed once → O(total events)
        """
        event1 = None
        event2 = None
        
        while True:
            # Get next event from each stream if needed
            if event1 is None and not stream1.empty():
                event1 = stream1.get()
            
            if event2 is None and not stream2.empty():
                event2 = stream2.get()
            
            # Merge logic
            if event1 and event2:
                if event1['timestamp'] <= event2['timestamp']:
                    self.output_queue.put(event1)
                    event1 = None
                else:
                    self.output_queue.put(event2)
                    event2 = None
            elif event1:
                self.output_queue.put(event1)
                event1 = None
            elif event2:
                self.output_queue.put(event2)
                event2 = None
            else:
                # Both streams empty - wait
                time.sleep(0.01)
```

### 5. External Merge Sort for Large Datasets

When dataset doesn't fit in memory, use external merge sort.

```python
import tempfile
import pickle

class ExternalMergeSorter:
    """
    Sort huge datasets that don't fit in RAM
    
    Use case: Sort 100GB of training data on machine with 16GB RAM
    
    Algorithm:
    1. Split data into chunks that fit in RAM
    2. Sort each chunk, write to disk
    3. Merge sorted chunks using merge algorithm
    """
    
    def __init__(self, chunk_size=10000):
        self.chunk_size = chunk_size
    
    def external_sort(self, input_file: str, output_file: str):
        """
        Sort large file using external merge sort
        """
        # Phase 1: Create sorted chunks
        chunk_files = self._create_sorted_chunks(input_file)
        
        # Phase 2: Merge chunks
        self._merge_chunks(chunk_files, output_file)
    
    def _create_sorted_chunks(self, input_file: str) -> List[str]:
        """Read input in chunks, sort each, write to temp files"""
        chunk_files = []
        
        with open(input_file, 'r') as f:
            while True:
                # Read chunk
                chunk = []
                for _ in range(self.chunk_size):
                    line = f.readline()
                    if not line:
                        break
                    chunk.append(line.strip())
                
                if not chunk:
                    break
                
                # Sort chunk
                chunk.sort()
                
                # Write to temp file
                temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                for line in chunk:
                    temp_file.write(line + '\n')
                temp_file.close()
                chunk_files.append(temp_file.name)
        
        return chunk_files
    
    def _merge_chunks(self, chunk_files: List[str], output_file: str):
        """
        Merge sorted chunks using K-way merge
        
        This is merge K sorted lists!
        """
        # Open all chunk files
        file_handles = [open(f, 'r') for f in chunk_files]
        
        # Min heap: (value, file_index)
        heap = []
        
        # Initialize with first line from each file
        for i, fh in enumerate(file_handles):
            line = fh.readline().strip()
            if line:
                heapq.heappush(heap, (line, i))
        
        # Merge
        with open(output_file, 'w') as out:
            while heap:
                value, file_idx = heapq.heappop(heap)
                out.write(value + '\n')
                
                # Get next line from same file
                next_line = file_handles[file_idx].readline().strip()
                if next_line:
                    heapq.heappush(heap, (next_line, file_idx))
        
        # Cleanup
        for fh in file_handles:
            fh.close()
```

**Key Insight:** The merge pattern scales from simple linked lists to **distributed data systems processing terabytes**. The algorithm stays the same, only the data structures change.

---

## Production Engineering Considerations

### Thread Safety

If merging in a multi-threaded environment, consider thread safety.

```python
from threading import Lock

class ThreadSafeMerger:
    """
    Thread-safe merging for concurrent access
    """
    def __init__(self):
        self.lock = Lock()
        self.result = None
    
    def merge(self, l1, l2):
        with self.lock:
            # Only one thread merges at a time
            self.result = mergeTwoLists(l1, l2)
        return self.result
```

### Memory Management in Production

```python
class ProductionMerger:
    """
    Production-grade merger with error handling and monitoring
    """
    
    def merge_with_monitoring(self, l1, l2, max_size=10000):
        """
        Merge with size limits and monitoring
        """
        # Validate inputs
        if not self._validate_sorted(l1):
            raise ValueError("List 1 not sorted")
        if not self._validate_sorted(l2):
            raise ValueError("List 2 not sorted")
        
        # Track metrics
        start_time = time.time()
        nodes_processed = 0
        
        dummy = ListNode(0)
        curr = dummy
        
        while l1 and l2:
            nodes_processed += 1
            
            # Safety check: prevent infinite lists
            if nodes_processed > max_size:
                raise RuntimeError(f"Exceeded max size {max_size}")
            
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        
        curr.next = l1 if l1 else l2
        
        # Log metrics
        duration = time.time() - start_time
        logger.info(f"Merged {nodes_processed} nodes in {duration:.3f}s")
        
        return dummy.next
    
    def _validate_sorted(self, head):
        """Validate list is sorted"""
        if not head:
            return True
        
        while head.next:
            if head.val > head.next.val:
                return False
            head = head.next
        
        return True
```

---

## Comprehensive Testing

### Test Utilities

```python
def list_to_linkedlist(arr):
    """
    Convert Python list to linked list
    
    Helper for testing
    """
    if not arr:
        return None
    head = ListNode(arr[0])
    curr = head
    for val in arr[1:]:
        curr.next = ListNode(val)
        curr = curr.next
    return head

def linkedlist_to_list(head):
    """
    Convert linked list to Python list
    
    For assertions
    """
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result

def print_list(head):
    """Print linked list"""
    values = linkedlist_to_list(head)
    print(" → ".join(map(str, values)))
```

### Test Suite

```python
import unittest

class TestMergeTwoLists(unittest.TestCase):
    
    def test_basic_merge(self):
        """Standard case: interleaved values"""
        l1 = list_to_linkedlist([1, 2, 4])
        l2 = list_to_linkedlist([1, 3, 4])
    merged = mergeTwoLists(l1, l2)
        self.assertEqual(linkedlist_to_list(merged), [1, 1, 2, 3, 4, 4])
    
    def test_both_empty(self):
        """Edge case: both lists empty"""
        self.assertIsNone(mergeTwoLists(None, None))
    
    def test_one_empty(self):
        """Edge case: one list empty"""
        l1 = list_to_linkedlist([1, 2, 3])
        self.assertEqual(linkedlist_to_list(mergeTwoLists(l1, None)), [1, 2, 3])
        self.assertEqual(linkedlist_to_list(mergeTwoLists(None, l1)), [1, 2, 3])
    
    def test_different_lengths(self):
        """Lists of very different lengths"""
        l1 = list_to_linkedlist([1])
        l2 = list_to_linkedlist([2, 3, 4, 5, 6, 7, 8])
        merged = mergeTwoLists(l1, l2)
        self.assertEqual(linkedlist_to_list(merged), [1, 2, 3, 4, 5, 6, 7, 8])
    
    def test_no_overlap(self):
        """No interleaving - all from one list first"""
        l1 = list_to_linkedlist([1, 2, 3])
        l2 = list_to_linkedlist([4, 5, 6])
        merged = mergeTwoLists(l1, l2)
        self.assertEqual(linkedlist_to_list(merged), [1, 2, 3, 4, 5, 6])
        
        # Reverse
        l1 = list_to_linkedlist([4, 5, 6])
        l2 = list_to_linkedlist([1, 2, 3])
        merged = mergeTwoLists(l1, l2)
        self.assertEqual(linkedlist_to_list(merged), [1, 2, 3, 4, 5, 6])
    
    def test_all_duplicates(self):
        """All same values"""
        l1 = list_to_linkedlist([1, 1, 1])
        l2 = list_to_linkedlist([1, 1, 1])
        merged = mergeTwoLists(l1, l2)
        self.assertEqual(linkedlist_to_list(merged), [1, 1, 1, 1, 1, 1])
    
    def test_single_nodes(self):
        """Single node lists"""
        l1 = list_to_linkedlist([1])
        l2 = list_to_linkedlist([2])
        merged = mergeTwoLists(l1, l2)
        self.assertEqual(linkedlist_to_list(merged), [1, 2])
    
    def test_negative_values(self):
        """Negative and mixed values"""
        l1 = list_to_linkedlist([-10, -5, 0])
        l2 = list_to_linkedlist([-7, -3, 5])
        merged = mergeTwoLists(l1, l2)
        self.assertEqual(linkedlist_to_list(merged), [-10, -7, -5, -3, 0, 5])
    
    def test_large_lists(self):
        """Performance test with large lists"""
        l1 = list_to_linkedlist(list(range(0, 10000, 2)))  # Even numbers
        l2 = list_to_linkedlist(list(range(1, 10000, 2)))  # Odd numbers
        merged = mergeTwoLists(l1, l2)
        result = linkedlist_to_list(merged)
        self.assertEqual(len(result), 10000)
        self.assertEqual(result, list(range(10000)))
    
    def test_recursive_version(self):
        """Test recursive implementation"""
        l1 = list_to_linkedlist([1, 3, 5])
        l2 = list_to_linkedlist([2, 4, 6])
        merged = mergeTwoListsRecursive(l1, l2)
        self.assertEqual(linkedlist_to_list(merged), [1, 2, 3, 4, 5, 6])

if __name__ == '__main__':
    unittest.main()
```

---

## Common Mistakes & How to Avoid

### Mistake 1: Forgetting to Advance Pointers

```python
# ❌ WRONG - infinite loop!
while l1 and l2:
    if l1.val <= l2.val:
        curr.next = l1
        # FORGOT: l1 = l1.next
    else:
        curr.next = l2
        l2 = l2.next
    curr = curr.next
```

**Fix:** Always advance the pointer after attaching:
```python
if l1.val <= l2.val:
    curr.next = l1
    l1 = l1.next  # ✅ Don't forget this
```

### Mistake 2: Not Handling Remaining Elements

```python
# ❌ WRONG - loses remaining elements
while l1 and l2:
    # merge logic
return dummy.next  # Missing remaining nodes!
```

**Fix:** Attach remaining nodes:
```python
while l1 and l2:
    # merge logic

# ✅ Attach remaining (at most one is non-None)
curr.next = l1 if l1 else l2
```

### Mistake 3: Not Returning `dummy.next`

```python
# ❌ WRONG - returns dummy node itself
return dummy  # This includes the dummy with val=0
```

**Fix:** Skip the dummy:
```python
return dummy.next  # ✅ Skip dummy, return actual head
```

### Mistake 4: Modifying Input Lists Unintentionally

```python
# If you need to preserve original lists
def merge_preserve_originals(l1, l2):
    # Create copies first
    l1_copy = copy_list(l1)
    l2_copy = copy_list(l2)
    return mergeTwoLists(l1_copy, l2_copy)
```

Our standard implementation **does modify** the original lists by rewiring pointers. This is usually fine, but be aware.

### Mistake 5: Wrong Comparison Operator

```python
# ❌ Using < instead of <=
if l1.val < l2.val:  # Wrong for stability
```

**Fix:** Use `<=` to maintain **stable merge** (preserves relative order of equal elements):
```python
if l1.val <= l2.val:  # ✅ Stable merge
```

---

## Interview Tips

### What Interviewers Look For

1. **Edge Case Handling**
   - Empty lists
   - Single elements
   - Very different lengths
   
2. **Pointer Management**
   - Clean, bug-free pointer manipulation
   - No off-by-one errors
   
3. **Code Clarity**
   - Use of dummy node
   - Clear variable names
   
4. **Complexity Analysis**
   - Correctly identify O(n + m) time, O(1) space

5. **Follow-up Questions**
   - Can you merge K lists?
   - What if lists aren't sorted?
   - How to merge in descending order?

### How to Explain Your Solution

**Template:**

1. **Approach:** "I'll use two pointers to traverse both lists, always picking the smaller element."

2. **Dummy Node:** "I'll use a dummy node to avoid special-casing the head."

3. **Walkthrough:** Walk through a small example (3-4 nodes each)

4. **Edge Cases:** "I handle empty lists by attaching the remaining list at the end."

5. **Complexity:** "Time O(n+m) since we visit each node once, space O(1) since we only use pointers."

### Extension Questions You Might Face

**Q: How would you merge K sorted lists?**
```python
def mergeKLists(lists):
    """
    Approach 1: Divide and conquer - O(N log k)
    Approach 2: Priority queue - O(N log k)
    
    I'd use divide and conquer to repeatedly merge pairs.
    """
    if not lists:
        return None
    
    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i+1] if i+1 < len(lists) else None
            merged.append(mergeTwoLists(l1, l2))
        lists = merged
    
    return lists[0]
```

**Q: What if lists aren't sorted?**
```python
def mergeUnsortedLists(l1, l2):
    """
    Can't use two-pointer merge. Instead:
    1. Convert to arrays
    2. Concatenate
    3. Sort: O((n+m) log(n+m))
    4. Convert back to linked list
    """
    arr1 = linkedlist_to_list(l1)
    arr2 = linkedlist_to_list(l2)
    merged_arr = sorted(arr1 + arr2)
    return list_to_linkedlist(merged_arr)
```

**Q: Can you do this without extra space (no dummy)?**
```python
def mergeWithoutDummy(l1, l2):
    """Yes, but requires more edge case handling"""
    if not l1:
        return l2
    if not l2:
        return l1
    
    # Determine head
    if l1.val <= l2.val:
        head = l1
        l1 = l1.next
    else:
        head = l2
        l2 = l2.next
    
    curr = head
    
    # Standard merge
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    curr.next = l1 if l1 else l2
    
    return head
```

---

## Key Takeaways

✅ **Two pointers** efficiently merge sorted sequences in O(n + m) time  
✅ **Dummy node** eliminates special-case handling and simplifies code  
✅ **In-place merge** achieves O(1) space by rewiring pointers  
✅ **Pattern extends** to merging K lists, data streams, and distributed systems  
✅ **Foundation of merge sort** and external sorting algorithms
✅ **Critical for ML pipelines** merging sorted shards, features, predictions

---

## Related Problems

Practice these to master the pattern:
- **[Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)** - Direct extension
- **[Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)** - Array version
- **[Sort List](https://leetcode.com/problems/sort-list/)** - Uses merge as subroutine
- **[Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)** - Similar two-pointer pattern

---

**Originally published at:** [arunbaby.com/dsa/0003-merge-sorted-lists](https://www.arunbaby.com/dsa/0003-merge-sorted-lists/)

*If you found this helpful, consider sharing it with others who might benefit.*
