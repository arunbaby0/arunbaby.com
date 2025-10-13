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

## Approach: Two Pointers

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(0)  # Dummy node simplifies edge cases
    curr = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    # Attach remaining nodes
    curr.next = l1 if l1 else l2
    
    return dummy.next
```

**Walkthrough:**
```
l1: 1 → 3 → 5
l2: 2 → 4 → 6

Step 1: 1 ≤ 2 → take 1 from l1
Step 2: 3 > 2 → take 2 from l2
Step 3: 3 ≤ 4 → take 3 from l1
Step 4: 5 > 4 → take 4 from l2
Step 5: 5 ≤ 6 → take 5 from l1
Step 6: l1 empty, attach l2 (6)

Result: 1 → 2 → 3 → 4 → 5 → 6
```

**Complexity:** O(n+m) time, O(1) space (in-place)

---

## Recursive Solution

```python
def mergeTwoListsRecursive(l1: ListNode, l2: ListNode) -> ListNode:
    if not l1:
        return l2
    if not l2:
        return l1
    
    if l1.val <= l2.val:
        l1.next = mergeTwoListsRecursive(l1.next, l2)
        return l1
    else:
        l2.next = mergeTwoListsRecursive(l1, l2.next)
        return l2
```

**Complexity:** O(n+m) time, O(n+m) space (call stack)

**Cleaner but uses stack space not ideal for very long lists.**

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
    heap = []
    
    # Add first node from each list
    for i, node in enumerate(lists):
        if node:
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

## Connection to ML Pipelines

### Merge Data Streams
```python
# Merge sorted data from multiple shards
def merge_data_shards(shard1, shard2):
    # Both shards sorted by timestamp
    merged = []
    i, j = 0, 0
    
    while i < len(shard1) and j < len(shard2):
        if shard1[i].timestamp <= shard2[j].timestamp:
            merged.append(shard1[i])
            i += 1
        else:
            merged.append(shard2[j])
            j += 1
    
    merged.extend(shard1[i:])
    merged.extend(shard2[j:])
    
    return merged
```

### Pipeline Stage Merging
```python
# Combine outputs from parallel feature extractors
class PipelineMerger:
    def merge(self, features_a, features_b):
        # Both sorted by sample_id
        merged_features = []
        
        while features_a and features_b:
            if features_a[0].id <= features_b[0].id:
                merged_features.append(features_a.pop(0))
            else:
                merged_features.append(features_b.pop(0))
        
        merged_features.extend(features_a or features_b)
        return merged_features
```

### Model Ensemble Merging
```python
# Merge predictions from two models (sorted by confidence)
def merge_predictions(model1_preds, model2_preds):
    merged = []
    i, j = 0, 0
    
    while i < len(model1_preds) and j < len(model2_preds):
        if model1_preds[i].confidence >= model2_preds[j].confidence:
            merged.append(model1_preds[i])
            i += 1
        else:
            merged.append(model2_preds[j])
            j += 1
    
    return merged + model1_preds[i:] + model2_preds[j:]
```

---

## Testing

```python
def list_to_linkedlist(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    curr = head
    for val in arr[1:]:
        curr.next = ListNode(val)
        curr = curr.next
    return head

def linkedlist_to_list(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result

def test_merge():
    l1 = list_to_linkedlist([1,2,4])
    l2 = list_to_linkedlist([1,3,4])
    merged = mergeTwoLists(l1, l2)
    assert linkedlist_to_list(merged) == [1,1,2,3,4,4]
    
    # Empty lists
    assert mergeTwoLists(None, None) is None
    assert linkedlist_to_list(mergeTwoLists(None, list_to_linkedlist([1]))) == [1]
```

---

## Key Takeaways

✅ **Two pointers** efficiently merge sorted sequences  
✅ **Dummy node** simplifies edge case handling  
✅ **Pattern extends** to merging K lists, data streams, pipelines  
✅ **In-place merge** achieves O(1) space  
✅ **Foundation of merge sort** and external sorting algorithms

---

## Related Problems
- [Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
- [Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)
- [Sort List](https://leetcode.com/problems/sort-list/)


---

**Originally published at:** [arunbaby.com/dsa/0003-merge-sorted-lists](https://www.arunbaby.com/dsa/0003-merge-sorted-lists/)

*If you found this helpful, consider sharing it with others who might benefit.*
