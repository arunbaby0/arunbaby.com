---
title: "Reverse Linked List"
day: 10
related_ml_day: 10
related_speech_day: 10
related_agents_day: 10
collection: dsa
categories:
 - dsa
tags:
 - linked-list
 - pointers
 - recursion
 - iteration
topic: Linked Lists
difficulty: Easy
companies: [Google, Meta, Amazon, Microsoft, Apple, Adobe, Bloomberg, Uber]
leetcode_link: "https://leetcode.com/problems/reverse-linked-list/"
time_complexity: "O(n)"
space_complexity: "O(1) iterative, O(n) recursive"
---

**Master linked list manipulation through reversal - a fundamental pattern for understanding pointer logic and in-place algorithms.**

## Problem

Given the `head` of a singly linked list, reverse the list, and return the reversed list.

**Example 1:**
``
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Visual:
1 → 2 → 3 → 4 → 5 → null
 ↓
5 → 4 → 3 → 2 → 1 → null
``

**Example 2:**
``
Input: head = [1,2]
Output: [2,1]
``

**Example 3:**
``
Input: head = []
Output: []
``

**Constraints:**
- The number of nodes in the list is in the range `[0, 5000]`
- `-5000 <= Node.val <= 5000`

**Follow-up:** Can you reverse the linked list both iteratively and recursively?

---

## Understanding Linked Lists

### Node Structure

``python
class ListNode:
 """Singly-linked list node"""
 
 def __init__(self, val=0, next=None):
 self.val = val
 self.next = next
 
 def __repr__(self):
 """String representation for debugging"""
 return f"ListNode({self.val})"

# Create a linked list: 1 → 2 → 3
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
``

### Helper Functions

``python
def create_linked_list(values):
 """Create linked list from list of values"""
 if not values:
 return None
 
 head = ListNode(values[0])
 current = head
 
 for val in values[1:]:
 current.next = ListNode(val)
 current = current.next
 
 return head

def list_to_array(head):
 """Convert linked list to array for easy visualization"""
 result = []
 current = head
 
 while current:
 result.append(current.val)
 current = current.next
 
 return result

def print_linked_list(head):
 """Pretty print linked list"""
 values = list_to_array(head)
 print(" → ".join(map(str, values)) + " → null")

# Usage
head = create_linked_list([1, 2, 3, 4, 5])
print_linked_list(head) # 1 → 2 → 3 → 4 → 5 → null
``

---

## Approach 1: Iterative (Three Pointers)

**The standard and most intuitive approach**

### Visualization

``
Initial: 1 → 2 → 3 → 4 → 5 → null
 prev curr next

Step 1: Reverse 1's pointer
null ← 1 2 → 3 → 4 → 5 → null
 prev curr next

Step 2: Reverse 2's pointer
null ← 1 ← 2 3 → 4 → 5 → null
 prev curr next

Step 3: Reverse 3's pointer
null ← 1 ← 2 ← 3 4 → 5 → null
 prev curr next

Step 4: Reverse 4's pointer
null ← 1 ← 2 ← 3 ← 4 5 → null
 prev curr next

Step 5: Reverse 5's pointer
null ← 1 ← 2 ← 3 ← 4 ← 5 null
 prev curr

Result: prev is new head
``

### Implementation

``python
def reverseList(head: ListNode) -> ListNode:
 """
 Iterative reversal using three pointers
 
 Time: O(n) - visit each node once
 Space: O(1) - only use 3 pointers
 
 Strategy:
 1. Track previous, current, and next nodes
 2. Reverse current's pointer to previous
 3. Move all pointers one step forward
 4. Repeat until end of list
 """
 prev = None
 curr = head
 
 while curr:
 # Save next node before we change the pointer
 next_temp = curr.next
 
 # Reverse the pointer
 curr.next = prev
 
 # Move prev and curr one step forward
 prev = curr
 curr = next_temp
 
 # prev is now the new head
 return prev

# Test
head = create_linked_list([1, 2, 3, 4, 5])
print("Original:", list_to_array(head))
reversed_head = reverseList(head)
print("Reversed:", list_to_array(reversed_head))
# Output:
# Original: [1, 2, 3, 4, 5]
# Reversed: [5, 4, 3, 2, 1]
``

### Step-by-Step Trace

``python
def reverseListVerbose(head: ListNode) -> ListNode:
 """Iterative reversal with detailed logging"""
 prev = None
 curr = head
 step = 0
 
 print("Initial state:")
 print(f" List: {list_to_array(head)}")
 
 while curr:
 step += 1
 print(f"\nStep {step}:")
 print(f" Current node: {curr.val}")
 
 # Save next
 next_temp = curr.next
 print(f" Saved next: {next_temp.val if next_temp else 'null'}")
 
 # Reverse pointer
 curr.next = prev
 print(f" Reversed {curr.val}'s pointer to {prev.val if prev else 'null'}")
 
 # Move forward
 prev = curr
 curr = next_temp
 
 # Show partial result
 if prev:
 print(f" Reversed portion: {list_to_array(prev)}")
 
 print(f"\nFinal: {list_to_array(prev)}")
 return prev
``

---

## Approach 2: Recursive

**Elegant but uses call stack**

### Visualization

``
reverseList(1 → 2 → 3 → 4 → 5)
 ↓
 reverseList(2 → 3 → 4 → 5)
 ↓
 reverseList(3 → 4 → 5)
 ↓
 reverseList(4 → 5)
 ↓
 reverseList(5)
 ↓
 return 5 (base case)
 ← reverse 4's pointer
 ← reverse 3's pointer
 ← reverse 2's pointer
 ← reverse 1's pointer
← return 5 as new head
``

### Implementation

``python
def reverseListRecursive(head: ListNode) -> ListNode:
 """
 Recursive reversal
 
 Time: O(n)
 Space: O(n) - recursion stack
 
 Key insight:
 - Reverse rest of list first
 - Then fix current node's pointer
 """
 # Base case: empty list or single node
 if not head or not head.next:
 return head
 
 # Recursively reverse rest of list
 new_head = reverseListRecursive(head.next)
 
 # Fix pointers
 # head.next is now the last node of reversed list
 # Make it point back to head
 head.next.next = head
 
 # Remove head's forward pointer
 head.next = None
 
 return new_head

# Test
head = create_linked_list([1, 2, 3, 4, 5])
reversed_head = reverseListRecursive(head)
print(list_to_array(reversed_head)) # [5, 4, 3, 2, 1]
``

### Understanding the Recursion

``python
def reverseListRecursiveVerbose(head: ListNode, depth=0) -> ListNode:
 """Recursive reversal with visualization"""
 indent = " " * depth
 
 # Base case
 if not head or not head.next:
 print(f"{indent}Base case: {head.val if head else 'null'}")
 return head
 
 print(f"{indent}Reversing from node {head.val}")
 print(f"{indent} Going deeper...")
 
 # Recursive call
 new_head = reverseListRecursiveVerbose(head.next, depth + 1)
 
 print(f"{indent} Returned from recursion")
 print(f"{indent} Fixing pointers for node {head.val}")
 print(f"{indent} Making {head.next.val} point to {head.val}")
 
 # Fix pointers
 head.next.next = head
 head.next = None
 
 return new_head

# Test
head = create_linked_list([1, 2, 3, 4, 5])
result = reverseListRecursiveVerbose(head)
``

---

## Approach 3: Using Stack

**Less efficient but intuitive**

``python
def reverseListStack(head: ListNode) -> ListNode:
 """
 Reverse using stack
 
 Time: O(n)
 Space: O(n) - stack storage
 
 Not optimal, but shows alternative thinking
 """
 if not head:
 return None
 
 # Push all nodes onto stack
 stack = []
 current = head
 
 while current:
 stack.append(current)
 current = current.next
 
 # Pop from stack to build reversed list
 new_head = stack.pop()
 current = new_head
 
 while stack:
 current.next = stack.pop()
 current = current.next
 
 # Important: set last node's next to None
 current.next = None
 
 return new_head

# Test
head = create_linked_list([1, 2, 3, 4, 5])
reversed_head = reverseListStack(head)
print(list_to_array(reversed_head)) # [5, 4, 3, 2, 1]
``

---

## Advanced Variations

### Variation 1: Reverse First K Nodes

``python
def reverseFirstK(head: ListNode, k: int) -> ListNode:
 """
 Reverse only first k nodes
 
 Example: [1,2,3,4,5], k=3 → [3,2,1,4,5]
 """
 if not head or k <= 1:
 return head
 
 # Reverse first k nodes
 prev = None
 curr = head
 count = 0
 
 # Reverse
 while curr and count < k:
 next_temp = curr.next
 curr.next = prev
 prev = curr
 curr = next_temp
 count += 1
 
 # Connect reversed part to rest of list
 if head:
 head.next = curr # head is now tail of reversed part
 
 return prev

# Test
head = create_linked_list([1, 2, 3, 4, 5])
result = reverseFirstK(head, 3)
print(list_to_array(result)) # [3, 2, 1, 4, 5]
``

### Variation 2: Reverse Between Positions

``python
def reverseBetween(head: ListNode, left: int, right: int) -> ListNode:
 """
 Reverse nodes from position left to right (1-indexed)
 
 Example: [1,2,3,4,5], left=2, right=4 → [1,4,3,2,5]
 
 LeetCode 92: Reverse Linked List II
 """
 if not head or left == right:
 return head
 
 # Dummy node to handle edge case where left=1
 dummy = ListNode(0)
 dummy.next = head
 
 # Move to node before left
 prev = dummy
 for _ in range(left - 1):
 prev = prev.next
 
 # Reverse from left to right
 reverse_start = prev.next
 curr = reverse_start.next
 
 for _ in range(right - left):
 # Extract curr
 reverse_start.next = curr.next
 
 # Insert curr after prev
 curr.next = prev.next
 prev.next = curr
 
 # Move to next node to reverse
 curr = reverse_start.next
 
 return dummy.next

# Test
head = create_linked_list([1, 2, 3, 4, 5])
result = reverseBetween(head, 2, 4)
print(list_to_array(result)) # [1, 4, 3, 2, 5]
``

### Variation 3: Reverse in K Groups

``python
def reverseKGroup(head: ListNode, k: int) -> ListNode:
 """
 Reverse nodes in k-group
 
 Example: [1,2,3,4,5], k=2 → [2,1,4,3,5]
 Example: [1,2,3,4,5], k=3 → [3,2,1,4,5]
 
 LeetCode 25: Reverse Nodes in k-Group
 """
 # Count total nodes
 count = 0
 current = head
 while current:
 count += 1
 current = current.next
 
 dummy = ListNode(0)
 dummy.next = head
 prev_group_end = dummy
 
 while count >= k:
 # Reverse k nodes
 group_start = prev_group_end.next
 prev = None
 curr = group_start
 
 for _ in range(k):
 next_temp = curr.next
 curr.next = prev
 prev = curr
 curr = next_temp
 
 # Connect reversed group
 prev_group_end.next = prev # prev is new group start
 group_start.next = curr # group_start is now group end
 
 # Move to next group
 prev_group_end = group_start
 count -= k
 
 return dummy.next

# Test
head = create_linked_list([1, 2, 3, 4, 5])
result = reverseKGroup(head, 2)
print(list_to_array(result)) # [2, 1, 4, 3, 5]

head = create_linked_list([1, 2, 3, 4, 5])
result = reverseKGroup(head, 3)
print(list_to_array(result)) # [3, 2, 1, 4, 5]
``

### Variation 4: Palindrome Check (Using Reversal)

``python
def isPalindrome(head: ListNode) -> bool:
 """
 Check if linked list is palindrome
 
 Strategy:
 1. Find middle of list (slow/fast pointers)
 2. Reverse second half
 3. Compare first and second half
 
 Time: O(n), Space: O(1)
 """
 if not head or not head.next:
 return True
 
 # Find middle using slow/fast pointers
 slow = fast = head
 while fast.next and fast.next.next:
 slow = slow.next
 fast = fast.next.next
 
 # Reverse second half
 second_half = reverseList(slow.next)
 
 # Compare both halves
 first_half = head
 while second_half:
 if first_half.val != second_half.val:
 return False
 first_half = first_half.next
 second_half = second_half.next
 
 return True

# Test
head = create_linked_list([1, 2, 3, 2, 1])
print(isPalindrome(head)) # True

head = create_linked_list([1, 2, 3, 4, 5])
print(isPalindrome(head)) # False
``

---

## Common Mistakes & Edge Cases

### Mistake 1: Forgetting to Save Next

``python
# WRONG: Loses reference to rest of list
def reverseListWrong(head):
 prev = None
 curr = head
 
 while curr:
 curr.next = prev # Lost reference to rest!
 prev = curr
 curr = curr.next # curr.next was just changed!
 
 return prev

# CORRECT: Save next before changing pointer
def reverseListCorrect(head):
 prev = None
 curr = head
 
 while curr:
 next_temp = curr.next # Save first!
 curr.next = prev
 prev = curr
 curr = next_temp
 
 return prev
``

### Mistake 2: Not Setting Last Node's Next to None

``python
# WRONG: Creates cycle
def reverseListCycle(head):
 if not head or not head.next:
 return head
 
 new_head = reverseListCycle(head.next)
 head.next.next = head
 # Missing: head.next = None
 
 return new_head # Creates cycle!

# CORRECT
def reverseListNoCycle(head):
 if not head or not head.next:
 return head
 
 new_head = reverseListNoCycle(head.next)
 head.next.next = head
 head.next = None # Break cycle
 
 return new_head
``

### Edge Cases Test Suite

``python
import unittest

class TestReverseLinkedList(unittest.TestCase):
 
 def test_empty_list(self):
 """Test with empty list"""
 self.assertIsNone(reverseList(None))
 
 def test_single_node(self):
 """Test with single node"""
 head = ListNode(1)
 result = reverseList(head)
 self.assertEqual(result.val, 1)
 self.assertIsNone(result.next)
 
 def test_two_nodes(self):
 """Test with two nodes"""
 head = create_linked_list([1, 2])
 result = reverseList(head)
 self.assertEqual(list_to_array(result), [2, 1])
 
 def test_multiple_nodes(self):
 """Test with multiple nodes"""
 head = create_linked_list([1, 2, 3, 4, 5])
 result = reverseList(head)
 self.assertEqual(list_to_array(result), [5, 4, 3, 2, 1])
 
 def test_duplicate_values(self):
 """Test with duplicate values"""
 head = create_linked_list([1, 1, 2, 2, 3])
 result = reverseList(head)
 self.assertEqual(list_to_array(result), [3, 2, 2, 1, 1])
 
 def test_all_same_values(self):
 """Test with all same values"""
 head = create_linked_list([5, 5, 5, 5])
 result = reverseList(head)
 self.assertEqual(list_to_array(result), [5, 5, 5, 5])
 
 def test_no_cycle_created(self):
 """Ensure no cycle is created"""
 head = create_linked_list([1, 2, 3])
 result = reverseList(head)
 
 # Traverse and count nodes
 count = 0
 curr = result
 while curr and count < 10: # Max 10 to detect cycle
 count += 1
 curr = curr.next
 
 self.assertEqual(count, 3) # Should be exactly 3 nodes

if __name__ == '__main__':
 unittest.main()
``

---

## Performance Comparison

``python
import time
import sys

def benchmark_reversal():
 """Compare different reversal approaches"""
 sizes = [100, 1000, 5000]
 iterations = 1000
 
 print("Reversal Method Comparison")
 print("=" * 60)
 print(f"{'Size':<10} {'Iterative':<15} {'Recursive':<15} {'Stack':<15}")
 print("-" * 60)
 
 for size in sizes:
 # Create test list
 head = create_linked_list(list(range(size)))
 
 # Benchmark iterative
 start = time.perf_counter()
 for _ in range(iterations):
 test_head = create_linked_list(list(range(size)))
 reverseList(test_head)
 iter_time = (time.perf_counter() - start) / iterations * 1000
 
 # Benchmark recursive (careful with stack overflow)
 if size <= 1000: # Limit recursion depth
 start = time.perf_counter()
 for _ in range(iterations):
 test_head = create_linked_list(list(range(size)))
 reverseListRecursive(test_head)
 rec_time = (time.perf_counter() - start) / iterations * 1000
 else:
 rec_time = float('inf') # Too deep for recursion
 
 # Benchmark stack
 start = time.perf_counter()
 for _ in range(iterations):
 test_head = create_linked_list(list(range(size)))
 reverseListStack(test_head)
 stack_time = (time.perf_counter() - start) / iterations * 1000
 
 print(f"{size:<10} {iter_time:<15.4f} {rec_time:<15.4f} {stack_time:<15.4f}")
 
 print("\n* Times in milliseconds")
 print("* Recursive shows 'inf' for large lists (stack overflow risk)")

benchmark_reversal()
``

---

## Connection to Caching (ML)

Linked lists are fundamental to cache implementations:

``python
class LRUNode:
 """Node for LRU cache doubly-linked list"""
 
 def __init__(self, key, value):
 self.key = key
 self.value = value
 self.prev = None
 self.next = None

class LRUCache:
 """
 LRU Cache using doubly-linked list
 
 Connection to reversal:
 - Both manipulate pointers
 - Both require careful pointer management
 - Understanding reversal helps understand cache eviction
 """
 
 def __init__(self, capacity: int):
 self.capacity = capacity
 self.cache = {} # key -> node
 
 # Dummy head and tail
 self.head = LRUNode(0, 0)
 self.tail = LRUNode(0, 0)
 self.head.next = self.tail
 self.tail.prev = self.head
 
 def _add_to_head(self, node):
 """Add node right after head (most recently used)"""
 node.next = self.head.next
 node.prev = self.head
 self.head.next.prev = node
 self.head.next = node
 
 def _remove_node(self, node):
 """Remove node from list"""
 prev_node = node.prev
 next_node = node.next
 prev_node.next = next_node
 next_node.prev = prev_node
 
 def get(self, key: int) -> int:
 """Get value and mark as recently used"""
 if key in self.cache:
 node = self.cache[key]
 self._remove_node(node)
 self._add_to_head(node)
 return node.value
 return -1
 
 def put(self, key: int, value: int) -> None:
 """Put key-value pair"""
 if key in self.cache:
 # Update existing
 node = self.cache[key]
 node.value = value
 self._remove_node(node)
 self._add_to_head(node)
 else:
 # Add new
 node = LRUNode(key, value)
 self.cache[key] = node
 self._add_to_head(node)
 
 # Evict if over capacity
 if len(self.cache) > self.capacity:
 # Remove least recently used (tail.prev)
 lru = self.tail.prev
 self._remove_node(lru)
 del self.cache[lru.key]

# Usage
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1)) # 1
cache.put(3, 3) # Evicts key 2
print(cache.get(2)) # -1 (not found)
``

---

## Production Patterns

### Pattern 1: Safe Reversal with Validation

``python
class LinkedListReverser:
 """
 Production-ready linked list reversal
 
 Features:
 - Input validation
 - Cycle detection
 - Logging
 - Error handling
 """
 
 def __init__(self):
 self.operations = 0
 
 def reverse(self, head: ListNode) -> ListNode:
 """Safely reverse linked list"""
 # Validate input
 if not self._validate_list(head):
 raise ValueError("Invalid linked list")
 
 # Check for cycles
 if self._has_cycle(head):
 raise ValueError("Cannot reverse list with cycle")
 
 # Perform reversal
 return self._reverse_iterative(head)
 
 def _validate_list(self, head: ListNode) -> bool:
 """Validate list structure"""
 if head is None:
 return True
 
 # Check for reasonable length (prevent infinite loop)
 max_length = 10000
 count = 0
 current = head
 
 while current and count < max_length:
 count += 1
 current = current.next
 
 return count < max_length
 
 def _has_cycle(self, head: ListNode) -> bool:
 """Detect cycle using Floyd's algorithm"""
 if not head:
 return False
 
 slow = fast = head
 
 while fast and fast.next:
 slow = slow.next
 fast = fast.next.next
 
 if slow == fast:
 return True
 
 return False
 
 def _reverse_iterative(self, head: ListNode) -> ListNode:
 """Iterative reversal with counting"""
 prev = None
 curr = head
 
 while curr:
 self.operations += 1
 next_temp = curr.next
 curr.next = prev
 prev = curr
 curr = next_temp
 
 return prev
 
 def get_stats(self):
 """Get operation statistics"""
 return {'operations': self.operations}

# Usage
reverser = LinkedListReverser()
head = create_linked_list([1, 2, 3, 4, 5])
result = reverser.reverse(head)
print(f"Reversed list: {list_to_array(result)}")
print(f"Operations: {reverser.get_stats()['operations']}")
``

### Pattern 2: Reversing in Streaming Context

``python
class StreamingListReverser:
 """
 Reverse linked list built from stream
 
 Useful when building list from incoming data
 """
 
 def __init__(self):
 self.head = None
 self.count = 0
 
 def add_value(self, value):
 """
 Add value to front (effectively building reversed list)
 
 More efficient than building forward then reversing
 """
 new_node = ListNode(value)
 new_node.next = self.head
 self.head = new_node
 self.count += 1
 
 def add_values(self, values):
 """Add multiple values"""
 for val in values:
 self.add_value(val)
 
 def get_list(self):
 """Get the reversed list"""
 return self.head
 
 def get_array(self):
 """Get as array"""
 return list_to_array(self.head)

# Usage: Build reversed list efficiently
reverser = StreamingListReverser()

# Add values from stream
for value in [5, 4, 3, 2, 1]:
 reverser.add_value(value)

# Result is already in order [1, 2, 3, 4, 5]
print(reverser.get_array()) # [1, 2, 3, 4, 5]
``

---

## Why Reversing Linked Lists Matters

### Beyond the Interview

Linked list reversal is more than just an interview question - it's a fundamental skill that teaches:

1. **Pointer manipulation** - Understanding how to change references carefully
2. **In-place algorithms** - Modifying data structures without extra space
3. **State management** - Tracking multiple pieces of information simultaneously
4. **Edge case handling** - Dealing with empty lists, single nodes, etc.

### Real-World Applications

``python
# Application 1: Undo/Redo functionality
class UndoStack:
 """
 Undo stack using linked list
 
 Reversing helps implement redo after undo
 """
 
 def __init__(self):
 self.undo_head = None
 self.redo_head = None
 
 def do_action(self, action):
 """Perform action and add to undo stack"""
 new_node = ListNode(action)
 new_node.next = self.undo_head
 self.undo_head = new_node
 
 # Clear redo stack
 self.redo_head = None
 
 def undo(self):
 """Undo last action"""
 if not self.undo_head:
 return None
 
 # Move from undo to redo
 action = self.undo_head.val
 
 new_redo = ListNode(action)
 new_redo.next = self.redo_head
 self.redo_head = new_redo
 
 self.undo_head = self.undo_head.next
 
 return action
 
 def redo(self):
 """Redo previously undone action"""
 if not self.redo_head:
 return None
 
 # Move from redo to undo
 action = self.redo_head.val
 self.do_action(action)
 self.redo_head = self.redo_head.next
 
 return action

# Application 2: Browser history
class BrowserHistory:
 """
 Browser forward/back navigation using two stacks
 
 Avoids requiring a doubly-linked prev pointer on ListNode
 """
 
 def __init__(self, homepage):
 self.back_stack = [homepage]
 self.forward_stack = []
 
 def visit(self, url):
 """Visit new URL"""
 self.back_stack.append(url)
 self.forward_stack.clear()
 
 def back(self, steps):
 """Go back steps pages"""
 while steps > 0 and len(self.back_stack) > 1:
 self.forward_stack.append(self.back_stack.pop())
 steps -= 1
 return self.back_stack[-1]
 
 def forward(self, steps):
 """Go forward steps pages"""
 while steps > 0 and self.forward_stack:
 self.back_stack.append(self.forward_stack.pop())
 steps -= 1
 return self.back_stack[-1]
``

---

## Deep Dive: Understanding Pointers

### Visualizing Pointer Changes

Let's understand exactly what happens to pointers during reversal:

``python
def reverseListDetailed(head: ListNode) -> ListNode:
 """
 Detailed reversal with ASCII visualization
 """
 if not head:
 print("Empty list - nothing to reverse")
 return None
 
 print("Initial state:")
 print_list_with_addresses(head)
 
 prev = None
 curr = head
 step = 0
 
 while curr:
 step += 1
 print(f"\n{'='*60}")
 print(f"STEP {step}")
 print(f"{'='*60}")
 
 # Show current state
 print(f"\nBefore pointer change:")
 print(f" prev -> {prev.val if prev else 'None'}")
 print(f" curr -> {curr.val}")
 print(f" curr.next -> {curr.next.val if curr.next else 'None'}")
 
 # Save next
 next_temp = curr.next
 print(f"\n Saved next_temp -> {next_temp.val if next_temp else 'None'}")
 
 # Reverse pointer
 print(f"\n Reversing: curr.next = prev")
 print(f" This makes {curr.val} point to {prev.val if prev else 'None'}")
 curr.next = prev
 
 # Show after reversal
 print(f"\nAfter pointer change:")
 print(f" {curr.val}.next now points to {curr.next.val if curr.next else 'None'}")
 
 # Move pointers
 print(f"\n Moving pointers forward:")
 print(f" prev = curr ({prev.val if prev else 'None'} -> {curr.val})")
 print(f" curr = next_temp ({curr.val} -> {next_temp.val if next_temp else 'None'})")
 
 prev = curr
 curr = next_temp
 
 # Show current reversed portion
 if prev:
 print(f"\n Reversed so far:")
 print_list_with_addresses(prev, limit=step)
 
 print(f"\n{'='*60}")
 print("FINAL RESULT")
 print(f"{'='*60}")
 print_list_with_addresses(prev)
 
 return prev

def print_list_with_addresses(head, limit=None):
 """Print list with pointer addresses for clarity"""
 current = head
 count = 0
 
 while current and (limit is None or count < limit):
 next_addr = id(current.next) if current.next else "None"
 print(f" [{id(current)}] {current.val} -> {next_addr}")
 current = current.next
 count += 1

# Example usage
print("DETAILED REVERSAL EXAMPLE")
print("="*60)
head = create_linked_list([1, 2, 3])
reversed_head = reverseListDetailed(head)
``

### Memory Layout Visualization

``python
class VisualListNode:
 """
 Enhanced ListNode with visualization
 """
 
 def __init__(self, val=0, next=None):
 self.val = val
 self.next = next
 self.id = id(self) # Memory address
 
 def visualize_memory(self):
 """Show memory layout"""
 print(f"┌─────────────────────┐")
 print(f"│ Node at {self.id:x} │")
 print(f"├─────────────────────┤")
 print(f"│ val: {self.val:8d} │")
 print(f"│ next: {id(self.next):x} │") if self.next else print(f"│ next: None │")
 print(f"└─────────────────────┘")

def visualize_reversal_step_by_step():
 """Complete visualization of reversal process"""
 # Create list
 nodes = [VisualListNode(i) for i in [1, 2, 3]]
 for i in range(len(nodes) - 1):
 nodes[i].next = nodes[i + 1]
 
 head = nodes[0]
 
 print("INITIAL STATE")
 print("="*60)
 for node in nodes:
 node.visualize_memory()
 if node.next:
 print(" ↓")
 
 # Reverse
 prev = None
 curr = head
 step = 0
 
 while curr:
 step += 1
 print(f"\nSTEP {step}: Reversing node {curr.val}")
 print("-"*60)
 
 next_temp = curr.next
 
 print(f"Breaking link: {curr.val} -/-> {next_temp.val if next_temp else 'None'}")
 print(f"Creating link: {curr.val} -> {prev.val if prev else 'None'}")
 
 curr.next = prev
 prev = curr
 curr = next_temp
 
 print("\nFINAL STATE")
 print("="*60)
 current = prev
 while current:
 current.visualize_memory()
 if current.next:
 print(" ↓")
 current = current.next

visualize_reversal_step_by_step()
``

---

## Interview Deep Dive

### Common Follow-up Questions

#### Q1: "Can you do it without recursion?"

**Answer:** Yes, using the iterative 3-pointer approach (O(1) space instead of O(n)).

``python
# Already covered - iterative is preferred in interviews
def reverseListIterative(head):
 prev = None
 curr = head
 
 while curr:
 next_temp = curr.next
 curr.next = prev
 prev = curr
 curr = next_temp
 
 return prev
``

#### Q2: "What if the list has a cycle?"

``python
def reverseListWithCycleDetection(head: ListNode) -> ListNode:
 """
 Reverse list, but first check for cycle
 
 If cycle exists, cannot reverse safely
 """
 # Detect cycle using Floyd's algorithm
 if has_cycle(head):
 raise ValueError("Cannot reverse list with cycle")
 
 # Safe to reverse
 return reverseList(head)

def has_cycle(head: ListNode) -> bool:
 """Floyd's cycle detection"""
 if not head:
 return False
 
 slow = fast = head
 
 while fast and fast.next:
 slow = slow.next
 fast = fast.next.next
 
 if slow == fast:
 return True
 
 return False
``

#### Q3: "How would you reverse only odd-positioned nodes?"

``python
def reverseOddPositions(head: ListNode) -> ListNode:
 """
 Reverse nodes at odd positions (1st, 3rd, 5th, ...)
 
 Example: 1->2->3->4->5 becomes 5->2->3->4->1
 Actually: 1->2->3->4->5 becomes 3->2->1->4->5
 
 More precisely: reverse 1st, 3rd, 5th positions in place
 """
 if not head or not head.next:
 return head
 
 # Collect odd-positioned nodes
 odd_nodes = []
 even_nodes = []
 
 current = head
 position = 1
 
 while current:
 if position % 2 == 1:
 odd_nodes.append(current)
 else:
 even_nodes.append(current)
 
 current = current.next
 position += 1
 
 # Reverse odd nodes
 odd_nodes.reverse()
 
 # Reconstruct list
 dummy = ListNode(0)
 current = dummy
 
 for i in range(max(len(odd_nodes), len(even_nodes))):
 if i < len(odd_nodes):
 current.next = odd_nodes[i]
 current = current.next
 
 if i < len(even_nodes):
 current.next = even_nodes[i]
 current = current.next
 
 current.next = None
 
 return dummy.next

# Test
head = create_linked_list([1, 2, 3, 4, 5])
result = reverseOddPositions(head)
print(list_to_array(result))
``

#### Q4: "Can you reverse in groups and connect them?"

``python
def reverseAndConnect(head: ListNode, k: int) -> ListNode:
 """
 Reverse in groups of k and connect reversed groups
 
 Example: [1,2,3,4,5,6,7,8], k=3
 Result: [3,2,1,6,5,4,8,7]
 
 Detailed walkthrough:
 1. Group 1: [1,2,3] -> [3,2,1]
 2. Group 2: [4,5,6] -> [6,5,4] 
 3. Group 3: [7,8] -> [8,7] (incomplete group)
 
 Connect: [3,2,1] -> [6,5,4] -> [8,7]
 """
 if not head or k <= 1:
 return head
 
 dummy = ListNode(0)
 dummy.next = head
 
 prev_group_end = dummy
 
 while True:
 # Check if we have k nodes remaining
 kth_node = prev_group_end
 for i in range(k):
 kth_node = kth_node.next
 if not kth_node:
 return dummy.next
 
 # Save next group start
 next_group_start = kth_node.next
 
 # Reverse current group
 group_start = prev_group_end.next
 prev = next_group_start
 curr = group_start
 
 for _ in range(k):
 next_temp = curr.next
 curr.next = prev
 prev = curr
 curr = next_temp
 
 # Connect reversed group
 prev_group_end.next = prev # prev is new group start
 prev_group_end = group_start # group_start is now group end
 
 return dummy.next

# Test with detailed output
head = create_linked_list([1, 2, 3, 4, 5, 6, 7, 8])
print("Original:", list_to_array(head))

result = reverseAndConnect(head, 3)
print("Reversed in groups of 3:", list_to_array(result))
``

---

## Advanced Techniques

### Technique 1: Reverse with Constraints

``python
def reverseWithMaxValue(head: ListNode, max_val: int) -> ListNode:
 """
 Reverse only nodes with value <= max_val
 
 Example: [1,5,2,6,3], max_val=4
 Result: [3,5,2,6,1] (reversed 1,2,3)
 """
 # Collect nodes to reverse
 to_reverse = []
 all_nodes = []
 
 current = head
 while current:
 all_nodes.append((current, current.val))
 if current.val <= max_val:
 to_reverse.append(current)
 current = current.next
 
 # Reverse qualified nodes' values
 values = [node.val for node in to_reverse]
 values.reverse()
 
 for i, node in enumerate(to_reverse):
 node.val = values[i]
 
 return head

# Test
head = create_linked_list([1, 5, 2, 6, 3])
result = reverseWithMaxValue(head, 4)
print(list_to_array(result)) # [3, 5, 2, 6, 1]
``

### Technique 2: Reverse Using Recursion with Tail

``python
def reverseListTailRecursive(head: ListNode) -> ListNode:
 """
 Tail-recursive reversal
 
 More efficient as can be optimized by compiler
 (though Python doesn't do tail call optimization)
 """
 def reverse_helper(curr, prev):
 # Base case
 if not curr:
 return prev
 
 # Save next
 next_node = curr.next
 
 # Reverse pointer
 curr.next = prev
 
 # Tail recursive call
 return reverse_helper(next_node, curr)
 
 return reverse_helper(head, None)
``

### Technique 3: Reverse Maintaining Original Order Information

``python
class ReversibleList:
 """
 Linked list that can be reversed and un-reversed
 
 Maintains original order information
 """
 
 def __init__(self, values):
 self.original_head = create_linked_list(values)
 self.current_head = self.original_head
 self.is_reversed = False
 
 def reverse(self):
 """Reverse the list"""
 self.current_head = reverseList(self.current_head)
 self.is_reversed = not self.is_reversed
 
 def restore_original(self):
 """Restore to original order"""
 if self.is_reversed:
 self.reverse()
 
 def get_array(self):
 """Get current list as array"""
 return list_to_array(self.current_head)
 
 def get_state(self):
 """Get current state"""
 return {
 'values': self.get_array(),
 'is_reversed': self.is_reversed
 }

# Usage
rlist = ReversibleList([1, 2, 3, 4, 5])
print("Original:", rlist.get_state())

rlist.reverse()
print("Reversed:", rlist.get_state())

rlist.restore_original()
print("Restored:", rlist.get_state())
``

---

## Performance Optimization

### Space-Optimized Variations

``python
import sys

def measure_space_complexity():
 """
 Measure actual space usage of different approaches
 """
 import tracemalloc
 
 # Create large list
 values = list(range(10000))
 
 print("Space Complexity Analysis")
 print("="*60)
 
 # Iterative approach
 tracemalloc.start()
 head = create_linked_list(values)
 before = tracemalloc.get_traced_memory()[0]
 
 reversed_head = reverseList(head)
 
 after = tracemalloc.get_traced_memory()[0]
 tracemalloc.stop()
 
 iterative_space = after - before
 print(f"Iterative: {iterative_space:,} bytes")
 
 # Recursive approach (will use more stack space)
 tracemalloc.start()
 head = create_linked_list(values)
 before = tracemalloc.get_traced_memory()[0]
 
 try:
 reversed_head = reverseListRecursive(head)
 after = tracemalloc.get_traced_memory()[0]
 recursive_space = after - before
 print(f"Recursive: {recursive_space:,} bytes")
 except RecursionError:
 print("Recursive: Stack overflow (list too large)")
 finally:
 tracemalloc.stop()

measure_space_complexity()
``

### Time Complexity Analysis

``python
def benchmark_reversal_comprehensive():
 """
 Comprehensive performance benchmark
 """
 import time
 import matplotlib.pyplot as plt
 
 sizes = [100, 500, 1000, 5000, 10000]
 iterations = 100
 
 results = {
 'iterative': [],
 'recursive': [],
 'stack': []
 }
 
 print(f"{'Size':<10} {'Iterative':>12} {'Recursive':>12} {'Stack':>12}")
 print("-" * 50)
 
 for size in sizes:
 # Iterative
 times = []
 for _ in range(iterations):
 head = create_linked_list(list(range(size)))
 start = time.perf_counter()
 reverseList(head)
 times.append(time.perf_counter() - start)
 
 iter_avg = sum(times) / len(times) * 1000 # ms
 results['iterative'].append(iter_avg)
 
 # Recursive (skip for large sizes)
 if size <= 1000:
 times = []
 for _ in range(iterations):
 head = create_linked_list(list(range(size)))
 start = time.perf_counter()
 reverseListRecursive(head)
 times.append(time.perf_counter() - start)
 
 rec_avg = sum(times) / len(times) * 1000
 results['recursive'].append(rec_avg)
 else:
 results['recursive'].append(None)
 
 # Stack-based
 times = []
 for _ in range(iterations):
 head = create_linked_list(list(range(size)))
 start = time.perf_counter()
 reverseListStack(head)
 times.append(time.perf_counter() - start)
 
 stack_avg = sum(times) / len(times) * 1000
 results['stack'].append(stack_avg)
 
 print(f"{size:<10} {iter_avg:>11.4f} {rec_avg if rec_avg else 'N/A':>11} {stack_avg:>11.4f}")
 
 return results

# Run benchmark
benchmark_results = benchmark_reversal_comprehensive()
``

---

## Key Takeaways

✅ **Pointer manipulation** - Core skill for linked list problems 
✅ **Iterative O(1) space** - Most efficient approach 
✅ **Recursive elegance** - Beautiful but O(n) space 
✅ **Save next first** - Critical pattern to avoid losing references 
✅ **Many variations** - Reverse k nodes, between positions, in groups 
✅ **Foundation for caching** - Understanding for LRU cache implementation 
✅ **Production considerations** - Validation, cycle detection, error handling 

---

## Related Problems

- **[Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)** - Reverse between positions
- **[Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)** - Reverse k nodes at a time
- **[Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)** - Uses reversal
- **[Reorder List](https://leetcode.com/problems/reorder-list/)** - L0→Ln→L1→Ln-1→...
- **[Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)** - Reverse in groups of 2
- **[Add Two Numbers](https://leetcode.com/problems/add-two-numbers/)** - Linked list manipulation
- **[Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)** - Pointer manipulation

---

**Originally published at:** [arunbaby.com/dsa/0010-reverse-linked-list](https://www.arunbaby.com/dsa/0010-reverse-linked-list/)

*If you found this helpful, consider sharing it with others who might benefit.*

