---
title: "Add Two Numbers"
day: 12
collection: dsa
categories:
  - dsa
tags:
  - linked-list
  - math
  - carry
  - elementary-math
subdomain: Linked Lists
tech_stack: [Python, Linked List, Math]
scale: "O(max(m,n)) time complexity"
companies: [Google, Meta, Amazon, Microsoft, Apple, Bloomberg]
related_ml_day: 12
related_speech_day: 12
related_agents_day: 12
---

**Master digit-by-digit addition with linked lists: Handle carry propagation elegantly. Classic problem teaching pointer manipulation and edge cases.**

## Problem Statement

You are given two **non-empty** linked lists representing two non-negative integers. The digits are stored in **reverse order**, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

### Examples

**Example 1:**
```
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807
```

Visual representation:
```
   2 → 4 → 3     (represents 342)
+  5 → 6 → 4     (represents 465)
--------------
   7 → 0 → 8     (represents 807)
```

**Example 2:**
```
Input: l1 = [0], l2 = [0]
Output: [0]
```

**Example 3:**
```
Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]
Explanation: 9999999 + 9999 = 10009998
```

### Constraints

- The number of nodes in each linked list is in the range `[1, 100]`
- `0 <= Node.val <= 9`
- It is guaranteed that the list represents a number that does not have leading zeros

---

## Understanding the Problem

### Why Reverse Order?

**The brilliant design choice**: Storing digits in reverse order makes addition **natural**!

**How we add numbers manually:**
```
  342
+ 465
-----

Step 1: Add rightmost digits: 2 + 5 = 7
Step 2: Add next digits:      4 + 6 = 10 (carry 1)
Step 3: Add leftmost digits:  3 + 4 + 1(carry) = 8

Result: 807
```

**With reverse-order linked lists:**
```
List 1: 2 → 4 → 3
List 2: 5 → 6 → 4

We traverse left-to-right, which is right-to-left in the number!
Perfect for addition!
```

**If they were in normal order** (left-to-right):
```
List 1: 3 → 4 → 2
List 2: 4 → 6 → 5

We'd need to traverse to the end first, then add backwards.
Much more complicated! ❌
```

### The Core Concept: Elementary Addition

Remember how you learned addition in elementary school?

```
   1 1        ← Carries
  342
+ 465
-----
  807
```

**Process:**
1. Start from rightmost (ones place)
2. Add digits: 2 + 5 = 7, write 7
3. Next column: 4 + 6 = 10, write 0, carry 1
4. Next column: 3 + 4 + 1 (carry) = 8, write 8

**Same algorithm, but with linked lists!**

### Key Challenges

**1. Different Lengths**

```
  12345
+    78
-------
  12423
```

Linked lists:
```
l1: 5 → 4 → 3 → 2 → 1
l2: 8 → 7
```

**Challenge**: l2 ends early. Need to handle remaining digits from l1.

**2. Carry Propagation**

```
  999
+   1
-----
 1000
```

The carry propagates all the way, creating a new digit!

```
l1: 9 → 9 → 9
l2: 1

Result: 0 → 0 → 0 → 1
        ↑ New node created by final carry!
```

**3. Final Carry**

```
  99
+ 99
----
 198
```

After adding all digits, we still have carry=1. Must create new node!

---

## Solution: Elementary Addition Algorithm

### Intuition

**Think of it like a zipper:**

```
List 1:  2 → 4 → 3 → None
List 2:  5 → 6 → 4 → None
         ↓   ↓   ↓
Result:  7 → 0 → 8 → None
```

**At each position:**
1. Get digit from l1 (or 0 if l1 ended)
2. Get digit from l2 (or 0 if l2 ended)
3. Add them plus any carry from previous: `sum = d1 + d2 + carry`
4. New digit = `sum % 10` (ones place)
5. New carry = `sum // 10` (tens place)
6. Create node with new digit
7. Move to next position

### Why This Works

**The magic of modulo and integer division:**

```python
sum = 15
digit = sum % 10    # = 5 (remainder)
carry = sum // 10   # = 1 (quotient)

Next position:
sum = next_d1 + next_d2 + 1 (carry)
```

**Example walkthrough:**

```
Position 0: 2 + 5 + 0(carry) = 7
  digit = 7 % 10 = 7
  carry = 7 // 10 = 0
  Create node: 7

Position 1: 4 + 6 + 0(carry) = 10
  digit = 10 % 10 = 0
  carry = 10 // 10 = 1
  Create node: 0

Position 2: 3 + 4 + 1(carry) = 8
  digit = 8 % 10 = 8
  carry = 8 // 10 = 0
  Create node: 8

Both lists ended, carry = 0, done!
Result: 7 → 0 → 8
```

### Implementation

```python
class ListNode:
    """
    Definition for singly-linked list node
    """
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    """
    Add two numbers represented as linked lists
    
    Time: O(max(m, n)) where m, n are lengths of l1, l2
    Space: O(max(m, n)) for the result list
    
    The algorithm is elegant because:
    1. We process lists left-to-right (which is right-to-left in the number)
    2. We handle carry naturally in each iteration
    3. We handle different lengths automatically
    """
    # Dummy head simplifies list construction
    # Why? No special case for creating the first node!
    dummy = ListNode(0)
    current = dummy
    
    # Carry from previous addition
    carry = 0
    
    # Continue while:
    # - l1 has more digits, OR
    # - l2 has more digits, OR
    # - We have a carry to propagate
    while l1 or l2 or carry:
        # Get current digits (0 if list ended)
        # Why 0? Because adding 0 doesn't change the sum!
        # This handles different lengths elegantly
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        # Add: digit1 + digit2 + carry from previous
        total = val1 + val2 + carry
        
        # Extract digit and carry using modulo and integer division
        # This is the elementary addition algorithm!
        # total can be 0-19 (max: 9+9+1)
        digit = total % 10   # Ones place: 0-9
        carry = total // 10  # Tens place: 0-1
        
        # Create new node with this digit
        current.next = ListNode(digit)
        current = current.next
        
        # Move to next positions (if they exist)
        # If a list ended, this becomes None, which stops at while check
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    
    # Return the actual head (skip dummy)
    # Why dummy? So we don't need special logic for first node!
    return dummy.next


# Helper functions for testing
def create_linked_list(nums):
    """
    Create linked list from array
    
    Example: [2, 4, 3] → 2 → 4 → 3 → None
    """
    if not nums:
        return None
    
    head = ListNode(nums[0])
    current = head
    
    for num in nums[1:]:
        current.next = ListNode(num)
        current = current.next
    
    return head

def list_to_array(head):
    """
    Convert linked list to array for easy viewing
    
    Example: 2 → 4 → 3 → None → [2, 4, 3]
    """
    result = []
    current = head
    
    while current:
        result.append(current.val)
        current = current.next
    
    return result

# Example usage
l1 = create_linked_list([2, 4, 3])  # represents 342
l2 = create_linked_list([5, 6, 4])  # represents 465

result = addTwoNumbers(l1, l2)
print(list_to_array(result))  # [7, 0, 8] represents 807
```

### Complexity Analysis

**Time Complexity: O(max(m, n))**
- We visit each node exactly once
- m = length of l1, n = length of l2
- We process max(m, n) digits

**Space Complexity: O(max(m, n))**
- Result list has at most max(m, n) + 1 nodes
- The +1 is for potential final carry
- Example: 999 + 1 = 1000 (4 nodes for 3-digit input)

**Why not O(m + n)?**
- We don't visit nodes twice, we visit max length once
- If m=5 and n=3, we visit 5 nodes total, not 8

---

## Step-by-Step Walkthrough

Let's trace through Example 1 in detail:

```
Input: l1 = [2,4,3], l2 = [5,6,4]
```

**Initial state:**
```
l1:      2 → 4 → 3 → None
l2:      5 → 6 → 4 → None
dummy:   0 → ?
current: ↑
carry:   0
```

**Iteration 1:**
```
val1 = 2, val2 = 5
total = 2 + 5 + 0 = 7
digit = 7 % 10 = 7
carry = 7 // 10 = 0

Create node: 7
dummy:   0 → 7 → ?
current:     ↑
l1:      4 → 3 → None
l2:      6 → 4 → None
```

**Iteration 2:**
```
val1 = 4, val2 = 6
total = 4 + 6 + 0 = 10
digit = 10 % 10 = 0
carry = 10 // 10 = 1

Create node: 0
dummy:   0 → 7 → 0 → ?
current:          ↑
l1:      3 → None
l2:      4 → None
```

**Iteration 3:**
```
val1 = 3, val2 = 4
total = 3 + 4 + 1 = 8
digit = 8 % 10 = 8
carry = 8 // 10 = 0

Create node: 8
dummy:   0 → 7 → 0 → 8 → ?
current:               ↑
l1:      None
l2:      None
```

**Loop condition check:**
```
l1 = None, l2 = None, carry = 0
All false! Exit loop.
```

**Return:**
```
dummy.next → 7 → 0 → 8 → None
```

**Verification:**
```
342 + 465 = 807 ✓
```

---

## Edge Cases & Common Mistakes

### Edge Case 1: Different Lengths

```python
# Input: [9,9,9,9] + [9,9]
# Expected: [8,9,9,9,1]

l1 = create_linked_list([9, 9, 9, 9])  # 9999
l2 = create_linked_list([9, 9])         # 99

result = addTwoNumbers(l1, l2)
print(list_to_array(result))  # [8, 9, 9, 9, 1]

# Verification: 9999 + 99 = 10098 ✓
```

**Why our algorithm handles this:**

```
Position 0: 9 + 9 = 18 → digit=8, carry=1
Position 1: 9 + 9 + 1 = 19 → digit=9, carry=1
Position 2: 9 + 0 + 1 = 10 → digit=0, carry=1  (l2 ended, use 0)
Position 3: 9 + 0 + 1 = 10 → digit=0, carry=1  (l2 still None)
Position 4: 0 + 0 + 1 = 1  → digit=1, carry=0  (both None, but carry!)

Result: [8,9,0,0,1] → Wait, that's wrong!
```

Let me trace this more carefully:

```
Position 0: 9 + 9 + 0 = 18 → digit=8, carry=1
Position 1: 9 + 9 + 1 = 19 → digit=9, carry=1
Position 2: 9 + 0 + 1 = 10 → digit=0, carry=1
Position 3: 9 + 0 + 1 = 10 → digit=0, carry=1
Position 4: 0 + 0 + 1 = 1  → digit=1, carry=0

Result: [8,9,0,0,1]
```

Actually that's: 10098 in reverse = [8,9,0,0,1]
But we want [8,9,9,9,1]...

Let me recalculate:
```
9999 + 99:
  9999
+   99
------
 10098

In reverse: [8,9,0,0,1]
```

Hmm, I made an error in my expected output above. Let me fix:

```python
# Input: [9,9,9,9] + [9,9]
# Expected: [8,9,0,0,1]  # represents 10098

l1 = create_linked_list([9, 9, 9, 9])  # 9999
l2 = create_linked_list([9, 9])         # 99

result = addTwoNumbers(l1, l2)
print(list_to_array(result))  # [8, 9, 0, 0, 1]

# Verification: 9999 + 99 = 10098 ✓
```

### Edge Case 2: Final Carry

**Common mistake: Forgetting final carry!**

```python
# Input: [9,9] + [9,9]
# Expected: [8,9,1]  # represents 198

# ❌ WRONG: Stopping when both lists end
def addTwoNumbers_wrong(l1, l2):
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while l1 and l2:  # ❌ Wrong condition!
        total = l1.val + l2.val + carry
        digit = total % 10
        carry = total // 10
        
        current.next = ListNode(digit)
        current = current.next
        
        l1 = l1.next
        l2 = l2.next
    
    return dummy.next  # ❌ Forgot final carry!

# This would return [8,9] instead of [8,9,1]
```

**What happens:**
```
Position 0: 9 + 9 + 0 = 18 → digit=8, carry=1
Position 1: 9 + 9 + 1 = 19 → digit=9, carry=1

Both lists end, but carry=1!
We need one more node: [8,9,1]
```

**Correct: Check carry in loop condition**

```python
while l1 or l2 or carry:  # ✓ Correct!
```

### Edge Case 3: Zero

```python
# Input: [0] + [0]
# Expected: [0]

l1 = create_linked_list([0])
l2 = create_linked_list([0])

result = addTwoNumbers(l1, l2)
print(list_to_array(result))  # [0]

# 0 + 0 = 0 ✓
```

### Edge Case 4: Single Digit + Multi Digit

```python
# Input: [9,9,9] + [1]
# Expected: [0,0,0,1]

l1 = create_linked_list([9, 9, 9])  # 999
l2 = create_linked_list([1])         # 1

result = addTwoNumbers(l1, l2)
print(list_to_array(result))  # [0, 0, 0, 1]

# 999 + 1 = 1000 ✓
```

**Trace:**
```
Position 0: 9 + 1 + 0 = 10 → digit=0, carry=1
Position 1: 9 + 0 + 1 = 10 → digit=0, carry=1
Position 2: 9 + 0 + 1 = 10 → digit=0, carry=1
Position 3: 0 + 0 + 1 = 1  → digit=1, carry=0

Result: [0,0,0,1] ✓
```

---

## Common Mistakes in Interviews

### Mistake 1: Not Using Dummy Node

```python
# ❌ WITHOUT dummy node - more complex!
def addTwoNumbers_no_dummy(l1, l2):
    carry = 0
    head = None
    tail = None
    
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        total = val1 + val2 + carry
        
        digit = total % 10
        carry = total // 10
        
        new_node = ListNode(digit)
        
        # Special case for first node!
        if not head:
            head = new_node
            tail = new_node
        else:
            tail.next = new_node
            tail = new_node
        
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    
    return head  # Have to track head separately!

# ✓ WITH dummy node - much cleaner!
def addTwoNumbers(l1, l2):
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        # ... same logic ...
        current.next = ListNode(digit)
        current = current.next
    
    return dummy.next  # One line!
```

**Why dummy is better:**
- No special case for first node
- Simpler code = fewer bugs
- One return statement
- Standard pattern for list construction

### Mistake 2: Forgetting to Handle None

```python
# ❌ WRONG: Crashes when list ends!
while l1 or l2:
    val1 = l1.val  # ❌ What if l1 is None?
    val2 = l2.val  # ❌ What if l2 is None?
    # ...

# ✓ CORRECT: Use conditional
while l1 or l2 or carry:
    val1 = l1.val if l1 else 0  # ✓ Safe!
    val2 = l2.val if l2 else 0  # ✓ Safe!
    # ...
```

### Mistake 3: Not Moving Pointers

```python
# ❌ WRONG: Infinite loop!
while l1 or l2 or carry:
    val1 = l1.val if l1 else 0
    val2 = l2.val if l2 else 0
    # ... create node ...
    
    # ❌ Forgot to move pointers!
    # l1 and l2 never become None!

# ✓ CORRECT: Always move pointers
    l1 = l1.next if l1 else None
    l2 = l2.next if l2 else None
```

---

## Follow-Up Questions

### Q1: What if numbers are in normal order (not reversed)?

**Input:** `3 → 4 → 2` represents 342

**Solution 1: Reverse both lists first**

```python
def reverse_list(head):
    """Reverse a linked list"""
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev

def addTwoNumbers_normal_order(l1, l2):
    """Add numbers in normal order"""
    # Reverse both lists
    l1_reversed = reverse_list(l1)
    l2_reversed = reverse_list(l2)
    
    # Add (same algorithm as before)
    result_reversed = addTwoNumbers(l1_reversed, l2_reversed)
    
    # Reverse result back
    result = reverse_list(result_reversed)
    
    return result

# Time: O(m + n) - three passes
# Space: O(1) - in-place reversal (not counting result)
```

**Solution 2: Use stack**

```python
def addTwoNumbers_with_stack(l1, l2):
    """Add numbers using stacks"""
    # Push all digits onto stacks
    stack1, stack2 = [], []
    
    while l1:
        stack1.append(l1.val)
        l1 = l1.next
    
    while l2:
        stack2.append(l2.val)
        l2 = l2.next
    
    # Add from top of stacks (rightmost digits)
    carry = 0
    head = None
    
    while stack1 or stack2 or carry:
        val1 = stack1.pop() if stack1 else 0
        val2 = stack2.pop() if stack2 else 0
        
        total = val1 + val2 + carry
        digit = total % 10
        carry = total // 10
        
        # Insert at head (to build in correct order)
        new_node = ListNode(digit)
        new_node.next = head
        head = new_node
    
    return head

# Time: O(m + n)
# Space: O(m + n) for stacks
```

### Q2: What if the result should be a single integer?

```python
def addTwoNumbers_to_int(l1, l2):
    """
    Convert lists to integers, add, return result
    
    Easier but doesn't work for very large numbers!
    """
    def list_to_int(head):
        """Convert linked list to integer"""
        num = 0
        multiplier = 1
        
        while head:
            num += head.val * multiplier
            multiplier *= 10
            head = head.next
        
        return num
    
    num1 = list_to_int(l1)
    num2 = list_to_int(l2)
    
    return num1 + num2

# Example
l1 = create_linked_list([2, 4, 3])  # 342
l2 = create_linked_list([5, 6, 4])  # 465

result = addTwoNumbers_to_int(l1, l2)
print(result)  # 807

# Time: O(m + n)
# Space: O(1)
# Limitation: Loses linked-list advantages, may overflow languages with fixed int size
```

### Q3: Can you do it recursively?

```python
def addTwoNumbers_recursive(l1, l2, carry=0):
    """
    Recursive solution
    
    Base case: Both lists empty and no carry
    Recursive case: Add current digits, recurse on next
    """
    # Base case: both lists ended and no carry
    if not l1 and not l2 and carry == 0:
        return None
    
    # Get current values
    val1 = l1.val if l1 else 0
    val2 = l2.val if l2 else 0
    
    # Add with carry
    total = val1 + val2 + carry
    digit = total % 10
    new_carry = total // 10
    
    # Create current node
    current = ListNode(digit)
    
    # Recurse on next nodes
    next1 = l1.next if l1 else None
    next2 = l2.next if l2 else None
    current.next = addTwoNumbers_recursive(next1, next2, new_carry)
    
    return current

# Example
l1 = create_linked_list([2, 4, 3])
l2 = create_linked_list([5, 6, 4])

result = addTwoNumbers_recursive(l1, l2)
print(list_to_array(result))  # [7, 0, 8]

# Time: O(max(m, n))
# Space: O(max(m, n)) for recursion stack
```

---

## Connection to Distributed Systems

This problem teaches concepts used in distributed computing!

### 1. Distributed Addition

In distributed systems, we often need to aggregate results from multiple nodes:

```python
class DistributedCounter:
    """
    Count across distributed nodes
    
    Each node has a digit, we add them up with carry
    Similar to linked list addition!
    """
    
    def __init__(self, nodes):
        self.nodes = nodes  # List of nodes with values
    
    def aggregate(self):
        """
        Aggregate counts from all nodes
        
        Like adding linked lists!
        """
        total = 0
        carry = 0
        
        for node in self.nodes:
            value = node.get_count()
            total = value + carry
            
            # Process carry
            carry = total // 10000  # Assuming each node counts to 10000
            node_result = total % 10000
            
            print(f"Node result: {node_result}, Carry: {carry}")
        
        return total

# This is conceptually similar to our linked list addition!
# Each node is like a digit, carry propagates between nodes
```

### 2. Pipeline Processing

```python
class ProcessingPipeline:
    """
    Multi-stage processing pipeline
    
    Each stage processes data and passes result + metadata to next
    Similar to carry propagation!
    """
    
    def process(self, data):
        result = data
        metadata = {}  # Like carry
        
        for stage in self.stages:
            result, metadata = stage.process(result, metadata)
            # Metadata (like carry) flows through pipeline
        
        return result
```

---

## Advanced: Handling Negative Numbers

**Challenge**: What if numbers can be negative?

**Example:**
```
l1 = [-2,4,3]  # Represents -342
l2 = [5,6,4]   # Represents 465
Result: 465 - 342 = 123 → [3,2,1]
```

**Approach:**

1. **Store sign separately**
2. **If signs same**: Add magnitudes, keep sign
3. **If signs different**: Subtract magnitudes, take sign of larger

```python
class SignedNumber:
    """
    Represent a signed number as linked list
    """
    
    def __init__(self, digits_list, is_negative=False):
        self.digits = digits_list  # ListNode
        self.is_negative = is_negative
    
    def __repr__(self):
        sign = "-" if self.is_negative else "+"
        return f"{sign}{list_to_str(self.digits)}"

def addSignedNumbers(num1, num2):
    """
    Add two signed numbers
    
    Handles positive and negative
    """
    # Case 1: Both positive or both negative
    if num1.is_negative == num2.is_negative:
        # Add magnitudes
        result_digits = addTwoNumbers(num1.digits, num2.digits)
        return SignedNumber(result_digits, num1.is_negative)
    
    # Case 2: Different signs (subtraction)
    else:
        # Determine which is larger
        if is_greater(num1.digits, num2.digits):
            # |num1| > |num2|, so result has num1's sign
            result_digits = subtractTwoNumbers(num1.digits, num2.digits)
            return SignedNumber(result_digits, num1.is_negative)
        else:
            # |num2| > |num1|, so result has num2's sign
            result_digits = subtractTwoNumbers(num2.digits, num1.digits)
            return SignedNumber(result_digits, num2.is_negative)

def is_greater(l1, l2):
    """
    Check if l1 > l2 (magnitudes)
    
    For reverse order lists
    """
    # Convert to integers for comparison
    # In production, do digit-by-digit comparison
    num1 = list_to_int_helper(l1)
    num2 = list_to_int_helper(l2)
    return num1 > num2

def list_to_int_helper(head):
    """Convert list to integer"""
    num = 0
    multiplier = 1
    while head:
        num += head.val * multiplier
        multiplier *= 10
        head = head.next
    return num

def subtractTwoNumbers(l1, l2):
    """Subtract l2 from l1 (assuming l1 >= l2)"""
    dummy = ListNode(0)
    current = dummy
    borrow = 0
    
    while l1 or l2 or borrow:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        diff = val1 - val2 - borrow
        
        if diff < 0:
            diff += 10
            borrow = 1
        else:
            borrow = 0
        
        current.next = ListNode(diff)
        current = current.next
        
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
    
    return dummy.next
```

---

## Interview Follow-Ups You Might Get

### Follow-Up 1: "Can you do it without creating new nodes?"

**Answer**: Yes, we can reuse one of the input lists.

```python
def addTwoNumbers_inplace(l1, l2):
    """
    Modify l1 in-place to store result
    
    Space: O(1) (no new nodes except when l1 ends)
    """
    result_head = l1
    current = l1
    prev = None
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        digit = total % 10
        
        if l1:
            # Reuse l1 node
            l1.val = digit
            prev = l1
            current = l1
            l1 = l1.next
        else:
            # l1 ended, need new node
            new_node = ListNode(digit)
            prev.next = new_node
            prev = new_node
        
        if l2:
            l2 = l2.next
    
    return result_head
```

**Pros**: O(1) space (excluding output)  
**Cons**: Destroys input list (side effects)

### Follow-Up 2: "What if lists can have leading zeros?"

**Example:**
```
l1 = [0,0,1]  # Represents 100
l2 = [0,1]    # Represents 10
```

**Answer**: Same algorithm works! Leading zeros don't affect addition.

But for output, you might want to remove trailing zeros (in reverse order):

```python
def remove_trailing_zeros(head):
    """
    Remove trailing zeros from result
    
    For reverse order: [0,0,1,0,0] → [0,0,1]
    """
    # Find last non-zero node
    current = head
    last_non_zero = None
    
    while current:
        if current.val != 0:
            last_non_zero = current
        current = current.next
    
    # Trim after last non-zero
    if last_non_zero:
        last_non_zero.next = None
    else:
        # All zeros - keep one zero
        return ListNode(0)
    
    return head
```

### Follow-Up 3: "How would you optimize for very long lists?"

**Answers:**

1. **Parallel processing** (as shown in Distributed Systems section)
2. **Chunk processing**: Process in chunks of 1000 digits, aggregate carries
3. **Hardware acceleration**: Use SIMD instructions for parallel digit addition

```python
def add_chunked(l1, l2, chunk_size=1000):
    """
    Process in chunks for better cache locality
    
    Useful for extremely long lists (millions of digits)
    """
    chunks1 = split_into_chunks(l1, chunk_size)
    chunks2 = split_into_chunks(l2, chunk_size)
    
    result_chunks = []
    carry = 0
    
    for c1, c2 in zip(chunks1, chunks2):
        chunk_result, carry = add_chunk_with_carry(c1, c2, carry)
        result_chunks.append(chunk_result)
    
    return merge_chunks(result_chunks)
```

---

## Key Takeaways

✅ **Dummy node pattern** - Simplifies list construction  
✅ **Handle different lengths** - Use 0 for ended lists  
✅ **Don't forget final carry** - Check in loop condition  
✅ **Modulo arithmetic** - Extract digit and carry elegantly  
✅ **Think elementary math** - Algorithm mirrors hand addition  

**Core pattern:**
```python
while l1 or l2 or carry:
    val1 = l1.val if l1 else 0
    val2 = l2.val if l2 else 0
    total = val1 + val2 + carry
    digit = total % 10
    carry = total // 10
    # Create node, move pointers
```

---

**Originally published at:** [arunbaby.com/dsa/0012-add-two-numbers](https://www.arunbaby.com/dsa/0012-add-two-numbers/)

*If you found this helpful, consider sharing it with others who might benefit.*

