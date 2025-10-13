---
title: "Valid Parentheses"
day: 2
collection: dsa
categories:
  - dsa
tags:
  - stack
  - strings
topic: Stack
difficulty: Easy
companies: [Google, Meta, Amazon, Microsoft]
leetcode_link: "https://leetcode.com/problems/valid-parentheses/"
time_complexity: "O(n)"
space_complexity: "O(n)"
related_ml_day: 2
related_speech_day: 2
---

**Why a simple stack solves bracket matching, expression parsing, and even neural network depth management in one elegant pattern.**

## Introduction

The Valid Parentheses problem introduces one of the most fundamental data structures in computer science: the **stack**. While the problem itself seems simple—matching brackets in a string—the underlying pattern is ubiquitous in software engineering:

- **Compilers** use stacks to parse expressions and ensure syntactic correctness
- **Web browsers** use stacks to manage the back button (page history)
- **Text editors** use stacks for undo/redo functionality
- **Operating systems** use stacks to manage function calls (call stack)
- **ML pipelines** use stacks to validate nested transformations

The beauty of stacks lies in their **Last-In-First-Out (LIFO)** property, which naturally matches the structure of nested operations. When you open a bracket `(`, you expect it to be closed `)` before any bracket opened before it. This LIFO behavior is precisely what stacks provide.

**What you'll learn:**
- Why stacks are the natural solution for matching problems
- How to implement stack-based solutions efficiently
- Common variations and extensions
- Real-world applications in ML systems and compilers
- Edge cases and production considerations
- Performance optimization techniques

---

## Problem Statement

Given a string containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets
2. Open brackets must be closed in the correct order
3. Every close bracket has a corresponding open bracket of the same type

### Examples

**Example 1:**
```
Input: s = "()"
Output: true
Explanation: Single pair of parentheses, properly matched
```

**Example 2:**
```
Input: s = "()[]{}"
Output: true
Explanation: Three pairs, each properly matched
```

**Example 3:**
```
Input: s = "(]"
Output: false
Explanation: Mismatched bracket types - opened '(' but closed ']'
```

**Example 4:**
```
Input: s = "([)]"
Output: false
Explanation: Wrong closing order - opened '[' but it's closed after ')'
```

**Example 5:**
```
Input: s = "{[]}"
Output: true
Explanation: Properly nested brackets
```

### Constraints
- `1 <= s.length <= 10^4`
- `s` consists of parentheses only `'()[]{}'`

---

## Understanding the Problem

### Why This is a Stack Problem

Consider the string `"([{}])"`:

```
Position:  0 1 2 3 4 5
String:    ( [ { } ] )
```

**Processing order:**
1. See `(` → Must remember to close it later
2. See `[` → Must remember to close it later
3. See `{` → Must remember to close it later
4. See `}` → Must match with most recent opening: `{` ✓
5. See `]` → Must match with most recent opening: `[` ✓
6. See `)` → Must match with most recent opening: `(` ✓

**Key observation:** We always match with the **most recent unclosed** opening bracket. This is exactly what stacks do!

### What Makes a String Invalid?

**Type 1: Wrong bracket type**
```
"(]"
Open: (
Close: ]
Error: Types don't match
```

**Type 2: Wrong closing order**
```
"([)]"
Opens: ( [
Next close: )
Error: Expected ] (most recent opening), got )
```

**Type 3: Unclosed opening brackets**
```
"((("
Opens: ( ( (
Closes: none
Error: Stack not empty at end
```

**Type 4: Extra closing brackets**
```
"())"
Opens: (
Closes: ) )
Error: Second ) has nothing to match
```

---

## Approach 1: Brute Force (Naive)

### The Idea

Repeatedly remove all adjacent valid pairs until no more removals are possible.

```python
def isValid(s: str) -> bool:
    """
    Brute force: Keep removing valid pairs
    """
    while True:
        old_len = len(s)
        
        # Remove all valid pairs
        s = s.replace('()', '')
        s = s.replace('[]', '')
        s = s.replace('{}', '')
        
        # If no removal happened, we're done
        if len(s) == old_len:
            break
    
    # Valid if string is now empty
    return len(s) == 0
```

### Example Walkthrough

```
Input: "([{}])"

Iteration 1:
  - Replace "{}": "([]))"
  - Length changed, continue

Iteration 2:
  - Replace "[]": "()"
  - Length changed, continue

Iteration 3:
  - Replace "()": ""
  - Length changed, continue

Iteration 4:
  - No replacements possible
  - String is empty → return True
```

### Complexity Analysis

**Time Complexity: O(n²)**
- Outer loop: Can run up to n/2 times (each iteration removes 2 characters minimum)
- Each iteration: O(n) to scan and replace substrings
- Total: O(n²)

**Space Complexity: O(n)**
- String replacements create new strings

### Why This is Inefficient

For a string like `"(((())))"`:
```
Iteration 1: "(((())))" → "(())"    # Remove 2 chars
Iteration 2: "(())"     → ""        # Remove 4 chars
```

We're doing O(n) work per iteration, and iterations scale with depth of nesting.

---

## Approach 2: Stack (Optimal)

### The Insight

Instead of removing pairs, **remember opening brackets on a stack** and match them with closing brackets as we encounter them.

### Algorithm

```python
def isValid(s: str) -> bool:
    """
    Stack-based solution: O(n) time, O(n) space
    
    Key idea: Stack naturally maintains LIFO order
    """
    # Stack to store opening brackets
    stack = []
    
    # Mapping of opening to closing brackets
    pairs = {
        '(': ')',
        '[': ']',
        '{': '}'
    }
    
    for char in s:
        if char in pairs:
            # Opening bracket: push to stack
            stack.append(char)
        else:
            # Closing bracket: must match top of stack
            if not stack:
                # No opening bracket to match
                return False
            
            opening = stack.pop()
            if pairs[opening] != char:
                # Wrong type of bracket
                return False
    
    # All brackets should be matched
    return len(stack) == 0
```

### Detailed Walkthrough

**Example 1: `"([{}])"`**

```
Initial: stack = []

char='(': Opening → stack = ['(']
char='[': Opening → stack = ['(', '[']
char='{': Opening → stack = ['(', '[', '{']
char='}': Closing
  - Stack not empty ✓
  - Pop '{', pairs['{'] = '}' = char ✓
  - stack = ['(', '[']
char=']': Closing
  - Stack not empty ✓
  - Pop '[', pairs['['] = ']' = char ✓
  - stack = ['(']
char=')': Closing
  - Stack not empty ✓
  - Pop '(', pairs['('] = ')' = char ✓
  - stack = []

Final: stack = [] (empty) → return True ✓
```

**Example 2: `"([)]"` (Invalid)**

```
Initial: stack = []

char='(': Opening → stack = ['(']
char='[': Opening → stack = ['(', '[']
char=')': Closing
  - Stack not empty ✓
  - Pop '[', pairs['['] = ']' ≠ ')' ✗
  - Return False

Error: Expected ']' to match '[', got ')'
```

**Example 3: `"((("` (Invalid - Unclosed)**

```
char='(': stack = ['(']
char='(': stack = ['(', '(']
char='(': stack = ['(', '(', '(']

End of string: stack = ['(', '(', '('] (not empty)
Return False ✗
```

**Example 4: `")))"` (Invalid - No Opening)**

```
char=')': Closing
  - Stack is empty ✗
  - Return False

Error: Closing bracket with no opening bracket
```

### Why Stack is Optimal

**1. Natural LIFO Matching**
- Most recent opening must be closed first
- Stack's pop() gives us exactly that

**2. O(1) Operations**
- Push: O(1)
- Pop: O(1)
- Check empty: O(1)

**3. Single Pass**
- We only iterate through the string once
- No need to repeatedly scan like brute force

**4. Early Exit**
- Can return False immediately on mismatch
- No need to process entire string

### Complexity Analysis

**Time Complexity: O(n)**
- Single pass through string
- Each character processed once
- Stack operations are O(1)

**Space Complexity: O(n)**
- In worst case, all characters are opening brackets
- Stack size: at most n/2 for valid strings, at most n for invalid
- Example: `"(((((("` → stack has 6 elements

---

## Deep Dive: Stack Data Structure

### What is a Stack?

A **stack** is a linear data structure following **Last-In-First-Out (LIFO)** principle.

**Operations:**
```python
stack = []

# Push: Add to top
stack.append('A')    # ['A']
stack.append('B')    # ['A', 'B']
stack.append('C')    # ['A', 'B', 'C']

# Pop: Remove from top
item = stack.pop()   # Returns 'C', stack = ['A', 'B']
item = stack.pop()   # Returns 'B', stack = ['A']

# Peek: View top without removing
top = stack[-1]      # Returns 'A', stack unchanged

# Check empty
is_empty = len(stack) == 0
```

### Stack vs Other Data Structures

| Operation | Stack | Queue | Array | Linked List |
|-----------|-------|-------|-------|-------------|
| Add to end | O(1) | O(1) | O(1)† | O(1) |
| Remove from end | O(1) | O(n) | O(1) | O(1) |
| Remove from front | O(n) | O(1) | O(n) | O(1) |
| Access middle | O(n) | O(n) | O(1) | O(n) |
| LIFO | Yes | No | No | No |
| FIFO | No | Yes | No | No |

† Amortized O(1) due to dynamic array resizing

### When to Use Stacks

**Use stacks when you need:**
- ✅ LIFO access pattern
- ✅ Undo/redo functionality
- ✅ Backtracking (DFS)
- ✅ Expression parsing
- ✅ Nested structure validation

**Don't use stacks when you need:**
- ❌ FIFO access (use queue)
- ❌ Random access to elements (use array)
- ❌ Minimum/maximum tracking (use heap)
- ❌ Sorted order maintenance (use tree)

---

## Alternative Implementations

### Using a List (Default Python)

```python
def isValid(s: str) -> bool:
    stack = []  # Python list as stack
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in s:
        if char in pairs:
            stack.append(char)
        else:
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return not stack  # Pythonic way to check empty
```

### Using collections.deque (More Efficient)

```python
from collections import deque

def isValid(s: str) -> bool:
    """
    Using deque for slightly better performance
    """
    stack = deque()  # Optimized for stack operations
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in s:
        if char in pairs:
            stack.append(char)
        else:
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return len(stack) == 0
```

**Why deque?**
- Optimized for append/pop from both ends
- O(1) guaranteed (list can occasionally be O(n) during resize)
- Better memory locality for very large stacks

**Performance comparison (1M operations):**
- List: ~0.120 seconds
- Deque: ~0.095 seconds (20% faster)

### Using String as Stack (Space-optimized)

```python
def isValid(s: str) -> bool:
    """
    Use string instead of list (immutable, but works for small inputs)
    Not recommended for production!
    """
    stack_str = ""
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in s:
        if char in pairs:
            stack_str += char
        else:
            if not stack_str or pairs[stack_str[-1]] != char:
                return False
            stack_str = stack_str[:-1]  # Remove last char
    
    return stack_str == ""
```

**Why this is worse:**
- String concatenation is O(n) in Python
- Creates new string on each modification
- Total complexity: O(n²) vs O(n)

---

## Variations and Extensions

### Variation 1: Return Index of First Mismatch

```python
def findMismatch(s: str) -> int:
    """
    Return index of first mismatched bracket, or -1 if valid
    
    Useful for syntax highlighting in IDEs
    """
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for i, char in enumerate(s):
        if char in pairs:
            # Store (bracket, index) pair
            stack.append((char, i))
        else:
            if not stack:
                # Closing bracket with no opening
                return i
            
            opening, opening_idx = stack.pop()
            if pairs[opening] != char:
                # Type mismatch
                return i
    
    # If stack not empty, return index of first unclosed bracket
    if stack:
        return stack[0][1]
    
    return -1  # Valid string

# Examples
print(findMismatch("()"))      # -1 (valid)
print(findMismatch("(]"))      # 1 (mismatch at index 1)
print(findMismatch("(()"))     # 0 (unclosed at index 0)
print(findMismatch(")"))       # 0 (no opening for closing)
```

### Variation 2: Count Minimum Removals

```python
def minRemoveToMakeValid(s: str) -> int:
    """
    Count minimum brackets to remove to make string valid
    
    Similar to edit distance for brackets
    """
    stack = []
    to_remove = 0
    
    for char in s:
        if char == '(':
            stack.append('(')
        elif char == ')':
            if stack:
                stack.pop()
            else:
                # Extra closing bracket
                to_remove += 1
    
    # Unclosed opening brackets
    to_remove += len(stack)
    
    return to_remove

# Examples
print(minRemoveToMakeValid("()"))      # 0
print(minRemoveToMakeValid("(()"))     # 1 (remove one '(')
print(minRemoveToMakeValid("())"))     # 1 (remove one ')')
print(minRemoveToMakeValid("()("))     # 1
```

### Variation 3: Remove Invalid Brackets

```python
def removeInvalidParentheses(s: str) -> str:
    """
    Remove minimum number of brackets to make valid
    
    Two-pass algorithm:
    1. Remove invalid closing brackets (left-to-right)
    2. Remove invalid opening brackets (right-to-left)
    """
    def removeInvalid(s, open_char, close_char):
        """
        Single pass to remove invalid closing brackets
        """
        count = 0
        result = []
        
        for char in s:
            if char == open_char:
                count += 1
            elif char == close_char:
                if count == 0:
                    # Invalid closing bracket, skip it
                    continue
                count -= 1
            
            result.append(char)
        
        return ''.join(result)
    
    # First pass: remove invalid closing
    s = removeInvalid(s, '(', ')')
    
    # Second pass: remove invalid opening (process reversed string)
    s = removeInvalid(s[::-1], ')', '(')[::-1]
    
    return s

# Examples
print(removeInvalidParentheses("()())()"))  # "()()()" or "(())()"
print(removeInvalidParentheses("(a)())()")) # "(a)()()"
print(removeInvalidParentheses(")("))       # ""
```

### Variation 4: Longest Valid Parentheses

```python
def longestValidParentheses(s: str) -> int:
    """
    Find length of longest valid parentheses substring
    
    Example: "(()" → 2 (substring "()")
             ")()())" → 4 (substring "()()")
    """
    stack = [-1]  # Initialize with base index
    max_length = 0
    
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:  # char == ')'
            stack.pop()
            if not stack:
                # No matching opening, new base
                stack.append(i)
            else:
                # Calculate length from last unmatched
                current_length = i - stack[-1]
                max_length = max(max_length, current_length)
    
    return max_length

# Examples
print(longestValidParentheses("(()"))      # 2
print(longestValidParentheses(")()())"))   # 4
print(longestValidParentheses(""))         # 0
```

### Variation 5: Generate All Valid Parentheses

```python
def generateParentheses(n: int) -> list[str]:
    """
    Generate all combinations of n pairs of valid parentheses
    
    Example: n=3 → ["((()))", "(()())", "(())()", "()(())", "()()()"]
    
    Uses backtracking with stack validation
    """
    result = []
    
    def backtrack(current, open_count, close_count):
        # Base case: used all n pairs
        if len(current) == 2 * n:
            result.append(current)
            return
        
        # Can add opening if we haven't used all n
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        
        # Can add closing if it would still be valid
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)
    
    backtrack('', 0, 0)
    return result

# Example
print(generateParentheses(3))
# Output: ['((()))', '(()())', '(())()', '()(())', '()()()']
```

---

## Edge Cases

### Edge Case 1: Empty String

```python
s = ""
# Depends on problem definition
# Usually: return True (vacuously valid)
```

### Edge Case 2: Single Character

```python
s = "("   # False (unclosed)
s = ")"   # False (no opening)
```

### Edge Case 3: Only Opening Brackets

```python
s = "((((("  # False (none closed)
stack = ['(', '(', '(', '(', '(']  # Not empty
```

### Edge Case 4: Only Closing Brackets

```python
s = ")))))"  # False (no opening to match)
# First ')' causes immediate failure
```

### Edge Case 5: Deeply Nested

```python
s = "(" * 5000 + ")" * 5000  # 10,000 characters
# Valid! Stack will grow to 5000, then empty
# Tests stack capacity and memory
```

### Edge Case 6: Alternating Pattern

```python
s = "()()()()"  # Valid
stack never grows beyond size 1
# Efficient: O(1) space in practice
```

### Edge Case 7: Completely Nested

```python
s = "(((())))"  # Valid
stack grows to n/2, then shrinks to 0
# Worst case for space: O(n/2) = O(n)
```

---

## Production Considerations

### Input Validation

```python
def isValidRobust(s: str) -> bool:
    """
    Production-ready with validation
    """
    # Validate input
    if s is None:
        raise TypeError("Input cannot be None")
    
    if not isinstance(s, str):
        raise TypeError(f"Expected string, got {type(s)}")
    
    # Empty string is valid
    if not s:
        return True
    
    # Quick check: odd length can't be valid
    if len(s) % 2 != 0:
        return False
    
    # Define valid characters
    valid_chars = set('()[]{}')
    pairs = {'(': ')', '[': ']', '{': '}'}
    closing = set(pairs.values())
    
    stack = []
    
    for i, char in enumerate(s):
        # Validate character
        if char not in valid_chars:
            raise ValueError(f"Invalid character '{char}' at index {i}")
        
        if char in pairs:
            # Opening bracket
            stack.append(char)
        elif char in closing:
            # Closing bracket
            if not stack:
                return False  # No opening to match
            
            opening = stack.pop()
            if pairs[opening] != char:
                return False  # Type mismatch
    
    return len(stack) == 0
```

### Performance Optimizations

**Optimization 1: Early Exit on Odd Length**

```python
# Odd length can never be valid
if len(s) & 1:  # Bitwise AND is faster than modulo
    return False
```

**Savings:** Skip processing for 50% of invalid inputs

**Optimization 2: Pre-allocate Stack Capacity**

```python
# Python lists auto-resize, but we can hint capacity
stack = []
# For C++/Java: reserve stack capacity upfront
# stack.reserve(len(s) // 2)
```

**Savings:** Reduces memory allocations during execution

**Optimization 3: Use Set for Closing Brackets**

```python
pairs = {'(': ')', '[': ']', '{': '}'}
closing = set(pairs.values())  # O(1) lookup

for char in s:
    if char in pairs:  # O(1)
        stack.append(char)
    elif char in closing:  # O(1) instead of O(3) list search
        # ...
```

**Savings:** Marginal but cleaner

**Optimization 4: Avoid Repeated Dict Lookups**

```python
# Instead of checking pairs[opening] multiple times
# Cache the result
expected_closing = pairs.get(stack[-1], None)
if expected_closing != char:
    return False
```

### Memory Optimization for Constrained Environments

```python
def isValidMemoryEfficient(s: str) -> bool:
    """
    Optimize for memory-constrained environments
    
    Trade-off: Slightly more complex code for lower memory
    """
    # Use indices instead of storing characters
    # Opening brackets: ( = 0, [ = 1, { = 2
    # Closing brackets: ) = 0, ] = 1, } = 2
    
    opening = {'(': 0, '[': 1, '{': 2}
    closing = {')': 0, ']': 1, '}': 2}
    
    # Stack stores integers (4 bytes) instead of chars
    stack = []
    
    for char in s:
        if char in opening:
            stack.append(opening[char])
        elif char in closing:
            if not stack or stack.pop() != closing[char]:
                return False
    
    return not stack
```

**Memory savings:**
- Storing `int` (4 bytes) vs `str` (28+ bytes in Python)
- For 10,000 character string: ~240 KB vs ~1.4 MB

---

## Real-World Applications

### Application 1: Expression Parser

**Problem:** Validate mathematical expressions

```python
def validateExpression(expr: str) -> bool:
    """
    Validate expression has balanced brackets
    
    Examples:
    - "(2 + 3) * 4" → Valid
    - "((2 + 3)" → Invalid
    - "2 + (3 * [4 - 5])" → Valid
    """
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in expr:
        if char in pairs:
            stack.append(char)
        elif char in pairs.values():
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return not stack

# Usage in calculator
def evaluate(expr: str):
    if not validateExpression(expr):
        raise SyntaxError("Invalid expression: unmatched brackets")
    
    # Proceed with evaluation
    return eval(expr)
```

### Application 2: HTML/XML Tag Validation

**Problem:** Check if HTML tags are properly nested

```python
import re

def validateHTML(html: str) -> bool:
    """
    Validate HTML tags are properly nested
    
    Example:
    - "<div><p>Hello</p></div>" → Valid
    - "<div><p>Hello</div></p>" → Invalid
    """
    # Extract tags
    tag_pattern = r'<(/?)(\w+)[^>]*>'
    tags = re.findall(tag_pattern, html)
    
    stack = []
    
    for is_closing, tag_name in tags:
        if not is_closing:
            # Opening tag
            stack.append(tag_name)
        else:
            # Closing tag
            if not stack or stack.pop() != tag_name:
                return False
    
    return not stack

# Examples
print(validateHTML("<div><p>Text</p></div>"))  # True
print(validateHTML("<div><p>Text</div></p>"))  # False
```

### Application 3: Function Call Stack Validation

**Problem:** Ensure function calls are properly matched with returns

```python
class FunctionCallTracker:
    """
    Track function call depth for debugging/profiling
    """
    def __init__(self):
        self.call_stack = []
    
    def enter_function(self, func_name: str):
        """Called when entering a function"""
        self.call_stack.append((func_name, time.time()))
        print(f"{'  ' * len(self.call_stack)}→ {func_name}")
    
    def exit_function(self, func_name: str):
        """Called when exiting a function"""
        if not self.call_stack:
            raise RuntimeError("exit_function called without matching enter")
        
        name, start_time = self.call_stack.pop()
        if name != func_name:
            raise RuntimeError(f"Expected to exit {name}, got {func_name}")
        
        duration = time.time() - start_time
        print(f"{'  ' * len(self.call_stack)}← {func_name} ({duration:.3f}s)")
    
    def is_balanced(self) -> bool:
        """Check if all function calls have been exited"""
        return len(self.call_stack) == 0

# Usage
tracker = FunctionCallTracker()

def func_a():
    tracker.enter_function("func_a")
    func_b()
    tracker.exit_function("func_a")

def func_b():
    tracker.enter_function("func_b")
    # ... do work ...
    tracker.exit_function("func_b")
```

### Application 4: ML Pipeline Validation

**Problem:** Ensure data transformation pipeline stages are properly nested

```python
class PipelineValidator:
    """
    Validate ML pipeline stages are properly structured
    
    Example pipeline:
    StartPipeline
      |- StartPreprocess
      |    |- StartNormalization
      |    |- EndNormalization
      |- EndPreprocess
      |- StartModel
      |    |- StartTraining
      |    |- EndTraining
      |- EndModel
    EndPipeline
    """
    def __init__(self):
        self.stage_stack = []
    
    def start_stage(self, stage_name: str):
        """Enter a pipeline stage"""
        self.stage_stack.append(stage_name)
        print(f"{'  ' * len(self.stage_stack)}Start: {stage_name}")
    
    def end_stage(self, stage_name: str):
        """Exit a pipeline stage"""
        if not self.stage_stack:
            raise ValueError(f"end_stage({stage_name}) called without matching start")
        
        expected = self.stage_stack.pop()
        if expected != stage_name:
            raise ValueError(f"Expected to end {expected}, got {stage_name}")
        
        print(f"{'  ' * len(self.stage_stack)}End: {stage_name}")
    
    def validate(self) -> bool:
        """Check if all stages properly closed"""
        if self.stage_stack:
            raise ValueError(f"Unclosed stages: {self.stage_stack}")
        return True

# Usage
validator = PipelineValidator()

# Valid pipeline
validator.start_stage("Pipeline")
validator.start_stage("Preprocess")
validator.start_stage("Normalize")
validator.end_stage("Normalize")
validator.end_stage("Preprocess")
validator.start_stage("Model")
validator.end_stage("Model")
validator.end_stage("Pipeline")

validator.validate()  # ✓ All stages properly nested
```

### Application 5: Undo/Redo Functionality

**Problem:** Implement undo/redo for text editor

```python
class TextEditor:
    """
    Text editor with undo/redo using two stacks
    """
    def __init__(self):
        self.text = ""
        self.undo_stack = []  # Stack of previous states
        self.redo_stack = []  # Stack of undone actions
    
    def type(self, char: str):
        """Add character"""
        # Save current state for undo
        self.undo_stack.append(self.text)
        
        # Clear redo stack (new action invalidates redo)
        self.redo_stack = []
        
        # Update text
        self.text += char
    
    def undo(self):
        """Undo last action"""
        if not self.undo_stack:
            print("Nothing to undo")
            return
        
        # Save current state for redo
        self.redo_stack.append(self.text)
        
        # Restore previous state
        self.text = self.undo_stack.pop()
    
    def redo(self):
        """Redo last undone action"""
        if not self.redo_stack:
            print("Nothing to redo")
            return
        
        # Save current state for undo
        self.undo_stack.append(self.text)
        
        # Restore redone state
        self.text = self.redo_stack.pop()
    
    def __str__(self):
        return self.text

# Example
editor = TextEditor()
editor.type('H')
editor.type('e')
editor.type('l')
editor.type('l')
editor.type('o')
print(editor)  # "Hello"

editor.undo()
print(editor)  # "Hell"

editor.redo()
print(editor)  # "Hello"
```

---

## Testing Strategy

### Comprehensive Test Suite

```python
import unittest

class TestValidParentheses(unittest.TestCase):
    
    def test_empty_string(self):
        """Empty string should be valid"""
        self.assertTrue(isValid(""))
    
    def test_single_pair(self):
        """Single pair of each type"""
        self.assertTrue(isValid("()"))
        self.assertTrue(isValid("[]"))
        self.assertTrue(isValid("{}"))
    
    def test_multiple_pairs(self):
        """Multiple pairs in sequence"""
        self.assertTrue(isValid("()[]{}"))
        self.assertTrue(isValid("()[]{()}"))
    
    def test_nested(self):
        """Nested brackets"""
        self.assertTrue(isValid("{[]}"))
        self.assertTrue(isValid("{" + "{}}"))  # Escaped for Jekyll
        self.assertTrue(isValid("([{}])"))
    
    def test_wrong_type(self):
        """Mismatched bracket types"""
        self.assertFalse(isValid("(]"))
        self.assertFalse(isValid("{)"))
        self.assertFalse(isValid("[}"))
    
    def test_wrong_order(self):
        """Wrong closing order"""
        self.assertFalse(isValid("([)]"))
        self.assertFalse(isValid("{[}]"))
    
    def test_unclosed(self):
        """Unclosed opening brackets"""
        self.assertFalse(isValid("(("))
        self.assertFalse(isValid("{[("))
    
    def test_extra_closing(self):
        """Extra closing brackets"""
        self.assertFalse(isValid("))"))
        self.assertFalse(isValid("())"))
    
    def test_deeply_nested(self):
        """Deep nesting"""
        s = "(" * 1000 + ")" * 1000
        self.assertTrue(isValid(s))
    
    def test_alternating(self):
        """Alternating pattern"""
        s = "()" * 1000
        self.assertTrue(isValid(s))
    
    def test_complex_valid(self):
        """Complex valid cases"""
        self.assertTrue(isValid("{[()()]}"))
        self.assertTrue(isValid("([]){}"))
        self.assertTrue(isValid("{[({})]}"))
    
    def test_complex_invalid(self):
        """Complex invalid cases"""
        self.assertFalse(isValid("((((()"))
        self.assertFalse(isValid("(((()))"))
        self.assertFalse(isValid("{[(])}"))

if __name__ == '__main__':
    unittest.main()
```

### Performance Benchmarking

```python
import time
import random

def benchmark(func, test_cases):
    """Benchmark function performance"""
    start = time.time()
    for test in test_cases:
        func(test)
    elapsed = time.time() - start
    return elapsed

# Generate test cases
def generate_valid_string(length):
    """Generate valid bracket string"""
    s = ""
    for _ in range(length // 2):
        s += "("
    for _ in range(length // 2):
        s += ")"
    return s

def generate_invalid_string(length):
    """Generate invalid bracket string"""
    brackets = "()[]{}"
    return ''.join(random.choice(brackets) for _ in range(length))

# Test cases
test_cases = [
    generate_valid_string(100) for _ in range(1000)
] + [
    generate_invalid_string(100) for _ in range(1000)
]

# Benchmark
time_stack = benchmark(isValid, test_cases)
print(f"Stack solution: {time_stack:.3f}s")

# Expected: ~0.02s for 2000 strings of length 100
```

---

## Key Takeaways

✅ **Stacks naturally solve LIFO problems** (brackets, function calls, undo)  
✅ **O(n) single-pass solution** is optimal for validation  
✅ **Hash map for pairs** makes code clean and extensible  
✅ **Pattern applies widely** in compilers, parsers, editors, ML pipelines  
✅ **Early exit optimizations** improve average-case performance  
✅ **Consider edge cases** (empty, single char, deeply nested)

---

## Related Problems

**LeetCode:**
- [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) (This problem)
- [22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)
- [32. Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/)
- [301. Remove Invalid Parentheses](https://leetcode.com/problems/remove-invalid-parentheses/)
- [1021. Remove Outermost Parentheses](https://leetcode.com/problems/remove-outermost-parentheses/)

**Stack Problems:**
- [155. Min Stack](https://leetcode.com/problems/min-stack/)
- [232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)
- [394. Decode String](https://leetcode.com/problems/decode-string/)
- [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

---

## Further Reading

**Books:**
- *Introduction to Algorithms* (CLRS) - Chapter 10: Elementary Data Structures
- *The Algorithm Design Manual* (Skiena) - Section 3.2: Stacks and Queues
- *Data Structures and Algorithm Analysis* (Weiss) - Chapter 3

**Articles:**
- [Understanding Stacks in Depth](https://en.wikipedia.org/wiki/Stack_(abstract_data_type))
- [Bracket Matching Algorithm](https://brilliant.org/wiki/stacks-queues/)

---

## Conclusion

The Valid Parentheses problem beautifully demonstrates how the right data structure makes a seemingly complex problem trivial. The stack's LIFO property is a perfect match for nested structures, eliminating the need for complex bookkeeping or multiple passes.

Beyond the specific problem, understanding stacks prepares you for:
- **Parsing and compilation** (expression evaluation, syntax analysis)
- **Backtracking algorithms** (DFS, path finding)
- **Memory management** (call stack, activation records)
- **Undo/redo systems** (editors, version control)

The patterns you've learned here—using stacks for matching, validation, and tracking nested structures—will appear repeatedly in system design, algorithm implementation, and production code.

Master the stack, and you've mastered a fundamental building block of computer science! 🚀

---

**Originally published at:** [arunbaby.com/dsa/0002-valid-parentheses](https://www.arunbaby.com/dsa/0002-valid-parentheses/)

*If you found this helpful, consider sharing it with others who might benefit.*
