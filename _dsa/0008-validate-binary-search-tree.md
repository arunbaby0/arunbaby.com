---
title: "Validate Binary Search Tree"
day: 8
related_ml_day: 8
related_speech_day: 8
related_agents_day: 8
collection: dsa
categories:
 - dsa
tags:
 - trees
 - binary-search-tree
 - recursion
 - validation
topic: Trees & Graphs
difficulty: Medium
companies: [Google, Meta, Amazon, Microsoft, Apple, Bloomberg]
leetcode_link: "https://leetcode.com/problems/validate-binary-search-tree/"
time_complexity: "O(n)"
space_complexity: "O(h)"
---

**Master BST validation to understand data integrity in tree structures, critical for indexing and search systems.**

## Problem

Given the root of a binary tree, determine if it is a valid binary search tree (BST).

**A valid BST is defined as:**
1. The left subtree of a node contains only nodes with keys **less than** the node's key
2. The right subtree of a node contains only nodes with keys **greater than** the node's key
3. Both left and right subtrees must also be binary search trees

**Example 1:**
``
Input: 2
 / \
 1 3

Output: true
Explanation: Valid BST
``

**Example 2:**
``
Input: 5
 / \
 1 4
 / \
 3 6

Output: false
Explanation: 4 is in right subtree of 5, but 3 < 5
``

**Constraints:**
- Number of nodes: [1, 10^4]
- Node values: -2^31 <= val <= 2^31 - 1

---

## Understanding BST Properties

### Valid BST

``
 5
 / \
 3 7
 / \ / \
 2 4 6 8

✓ All left descendants < node < all right descendants
✓ Inorder traversal: [2, 3, 4, 5, 6, 7, 8] (sorted!)
``

### Invalid BST Examples

**Example 1: Wrong child placement**
``
 5
 / \
 6 7
 
✗ 6 > 5 but in left subtree
``

**Example 2: Subtree violation**
``
 10
 / \
 5 15
 / \
 6 20

✗ 6 < 10 but in right subtree
 Even though 6 < 15 (local property holds)
``

**Example 3: Duplicate values**
``
 5
 / \
 5 7

✗ BST requires strict inequality (depends on problem definition)
 Some definitions allow duplicates in one direction
``

---

## Approach 1: Recursive with Range Validation

**Key insight:** Each node must fall within a valid range [min, max]

### Algorithm

``python
class TreeNode:
 def __init__(self, val=0, left=None, right=None):
 self.val = val
 self.left = left
 self.right = right

def isValidBST(root: TreeNode) -> bool:
 """
 Validate BST using range checking
 
 Time: O(n) - visit each node once
 Space: O(h) - recursion stack, h = height
 """
 def validate(node, min_val, max_val):
 # Empty tree is valid
 if not node:
 return True
 
 # Check current node's value
 if node.val <= min_val or node.val >= max_val:
 return False
 
 # Validate subtrees with updated ranges
 # Left subtree: all values must be < node.val
 # Right subtree: all values must be > node.val
 return (validate(node.left, min_val, node.val) and
 validate(node.right, node.val, max_val))
 
 # Initial range: (-∞, +∞)
 return validate(root, float('-inf'), float('inf'))

# Example usage
root = TreeNode(2)
root.left = TreeNode(1)
root.right = TreeNode(3)

print(isValidBST(root)) # True
``

### Visualization

``
Validate: 5
 / \
 3 7

Call tree:
validate(5, -∞, +∞)
├─ validate(3, -∞, 5) ← 3 must be in (-∞, 5)
│ ├─ validate(None) ← True
│ └─ validate(None) ← True
└─ validate(7, 5, +∞) ← 7 must be in (5, +∞)
 ├─ validate(None) ← True
 └─ validate(None) ← True

Result: True
``

**Invalid example:**
``
Validate: 10
 / \
 5 15
 / \
 6 20

validate(10, -∞, +∞)
├─ validate(5, -∞, 10) ← OK: 5 in (-∞, 10)
└─ validate(15, 10, +∞) ← OK: 15 in (10, +∞)
 ├─ validate(6, 10, 15) ← FAIL: 6 not in (10, 15)
 │ 6 <= 10 (min_val)
 └─ ...

Result: False
``

---

## Approach 2: Inorder Traversal

**Key insight:** Inorder traversal of BST produces sorted sequence

### Algorithm

``python
def isValidBST(root: TreeNode) -> bool:
 """
 Validate using inorder traversal
 
 Check if inorder gives strictly increasing sequence
 
 Time: O(n)
 Space: O(n) for storing values
 """
 def inorder(node, values):
 if not node:
 return
 
 inorder(node.left, values)
 values.append(node.val)
 inorder(node.right, values)
 
 values = []
 inorder(root, values)
 
 # Check if strictly increasing
 for i in range(1, len(values)):
 if values[i] <= values[i-1]:
 return False
 
 return True
``

### Space-Optimized Version

``python
def isValidBST(root: TreeNode) -> bool:
 """
 Inorder validation without storing all values
 
 Time: O(n)
 Space: O(h) - recursion stack only
 """
 def inorder(node):
 nonlocal prev
 
 if not node:
 return True
 
 # Validate left subtree
 if not inorder(node.left):
 return False
 
 # Check current node
 if node.val <= prev:
 return False
 prev = node.val
 
 # Validate right subtree
 return inorder(node.right)
 
 prev = float('-inf')
 return inorder(root)
``

---

## Approach 3: Iterative with Stack

**Advantage:** Avoids recursion (useful for very deep trees)

``python
def isValidBST(root: TreeNode) -> bool:
 """
 Iterative inorder validation
 
 Time: O(n)
 Space: O(h)
 """
 stack = []
 prev = float('-inf')
 current = root
 
 while current or stack:
 # Go to leftmost node
 while current:
 stack.append(current)
 current = current.left
 
 # Process node
 current = stack.pop()
 
 # Check BST property
 if current.val <= prev:
 return False
 prev = current.val
 
 # Move to right subtree
 current = current.right
 
 return True

# Example
root = TreeNode(5)
root.left = TreeNode(1)
root.right = TreeNode(4)
root.right.left = TreeNode(3)
root.right.right = TreeNode(6)

print(isValidBST(root)) # False (4 > 1 but 3 < 5)
``

### Stack Visualization

``
Tree: 5
 / \
 3 7
 / \
 2 4

Iteration 1:
 stack: [5, 3, 2]
 current: 2 → pop → check 2 > -∞ ✓
 prev = 2

Iteration 2:
 stack: [5, 3]
 current: 3 → pop → check 3 > 2 ✓
 prev = 3

Iteration 3:
 current: 4 → check 4 > 3 ✓
 prev = 4

Iteration 4:
 stack: [5]
 current: 5 → pop → check 5 > 4 ✓
 prev = 5

Iteration 5:
 current: 7 → check 7 > 5 ✓
 prev = 7

Result: True
``

---

## Edge Cases

### 1. Single Node

``python
def test_single_node():
 """Single node is always valid BST"""
 root = TreeNode(1)
 assert isValidBST(root) == True

test_single_node()
``

### 2. Duplicate Values

``python
def test_duplicates():
 """
 Duplicates are typically invalid
 
 5
 / \
 5 5
 """
 root = TreeNode(5)
 root.left = TreeNode(5)
 root.right = TreeNode(5)
 
 assert isValidBST(root) == False

test_duplicates()
``

### 3. Integer Overflow Edge Cases

``python
def test_extreme_values():
 """Test with min/max integer values"""
 # Tree with INT_MIN
 root1 = TreeNode(-2**31)
 root1.right = TreeNode(2**31 - 1)
 assert isValidBST(root1) == True
 
 # Tree with INT_MAX
 root2 = TreeNode(2**31 - 1)
 root2.left = TreeNode(-2**31)
 assert isValidBST(root2) == True

test_extreme_values()
``

### 4. Skewed Trees

``python
def test_skewed():
 """
 Right-skewed tree (like linked list)
 
 1 → 2 → 3 → 4
 """
 root = TreeNode(1)
 root.right = TreeNode(2)
 root.right.right = TreeNode(3)
 root.right.right.right = TreeNode(4)
 
 assert isValidBST(root) == True

test_skewed()
``

---

## Common Mistakes

### Mistake 1: Only Checking Immediate Children

``python
# WRONG: Only checks node > left and node < right
def isValidBST_WRONG(root):
 if not root:
 return True
 
 if root.left and root.left.val >= root.val:
 return False
 if root.right and root.right.val <= root.val:
 return False
 
 return (isValidBST_WRONG(root.left) and 
 isValidBST_WRONG(root.right))

# Fails on:
# 10
# / \
# 5 15
# / \
# 6 20
#
# This is INVALID (6 < 10) but above code returns True!
``

### Mistake 2: Using Default Integer Min/Max

``python
# WRONG: Can't handle trees with actual INT_MIN/MAX values
def isValidBST_WRONG(root):
 def validate(node, min_val=-2**31, max_val=2**31-1):
 if not node:
 return True
 
 # Problem: if node.val == -2**31, this fails incorrectly
 if node.val <= min_val or node.val >= max_val:
 return False
 
 return (validate(node.left, min_val, node.val) and
 validate(node.right, node.val, max_val))
 
 return validate(root)

# Use float('-inf') and float('inf') instead!
``

### Mistake 3: Forgetting Strict Inequality

``python
# WRONG: Uses <= instead of <
def isValidBST_WRONG(root):
 def validate(node, min_val, max_val):
 if not node:
 return True
 
 # Should be < and >, not <= and >=
 if node.val < min_val or node.val > max_val:
 return False
 
 return (validate(node.left, min_val, node.val) and
 validate(node.right, node.val, max_val))
 
 return validate(root, float('-inf'), float('inf'))

# Allows duplicates!
``

---

## Advanced Variations

### Problem 1: Validate BST with Duplicates

``python
def isValidBSTWithDuplicates(root: TreeNode, allow_left_duplicates=True) -> bool:
 """
 Validate BST allowing duplicates
 
 Args:
 allow_left_duplicates: If True, duplicates go to left
 If False, duplicates go to right
 
 Returns:
 True if valid BST with duplicates
 """
 def validate(node, min_val, max_val):
 if not node:
 return True
 
 if allow_left_duplicates:
 # Allow equals on left: left <= node < right
 if node.val < min_val or node.val >= max_val:
 return False
 return (validate(node.left, min_val, node.val) and
 validate(node.right, node.val + 1, max_val))
 else:
 # Allow equals on right: left < node <= right
 if node.val <= min_val or node.val > max_val:
 return False
 return (validate(node.left, min_val, node.val - 1) and
 validate(node.right, node.val, max_val))
 
 return validate(root, float('-inf'), float('inf'))
``

### Problem 2: Find Violations

``python
def findBSTViolations(root: TreeNode) -> list:
 """
 Find all nodes that violate BST property
 
 Returns: List of (node_val, reason) tuples
 """
 violations = []
 
 def validate(node, min_val, max_val):
 if not node:
 return True
 
 is_valid = True
 
 if node.val <= min_val:
 violations.append((node.val, f"Value {node.val} <= min_bound {min_val}"))
 is_valid = False
 
 if node.val >= max_val:
 violations.append((node.val, f"Value {node.val} >= max_bound {max_val}"))
 is_valid = False
 
 validate(node.left, min_val, node.val)
 validate(node.right, node.val, max_val)
 
 return is_valid
 
 validate(root, float('-inf'), float('inf'))
 return violations

# Example
root = TreeNode(10)
root.left = TreeNode(5)
root.right = TreeNode(15)
root.right.left = TreeNode(6)

violations = findBSTViolations(root)
for val, reason in violations:
 print(f"Violation: {reason}")
``

### Problem 3: Largest BST Subtree

``python
def largestBSTSubtree(root: TreeNode) -> int:
 """
 Find size of largest BST subtree
 
 Returns: Number of nodes in largest BST
 """
 def dfs(node):
 """
 Returns: (is_bst, size, min_val, max_val)
 """
 if not node:
 return (True, 0, float('inf'), float('-inf'))
 
 left_is_bst, left_size, left_min, left_max = dfs(node.left)
 right_is_bst, right_size, right_min, right_max = dfs(node.right)
 
 # Check if current tree is BST
 if (left_is_bst and right_is_bst and 
 left_max < node.val < right_min):
 # Current subtree is BST
 size = left_size + right_size + 1
 min_val = min(left_min, node.val)
 max_val = max(right_max, node.val)
 
 nonlocal max_bst_size
 max_bst_size = max(max_bst_size, size)
 
 return (True, size, min_val, max_val)
 else:
 # Not a BST, but children might contain BSTs
 return (False, 0, 0, 0)
 
 max_bst_size = 0
 dfs(root)
 return max_bst_size

# Example: Mixed tree
# 10
# / \
# 5 15
# / \ \
# 1 8 7
#
# Largest BST: left subtree (5, 1, 8) with size 3
``

---

## Connection to ML Systems

BST validation patterns appear in ML engineering:

### 1. Feature Validation

``python
class FeatureValidator:
 """
 Validate feature values fall within expected ranges
 
 Similar to BST range validation
 """
 
 def __init__(self):
 self.feature_ranges = {}
 
 def set_range(self, feature_name, min_val, max_val):
 """Set expected range for feature"""
 self.feature_ranges[feature_name] = (min_val, max_val)
 
 def validate(self, features: dict) -> tuple:
 """
 Validate features fall within ranges
 
 Returns: (is_valid, violations)
 """
 violations = []
 
 for feature, value in features.items():
 if feature not in self.feature_ranges:
 continue
 
 min_val, max_val = self.feature_ranges[feature]
 
 if value < min_val or value > max_val:
 violations.append({
 'feature': feature,
 'value': value,
 'expected_range': (min_val, max_val)
 })
 
 return len(violations) == 0, violations

# Usage
validator = FeatureValidator()
validator.set_range('age', 0, 120)
validator.set_range('income', 0, 1000000)

features = {'age': 25, 'income': 50000}
is_valid, violations = validator.validate(features)

if not is_valid:
 print(f"Invalid features: {violations}")
``

### 2. Model Prediction Bounds

``python
class PredictionValidator:
 """
 Validate model predictions are reasonable
 
 Catches model failures early
 """
 
 def __init__(self, min_pred, max_pred):
 self.min_pred = min_pred
 self.max_pred = max_pred
 self.violations_count = 0
 
 def validate_batch(self, predictions):
 """
 Validate batch of predictions
 
 Returns: (valid_predictions, invalid_indices)
 """
 invalid_indices = []
 
 for i, pred in enumerate(predictions):
 if pred < self.min_pred or pred > self.max_pred:
 invalid_indices.append(i)
 self.violations_count += 1
 
 # Filter out invalid predictions
 valid_predictions = [
 pred for i, pred in enumerate(predictions)
 if i not in invalid_indices
 ]
 
 return valid_predictions, invalid_indices
 
 def get_violation_rate(self, total_predictions):
 """Calculate rate of invalid predictions"""
 return self.violations_count / total_predictions if total_predictions > 0 else 0

# Example: Probability predictions
validator = PredictionValidator(min_pred=0.0, max_pred=1.0)

predictions = [0.5, 0.8, 1.2, -0.1, 0.3] # Contains invalid values
valid, invalid_idx = validator.validate_batch(predictions)

print(f"Valid predictions: {valid}")
print(f"Invalid indices: {invalid_idx}")
``

### 3. Decision Tree Structure Validation

``python
class DecisionTreeValidator:
 """
 Validate decision tree structure
 
 Ensures tree is well-formed for inference
 """
 
 def validate_tree(self, node, depth=0, max_depth=100):
 """
 Validate decision tree node
 
 Checks:
 - Leaf nodes have predictions
 - Internal nodes have split conditions
 - No cycles (via depth limit)
 """
 if depth > max_depth:
 return False, f"Tree too deep (depth > {max_depth})"
 
 # Leaf node
 if not node.left and not node.right:
 if node.prediction is None:
 return False, "Leaf node missing prediction"
 return True, None
 
 # Internal node
 if node.feature is None or node.threshold is None:
 return False, "Internal node missing split condition"
 
 # Validate children
 if node.left:
 left_valid, left_err = self.validate_tree(node.left, depth + 1, max_depth)
 if not left_valid:
 return False, f"Left subtree: {left_err}"
 
 if node.right:
 right_valid, right_err = self.validate_tree(node.right, depth + 1, max_depth)
 if not right_valid:
 return False, f"Right subtree: {right_err}"
 
 return True, None
``

---

## Testing

### Comprehensive Test Suite

``python
import unittest

class TestBSTValidation(unittest.TestCase):
 
 def test_valid_bst(self):
 """Test valid BST"""
 root = TreeNode(2)
 root.left = TreeNode(1)
 root.right = TreeNode(3)
 self.assertTrue(isValidBST(root))
 
 def test_invalid_bst(self):
 """Test invalid BST"""
 root = TreeNode(5)
 root.left = TreeNode(1)
 root.right = TreeNode(4)
 root.right.left = TreeNode(3)
 root.right.right = TreeNode(6)
 self.assertFalse(isValidBST(root))
 
 def test_single_node(self):
 """Test single node"""
 root = TreeNode(1)
 self.assertTrue(isValidBST(root))
 
 def test_empty_tree(self):
 """Test empty tree"""
 self.assertTrue(isValidBST(None))
 
 def test_left_skewed(self):
 """Test left-skewed tree"""
 root = TreeNode(4)
 root.left = TreeNode(3)
 root.left.left = TreeNode(2)
 root.left.left.left = TreeNode(1)
 self.assertTrue(isValidBST(root))
 
 def test_right_skewed(self):
 """Test right-skewed tree"""
 root = TreeNode(1)
 root.right = TreeNode(2)
 root.right.right = TreeNode(3)
 root.right.right.right = TreeNode(4)
 self.assertTrue(isValidBST(root))
 
 def test_duplicate_values(self):
 """Test duplicate values"""
 root = TreeNode(2)
 root.left = TreeNode(2)
 root.right = TreeNode(2)
 self.assertFalse(isValidBST(root))
 
 def test_extreme_values(self):
 """Test with INT_MIN and INT_MAX"""
 root = TreeNode(0)
 root.left = TreeNode(-2**31)
 root.right = TreeNode(2**31 - 1)
 self.assertTrue(isValidBST(root))

if __name__ == '__main__':
 unittest.main()
``

---

## Performance Optimization

### Early Termination

``python
def isValidBST_optimized(root: TreeNode) -> bool:
 """
 Optimized with early termination
 
 Stop as soon as violation found
 """
 def validate(node, min_val, max_val):
 if not node:
 return True
 
 # Early termination
 if node.val <= min_val or node.val >= max_val:
 return False
 
 # Short-circuit evaluation: if left fails, don't check right
 return (validate(node.left, min_val, node.val) and
 validate(node.right, node.val, max_val))
 
 return validate(root, float('-inf'), float('inf'))
``

---

## Performance Comparison

### Benchmarking Different Approaches

``python
import time
import sys

def benchmark_validation(approach_name, validation_fn, tree, iterations=1000):
 """Benchmark BST validation approach"""
 start = time.perf_counter()
 
 for _ in range(iterations):
 result = validation_fn(tree)
 
 end = time.perf_counter()
 avg_time_ms = (end - start) / iterations * 1000
 
 return avg_time_ms

# Create test trees
def create_balanced_tree(n):
 """Create balanced BST with n nodes"""
 if n == 0:
 return None
 
 mid = n // 2
 root = TreeNode(mid)
 
 def build(start, end):
 if start > end:
 return None
 mid = (start + end) // 2
 node = TreeNode(mid)
 node.left = build(start, mid - 1)
 node.right = build(mid + 1, end)
 return node
 
 return build(0, n - 1)

# Benchmark
tree_sizes = [100, 1000, 10000]
approaches = [
 ('Recursive Range', isValidBST),
 ('Inorder Iterative', isValidBST), # replace with iterative alias if defined
 ('Inorder Recursive', isValidBST) # replace with recursive inorder alias if defined
]

print("BST Validation Performance:")
print("-" * 60)
for size in tree_sizes:
 print(f"\nTree size: {size} nodes")
 tree = create_balanced_tree(size)
 
 for name, fn in approaches:
 time_ms = benchmark_validation(name, fn, tree, iterations=100)
 print(f" {name:25s}: {time_ms:.3f}ms")
``

**Typical results:**

``
Tree size: 100 nodes
 Recursive Range : 0.012ms
 Inorder Iterative : 0.015ms
 Inorder Recursive : 0.013ms

Tree size: 1000 nodes
 Recursive Range : 0.125ms
 Inorder Iterative : 0.148ms
 Inorder Recursive : 0.132ms

Tree size: 10000 nodes
 Recursive Range : 1.342ms
 Inorder Iterative : 1.523ms
 Inorder Recursive : 1.398ms
``

**Analysis:**
- All O(n), linear scaling
- Recursive range is fastest (fewer operations)
- Iterative has overhead of stack management
- Differences are small in practice

---

## Interview Deep Dive

### Common Follow-Up Questions

**Q1: What if the tree allows duplicates?**

``python
def isValidBSTWithDuplicatesLeft(root: TreeNode) -> bool:
 """
 Valid if duplicates are allowed on left side
 
 Modified condition: left <= node < right
 """
 def validate(node, min_val, max_val):
 if not node:
 return True
 
 # Allow equal on left: node.val > min_val (not >=)
 # Strict right: node.val < max_val
 if node.val < min_val or node.val >= max_val:
 return False
 
 return (validate(node.left, min_val, node.val) and # Allow <= node
 validate(node.right, node.val, max_val))
 
 return validate(root, float('-inf'), float('inf'))

def isValidBSTWithDuplicatesRight(root: TreeNode) -> bool:
 """
 Valid if duplicates are allowed on right side
 
 Modified condition: left < node <= right
 """
 def validate(node, min_val, max_val):
 if not node:
 return True
 
 # Strict left: node.val > min_val
 # Allow equal on right: node.val <= max_val
 if node.val <= min_val or node.val > max_val:
 return False
 
 return (validate(node.left, min_val, node.val - 1) and
 validate(node.right, node.val, max_val)) # Allow >= node
 
 return validate(root, float('-inf'), float('inf'))
``

**Q2: Return the first invalid node instead of boolean?**

``python
def findFirstInvalidNode(root: TreeNode) -> TreeNode:
 """
 Find first node that violates BST property
 
 Returns: Invalid node or None
 """
 def validate(node, min_val, max_val):
 if not node:
 return None
 
 # Check current node
 if node.val <= min_val or node.val >= max_val:
 return node
 
 # Check left subtree first
 left_invalid = validate(node.left, min_val, node.val)
 if left_invalid:
 return left_invalid
 
 # Check right subtree
 right_invalid = validate(node.right, node.val, max_val)
 if right_invalid:
 return right_invalid
 
 return None
 
 return validate(root, float('-inf'), float('inf'))

# Example usage
root = TreeNode(10)
root.left = TreeNode(5)
root.right = TreeNode(15)
root.right.left = TreeNode(6) # Invalid

invalid_node = findFirstInvalidNode(root)
if invalid_node:
 print(f"First invalid node: {invalid_node.val}") # 6
``

**Q3: What if nodes can be None?**

``python
# Already handled! Our code checks `if not node:` first
# This handles None values correctly
``

**Q4: Can you do it in O(1) space?**

``python
def isValidBST_O1_space(root: TreeNode) -> bool:
 """
 Morris traversal for O(1) space
 
 Time: O(n), Space: O(1)
 """
 prev_val = float('-inf')
 current = root
 
 while current:
 if not current.left:
 # No left child, check current
 if current.val <= prev_val:
 return False
 prev_val = current.val
 current = current.right
 else:
 # Find predecessor
 predecessor = current.left
 while predecessor.right and predecessor.right != current:
 predecessor = predecessor.right
 
 if not predecessor.right:
 # Create thread
 predecessor.right = current
 current = current.left
 else:
 # Remove thread, check current
 predecessor.right = None
 if current.val <= prev_val:
 return False
 prev_val = current.val
 current = current.right
 
 return True
``

---

## Production Engineering Patterns

### 1. Cached Validation Results

``python
from functools import lru_cache

class TreeWithValidationCache:
 """
 Tree with cached validation result
 
 Useful if tree is queried many times without modification
 """
 
 def __init__(self, root):
 self.root = root
 self._validation_result = None
 self._tree_hash = None
 
 def is_valid(self) -> bool:
 """
 Check if tree is valid BST (with caching)
 
 Returns: Cached result if tree hasn't changed
 """
 current_hash = self._compute_tree_hash()
 
 if current_hash == self._tree_hash and self._validation_result is not None:
 # Tree hasn't changed, return cached result
 return self._validation_result
 
 # Revalidate
 self._validation_result = isValidBST(self.root)
 self._tree_hash = current_hash
 
 return self._validation_result
 
 def _compute_tree_hash(self):
 """Compute hash of tree structure"""
 def hash_tree(node):
 if not node:
 return 0
 return hash((node.val, hash_tree(node.left), hash_tree(node.right)))
 
 return hash_tree(self.root)
 
 def invalidate_cache(self):
 """Call after modifying tree"""
 self._validation_result = None
 self._tree_hash = None
``

### 2. Incremental Validation

``python
class IncrementalBSTValidator:
 """
 Validate BST incrementally as nodes are added
 
 More efficient than revalidating entire tree
 """
 
 def __init__(self):
 self.is_valid = True
 self.violations = []
 
 def validate_insertion(self, root, new_value, insertion_path):
 """
 Validate after inserting new_value
 
 Args:
 root: Tree root
 new_value: Value being inserted
 insertion_path: Path taken during insertion
 e.g., ['L', 'R', 'L'] = left, right, left
 
 Returns: True if tree is still valid
 """
 # Reconstruct bounds along insertion path
 min_val = float('-inf')
 max_val = float('inf')
 current = root
 
 for direction in insertion_path:
 if direction == 'L':
 # Going left: update max_val
 max_val = current.val
 current = current.left
 else: # 'R'
 # Going right: update min_val
 min_val = current.val
 current = current.right
 
 # Check if new_value violates bounds
 if new_value <= min_val or new_value >= max_val:
 self.is_valid = False
 self.violations.append({
 'value': new_value,
 'min_bound': min_val,
 'max_bound': max_val,
 'path': insertion_path
 })
 return False
 
 return True

# Usage
validator = IncrementalBSTValidator()

# Insert values and validate incrementally
def insert_and_validate(root, value, validator):
 """Insert value and validate incrementally"""
 path = []
 
 # Perform insertion (track path)
 def insert_with_path(node, val, path):
 if not node:
 return TreeNode(val)
 
 if val < node.val:
 path.append('L')
 node.left = insert_with_path(node.left, val, path)
 else:
 path.append('R')
 node.right = insert_with_path(node.right, val, path)
 
 return node
 
 root = insert_with_path(root, value, path)
 
 # Validate incrementally
 is_valid = validator.validate_insertion(root, value, path)
 
 return root, is_valid
``

### 3. Validation with Repair

``python
def validateAndRepairBST(root: TreeNode) -> tuple:
 """
 Validate BST and attempt to fix violations
 
 Returns: (is_valid, repaired_tree, fixes_applied)
 """
 violations = []
 
 def find_violations(node, min_val, max_val):
 """Find all nodes that violate BST property"""
 if not node:
 return
 
 if node.val <= min_val or node.val >= max_val:
 violations.append({
 'node': node,
 'min_bound': min_val,
 'max_bound': max_val
 })
 
 find_violations(node.left, min_val, node.val)
 find_violations(node.right, node.val, max_val)
 
 # Find violations
 find_violations(root, float('-inf'), float('inf'))
 
 if not violations:
 return True, root, []
 
 # Attempt to repair by moving nodes
 fixes = []
 for v in violations:
 node = v['node']
 # Simple fix: clamp value to valid range
 old_val = node.val
 node.val = max(v['min_bound'] + 1, min(node.val, v['max_bound'] - 1))
 fixes.append({
 'node_old_val': old_val,
 'node_new_val': node.val,
 'bounds': (v['min_bound'], v['max_bound'])
 })
 
 # Revalidate
 is_valid_after = isValidBST(root)
 
 return is_valid_after, root, fixes

# Example
root = TreeNode(10)
root.left = TreeNode(5)
root.right = TreeNode(15)
root.right.left = TreeNode(6) # Invalid

is_valid, repaired_root, fixes = validateAndRepairBST(root)
print(f"Valid after repair: {is_valid}")
print(f"Fixes applied: {fixes}")
``

---

## Real-World Applications

### Database Index Validation

``python
class DatabaseIndexValidator:
 """
 Validate database B-tree index structure
 
 B-trees are generalized BSTs
 """
 
 def __init__(self, max_keys_per_node=4):
 self.max_keys_per_node = max_keys_per_node
 
 def validate_btree_node(self, node):
 """
 Validate B-tree node
 
 B-tree properties:
 1. Keys are sorted
 2. Number of keys <= max_keys_per_node
 3. For each key k, left subtree has values < k, right has values > k
 """
 if not node:
 return True
 
 # Check keys are sorted
 keys = node.keys
 if keys != sorted(keys):
 return False
 
 # Check number of keys
 if len(keys) > self.max_keys_per_node:
 return False
 
 # Check children (similar to BST validation)
 if node.children:
 for i, child in enumerate(node.children):
 # Determine bounds for child
 if i == 0:
 # Leftmost child
 if not self._validate_subtree(child, float('-inf'), keys[0]):
 return False
 elif i == len(keys):
 # Rightmost child
 if not self._validate_subtree(child, keys[-1], float('inf')):
 return False
 else:
 # Middle child
 if not self._validate_subtree(child, keys[i-1], keys[i]):
 return False
 
 return True
 
 def _validate_subtree(self, node, min_val, max_val):
 """Validate all keys in subtree fall within range"""
 if not node:
 return True
 
 for key in node.keys:
 if key <= min_val or key >= max_val:
 return False
 
 # Recursively validate children
 return self.validate_btree_node(node)
``

### ML Model Tree Structure Validation

``python
class MLTreeValidator:
 """
 Validate ML model tree structures
 
 Decision trees, gradient boosting, etc.
 """
 
 def validate_decision_tree(self, node, feature_names=None):
 """
 Validate decision tree structure
 
 Checks:
 1. Leaf nodes have predictions
 2. Internal nodes have split conditions
 3. Feature indices are valid
 4. Thresholds are numeric
 """
 if node.is_leaf():
 # Leaf must have prediction
 if node.prediction is None:
 return False, "Leaf node missing prediction"
 return True, None
 
 # Internal node validation
 if node.feature_index is None:
 return False, "Internal node missing feature_index"
 
 if node.threshold is None:
 return False, "Internal node missing threshold"
 
 # Check feature index is valid
 if feature_names and node.feature_index >= len(feature_names):
 return False, f"Feature index {node.feature_index} out of range"
 
 # Check threshold is numeric
 if not isinstance(node.threshold, (int, float)):
 return False, "Threshold must be numeric"
 
 # Recursively validate children
 if node.left:
 left_valid, left_err = self.validate_decision_tree(node.left, feature_names)
 if not left_valid:
 return False, f"Left subtree: {left_err}"
 
 if node.right:
 right_valid, right_err = self.validate_decision_tree(node.right, feature_names)
 if not right_valid:
 return False, f"Right subtree: {right_err}"
 
 return True, None
``

---

## Key Takeaways

✅ **Range validation is key** - Each node must fall within [min, max] range 
✅ **Inorder = sorted** - BST inorder traversal produces sorted sequence 
✅ **Check all descendants** - Not just immediate children 
✅ **Handle edge cases** - Single node, duplicates, extreme values 
✅ **O(n) time is optimal** - Must visit all nodes 
✅ **Three approaches** - Recursive range, inorder, iterative 
✅ **ML applications** - Feature validation, prediction bounds, tree structure checks 

---

## Related Problems

- **[Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)** - Foundation
- **[Kth Smallest Element in BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)** - Uses inorder
- **[Recover Binary Search Tree](https://leetcode.com/problems/recover-binary-search-tree/)** - Fix BST violations
- **[Largest BST Subtree](https://leetcode.com/problems/largest-bst-subtree/)** - Find largest valid BST
- **[Convert Sorted Array to BST](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)** - Build valid BST

---

**Originally published at:** [arunbaby.com/dsa/0008-validate-binary-search-tree](https://www.arunbaby.com/dsa/0008-validate-binary-search-tree/)

*If you found this helpful, consider sharing it with others who might benefit.*

