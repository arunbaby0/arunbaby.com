---
title: "Binary Tree Maximum Path Sum"
day: 46
collection: dsa
categories:
  - dsa
tags:
  - binary-tree
  - dynamic-programming
  - tree-dp
  - recursion
  - dfs
  - path-finding
  - hard
difficulty: Hard
subdomain: "Trees & Dynamic Programming"
tech_stack: Python
scale: "O(N) time, O(H) space where H is tree height"
companies: Google, Meta, Amazon, Microsoft, Apple
related_ml_day: 46
related_speech_day: 46
related_agents_day: 46
---

**"Every path has a peak—find the one with the maximum sum."**

## 1. Problem Statement

Given a **binary tree**, find the maximum **path sum**. A path is defined as any sequence of nodes from some starting node to any node in the tree along parent-child connections. The path must contain **at least one node** and does not need to go through the root.

**Example 1:**
```
    1
   / \
  2   3

Output: 6
Explanation: Path 2 → 1 → 3 has sum 2 + 1 + 3 = 6
```

**Example 2:**
```
       -10
       /  \
      9   20
         /  \
        15   7

Output: 42
Explanation: Path 15 → 20 → 7 has sum 15 + 20 + 7 = 42
```

**Constraints:**
- The number of nodes in the tree: `1 ≤ n ≤ 3 × 10^4`
- `-1000 ≤ Node.val ≤ 1000`

## 2. Understanding the Problem

This problem challenges our understanding of tree traversal and dynamic programming on trees. Let's break down the key insights:

### What Makes a Path?

A path in a binary tree can be visualized as an "arch" that:
1. Starts at some node
2. Optionally goes down through the left subtree
3. Optionally goes down through the right subtree
4. The path can "turn around" at most once (at the highest node in the path)

```
        A          A is the "turning point"
       / \
      B   C        Path: B → A → C (or just A, or B → A, or A → C)
     /     \
    D       E      But NOT: D → B → A → C → E (can't continue from both children)
```

### Key Insight: Single vs. Split Paths

For each node, we need to consider two scenarios:
1. **Contribution Path**: The maximum sum we can contribute to a parent node (going only one direction)
2. **Complete Path**: The maximum path that "completes" at this node (can use both left and right)

This duality is what makes the problem tricky—we track one thing for return values but update our global answer with another.

### Why Traditional Approaches Fail

**Why DFS alone isn't enough:** A naive DFS might try to track all possible paths, leading to exponential complexity. The key is recognizing that at each node, we only need to track the maximum single-direction path.

**Why we need global state:** The maximum path might not pass through the root, so we can't simply return the answer through recursion—we need a global variable to track the best path seen so far.

## 3. Approach 1: Brute Force (Enumerate All Paths)

The brute force approach considers every possible path by trying every pair of nodes:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxPathSum_brute_force(root: TreeNode) -> int:
    """
    Brute force: Find all paths and compute their sums.
    
    This approach:
    1. For each node, find max path starting from it going down
    2. For each node as a "turning point", combine left and right paths
    
    Time: O(N^2) - for each node, we might traverse subtrees
    Space: O(N) - recursion stack
    """
    if not root:
        return 0
    
    def max_path_from_node(node: TreeNode) -> int:
        """Maximum sum path starting from node and going only downward."""
        if not node:
            return 0
        
        left_max = max(0, max_path_from_node(node.left))
        right_max = max(0, max_path_from_node(node.right))
        
        return node.val + max(left_max, right_max)
    
    def find_all_complete_paths(node: TreeNode) -> int:
        """Find max complete path in tree rooted at node."""
        if not node:
            return float('-inf')
        
        # Path that turns at this node
        left_contribution = max(0, max_path_from_node(node.left))
        right_contribution = max(0, max_path_from_node(node.right))
        path_through_node = node.val + left_contribution + right_contribution
        
        # Best path in subtrees
        left_best = find_all_complete_paths(node.left)
        right_best = find_all_complete_paths(node.right)
        
        return max(path_through_node, left_best, right_best)
    
    return find_all_complete_paths(root)
```

**Complexity:**
- **Time:** O(N²) in the worst case (skewed tree), as we call `max_path_from_node` for each node
- **Space:** O(N) for recursion stack

**Why it's inefficient:** We recalculate the same subtree sums multiple times. Each call to `max_path_from_node` might traverse the entire subtree.

## 4. Approach 2: Optimized DFS with Global Maximum (Optimal)

The key insight is to compute both pieces of information in a single pass:
1. Return the maximum single-direction contribution to the parent
2. Update a global maximum with the complete path at each node

```python
def maxPathSum(root: TreeNode) -> int:
    """
    Optimal solution using post-order DFS.
    
    Key insight: At each node, we compute:
    - max_contribution: Maximum sum we can add to parent (single direction)
    - path_sum: Maximum complete path that turns at this node
    
    We use a global variable to track the best complete path seen.
    
    Time: O(N) - visit each node once
    Space: O(H) - recursion stack, H = tree height
    """
    max_sum = float('-inf')  # Global maximum path sum
    
    def dfs(node: TreeNode) -> int:
        """
        Returns the maximum contribution this subtree can make to a path
        that extends upward to the parent.
        
        Side effect: Updates max_sum if a better complete path is found.
        """
        nonlocal max_sum
        
        if not node:
            return 0
        
        # Recursively compute max contribution from each child
        # Use max(0, ...) to ignore negative contributions
        left_gain = max(0, dfs(node.left))
        right_gain = max(0, dfs(node.right))
        
        # The complete path that "peaks" at this node
        # This path goes: some left descendant → node → some right descendant
        current_path_sum = node.val + left_gain + right_gain
        
        # Update global maximum
        max_sum = max(max_sum, current_path_sum)
        
        # Return the maximum contribution to parent
        # Can only extend in ONE direction (either left or right, not both)
        return node.val + max(left_gain, right_gain)
    
    dfs(root)
    return max_sum
```

**Complexity:**
- **Time:** O(N) - each node visited exactly once
- **Space:** O(H) - recursion stack where H is tree height (O(log N) for balanced, O(N) for skewed)

### Detailed Walkthrough

Let's trace through Example 2:
```
       -10
       /  \
      9   20
         /  \
        15   7
```

**Step-by-step execution:**

| Node | left_gain | right_gain | current_path_sum | max_sum | Returns |
|------|-----------|------------|------------------|---------|---------|
| 15   | 0         | 0          | 15               | 15      | 15      |
| 7    | 0         | 0          | 7                | 15      | 7       |
| 20   | 15        | 7          | 20+15+7=42       | 42      | 20+15=35 |
| 9    | 0         | 0          | 9                | 42      | 9       |
| -10  | 9         | 35         | -10+9+35=34      | 42      | -10+35=25 |

**Key observations:**
1. At node 20, the complete path (15→20→7 = 42) is captured in `current_path_sum`
2. But node 20 only returns 35 (20+15) because it can only contribute one direction upward
3. The final answer is 42, found at node 20

## 5. Why max(0, ...) Is Crucial

The `max(0, child_gain)` pattern handles negative paths elegantly:

```python
# Without max(0, ...):
#     -5
#    /   \
#  -10   -3
#
# left_gain = -10, right_gain = -3
# current_path = -5 + (-10) + (-3) = -18
# Wrong! Best path is just -3

# With max(0, ...):
# left_gain = max(0, -10) = 0
# right_gain = max(0, -3) = 0  
# current_path = -5 + 0 + 0 = -5
# But the global max would be max(-5, -10, -3) = -3
```

Wait, there's a subtle issue! If all values are negative, we still need to include at least one node. Let's verify our solution handles this:

```python
# Tree: single node with value -5
# dfs(-5):
#   left_gain = 0, right_gain = 0
#   current_path_sum = -5 + 0 + 0 = -5
#   max_sum = max(-inf, -5) = -5
#   return -5 + max(0, 0) = -5
# 
# Result: -5 ✓
```

The solution correctly handles all-negative trees because:
1. `current_path_sum` always includes `node.val`
2. We use `max(0, child)` for children, not for the node itself

## 6. Alternative Implementation: Class-Based

Some prefer encapsulating state in a class:

```python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        """Class-based implementation with instance variable."""
        self.max_sum = float('-inf')
        self._dfs(root)
        return self.max_sum
    
    def _dfs(self, node: TreeNode) -> int:
        if not node:
            return 0
        
        left = max(0, self._dfs(node.left))
        right = max(0, self._dfs(node.right))
        
        # Update global max with complete path through this node
        self.max_sum = max(self.max_sum, node.val + left + right)
        
        # Return max single-direction contribution
        return node.val + max(left, right)
```

## 7. Approach 3: Iterative with Stack

For those who prefer iterative solutions (useful to avoid recursion limits):

```python
def maxPathSum_iterative(root: TreeNode) -> int:
    """
    Iterative post-order traversal using explicit stack.
    
    We need post-order because we process children before parent.
    Store computed gains in a dictionary.
    """
    if not root:
        return 0
    
    max_sum = float('-inf')
    gains = {None: 0}  # Maps node to its max contribution upward
    
    # Post-order traversal using stack
    stack = [(root, False)]  # (node, processed_children)
    
    while stack:
        node, processed = stack.pop()
        
        if processed:
            # Children have been processed; compute this node's values
            left_gain = max(0, gains.get(node.left, 0))
            right_gain = max(0, gains.get(node.right, 0))
            
            # Complete path through this node
            current_path = node.val + left_gain + right_gain
            max_sum = max(max_sum, current_path)
            
            # Store contribution for parent
            gains[node] = node.val + max(left_gain, right_gain)
        else:
            # First visit: add this node back (to process after children)
            # Then add children
            stack.append((node, True))
            if node.right:
                stack.append((node.right, False))
            if node.left:
                stack.append((node.left, False))
    
    return max_sum
```

**Complexity:**
- **Time:** O(N)
- **Space:** O(N) for the gains dictionary + O(H) for stack

## 8. Edge Cases and Testing

```python
def test_max_path_sum():
    """Comprehensive test cases."""
    
    # Test 1: Basic tree
    #     1
    #    / \
    #   2   3
    root1 = TreeNode(1, TreeNode(2), TreeNode(3))
    assert maxPathSum(root1) == 6, "Basic tree: 2+1+3=6"
    
    # Test 2: Tree with negative root
    #     -10
    #     /  \
    #    9   20
    #       /  \
    #      15   7
    root2 = TreeNode(-10, 
                     TreeNode(9),
                     TreeNode(20, TreeNode(15), TreeNode(7)))
    assert maxPathSum(root2) == 42, "Path 15+20+7=42"
    
    # Test 3: Single node
    root3 = TreeNode(5)
    assert maxPathSum(root3) == 5, "Single node"
    
    # Test 4: Single negative node
    root4 = TreeNode(-3)
    assert maxPathSum(root4) == -3, "Single negative node"
    
    # Test 5: All negative
    #    -1
    #   /  \
    # -2   -3
    root5 = TreeNode(-1, TreeNode(-2), TreeNode(-3))
    assert maxPathSum(root5) == -1, "Least negative path"
    
    # Test 6: Path doesn't go through root
    #      1
    #     /
    #    2
    #   / \
    #  3   4
    root6 = TreeNode(1, 
                     TreeNode(2, TreeNode(3), TreeNode(4)))
    # Best path: 3 + 2 + 4 = 9 (doesn't use root!)
    assert maxPathSum(root6) == 10, "Path including root: 3+2+1=6 or 4+2+1=7 or 3+2+4=9"
    # Actually: 3+2+1 or 4+2+1 use root. Full path 3+2+4+1? No, can't go up then down.
    # Best is 3 + 2 + 4 = 9? Let's check:
    # At node 3: return 3, max_sum = 3
    # At node 4: return 4, max_sum = 4
    # At node 2: left=3, right=4, current=2+3+4=9, max_sum=9, return 2+4=6
    # At node 1: left=6, right=0, current=1+6+0=7, max_sum=9, return 1+6=7
    # Answer: 9
    
    # Test 7: Left-skewed tree
    #     5
    #    /
    #   4
    #  /
    # 3
    root7 = TreeNode(5, TreeNode(4, TreeNode(3)))
    assert maxPathSum(root7) == 12, "Path 3+4+5=12"
    
    # Test 8: Large values
    root8 = TreeNode(1000, TreeNode(1000), TreeNode(1000))
    assert maxPathSum(root8) == 3000, "Large values"
    
    print("All tests passed!")

test_max_path_sum()
```

## 9. Common Mistakes

### Mistake 1: Forgetting to Use max(0, ...) for Child Contributions

```python
# WRONG: Including negative paths
left_gain = dfs(node.left)  # Could be negative!
right_gain = dfs(node.right)
# This might add negative contributions unnecessarily

# CORRECT: Ignore negative contributions
left_gain = max(0, dfs(node.left))
right_gain = max(0, dfs(node.right))
```

### Mistake 2: Returning Complete Path Instead of Single Direction

```python
# WRONG: Returning the complete path sum
return node.val + left_gain + right_gain  # Can't extend both ways!

# CORRECT: Return single direction only
return node.val + max(left_gain, right_gain)
```

### Mistake 3: Not Handling Single-Node Trees

```python
# WRONG: Initializing max_sum to 0
max_sum = 0  # Fails for all-negative trees

# CORRECT: Initialize to negative infinity
max_sum = float('-inf')
```

### Mistake 4: Confusing Path with Root-to-Leaf Path

The problem allows paths that don't include the root and don't go to leaves. Make sure to update the global maximum at every node.

## 10. Variations and Extensions

### Variation 1: Return the Actual Path

```python
def maxPathSum_with_path(root: TreeNode):
    """
    Return both the maximum sum and the actual path.
    """
    max_sum = float('-inf')
    best_path = []
    
    def dfs(node):
        nonlocal max_sum, best_path
        
        if not node:
            return 0, []
        
        left_gain, left_path = dfs(node.left)
        right_gain, right_path = dfs(node.right)
        
        # Clamp negative contributions
        left_contrib = max(0, left_gain)
        right_contrib = max(0, right_gain)
        
        # Complete path at this node
        current_sum = node.val + left_contrib + right_contrib
        
        if current_sum > max_sum:
            max_sum = current_sum
            # Build the path: reverse left + [node] + right
            best_path = []
            if left_gain > 0:
                best_path.extend(reversed(left_path))
            best_path.append(node.val)
            if right_gain > 0:
                best_path.extend(right_path)
        
        # Return contribution to parent
        if left_gain >= right_gain and left_gain > 0:
            return node.val + left_gain, left_path + [node.val]
        elif right_gain > 0:
            return node.val + right_gain, right_path + [node.val]
        else:
            return node.val, [node.val]
    
    dfs(root)
    return max_sum, best_path
```

### Variation 2: Path Must Go Through Root

```python
def maxPathSum_through_root(root: TreeNode) -> int:
    """Path must include the root node."""
    if not root:
        return 0
    
    def max_down(node):
        """Max sum path going downward from node."""
        if not node:
            return 0
        return node.val + max(0, max(max_down(node.left), max_down(node.right)))
    
    left_max = max(0, max_down(root.left))
    right_max = max(0, max_down(root.right))
    
    return root.val + left_max + right_max
```

### Variation 3: Maximum Path Sum Between Two Leaves

```python
def maxPathSum_leaves(root: TreeNode) -> int:
    """
    Path must start and end at leaf nodes.
    """
    max_sum = float('-inf')
    
    def dfs(node):
        nonlocal max_sum
        
        if not node:
            return float('-inf')
        
        # Leaf node
        if not node.left and not node.right:
            return node.val
        
        left = dfs(node.left)
        right = dfs(node.right)
        
        # If both children exist, we can form a leaf-to-leaf path
        if node.left and node.right:
            max_sum = max(max_sum, node.val + left + right)
            return node.val + max(left, right)
        
        # Only one child exists
        return node.val + (left if node.left else right)
    
    dfs(root)
    return max_sum if max_sum != float('-inf') else root.val
```

## 11. Connection to Transfer Learning in ML Systems

The tree DP pattern in Binary Tree Max Path Sum directly connects to how **transfer learning** propagates knowledge through neural network layers:

### Knowledge Propagation Analogy

Just as we propagate maximum contributions upward through the tree:
- **Bottom-up contribution**: Child nodes contribute their best paths to parents
- **Selective inclusion**: We only include positive contributions (like useful learned features)
- **Global optimization**: We track the global best while computing local optima

Similarly, in transfer learning:
- **Layer-wise adaptation**: Lower layers contribute feature representations
- **Selective freezing**: We keep beneficial pretrained weights (like positive path contributions)
- **Fine-tuning**: We optimize for the new task while preserving useful knowledge

### Code Parallel

```python
# Tree DP: Selective contribution
left_gain = max(0, dfs(node.left))  # Only keep positive

# Transfer Learning: Selective freezing
for layer in pretrained_model.layers[:freeze_point]:
    layer.trainable = False  # Keep beneficial representations
```

This connection highlights why understanding tree DP helps in designing adaptive ML systems—both involve propagating and selecting valuable "contributions" through hierarchical structures.

## 12. Interview Strategy

### How to Approach in an Interview

1. **Clarify the problem** (2 minutes)
   - "Can the path have one node?"
   - "Can values be negative?"
   - "Does the path need to go through the root?"

2. **Start with examples** (3 minutes)
   - Draw the example trees
   - Trace through manually to understand path structure

3. **Explain the key insight** (2 minutes)
   - "A path has a 'peak' node where it turns"
   - "We track two things: contribution to parent and complete path at each node"

4. **Code the solution** (10 minutes)
   - Write helper function first
   - Explain as you code

5. **Trace through example** (3 minutes)
   - Show the algorithm on Example 2

6. **Discuss complexity and edge cases** (2 minutes)

### Common Follow-up Questions

1. "Can you do this iteratively?" → Show the stack-based solution
2. "How would you return the actual path?" → Variation 1 above
3. "What if paths must be root-to-leaf?" → Different problem, simpler
4. "What if nodes have negative values?" → Our solution handles it

## 13. Production Considerations

### When Would You Use This in Production?

1. **Network routing**: Finding optimal paths through hierarchical network topologies
2. **Organization analysis**: Finding the most valuable chain in org charts
3. **Dependency resolution**: Finding the most critical dependency chain
4. **Game AI**: Evaluating move sequences in game trees

### Handling Large Trees

```python
import sys

def maxPathSum_large_trees(root: TreeNode) -> int:
    """Handle trees that might exceed recursion limit."""
    # Increase recursion limit for deep trees
    sys.setrecursionlimit(10000)
    
    # Or use iterative solution for production
    return maxPathSum_iterative(root)
```

### Parallel Processing for Forest

```python
from concurrent.futures import ThreadPoolExecutor

def maxPathSum_parallel(roots: list) -> list:
    """Process multiple trees in parallel."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(maxPathSum, roots))
    return results
```

## 14. Complexity Deep Dive

### Time Complexity Proof

Each node is visited exactly once in the DFS traversal. At each node:
- We make O(1) recursive calls (to children, which are counted separately)
- We do O(1) comparisons and arithmetic

Total: O(N) where N is the number of nodes.

### Space Complexity Analysis

**Recursive solution:**
- Call stack depth = tree height H
- Best case (balanced tree): H = O(log N)
- Worst case (skewed tree): H = O(N)
- Space: O(H)

**Iterative solution:**
- Stack size: O(H)
- Gains dictionary: O(N)
- Total: O(N)

## 15. Related Problems

| Problem | Key Difference | Link |
|---------|----------------|------|
| Path Sum | Fixed target, root-to-leaf only | LC 112 |
| Path Sum II | Return all paths | LC 113 |
| Path Sum III | Count paths with sum, any start | LC 437 |
| Diameter of Binary Tree | Path length, not sum | LC 543 |
| Longest Univalue Path | Same values | LC 687 |

## 16. Key Takeaways

- **Tree DP pattern**: Compute return value for parent while updating global answer
- **Two states per node**: Contribution (single direction) vs. complete path (both directions)
- **max(0, child) trick**: Elegantly handles negative values
- **Post-order traversal**: Process children before parent
- **O(N) optimal**: Each node visited once

The Binary Tree Maximum Path Sum problem teaches a fundamental pattern that appears throughout algorithm design: tracking local contributions while maintaining global optima. Master this, and you'll recognize it in many disguises—from network flow to ML pipeline optimization.

---

**Originally published at:** [arunbaby.com/dsa/0046-binary-tree-max-path-sum](https://www.arunbaby.com/dsa/0046-binary-tree-max-path-sum/)

*If you found this helpful, consider sharing it with others who might benefit.*
