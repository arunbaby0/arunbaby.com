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

## 1. Introduction: Why This Problem Matters

Imagine you're analyzing a network of servers, where each server has a "value" representing its processing capacity (positive) or its overhead cost (negative). You want to find the most valuable path through this network—a sequence of connected servers that maximizes the total value.

This is exactly what the **Binary Tree Maximum Path Sum** problem asks us to solve. It's a classic interview question at top tech companies because it tests your understanding of:

- **Tree traversal and recursion**: How to think about problems in terms of subproblems
- **Dynamic programming on trees**: How to carry information up and down a tree
- **The distinction between "contribution" and "completion"**: A subtle but crucial concept

What makes this problem tricky is that the maximum path doesn't have to go through the root. It can start and end anywhere in the tree. This seemingly small detail completely changes how we need to approach the problem.

---

## 2. Understanding the Problem

### 2.1 What is a "Path" in a Binary Tree?

Before we dive into the solution, let's make sure we understand exactly what we're looking for.

A **path** in a binary tree is:
- A sequence of nodes where each consecutive pair is connected by an edge
- The path can start at any node and end at any node
- **Crucially**: The path cannot branch. Once you go down a path, you can't split and go both left and right.

Let me illustrate this with a visual:

```
Valid paths:        Invalid path (branches):
    
    1                      1
   / \                    /|\
  2   3                  2 1 3   ← Can't go both ways!
     / \                    
    4   5              
    
Paths: [2,1,3], [4,3,5], [1,3,4], etc.
```

### 2.2 The Problem Statement

Given a binary tree, find the **maximum path sum**. The path:
- Must contain at least one node
- Can start and end at any nodes
- Follows parent-child connections (no jumping)

**Example 1: Simple case**
```
    1
   / \
  2   3

Maximum path sum: 6 (path: 2 → 1 → 3)
```

**Example 2: With negative values**
```
       -10
       /  \
      9   20
         /  \
        15   7

Maximum path sum: 42 (path: 15 → 20 → 7)
```

Notice in Example 2 that the maximum path doesn't include the root (-10) because including it would reduce the sum!

### 2.3 Why This Problem is Tricky

At first glance, you might think: "Just find the path from any node to any other node with the maximum sum." But there are **exponentially many paths** in a tree! For a tree with N nodes, checking all possible paths would take O(N²) or worse.

The key insight is that we need to think about this problem differently—not as "finding paths" but as "computing values at each node that help us track the global maximum."

---

## 3. Building Intuition: The Arc Concept

### 3.1 Every Path is an "Arc" with a Peak

Here's the mental model that unlocks this problem: **every path in a tree forms an arc with exactly one highest point** (we'll call it the "peak" or "turning point").

```
        Peak
       /    \
      ↗      ↘
    Left    Right
   branch   branch
```

At any node in the tree, a path passing through that node can:
1. Come up from the left subtree
2. Pass through this node (the peak)
3. Go down into the right subtree

Or it could only go left, or only go right, or just be the node itself.

### 3.2 Two Different Questions at Each Node

This leads us to ask **two different questions** at each node:

**Question 1: What's the maximum path that "peaks" at this node?**
This is a path that might use both left and right children. This value helps us update our global maximum.

**Question 2: What's the maximum contribution this subtree can make to a path that peaks higher up?**
This is the value we return to the parent. It can only go in ONE direction (left or right, not both), because the path can't branch.

Let me illustrate:

```
        A  ← Parent asking: "What can child B contribute to me?"
       /
      B  ← B can contribute: B.val + max(left_contribution, right_contribution)
     / \           BUT NOT: B.val + left + right (that would branch!)
    L   R
```

This distinction between "complete path at this node" and "contribution to parent" is the crux of the solution.

---

## 4. The Algorithm: Step by Step

### 4.1 High-Level Approach

We'll use **post-order traversal** (process children before parent) with a clever twist:

1. At each node, compute the maximum contribution this subtree can offer to its parent
2. While doing this, also compute the maximum complete path that peaks at this node
3. Keep track of the global maximum path sum across all nodes

### 4.2 Handling Negative Contributions

What if a child's contribution is negative? Should we include it?

**No!** If a subtree contributes a negative value, we're better off not including it at all. We use `max(0, child_contribution)` to handle this elegantly.

For example:
```
      5
     / \
   -3   2

Left contribution: max(0, -3) = 0  (ignore the left child)
Right contribution: max(0, 2) = 2  (include the right child)

Maximum path at node 5: 5 + 0 + 2 = 7
Contribution to parent: 5 + max(0, 2) = 7
```

### 4.3 The Key Formulas

At each node, we compute:

**1. Gain from left child:**
```
left_gain = max(0, left_child_contribution)
```

**2. Gain from right child:**
```
right_gain = max(0, right_child_contribution)
```

**3. Maximum complete path peaking at this node:**
```
peak_path_sum = node.val + left_gain + right_gain
```
This is a complete path that uses both children (if beneficial).

**4. Contribution to parent:**
```
contribution_to_parent = node.val + max(left_gain, right_gain)
```
This can only go in ONE direction to avoid branching.

### 4.4 Visual Walkthrough

Let's trace through Example 2:

```
       -10
       /  \
      9   20
         /  \
        15   7
```

**Processing order** (post-order): 9, 15, 7, 20, -10

**Step 1: Process node 9 (leaf)**
- Left gain: 0 (no left child)
- Right gain: 0 (no right child)
- Peak path sum: 9 + 0 + 0 = 9
- Global max so far: 9
- Contribution to parent: 9 + max(0, 0) = 9

**Step 2: Process node 15 (leaf)**
- Left gain: 0
- Right gain: 0
- Peak path sum: 15
- Global max so far: max(9, 15) = 15
- Contribution to parent: 15

**Step 3: Process node 7 (leaf)**
- Left gain: 0
- Right gain: 0
- Peak path sum: 7
- Global max so far: max(15, 7) = 15
- Contribution to parent: 7

**Step 4: Process node 20**
- Left gain: max(0, 15) = 15
- Right gain: max(0, 7) = 7
- Peak path sum: 20 + 15 + 7 = **42** ✨
- Global max so far: max(15, 42) = 42
- Contribution to parent: 20 + max(15, 7) = 35

**Step 5: Process node -10 (root)**
- Left gain: max(0, 9) = 9
- Right gain: max(0, 35) = 35
- Peak path sum: -10 + 9 + 35 = 34
- Global max so far: max(42, 34) = **42** (unchanged)
- Contribution to parent: N/A (this is the root)

**Final answer: 42** (the path 15 → 20 → 7)

---

## 5. The Solution

Now that we understand the algorithm, let's look at the implementation. Notice how concise it is—the complexity is in the thinking, not the code!

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def maxPathSum(root: TreeNode) -> int:
    """
    Find the maximum path sum in a binary tree.
    
    The key insight is distinguishing between:
    1. The max path "peaking" at each node (for global max)
    2. The max "contribution" to parent (for recursion)
    """
    # We use a list to allow modification in nested function
    max_sum = [float('-inf')]
    
    def compute_max_contribution(node):
        """
        Returns the maximum contribution this subtree can make
        to a path that extends upward to the parent.
        
        Side effect: Updates max_sum if we find a better path.
        """
        if not node:
            return 0
        
        # Recursively get contributions from children
        # Use max(0, ...) to ignore negative contributions
        left_gain = max(0, compute_max_contribution(node.left))
        right_gain = max(0, compute_max_contribution(node.right))
        
        # The best complete path through this node
        peak_path_sum = node.val + left_gain + right_gain
        
        # Update global maximum
        max_sum[0] = max(max_sum[0], peak_path_sum)
        
        # Return contribution to parent (can only go one direction)
        return node.val + max(left_gain, right_gain)
    
    compute_max_contribution(root)
    return max_sum[0]
```

### 5.1 Why Use a List for max_sum?

You might wonder why we use `max_sum = [float('-inf')]` instead of just `max_sum = float('-inf')`. 

In Python, when you assign a new value to a variable inside a nested function, Python treats it as a new local variable. Using a list (or `nonlocal` keyword) lets us modify the outer variable.

```python
# This doesn't work:
max_sum = float('-inf')
def inner():
    max_sum = 10  # Creates a NEW local variable!

# This works:
max_sum = [float('-inf')]
def inner():
    max_sum[0] = 10  # Modifies the existing list
```

---

## 6. Common Mistakes and Edge Cases

### 6.1 Mistake: Including Negative Paths

**Wrong thinking**: "I should include all children in the path."

**Correct thinking**: Use `max(0, child_contribution)` to exclude negative contributions.

### 6.2 Mistake: Returning the Complete Path Instead of Contribution

**Wrong thinking**: "Return `node.val + left + right` to the parent."

**Correct thinking**: Return `node.val + max(left, right)` because paths can't branch.

### 6.3 Edge Case: All Negative Values

What if every node has a negative value?

```
      -3
     /  \
   -2   -1
```

The answer should be **-1** (just the node with the largest value, since we must include at least one node).

Our algorithm handles this correctly because:
- `max_sum` starts at negative infinity
- We always update it with `peak_path_sum` which includes at least the node's value
- The maximum single node (-1) becomes our answer

### 6.4 Edge Case: Single Node

```
    5

Answer: 5
```

Our algorithm handles this: left_gain = 0, right_gain = 0, peak_path_sum = 5.

---

## 7. Complexity Analysis

### Time Complexity: O(N)

We visit each node exactly once during the post-order traversal. At each node, we do O(1) work (a few comparisons and additions).

### Space Complexity: O(H) where H is the tree height

The space is used by the recursion stack. In the worst case (a completely skewed tree), H = N, so space is O(N). For a balanced tree, H = log(N), so space is O(log N).

```
Skewed tree (H = N):     Balanced tree (H = log N):
    1                           4
     \                        /   \
      2                      2     6
       \                    / \   / \
        3                  1   3 5   7
```

---

## 8. Why This Pattern Matters: Connection to Transfer Learning

This problem teaches a pattern that appears throughout computer science and machine learning: **the distinction between local computation and global tracking**.

In the context of **Transfer Learning** (Day 46 ML topic):

| Binary Tree Max Path Sum | Transfer Learning |
|--------------------------|-------------------|
| Local contribution to parent | Layer-specific features |
| Global maximum path | End-to-end model performance |
| Choosing max(left, right) | Selecting which features to transfer |
| Ignoring negative contributions | Filtering out harmful knowledge transfer |

Just as we track a global maximum while computing local contributions, transfer learning tracks global model performance while deciding which layer-specific knowledge to reuse.

---

## 9. Interview Tips

### 9.1 How to Approach This in an Interview

1. **Start with examples**: Draw a small tree and manually trace through what paths exist
2. **Identify the insight**: Explain the distinction between "peak path" and "contribution"
3. **State the recurrence**: Write out the formulas before coding
4. **Handle edge cases**: Mention negative values and single nodes
5. **Analyze complexity**: Time O(N), Space O(H)

### 9.2 Common Follow-Up Questions

**Q: What if we need to return the actual path, not just the sum?**
A: Track the nodes as we traverse. When we update the global max, also store the path.

**Q: What if we want the k largest path sums?**
A: Use a min-heap of size k. Push each peak_path_sum and maintain heap size.

**Q: Can you solve this iteratively?**
A: Yes, using a stack for post-order traversal, but it's more complex and less intuitive.

---

## 10. Practice Problems

Once you understand this pattern, try these related problems:

1. **Binary Tree Diameter** - Similar concept, but counting edges instead of summing values
2. **Path Sum III** - Count paths that sum to a target (uses prefix sums)
3. **Longest Univalue Path** - Same pattern, different condition for extension

---

## 11. Summary

The Binary Tree Maximum Path Sum problem teaches us a powerful pattern:

1. **Think in terms of "contribution" vs "completion"**: What value do we pass up to the parent vs. what value completes a path at this node?

2. **Use post-order traversal for tree DP**: Process children first, then use their results to compute the parent's values.

3. **Track global state while computing local values**: The recursion computes contributions, but we update a global variable for the answer.

4. **Handle negative values gracefully**: Using `max(0, child)` elegantly handles the case where including a subtree would hurt our sum.

This pattern—computing local values while tracking a global optimum—appears in many contexts: network routing, game theory, and yes, transfer learning in machine learning.

---

**Originally published at:** [arunbaby.com/dsa/0046-binary-tree-max-path-sum](https://www.arunbaby.com/dsa/0046-binary-tree-max-path-sum/)

*If you found this helpful, consider sharing it with others who might benefit.*
