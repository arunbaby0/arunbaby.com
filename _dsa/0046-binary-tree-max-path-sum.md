---
title: "Binary Tree Maximum Path Sum"
day: 46
related_ml_day: 46
related_speech_day: 46
related_agents_day: 46
collection: dsa
categories:
 - dsa
tags:
 - binary-tree
 - dynamic-programming
 - recursion
 - dfs
 - hard
difficulty: Hard
subdomain: "Trees & DP"
tech_stack: Python
scale: "O(N) time, O(H) space"
companies: Google, Meta, Amazon, Microsoft, Apple
---

**"Find the path to success—even if you have to start from the bottom, go up, and come back down."**

## 1. Problem Statement

The **Binary Tree Maximum Path Sum** problem is a classic hard interview question that tests your mastery of tree traversal and recursion state management.

Given the `root` of a binary tree, return the *maximum path sum* of any non-empty path.

**Constraints & Definitions:**
- A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them.
- A node can only appear in the sequence **at most once**.
- The path does **not** need to pass through the root.
- The path must contain at least one node.
- Node values can be **negative**, positive, or zero.

**Example 1:**
``
 1
 / \
 2 3
``
Output: `6` (Path: `2 -> 1 -> 3`)

**Example 2:**
``
 -10
 / \
 9 20
 / \
 15 7
``
Output: `42` (Path: `15 -> 20 -> 7`)

This problem is famous because the intuitive "top-down" approach fails. You cannot simply decide at the root which way to go, because the optimal path might exist entirely within a subtree (as seen in Example 2, where the root `-10` is excluded).

---

## 2. Understanding the Problem

### 2.1 The "Path" Concept
In most tree problems, a "path" usually means "root to leaf". In this problem, a path is strictly defined by graph connectivity. It can be:
1. **Child -> Node -> Parent**: Going up.
2. **Parent -> Node -> Child**: Going down.
3. **Left Child -> Node -> Right Child**: A "bridge" or "arch".

However, there is a strict limitation: **No branching**. You cannot go `Parent -> Node` AND `Node -> Left` AND `Node -> Right` simultaneously. That would form a "Y" shape, not a path.

### 2.2 The "Arc" Visualization
Every valid path in this problem has a unique "highest node" (closest to the root). We call this the **Peak** or **Anchor** of the path.
- From the Peak, the path goes down the left side (optional).
- From the Peak, the path goes down the right side (optional).

Therefore, any path can be mathematically described as:
`Path(Peak) = Value(Peak) + MaxChain(Left Child) + MaxChain(Right Child)`

Where `MaxChain` is a path starting at a child and going *only* downwards.

### 2.3 The Core Conflict
This definition creates a conflict:
- To calculate the global maximum path, we want to connect `Left + Node + Right`.
- But to hold up our end of the bargain for our *Parent*, we can only offer `Node + max(Left, Right)`. We cannot offer both, because that would force the path to split at our node (coming from parent, going to both children).

This distinction—**"What I calculate for myself"** vs. **"What I return to my parent"**—is the heart of the solution.

---

## 3. Approach 1: Brute Force

A naive approach might be to treat every single node as a potential "Peak".

### Algorithm
1. Traverse the tree (e.g., Pre-order).
2. For every node `N`:
 - Calculate the max path starting at `N.left` and going down.
 - Calculate the max path starting at `N.right` and going down.
 - Total Sum = `N.val + LeftSum + RightSum`.
 - Update global max.
3. This requires a helper function `getMaxDownPath(node)` which does its own traversal.

### Complexity
- **Time**: O(N^2). For every node, we traverse its entire subtree. In skewed trees (linked list), this is quadratic.
- **Space**: O(H) for recursion stack.

This is acceptable for small trees (`N < 1000`), but fails for large datasets. We need a single-pass solution.

---

## 4. Approach 2: Optimal Recursive Solution (One Pass)

We can solve this in O(N) by using a **Post-order Traversal** (Bottom-Up).
The recursion will return the "Contribution" value to the parent, while simultaneously updating the "Global Maximum" using the "Arc" value.

### Key Logic
At every node `root`, asking for `dfs(root)`:
1. Recursively get `left_gain = dfs(root.left)`.
2. Recursively get `right_gain = dfs(root.right)`.
3. **Crucial Check**: If a child's gain is negative, ignore it! A negative path reduces our sum. We are better off not including that branch. `left_gain = max(left_gain, 0)`.
4. **Local Arc Sum**: `current_node_max = root.val + left_gain + right_gain`.
 - Update the global maximum if this is the best we've seen.
5. **Return Value**: `root.val + max(left_gain, right_gain)`.
 - We can only ascend one branch to the parent.

---

## 5. Implementation

Here is the robust, production-ready implementation.

``python
import math

# Definition for a binary tree node.
class TreeNode:
 def __init__(self, val=0, left=None, right=None):
 self.val = val
 self.left = left
 self.right = right

class Solution:
 def maxPathSum(self, root: TreeNode) -> int:
 """
 Calculates the maximum path sum in a binary tree.
 
 Args:
 root: The root node of the binary tree.
 
 Returns:
 int: The maximum path sum found.
 """
 # Initialize global max with absolute minimum
 # Using a list allows us to update it within the nested function scope
 self.global_max = -math.inf
 
 def get_max_gain(node):
 """
 Recursive function that returns the maximum contribution a node 
 can make to its parent's path.
 
 Side effect: Updates self.global_max with the best 'arch' path 
 peaking at this node.
 """
 if not node:
 return 0
 
 # Recursive step: get gains from left and right subtrees
 # If a subtree returns a negative gain, we treat it as 0
 # (i.e., we choose not to include that path segment).
 left_gain = max(get_max_gain(node.left), 0)
 right_gain = max(get_max_gain(node.right), 0)
 
 # Case 1: The path curves at this node (Left -> Node -> Right)
 # This path CANNOT be extended to the parent, but it might be the global max.
 price_of_new_path = node.val + left_gain + right_gain
 
 # Update the global result if this curve is better than anything seen so far
 self.global_max = max(self.global_max, price_of_new_path)
 
 # Case 2: The path continues upward (Node -> Parent)
 # We must choose the heavier branch (Left or Right) to extend.
 return node.val + max(left_gain, right_gain)
 
 get_max_gain(root)
 
 return self.global_max

# Example Helper to build simple tree
def run_example():
 # Tree: [1, 2, 3]
 root = TreeNode(1)
 root.left = TreeNode(2)
 root.right = TreeNode(3)
 sol = Solution()
 print(f"Max Path Sum: {sol.maxPathSum(root)}") # Expected: 6

if __name__ == "__main__":
 run_example()
``

### Why use a class variable?
In the recursive function, we encounter two different values of interest:
1. The value to **pass up** (recursion return).
2. The value to **record** (global max).
Using `self.global_max` decouples these two concerns cleanly.

---

## 6. Testing Strategy

When testing recursive tree algorithms, "visual" coverage is key.

### Test Case 1: The "Tent" (Simple curve)
``
 1
 / \
 2 3
``
- Left gain: 2
- Right gain: 3
- Arc: 1 + 2 + 3 = 6. Return: 1 + 3 = 4.
- **Max: 6**.

### Test Case 2: Negative Root (Disconnected Subtrees)
``
 -10
 / \
 9 20
 / \
 15 7
``
- Node 9: Returns 9. Max so far: 9.
- Node 15: Returns 15. Max so far: 15.
- Node 7: Returns 7. Max so far: 15.
- Node 20:
 - Left gain: 15.
 - Right gain: 7.
 - Arc: 20 + 15 + 7 = 42. Max so far: 42.
 - Return: 20 + 15 = 35.
- Node -10:
 - Left gain: 9.
 - Right gain: 35.
 - Arc: -10 + 9 + 35 = 34.
 - Return: -10 + 35 = 25.
- **Result: 42**. Note that the root -10 was examined but the "Arc" through it (34) was inferior to the subtree arc (42).

### Test Case 3: All Negative
``
 -3
``
- Recursive base case returns 0? NO.
- `left_gain = max(0, 0) = 0`.
- Arc: -3 + 0 + 0 = -3.
- **Result: -3**.
*Edge Case Warning*: If your code initializes `global_max = 0`, you will fail this case (returning 0 instead of -3). Always initialize with `-infinity`.

---

## 7. Complexity Analysis

### Time Complexity: O(N)
- We touch every node exactly once.
- At each node, we perform constant time operations (`+, \max`).
- Therefore, time scales linearly with the number of nodes `N`.

### Space Complexity: O(H)
- `H` is the height of the tree.
- This is the cost of the implicit recursion stack.
- **Best Case (Balanced Tree):** `H = \log N`.
- **Worst Case (Skewed Tree):** `H = N`.

---

## 8. Production Considerations

In a real-world system (like a network routing table or organizational hierarchy analyzer), how does this perform?

### 8.1 Stack Overflow Risk
For extremely deep trees (`N > 10,000` in a line), standard Python recursion limit (1000) will be hit.
**Solution:**
- Increase recursion limit: `sys.setrecursionlimit(20000)`.
- **Better:** Iterative DFS using an explicit stack is preferred for production stability, though significantly harder to implement for post-order logic.

### 8.2 Parallelism
Can we calculate this in parallel?
- Subtrees are independent.
- If the tree is distributed (e.g., a DOM tree or a distributed database index), we can send `get_max_gain` queries to `Node.left` and `Node.right` workers in parallel (MapReduce style).

---

## 9. Connections to ML Systems

This algorithm is conceptually identical to **Backpropagation through Time (BPTT)** or recursive neural networks (RvNNs, distinct from RNNs).

### 9.1 Socher's Recursive Neural Networks (Tree-LSTMs)
In Sentiment Analysis, we often parse sentences into syntax trees.
- "The movie was [not [good]]".
- We need to compute the "sentiment vector" of `"not good"` based on its children `"not"` and `"good"`.
- This is exactly the bottom-up traversal we just wrote!
- Instead of `val + left + right` (scalar sum), ML systems use `Op(val, W_L * left_vec, W_R * right_vec)` (tensor operations).

### 9.2 Transfer Learning Theme
The theme for this section of the blog is **Transfer Learning**. In this algorithm, information "transfers" strictly from child to parent. The parent cannot solve its problem without the "pre-trained knowledge" (max path) of its children. This mirrors how a pre-trained ResNet feature extractor feeds into a new classifier head.

---

## 10. Interview Strategy

If asked this in an interview:

1. **Don't Rush Code**: Draw the "Arc" vs "Branch" distinction on the whiteboard first. This shows you understand the edge cases.
2. **Highlight the "Negative" Case**: ask "Are node values negative?". This is the main trap.
3. **Global vs Local**: Explicitly mention why you need a global variable. Interviewers love asking "Can you do this without a global variable?" (Yes, return a distinct Tuple `(max_branch, max_arc)` from search step).
4. **Iterative Follow-up**: If they ask for iterative, suggest using a `visited` set to mimic post-order traversal with a stack, though admit it makes the code much messier.

---

## 11. Key Takeaways

1. **Tree DP Pattern**: This fits the "Tree Dynamic Programming" pattern: `Res(Node) = F(Node, Res(Left), Res(Right))`.
2. **Path != Traverse**: A path has structural constraints (no branching).
3. **Max(0, x)**: A powerful idiom for "optional inclusion".
4. **Separation of Concerns**: The value you *return* (recursion contract) is often different from the value you *calculate* (business logic).

---

**Originally published at:** [arunbaby.com/dsa/0046-binary-tree-max-path-sum](https://www.arunbaby.com/dsa/0046-binary-tree-max-path-sum/)

*If you found this helpful, consider sharing it with others who might benefit.*
