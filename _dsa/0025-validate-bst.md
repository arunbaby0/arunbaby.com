---
title: "Validate Binary Search Tree"
day: 25
collection: dsa
categories:
  - dsa
tags:
  - binary-search-tree
  - recursion
  - dfs
  - bfs
  - medium
subdomain: "Tree Algorithms"
tech_stack: [Python, C++, Java]
scale: "O(N) time, O(H) space"
companies: [Facebook, Amazon, Microsoft, Bloomberg, LinkedIn]
related_dsa_day: 25
related_ml_day: 25
related_speech_day: 25
---

**The gatekeeper of data integrity. How do we ensure our sorted structures are actually sorted?**

## Problem

Given the `root` of a binary tree, determine if it is a valid binary search tree (BST).

A **valid BST** is defined as follows:
1.  The left subtree of a node contains only nodes with keys **less than** the node's key.
2.  The right subtree of a node contains only nodes with keys **greater than** the node's key.
3.  Both the left and right subtrees must also be binary search trees.

**Example 1:**
```
    2
   / \
  1   3
```
**Input:** `root = [2,1,3]`
**Output:** `true`

**Example 2:**
```
    5
   / \
  1   4
     / \
    3   6
```
**Input:** `root = [5,1,4,null,null,3,6]`
**Output:** `false`
**Explanation:** The root node's value is 5 but its right child's value is 4.

## Intuition

A Binary Search Tree (BST) is the backbone of efficient search. It guarantees `O(log N)` lookup. But this guarantee only holds if the tree is valid. If a single node is out of place, the search algorithm breaks.

The most common mistake beginners make is checking only the immediate children:
`node.left.val < node.val < node.right.val`

**This is wrong.**
Consider this tree:
```
    5
   / \
  4   6
     / \
    3   7
```
- Node 5 is valid (4 < 5 < 6).
- Node 6 is valid (3 < 6 < 7).
- **But the tree is invalid!** The node `3` is in the right subtree of `5`, but `3 < 5`.

**Key Insight:** Every node defines a **range** `(min, max)` for its children.
- The root can be anything `(-inf, +inf)`.
- If we go left, the upper bound tightens: `(-inf, root.val)`.
- If we go right, the lower bound tightens: `(root.val, +inf)`.

## Approach 1: Recursive Traversal with Valid Range

We pass the valid range `(low, high)` down the recursion stack.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        
        def validate(node, low, high):
            # Empty trees are valid BSTs
            if not node:
                return True
            
            # The current node's value must be within the range (low, high)
            if not (low < node.val < high):
                return False
            
            # Recursively validate subtrees
            # Left child: range becomes (low, node.val)
            # Right child: range becomes (node.val, high)
            return (validate(node.left, low, node.val) and
                    validate(node.right, node.val, high))
            
        return validate(root, float('-inf'), float('inf'))
```

**Complexity Analysis:**
- **Time:** `O(N)`. We visit every node exactly once.
- **Space:** `O(H)`, where `H` is the height of the tree (recursion stack). In worst case (skewed tree), `O(N)`.

## Approach 2: Inorder Traversal

A fundamental property of a BST is that its **Inorder Traversal** (Left -> Root -> Right) produces a **sorted array**.

We can traverse the tree inorder and check if the values are strictly increasing.

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        self.prev = float('-inf')
        self.valid = True
        
        def inorder(node):
            if not node or not self.valid:
                return
            
            inorder(node.left)
            
            # Check if current value is greater than previous value
            if node.val <= self.prev:
                self.valid = False
                return
            self.prev = node.val
            
            inorder(node.right)
            
        inorder(root)
        return self.valid
```

**Optimization:** We don't need to store the whole array. We just need to keep track of the `prev` element.

## Approach 3: Iterative Inorder (Stack)

Recursion uses the system stack. We can simulate this with an explicit stack to avoid stack overflow errors in languages with limited recursion depth.

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        stack = []
        prev = float('-inf')
        current = root
        
        while stack or current:
            while current:
                stack.append(current)
                current = current.left
            
            current = stack.pop()
            
            # If next element in inorder traversal
            # is smaller than the previous one, that's invalid.
            if current.val <= prev:
                return False
            prev = current.val
            
            current = current.right
            
        return True
```

## Advanced Variant: Morris Traversal (O(1) Space)

Can we do this without a stack or recursion? Yes, using **Morris Traversal**.
It modifies the tree structure temporarily (threading) to traverse it, then restores it.

**Idea:**
For a node, find its **predecessor** (rightmost node of the left subtree). Make the predecessor's right child point to the current node. This creates a "back link" to return to the root after visiting the left subtree.

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        current = root
        prev = float('-inf')
        
        while current:
            if not current.left:
                # Process node
                if current.val <= prev:
                    return False
                prev = current.val
                current = current.right
            else:
                # Find predecessor
                pre = current.left
                while pre.right and pre.right != current:
                    pre = pre.right
                
                if not pre.right:
                    # Create thread
                    pre.right = current
                    current = current.left
                else:
                    # Break thread (restore tree)
                    pre.right = None
                    # Process node
                    if current.val <= prev:
                        return False
                    prev = current.val
                    current = current.right
                    
        return True
```

**Pros:** `O(1)` Space!
**Cons:** Modifies the tree (not thread-safe). Slower due to pointer manipulation.

## System Design: Validating Database Indexes

**Interviewer:** "How does a database like PostgreSQL ensure its B-Tree indexes are not corrupted?"

**Candidate:**
1.  **Checksums:** Every page on disk has a CRC32 checksum. If the bits rot, we know.
2.  **In-Memory Validation:** When reading a page, the DB checks if `min_key <= all_keys <= max_key`.
3.  **`amcheck` (Postgres):** A utility that runs `Validate BST` logic on the B-Tree structure.
    - It verifies parent-child relationships.
    - It verifies that the "High Key" of the left sibling is less than the "Low Key" of the right sibling.

**Scenario:** You are building a distributed KV store (like DynamoDB).
- **Problem:** A bit flip in RAM changes a pointer. Now a subtree is "lost" (unreachable).
- **Detection:** Run a background "Scrubber" process that traverses the tree (like Approach 1) and verifies integrity.
- **Repair:** If an inconsistency is found, rebuild the index from the WAL (Write Ahead Log).

## Advanced Variant 1: Recover Binary Search Tree

**Problem:** Two nodes of a BST are swapped by mistake. Recover the tree without changing its structure.
**Constraint:** Use `O(1)` space.

**Intuition:**
If we do an Inorder Traversal of a valid BST, we get a sorted array: `[1, 2, 3, 4, 5]`.
If two nodes are swapped (e.g., 2 and 4), we get: `[1, 4, 3, 2, 5]`.
Notice the inversions:
1.  `4 > 3` (First violation). The larger value (`4`) is the first swapped node.
2.  `3 > 2` (Second violation). The smaller value (`2`) is the second swapped node.

We can find these two nodes using Morris Traversal (to keep `O(1)` space) and then swap their values.

```python
class Solution:
    def recoverTree(self, root: TreeNode) -> None:
        self.first = None
        self.second = None
        self.prev = TreeNode(float('-inf'))
        
        # Morris Traversal
        curr = root
        while curr:
            if not curr.left:
                self.detect_swap(curr)
                curr = curr.right
            else:
                pre = curr.left
                while pre.right and pre.right != curr:
                    pre = pre.right
                
                if not pre.right:
                    pre.right = curr
                    curr = curr.left
                else:
                    pre.right = None
                    self.detect_swap(curr)
                    curr = curr.right
        
        # Swap values
        self.first.val, self.second.val = self.second.val, self.first.val
    
    def detect_swap(self, curr):
        if curr.val < self.prev.val:
            if not self.first:
                self.first = self.prev
            self.second = curr
        self.prev = curr
```

## Advanced Variant 2: BST Iterator

**Problem:** Implement an iterator over a BST with `next()` and `hasNext()` methods.
**Constraint:** `next()` and `hasNext()` should run in `O(1)` average time and use `O(H)` memory.

**Intuition:**
We can't flatten the tree into a list (that takes `O(N)` memory).
Instead, we simulate the recursion stack.
1.  Initialize: Push all left children of the root onto the stack.
2.  `next()`: Pop a node. If it has a right child, push all *its* left children onto the stack.

```python
class BSTIterator:
    def __init__(self, root: TreeNode):
        self.stack = []
        self._push_left(root)

    def _push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        node = self.stack.pop()
        if node.right:
            self._push_left(node.right)
        return node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0
```

## Advanced Variant 3: Largest BST Subtree

**Problem:** Given a Binary Tree, find the largest subtree which is a Binary Search Tree. Return the size (number of nodes).

**Intuition:**
A Bottom-Up approach is best (Postorder Traversal).
For each node, we need to know:
1.  Is my left subtree a BST?
2.  Is my right subtree a BST?
3.  What is the max value in left subtree? (Must be `< node.val`)
4.  What is the min value in right subtree? (Must be `> node.val`)
5.  What is the size of the subtree?

If all conditions are met, we are a BST of size `left_size + right_size + 1`.
If not, we return `size = -1` (or some flag) to indicate invalidity, but we pass up the max size found so far.

```python
class Solution:
    def largestBSTSubtree(self, root: TreeNode) -> int:
        self.max_size = 0
        
        def postorder(node):
            if not node:
                # min_val, max_val, size
                return float('inf'), float('-inf'), 0
            
            l_min, l_max, l_size = postorder(node.left)
            r_min, r_max, r_size = postorder(node.right)
            
            # Check if valid BST
            if l_max < node.val < r_min:
                size = l_size + r_size + 1
                self.max_size = max(self.max_size, size)
                # Return updated range and size
                return min(l_min, node.val), max(r_max, node.val), size
            else:
                # Not a BST, return invalid range but keep size 0 to indicate failure
                return float('-inf'), float('inf'), 0
                
        postorder(root)
        return self.max_size
```

## Engineering Deep Dive: Floating Point Precision

In the real world, BSTs often store `float` or `double` values (e.g., timestamps, prices).
**The Problem:** `a < b` is dangerous with floats.
`0.1 + 0.2 == 0.3` is `False` in Python/C++.

**Solution:** Epsilon Comparison.
Instead of `val < high`, use `val < high - epsilon`.
Or better, store values as integers (e.g., micros since epoch, cents) whenever possible.

## System Design: Concurrency Control

**Interviewer:** "How do you validate a BST that is being updated by 1000 threads?"

**Candidate:**
1.  **Global Lock (Mutex):** Simple but slow. No concurrency.
2.  **Reader-Writer Lock:** Multiple readers (validators) can run in parallel. Writers (inserts/deletes) block readers.
    - `ValidateBST` acquires a Read Lock.
3.  **Optimistic Concurrency Control (OCC):**
    - Version the tree nodes.
    - Validate without locks.
    - Check if version changed during validation. If yes, retry.
4.  **Copy-on-Write (CoW):**
    - Used in functional databases (CouchDB).
    - Validation runs on a snapshot. Updates create new nodes.

## Appendix A: Handling Duplicates

The standard definition says "strictly less/greater".
What if duplicates are allowed? `left <= node < right`?
- **Inorder Traversal:** Still works. The sequence will be non-decreasing (`1, 2, 2, 3`).
- **Range Approach:** Change `<` to `<=`.

**Interview Tip:** Always clarify with the interviewer: "Do we allow duplicate values?"

## Appendix B: Comprehensive Test Cases

1.  **Valid:** `[2, 1, 3]` -> True
2.  **Invalid (Right Child Small):** `[5, 1, 4]` -> False
3.  **Invalid (Deep Left):** `[5, 4, 6, null, null, 3, 7]` -> False (3 is in right subtree of 5).
4.  **Single Node:** `[1]` -> True
5.  **Duplicates:** `[1, 1]` -> False (per standard definition).
6.  **Int Limits:** `[2147483647]` -> True. (Use `long` or `float('inf')` for bounds).
7.  **Skewed:** `[1, null, 2, null, 3]` -> True.
8.  **Float Values:** `[1.1, 0.9, 1.2]` -> True.
9.  **Negative Values:** `[-1, -2, 0]` -> True.

## Advanced Variant 4: Validate AVL Tree

**Problem:** Check if a BST is a valid AVL Tree.
**Condition:** For every node, the height difference between left and right subtrees is at most 1.

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def check(node):
            if not node:
                return 0
            
            left_h = check(node.left)
            if left_h == -1: return -1
            
            right_h = check(node.right)
            if right_h == -1: return -1
            
            if abs(left_h - right_h) > 1:
                return -1
            
            return max(left_h, right_h) + 1
            
        return check(root) != -1
```

## Advanced Variant 5: Validate Red-Black Tree

**Problem:** Check if a BST is a valid Red-Black Tree.
**Properties:**
1.  Every node is Red or Black.
2.  Root is Black.
3.  Leaves (NIL) are Black.
4.  If a node is Red, both children are Black.
5.  Every path from a node to any of its descendant NIL nodes contains the same number of Black nodes.

This requires passing two values up the recursion: `(is_valid, black_height)`.

## Deep Dive: Threaded Binary Trees

We used Morris Traversal earlier. This is based on **Threaded Binary Trees**.
A "Thread" is a pointer to the in-order successor (or predecessor) stored in the `right` (or `left`) child pointer if it would otherwise be null.

**Types:**
1.  **Single Threaded:** Only right null pointers point to successor.
2.  **Double Threaded:** Left null pointers point to predecessor too.

**Why?**
-   Avoids recursion (Stack overflow).
-   Avoids stack (Memory overhead).
-   Faster traversal (No push/pop).

## System Design: Distributed BST (DHT)

**Interviewer:** "How do you validate a BST that spans 1000 servers?"
**Candidate:** "That's a Distributed Hash Table (DHT) like Chord or Dynamo."

**Chord Ring Validation:**
1.  **Stabilization Protocol:** Every node periodically asks its successor: "Who is your predecessor?"
2.  **Rectification:** If `successor.predecessor` is not me, but someone between us, I update my successor.
3.  **Global Consistency:** We can't pause the world to validate. We rely on **Eventual Consistency**.
4.  **Anti-Entropy:** Merkle Trees are used to compare data ranges between nodes efficiently.

## Engineering Deep Dive: Cache Locality

Standard BSTs are bad for CPU Cache.
-   Nodes are allocated on the heap (random addresses).
-   `node.left` might be at `0x1000`, `node.right` at `0x9000`.
-   Traversing causes **Cache Misses**.

**Solution: B-Trees (or B+ Trees).**
-   Store multiple keys (e.g., 100) in a single node (contiguous memory).
-   Fits in a Cache Line (64 bytes) or Page (4KB).
-   This is why Databases use B-Trees, not BSTs.

## Iterative Postorder Traversal (One Stack)

The hardest traversal to implement iteratively.
We need to know if we are visiting a node from the left (go right next) or from the right (process node next).

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        stack = []
        res = []
        curr = root
        last_visited = None
        
        while stack or curr:
            if curr:
                stack.append(curr)
                curr = curr.left
            else:
                peek = stack[-1]
                # If right child exists and traversing from left child, then move right
                if peek.right and last_visited != peek.right:
                    curr = peek.right
                else:
                    res.append(peek.val)
                    last_visited = stack.pop()
                    
        return res
```

## Advanced Variant 6: Cartesian Trees

**Definition:** A tree that is a Heap (by value) and a BST (by index/key).
**Use Case:** Range Minimum Query (RMQ) -> LCA in Cartesian Tree.

**Construction:**
Given an array `[3, 2, 1, 5, 4]`.
1.  Find min element `1`. This is Root.
2.  Left child is Cartesian Tree of `[3, 2]`.
3.  Right child is Cartesian Tree of `[5, 4]`.

**Validation:**
Check if it satisfies both Heap property and BST property.

## Advanced Variant 7: Splay Trees

**Definition:** A self-adjusting BST.
**Key Operation:** `Splay(node)`. Moves `node` to the root using rotations.
**Property:** Recently accessed elements are near the root.
**Amortized Complexity:** `O(log N)`.

**Rotations:**
1.  **Zig:** Single rotation (like AVL).
2.  **Zig-Zig:** Two rotations (same direction).
3.  **Zig-Zag:** Two rotations (opposite direction).

**Validation:**
Standard BST validation works. But we also care about **Balance**.
A Splay Tree can be a linked list (`O(N)` worst case), but the *amortized* cost is logarithmic.

## Advanced Variant 8: Treaps (Tree + Heap)

**Definition:** A Randomized BST.
-   **Keys:** Follow BST property.
-   **Priorities:** Randomly assigned. Follow Heap property.

**Why?**
Random priorities ensure the tree is balanced with high probability (`O(log N)` height).
It avoids the complex rotation logic of AVL/Red-Black trees.

**Validation:**
1.  Check BST property on Keys.
2.  Check Heap property on Priorities.

## Deep Dive: Persistent Binary Search Trees

**Problem:** We want to keep the history of the tree.
**Scenario:** "What was the state of the DB at 10:00 AM?"

**Implementation:**
**Path Copying:**
When modifying a node, we don't overwrite it.
We create a **copy** of the node, and a copy of its parent, all the way to the root.
The new root represents the new version. The old root represents the old version.
They share the unchanged subtrees.

**Space Complexity:** `O(log N)` extra space per update.
**Time Complexity:** `O(log N)` per update.

**Use Case:** Functional Programming (Haskell), Git, MVCC Databases.

## Dynamic Programming: Optimal Binary Search Trees

**Problem:** Given keys `k1, k2, ..., kn` and their frequencies `f1, f2, ..., fn`.
Construct a BST that minimizes the weighted search cost.
`Cost = Sum(depth(node) * frequency(node))`

**Intuition:**
High frequency nodes should be near the root.
This is similar to **Huffman Coding**, but the order of keys is fixed (must be BST).

**DP State:**
`dp[i][j]` = Min cost to construct OBST from keys `i` to `j`.
`dp[i][j] = Sum(freq[i...j]) + min(dp[i][r-1] + dp[r+1][j])` for `r` from `i` to `j`.

**Complexity:** `O(N^3)` time, `O(N^2)` space.
**Knuth's Optimization:** Reduces time to `O(N^2)`.

## Appendix C: Common BST Patterns

1.  **Inorder is Sorted:** The most useful property.
2.  **Preorder/Postorder Serialization:** A BST can be uniquely reconstructed from its Preorder traversal (if we know the bounds).
3.  **Successor/Predecessor:** Finding the next/prev value is `O(H)`.
4.  **LCA (Lowest Common Ancestor):** In a BST, the LCA of `p` and `q` is the first node `n` such that `min(p, q) < n < max(p, q)`.

## Conclusion

Validating a BST is the "Hello World" of Tree algorithms, but it teaches us the most important lesson in recursion: **Passing State**.
- In Preorder/Postorder, we often just pass the node.
- In `isValidBST`, we must pass the *context* (min/max constraints).

This pattern—passing constraints down the tree—appears everywhere, from **Alpha-Beta Pruning** in Game AI to **Constraint Satisfaction Problems** in AI planning.
