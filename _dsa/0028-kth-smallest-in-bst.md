---
title: "Kth Smallest Element in a BST"
day: 28
collection: dsa
categories:
  - dsa
tags:
  - bst
  - recursion
  - stack
  - medium
subdomain: "Tree Algorithms"
tech_stack: [Python, Java, C++]
scale: "O(H + k) time, O(H) space"
companies: [Uber, Google, Amazon, Facebook]
related_dsa_day: 28
related_ml_day: 28
related_speech_day: 28
---

**Finding the median or the 99th percentile is easy in a sorted array. Can we do it in a tree?**

## Problem Statement

Given the `root` of a Binary Search Tree (BST) and an integer `k`, return the `k`th smallest value (1-indexed) of all the values of the nodes in the tree.

**Example 1:**
```
   3
  / \
 1   4
  \
   2
```
**Input:** `root = [3,1,4,null,2]`, `k = 1`
**Output:** `1`

**Example 2:**
**Input:** `root = [5,3,6,2,4,null,null,1]`, `k = 3`
**Output:** `3`

**Constraints:**
-   The number of nodes in the tree is `n`.
-   `1 <= k <= n <= 10^4`
-   `0 <= Node.val <= 10^4`

## Intuition

The defining property of a BST is:
> Left Subtree < Root < Right Subtree

If we perform an **Inorder Traversal** (Left -> Root -> Right), we visit the nodes in **sorted ascending order**.
So, the problem reduces to: "Perform an Inorder traversal and stop at the `k`th node."

## Approach 1: Recursive Inorder Traversal

We can traverse the entire tree, store the elements in a list, and return `list[k-1]`.

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.result = []
        
        def inorder(node):
            if not node: return
            inorder(node.left)
            self.result.append(node.val)
            inorder(node.right)
            
        inorder(root)
        return self.result[k-1]
```

**Complexity:**
-   **Time:** \(O(N)\). We visit every node.
-   **Space:** \(O(N)\). We store every node.

## Approach 2: Iterative Inorder (Optimal)

We don't need to visit the whole tree. We can stop as soon as we find the `k`th element.
Using a Stack, we can simulate the recursion and return early.

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        curr = root
        
        while stack or curr:
            # 1. Go as left as possible
            while curr:
                stack.append(curr)
                curr = curr.left
            
            # 2. Process node
            curr = stack.pop()
            k -= 1
            if k == 0:
                return curr.val
            
            # 3. Go right
            curr = curr.right
            
        return -1 # Should not reach here
```

**Complexity:**
-   **Time:** \(O(H + k)\). We go down the height \(H\) to reach the leftmost node, then process \(k\) nodes.
-   **Space:** \(O(H)\) for the stack.

## Follow-up: Frequent Inserts/Deletes

**Question:** What if the BST is modified often (insert/delete operations) and you need to find the kth smallest frequently? How would you optimize the `kthSmallest` operation?

**Answer:**
The \(O(H+k)\) approach is too slow if \(k\) is large (e.g., \(N/2\)).
We can optimize this to \(O(H)\) by **Augmenting the BST**.
Each node should store a new field: `count` (size of the subtree rooted at this node).

**Algorithm:**
Let `left_count = node.left.count` (or 0 if null).
1.  If `k == left_count + 1`: The current node is the answer.
2.  If `k <= left_count`: The answer is in the left subtree. Recurse `left` with `k`.
3.  If `k > left_count + 1`: The answer is in the right subtree. Recurse `right` with `k - (left_count + 1)`.

**Complexity:**
-   **Time:** \(O(H)\) (Logarithmic).
-   **Space:** \(O(N)\) to store the counts.

## Real-World Application: Database Indexing

This is exactly how databases (like PostgreSQL or MySQL) implement `OFFSET` and `LIMIT`.
-   B-Trees store counts in internal nodes.
-   `SELECT * FROM users ORDER BY age LIMIT 1 OFFSET 1000` doesn't scan 1000 rows. It traverses the B-Tree using the counts to jump directly to the 1001st entry.

## Connections to ML Systems

In **Ranking Systems** (Day 28 ML), we often need to retrieve the "Top K" items.
While we usually use Heaps for "Top K", BSTs (or Balanced BSTs) are useful when we need to support dynamic updates and arbitrary rank queries (e.g., "What is the rank of this item?").

## Approach 3: Morris Traversal (O(1) Space)

Can we do this without a stack or recursion? Yes, using **Morris Traversal**.
This algorithm modifies the tree structure temporarily (threading) to traverse it, then restores it.

**Algorithm:**
1.  Initialize `curr` as `root`.
2.  While `curr` is not NULL:
    -   If `curr.left` is NULL:
        -   **Visit(curr)** (Increment count).
        -   If count == k, return `curr.val`.
        -   `curr = curr.right`
    -   Else:
        -   Find the **predecessor** (rightmost node in left subtree).
        -   If `predecessor.right` is NULL:
            -   Make `predecessor.right = curr` (Create thread).
            -   `curr = curr.left`
        -   Else (`predecessor.right == curr`):
            -   `predecessor.right = NULL` (Remove thread).
            -   **Visit(curr)**.
            -   If count == k, return `curr.val`.
            -   `curr = curr.right`

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        curr = root
        count = 0
        
        while curr:
            if not curr.left:
                count += 1
                if count == k: return curr.val
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
                    count += 1
                    if count == k: return curr.val
                    curr = curr.right
        return -1
```

**Complexity:**
-   **Time:** \(O(N)\). Each edge is traversed at most 3 times.
-   **Space:** \(O(1)\). No stack, no recursion.

## Deep Dive: Augmented BST Implementation

The \(O(H)\) approach requires maintaining the `count` field.
Here is how you would implement the `insert` operation to maintain this invariant.

```python
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None
        self.count = 1  # Size of subtree

class BST:
    def insert(self, root, val):
        if not root:
            return TreeNode(val)
        
        if val < root.val:
            root.left = self.insert(root.left, val)
        else:
            root.right = self.insert(root.right, val)
            
        # Update count after insertion
        root.count = 1 + self.getSize(root.left) + self.getSize(root.right)
        return root

    def getSize(self, node):
        return node.count if node else 0

    def kthSmallest(self, root, k):
        left_count = self.getSize(root.left)
        
        if k == left_count + 1:
            return root.val
        elif k <= left_count:
            return self.kthSmallest(root.left, k)
        else:
            return self.kthSmallest(root.right, k - left_count - 1)
```

**Trade-off:**
-   **Pros:** \(O(\log N)\) query time.
-   **Cons:** Insertion/Deletion becomes slightly slower (constant factor) due to updating counts.
-   **Cons:** Extra \(O(N)\) space for the `count` field.

## Comparison with Other Data Structures

| Data Structure | Kth Smallest Time | Update Time | Space |
| :--- | :--- | :--- | :--- |
| **Sorted Array** | \(O(1)\) | \(O(N)\) | \(O(N)\) |
| **Min Heap** | \(O(K \log N)\) | \(O(\log N)\) | \(O(N)\) |
| **Standard BST** | \(O(N)\) | \(O(H)\) | \(O(N)\) |
| **Augmented BST** | \(O(H)\) | \(O(H)\) | \(O(N)\) |
| **Segment Tree** | \(O(\log N)\) | \(O(\log N)\) | \(O(N)\) |

**Why not a Heap?**
A Min-Heap gives access to the minimum in \(O(1)\). To find the Kth smallest, we must pop K times. This destroys the heap (or requires copying it). Complexity: \(O(K \log N)\).
If \(K \approx N\), this is \(O(N \log N)\), which is worse than the BST's \(O(N)\).

## Real-World Application: Order Statistics Trees

In **Trading Systems**, we often need the "Median Price" of the last 1000 trades.
-   An Augmented BST (Order Statistic Tree) allows us to insert new trades and query the median in \(O(\log N)\).
-   Python's `sortedcontainers` library implements this efficiently.

## Implementation in C++

C++ `std::set` is usually a Red-Black Tree, but it doesn't expose the subtree size.
However, the GCC Policy-Based Data Structures (PBDS) library does!

```cpp
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;

typedef tree<
    int,
    null_type,
    less<int>,
    rb_tree_tag,
    tree_order_statistics_node_update>
    ordered_set;

// Usage:
ordered_set os;
os.insert(10);
os.insert(20);
os.insert(5);

// find_by_order returns iterator to kth element (0-indexed)
cout << *os.find_by_order(1) << endl; // Output: 10
```

## Top Interview Questions

**Q1: What if K is invalid (k < 1 or k > N)?**
*Answer:* The problem constraints say \(1 \le k \le N\), so it's always valid. In a real system, we should throw an exception or return an error code.

**Q2: How does the Augmented BST handle duplicates?**
*Answer:*
Standard BSTs don't allow duplicates.
If we need duplicates, we can:
1.  Store a `frequency` count in each node.
2.  Use `less_equal` logic (put equal values to the right).
The `kthSmallest` logic needs to be adjusted to account for `frequency`.

**Q3: Can we optimize the Iterative approach if we run it multiple times?**
*Answer:*
If the tree structure is static, we can cache the Inorder traversal in an array.
If the tree changes, we are back to the Augmented BST solution.

## Deep Dive: The Iterator Pattern

The Iterative approach essentially implements a **BST Iterator**.
This is a common design pattern. Instead of finding the Kth element, we might want to iterate through the tree one by one.

```python
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        self._push_left(root)

    def _push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        node = self.stack.pop()
        self._push_left(node.right)
        return node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0

# Usage for Kth Smallest:
# iterator = BSTIterator(root)
# for _ in range(k):
#     val = iterator.next()
# return val
```

**Why is this better?**
-   **Memory Efficiency:** It only stores \(O(H)\) nodes.
-   **Lazy Evaluation:** It computes the next node only when asked. If we stop at \(K\), we don't process the rest of the tree (unlike the Recursive approach which might visit everything if not careful).
-   **Composability:** We can pass this iterator to other functions (e.g., "Merge two BSTs").

## Deep Dive: Handling Dynamic Updates (Rebalancing)

In the Augmented BST approach, we maintain `count`.
But what if the tree becomes skewed?
-   Insert `1, 2, 3, 4, 5`.
-   Tree becomes a linked list.
-   Height \(H = N\).
-   Query time becomes \(O(N)\).

**Solution:** Use a **Self-Balancing BST** (AVL Tree or Red-Black Tree).
When we rotate nodes to rebalance, we must update the `count` fields.

**Rotation Logic:**
```
    y          x
   / \        / \
  x   C  <-> A   y
 / \            / \
A   B          B   C
```
When rotating Right (y -> x):
1.  `x.count = y.count` (x takes y's place, so it has the same total size).
2.  `y.count = 1 + size(B) + size(C)` (y is now root of B and C).

This ensures that even with frequent updates, the height remains \(O(\log N)\), and our Kth Smallest query remains \(O(\log N)\).

## Deep Dive: Python Generators for Inorder

Python's `yield` keyword makes writing the iterator trivial.
This is arguably the most "Pythonic" way to solve the problem.

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder_gen(node):
            if not node: return
            yield from inorder_gen(node.left)
            yield node.val
            yield from inorder_gen(node.right)
            
        gen = inorder_gen(root)
        for _ in range(k):
            val = next(gen)
        return val
```

**Under the Hood:**
Python handles the stack frames for us. `yield from` delegates to a sub-generator.
**Performance:** Slightly slower than the manual stack due to generator overhead, but much cleaner code.

## Deep Dive: BST vs B-Tree (Database Indexing)

We mentioned databases use B-Trees. Why not BSTs?
1.  **Disk I/O:** BST nodes are scattered in memory. B-Tree nodes contain thousands of keys in a single block (Page).
2.  **Height:** A BST with \(N=10^9\) has height \(\approx 30\). A B-Tree with branching factor \(B=1000\) has height \(\approx 3\).
3.  **Locality:** B-Trees are cache-friendly.

**Relevance to Kth Smallest:**
In a B-Tree, each internal node stores the count of keys in its subtrees.
The logic is identical to our Augmented BST, just with \(M\) children instead of 2.

## Deep Dive: Threaded Binary Trees

Morris Traversal is based on the concept of **Threaded Binary Trees**.
In a standard BST, \(N+1\) pointers are NULL (the leaves). This is wasted space.
**Idea:** Use the NULL right pointers to point to the **Inorder Successor**.
**Idea:** Use the NULL left pointers to point to the **Inorder Predecessor**.

**Benefits:**
-   Traversals become purely iterative (no stack needed).
-   Finding the successor is \(O(1)\) (mostly).

**Drawbacks:**
-   Insertion/Deletion is more complex (need to update threads).
-   We need a bit flag to distinguish between a "Child Pointer" and a "Thread".

## Deep Dive: Why not a Segment Tree?

A Segment Tree can also solve "Kth Smallest" in \(O(\log N)\).
-   **Setup:** Map the value range \([0, 10^4]\) to the leaves of the segment tree.
-   **Node Value:** Count of elements in the range \([L, R]\).
-   **Query:** Similar to Augmented BST. If `left_child.count >= k`, go left. Else go right.

**Comparison:**
-   **Segment Tree:** Good if the *range of values* is small and fixed. Bad if values are sparse or floats.
-   **Augmented BST:** Good for arbitrary values. Space depends on number of elements, not value range.

## Deep Dive: Python Generators for Inorder

Python's `yield` keyword makes writing the iterator trivial.
This is arguably the most "Pythonic" way to solve the problem.

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder_gen(node):
            if not node: return
            yield from inorder_gen(node.left)
            yield node.val
            yield from inorder_gen(node.right)
            
        gen = inorder_gen(root)
        for _ in range(k):
            val = next(gen)
        return val
```

**Under the Hood:**
Python handles the stack frames for us. `yield from` delegates to a sub-generator.
**Performance:** Slightly slower than the manual stack due to generator overhead, but much cleaner code.

## Deep Dive: BST vs B-Tree (Database Indexing)

We mentioned databases use B-Trees. Why not BSTs?
1.  **Disk I/O:** BST nodes are scattered in memory. B-Tree nodes contain thousands of keys in a single block (Page).
2.  **Height:** A BST with \(N=10^9\) has height \(\approx 30\). A B-Tree with branching factor \(B=1000\) has height \(\approx 3\).
3.  **Locality:** B-Trees are cache-friendly.

**Relevance to Kth Smallest:**
In a B-Tree, each internal node stores the count of keys in its subtrees.
The logic is identical to our Augmented BST, just with \(M\) children instead of 2.

## Deep Dive: Threaded Binary Trees

Morris Traversal is based on the concept of **Threaded Binary Trees**.
In a standard BST, \(N+1\) pointers are NULL (the leaves). This is wasted space.
**Idea:** Use the NULL right pointers to point to the **Inorder Successor**.
**Idea:** Use the NULL left pointers to point to the **Inorder Predecessor**.

**Benefits:**
-   Traversals become purely iterative (no stack needed).
-   Finding the successor is \(O(1)\) (mostly).

**Drawbacks:**
-   Insertion/Deletion is more complex (need to update threads).
-   We need a bit flag to distinguish between a "Child Pointer" and a "Thread".

## Deep Dive: Why not a Segment Tree?

A Segment Tree can also solve "Kth Smallest" in \(O(\log N)\).
-   **Setup:** Map the value range \([0, 10^4]\) to the leaves of the segment tree.
-   **Node Value:** Count of elements in the range \([L, R]\).
-   **Query:** Similar to Augmented BST. If `left_child.count >= k`, go left. Else go right.

**Comparison:**
-   **Segment Tree:** Good if the *range of values* is small and fixed. Bad if values are sparse or floats.
-   **Augmented BST:** Good for arbitrary values. Space depends on number of elements, not value range.

## Deep Dive: Python Generators for Inorder

Python's `yield` keyword makes writing the iterator trivial.
This is arguably the most "Pythonic" way to solve the problem.

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder_gen(node):
            if not node: return
            yield from inorder_gen(node.left)
            yield node.val
            yield from inorder_gen(node.right)
            
        gen = inorder_gen(root)
        for _ in range(k):
            val = next(gen)
        return val
```

**Under the Hood:**
Python handles the stack frames for us. `yield from` delegates to a sub-generator.
**Performance:** Slightly slower than the manual stack due to generator overhead, but much cleaner code.

## Deep Dive: BST vs B-Tree (Database Indexing)

We mentioned databases use B-Trees. Why not BSTs?
1.  **Disk I/O:** BST nodes are scattered in memory. B-Tree nodes contain thousands of keys in a single block (Page).
2.  **Height:** A BST with \(N=10^9\) has height \(\approx 30\). A B-Tree with branching factor \(B=1000\) has height \(\approx 3\).
3.  **Locality:** B-Trees are cache-friendly.

**Relevance to Kth Smallest:**
In a B-Tree, each internal node stores the count of keys in its subtrees.
The logic is identical to our Augmented BST, just with \(M\) children instead of 2.

## Deep Dive: Threaded Binary Trees

Morris Traversal is based on the concept of **Threaded Binary Trees**.
In a standard BST, \(N+1\) pointers are NULL (the leaves). This is wasted space.
**Idea:** Use the NULL right pointers to point to the **Inorder Successor**.
**Idea:** Use the NULL left pointers to point to the **Inorder Predecessor**.

**Benefits:**
-   Traversals become purely iterative (no stack needed).
-   Finding the successor is \(O(1)\) (mostly).

**Drawbacks:**
-   Insertion/Deletion is more complex (need to update threads).
-   We need a bit flag to distinguish between a "Child Pointer" and a "Thread".

## Deep Dive: Why not a Segment Tree?

A Segment Tree can also solve "Kth Smallest" in \(O(\log N)\).
-   **Setup:** Map the value range \([0, 10^4]\) to the leaves of the segment tree.
-   **Node Value:** Count of elements in the range \([L, R]\).
-   **Query:** Similar to Augmented BST. If `left_child.count >= k`, go left. Else go right.

**Comparison:**
-   **Segment Tree:** Good if the *range of values* is small and fixed. Bad if values are sparse or floats.
-   **Augmented BST:** Good for arbitrary values. Space depends on number of elements, not value range.

## Deep Dive: Python Generators for Inorder

Python's `yield` keyword makes writing the iterator trivial.
This is arguably the most "Pythonic" way to solve the problem.

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder_gen(node):
            if not node: return
            yield from inorder_gen(node.left)
            yield node.val
            yield from inorder_gen(node.right)
            
        gen = inorder_gen(root)
        for _ in range(k):
            val = next(gen)
        return val
```

**Under the Hood:**
Python handles the stack frames for us. `yield from` delegates to a sub-generator.
**Performance:** Slightly slower than the manual stack due to generator overhead, but much cleaner code.

## Deep Dive: BST vs B-Tree (Database Indexing)

We mentioned databases use B-Trees. Why not BSTs?
1.  **Disk I/O:** BST nodes are scattered in memory. B-Tree nodes contain thousands of keys in a single block (Page).
2.  **Height:** A BST with \(N=10^9\) has height \(\approx 30\). A B-Tree with branching factor \(B=1000\) has height \(\approx 3\).
3.  **Locality:** B-Trees are cache-friendly.

**Relevance to Kth Smallest:**
In a B-Tree, each internal node stores the count of keys in its subtrees.
The logic is identical to our Augmented BST, just with \(M\) children instead of 2.

## Deep Dive: Threaded Binary Trees

Morris Traversal is based on the concept of **Threaded Binary Trees**.
In a standard BST, \(N+1\) pointers are NULL (the leaves). This is wasted space.
**Idea:** Use the NULL right pointers to point to the **Inorder Successor**.
**Idea:** Use the NULL left pointers to point to the **Inorder Predecessor**.

**Benefits:**
-   Traversals become purely iterative (no stack needed).
-   Finding the successor is \(O(1)\) (mostly).

**Drawbacks:**
-   Insertion/Deletion is more complex (need to update threads).
-   We need a bit flag to distinguish between a "Child Pointer" and a "Thread".

## Deep Dive: Why not a Segment Tree?

A Segment Tree can also solve "Kth Smallest" in \(O(\log N)\).
-   **Setup:** Map the value range \([0, 10^4]\) to the leaves of the segment tree.
-   **Node Value:** Count of elements in the range \([L, R]\).
-   **Query:** Similar to Augmented BST. If `left_child.count >= k`, go left. Else go right.

**Comparison:**
-   **Segment Tree:** Good if the *range of values* is small and fixed. Bad if values are sparse or floats.
-   **Augmented BST:** Good for arbitrary values. Space depends on number of elements, not value range.

## Deep Dive: Python Generators for Inorder

Python's `yield` keyword makes writing the iterator trivial.
This is arguably the most "Pythonic" way to solve the problem.

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder_gen(node):
            if not node: return
            yield from inorder_gen(node.left)
            yield node.val
            yield from inorder_gen(node.right)
            
        gen = inorder_gen(root)
        for _ in range(k):
            val = next(gen)
        return val
```

**Under the Hood:**
Python handles the stack frames for us. `yield from` delegates to a sub-generator.
**Performance:** Slightly slower than the manual stack due to generator overhead, but much cleaner code.

## Deep Dive: BST vs B-Tree (Database Indexing)

We mentioned databases use B-Trees. Why not BSTs?
1.  **Disk I/O:** BST nodes are scattered in memory. B-Tree nodes contain thousands of keys in a single block (Page).
2.  **Height:** A BST with \(N=10^9\) has height \(\approx 30\). A B-Tree with branching factor \(B=1000\) has height \(\approx 3\).
3.  **Locality:** B-Trees are cache-friendly.

**Relevance to Kth Smallest:**
In a B-Tree, each internal node stores the count of keys in its subtrees.
The logic is identical to our Augmented BST, just with \(M\) children instead of 2.

## Deep Dive: Threaded Binary Trees

Morris Traversal is based on the concept of **Threaded Binary Trees**.
In a standard BST, \(N+1\) pointers are NULL (the leaves). This is wasted space.
**Idea:** Use the NULL right pointers to point to the **Inorder Successor**.
**Idea:** Use the NULL left pointers to point to the **Inorder Predecessor**.

**Benefits:**
-   Traversals become purely iterative (no stack needed).
-   Finding the successor is \(O(1)\) (mostly).

**Drawbacks:**
-   Insertion/Deletion is more complex (need to update threads).
-   We need a bit flag to distinguish between a "Child Pointer" and a "Thread".

## Deep Dive: Why not a Segment Tree?

A Segment Tree can also solve "Kth Smallest" in \(O(\log N)\).
-   **Setup:** Map the value range \([0, 10^4]\) to the leaves of the segment tree.
-   **Node Value:** Count of elements in the range \([L, R]\).
-   **Query:** Similar to Augmented BST. If `left_child.count >= k`, go left. Else go right.

**Comparison:**
-   **Segment Tree:** Good if the *range of values* is small and fixed. Bad if values are sparse or floats.
-   **Augmented BST:** Good for arbitrary values. Space depends on number of elements, not value range.

## Deep Dive: Time and Space Complexity Analysis

Let's rigorously analyze the complexity.

**Recursive Approach:**
-   **Time:** We visit nodes until we hit `k`.
    -   Best Case (k=1): \(O(1)\) (if we are lucky and root has no left child).
    -   Worst Case (k=N): \(O(N)\).
    -   Average Case: \(O(N)\).
-   **Space:** Recursion stack.
    -   Balanced Tree: \(O(\log N)\).
    -   Skewed Tree: \(O(N)\).

**Iterative Approach:**
-   **Time:** \(O(H + k)\).
    -   We traverse down to the leftmost node: \(O(H)\).
    -   We then pop `k` times. Each pop might involve pushing right children.
    -   Amortized cost of `next()` is \(O(1)\).
    -   Total: \(O(H + k)\).
-   **Space:** \(O(H)\).

**Augmented BST:**
-   **Time:** \(O(H)\).
    -   At each step, we go down one level.
    -   We perform constant work (comparisons).
    -   Total steps = Height.
-   **Space:** \(O(N)\) to store the `count` in every node.

## Deep Dive: Why Inorder? (A Proof)

Why does Inorder traversal yield sorted values?
**Theorem:** For any BST, Inorder traversal visits nodes in non-decreasing order.

**Proof (by Induction):**
1.  **Base Case:** Empty tree. Sequence is empty (sorted).
2.  **Inductive Step:**
    -   Assume true for subtrees of height \(h\).
    -   Consider tree of height \(h+1\) with root \(R\), left subtree \(L\), right subtree \(R_{ight}\).
    -   **BST Property:** \(\forall x \in L, x < R\). \(\forall y \in R_{ight}, y > R\).
    -   **Inorder:** `Inorder(L) + [R] + Inorder(R_{ight})`.
    -   By hypothesis, `Inorder(L)` is sorted. `Inorder(R_{ight})` is sorted.
    -   Since all elements in \(L\) are smaller than \(R\), and all elements in \(R_{ight}\) are larger than \(R\), the concatenation is sorted.

## Implementation in Java

Java's `Stack` class is legacy. Use `Deque` (ArrayDeque).

```java
class Solution {
    public int kthSmallest(TreeNode root, int k) {
        Deque<TreeNode> stack = new ArrayDeque<>();
        TreeNode curr = root;
        
        while (curr != null || !stack.isEmpty()) {
            while (curr != null) {
                stack.push(curr);
                curr = curr.left;
            }
            
            curr = stack.pop();
            k--;
            if (k == 0) {
                return curr.val;
            }
            
            curr = curr.right;
        }
        return -1;
    }
}
```

## Key Takeaways

-   **Inorder Traversal** of a BST gives sorted order.
-   **Iterative Traversal** allows early exit, saving time.
-   **Augmented Trees** allow \(O(\log N)\) selection, at the cost of complex maintenance during inserts/deletes.

---

**Originally published at:** [arunbaby.com/dsa/0028-kth-smallest-in-bst](https://www.arunbaby.com/dsa/0028-kth-smallest-in-bst/)

*If you found this helpful, consider sharing it with others who might benefit.*


