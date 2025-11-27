---
title: "Lowest Common Ancestor of a Binary Tree"
day: 29
collection: dsa
categories:
  - dsa
tags:
  - binary tree
  - recursion
  - tree traversal
  - dfs
difficulty: Medium
related_dsa_day: 29
related_ml_day: 29
related_speech_day: 29
---

**"Find the point where two paths in a tree first meet."**

## 1. Problem Statement

Given a binary tree, find the **lowest common ancestor (LCA)** of two given nodes in the tree.

The lowest common ancestor is defined between two nodes `p` and `q` as the lowest node in T that has both `p` and `q` as descendants (where we allow **a node to be a descendant of itself**).

**Example:**
```
        3
       / \
      5   1
     / \ / \
    6  2 0  8
      / \
     7   4
```

- LCA(5, 1) = 3
- LCA(5, 4) = 5 (a node can be its own ancestor)
- LCA(6, 4) = 5

## 2. Recursive Solution (Most Intuitive)

**Intuition:**
- If the current node is NULL, return NULL.
- If the current node is `p` or `q`, return the current node.
- Recursively search left and right subtrees.
- If both subtrees return non-NULL, current node is the LCA.
- If only one subtree returns non-NULL, propagate it upward.

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # Base case: reached NULL or found one of the targets
        if not root or root == p or root == q:
            return root
        
        # Search in left and right subtrees
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        # If both sides found something, current node is LCA
        if left and right:
            return root
        
        # Otherwise, return whichever side found something
        return left if left else right
```

**Time Complexity:** \\(O(N)\\) where N is the number of nodes (we visit each node once).
**Space Complexity:** \\(O(H)\\) for recursion stack, where H is the height (\\(O(\log N)\\) for balanced, \\(O(N)\\) for skewed).

## 3. Path Storage Solution

**Idea:** Find the path from root to `p` and root to `q`, then find the last common node.

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def find_path(root, target, path):
            if not root:
                return False
            
            path.append(root)
            
            if root == target:
                return True
            
            if find_path(root.left, target, path) or find_path(root.right, target, path):
                return True
            
            path.pop()
            return False
        
        path_p, path_q = [], []
        find_path(root, p, path_p)
        find_path(root, q, path_q)
        
        lca = None
        for i in range(min(len(path_p), len(path_q))):
            if path_p[i] == path_q[i]:
                lca = path_p[i]
            else:
                break
        
        return lca
```

**Time Complexity:** \\(O(N)\\) (two DFS traversals).
**Space Complexity:** \\(O(N)\\) (storing paths).

## 4. Iterative Solution with Parent Pointers

**Idea:** Use a parent map to track each node's parent, then trace back from `p` and `q` to find the first common ancestor.

```python
from collections import deque

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # Build parent pointers using BFS
        parent = {root: None}
        queue = deque([root])
        
        while p not in parent or q not in parent:
            node = queue.popleft()
            if node.left:
                parent[node.left] = node
                queue.append(node.left)
            if node.right:
                parent[node.right] = node
                queue.append(node.right)
        
        # Collect all ancestors of p
        ancestors = set()
        while p:
            ancestors.add(p)
            p = parent[p]
        
        # Find first ancestor of q that's also an ancestor of p
        while q not in ancestors:
            q = parent[q]
        
        return q
```

**Time Complexity:** \\(O(N)\\).
**Space Complexity:** \\(O(N)\\) (parent map and ancestor set).

## 5. Edge Cases

```python
# Test cases
def test_lca():
    # Case 1: One node is ancestor of the other
    # LCA(5, 4) = 5
    
    # Case 2: Nodes in different subtrees
    # LCA(6, 0) = 3
    
    # Case 3: One node is root
    # LCA(3, 4) = 3
    
    # Case 4: Both nodes are same
    # LCA(5, 5) = 5
```

## Deep Dive: Why the Recursive Solution Works

The key insight is the **bottom-up propagation**:

**Case 1: `p` and `q` are in different subtrees**
```
      LCA
      / \
     p   q
```
- Left subtree returns `p`.
- Right subtree returns `q`.
- Since both are non-NULL, LCA is the current node.

**Case 2: `p` is ancestor of `q`**
```
      p
       \
        q
```
- When we hit `p`, we return `p` immediately.
- The subtree containing `q` also eventually returns `p` (propagated up).
- Since only one side returns non-NULL, we return `p`.

**Mathematical Proof:**
Let \\( T(n) \\) be a binary tree rooted at \\( n \\).
Define \\( \\text{LCA}(p, q) \\) as the deepest node \\( n \\) such that \\( p \\in T(n) \\) and \\( q \\in T(n) \\).

**Claim:** The recursive algorithm correctly finds \\( \\text{LCA}(p, q) \\).

**Proof by Induction:**
- **Base:** If \\( n = p \\) or \\( n = q \\) or \\( n = \\text{NULL} \\), return \\( n \\). Correct.
- **Inductive Step:** 
  - If \\( \\text{left} \\neq \\text{NULL} \\) and \\( \\text{right} \\neq \\text{NULL} \\), then \\( p \\) and \\( q \\) are in different subtrees. Thus, \\( n \\) is the LCA.
  - If only \\( \\text{left} \\neq \\text{NULL} \\), both \\( p \\) and \\( q \\) are in the left subtree, so we propagate the LCA from the left subtree.

## Deep Dive: LCA in a Binary Search Tree (BST)

If the tree is a BST, we can optimize using the BST property.

```python
def lowestCommonAncestor_BST(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
```

**Time Complexity:** \\(O(H)\\) where H is height.
**Space Complexity:** \\(O(1)\\) (iterative, no recursion).

**Why BST is Special:**
- If both `p` and `q` are smaller than `root`, LCA must be in the left subtree.
- If both are larger, LCA must be in the right subtree.
- Otherwise, `root` is the LCA (one is on each side, or `root` is one of them).

## Deep Dive: LCA in a Directed Acyclic Graph (DAG)

In a DAG, a node can have multiple parents. LCA becomes more complex.

**Approach: Topological Sort + DFS**
1. Find all ancestors of `p` using DFS.
2. Find all ancestors of `q` using DFS.
3. The LCA is the common ancestor with the maximum topological order (deepest).

**Time Complexity:** \\(O(V + E)\\) where V is vertices and E is edges.

## Deep Dive: Range Minimum Query (RMQ) and LCA

There's a deep connection: **LCA can be reduced to RMQ**.

**Euler Tour Technique:**
1. Perform a DFS and record the sequence of nodes (visiting each node when entering and leaving).
2. For each node, record its first occurrence in the Euler tour.
3. LCA(p, q) = Node with minimum depth in the Euler tour between first[p] and first[q].

**Example:**
```
Tree:      1
          / \
         2   3
        / \
       4   5

Euler Tour: [1, 2, 4, 2, 5, 2, 1, 3, 1]
Depths:     [0, 1, 2, 1, 2, 1, 0, 1, 0]
First occurrence: 1->0, 2->1, 3->7, 4->2, 5->4

LCA(4, 5):
  first[4] = 2, first[5] = 4
  Min depth in range [2, 4] is at index 3 (depth 1, node 2)
  LCA = 2
```

**With RMQ Preprocessing:**
- Preprocess the depth array with Sparse Table or Segment Tree.
- Answer LCA queries in \\(O(1)\\) after \\(O(N \log N)\\) preprocessing.

## Deep Dive: Tarjan's Offline LCA Algorithm

If we have many LCA queries offline (all queries known in advance), **Tarjan's algorithm** uses **Disjoint Set Union (DSU)**.

**Algorithm:**
```python
def tarjan_lca(root, queries):
    parent = {}
    ancestor = {}
    color = {}  # 0: white, 1: gray, 2: black
    result = {}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        parent[find(x)] = find(y)
    
    def dfs(node):
        parent[node] = node
        ancestor[node] = node
        color[node] = 1  # gray
        
        for child in [node.left, node.right]:
            if child:
                dfs(child)
                union(child, node)
                ancestor[find(node)] = node
        
        color[node] = 2  # black
        
        for (u, v) in queries:
            if u == node and color.get(v) == 2:
                result[(u, v)] = ancestor[find(v)]
            if v == node and color.get(u) == 2:
                result[(u, v)] = ancestor[find(u)]
    
    dfs(root)
    return result
```

**Time Complexity:** \\(O((N + Q) \cdot \alpha(N))\\) where Q is number of queries and \\(\alpha\\) is the inverse Ackermann function (nearly constant).

## Deep Dive: Lowest Common Ancestor of K Nodes

**Problem:** Find the LCA of K nodes \\( \{p_1, p_2, \ldots, p_k\} \\).

**Approach 1: Iterative LCA**
```python
def lca_of_k_nodes(root, nodes):
    lca = nodes[0]
    for i in range(1, len(nodes)):
        lca = lowestCommonAncestor(root, lca, nodes[i])
    return lca
```
**Time Complexity:** \\(O(K \cdot N)\\) in worst case.

**Approach 2: DFS with Counter**
```python
def lca_of_k_nodes_optimized(root, nodes):
    node_set = set(nodes)
    
    def dfs(node):
        if not node:
            return 0, None
        
        count = 1 if node in node_set else 0
        left_count, left_lca = dfs(node.left)
        right_count, right_lca = dfs(node.right)
        
        total_count = count + left_count + right_count
        
        if total_count == len(node_set) and not hasattr(dfs, 'lca'):
            dfs.lca = node
        
        return total_count, dfs.lca if hasattr(dfs, 'lca') else None
    
    _, lca = dfs(root)
    return lca
```
**Time Complexity:** \\(O(N)\\) single pass.

## Deep Dive: LCA with Node Values (Not References)

**Problem:** Given a tree and two integer values, find their LCA.

**Challenge:** We need to search for the nodes first.

```python
def lca_by_values(root, val1, val2):
    def find_lca(node):
        if not node or node.val == val1 or node.val == val2:
            return node
        
        left = find_lca(node.left)
        right = find_lca(node.right)
        
        if left and right:
            return node
        return left if left else right
    
    return find_lca(root)
```

**Caveat:** What if one value doesn't exist?
- The above code would return the other node as LCA (incorrect).
- **Fix:** Verify both values exist first with a separate traversal.

## Deep Dive: LCA in a Binary Tree with Parent Pointers

If each node has a `parent` pointer, the problem becomes **finding the intersection of two linked lists**.

```python
def lca_with_parent_pointers(p, q):
    def get_depth(node):
        depth = 0
        while node:
            depth += 1
            node = node.parent
        return depth
    
    depth_p = get_depth(p)
    depth_q = get_depth(q)
    
    # Move the deeper node up to the same level
    while depth_p > depth_q:
        p = p.parent
        depth_p -= 1
    
    while depth_q > depth_p:
        q = q.parent
        depth_q -= 1
    
    # Move both up until they meet
    while p != q:
        p = p.parent
        q = q.parent
    
    return p
```

**Time Complexity:** \\(O(H)\\).
**Space Complexity:** \\(O(1)\\).

## Comparison Table

| Approach | Time | Space | Pros | Cons |
|:---|:---|:---|:---|:---|
| **Recursive** | \\(O(N)\\) | \\(O(H)\\) | Elegant, simple | Recursion overhead |
| **Path Storage** | \\(O(N)\\) | \\(O(N)\\) | Easy to understand | Extra space for paths |
| **Parent Pointers (BFS)** | \\(O(N)\\) | \\(O(N)\\) | Iterative | Requires building parent map |
| **BST Optimized** | \\(O(H)\\) | \\(O(1)\\) | Fast for BST | Only works for BST |
| **Tarjan (Offline)** | \\(O((N+Q)\alpha(N))\\) | \\(O(N)\\) | Multiple queries | Requires all queries upfront |
| **RMQ Reduction** | \\(O(1)\\) query | \\(O(N \log N)\\) | Very fast queries | Complex preprocessing |

## Implementation in Other Languages

**C++:**
```cpp
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || root == p || root == q) return root;
        
        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p, q);
        
        if (left && right) return root;
        return left ? left : right;
    }
};
```

**Java:**
```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        
        if (left != null && right != null) return root;
        return left != null ? left : right;
    }
}
```

## Top Interview Questions

**Q1: What if the tree is very deep and we hit stack overflow?**
*Answer:*
Use the iterative solution with parent pointers or convert the recursive solution to iterative using an explicit stack.

**Q2: Can LCA be \\(O(1)\\) query time?**
*Answer:*
Yes, with \\(O(N \log N)\\) preprocessing using the RMQ reduction (Euler tour + Sparse Table).

**Q3: What if we're given a forest (multiple trees) instead of a single tree?**
*Answer:*
If `p` and `q` are not in the same tree, return NULL. Otherwise, find the root of their tree and apply LCA.

**Q4: How do you handle the case where one of the nodes doesn't exist?**
*Answer:*
Add a validation step to ensure both nodes exist in the tree before running LCA.

## Key Takeaways

1. **Recursive Solution is Elegant:** The post-order traversal naturally solves LCA.
2. **BST Optimization:** Leverage BST properties for \\(O(H)\\) time.
3. **RMQ Connection:** LCA and Range Minimum Query are equivalent problems.
4. **Offline Queries:** Tarjan's algorithm with DSU is optimal for batch queries.
5. **Parent Pointers:** Reduce to "intersection of two linked lists" problem.

## Summary

| Aspect | Insight |
|:---|:---|
| **Core Idea** | Find the deepest node that is an ancestor of both targets |
| **Key Trick** | Post-order DFS with bottom-up propagation |
| **BST Optimization** | Navigate by comparing values |
| **Advanced** | RMQ reduction for \\(O(1)\\) queries |

---

**Originally published at:** [arunbaby.com/dsa/0029-lowest-common-ancestor](https://www.arunbaby.com/dsa/0029-lowest-common-ancestor/)

*If you found this helpful, consider sharing it with others who might benefit.*


