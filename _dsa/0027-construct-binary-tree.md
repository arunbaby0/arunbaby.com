---
title: "Construct Binary Tree from Preorder and Inorder Traversal"
day: 27
collection: dsa
categories:
  - dsa
tags:
  - binary-tree
  - recursion
  - hash-table
  - stack
  - medium
subdomain: "Tree Algorithms"
tech_stack: [Python, Java, C++]
scale: "O(N) time, O(N) space"
companies: [Amazon, Microsoft, Bloomberg, Google, Apple]
related_ml_day: 27
related_speech_day: 27
---

**Given two arrays, can you rebuild the original tree? It's like solving a jigsaw puzzle where the pieces are numbers.**

## Problem Statement

Given two integer arrays `preorder` and `inorder` where `preorder` is the preorder traversal of a binary tree and `inorder` is the inorder traversal of the same tree, construct and return the binary tree.

**Example 1:**
```
    3
   / \
  9  20
    /  \
   15   7
```
**Input:** `preorder = [3,9,20,15,7]`, `inorder = [9,3,15,20,7]`
**Output:** `[3,9,20,null,null,15,7]`

**Example 2:**
**Input:** `preorder = [-1]`, `inorder = [-1]`
**Output:** `[-1]`

**Constraints:**
- `1 <= preorder.length <= 3000`
- `inorder.length == preorder.length`
- `-3000 <= preorder[i], inorder[i] <= 3000`
- `preorder` and `inorder` consist of **unique** values.
- Each value of `inorder` also appears in `preorder`.
- `preorder` is guaranteed to be the preorder traversal of the tree.
- `inorder` is guaranteed to be the inorder traversal of the tree.

## Intuition

To reconstruct a tree, we need to know:
1.  **Who is the Root?**
2.  **What is in the Left Subtree?**
3.  **What is in the Right Subtree?**

Let's look at the properties of the traversals:
-   **Preorder (Root -> Left -> Right):** The **first element** is always the root.
-   **Inorder (Left -> Root -> Right):** The root splits the array into the left subtree (values to the left of root) and the right subtree (values to the right of root).

**Visualizing the Split:**
`preorder = [3, 9, 20, 15, 7]`
`inorder  = [9, 3, 15, 20, 7]`

1.  **Root:** `preorder[0]` is `3`.
2.  **Find Root in Inorder:** `3` is at index `1` in `inorder`.
3.  **Left Subtree:** Everything to the left of `3` in `inorder` (`[9]`).
4.  **Right Subtree:** Everything to the right of `3` in `inorder` (`[15, 20, 7]`).

Now we have the sizes.
-   Left Subtree Size: 1 node.
-   Right Subtree Size: 3 nodes.

We can recursively apply this logic to the `preorder` array to find the corresponding left and right segments.

## Approach 1: Recursive Slicing (Naive)

We can implement the intuition directly by slicing the arrays.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        
        # 1. The first element of preorder is the root
        root_val = preorder[0]
        root = TreeNode(root_val)
        
        # 2. Find the root in inorder to split left/right
        mid = inorder.index(root_val)
        
        # 3. Recursively build
        # Left: preorder[1:mid+1], inorder[:mid]
        # Right: preorder[mid+1:], inorder[mid+1:]
        root.left = self.buildTree(preorder[1:mid+1], inorder[:mid])
        root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:])
        
        return root
```

**Complexity Analysis:**
-   **Time:** \(O(N^2)\). In the worst case (skewed tree), `inorder.index()` takes \(O(N)\) and we do it \(N\) times. Also, array slicing takes \(O(N)\).
-   **Space:** \(O(N^2)\) due to creating new arrays for every recursive call.

## Approach 2: Optimization with Hash Map (Optimal Recursion)

We can optimize two things:
1.  **Finding the Root:** Use a Hash Map to store `val -> index` for `inorder`. This makes lookup \(O(1)\).
2.  **Slicing:** Instead of creating new arrays, pass `left` and `right` pointers (indices) for the current range in `inorder`. We also track the current index in `preorder`.

**Algorithm:**
1.  Build a map `inorder_map = {val: index}`.
2.  Keep a global variable `preorder_index` starting at 0.
3.  Define `helper(left, right)` which builds a subtree using `inorder[left:right+1]`.
4.  Get `root_val = preorder[preorder_index]`. Increment `preorder_index`.
5.  Get `inorder_index` from the map.
6.  Recursively build left child with range `(left, inorder_index - 1)`.
7.  Recursively build right child with range `(inorder_index + 1, right)`.

### Implementation

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # Map for O(1) lookup
        inorder_map = {val: idx for idx, val in enumerate(inorder)}
        
        # Use a mutable container (list) or class variable for the index
        # so it updates across recursive calls
        self.preorder_index = 0
        
        def array_to_tree(left: int, right: int) -> Optional[TreeNode]:
            # Base case: no elements to construct the tree
            if left > right:
                return None
            
            # Select the preorder_index element as the root and increment it
            root_val = preorder[self.preorder_index]
            self.preorder_index += 1
            root = TreeNode(root_val)
            
            # Root splits inorder list into left and right subtrees
            inorder_index = inorder_map[root_val]
            
            # Recursion
            # IMPORTANT: We must build the LEFT subtree first because
            # preorder traversal visits Left after Root.
            root.left = array_to_tree(left, inorder_index - 1)
            root.right = array_to_tree(inorder_index + 1, right)
            
            return root
        
        return array_to_tree(0, len(preorder) - 1)
```

## Approach 3: Iterative Solution (Stack)

Recursion uses the system stack. We can simulate this with an explicit stack.
This is tricky but insightful.

**Intuition:**
1.  Keep pushing nodes from `preorder` onto the stack. These are potential left children.
2.  If the current node equals the `inorder` head, it means we have finished the left subtree. Pop from stack and switch to the right child.

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder: return None
        
        root = TreeNode(preorder[0])
        stack = [root]
        inorder_index = 0
        
        for i in range(1, len(preorder)):
            pre_val = preorder[i]
            node = stack[-1]
            
            if node.val != inorder[inorder_index]:
                # We are still going down the left branch
                node.left = TreeNode(pre_val)
                stack.append(node.left)
            else:
                # We hit the leftmost node. 
                # Pop until we find the node that this new value is a RIGHT child of.
                while stack and stack[-1].val == inorder[inorder_index]:
                    node = stack.pop()
                    inorder_index += 1
                
                node.right = TreeNode(pre_val)
                stack.append(node.right)
                
        return root
```

### Deep Dive: Skewed Trees Analysis

Understanding skewed trees helps debug recursion depth issues.

### 1. Left-Skewed Tree (`1 -> 2 -> 3`)
-   **Preorder:** `[1, 2, 3]`
-   **Inorder:** `[3, 2, 1]`
-   **Execution:**
    1.  Root `1`. Split Inorder at `1`. Left=`[3, 2]`. Right=`[]`.
    2.  Recurse Left. Root `2`. Split Inorder at `2`. Left=`[3]`. Right=`[]`.
    3.  Recurse Left. Root `3`. Split Inorder at `3`. Left=`[]`. Right=`[]`.
    -   **Stack Depth:** 3 (Linear).

### 2. Right-Skewed Tree (`1 -> 2 -> 3`)
-   **Preorder:** `[1, 2, 3]`
-   **Inorder:** `[1, 2, 3]`
-   **Execution:**
    1.  Root `1`. Split Inorder at `1`. Left=`[]`. Right=`[2, 3]`.
    2.  Recurse Right. Root `2`. Split Inorder at `2`. Left=`[]`. Right=`[3]`.
    -   **Stack Depth:** 3 (Linear).

**Key Insight:**
In both cases, the recursion depth is \(O(N)\).
For a balanced tree, the depth is \(O(\log N)\).
This is why `buildTree` is vulnerable to Stack Overflow on worst-case inputs (linked lists), even though the time complexity is the same.

## Deep Dive: Locality of Reference (Preorder vs Level Order)

Why do we prefer Preorder for serialization?
-   **Preorder:** `[Root, Left Subtree, Right Subtree]`.
    -   The Left Subtree is a contiguous block in the array.
    -   This is **Cache Friendly**. When we process the left child, its descendants are likely already in the CPU Cache.
-   **Level Order:** `[Root, Left, Right, LL, LR, RL, RR]`.
    -   Children are far away from parents.
    -   Grandchildren are even further.
    -   This causes **Cache Misses** during reconstruction.

### Deep Dive: Visualizing the Iterative Approach

The iterative solution is notoriously hard to understand. Let's trace it with an example.
`Preorder: [3, 9, 20, 15, 7]`
`Inorder:  [9, 3, 15, 20, 7]`

**State:** `Stack = []`, `inorder_idx = 0` (`9`)

1.  **Process 3 (Root):**
    -   Push `3` to Stack. `Stack=[3]`.
    -   `3 != Inorder[0] (9)`.
    -   **Meaning:** `3` is not a leaf (or at least, not the leftmost leaf). We keep going left.

2.  **Process 9:**
    -   `9` is left child of `3`.
    -   Push `9`. `Stack=[3, 9]`.
    -   `9 == Inorder[0] (9)`.
    -   **Meaning:** We hit the leftmost node! We can't go left anymore.

3.  **Process 20:**
    -   `Stack Top (9) == Inorder (9)`.
    -   Pop `9`. `Stack=[3]`. `inorder_idx` -> `1` (`3`).
    -   `Stack Top (3) == Inorder (3)`.
    -   Pop `3`. `Stack=[]`. `inorder_idx` -> `2` (`15`).
    -   **Meaning:** We are backtracking. We finished `9`'s subtree. We finished `3`'s left subtree.
    -   Now `20` must be the **Right Child** of the last popped node (`3`).
    -   Push `20`. `Stack=[20]`.

4.  **Process 15:**
    -   `20 != Inorder (15)`.
    -   `15` is left child of `20`.
    -   Push `15`. `Stack=[20, 15]`.

**Key Insight:**
The stack stores the **"Spine"** of the left branch.
When `stack.top() == inorder[idx]`, it means we have finished the left child and need to backtrack to find the parent who needs a right child.

## Complexity Analysis

-   **Time Complexity:** \(O(N)\).
    -   We visit every node exactly once.
    -   In the iterative approach, each node is pushed and popped exactly once.
-   **Space Complexity:** \(O(N)\).
    -   Hash Map stores \(N\) entries.
    -   Stack stores \(O(H)\) nodes.

## Variant 1: Construct from Inorder and Postorder

**Problem:** Given `inorder` and `postorder`, construct the tree.
**Difference:**
-   **Postorder (Left -> Right -> Root):** The **last element** is the root.
-   We must process the `postorder` array from **right to left**.
-   We must build the **Right** subtree before the Left subtree (because traversing backwards from the end of Postorder, we hit Right children first).

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        inorder_map = {val: idx for idx, val in enumerate(inorder)}
        self.post_idx = len(postorder) - 1
        
        def helper(left, right):
            if left > right: return None
            
            root_val = postorder[self.post_idx]
            self.post_idx -= 1
            root = TreeNode(root_val)
            
            idx = inorder_map[root_val]
            
            # Build Right first!
            root.right = helper(idx + 1, right)
            root.left = helper(left, idx - 1)
            return root
            
        return helper(0, len(inorder) - 1)
```

## Variant 2: Construct from Preorder and Postorder

**Problem:** Given `preorder` and `postorder`.
**Ambiguity:** If a node has only one child, we don't know if it's left or right.
Example: `1 -> 2`.
Pre: `[1, 2]`. Post: `[2, 1]`.
Is `2` the left child of `1`? Or the right child?
Usually, we assume it's the **Left** child to make it unique.

**Logic:**
-   `root = preorder[0]`
-   `left_child_root = preorder[1]`
-   Find `left_child_root` in `postorder`. This marks the boundary of the left subtree in `postorder`.

## Deep Dive: The Challenge of Iterative Postorder

We showed Iterative Preorder (easy). Iterative Postorder is much harder.
**Why?**
In Preorder (Root-Left-Right), we process the Root immediately.
In Postorder (Left-Right-Root), we must visit Left, then Right, and *only then* process Root.
When we pop a node from the stack, we don't know if we are coming up from the Left child (so go Right) or from the Right child (so process Root).

**Solution:**
1.  **Two Stacks:** Reverse Preorder (Root-Right-Left) -> Postorder (Left-Right-Root).
2.  **One Stack + Pointers:** Keep track of `last_visited` node to know where we came from.

## Deep Dive: Handling Duplicates (The Impossible Case)

Why did the problem statement say "unique values"?
Consider `Preorder = [1, 1]` and `Inorder = [1, 1]`.
-   Root is `1`.
-   Inorder split: Left=`[]`? or Left=`[1]`?
-   We don't know if the second `1` is the left child or right child.
-   **Conclusion:** You cannot uniquely reconstruct a tree with duplicates using standard traversals unless you have **Node IDs** or **Null Markers**.

## Deep Dive: The Ambiguity of Preorder + Postorder

Why can't we uniquely reconstruct a tree from Preorder and Postorder?

Consider two trees:
**Tree A:**
```
    1
   /
  2
```
`Pre: [1, 2]`
`Post: [2, 1]`

**Tree B:**
```
    1
     \
      2
```
`Pre: [1, 2]`
`Post: [2, 1]`

The traversals are **identical**.
-   In Preorder, `2` comes after `1`, so it's a child.
-   In Postorder, `2` comes before `1`, so it's a child.
-   But neither tells us *which* child (Left or Right).

**Resolution:**
To solve LeetCode "Construct Binary Tree from Preorder and Postorder Traversal", we must **assume** that if a node has only one child, it is the **Left** child. With this constraint, the solution becomes unique.

## Variant 3: Construct BST from Preorder

**Problem:** Given `preorder` of a **Binary Search Tree**.
**Optimization:** We don't need `inorder`. We know `inorder` is just `sorted(preorder)`.
But we can do better than \(O(N \log N)\) sorting. We can do \(O(N)\).

**Method:** Use the **Upper Bound** constraint (similar to "Validate BST").
-   Root can range `(-inf, inf)`.
-   Left child range `(-inf, root.val)`.
-   Right child range `(root.val, inf)`.

```python
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        self.idx = 0
        
        def helper(upper_bound=float('inf')):
            if self.idx == len(preorder) or preorder[self.idx] > upper_bound:
                return None
            
            root_val = preorder[self.idx]
            self.idx += 1
            root = TreeNode(root_val)
            
            root.left = helper(root_val)
            root.right = helper(upper_bound)
            
            return root
            
        return helper()

## Deep Dive: Trampolines (Fixing Recursion in Python)

Since Python doesn't have Tail Call Optimization, we can implement a "Trampoline".
A trampoline runs a loop that iteratively calls functions returned by the recursive function.

```python
import types

def trampoline(f):
    def wrapped(*args, **kwargs):
        g = f(*args, **kwargs)
        while isinstance(g, types.GeneratorType):
            g = next(g)
        return g
    return wrapped

# Recursive function must yield the next call
def factorial(n, acc=1):
    if n == 0: return acc
    yield factorial(n-1, n*acc)

# Usage
print(trampoline(factorial)(10000)) # No Stack Overflow!
```

**Relevance:** You can wrap the `buildTree` helper in a trampoline to make it stack-safe without rewriting it as iterative.

## Deep Dive: Mathematical Proof of Uniqueness

**Theorem:** A binary tree is uniquely determined by its Preorder and Inorder traversals.

**Proof (by Induction):**
1.  **Base Case:** \(N=0\) (Empty) or \(N=1\) (Single Node). Trivial.
2.  **Inductive Step:** Assume it holds for \(k < N\).
    -   Preorder: `[Root, L_1...L_k, R_1...R_m]`
    -   Inorder: `[L_1...L_k, Root, R_1...R_m]`
    -   The `Root` is fixed (first element of Preorder).
    -   The `Root` splits Inorder into two unique sets \(L\) and \(R\).
    -   Since values are unique, \(L\) and \(R\) are disjoint.
    -   The size of \(L\) is fixed. This uniquely determines the split point in Preorder.
    -   By the inductive hypothesis, the Left Subtree (size \(k\)) and Right Subtree (size \(m\)) are unique.
    -   Therefore, the whole tree is unique.

**Counter-Example (Preorder + Postorder):**
-   Preorder: `Root, Left, Right`
-   Postorder: `Left, Right, Root`
-   If `Right` is empty:
    -   Pre: `Root, Left`
    -   Post: `Left, Root`
-   If `Left` is empty:
    -   Pre: `Root, Right`
    -   Post: `Right, Root`
-   The sequences look identical if we rename `Left` to `Right`. Hence, ambiguity.

## Deep Dive: Skewed Trees Analysiszation Formats

We used a simple string format `"1,2,#,#,3"`. In production, we need efficiency.

### 1. JSON (`{"val": 1, "left": ...}`)
-   **Pros:** Human readable. Easy to debug.
-   **Cons:** Verbose. "val", "left", "right" keys are repeated millions of times. Slow parsing.

### 2. Binary (Protobuf / FlatBuffers)
-   **Idea:** Define a schema.
    ```protobuf
    message Node {
      int32 val = 1;
      Node left = 2;
      Node right = 3;
    }
    ```
-   **Pros:** Extremely compact. Fast. Type-safe.
-   **Cons:** Requires schema compilation. Not human readable.

### 3. Array-based (Heap Layout)
-   **Idea:** If the tree is complete (like a Heap), we can store it in an array.
    -   Root at `i`.
    -   Left at `2*i + 1`.
    -   Right at `2*i + 2`.
-   **Pros:** Zero pointer overhead. Cache friendly.
-   **Cons:** Wastes space for sparse trees (skewed trees).

## Implementation in C++

Python hides the complexity of memory management. In C++, we must be careful.

```cpp
#include <vector>
#include <unordered_map>
#include <stack>

using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    unordered_map<int, int> inorder_map;
    int preorder_index = 0;

    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        // 1. Build Map
        for(int i=0; i<inorder.size(); ++i) {
            inorder_map[inorder[i]] = i;
        }
        preorder_index = 0;
        return arrayToTree(preorder, 0, preorder.size() - 1);
    }

    TreeNode* arrayToTree(vector<int>& preorder, int left, int right) {
        if (left > right) return NULL;

        // 2. Pick root
        int root_val = preorder[preorder_index];
        preorder_index++;
        TreeNode* root = new TreeNode(root_val);

        // 3. Find split point
        int inorder_index = inorder_map[root_val];

        // 4. Recurse
        root->left = arrayToTree(preorder, left, inorder_index - 1);
        root->right = arrayToTree(preorder, inorder_index + 1, right);

        return root;
    }
};
```

**Memory Note:** In a real interview, ask if you need to `delete` the tree. If so, you need a Postorder traversal destructor.

## Advanced Variant 4: Construct N-ary Tree from Preorder

**Problem:** Given the `preorder` traversal of an N-ary tree (where each node has a list of children).
**Note:** We need more information than just preorder to reconstruct an N-ary tree uniquely. Usually, we are given `preorder` and the `number of children` for each node, or the serialization includes `null` markers.

**Scenario:** Serialization is `[1, [3, [5, 6]], 2, 4]`.
If we just have `[1, 3, 5, 6, 2, 4]`, we can't do it.
But if we have `[1, 3, 5, null, 6, null, null, 2, null, 4, null]`, we can.

**Algorithm:**
1.  Pop `val` from queue. Create `node`.
2.  While next value is not `null`, add `helper()` to `node.children`.
3.  If next is `null`, pop it and return `node`.

```python
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

class Solution:
    def deserialize(self, data: str) -> 'Node':
        if not data: return None
        tokens = deque(data.split(","))
        
        def helper():
            if not tokens: return None
            val = tokens.popleft()
            if val == "#": return None
            
            node = Node(int(val), [])
            while tokens and tokens[0] != "#":
                child = helper()
                if child:
                    node.children.append(child)
            
            if tokens: tokens.popleft() # Pop the '#' that ended this level
            return node
            
        return helper()
```

## Real-World Application: Serialization

This algorithm is fundamental to **Serialization and Deserialization**.
When you save a tree structure to a file (or send it over a network), you can't just save the pointers. You save the traversal.
To reconstruct it later, you need either:
1.  Preorder + Inorder (if unique values).
2.  Postorder + Inorder (if unique values).
3.  Preorder with `null` markers (Approach used in Day 26).

This specific problem (Preorder + Inorder) is often used in **Compiler Design** to reconstruct the Abstract Syntax Tree (AST) from parsed tokens.

## Connections to ML Systems

In **Decision Trees** (like Random Forest or XGBoost), the model is essentially a binary tree.
-   **Training:** We build the tree top-down (like Preorder). We pick a feature to split on (Root), then split the data into Left and Right.
-   **Inference:** We traverse the tree from Root to Leaf.

While we don't usually reconstruct Decision Trees from traversals, the *concept* of recursively partitioning data based on a "root" decision is identical.

## Interview Strategy

-   **Start with the Example:** Walk through the example manually. Draw the arrays and cross out numbers as you "use" them.
-   **Explain the "Why":** Why do we need Inorder? Because Preorder tells us *what* the root is, but not *how many* nodes are in the left child. Inorder gives us the size.
-   **Mention the Optimization:** Start with the \(O(N^2)\) slicing idea, then quickly pivot to "We can optimize the search with a Hash Map and the slicing with indices."
-   **Iterative:** Only mention the stack approach if asked or if you are very confident. It's easy to bug up.

## Implementation in Java

Java is verbose, but explicit.

```java
import java.util.HashMap;
import java.util.Map;

public class Solution {
    private Map<Integer, Integer> inorderMap;
    private int preorderIndex;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        inorderMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            inorderMap.put(inorder[i], i);
        }
        
        preorderIndex = 0;
        return arrayToTree(preorder, 0, preorder.length - 1);
    }

    private TreeNode arrayToTree(int[] preorder, int left, int right) {
        if (left > right) {
            return null;
        }

        int rootValue = preorder[preorderIndex++];
        TreeNode root = new TreeNode(rootValue);

        int inorderIndex = inorderMap.get(rootValue);

        // Recursive calls
        root.left = arrayToTree(preorder, left, inorderIndex - 1);
        root.right = arrayToTree(preorder, inorderIndex + 1, right);

        return root;
    }
}
```

## Top Interview Questions

**Q1: Can you solve this with O(1) space (excluding recursion stack)?**
*Answer:*
Yes, but it's very complex. You can use the **Morris Traversal** idea to modify the tree pointers temporarily, but reconstructing a tree while modifying it is dangerous.
The Iterative approach uses \(O(H)\) stack space.
The Recursive approach uses \(O(H)\) stack space + \(O(N)\) map space.
You *can* avoid the Map by searching linearly \(O(N)\), but that degrades time to \(O(N^2)\).
So, \(O(N)\) space is the practical lower bound for an \(O(N)\) time solution.

**Q2: What if the tree contains duplicate values?**
*Answer:*
If duplicates exist, the problem is **unsolvable** (ill-posed) with just Preorder and Inorder.
Example:
Tree 1: Root(1) -> Left(1). Pre: `[1, 1]`, In: `[1, 1]`.
Tree 2: Root(1) -> Right(1). Pre: `[1, 1]`, In: `[1, 1]`.
You cannot distinguish them. You would need unique IDs for nodes.

**Q3: Can you reconstruct a tree from Level Order and Inorder?**
*Answer:*
Yes.
1.  Root is `LevelOrder[0]`.
2.  Find Root in `Inorder`. Split into Left/Right sets.
3.  Filter `LevelOrder` into `LeftLevelOrder` (elements present in Left Inorder set) and `RightLevelOrder`.
4.  Recurse.
**Complexity:** \(O(N^2)\) because filtering the level order array takes \(O(N)\) at each step.

**Q4: Why do we need Inorder? Why isn't Preorder enough?**
*Answer:*
Preorder is `Root, Left, Right`.
It tells us the Root is first. But it doesn't tell us where `Left` ends and `Right` begins.
`[1, 2, 3]`. Is `2` left of `1`? Or right? Is `3` child of `2`?
Inorder gives us the boundary. `Left < Root < Right`.

## Deep Dive: The Danger of Recursion (Stack Overflow)

Why do some interviewers hate recursion?
Because the **Call Stack** is limited (usually 1MB - 8MB).

**Scenario:**
-   You have a skewed tree (linked list) of depth 100,000.
-   Each recursive call pushes a stack frame (return address, local variables).
-   **Result:** `RecursionError: maximum recursion depth exceeded` or `Segmentation Fault`.

**Mitigation:**
1.  **Tail Recursion Optimization (TRO):** Some languages (Scala, C++) optimize tail calls into loops. Python/Java **do not**.
2.  **Trampolines:** A technique to simulate TRO in Python (using generators/decorators).
3.  **Iterative Approach:** Always safe. Uses the **Heap** (which is GBs in size) instead of the Stack.

## Comparison of Approaches

| Feature | Recursive | Iterative (Stack) | Morris Traversal |
| :--- | :--- | :--- | :--- |
| **Time Complexity** | \(O(N)\) | \(O(N)\) | \(O(N)\) |
| **Space Complexity** | \(O(H)\) | \(O(H)\) | \(O(1)\) |
| **Code Simplicity** | High (5 lines) | Medium (20 lines) | Low (Complex logic) |
| **Stack Safety** | Low (Overflow risk) | High (Heap memory) | High (No stack) |
| **Tree Modification** | No | No | Yes (Temporary threads) |

**Verdict:**
-   **Competitions:** Use Recursive (fast to write).
-   **Production:** Use Iterative (safe).
-   **Embedded/Kernel:** Use Morris (memory constrained).

## Deep Dive: Morris Traversal (O(1) Space)

While not directly applicable to *reconstruction*, Morris Traversal is the gold standard for *traversing* a tree with O(1) space. It uses **Threading**.

**Idea:**
-   Make use of the `null` pointers in leaf nodes.
-   If a node has a left child, find the **rightmost** node in the left subtree (the "predecessor").
-   Make the predecessor's `right` pointer point to the current node.
-   This creates a temporary cycle (thread) that allows us to return to the root after finishing the left subtree.

```python
def morris_inorder(root):
    curr = root
    while curr:
        if not curr.left:
            print(curr.val)
            curr = curr.right
        else:
            # Find predecessor
            pre = curr.left
            while pre.right and pre.right != curr:
                pre = pre.right
            
            if not pre.right:
                # Create thread
                pre.right = curr
                curr = curr.left
            else:
                # Remove thread (restore tree)
                pre.right = None
                print(curr.val)
                curr = curr.right
```

## Further Reading

1.  **CLRS:** Introduction to Algorithms, Chapter 12 (Binary Search Trees).
2.  **Knuth:** The Art of Computer Programming, Vol 1 (Fundamental Algorithms).
3.  **LeetCode:** [105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
4.  **LeetCode:** [106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)
5.  **LeetCode:** [889. Construct Binary Tree from Preorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/)

## Key Takeaways

-   **Preorder** = Root first.
-   **Inorder** = Root in middle.
-   **Postorder** = Root last.
-   **Hash Map** reduces search from \(O(N)\) to \(O(1)\).
-   **Indices** avoid the overhead of array slicing.
-   **Ambiguity:** You generally need Inorder + (Preorder OR Postorder) to uniquely reconstruct a binary tree.

---

**Originally published at:** [arunbaby.com/dsa/0027-construct-binary-tree](https://www.arunbaby.com/dsa/0027-construct-binary-tree/)

*If you found this helpful, consider sharing it with others who might benefit.*


