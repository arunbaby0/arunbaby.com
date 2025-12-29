---
title: "Serialize and Deserialize Binary Tree"
day: 47
related_ml_day: 47
related_speech_day: 47
related_agents_day: 47
collection: dsa
categories:
 - dsa
tags:
 - tree
 - bfs
 - dfs
 - serialization
 - system-design
difficulty: Hard
subdomain: "Trees & Design"
tech_stack: Python
scale: "Handling 1M+ nodes efficiently"
companies: Uber, Google, Amazon, Facebook
---

**"Data is only useful if it can survive the journey from RAM to Disk and back again."**

## 1. Problem Statement

This is one of the most practical "System Design" questions disguised as a coding problem.
**Serialization** is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

**Example:**
``
 1
 / \
 2 3
 / \
 4 5
``
**Input**: `root = [1,2,3,null,null,4,5]`
**Output**: `[1,2,3,null,null,4,5]`

Constraints:
- The number of nodes in the tree is in the range `[0, 10^4]`.
- Value range: `-1000 <= Node.val <= 1000`.

---

## 2. Understanding the Problem

### 2.1 The Challenge of Ambiguity
If I tell you a tree has nodes `[1, 2, 3]`, that is ambiguous.
- Is 2 left or right of 1?
- Is 3 a child of 1 or 2?

Standard traversals (Inorder, Preorder) are not unique on their own.
- Preorder `[1, 2]` could be `1->left(2)` or `1->right(2)`.
- **Key Insight**: To make a traversal unique, we must record the **Null Pointers**.

If we record `[1, 2, #, #, #]` (where # is null), we know exactly that 1 has a left child 2, and 2 has no children.

### 2.2 System Design Context
In a real system (like DOM tree serialization or formatting a JSON response), we care about:
1. **Compactness**: The string shouldn't be 10x larger than the data.
2. **Streaming**: Can we start processing the root before we receive the leaves? (DFS allows this).
3. **Human Readability**: JSON is readable, Binary Protobuf is compact.

---

## 3. Approach 1: Preorder Traversal (DFS) -- The "Streaming" Way

This is the standard recursive solution. We visit the root, then recursively serialize the left subtree, then the right subtree.

**Serialization Strategy:**
- If Node is null: Append `"N,"` (or any sentinel).
- If Node exists: Append `str(val) + ","`.
- Recurse Left.
- Recurse Right.

Example for the tree above: `1,2,N,N,3,4,N,N,5,N,N,`

**Deserialization Strategy:**
- Split the string into a Queue (values list).
- Pop the first value.
- If "N": return None.
- Else: Create Node(val).
 - `node.left` = recurse()
 - `node.right` = recurse()
- Return Node.

Since the list is built in Preorder, the recursion naturally consumes items in the exact order they are needed to rebuild the tree top-down.

---

## 4. Approach 2: Level Order Traversal (BFS) -- The "Layered" Way

This is how LeetCode visualizes trees (e.g., `[1,2,3,null,null,4,5]`). It saves space if the tree is dense (complete binary tree) but wastes space if the tree is sparse/skewed.

**Strategy:**
- Use a Queue.
- Push Root.
- While Queue:
 - Pop Node.
 - If Node: Append val. Push Left, Push Right.
 - Else: Append "N".

**Deserialization:**
- This is trickier. You need a Queue for the *parent nodes* waiting for children.
- Root = List[0]. Push to Queue.
- Pointer `i = 1`.
- While Queue:
 - Parent = Pop().
 - Left = List[i]. If not "N", attach to Parent.left, Push Left.
 - Right = List[i+1]. If not "N", attach to Parent.right, Push Right.

**Verdict**: DFS (Approach 1) is generally cleaner to implement recursively. BFS is better if you want to verify the top of the tree without reading the whole string.

---

## 5. Implementation: Preorder DFS (Python)

We will implement the DFS approach because it handles skewed trees gracefully and the code is elegant.

``python
class Codec:

 def serialize(self, root):
 """Encodes a tree to a single string.
 
 :type root: TreeNode
 :rtype: str
 """
 path = []
 
 def dfs(node):
 if not node:
 path.append("#")
 return
 
 # Preorder: Root -> Left -> Right
 path.append(str(node.val))
 dfs(node.left)
 dfs(node.right)
 
 dfs(root)
 # Join with comma to handle multi-digit numbers
 return ",".join(path)

 def deserialize(self, data):
 """Decodes your encoded data to tree.
 
 :type data: str
 :rtype: TreeNode
 """
 # Create an iterator (or queue) for O(1) popping
 values = iter(data.split(","))
 
 def build():
 try:
 val = next(values)
 except StopIteration:
 return None
 
 # Base Case: Null pointer
 if val == "#":
 return None
 
 # Recursive Step
 node = TreeNode(int(val))
 node.left = build() # Consumes the next chunk of the stream
 node.right = build() # Consumes the remainder
 return node
 
 return build()
``

---

## 6. Testing Strategy

### Test Case 1: Skewed Tree (Linked List)
`1 -> 2 -> 3`
- Serialize: `1,2,3,N,N,N,N`
- Deserialize:
 - Pop 1 (Root)
 - Left = Pop 2
 - Left of 2 = Pop 3
 - Left of 3 = Pop N
 - Right of 3 = Pop N
 - Right of 2 = Pop N...
 - Works perfectly.

### Test Case 2: Full Null Tree
`root = None`
- Serialize: `#`
- Deserialize: Returns None immediately.

### Test Case 3: Negative Values
Tree values can be negative (`-55`). Using `,` delimiter ensures we parse `-55` correctly and not split it as `-` and `55`.

---

## 7. Complexity Analysis

### Time Complexity
- **Serialize**: O(N). We visit every node once. String concatenation (with `join`) is linear.
- **Deserialize**: O(N). We process every value in the string exactly once.

### Space Complexity
- **Serialize**: O(N) for the recursion stack (skewed tree) and the output string.
- **Deserialize**: O(N) for the recursion stack and splitting the string.

---

## 8. Production Considerations

1. **Format Selection**:
 - **CSV/String**: Human readable, easy to debug. Used in this solution.
 - **JSON**: Standard, but verbose (`{"val":1, "left":...}`). huge overhead.
 - **Protobuf/Binary**: Ideal for production. Uses Varints (variable integers) to store small numbers in 1 byte. No commas needed. 10x smaller.

2. **Recursion Limit**:
 - Python's default is 1000. For a deep tree (skewed), this crashes. In production, use an **Iterative DFS** with an explicit stack to handle trees with depth > 1000.

---

## 9. Connections to ML Systems

This problem is the exact algorithmic counterpart to **Model Serialization** (ML System Design).
- **Tree**: Neural Network Graph (Layers are nodes).
- **Weights**: Node Values.
- **Topology**: Left/Right pointers.
- **ONNX Format**: A standardized "string" representation of the computation graph so it can be moved from PyTorch (Python) to C++ Runtime.

Also relates to **Speech Model Export** (Speech Tech), where we serialize the state of streaming Transducers.

---

## 10. Interview Strategy

1. **Clarify Format**: Ask "Can I use JSON?" or "Do I need a binary format?". Usually, they want the logic, not the library.
2. **Mention Traversal Choice**: "I will use Preorder DFS because it serializes the root first, making deserialization straightforward since we always know what node we are building."
3. **Handle Ambiguity**: Explicitly state "I will use a sentinel character '#' to denote nulls to preserve uniqueness."

---

## 11. Key Takeaways

1. **Sentinels are Mandatory**: You cannot serialize a structure uniquely without representing the "absence" of data (nulls).
2. **Streaming Nature**: Preorder traversal is "stream-friendly". You can start building the tree as soon as the first byte arrives.
3. **State Management**: Deserialization is just "replaying" the construction steps recorded during serialization.

---

**Originally published at:** [arunbaby.com/dsa/0047-serialize-deserialize-tree](https://www.arunbaby.com/dsa/0047-serialize-deserialize-tree/)

*If you found this helpful, consider sharing it with others who might benefit.*
