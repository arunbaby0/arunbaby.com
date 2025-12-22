---
title: "Serialize and Deserialize Binary Tree"
day: 47
collection: dsa
categories:
  - dsa
tags:
  - binary-tree
  - serialization
  - dfs
  - bfs
  - design
difficulty: Hard
subdomain: "Trees & Design"
tech_stack: Python
scale: "O(N) time and space"
companies: Google, Meta, Amazon, Microsoft, LinkedIn
related_ml_day: 47
related_speech_day: 47
related_agents_day: 47
---

**"How do you save a tree to disk and reconstruct it perfectly?"**

## 1. Introduction: The Problem of Persistence

Trees live in memory. They're made of nodes with pointers—left child, right child, parent. But what if you need to:

- Save a tree to a file and load it later?
- Send a tree over a network to another machine?
- Store a tree in a database?
- Cache a tree for quick retrieval?

Pointers don't survive these operations. You can't save a memory address to a file and expect it to be valid when you load it back. You need a way to convert the tree into a persistent format and reconstruct it later.

This is the problem of **serialization** (converting to a storable format) and **deserialization** (reconstructing from that format).

---

## 2. Understanding the Challenge

### 2.1 What Makes Trees Hard to Serialize?

Unlike arrays or strings, trees have **structure** that must be preserved:

```
Original tree:
      1
     / \
    2   3
       / \
      4   5

We need to capture:
- Node values: 1, 2, 3, 4, 5
- Parent-child relationships: 1→2, 1→3, 3→4, 3→5
- Missing children: Node 2 has no children
```

If we just output "1, 2, 3, 4, 5", we lose all structural information. How do we encode the structure?

### 2.2 The Key Insight: Mark Missing Nodes

The trick is to include **markers for missing children**. Instead of just listing values, we also record where children are absent:

```
      1
     / \
    2   3
       / \
      4   5

With null markers (using "null"):
1, 2, null, null, 3, 4, null, null, 5, null, null
```

Now we can reconstruct the tree:
- "1" → Root
- "2" → Left child of 1
- "null" → 2 has no left child
- "null" → 2 has no right child
- "3" → Right child of 1
- And so on...

### 2.3 Traversal Order Matters

We must serialize and deserialize in the **same traversal order**. Common choices:

**Pre-order (root → left → right):**
Most intuitive for reconstruction. Process the root first, then recursively handle subtrees.

**Level-order (BFS):**
Process level by level. Natural for queue-based reconstruction.

**In-order + pre-order combination:**
Theoretically works but more complex. Rarely used for serialization.

We'll focus on pre-order (DFS) and level-order (BFS) as these are the most practical.

---

## 3. Approach 1: Pre-Order Traversal (DFS)

### 3.1 The Intuition

Pre-order traversal visits nodes in a specific sequence: root, then left subtree, then right subtree. If we include null markers, this sequence uniquely defines the tree.

**Why pre-order works:**
1. We always know the first value is the root
2. Everything after the root (until we've processed the left subtree) belongs to the left
3. Everything remaining belongs to the right
4. Null markers tell us where subtrees end

### 3.2 Serialization Process

Let's trace through serialization:

```
Tree:
      1
     / \
    2   3
       / \
      4   5

Pre-order traversal with nulls:
```

**Step 1:** Visit 1 → Output: "1"  
**Step 2:** Go left to 2 → Output: "1,2"  
**Step 3:** Go left from 2 (null) → Output: "1,2,null"  
**Step 4:** Go right from 2 (null) → Output: "1,2,null,null"  
**Step 5:** Back to 1, go right to 3 → Output: "1,2,null,null,3"  
**Step 6:** Go left from 3 to 4 → Output: "1,2,null,null,3,4"  
**Step 7:** Go left from 4 (null) → Output: "1,2,null,null,3,4,null"  
**Step 8:** Go right from 4 (null) → Output: "1,2,null,null,3,4,null,null"  
**Step 9:** Back to 3, go right to 5 → Output: "1,2,null,null,3,4,null,null,5"  
**Step 10:** Go left from 5 (null) → Output: "1,2,null,null,3,4,null,null,5,null"  
**Step 11:** Go right from 5 (null) → Output: "1,2,null,null,3,4,null,null,5,null,null"

**Final serialization:** `"1,2,null,null,3,4,null,null,5,null,null"`

### 3.3 Deserialization Process

Deserialization is the reverse. We consume tokens one by one:

**Input:** `"1,2,null,null,3,4,null,null,5,null,null"`  
**Token list:** [1, 2, null, null, 3, 4, null, null, 5, null, null]

**Step 1:** Pop "1" → Create node 1, this is root  
**Step 2:** Recursively build left subtree...  
  - Pop "2" → Create node 2, set as left child of 1  
  - Pop "null" → Node 2 has no left child  
  - Pop "null" → Node 2 has no right child  
**Step 3:** Recursively build right subtree of 1...  
  - Pop "3" → Create node 3, set as right child of 1  
  - Recursively build left of 3...  
    - Pop "4" → Create node 4  
    - Pop "null" → No left child  
    - Pop "null" → No right child  
  - Recursively build right of 3...  
    - Pop "5" → Create node 5  
    - Pop "null" → No left child  
    - Pop "null" → No right child

**Result:** Tree reconstructed perfectly!

### 3.4 The Implementation

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Codec:
    """Serialize and deserialize binary tree using pre-order DFS."""
    
    def serialize(self, root: TreeNode) -> str:
        """Convert tree to string representation."""
        result = []
        
        def dfs(node):
            if not node:
                result.append("null")
                return
            result.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        
        dfs(root)
        return ",".join(result)
    
    def deserialize(self, data: str) -> TreeNode:
        """Reconstruct tree from string representation."""
        tokens = iter(data.split(","))
        
        def build():
            val = next(tokens)
            if val == "null":
                return None
            node = TreeNode(int(val))
            node.left = build()
            node.right = build()
            return node
        
        return build()
```

### 3.5 Why This Works

The magic is in the iterator (`iter()`). Each call to `build()`:
1. Consumes exactly one token for the current node
2. Recursively consumes tokens for the left subtree
3. Recursively consumes tokens for the right subtree

Since both serialization and deserialization follow the same order (pre-order), they're perfectly synchronized.

---

## 4. Approach 2: Level-Order Traversal (BFS)

### 4.1 The Intuition

Level-order traversal processes nodes level by level, left to right. This matches how we might naturally draw or describe a tree.

```
      1        Level 0
     / \
    2   3      Level 1
       / \
      4   5    Level 2
```

Level-order: [1, 2, 3, null, null, 4, 5]

Wait—this is fewer values than pre-order! That's because in level-order, we don't need to include nulls for children of null nodes.

### 4.2 Serialization Process

**Step 1:** Start with root in queue  
**Step 2:** For each node in queue:
- Output its value (or "null")
- If not null, add its children to queue

```
Tree:
      1
     / \
    2   3
       / \
      4   5

Queue processing:
[1] → Output "1", add children [2, 3]
[2, 3] → Pop 2, output "2", add children [null, null]
[3, null, null] → Pop 3, output "3", add children [4, 5]
[null, null, 4, 5] → Pop null, output "null"
[null, 4, 5] → Pop null, output "null"
[4, 5] → Pop 4, output "4", add children [null, null]
[5, null, null] → Pop 5, output "5", add children [null, null]
[null, null, null, null] → All nulls, done
```

**Result:** `"1,2,3,null,null,4,5,null,null,null,null"`

(We can optimize by trimming trailing nulls: `"1,2,3,null,null,4,5"`)

### 4.3 Deserialization Process

Use a queue to track which nodes need children assigned:

**Input:** `"1,2,3,null,null,4,5"`

**Step 1:** Create root from first value (1)  
**Step 2:** Process pairs of values for each node's children:
- Node 1: left=2, right=3
- Node 2: left=null, right=null
- Node 3: left=4, right=5
- Node 4: (no more values, or implied nulls)
- Node 5: (no more values, or implied nulls)

### 4.4 When to Use Each Approach

| Aspect | Pre-order (DFS) | Level-order (BFS) |
|--------|-----------------|-------------------|
| Implementation | Simpler recursion | Queue-based iteration |
| Memory | Call stack (O(H)) | Queue (O(W)) |
| Best for | Deep trees | Wide trees |
| Trailing nulls | Always included | Can be trimmed |
| Conceptual | Follows recursion | Follows tree "shape" |

In practice, pre-order is simpler to implement and understand. Level-order matches JSON formats used by platforms like LeetCode.

---

## 5. Edge Cases to Consider

### 5.1 Empty Tree

The most common edge case:

```python
serialize(None) → "null"
deserialize("null") → None
```

Always handle this first!

### 5.2 Single Node Tree

```python
serialize(TreeNode(1)) → "1,null,null"  # Pre-order
deserialize("1,null,null") → TreeNode(1)
```

### 5.3 Negative Values

Values can be negative. Make sure serialization handles them:

```python
serialize(TreeNode(-1)) → "-1,null,null"  # The negative sign is part of the value
```

### 5.4 Large Values

Values might be very large numbers. Ensure parsing handles them:

```python
# Integer overflow is typically not an issue in Python
# but matters in languages like Java or C++
```

### 5.5 Skewed Trees

Left-skewed or right-skewed trees work fine but produce many nulls:

```
    1
     \
      2
       \
        3

Pre-order: "1,null,2,null,3,null,null"
```

---

## 6. Optimization Ideas

### 6.1 Space Optimization

The null markers consume significant space. Alternative approaches:

**Bit encoding:** Use bits to indicate which children exist:
- 00 = no children
- 01 = right only
- 10 = left only
- 11 = both children

This eliminates explicit null markers but requires additional metadata.

**Structure + values:** Encode structure separately from values:
- Structure: "110010..." (1 = has left, 0 = no left, etc.)
- Values: "1,2,3,4,5"

### 6.2 Compression

For large trees, apply standard compression (gzip, lz4) after serialization. This is especially effective because serialized trees often have repetitive patterns.

### 6.3 Binary Format

Instead of strings, use binary encoding:
- 4 bytes per value (or variable-length encoding)
- 1 bit per null marker

This reduces size significantly at the cost of human readability.

---

## 7. Real-World Applications

### 7.1 Database Storage

Trees are used in:
- Expression trees (SQL query plans)
- Document structures (XML, JSON)
- File systems (directory trees)

All of these need serialization for persistence.

### 7.2 Distributed Systems

When processing trees across machines:
- Serialize on source machine
- Send over network
- Deserialize on destination machine

### 7.3 Caching

Cache computed tree structures:
- Serialize tree to Redis
- Deserialize on cache hit
- Avoid recomputation

### 7.4 Version Control

Store tree history:
- Serialize each version
- Compare serializations for diffs
- Reconstruct any historical version

---

## 8. Connection to Model Serialization (ML Day 47)

Interesting parallel: ML models are also complex structures that need serialization!

| Tree Serialization | Model Serialization |
|-------------------|---------------------|
| Node values | Weight matrices |
| Tree structure | Layer connections |
| Pointers | Tensor shapes |
| Null markers | Empty layers/skip connections |
| Format: String | Format: Protobuf/Pickle/ONNX |

Both face the same fundamental challenge: converting in-memory structures with references into flat, portable formats.

---

## 9. Complexity Analysis

### Time Complexity: O(N)

- Serialization: Visit each node once
- Deserialization: Process each token once

Where N is the number of nodes.

### Space Complexity: O(N)

- Serialization: Storage for output string (~3N characters with nulls)
- Deserialization: Recursion stack (O(H)) + tree storage (O(N))

Where H is tree height. In worst case (skewed tree), H = N.

---

## 10. Interview Tips

### 10.1 Clarifying Questions

Ask:
- What values can nodes contain? (integers, strings, negatives?)
- Should it handle empty trees?
- Any size constraints?
- Preferred format? (String, JSON, binary?)

### 10.2 Start with Pre-order

Pre-order is simpler to explain and implement. Mention you could also use level-order if asked.

### 10.3 Don't Forget Edge Cases

Test your solution mentally with:
- Empty tree
- Single node
- Skewed tree (all left or all right)
- Negative values

### 10.4 Follow-up Questions

Common follow-ups:
- "How would you serialize a general tree (not binary)?" → Include child count
- "How would you serialize a graph?" → Need to handle cycles—use node IDs
- "How would you minimize the serialized size?" → Compression, binary format

---

## 11. Summary

Serialization and deserialization is fundamentally about:

1. **Traversing the tree in a consistent order** (pre-order or level-order)
2. **Marking absent children** so we know where branches end
3. **Reconstructing by consuming tokens in the same order**

The key insight is that **including null markers makes the serialization uniquely decodable**. Without them, we can't tell where one subtree ends and another begins.

This pattern—converting structured data to flat formats and back—is universal in computing. Master it for trees, and you'll recognize it everywhere: in database systems, network protocols, file formats, and yes, in serializing ML models.

---

**Originally published at:** [arunbaby.com/dsa/0047-serialize-deserialize-tree](https://www.arunbaby.com/dsa/0047-serialize-deserialize-tree/)

*If you found this helpful, consider sharing it with others who might benefit.*
