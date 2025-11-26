---
title: "Binary Tree Level Order Traversal"
day: 26
collection: dsa
categories:
  - dsa
tags:
  - binary-tree
  - bfs
  - queue
  - medium
subdomain: "Tree Algorithms"
tech_stack: [Python, C++, Java]
scale: "O(N) time, O(N) space"
companies: [Facebook, Amazon, Microsoft, LinkedIn, Google]
related_dsa_day: 26
related_ml_day: 26
related_speech_day: 26
---

**How do you print a corporate hierarchy level by level? CEO first, then VPs, then Managers...**

## Problem

Given the `root` of a binary tree, return the *level order traversal* of its nodes' values. (i.e., from left to right, level by level).

**Example 1:**
```
    3
   / \
  9  20
    /  \
   15   7
```
**Input:** `root = [3,9,20,null,null,15,7]`
**Output:** `[[3],[9,20],[15,7]]`

**Example 2:**
**Input:** `root = [1]`
**Output:** `[[1]]`

## Intuition

Depth First Search (DFS) dives deep. It goes `Root -> Left -> Left...` until it hits a leaf.
Breadth First Search (BFS) explores wide. It visits all neighbors at the current depth before moving deeper.

For a tree, BFS naturally produces a Level Order Traversal.
The key data structure for BFS is the **Queue** (FIFO - First In, First Out).
- We enter the queue at the back.
- We leave the queue from the front.
- This ensures that nodes at depth `d` are processed before nodes at depth `d+1`.

## Approach 1: Iterative BFS using Queue

We use a `deque` (double-ended queue) in Python for efficient `popleft()`.

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(current_level)
            
        return result
```

**Complexity Analysis:**
- **Time:** `O(N)`. We visit every node once.
- **Space:** `O(N)` (or `O(W)` where W is max width). In a perfect binary tree, the last level has `N/2` nodes.

## Approach 2: Recursive DFS (Preorder)

Can we do this with DFS? Yes, but it's less intuitive.
We pass the `level` index in the recursion.
`dfs(node, level)` adds `node.val` to `result[level]`.

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        result = []
        
        def dfs(node, level):
            if not node:
                return
            
            # Ensure the list for this level exists
            if len(result) == level:
                result.append([])
            
            result[level].append(node.val)
            
            dfs(node.left, level + 1)
            dfs(node.right, level + 1)
            
        dfs(root, 0)
        return result
```

**Pros:** Simpler code (no queue).
**Cons:** Uses system stack `O(H)` space. BFS uses heap space.

## Variant: Zigzag Level Order Traversal

**Problem:** Return the zigzag level order traversal.
Level 0: Left -> Right
Level 1: Right -> Left
Level 2: Left -> Right

**Solution:**
Use a standard BFS.
Keep a flag `left_to_right`.
If `left_to_right` is False, append to `current_level` in reverse (or use `deque.appendleft`).

```python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        
        res = []
        q = deque([root])
        left_to_right = True
        
        while q:
            level_size = len(q)
            level_nodes = deque() # Use deque for O(1) appendleft
            
            for _ in range(level_size):
                node = q.popleft()
                
                if left_to_right:
                    level_nodes.append(node.val)
                else:
                    level_nodes.appendleft(node.val)
                    
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
            
            res.append(list(level_nodes))
            left_to_right = not left_to_right
            
        return res
```

## Variant 2: N-ary Tree Level Order Traversal

**Problem:** Given an N-ary tree (where each node has a list of `children`), return the level order traversal.

**Intuition:**
Same as Binary Tree, but instead of adding `left` and `right`, we iterate through `children`.

```python
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root: return []
        
        res = []
        q = deque([root])
        
        while q:
            level = []
            for _ in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.children:
                    q.extend(node.children)
            res.append(level)
            
        return res
```

## Variant 3: Binary Tree Level Order Traversal II (Bottom-Up)

**Problem:** Return the traversal from leaf to root. `[[15,7], [9,20], [3]]`.

**Solution:**
Standard BFS, but `result.insert(0, level)` or `result.reverse()` at the end.
`reverse()` is `O(N)` but amortized `O(1)` per level. `insert(0)` is `O(N)` per level (Total `O(N^2)`). **Always use reverse.**

## Variant 4: Binary Tree Right Side View

**Problem:** Imagine standing on the right side of the tree. Return the values of the nodes you can see.
Basically, the **last node** of each level.

```python
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root: return []
        
        res = []
        q = deque([root])
        
        while q:
            level_len = len(q)
            for i in range(level_len):
                node = q.popleft()
                # If it's the last node in the current level
                if i == level_len - 1:
                    res.append(node.val)
                
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
        return res
```

## Variant 5: Cousins in Binary Tree

**Problem:** Two nodes are cousins if they have the same depth but different parents.
Given `root`, `x`, and `y`, return `True` if they are cousins.

**Intuition:**
BFS is perfect for tracking depth. We also need to track the parent.
We can store `(node, parent)` in the queue.

```python
class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        q = deque([(root, None)])
        
        while q:
            level_size = len(q)
            found_x = False
            found_y = False
            x_parent = None
            y_parent = None
            
            for _ in range(level_size):
                node, parent = q.popleft()
                
                if node.val == x:
                    found_x = True
                    x_parent = parent
                if node.val == y:
                    found_y = True
                    y_parent = parent
                
                if node.left: q.append((node.left, node))
                if node.right: q.append((node.right, node))
            
            # Check after finishing the level
            if found_x and found_y:
                return x_parent != y_parent
            
            # If one found but not the other, they are at different depths
            if found_x or found_y:
                return False
                
        return False
```

## Advanced Variant 6: Maximum Width of Binary Tree

**Problem:** The maximum width among all levels.
The width of one level is defined as the length between the end-nodes (the leftmost and rightmost non-null nodes), where the null nodes between the end-nodes are also counted into the length calculation.

**Intuition:**
This is tricky because of the "null nodes are counted" part.
We can index the nodes like a Heap.
- Root index: `1`
- Left child: `2*i`
- Right child: `2*i + 1`
Width = `index_right - index_left + 1`.

```python
class Solution:
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        if not root: return 0
        
        max_width = 0
        # Queue stores (node, index)
        q = deque([(root, 0)])
        
        while q:
            level_len = len(q)
            _, level_start_index = q[0]
            
            for i in range(level_len):
                node, index = q.popleft()
                
                if node.left:
                    q.append((node.left, 2*index))
                if node.right:
                    q.append((node.right, 2*index + 1))
                    
            # Calculate width for this level
            # Current index is the last one popped
            max_width = max(max_width, index - level_start_index + 1)
            
        return max_width
```

## System Design: Distributed Graph Traversal (Pregel)

**Interviewer:** "How do you run BFS on a graph with 1 Trillion nodes (Facebook Friend Graph)?"
**Candidate:** "You can't fit it in RAM. You need **Pregel** (Google's Bulk Synchronous Parallel model)."

**The Pregel Model:**
1.  **Supersteps:** Computation happens in rounds.
2.  **Vertex-Centric:** Each vertex runs a function `Compute(messages)`.
3.  **Message Passing:** Vertices send messages to neighbors (to be received in the next Superstep).

**BFS in Pregel:**
- **Superstep 0:** Source vertex sets `min_dist = 0` and sends `dist=1` to neighbors.
- **Superstep 1:** Neighbors receive `dist=1`. If `current_dist > 1`, update `current_dist = 1` and send `dist=2` to neighbors.
- **Halt:** When no nodes update their distance.

## Deep Dive: Vertical Order Traversal

**Problem:** Print the tree in vertical columns.
If two nodes are in the same row and column, the order should be from left to right.

**Intuition:**
We need coordinates `(row, col)`.
- Root: `(0, 0)`
- Left: `(row+1, col-1)`
- Right: `(row+1, col+1)`

We can use BFS to traverse. We store `(node, col)` in the queue.
We need a Hash Map `col -> list of nodes`.
Finally, sort the map by keys (column index).

```python
class Solution:
    def verticalOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        
        column_table = defaultdict(list)
        q = deque([(root, 0)])
        min_col = 0
        max_col = 0
        
        while q:
            node, col = q.popleft()
            column_table[col].append(node.val)
            
            min_col = min(min_col, col)
            max_col = max(max_col, col)
            
            if node.left: q.append((node.left, col - 1))
            if node.right: q.append((node.right, col + 1))
            
        return [column_table[x] for x in range(min_col, max_col + 1)]
```

## Appendix B: Boundary Traversal

**Problem:** Print the boundary of the tree (Left Boundary + Leaves + Right Boundary).
**Intuition:**
1.  **Left Boundary:** Keep going left. If no left, go right. (Exclude leaf).
2.  **Leaves:** DFS/Preorder. Add if `!left` and `!right`.
3.  **Right Boundary:** Keep going right. If no right, go left. (Exclude leaf). Add in reverse order.

This is a classic "Hard" problem that tests modular thinking. Don't try to do it in one pass. Break it down.

## Advanced Variant 7: Diagonal Traversal

**Problem:** Print the tree diagonally.
```
    8
   / \
  3   10
 / \    \
1   6    14
   / \   /
  4   7 13
```
**Output:** `[[8, 10, 14], [3, 6, 7, 13], [1, 4]]`

**Intuition:**
- Root is at `d=0`.
- Left child is at `d+1`.
- Right child is at `d` (same diagonal).

We can use a Queue. But instead of just popping, we iterate through the **right child chain** and add all of them to the current diagonal list, while pushing their left children to the queue for the next diagonal.

```python
class Solution:
    def diagonalTraversal(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        
        res = []
        q = deque([root])
        
        while q:
            level_size = len(q)
            curr_diagonal = []
            
            for _ in range(level_size):
                node = q.popleft()
                
                # Process the current node and all its right children
                while node:
                    curr_diagonal.append(node.val)
                    if node.left:
                        q.append(node.left)
                    node = node.right
            
            res.append(curr_diagonal)
            
        return res
```

## Advanced Variant 8: Serialize and Deserialize Binary Tree

**Problem:** Convert a tree to a string and back.
**Method:** Level Order Traversal (BFS).

**Serialization:**
Use a Queue. If a node is `None`, append "null".
`[1, 2, 3, null, null, 4, 5]`

**Deserialization:**
Use a Queue.
1.  Read root `1`. Push to queue.
2.  Pop `1`. Read next two values `2`, `3`. Attach as left/right. Push `2`, `3`.
3.  Pop `2`. Read `null`, `null`. Attach.
4.  Pop `3`. Read `4`, `5`. Attach. Push `4`, `5`.

```python
class Codec:
    def serialize(self, root):
        if not root: return ""
        q = deque([root])
        res = []
        while q:
            node = q.popleft()
            if node:
                res.append(str(node.val))
                q.append(node.left)
                q.append(node.right)
            else:
                res.append("null")
        return ",".join(res)

    def deserialize(self, data):
        if not data: return None
        vals = data.split(",")
        root = TreeNode(int(vals[0]))
        q = deque([root])
        i = 1
        while q:
            node = q.popleft()
            
            # Left Child
            if vals[i] != "null":
                node.left = TreeNode(int(vals[i]))
                q.append(node.left)
            i += 1
            
            # Right Child
            if vals[i] != "null":
                node.right = TreeNode(int(vals[i]))
                q.append(node.right)
            i += 1
        return root
```

## Deep Dive: Tree BFS vs. Graph BFS

**Tree BFS:**
- No cycles.
- No `visited` set needed.
- Exactly one path to each node.

**Graph BFS:**
- Cycles exist.
- **Must** use `visited` set to avoid infinite loops.
- Multiple paths exist. BFS finds the shortest path (in unweighted graphs).

**Bidirectional BFS:**
To find the shortest path between `A` and `B` in a massive graph.
Run BFS from `A` forward and from `B` backward.
Meet in the middle.
**Complexity:** `O(b^(d/2))` instead of `O(b^d)`. Huge saving!

## Appendix C: The "Rotting Oranges" Pattern

**Problem:** Given a grid where `2` is rotten orange, `1` is fresh, `0` is empty.
Every minute, a rotten orange rots its 4-directional neighbors.
Return min minutes until all fresh oranges rot.

**Intuition:**
This is **Multi-Source BFS**.
1.  Push *all* initially rotten oranges into the Queue at `t=0`.
2.  Run standard BFS.
3.  The number of levels is the time.

This pattern appears in:
- "Walls and Gates"
- "01 Matrix"
- "Map of Highest Peak"

## Appendix D: Interview Questions

1.  **Q:** "Can you perform Level Order Traversal without a Queue?"
    **A:** Yes, using Recursion (DFS) and passing the level index (Approach 2). Or using two arrays (current_level, next_level).

2.  **Q:** "What is the space complexity of BFS?"
    **A:** `O(W)` where `W` is the maximum width. In a full binary tree, the last level has `N/2` leaves, so `O(N)`.

3.  **Q:** "When should you use DFS vs BFS?"
    **A:**
    - **BFS:** Shortest path, levels, closer to root.
    - **DFS:** Exhaustive search, backtracking, path finding, closer to leaves.

## Advanced Variant 9: Populating Next Right Pointers in Each Node

**Problem:** You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. Populate each `next` pointer to point to its next right node. If there is no next right node, the next pointer should be set to `NULL`.

**Intuition:**
Level Order Traversal is the obvious choice.
But can we do it with **O(1) Space**?
Yes. We can use the `next` pointers we already established in the *previous* level to traverse the *current* level.

```python
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root: return None
        
        leftmost = root
        
        while leftmost.left:
            head = leftmost
            while head:
                # Connection 1: Left -> Right (Same Parent)
                head.left.next = head.right
                
                # Connection 2: Right -> Next's Left (Different Parent)
                if head.next:
                    head.right.next = head.next.left
                
                head = head.next
            
            leftmost = leftmost.left
            
        return root
```

## Advanced Variant 10: Average of Levels in Binary Tree

**Problem:** return the average value of the nodes on each level in the form of an array.

**Intuition:**
Standard BFS. Sum the values in the level loop, divide by `level_size`.

```python
class Solution:
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        if not root: return []
        res = []
        q = deque([root])
        
        while q:
            level_sum = 0
            level_count = len(q)
            
            for _ in range(level_count):
                node = q.popleft()
                level_sum += node.val
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
                
            res.append(level_sum / level_count)
            
        return res
```

## Advanced Variant 11: Find Bottom Left Tree Value

**Problem:** Given the root of a binary tree, return the leftmost value in the last row of the tree.

**Intuition:**
Right-to-Left BFS.
The last node visited will be the bottom-left node.

```python
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        q = deque([root])
        node = root
        while q:
            node = q.popleft()
            # Add Right first, then Left
            if node.right: q.append(node.right)
            if node.left: q.append(node.left)
        return node.val
```

## Deep Dive: Queue Implementation (Array vs. Linked List)

**Array (Python List):**
- `pop(0)` is `O(N)` because we have to shift all elements. **Bad.**
- `pop()` is `O(1)`.

**Linked List (Python deque):**
- Doubly Linked List.
- `popleft()` is `O(1)`. **Good.**
- `append()` is `O(1)`.

**Circular Buffer (Ring Buffer):**
- Fixed size array.
- `head` and `tail` pointers wrap around.
- Used in low-latency systems (Network Drivers). No dynamic allocation overhead.

## System Design: Distributed Queue (Kafka vs. SQS)

**Interviewer:** "We need a queue for our Distributed BFS. Should we use Kafka or SQS?"

**Candidate:**
1.  **SQS (Simple Queue Service):**
    - **Pros:** Infinite scaling, no management.
    - **Cons:** No ordering guarantee (standard), expensive at high throughput.
    - **Use Case:** Task Queue (Celery).

2.  **Kafka:**
    - **Pros:** High throughput (millions/sec), replayable (log), ordered within partition.
    - **Cons:** Hard to manage (Zookeeper), fixed partitions.
    - **Use Case:** Event Streaming, Data Pipeline.

**Decision:** For BFS, we usually need a **Priority Queue** (to prioritize high-rank pages), so neither is perfect. We might use **Redis Sorted Sets**.

## Advanced Variant 12: Deepest Leaves Sum

**Problem:** Return the sum of values of the deepest leaves.

**Intuition:**
Standard BFS. Reset `sum` at the start of each level. The last `sum` is the answer.

```python
class Solution:
    def deepestLeavesSum(self, root: TreeNode) -> int:
        if not root: return 0
        q = deque([root])
        level_sum = 0
        
        while q:
            level_sum = 0
            for _ in range(len(q)):
                node = q.popleft()
                level_sum += node.val
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
        return level_sum
```

## Appendix E: The "Word Ladder" Pattern

**Problem:** Transform "hit" to "cog" by changing one letter at a time. Each intermediate word must exist in the dictionary. Return shortest path.

**Intuition:**
This is BFS on an implicit graph.
- **Nodes:** Words.
- **Edges:** Words differing by 1 letter.
- **Start:** "hit".
- **Target:** "cog".

**Optimization:**
Pre-process the dictionary into generic states:
`hot` -> `*ot`, `h*t`, `ho*`.
Map `*ot` -> `[hot, dot, lot]`.
This allows `O(1)` neighbor finding.

## Deep Dive: Python `deque` Internals

Why is `deque` faster than `list` for popping from the front?
**List:** Contiguous memory array. `pop(0)` requires shifting `N-1` elements. `O(N)`.
**Deque:** Doubly Linked List of **Blocks** (Arrays).
- Each block stores 64 elements.
- `popleft()` just increments a pointer in the first block.
- If the block becomes empty, we unlink it. `O(1)`.
- **Cache Locality:** Better than a standard Linked List (one node per element) because of the block structure.

## Advanced Variant 13: Check Completeness of a Binary Tree

**Problem:** Check if the tree is a **Complete Binary Tree** (filled left to right).
**Intuition:**
Level Order Traversal.
If we see a `null` node, we should **never** see a non-null node again.
If we do, it's not complete.

```python
class Solution:
    def isCompleteTree(self, root: TreeNode) -> bool:
        q = deque([root])
        seen_null = False
        
        while q:
            node = q.popleft()
            
            if not node:
                seen_null = True
            else:
                if seen_null:
                    return False
                q.append(node.left)
                q.append(node.right)
                
        return True
```

## Advanced Variant 14: Maximum Level Sum of a Binary Tree

**Problem:** Return the level number (1-indexed) with the maximum sum.

**Intuition:**
Standard BFS. Track `max_sum` and `max_level`.

```python
class Solution:
    def maxLevelSum(self, root: TreeNode) -> int:
        if not root: return 0
        q = deque([root])
        max_sum = float('-inf')
        max_level = 1
        curr_level = 1
        
        while q:
            level_sum = 0
            level_len = len(q)
            
            for _ in range(level_len):
                node = q.popleft()
                level_sum += node.val
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
            
            if level_sum > max_sum:
                max_sum = level_sum
                max_level = curr_level
                
            curr_level += 1
            
        return max_level
```

## Advanced Variant 15: Even Odd Tree

**Problem:**
- Even-indexed levels: Strictly increasing, odd values.
- Odd-indexed levels: Strictly decreasing, even values.

**Intuition:**
BFS with a toggle flag. Check conditions inside the loop.

```python
class Solution:
    def isEvenOddTree(self, root: TreeNode) -> bool:
        q = deque([root])
        level = 0
        
        while q:
            prev = float('-inf') if level % 2 == 0 else float('inf')
            
            for _ in range(len(q)):
                node = q.popleft()
                
                if level % 2 == 0:
                    # Even Level: Odd values, Increasing
                    if node.val % 2 == 0 or node.val <= prev:
                        return False
                else:
                    # Odd Level: Even values, Decreasing
                    if node.val % 2 != 0 or node.val >= prev:
                        return False
                        
                prev = node.val
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
                
            level += 1
            
        return True
```

## System Design: Rate Limiter (Token Bucket)

**Interviewer:** "Design a Rate Limiter."
**Candidate:** "We can use a **Token Bucket** algorithm."

**Concept:**
- A bucket holds `N` tokens.
- Tokens are added at rate `R` per second.
- A request consumes 1 token.
- If bucket is empty, reject request.

**Implementation:**
We don't need a literal Queue. We can use a counter and a timestamp.
`current_tokens = min(capacity, previous_tokens + (now - last_refill_time) * rate)`

**Distributed Rate Limiter:**
Use Redis (Lua Script) to make the read-update-write atomic.
Or use a **Sliding Window Log** (Queue of timestamps) for strict accuracy (but high memory).

## Advanced Variant 16: Pseudo-Palindromic Paths

**Problem:** Return the number of paths from root to leaf where the path values can form a palindrome.
**Intuition:**
A path can form a palindrome if at most one number has an odd frequency.
We can use BFS (or DFS) and a bitmask to track parity of counts.
`mask ^= (1 << node.val)`.
If `mask & (mask - 1) == 0`, it's a palindrome.

```python
class Solution:
    def pseudoPalindromicPaths (self, root: TreeNode) -> int:
        count = 0
        # (node, mask)
        q = deque([(root, 0)])
        
        while q:
            node, mask = q.popleft()
            mask ^= (1 << node.val)
            
            if not node.left and not node.right:
                if mask & (mask - 1) == 0:
                    count += 1
            
            if node.left: q.append((node.left, mask))
            if node.right: q.append((node.right, mask))
            
        return count
```

## Conclusion

Level Order Traversal is the "Hello World" of BFS.
Mastering the `while queue: ... for _ in range(len(queue)):` pattern is crucial. It appears in graph problems, tree problems, and even matrix problems (Rotting Oranges).
