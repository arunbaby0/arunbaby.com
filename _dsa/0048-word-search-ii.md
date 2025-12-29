---
title: "Word Search II"
day: 48
collection: dsa
categories:
  - dsa
tags:
  - trie
  - backtracking
  - dfs
  - matrix
  - hard
difficulty: Hard
subdomain: "String & Trie"
tech_stack: Python
scale: "Searching 10k words in 100x100 grid"
companies: Microsoft, Uber, Airbnb, Pinterest
related_ml_day: 48
related_speech_day: 48
related_agents_day: 48
---

**"Don't look for one needle in a haystack. Magnetize the hay to find all needles at once."**

## 1. Problem Statement

This is the "Boss Level" of grid-based search problems.
Given an `m x n` board of characters and a list of strings `words`, return all words on the board.

**Rules:**
1.  Each word must be constructed from letters of sequentially adjacent cells (horizontally or vertically neighboring).
2.  The same letter cell may not be used more than once in a word.
3.  The output should contain all unique words found.

**Example:**
```
Board:
[
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]
Words: ["oath","pea","eat","rain"]
Output: ["eat","oath"]
```
Note: "pea" is not present. "rain" is not present.

---

## 2. Understanding the Problem

### 2.1 The Native vs. Optimized Mindset
-   **Word Search I**: Find *one* word. We iterate the grid and run DFS for that word.
-   **Word Search II**: Find *K* words ($K$ can be 30,000).

If we define $M, N$ as grid dimensions and $L$ as max word length:
-   **Naive Approach**: For each word, scan grid. Total Time: $K \times (M \times N \times 4^L)$.
-   This is horribly inefficient. We are visiting the same cells `(0,0)` 'o' -> `(0,1)` 'a' thousands of times, once for "oath", once for "oats", once for "oatmeal"...

### 2.2 The Inversion of Control
Instead of "For each word, search the grid", we flip it:
"For each path in the grid, does it form *any* word?"

To support this query ("Does this prefix exist in my list?"), we need a **Trie** (Prefix Tree).

---

## 3. Approach 1: Naive (Repeated DFS) -- Don't do this
Run the solution for "Word Search I" inside a loop.
Most interviewers will fail you immediately for this because it ignores the shared structure of the dictionary words.

---

## 4. Approach 2: Backtracking with Trie

We build a Trie from the dictionary words. Then we start DFS from every cell in the grid.
As we traverse the grid (e.g., `o -> a -> t`), we move a pointer down the Trie.

### Key Logic
1.  **Build Trie**: Insert all `words` into Trie. Mark leaf nodes with the actual `word`.
2.  **Grid DFS**:
    -   Start at `(r, c)`.
    -   Check if `board[r][c]` is a child of `Current_Trie_Node`.
    -   If Yes: Move to child. Mark visited. Recurse.
    -   If No: Stop (Pruning).
3.  **Optimization (Leaf Pruning)**:
    -   Once we find a word like "oath", we add it to results.
    -   **Crucial**: We should *remove* "oath" from the Trie (or mark it found) so we don't find it again via a different path.
    -   Even better: If a node has no children after finding a word, delete the node. This progressively shrinks the search space!

---

## 5. Implementation

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None # Stores the word at the leaf node

class Solution:
    def findWords(self, board, words):
        # 1. Build the Trie
        root = TrieNode()
        for w in words:
            node = root
            for char in w:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = w # End of word
            
        self.res = []
        rows, cols = len(board), len(board[0])
        
        def dfs(r, c, parent_node):
            letter = board[r][c]
            curr_node = parent_node.children[letter]
            
            # Check for match
            if curr_node.word:
                self.res.append(curr_node.word)
                curr_node.word = None # De-duplicate: Ensure we don't add it again
            
            # DFS Traversal logic
            board[r][c] = '#' # Mark visited
            
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if board[nr][nc] in curr_node.children:
                        dfs(nr, nc, curr_node)
            
            board[r][c] = letter # Backtrack
            
            # 3. Optimization: Incremental Pruning
            # If leaf node and no word, prune it to optimize future searches
            if not curr_node.children:
                del parent_node.children[letter]

        # Trigger DFS
        for r in range(rows):
            for c in range(cols):
                if board[r][c] in root.children:
                    dfs(r, c, root)
                    
        return self.res
```

---

## 6. Testing Strategy

### Test Case 1: Prefix shared
Words: `["oath", "oat"]`.
Board: Contains `o -> a -> t -> h`.
-   DFS reaches 't'. Finds "oat". Adds to result. Removes `word` marker.
-   DFS continues to 'h'. Finds "oath". Adds to result.
-   Backtrack.

### Test Case 2: No match
Words: `["abcd"]`. Board: `['a', 'b', 'x', 'd']`.
-   DFS matches `a -> b`.
-   Next neighbor `x` is not in Trie child of `b`.
-   Stop immediately. (This shows the power of Trie pruning vs Naive).

---

## 7. Complexity Analysis

-   **Time**: $O(M \times N \times 4^L)$.
    -   In worst case, we might traverse depth $L$ for every cell.
    -   The Trie makes this much faster on average because branches are pruned early.
-   **Space**: $O(K \times L)$ for Trie storage ($K$ words, length $L$).

---

## 8. Production Considerations

1.  **Thread Safety**: In a real "Boggle Solver" server, building the Trie is a one-time startup cost. The DFS is per-request. The Trie must be read-only during requests.
2.  **Memory**: If dictionary is huge (Oxford Dictionary), Trie can take GBs. We might use a **DAWG (Directed Acyclic Word Graph)** to compress shared suffixes (e.g., "-ing").

---

## 9. Connections to ML Systems

This is the exact algorithm used in **Typeahead Systems** (ML System Design Track).
-   **Problem**: User typed "Am".
-   **Grid**: The keyboard/input.
-   **Trie**: The database of all Amazon Products.
-   **Task**: Find the most likely completion.
This DSA problem is the *Search* component of that system.

Also relates to **Phonetic Search** (Speech Tech) where we search Tries by sound.

---

## 10. Interview Strategy

1.  **Start with Naive**: Briefly mention "I could just run DFS for each word", then immediately pivot. "But that's inefficient because..."
2.  **Draw the Trie**: Show how "ant" and "and" share the "an" node.
3.  **Explain Pruning**: This is the differentiator. "If I remove the leaf node after finding a word, the Trie gets smaller." This shows senior-level optimization thinking.

---

## 11. Key Takeaways

1.  **State Machines**: A Trie acts as a State Machine for the DFS. "Am I allowed to step on 'X'?" depends on my current Trie node.
2.  **Preprocessing wins**: Spend $O(K)$ time building a Trie to save exponential time during search.
3.  **Backtracking Template**: `Mark -> Explore -> Unmark` is the universal pattern.

---

**Originally published at:** [arunbaby.com/dsa/0048-word-search-ii](https://www.arunbaby.com/dsa/0048-word-search-ii/)

*If you found this helpful, consider sharing it with others who might benefit.*
