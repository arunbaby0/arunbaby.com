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
  - string-search
difficulty: Hard
subdomain: "String & Trie"
tech_stack: Python
scale: "O(M*N * 4^L) worst case, highly optimized"
companies: Microsoft, Uber, Airbnb, Pinterest
related_ml_day: 48
related_speech_day: 48
related_agents_day: 48
---

**"Don't look for one needle in a haystack. Magnetize the hay to find all needles at once."**

## 1. The Problem: Boggle on Steroids

Imagine you are playing the game Boggle. You have a grid of letters and a possibly huge list of valid words (a dictionary). Your task is to find **all** the words from the list that can be formed in the grid by connecting adjacent letters (horizontally or vertically).

**The Rules:**
- You can move up, down, left, or right.
- You cannot use the same cell twice in a single word path.
- Input: An `m x n` board of characters and a list of strings `words`.
- Output: All words from the list that exist in the board.

**Example:**

```
Board:
[
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]

Words: ["oath", "pea", "eat", "rain"]

Output: ["eat", "oath"]
```

("pea" is not there; "rain" is distinct.)

---

## 2. Approach 1: The Naive Solution (Repeated Search)

If you've solved "Word Search I" (where you look for a *single* word), your instinct might be: "I'll just run that algorithm for every word in the dictionary!"

**The Algorithm:**
1. Iterate through every word in the `words` list.
2. For each word, scan the entire grid to find a starting character.
3. If found, run DFS (Depth-First Search) to try and match the rest of the word.

**Why this fails at scale:**
Let's say the board is `50x50` and you have `10,000` words.
If many words start with widely used prefixes (like "react", "reason", "real", "read"), you end up traversing the exact same path on the board ("r" -> "e" -> "a") thousands of times—once for each word.

This redundancy is killer. We need a way to group words that share prefixes so we traverse the common paths only once.

---

## 3. The Optimized Solution: Backtracking with a Trie

### 3.1 What is a Trie?

A **Trie** (pronounced "try" or "tree") is a tree data structure specialized for strings. Instead of storing whole strings in nodes, we store characters on edges (or nodes). Paths from the root down to a node represent prefixes.

**Visualizing the Trie:**
If our dictionary is `["oath", "pea", "eat", "rain"]`:

```
        (root)
       /   |   \
      o    p    e
     /     |     \
    a      e      a
   /       |       \
  t        a        t (*)
 /
h (*)
```
*(*) marks the end of a valid word.*

Notice how `"eat"` and `"eating"` (if we had it) would share the `e-a-t` path.

### 3.2 The Key Strategy

Instead of iterating through the *words*, we iterate through the **grid**.
We start a DFS from every cell in the grid, and simultaneously traverse the Trie.

1. **Synchronized Traversal**:
   - If we are at cell `(r, c)` with letter `'o'`, we check if the Trie root has a child `'o'`.
   - If yes, we move to that Trie node and move to a neighbor on the grid.
2. **Instant Validation**:
   - If the current Trie node marks the "end of a word", we found one! Add it to results.
   - If the Trie node has no children matching any grid neighbor, we stop immediately. We don't waste time exploring deeper.

### 3.3 Optimization: Pruning the Trie

This is the "pro tip" for this problem.
Once we find a word like `"oath"`, we add it to our result list. **We don't need to find it again.**
If we leave `"oath"` in the Trie, our DFS usually continues searching or re-finds it from other paths.

**Pruning:** When we find a word, we can theoretically remove it from the Trie. If a leaf node is removed and its parent has no other children, the parent can be removed too. This progressively shrinks the search space as we find words, making the algorithm faster the longer it runs!

---

## 4. Design & Implementation

Let's build this step by step.

### 4.1 The Trie Node

To keep things efficient, we'll store the actual word at the endpoint node. This saves us from having to reconstruct the string during DFS.

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # Map char -> TrieNode
        self.word = None    # Stores the word if this node is an end
```

### 4.2 The Main Algorithm class

```python
class Solution:
    def findWords(self, board, words):
        # Step 1: Build the Trie
        root = TrieNode()
        for w in words:
            node = root
            for char in w:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = w  # Mark end of word

        self.matches = []
        rows, cols = len(board), len(board[0])
        
        # Step 2: DFS function
        def dfs(r, c, node):
            letter = board[r][c]
            curr_node = node.children[letter]
            
            # Check if we found a word
            if curr_node.word:
                self.matches.append(curr_node.word)
                curr_node.word = None  # Avoid duplicates!
            
            # Mark path as visited for current recursion stack
            board[r][c] = '#' 
            
            # Explore neighbors
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    board[nr][nc] in curr_node.children):
                    dfs(nr, nc, curr_node)
            
            # Backtrack: Unmark path
            board[r][c] = letter
            
            # Optimization: Leaf pruning
            # If current node has no children (and word is consumed),
            # we can prune it from parent to optimize future searches.
            if not curr_node.children:
                del node.children[letter]

        # Step 3: Trigger DFS from every cell
        for r in range(rows):
            for c in range(cols):
                if board[r][c] in root.children:
                    dfs(r, c, root)
                    
        return self.matches
```

---

## 5. Walkthrough: How it Runs

Consider the board:
```
o  a
t  h
```
Words: `["oath", "oat"]`

1. **Trie Build**:
   `root -> o -> a -> t (end "oat") -> h (end "oath")`

2. **Scan Grid**:
   - Start at `(0,0)` which is `'o'`. It exists in `root.children`.
   - **DFS('o')**:
     - Move to Trie node `o`.
     - Valid neighbors of `(0,0)`: `(0,1)` is `'a'`, `(1,0)` is `'t'`.
     - Trie node `o` has child `a`? Yes.
   - **DFS('a')** (at `0,1`):
     - Trie node `a` has child `t`? Yes. neighbor `(1,0)` is `t`.
   - **DFS('t')** (at `1,0`):
     - **Found word!** `curr_node.word` is `"oat"`. Add to list. Set `curr_node.word = None`.
     - Trie node `t` has child `h`? Yes. neighbor `(1,1)` is `h`.
   - **DFS('h')** (at `1,1`):
     - **Found word!** `curr_node.word` is `"oath"`. Add to list.
     - `curr_node` (path o-a-t-h) has no children.
     - **Pruning**: Remove `h` from parent `t`'s children.
   - Backtrack to `t`:
     - Since we removed `h`, and `t` has no other children, we remove `t` from parent `a`.
   - Backtrack to `a`:
     - Remove `a`... and so on.

The pruning effectively "eats" the Trie branches as we solve the puzzle, making the board emptier of possibilities as we go.

---

## 6. Common Pitfalls

1. **Not handling duplicates**: The problem asks for unique words. If the board has two "eat" formations, you shouldn't return "eat" twice.
   - *Fix*: Set `node.word = None` immediately after adding it to results.
   
2. **Revisiting cells**: You cannot use the same cell `(r,c)` twice in one word.
   - *Fix*: Use a `visited` set or temporarily mutate the board (e.g., change `'a'` to `'#'`) during DFS.

3. **Time Limit Exceeded (TLE)**: Without Trie pruning, standard Trie DFS can still be slow on massive test cases where almost every path is a valid prefix. Pruning is often required for the fastest solution.

---

## 7. Complexity Analysis

- **Time Complexity**: `O(M * N * 3^L)` worst case.
  - `M * N`: We start a search from every cell.
  - `3^L`: In the DFS, we have at most 4 directions, but we came from 1, so we look at 3 neighbors. `L` is the max length of a word.
  - In practice, the Trie limits this drastically.
  
- **Space Complexity**: `O(K)` where `K` is the total number of characters in the dictionary (to build the Trie). The recursion stack takes `O(L)`.

---

## 8. Real World Connections

This isn't just a puzzle game algorithm. This pattern—**traversing a data structure guided by a state machine (Trie)**—appears in:

1. **Boggle Solvers**: Obviously.
2. **Spell Checkers**: To quickly validate if a typed sequence is a valid prefix.
3. **T9 Text Prediction**: Finding valid words from number sequences on old phones.
4. **DNA Sequencing**: Searching for multiple genetic markers (sequences) in a long DNA strand.

---

**Originally published at:** [arunbaby.com/dsa/0048-word-search-ii](https://www.arunbaby.com/dsa/0048-word-search-ii/)

*If you found this helpful, consider sharing it with others who might benefit.*
