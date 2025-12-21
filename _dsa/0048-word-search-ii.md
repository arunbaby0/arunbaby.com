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
  - word-search
  - prefix-tree
difficulty: Hard
subdomain: "Tries & Backtracking"
tech_stack: Python
scale: "O(M×N×4^L) worst case, Trie pruning helps significantly"
companies: Google, Meta, Amazon, Microsoft, Apple
related_ml_day: 48
related_speech_day: 48
related_agents_day: 48
---

**"Find all words hiding in a grid—with a Trie to light the way."**

## 1. Problem Statement

Given an `m x n` board of characters and a list of strings `words`, return all words on the board. Each word must be constructed from letters of sequentially adjacent cells (horizontal or vertical neighbors). The same cell may not be used more than once in a word.

**Example:**
```
board = [
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]
words = ["oath","pea","eat","rain"]

Output: ["eat","oath"]
```

**Constraints:**
- `m, n ≤ 12`
- `1 ≤ words.length ≤ 3 × 10^4`
- `1 ≤ words[i].length ≤ 10`

## 2. Understanding the Problem

This is Word Search I on steroids—instead of checking one word, we check thousands. The naive approach would apply Word Search I for each word: O(words × M × N × 4^L). With 30K words, this is too slow.

### Key Insight: Trie for Prefix Matching

Build all words into a Trie, then DFS through the board. At each cell, check if the current path is a prefix in the Trie. If not, prune immediately.

```
Words: ["oath", "oat", "pea"]

Trie:
       root
      /    \
     o      p
     |      |
     a      e
     |      |
     t*     a*
     |
     h*
```

## 3. Approach: Trie + DFS Backtracking

```python
from typing import List

class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None  # Store full word at end


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Find all words from list that exist in the board.
        
        Approach:
        1. Build Trie from all words
        2. DFS from each cell, following Trie paths
        3. Prune when no prefix match
        
        Time: O(M×N×4^L) worst case, but Trie pruning helps significantly
        Space: O(total characters in words) for Trie
        """
        # Build Trie
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word
        
        m, n = len(board), len(board[0])
        result = []
        
        def dfs(row, col, node):
            """DFS with Trie guidance."""
            char = board[row][col]
            
            # Check if current char in Trie path
            if char not in node.children:
                return
            
            next_node = node.children[char]
            
            # Found a word?
            if next_node.word:
                result.append(next_node.word)
                next_node.word = None  # Avoid duplicates
            
            # Mark visited
            board[row][col] = '#'
            
            # Explore neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < m and 0 <= nc < n and board[nr][nc] != '#':
                    dfs(nr, nc, next_node)
            
            # Restore
            board[row][col] = char
            
            # Optimization: remove leaf nodes with no words
            if not next_node.children and not next_node.word:
                del node.children[char]
        
        # Start DFS from each cell
        for i in range(m):
            for j in range(n):
                dfs(i, j, root)
        
        return result
```

## 4. Detailed Walkthrough

```
board = [['o','a','a','n'],
         ['e','t','a','e'],
         ['i','h','k','r'],
         ['i','f','l','v']]

words = ["oath","pea","eat","rain"]

Trie after building:
root → o → a → t → h (word: "oath")
     |
     → p → e → a (word: "pea")
     |
     → e → a → t (word: "eat")
     |
     → r → a → i → n (word: "rain")

DFS from (0,0) = 'o':
- 'o' in Trie? Yes
- Neighbors: (0,1)='a', (1,0)='e'
- DFS (0,1) = 'a': 'a' after 'o'? Yes
  - DFS (0,2) = 'a': 'a' after 'oa'? No, prune
  - DFS (1,1) = 't': 't' after 'oa'? Yes
    - Found "oat"? No word marker
    - DFS (2,1) = 'h': 'h' after 'oat'? Yes
      - Found "oath"! Add to result
      
DFS from (1,1) = 't' eventually finds "eat": e(1,0) → a(0,1) → t(1,1)
```

## 5. Optimizations

### 5.1 Trie Pruning

```python
# After finding a word, prune empty branches
if not next_node.children and not next_node.word:
    del node.children[char]
```

This is crucial—as we find words, we remove them from the Trie, reducing future search space.

### 5.2 Early Termination

```python
# Before starting DFS, check if any words remain
if not root.children:
    return result
```

### 5.3 Board Character Frequency Check

```python
from collections import Counter

def can_form_word(word, board_chars):
    word_count = Counter(word)
    return all(board_chars[c] >= word_count[c] for c in word_count)

# Filter words that can't possibly be formed
board_chars = Counter(c for row in board for c in row)
words = [w for w in words if can_form_word(w, board_chars)]
```

## 6. Alternative: Iterative with Stack

```python
def findWords_iterative(board, words):
    """Iterative version using explicit stack."""
    root = build_trie(words)
    m, n = len(board), len(board[0])
    result = []
    
    for i in range(m):
        for j in range(n):
            # Stack: (row, col, node, path, visited)
            stack = [(i, j, root, "", set())]
            
            while stack:
                row, col, node, path, visited = stack.pop()
                
                if (row, col) in visited:
                    continue
                
                char = board[row][col]
                if char not in node.children:
                    continue
                
                next_node = node.children[char]
                new_path = path + char
                new_visited = visited | {(row, col)}
                
                if next_node.word:
                    result.append(next_node.word)
                    next_node.word = None
                
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < m and 0 <= nc < n:
                        stack.append((nr, nc, next_node, new_path, new_visited))
    
    return result
```

## 7. Complexity Analysis

**Time:** O(M × N × 4^L)
- M × N starting positions
- 4^L possible paths (4 directions, L = max word length)
- Trie pruning typically reduces this significantly

**Space:** O(Total characters in words)
- Trie storage
- O(L) recursion depth

## 8. Common Mistakes

1. **Forgetting to unmark visited cells** during backtrack
2. **Not handling duplicates** - same word found via different paths
3. **Modifying board permanently** - always restore after DFS
4. **Not pruning empty Trie branches** - significant performance loss

## 9. Testing

```python
def test_word_search():
    s = Solution()
    
    # Basic test
    board1 = [['o','a','a','n'],['e','t','a','e'],['i','h','k','r'],['i','f','l','v']]
    words1 = ["oath","pea","eat","rain"]
    assert set(s.findWords(board1, words1)) == {"eat", "oath"}
    
    # Single cell
    board2 = [['a']]
    words2 = ["a", "ab"]
    assert s.findWords(board2, words2) == ["a"]
    
    # No matches
    board3 = [['a','b'],['c','d']]
    words3 = ["xyz"]
    assert s.findWords(board3, words3) == []
    
    print("All tests passed!")
```

## 10. Connection to Trie-based Search in ML

Word Search II demonstrates Trie's power for **multi-pattern matching**—a technique used extensively in:

- **Autocomplete**: Match user prefix against dictionary
- **Spell checkers**: Find similar words
- **Phonetic search**: Match speech transcripts

The key insight—**prune early using prefix structure**—applies directly to trie-based search systems in ML.

---

**Originally published at:** [arunbaby.com/dsa/0048-word-search-ii](https://www.arunbaby.com/dsa/0048-word-search-ii/)

*If you found this helpful, consider sharing it with others who might benefit.*
