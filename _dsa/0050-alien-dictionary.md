---
title: "Alien Dictionary"
day: 50
collection: dsa
categories:
  - dsa
tags:
  - graphs
  - topological-sort
  - dfs
  - bfs
  - string
  - ordering
difficulty: Hard
subdomain: "Graphs & Strings"
tech_stack: Python
scale: "O(C) where C is total characters across all words"
companies: Google, Meta, Amazon, Microsoft, Airbnb
related_ml_day: 50
related_speech_day: 50
related_agents_day: 50
---

**"Given sorted words in an alien language, deduce the alphabet order."**

## 1. Problem Statement

There is a new alien language that uses the English alphabet. The order of letters is unknown. You're given a list of strings `words` from the dictionary, where the strings are **sorted lexicographically** by the rules of this new language.

Derive the order of letters in this language. If invalid, return `""`. If multiple valid orderings exist, return any one.

**Example 1:**
```
words = ["wrt", "wrf", "er", "ett", "rftt"]
Output: "wertf"

Explanation:
- "wrt" < "wrf" → 't' < 'f'
- "wrt" < "er" → 'w' < 'e'  
- "er" < "ett" → 'r' < 't'
- "ett" < "rftt" → 'e' < 'r'

Order: w → e → r → t → f
```

**Example 2:**
```
words = ["z", "x"]
Output: "zx"
```

**Example 3:**
```
words = ["z", "x", "z"]
Output: "" (invalid: z can't be both before and after x)
```

## 2. Understanding the Problem

This is topological sort with edge extraction from sorted words:

1. **Extract ordering constraints** by comparing adjacent words
2. **Build a directed graph** where edge (a, b) means a < b
3. **Topological sort** to get valid character ordering
4. **Detect cycles** = invalid input

### Key Insight: Adjacent Word Comparison

```
"wrt" vs "wrf":
w == w (continue)
r == r (continue)
t != f → t comes before f

Only the FIRST different character tells us ordering!
```

## 3. Solution

```python
from typing import List
from collections import defaultdict, deque

class Solution:
    def alienOrder(self, words: List[str]) -> str:
        """
        Derive alien alphabet order from sorted words.
        
        Steps:
        1. Build graph from adjacent word comparisons
        2. Topological sort with cycle detection
        
        Time: O(C) where C = total chars in all words
        Space: O(1) since alphabet size is fixed (26)
        """
        # Initialize graph with all unique characters
        graph = defaultdict(set)  # char -> set of chars that come after
        in_degree = {c: 0 for word in words for c in word}
        
        # Build graph by comparing adjacent words
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            
            # Check for invalid case: "abc" before "ab"
            if len(word1) > len(word2) and word1[:len(word2)] == word2:
                return ""
            
            # Find first different character
            for c1, c2 in zip(word1, word2):
                if c1 != c2:
                    # c1 comes before c2
                    if c2 not in graph[c1]:
                        graph[c1].add(c2)
                        in_degree[c2] += 1
                    break  # Only first difference matters!
        
        # Kahn's algorithm for topological sort
        queue = deque([c for c in in_degree if in_degree[c] == 0])
        result = []
        
        while queue:
            char = queue.popleft()
            result.append(char)
            
            for next_char in graph[char]:
                in_degree[next_char] -= 1
                if in_degree[next_char] == 0:
                    queue.append(next_char)
        
        # Check for cycle
        if len(result) != len(in_degree):
            return ""
        
        return "".join(result)
```

## 4. DFS Alternative

```python
def alienOrder_dfs(self, words: List[str]) -> str:
    """DFS-based topological sort with cycle detection."""
    # Build graph
    graph = defaultdict(set)
    chars = set(c for word in words for c in word)
    
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        
        if len(w1) > len(w2) and w1[:len(w2)] == w2:
            return ""
        
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                graph[c1].add(c2)
                break
    
    # DFS with cycle detection
    # 0: unvisited, 1: visiting, 2: visited
    state = {c: 0 for c in chars}
    result = []
    
    def dfs(char):
        if state[char] == 1:
            return False  # Cycle!
        if state[char] == 2:
            return True
        
        state[char] = 1
        for next_char in graph[char]:
            if not dfs(next_char):
                return False
        
        state[char] = 2
        result.append(char)  # Add in reverse post-order
        return True
    
    for char in chars:
        if not dfs(char):
            return ""
    
    return "".join(reversed(result))
```

## 5. Walkthrough

```
words = ["wrt", "wrf", "er", "ett", "rftt"]

Step 1: Extract edges
- "wrt" vs "wrf": t → f
- "wrf" vs "er": w → e
- "er" vs "ett": r → t
- "ett" vs "rftt": e → r

Step 2: Build graph
w → [e]
e → [r]
r → [t]
t → [f]
f → []

Step 3: In-degrees
w: 0, e: 1, r: 1, t: 1, f: 1

Step 4: Kahn's algorithm
Queue: [w]
- Pop w, result = [w], reduce e to 0
Queue: [e]
- Pop e, result = [w,e], reduce r to 0
Queue: [r]
- Pop r, result = [w,e,r], reduce t to 0
Queue: [t]
- Pop t, result = [w,e,r,t], reduce f to 0
Queue: [f]
- Pop f, result = [w,e,r,t,f]

Output: "wertf"
```

## 6. Edge Cases

### Invalid: Prefix comes after full word

```python
words = ["abc", "ab"]
# Invalid! "abc" cannot come before "ab" in ANY ordering
# (a prefix can't come after its extension)
```

### Cycle detection

```python
words = ["z", "x", "z"]
# z → x (from ["z", "x"])
# x → z (from ["x", "z"])
# Cycle! Return ""
```

### Single word

```python
words = ["abc"]
# Output: "abc" or "bca" or "cab" (any order with a, b, c)
# Multiple valid answers since no constraints
```

### Multiple valid orderings

```python
words = ["ab", "cd"]
# a → nothing, b → nothing
# c → nothing, d → nothing
# Only constraint: a before b (implied by "ab")
# Valid: "abcd", "acbd", "cabd", etc.
```

## 7. Testing

```python
def test_alien_dictionary():
    s = Solution()
    
    # Basic
    assert s.alienOrder(["wrt","wrf","er","ett","rftt"]) == "wertf"
    
    # Simple
    assert s.alienOrder(["z","x"]) == "zx"
    
    # Invalid order
    assert s.alienOrder(["z","x","z"]) == ""
    
    # Invalid prefix
    assert s.alienOrder(["abc","ab"]) == ""
    
    # Single letter
    assert s.alienOrder(["z"]) == "z"
    
    # All same
    assert s.alienOrder(["a","a"]) == "a"
    
    print("All tests passed!")
```

## 8. Complexity Analysis

**Time:** O(C)
- C = total characters across all words
- Each character pair compared once
- Topological sort is O(V + E) where V ≤ 26, E ≤ 26²

**Space:** O(1) or O(26²)
- Graph has at most 26 nodes and 26² edges
- Constant if we consider alphabet size fixed

## 9. Common Mistakes

1. **Only using first word pair**: Must compare ALL adjacent pairs
2. **Not handling prefix case**: "abc" before "ab" is invalid
3. **Missing characters**: Include ALL characters, not just those in edges
4. **Stopping after first difference**: Correct—only first diff matters

## 10. Connection to Language Models

Alien Dictionary is about **inferring structure from examples**—exactly what language models do:

| Alien Dictionary | Language Modeling |
|-----------------|-------------------|
| Word order → Char order | Text → Probability distribution |
| Constraints | Training data |
| Topological sort | Learning algorithm |
| Invalid (cycle) | Contradictory data |

Both extract **hidden rules** from observed sequences.

---

**Originally published at:** [arunbaby.com/dsa/0050-alien-dictionary](https://www.arunbaby.com/dsa/0050-alien-dictionary/)

*If you found this helpful, consider sharing it with others who might benefit.*
