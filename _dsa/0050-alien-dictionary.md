---
title: "Alien Dictionary"
day: 50
collection: dsa
categories:
  - dsa
tags:
  - graph
  - topological-sort
  - bfs
  - hard
  - string
difficulty: Hard
subdomain: "Graph Construction"
tech_stack: Python
scale: "Reconstructing alphabet from dictionary"
companies: Facebook, Google, Airbnb, Amazon
related_ml_day: 50
related_speech_day: 50
related_agents_day: 50
---

**"Language is just a graph of symbols. If you know the order, you know the language."**

## 1. Problem Statement

This is a legendary Facebook interview question. It combines String processing with Graph Theory.

There is a new alien language that uses the English alphabet. However, the order among the letters is unknown to you.
You are given a list of strings `words` from the alien language's dictionary, where the strings in `words` are **sorted lexicographically** by the rules of this new language.

Return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. If there is no solution, return `""`.

**Example 1:**
-   Input: `words = ["wrt","wrf","er","ett","rftt"]`
-   Output: `"wertf"`
-   Explanation:
    -   `wrt` vs `wrf`: `t` comes before `f`. (`t -> f`)
    -   `wrf` vs `er`: `w` comes before `e`. (`w -> e`)
    -   `er` vs `ett`: `r` comes before `t`. (`r -> t`)
    -   `ett` vs `rftt`: `e` comes before `r`. (`e -> r`)
    -   Combined: `w -> e -> r -> t -> f`.

---

## 2. Understanding the Problem

### 2.1 The Lexicographical Comparison
In a sorted dictionary (English):
-   "Apple" comes before "Approve".
-   Why? First mismatch. `l` vs `r`. `l` < `r`.
-   The characters *after* the first mismatch don't matter!

**Algorithm**: To find the rules, we only need to compare **adjacent words** in the list.
Comparing `word[0]` vs `word[1]` gives us one rule.
Comparing `word[0]` vs `word[99]` gives us redundant info (transitive property).

### 2.2 Graph Representation
-   Nodes: Unique characters in the list.
-   Edges: `u -> v` means `u` comes before `v` in the alphabet.
-   Task: Find a Topological Sort of this graph.

---

## 3. Approach 1: Graph Construction + Kahn's Algo

We break this into three phases.

### Phase 1: Initialize
Scan every word to find all unique characters. Initialize `Indegree = {c: 0}` and `Adj = {c: []}`.

### Phase 2: Build Edges
Iterate `words` from `0` to `N-1`.
Compare `w1` and `w2`.
-   Find first index `j` where `w1[j] != w2[j]`.
-   Add edge `w1[j] -> w2[j]`.
-   Increment Indegree of `w2[j]`.
-   **Break**. (Don't check further characters).

*Edge Case*: "Prefix Problem".
If `w1 = "abc"` and `w2 = "ab"`.
In a valid dictionary, prefix ("ab") must come BEFORE "abc".
If "abc" comes before "ab", the input is invalid. Return `""`.

### Phase 3: Topological Sort
Run Kahn's Algorithm (BFS) (Day 49).
If result length < num unique chars, there is a cycle (Invalid).

---

## 4. Implementation

```python
from collections import deque, defaultdict

class Solution:
    def alienOrder(self, words: list[str]) -> str:
        # 1. Initialize data structures
        adj = defaultdict(set)
        in_degree = {c: 0 for w in words for c in w}
        
        # 2. Build Graph
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            min_len = min(len(w1), len(w2))
            
            # Check for prefix error ("abc" before "ab")
            if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
                return ""
            
            for j in range(min_len):
                if w1[j] != w2[j]:
                    if w2[j] not in adj[w1[j]]:
                        adj[w1[j]].add(w2[j])
                        in_degree[w2[j]] += 1
                    break # Only the first mismatch matters
                    
        # 3. Topological Sort (Kahn's)
        queue = deque([c for c in in_degree if in_degree[c] == 0])
        result = []
        
        while queue:
            char = queue.popleft()
            result.append(char)
            
            for neighbor in adj[char]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        # 4. Cycle Check
        if len(result) < len(in_degree):
            return "" # Cycle detected
            
        return "".join(result)
```

---

## 5. Testing Strategy

### Test Case 1: Simple Line
Input: `["z", "x"]`.
-   `z` vs `x`. Edge `z -> x`.
-   Queue: `[z]`. Result `zx`.

### Test Case 2: Cycle
Input: `["z", "x", "z"]`.
-   `z -> x`.
-   `x -> z`.
-   Cycle. Queue empties early. Result length < 3. Return `""`.

### Test Case 3: Multiple Components
Input: `["a", "b", "c", "d"]` (where `a->b` and `c->d`, but no link between systems).
-   Result order between disconnected components doesn't matter (e.g., `ab cd` or `acbd`).
-   Any valid topo sort is accepted.

---

## 6. Complexity Analysis

Let $C$ be total length of all words (total characters).
Let $U$ be number of unique characters (at most 26).

-   **Time**: $O(C)$.
    -   We scan the words list. Comparing two words takes $O(L)$. Total edge building is $O(C)$.
    -   Topo sort takes $O(U + E)$. Since $U \le 26$, this is constant time relative to input size!
    -   Dominant term is scanning the input.
-   **Space**: $O(1)$ or $O(U + E)$. Since $U$ is fixed at 26, the graph is tiny.

---

## 7. Production Considerations

This logic is used in **Versioning Systems** (SemVer).
"Determine if v1.10.2 > v1.2.9".
It parses the string into segments and performs the exact same "First Mismatch" logic used here.

---

## 8. Connections to ML Systems

This relates to **Character-Level LMs** (ML Day 50).
-   In Alien Dictionary, we **infer** the rules of the language from raw data.
-   In Char-RNN, the model **infers** the probability $P(char_t | char_{t-1})$.
-   Both are "Language Modeling" problems at the atomic level.

---

## 9. Interview Strategy

1.  **Don't skip the Prefix Check**: This is the most common reason to fail. Mention `abc` vs `ab` explicitly.
2.  **Explain "Set" for Edges**: Using `list` for adjacency can lead to duplicate edges if the same rule appears twice (`wrt, wrf` and later `art, arf`). Use a `Set` or check before adding.
3.  **Variable Names**: Use `adj` and `in_degree`. Clear naming helps debugging.

---

## 10. Key Takeaways

1.  **Information Theory**: You only get information from the *first difference*. Everything after is noise.
2.  **Graph Construction**: Sometimes the graph isn't given; you have to build it by observing constraints.
3.  **Cycles**: Order means "No Cycles". Always cycle check.

---

**Originally published at:** [arunbaby.com/dsa/0050-alien-dictionary](https://www.arunbaby.com/dsa/0050-alien-dictionary/)

*If you found this helpful, consider sharing it with others who might benefit.*
