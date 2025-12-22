---
title: "Alien Dictionary"
day: 50
collection: dsa
categories:
  - dsa
tags:
  - graph
  - topological-sort
  - string-processing
  - directed-graph
  - hard
difficulty: Hard
subdomain: "Graph & Strings"
tech_stack: Python
scale: "O(C) where C is total characters in input"
companies: Facebook, Google, Airbnb, Amazon
related_ml_day: 50
related_speech_day: 50
related_agents_day: 50
---

**"If 'apple' comes before 'banana' but 'bat' comes before 'arc', what is the alphabet?"**

## 1. Introduction: The Decryption Problem

You have fond a list of words from an Alien Language. You don't know their alphabet order. However, you know the list of words is **sorted lexicographically** (alphabetically) according to their rules.

**Input**:
`["wrt", "wrf", "er", "ett", "rftt"]`

**Output**: `"wertf"`

Why?
- `wrt` comes before `wrf` -> `t` comes before `f`.
- `wrf` comes before `er` -> `w` comes before `e`.
- `er` comes before `ett` -> `r` comes before `t`.
- `ett` comes before `rftt` -> `e` comes before `r`.

Combining these clues: `w -> e -> r -> t -> f`.

This is the **Alien Dictionary** problem. It is a fantastic test of your ability to model a problem. It combines **String Processing** (extracting rules) with **Graph Theory** (ordering dependencies).

---

## 2. Breaking it Down

This is a 2-step problem.

### Step 1: Build the Graph (Extract Dependencies)
We need to turn the word list into a dependency graph.
Dependency Rule: "Letter A comes before Letter B" is an edge `A -> B`.

**How to find an edge?**
Compare **adjacent words** in the list.
1. Compare `wrt` and `wrf`.
   - `w` == `w` (skip)
   - `r` == `r` (skip)
   - `t` != `f`. Since `wrt` is first, `t` must be smaller than `f`.
   - **Edge**: `t -> f`.
   - *Stop comparison for this pair*. (Characters after the first difference don't matter for sorting).

2. Compare `wrf` and `er`.
   - `w` != `e`.
   - **Edge**: `w -> e`.
   - Stop.

**Crucial Edge Case**: Prefix ordering.
If words are `["apple", "app"]`, this is **Invalid**. In a valid dictionary, the shorter prefix (`app`) must always come before the longer word (`apple`). If we see this, return `""`.

### Step 2: Topological Sort
Once we have the graph, finding the alphabet is just finding a **Topological Sort** (Order where parents come before children).

We can use **Kahn's Algorithm** (BFS) just like in Course Schedule (Day 49).

---

## 3. The Algorithm

1. **Initialize**: Map `graph` using a Set (to avoid duplicates) and `indegree` map for *every unique character* found in words.
2. **Build Graph**: Zip adjacent words together. Find first diff. update graph and indegrees.
3. **BFS**:
   - Queue = chars with 0 in-degree.
   - Result = string buffer.
   - While Queue not empty:
     - Pop char `C`. Append to Result.
     - For neighbor `N` in `graph[C]`:
       - `indegree[N] -= 1`
       - If `indegree[N] == 0`, push `N` to Queue.
4. **Validation**:
   - If `len(result) < num_unique_chars`, there is a cycle (e.g., A < B and B < A). Return `""`.

---

## 4. Visualizing the Example

Input: `["z", "x", "z"]` (Wait, `z` before `x` then `x` before `z`?)
1. Pair (`z`, `x`) -> Edge `z -> x`.
2. Pair (`x`, `z`) -> Edge `x -> z`.
Graph: `z <-> x` (Cycle).
Result: `""`.

Input: `["wrt", "wrf", "er", "ett", "rftt"]`
1. `wrt, wrf` -> `t -> f`.
2. `wrf, er` -> `w -> e`.
3. `er, ett` -> `r -> t`.
4. `ett, rftt` -> `e -> r`.

Graph:
`w -> e`
`e -> r`
`r -> t`
`t -> f`
In-degrees: `w:0, e:1, r:1, t:1, f:1`.

Queue: `[w]`
- Pop `w`. Result: "w". Reduce `e`. Queue: `[e]`
- Pop `e`. Result: "we". Reduce `r`. Queue: `[r]`
- Pop `r`. Result: "wer". Reduce `t`. Queue: `[t]`
- Pop `t`. Result: "wert". Reduce `f`. Queue: `[f]`
- Pop `f`. Result: "wertf".

---

## 5. Implementation

```python
from collections import deque, defaultdict

class Solution:
    def alienOrder(self, words: List[str]) -> str:
        # 0. Initialize Data Structures
        adj = defaultdict(set)
        in_degree = {c: 0 for word in words for c in word}
        
        # 1. Build Graph
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            min_len = min(len(w1), len(w2))
            
            # Check for invalid prefix case ("apple", "app")
            if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
                return ""
                
            for j in range(min_len):
                if w1[j] != w2[j]:
                    if w2[j] not in adj[w1[j]]:
                        adj[w1[j]].add(w2[j])
                        in_degree[w2[j]] += 1
                    break # Only the first difference matters!
                    
        # 2. Topological Sort (Kahn's)
        queue = deque([c for c in in_degree if in_degree[c] == 0])
        result = []
        
        while queue:
            c = queue.popleft()
            result.append(c)
            
            for neighbor in adj[c]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        # 3. Check for cycles
        if len(result) < len(in_degree):
            return ""
            
        return "".join(result)
```

---

## 6. Time & Space Complexity

- **Time**: `O(C)`
  - C is the total number of characters in all words.
  - We iterate through the word list once to build the graph.
  - The graph has at most 26 nodes (lowercase English letters). BFS on a small graph is constant time relative to the input C.
  
- **Space**: `O(1)` or `O(U)`
  - U is unique characters (max 26). The graph is tiny.
  - Adjacency list size is bounded by 26*26.

---

## 7. Real World Connection

This isn't just about Aliens.
This logic is used in **Data Reconciliation**.
If you have two different sorted lists of data (e.g., versions of a file), and you want to deduce the implicit sorting rule (Order by Date? Order by Size?), you can use this differential analysis.

It is also similar to how **LLMs learn implicit rules** from sequence data. They see `A` mostly followed by `B`, and learn the structure. Here, we deterministically extract the rigid structure.

---

**Originally published at:** [arunbaby.com/dsa/0050-alien-dictionary](https://www.arunbaby.com/dsa/0050-alien-dictionary/)

*If you found this helpful, consider sharing it with others who might benefit.*
