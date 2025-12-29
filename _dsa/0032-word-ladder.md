---
title: "Word Ladder (BFS)"
day: 32
related_ml_day: 32
related_speech_day: 32
related_agents_day: 32
collection: dsa
categories:
 - dsa
tags:
 - graph
 - bfs
 - shortest path
 - string
difficulty: Hard
---

**"Transforming 'cold' to 'warm' one letter at a time."**

## 1. Problem Statement

A **transformation sequence** from word `beginWord` to word `endWord` using a dictionary `wordList` is a sequence of words `beginWord -> s1 -> s2 -> ... -> sk` such that:
1. Every adjacent pair of words differs by a single letter.
2. Every `si` for `1 <= i <= k` is in `wordList`. Note that `beginWord` does not need to be in `wordList`.
3. `sk == endWord`.

Given two words, `beginWord` and `endWord`, and a dictionary `wordList`, return the **number of words** in the shortest transformation sequence from `beginWord` to `endWord`, or `0` if no such sequence exists.

**Example 1:**
``
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: hit -> hot -> dot -> dog -> cog
``

**Example 2:**
``
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: 0
Explanation: The endWord "cog" is not in wordList, therefore no valid transformation sequence exists.
``

## 2. BFS Solution (Shortest Path in Unweighted Graph)

**Intuition:**
- Treat each word as a node in a graph.
- An edge exists between two nodes if they differ by exactly one letter.
- The problem becomes finding the **shortest path** from `beginWord` to `endWord`.
- **BFS (Breadth-First Search)** is the standard algorithm for finding the shortest path in an unweighted graph because it explores nodes layer by layer.

**Graph Construction Strategy:**
There are two main ways to find neighbors of a word:
1. **Compare with all other words:** For a word `W`, iterate through the entire `wordList` and check if the difference is 1 character. This takes O(N \cdot M), where `N` is the number of words and `M` is the word length. Doing this for every word in BFS gives O(N^2 \cdot M). This is too slow if `N` is large (e.g., 5000 words).
2. **Generate all possible neighbors:** For a word `W`, change each of its characters to 'a' through 'z'. This generates `26 \cdot M` potential words. Check if each potential word exists in `wordSet`. This takes O(26 \cdot M \cdot M) (hashing takes O(M)). This is much faster when `N` is large but `M` is small (e.g., `M=5`).

**Algorithm:**
1. Convert `wordList` to a set `wordSet` for O(1) lookups.
2. Initialize a queue with `(beginWord, 1)`. The level starts at 1 because the sequence length includes the start word.
3. Initialize a `visited` set to avoid cycles and redundant processing.
4. While the queue is not empty:
 - Dequeue the current word and its level.
 - If the word is `endWord`, return the level.
 - Generate all possible neighbors by changing one character at a time.
 - If a neighbor is in `wordSet` and not visited, add it to the queue and mark as visited.

``python
from collections import deque
from typing import List

class Solution:
 def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
 wordSet = set(wordList)
 if endWord not in wordSet:
 return 0
 
 queue = deque([(beginWord, 1)]) # (current_word, level)
 visited = set([beginWord])
 
 while queue:
 word, level = queue.popleft()
 
 if word == endWord:
 return level
 
 # Generate all possible neighbors
 for i in range(len(word)):
 original_char = word[i]
 for char_code in range(ord('a'), ord('z') + 1):
 char = chr(char_code)
 if char == original_char:
 continue
 
 # Create new word: hit -> ait, bit, ... hat, hbt ...
 new_word = word[:i] + char + word[i+1:]
 
 if new_word in wordSet and new_word not in visited:
 visited.add(new_word)
 queue.append((new_word, level + 1))
 
 return 0
``

**Time Complexity Analysis:**
- **Preprocessing:** Converting list to set takes O(N \cdot M).
- **BFS Traversal:** In the worst case, we visit every word in the `wordList`.
- **Neighbor Generation:** For each word, we iterate through its length `M`. For each position, we try 26 characters. Creating the new string takes O(M). Checking existence in the set takes O(M) (average).
- Total Complexity: O(N \cdot M^2).
 - If `N = 5000` and `M = 5`, `N \cdot M^2 = 5000 \cdot 25 = 125,000` operations.
 - Comparing all pairs would be `N^2 \cdot M = 25,000,000 \cdot 5 = 125,000,000`.
 - The generation approach is significantly faster for typical constraints.

**Space Complexity:**
- O(N \cdot M) to store `wordSet`, `visited` set, and the queue.

## 3. Bi-directional BFS (Optimization)

**Intuition:**
Standard BFS searches a tree that grows exponentially. If the branching factor is `b` and the distance to the target is `d`, BFS visits roughly `b^d` nodes.
**Bi-directional BFS** runs two simultaneous searches: one from the start and one from the end. They meet in the middle.
- Search 1: Start -> Middle (distance `d/2`)
- Search 2: End -> Middle (distance `d/2`)
- Total nodes visited: `b^{d/2} + b^{d/2} = 2 \cdot b^{d/2}`.
- This is exponentially smaller than `b^d`.

**Algorithm:**
1. Maintain two sets: `beginSet` (initially `{beginWord}`) and `endSet` (initially `{endWord}`).
2. Maintain a `visited` set containing all words visited by either search.
3. In each step, always expand the **smaller set**. This keeps the search balanced and minimizes the number of generated neighbors.
4. For each word in the current set, generate neighbors.
5. If a neighbor is found in the **opposite set**, the two searches have met! Return `level + 1`.
6. Otherwise, if the neighbor is valid (in `wordSet` and not visited), add it to the next layer.

``python
class Solution:
 def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
 wordSet = set(wordList)
 if endWord not in wordSet:
 return 0
 
 beginSet = {beginWord}
 endSet = {endWord}
 visited = {beginWord, endWord}
 length = 1
 
 while beginSet and endSet:
 # Always expand the smaller set to minimize work
 if len(beginSet) > len(endSet):
 beginSet, endSet = endSet, beginSet
 
 newSet = set()
 for word in beginSet:
 for i in range(len(word)):
 original_char = word[i]
 for char_code in range(ord('a'), ord('z') + 1):
 char = chr(char_code)
 if char == original_char:
 continue
 
 new_word = word[:i] + char + word[i+1:]
 
 # If the neighbor is in the opposite set, we connected the paths
 if new_word in endSet:
 return length + 1
 
 if new_word in wordSet and new_word not in visited:
 visited.add(new_word)
 newSet.add(new_word)
 
 beginSet = newSet
 length += 1
 
 return 0
``

**Performance Comparison:**
Imagine a graph where each node has 10 neighbors (`b=10`) and the shortest path is 6 steps (`d=6`).
- **Standard BFS:** Visits `\approx 10^6 = 1,000,000` nodes.
- **Bi-directional BFS:** Visits `\approx 2 \times 10^3 = 2,000` nodes.
- **Speedup:** 500x faster!

## 4. Word Ladder II (Return All Shortest Paths)

**Problem:** Instead of just the length, return *all* shortest transformation sequences.
Example: `hit -> hot -> dot -> dog -> cog` AND `hit -> hot -> lot -> log -> cog`.

**Challenge:**
- We need to store the path structure.
- Standard BFS only stores the distance.
- Storing full paths in the queue consumes exponential memory.

**Optimized Approach:**
1. **BFS for Graph Building:** Run BFS from `beginWord` to find the shortest distance to `endWord`. While doing this, build a **DAG (Directed Acyclic Graph)** or a `parents` map where `parents[child]` contains all `parents` that lead to `child` with the shortest distance.
 - Crucial: Only add edges from level `L` to level `L+1`. Do not add edges within the same level or back to previous levels.
2. **DFS for Path Reconstruction:** Run DFS (Backtracking) from `endWord` back to `beginWord` using the `parents` map to reconstruct all paths.

``python
from collections import defaultdict

class Solution:
 def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
 wordSet = set(wordList)
 if endWord not in wordSet:
 return []
 
 # BFS initialization
 layer = {beginWord}
 parents = defaultdict(set) # word -> set of parents
 wordSet.discard(beginWord) # Remove start word to avoid cycles
 
 found = False
 while layer and not found:
 next_layer = set()
 # Remove words in current layer from wordSet to prevent visiting them again in future layers
 # But allow visiting them multiple times in the *current* layer (for multiple paths)
 for word in layer:
 if word in wordSet:
 wordSet.remove(word)
 
 # Actually, we need to remove words visited in the *current* layer from wordSet *after* processing the layer
 # Let's refine the logic:
 # 1. Generate next layer
 # 2. If endWord found, stop after this layer
 # 3. Remove next_layer words from wordSet
 
 current_layer_visited = set()
 
 for word in layer:
 for i in range(len(word)):
 for char_code in range(ord('a'), ord('z') + 1):
 new_word = word[:i] + chr(char_code) + word[i+1:]
 
 if new_word == endWord:
 parents[endWord].add(word)
 found = True
 elif new_word in wordSet:
 next_layer.add(new_word)
 parents[new_word].add(word)
 current_layer_visited.add(new_word)
 
 wordSet -= current_layer_visited
 layer = next_layer
 
 if not found:
 return []
 
 # DFS Backtracking to reconstruct paths
 res = []
 def backtrack(current_word, path):
 if current_word == beginWord:
 res.append(path[::-1]) # Reverse path to get begin -> end
 return
 
 for parent in parents[current_word]:
 path.append(parent)
 backtrack(parent, path)
 path.pop()
 
 backtrack(endWord, [endWord])
 return res
``

**Complexity of Word Ladder II:**
- **Time:** The number of shortest paths can be exponential. In the worst case, we might traverse a huge number of paths. However, the BFS part is still polynomial. The DFS part depends on the output size.
- **Space:** Storing the `parents` map is proportional to the number of edges in the shortest-path DAG.

## 5. Deep Dive: Pre-processing for Faster Neighbor Generation

If `wordList` is very sparse (e.g., words are 10 chars long, but only 1000 words exist), the `26 \cdot M` generation strategy might generate many invalid words.
We can pre-process the dictionary using a **wildcard map**.

**Concept:**
- Word: `hot`
- Intermediate states: `*ot`, `h*t`, `ho*`
- Map:
 - `*ot` -> `[hot, dot, lot]`
 - `h*t` -> `[hot, hit, hat]`
 - `ho*` -> `[hot, how]`

**Algorithm:**
1. Build the `all_combo_dict`.
2. In BFS, for a word `hot`, look up `*ot`, `h*t`, `ho*` in the dictionary.
3. The values are the direct neighbors.

``python
from collections import defaultdict

class Solution:
 def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
 if endWord not in wordList:
 return 0
 
 # Pre-processing
 L = len(beginWord)
 all_combo_dict = defaultdict(list)
 for word in wordList:
 for i in range(L):
 all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)
 
 # BFS
 queue = deque([(beginWord, 1)])
 visited = {beginWord}
 
 while queue:
 current_word, level = queue.popleft()
 for i in range(L):
 intermediate_word = current_word[:i] + "*" + current_word[i+1:]
 
 for neighbor in all_combo_dict[intermediate_word]:
 if neighbor == endWord:
 return level + 1
 if neighbor not in visited:
 visited.add(neighbor)
 queue.append((neighbor, level + 1))
 
 # Optimization: Clear the list to prevent reprocessing
 # all_combo_dict[intermediate_word] = [] 
 
 return 0
``

**Trade-offs:**
- **Pros:** Very fast neighbor lookup O(M) instead of O(26 \cdot M).
- **Cons:** High memory usage to store the dictionary. If words are dense (many words match `*ot`), the adjacency lists become long.

## 6. Deep Dive: A* Search (Heuristic Search)

BFS is "blind"—it explores in all directions equally. **A* Search** uses a heuristic to prioritize paths that seem closer to the target.

**Heuristic Function `h(n)`:**
We need an admissible heuristic (never overestimates the cost).
- **Hamming Distance:** The number of positions where characters differ.
 - `hit` vs `cog`: 3 differences.
 - `hot` vs `cog`: 2 differences.
 - `hot` is "closer" to `cog` than `hit`.

**Algorithm:**
Use a Priority Queue instead of a standard Queue. Priority = `g(n) + h(n)`, where:
- `g(n)`: Cost from start to current node (current level).
- `h(n)`: Estimated cost from current node to end.

``python
import heapq

class Solution:
 def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
 wordSet = set(wordList)
 if endWord not in wordSet:
 return 0
 
 def heuristic(word):
 return sum(c1 != c2 for c1, c2 in zip(word, endWord))
 
 # Priority Queue: (estimated_total_cost, current_cost, word)
 pq = [(heuristic(beginWord) + 1, 1, beginWord)]
 visited = {beginWord: 1} # word -> cost
 
 while pq:
 _, cost, word = heapq.heappop(pq)
 
 if word == endWord:
 return cost
 
 if cost > visited.get(word, float('inf')):
 continue
 
 for i in range(len(word)):
 original_char = word[i]
 for char_code in range(ord('a'), ord('z') + 1):
 char = chr(char_code)
 if char == original_char:
 continue
 
 new_word = word[:i] + char + word[i+1:]
 
 if new_word in wordSet:
 new_cost = cost + 1
 if new_cost < visited.get(new_word, float('inf')):
 visited[new_word] = new_cost
 priority = new_cost + heuristic(new_word)
 heapq.heappush(pq, (priority, new_cost, new_word))
 
 return 0
``

**Why A* isn't always better here:**
In high-dimensional spaces like this (where branching factor is ~25), the heuristic (Hamming distance) is often not strong enough to prune the search space significantly compared to Bi-directional BFS. Bi-directional BFS is usually the winner for this specific problem.

## 7. Deep Dive: Real-world Applications

While transforming "hit" to "cog" is a puzzle, the underlying concepts have serious applications:

1. **Genetic Sequencing (Edit Distance):**
 - DNA sequences can be modeled as strings.
 - Mutations are single-character changes.
 - Finding the shortest path between two DNA sequences helps trace evolutionary history.

2. **Spell Checking:**
 - If a user types "wrd", we want to find the closest valid word.
 - This is a BFS of depth 1 or 2 from the typo.

3. **Error Correction Codes:**
 - Hamming codes allow detecting and correcting single-bit errors.
 - This is essentially finding the nearest valid "codeword" in the graph of all possible binary strings.

4. **Semantic Word Embeddings:**
 - In NLP, we often want to move from one concept to another.
 - "King" - "Man" + "Woman" = "Queen".
 - This is a continuous version of a word ladder in vector space.

## 8. Deep Dive: Variations and Constraints

**Variation 1: Variable Word Lengths**
- **Rule:** You can insert, delete, or replace a character.
- **Graph:** Neighbors include `hot` -> `ho` (delete), `hot` -> `shot` (insert), `hot` -> `hat` (replace).
- **Complexity:** Branching factor increases significantly (`26 \cdot M` replacements + `M` deletions + `26 \cdot (M+1)` insertions).

**Variation 2: Weighted Edges**
- **Rule:** Changing a vowel costs 2, consonant costs 1.
- **Algorithm:** Use **Dijkstra's Algorithm** instead of BFS.

**Variation 3: Constraint Satisfaction**
- **Rule:** Must pass through a specific word (e.g., `hit` -> ... -> `dot` -> ... -> `cog`).
- **Algorithm:** Run BFS twice: `hit` -> `dot` AND `dot` -> `cog`. Sum the lengths.

## Comparison Table

| Approach | Time Complexity | Space Complexity | Pros | Cons |
|:---|:---|:---|:---|:---|
| **Simple BFS** | O(M^2 N) | O(MN) | Simple, guarantees shortest path | Slow for large graphs |
| **Bi-directional BFS** | O(M^2 N^{0.5}) | O(MN^{0.5}) | Fastest for 2-point search | More code, needs start/end known |
| **Word Ladder II** | Exponential | Exponential | Finds ALL paths | Very memory intensive |
| **A* Search** | Depends on heuristic | O(MN) | Good for single path | Heuristic overhead, PQ overhead |
| **Pre-processed Map** | O(M^2 N) | O(M^2 N) | Fast neighbor lookup | High memory usage |

## Implementation in Other Languages

**C++:**
``cpp
class Solution {
public:
 int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
 unordered_set<string> wordSet(wordList.begin(), wordList.end());
 if (wordSet.find(endWord) == wordSet.end()) return 0;
 
 queue<pair<string, int>> q;
 q.push({beginWord, 1});
 
 while (!q.empty()) {
 auto [word, level] = q.front();
 q.pop();
 
 if (word == endWord) return level;
 
 for (int i = 0; i < word.size(); ++i) {
 char original = word[i];
 for (char c = 'a'; c <= 'z'; ++c) {
 if (c == original) continue;
 word[i] = c;
 if (wordSet.count(word)) {
 q.push({word, level + 1});
 wordSet.erase(word); // Mark as visited
 }
 }
 word[i] = original;
 }
 }
 return 0;
 }
};
``

**Java:**
``java
class Solution {
 public int ladderLength(String beginWord, String endWord, List<String> wordList) {
 Set<String> wordSet = new HashSet<>(wordList);
 if (!wordSet.contains(endWord)) return 0;
 
 Queue<String> queue = new LinkedList<>();
 queue.offer(beginWord);
 int level = 1;
 
 while (!queue.isEmpty()) {
 int size = queue.size();
 for (int i = 0; i < size; i++) {
 String currentWord = queue.poll();
 char[] wordChars = currentWord.toCharArray();
 
 for (int j = 0; j < wordChars.length; j++) {
 char originalChar = wordChars[j];
 for (char c = 'a'; c <= 'z'; c++) {
 if (c == originalChar) continue;
 wordChars[j] = c;
 String newWord = new String(wordChars);
 
 if (newWord.equals(endWord)) return level + 1;
 
 if (wordSet.contains(newWord)) {
 queue.offer(newWord);
 wordSet.remove(newWord);
 }
 }
 wordChars[j] = originalChar;
 }
 }
 level++;
 }
 return 0;
 }
}
``

## Top Interview Questions

**Q1: What if the word list is too large to fit in memory?**
*Answer:*
If the list is on disk, we can't use a hash set for O(1) lookups.
1. **Bloom Filter:** Load a Bloom Filter into memory. It can tell us if a word is *definitely not* in the set or *probably* in the set. If probably, check disk.
2. **Trie (Prefix Tree):** Store words in a Trie to save space (shared prefixes).
3. **Distributed Search:** Partition the words across multiple machines (sharding).

**Q2: How would you handle words of different lengths?**
*Answer:*
The problem definition implies equal lengths. If we allow insertions/deletions (Edit Distance = 1), we generate neighbors by:
1. **Substitution:** `hit` -> `hat`
2. **Insertion:** `hit` -> `hits`, `chit`
3. **Deletion:** `hit` -> `hi`, `it`
We must check if these new words exist in the dictionary.

**Q3: Can we use DFS?**
*Answer:*
DFS is **not suitable** for finding the shortest path in an unweighted graph.
- DFS dives deep. It might find a path of length 100 before finding the optimal path of length 5.
- To find the shortest path with DFS, you'd have to explore *all* paths and compare them, which is O(N!). BFS finds the shortest path as soon as it reaches the target.

**Q4: What is the maximum number of edges from a word?**
*Answer:*
For a word of length `M` and an alphabet size of 26:
- Each of the `M` positions can be changed to 25 other characters.
- Total potential neighbors = `M \times 25`.
- However, the number of *valid* edges depends on how many of these potential neighbors actually exist in `wordList`.

**Q5: How does Bi-directional BFS handle the meeting point?**
*Answer:*
The meeting point is not necessarily a single node. It's an edge.
- Search A reaches node `U`.
- Search B reaches node `V`.
- If `V` is a neighbor of `U`, the paths connect.
- The total length is `level(U) + level(V)`. In my implementation, I check `if new_word in endSet`, which effectively checks for this connection.

## Key Takeaways

1. **Graph Modeling:** The core skill is recognizing that words are nodes and edits are edges.
2. **BFS for Shortest Path:** Always the first choice for unweighted shortest path problems.
3. **Bi-directional BFS:** A critical optimization for search problems where start and end are known. It reduces the search space from `b^d` to `2 \cdot b^{d/2}`.
4. **Neighbor Generation:** Iterating 'a'-'z' (`26M`) is usually faster than iterating the word list (`N`) when `N` is large.
5. **Word Ladder II:** Requires a two-phase approach: BFS to build the shortest-path graph, then DFS to extract paths.

## Summary

| Aspect | Insight |
|:---|:---|
| **Core Problem** | Shortest path in a graph of words |
| **Best Algorithm** | Bi-directional BFS |
| **Key Optimization** | Generate neighbors by swapping chars, not iterating list |
| **Applications** | Spell checking, genetic sequencing, puzzle solving |

---

**Originally published at:** [arunbaby.com/dsa/0032-word-ladder](https://www.arunbaby.com/dsa/0032-word-ladder/)

*If you found this helpful, consider sharing it with others who might benefit.*
