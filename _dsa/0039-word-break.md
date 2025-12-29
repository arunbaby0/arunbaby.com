---
title: "Word Break"
day: 39
related_ml_day: 39
related_speech_day: 39
related_agents_day: 39
collection: dsa
categories:
 - dsa
tags:
 - dynamic-programming
 - trie
 - bfs
 - string
difficulty: Medium
---

**"Making sense of a stream of characters."**

## 1. Problem Statement

Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

**Example 1:**
``
Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
``

**Example 2:**
``
Input: s = "applepenapple", wordDict = ["apple", "pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.
``

**Example 3:**
``
Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
Output: false
``

## 2. Intuition

This problem asks if we can split the string `s` into valid substrings. This structure suggests **Dynamic Programming** or **Recursion**.

If we want to know if `s[0...n]` is valid, we can check if there exists a split point `j` such that `s[0...j]` is a valid word AND `s[j...n]` can be broken into valid words.

This gives us the optimal substructure property:
`WordBreak(s) = (s[0:i] in Dict) AND WordBreak(s[i:])` for some `i`.

## 3. Approach 1: Recursion with Memoization (Top-Down DP)

We define a function `canBreak(start_index)` that returns true if the substring `s[start_index:]` can be segmented.

**Algorithm:**
1. Base Case: If `start_index == len(s)`, we have successfully segmented the entire string. Return `True`.
2. Recursive Step: Iterate through all possible end indices `end` from `start_index + 1` to `len(s)`.
3. Check if `s[start_index:end]` is in `wordDict`.
4. If it is, recursively check `canBreak(end)`.
5. If both are true, return `True`.
6. Memoize the result for `start_index` to avoid re-computation.

``python
class Solution:
 def wordBreak(self, s: str, wordDict: List[str]) -> bool:
 word_set = set(wordDict) # O(1) lookups
 memo = {}

 def backtrack(start):
 if start == len(s):
 return True
 
 if start in memo:
 return memo[start]
 
 for end in range(start + 1, len(s) + 1):
 word = s[start:end]
 if word in word_set and backtrack(end):
 memo[start] = True
 return True
 
 memo[start] = False
 return False
 
 return backtrack(0)
``

**Complexity:**
- **Time:** O(N^3). There are `N` states. For each state, we iterate `N` times. String slicing takes O(N). Total O(N^3).
- **Space:** O(N) for recursion stack and memoization.

## 4. Approach 2: Tabulation (Bottom-Up DP)

We can solve this iteratively. Let `dp[i]` be `True` if the substring `s[0...i]` can be segmented.

**Definition:**
`dp[i]` = True if `s[0...i]` can be broken into valid words.

**Transition:**
`dp[i]` is True if there exists a `j < i` such that `dp[j]` is True AND `s[j...i]` is in `wordDict`.

**Initialization:**
`dp[0] = True` (Empty string is valid).

``python
class Solution:
 def wordBreak(self, s: str, wordDict: List[str]) -> bool:
 word_set = set(wordDict)
 n = len(s)
 dp = [False] * (n + 1)
 dp[0] = True
 
 for i in range(1, n + 1):
 for j in range(i):
 # If s[0...j] is valid AND s[j...i] is a word
 if dp[j] and s[j:i] in word_set:
 dp[i] = True
 break # Optimization: found one valid split, move to next i
 
 return dp[n]
``

**Complexity:**
- **Time:** O(N^3). Nested loops (`N^2`) + substring slicing/hashing (`N`).
- **Space:** O(N) for the DP array.

## 5. Approach 3: BFS (Breadth-First Search)

We can model this as a graph problem.
- **Nodes:** Indices `0` to `n`.
- **Edges:** Directed edge from `i` to `j` if `s[i:j]` is a valid word.
- **Goal:** Is there a path from `0` to `n`?

``python
from collections import deque

class Solution:
 def wordBreak(self, s: str, wordDict: List[str]) -> bool:
 word_set = set(wordDict)
 queue = deque([0])
 visited = {0}
 n = len(s)
 
 while queue:
 start = queue.popleft()
 if start == n:
 return True
 
 for end in range(start + 1, n + 1):
 if end in visited:
 continue
 
 if s[start:end] in word_set:
 if end == n:
 return True
 queue.append(end)
 visited.add(end)
 
 return False
``

**Complexity:**
- **Time:** O(N^3) in worst case (dense graph).
- **Space:** O(N) for queue and visited set.

## 6. Optimization: Trie (Prefix Tree)

Instead of checking every substring `s[j:i]`, which involves slicing and hashing, we can use a Trie to efficiently traverse potential words.

**Algorithm:**
1. Build a Trie from `wordDict`.
2. Use DP. For each `i` where `dp[i]` is True, traverse the Trie starting from `s[i]`.
3. If we reach a Trie node marked `is_end`, we mark `dp[i + length]` as True.

``python
class TrieNode:
 def __init__(self):
 self.children = {}
 self.is_end = False

class Solution:
 def wordBreak(self, s: str, wordDict: List[str]) -> bool:
 root = TrieNode()
 for word in wordDict:
 node = root
 for char in word:
 if char not in node.children:
 node.children[char] = TrieNode()
 node = node.children[char]
 node.is_end = True
 
 n = len(s)
 dp = [False] * (n + 1)
 dp[0] = True
 
 for i in range(n):
 if not dp[i]:
 continue
 
 # Traverse Trie starting from s[i]
 node = root
 for j in range(i, n):
 char = s[j]
 if char not in node.children:
 break
 node = node.children[char]
 if node.is_end:
 dp[j + 1] = True
 
 return dp[n]
``

**Complexity:**
- **Time:** O(N^2 + M \cdot K), where `M` is number of words, `K` is avg word length (Trie build). The DP part is O(N^2) because we don't do string slicing/hashing.
- **Space:** O(M \cdot K) for Trie + O(N) for DP.
- **Note:** This is significantly faster if the dictionary is large but words are short.

## 7. Deep Dive: Word Break II (Reconstructing Sentences)

**Problem:** Return *all* possible sentences.
**Example:** `s = "catsanddog"`, `dict = ["cat", "cats", "and", "sand", "dog"]`
**Output:** `["cats and dog", "cat sand dog"]`

**Approach:** Backtracking with Memoization.
- Instead of storing `True/False`, store the list of valid sentences for each suffix.

``python
class Solution:
 def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
 word_set = set(wordDict)
 memo = {}
 
 def backtrack(start):
 if start == len(s):
 return [""]
 
 if start in memo:
 return memo[start]
 
 res = []
 for end in range(start + 1, len(s) + 1):
 word = s[start:end]
 if word in word_set:
 suffixes = backtrack(end)
 for suffix in suffixes:
 if suffix:
 res.append(word + " " + suffix)
 else:
 res.append(word)
 
 memo[start] = res
 return res
 
 return backtrack(0)
``

**Complexity:**
- **Time:** O(N \cdot 2^N) in worst case (e.g., `s="aaaaa"`, `dict=["a", "aa", "aaa"]`).
- **Space:** Exponential to store all results.

## 8. Real-World Application: Search Query Segmentation

When you type "newyorktimes" into Google, it understands "new york times". This is Word Break.

**Challenges in Production:**
1. **Unknown Words (OOV):** Names, typos, new slang.
 - *Solution:* Use statistical language models (n-grams) to score segmentations, not just binary dictionary lookup.
2. **Ambiguity:** "expertsexchange" -> "experts exchange" vs "expert sex change".
 - *Solution:* Use frequency counts. `P(\text{"experts"}) \cdot P(\text{"exchange"}) > P(\text{"expert"}) \cdot P(\text{"sex"}) \cdot P(\text{"change"})`.
3. **Latency:** Must be sub-millisecond.
 - *Solution:* Aho-Corasick algorithm or optimized Tries.

## 9. Deep Dive: Aho-Corasick Algorithm

For massive scale dictionary matching, **Aho-Corasick** is the gold standard.
- It builds a Finite Automaton from the dictionary.
- It adds "failure links" to the Trie.
- Allows finding all dictionary word occurrences in `s` in O(N + \text{matches}) time.
- Used in `grep`, intrusion detection systems (Snort), and virus scanners.

## 10. System Design: Spell Checker

**Scenario:** Build a spell checker that suggests corrections and segmentations.

**Components:**
1. **Error Model:** Probability of typing `x` when you meant `y` (Edit Distance).
2. **Language Model:** Probability of word sequence (N-grams or LSTM/Transformer).
3. **Candidate Generator:**
 - Generate candidates within edit distance 1 or 2.
 - Generate segmentation candidates (Word Break).
4. **Ranker:**
 - Score = `\log P(\text{Error}) + \log P(\text{Language})`.
 - Return top-k.

**Optimization:**
- **Bloom Filter:** Quickly check if a word exists in the dictionary before expensive lookup.
- **SymSpell:** Pre-generate all deletions of dictionary words for fast lookup.

## 11. Deep Dive: Maximum Word Break (Max Match)

Sometimes we don't care about *any* valid segmentation, but the one with the **longest words** (Max Match) or **most words**.

**Max Match Algorithm (Greedy):**
- Start at index 0.
- Find the longest word in dictionary starting at 0.
- Move index.
- **Pros:** Very fast O(N).
- **Cons:** Fails on "thetable" -> "theta" "ble" (if "theta" and "ble" are words, but "table" is better).
- **Use Case:** Chinese/Japanese tokenization baselines.

- **Use Case:** Chinese/Japanese tokenization baselines.

## 12. Deep Dive: Trie Implementation Details

While the basic Trie is simple, optimizing it for production requires care.

**1. Array vs Hash Map:**
- **Hash Map (`dict` in Python):** Flexible, handles Unicode. Memory overhead is high.
- **Array (`Node[26]`):** Fast access, low memory overhead. Only works for 'a'-'z'.

**2. Iterative vs Recursive:**
- **Recursive:** Elegant but risks stack overflow for long words.
- **Iterative:** Preferred for production.

**Optimized Trie Code (Array-based):**
``python
class TrieNode:
 __slots__ = 'children', 'is_end'
 def __init__(self):
 self.children = [None] * 26
 self.is_end = False

class Trie:
 def __init__(self):
 self.root = TrieNode()
 
 def insert(self, word):
 node = self.root
 for char in word:
 idx = ord(char) - ord('a')
 if not node.children[idx]:
 node.children[idx] = TrieNode()
 node = node.children[idx]
 node.is_end = True
``

**3. Compressed Trie (Radix Tree):**
- If a node has only one child, merge them.
- `root -> "a" -> "p" -> "p" -> "l" -> "e"` becomes `root -> "apple"`.
- **Benefit:** Reduces depth and memory usage.

## 13. Deep Dive: Aho-Corasick Algorithm in Depth

Aho-Corasick generalizes KMP algorithm to multiple patterns.

**Key Components:**
1. **Trie:** Standard prefix tree of all dictionary words.
2. **Failure Links:** For each node `u` representing string `S`, the failure link points to the longest proper suffix of `S` that is also a prefix of some pattern in the Trie.
3. **Output Links:** Shortcut to the nearest "is_end" node reachable via failure links.

**Construction (BFS):**
1. Root's failure link points to Root.
2. For nodes at depth 1, failure link points to Root.
3. For node `v` (child of `u` via char `c`):
 - Follow `u`'s failure link to `f(u)`.
 - Check if `f(u)` has child via `c`.
 - If yes, `f(v) = f(u).child(c)`.
 - If no, recurse up failure links until Root.

**Usage in Word Break:**
- Instead of restarting search from root for every `i`, we follow failure links.
- This allows us to find *all* matching words ending at `i` in O(1) amortized time per character.
- **Complexity:** O(N + \text{Total Occurrences}).

**Code Sketch:**
``python
def build_failure_links(root):
 queue = deque()
 for char, node in root.children.items():
 node.fail = root
 queue.append(node)
 
 while queue:
 curr = queue.popleft()
 for char, child in curr.children.items():
 # Find failure link for child
 f = curr.fail
 while f != root and char not in f.children:
 f = f.fail
 child.fail = f.children.get(char, root)
 child.output = child if child.is_end else child.fail.output
 queue.append(child)
``

## 14. Deep Dive: Word Break II Optimization

The backtracking solution for Word Break II can be slow.

**Pruning:**
- Before backtracking, run the simple DP (Word Break I) to check if a solution exists.
- If `dp[n]` is False, return empty list immediately.
- This avoids exploring the recursion tree for impossible cases.

**Max Length Optimization:**
- Let `max_len` be the length of the longest word in dictionary.
- When iterating `end` from `start + 1`, stop at `start + max_len + 1`.
- This reduces the inner loop from O(N) to O(K).

``python
class Solution:
 def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
 word_set = set(wordDict)
 max_len = max(len(w) for w in wordDict) if wordDict else 0
 memo = {}
 
 # Pruning check
 if not self.canBreak(s, word_set):
 return []

 def backtrack(start):
 if start == len(s):
 return [""]
 if start in memo: return memo[start]
 
 res = []
 # Optimization: Only check up to max_len
 for end in range(start + 1, min(len(s), start + max_len) + 1):
 word = s[start:end]
 if word in word_set:
 suffixes = backtrack(end)
 for suffix in suffixes:
 if suffix: res.append(word + " " + suffix)
 else: res.append(word)
 memo[start] = res
 return res
 
 return backtrack(0)
``

## 15. System Design: Search Query Segmentation (Viterbi)

**Problem:** User types "newyorktimes". We want "new york times".

**Probabilistic Model:**
- We want to find segmentation `S = w_1, w_2, ..., w_k` that maximizes `P(S | \text{input})`.
- Using Bayes Rule and ignoring denominator: `\text{argmax}_S P(\text{input} | S) P(S)`.
- `P(\text{input} | S) \approx 1` if concatenation of `S` equals input.
- `P(S) \approx \prod P(w_i)` (Unigram model) or `\prod P(w_i | w_{i-1})` (Bigram model).

**Viterbi Algorithm (DP on Log Probabilities):**
- `dp[i]` = Max log-probability of segmenting `s[0...i]`.
- `parent[i]` = The index of the split that gave this max probability.

**Data:**
- Corpus of billions of web pages.
- Count word frequencies.
- `P(w) = \frac{\text{count}(w)}{N}`.

**Smoothing:**
- What if a word is not in our dictionary?
- Assign a small probability `\epsilon` based on character length.
- Or use a character-level language model.

**Code:**
``python
import math

class Segmenter:
 def __init__(self, word_counts, total_count):
 self.word_counts = word_counts
 self.total = total_count
 self.min_prob = math.log(1 / (total_count * 100)) # Smoothing

 def get_prob(self, word):
 return math.log(self.word_counts.get(word, 0) + 1) - math.log(self.total)

 def segment(self, s):
 n = len(s)
 dp = [-float('inf')] * (n + 1)
 parent = [0] * (n + 1)
 dp[0] = 0
 
 for i in range(1, n + 1):
 for j in range(max(0, i - 20), i): # Limit word length
 word = s[j:i]
 prob = dp[j] + self.get_prob(word)
 if prob > dp[i]:
 dp[i] = prob
 parent[i] = j
 
 # Reconstruct
 res = []
 curr = n
 while curr > 0:
 prev = parent[curr]
 res.append(s[prev:curr])
 curr = prev
 return res[::-1]
``

## 16. Advanced: Handling Compound Words (German)

**Problem:** German has infinite compound words ("Donaudampfschifffahrtskapitän").
- Dictionary cannot contain all of them.

**Solution:**
- Recursive decomposition.
- If a word is not in dictionary, try to split it.
- **Morpheme Analysis:** Split into smallest meaningful units.
- **Rule-based:** German has "Fugen-s" (connective 's'). "Liebe" + "Brief" = "Liebesbrief".
- The segmenter must handle these connective characters.

## 17. Case Study: Spell Checker Implementation

**Norvig's Spell Checker:**
1. **Deletions:** "helo" -> "hel", "heo", "hlo", "elo".
2. **Transpositions:** "helo" -> "ehlo", "hleo", "heol".
3. **Replacements:** "helo" -> "aelo", "belo"...
4. **Insertions:** "helo" -> "ahelo", "bhelo"...

**Integration with Word Break:**
- If the input is "thequickbrown", Norvig's approach fails (too many edits).
- We first run **Word Break**.
- If Word Break fails, we try to correct *subsegments*.
- "thequikbrown" -> "the" (valid) + "quik" (invalid) + "brown" (valid).
- Run spell check on "quik" -> "quick".

- **Use Case:** Chinese/Japanese tokenization baselines.

## 18. Deep Dive: Suffix Trees and Suffix Arrays

While Tries are great for dictionary lookups, **Suffix Trees** are the ultimate tool for substring problems.

**Suffix Tree:**
- A compressed Trie of *all suffixes* of a text `T`.
- Can check if `P` is a substring of `T` in O(|P|).
- **Construction:** Ukkonen's Algorithm (O(N)). Complex to implement.

**Suffix Array:**
- An array of integers representing the starting indices of all suffixes of `T`, sorted lexicographically.
- **Example:** `banana`
 - Suffixes: `banana`, `anana`, `nana`, `ana`, `na`, `a`.
 - Sorted: `a` (5), `ana` (3), `anana` (1), `banana` (0), `na` (4), `nana` (2).
 - SA: `[5, 3, 1, 0, 4, 2]`.
- **LCP Array (Longest Common Prefix):** Stores length of LCP between adjacent suffixes in SA.
- **Usage in Word Break:**
 - If we concatenate all dictionary words into a mega-string `D`, we can build a Suffix Array.
 - We can find occurrences of dictionary words in `S` using binary search on the SA.

## 19. Deep Dive: Rabin-Karp Algorithm (Rolling Hash)

For the "substring check" step in the DP (`s[j:i] \in \text{Dict}`), we can use hashing.

**Rolling Hash:**
- Compute hash of window `s[j:i]` in O(1) using the hash of `s[j:i-1]`.
- `H(s[j:i]) = (H(s[j:i-1]) \times B + s[i]) \pmod M`.

**Algorithm:**
1. Compute hashes of all dictionary words and store in a Set (O(L)).
2. For each starting position `j` in `s`:
 - Compute rolling hashes for substrings starting at `j`.
 - If hash matches, do a full string check (to avoid collisions).
 - Update DP.

**Pros:** Faster than slicing if many words have same length.
**Cons:** Hash collisions.

## 20. System Design: Scalable Autocomplete System

**Scenario:** Type "word br..." -> Suggest "word break", "word break ii".

**Requirements:**
- **Latency:** < 50ms (p99).
- **Throughput:** 50k QPS.
- **Freshness:** New trending queries appear within minutes.

**Architecture:**

1. **Data Structure:**
 - **Trie:** Nodes store characters.
 - **Top-K Cache:** Each node stores the top 5 most popular completions ending in that subtree.
 - **Optimization:** Store pointers to DB IDs instead of full strings to save RAM.

2. **Storage:**
 - **Redis:** In-memory Trie for hot prefixes.
 - **Cassandra:** Persistent storage of query logs and frequencies.

3. **Ranking Service:**
 - Score = `w_1 \cdot \text{Frequency} + w_2 \cdot \text{Recency} + w_3 \cdot \text{Personalization}`.
 - Offline job (Spark) updates frequencies hourly.

4. **Handling Typos (Fuzzy Search):**
 - If exact prefix not found, search nodes within Edit Distance 1.
 - Use **Levenshtein Automata**.

## 21. Advanced: Parallel Word Break (MapReduce)

**Problem:** Segment a genome string of length `10^9`.
- DP is O(N^2), too slow.
- Dependencies: `dp[i]` depends on `dp[j]`.

**Parallel Algorithm:**
1. **Split:** Divide string into chunks of size `K` (e.g., 1MB).
2. **Map:** For each chunk, compute a **Transition Matrix** or **Reachability Graph**.
 - Input: Possible start states (offsets into the chunk).
 - Output: Possible end states (offsets out of the chunk).
3. **Reduce:** Multiply matrices (or compose graphs) to find reachability from start of string to end.

**Matrix Multiplication (Tropical Semiring):**
- `(A \otimes B)_{ij} = \max_k (A_{ik} + B_{kj})`.
- Allows combining partial segmentations.

## 22. Deep Dive: Generalized Word Break (2D Grid)

**Problem:** Given a 2D grid of characters (Boggle), find if a word exists.
- This is **DFS/Backtracking**, not standard DP.

**Optimization:**
- Build Trie of dictionary words.
- Start DFS from every cell.
- Pass current Trie node to neighbor.
- **Pruning:** If current path is not a prefix of *any* word (Trie node is None), stop.

**Complexity:** O(R \cdot C \cdot 4^L), where `L` is max word length.

## 23. Interview Questions (Hard)

**Q4: Word Break III - Minimum Cost Segmentation**
*Problem:* Each word has a cost. Insert spaces to minimize total cost.
*Solution:* `dp[i] = min(dp[j] + cost(s[j:i]))`. Use Trie to find valid `s[j:i]`.

**Q5: Palindrome Partitioning**
*Problem:* Split string such that every substring is a palindrome.
*Solution:* Similar to Word Break. Precompute `isPalindrome[j][i]` in O(N^2). Then `dp[i] = min(dp[j] + 1)` if `isPalindrome[j][i]`.

**Q6: Word Break with Wildcards**
*Problem:* `s` contains `?` which can be any char.
*Solution:* Trie traversal matches all children for `?`. DP state remains same.

**Q7: Streaming Word Break**
*Problem:* `s` comes in as a stream. Return True as soon as a valid segmentation is possible for current prefix.
*Solution:* Maintain a set of "active" Trie pointers. For each new char, advance all pointers. If any pointer reaches "is_end", add Root to active set.

**Q8: Longest Word in Dictionary that can be built from other words**
*Problem:* Given list of words, find longest one made of other words in list.
*Solution:* Sort by length. For each word, run `WordBreak(word, dict \ {word})`. First one that returns True is answer.

## 24. Interview Questions

**Q1: How to handle very large dictionaries that don't fit in RAM?**
*Answer:*
- **Disk-based Trie:** Store Trie nodes on disk (B-Tree).
- **Bloom Filter:** Keep a Bloom filter in RAM to rule out non-existent words quickly.
- **Sharding:** Split dictionary by prefix (A-M on Server 1, N-Z on Server 2).

**Q2: What if the dictionary words have costs? Find min cost segmentation.**
*Answer:*
- Modify DP: `dp[i] = min(dp[j] + cost(s[j:i]))` for valid `j`.
- This becomes a Shortest Path problem on the DAG.

**Q3: Word Break with a limit on number of words?**
*Answer:*
- Add a state to DP: `dp[i][k]` = True if `s[0...i]` can be segmented into exactly `k` words.

## 25. Common Mistakes
1. **Greedy Approach:** Trying to match the longest word first. Fails for `s="goals"`, `dict=["go", "goal", "goals", "ls"]`. Greedy takes "goals", leaves "". Correct. Wait, `s="aaaa"`, `dict=["aaaa", "aaa"]`. Greedy takes "aaaa". Correct.
 - Counter-example: `s="abcd"`, `dict=["ab", "abc", "cd", "d"]`. Greedy takes "abc", leaves "d". Valid.
 - Counter-example: `s="abcd"`, `dict=["a", "abc", "b", "cd"]`. Greedy takes "abc", leaves "d" (fail). Correct is "a", "b", "cd".
2. **Infinite Recursion:** Forgetting to memoize. O(2^N) complexity.
3. **Off-by-one Errors:** String slicing `s[start:end]` vs `s[start:end+1]`.

## 26. Performance Benchmarking

**Scenario:** `s` length 1000, Dictionary size 10,000.

| Approach | Time | Space |
| :--- | :--- | :--- |
| **Recursion (No Memo)** | Timeout | O(N) |
| **Recursion + Memo** | 50ms | O(N) |
| **Tabulation** | 45ms | O(N) |
| **Trie Optimization** | 15ms | O(M \cdot K) |

**Takeaway:** Trie optimization is crucial when the dictionary is large and string operations are the bottleneck.

## 27. Ethical Considerations

**1. Content Filtering:**
- Word Break is used to detect "bypassed" profanity (e.g., "assassin" -> "ass assin").
- **Risk:** Scunthorpe problem (blocking valid words).
- **Mitigation:** Context-aware filtering, allow-lists.

**2. Search Segmentation Bias:**
- If "blacklivesmatter" is segmented as "black lives matter" vs "black lives mattering", it affects search results.
- **Impact:** Can suppress or amplify social movements.

## 28. Further Reading

1. **"Speech and Language Processing" (Jurafsky & Martin):** Chapter on N-grams and Tokenization.
2. **"Introduction to Information Retrieval" (Manning):** Tokenization strategies.
3. **"Aho-Corasick Algorithm":** Efficient string matching.

## 29. Conclusion

Word Break is more than just a DP problem; it's the foundation of how computers understand continuous text. Whether it's a search engine parsing your query, a spell checker fixing your typos, or a content filter scanning for banned words, the ability to segment strings efficiently is critical. By mastering the DP approach and optimizing with Tries, you gain the tools to handle text at scale.

## 30. Summary

| Approach | Time | Space | Notes |
| :--- | :--- | :--- | :--- |
| **Recursion + Memo** | O(N^3) | O(N) | Easy to implement |
| **Tabulation** | O(N^3) | O(N) | Iterative, avoids stack overflow |
| **BFS** | O(N^3) | O(N) | Graph perspective |
| **Trie** | O(N^2) | O(MK) | Fastest for large dicts |

---

**Originally published at:** [arunbaby.com/dsa/0039-word-break](https://www.arunbaby.com/dsa/0039-word-break/)
