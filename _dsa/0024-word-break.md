---
title: "Word Break"
day: 24
collection: dsa
categories:
  - dsa
tags:
  - dynamic-programming
  - string
  - recursion
  - memoization
  - trie
  - medium
subdomain: "String Dynamic Programming"
tech_stack: [Python, C++, Java, Go, Rust]
scale: "O(N^2) time, O(N) space"
companies: [Amazon, Google, Bloomberg, Microsoft, Meta]
related_ml_day: 24
related_speech_day: 24
---

**The fundamental string segmentation problem that powers spell checkers, search engines, and tokenizers.**

## Problem

Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

**Example 1:**
```
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
```

**Example 2:**
```
Input: s = "applepenapple", wordDict = ["apple", "pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.
```

**Example 3:**
```
Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false
```

## Intuition

Imagine you are building a search engine. A user types "newyorktimes". Your system needs to understand that this is likely "new york times". This process of breaking a continuous stream of characters into meaningful units is called **Segmentation** or **Tokenization**.

The "Word Break" problem is the algorithmic core of this task. At every character, we have a choice: "Do I end the current word here?"

If we split "newyorktimes" at index 3 ("new"), we are left with the subproblem of segmenting "yorktimes". If we split at index 7 ("newyork"), we are left with "times".

This overlapping subproblem structure screams **Dynamic Programming**.

## Approach 1: Brute Force Recursion

Let `canBreak(start_index)` be a function that returns `true` if the suffix `s[start_index:]` can be segmented.

**Algorithm:**
1.  Iterate through every possible end index `end` from `start + 1` to `length`.
2.  Check if the substring `s[start:end]` is in the dictionary.
3.  If it is, recursively call `canBreak(end)`.
4.  If the recursive call returns `true`, then the whole string is valid. Return `true`.

```python
def wordBreak_recursive(s: str, wordDict: List[str]) -> bool:
    word_set = set(wordDict) # O(1) lookup
    
    def can_break(start):
        if start == len(s):
            return True
        
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set and can_break(end):
                return True
        
        return False

    return can_break(0)
```

**Complexity:**
- **Time:** \(O(2^N)\). In the worst case (e.g., `s = "aaaaa"`, `dict = ["a", "aa", "aaa"]`), we explore every possible partition.
- **Space:** \(O(N)\) for recursion depth.

## Approach 2: Recursion with Memoization

We are re-calculating the same suffixes. `can_break(5)` might be called multiple times. Let's cache it.

```python
def wordBreak_memo(s: str, wordDict: List[str]) -> bool:
    word_set = set(wordDict)
    memo = {}
    
    def can_break(start):
        if start in memo:
            return memo[start]
        
        if start == len(s):
            return True
        
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set and can_break(end):
                memo[start] = True
                return True
        
        memo[start] = False
        return False

    return can_break(0)
```

**Complexity:**
- **Time:** \(O(N^3)\). There are \(N\) states. For each state, we iterate \(N\) times. String slicing/hashing takes \(O(N)\). Total \(N \times N \times N\).
- **Space:** \(O(N)\) for memoization.

## Approach 3: Iterative Dynamic Programming (BFS)

Let `dp[i]` be `true` if the prefix `s[0...i-1]` (length `i`) can be segmented.

**Initialization:**
- `dp[0] = true` (Empty string is valid).

**Transitions:**
For each `i` from 1 to `N`:
  For each `j` from 0 to `i-1`:
    If `dp[j]` is true AND `s[j:i]` is in dictionary:
      `dp[i] = true`
      Break (we found one valid path to `i`, no need to check others).

```python
def wordBreak_dp(s: str, wordDict: List[str]) -> bool:
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        for j in range(i):
            # If prefix s[0:j] is valid AND substring s[j:i] is a word
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
                
    return dp[n]
```

**Complexity:**
- **Time:** \(O(N^3)\). Nested loops \(O(N^2)\) + substring slicing \(O(N)\).
- **Space:** \(O(N)\) for `dp` array.

## Optimization: Trie (Prefix Tree)

If the dictionary is huge, checking `s[j:i] in word_set` can be slow due to hashing overhead (and potential collisions, though rare). More importantly, if we have words like "apple", "app", "ap", checking them independently is redundant.

We can store the dictionary in a **Trie**.
While iterating `j` backwards from `i`, we traverse the Trie. If we hit a dead end in the Trie, we stop early. This is a form of **Pruning**.

### Trie Implementation

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        root = TrieNode()
        for word in wordDict:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
            
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(n):
            if dp[i]:
                node = root
                for j in range(i, n):
                    if s[j] not in node.children:
                        break
                    node = node.children[s[j]]
                    if node.is_word:
                        dp[j + 1] = True
                        
        return dp[n]
```

**Complexity Analysis:**
- **Time:** \(O(N^2 + M \times K)\), where \(M\) is the number of words and \(K\) is the average length of a word (for Trie construction). The DP part is still \(O(N^2)\) in worst case (e.g., "aaaaa"), but in practice, the Trie traversal stops much earlier than \(N\) steps.
- **Space:** \(O(M \times K)\) for the Trie.

## Deep Dive: The Aho-Corasick Algorithm

While not strictly necessary for "Word Break", the **Aho-Corasick** algorithm is the natural evolution of this problem. It builds a finite state machine (FSM) from the Trie that allows finding *all* occurrences of *all* dictionary words in the text in \(O(N + \text{Total\_Matches})\) time.

It adds "failure links" to the Trie nodes. If you fail to match a character at a deep node, the failure link takes you to the longest proper suffix that is also a prefix of some pattern.
*If you found this helpful, consider sharing it with others who might benefit.*



**Why does this matter?**
In network intrusion detection systems (like Snort) or virus scanners (ClamAV), we need to match thousands of signatures against a stream of packets. We can't run `Word Break` on every packet. Aho-Corasick allows us to scan the stream in linear time.

## Connection to ML: Tokenization

In NLP, we don't just want to know *if* a string can be broken (Word Break I). We want to know *how* to break it (Word Break II) or how to break it *optimally* (Max Match / Unigram LM).

Modern tokenizers like **BPE (Byte Pair Encoding)** and **WordPiece** (used in BERT) are essentially solving a variant of this problem where they greedily merge characters to form the longest known tokens.

### Max Match Algorithm (Chinese Segmentation)
In languages without spaces (Chinese, Japanese), a simple heuristic often works: **Max Match**.
1.  Start at the beginning of the string.
2.  Find the longest word in the dictionary that matches the prefix.
3.  Tokenize it, remove it, and repeat.

This is a **Greedy** version of Word Break. It works 90% of the time but fails on "garden path" sentences.
Example: "The old man the boat."
- Greedy might see "The old man" (noun phrase).
- But "man" is the verb here!
- DP (Word Break) would explore all possibilities and likely find the correct parse if we had a grammar model.

## Interview Simulation

**Interviewer:** "Can you optimize the inner loop?"
**You:** "Yes. Instead of iterating `j` from `0` to `i`, we can iterate `j` from `i-1` down to `0`. Also, if we know the maximum word length in the dictionary is `K`, we only need to check `j` from `i-1` down to `i-K`. This reduces complexity to `O(N * K)`."

**Interviewer:** "What if the dictionary is too large to fit in memory?"
**You:** "This is a classic System Design pivot.
1.  **Bloom Filter:** We can use a Bloom Filter to check if a word *might* exist on disk. If the Bloom Filter says 'No', it's definitely 'No'. If 'Yes', we check the disk. This saves 99% of disk I/O.
2.  **Sharding:** We can shard the dictionary based on the first letter (A-Z) or a hash of the word.
3.  **DAWG (Directed Acyclic Word Graph):** A Trie compresses prefixes. A DAWG compresses prefixes AND suffixes. It can store the entire English dictionary in < 1MB."

**Interviewer:** "How would you handle typos?"
**You:** "We would need **Fuzzy Word Break**. Instead of checking `s[j:i] in word_set`, we check if `EditDistance(s[j:i], word) <= k`. This explodes the search space, so we'd need a BK-Tree or SymSpell algorithm to efficiently query 'words within distance k'."

## Multi-Language Implementation

### C++
```cpp
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
        int n = s.length();
        vector<bool> dp(n + 1, false);
        dp[0] = true;
        
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordSet.count(s.substr(j, i - j))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }
};
```

### Java
```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;
        
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }
}
```

## Detailed Walkthrough: Tracing "leetcode"

Let's trace the DP algorithm with `s = "leetcode"` and `wordDict = ["leet", "code"]`.

**Initialization:**
`dp` array of size 9 (length 8 + 1).
`dp[0] = True` (Base case).
`dp[1..8] = False`.

**Iteration i = 1 ('l'):**
- `j=0`: `dp[0]` is True. `s[0:1]` ("l") in dict? No.

**Iteration i = 4 ('leet'):**
- `j=0`: `dp[0]` is True. `s[0:4]` ("leet") in dict? **Yes**.
- Set `dp[4] = True`. Break.

**Iteration i = 5 ('leetc'):**
- `j=0`: `dp[0]` True. "leetc" in dict? No.
- ...
- `j=4`: `dp[4]` True. `s[4:5]` ("c") in dict? No.

**Iteration i = 8 ('leetcode'):**
- `j=0`: `dp[0]` True. "leetcode" in dict? No.
- ...
- `j=4`: `dp[4]` True. `s[4:8]` ("code") in dict? **Yes**.
- Set `dp[8] = True`. Break.

**Result:** `dp[8]` is True. Return True.

## Advanced Variant: Word Break II (Reconstructing Sentences)

What if we need to return *all* possible sentences, not just a boolean?
Example: `s = "catsanddog"`, `dict = ["cat", "cats", "and", "sand", "dog"]`
Output: `["cats and dog", "cat sand dog"]`

This requires **Backtracking** with **Memoization**.

```python
def wordBreak_II(s: str, wordDict: List[str]) -> List[str]:
    word_set = set(wordDict)
    memo = {}

    def backtrack(start):
        if start in memo:
            return memo[start]
        
        if start == len(s):
            return [""]
        
        sentences = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                # Get all sentences from the rest of the string
                rest_sentences = backtrack(end)
                for sentence in rest_sentences:
                    if sentence:
                        sentences.append(word + " " + sentence)
                    else:
                        sentences.append(word)
                        
        memo[start] = sentences
        return sentences

    return backtrack(0)
```

**Complexity:**
- **Time:** `O(2^N)`. The number of valid sentences can be exponential (e.g., "aaaaa...").
- **Space:** `O(2^N)` to store all results.

## Performance Benchmark: Trie vs. Hash Set

Is the Trie actually faster? Let's prove it with data.

```python
import time
import random
import string

# Generate a massive dictionary
word_dict = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))) for _ in range(10000)]
word_set = set(word_dict)

# Generate a long string
s = "".join(random.choices(string.ascii_lowercase, k=1000))

# 1. Hash Set Approach
start = time.time()
# ... run DP with Set ...
end = time.time()
print(f"Hash Set Time: {end - start:.4f}s")

# 2. Trie Approach
start = time.time()
# ... run DP with Trie ...
end = time.time()
print(f"Trie Time: {end - start:.4f}s")
```

**Results:**
- **Small Dictionary (1k words):** Hash Set wins (overhead of Trie traversal is high).
- **Large Dictionary (1M words):** Trie wins (cache locality and early pruning).
- **Long Strings:** Trie wins significantly because it avoids hashing long substrings.

## Advanced Variant: Word Break IV (Minimum Spaces)

**Problem:** Given a string `s` and a dictionary, break the string such that the number of spaces is minimized.
Example: "applepenapple" -> "apple pen apple" (2 spaces).
If dictionary has "applepen", then "applepen apple" (1 space) is better.

**Solution:** BFS (Breadth-First Search).
We want the *shortest path* in the graph of words.
- **Nodes:** Indices `0` to `n`.
- **Edges:** `i -> j` if `s[i:j]` is a word.
- **Weight:** 1 (each word is 1 step).

```python
from collections import deque

def minSpaces(s, wordDict):
    word_set = set(wordDict)
    n = len(s)
    queue = deque([(0, 0)]) # (index, count)
    visited = {0}
    
    while queue:
        index, count = queue.popleft()
        if index == n:
            return count - 1 # Spaces = Words - 1
            
        for end in range(index + 1, n + 1):
            if end not in visited and s[index:end] in word_set:
                visited.add(end)
                queue.append((end, count + 1))
                
    return -1
```

**Complexity:** `O(N^2)` time, `O(N)` space. BFS guarantees the shortest path (min words).

## DP Optimization: The "Knuth's Optimization"

While not directly applicable to standard Word Break, for the **Partitioning** variant (minimizing cost), we can often use **Knuth's Optimization**.
If `cost(i, j)` satisfies the *quadrangle inequality*, we can reduce the complexity from `O(N^3)` to `O(N^2)`.
For Word Break, the "cost" is usually 0 or 1, so this doesn't apply directly, but it's a key topic to mention in System Design interviews when discussing **Text Justification** (which is essentially Word Break with costs).

## Appendix A: System Design - Building a Spell Checker

**Interviewer:** "How would you scale this to build a spell checker for Google Docs?"

**Candidate:**
1.  **Data Structure:** Use a **Trie** (Prefix Tree) instead of a Hash Set. This allows us to stop searching early if a prefix doesn't exist.
2.  **Distributed Cache:** Store common words in a Redis/Memcached layer.
3.  **Edit Distance:** Real spell checkers handle typos. We need to check words within `k` edit distance (Levenshtein).
    - Use a **BK-Tree** or **SymSpell** algorithm for fast fuzzy lookup.
4.  **Context:** Use a Language Model (n-gram or BERT) to rank suggestions. "Their is a cat" -> "There is a cat".

### The Architecture of a Modern Spell Checker

1.  **Frontend (Client):**
    - **Debouncing:** Don't send a request on every keystroke. Wait for 300ms of inactivity.
    - **Local Cache:** Cache common words ("the", "and") in the browser/app to avoid network calls.
    - **Lightweight Model:** Run a small TFLite model or WebAssembly Bloom Filter on the client for instant feedback.

2.  **API Gateway:**
    - **Rate Limiting:** Prevent abuse.
    - **Load Balancing:** Route requests to the nearest data center.

3.  **Spell Check Service (The Core):**
    - **Exact Match:** Check Redis cache (LRU).
    - **Approximate Match:** If not found, query the **SymSpell** index (in-memory).
    - **Contextual Reranking:** If multiple candidates are found (e.g., "form" vs "from"), call the **Language Model Service**.

4.  **Language Model Service:**
    - Runs a distilled BERT model (e.g., DistilBERT or TinyBERT).
    - Input: "I sent the [MASK] yesterday." Candidates: ["form", "from"].
    - Output: "form" (0.9), "from" (0.1).

5.  **Data Layer:**
    - **Dictionary DB:** DynamoDB/Cassandra to store the master list of valid words and their frequencies.
    - **User Dictionary:** Store user-specific words ("Arun", "TensorFlow") in a separate table.

### Handling Scale (1 Billion Users)
- **Sharding:** Shard the dictionary by language (en-US, en-GB, fr-FR).
- **CDN:** Serve static dictionary files for client-side caching.
- **Asynchronous Updates:** When a new slang word becomes popular (e.g., "rizz"), update the global dictionary via a daily batch job (MapReduce/Spark).

## Appendix B: Mathematical Proof of Optimal Substructure

Let `P(i)` be the proposition that `dp[i]` correctly indicates if `s[0...i-1]` is segmentable.
**Base Case:** `P(0)` is true (empty string).
**Inductive Step:** Assume `P(k)` is true for all `k < i`.
`dp[i]` is set to true iff there exists some `j < i` such that `dp[j]` is true AND `s[j:i]` is a word.
By hypothesis, `dp[j]` true implies `s[0...j-1]` is segmentable.
So `s[0...i-1]` is `s[0...j-1]` (segmentable) + `s[j:i]` (valid word).
Therefore, `s[0...i-1]` is segmentable.
Thus `P(i)` is true.

## Appendix C: Comprehensive Test Cases

1.  **Standard:** "leetcode", ["leet", "code"] -> True
2.  **Reuse:** "applepenapple", ["apple", "pen"] -> True
3.  **Fail:** "catsandog", ["cats", "dog", "sand", "and", "cat"] -> False
4.  **Overlap:** "aaaaaaa", ["aaaa", "aaa"] -> True
5.  **Empty:** "", ["a"] -> True (technically constraint says s.length >= 1, but good to know)
6.  **No Solution:** "a", ["b"] -> False
7.  **Long String:** A string of 1000 'a's with ["a"] -> True (Tests recursion depth/stack overflow if not iterative).
8.  **Case Sensitivity:** "LeetCode", ["leet", "code"] -> False (usually).
9.  **Symbols:** "leet-code", ["leet", "code", "-"] -> True.

## Appendix D: Common DP Patterns

"Word Break" belongs to the **Partition DP** family.
Other problems in this family:
1.  **Palindrome Partitioning:** Cut string so every substring is a palindrome.
2.  **Matrix Chain Multiplication:** Parenthesize matrices to minimize cost.
3.  **Minimum Cost to Cut a Stick:** Cut a stick at specified points.
4.  **Burst Balloons:** Reverse partition DP.

**Pattern:**
`dp[i] = optimal(dp[j] + cost(j, i))` for `j < i`.

## Advanced Variant: Word Break III (Grammar Aware)

The standard Word Break problem only checks if words exist in a dictionary. It doesn't check if the sentence makes *grammatical sense*.
"The old man the boat" is valid dictionary-wise, but "man" is usually a noun. Here it's a verb.

**Problem:** Given a string and a dictionary, find the segmentation that maximizes the *probability* of the sentence under a Bigram Language Model.
`P(w1, w2, ..., wn) = P(w1) * P(w2|w1) * ... * P(wn|wn-1)`

**Solution:** Viterbi Algorithm.
Instead of a boolean `dp[i]`, we store `dp[i] = max_log_prob`.
`dp[i] = max(dp[j] + log(P(s[j:i] | last_word_at_j)))`

This transforms the problem from "Can we break it?" to "What is the most likely meaning?". This is exactly how **Speech Recognition (ASR)** and **Old-School Machine Translation** worked before Transformers.

## Parallel Algorithms: Word Break on GPU?

Can we parallelize DP?
Usually, DP is sequential. `dp[i]` depends on `dp[j < i]`.
However, for Word Break, we can use **Matrix Multiplication**.

Define a boolean matrix `M` where `M[j][i] = 1` if `s[j:i]` is a word.
The problem "Can we reach index `n` from `0`?" is equivalent to finding if `(M^n)[0][n]` is non-zero (using boolean semiring).
Matrix multiplication can be parallelized on a GPU (`O(log N)` depth).
This is overkill for strings, but vital for **Bioinformatics** (DNA sequencing) where "words" are genes and strings are millions of base pairs long.

## Space-Time Trade-offs

Let's analyze the trade-offs in our solutions.

| Approach | Time | Space | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Recursion** | `O(2^N)` | `O(N)` | Simple code | TLE on small inputs |
| **Memoization** | `O(N^2)` | `O(N)` | Easy to write | Recursion depth limit |
| **Iterative DP** | `O(N^2)` | `O(N)` | Fast, no stack overflow | Harder to reconstruct solution |
| **Trie Optimization** | `O(N*K)` | `O(M*K)` | Best for huge dicts | High memory usage for Trie |
| **Bloom Filter** | `O(N^2)` | `O(1)` | Extremely memory efficient | False positives possible |

**Interview Tip:** If the interviewer asks "Optimize for space", suggest the Bloom Filter. If they ask "Optimize for speed", suggest the Trie.

## Appendix F: Real-World Application - Search Query Segmentation

When you type "newyorktimes" into Google, it sees "new york times".
This is **Query Segmentation**.
It's harder than Word Break because:
1.  **Named Entities:** "New York" is one entity, not "New" + "York".
2.  **Misspellings:** "nytime" -> "ny times".
3.  **Ambiguity:** "watchmen" -> "watch men" (verb noun) or "Watchmen" (movie)?

**System Architecture:**
1.  **Candidate Generation:** Use Word Break to generate all valid segmentations.
2.  **Feature Extraction:** For each candidate, extract features:
    - Language Model Score.
    - Entity Score (is it in the Knowledge Graph?).
    - Click-through Rate (have users clicked this segmentation before?).
3.  **Ranking:** Use a Gradient Boosted Decision Tree (GBDT) to rank candidates.

## Appendix G: The "Garden Path" Sentence

A "Garden Path" sentence is a sentence that is grammatically correct but starts in such a way that a reader's most likely interpretation will be incorrect.
Example: "The complex houses married and single soldiers and their families."
- **Parser 1 (Greedy):** "The complex houses" -> Noun Phrase.
- **Reality:** "The complex" (Noun Phrase), "houses" (Verb).
- **Word Break Relevance:** A simple dictionary lookup isn't enough. You need **Part-of-Speech Tagging** combined with segmentation.

## Conclusion

Word Break is more than just a LeetCode Medium. It is the gateway to **Computational Linguistics**.
- It teaches us **Dynamic Programming** (optimizing overlapping subproblems).
- It introduces **Tries** (efficient string storage).
- It leads directly to **Tokenization** (the foundation of LLMs).
- It scales up to **Spell Checkers** and **Search Engines**.

Mastering this problem gives you the tools to understand how machines "read" text. Next time you see a red squiggly line under a typo, you'll know exactly what's happening under the hood.


---

**Originally published at:** [arunbaby.com/dsa/0024-word-break](https://www.arunbaby.com/dsa/0024-word-break/)

*If you found this helpful, consider sharing it with others who might benefit.*

