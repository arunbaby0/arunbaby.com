---
title: "Regular Expression Matching"
day: 58
collection: dsa
categories:
  - dsa
tags:
  - dynamic-programming
  - recursion
  - string-matching
  - hard
  - state-machines
difficulty: Hard
subdomain: "Dynamic Programming"
tech_stack: Python
scale: "Handling strings and patterns up to 1000 characters with O(N*M) time"
companies: [Google, Meta, Amazon, Microsoft, Uber, Airbnb]
related_ml_day: 58
related_speech_day: 58
related_agents_day: 58
---

**"Regular Expression Matching is where string manipulation meets automata theory. It requires translating a sequence of patterns into a resilient state machine that can handle non-deterministic choices."**

## 1. Introduction: The Power of Pattern

From the `grep` command in your terminal to the tokenizers in a Large Language Model, **Regular Expressions (RegEx)** are the backbone of text processing. While most engineers use them every day, few understand what's happening under the hood.

At its core, RegEx matching is a **Search Problem**. Unlike simple string search (where we look for an exact substring), RegEx introduced **Wildcards** (`.`) and **Quantifiers** (`*`). This makes the search space branch: when you see `a*`, you could choose to match zero 'a's, one 'a', or ten 'a's. 

Today, we solve the "True" RegEx matching problem using **Dynamic Programming**, exploring how to manage overlapping subproblems and complex state transitions.

---

## 2. The Problem Statement

Implement regular expression matching with support for `.` and `*`.
- `.` Matches any single character.
- `*` Matches zero or more of the preceding element.

The matching should cover the **entire** input string (not just a partial match).

**Example 1:**
- Input: `s = "aa", p = "a"`
- Output: `false`

**Example 2:**
- Input: `s = "aa", p = "a*"`
- Output: `true` (Matches zero or more 'a's)

**Example 3:**
- Input: `s = "ab", p = ".*"`
- Output: `true` (Matches any sequence)

---

## 3. Thematic Link: Advanced DP and State Machines

Today, on Day 58, we explore **State-Driven Intelligence**:
- **DSA**: We use DP to track the valid states of a matching engine.
- **ML System Design**: Advanced NLP pipelines use finite state machines for tokenization and entity recognition.
- **Speech Tech**: Conversational AI systems transition through "Dialog States" based on user intent.
- **Agents**: Ethical AI agents use rule-based state machines to enforce safety guardrails.

---

## 4. Approach 1: Top-Down Recursion with Memoization

The most intuitive way to think about `*` is as a branching decision.

### 4.1 The Recurrence Relation
Let `dp(i, j)` be true if `s[i:]` matches `p[j:]`.
1.  **Check the first character**: `first_match = (i < len(s) and p[j] in {s[i], '.'})`.
2.  **Handle the `*` (Lookahead)**: If `p[j+1] == '*'`:
    - Option A: **Ignore the `*`** (match zero): `dp(i, j + 2)`.
    - Option B: **Use the `*`** (if first match is true!): `first_match and dp(i + 1, j)`.
3.  **No `*`**: Just move both pointers forward: `first_match and dp(i + 1, j + 1)`.

### 4.2 Implementation

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        memo = {}

        def dp(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            # Base Case: If pattern is exhausted, string must be exhausted too
            if j == len(p):
                return i == len(s)

            first_match = i < len(s) and p[j] in {s[i], "."}

            if j + 1 < len(p) and p[j+1] == "*":
                # Decision: Match 0 or Match 1+
                res = dp(i, j + 2) or (first_match and dp(i + 1, j))
            else:
                res = first_match and dp(i + 1, j + 1)

            memo[(i, j)] = res
            return res

        return dp(0, 0)
```

---

## 5. Approach 2: Bottom-Up Dynamic Programming (State Machine)

The recursive approach is expressive, but it has overhead. A 2D table approach is often preferred in production for its $O(1)$ memory access speed.

### 5.1 The Table Design
`T[i][j]` represents if `s[:i]` matches `p[:j]`.

| | `""` | `a` | `*` | `b` |
| :--- | :--- | :--- | :--- | :--- |
| **`""`** | T | F | T | F |
| **`a`** | F | T | T | F |
| **`aa`**| F | F | T | F |

### 5.2 Implementation details

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        n, m = len(s), len(p)
        dp = [[False] * (m + 1) for _ in range(n + 1)]
        
        # Base Case: Empty string matches empty pattern
        dp[0][0] = True
        
        # Fill for empty string but non-empty pattern (e.g., "a*b*")
        for j in range(2, m + 1):
            if p[j-1] == "*":
                dp[0][j] = dp[0][j-2]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if p[j-1] == s[i-1] or p[j-1] == ".":
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == "*":
                    # Option 1: Match 0 preceding elements (skip p[j-2]*)
                    match_zero = dp[i][j-2]
                    # Option 2: Match 1+ preceding elements
                    # (if current char in s matches the char before *)
                    preceding_char = p[j-2]
                    match_one_plus = (s[i-1] == preceding_char or preceding_char == ".") and dp[i-1][j]
                    
                    dp[i][j] = match_zero or match_one_plus
        
        return dp[n][m]
```

---

## 6. Implementation Deep Dive: Handling the Quantifier

The logic `dp[i][j] = dp[i][j-2] or (first_match and dp[i-1][j])` is the heart of the algorithm.
- `dp[i][j-2]` is like saying: "I don't need this `*` construct. Pretend it's not here." 
- `dp[i-1][j]` is like saying: "I just matched a character using this `*`. I am now in the same pattern state but one character further in the string."

This is equivalent to a **Non-deterministic Finite Automaton (NFA)**. When the machine sees a `*`, it splits into two parallel universes. If *any* universe reaches the final state, the string matches.

---

## 7. Comparative Performance Analysis

| Metric | Recursion (No Memo) | DP (Memo/Table) | NFA (Automata) |
| :--- | :--- | :--- | :--- |
| **Time** | $O(2^{N+M})$ | $O(N \cdot M)$ | $O(N \cdot M)$ |
| **Space** | $O(N+M)$ | $O(N \cdot M)$ | $O(M)$ |

**Note on Memory**: We can optimize the DP table to $O(M)$ space because `dp[i]` only depends on `dp[i-1]`.

---

## 8. Real-world Applications: NLP and Parsing

In our **ML System Design** post today, we discussed **Advanced NLP Pipelines**.
- **RegEx in Tokenization**: Before a Transformer sees text, it is split into tokens. RegEx is used to handle punctuations, URLs, and numeric formats.
- **Rule-based Entity Extraction**: In high-precision domains (Legal, Medical), we often use RegEx-based state machines to extract dates, IDs, and symptoms because they are more reliable than neural networks for specific patterns.

---

## 9. Interview Strategy: The "Empty Input" Case

1.  **Test the Extremes**: Ask "What if `s` is empty and `p` is `a*`?" (Result: True).
2.  **Explain the Table**: Before coding the 2D DP, explain what `dp[0][j]` means. It handles the "optional" patterns at the start.
3.  **Trace a '.*'**: Show how `.*` can swallow any character by self-looping in the DP table.
4.  **Complexity Discussion**: Mention the space optimization to $O(M)$ if they ask for more efficiency.

---

## 10. Common Pitfalls

1.  **Off-by-one errors**: The DP table is `(n+1) x (m+1)`. Be careful with `index` vs `length`.
2.  **The j-2 risk**: When checking `p[j-1] == "*"`, ensure `j >= 2`. The problem usually guarantees `p` is valid (no `*` at the start).
3.  **Forgetting 'match zero'**: Many developers forget that `a*` can match nothing.

---

## 11. Key Takeaways

1.  **DP is about Decision Trees**: Every `*` is a branch. DP collapses the branches.
2.  **State Machines are strings**: A RegEx is a compressed instructions for a machine.
3.  **Correctness > Speed**: For RegEx, the edge cases (empty strings, trailing wildcards) are where most bugs live.

---

**Originally published at:** [arunbaby.com/dsa/0058-regular-expression-matching](https://www.arunbaby.com/dsa/0058-regular-expression-matching/)

*If you found this helpful, consider sharing it with others who might benefit.*
