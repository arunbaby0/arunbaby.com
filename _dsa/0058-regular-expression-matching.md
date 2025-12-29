---
title: "Regular Expression Matching"
day: 58
related_ml_day: 58
related_speech_day: 58
related_agents_day: 58
collection: dsa
categories:
  - dsa
tags:
  - dynamic-programming
  - string-matching
  - hard
  - state-machines
difficulty: Hard
subdomain: "Dynamic Programming"
tech_stack: Python
scale: "Handling strings and patterns up to 1000 characters with O(N*M) time"
companies: [Google, Meta, Amazon, Microsoft, Uber, Airbnb]
---

**"Regular Expression Matching is where string manipulation meets automata theory. It requires translating a sequence of patterns into a resilient state machine."**

## 1. Introduction: The Power of Pattern

From terminal commands to tokenizers in Large Language Models, **Regular Expressions (RegEx)** are the backbone of text processing. While most engineers use them every day, few understand what's happening under the hood.

At its core, RegEx matching is a **Search Problem**. Unlike simple string search, RegEx introduced **Wildcards** (`.`) and **Quantifiers** (`*`). This makes the search space branch: when you see `a*`, you could choose to match zero 'a's, one 'a', or many. 

We solve the RegEx matching problem using **Dynamic Programming**, exploring how to manage overlapping subproblems and complex state transitions.

---

## 2. The Problem Statement

Implement regular expression matching with support for `.` and `*`.
- `.` Matches any single character.
- `*` Matches zero or more of the preceding element.

The matching should cover the **entire** input string.

---

## 3. Thematic Link: Advanced DP and State Machines

We explore **State-Driven Intelligence**:
- **DSA**: We use DP to track the valid states of a matching engine.
- **ML System Design**: Advanced NLP pipelines use finite state machines for tokenization and entity recognition.
- **Speech Tech**: Conversational AI systems transition through dialog states based on user intent.
- **Agents**: Ethical AI agents use rule-based state machines to enforce safety guardrails.

---

## 4. Approach 1: Top-Down Recursion with Memoization

The most intuitive way to think about `*` is as a branching decision.

### 4.1 The Recurrence Relation
Let `dp(i, j)` be true if `s[i:]` matches `p[j:]`.
1. **Check the first character**: `first_match = (i < len(s) and p[j] in {s[i], '.'})`.
2. **Handle the * lookahead**: If `p[j+1] == '*'`:
  - Option A: Ignore the `*` construct (match zero): `dp(i, j + 2)`.
  - Option B: Use the `*` construct (if first match is true): `first_match and dp(i + 1, j)`.
3. **No * quantifier**: Just move both pointers forward: `first_match and dp(i + 1, j + 1)`.

### 4.2 Implementation

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        memo = {}

        def dp(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if j == len(p):
                return i == len(s)

            first_match = i < len(s) and p[j] in {s[i], "."}

            if j + 1 < len(p) and p[j+1] == "*":
                res = dp(i, j + 2) or (first_match and dp(i + 1, j))
            else:
                res = first_match and dp(i + 1, j + 1)

            memo[(i, j)] = res
            return res

        return dp(0, 0)
```

---

## 5. Approach 2: Bottom-Up Dynamic Programming

A 2D table approach is often preferred for its iterative efficiency and O(1) memory access speed.

### 5.1 The Table Design
`T[i][j]` represents if `s[:i]` matches `p[:j]`.

### 5.2 Implementation

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        n, m = len(s), len(p)
        dp = [[False] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = True
        
        for j in range(2, m + 1):
            if p[j-1] == "*":
                dp[0][j] = dp[0][j-2]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if p[j-1] == s[i-1] or p[j-1] == ".":
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == "*":
                    match_zero = dp[i][j-2]
                    preceding_char = p[j-2]
                    match_one_plus = (s[i-1] == preceding_char or preceding_char == ".") and dp[i-1][j]
                    dp[i][j] = match_zero or match_one_plus
        
        return dp[n][m]
```

---

## 6. Implementation Deep Dive: Handling the Quantifier

The logic `dp[i][j] = dp[i][j-2] or (first_match and dp[i-1][j])` is the heart of the algorithm.
- `dp[i][j-2]` means skipping the `*` and its preceding character entirely.
- `dp[i-1][j]` means we used the `*` to match a character, and we're looking to match the rest of the string with the same pattern.

This is equivalent to a **Non-deterministic Finite Automaton (NFA)**. When the machine sees a `*`, it effectively splits. If any path reaches the final state, the string matches.

---

## 7. Comparative Performance Analysis

| Metric | Recursion (No Memo) | DP (Memo/Table) | NFA (Automata) |
| :--- | :--- | :--- | :--- |
| **Time** | O(2^{N+M}) | O(NM) | O(NM) |
| **Space** | O(N+M) | O(NM) | O(M) |

---

## 8. Real-world Applications: NLP and Parsing

RegEx is used extensively in **NLP Pipelines**:
- **Tokenization**: Handling punctuations, URLs, and numeric formats.
- **Rule-based Entity Extraction**: In high-precision domains like legal or medical tech, RegEx state machines extract specific dates and IDs more reliably than raw neural nets.

---

## 9. Interview Strategy: The "Empty Input" Case

1. **Test the Extremes**: Ask "What if `s` is empty and `p` is `a*`?" (The result should be true).
2. **Explain the Table**: Define what `dp[0][j]` means—it handles "optional" patterns at the start.
3. **Trace a wildcard**: Show how `.*` can swallow any character in the DP table.

---

## 10. Common Pitfalls

1. **Off-by-one errors**: The DP table is `(n+1) x (m+1)`. Be careful with 1-based indexing for the table vs 0-based for the string.
2. **The j-2 risk**: When checking `p[j-1] == "*"`, ensure `j >= 2`.
3. **Forgetting 'match zero'**: Always remember that `a*` can match an empty string.

---

## 11. Key Takeaways

1. **DP is about Decision Trees**: Every `*` is a branch; DP collapses the branches.
2. **State Machines are patterns**: A RegEx is a set of compressed instructions for a matching machine.
3. **Correctness > Speed**: Edge cases like empty strings and wildcards are where most bugs live in string matching.

---

**Originally published at:** [arunbaby.com/dsa/0058-regular-expression-matching](https://www.arunbaby.com/dsa/0058-regular-expression-matching/)

*If you found this helpful, consider sharing it with others who might benefit.*
