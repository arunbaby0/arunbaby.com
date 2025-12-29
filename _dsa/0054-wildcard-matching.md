---
title: "Wildcard Matching"
day: 54
collection: dsa
categories:
  - dsa
tags:
  - dynamic-programming
  - string
  - pattern-matching
  - hard
  - edge-cases
  - state-machine
difficulty: Hard
subdomain: "String DP"
tech_stack: Python
scale: "O(MN) DP, optimize to O(N) space"
companies: Google, Meta, Amazon, Microsoft
related_dsa_day: 54
related_ml_day: 54
related_speech_day: 54
related_agents_day: 54
---

**"Wildcard matching is a tiny regex engine: define the states, then let DP do the rest."**

## 1. Problem Statement

Given an input string `s` and a pattern `p`, implement wildcard matching with:
- `?` matches any single character
- `*` matches any sequence of characters (including empty)

Return `True` if the entire string matches the pattern.

Examples:
- `s="aa"`, `p="a"` → `False`
- `s="aa"`, `p="*"` → `True`
- `s="cb"`, `p="?a"` → `False`
- `s="adceb"`, `p="*a*b"` → `True`

This is a classic DP/state-machine interview question.

---

## 2. Understanding the Problem

### 2.1 “Entire string” match vs substring match

We must match the entire string:
- pattern must consume all of `s`
- not “find a substring match”

### 2.3 The core ambiguity: `*` is both empty and “many”

The reason this problem is hard is `*`.
If `*` were only “one character” or only “many characters”, you could do a simple scan.
But `*` can be:
- empty (consume nothing)
- one char
- two chars
- … up to the rest of the string

So the algorithm needs a principled way to explore many possibilities without exponential blowup.
That’s exactly what DP gives you: it explores all possibilities while reusing overlapping subproblems.

### 2.4 Think like an automaton (state machine)

Even though we implement this with DP, it’s conceptually a state machine:
- state = (how many chars of `s` consumed, how many chars of `p` consumed)
- transitions depend on the next pattern token:
  - literal / `?`: consume one of each
  - `*`: either advance pattern (empty match) OR advance string (consume one more char under the star)

If you’ve ever built a lexer, router, or policy engine, this should feel familiar.

### 2.2 Thematic link: pattern recognition and state machines

This connects to today’s cross-track theme:
- pattern matching in ML (matching patterns across data)
- acoustic pattern matching in speech (matching phonetic/acoustic signatures)
- scaling multi-agent systems (agents coordinate via protocols—state machines again)

---

## 3. Approach 1: Brute Force Backtracking

### 3.1 Idea

Recursively try to match:
- if pattern char is literal or `?`, consume one char
- if pattern char is `*`, try matching it as empty or consuming 1+ chars

### 3.2 Complexity
- worst-case exponential due to `*` branching

Useful for intuition, not acceptable for worst-case constraints.

### 3.3 Backtracking + memoization (top-down DP)

An intermediate step that’s worth mentioning:
- recursion on `(i, j)` with memoization

This transforms exponential brute force into O(MN) by caching states.
In interviews, this is a good “bridge” answer if you’re more comfortable with recursion than 2D tables.

---

## 4. Approach 2: Dynamic Programming (2D)

### 4.1 DP state

Let `dp[i][j]` mean:
- does `s[:i]` match `p[:j]`?

Answer is `dp[m][n]`.

### 4.2 Transitions

Let `p[j-1]` be the current pattern char, `s[i-1]` the current string char.

- If `p[j-1]` is a literal or `?`:
  - match if previous matched and current chars compatible:
  - `dp[i][j] = dp[i-1][j-1] and (p[j-1] == '?' or p[j-1] == s[i-1])`

- If `p[j-1]` is `*`:
  - `*` matches empty: `dp[i][j] |= dp[i][j-1]`
  - `*` matches one more char: `dp[i][j] |= dp[i-1][j]`

So:
- `dp[i][j] = dp[i][j-1] or dp[i-1][j]`

### 4.3 Base cases

- `dp[0][0] = True` (empty matches empty)
- `dp[0][j]` is True only if `p[:j]` is all `*`

### 4.3.1 Why the empty-string initialization matters

This is the most common source of bugs.
Example:
- `s=""`, `p="**"` should be True
- `s=""`, `p="*a"` should be False

So you must propagate `dp[0][j] = dp[0][j-1]` only while the pattern remains all `*`.
The first non-`*` breaks the chain forever for the empty string row.

### 4.4 Complexity
- **Time**: \(O(MN)\)
- **Space**: \(O(MN)\)

---

## 5. Space Optimization (1D DP)

We only need previous row, so we can reduce to O(N) space.

### 5.1 1D DP correctness intuition

In 2D DP, each cell `dp[i][j]` depends on:
- `dp[i-1][j-1]` (diagonal)
- `dp[i][j-1]` (left)
- `dp[i-1][j]` (up)

When you compress to 1D, you must preserve the “diagonal” value from the previous row.
That’s why the implementation uses a `prev_diag` variable.

If you forget this, you accidentally mix states from the current row with the previous row and the DP becomes incorrect.

### 5.2 A greedy linear-time alternative (bonus)

There is also a well-known greedy approach that runs in O(M+N) time:
- scan `s` and `p` with pointers
- when you see `*`, record its position and the match position in `s`
- if later mismatch occurs, “backtrack” to the last `*` and let it absorb one more char

This works because `*` is the only source of ambiguity and greedy only needs to remember the last star.

However, DP is usually preferred in interviews because:
- easier to prove
- generalizes to other pattern problems

---

## 6. Implementation (Python)

```python
from typing import List


class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)

        # dp[j] represents dp for current i and pattern prefix length j
        dp = [False] * (n + 1)
        dp[0] = True

        # Initialize dp for empty s: only '*' can match empty
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                dp[j] = dp[j - 1]
            else:
                dp[j] = False

        for i in range(1, m + 1):
            prev_diag = dp[0]  # dp[i-1][0]
            dp[0] = False      # non-empty s can't match empty pattern

            for j in range(1, n + 1):
                temp = dp[j]   # dp[i-1][j] before overwrite
                if p[j - 1] == "*":
                    dp[j] = dp[j] or dp[j - 1]  # dp[i-1][j] or dp[i][j-1]
                else:
                    match = (p[j - 1] == "?" or p[j - 1] == s[i - 1])
                    dp[j] = prev_diag and match  # dp[i-1][j-1] and char match
                prev_diag = temp

        return dp[n]
```

---

## 7. Implementation 2 (Optional): Greedy Two-Pointer

This is a classic solution with linear time in practice.

```python
class SolutionGreedy:
    def isMatch(self, s: str, p: str) -> bool:
        i = j = 0
        star = -1
        match = 0

        while i < len(s):
            if j < len(p) and (p[j] == "?" or p[j] == s[i]):
                i += 1
                j += 1
            elif j < len(p) and p[j] == "*":
                star = j
                match = i
                j += 1
            elif star != -1:
                # Backtrack: let the last '*' match one more character
                j = star + 1
                match += 1
                i = match
            else:
                return False

        # Remaining pattern must be all '*'
        while j < len(p) and p[j] == "*":
            j += 1
        return j == len(p)
```

When to use it:
- large strings where O(MN) is too slow
- repeated matching against many strings (still consider compilation/automata)

---

## 8. Testing

Test cases:
- `("", "")` → True
- `("", "*")` → True
- `("", "?")` → False
- `("aa", "a")` → False
- `("aa", "*")` → True
- `("adceb", "*a*b")` → True
- `("acdcb", "a*c?b")` → False

---

## 9. Complexity Analysis

- **Time**: \(O(MN)\)
- **Space**: \(O(N)\) with 1D DP

---

## 10. Production Considerations

In production, wildcard matching appears in:
- routing rules
- log filtering
- permission policies

Treat patterns as untrusted input:
- bound pattern length
- bound runtime (avoid pathological patterns)
- consider compiling patterns into automata for repeated matching

### 10.1 Pattern normalization (collapse multiple `*`)

One easy optimization:
- convert `"a**b***c"` to `"a*b*c"`

Multiple consecutive stars are equivalent to a single star, but they inflate the DP table and can slow greedy matchers too.

### 10.2 Security: ReDoS analogs

Even though wildcards are simpler than regex, user-supplied patterns can still cause worst-case behavior:
- DP is O(MN) (safe but can be large)
- greedy is O(M+N) but can still do extra work with many mismatches

So in production:
- enforce size limits (pattern length and string length)
- enforce timeouts for complex matching modules

### 10.3 “Compile once, match many” for rule engines

If you match thousands of strings against the same pattern, don’t run DP each time.
Instead:
- pre-process pattern
- use greedy or compile into a small automaton

This is exactly what production rule engines do: they compile rules into efficient runtime structures.

---

## 11. Connections to ML Systems

This connects to pattern matching in ML:
- features often match patterns (regex-like) during data cleaning
- labeling functions in weak supervision often use wildcard/regex rules
- rule engines and validators often rely on automata-like matching

---

## 12. Interview Strategy

- Clarify “entire string match”.
- Explain `*` as two options: empty or consume one char (dp[i][j-1] vs dp[i-1][j]).
- Handle empty string initialization carefully (`dp[0][j]`).

### 12.1 Common interviewer follow-ups

- “Can you optimize space?” → 1D DP
- “Can you do it faster than O(MN)?” → greedy two-pointer
- “How would you test it?” → edge cases around empty strings and stars, plus randomized differential tests vs DP baseline

### 12.2 The one bug to avoid

If you mess up `dp[0][j]` initialization, many cases fail silently.
Always test:
- empty string vs star patterns
- patterns that start with stars but later have literals

---

## 13. Key Takeaways

1. Wildcard matching is a state machine; DP enumerates states safely.
2. `*` creates two transitions: match empty or consume one more character.
3. Space can be reduced to O(N) with a rolling DP array.

### 13.1 Appendix: why this matters outside interviews

Wildcard/pattern matching appears in:
- access control policies
- routing rules in gateways
- monitoring filters

The same principles apply:
- define semantics precisely
- make runtime safe (budgets, compilation)
- make behavior observable (which rule matched?)

---

**Originally published at:** [arunbaby.com/dsa/0054-wildcard-matching](https://www.arunbaby.com/dsa/0054-wildcard-matching/)

*If you found this helpful, consider sharing it with others who might benefit.*

