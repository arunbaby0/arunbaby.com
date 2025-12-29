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
related_ml_day: 54
related_speech_day: 54
related_agents_day: 54
---

**"Wildcard matching is more than a string puzzle—it is the foundation of every file system glob, every firewall rule, and every log-routing engine you use today."**

## 1. Problem Statement

Given an input string `s` and a pattern `p`, implement wildcard matching with support for `?` and `*`.

- `?` matches any single character.
- `*` matches any sequence of characters (including the empty sequence).

The matching must cover the entire input string (not just a substring).

### 1.1 Formal Definition
Given:
- $S = s_1 s_2 \dots s_m$
- $P = p_1 p_2 \dots p_n$

Determine if $S$ matches $P$.

**Example 1**
- Input: `s = "aa"`, `p = "a"`
- Output: `false`
- Explanation: "a" does not match the entire string "aa".

**Example 2**
- Input: `s = "aa"`, `p = "*"`
- Output: `true`
- Explanation: '*' matches any sequence.

**Example 3**
- Input: `s = "cb"`, `p = "?a"`
- Output: `false`
- Explanation: '?' matches 'c', but 'a' does not match 'b'.

**Example 4**
- Input: `s = "adceb"`, `p = "*a*b"`
- Output: `true`
- Explanation: The first '*' matches the empty string, 'a' matches 'a', the second '*' matches "dce", and 'b' matches 'b'.

---

## 2. Understanding the Problem

### 2.1 The Nature of Wildcards
In the world of computer science, wildcard matching is a subset of Regular Expression matching. While full regex (PCRE, POSIX) supports complex quantifiers like `+`, `{n,m}`, and lookaheads, wildcards (often called **globbing**) are simpler and more prevalent in shell environments (e.g., `rm *.txt`).

The simplicity of wildcards often masks their computational complexity. The `*` character introduced **non-determinism**. When a matcher encounters a `*`, it doesn't know how many characters to consume. It could consume zero, one, or one hundred. This creates a branching path of possibilities.

### 2.2 Why This Problem is Hard
The difficulty lies entirely in the `*` operator.
- If we only had literal characters, we would just use a single loop.
- If we only had `?`, we would still use a single loop (checking for equality or `?`).
- With `*`, we face an **ambiguity** that requires exploring multiple futures. If we choose to let `*` match "a" but it should have matched "ab", our linear scan fails.

### 2.3 Visualizing the State Machine
Imagine a state machine where each character in the pattern is a state.

```text
Pattern: a * b

State 0: Start
State 1: Matched 'a'
State 2: The '*' loop (can stay here for any character)
State 3: Matched 'b' (Final State)

Transitions:
(S0) --'a'--> (S1)
(S1) --empty--> (S2)
(S2) --any char--> (S2)
(S2) --'b'--> (S3)
```

In the state machine above, when we are in State 2, we have a choice: stay in State 2 (consuming a character) or move to State 3 if the current character is 'b'. This choice is what leads to the need for Dynamic Programming or backtracking.

### 2.4 Thematic link: Pattern Recognition and Determinism
Today's shared theme across tracks is **Pattern Matching and State Machines**:
- **DSA**: We are building a discrete pattern matcher for strings.
- **ML System Design**: Pattern matching is used in feature engineering (detecting sequences in time-series) and data validation (regex-based schema checks).
- **Speech Tech**: Acoustic pattern matching (finding a specific wake-word like "Hey Siri") uses HMMs (Hidden Markov Models) which are essentially probabilistic state machines.
- **Agents**: Scaling multi-agent systems requires agents to match message patterns to internal handlers (a form of "routing" or "dispatching").

---

## 3. Approach 1: Recursive Backtracking (Brute Force)

### 3.1 The Logic
A natural way to solve this is to define a recursive function `isMatch(s_idx, p_idx)`.

1. If both indices reach the end, return `True`.
2. If the pattern ends but the string doesn't, return `False`.
3. If the string ends but the pattern remains:
   - If the remaining pattern characters are all `*`, return `True` (as `*` can match empty).
   - Otherwise, return `False`.
4. If `p[p_idx]` is a literal or `?`:
   - Match if `s[s_idx] == p[p_idx]` or `p[p_idx] == '?'`.
   - Recurse to `isMatch(s_idx + 1, p_idx + 1)`.
5. If `p[p_idx]` is `*`:
   - Choice A: `*` matches empty. Recurse to `isMatch(s_idx, p_idx + 1)`.
   - Choice B: `*` matches the current character and potentially more. Recurse to `isMatch(s_idx + 1, p_idx)`.

### 3.2 Complexity Analysis
- **Time**: $O(2^{M+N})$ in the worst case (e.g., `s = "aaaaa"`, `p = "*****b"`). Every `*` creates two branches.
- **Space**: $O(M+N)$ for the recursion stack.

### 3.3 Adding Memoization (Top-Down DP)
To optimize, we cache the results of `(s_idx, p_idx)`. There are only $M \times N$ possible states.
- **Time**: $O(M \times N)$.
- **Space**: $O(M \times N)$.

---

## 4. Approach 2: Dynamic Programming (Bottom-Up)

Bottom-up DP is often preferred in high-performance systems to avoid recursion depth issues and to leverage cache locality.

### 4.1 The DP State
Let `dp[i][j]` be a boolean indicating whether `s[0...i-1]` matches `p[0...j-1]`.

The table size will be `(len(s) + 1) x (len(p) + 1)`.

### 4.2 Base Cases
1. `dp[0][0] = True`: An empty string matches an empty pattern.
2. `dp[i][0] = False` for $i > 0$: A non-empty string cannot match an empty pattern.
3. `dp[0][j]`: An empty string can only match a pattern if it consists entirely of `*`.
   - `dp[0][j] = dp[0][j-1]` if `p[j-1] == '*'`.

### 4.3 Transitions
For `dp[i][j]`, check `p[j-1]`:

1. **If `p[j-1]` is a literal or `?`**:
   - We match if the current characters match AND the previous prefixes matched.
   - `dp[i][j] = dp[i-1][j-1]` and (`p[j-1] == s[i-1]` or `p[j-1] == '?'`)

2. **If `p[j-1]` is `*`**:
   - `*` acts as empty: `dp[i][j] = dp[i][j-1]`
   - `*` acts as one or more characters: `dp[i][j] = dp[i-1][j]`
   - Combining them: `dp[i][j] = dp[i][j-1] or dp[i-1][j]`

### 4.4 ASCII Visualization of DP Table
Let's match `s = "adceb"`, `p = "*a*b"`.

| | '' | * | a | * | b |
|---|---|---|---|---|---|
| **''** | T | T | F | F | F |
| **a** | F | T | T | T | F |
| **d** | F | T | F | T | F |
| **c** | F | T | F | T | F |
| **e** | F | T | F | T | F |
| **b** | F | T | F | T | T |

**Final Answer: `dp[5][4] = True`**

Notes on the table:
- The first row handles the empty string. Notice how `*` propagate `True`.
- The first column (after `[0][0]`) is always `False`.
- For `*`, we look **Up** (matching one char) or **Left** (matching empty).

---

## 5. Space Optimization: The 1D Rolling Array

Looking at the transitions:
- `dp[i][j]` depends on `dp[i-1][j-1]`, `dp[i][j-1]`, and `dp[i-1][j]`.
- This means we only need the **previous row** and the **current row**.

We can reduce space to $O(N)$ by using a single array of size $N+1$. However, we must be careful with the "diagonal" dependency (`dp[i-1][j-1]`). We usually store the previous row's value in a temporary variable before overwriting.

---

## 6. Theory: Automata and Non-Determinism

### 6.1 NFA vs DFA
In automata theory, a wildcard pattern can be represented as a **Non-deterministic Finite Automaton (NFA)**.
- A **Non-deterministic** machine can be in multiple states at once. When we see `*`, the machine "splits" into two: one state remains at the `*` loop, and another moves to the next character in the pattern.
- A **Deterministic Finite Automaton (DFA)** can only be in one state at any time.

Any NFA can be converted into a DFA using the **powerset construction**. However, the number of states in the DFA can be exponential relative to the NFA ($2^N$). For wildcard matching, the DFA is usually manageable, but the DP approach effectively simulates the NFA in $O(MN)$ time.

### 6.2 Thompson's Construction
Thompson's algorithm is a method for transforming a regular expression into an NFA. For wildcards:
- `?` is a single transition with a wildcard label.
- `*` is a state with an epsilon transition to itself (loop) and a transition to the next state.

In production engines like Go's `regexp` or Google's `RE2`, this NFA simulation is used to guarantee linear time matching relative to the input string length, avoiding the "exponential blowup" seen in some backtracking engines.

---

## 7. Approach 3: Greedy Two-Pointer (The "Optimal" Interview Answer)

While DP is $O(MN)$, there is a clever $O(MN)$ worst-case but $O(M+N)$ average-case greedy approach. It works because wildcards have a property: once a `*` matches a certain prefix, if we fail later, we only need to backtrack to the **most recent** `*` and try matching it against one more character.

### 7.1 The Logic
Keep pointers `s_ptr`, `p_ptr`, `star_ptr` (index of last `*`), and `match_ptr` (index in `s` where `*` started matching).

1. If `s[s_ptr]` matches `p[p_ptr]` (literal or `?`), increment both.
2. If `p[p_ptr]` is `*`:
   - Record `star_ptr = p_ptr` and `match_ptr = s_ptr`.
   - Increment `p_ptr` (try matching `*` as empty first).
3. If mismatch occurs:
   - If we have a `star_ptr`, backtrack!
   - `p_ptr = star_ptr + 1`.
   - `match_ptr += 1` (the star now matches one more char).
   - `s_ptr = match_ptr`.
4. If no `star_ptr` and mismatch, return `False`.

### 7.2 Why this works
This is a form of backtracking that only remembers the **last** bit of non-determinism. Because `*` matches anything, any string matched by an earlier `*` could also have been matched by a later `*`. Thus, we only ever need to "shift" the most recent star.

---

## 8. Implementation

### 8.1 Approach 1: Top-Down Recursion with Memoization

This approach is the most direct translation of the mathematical recurrence. It uses a cache to avoid redundant sub-problems.

```python
class SolutionRecursive:
    """
    Top-Down Memoization.
    Time: O(M * N)
    Space: O(M * N) for the memoization table and recursion stack.
    """
    def isMatch(self, s: str, p: str) -> bool:
        memo = {}

        def dp(i: int, j: int) -> bool:
            # Check cache
            if (i, j) in memo:
                return memo[(i, j)]
            
            # Base Case: Both reached end
            if i == len(s) and j == len(p):
                return True
            # Pattern ended but string hasn't
            if j == len(p):
                return False
            # String ended but pattern hasn't
            if i == len(s):
                # Pattern must only contain '*' to match empty string
                return p[j] == '*' and dp(i, j + 1)

            # Recursive step
            res = False
            if p[j] == s[i] or p[j] == '?':
                res = dp(i + 1, j + 1)
            elif p[j] == '*':
                # Match empty OR match one/more characters
                res = dp(i, j + 1) or dp(i + 1, j)
            
            memo[(i, j)] = res
            return res

        return dp(0, 0)
```

### 8.2 Approach 2: Bottom-Up Dynamic Programming (Standard)

The iterative version is usually preferred in production for its predictable memory layout and lack of stack overflow risks.

```python
from typing import List

class SolutionDP:
    """
    Standard Bottom-Up DP solution.
    Time Complexity: O(M * N)
    Space Complexity: O(M * N)
    """
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        
        # Initialize DP table: dp[i][j] means s[:i] matches p[:j]
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        # Base Case: Empty string matches empty pattern
        dp[0][0] = True
        
        # Base Case: Empty string matching pattern with '*'
        # '*' can match zero characters, so it inherits from the previous pattern index.
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-1]
        
        # Fill the table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j-1] == s[i-1] or p[j-1] == '?':
                    # Characters match, carry over the result from both prefixes
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == '*':
                    # Branching logic for '*':
                    # Case 1: '*' matches empty (treat it as skipping the star)
                    # Case 2: '*' matches one or more (treat it as consuming s[i-1])
                    dp[i][j] = dp[i][j-1] or dp[i-1][j]
        
        return dp[m][n]
```

### 8.3 Approach 3: Greedy Two-Pointer (Optimal Space)

This is the "trick" solution that reduces space to $O(1)$ by manually managing the backtracking to the last seen `*`.

```python
class SolutionGreedy:
    """
    Greedy backtracking with O(1) extra space.
    Time Complexity: O(M * N) worst case, but near O(M+N) on average.
    Space Complexity: O(1)
    """
    def isMatch(self, s: str, p: str) -> bool:
        s_ptr = p_ptr = 0
        star_idx = -1
        last_s_match = 0
        
        while s_ptr < len(s):
            # 1. Match constant char or '?'
            if p_ptr < len(p) and (p[p_ptr] == s[s_ptr] or p[p_ptr] == '?'):
                s_ptr += 1
                p_ptr += 1
            # 2. Match '*'
            elif p_ptr < len(p) and p[p_ptr] == '*':
                # Record the star position and the current string position
                star_idx = p_ptr
                last_s_match = s_ptr
                p_ptr += 1 # Try matching empty first
            # 3. Mismatch, but we have a previous '*' to backtrack to
            elif star_idx != -1:
                # Backtrack pattern pointer to just after the star
                p_ptr = star_idx + 1
                # Increment the string pointer to the next possible match for the star
                last_s_match += 1
                s_ptr = last_s_match
            # 4. Total mismatch
            else:
                return False
        
        # Check if remaining characters in pattern are all '*'
        while p_ptr < len(p) and p[p_ptr] == '*':
            p_ptr += 1
            
        return p_ptr == len(p)
```

---

## 9. Advanced: The Bitap Algorithm

For fuzzy string matching and wildcard patterns, the **Bitap Algorithm** (also known as the Shift-Or algorithm) is an extremely fast bit manipulation technique.

### 9.1 How it Works
The Bitap algorithm maintains a bitmask of potential matches. Each bit in the mask represents whether a prefix of the pattern matches the current prefix of the string.
- When a `?` is encountered, we shift the bitmask.
- When a `*` is encountered, we perform bitwise operations to allow any number of characters.

### 9.2 Why use it?
- **Speed**: It uses CPU bitwise instructions which are incredibly fast.
- **Hardware Friendship**: It is very easy to implement in hardware (FGPAs) or low-level SIMD instructions.
- **Fuzzy matching**: It can be easily extended to support "Levenstein distance" (allowing $k$ errors).

While Bitap is usually overkill for interview wildcard matching, mentioning it as a "high-performance alternative" shows a level of depth that many candidates lack.

---

## 10. Complexity Analysis

| Approach | Time Complexity | Space Complexity | Notes |
|---|---|---|---|
| **Brute Force** | $O(2^{M+N})$ | $O(M+N)$ | Terrible for long strings with many stars. |
| **Top-Down DP** | $O(MN)$ | $O(MN)$ | Good if many states are unreachable. |
| **Bottom-Up DP** | $O(MN)$ | $O(MN)$ | Predictable and cache-friendly. |
| **Space-Optimized DP** | $O(MN)$ | $O(N)$ | The standard for memory-constrained systems. |
| **Greedy Pointer** | $O(MN)$ worst-case | $O(1)$ | Best in practice for most strings. |

---

## 11. Production Considerations

### 11.1 History: Unix Globs
The term "glob" comes from the very first versions of Unix. There was a standalone program called `/etc/glob` that would expand wildcard patterns for the shell. Later, this logic was moved into the C library as `glob()`. Wildcard matching is thus "baked into" the DNA of modern computing.

### 11.2 Pattern Normalization (Optimization)
Before running the matching algorithm, normalize your pattern:
- Collapse multiple consecutive stars into one. `**` behaves exactly like `*`.
- Complexity impact: This reduces the $N$ in $O(MN)$, potentially saving significant time in complex patterns.

```python
def normalize(p: str) -> str:
    if not p: return p
    res = [p[0]]
    for i in range(1, len(p)):
        if p[i] == '*' and p[i-1] == '*':
            continue
        res.append(p[i])
    return "".join(res)
```

### 11.3 Security: Avoiding ReDoS (Regex Denial of Service)
While Wildcard matching is safer than full Regex (which can have exponential $O(2^N)$ backtracking in some engines), $O(MN)$ can still be abused.
- If a user provides a pattern of $10^5$ characters and a string of $10^5$ characters, the DP table would require $10^{10}$ booleans (~10GB of RAM).
- **Defense**: Enforce limits on pattern and string lengths in your API. Usually, 1024 or 2048 is more than enough for globbing.

### 11.4 Wildcards in Databases: SQL `LIKE`
In SQL, the `%` character acts as a wildcard (matching zero or more characters) and `_` acts as a single-character wildcard (equivalent to `?`).
- `SELECT * FROM users WHERE email LIKE 'admin_%@google.com'`
- Databases often optimize these queries by using indexes. If the pattern **starts** with literal characters (e.g., `admin_%`), the database can use a B-Tree index to perform a range scan. If it starts with a wildcard (e.g., `%@google.com`), it must perform a full table scan, which is $O(N)$ relative to the number of rows.

### 11.5 Compilation to Machine Code
For extremely high-performance filtering (e.g., cloud networking layers), wildcard patterns are often compiled into specialized machine code or eBPF programs. This allows matching at line speed (100Gbps+) by reducing the matching logic to a series of optimized jumps and comparisons.

### 11.6 Case Study: AWS IAM Policy Evaluation
In AWS Identity and Access Management (IAM), policies allow you to define permissions for users and roles. These policies often use wildcards in the `Resource` field.
Example: `arn:aws:s3:::my-bucket/logs/2023/*`
- When a request comes in for `arn:aws:s3:::my-bucket/logs/2023/error.log`, AWS must match it against the policy.
- AWS evaluates thousands of these policies per second.
- To handle this scale, they don't just run a naive DP. They use **optimized state machines** that can evaluate multiple patterns at once.
- Security is paramount: the matcher must be **strictly linear** to prevent "denial of service" attacks using complex patterns.

---

## 12. Implementation Deep Dive: Line-by-Line

### 12.1 Explaining the DP Implementation
Let's look at the core loop of the DP solution again:

```python
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if p[j-1] == s[i-1] or p[j-1] == '?':
            dp[i][j] = dp[i-1][j-1]
```
- **Line 1 & 2**: We iterate through every character of the string (`i`) and every character of the pattern (`j`).
- **Line 3**: We check for a "local match". If the characters are the same, or the pattern has a `?`.
- **Line 4**: If it's a local match, then the logic is: "Does this prefix match?" depends entirely on "Did the previous prefix (excluding these two chars) match?". This is why we look at the diagonal `dp[i-1][j-1]`.

```python
        elif p[j-1] == '*':
            dp[i][j] = dp[i][j-1] or dp[i-1][j]
```
- **Line 5**: If the pattern is a star...
- **Line 6**: This is the heart of the branching logic.
  - `dp[i][j-1]`: Can we match by treating the `*` as an empty string? (i.e., ignore the star and check if the current string prefix matched the pattern prefix *before* the star).
  - `dp[i-1][j]`: Can we match by letting the `*` consume the current character `s[i-1]`? (i.e., if the shorter string `s[:i-1]` already matched the current pattern prefix including the star, then the star just "eats" one more character).

### 12.2 Explaining the Greedy Implementation
The greedy pointer approach is often more confusing. Let's break down the "backtrack" logic:

```python
elif star_idx != -1:
    p_ptr = star_idx + 1
    s_tmp_idx += 1
    s_ptr = s_tmp_idx
```
- **Line 1**: We only reach this `elif` if there was a mismatch at the current `p_ptr` and `s_ptr`.
- **Line 2**: We reset the pattern pointer to the character **immediately following** the last star we saw.
- **Line 3**: We increment `s_tmp_idx`. This variable tracks the "end of the match" for the star. By incrementing it, we are effectively saying: "The star didn't match enough characters; let's try making it match one more."
- **Line 4**: We reset the string pointer to this new starting point.

This "backtracking to the last star" is what allows the algorithm to explore different "lengths" for the star's match without the full $O(MN)$ overhead of a DP table in most cases.

---

## 13. Connections to ML Systems

### 13.1 Dynamic Matching in Feature Engineering
In sequence-based ML (like NLP or Clickstream analysis), we often use wildcard-like patterns to define features.
- "Did the user perform sequence `A -> * -> B`?"
- This is essentially wildcard matching. The DP approach we discussed is the foundation of **DTW (Dynamic Time Warping)**, a popular algorithm for matching two temporal sequences that may vary in speed.

### 13.2 Data Validation Guardrails
Multi-agent systems and ML pipelines are notoriously "silent fail" prone. If a data schema changes slightly, the model might still produce results, but they will be garbage (**GIGO - Garbage In, Garbage Out**).
- We use pattern matching in **Data Contracts**.
- Example: "Ensure the `source_id` follows the pattern `REGION_*_TYPE_??`". This ensures that incoming data flows are structured correctly before hitting the model.

### 13.3 Search and Retrieval
In Agentic RAG (Retrieval-Augmented Generation), we often filter metadata.
- "Find all documents where `category` matches `finance/*`".
- If the metadata store is a SQL database, this is converted to a `LIKE 'finance/%'` query.
- Knowing the underlying matching logic helps you understand how the index will (or won't) be used.

---

## 14. Key Takeaways

1. **Fundamental State Machine**: Wildcard matching is the bridge between simple string comparison and full Regular Expressions.
2. **DP is Robust**: The $O(MN)$ DP approach is the most reliable way to handle the non-determinism of `*`.
3. **Space Matters**: Reducing $O(MN)$ to $O(N)$ space is a critical optimization for production scale.
4. **History Matters**: From Unix `glob` to modern SQL, wildcard matching is one of the most successful abstractions in computer science.

---

**Originally published at:** [arunbaby.com/dsa/0054-wildcard-matching](https://www.arunbaby.com/dsa/0054-wildcard-matching/)

*If you found this helpful, consider sharing it with others who might benefit.*
