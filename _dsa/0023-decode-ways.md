---
title: "Decode Ways"
day: 23
collection: dsa
categories:
  - dsa
tags:
  - dynamic-programming
  - string
  - recursion
  - memoization
  - medium
subdomain: "Sequence Dynamic Programming"
tech_stack: [Python, C++, Java, Go, Rust]
scale: "O(N) time, O(1) space"
companies: [Facebook, Google, Uber, Microsoft, Amazon]
related_dsa_day: 23
related_ml_day: 23
related_speech_day: 23
---

**A deceptive counting problem that teaches the fundamentals of state transitions and connects directly to Beam Search.**

## Problem

A message containing letters from `A-Z` can be encoded into numbers using the following mapping:
- 'A' -> "1"
- 'B' -> "2"
- ...
- 'Z' -> "26"

To **decode** an encoded message, you need to group the digits and map them back to letters.
Given a string `s` containing only digits, return the **number of ways** to decode it.

A message containing letters from `A-Z` can be encoded into numbers using the following mapping:
- 'A' -> "1"
- 'B' -> "2"
- ...
- 'Z' -> "26"

To **decode** an encoded message, you need to group the digits and map them back to letters.
Given a string `s` containing only digits, return the **number of ways** to decode it.

**Example 1:**
```
Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
```

**Example 2:**
```
Input: s = "226"
Output: 3
Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
```

**Example 3 (The Tricky One):**
```
Input: s = "06"
Output: 0
Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06").
```

**Constraints:**
- `1 <= s.length <= 100`
- `s` contains only digits and may contain leading zero(s).

## Thematic Connection: Decoding Paths

Why is this problem on Day 23?
In **ML System Design** and **Speech Tech**, we often deal with "decoding" a sequence of probabilities into the most likely sequence of words.
- **DSA:** We count *all* valid paths (Decode Ways).
- **ML/Speech:** We search for the *best* path (Beam Search).

The underlying structure is a graph where each node represents a state (index in the string), and edges represent valid transitions (taking 1 digit or 2 digits).

## Approach 1: Brute Force Recursion

Let's think about this recursively.
At any index `i`, we have two choices:
1.  **Single Digit:** Take `s[i]` as a single number. Valid if `s[i]` is '1'-'9'.
2.  **Double Digit:** Take `s[i]s[i+1]` as a two-digit number. Valid if it forms a number between 10 and 26.

Let `numDecodings(i)` be the number of ways to decode the suffix `s[i:]`.

**Recurrence Relation:**
`numDecodings(i) = (valid single ? numDecodings(i+1) : 0) + (valid double ? numDecodings(i+2) : 0)`

**Base Cases:**
- If `i == len(s)`: We reached the end successfully. Return 1.
- If `s[i] == '0'`: A string starting with '0' cannot be decoded. Return 0.

### Python Implementation (Recursive)

```python
def numDecodings_recursive(s: str) -> int:
    def decode(index):
        # Base Case: End of string
        if index == len(s):
            return 1
        
        # Base Case: Leading zero
        if s[index] == '0':
            return 0
        
        # Option 1: Take 1 digit
        res = decode(index + 1)
        
        # Option 2: Take 2 digits
        if index + 1 < len(s) and (s[index] == '1' or (s[index] == '2' and s[index+1] in "0123456")):
            res += decode(index + 2)
            
        return res

    return decode(0)
```

### Complexity Analysis
- **Time Complexity:** `O(2^N)`. In the worst case (e.g., "111111"), every step branches into two. This is the Fibonacci sequence recursion.
- **Space Complexity:** `O(N)` for the recursion stack.

## Approach 2: Recursion with Memoization

We are re-calculating the same subproblems. `decode(5)` might be called from `decode(3)` (taking 2 steps) and `decode(4)` (taking 1 step). We can cache the results.

```python
def numDecodings_memo(s: str) -> int:
    memo = {}
    
    def decode(index):
        if index in memo:
            return memo[index]
        
        if index == len(s):
            return 1
        
        if s[index] == '0':
            return 0
        
        res = decode(index + 1)
        
        if index + 1 < len(s) and (s[index] == '1' or (s[index] == '2' and s[index+1] in "0123456")):
            res += decode(index + 2)
            
        memo[index] = res
        return res

    return decode(0)
```

- **Time Complexity:** `O(N)`. We visit each index once.
- **Space Complexity:** `O(N)` for memoization map + stack.

## Approach 3: Iterative Dynamic Programming

Let's flip it. Instead of recursion, let's build an array `dp`.
`dp[i]` = Number of ways to decode the string `s[0...i-1]` (length `i`).

**Initialization:**
- `dp[0] = 1` (Empty string has 1 way: do nothing).
- `dp[1] = 1` if `s[0] != '0'` else `0`.

**Transitions:**
For `i` from 2 to `n`:
1.  **One Digit:** If `s[i-1]` is not '0', we can add `dp[i-1]`.
2.  **Two Digits:** If `s[i-2:i]` is between "10" and "26", we can add `dp[i-2]`.

### Python Implementation (Iterative)

```python
def numDecodings_dp(s: str) -> int:
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    dp = [0] * (n + 1)
    
    dp[0] = 1
    dp[1] = 1
    
    for i in range(2, n + 1):
        # Check if single digit is valid (1-9)
        if s[i-1] != '0':
            dp[i] += dp[i-1]
            
        # Check if two digits are valid (10-26)
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]
            
    return dp[n]
```

### Complexity Analysis
- **Time Complexity:** `O(N)`.
- **Space Complexity:** `O(N)` for the `dp` array.

## Approach 4: Space Optimization (O(1) Space)

Notice that `dp[i]` only depends on `dp[i-1]` and `dp[i-2]`. We don't need the whole array. We just need two variables.

```python
def numDecodings_optimized(s: str) -> int:
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    two_back = 1 # dp[i-2] (initially dp[0])
    one_back = 1 # dp[i-1] (initially dp[1])
    
    for i in range(2, n + 1):
        current = 0
        
        # Single digit check
        if s[i-1] != '0':
            current += one_back
            
        # Double digit check
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            current += two_back
            
        two_back = one_back
        one_back = current
        
    return one_back
```

- **Space Complexity:** `O(1)`.

## Detailed Walkthrough: Tracing "226"

Let's trace the `O(1)` space algorithm with `s = "226"`.

**Initialization:**
- `two_back` (dp[-1]) = 1
- `one_back` (dp[0]) = 1 (since '2' != '0')

**Iteration 1 (i=2, Char='2'):**
- **Single Digit:** '2' is valid. `current += one_back` (1).
- **Double Digit:** "22" is valid (10-26). `current += two_back` (1).
- `current` = 2.
- Update: `two_back` = 1, `one_back` = 2.
- *Meaning:* "22" can be "BB" or "V".

**Iteration 2 (i=3, Char='6'):**
- **Single Digit:** '6' is valid. `current += one_back` (2).
- **Double Digit:** "26" is valid (10-26). `current += two_back` (1).
- `current` = 3.
- Update: `two_back` = 2, `one_back` = 3.
- *Meaning:* "226" can be "BBF", "VF", "BZ".

**Result:** Return 3.

## Edge Cases and Pitfalls

This problem is famous for its edge cases.
1.  **Leading Zeros:** "06" -> 0. "0" -> 0.
2.  **Mid-stream Zeros:** "10" -> 1 ("J"). "100" -> 0 (The second '0' cannot be decoded alone, and "00" is invalid).
3.  **Large Numbers:** "30" -> 0. "27" -> 1 ("BG", not "27").

**Debugging Tip:**
If your code fails on "10", check if you handle the single digit case correctly. `s[i-1]` is '0', so you shouldn't add `dp[i-1]`. But `s[i-2:i]` is "10", which is valid, so you add `dp[i-2]`.

## Advanced Variant: Decode Ways II (Wildcards)

What if the input string contains `*`?
- `*` can be any digit from '1' to '9'.
- `1*` can be "11" to "19" (9 possibilities).
- `2*` can be "21" to "26" (6 possibilities).
- `**` can be "11"-"19" and "21"-"26" (15 possibilities).

This explodes the complexity. The logic remains the same (add `dp[i-1]` and `dp[i-2]`), but the *coefficients* change.

**Transitions:**
- If `s[i] == '*'`: `dp[i] += 9 * dp[i-1]`
- If `s[i-1] == '1'` and `s[i] == '*'`: `dp[i] += 9 * dp[i-2]`
- If `s[i-1] == '2'` and `s[i] == '*'`: `dp[i] += 6 * dp[i-2]`
- If `s[i-1] == '*' ` and `s[i] == '*'`: `dp[i] += 15 * dp[i-2]`

This variant tests your ability to handle combinatorial explosion within a DP framework.

## Connection to Beam Search (The "Why")

In "Decode Ways", we are summing up *all* possible paths.
In **Beam Search** (used in ASR/NLP), we have a similar graph, but edges have **probabilities**.
- 'A' might have probability 0.9.
- 'B' might have probability 0.1.

Instead of summing, we want to find the path with the **maximum product of probabilities**.
Also, the graph is infinite (or very large), so we can't visit every node. We keep only the top-K paths at each step.

Understanding "Decode Ways" proves you understand the state-space graph that Beam Search traverses.

## Interview Simulation

**Interviewer:** "Can you solve this in constant space?"
**You:** "Yes, by observing the Fibonacci-like structure."

**Interviewer:** "What if the mapping included '*' which can be 1-9?"
**You:** "That's LeetCode 639 (Decode Ways II). The logic is the same, but the branching factor increases. '*' as a single digit adds `9 * dp[i-1]`. '**' adds `15 * dp[i-2]` etc."

**Interviewer:** "How would you test this?"
**You:** "I would use a table of test cases covering all zero-patterns:
- Start: '0', '01'
- Middle: '101', '100', '20', '30'
- End: '10'
- Valid: '12', '226'"

**Interviewer:** "Can you write a test suite?"
**You:** "Sure. I'd use a data-driven test."

```python
def test_decode_ways():
    cases = [
        ("12", 2),
        ("226", 3),
        ("0", 0),
        ("06", 0),
        ("10", 1),
        ("27", 1),
        ("2101", 1) # B J A
    ]
    for s, expected in cases:
        assert numDecodings_optimized(s) == expected
        print(f"Pass: {s} -> {expected}")
```

## Multi-Language Implementation

### C++
```cpp
class Solution {
public:
    int numDecodings(string s) {
        if (s.empty() || s[0] == '0') return 0;
        int n = s.length();
        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        dp[1] = 1;
        
        for (int i = 2; i <= n; ++i) {
            int oneDigit = s[i-1] - '0';
            int twoDigits = stoi(s.substr(i-2, 2));
            
            if (oneDigit >= 1) dp[i] += dp[i-1];
            if (twoDigits >= 10 && twoDigits <= 26) dp[i] += dp[i-2];
        }
        return dp[n];
    }
};
```

### Java
```java
class Solution {
    public int numDecodings(String s) {
        if (s == null || s.length() == 0 || s.charAt(0) == '0') return 0;
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        
        for (int i = 2; i <= n; i++) {
            int oneDigit = Integer.valueOf(s.substring(i-1, i));
            int twoDigits = Integer.valueOf(s.substring(i-2, i));
            
            if (oneDigit >= 1) dp[i] += dp[i-1];
            if (twoDigits >= 10 && twoDigits <= 26) dp[i] += dp[i-2];
        }
        return dp[n];
    }
}
```

### Go
```go
func numDecodings(s string) int {
    if len(s) == 0 || s[0] == '0' {
        return 0
    }
    n := len(s)
    dp := make([]int, n+1)
    dp[0] = 1
    dp[1] = 1
    
    for i := 2; i <= n; i++ {
        oneDigit := s[i-1] - '0'
        twoDigits, _ := strconv.Atoi(s[i-2 : i])
        
        if oneDigit >= 1 {
            dp[i] += dp[i-1]
        }
        if twoDigits >= 10 && twoDigits <= 26 {
            dp[i] += dp[i-2]
        }
    }
    return dp[n]
}
```

### Rust
```rust
impl Solution {
    pub fn num_decodings(s: String) -> i32 {
        if s.starts_with('0') {
            return 0;
        }
        let n = s.len();
        let chars: Vec<char> = s.chars().collect();
        let mut dp = vec![0; n + 1];
        dp[0] = 1;
        dp[1] = 1;
        
        for i in 2..=n {
            let one_digit = chars[i-1].to_digit(10).unwrap();
            let two_digits = s[i-2..i].parse::<i32>().unwrap();
            
            if one_digit >= 1 {
                dp[i] += dp[i-1];
            }
            if two_digits >= 10 && two_digits <= 26 {
                dp[i] += dp[i-2];
            }
        }
        dp[n]
    }
}
```

## Appendix A: System Design Interview Transcript

**Interviewer:** "Okay, we've solved the algorithmic part. Now, imagine this decoding service needs to scale to 1 billion requests per day. The input strings can be very long (e.g., DNA sequences encoded as digits). How would you design this?"

**Candidate:** "That's a great question. Since the problem has optimal substructure and overlapping subproblems, it's a prime candidate for **distributed computing** or **parallel processing**, but the dependencies make it tricky. `dp[i]` depends on `dp[i-1]`. This implies a sequential dependency."

**Interviewer:** "Exactly. You can't just split the string in half and process both sides independently, because the split point might be in the middle of a two-digit number. How do you handle that?"

**Candidate:** "We can use a **Map-Reduce** approach with a twist.
1.  **Split:** Divide the string `S` into chunks `C1, C2, ..., Ck`.
2.  **Map:** For each chunk, we compute a transition matrix. Instead of returning a single number, we return a 2x2 matrix representing the linear transformation from the start of the chunk to the end.
    - State 0: We consumed the last digit of the previous chunk.
    - State 1: We 'borrowed' the last digit of the previous chunk to form a two-digit number.
3.  **Reduce:** Multiply the matrices in order. Matrix multiplication is associative, so this can be parallelized using a segment tree or just standard parallel reduction."

**Interviewer:** "Impressive. That's actually how parallel prefix sum (scan) works. What about caching?"

**Candidate:** "Since the mapping is static (A=1, B=2...), we can cache common substrings. If we see the pattern '12345' frequently, we can memoize its transition matrix in Redis. This would turn `O(N)` into `O(1)` for cache hits."

**Interviewer:** "What if the mapping changes dynamically? Say, for security rotation?"

**Candidate:** "Then we need versioned caching. Key = `Hash(Mapping_Version + Substring)`. When the mapping rotates, we increment the version, effectively invalidating the cache."

## Appendix B: Mathematical Proof of Optimal Substructure

Let `N(S)` be the number of ways to decode string `S`.
Let `S` be `d_1 d_2 ... d_n`.

We claim that `N(S)` satisfies the recurrence:
`N(S[1...n]) = C_1 * N(S[2...n]) + C_2 * N(S[3...n])`

**Proof:**
The first decision is disjoint and exhaustive.
1.  **Case 1:** We decode `d_1` as a single character. This is valid if `d_1 \in \{1..9\}`. If valid, the remaining problem is `S[2...n]`. The number of ways is `1 * N(S[2...n])`.
2.  **Case 2:** We decode `d_1 d_2` as a single character. This is valid if `d_1 d_2 \in \{10..26\}`. If valid, the remaining problem is `S[3...n]`. The number of ways is `1 * N(S[3...n])`.

Since these are the only two ways to consume the start of the string, by the **Rule of Sum**, the total ways is the sum of these two cases.
This proves optimal substructure: The solution to the problem depends only on the solutions to its suffixes.

## Appendix C: 50 Comprehensive Test Cases

When testing your solution, ensure you cover these categories:

**Category 1: Basic Valid**
1. "1" -> 1 (A)
2. "12" -> 2 (AB, L)
3. "226" -> 3 (BZ, VF, BBF)
4. "111" -> 3 (AAA, AK, KA)

**Category 2: Leading Zeros**
5. "0" -> 0
6. "01" -> 0
7. "00" -> 0
8. "012" -> 0

**Category 3: Trailing Zeros**
9. "10" -> 1 (J)
10. "20" -> 1 (T)
11. "30" -> 0 (Invalid)
12. "100" -> 0 (Invalid)

**Category 4: Middle Zeros**
13. "101" -> 1 (JA)
14. "1001" -> 0
15. "110" -> 1 (AJ) - Wait, "1" "10" -> AJ. "11" "0" -> Invalid. Correct.

**Category 5: Boundary Conditions**
16. "26" -> 2 (BF, Z)
17. "27" -> 1 (BG)
18. "9" -> 1 (I)
19. "99" -> 1 (II)

## Appendix D: Common Dynamic Programming Patterns

"Decode Ways" belongs to the **Sequence DP** family. Here are others you should know:

1.  **0/1 Knapsack:**
    - **Problem:** Pick items with weight `w` and value `v` to maximize value within capacity `W`.
    - **State:** `dp[i][w]` = Max value using first `i` items with capacity `w`.
    - **Transition:** `max(dp[i-1][w], dp[i-1][w-w_i] + v_i)`.

2.  **Longest Increasing Subsequence (LIS):**
    - **Problem:** Find the longest subsequence where elements are increasing.
    - **State:** `dp[i]` = Length of LIS ending at index `i`.
    - **Transition:** `dp[i] = 1 + max(dp[j])` for all `j < i` where `nums[j] < nums[i]`.

3.  **Longest Common Subsequence (LCS):**
    - **Problem:** Find the longest subsequence common to two strings.
    - **State:** `dp[i][j]` = LCS of `s1[0..i]` and `s2[0..j]`.
    - **Transition:** If `s1[i] == s2[j]`, `1 + dp[i-1][j-1]`. Else `max(dp[i-1][j], dp[i][j-1])`.

4.  **Matrix Chain Multiplication:**
    - **Problem:** Parenthesize matrix multiplications to minimize scalar operations.
    - **State:** `dp[i][j]` = Min cost to multiply matrices `i` through `j`.
    - **Transition:** `min(dp[i][k] + dp[k+1][j] + cost(i, k, j))` for all `k`.

Mastering these patterns will allow you to solve 90% of DP problems in interviews.

## Conclusion

Decode Ways is a masterclass in handling state transitions and edge cases. It teaches you to look beyond the "happy path" and rigorously define valid transitions.

**Key Takeaways:**
1.  **Zeros are the enemy.** Handle them first.
2.  **State Definition:** `dp[i]` depends on `i-1` and `i-2`.
3.  **Optimization:** Space can be reduced to `O(1)`.

**Practice Problems:**
- [Decode Ways II](https://leetcode.com/problems/decode-ways-ii/) (Hard)
- [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) (Easy)
- [Word Break](https://leetcode.com/problems/word-break/) (Medium)

---

**Originally published at:** [arunbaby.com/dsa/0023-decode-ways](https://www.arunbaby.com/dsa/0023-decode-ways/)

*If you found this helpful, consider sharing it with others who might benefit.*
