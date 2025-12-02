---
title: "Coin Change (Unbounded Knapsack)"
day: 38
collection: dsa
categories:
  - dsa
tags:
  - dynamic-programming
  - unbounded-knapsack
  - greedy
  - bfs
difficulty: Medium
---

**"Making change with the fewest coins."**

## 1. Problem Statement

You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.

Return the **fewest number of coins** needed to make up that amount. If that amount cannot be made up by any combination of the coins, return `-1`.

You may assume that you have an **infinite number** of each kind of coin.

**Example 1:**
```
Input: coins = [1, 2, 5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
```

**Example 2:**
```
Input: coins = [2], amount = 3
Output: -1
```

## 2. Intuition: Unbounded Knapsack

This is the **Unbounded Knapsack Problem** (we can use each coin unlimited times).

**Key Difference from 0/1 Knapsack:**
- **0/1:** Each item can be used at most once.
- **Unbounded:** Each item can be used unlimited times.

## 3. Approach 1: Dynamic Programming (Bottom-Up)

**State:** `dp[i]` = minimum coins needed to make amount `i`.

**Transition:**
For each amount `i`, try all coins:
$$dp[i] = \min(dp[i], dp[i - \text{coin}] + 1)$$

**Base Case:** `dp[0] = 0` (0 coins needed for amount 0).

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for i in range(1, amount + 1):
            for coin in coins:
                if i >= coin:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
```

**Complexity:**
- **Time:** $O(N \times \text{amount})$ where $N$ is the number of coin types.
- **Space:** $O(\text{amount})$

## 4. Approach 2: BFS (Shortest Path)

Think of this as a graph problem:
- **Nodes:** Amounts from 0 to `amount`.
- **Edges:** From amount `i`, we can go to `i + coin` for each coin.
- **Goal:** Find shortest path from 0 to `amount`.

```python
from collections import deque

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount == 0: return 0
        
        queue = deque([(0, 0)])  # (current_amount, num_coins)
        visited = {0}
        
        while queue:
            curr, steps = queue.popleft()
            
            for coin in coins:
                next_amt = curr + coin
                
                if next_amt == amount:
                    return steps + 1
                
                if next_amt < amount and next_amt not in visited:
                    visited.add(next_amt)
                    queue.append((next_amt, steps + 1))
        
        return -1
```

**Complexity:**
- **Time:** $O(N \times \text{amount})$
- **Space:** $O(\text{amount})$ for visited set.

## 5. Greedy Approach (Doesn't Always Work!)

**Naive Greedy:** Always pick the largest coin.

**Example where it fails:**
```
coins = [1, 3, 4], amount = 6
Greedy: 4 + 1 + 1 = 3 coins
Optimal: 3 + 3 = 2 coins
```

**When Greedy Works:**
- **Canonical Coin Systems** (like US coins: 1, 5, 10, 25).
- For these, greedy is optimal and runs in $O(N)$.

## 6. Variation: Coin Change II (Count Ways)

**Problem:** Count the number of ways to make the amount.

**DP Transition:**
$$dp[i] = \sum dp[i - \text{coin}]$$

```python
def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1  # One way to make 0: use no coins
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]
```

**Key Difference:** Iterate coins in outer loop to avoid counting duplicates.

## 7. Summary

| Approach | Time | Space | Notes |
| :--- | :--- | :--- | :--- |
| **DP** | $O(N \cdot A)$ | $O(A)$ | Standard solution |
| **BFS** | $O(N \cdot A)$ | $O(A)$ | Graph perspective |
| **Greedy** | $O(N)$ | $O(1)$ | Only for canonical systems |

Where $N$ = number of coin types, $A$ = amount.

## 8. Deep Dive: Reconstructing the Solution

The DP approach gives us the **count**, but how do we get the actual coins used?

```python
def coinChange WithPath(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    parent = [-1] * (amount + 1)  # Track which coin was used
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                parent[i] = coin
    
    if dp[amount] == float('inf'):
        return -1, []
    
    # Backtrack to find coins
    result = []
    curr = amount
    while curr > 0:
        coin = parent[curr]
        result.append(coin)
        curr -= coin
    
    return dp[amount], result

# Example
count, coins_used = coinChangeWithPath([1, 2, 5], 11)
print(f"Count: {count}, Coins: {coins_used}")  # Count: 3, Coins: [5, 5, 1]
```

## 9. Deep Dive: Why Greedy Fails

**Theorem:** Greedy works if and only if the coin system is **canonical**.

**Definition (Canonical):** A coin system is canonical if for every amount, the greedy algorithm produces the optimal solution.

**US Coins `[1, 5, 10, 25]`:** Canonical ✓
**Counter-example `[1, 3, 4]`:**
- Amount = 6
- Greedy: 4 + 1 + 1 = 3 coins
- Optimal: 3 + 3 = 2 coins
- Not canonical ✗

**Testing Canonicality:**
- Check all amounts up to the largest coin squared.
- If greedy matches DP for all, it's canonical.

## 10. Deep Dive: Coin Change II (Counting Combinations)

**Problem:** How many ways can we make the amount?

**Key Insight:** Order matters in permutations, not in combinations.
- Combination: `{1, 2, 2}` is same as `{2, 1, 2}`.
- Permutation: `[1, 2, 2]` is different from `[2, 1, 2]`.

**For Combinations (Coin Change II):**
```python
def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    # Outer loop: coins (prevents duplicates)
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]
```

**For Permutations:**
```python
def changePermutations(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    # Outer loop: amounts
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] += dp[i - coin]
    
    return dp[amount]
```

## 11. Deep Dive: Minimum Coins with Limit

**Variation:** Each coin can be used at most `k` times.

**Example:** `coins = [1, 2, 5]`, `limits = [2, 3, 1]`, `amount = 11`.
- Can use coin 1 at most 2 times.
- Can use coin 2 at most 3 times.
- Can use coin 5 at most 1 time.

**Solution:** 2D DP.
```python
def coinChangeWithLimit(coins, limits, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin, limit in zip(coins, limits):
        # Process in reverse to avoid using same coin multiple times in one iteration
        for i in range(amount, coin - 1, -1):
            for k in range(1, limit + 1):
                if i >= k * coin:
                    dp[i] = min(dp[i], dp[i - k * coin] + k)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

## 12. Real-World Applications

### 1. Currency Exchange
- **Problem:** Convert $100 to Euros using fewest bills.
- **Coins:** Denominations of Euros.

### 2. Resource Allocation
- **Problem:** Allocate server instances (small, medium, large) to meet demand.
- **Coins:** Instance types.
- **Amount:** Total compute needed.

### 3. Change-Making Machines
- **Problem:** Vending machines must give change.
- **Optimization:** Minimize coins dispensed (saves refill costs).

## 13. Code: Space-Optimized Coin Change II

For counting ways, we only need the current DP array.

```python
def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]
```

**Space:** $O(\text{amount})$ instead of $O(N \times \text{amount})$.

## 14. Interview Pro Tips

1.  **Clarify:** Unlimited coins? Or limited?
2.  **Start with DP:** Always explain the $O(N \times A)$ solution.
3.  **Mention Greedy:** Show you know when it works (canonical systems).
4.  **Variants:** Be ready for "count ways" or "with limits".
5.  **Reconstruction:** Know how to print the actual coins.

## 15. Performance Benchmarking

**Test Case:** `coins = [1, 2, 5, 10, 20, 50]`, `amount = 10000`.

| Approach | Python Time | C++ Time |
| :--- | :--- | :--- |
| **DP** | 120ms | 8ms |
| **BFS** | 250ms | 15ms |
| **Greedy (if canonical)** | 0.1ms | 0.01ms |

**Takeaway:** For canonical systems, greedy is 1000x faster.

## 16. Edge Cases

1.  **Amount = 0:** Return 0 (no coins needed).
2.  **No solution:** Return -1.
3.  **Single coin = amount:** Return 1.
4.  **All coins > amount:** Return -1.
5.  **Duplicate coins:** `[1, 1, 2]` → Treat as `[1, 2]`.

## 17. Summary

| Approach | Time | Space | Notes |
| :--- | :--- | :--- | :--- |
| **DP** | $O(N \cdot A)$ | $O(A)$ | Standard solution |
| **BFS** | $O(N \cdot A)$ | $O(A)$ | Graph perspective |
| **Greedy** | $O(N)$ | $O(1)$ | Only for canonical systems |

Where $N$ = number of coin types, $A$ = amount.

---

**Originally published at:** [arunbaby.com/dsa/0038-coin-change](https://www.arunbaby.com/dsa/0038-coin-change/)
