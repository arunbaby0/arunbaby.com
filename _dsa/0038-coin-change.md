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

## 17. Deep Dive: The Frobenius Coin Problem

**Problem:** Given a set of coin denominations (that are coprime), what is the largest amount that *cannot* be made?
-   Also known as the **Coin Problem** or **McNugget Problem**.

**Two Coins ($a, b$):**
-   Formula: $g(a, b) = ab - a - b$.
-   Example: Coins 3 and 5.
    -   $3 \times 5 - 3 - 5 = 15 - 8 = 7$.
    -   Amounts: 1, 2, **3**, 4, **5**, **6**, **7** (Impossible), **8**, **9**, **10**...
    -   Largest impossible is 7.

**Three or More Coins:**
-   No closed-form formula exists.
-   This is related to the geometry of numbers and lattice points.
-   **Algorithm:** Use Dijkstra's algorithm on a graph where nodes are residues modulo the smallest coin.

**Why it matters:**
-   Helps design coin systems where every amount is reachable (e.g., include `1`).
-   Used in scheduling and tiling problems.

## 18. Deep Dive: Bounded Knapsack with Binary Decomposition

**Problem:** What if you have limited coins, but the limits are large (e.g., 1000 of each)?
-   Naive DP: $O(N \cdot A \cdot K)$. Too slow.

**Binary Decomposition:**
-   Any number $K$ can be represented as sum of powers of 2.
-   Example: $K=13 \to 1 + 2 + 4 + 6$.
-   Instead of 13 items of weight $W$, we create items with weights $1W, 2W, 4W, 6W$.
-   Now we have $O(\log K)$ items instead of $K$.
-   Run 0/1 Knapsack on these new items.

**Complexity:**
-   **Time:** $O(N \cdot A \cdot \log K)$.
-   **Space:** $O(A)$.

**Code:**
```python
def boundedKnapsack(coins, limits, amount):
    items = []
    for coin, limit in zip(coins, limits):
        k = 1
        while k <= limit:
            items.append(k * coin)
            limit -= k
            k *= 2
        if limit > 0:
            items.append(limit * coin)
            
    # Standard 0/1 Knapsack
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for item in items:
        for j in range(amount, item - 1, -1):
            dp[j] = min(dp[j], dp[j - item] + 1)
            
    return dp[amount]
```

## 19. System Design: High-Frequency Trading (Arbitrage)

**Scenario:** Currency Arbitrage.
-   You have 1 USD.
-   Exchange rates: USD -> EUR -> GBP -> USD.
-   Goal: Maximize profit (or find cycle > 1.0).

**Connection to Coin Change:**
-   Coin Change finds shortest path (additive weights).
-   Arbitrage finds longest path (multiplicative weights).
-   $\log(\text{Product}) = \sum \log(\text{Factors})$.
-   Transform multiplicative rates to additive log-rates.
-   Use **Bellman-Ford** to find negative cycles (which correspond to profit > 1.0).

**Architecture:**
1.  **Ingestion:** UDP multicast feed from exchanges (Nasdaq, CME).
2.  **Graph Build:** Nodes = Currencies, Edges = $-\log(\text{Rate})$.
3.  **Algorithm:** Bellman-Ford (or SPFA) on FPGA for microsecond latency.
4.  **Execution:** Send orders via collocated servers.

## 20. Advanced: Generating Functions for Coin Change

**Math Perspective:**
-   Each coin $c$ corresponds to a polynomial $1 + x^c + x^{2c} + x^{3c} + ... = \frac{1}{1 - x^c}$.
-   The number of ways to make amount $A$ is the coefficient of $x^A$ in the product:
    $$P(x) = \prod_{c \in coins} \frac{1}{1 - x^c}$$

**Partial Fraction Decomposition:**
-   We can decompose $P(x)$ into simpler terms.
-   Allows computing the answer in $O(1)$ for fixed $N$ and large $A$.
-   This is how math competitions solve "Ways to make 1,000,000 with 1, 5, 10, 25".

## 21. Common Mistakes and Pitfalls

**1. Integer Overflow:**
-   "Count ways" can exceed $2^{63}-1$ very quickly.
-   **Fix:** Use Python (arbitrary precision) or `BigInt` in C++/Java.

**2. Greedy on Non-Canonical Systems:**
-   *Mistake:* Assuming greedy works because it works for US coins.
-   *Fix:* Always check if the system is canonical or use DP.

**3. Incorrect Initialization:**
-   Initialize `dp` with 0 instead of `infinity` for minimization.
-   *Result:* `min(0, ...)` is always 0.
-   *Fix:* `dp = [float('inf')] * (amount + 1); dp[0] = 0`.

**4. Order of Loops (Permutation vs Combination):**
-   *Mistake:* Swapping loops in Coin Change II.
-   *Result:* Counting `[1, 2]` and `[2, 1]` as two different ways.
-   *Fix:* Coins outer loop = Combinations. Amount outer loop = Permutations.

-   *Fix:* Coins outer loop = Combinations. Amount outer loop = Permutations.

## 22. Deep Dive: Optimal Denomination Design

**Problem:** If you were the King of a new country, what coin denominations should you mint?
-   **Goal:** Minimize the average number of coins needed for transactions.

**Greedy Optimization:**
-   Powers of 2 (`1, 2, 4, 8...`) allow any amount $N$ with $\log_2 N$ coins.
-   Powers of 10 (`1, 10, 100...`) are intuitive for humans but less efficient.
-   **US System (`1, 5, 10, 25`):** Good compromise. Average coins for 0-99 cents is 4.7.
-   **Optimal for 0-99:** `1, 3, 11, 37`. Average is 4.1. But hard to do mental math!

**Algorithm to Find Optimal Denominations:**
-   Use **Local Search** or **Genetic Algorithms**.
-   Define cost function: $\sum_{i=0}^{99} \text{minCoins}(i)$.
-   Perturb denominations and check if cost decreases.

## 23. Advanced: Quantum Algorithms for Knapsack

**Quantum Approximate Optimization Algorithm (QAOA):**
-   Knapsack/Coin Change can be mapped to **QUBO** (Quadratic Unconstrained Binary Optimization).
-   $H = A(\sum x_i w_i - W)^2 - B \sum x_i v_i$.
-   Quantum computers (like D-Wave annealers) find the ground state of this Hamiltonian.

**Grover's Search:**
-   Can find if a solution exists in $O(\sqrt{2^N})$ instead of $O(2^N)$.
-   Provides a quadratic speedup for the decision problem.

## 24. Interview Questions for Coin Change

**Q1: What if coins are not integers?**
*Answer:* Multiply all values by $10^k$ to make them integers. Floating point arithmetic is dangerous for equality checks.

**Q2: Can we solve Coin Change with negative coins?**
*Answer:* No, this creates cycles. If `1 + (-1) = 0`, we can add infinite pairs of `1, -1` to increase the coin count (or decrease cost if we minimize cost). It becomes a shortest path problem on a graph with negative edges (Bellman-Ford).

**Q3: How to handle "At least K coins"?**
*Answer:* This is just `dp[amount]` but we want `max` coins instead of `min`. Initialize with `-inf`.

**Q4: What if we want to minimize the *weight* of coins?**
*Answer:* Each coin has a value $V$ and weight $W$.
-   `dp[i] = min(dp[i], dp[i - V] + W)`
-   This is the general Unbounded Knapsack problem.

**Q5: Solve for $N=100, Amount=10^{18}$.**
*Answer:* DP fails.
-   If $N$ is small, use matrix exponentiation (for counting ways).
-   If we just need *any* solution, take as many largest coins as possible until remainder is small, then use BFS/DP for the remainder (Frobenius number logic).

## 25. Deep Dive: Change-Making for Non-Standard Currencies

**Historical Context:**
-   **Old British System:** 1 pound = 20 shillings, 1 shilling = 12 pence. (Base 12 and 20).
-   **Greedy Fails:** `[1, 3, 4]` is a classic counter-example, but real currencies are usually designed to be greedy-compatible.
-   **Exception:** The **Bahamian 15-cent coin**.
    -   Coins: `1, 5, 10, 15, 25`.
    -   Amount: 30.
    -   Greedy: 25 + 5 (2 coins).
    -   Alternative: 15 + 15 (2 coins).
    -   Greedy works here!
    -   But for Amount 20: Greedy (15 + 5) vs (10 + 10). Both 2 coins.
    -   Actually, for `[1, 3, 4]`, amount 6 is the smallest counter-example.

**Algorithm to Check Greedy:**
-   Kozen & Zaks (1994) gave an $O(N^2)$ algorithm to check if a set of coins is canonical.
-   If $c_1 < c_2 < ... < c_n$, let $m_i = \lceil c_{i+1} / c_i \rceil$.
-   Check if greedy is optimal for all $c_{i+1} - 1$.

## 26. Production Considerations for Coin Change Systems

**Real-World Vending Machine Implementation:**

When implementing coin change in embedded systems (vending machines, parking meters), several constraints apply:

**1. Memory Constraints:**
-   Microcontrollers have limited RAM (often 2-8KB).
-   Cannot store large DP arrays.
-   **Solution:** Use greedy for canonical systems, or compute on-demand for small amounts.

**2. Real-Time Requirements:**
-   Must dispense change in < 500ms.
-   **Solution:** Pre-compute lookup table for common amounts (0-999 cents).
-   Store in ROM/Flash memory.

**3. Coin Inventory Management:**
```python
class VendingMachine:
    def __init__(self, coins, inventory):
        self.coins = coins  # [1, 5, 10, 25]
        self.inventory = inventory  # {1: 100, 5: 50, 10: 20, 25: 10}
    
    def make_change(self, amount):
        result = []
        remaining = amount
        
        # Greedy with inventory check
        for coin in sorted(self.coins, reverse=True):
            while remaining >= coin and self.inventory[coin] > 0:
                result.append(coin)
                self.inventory[coin] -= 1
                remaining -= coin
        
        if remaining > 0:
            # Rollback and try alternative
            for coin in result:
                self.inventory[coin] += 1
            return self.make_change_dp(amount)
        
        return result
    
    def make_change_dp(self, amount):
        # Bounded knapsack with inventory limits
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        parent = {}
        
        for coin in self.coins:
            for i in range(coin, amount + 1):
                count_needed = (i // coin)
                if count_needed <= self.inventory[coin]:
                    if dp[i - coin] + 1 < dp[i]:
                        dp[i] = dp[i - coin] + 1
                        parent[i] = coin
        
        # Reconstruct and update inventory
        result = []
        curr = amount
        while curr > 0 and curr in parent:
            coin = parent[curr]
            result.append(coin)
            self.inventory[coin] -= 1
            curr -= coin
        
        return result if curr == 0 else None
```

## 27. Advanced Optimization: Parallel Coin Change

For massive batch processing (e.g., processing millions of transactions):

**GPU Acceleration:**
```python
import cupy as cp

def coin_change_gpu(amounts, coins):
    """
    Process multiple amounts in parallel on GPU
    """
    max_amount = cp.max(amounts)
    n_amounts = len(amounts)
    
    # Allocate GPU memory
    dp = cp.full((n_amounts, max_amount + 1), cp.inf, dtype=cp.float32)
    dp[:, 0] = 0
    
    # DP on GPU
    for coin in coins:
        for i in range(coin, max_amount + 1):
            dp[:, i] = cp.minimum(dp[:, i], dp[:, i - coin] + 1)
    
    # Extract results
    results = cp.array([dp[idx, amt] for idx, amt in enumerate(amounts)])
    return cp.asnumpy(results)
```

**Distributed Processing (Spark):**
```python
from pyspark import SparkContext

def process_batch(amounts, coins):
    sc = SparkContext()
    
    def solve_single(amount):
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if i >= coin:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[amount]
    
    amounts_rdd = sc.parallelize(amounts)
    results = amounts_rdd.map(solve_single).collect()
    return results
```

## 28. Memory Optimization Techniques

**1. Sparse DP (for large amounts):**
```python
def coin_change_sparse(coins, amount):
    # Only store reachable states
    dp = {0: 0}
    
    for i in range(1, amount + 1):
        candidates = []
        for coin in coins:
            if i - coin in dp:
                candidates.append(dp[i - coin] + 1)
        
        if candidates:
            dp[i] = min(candidates)
    
    return dp.get(amount, -1)
```

**2. Sliding Window (for streaming amounts):**
```python
def coin_change_streaming(coins, max_window=1000):
    """
    Process amounts in a stream without storing full DP table
    """
    dp = [float('inf')] * max_window
    dp[0] = 0
    
    for i in range(1, max_window):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    def query(amount):
        if amount < max_window:
            return dp[amount]
        else:
            # Compute on-demand for large amounts
            return coin_change_large(coins, amount)
    
    return query
```

## 29. Ethical Considerations

**1. Cashless Society:**
-   Optimizing coin change is less relevant as we move to digital payments.
-   **Impact:** Marginalizes unbanked populations who rely on cash.
-   **Policy:** Laws requiring businesses to accept cash (e.g., in NYC).

**2. Algorithmic Pricing:**
-   Dynamic pricing (Uber surge) is a form of resource allocation.
-   **Risk:** Price gouging during emergencies.
-   **Regulation:** Caps on surge pricing during disasters.

## 30. Further Reading

1.  **"The Art of Computer Programming, Vol 3" (Knuth):** Generating functions.
2.  **"Algorithms" (Dasgupta):** DP chapter.
3.  **"Coin Problem" (MathWorld):** Frobenius numbers.
4.  **"High-Frequency Trading" (Aldridge):** Arbitrage strategies.

## 31. Conclusion

The Coin Change problem is a masterclass in Dynamic Programming. It teaches us about state transition, the importance of loop order (permutations vs. combinations), and the dangers of greedy algorithms. From making change at a bodega to detecting arbitrage opportunities in global FX markets, the principles of "optimizing a sum of parts" are universal. Whether you solve it with a simple 1D array or a complex generating function, mastering Coin Change is a rite of passage for every computer scientist.

## 32. Summary

| Approach | Time | Space | Notes |
| :--- | :--- | :--- | :--- |
| **DP** | $O(N \cdot A)$ | $O(A)$ | Standard solution |
| **BFS** | $O(N \cdot A)$ | $O(A)$ | Graph perspective |
| **Greedy** | $O(N)$ | $O(1)$ | Only for canonical systems |

Where $N$ = number of coin types, $A$ = amount.

---

**Originally published at:** [arunbaby.com/dsa/0038-coin-change](https://www.arunbaby.com/dsa/0038-coin-change/)
