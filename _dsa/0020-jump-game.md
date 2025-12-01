---
title: "Jump Game"
day: 20
collection: dsa
categories:
  - dsa
tags:
  - greedy
  - array
  - dynamic-programming
  - optimization
  - medium
subdomain: "Greedy Algorithms"
tech_stack: [Python]
scale: "O(N) time, O(1) space"
companies: [Google, Meta, Amazon, Microsoft, Apple, Bloomberg]
related_ml_day: 20
related_speech_day: 20
---

**Master greedy decision-making to determine reachability—the same adaptive strategy used in online learning and real-time speech systems.**

## Problem Statement

You are given an integer array `nums`. You are initially positioned at the array's **first index**, and each element in the array represents your **maximum** jump length at that position.

Return `true` if you can reach the last index, or `false` otherwise.

### Examples

**Example 1:**
```
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

**Example 2:**
```
Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump 
length is 0, which makes it impossible to reach the last index.
```

**Example 3:**
```
Input: nums = [0]
Output: true
Explanation: Already at the last index.
```

### Constraints

- `1 <= nums.length <= 10^4`
- `0 <= nums[i] <= 10^5`

## Understanding the Problem

This is a **reachability problem** that teaches:
1. **Greedy optimization** - making locally optimal choices
2. **Forward thinking** - tracking maximum reachable position
3. **Early termination** - stopping as soon as we know the answer
4. **Adaptive decision-making** - updating strategy as we progress

### Key Insight

At each position `i`, we can jump to any index in the range `[i+1, i+nums[i]]`. The question is: can we build a path from index 0 to index `n-1`?

**Greedy observation:** We don't need to find the exact path—we just need to know if the last index is **reachable**.

We can track the **maximum index we can reach so far** and update it as we iterate through the array.

### Why This Problem Matters

1. **Greedy algorithms:** Learn when greedy choices lead to optimal solutions
2. **Reachability analysis:** Common in graph problems, state machines
3. **Optimization under constraints:** Maximize reach with limited jumps
4. **Real-world applications:**
   - Network routing (can packet reach destination?)
   - Resource allocation (can we complete all tasks?)
   - Game AI (can player win from current state?)
   - Online learning (can model adapt to new patterns?)

### The Adaptive Strategy Connection

| Jump Game | Online Learning | Adaptive Speech |
|-----------|-----------------|-----------------|
| Greedily track max reach | Adapt model to new data | Adapt to speaker/noise |
| Update strategy at each position | Update weights incrementally | Update acoustic model online |
| Forward-looking optimization | Look-ahead predictions | Predictive adaptation |
| Early termination if stuck | Early stopping if degrading | Fallback if quality drops |

All three use **adaptive, greedy strategies** to optimize in dynamic environments.

## Approach 1: Brute Force - Try All Paths

### Intuition

Use recursion/backtracking to try all possible paths from index 0 to index n-1.

### Implementation

```python
from typing import List

def canJump_bruteforce(nums: List[int]) -> bool:
    """
    Brute force: recursively try all paths.
    
    Time: O(2^N) - exponential, each position has multiple choices
    Space: O(N) - recursion depth
    
    Why this approach?
    - Shows the search space
    - Demonstrates need for optimization
    - Helps understand the problem structure
    
    Problem:
    - Extremely slow for large inputs
    - Explores redundant paths
    """
    def can_reach(position: int) -> bool:
        """Check if we can reach last index from position."""
        # Base case: reached the end
        if position >= len(nums) - 1:
            return True
        
        # Try all possible jumps from current position
        max_jump = nums[position]
        
        for jump in range(1, max_jump + 1):
            if can_reach(position + jump):
                return True
        
        return False
    
    return can_reach(0)


# Test
print(canJump_bruteforce([2,3,1,1,4]))  # True
print(canJump_bruteforce([3,2,1,0,4]))  # False
```

### Analysis

**Time Complexity: O(2^N)**
- At each position, we might have up to N choices
- Exponential branching factor

**Space Complexity: O(N)**
- Recursion depth

**For N=10,000:** Completely infeasible!

## Approach 2: Dynamic Programming (Top-Down with Memoization)

### Intuition

The brute force approach has **overlapping subproblems** - we might check if index `i` is reachable multiple times.

Use memoization to cache results.

### Implementation

```python
def canJump_dp_memo(nums: List[int]) -> bool:
    """
    DP with memoization.
    
    Time: O(N^2) - each position visited once, try up to N jumps
    Space: O(N) - memoization cache + recursion
    
    Optimization over brute force:
    - Cache results to avoid recomputation
    - Still explores many paths
    """
    memo = {}
    
    def can_reach(position: int) -> bool:
        # Check cache
        if position in memo:
            return memo[position]
        
        # Base case
        if position >= len(nums) - 1:
            return True
        
        # Try all jumps
        max_jump = nums[position]
        
        for jump in range(1, max_jump + 1):
            if can_reach(position + jump):
                memo[position] = True
                return True
        
        memo[position] = False
        return False
    
    return can_reach(0)
```

### Analysis

**Time Complexity: O(N^2)**
- N positions
- Each position tries up to N jumps
- Better than exponential, but still slow

**Space Complexity: O(N)**
- Memoization cache
- Recursion stack

## Approach 3: Dynamic Programming (Bottom-Up)

### Intuition

Build up from the end: mark each position as GOOD (can reach end) or BAD (cannot reach end).

### Implementation

```python
def canJump_dp_bottomup(nums: List[int]) -> bool:
    """
    Bottom-up DP.
    
    Time: O(N^2)
    Space: O(N)
    
    dp[i] = True if we can reach last index from position i
    """
    n = len(nums)
    dp = [False] * n
    dp[n-1] = True  # Last position can "reach" itself
    
    # Work backwards
    for i in range(n - 2, -1, -1):
        max_jump = nums[i]
        
        # Check if any position we can jump to is GOOD
        for jump in range(1, min(max_jump + 1, n - i)):
            if dp[i + jump]:
                dp[i] = True
                break
    
    return dp[0]
```

## Approach 4: Greedy (Optimal)

### The Key Insight

**We don't need to find the exact path—just whether the last index is reachable!**

**Greedy strategy:** Track the **maximum index we can reach so far** as we iterate left to right.

- At each position `i`, update `max_reach = max(max_reach, i + nums[i])`.
- If `max_reach >= n-1`, we can reach the end.
- If at any point `i > max_reach`, we're stuck (gap we can't cross).

This is **greedy** because we make the locally optimal choice at each step: extend our reach as far as possible.

### Implementation

```python
def canJump(nums: List[int]) -> bool:
    """
    Greedy solution - optimal.
    
    Time: O(N) - single pass
    Space: O(1) - only tracking max_reach
    
    Strategy:
    - Track the farthest position we can reach
    - Update it as we iterate
    - If we reach or pass the last index, return True
    - If current position exceeds max reachable, return False
    
    Why this works:
    - We only care about maximum reach, not the exact path
    - Greedy choice: always extend reach as far as possible
    - Early termination: stop as soon as we know the answer
    """
    if len(nums) <= 1:
        return True
    
    max_reach = 0  # Farthest index we can reach
    
    for i in range(len(nums)):
        # If current position is beyond our reach, we're stuck
        if i > max_reach:
            return False
        
        # Update maximum reachable position
        max_reach = max(max_reach, i + nums[i])
        
        # Early termination: if we can reach the end, done
        if max_reach >= len(nums) - 1:
            return True
    
    return max_reach >= len(nums) - 1
```

### Step-by-Step Visualization

**Example 1:** `nums = [2,3,1,1,4]`

```
Initial: max_reach = 0

i=0, nums[0]=2:
  i (0) <= max_reach (0) ✓
  max_reach = max(0, 0+2) = 2
  Can reach indices: [0, 1, 2]

i=1, nums[1]=3:
  i (1) <= max_reach (2) ✓
  max_reach = max(2, 1+3) = 4
  Can reach indices: [0, 1, 2, 3, 4] ✓ (includes last index 4)
  Return True
```

**Example 2:** `nums = [3,2,1,0,4]`

```
Initial: max_reach = 0

i=0, nums[0]=3:
  i (0) <= max_reach (0) ✓
  max_reach = max(0, 0+3) = 3
  Can reach: [0, 1, 2, 3]

i=1, nums[1]=2:
  i (1) <= max_reach (3) ✓
  max_reach = max(3, 1+2) = 3
  Can reach: [0, 1, 2, 3]

i=2, nums[2]=1:
  i (2) <= max_reach (3) ✓
  max_reach = max(3, 2+1) = 3
  Can reach: [0, 1, 2, 3]

i=3, nums[3]=0:
  i (3) <= max_reach (3) ✓
  max_reach = max(3, 3+0) = 3
  Can reach: [0, 1, 2, 3] (stuck at index 3!)

i=4, nums[4]=4:
  i (4) > max_reach (3) ✗
  We can't reach index 4
  Return False
```

### Why Greedy Works

**Proof sketch:**
1. If index `j` is reachable, then any index `k` where `k <= j + nums[j]` is also reachable.
2. Therefore, tracking the maximum reachable index is sufficient.
3. We never need to backtrack or try different paths.
4. The greedy choice (always extend reach maximally) guarantees we find the answer.

## Implementation: Production-Grade Solution

```python
from typing import List, Optional
import logging

class JumpGameSolver:
    """
    Production-ready Jump Game solver with multiple strategies.
    
    Features:
    - Input validation
    - Multiple algorithms
    - Performance metrics
    - Detailed logging
    """
    
    def __init__(self, strategy: str = "greedy"):
        """
        Initialize solver.
        
        Args:
            strategy: "greedy", "dp", or "bruteforce"
        """
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        self.iterations = 0
    
    def can_jump(self, nums: List[int]) -> bool:
        """
        Determine if last index is reachable.
        
        Args:
            nums: Array of jump lengths
            
        Returns:
            True if last index reachable, False otherwise
            
        Raises:
            ValueError: If input is invalid
        """
        # Validate input
        if not nums:
            raise ValueError("nums cannot be empty")
        
        if not all(isinstance(x, int) and x >= 0 for x in nums):
            raise ValueError("All elements must be non-negative integers")
        
        # Reset metrics
        self.iterations = 0
        
        # Choose strategy
        if self.strategy == "greedy":
            result = self._greedy(nums)
        elif self.strategy == "dp":
            result = self._dp_bottomup(nums)
        elif self.strategy == "bruteforce":
            result = self._bruteforce(nums)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.logger.info(
            f"Can jump: {result}, Strategy: {self.strategy}, "
            f"Iterations: {self.iterations}"
        )
        
        return result
    
    def _greedy(self, nums: List[int]) -> bool:
        """Greedy approach - optimal."""
        if len(nums) <= 1:
            return True
        
        max_reach = 0
        
        for i in range(len(nums)):
            self.iterations += 1
            
            if i > max_reach:
                return False
            
            max_reach = max(max_reach, i + nums[i])
            
            if max_reach >= len(nums) - 1:
                return True
        
        return max_reach >= len(nums) - 1
    
    def _dp_bottomup(self, nums: List[int]) -> bool:
        """DP approach."""
        n = len(nums)
        dp = [False] * n
        dp[n-1] = True
        
        for i in range(n - 2, -1, -1):
            self.iterations += 1
            max_jump = nums[i]
            
            for jump in range(1, min(max_jump + 1, n - i)):
                if dp[i + jump]:
                    dp[i] = True
                    break
        
        return dp[0]
    
    def _bruteforce(self, nums: List[int]) -> bool:
        """Brute force recursive approach."""
        def can_reach(position: int) -> bool:
            self.iterations += 1
            
            if position >= len(nums) - 1:
                return True
            
            max_jump = nums[position]
            
            for jump in range(1, max_jump + 1):
                if can_reach(position + jump):
                    return True
            
            return False
        
        return can_reach(0)
    
    def find_path(self, nums: List[int]) -> Optional[List[int]]:
        """
        Find an actual path to the last index (if exists).
        
        Returns:
            List of indices representing the path, or None if impossible
        """
        if not self.can_jump(nums):
            return None
        
        n = len(nums)
        if n == 1:
            return [0]
        
        # Use greedy to find a path
        path = [0]
        current = 0
        
        while current < n - 1:
            max_jump = nums[current]
            
            # Greedy choice: jump to position that maximizes next reach
            best_next = current + 1
            best_reach = best_next + nums[best_next]
            
            for jump in range(1, max_jump + 1):
                next_pos = current + jump
                if next_pos >= n - 1:
                    path.append(n - 1)
                    return path
                
                next_reach = next_pos + nums[next_pos]
                if next_reach > best_reach:
                    best_reach = next_reach
                    best_next = next_pos
            
            path.append(best_next)
            current = best_next
        
        return path
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        return {
            "strategy": self.strategy,
            "iterations": self.iterations
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_cases = [
        [2,3,1,1,4],
        [3,2,1,0,4],
        [0],
        [1,2,3],
        [1,1,1,1],
    ]
    
    solver = JumpGameSolver(strategy="greedy")
    
    for nums in test_cases:
        print(f"\nInput: {nums}")
        result = solver.can_jump(nums)
        print(f"Can jump: {result}")
        
        if result:
            path = solver.find_path(nums)
            print(f"Path: {path}")
        
        print(f"Stats: {solver.get_stats()}")
```

## Testing

### Comprehensive Test Suite

```python
import pytest

class TestJumpGame:
    """Comprehensive test suite for Jump Game."""
    
    @pytest.fixture
    def solver(self):
        return JumpGameSolver(strategy="greedy")
    
    def test_basic_examples(self, solver):
        """Test basic examples from problem."""
        assert solver.can_jump([2,3,1,1,4]) == True
        assert solver.can_jump([3,2,1,0,4]) == False
        assert solver.can_jump([0]) == True
    
    def test_edge_cases(self, solver):
        """Test edge cases."""
        # Single element
        assert solver.can_jump([5]) == True
        
        # Two elements
        assert solver.can_jump([1,0]) == True
        assert solver.can_jump([0,1]) == False
        
        # All zeros except first
        assert solver.can_jump([2,0,0]) == True
        assert solver.can_jump([1,0,0]) == False
    
    def test_always_reachable(self, solver):
        """Test cases where all jumps are >= 1."""
        assert solver.can_jump([1,1,1,1,1]) == True
        assert solver.can_jump([2,2,2,2,2]) == True
    
    def test_large_jumps(self, solver):
        """Test with very large jumps."""
        assert solver.can_jump([10000]) == True
        assert solver.can_jump([10000, 0, 0, 0, 0]) == True
    
    def test_barrier(self, solver):
        """Test with zero barrier."""
        assert solver.can_jump([1,0,1]) == False
        assert solver.can_jump([2,0,1]) == True
        assert solver.can_jump([1,1,0,1]) == True
    
    def test_strategy_equivalence(self):
        """Test that all strategies give same results."""
        test_cases = [
            [2,3,1,1,4],
            [3,2,1,0,4],
            [0],
            [1,2,3],
        ]
        
        for nums in test_cases:
            greedy_result = JumpGameSolver("greedy").can_jump(nums)
            dp_result = JumpGameSolver("dp").can_jump(nums)
            
            assert greedy_result == dp_result
    
    def test_find_path(self, solver):
        """Test path finding."""
        nums = [2,3,1,1,4]
        path = solver.find_path(nums)
        
        assert path is not None
        assert path[0] == 0
        assert path[-1] == len(nums) - 1
        
        # Verify path is valid
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            max_jump = nums[current]
            assert next_pos <= current + max_jump
    
    def test_no_path(self, solver):
        """Test when no path exists."""
        nums = [3,2,1,0,4]
        path = solver.find_path(nums)
        assert path is None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Complexity Analysis

### Greedy Approach (Optimal)

**Time Complexity: O(N)**
- Single pass through the array
- Each position visited once
- O(1) work per position

**Space Complexity: O(1)**
- Only a few variables: `max_reach`, loop index
- No additional data structures

### Comparison

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Brute Force | O(2^N) | O(N) | Too slow |
| DP (Memo) | O(N^2) | O(N) | Correct but slow |
| DP (Bottom-up) | O(N^2) | O(N) | Correct but slow |
| Greedy | O(N) | O(1) | **Optimal** |

## Production Considerations

### 1. Handling Very Large Arrays

For arrays that don't fit in memory:

```python
def canJump_streaming(nums_iterator):
    """
    Check if jump is possible with streaming input.
    
    Useful when nums is too large to load at once.
    """
    max_reach = 0
    position = 0
    
    for num in nums_iterator:
        if position > max_reach:
            return False
        
        max_reach = max(max_reach, position + num)
        position += 1
    
    return max_reach >= position - 1
```

### 2. Minimum Jumps (Extension)

If reachable, what's the minimum number of jumps?

```python
def minJumps(nums: List[int]) -> int:
    """
    Find minimum number of jumps to reach last index.
    
    Time: O(N)
    Space: O(1)
    
    Greedy approach:
    - Track current jump range
    - Count jumps when forced to jump again
    """
    if len(nums) <= 1:
        return 0
    
    jumps = 0
    current_end = 0
    farthest = 0
    
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        
        # If we've reached the end of current jump range
        if i == current_end:
            jumps += 1
            current_end = farthest
            
            # Early termination
            if current_end >= len(nums) - 1:
                break
    
    return jumps if current_end >= len(nums) - 1 else -1
```

### 3. Validation and Error Handling

```python
class JumpValidator:
    """Validate jump game inputs and scenarios."""
    
    @staticmethod
    def validate_input(nums: List[int]) -> tuple[bool, Optional[str]]:
        """
        Validate input array.
        
        Returns:
            (is_valid, error_message)
        """
        if not nums:
            return False, "Array cannot be empty"
        
        if len(nums) > 10**4:
            return False, f"Array too large: {len(nums)} (max 10^4)"
        
        for i, num in enumerate(nums):
            if not isinstance(num, int):
                return False, f"Invalid type at index {i}: {type(num)}"
            
            if num < 0:
                return False, f"Negative value at index {i}: {num}"
            
            if num > 10**5:
                return False, f"Value too large at index {i}: {num}"
        
        return True, None
```

### 4. Performance Monitoring

```python
import time
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Track performance metrics."""
    execution_time_ms: float
    iterations: int
    input_size: int
    result: bool
    
    @property
    def iterations_per_element(self) -> float:
        return self.iterations / self.input_size if self.input_size > 0 else 0
    
    def __str__(self) -> str:
        return (
            f"Performance Metrics:\n"
            f"  Execution time: {self.execution_time_ms:.3f}ms\n"
            f"  Iterations: {self.iterations}\n"
            f"  Input size: {self.input_size}\n"
            f"  Iterations/element: {self.iterations_per_element:.2f}\n"
            f"  Result: {self.result}"
        )


def canJump_with_metrics(nums: List[int]) -> tuple[bool, PerformanceMetrics]:
    """Solve problem and return metrics."""
    start = time.perf_counter()
    iterations = 0
    
    max_reach = 0
    
    for i in range(len(nums)):
        iterations += 1
        
        if i > max_reach:
            result = False
            break
        
        max_reach = max(max_reach, i + nums[i])
        
        if max_reach >= len(nums) - 1:
            result = True
            break
    else:
        result = max_reach >= len(nums) - 1
    
    execution_time = (time.perf_counter() - start) * 1000
    
    metrics = PerformanceMetrics(
        execution_time_ms=execution_time,
        iterations=iterations,
        input_size=len(nums),
        result=result
    )
    
    return result, metrics
```

## Connections to ML Systems

The **greedy decision-making and adaptive strategy** from this problem applies directly to online learning and adaptive systems:

### 1. Online Learning Systems

**Similarity to Jump Game:**
- **Jump Game:** Greedily extend reach at each position
- **Online Learning:** Greedily update model with each new data point

```python
class OnlineLearner:
    """
    Online learning with greedy updates.
    
    Similar to Jump Game:
    - Process data sequentially (like iterating through array)
    - Make greedy decisions at each step (like updating max_reach)
    - Adapt to new information (like extending reach)
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.weights = None
        self.learning_rate = learning_rate
        self.performance_history = []
    
    def update(self, features, label):
        """
        Greedy update with new sample.
        
        Like Jump Game's greedy reach extension:
        - Each new sample updates our 'reach' (model capability)
        - Greedy choice: always move towards better performance
        """
        if self.weights is None:
            self.weights = np.zeros(len(features))
        
        # Predict
        prediction = np.dot(self.weights, features)
        
        # Compute error
        error = label - prediction
        
        # Greedy update: move weights to reduce error
        self.weights += self.learning_rate * error * features
        
        # Track 'reach' (model performance)
        self.performance_history.append(abs(error))
    
    def can_converge(self, threshold: float = 0.01) -> bool:
        """
        Check if model is converging.
        
        Like checking if we can 'reach' the goal (low error).
        """
        if len(self.performance_history) < 10:
            return False
        
        recent_errors = self.performance_history[-10:]
        avg_error = np.mean(recent_errors)
        
        return avg_error < threshold
```

### 2. Adaptive Thresholding

In real-time systems, we often need to decide: can we meet SLA with current resources?

```python
class AdaptiveThresholdManager:
    """
    Manage adaptive thresholds based on performance.
    
    Similar to Jump Game's reachability check.
    """
    
    def __init__(self, target_latency_ms: float = 100.0):
        self.target_latency = target_latency_ms
        self.current_load = 0
        self.max_capacity = 100
    
    def can_handle_request(self, request_cost: int) -> bool:
        """
        Greedy check: can we handle this request?
        
        Like Jump Game: can we 'reach' a state where this request completes?
        """
        # Current position = current_load
        # Jump = request_cost
        # Goal = stay under max_capacity
        
        potential_load = self.current_load + request_cost
        
        if potential_load > self.max_capacity:
            return False  # Can't reach goal
        
        # Greedy decision: accept request
        self.current_load = potential_load
        return True
    
    def release_resources(self, amount: int):
        """Release resources (like moving forward in jump game)."""
        self.current_load = max(0, self.current_load - amount)
```

### 3. Checkpoint Reachability

In distributed training, we need to determine: can we reach next checkpoint before timeout/failure?

```python
def can_reach_checkpoint(
    current_step: int,
    checkpoint_step: int,
    steps_per_second: float,
    time_remaining_sec: float
) -> bool:
    """
    Check if we can reach checkpoint in time.
    
    Similar to Jump Game:
    - Current position = current_step
    - Goal = checkpoint_step
    - Constraint = time_remaining
    - 'Jump length' = steps we can complete in time
    """
    steps_needed = checkpoint_step - current_step
    steps_possible = int(steps_per_second * time_remaining_sec)
    
    return steps_possible >= steps_needed
```

### Key Parallels

| Jump Game | Online Learning | Adaptive Systems |
|-----------|-----------------|------------------|
| Track max reach | Track model performance | Track system capacity |
| Greedy extension | Greedy weight updates | Greedy resource allocation |
| Early termination | Early stopping | Circuit breaking |
| Forward-looking | Look-ahead prediction | Predictive scaling |

## Interview Strategy

### How to Approach This in an Interview

**1. Clarify (1 min)**
```
- Can nums contain zeros? (Yes)
- Can nums be empty? (No, constraint says length >= 1)
- Jump length is maximum, not fixed? (Yes, can jump 1 to nums[i])
```

**2. Explain Intuition (2 min)**
```
"I'll track the farthest position I can reach as I iterate through
the array. At each position, I update the maximum reach based on
current position + jump length. If I ever encounter a position
beyond my reach, I'm stuck. Otherwise, if max reach includes the
last index, return true."
```

**3. Discuss Approaches (2 min)**
```
1. Brute force: Try all paths - O(2^N), too slow
2. DP: Track reachability for each position - O(N^2)
3. Greedy: Track max reach - O(N), optimal

I'll implement the greedy approach.
```

**4. Code (8-10 min)**
- Clear variable names
- Handle edge cases
- Add comments

**5. Test (3 min)**
- Walk through Example 1 and 2
- Test edge case: single element

**6. Optimize (2 min)**
- Already optimal!
- Discuss early termination

### Common Mistakes

1. **Checking every reachable position instead of just max:**
   ```python
   # Inefficient
   for i in range(len(nums)):
       for j in range(i+1, min(i+nums[i]+1, len(nums))):
           # Check each position
   
   # Efficient
   max_reach = max(max_reach, i + nums[i])
   ```

2. **Not checking if current position is reachable:**
   ```python
   # Wrong: might process unreachable positions
   for i in range(len(nums)):
       max_reach = max(max_reach, i + nums[i])
   
   # Correct: check if i is reachable
   for i in range(len(nums)):
       if i > max_reach:
           return False
       max_reach = max(max_reach, i + nums[i])
   ```

3. **Forgetting early termination:**
   - Can return True as soon as `max_reach >= len(nums) - 1`

4. **Off-by-one errors:**
   - Last index is `len(nums) - 1`, not `len(nums)`

### Follow-up Questions

**Q1: Find minimum number of jumps?**

See `minJumps` implementation above.

**Q2: Can you jump backwards?**

Different problem—need to explore all paths (DP or BFS).

**Q3: What if each element is a cost, and you want minimum cost path?**

Dijkstra's algorithm or DP with cost accumulation.

**Q4: What if there are multiple goals (reachable indices)?**

Modify to track all reachable positions, use a set or boolean array.

## Additional Practice & Variants

### 1. Jump Game II (LeetCode 45)

**Problem:** Find the **minimum** number of jumps to reach the last index (guaranteed reachable).

```python
def jump(nums: List[int]) -> int:
    """
    Minimum jumps to reach end.
    
    Greedy approach:
    - Track current jump's max reach
    - When we must jump again, increment counter
    
    Time: O(N), Space: O(1)
    """
    if len(nums) <= 1:
        return 0
    
    jumps = 0
    current_end = 0
    farthest = 0
    
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        
        if i == current_end:
            jumps += 1
            current_end = farthest
    
    return jumps
```

### 2. Jump Game III (LeetCode 1306)

**Problem:** Given array `arr` and start index, you can jump to `i + arr[i]` or `i - arr[i]`. Can you reach any index with value 0?

```python
def canReach(arr: List[int], start: int) -> bool:
    """
    Can reach index with value 0 (bidirectional jumps).
    
    Use BFS or DFS since we can jump both directions.
    """
    visited = set()
    queue = [start]
    
    while queue:
        pos = queue.pop(0)
        
        if arr[pos] == 0:
            return True
        
        if pos in visited:
            continue
        
        visited.add(pos)
        
        # Try both jumps
        for next_pos in [pos + arr[pos], pos - arr[pos]]:
            if 0 <= next_pos < len(arr) and next_pos not in visited:
                queue.append(next_pos)
    
    return False
```

### 3. Jump Game IV (LeetCode 1345)

**Problem:** Can jump to `i-1`, `i+1`, or any index `j` where `arr[j] == arr[i]`. Find minimum jumps.

Uses BFS with value-based teleportation.

## Key Takeaways

✅ **Greedy approach is optimal** - track maximum reachable position

✅ **Single pass, O(1) space** - no need for DP or recursion

✅ **Early termination** - return as soon as we know the answer

✅ **Forward-looking strategy** - always extend reach maximally

✅ **Reachability problems** often have elegant greedy solutions

✅ **Same pattern in online learning** - greedy updates, adaptive strategies

✅ **Same pattern in adaptive systems** - greedy resource allocation, capacity checks

✅ **Testing critical** - edge cases (zeros, barriers, large jumps)

✅ **Extensions interesting** - minimum jumps, bidirectional, value-based teleportation

✅ **Production applications** - checkpoint reachability, SLA compliance, resource planning

### Mental Model

Think of this problem as:
- **Jump Game:** Greedy reach extension with early termination
- **Online Learning:** Greedy model updates with convergence checks
- **Adaptive Speech:** Greedy model adaptation with quality monitoring

All use the pattern: **make greedy decisions, adapt based on new information, terminate when goal is reached or unreachable.**

### Connection to Thematic Link: Greedy Decisions and Adaptive Strategies

All three Day 20 topics share **greedy, adaptive optimization**:

**DSA (Jump Game):**
- Greedy decision: extend max reach at each step
- Adaptive: update strategy based on current position
- Forward-looking: anticipate future reachability

**ML System Design (Online Learning Systems):**
- Greedy decision: update model with each new sample
- Adaptive: adjust to distribution shifts
- Forward-looking: predict future patterns

**Speech Tech (Adaptive Speech Models):**
- Greedy decision: update model based on recent audio
- Adaptive: adjust to speaker/noise/accent changes
- Forward-looking: anticipate user corrections

The **unifying principle**: make locally optimal, greedy decisions while adapting to new information—crucial for systems that must respond to changing conditions in real-time.

---

**Originally published at:** [arunbaby.com/dsa/0020-jump-game](https://www.arunbaby.com/dsa/0020-jump-game/)

*If you found this helpful, consider sharing it with others who might benefit.*




