---
title: "Maximum Subarray (Kadane's Algorithm)"
day: 5
collection: dsa
categories:
  - dsa
tags:
  - dynamic-programming
  - arrays
  - kadane-algorithm
topic: Dynamic Programming
difficulty: Medium
companies: [Google, Meta, Amazon, Microsoft, LinkedIn]
leetcode_link: "https://leetcode.com/problems/maximum-subarray/"
time_complexity: "O(n)"
space_complexity: "O(1)"
related_ml_day: 5
related_speech_day: 5
related_agents_day: 5
---

**Master the pattern behind online algorithms, streaming analytics, and dynamic programming, a single elegant idea powering countless production systems.**

## Problem

Given an integer array `nums`, find the subarray with the largest sum, and return its sum.

A **subarray** is a contiguous non-empty sequence of elements within an array.

**Example 1:**
```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.
```

**Example 2:**
```
Input: nums = [1]
Output: 1
Explanation: The subarray [1] has the largest sum 1.
```

**Example 3:**
```
Input: nums = [5,4,-1,7,8]
Output: 23
Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
```

**Constraints:**
- `1 <= nums.length <= 10^5`
- `-10^4 <= nums[i] <= 10^4`

---

## Intuition

**Key Insight:** At each position, decide whether to:
1. **Extend** the current subarray by including this element
2. **Start fresh** from this element

**Why this works:** If the sum up to the previous element is negative, it can only hurt the sum, better to start fresh.

This is **Kadane's Algorithm**: a classic example of greedy + dynamic programming.

---

## Approach 1: Brute Force (Not Optimal)

Try all possible subarrays.

### Implementation

```python
from typing import List

def maxSubArrayBruteForce(nums: List[int]) -> int:
    """
    Try all subarrays
    
    Time: O(n²)
    Space: O(1)
    """
    n = len(nums)
    max_sum = float('-inf')
    
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += nums[j]
            max_sum = max(max_sum, current_sum)
    
    return max_sum

# Example
print(maxSubArrayBruteForce([-2,1,-3,4,-1,2,1,-5,4]))  # 6
```

**Time Complexity:** O(n²)  
**Space Complexity:** O(1)

**Why it's bad:** For n = 100,000, this requires 10 billion operations, too slow for production.

---

## Approach 2: Kadane's Algorithm (Optimal)

Track the maximum sum ending at each position.

### Implementation

```python
from typing import List

def maxSubArray(nums: List[int]) -> int:
    """
    Kadane's Algorithm
    
    Time: O(n) - single pass
    Space: O(1) - two variables
    
    Algorithm:
    1. Track current_sum (max sum ending here)
    2. At each element: extend OR start fresh
    3. Track global max_sum
    """
    if not nums:
        return 0
    
    current_sum = nums[0]
    max_sum = nums[0]
    
    for i in range(1, len(nums)):
        # Key decision: extend OR start fresh
        current_sum = max(nums[i], current_sum + nums[i])
        
        # Update global maximum
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

### Detailed Walkthrough

```
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]

Initial:
  current_sum = -2
  max_sum = -2

i=1, nums[i]=1:
  current_sum = max(1, -2+1) = max(1, -1) = 1  (start fresh!)
  max_sum = max(-2, 1) = 1

i=2, nums[i]=-3:
  current_sum = max(-3, 1-3) = max(-3, -2) = -2  (extend, even though negative)
  max_sum = max(1, -2) = 1

i=3, nums[i]=4:
  current_sum = max(4, -2+4) = max(4, 2) = 4  (start fresh!)
  max_sum = max(1, 4) = 4

i=4, nums[i]=-1:
  current_sum = max(-1, 4-1) = max(-1, 3) = 3  (extend)
  max_sum = max(4, 3) = 4

i=5, nums[i]=2:
  current_sum = max(2, 3+2) = max(2, 5) = 5  (extend)
  max_sum = max(4, 5) = 5

i=6, nums[i]=1:
  current_sum = max(1, 5+1) = max(1, 6) = 6  (extend)
  max_sum = max(5, 6) = 6

i=7, nums[i]=-5:
  current_sum = max(-5, 6-5) = max(-5, 1) = 1  (extend)
  max_sum = max(6, 1) = 6

i=8, nums[i]=4:
  current_sum = max(4, 1+4) = max(4, 5) = 5  (extend)
  max_sum = max(6, 5) = 6

Final: max_sum = 6
Subarray: [4, -1, 2, 1]
```

### Why This Works

**Invariant:** `current_sum` always holds the maximum sum of a subarray ending at position `i`.

**Correctness:**
- If `current_sum < 0`, it can't help future elements → start fresh
- We consider every possible ending position
- `max_sum` tracks the best across all positions

**Greedy Choice:** At each step, make locally optimal decision (extend or start fresh).

---

## Approach 3: Dynamic Programming Formulation

View as a DP problem for deeper understanding.

### Formulation

**State:** `dp[i]` = maximum sum of subarray ending at index `i`

**Recurrence:**
```
dp[i] = max(nums[i], dp[i-1] + nums[i])
```

**Base case:** `dp[0] = nums[0]`

**Answer:** `max(dp[0], dp[1], ..., dp[n-1])`

### Implementation

```python
def maxSubArrayDP(nums: List[int]) -> int:
    """
    Explicit DP formulation
    
    Time: O(n)
    Space: O(n) → can optimize to O(1)
    """
    n = len(nums)
    if n == 0:
        return 0
    
    # DP table
    dp = [0] * n
    dp[0] = nums[0]
    
    # Fill table
    for i in range(1, n):
        dp[i] = max(nums[i], dp[i-1] + nums[i])
    
    # Answer is max of all dp values
    return max(dp)
```

**Optimization:** Since `dp[i]` only depends on `dp[i-1]`, we can use O(1) space → this becomes identical to Kadane's algorithm!

---

## Returning the Actual Subarray

Modify algorithm to track indices.

```python
def maxSubArrayWithIndices(nums: List[int]) -> tuple[int, int, int]:
    """
    Return (max_sum, start_index, end_index)
    """
    if not nums:
        return (0, -1, -1)
    
    current_sum = nums[0]
    max_sum = nums[0]
    
    # Track indices
    start = 0
    end = 0
    temp_start = 0
    
    for i in range(1, len(nums)):
        # If starting fresh, update temp_start
        if nums[i] > current_sum + nums[i]:
            current_sum = nums[i]
            temp_start = i
        else:
            current_sum = current_sum + nums[i]
        
        # Update global max
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
    
    return (max_sum, start, end)

# Usage
nums = [-2,1,-3,4,-1,2,1,-5,4]
max_sum, start, end = maxSubArrayWithIndices(nums)
print(f"Max sum: {max_sum}")
print(f"Subarray: nums[{start}:{end+1}] = {nums[start:end+1]}")
# Output:
# Max sum: 6
# Subarray: nums[3:7] = [4, -1, 2, 1]
```

---

## Edge Cases & Testing

### Edge Cases

```python
def test_edge_cases():
    # Single element
    assert maxSubArray([1]) == 1
    assert maxSubArray([-1]) == -1
    
    # All negative
    assert maxSubArray([-2, -3, -1, -4]) == -1
    
    # All positive
    assert maxSubArray([1, 2, 3, 4]) == 10
    
    # Mixed
    assert maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    
    # Alternating signs
    assert maxSubArray([5, -3, 5]) == 7
    
    # Zero in array
    assert maxSubArray([0, -3, 1, 1]) == 2
    
    # Large numbers
    assert maxSubArray([10000, -1, 10000]) == 19999
```

### Comprehensive Test Suite

```python
import unittest
from typing import List

class TestMaxSubArray(unittest.TestCase):
    
    def test_example1(self):
        self.assertEqual(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]), 6)
    
    def test_example2(self):
        self.assertEqual(maxSubArray([1]), 1)
    
    def test_example3(self):
        self.assertEqual(maxSubArray([5,4,-1,7,8]), 23)
    
    def test_all_negative(self):
        # When all negative, return the largest (least negative) element
        self.assertEqual(maxSubArray([-3, -2, -5, -1]), -1)
    
    def test_all_positive(self):
        # When all positive, sum is the entire array
        self.assertEqual(maxSubArray([1, 2, 3, 4, 5]), 15)
    
    def test_alternating(self):
        self.assertEqual(maxSubArray([1, -1, 1, -1, 1]), 1)
    
    def test_zeros(self):
        self.assertEqual(maxSubArray([0, 0, 0]), 0)
    
    def test_large_array(self):
        # Performance test: 100k elements
        import random
        large = [random.randint(-100, 100) for _ in range(100000)]
        result = maxSubArray(large)  # Should complete quickly
        self.assertIsInstance(result, int)

if __name__ == '__main__':
    unittest.main()
```

---

## Variations

### Variation 1: Circular Array

Array is circular (can wrap around).

```python
def maxSubarraySumCircular(nums: List[int]) -> int:
    """
    Maximum sum in circular array
    
    Strategy:
    1. Max subarray not wrapping = standard Kadane's
    2. Max subarray wrapping = total_sum - min_subarray
    3. Return max of both
    
    Time: O(n)
    Space: O(1)
    """
    def kadane_max(arr):
        current = arr[0]
        maximum = arr[0]
        for i in range(1, len(arr)):
            current = max(arr[i], current + arr[i])
            maximum = max(maximum, current)
        return maximum
    
    def kadane_min(arr):
        current = arr[0]
        minimum = arr[0]
        for i in range(1, len(arr)):
            current = min(arr[i], current + arr[i])
            minimum = min(minimum, current)
        return minimum
    
    total_sum = sum(nums)
    
    # Case 1: Max subarray not wrapping
    max_kadane = kadane_max(nums)
    
    # Case 2: Max subarray wrapping
    # = total_sum - min_subarray
    min_kadane = kadane_min(nums)
    max_wrap = total_sum - min_kadane
    
    # Edge case: all elements negative
    if max_wrap == 0:
        return max_kadane
    
    return max(max_kadane, max_wrap)

# Example
print(maxSubarraySumCircular([5, -3, 5]))  # 10 (5 + 5, wrapping)
print(maxSubarraySumCircular([1, -2, 3, -2]))  # 3 (just [3])
```

### Variation 2: Maximum Product Subarray

Find subarray with maximum product instead of sum.

```python
def maxProduct(nums: List[int]) -> int:
    """
    Maximum product subarray
    
    Track both max and min (for handling negatives)
    
    Time: O(n)
    Space: O(1)
    """
    if not nums:
        return 0
    
    max_so_far = nums[0]
    min_so_far = nums[0]
    result = nums[0]
    
    for i in range(1, len(nums)):
        # If current number is negative, swap max and min
        if nums[i] < 0:
            max_so_far, min_so_far = min_so_far, max_so_far
        
        # Update max and min
        max_so_far = max(nums[i], max_so_far * nums[i])
        min_so_far = min(nums[i], min_so_far * nums[i])
        
        # Update result
        result = max(result, max_so_far)
    
    return result

# Example
print(maxProduct([2, 3, -2, 4]))  # 6 (subarray [2,3])
print(maxProduct([-2, 0, -1]))  # 0
```

---

## Connection to ML Systems

Kadane's algorithm pattern appears everywhere in ML:

### 1. Streaming Metrics

```python
class StreamingMetrics:
    """
    Track running statistics using Kadane-like pattern
    
    Use case: Monitor model performance in real-time
    """
    
    def __init__(self):
        self.current_window_sum = 0
        self.best_window_sum = float('-inf')
        self.window_start = 0
        self.best_window_start = 0
        self.best_window_end = 0
        self.position = 0
    
    def add_metric(self, value):
        """
        Add new metric value
        
        Tracks best performing window
        """
        # Kadane's pattern: extend or start fresh
        if self.current_window_sum < 0:
            self.current_window_sum = value
            self.window_start = self.position
        else:
            self.current_window_sum += value
        
        # Update best window
        if self.current_window_sum > self.best_window_sum:
            self.best_window_sum = self.current_window_sum
            self.best_window_start = self.window_start
            self.best_window_end = self.position
        
        self.position += 1
    
    def get_best_window(self):
        """Get indices of best performing window"""
        return {
            'sum': self.best_window_sum,
            'start': self.best_window_start,
            'end': self.best_window_end,
            'length': self.best_window_end - self.best_window_start + 1
        }

# Usage: Track model accuracy improvements
metrics = StreamingMetrics()

# Simulate daily accuracy changes
accuracy_deltas = [0.02, 0.01, -0.03, 0.05, 0.03, 0.01, -0.02, 0.04]

for delta in accuracy_deltas:
    metrics.add_metric(delta)

best = metrics.get_best_window()
print(f"Best improvement window: days {best['start']} to {best['end']}")
print(f"Total improvement: {best['sum']:.2f}")
```

### 2. A/B Test Analysis

```python
from typing import List

class ABTestWindowAnalyzer:
    """
    Find best time window for A/B test metric
    
    Use Kadane's to find period with max lift
    """
    
    def find_best_test_period(self, daily_lifts: List[float]) -> dict:
        """
        Find consecutive days with maximum cumulative lift
        
        Args:
            daily_lifts: Daily lift (treatment - control) metrics
        
        Returns:
            Best testing period details
        """
        if not daily_lifts:
            return None
        
        current_sum = daily_lifts[0]
        max_sum = daily_lifts[0]
        
        start = 0
        end = 0
        temp_start = 0
        
        for i in range(1, len(daily_lifts)):
            # Kadane's pattern
            if daily_lifts[i] > current_sum + daily_lifts[i]:
                current_sum = daily_lifts[i]
                temp_start = i
            else:
                current_sum += daily_lifts[i]
            
            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i
        
        return {
            'max_cumulative_lift': max_sum,
            'start_day': start,
            'end_day': end,
            'duration_days': end - start + 1,
            'average_daily_lift': max_sum / (end - start + 1)
        }

# Usage
analyzer = ABTestWindowAnalyzer()

# Daily conversion rate lift (treatment - control)
daily_lifts = [0.002, 0.001, -0.001, 0.005, 0.003, 0.002, -0.002, 0.004]

result = analyzer.find_best_test_period(daily_lifts)
print(f"Best test period: days {result['start_day']} to {result['end_day']}")
print(f"Cumulative lift: {result['max_cumulative_lift']:.4f}")
print(f"Average daily lift: {result['average_daily_lift']:.4f}")
```

### 3. Batch Processing Optimization

```python
from typing import List

class BatchSizeOptimizer:
    """
    Find optimal batch size range for processing
    
    Use Kadane's pattern to optimize throughput
    """
    
    def find_optimal_batching(self, processing_gains: List[float]) -> dict:
        """
        Find range of batch sizes with max throughput gain
        
        Args:
            processing_gains: Throughput gain per batch size increment
        
        Returns:
            Optimal batch size range
        """
        # Kadane's to find best subarray
        current_gain = 0
        max_gain = float('-inf')
        start_size = 0
        end_size = 0
        temp_start = 0
        
        for i, gain in enumerate(processing_gains):
            if current_gain < 0:
                current_gain = gain
                temp_start = i
            else:
                current_gain += gain
            
            if current_gain > max_gain:
                max_gain = current_gain
                start_size = temp_start
                end_size = i
        
        return {
            'optimal_min_batch': start_size,
            'optimal_max_batch': end_size,
            'total_gain': max_gain
        }

# Usage
optimizer = BatchSizeOptimizer()

# Throughput gains for batch sizes 1-10
# (e.g., batch size 1→2 gains 0.1, 2→3 gains 0.2, etc.)
gains = [0.1, 0.2, 0.15, -0.05, -0.1, 0.3, 0.2, 0.1, -0.15, -0.2]

result = optimizer.find_optimal_batching(gains)
print(f"Optimal batch size range: {result['optimal_min_batch']}-{result['optimal_max_batch']}")
print(f"Expected throughput gain: {result['total_gain']:.2f}")
```

---

## Advanced Applications in ML Systems

### Time-Series Analysis

Kadane's algorithm for finding anomalies in time-series data.

```python
class TimeSeriesAnomalyDetector:
    """
    Detect anomalous periods in time-series
    
    Uses modified Kadane's to find sustained deviations
    """
    
    def __init__(self, baseline_mean=0.0):
        self.baseline = baseline_mean
    
    def detect_anomalous_period(
        self,
        values: List[float],
        threshold: float = 2.0
    ) -> Dict:
        """
        Find period with maximum cumulative deviation from baseline
        
        Args:
            values: Time-series values
            threshold: Deviation threshold to report
        
        Returns:
            {
                'max_deviation': float,
                'start_idx': int,
                'end_idx': int,
                'is_anomalous': bool
            }
        """
        # Convert to deviations from baseline
        deviations = [v - self.baseline for v in values]
        
        # Apply Kadane's
        current_sum = deviations[0]
        max_sum = deviations[0]
        start = 0
        end = 0
        temp_start = 0
        
        for i in range(1, len(deviations)):
            if deviations[i] > current_sum + deviations[i]:
                current_sum = deviations[i]
                temp_start = i
            else:
                current_sum += deviations[i]
            
            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i
        
        return {
            'max_deviation': max_sum,
            'start_idx': start,
            'end_idx': end,
            'duration': end - start + 1,
            'is_anomalous': max_sum > threshold,
            'values_in_period': values[start:end+1]
        }

# Usage: Detect CPU spike periods
cpu_usage = [45, 50, 48, 75, 80, 85, 90, 78, 52, 48, 50]
detector = TimeSeriesAnomalyDetector(baseline_mean=50)

result = detector.detect_anomalous_period(cpu_usage, threshold=100)
if result['is_anomalous']:
    print(f"Anomaly detected: indices {result['start_idx']}-{result['end_idx']}")
    print(f"Max deviation: {result['max_deviation']:.1f}")
    print(f"Duration: {result['duration']} time steps")
```

### Feature Importance over Time

Track feature contribution windows in ML models.

```python
class FeatureContributionTracker:
    """
    Track windows where features contribute most to predictions
    
    Use Kadane's pattern to find impactful periods
    """
    
    def __init__(self):
        self.feature_impacts = {}
    
    def track_feature_impact(
        self,
        feature_name: str,
        daily_impacts: List[float]
    ) -> Dict:
        """
        Find period where feature had maximum cumulative impact
        
        Args:
            feature_name: Name of feature
            daily_impacts: Daily SHAP values or feature importance
        
        Returns:
            Analysis of most impactful period
        """
        if not daily_impacts:
            return None
        
        # Kadane's algorithm
        current_sum = daily_impacts[0]
        max_sum = daily_impacts[0]
        start = 0
        end = 0
        temp_start = 0
        
        for i in range(1, len(daily_impacts)):
            if daily_impacts[i] > current_sum + daily_impacts[i]:
                current_sum = daily_impacts[i]
                temp_start = i
            else:
                current_sum += daily_impacts[i]
            
            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i
        
        # Also track minimum impact period (negative contribution)
        current_min = daily_impacts[0]
        min_sum = daily_impacts[0]
        min_start = 0
        min_end = 0
        temp_min_start = 0
        
        for i in range(1, len(daily_impacts)):
            if daily_impacts[i] < current_min + daily_impacts[i]:
                current_min = daily_impacts[i]
                temp_min_start = i
            else:
                current_min += daily_impacts[i]
            
            if current_min < min_sum:
                min_sum = current_min
                min_start = temp_min_start
                min_end = i
        
        return {
            'feature_name': feature_name,
            'max_positive_impact': {
                'cumulative': max_sum,
                'start_day': start,
                'end_day': end,
                'duration': end - start + 1,
                'avg_daily': max_sum / (end - start + 1)
            },
            'max_negative_impact': {
                'cumulative': min_sum,
                'start_day': min_start,
                'end_day': min_end,
                'duration': min_end - min_start + 1,
                'avg_daily': min_sum / (min_end - min_start + 1)
            }
        }

# Usage
tracker = FeatureContributionTracker()

# SHAP values for a feature over 30 days
shap_values = [0.1, 0.2, 0.15, -0.05, 0.3, 0.25, 0.2, -0.1, -0.15, 0.05,
               0.1, 0.12, 0.18, 0.22, 0.19, -0.08, 0.1, 0.15, 0.2, 0.25,
               0.3, 0.28, -0.12, -0.2, 0.05, 0.1, 0.15, 0.12, 0.08, 0.1]

analysis = tracker.track_feature_impact('user_engagement_score', shap_values)

print(f"Feature: {analysis['feature_name']}")
print(f"Most impactful period: days {analysis['max_positive_impact']['start_day']}"
      f" to {analysis['max_positive_impact']['end_day']}")
print(f"Cumulative impact: {analysis['max_positive_impact']['cumulative']:.3f}")
```

### Sliding Window with Constraints

Maximum subarray with length constraints.

```python
def maxSubArrayWithConstraints(
    nums: List[int],
    min_length: int = 1,
    max_length: int = None
) -> tuple[int, int, int]:
    """
    Find maximum subarray with length constraints
    
    Args:
        nums: Input array
        min_length: Minimum subarray length
        max_length: Maximum subarray length (None = no limit)
    
    Returns:
        (max_sum, start_idx, end_idx)
    """
    n = len(nums)
    if n < min_length:
        return (float('-inf'), -1, -1)
    
    max_sum = float('-inf')
    best_start = 0
    best_end = 0
    
    # For each starting position
    for start in range(n):
        current_sum = 0
        
        # Try different ending positions
        for end in range(start, n):
            current_sum += nums[end]
            length = end - start + 1
            
            # Check constraints
            if max_length and length > max_length:
                break
            
            if length >= min_length and current_sum > max_sum:
                max_sum = current_sum
                best_start = start
                best_end = end
    
    return (max_sum, best_start, best_end)

# Usage: Find best 3-5 day trading window
prices_changes = [5, -2, 8, -3, 4, -1, 7, -2, 3]
max_profit, start, end = maxSubArrayWithConstraints(
    prices_changes,
    min_length=3,
    max_length=5
)

print(f"Best window: days {start} to {end}")
print(f"Total gain: {max_profit}")
print(f"Window length: {end - start + 1} days")
```

### Divide and Conquer Solution

O(n log n) approach for understanding recursion.

```python
def maxSubArrayDivideConquer(nums: List[int]) -> int:
    """
    Divide and conquer approach
    
    Time: O(n log n)
    Space: O(log n) for recursion stack
    
    Educational value: Shows different algorithmic paradigm
    """
    
    def maxCrossingSum(nums, left, mid, right):
        """
        Find max sum crossing the midpoint
        """
        # Left side of mid
        left_sum = float('-inf')
        current_sum = 0
        for i in range(mid, left - 1, -1):
            current_sum += nums[i]
            left_sum = max(left_sum, current_sum)
        
        # Right side of mid
        right_sum = float('-inf')
        current_sum = 0
        for i in range(mid + 1, right + 1):
            current_sum += nums[i]
            right_sum = max(right_sum, current_sum)
        
        return left_sum + right_sum
    
    def maxSubArrayRecursive(nums, left, right):
        """
        Recursive divide and conquer
        """
        # Base case
        if left == right:
            return nums[left]
        
        # Divide
        mid = (left + right) // 2
        
        # Conquer: three cases
        # 1. Max subarray in left half
        left_max = maxSubArrayRecursive(nums, left, mid)
        
        # 2. Max subarray in right half
        right_max = maxSubArrayRecursive(nums, mid + 1, right)
        
        # 3. Max subarray crossing midpoint
        cross_max = maxCrossingSum(nums, left, mid, right)
        
        # Return maximum of three
        return max(left_max, right_max, cross_max)
    
    return maxSubArrayRecursive(nums, 0, len(nums) - 1)

# Example
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(f"Max subarray sum: {maxSubArrayDivideConquer(nums)}")  # 6
```

---

## Interview Tips & Common Patterns

### Recognizing Kadane's Pattern

**When to use Kadane's:**
- "Maximum/minimum subarray sum"
- "Best consecutive period"
- "Optimal window with contiguous elements"
- "Track running optimum with reset option"

**Key characteristics:**
- Contiguous subsequence required
- Looking for optimum (max/min)
- Can "start fresh" at any point
- Single pass possible

### Follow-up Questions to Expect

**Q1: What if array can be empty?**
```python
def maxSubArrayEmptyAllowed(nums: List[int]) -> int:
    """
    Allow empty subarray (return 0 if all negative)
    """
    if not nums:
        return 0
    
    max_sum = 0  # Empty subarray
    current_sum = 0
    
    for num in nums:
        current_sum = max(0, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

**Q2: Return all maximum subarrays (in case of ties)?**
```python
def findAllMaxSubarrays(nums: List[int]) -> List[tuple[int, int]]:
    """
    Find all subarrays with maximum sum
    """
    # First, find max sum
    max_sum = maxSubArray(nums)
    
    # Find all subarrays with this sum
    result = []
    n = len(nums)
    
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += nums[j]
            if current_sum == max_sum:
                result.append((i, j))
    
    return result

# Example
nums = [1, 2, -3, 4]
print(findAllMaxSubarrays(nums))  # [(0, 1), (3, 3)] - both sum to 4
```

**Q3: 2D version (maximum sum rectangle)?**
```python
def maxSumRectangle(matrix: List[List[int]]) -> int:
    """
    Find maximum sum rectangle in 2D matrix
    
    Strategy: Fix left and right columns, apply Kadane's on rows
    
    Time: O(n² * m) where matrix is n x m
    """
    if not matrix or not matrix[0]:
        return 0
    
    rows = len(matrix)
    cols = len(matrix[0])
    max_sum = float('-inf')
    
    # Try all pairs of columns
    for left in range(cols):
        # Temp array to store row sums
        temp = [0] * rows
        
        for right in range(left, cols):
            # Add current column to temp
            for row in range(rows):
                temp[row] += matrix[row][right]
            
            # Apply Kadane's on temp (1D problem)
            current_sum = temp[0]
            current_max = temp[0]
            
            for i in range(1, rows):
                current_sum = max(temp[i], current_sum + temp[i])
                current_max = max(current_max, current_sum)
            
            max_sum = max(max_sum, current_max)
    
    return max_sum

# Example
matrix = [
    [1, 2, -1, -4],
    [-8, -3, 4, 2],
    [3, 8, 10, -8]
]
print(maxSumRectangle(matrix))  # 19 (rectangle from (0,1) to (2,2))
```

---

## Production Considerations

### Handling Real-World Data

```python
import math

class RobustMaxSubArray:
    """
    Production-ready maximum subarray with validation
    """
    
    def max_subarray(self, nums: List[float]) -> float:
        """
        Handle floating point values, NaN, inf
        """
        # Filter out invalid values
        valid_nums = [
            x for x in nums
            if x is not None and not math.isnan(x) and not math.isinf(x)
        ]
        
        if not valid_nums:
            return 0.0
        
        # Standard Kadane's
        current_sum = valid_nums[0]
        max_sum = valid_nums[0]
        
        for i in range(1, len(valid_nums)):
            current_sum = max(valid_nums[i], current_sum + valid_nums[i])
            max_sum = max(max_sum, current_sum)
        
        return round(max_sum, 6)  # Round for float precision
    
    def max_subarray_with_metadata(self, nums: List[float]) -> Dict:
        """
        Return comprehensive analysis
        """
        if not nums:
            return {
                'max_sum': 0,
                'start': -1,
                'end': -1,
                'length': 0,
                'percentage_of_total': 0
            }
        
        current_sum = nums[0]
        max_sum = nums[0]
        start = 0
        end = 0
        temp_start = 0
        
        for i in range(1, len(nums)):
            if nums[i] > current_sum + nums[i]:
                current_sum = nums[i]
                temp_start = i
            else:
                current_sum += nums[i]
            
            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i
        
        total_sum = sum(nums)
        
        return {
            'max_sum': max_sum,
            'start': start,
            'end': end,
            'length': end - start + 1,
            'percentage_of_array': (end - start + 1) / len(nums) * 100,
            'percentage_of_total': (max_sum / total_sum * 100) if total_sum != 0 else 0,
            'subarray': nums[start:end+1]
        }
```

### Performance Monitoring

```python
import time

class PerformanceTracker:
    """
    Track algorithm performance
    """
    
    def benchmark(self, sizes):
        """Benchmark on different input sizes"""
        for size in sizes:
            nums = [(-1) ** i * (i % 100) for i in range(size)]
            
            start = time.perf_counter()
            result = maxSubArray(nums)
            end = time.perf_counter()
            
            elapsed_ms = (end - start) * 1000
            throughput = size / (end - start) / 1_000_000  # M elements/sec
            
            print(f"n={size:>7}: {elapsed_ms:>8.3f}ms, {throughput:>6.2f} M/s")
    
    def compare_approaches(self, nums):
        """Compare different approaches"""
        approaches = {
            "Kadane's O(n)": maxSubArray,
            "Brute Force O(n²)": maxSubArrayBruteForce,
            "Divide & Conquer O(n log n)": maxSubArrayDivideConquer,
        }
        
        print(f"Array size: {len(nums)}")
        print("-" * 50)
        
        for name, func in approaches.items():
            start = time.perf_counter()
            result = func(nums)
            end = time.perf_counter()
            
            elapsed_ms = (end - start) * 1000
            print(f"{name:30} {elapsed_ms:>8.3f}ms  Result: {result}")

# Run benchmark
tracker = PerformanceTracker()
tracker.benchmark([100, 1_000, 10_000, 100_000, 1_000_000])

# Compare on smaller array
small_array = [(-1) ** i * (i % 10) for i in range(1000)]
tracker.compare_approaches(small_array)
```

### Monitoring in Production

```python
class MaxSubarrayMonitor:
    """
    Monitor Kadane's algorithm in production
    
    Track performance, edge cases, and anomalies
    """
    
    def __init__(self):
        self.execution_count = 0
        self.total_time = 0
        self.edge_case_count = 0
        self.all_negative_count = 0
        self.all_positive_count = 0
    
    def monitored_max_subarray(self, nums: List[int]) -> Dict:
        """
        Wrap max_subarray with monitoring
        """
        self.execution_count += 1
        
        start = time.perf_counter()
        
        # Edge case detection
        if not nums:
            self.edge_case_count += 1
            return {'result': 0, 'edge_case': 'empty_array'}
        
        if all(x < 0 for x in nums):
            self.all_negative_count += 1
        
        if all(x > 0 for x in nums):
            self.all_positive_count += 1
        
        # Execute algorithm
        result = maxSubArray(nums)
        
        end = time.perf_counter()
        self.total_time += (end - start)
        
        return {
            'result': result,
            'execution_time_ms': (end - start) * 1000,
            'array_size': len(nums)
        }
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        if self.execution_count == 0:
            return {}
        
        return {
            'total_executions': self.execution_count,
            'avg_time_ms': (self.total_time / self.execution_count) * 1000,
            'edge_cases': self.edge_case_count,
            'all_negative_arrays': self.all_negative_count,
            'all_positive_arrays': self.all_positive_count,
            'edge_case_rate': self.edge_case_count / self.execution_count * 100
        }
```

---

## Key Takeaways

✅ **Kadane's algorithm** is a perfect example of greedy + DP  
✅ **Single pass O(n)** with O(1) space, optimal for streaming data  
✅ **Local optimality** → global optimality when problem has optimal substructure  
✅ **Pattern extends** to circular arrays, max product, and many ML applications  
✅ **Production systems** use this pattern for online metrics, A/B tests, and batch optimization  
✅ **Connection to DP** helps understand state transitions and decision making  
✅ **Similar to stock problem** (Day 4), both track running optimum in single pass  

---

## Related Problems

Master these variations:
- **[Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)** - Track both max and min
- **[Maximum Subarray Sum Circular](https://leetcode.com/problems/maximum-sum-circular-subarray/)** - Circular array variation
- **[Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)** - Same pattern (Day 4)
- **[Maximum Sum of Two Non-Overlapping Subarrays](https://leetcode.com/problems/maximum-sum-of-two-non-overlapping-subarrays/)**
- **[Longest Turbulent Subarray](https://leetcode.com/problems/longest-turbulent-subarray/)** - Similar DP pattern

---

**Originally published at:** [arunbaby.com/dsa/0005-maximum-subarray](https://www.arunbaby.com/dsa/0005-maximum-subarray/)

*If you found this helpful, consider sharing it with others who might benefit.*

