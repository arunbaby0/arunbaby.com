---
title: "Binary Search"
day: 9
collection: dsa
categories:
  - dsa
tags:
  - binary-search
  - searching
  - divide-and-conquer
  - arrays
topic: Sorting & Searching
difficulty: Easy
companies: [Google, Meta, Amazon, Microsoft, Apple, Bloomberg, Adobe]
leetcode_link: "https://leetcode.com/problems/binary-search/"
time_complexity: "O(log n)"
space_complexity: "O(1) iterative, O(log n) recursive"
related_ml_day: 9
related_speech_day: 9
related_agents_day: 9
---

**Master binary search to understand logarithmic algorithms and efficient searching, foundational for optimization and search systems.**

## Problem

Given an array of integers `nums` which is sorted in ascending order, and an integer `target`, write a function to search `target` in `nums`. If `target` exists, then return its index. Otherwise, return `-1`.

You must write an algorithm with **O(log n)** runtime complexity.

**Example 1:**
```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4
```

**Example 2:**
```
Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
```

**Constraints:**
- 1 <= nums.length <= 10^4
- -10^4 < nums[i], target < 10^4
- All integers in `nums` are **unique**
- `nums` is sorted in ascending order

---

## Understanding Binary Search

### The Core Idea

Binary search repeatedly divides the search space in half:

```
Array: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
Target: 11

Step 1: Check middle (9)
  [1, 3, 5, 7, 9] | 11 | [11, 13, 15, 17, 19]
  11 > 9 → search right half

Step 2: Check middle of right half (15)
  [11, 13] | 15 | [17, 19]
  11 < 15 → search left half

Step 3: Check middle of left half (11)
  Found! Index = 5
```

**Key insight:** Each comparison eliminates half of the remaining elements.

### Why O(log n)?

With each step, we cut the search space in half:
- n elements → n/2 → n/4 → n/8 → ... → 1
- Number of steps = log₂(n)

```
Array size    Steps needed
10            4
100           7
1,000         10
1,000,000     20
1,000,000,000 30
```

---

## Approach 1: Iterative Binary Search

**Most common and recommended approach**

```python
def binarySearch(nums: list[int], target: int) -> int:
    """
    Iterative binary search
    
    Time: O(log n)
    Space: O(1)
    
    Args:
        nums: Sorted array
        target: Value to find
    
    Returns:
        Index of target or -1 if not found
    """
    left = 0
    right = len(nums) - 1
    
    while left <= right:
        # Calculate middle (avoids overflow)
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            # Target is in right half
            left = mid + 1
        else:
            # Target is in left half
            right = mid - 1
    
    # Target not found
    return -1

# Test cases
print(binarySearch([-1, 0, 3, 5, 9, 12], 9))   # 4
print(binarySearch([-1, 0, 3, 5, 9, 12], 2))   # -1
print(binarySearch([5], 5))                      # 0
```

### Why `mid = left + (right - left) // 2`?

```python
# Simple but can overflow with large indices
mid = (left + right) // 2  # BAD: left + right might overflow

# Safe version
mid = left + (right - left) // 2  # GOOD: no overflow

# Example where overflow matters (in languages like Java):
# left = 2^30, right = 2^30
# left + right = 2^31 (overflow in 32-bit int!)
# left + (right - left) // 2 = safe
```

### Loop Invariant

**The target, if it exists, is always in the range [left, right]:**

```
Initial: left=0, right=n-1
  Target ∈ [0, n-1] ✓

After each iteration:
  - If nums[mid] < target: left = mid + 1
    Target > nums[mid], so Target ∈ [mid+1, right] ✓
  
  - If nums[mid] > target: right = mid - 1
    Target < nums[mid], so Target ∈ [left, mid-1] ✓

Termination: left > right
  Search space is empty, target not found ✓
```

---

## Approach 2: Recursive Binary Search

**More elegant, uses call stack**

```python
def binarySearchRecursive(nums: list[int], target: int) -> int:
    """
    Recursive binary search
    
    Time: O(log n)
    Space: O(log n) - recursion stack
    """
    def search(left: int, right: int) -> int:
        # Base case: search space is empty
        if left > right:
            return -1
        
        # Calculate middle
        mid = left + (right - left) // 2
        
        # Found target
        if nums[mid] == target:
            return mid
        
        # Recursively search appropriate half
        if nums[mid] < target:
            return search(mid + 1, right)  # Search right
        else:
            return search(left, mid - 1)   # Search left
    
    return search(0, len(nums) - 1)

# Test
print(binarySearchRecursive([1, 2, 3, 4, 5], 3))  # 2
```

### Recursion Tree

```
search([1,2,3,4,5,6,7,8,9], target=7)
├─ mid=5 (value=5), 7>5
└─ search([6,7,8,9])
   ├─ mid=7 (value=7)
   └─ Found! Return 6
```

---

## Approach 3: Python's bisect Module

**Production-ready implementation**

```python
import bisect

def binarySearchBuiltin(nums: list[int], target: int) -> int:
    """
    Using Python's bisect module
    
    bisect.bisect_left returns the insertion point
    """
    idx = bisect.bisect_left(nums, target)
    
    if idx < len(nums) and nums[idx] == target:
        return idx
    return -1

# Alternative: find insertion point
def findInsertPosition(nums: list[int], target: int) -> int:
    """Find position where target should be inserted"""
    return bisect.bisect_left(nums, target)

# Test
nums = [1, 3, 5, 7, 9]
print(binarySearchBuiltin(nums, 5))        # 2
print(findInsertPosition(nums, 6))         # 3 (insert between 5 and 7)
```

---

## Binary Search Variants

### Variant 1: Find First Occurrence

```python
def findFirst(nums: list[int], target: int) -> int:
    """
    Find first occurrence of target
    
    Example: [1, 2, 2, 2, 3], target=2 → return 1
    """
    left, right = 0, len(nums) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

# Test
print(findFirst([1, 2, 2, 2, 3], 2))  # 1
```

### Variant 2: Find Last Occurrence

```python
def findLast(nums: list[int], target: int) -> int:
    """
    Find last occurrence of target
    
    Example: [1, 2, 2, 2, 3], target=2 → return 3
    """
    left, right = 0, len(nums) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

# Test
print(findLast([1, 2, 2, 2, 3], 2))  # 3
```

### Variant 3: Search in Rotated Sorted Array

```python
def searchRotated(nums: list[int], target: int) -> int:
    """
    Search in rotated sorted array
    
    Example: [4,5,6,7,0,1,2], target=0 → return 4
    
    The array was originally [0,1,2,4,5,6,7] and rotated
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Determine which half is sorted
        if nums[left] <= nums[mid]:
            # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1  # Target in left half
            else:
                left = mid + 1   # Target in right half
        else:
            # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1   # Target in right half
            else:
                right = mid - 1  # Target in left half
    
    return -1

# Test
print(searchRotated([4,5,6,7,0,1,2], 0))  # 4
```

### Variant 4: Find Peak Element

```python
def findPeakElement(nums: list[int]) -> int:
    """
    Find peak element (greater than neighbors)
    
    Example: [1,2,3,1] → return 2 (index of 3)
    
    Time: O(log n)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[mid + 1]:
            # Peak is in left half (or mid is peak)
            right = mid
        else:
            # Peak is in right half
            left = mid + 1
    
    return left

# Test
print(findPeakElement([1, 2, 3, 1]))     # 2
print(findPeakElement([1, 2, 1, 3, 5]))  # 4 (index of 5)
```

### Variant 5: Square Root (Binary Search on Answer)

```python
def mySqrt(x: int) -> int:
    """
    Find square root (floor value)
    
    Example: x=8 → return 2 (since 2² = 4 < 8 < 9 = 3²)
    
    Binary search on the answer!
    """
    if x < 2:
        return x
    
    left, right = 1, x // 2
    
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # Floor value

# Test
print(mySqrt(4))   # 2
print(mySqrt(8))   # 2
print(mySqrt(16))  # 4
```

---

## Common Mistakes

### Mistake 1: Wrong Loop Condition

```python
# WRONG: Misses single element case
while left < right:  # Should be left <= right
    mid = left + (right - left) // 2
    # ...

# Example where it fails:
nums = [5], target = 5
# left=0, right=0, loop doesn't execute!
```

### Mistake 2: Infinite Loop

```python
# WRONG: Can cause infinite loop
while left < right:
    mid = (left + right) // 2
    if nums[mid] < target:
        left = mid  # Should be mid + 1!
    else:
        right = mid - 1
```

### Mistake 3: Integer Overflow (Other Languages)

```python
# In Python, no overflow, but in Java/C++:
mid = (left + right) / 2  # Can overflow

# Better:
mid = left + (right - left) / 2
```

### Mistake 4: Off-by-One Errors

```python
# WRONG: Initial right value
right = len(nums)  # Should be len(nums) - 1

# Or need to adjust loop condition:
while left < right:  # Not left <= right
```

---

## Edge Cases

### Test Suite

```python
import unittest

class TestBinarySearch(unittest.TestCase):
    
    def test_empty_array(self):
        """Test with empty array"""
        self.assertEqual(binarySearch([], 5), -1)
    
    def test_single_element_found(self):
        """Test single element - found"""
        self.assertEqual(binarySearch([5], 5), 0)
    
    def test_single_element_not_found(self):
        """Test single element - not found"""
        self.assertEqual(binarySearch([5], 3), -1)
    
    def test_first_element(self):
        """Test target is first element"""
        self.assertEqual(binarySearch([1, 2, 3, 4, 5], 1), 0)
    
    def test_last_element(self):
        """Test target is last element"""
        self.assertEqual(binarySearch([1, 2, 3, 4, 5], 5), 4)
    
    def test_middle_element(self):
        """Test target is middle element"""
        self.assertEqual(binarySearch([1, 2, 3, 4, 5], 3), 2)
    
    def test_not_found_smaller(self):
        """Test target smaller than all elements"""
        self.assertEqual(binarySearch([5, 6, 7, 8], 2), -1)
    
    def test_not_found_larger(self):
        """Test target larger than all elements"""
        self.assertEqual(binarySearch([1, 2, 3, 4], 10), -1)
    
    def test_not_found_middle(self):
        """Test target in middle but not present"""
        self.assertEqual(binarySearch([1, 3, 5, 7, 9], 6), -1)
    
    def test_negative_numbers(self):
        """Test with negative numbers"""
        self.assertEqual(binarySearch([-5, -3, -1, 0, 2], -3), 1)
    
    def test_duplicates(self):
        """Test with duplicates (finds any occurrence)"""
        result = binarySearch([1, 2, 2, 2, 3], 2)
        self.assertIn(result, [1, 2, 3])
    
    def test_large_array(self):
        """Test with large array"""
        nums = list(range(1000000))
        self.assertEqual(binarySearch(nums, 999999), 999999)

if __name__ == '__main__':
    unittest.main()
```

---

## Performance Analysis

### Time Complexity Proof

```
T(n) = T(n/2) + O(1)

By Master Theorem:
a = 1, b = 2, f(n) = O(1)
log_b(a) = log_2(1) = 0
f(n) = O(n^0) = O(1)

Therefore: T(n) = O(log n)
```

### Comparison with Linear Search

```python
import time
import random

def benchmark_search():
    """Compare binary vs linear search"""
    sizes = [100, 1000, 10000, 100000, 1000000]
    
    print("Size      Binary      Linear")
    print("-" * 40)
    
    for size in sizes:
        nums = list(range(size))
        target = size - 1  # Worst case for linear
        
        # Binary search
        start = time.perf_counter()
        for _ in range(1000):
            binarySearch(nums, target)
        binary_time = time.perf_counter() - start
        
        # Linear search
        start = time.perf_counter()
        for _ in range(1000):
            nums.index(target)
        linear_time = time.perf_counter() - start
        
        print(f"{size:7d}   {binary_time:8.4f}s  {linear_time:8.4f}s")

# Example output:
# Size      Binary      Linear
# ----------------------------------------
#     100     0.0003s    0.0010s
#    1000     0.0004s    0.0095s
#   10000     0.0005s    0.0950s
#  100000     0.0006s    0.9500s
# 1000000     0.0007s    9.5000s
```

---

## Connection to ML Systems

Binary search patterns appear throughout ML engineering:

### 1. Hyperparameter Tuning

```python
import copy

def find_optimal_learning_rate(model, train_fn, validate_fn, 
                                min_lr=1e-6, max_lr=1.0):
    """
    Binary search for optimal learning rate
    
    Similar to binary search on answer space
    """
    best_lr = min_lr
    best_loss = float('inf')
    
    left, right = min_lr, max_lr
    
    while right - left > 1e-7:
        mid = (left + right) / 2
        
        # Train with this learning rate
        model_copy = copy.deepcopy(model)
        train_fn(model_copy, learning_rate=mid)
        loss = validate_fn(model_copy)
        
        if loss < best_loss:
            best_loss = loss
            best_lr = mid
        
        # Adjust search space based on loss gradient
        # (simplified - real implementation would be more sophisticated)
        if loss > best_loss * 1.1:
            right = mid
        else:
            left = mid
    
    return best_lr
```

### 2. Threshold Optimization

```python
def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Ternary search for optimal classification threshold (smooth metric)
    
    Finds threshold that maximizes given metric
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    def evaluate_threshold(threshold):
        y_pred = (y_pred_proba >= threshold).astype(int)
        if metric == 'f1':
            return f1_score(y_true, y_pred)
        elif metric == 'precision':
            return precision_score(y_true, y_pred)
        elif metric == 'recall':
            return recall_score(y_true, y_pred)
    
    # Ternary search across [0, 1]
    left, right = 0.0, 1.0
    best_threshold = 0.5
    best_score = evaluate_threshold(0.5)
    
    # Sample points and use binary search logic
    for _ in range(20):  # 20 iterations for precision
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        score1 = evaluate_threshold(mid1)
        score2 = evaluate_threshold(mid2)
        
        if score1 > best_score:
            best_score = score1
            best_threshold = mid1
        
        if score2 > best_score:
            best_score = score2
            best_threshold = mid2
        
        # Ternary search logic
        if score1 > score2:
            right = mid2
        else:
            left = mid1
    
    return best_threshold, best_score

# Usage
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
y_pred_proba = np.array([0.2, 0.8, 0.6, 0.3, 0.9, 0.1, 0.7, 0.55])

threshold, score = find_optimal_threshold(y_true, y_pred_proba, metric='f1')
print(f"Optimal threshold: {threshold:.3f}, F1 score: {score:.3f}")
```

### 3. Model Version Search

```python
import bisect

class ModelVersionSelector:
    """
    Binary search through model versions to find regression point
    
    Similar to git bisect
    """
    
    def __init__(self, versions):
        """
        Args:
            versions: List of model versions sorted by timestamp
        """
        self.versions = versions
    
    def find_regression(self, test_fn):
        """
        Find first version where test fails
        
        Args:
            test_fn: Function that tests model, returns True if passes
        
        Returns:
            First failing version or None
        """
        left, right = 0, len(self.versions) - 1
        first_bad = None
        
        while left <= right:
            mid = left + (right - left) // 2
            version = self.versions[mid]
            
            print(f"Testing version {version}...")
            if test_fn(version):
                # This version is good, search right
                left = mid + 1
            else:
                # This version is bad, could be first bad
                first_bad = version
                right = mid - 1
        
        return first_bad

# Usage
versions = ['v1.0', 'v1.1', 'v1.2', 'v1.3', 'v1.4', 'v1.5']

def test_model(version):
    """Test if model version performs well"""
    # Load model and run tests
    accuracy = evaluate_model(version)
    return accuracy >= 0.95

selector = ModelVersionSelector(versions)
bad_version = selector.find_regression(test_model)
print(f"Regression introduced in: {bad_version}")
```

---

## Production Patterns

### 1. Bisect for Bucketing

```python
class FeatureBucketer:
    """
    Bucket continuous features using binary search
    
    Faster than linear scan for many buckets
    """
    
    def __init__(self, bucket_boundaries):
        """
        Args:
            bucket_boundaries: Sorted list of bucket boundaries
                              e.g., [0, 10, 50, 100, 500, 1000]
        """
        self.boundaries = sorted(bucket_boundaries)
    
    def get_bucket(self, value):
        """
        Find bucket for value using binary search
        
        Time: O(log b) where b = number of buckets
        
        Returns:
            Bucket index
        """
        import bisect
        return bisect.bisect_left(self.boundaries, value)
    
    def bucket_features(self, values):
        """Bucket array of values"""
        return [self.get_bucket(v) for v in values]

# Usage
bucketer = FeatureBucketer([0, 1000, 5000, 10000, 50000, 100000])

# Bucket user income
incomes = [500, 2000, 7500, 15000, 75000, 200000]
buckets = bucketer.bucket_features(incomes)
print(buckets)  # [0, 1, 2, 3, 4, 5]
```

### 2. Cache with Binary Search

```python
import bisect

class SortedCache:
    """
    Cache with binary search for fast lookups
    
    Useful when keys are numeric and can be sorted
    """
    
    def __init__(self, max_size=1000):
        self.keys = []
        self.values = []
        self.max_size = max_size
    
    def get(self, key):
        """
        Get value with binary search
        
        Time: O(log n)
        """
        idx = bisect.bisect_left(self.keys, key)
        if idx < len(self.keys) and self.keys[idx] == key:
            return self.values[idx]
        return None
    
    def put(self, key, value):
        """
        Insert key-value pair maintaining sorted order
        
        Time: O(n) for insertion, but O(log n) lookups
        """
        idx = bisect.bisect_left(self.keys, key)
        
        if idx < len(self.keys) and self.keys[idx] == key:
            # Key exists, update value
            self.values[idx] = value
        else:
            # Insert new key-value pair
            self.keys.insert(idx, key)
            self.values.insert(idx, value)
            
            # Evict if over capacity (remove oldest)
            if len(self.keys) > self.max_size:
                self.keys.pop(0)
                self.values.pop(0)

# Usage for caching predictions
cache = SortedCache()

def get_prediction_cached(user_id, model):
    """Get prediction with caching"""
    # Check cache
    cached = cache.get(user_id)
    if cached is not None:
        return cached
    
    # Compute prediction
    prediction = model.predict([user_id])
    
    # Cache result
    cache.put(user_id, prediction)
    
    return prediction
```

---

## Advanced Applications

### 1. Exponential Search

**When search space is unbounded**

```python
def exponential_search(arr, target):
    """
    Exponential search for unbounded/infinite arrays
    
    Step 1: Find range where target might exist
    Step 2: Binary search in that range
    
    Time: O(log n) where n is position of target
    """
    if not arr:
        return -1
    
    # Check first element
    if arr[0] == target:
        return 0
    
    # Find range by repeatedly doubling index
    i = 1
    while i < len(arr) and arr[i] <= target:
        i *= 2
    
    # Binary search in range [i//2, min(i, len(arr)-1)]
    left = i // 2
    right = min(i, len(arr) - 1)
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Test
print(exponential_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 7))  # 6
```

### 2. Interpolation Search

**Faster than binary search for uniformly distributed data**

```python
def interpolation_search(arr, target):
    """
    Interpolation search
    
    Instead of middle, guess position based on value
    
    Time: O(log log n) for uniformly distributed data
          O(n) worst case
    """
    left, right = 0, len(arr) - 1
    
    while left <= right and arr[left] <= target <= arr[right]:
        # Calculate interpolated position
        if arr[left] == arr[right]:
            if arr[left] == target:
                return left
            return -1
        
        # Interpolation formula
        pos = left + int(
            (target - arr[left]) / (arr[right] - arr[left]) * (right - left)
        )
        
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    
    return -1

# Test - works best on uniformly distributed data
arr = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(interpolation_search(arr, 70))  # 7
```

### 3. Ternary Search

**For unimodal functions (single peak/valley)**

```python
def ternary_search(f, left, right, epsilon=1e-9):
    """
    Ternary search for finding maximum of unimodal function
    
    Divides search space into 3 parts
    
    Time: O(log3 n) ≈ O(log n)
    
    Args:
        f: Unimodal function
        left: Left bound
        right: Right bound
        epsilon: Precision
    
    Returns:
        x that maximizes f(x)
    """
    while right - left > epsilon:
        # Two midpoints
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        if f(mid1) < f(mid2):
            # Maximum is in [mid1, right]
            left = mid1
        else:
            # Maximum is in [left, mid2]
            right = mid2
    
    return (left + right) / 2

# Example: Find maximum of parabola
def f(x):
    return -(x - 5)**2 + 10  # Maximum at x=5

max_x = ternary_search(f, 0, 10)
print(f"Maximum at x = {max_x:.6f}, f(x) = {f(max_x):.6f}")
```

---

## Binary Search in 2D

### Search in 2D Matrix

```python
def searchMatrix(matrix, target):
    """
    Search in row-wise and column-wise sorted matrix
    
    Example:
    [
      [1,  4,  7,  11],
      [2,  5,  8,  12],
      [3,  6,  9,  16],
      [10, 13, 14, 17]
    ]
    
    Approach: Start from top-right corner
    - If target < current: go left
    - If target > current: go down
    
    Time: O(m + n)
    """
    if not matrix or not matrix[0]:
        return False
    
    m, n = len(matrix), len(matrix[0])
    row, col = 0, n - 1
    
    while row < m and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1  # Go left
        else:
            row += 1  # Go down
    
    return False

# Test
matrix = [
    [1,  4,  7,  11],
    [2,  5,  8,  12],
    [3,  6,  9,  16],
    [10, 13, 14, 17]
]
print(searchMatrix(matrix, 5))  # True
print(searchMatrix(matrix, 20)) # False
```

### K-th Smallest in Sorted Matrix

```python
def kthSmallest(matrix, k):
    """
    Find k-th smallest element in sorted matrix
    
    Binary search on value range, not index!
    
    Time: O(n * log(max - min))
    """
    n = len(matrix)
    left, right = matrix[0][0], matrix[n-1][n-1]
    
    def count_less_equal(mid):
        """Count elements <= mid"""
        count = 0
        row, col = n - 1, 0
        
        while row >= 0 and col < n:
            if matrix[row][col] <= mid:
                count += row + 1
                col += 1
            else:
                row -= 1
        
        return count
    
    # Binary search on value
    while left < right:
        mid = left + (right - left) // 2
        
        if count_less_equal(mid) < k:
            left = mid + 1
        else:
            right = mid
    
    return left

# Test
matrix = [
    [1,  5,  9],
    [10, 11, 13],
    [12, 13, 15]
]
print(kthSmallest(matrix, 8))  # 13
```

---

## Interview Strategies

### Problem Recognition

**When to use binary search:**

1. **Sorted array** - Classic binary search
2. **Monotonic function** - Search on answer space
3. **Find boundary** - First/last occurrence
4. **Minimize/maximize** - Optimization problems
5. **Search space reducible** - Can eliminate half each time

### Common Patterns

**Pattern 1: Find exact value**
```python
while left <= right:
    mid = left + (right - left) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
return -1
```

**Pattern 2: Find lower bound (first >= target)**
```python
while left < right:
    mid = left + (right - left) // 2
    if arr[mid] < target:
        left = mid + 1
    else:
        right = mid
return left
```

**Pattern 3: Find upper bound (first > target)**
```python
while left < right:
    mid = left + (right - left) // 2
    if arr[mid] <= target:
        left = mid + 1
    else:
        right = mid
return left
```

### Debug Checklist

```python
# Before submitting, check:
✓ Loop condition: left <= right or left < right?
✓ Mid calculation: avoiding overflow?
✓ Update: left = mid + 1 or left = mid?
✓ Update: right = mid - 1 or right = mid?
✓ Return value: left, right, or -1?
✓ Edge cases: empty array, single element?
✓ Infinite loop: does search space always shrink?
```

---

## Production Code Examples

### Binary Search with Logging

```python
import logging
from typing import List, Optional

class ProductionBinarySearch:
    """
    Production-ready binary search with logging and error handling
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def search(self, nums: List[int], target: int) -> Optional[int]:
        """
        Search for target in sorted array
        
        Returns:
            Index of target or None if not found
        
        Raises:
            ValueError: If array is not sorted
        """
        # Validate input
        if not nums:
            self.logger.warning("Empty array provided")
            return None
        
        # Check if sorted (optional, expensive for large arrays)
        if not self._is_sorted(nums):
            self.logger.error("Array is not sorted")
            raise ValueError("Array must be sorted")
        
        # Binary search
        left, right = 0, len(nums) - 1
        iterations = 0
        
        while left <= right:
            iterations += 1
            mid = left + (right - left) // 2
            
            self.logger.debug(
                f"Iteration {iterations}: left={left}, right={right}, "
                f"mid={mid}, value={nums[mid]}"
            )
            
            if nums[mid] == target:
                self.logger.info(
                    f"Found target {target} at index {mid} "
                    f"after {iterations} iterations"
                )
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        self.logger.info(
            f"Target {target} not found after {iterations} iterations"
        )
        return None
    
    def _is_sorted(self, nums: List[int]) -> bool:
        """Check if array is sorted"""
        for i in range(1, len(nums)):
            if nums[i] < nums[i-1]:
                return False
        return True

# Usage
searcher = ProductionBinarySearch()
result = searcher.search([1, 3, 5, 7, 9], 7)
```

### Thread-Safe Binary Search Cache

```python
import threading
from bisect import bisect_left

class ThreadSafeSortedCache:
    """
    Thread-safe cache with binary search lookups
    
    Useful for high-concurrency environments
    """
    
    def __init__(self):
        self.keys = []
        self.values = []
        self.lock = threading.RLock()
    
    def get(self, key):
        """
        Thread-safe get with binary search
        
        Time: O(log n)
        """
        with self.lock:
            if not self.keys:
                return None
            
            idx = bisect_left(self.keys, key)
            if idx < len(self.keys) and self.keys[idx] == key:
                return self.values[idx]
            return None
    
    def put(self, key, value):
        """
        Thread-safe put maintaining sorted order
        
        Time: O(n) for insertion
        """
        with self.lock:
            idx = bisect_left(self.keys, key)
            
            if idx < len(self.keys) and self.keys[idx] == key:
                # Update existing key
                self.values[idx] = value
            else:
                # Insert new key-value
                self.keys.insert(idx, key)
                self.values.insert(idx, value)
    
    def range_query(self, start_key, end_key):
        """
        Get all values in key range [start_key, end_key]
        
        Time: O(log n + k) where k is number of results
        """
        with self.lock:
            start_idx = bisect_left(self.keys, start_key)
            end_idx = bisect_right(self.keys, end_key)
            return list(zip(
                self.keys[start_idx:end_idx],
                self.values[start_idx:end_idx]
            ))

# Usage
cache = ThreadSafeSortedCache()

# Thread-safe operations
cache.put(10, "value10")
cache.put(5, "value5")
cache.put(15, "value15")

print(cache.get(10))  # "value10"
print(cache.range_query(5, 15))  # All values in range
```

---

## Real-World Case Studies

### Case Study 1: Netflix Content Search

```python
class NetflixContentSearch:
    """
    Simplified Netflix content search using binary search
    
    Real system uses more sophisticated indexing
    """
    
    def __init__(self, content_library):
        """
        Args:
            content_library: List of (timestamp, content_id) tuples
                            Sorted by timestamp
        """
        self.timestamps = [item[0] for item in content_library]
        self.content_ids = [item[1] for item in content_library]
    
    def find_content_at_time(self, target_time):
        """
        Find content released at or just before target_time
        
        Example: User wants to see content from 2020
        Returns most recent content <= 2020
        """
        idx = bisect.bisect_right(self.timestamps, target_time) - 1
        
        if idx >= 0:
            return self.content_ids[idx]
        return None
    
    def find_content_in_range(self, start_time, end_time):
        """
        Find all content released in time range
        
        Example: All shows from 2019-2021
        """
        start_idx = bisect.bisect_left(self.timestamps, start_time)
        end_idx = bisect.bisect_right(self.timestamps, end_time)
        
        return self.content_ids[start_idx:end_idx]

# Usage
library = [
    (2018, "content1"),
    (2019, "content2"),
    (2020, "content3"),
    (2021, "content4"),
    (2022, "content5")
]

search = NetflixContentSearch(library)
print(search.find_content_at_time(2020))  # content3
print(search.find_content_in_range(2019, 2021))  # [content2, content3, content4]
```

### Case Study 2: Database Query Optimization

```python
class DatabaseIndexSearch:
    """
    Binary search on database index
    
    Similar to B-tree index lookups
    """
    
    def __init__(self, index_pages):
        """
        Args:
            index_pages: List of (key, page_id) tuples
        """
        self.index = sorted(index_pages, key=lambda x: x[0])
    
    def find_page(self, search_key):
        """
        Find page containing search_key
        
        Returns page_id for further disk I/O
        """
        left, right = 0, len(self.index) - 1
        result_page = None
        
        while left <= right:
            mid = left + (right - left) // 2
            key, page_id = self.index[mid]
            
            if key <= search_key:
                result_page = page_id
                left = mid + 1
            else:
                right = mid - 1
        
        return result_page
    
    def estimate_io_cost(self, search_key):
        """
        Estimate I/O cost of query
        
        Binary search reduces disk reads from O(n) to O(log n)
        """
        page_id = self.find_page(search_key)
        
        # In real database, would calculate actual I/O cost
        index_ios = int(np.log2(len(self.index))) + 1
        data_ios = 1  # One page read for data
        
        return {
            'index_ios': index_ios,
            'data_ios': data_ios,
            'total_ios': index_ios + data_ios,
            'page_id': page_id
        }

# Usage
index = [(10, 'page1'), (20, 'page2'), (30, 'page3'), (40, 'page4')]
db = DatabaseIndexSearch(index)

cost = db.estimate_io_cost(25)
print(f"Query cost: {cost['total_ios']} I/O operations")
```

---

## Performance Profiling

### Benchmark Suite

```python
import time
import random
import numpy as np
import matplotlib.pyplot as plt

def benchmark_search_algorithms():
    """
    Comprehensive benchmark of search algorithms
    """
    sizes = [100, 1000, 10000, 100000, 1000000]
    
    results = {
        'binary': [],
        'linear': [],
        'exponential': [],
        'interpolation': []
    }
    
    for size in sizes:
        # Create sorted array
        arr = list(range(size))
        target = size - 1  # Worst case
        
        # Benchmark binary search
        start = time.perf_counter()
        for _ in range(1000):
            binarySearch(arr, target)
        results['binary'].append(time.perf_counter() - start)
        
        # Benchmark linear search
        start = time.perf_counter()
        for _ in range(1000):
            arr.index(target)
        results['linear'].append(time.perf_counter() - start)
        
        # Benchmark exponential search
        start = time.perf_counter()
        for _ in range(1000):
            exponential_search(arr, target)
        results['exponential'].append(time.perf_counter() - start)
        
        # Benchmark interpolation search
        start = time.perf_counter()
        for _ in range(1000):
            interpolation_search(arr, target)
        results['interpolation'].append(time.perf_counter() - start)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for algo, times in results.items():
        plt.plot(sizes, times, marker='o', label=algo)
    
    plt.xlabel('Array Size')
    plt.ylabel('Time (seconds for 1000 iterations)')
    plt.title('Search Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('search_benchmark.png')
    
    return results
```

---

## Key Takeaways

✅ **O(log n) is powerful** - Scales to billions of elements  
✅ **Requires sorted data** - Worth the sorting cost for multiple searches  
✅ **Two pointers pattern** - Left and right converge to answer  
✅ **Watch for edge cases** - Empty arrays, single elements, boundaries  
✅ **Many variants** - First/last occurrence, rotated arrays, search on answer  
✅ **ML applications** - Hyperparameter tuning, threshold optimization, bucketing  
✅ **Production use** - Feature bucketing, caching, version selection  

---

## Related Problems

- **[Search Insert Position](https://leetcode.com/problems/search-insert-position/)** - Find insertion index
- **[Find First and Last Position](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)** - Range search
- **[Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)** - Modified binary search
- **[Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)** - Find pivot
- **[Find Peak Element](https://leetcode.com/problems/find-peak-element/)** - Local maximum
- **[Sqrt(x)](https://leetcode.com/problems/sqrtx/)** - Binary search on answer
- **[Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)** - Binary search on speed

---

**Originally published at:** [arunbaby.com/dsa/0009-binary-search](https://www.arunbaby.com/dsa/0009-binary-search/)

*If you found this helpful, consider sharing it with others who might benefit.*

