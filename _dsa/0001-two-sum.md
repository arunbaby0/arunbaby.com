---
title: "Two Sum"
day: 1
related_ml_day: 1
related_speech_day: 1
related_agents_day: 1
collection: dsa
categories:
 - dsa
tags:
 - hash-tables
 - arrays
topic: Hash Tables
difficulty: Easy
companies: [Google, Meta, Amazon, Apple]
leetcode_link: "https://leetcode.com/problems/two-sum/"
time_complexity: "O(n)"
space_complexity: "O(n)"

---

**The hash table trick that makes O(n²) become O(n) and why this pattern appears everywhere from feature stores to embedding lookups.**

## Introduction

Two Sum is often the first problem engineers encounter when starting their algorithm journey, but don't let its "Easy" label fool you. This problem introduces one of the most powerful patterns in computer science: **trading space for time using hash tables**. This pattern isn't just academic it powers real production systems handling millions of requests per second, from recommendation engines to real-time analytics.

In this comprehensive guide, we'll explore:
- Why the naive O(n²) solution fails at scale
- How hash tables enable O(1) lookups
- The underlying mechanics of hash tables
- When and why to use this pattern
- Real-world applications in ML systems
- Production considerations and edge cases
- Common pitfalls and how to avoid them

## Problem Statement

**Given an array of integers `nums` and an integer `target`, return the indices of the two numbers that add up to `target`.**

### Constraints and Assumptions
- Each input has **exactly one solution**
- You **cannot use the same element twice**
- You can return the answer in any order
- `2 <= nums.length <= 10^4`
- `-10^9 <= nums[i] <= 10^9`
- `-10^9 <= target <= 10^9`

### Examples

**Example 1:**
``python
Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]
Explanation: nums[0] + nums[1] = 2 + 7 = 9
``

**Example 2:**
``python
Input: nums = [3, 2, 4], target = 6
Output: [1, 2]
Explanation: nums[1] + nums[2] = 2 + 4 = 6
Note: We can't use [0, 0] because we can't use the same element twice
``

**Example 3:**
``python
Input: nums = [3, 3], target = 6
Output: [0, 1]
Explanation: Even though both values are 3, they're at different indices
``

---

## Approach 1: Brute Force (The Naive Solution)

### The Idea

The most straightforward approach is to check every possible pair of numbers to see if they sum to the target. This is what most beginners think of first, and it's a perfectly valid starting point.

### Implementation

``python
def twoSum(nums: list[int], target: int) -> list[int]:
 """
 Brute force: Check all possible pairs
 
 Args:
 nums: List of integers
 target: Target sum
 
 Returns:
 List containing two indices [i, j] where nums[i] + nums[j] = target
 """
 n = len(nums)
 
 # Outer loop: select first number
 for i in range(n):
 # Inner loop: select second number
 # Start from i+1 to avoid using same element twice
 for j in range(i + 1, n):
 if nums[i] + nums[j] == target:
 return [i, j]
 
 # Should never reach here given problem constraints
 return []
``

### Step-by-Step Walkthrough

Let's trace through `nums = [2, 7, 11, 15]`, `target = 9`:

``
Iteration 1: i=0, nums[i]=2
 j=1: nums[1]=7 → 2+7=9 ✓ FOUND! Return [0,1]
``

That was quick! But let's see a case where it's slower:

``
nums = [1, 2, 3, 4, 5], target = 9

Iteration 1: i=0, nums[i]=1
 j=1: 1+2=3 ✗
 j=2: 1+3=4 ✗
 j=3: 1+4=5 ✗
 j=4: 1+5=6 ✗

Iteration 2: i=1, nums[i]=2
 j=2: 2+3=5 ✗
 j=3: 2+4=6 ✗
 j=4: 2+5=7 ✗

Iteration 3: i=2, nums[i]=3
 j=3: 3+4=7 ✗
 j=4: 3+5=8 ✗

Iteration 4: i=3, nums[i]=4
 j=4: 4+5=9 ✓ FOUND! Return [3,4]
``

We had to check 9 pairs before finding the answer!

### Complexity Analysis

**Time Complexity: O(n²)**
- Outer loop runs n times
- For each outer iteration, inner loop runs (n-1), (n-2), ..., 1 times
- Total comparisons: (n-1) + (n-2) + ... + 1 = n(n-1)/2 ≈ n²/2
- In Big-O notation, we drop constants, so O(n²)

**Space Complexity: O(1)**
- We only use a fixed amount of extra space (variables i, j)
- No data structures that grow with input size

### Why This Fails at Scale

Let's see what happens with different input sizes:

| Array Size | Comparisons | Time @ 1B ops/sec |
|------------|-------------|-------------------|
| 100 | 4,950 | 0.005 ms |
| 1,000 | 499,500 | 0.5 ms |
| 10,000 | 49,995,000 | 50 ms |
| 100,000 | 4,999,950,000 | 5 seconds |
| 1,000,000 | ~500 billion | 8+ minutes |

**Problem:** As input doubles, runtime quadruples. This is catastrophic for large inputs.

**When it's acceptable:**
- Tiny arrays (n < 100)
- One-time offline computation
- Prototyping/testing
- Interview follow-up after optimal solution

**When it's unacceptable:**
- Production systems with unpredictable input sizes
- Real-time/latency-sensitive applications
- Repeated queries on same data
- Any n > 10,000

---

## Approach 2: Hash Table (The Optimal Solution)

### The Breakthrough Insight

The key realization: **For each number `nums[i]`, we need to find if `target - nums[i]` exists in the array.**

Instead of searching through the entire array each time (O(n)), we can use a **hash table** to check existence in O(1).

### What is a Hash Table?

Before diving into the solution, let's understand the data structure that makes it possible.

**Hash Table (Dictionary/Map):** A data structure that maps keys to values with O(1) average-case lookup time.

**How it works:**
1. **Hash Function:** Converts a key into an array index
2. **Array Storage:** Stores values at computed indices
3. **Collision Handling:** Manages when two keys hash to same index

**Example:**
``python
# Python dictionary is a hash table
seen = {}
seen[2] = 0 # Key 2 maps to value 0
seen[7] = 1 # Key 7 maps to value 1

# Later, check if 7 exists
if 7 in seen: # O(1) operation!
 print(f"Found at index {seen[7]}")
``

**Under the Hood:**
``
Key → Hash Function → Index in array

Example: hash(2) → 12345 % array_size → index 5
 hash(7) → 98765 % array_size → index 3

Array: [_, _, _, (7→1), _, (2→0), _, ...]
 0 1 2 3 4 5 6
``

### The Algorithm

**Strategy:** Build the hash table as we iterate, checking for complements.

``python
def twoSum(nums: list[int], target: int) -> list[int]:
 """
 Optimal solution using hash table
 
 Time: O(n), Space: O(n)
 """
 # Dictionary to store: number → index
 seen = {}
 
 for i, num in enumerate(nums):
 # Calculate what number we need
 complement = target - num
 
 # Check if we've seen the complement before
 if complement in seen:
 # Found it! Return both indices
 return [seen[complement], i]
 
 # Haven't found complement yet, store current number
 seen[num] = i
 
 # Problem guarantees a solution exists
 return []
``

### Detailed Walkthrough

Let's trace `nums = [2, 7, 11, 15]`, `target = 9`:

``
Initial state:
seen = {}

Iteration 1: i=0, num=2
 complement = 9 - 2 = 7
 Is 7 in seen? No
 Store: seen[2] = 0
 seen = {2: 0}

Iteration 2: i=1, num=7
 complement = 9 - 7 = 2
 Is 2 in seen? Yes! (at index 0)
 Return [0, 1] ✓
``

**Another example:** `nums = [3, 2, 4]`, `target = 6`:

``
Initial: seen = {}

Iteration 1: i=0, num=3
 complement = 6 - 3 = 3
 Is 3 in seen? No
 seen = {3: 0}

Iteration 2: i=1, num=2
 complement = 6 - 2 = 4
 Is 4 in seen? No
 seen = {3: 0, 2: 1}

Iteration 3: i=2, num=4
 complement = 6 - 4 = 2
 Is 2 in seen? Yes! (at index 1)
 Return [1, 2] ✓
``

### Why This Works

**Key observations:**
1. **Single pass:** We only iterate through the array once
2. **O(1) lookups:** Hash table checks are constant time
3. **Build as we go:** No need to pre-populate the hash table
4. **Order independent:** Works regardless of element order

**Mathematical proof:**
- If `nums[i] + nums[j] = target`
- Then `nums[j] = target - nums[i]`
- When we reach `nums[j]`, we check if `(target - nums[j])` exists
- This equals `nums[i]`, which we stored earlier
- Therefore, we'll find the pair when we encounter the second number

### Complexity Analysis

**Time Complexity: O(n)**
- Single loop through n elements: O(n)
- Hash table operations (insert, lookup): O(1) average
- Total: O(n) × O(1) = O(n)

**Space Complexity: O(n)**
- Hash table stores at most n elements
- In worst case (no solution found until end), we store all n numbers

**Best case:** Solution found immediately → O(1) time, O(1) space
**Average case:** Solution found midway → O(n/2) ≈ O(n) time, O(n/2) ≈ O(n) space
**Worst case:** Solution at end → O(n) time, O(n) space

### Performance Comparison

| Array Size | Brute Force | Hash Table | Speedup |
|------------|-------------|------------|---------|
| 100 | 0.005 ms | 0.001 ms | 5x |
| 1,000 | 0.5 ms | 0.01 ms | 50x |
| 10,000 | 50 ms | 0.1 ms | 500x |
| 100,000 | 5 sec | 1 ms | 5000x |
| 1,000,000 | 8 min | 10 ms | 50000x |

**The speedup grows linearly with input size!**

---

## Deep Dive: Hash Table Mechanics

### How Hash Functions Work

A hash function converts arbitrary data into a fixed-size integer:

``python
def simple_hash(key, table_size):
 """
 Simplified hash function for integers
 """
 return key % table_size

# Example
table_size = 10
print(simple_hash(23, table_size)) # 3
print(simple_hash(47, table_size)) # 7
print(simple_hash(33, table_size)) # 3 ← Collision!
``

**Real hash functions are more sophisticated:**
- Python uses SipHash for strings/bytes; integers hash to their value (with a special-case for -1)
- Involves bit manipulation and prime numbers
- Designed to minimize collisions
- Must be deterministic (same input → same output)

### Collision Handling

**Problem:** Two different keys might hash to the same index.

**Solution 1: Chaining**
``
Index 0: []
Index 1: [(7, idx_a), (17, idx_b)] ← Both hash to 1
Index 2: []
Index 3: [(3, idx_c)]
Index 4: [(4, idx_d), (14, idx_e)] ← Both hash to 4
``

Each slot holds a linked list. Lookup requires traversing the list.

**Solution 2: Open Addressing**
``
If slot is occupied, try next slot:
- Linear probing: try slot+1, slot+2, ...
- Quadratic probing: try slot+1², slot+2², ...
- Double hashing: use second hash function
``

**Python's approach:** Uses open addressing with a deterministic perturbation-based probing sequence.

### Why Hash Tables are O(1)

**Average case:** 
- Good hash function distributes keys uniformly
- Low load factor (< 0.75) means few collisions
- Most lookups hit immediately

**Worst case:**
- All keys hash to same index → O(n) lookup
- But hash functions are designed to make this extremely unlikely
- Python automatically resizes table when load factor exceeds threshold

**Load Factor:**
``
load_factor = num_elements / table_size

Example:
- ~66 elements in table of size 100 → load factor ≈ 0.66
- When load factor exceeds roughly 2/3, CPython grows the table (with overallocation)
- This keeps lookup times close to O(1)
``

---

## Variants and Extensions

### Variant 1: Return Values Instead of Indices

``python
def twoSumValues(nums: list[int], target: int) -> list[int]:
 """
 Return the actual values, not indices
 """
 seen = set()
 
 for num in nums:
 complement = target - num
 if complement in seen:
 return [complement, num]
 seen.add(num)
 
 return []

# Example
nums = [2, 7, 11, 15], target = 9
result = twoSumValues(nums, 9) # [2, 7]
``

**When to use:** You only need the values, not their positions.

### Variant 2: Return All Pairs

``python
def twoSumAllPairs(nums: list[int], target: int) -> list[list[int]]:
 """
 Find all pairs that sum to target (may have duplicates)
 """
 seen = {}
 pairs = []
 
 for i, num in enumerate(nums):
 complement = target - num
 
 # If complement exists, found a pair
 if complement in seen:
 for prev_idx in seen[complement]:
 pairs.append([prev_idx, i])
 
 # Store current number's index
 if num not in seen:
 seen[num] = []
 seen[num].append(i)
 
 return pairs

# Example
nums = [1, 1, 1, 2, 2], target = 3
result = twoSumAllPairs(nums, 3)
# [[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]]
``

### Variant 3: Sorted Input (Two Pointers)

**If the array is sorted**, we can use a more space-efficient approach:

``python
def twoSumSorted(nums: list[int], target: int) -> list[int]:
 """
 Two pointers approach for sorted array
 
 Time: O(n), Space: O(1)
 """
 left = 0
 right = len(nums) - 1
 
 while left < right:
 current_sum = nums[left] + nums[right]
 
 if current_sum == target:
 return [left, right]
 elif current_sum < target:
 # Sum too small, need larger number
 left += 1
 else:
 # Sum too large, need smaller number
 right -= 1
 
 return []
``

**Why this works:**
- Start with smallest and largest numbers
- If sum is too small, increase left pointer (make sum larger)
- If sum is too large, decrease right pointer (make sum smaller)
- Guaranteed to find solution in one pass

**Trade-off:**
- Pro: O(1) space (no hash table)
- Con: Requires sorted input (sorting is O(n log n))
- Use when: Array already sorted or space is critical

**Example walkthrough:**
``
nums = [1, 2, 3, 4, 5], target = 9

Step 1: left=0, right=4
 sum = 1 + 5 = 6 < 9 → left++

Step 2: left=1, right=4
 sum = 2 + 5 = 7 < 9 → left++

Step 3: left=2, right=4
 sum = 3 + 5 = 8 < 9 → left++

Step 4: left=3, right=4
 sum = 4 + 5 = 9 = target ✓ Return [3, 4]
``

### Variant 4: Count Number of Pairs

``python
def countPairs(nums: list[int], target: int) -> int:
 """
 Count how many pairs sum to target
 """
 seen = {}
 count = 0
 
 for num in nums:
 complement = target - num
 
 # If complement exists, all its occurrences form pairs
 if complement in seen:
 count += seen[complement]
 
 # Increment count for current number
 seen[num] = seen.get(num, 0) + 1
 
 return count

# Example
nums = [1, 1, 1, 2, 2], target = 3
count = countPairs(nums, 3) # 6 pairs
``

---

## Edge Cases and Pitfalls

### Edge Case 1: Empty or Single Element Array

``python
def twoSum(nums: list[int], target: int) -> list[int]:
 if not nums or len(nums) < 2:
 raise ValueError("Array must have at least 2 elements")
 
 seen = {}
 for i, num in enumerate(nums):
 complement = target - num
 if complement in seen:
 return [seen[complement], i]
 seen[num] = i
 
 raise ValueError("No solution found")
``

**Problem guarantees:** The problem states there's always exactly one solution, so we shouldn't reach the exception in valid inputs.

### Edge Case 2: Using Same Element Twice

``python
# Wrong!
nums = [3, 3], target = 6
# If we're not careful, might try to use index 0 twice

# Correct approach: Our solution naturally handles this
# because we only add to `seen` after checking for complement
``

**Why our solution works:**
``
i=0, num=3:
 complement = 3
 3 not in seen yet
 seen = {3: 0}

i=1, num=3:
 complement = 3
 3 IS in seen (at index 0)
 Return [0, 1] ✓
``

### Edge Case 3: Negative Numbers

``python
nums = [-1, -2, -3, -4, -5], target = -8
# Works perfectly! Hash tables handle negative numbers fine

complement = -8 - (-5) = -3
# No special handling needed
``

### Edge Case 4: Zero in Array

``python
nums = [0, 4, 3, 0], target = 0
# target = 0 means we need two numbers that sum to 0
# i.e., opposites or two zeros

# Our solution handles this correctly
``

### Edge Case 5: Large Numbers

``python
nums = [1000000000, -1000000000, 1], target = 1
# Hash tables handle large integers efficiently
# Python has arbitrary-precision integers, no overflow
``

**In other languages (C++, Java):**
``cpp
// Be careful with overflow when computing complement via subtraction
long long complement = static_cast<long long>(target) - static_cast<long long>(nums[i]);

// Safer approach: use 64-bit math throughout to avoid overflow
// If you must check for addition overflow explicitly:
if (nums[i] > 0 && target > INT_MAX - nums[i]) { /* handle overflow */ }
if (nums[i] < 0 && target < INT_MIN - nums[i]) { /* handle underflow */ }
``

### Common Mistake 1: Overwriting Indices

``python
# Wrong!
def twoSumWrong(nums, target):
 seen = {}
 
 # Pre-populate hash table
 for i, num in enumerate(nums):
 seen[num] = i
 
 # Search for complement
 for i, num in enumerate(nums):
 complement = target - num
 if complement in seen and seen[complement] != i:
 return [i, seen[complement]]
 
 return []

# Problem: If there are duplicates, we overwrite indices
nums = [3, 2, 4], target = 6
# After pre-population: seen = {3: 0, 2: 1, 4: 2}
# When we check nums[1]=2, complement=4, we find it
# But we should not have used i=0 for num=3
``

**Fix:** Build hash table as we search (our original solution).

### Common Mistake 2: Forgetting to Check for Same Index

``python
# Wrong!
def twoSumWrong(nums, target):
 seen = {}
 for i, num in enumerate(nums):
 seen[num] = i
 
 for i, num in enumerate(nums):
 complement = target - num
 if complement in seen: # Missing check!
 return [i, seen[complement]]
 
 return []

# Problem with [3], target = 6:
# complement = 6 - 3 = 3
# 3 is in seen at index 0
# Would return [0, 0] ✗
``

**Fix:** Check `seen[complement] != i`.

---

## Production Considerations

### Input Validation

``python
from typing import List, Optional

def twoSum(nums: Optional[List[int]], target: int) -> List[int]:
 """
 Production-grade implementation with validation
 """
 # Validate inputs
 if nums is None:
 raise TypeError("nums cannot be None")
 
 if not isinstance(nums, list):
 raise TypeError(f"nums must be a list, got {type(nums)}")
 
 if len(nums) < 2:
 raise ValueError(f"nums must have at least 2 elements, got {len(nums)}")
 
 if not isinstance(target, (int, float)):
 raise TypeError(f"target must be a number, got {type(target)}")
 
 # Main logic
 seen = {}
 for i, num in enumerate(nums):
 if not isinstance(num, (int, float)):
 raise TypeError(f"nums[{i}] must be a number, got {type(num)}")
 
 complement = target - num
 
 if complement in seen:
 return [seen[complement], i]
 
 seen[num] = i
 
 raise ValueError("No solution found")
``

### Logging and Monitoring

``python
import logging
import time

def twoSum(nums: List[int], target: int) -> List[int]:
 """
 Production version with logging
 """
 logger = logging.getLogger(__name__)
 start_time = time.time()
 
 logger.debug(f"Starting twoSum with {len(nums)} elements, target={target}")
 
 seen = {}
 for i, num in enumerate(nums):
 complement = target - num
 
 if complement in seen:
 elapsed = (time.time() - start_time) * 1000
 logger.info(f"Found solution in {elapsed:.2f}ms after checking {i+1} elements")
 return [seen[complement], i]
 
 seen[num] = i
 
 elapsed = (time.time() - start_time) * 1000
 logger.warning(f"No solution found after {elapsed:.2f}ms")
 raise ValueError("No solution found")
``

### Thread Safety

``python
from threading import Lock
from typing import Dict

class TwoSumCache:
 """
 Thread-safe cache for repeated two-sum queries on same array
 """
 def __init__(self):
 self._cache: Dict[tuple, List[int]] = {}
 self._lock = Lock()
 
 def two_sum(self, nums: List[int], target: int) -> List[int]:
 # Create cache key (tuple of nums and target)
 cache_key = (tuple(nums), target)
 
 # Check cache (thread-safe)
 with self._lock:
 if cache_key in self._cache:
 return self._cache[cache_key].copy()
 
 # Compute result
 result = self._two_sum_impl(nums, target)
 
 # Store in cache (thread-safe)
 with self._lock:
 self._cache[cache_key] = result.copy()
 
 return result
 
 def _two_sum_impl(self, nums: List[int], target: int) -> List[int]:
 seen = {}
 for i, num in enumerate(nums):
 complement = target - num
 if complement in seen:
 return [seen[complement], i]
 seen[num] = i
 raise ValueError("No solution found")
``

### Memory Management

``python
def twoSumMemoryEfficient(nums: List[int], target: int) -> List[int]:
 """
 More memory-efficient for very large arrays
 """
 # Instead of storing all elements, we can estimate capacity
 seen = {}
 
 # Pre-allocate to reduce resizing
 # (Python does this automatically, but you can hint)
 expected_size = min(len(nums), 10000) # Cap at 10k
 
 for i, num in enumerate(nums):
 complement = target - num
 
 if complement in seen:
 result = [seen[complement], i]
 
 # Clear hash table to free memory
 seen.clear()
 
 return result
 
 seen[num] = i
 
 # Optional: Limit hash table size in streaming scenarios
 if len(seen) > expected_size:
 # This is a heuristic; adjust based on your use case
 pass
 
 raise ValueError("No solution found")
``

---

## Connections to Real-World Systems

### 1. Feature Stores in ML

**Problem:** For each user request, quickly look up precomputed features.

``python
class FeatureStore:
 def __init__(self):
 # Hash table mapping user_id → features
 self.user_features = {}
 
 def get_features(self, user_id: int) -> dict:
 """O(1) lookup, just like Two Sum!"""
 if user_id in self.user_features:
 return self.user_features[user_id]
 
 # Compute and cache
 features = self._compute_features(user_id)
 self.user_features[user_id] = features
 return features
 
 def _compute_features(self, user_id: int) -> dict:
 # Expensive computation
 return {
 'age': 28,
 'engagement_score': 0.75,
 'last_active': '2025-10-13'
 }

# Usage
store = FeatureStore()
features = store.get_features(user_id=12345) # O(1)!
``

**Scale:** Feature stores at companies like Uber and Netflix serve millions of lookups per second using this exact pattern.

### 2. Embedding Lookups

**Problem:** Given a token ID, retrieve its embedding vector.

``python
import numpy as np

class EmbeddingTable:
 def __init__(self, vocab_size: int, embedding_dim: int):
 # Hash table: token_id → embedding vector
 self.embeddings = {}
 
 # Initialize with random embeddings
 for token_id in range(vocab_size):
 self.embeddings[token_id] = np.random.randn(embedding_dim)
 
 def lookup(self, token_id: int) -> np.ndarray:
 """O(1) embedding lookup"""
 return self.embeddings[token_id]

# Usage in neural network
embedding_table = EmbeddingTable(vocab_size=50000, embedding_dim=300)

# During inference
token_id = 4567
embedding = embedding_table.lookup(token_id) # O(1)!
``

**Real systems:** GPT, BERT, and other transformer models perform millions of embedding lookups per second.

### 3. Cache Systems

**Problem:** Store frequently accessed data for O(1) retrieval.

``python
from collections import OrderedDict

class LRUCache:
 def __init__(self, capacity: int):
 self.cache = OrderedDict()
 self.capacity = capacity
 
 def get(self, key: int) -> int:
 """O(1) lookup with LRU tracking"""
 if key not in self.cache:
 return -1
 
 # Move to end (mark as recently used)
 self.cache.move_to_end(key)
 return self.cache[key]
 
 def put(self, key: int, value: int) -> None:
 """O(1) insertion with LRU eviction"""
 if key in self.cache:
 # Update existing key
 self.cache.move_to_end(key)
 else:
 # Add new key
 if len(self.cache) >= self.capacity:
 # Evict least recently used
 self.cache.popitem(last=False)
 
 self.cache[key] = value

# Usage
cache = LRUCache(capacity=1000)
cache.put(user_id=123, value={"name": "Alice"})
user_data = cache.get(user_id=123) # O(1)!
``

**Production examples:** Redis, Memcached, and CDN caches use hash tables for O(1) lookups.

### 4. Deduplication

**Problem:** Remove duplicate entries from a stream of data.

``python
def deduplicate_stream(data_stream):
 """
 Remove duplicates from stream in O(n) time
 """
 seen = set() # Hash set (hash table with no values)
 unique_items = []
 
 for item in data_stream:
 if item not in seen: # O(1) check
 unique_items.append(item)
 seen.add(item) # O(1) insertion
 
 return unique_items

# Usage in data pipeline
raw_events = [
 {"user_id": 1, "action": "click"},
 {"user_id": 2, "action": "view"},
 {"user_id": 1, "action": "click"}, # Duplicate
 {"user_id": 3, "action": "purchase"}
]

unique_events = deduplicate_stream(raw_events)
# O(n) time instead of O(n²) with nested loops!
``

### 5. Join Operations in Databases

**Problem:** SQL JOIN operations use hash tables for efficiency.

``python
def hash_join(table1, table2, join_key):
 """
 Simplified hash join algorithm (used in databases)
 
 Similar to Two Sum: build hash table from one table,
 probe with the other
 """
 # Build phase: Create hash table from smaller table
 hash_table = {}
 for row in table1:
 key = row[join_key]
 if key not in hash_table:
 hash_table[key] = []
 hash_table[key].append(row)
 
 # Probe phase: Lookup each row from table2
 result = []
 for row in table2:
 key = row[join_key]
 if key in hash_table: # O(1) lookup!
 for matching_row in hash_table[key]:
 result.append({**matching_row, **row})
 
 return result

# Example
users = [
 {"user_id": 1, "name": "Alice"},
 {"user_id": 2, "name": "Bob"}
]

orders = [
 {"user_id": 1, "order_id": 101},
 {"user_id": 1, "order_id": 102},
 {"user_id": 2, "order_id": 103}
]

# O(n + m) hash join vs O(n * m) nested loop join
joined = hash_join(users, orders, "user_id")
``

**Database systems** (PostgreSQL, MySQL) use hash joins when appropriate, achieving massive speedups over nested loop joins.

---

## When NOT to Use Hash Tables

Despite their power, hash tables aren't always the answer:

### 1. Need Sorted Order

``python
# If you need results in sorted order, hash tables won't help
# Use sorting + two pointers instead

def twoSumSortedResult(nums, target):
 # Create list of (value, index) pairs
 indexed = [(num, i) for i, num in enumerate(nums)]
 
 # Sort by value
 indexed.sort()
 
 left, right = 0, len(indexed) - 1
 while left < right:
 curr_sum = indexed[left][0] + indexed[right][0]
 if curr_sum == target:
 return sorted([indexed[left][1], indexed[right][1]])
 elif curr_sum < target:
 left += 1
 else:
 right -= 1
 
 return []
``

### 2. Memory Constrained

``python
# Embedded systems, mobile devices with limited memory
# If O(n) extra space is too much, use two pointers on sorted array

def twoSumLowMemory(nums, target):
 # Sort in-place (if allowed to modify input)
 sorted_indices = sorted(range(len(nums)), key=lambda i: nums[i])
 
 left, right = 0, len(nums) - 1
 while left < right:
 l_idx, r_idx = sorted_indices[left], sorted_indices[right]
 curr_sum = nums[l_idx] + nums[r_idx]
 
 if curr_sum == target:
 return [l_idx, r_idx]
 elif curr_sum < target:
 left += 1
 else:
 right -= 1
 
 return []
``

### 3. Small Inputs

``python
# For n < 100, brute force might be faster
# No hash table overhead, better cache locality

def twoSumSmallInput(nums, target):
 if len(nums) < 100:
 # Brute force for small inputs
 for i in range(len(nums)):
 for j in range(i+1, len(nums)):
 if nums[i] + nums[j] == target:
 return [i, j]
 else:
 # Hash table for large inputs
 return twoSum(nums, target)
``

---

## Testing and Validation

### Comprehensive Test Suite

``python
import unittest

class TestTwoSum(unittest.TestCase):
 def test_basic_case(self):
 """Test example from problem statement"""
 nums = [2, 7, 11, 15]
 target = 9
 result = twoSum(nums, target)
 self.assertEqual(sorted(result), [0, 1])
 self.assertEqual(nums[result[0]] + nums[result[1]], target)
 
 def test_duplicates(self):
 """Test with duplicate values"""
 nums = [3, 3]
 target = 6
 result = twoSum(nums, target)
 self.assertEqual(sorted(result), [0, 1])
 
 def test_negative_numbers(self):
 """Test with negative numbers"""
 nums = [-1, -2, -3, -4, -5]
 target = -8
 result = twoSum(nums, target)
 self.assertEqual(nums[result[0]] + nums[result[1]], target)
 
 def test_zero_target(self):
 """Test with zero as target"""
 nums = [-3, 0, 3, 4]
 target = 0
 result = twoSum(nums, target)
 self.assertEqual(nums[result[0]] + nums[result[1]], 0)
 
 def test_large_numbers(self):
 """Test with large numbers"""
 nums = [1000000000, -1000000000, 1]
 target = 1
 result = twoSum(nums, target)
 self.assertEqual(nums[result[0]] + nums[result[1]], 1)
 
 def test_minimum_size(self):
 """Test with minimum array size"""
 nums = [1, 2]
 target = 3
 result = twoSum(nums, target)
 self.assertEqual(sorted(result), [0, 1])
 
 def test_unordered(self):
 """Test that order doesn't matter"""
 nums = [15, 11, 7, 2]
 target = 9
 result = twoSum(nums, target)
 self.assertEqual(nums[result[0]] + nums[result[1]], 9)
 
 def test_performance(self):
 """Test performance with large input"""
 import time
 
 # Generate large array
 nums = list(range(10000))
 target = 19999 # Last two elements
 
 start = time.time()
 result = twoSum(nums, target)
 elapsed = time.time() - start
 
 self.assertEqual(nums[result[0]] + nums[result[1]], target)
 self.assertLess(elapsed, 0.1, "Should complete in < 100ms")

if __name__ == '__main__':
 unittest.main()
``

---

## Summary and Key Takeaways

### Core Concepts

✅ **Hash tables enable O(1) lookups**, reducing O(n²) to O(n)
✅ **Space-time tradeoff**: We use O(n) space to achieve O(n) time
✅ **Build as you go**: No need to pre-populate the hash table
✅ **Complement pattern**: For each element, check if its "partner" exists

### When to Use This Pattern

**Use hash tables when:**
- Need fast lookups (O(1) vs O(n))
- Memory is available
- Order doesn't matter
- Working with large datasets

**Use two pointers when:**
- Input is already sorted
- Space is constrained
- Need sorted output
- Input is small (n < 100)

### Production Lessons

1. **Always validate inputs** in production code
2. **Consider edge cases** (empty, single element, duplicates, negatives)
3. **Monitor performance** with logging and metrics
4. **Handle errors gracefully** with clear error messages
5. **Document assumptions** (e.g., "exactly one solution exists")

### Related Patterns

This hash table pattern appears in:
- **3Sum, 4Sum, K-Sum** problems
- **Feature stores** in ML systems
- **Embedding tables** in NLP
- **Cache systems** (LRU, LFU)
- **Deduplication** pipelines
- **Database joins** (hash join)

### Further Practice

**Next steps:**
1. Solve [3Sum](https://leetcode.com/problems/3sum/) (extends Two Sum)
2. Implement [LRU Cache](https://leetcode.com/problems/lru-cache/) (uses hash table + doubly linked list)
3. Study [Group Anagrams](https://leetcode.com/problems/group-anagrams/) (hash table with string keys)
4. Read about [Consistent Hashing](https://en.wikipedia.org/wiki/Consistent_hashing) (used in distributed systems)

**Books and resources:**
- *Introduction to Algorithms* (CLRS) - Chapter on Hash Tables
- *Designing Data-Intensive Applications* by Martin Kleppmann
- *The Algorithm Design Manual* by Steven Skiena

---

## Conclusion

Two Sum may seem simple, but it introduces one of the most important patterns in computer science: **using hash tables to trade space for time**. This pattern powers countless production systems, from recommendation engines serving millions of users to real-time analytics processing billions of events.

The next time you reach for a nested loop, ask yourself: "Could a hash table make this O(n) instead of O(n²)?" Often, the answer is yes and the performance difference can be transformational.

Remember: **Algorithms aren't just for interviews. They're the foundation of scalable, efficient production systems.**

Happy coding! 🚀

---

**Originally published at:** [arunbaby.com/dsa/0001-two-sum](https://www.arunbaby.com/dsa/0001-two-sum/)

*If you found this helpful, consider sharing it with others who might benefit.*
