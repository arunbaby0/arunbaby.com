---
title: "Group Anagrams"
day: 15
collection: dsa
categories:
  - dsa
tags:
  - hash-table
  - string
  - sorting
  - grouping
  - anagrams
  - medium-easy
subdomain: "Hash Tables & Strings"
tech_stack: [Python]
scale: "O(NK log K) time, O(NK) space"
companies: [Google, Meta, Amazon, Microsoft, Apple, Bloomberg]
related_dsa_day: 15
related_ml_day: 15
related_speech_day: 15
---

**Master hash-based grouping to solve anagrams—the foundation of clustering systems and speaker diarization in production ML.**

## Problem Statement

Given an array of strings `strs`, group the **anagrams** together. You can return the answer in **any order**.

An **anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

### Examples

**Example 1:**
```
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

**Example 2:**
```
Input: strs = [""]
Output: [[""]]
```

**Example 3:**
```
Input: strs = ["a"]
Output: [["a"]]
```

### Constraints

- `1 <= strs.length <= 10^4`
- `0 <= strs[i].length <= 100`
- `strs[i]` consists of lowercase English letters

## Understanding the Problem

This is a **fundamental grouping problem** that teaches us:
1. **How to identify similar items** (anagrams share same characters)
2. **How to use hash tables for efficient grouping**
3. **How to design good hash keys** for complex objects
4. **Pattern recognition** for clustering algorithms

### What Are Anagrams?

Two strings are anagrams if they contain the **same characters with the same frequencies**, just in different order.

**Examples:**
- "listen" and "silent" → anagrams (same letters: e,i,l,n,s,t)
- "eat", "tea", "ate" → all anagrams
- "cat" and "rat" → NOT anagrams (different letters)

### Key Insight

Anagrams share a **unique signature**:
- Sorted characters: "eat" → "aet", "tea" → "aet"
- Character count: both have {a:1, e:1, t:1}

We can use this signature as a **hash key** to group anagrams together.

### Why This Problem Matters

1. **Hash table mastery:** Learn to design effective hash keys
2. **Grouping pattern:** Fundamental for clustering algorithms
3. **String manipulation:** Common in text processing
4. **Real-world applications:**
   - Text deduplication
   - Spam detection (similar messages)
   - DNA sequence analysis
   - Document clustering
   - Speaker identification (similar voice characteristics)

### The Clustering Connection

The grouping pattern in this problem is **identical** to clustering in ML:

| Group Anagrams | Clustering Systems | Speaker Diarization |
|----------------|-------------------|---------------------|
| Group strings by characters | Group data points by features | Group speech by speaker |
| Hash key: sorted string | Cluster ID: centroid | Speaker ID: voice embedding |
| O(NK log K) grouping | O(N × K) clustering | O(N × M) diarization |

All three use **hash-based or similarity-based grouping** to organize items.

## Approach 1: Brute Force - Compare All Pairs

### Intuition

Compare every pair of strings to check if they're anagrams, then group them.

### Implementation

```python
from typing import List
from collections import defaultdict

def groupAnagrams_bruteforce(strs: List[str]) -> List[List[str]]:
    """
    Brute force: compare all pairs.
    
    Time: O(N^2 × K) where N = number of strings, K = max string length
    Space: O(NK)
    
    Why this approach?
    - Simple to understand
    - Shows the naive solution
    - Demonstrates need for optimization
    
    Problem:
    - Too slow for large inputs
    - Redundant comparisons
    """
    def are_anagrams(s1: str, s2: str) -> bool:
        """Check if two strings are anagrams."""
        if len(s1) != len(s2):
            return False
        
        # Count characters in each string
        from collections import Counter
        return Counter(s1) == Counter(s2)
    
    # Track which strings have been grouped
    grouped = [False] * len(strs)
    result = []
    
    for i in range(len(strs)):
        if grouped[i]:
            continue
        
        # Start new group with current string
        group = [strs[i]]
        grouped[i] = True
        
        # Find all anagrams of current string
        for j in range(i + 1, len(strs)):
            if not grouped[j] and are_anagrams(strs[i], strs[j]):
                group.append(strs[j])
                grouped[j] = True
        
        result.append(group)
    
    return result


# Test
test_input = ["eat","tea","tan","ate","nat","bat"]
print(groupAnagrams_bruteforce(test_input))
# Output: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

### Analysis

**Time Complexity: O(N² × K)**
- N² pairs to compare
- Each comparison: O(K) to count characters

**Space Complexity: O(NK)**
- Store all strings in result

**For N=10,000, K=100:**
- Operations: 10,000² × 100 = 10 billion
- Too slow!

## Approach 2: Sorting as Hash Key (Standard Solution)

### The Key Insight

**Anagrams become identical when sorted!**

- "eat" → sort → "aet"
- "tea" → sort → "aet"
- "ate" → sort → "aet"

We can use the **sorted string as a hash key** to group anagrams.

### Implementation

```python
from collections import defaultdict
from typing import List

def groupAnagrams(strs: List[str]) -> List[List[str]]:
    """
    Optimal solution using sorted string as hash key.
    
    Time: O(N × K log K) where N = number of strings, K = max string length
    Space: O(NK)
    
    Algorithm:
    1. For each string, create hash key by sorting it
    2. Use hash table to group strings with same key
    3. Return groups as list
    
    Why this works:
    - Sorting is canonical representation of anagram
    - Hash table provides O(1) lookup
    - Single pass through all strings
    """
    # Hash table: sorted_string -> list of original strings
    anagram_map = defaultdict(list)
    
    for s in strs:
        # Sort the string to create hash key
        # "eat" -> ['a', 'e', 't'] -> "aet"
        sorted_str = ''.join(sorted(s))
        
        # Group by sorted key
        anagram_map[sorted_str].append(s)
    
    # Return all groups
    return list(anagram_map.values())


# Test cases
test_cases = [
    ["eat","tea","tan","ate","nat","bat"],
    [""],
    ["a"],
    ["abc", "bca", "cab", "xyz", "zyx", "yxz"],
]

for test in test_cases:
    result = groupAnagrams(test)
    print(f"Input: {test}")
    print(f"Output: {result}\n")
```

### Step-by-Step Visualization

```
Input: ["eat","tea","tan","ate","nat","bat"]

Step 1: Process "eat"
  sorted("eat") = "aet"
  anagram_map = {"aet": ["eat"]}

Step 2: Process "tea"
  sorted("tea") = "aet"
  anagram_map = {"aet": ["eat", "tea"]}

Step 3: Process "tan"
  sorted("tan") = "ant"
  anagram_map = {"aet": ["eat", "tea"], "ant": ["tan"]}

Step 4: Process "ate"
  sorted("ate") = "aet"
  anagram_map = {"aet": ["eat", "tea", "ate"], "ant": ["tan"]}

Step 5: Process "nat"
  sorted("nat") = "ant"
  anagram_map = {"aet": ["eat", "tea", "ate"], "ant": ["tan", "nat"]}

Step 6: Process "bat"
  sorted("bat") = "abt"
  anagram_map = {
    "aet": ["eat", "tea", "ate"],
    "ant": ["tan", "nat"],
    "abt": ["bat"]
  }

Output: [["eat","tea","ate"], ["tan","nat"], ["bat"]]
```

## Approach 3: Character Count as Hash Key (Optimal for Large K)

### Alternative Hash Key

Instead of sorting, we can use **character frequencies** as the hash key.

**Why?** When strings are very long (K >> 26), counting is faster than sorting.

### Implementation

```python
from collections import defaultdict
from typing import List

def groupAnagrams_count(strs: List[str]) -> List[List[str]]:
    """
    Use character count as hash key.
    
    Time: O(NK) where N = number of strings, K = max string length
    Space: O(NK)
    
    Advantage over sorting:
    - O(K) instead of O(K log K) per string
    - Better for very long strings
    
    Hash key format:
    - Tuple of 26 integers (a-z counts)
    - e.g., "aab" -> (2, 1, 0, 0, ..., 0)
    """
    anagram_map = defaultdict(list)
    
    for s in strs:
        # Count characters (a-z)
        count = [0] * 26
        
        for char in s:
            count[ord(char) - ord('a')] += 1
        
        # Use tuple as hash key (lists aren't hashable)
        key = tuple(count)
        
        anagram_map[key].append(s)
    
    return list(anagram_map.values())


# Test
test = ["eat","tea","tan","ate","nat","bat"]
result = groupAnagrams_count(test)
print(f"Result: {result}")
```

### Character Count Visualization

```
"eat" -> count array:
Index:  0  1  2  3  4  5  ... 19 20 21 ...
Char:   a  b  c  d  e  f  ... t  u  v  ...
Count: [1, 0, 0, 0, 1, 0, ... 1, 0, 0, ...]
       (a=1, e=1, t=1)

Key = (1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0)

"tea" -> same count array -> same key!
"tan" -> different count array -> different key
```

## Implementation: Production-Grade Solution

```python
from collections import defaultdict
from typing import List, Dict, Optional
import logging
from enum import Enum

class GroupingStrategy(Enum):
    """Strategy for creating hash keys."""
    SORTED = "sorted"
    COUNT = "count"
    PRIME = "prime"  # Advanced: prime number hash

class AnagramGrouper:
    """
    Production-ready anagram grouper with multiple strategies.
    
    Features:
    - Multiple grouping strategies
    - Input validation
    - Performance metrics
    - Detailed logging
    """
    
    def __init__(self, strategy: GroupingStrategy = GroupingStrategy.SORTED):
        """
        Initialize grouper.
        
        Args:
            strategy: Grouping strategy to use
        """
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # Metrics
        self.comparisons = 0
        self.groups_created = 0
    
    def group_anagrams(self, strs: List[str]) -> List[List[str]]:
        """
        Group anagrams using selected strategy.
        
        Args:
            strs: List of strings to group
            
        Returns:
            List of groups (each group is a list of anagrams)
            
        Raises:
            ValueError: If input is invalid
        """
        # Validate input
        if not isinstance(strs, list):
            raise ValueError("Input must be a list of strings")
        
        if not all(isinstance(s, str) for s in strs):
            raise ValueError("All elements must be strings")
        
        # Reset metrics
        self.comparisons = 0
        self.groups_created = 0
        
        # Choose strategy
        if self.strategy == GroupingStrategy.SORTED:
            result = self._group_by_sorted(strs)
        elif self.strategy == GroupingStrategy.COUNT:
            result = self._group_by_count(strs)
        elif self.strategy == GroupingStrategy.PRIME:
            result = self._group_by_prime(strs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.groups_created = len(result)
        
        self.logger.info(
            f"Grouped {len(strs)} strings into {self.groups_created} groups "
            f"using {self.strategy.value} strategy"
        )
        
        return result
    
    def _group_by_sorted(self, strs: List[str]) -> List[List[str]]:
        """Group using sorted string as key."""
        anagram_map = defaultdict(list)
        
        for s in strs:
            key = ''.join(sorted(s))
            anagram_map[key].append(s)
            self.comparisons += 1
        
        return list(anagram_map.values())
    
    def _group_by_count(self, strs: List[str]) -> List[List[str]]:
        """Group using character count as key."""
        anagram_map = defaultdict(list)
        
        for s in strs:
            # Count characters
            count = [0] * 26
            for char in s:
                if 'a' <= char <= 'z':
                    count[ord(char) - ord('a')] += 1
                else:
                    # Handle uppercase or non-alphabetic
                    char_lower = char.lower()
                    if 'a' <= char_lower <= 'z':
                        count[ord(char_lower) - ord('a')] += 1
            
            key = tuple(count)
            anagram_map[key].append(s)
            self.comparisons += 1
        
        return list(anagram_map.values())
    
    def _group_by_prime(self, strs: List[str]) -> List[List[str]]:
        """
        Group using prime number hash.
        
        Assign each letter a prime number:
        a=2, b=3, c=5, d=7, e=11, ...
        
        Hash = product of primes for each character.
        
        Advantage: Unique hash for each anagram group
        Disadvantage: Can overflow for long strings
        """
        # Prime numbers for a-z
        primes = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97, 101
        ]
        
        anagram_map = defaultdict(list)
        
        for s in strs:
            # Calculate prime product
            hash_value = 1
            
            for char in s:
                if 'a' <= char <= 'z':
                    hash_value *= primes[ord(char) - ord('a')]
            
            anagram_map[hash_value].append(s)
            self.comparisons += 1
        
        return list(anagram_map.values())
    
    def find_anagrams_of(self, target: str, strs: List[str]) -> List[str]:
        """
        Find all anagrams of a target string in a list.
        
        Args:
            target: Target string
            strs: List of strings to search
            
        Returns:
            List of strings that are anagrams of target
        """
        # Get hash key for target
        if self.strategy == GroupingStrategy.SORTED:
            target_key = ''.join(sorted(target))
        elif self.strategy == GroupingStrategy.COUNT:
            count = [0] * 26
            for char in target:
                if 'a' <= char <= 'z':
                    count[ord(char) - ord('a')] += 1
            target_key = tuple(count)
        else:
            target_key = None
        
        # Find matching strings
        result = []
        
        for s in strs:
            if self.strategy == GroupingStrategy.SORTED:
                key = ''.join(sorted(s))
            elif self.strategy == GroupingStrategy.COUNT:
                count = [0] * 26
                for char in s:
                    if 'a' <= char <= 'z':
                        count[ord(char) - ord('a')] += 1
                key = tuple(count)
            
            if key == target_key:
                result.append(s)
        
        return result
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        return {
            "strategy": self.strategy.value,
            "comparisons": self.comparisons,
            "groups_created": self.groups_created,
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test data
    test_strs = ["eat","tea","tan","ate","nat","bat"]
    
    # Test different strategies
    for strategy in GroupingStrategy:
        print(f"\n=== Testing {strategy.value} strategy ===")
        
        grouper = AnagramGrouper(strategy=strategy)
        result = grouper.group_anagrams(test_strs)
        
        print(f"Result: {result}")
        print(f"Stats: {grouper.get_stats()}")
        
        # Test finding anagrams
        anagrams_of_eat = grouper.find_anagrams_of("eat", test_strs)
        print(f"Anagrams of 'eat': {anagrams_of_eat}")
```

## Testing

### Comprehensive Test Suite

```python
import pytest
from typing import List

class TestAnagramGrouper:
    """Comprehensive test suite for anagram grouping."""
    
    @pytest.fixture
    def grouper(self):
        return AnagramGrouper(strategy=GroupingStrategy.SORTED)
    
    def test_basic_examples(self, grouper):
        """Test basic examples from problem."""
        # Example 1
        result = grouper.group_anagrams(["eat","tea","tan","ate","nat","bat"])
        
        # Convert to sets for comparison (order doesn't matter)
        result_sets = [set(group) for group in result]
        expected_sets = [
            {"eat", "tea", "ate"},
            {"tan", "nat"},
            {"bat"}
        ]
        
        assert len(result_sets) == len(expected_sets)
        for expected in expected_sets:
            assert expected in result_sets
    
    def test_empty_string(self, grouper):
        """Test with empty string."""
        result = grouper.group_anagrams([""])
        assert result == [[""]]
    
    def test_single_string(self, grouper):
        """Test with single string."""
        result = grouper.group_anagrams(["a"])
        assert result == [["a"]]
    
    def test_no_anagrams(self, grouper):
        """Test when no strings are anagrams."""
        result = grouper.group_anagrams(["abc", "def", "ghi"])
        assert len(result) == 3
        assert all(len(group) == 1 for group in result)
    
    def test_all_anagrams(self, grouper):
        """Test when all strings are anagrams."""
        result = grouper.group_anagrams(["abc", "bca", "cab", "acb"])
        assert len(result) == 1
        assert len(result[0]) == 4
    
    def test_long_strings(self, grouper):
        """Test with long strings."""
        long1 = "a" * 100
        long2 = "a" * 100
        long3 = "b" * 100
        
        result = grouper.group_anagrams([long1, long2, long3])
        assert len(result) == 2
    
    def test_strategy_equivalence(self):
        """Test that all strategies produce equivalent results."""
        test_input = ["eat","tea","tan","ate","nat","bat"]
        
        results = {}
        
        for strategy in GroupingStrategy:
            grouper = AnagramGrouper(strategy=strategy)
            result = grouper.group_anagrams(test_input)
            
            # Convert to frozensets for comparison
            result_sets = frozenset(
                frozenset(group) for group in result
            )
            results[strategy] = result_sets
        
        # All strategies should produce same groupings
        assert len(set(results.values())) == 1
    
    def test_case_insensitive(self):
        """Test case handling."""
        grouper = AnagramGrouper(strategy=GroupingStrategy.COUNT)
        result = grouper.group_anagrams(["Eat", "Tea", "eat"])
        
        # Should group regardless of case
        assert len(result) <= 2  # Depends on implementation
    
    def test_invalid_input(self, grouper):
        """Test input validation."""
        with pytest.raises(ValueError):
            grouper.group_anagrams("not a list")
        
        with pytest.raises(ValueError):
            grouper.group_anagrams([1, 2, 3])
    
    def test_find_anagrams(self, grouper):
        """Test finding specific anagrams."""
        strs = ["eat","tea","tan","ate","nat","bat"]
        
        anagrams = grouper.find_anagrams_of("eat", strs)
        assert set(anagrams) == {"eat", "tea", "ate"}
        
        anagrams = grouper.find_anagrams_of("tan", strs)
        assert set(anagrams) == {"tan", "nat"}


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Complexity Analysis

### Sorting Approach

**Time Complexity: O(N × K log K)**
- N strings to process
- Each string of length K needs sorting: O(K log K)
- Hash table operations: O(1) average

**Space Complexity: O(NK)**
- Store all strings in hash table: O(NK)
- Hash keys: O(NK)

### Character Count Approach

**Time Complexity: O(NK)**
- N strings to process
- Each string of length K needs counting: O(K)
- Better than sorting for large K!

**Space Complexity: O(NK)**
- Store all strings: O(NK)
- Hash keys (26-element tuples): O(N)

### Comparison

| Approach | Time | Space | Best For |
|----------|------|-------|----------|
| Brute Force | O(N²K) | O(NK) | Never (too slow) |
| Sorting | O(NK log K) | O(NK) | Short to medium strings |
| Count | O(NK) | O(NK) | Long strings (K >> 26) |
| Prime | O(NK) | O(NK) | Theoretical interest |

**When K is small (≤100):** Sorting is simpler and sufficient.
**When K is large (>1000):** Character count is faster.

## Production Considerations

### 1. Unicode Support

```python
def group_anagrams_unicode(strs: List[str]) -> List[List[str]]:
    """
    Group anagrams with full Unicode support.
    
    Handles:
    - Non-ASCII characters
    - Emojis
    - Accented characters
    """
    from collections import Counter
    
    anagram_map = defaultdict(list)
    
    for s in strs:
        # Use Counter for Unicode-safe counting
        # Convert to frozenset of items for hashing
        key = tuple(sorted(Counter(s).items()))
        anagram_map[key].append(s)
    
    return list(anagram_map.values())


# Test with Unicode
unicode_strs = ["café", "éfac", "hello", "हेलो", "😀😁", "😁😀"]
result = group_anagrams_unicode(unicode_strs)
print(result)
```

### 2. Case-Insensitive Grouping

```python
def group_anagrams_case_insensitive(strs: List[str]) -> List[List[str]]:
    """Group anagrams ignoring case."""
    anagram_map = defaultdict(list)
    
    for s in strs:
        # Normalize to lowercase before sorting
        key = ''.join(sorted(s.lower()))
        anagram_map[key].append(s)
    
    return list(anagram_map.values())


# Test
result = group_anagrams_case_insensitive(["Eat", "Tea", "eat", "tea"])
print(result)  # All in one group
```

### 3. Streaming / Online Grouping

```python
class StreamingAnagramGrouper:
    """
    Group anagrams in streaming fashion.
    
    Useful when:
    - Data doesn't fit in memory
    - Processing real-time stream
    - Need incremental results
    """
    
    def __init__(self):
        self.groups = defaultdict(list)
        self.group_ids = {}
        self.next_id = 0
    
    def add_string(self, s: str) -> int:
        """
        Add string to grouping.
        
        Returns:
            Group ID for this string
        """
        key = ''.join(sorted(s))
        
        if key not in self.group_ids:
            self.group_ids[key] = self.next_id
            self.next_id += 1
        
        group_id = self.group_ids[key]
        self.groups[group_id].append(s)
        
        return group_id
    
    def get_group(self, group_id: int) -> List[str]:
        """Get strings in a specific group."""
        return self.groups.get(group_id, [])
    
    def get_all_groups(self) -> List[List[str]]:
        """Get all groups."""
        return list(self.groups.values())


# Usage
streamer = StreamingAnagramGrouper()

for word in ["eat", "tea", "tan", "ate"]:
    group_id = streamer.add_string(word)
    print(f"'{word}' -> group {group_id}")

print(f"Final groups: {streamer.get_all_groups()}")
```

### 4. Performance Monitoring

```python
import time
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Performance metrics for grouping operation."""
    total_strings: int
    total_groups: int
    execution_time_ms: float
    avg_group_size: float
    largest_group_size: int
    
    def __str__(self):
        return (
            f"Performance Metrics:\n"
            f"  Strings processed: {self.total_strings}\n"
            f"  Groups created: {self.total_groups}\n"
            f"  Execution time: {self.execution_time_ms:.2f}ms\n"
            f"  Avg group size: {self.avg_group_size:.2f}\n"
            f"  Largest group: {self.largest_group_size}"
        )


def group_anagrams_with_metrics(strs: List[str]) -> tuple[List[List[str]], PerformanceMetrics]:
    """Group anagrams and return performance metrics."""
    start_time = time.perf_counter()
    
    # Group anagrams
    anagram_map = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        anagram_map[key].append(s)
    
    result = list(anagram_map.values())
    
    # Calculate metrics
    execution_time = (time.perf_counter() - start_time) * 1000
    
    group_sizes = [len(group) for group in result]
    
    metrics = PerformanceMetrics(
        total_strings=len(strs),
        total_groups=len(result),
        execution_time_ms=execution_time,
        avg_group_size=sum(group_sizes) / len(group_sizes) if group_sizes else 0,
        largest_group_size=max(group_sizes) if group_sizes else 0
    )
    
    return result, metrics
```

## Connections to ML Systems

The **hash-based grouping** pattern from this problem is fundamental to clustering systems:

### 1. Clustering Systems

**Similarity to Group Anagrams:**
- **Anagrams:** Group strings by character composition
- **Clustering:** Group data points by feature similarity

```python
import numpy as np
from collections import defaultdict

class SimpleClusterer:
    """
    Simple clustering using hash-based grouping.
    
    Similar to anagram grouping:
    - Hash key: quantized feature vector
    - Grouping: points with same hash go to same cluster
    """
    
    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
    
    def cluster(self, points: np.ndarray) -> List[List[int]]:
        """
        Cluster points using locality-sensitive hashing.
        
        Args:
            points: Array of shape (n_samples, n_features)
            
        Returns:
            List of clusters (each cluster is list of point indices)
        """
        clusters = defaultdict(list)
        
        for idx, point in enumerate(points):
            # Create hash key by quantizing features
            # (similar to sorting characters in anagram)
            hash_key = tuple(
                int(feature * self.num_bins) for feature in point
            )
            
            clusters[hash_key].append(idx)
        
        return list(clusters.values())


# Example: Cluster 2D points
points = np.array([
    [0.1, 0.1],  # Cluster 1
    [0.12, 0.11],  # Cluster 1
    [0.8, 0.9],  # Cluster 2
    [0.82, 0.88],  # Cluster 2
])

clusterer = SimpleClusterer(num_bins=10)
clusters = clusterer.cluster(points)
print(f"Clusters: {clusters}")
```

### 2. Duplicate Detection

```python
class DocumentDeduplicator:
    """
    Detect duplicate or near-duplicate documents.
    
    Uses same pattern as anagram grouping:
    - Hash key: document signature
    - Grouping: similar documents
    """
    
    def __init__(self):
        self.doc_groups = defaultdict(list)
    
    def add_document(self, doc_id: str, text: str):
        """Add document to deduplication system."""
        # Create signature (like anagram key)
        signature = self._create_signature(text)
        self.doc_groups[signature].append(doc_id)
    
    def _create_signature(self, text: str) -> str:
        """
        Create document signature.
        
        Methods:
        1. Word frequency (like character count)
        2. N-gram hashing
        3. MinHash
        """
        # Simple: use sorted bag of words
        words = text.lower().split()
        return ' '.join(sorted(words))
    
    def find_duplicates(self) -> List[List[str]]:
        """Find groups of duplicate documents."""
        return [
            group for group in self.doc_groups.values()
            if len(group) > 1
        ]


# Usage
dedup = DocumentDeduplicator()
dedup.add_document("doc1", "the quick brown fox")
dedup.add_document("doc2", "quick brown fox the")  # Duplicate!
dedup.add_document("doc3", "hello world")

duplicates = dedup.find_duplicates()
print(f"Duplicate groups: {duplicates}")
```

### 3. Feature Hashing

```python
class FeatureHasher:
    """
    Hash high-dimensional features to lower dimensions.
    
    Similar to anagram grouping:
    - Hash collisions group similar features
    - Dimensionality reduction via hashing
    """
    
    def __init__(self, n_features: int = 100):
        self.n_features = n_features
    
    def transform(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """
        Transform feature dictionary to fixed-size vector.
        
        Args:
            feature_dict: {feature_name: value}
            
        Returns:
            Dense feature vector
        """
        # Initialize vector
        vector = np.zeros(self.n_features)
        
        # Hash each feature to a bin
        for feature_name, value in feature_dict.items():
            # Hash feature name to index
            # (like sorting string to create key)
            hash_idx = hash(feature_name) % self.n_features
            vector[hash_idx] += value
        
        return vector


# Example: Text features
hasher = FeatureHasher(n_features=10)

doc1_features = {"word_cat": 2, "word_dog": 1, "word_house": 1}
doc2_features = {"word_cat": 1, "word_dog": 2, "word_car": 1}

vec1 = hasher.transform(doc1_features)
vec2 = hasher.transform(doc2_features)

print(f"Vector 1: {vec1}")
print(f"Vector 2: {vec2}")
print(f"Similarity: {np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))}")
```

### Key Parallels

| Group Anagrams | Clustering/ML |
|----------------|---------------|
| Sorted string as hash key | Feature signature as hash key |
| Group identical keys | Group similar signatures |
| O(1) hash lookup | O(1) hash lookup |
| Character frequency | Feature frequency |
| String similarity | Data point similarity |

## Interview Strategy

### How to Approach in an Interview

**1. Clarify (1 min)**
```
- Are all strings lowercase? (Yes, per constraints)
- Can strings be empty? (Yes)
- Does order of output matter? (No)
- Memory constraints? (Reasonable for N ≤ 10^4)
```

**2. Explain Intuition (2 min)**
```
"Anagrams have the same characters, just rearranged. If we sort
each string, anagrams become identical. We can use this sorted
string as a hash key to group them together in O(1) time per lookup."
```

**3. Discuss Approaches (2 min)**
```
1. Brute force: Compare all pairs - O(N²K)
2. Sorting as key: Sort each string - O(NK log K)
3. Count as key: Count characters - O(NK)

I'll implement approach 2 (sorting) as it's simpler and efficient
enough for the constraints.
```

**4. Code (10 min)**
- Clear variable names
- Add comments
- Handle edge cases

**5. Test (3 min)**
- Walk through example
- Edge cases: empty string, single string

**6. Optimize (2 min)**
- Discuss count approach for very long strings

### Common Mistakes

1. **Forgetting to handle empty strings**
   ```python
   # Wrong: crashes on empty string
   key = sorted(s)[0]
   
   # Correct: works for empty
   key = ''.join(sorted(s))
   ```

2. **Using list as hash key**
   ```python
   # Wrong: lists aren't hashable
   key = sorted(s)  # Returns list
   
   # Correct: convert to string or tuple
   key = ''.join(sorted(s))
   key = tuple(sorted(s))
   ```

3. **Not considering case sensitivity**
   - Problem says lowercase only, but clarify in interview

4. **Inefficient anagram checking in brute force**
   ```python
   # Inefficient
   def are_anagrams(s1, s2):
       return sorted(s1) == sorted(s2)
   
   # Better: use Counter
   from collections import Counter
   return Counter(s1) == Counter(s2)
   ```

### Follow-up Questions

**Q1: How would you find the largest group of anagrams?**
```python
def find_largest_anagram_group(strs: List[str]) -> List[str]:
    """Find the group with most anagrams."""
    anagram_map = defaultdict(list)
    
    for s in strs:
        key = ''.join(sorted(s))
        anagram_map[key].append(s)
    
    # Return largest group
    return max(anagram_map.values(), key=len)
```

**Q2: Can you do this without sorting?**

Yes! Use character count (shown in Approach 3).

**Q3: How would you handle very large datasets?**
```python
"""
For datasets that don't fit in memory:
1. Use external sorting / MapReduce
2. Process in batches
3. Use database with hash index
4. Stream processing with approximate grouping
"""

def group_anagrams_distributed(strs_iterator):
    """
    Process large dataset in streaming fashion.
    
    MapReduce approach:
    - Map: (string -> (sorted_key, string))
    - Reduce: Group by sorted_key
    """
    pass
```

**Q4: What if we want fuzzy matching (allow 1-2 character differences)?**
```python
def group_similar_strings(strs: List[str], max_diff: int = 1) -> List[List[str]]:
    """
    Group strings that are similar (not exact anagrams).
    
    This requires different approach:
    - Locality-sensitive hashing
    - Edit distance clustering
    - N-gram similarity
    """
    # More complex - would need LSH or edit distance
    pass
```

## Key Takeaways

✅ **Hash tables are perfect for grouping** - O(1) lookup and insertion

✅ **Good hash keys are canonical** - sorted string represents all anagrams

✅ **Two main approaches:** Sorting O(NK log K) vs Counting O(NK)

✅ **Character count is faster** for very long strings (K >> 26)

✅ **Pattern applies broadly:** Document clustering, duplicate detection, feature hashing

✅ **Production considerations:** Unicode support, case-insensitivity, streaming

✅ **Same pattern in clustering:** Hash key = feature signature, grouping = clustering

✅ **Same pattern in speaker diarization:** Hash key = voice embedding, grouping = speaker clusters

✅ **Defaultdict is cleaner** than manually checking key existence

✅ **Testing is crucial:** Edge cases (empty strings, single string, all anagrams)

### Mental Model

Think of this problem as:
- **Anagrams:** Hash-based string grouping
- **Clustering:** Hash-based data point grouping
- **Speaker Diarization:** Hash-based audio segment grouping

All use the same pattern: **create signature → hash → group by signature**

## Additional Practice & Variants

If you want to deepen your understanding and move closer to production use cases, here are some structured follow-ups:

- **Variant 1 – Top-K Anagram Groups:**
  - Given a list of words, return only the **K largest** anagram groups.
  - Forces you to combine grouping with **heap / sorting** logic.
  - Think about how to stream this when the vocabulary is huge.

- **Variant 2 – Online Anagram Service:**
  - Build an HTTP service with two endpoints:
    - `POST /word` to insert a new word into the system.
    - `GET /anagrams?word=...` to retrieve all known anagrams of a given word.
  - Internally, you still use the same **signature → list-of-words** hash map,
    but now you must think about:
    - Concurrency (multiple writers/readers),
    - Persistence (Redis / database),
    - Eviction or TTLs if memory is constrained.

- **Variant 3 – Fuzzy Anagrams:**
  - Allow up to 1 character insertion/deletion/substitution (edit distance 1).
  - You can still start from the exact-anagram hash grouping,
    but now you need a **secondary similarity check** inside each bucket.
  - This parallels approximate matching in search systems and LSH in clustering.

Beyond interviews, this problem is a good mental model for any system where you:
1. Define a **signature** for items (exact or approximate),
2. Use that signature as a **hash key or index**,
3. Group or retrieve items efficiently based on that signature.

Once you can explain and implement this pattern fluently, you are in a good place
to reason about higher-level systems like feature stores, deduplication pipelines,
and similarity-based retrieval engines.

---

**Originally published at:** [arunbaby.com/dsa/0015-group-anagrams](https://www.arunbaby.com/dsa/0015-group-anagrams/)

*If you found this helpful, consider sharing it with others who might benefit.*






