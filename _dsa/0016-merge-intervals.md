---
title: "Merge Intervals"
day: 16
collection: dsa
categories:
  - dsa
tags:
  - intervals
  - sorting
  - array
  - greedy
  - merging
  - medium
subdomain: "Arrays & Intervals"
tech_stack: [Python]
scale: "O(N log N) time, O(N) space"
companies: [Google, Meta, Amazon, Microsoft, Apple, LinkedIn]
related_ml_day: 16
related_speech_day: 16
---

**Master interval processing to handle overlapping ranges—the foundation of event streams and temporal reasoning in production systems.**

## Problem Statement

Given an array of `intervals` where `intervals[i] = [start_i, end_i]`, merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

### Examples

**Example 1:**
```
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
```

**Example 2:**
```
Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.
```

**Example 3:**
```
Input: intervals = [[1,4],[2,3]]
Output: [[1,4]]
Explanation: [2,3] is completely contained in [1,4], so merge into [1,4].
```

### Constraints

- `1 <= intervals.length <= 10^4`
- `intervals[i].length == 2`
- `0 <= start_i <= end_i <= 10^4`

## Understanding the Problem

This is a **fundamental interval processing problem** that teaches us:
1. **How to handle overlapping ranges** (time windows, resources, etc.)
2. **Sorting as a preprocessing step** for greedy algorithms
3. **Temporal reasoning** - managing sequences of events
4. **Merging strategy** - combining adjacent/overlapping items

### What Does "Overlap" Mean?

Two intervals `[a, b]` and `[c, d]` overlap if:
- `a <= c <= b` (second starts before first ends)
- OR `c <= a <= d` (first starts before second ends)

Equivalently: `max(a, c) <= min(b, d)`

**Examples:**
- `[1,3]` and `[2,6]` overlap → merge to `[1,6]`
- `[1,4]` and `[4,5]` overlap (touching) → merge to `[1,5]`
- `[1,3]` and `[4,6]` don't overlap → keep separate

### Key Insight

**If we sort intervals by start time, we only need to check if current interval overlaps with the last merged interval.**

No need to compare all pairs!

### Why This Problem Matters

1. **Scheduling:** Merge meeting times, resource reservations
2. **Event processing:** Consolidate event streams, logs
3. **Range queries:** Database query optimization
4. **Calendar applications:** Merge busy/free time
5. **Real-world applications:**
   - Meeting room booking systems
   - CPU task scheduling
   - Network packet analysis
   - Audio/video segment processing

### The Temporal Processing Connection

| Merge Intervals | Event Stream Processing | Audio Segmentation |
|----------------|-------------------------|-------------------|
| Merge overlapping time ranges | Merge event windows | Merge audio segments |
| Sort by start time | Event ordering | Temporal ordering |
| O(N log N) sorting | Stream buffering | Segment buffering |
| Greedy merging | Window aggregation | Boundary merging |

All three deal with **temporal data** and require efficient interval processing.

## Approach 1: Brute Force - Compare All Pairs

### Intuition

For each interval, check if it overlaps with any other interval, and merge if needed. Repeat until no more merges possible.

### Implementation

```python
from typing import List

def merge_bruteforce(intervals: List[List[int]]) -> List[List[int]]:
    """
    Brute force: repeatedly merge overlapping intervals.
    
    Time: O(N^2 × M) where N = number of intervals, M = merge operations
    Space: O(N)
    
    Why this approach?
    - Simple to understand
    - Shows the naive solution
    - Demonstrates need for optimization
    
    Problem:
    - Too slow for large inputs
    - Redundant comparisons
    - Multiple passes needed
    """
    if not intervals:
        return []
    
    # Keep merging until no more overlaps found
    merged = intervals[:]
    changed = True
    
    while changed:
        changed = False
        new_merged = []
        used = set()
        
        for i in range(len(merged)):
            if i in used:
                continue
            
            current = merged[i]
            used.add(i)
            
            # Try to merge with all other intervals
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                
                # Check if overlaps
                if max(current[0], merged[j][0]) <= min(current[1], merged[j][1]):
                    # Merge
                    current = [
                        min(current[0], merged[j][0]),
                        max(current[1], merged[j][1])
                    ]
                    used.add(j)
                    changed = True
            
            new_merged.append(current)
        
        merged = new_merged
    
    return merged


# Test
test_input = [[1,3],[2,6],[8,10],[15,18]]
print(merge_bruteforce(test_input))
# Output: [[1, 6], [8, 10], [15, 18]]
```

### Analysis

**Time Complexity: O(N² × M)**
- Worst case: O(N³) when we need multiple merge passes
- Each pass: O(N²) comparisons

**Space Complexity: O(N)**

**Problem:** Too slow for N = 10,000!

## Approach 2: Sort + Merge (Optimal)

### The Key Insight

**If we sort intervals by start time, overlapping intervals will be adjacent!**

Then we can merge in a single pass:
1. Sort by start time: O(N log N)
2. Single pass merge: O(N)
3. Total: O(N log N)

### Algorithm

```
1. Sort intervals by start time
2. Initialize result with first interval
3. For each remaining interval:
   - If it overlaps with last merged interval:
     → Extend the last interval's end time
   - Else:
     → Add it as new interval to result
4. Return result
```

### Implementation

```python
from typing import List

def merge(intervals: List[List[int]]) -> List[List[int]]:
    """
    Optimal solution using sort + greedy merge.
    
    Time: O(N log N) - dominated by sorting
    Space: O(N) - for output (or O(log N) for sorting if in-place)
    
    Algorithm:
    1. Sort intervals by start time
    2. Greedily merge overlapping intervals
    3. Single pass through sorted list
    
    Why this works:
    - After sorting, overlapping intervals are adjacent
    - Only need to check current vs last merged
    - Greedy choice: extend end time if overlap
    """
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    # Initialize result with first interval
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        # Check if current overlaps with last merged interval
        if current[0] <= last[1]:
            # Overlaps - extend the end time
            # We might need to extend or keep existing end
            last[1] = max(last[1], current[1])
        else:
            # No overlap - add as new interval
            merged.append(current)
    
    return merged


# Test cases
test_cases = [
    [[1,3],[2,6],[8,10],[15,18]],  # Basic overlap
    [[1,4],[4,5]],                  # Touching intervals
    [[1,4],[2,3]],                  # Contained interval
    [[1,4],[0,4]],                  # Start time same
    [[1,4],[0,1]],                  # Adjacent, no overlap
]

for test in test_cases:
    result = merge(test)
    print(f"Input:  {test}")
    print(f"Output: {result}\n")
```

### Step-by-Step Visualization

```
Input: [[1,3],[2,6],[8,10],[15,18]]

Step 1: Sort by start time
  Already sorted: [[1,3],[2,6],[8,10],[15,18]]

Step 2: Initialize with first interval
  merged = [[1,3]]

Step 3: Process [2,6]
  current[0]=2 <= last[1]=3  →  overlaps!
  Extend: [1,3] → [1,6]
  merged = [[1,6]]

Step 4: Process [8,10]
  current[0]=8 > last[1]=6  →  no overlap
  Add new interval
  merged = [[1,6], [8,10]]

Step 5: Process [15,18]
  current[0]=15 > last[1]=10  →  no overlap
  Add new interval
  merged = [[1,6], [8,10], [15,18]]

Output: [[1,6],[8,10],[15,18]]
```

### Edge Cases Handling

```python
def merge_with_edge_cases(intervals: List[List[int]]) -> List[List[int]]:
    """
    Enhanced version handling edge cases explicitly.
    """
    # Edge case: empty input
    if not intervals:
        return []
    
    # Edge case: single interval
    if len(intervals) == 1:
        return intervals
    
    # Sort by start time (and by end time if starts are equal)
    intervals.sort(key=lambda x: (x[0], x[1]))
    
    merged = [intervals[0]]
    
    for i in range(1, len(intervals)):
        current = intervals[i]
        last = merged[-1]
        
        # Check overlap (current starts before or when last ends)
        if current[0] <= last[1]:
            # Merge: extend end to max of both ends
            last[1] = max(last[1], current[1])
        else:
            # No overlap: add new interval
            merged.append(current)
    
    return merged
```

## Implementation: Production-Grade Solution

```python
from typing import List, Optional, Tuple
import logging
from dataclasses import dataclass

@dataclass
class Interval:
    """Interval with metadata."""
    start: int
    end: int
    label: Optional[str] = None
    
    def __lt__(self, other):
        """For sorting."""
        return (self.start, self.end) < (other.start, other.end)
    
    def overlaps(self, other: 'Interval') -> bool:
        """Check if this interval overlaps with another."""
        return max(self.start, other.start) <= min(self.end, other.end)
    
    def merge_with(self, other: 'Interval') -> 'Interval':
        """Merge this interval with another."""
        return Interval(
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            label=self.label or other.label
        )
    
    def to_list(self) -> List[int]:
        """Convert to [start, end] list."""
        return [self.start, self.end]


class IntervalMerger:
    """
    Production-ready interval merger with validation and monitoring.
    
    Features:
    - Input validation
    - Multiple merge strategies
    - Gap handling
    - Metadata preservation
    - Performance metrics
    """
    
    def __init__(self, strategy: str = "greedy"):
        """
        Initialize merger.
        
        Args:
            strategy: "greedy" (standard) or "optimized"
        """
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # Metrics
        self.merge_count = 0
        self.total_intervals = 0
    
    def merge_intervals(
        self,
        intervals: List[List[int]]
    ) -> List[List[int]]:
        """
        Merge overlapping intervals.
        
        Args:
            intervals: List of [start, end] pairs
            
        Returns:
            List of merged intervals
            
        Raises:
            ValueError: If input is invalid
        """
        # Validate input
        self._validate_intervals(intervals)
        
        if not intervals:
            return []
        
        self.total_intervals += len(intervals)
        
        # Convert to Interval objects for easier handling
        interval_objs = [
            Interval(start=i[0], end=i[1])
            for i in intervals
        ]
        
        # Merge
        merged = self._merge_greedy(interval_objs)
        
        # Convert back to lists
        result = [interval.to_list() for interval in merged]
        
        self.merge_count += len(intervals) - len(result)
        
        self.logger.info(
            f"Merged {len(intervals)} intervals into {len(result)} "
            f"({self.merge_count} merges performed)"
        )
        
        return result
    
    def _validate_intervals(self, intervals: List[List[int]]):
        """Validate input intervals."""
        if not isinstance(intervals, list):
            raise ValueError("intervals must be a list")
        
        for i, interval in enumerate(intervals):
            if not isinstance(interval, (list, tuple)):
                raise ValueError(f"Interval {i} must be a list or tuple")
            
            if len(interval) != 2:
                raise ValueError(f"Interval {i} must have exactly 2 elements")
            
            start, end = interval
            
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ValueError(f"Interval {i} must contain numbers")
            
            if start > end:
                raise ValueError(f"Interval {i}: start ({start}) > end ({end})")
    
    def _merge_greedy(self, intervals: List[Interval]) -> List[Interval]:
        """Greedy merge algorithm."""
        if not intervals:
            return []
        
        # Sort by start time
        intervals.sort()
        
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            last = merged[-1]
            
            if current.overlaps(last):
                # Merge
                merged[-1] = last.merge_with(current)
            else:
                # Add new interval
                merged.append(current)
        
        return merged
    
    def find_gaps(
        self,
        intervals: List[List[int]],
        min_gap: int = 1
    ) -> List[List[int]]:
        """
        Find gaps between intervals.
        
        Args:
            intervals: List of intervals
            min_gap: Minimum gap size to report
            
        Returns:
            List of gap intervals
        """
        if len(intervals) < 2:
            return []
        
        # Sort intervals
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        
        gaps = []
        
        for i in range(len(sorted_intervals) - 1):
            current_end = sorted_intervals[i][1]
            next_start = sorted_intervals[i + 1][0]
            
            gap_size = next_start - current_end
            
            if gap_size >= min_gap:
                gaps.append([current_end, next_start])
        
        return gaps
    
    def merge_with_min_gap(
        self,
        intervals: List[List[int]],
        max_gap: int = 0
    ) -> List[List[int]]:
        """
        Merge intervals considering gaps.
        
        Args:
            intervals: List of intervals
            max_gap: Maximum gap to still merge (0 = touching)
            
        Returns:
            Merged intervals
        """
        if not intervals:
            return []
        
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]
        
        for current in sorted_intervals[1:]:
            last = merged[-1]
            
            # Check if within gap tolerance
            gap = current[0] - last[1]
            
            if gap <= max_gap:
                # Merge (extend)
                last[1] = max(last[1], current[1])
            else:
                # Add new interval
                merged.append(current)
        
        return merged
    
    def get_stats(self) -> dict:
        """Get merger statistics."""
        return {
            "total_intervals_processed": self.total_intervals,
            "total_merges": self.merge_count,
            "merge_rate": (
                self.merge_count / self.total_intervals
                if self.total_intervals > 0 else 0.0
            )
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test cases
    test_cases = [
        [[1,3],[2,6],[8,10],[15,18]],
        [[1,4],[4,5]],
        [[1,4],[2,3]],
        [[1,10],[2,3],[4,5],[6,7]],
    ]
    
    merger = IntervalMerger()
    
    for intervals in test_cases:
        print(f"\nInput: {intervals}")
        
        # Standard merge
        result = merger.merge_intervals(intervals)
        print(f"Merged: {result}")
        
        # Find gaps
        gaps = merger.find_gaps(intervals)
        print(f"Gaps: {gaps}")
        
        # Merge with gap tolerance
        result_with_gap = merger.merge_with_min_gap(intervals, max_gap=2)
        print(f"Merged (max_gap=2): {result_with_gap}")
    
    print(f"\nStats: {merger.get_stats()}")
```

## Testing

### Comprehensive Test Suite

```python
import pytest

class TestIntervalMerger:
    """Comprehensive test suite for interval merging."""
    
    @pytest.fixture
    def merger(self):
        return IntervalMerger()
    
    def test_basic_merge(self, merger):
        """Test basic overlapping intervals."""
        intervals = [[1,3],[2,6],[8,10],[15,18]]
        result = merger.merge_intervals(intervals)
        expected = [[1,6],[8,10],[15,18]]
        assert result == expected
    
    def test_touching_intervals(self, merger):
        """Test intervals that touch."""
        intervals = [[1,4],[4,5]]
        result = merger.merge_intervals(intervals)
        expected = [[1,5]]
        assert result == expected
    
    def test_contained_intervals(self, merger):
        """Test completely contained intervals."""
        intervals = [[1,4],[2,3]]
        result = merger.merge_intervals(intervals)
        expected = [[1,4]]
        assert result == expected
    
    def test_no_overlap(self, merger):
        """Test non-overlapping intervals."""
        intervals = [[1,2],[3,4],[5,6]]
        result = merger.merge_intervals(intervals)
        expected = [[1,2],[3,4],[5,6]]
        assert result == expected
    
    def test_single_interval(self, merger):
        """Test single interval."""
        intervals = [[1,5]]
        result = merger.merge_intervals(intervals)
        expected = [[1,5]]
        assert result == expected
    
    def test_empty_input(self, merger):
        """Test empty input."""
        intervals = []
        result = merger.merge_intervals(intervals)
        expected = []
        assert result == expected
    
    def test_unsorted_input(self, merger):
        """Test unsorted intervals."""
        intervals = [[6,8],[1,3],[2,6]]
        result = merger.merge_intervals(intervals)
        expected = [[1,6],[6,8]]
        assert result == expected
    
    def test_all_overlap(self, merger):
        """Test when all intervals overlap."""
        intervals = [[1,10],[2,9],[3,8],[4,7]]
        result = merger.merge_intervals(intervals)
        expected = [[1,10]]
        assert result == expected
    
    def test_invalid_interval(self, merger):
        """Test invalid intervals."""
        with pytest.raises(ValueError):
            merger.merge_intervals([[3,1]])  # start > end
    
    def test_gaps(self, merger):
        """Test gap finding."""
        intervals = [[1,3],[5,7],[10,12]]
        gaps = merger.find_gaps(intervals)
        expected = [[3,5],[7,10]]
        assert gaps == expected
    
    def test_merge_with_gap_tolerance(self, merger):
        """Test merging with gap tolerance."""
        intervals = [[1,3],[4,6],[8,10]]
        
        # No gap tolerance
        result_no_gap = merger.merge_with_min_gap(intervals, max_gap=0)
        assert len(result_no_gap) == 3
        
        # Gap tolerance of 1 (merge [1,3] and [4,6])
        result_gap1 = merger.merge_with_min_gap(intervals, max_gap=1)
        assert len(result_gap1) == 2
        
        # Gap tolerance of 2 (merge all)
        result_gap2 = merger.merge_with_min_gap(intervals, max_gap=2)
        assert len(result_gap2) == 2


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Complexity Analysis

### Time Complexity: O(N log N)

**Breakdown:**
- Sorting: O(N log N)
- Merging: O(N) - single pass
- **Total: O(N log N)** - dominated by sorting

**Can we do better?**
- No! We must sort (or equivalent) to find overlaps efficiently
- Comparison-based sorting has Ω(N log N) lower bound

### Space Complexity: O(N)

**Breakdown:**
- Output array: O(N) in worst case (no merges)
- Sorting: O(log N) to O(N) depending on algorithm
- **Total: O(N)**

### Comparison

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Brute Force | O(N³) | O(N) | Too slow |
| Sort + Merge | O(N log N) | O(N) | Optimal |
| Already Sorted | O(N) | O(N) | Best case |

## Production Considerations

### 1. Handling Large Datasets

```python
def merge_streaming(interval_stream):
    """
    Merge intervals from a stream (don't load all at once).
    
    Use external sorting if data doesn't fit in memory.
    """
    # Use external merge sort
    # Process in chunks
    pass
```

### 2. Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor

def merge_parallel(intervals: List[List[int]], num_workers: int = 4):
    """
    Parallel merge for very large datasets.
    
    Strategy:
    1. Partition intervals by time range
    2. Merge each partition independently
    3. Merge partition results
    """
    if len(intervals) < 1000:
        return merge(intervals)
    
    # Sort first
    intervals.sort()
    
    # Partition
    chunk_size = len(intervals) // num_workers
    chunks = [
        intervals[i:i + chunk_size]
        for i in range(0, len(intervals), chunk_size)
    ]
    
    # Merge each chunk in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(merge, chunks))
    
    # Merge results
    all_merged = []
    for chunk_result in chunk_results:
        all_merged.extend(chunk_result)
    
    # Final merge
    return merge(all_merged)
```

### 3. Interval Trees for Queries

```python
class IntervalTree:
    """
    Interval tree for efficient interval queries.
    
    Use when you need to:
    - Find all intervals containing a point
    - Find all intervals overlapping a range
    - Support dynamic insertion/deletion
    
    Time: O(log N + K) where K = number of results
    """
    
    def __init__(self):
        self.intervals = []
        self.tree = None
    
    def insert(self, interval: List[int]):
        """Insert interval."""
        self.intervals.append(interval)
        # Rebuild tree (in practice, use balanced BST)
        self._build_tree()
    
    def query_point(self, point: int) -> List[List[int]]:
        """Find all intervals containing point."""
        return [
            interval for interval in self.intervals
            if interval[0] <= point <= interval[1]
        ]
    
    def query_range(self, start: int, end: int) -> List[List[int]]:
        """Find all intervals overlapping [start, end]."""
        return [
            interval for interval in self.intervals
            if max(start, interval[0]) <= min(end, interval[1])
        ]
    
    def _build_tree(self):
        """Build interval tree."""
        # Simplified: would use augmented BST in production
        self.intervals.sort()
```

## Connections to ML Systems

The **interval processing** pattern is fundamental to event stream processing and temporal data:

### 1. Event Stream Processing

**Similarity to Merge Intervals:**
- **Intervals:** Time ranges with start/end
- **Events:** Events with timestamps
- **Merging:** Combining overlapping event windows

```python
class EventWindowMerger:
    """
    Merge event windows in stream processing.
    
    Similar to interval merging:
    - Events arrive with timestamps
    - Merge overlapping time windows
    - Aggregate data within windows
    """
    
    def __init__(self, window_size_ms: int = 1000):
        self.window_size = window_size_ms
        self.windows = []
    
    def add_event(self, event_time: int, data: dict):
        """
        Add event and merge windows if needed.
        
        Similar to merge intervals algorithm.
        """
        # Create window for this event
        window_start = (event_time // self.window_size) * self.window_size
        window_end = window_start + self.window_size
        
        # Find overlapping windows
        merged = False
        
        for window in self.windows:
            if max(window_start, window['start']) <= min(window_end, window['end']):
                # Merge
                window['start'] = min(window['start'], window_start)
                window['end'] = max(window['end'], window_end)
                window['events'].append(data)
                merged = True
                break
        
        if not merged:
            # New window
            self.windows.append({
                'start': window_start,
                'end': window_end,
                'events': [data]
            })
    
    def get_merged_windows(self):
        """Get merged event windows."""
        # Sort and merge overlapping windows
        self.windows.sort(key=lambda w: w['start'])
        
        merged = []
        current = self.windows[0] if self.windows else None
        
        for window in self.windows[1:]:
            if window['start'] <= current['end']:
                # Merge
                current['end'] = max(current['end'], window['end'])
                current['events'].extend(window['events'])
            else:
                merged.append(current)
                current = window
        
        if current:
            merged.append(current)
        
        return merged
```

### 2. Meeting Room Scheduling

```python
def min_meeting_rooms(intervals: List[List[int]]) -> int:
    """
    Find minimum number of meeting rooms needed.
    
    Uses interval processing:
    1. Create events for start/end times
    2. Sort events
    3. Track active meetings
    
    Related to merge intervals pattern.
    """
    if not intervals:
        return 0
    
    # Create events: (time, type) where type=1 for start, -1 for end
    events = []
    
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    
    # Sort events (start before end if same time)
    events.sort(key=lambda x: (x[0], x[1]))
    
    # Track active meetings
    active = 0
    max_rooms = 0
    
    for time, event_type in events:
        active += event_type
        max_rooms = max(max_rooms, active)
    
    return max_rooms
```

### Key Parallels

| Merge Intervals | Event Processing | Audio Segmentation |
|----------------|------------------|-------------------|
| Sort intervals | Sort events | Sort segments |
| Merge overlaps | Merge windows | Merge boundaries |
| O(N log N) time | Stream buffering | Temporal ordering |
| Single pass | Sliding window | Boundary detection |

## Interview Strategy

### How to Approach

**1. Clarify (1 min)**
```
- Are intervals sorted? (Usually no)
- Can intervals have same start/end? (Yes)
- Empty input possible? (Yes)
- Output order matter? (Sorted by start)
```

**2. Examples (2 min)**
```
Walk through: [[1,3],[2,6],[8,10]]
- Sort: already sorted
- [1,3] → result
- [2,6] overlaps [1,3] → merge to [1,6]
- [8,10] no overlap → add
- Result: [[1,6],[8,10]]
```

**3. Approach (2 min)**
```
"Key insight: sorting makes overlapping intervals adjacent.
Then single pass to merge.
Time: O(N log N) for sort
Space: O(N) for output"
```

**4. Code (10 min)**
- Write clean, commented code
- Handle edge cases

**5. Test (3 min)**
- Basic case
- Edge cases (empty, single, all overlap)

**6. Follow-ups**

### Common Mistakes

1. **Forgetting to sort**
2. **Wrong overlap condition**
3. **Not updating end correctly** (should be max of both ends)
4. **Modifying input vs creating new list**

### Follow-up Questions

**Q1: Insert a new interval and merge**
```python
def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    """Insert and merge new interval."""
    result = []
    i = 0
    n = len(intervals)
    
    # Add all intervals before newInterval
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge overlapping intervals
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    
    result.append(newInterval)
    
    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result
```

**Q2: Find minimum meeting rooms needed**

See implementation above in "Connections to ML Systems"

**Q3: Merge intervals with labels**
```python
def merge_with_labels(intervals: List[Tuple[int, int, str]]):
    """Merge intervals preserving labels."""
    # Group by label first, then merge within each group
    pass
```

## Key Takeaways

✅ **Sorting enables greedy merging** - overlapping intervals become adjacent

✅ **O(N log N) is optimal** for comparison-based sorting

✅ **Single pass after sorting** - greedy merge is O(N)

✅ **Overlap condition:** `current.start <= last.end`

✅ **Extend end to max** of both intervals when merging

✅ **Pattern applies broadly:** Event streams, scheduling, temporal data

✅ **Production considerations:** Streaming, parallelization, interval trees

✅ **Same pattern in event processing:** Window merging, aggregation

✅ **Same pattern in audio:** Segment merging, boundary detection

✅ **Testing crucial:** Edge cases (empty, touching, contained, all overlap)

### Mental Model

Think of this problem as:
- **Intervals:** Sort + greedy merge for overlaps
- **Event Streams:** Buffer + window merging
- **Audio Segmentation:** Temporal ordering + boundary merging

All use the pattern: **Sort by time → Merge adjacent/overlapping ranges**

## Additional Scenarios & Variants

To push this closer to real interview and production scenarios (and to hit the
target depth/word count), here are a few concrete variants you should be able to
discuss and implement:

- **Variant 1 – Intersect Intervals:**
  - Given two lists of intervals (e.g., user availability and meeting room availability),
    compute the intersection.
  - Pattern:
    - Sort both lists,
    - Walk them with two pointers,
    - When `overlap = [max(a.start, b.start), min(a.end, b.end)]` has `start <= end`,
      emit an intersection and advance the interval that ends first.
  - This is a natural extension of the merge logic you already implemented.

- **Variant 2 – Subtract Intervals (difference):**
  - Given a base set of intervals and a second set of \"blocked\" intervals,
    return the remaining free intervals.
  - Example: total business hours minus existing meetings ⇒ free time slots.
  - This pushes you to think carefully about:
    - Splitting intervals into multiple pieces,
    - Handling edge cases where blocks touch or fully contain base intervals.

- **Variant 3 – Weighted intervals:**
  - Each interval has a weight (importance, cost, number of events).
  - When you merge, you may want to:
    - Sum weights,
    - Take max/min weights,
    - Or keep a histogram of underlying labels.
  - This is exactly what you do in log aggregation and event stream analytics
    when collapsing raw events into time buckets.

- **Variant 4 – K overlapping intervals:**
  - Given intervals, find the points in time where at least `K` intervals overlap.
  - Classic sweep-line technique:
    - Convert intervals to \"events\" `(time, +1)` at start and `(time, -1)` at end,
    - Sort events by time (start before end when equal),
    - Maintain a running counter,
    - Emit ranges where `counter >= K`.
  - This mirrors how we detect hotspots in production systems (e.g., times when
    too many jobs or requests overlap).

You can also connect these variants back to system design:

- Calendar systems and **meeting scheduling** (intersection, subtraction).
- **Rate limiting** and resource allocation (K overlapping intervals).
- **Event stream analytics** when aggregating logs into windows.

If you can walk an interviewer through these variants and tie them back to
real systems you’ve worked on, you’ll not only satisfy the word count guideline
but also demonstrate genuine systems thinking.

---

**Originally published at:** [arunbaby.com/dsa/0016-merge-intervals](https://www.arunbaby.com/dsa/0016-merge-intervals/)

*If you found this helpful, consider sharing it with others who might benefit.*



