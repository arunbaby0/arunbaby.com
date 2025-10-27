---
title: "Container With Most Water"
day: 13
collection: dsa
categories:
  - dsa
tags:
  - two-pointers
  - greedy
  - array
  - optimization
  - interview-classic
  - medium-easy
subdomain: "Array Algorithms"
tech_stack: [Python]
scale: "O(N) time, O(1) space"
companies: [Google, Meta, Amazon, Microsoft, Apple]
related_dsa_day: 13
related_ml_day: 13
related_speech_day: 13
---

**Master the two-pointer greedy technique that powers resource optimization in production ML systems.**

## Problem Statement

You are given an integer array `height` of length `n`. There are `n` vertical lines drawn such that the two endpoints of the `i`-th line are `(i, 0)` and `(i, height[i])`.

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

**Note:** You may not slant the container.

### Examples

**Example 1:**
```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The vertical lines are represented by [1,8,6,2,5,4,8,3,7]. 
In this case, the max area of water (blue section) the container can contain is 49.
```

**Example 2:**
```
Input: height = [1,1]
Output: 1
```

**Example 3:**
```
Input: height = [4,3,2,1,4]
Output: 16
```

### Constraints

- `n == height.length`
- `2 <= n <= 10^5`
- `0 <= height[i] <= 10^4`

## Understanding the Problem

At first glance, this seems like a simple geometry problem, but it's actually teaching us a profound lesson about **greedy optimization** that applies directly to resource allocation in production systems.

### Core Insight

The water container is defined by:
- **Width:** Distance between two lines (indices) `j - i`
- **Height:** The minimum of the two heights `min(height[i], height[j])`
- **Area:** `width × height = (j - i) × min(height[i], height[j])`

The key realization: **The shorter line always limits the water capacity**. This is analogous to:
- **In ML systems:** The slowest component determines throughput
- **In speech processing:** The weakest model in the pipeline limits accuracy
- **In resource allocation:** The bottleneck resource constrains performance

### Why This Problem Matters

1. **Greedy decision-making:** When to move which pointer?
2. **Optimization under constraints:** Maximize area with competing factors (width vs height)
3. **Two-pointer technique:** A fundamental pattern for O(N) solutions
4. **Real-world modeling:** Resource allocation, capacity planning, bottleneck analysis

## Approach 1: Brute Force

### Intuition

Try every possible pair of lines and calculate the area for each. Keep track of the maximum.

### Implementation

```python
def maxArea_bruteforce(height: list[int]) -> int:
    """
    Brute force solution: try all pairs.
    
    Time: O(N^2) - nested loops
    Space: O(1) - only storing max_area
    
    Why this approach?
    - Guarantees finding the optimal solution
    - Easy to understand and implement
    - Good starting point in interviews
    
    What's the problem?
    - Too slow for large inputs (n up to 10^5)
    - Wastes computation on obviously suboptimal pairs
    """
    n = len(height)
    max_area = 0
    
    # Try every pair (i, j) where i < j
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate area for this pair
            width = j - i
            h = min(height[i], height[j])
            area = width * h
            
            # Update maximum
            max_area = max(max_area, area)
    
    return max_area
```

### Test the Brute Force

```python
# Test cases
test_cases = [
    ([1,8,6,2,5,4,8,3,7], 49),
    ([1,1], 1),
    ([4,3,2,1,4], 16),
    ([1,2,1], 2),
]

for heights, expected in test_cases:
    result = maxArea_bruteforce(heights)
    status = "✓" if result == expected else "✗"
    print(f"{status} Input: {heights}")
    print(f"  Expected: {expected}, Got: {result}\n")
```

### Complexity Analysis

- **Time:** O(N²) - we examine all (N choose 2) = N(N-1)/2 pairs
- **Space:** O(1) - only a few variables

**Problem:** With N = 10^5, we'd need ~5 billion operations. Too slow!

## Approach 2: Two-Pointer Greedy (Optimal)

### The Key Insight

Here's the brilliant observation that makes this problem solvable in O(N):

**If we have two pointers at positions `left` and `right`:**
- The area is `(right - left) × min(height[left], height[right])`
- As we move pointers inward, **width always decreases**
- To potentially increase area, we need to increase height
- **Moving the taller pointer can only decrease area** (width decreases, height can't increase)
- **Moving the shorter pointer might increase area** (width decreases, but height might increase enough to compensate)

This is the **greedy choice**: always move the pointer at the shorter line.

### Why Does This Work?

Let's prove we don't miss the optimal solution:

1. Start with `left = 0`, `right = n-1` (maximum width)
2. Say `height[left] < height[right]`
3. Consider any container using `left` and some `k` where `left < k < right`:
   - Width is smaller: `k - left < right - left`
   - Height is at most `height[left]` (the limiting factor)
   - So area is at most `(k - left) × height[left]`
   - This is definitely ≤ `(right - left) × height[left]` (current area)
4. Therefore, we can safely discard `left` and never consider it again!

This greedy property ensures we examine all potentially optimal pairs.

### Implementation

```python
def maxArea(height: list[int]) -> int:
    """
    Two-pointer greedy solution.
    
    Time: O(N) - single pass with two pointers
    Space: O(1) - only a few variables
    
    Algorithm:
    1. Start with widest container (left=0, right=n-1)
    2. Calculate current area
    3. Move the pointer at the shorter line inward
    4. Repeat until pointers meet
    
    Why this works:
    - We never miss the optimal solution (proven above)
    - Each step eliminates one line from consideration
    - Greedy choice: always improve the bottleneck (shorter line)
    """
    left = 0
    right = len(height) - 1
    max_area = 0
    
    while left < right:
        # Calculate current area
        # Width decreases as pointers move inward
        width = right - left
        
        # Height is limited by the shorter line
        current_height = min(height[left], height[right])
        
        # Calculate and update maximum area
        current_area = width * current_height
        max_area = max(max_area, current_area)
        
        # Greedy choice: move the pointer at the shorter line
        # Why? Moving the taller pointer can only make things worse
        # (width decreases and height can't improve)
        if height[left] < height[right]:
            left += 1  # Try to find a taller left line
        else:
            right -= 1  # Try to find a taller right line
    
    return max_area
```

### Step-by-Step Visualization

Let's trace through `height = [1,8,6,2,5,4,8,3,7]`:

```
Initial: left=0 (height=1), right=8 (height=7)
         Area = 8 × min(1,7) = 8 × 1 = 8
         Move left (shorter)

Step 1:  left=1 (height=8), right=8 (height=7)
         Area = 7 × min(8,7) = 7 × 7 = 49 ← Maximum!
         Move right (shorter)

Step 2:  left=1 (height=8), right=7 (height=3)
         Area = 6 × min(8,3) = 6 × 3 = 18
         Move right (shorter)

Step 3:  left=1 (height=8), right=6 (height=8)
         Area = 5 × min(8,8) = 5 × 8 = 40
         Move either (equal), let's move right

Step 4:  left=1 (height=8), right=5 (height=4)
         Area = 4 × min(8,4) = 4 × 4 = 16
         Move right (shorter)

... continues until left meets right

Maximum area found: 49
```

### Optimized Implementation with Early Termination

```python
def maxArea_optimized(height: list[int]) -> int:
    """
    Enhanced version with potential early termination.
    
    Additional optimization:
    - If we find a container with max possible theoretical area,
      we can stop early (though rare in practice)
    """
    left = 0
    right = len(height) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        
        # Calculate area with current configuration
        if height[left] < height[right]:
            # Left is the bottleneck
            current_area = width * height[left]
            max_area = max(max_area, current_area)
            left += 1
        else:
            # Right is the bottleneck (or equal)
            current_area = width * height[right]
            max_area = max(max_area, current_area)
            right -= 1
        
        # Early termination (optional):
        # If theoretical maximum remaining area can't beat current max, stop
        # Theoretical max = remaining_width × max(remaining_heights)
        # This is rarely beneficial but shows advanced thinking
    
    return max_area
```

## Implementation: Production-Grade Solution

Here's a complete implementation with error handling, logging, and optimizations:

```python
from typing import List, Optional
import logging

class ContainerSolver:
    """
    Production-ready container with most water solver.
    
    Features:
    - Input validation
    - Multiple solution strategies
    - Performance metrics
    - Detailed logging
    """
    
    def __init__(self, strategy: str = "two_pointer"):
        """
        Initialize solver with specified strategy.
        
        Args:
            strategy: "brute_force" or "two_pointer" (default)
        """
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        self.comparisons = 0  # Track operations
    
    def max_area(self, height: List[int]) -> int:
        """
        Find maximum water container area.
        
        Args:
            height: List of line heights
            
        Returns:
            Maximum area
            
        Raises:
            ValueError: If input is invalid
        """
        # Validate input
        if not height or len(height) < 2:
            raise ValueError("Need at least 2 lines to form a container")
        
        if not all(isinstance(h, int) and h >= 0 for h in height):
            raise ValueError("All heights must be non-negative integers")
        
        # Reset metrics
        self.comparisons = 0
        
        # Choose strategy
        if self.strategy == "brute_force":
            result = self._brute_force(height)
        else:
            result = self._two_pointer(height)
        
        self.logger.info(f"Found max area {result} with {self.comparisons} comparisons")
        return result
    
    def _brute_force(self, height: List[int]) -> int:
        """Brute force O(N^2) solution."""
        n = len(height)
        max_area = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                width = j - i
                h = min(height[i], height[j])
                area = width * h
                max_area = max(max_area, area)
                self.comparisons += 1
        
        return max_area
    
    def _two_pointer(self, height: List[int]) -> int:
        """Two-pointer O(N) solution."""
        left = 0
        right = len(height) - 1
        max_area = 0
        
        while left < right:
            # Calculate current area
            width = right - left
            current_height = min(height[left], height[right])
            current_area = width * current_height
            max_area = max(max_area, current_area)
            self.comparisons += 1
            
            # Move pointer at shorter line
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_area
    
    def get_container_details(self, height: List[int]) -> dict:
        """
        Get detailed information about the optimal container.
        
        Returns:
            Dictionary with area, indices, and metadata
        """
        if len(height) < 2:
            return {"error": "Invalid input"}
        
        left = 0
        right = len(height) - 1
        max_area = 0
        best_left = 0
        best_right = 0
        
        while left < right:
            width = right - left
            current_height = min(height[left], height[right])
            current_area = width * current_height
            
            if current_area > max_area:
                max_area = current_area
                best_left = left
                best_right = right
            
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return {
            "max_area": max_area,
            "left_index": best_left,
            "right_index": best_right,
            "left_height": height[best_left],
            "right_height": height[best_right],
            "width": best_right - best_left,
            "effective_height": min(height[best_left], height[best_right])
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test cases
    test_cases = [
        [1,8,6,2,5,4,8,3,7],  # Example 1
        [1,1],                 # Example 2
        [4,3,2,1,4],          # Example 3
    ]
    
    solver = ContainerSolver(strategy="two_pointer")
    
    for heights in test_cases:
        print(f"\nHeight array: {heights}")
        
        # Get maximum area
        max_area = solver.max_area(heights)
        print(f"Maximum area: {max_area}")
        
        # Get detailed information
        details = solver.get_container_details(heights)
        print(f"Details: {details}")
        print(f"Comparisons made: {solver.comparisons}")
```

## Testing

### Comprehensive Test Suite

```python
import pytest
from typing import List, Tuple

class TestContainerWithMostWater:
    """Comprehensive test suite for container problem."""
    
    @pytest.fixture
    def solver(self):
        return ContainerSolver(strategy="two_pointer")
    
    def test_basic_examples(self, solver):
        """Test provided examples."""
        test_cases = [
            ([1,8,6,2,5,4,8,3,7], 49),
            ([1,1], 1),
            ([4,3,2,1,4], 16),
        ]
        
        for heights, expected in test_cases:
            assert solver.max_area(heights) == expected
    
    def test_edge_cases(self, solver):
        """Test edge cases."""
        # Minimum length
        assert solver.max_area([1, 2]) == 1
        
        # All same height
        assert solver.max_area([5, 5, 5, 5]) == 15  # width=3, height=5
        
        # Increasing sequence
        assert solver.max_area([1, 2, 3, 4, 5]) == 6  # indices 0,4: 4×min(1,5)=4 or 1,4: 3×min(2,5)=6
        
        # Decreasing sequence
        assert solver.max_area([5, 4, 3, 2, 1]) == 6  # symmetric
    
    def test_large_input(self, solver):
        """Test with large input."""
        # Create array with 10^5 elements
        heights = list(range(1, 100001))
        result = solver.max_area(heights)
        assert result > 0
        assert solver.comparisons < 100000  # Should be O(N)
    
    def test_invalid_input(self, solver):
        """Test input validation."""
        with pytest.raises(ValueError):
            solver.max_area([])
        
        with pytest.raises(ValueError):
            solver.max_area([1])
        
        with pytest.raises(ValueError):
            solver.max_area([1, -1, 2])  # Negative height
    
    def test_strategy_equivalence(self):
        """Test that both strategies give same results."""
        heights = [1,8,6,2,5,4,8,3,7]
        
        bf_solver = ContainerSolver(strategy="brute_force")
        tp_solver = ContainerSolver(strategy="two_pointer")
        
        assert bf_solver.max_area(heights) == tp_solver.max_area(heights)
    
    def test_performance_difference(self):
        """Demonstrate performance difference."""
        heights = list(range(1, 1001))
        
        bf_solver = ContainerSolver(strategy="brute_force")
        tp_solver = ContainerSolver(strategy="two_pointer")
        
        bf_result = bf_solver.max_area(heights)
        bf_comparisons = bf_solver.comparisons
        
        tp_result = tp_solver.max_area(heights)
        tp_comparisons = tp_solver.comparisons
        
        assert bf_result == tp_result
        assert tp_comparisons < bf_comparisons  # Much fewer comparisons
        print(f"Brute force: {bf_comparisons} comparisons")
        print(f"Two pointer: {tp_comparisons} comparisons")
        print(f"Speedup: {bf_comparisons / tp_comparisons:.2f}x")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Complexity Analysis

### Two-Pointer Solution (Optimal)

**Time Complexity: O(N)**

- Single pass through the array
- Each iteration moves one pointer
- Total iterations = N-1 (when pointers meet)
- Each iteration: O(1) operations (comparison, arithmetic)

**Space Complexity: O(1)**

- Only a few variables: `left`, `right`, `max_area`, `width`, `height`
- No additional data structures
- Space usage independent of input size

### Brute Force Solution

**Time Complexity: O(N²)**

- Nested loops: outer loop runs N times, inner loop runs up to N times
- Total pairs: N × (N-1) / 2 ≈ N²/2
- Each pair: O(1) to calculate area

**Space Complexity: O(1)**

- Same as optimal solution

### Comparison

| Metric | Brute Force | Two-Pointer | Improvement |
|--------|-------------|-------------|-------------|
| Time | O(N²) | O(N) | N times faster |
| Space | O(1) | O(1) | Same |
| Comparisons (N=1000) | ~500,000 | ~1,000 | 500x fewer |
| Comparisons (N=10⁵) | ~5×10⁹ | ~10⁵ | 50,000x fewer |

## Production Considerations

### 1. Input Validation

```python
def validate_height_array(height: List[int]) -> Tuple[bool, Optional[str]]:
    """
    Validate input for production use.
    
    Returns:
        (is_valid, error_message)
    """
    if not height:
        return False, "Height array cannot be empty"
    
    if len(height) < 2:
        return False, "Need at least 2 lines"
    
    if len(height) > 10**5:
        return False, "Input too large (max 10^5 elements)"
    
    for i, h in enumerate(height):
        if not isinstance(h, (int, float)):
            return False, f"Invalid type at index {i}: {type(h)}"
        
        if h < 0:
            return False, f"Negative height at index {i}: {h}"
        
        if h > 10**4:
            return False, f"Height too large at index {i}: {h} (max 10^4)"
    
    return True, None
```

### 2. Monitoring and Metrics

```python
from dataclasses import dataclass
from time import time

@dataclass
class PerformanceMetrics:
    """Track performance metrics."""
    execution_time_ms: float
    comparisons: int
    input_size: int
    max_area: int
    
    @property
    def comparisons_per_element(self) -> float:
        return self.comparisons / self.input_size if self.input_size > 0 else 0
    
    def __str__(self) -> str:
        return (
            f"Performance Metrics:\n"
            f"  Execution time: {self.execution_time_ms:.3f}ms\n"
            f"  Comparisons: {self.comparisons}\n"
            f"  Input size: {self.input_size}\n"
            f"  Comparisons/element: {self.comparisons_per_element:.2f}\n"
            f"  Max area: {self.max_area}"
        )


def max_area_with_metrics(height: List[int]) -> Tuple[int, PerformanceMetrics]:
    """
    Solve problem and return metrics.
    """
    start = time()
    comparisons = 0
    
    left = 0
    right = len(height) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        current_height = min(height[left], height[right])
        current_area = width * current_height
        max_area = max(max_area, current_area)
        comparisons += 1
        
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    execution_time = (time() - start) * 1000  # Convert to ms
    
    metrics = PerformanceMetrics(
        execution_time_ms=execution_time,
        comparisons=comparisons,
        input_size=len(height),
        max_area=max_area
    )
    
    return max_area, metrics
```

### 3. Handling Real-World Data

```python
import numpy as np

def max_area_numpy(height: np.ndarray) -> int:
    """
    NumPy-optimized version for large-scale processing.
    
    Useful when:
    - Processing multiple arrays in batch
    - Integration with ML pipelines
    - Need vectorized operations
    """
    if len(height) < 2:
        return 0
    
    left = 0
    right = len(height) - 1
    max_area = 0
    
    # Convert to int32 for efficiency
    height = height.astype(np.int32)
    
    while left < right:
        width = right - left
        current_height = min(height[left], height[right])
        current_area = width * current_height
        max_area = max(max_area, current_area)
        
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return int(max_area)


def batch_max_area(height_arrays: List[List[int]]) -> List[int]:
    """
    Process multiple height arrays in batch.
    
    Useful for:
    - ML feature engineering
    - Batch processing of user inputs
    - A/B testing different configurations
    """
    return [max_area_optimized(heights) for heights in height_arrays]
```

### 4. Error Handling and Resilience

```python
from enum import Enum
from typing import Union

class ErrorCode(Enum):
    SUCCESS = 0
    INVALID_INPUT = 1
    EMPTY_ARRAY = 2
    INSUFFICIENT_ELEMENTS = 3
    OUT_OF_BOUNDS = 4


class ContainerResult:
    """Result wrapper with error handling."""
    
    def __init__(self, value: int, error_code: ErrorCode, message: str = ""):
        self.value = value
        self.error_code = error_code
        self.message = message
    
    @property
    def is_success(self) -> bool:
        return self.error_code == ErrorCode.SUCCESS
    
    def __repr__(self) -> str:
        if self.is_success:
            return f"ContainerResult(value={self.value})"
        return f"ContainerResult(error={self.error_code.name}, message='{self.message}')"


def safe_max_area(height: List[int]) -> ContainerResult:
    """
    Production-safe version with comprehensive error handling.
    """
    # Validate input
    if not height:
        return ContainerResult(0, ErrorCode.EMPTY_ARRAY, "Height array is empty")
    
    if len(height) < 2:
        return ContainerResult(
            0, 
            ErrorCode.INSUFFICIENT_ELEMENTS,
            f"Need at least 2 elements, got {len(height)}"
        )
    
    # Check for invalid values
    for i, h in enumerate(height):
        if not isinstance(h, (int, float)):
            return ContainerResult(
                0,
                ErrorCode.INVALID_INPUT,
                f"Invalid type at index {i}: {type(h)}"
            )
        if h < 0 or h > 10**4:
            return ContainerResult(
                0,
                ErrorCode.OUT_OF_BOUNDS,
                f"Height {h} at index {i} out of bounds [0, 10000]"
            )
    
    # Compute result
    try:
        left = 0
        right = len(height) - 1
        max_area = 0
        
        while left < right:
            width = right - left
            current_height = min(height[left], height[right])
            current_area = width * current_height
            max_area = max(max_area, current_area)
            
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return ContainerResult(max_area, ErrorCode.SUCCESS)
    
    except Exception as e:
        return ContainerResult(0, ErrorCode.INVALID_INPUT, f"Unexpected error: {str(e)}")
```

## Connections to ML Systems

The **greedy optimization and resource management** principles from this problem directly apply to ML system design:

### 1. Resource Allocation (Day 13 ML System Design)

Just like finding the maximum water container:

**Container Problem:** Maximize area given two heights (bottleneck determines capacity)

**ML Resource Allocation:**
- **CPU/GPU allocation:** Limited by the slowest model in the pipeline
- **Memory management:** Constrained by the component with highest memory footprint
- **Throughput optimization:** Bottlenecked by the slowest processing stage

```python
# Analogy: Allocating compute resources
class ResourceAllocator:
    """
    Similar greedy strategy for ML resource allocation.
    """
    def optimize_allocation(self, component_needs: List[int], total_budget: int):
        """
        Greedy allocation: prioritize bottlenecks (like moving shorter pointer).
        
        Args:
            component_needs: Resource requirements for each component
            total_budget: Total available resources
        """
        # Sort by need (like sorting heights)
        sorted_needs = sorted(enumerate(component_needs), key=lambda x: x[1])
        
        allocation = [0] * len(component_needs)
        remaining = total_budget
        
        # Greedy: allocate to bottlenecks first
        for idx, need in sorted_needs:
            alloc = min(need, remaining)
            allocation[idx] = alloc
            remaining -= alloc
            
            if remaining == 0:
                break
        
        return allocation
```

### 2. Batch Size Optimization

In ML training, choosing batch size is analogous:
- **Large batch:** More GPU utilization but might not fit in memory (like wide container but shallow)
- **Small batch:** Better convergence but slower training (like narrow container but potentially tall)
- **Optimal:** Two-pointer approach to find the sweet spot

### 3. Model Serving

When serving ML models:

```python
class ModelServingOptimizer:
    """
    Optimize model serving using greedy strategies.
    
    Similar to container problem:
    - Throughput (width) vs Quality (height)
    - Find optimal trade-off point
    """
    def find_optimal_config(self, latency_sla: int, quality_threshold: float):
        """
        Use greedy approach to find optimal model configuration.
        
        Analogy:
        - Left pointer: simpler/faster models
        - Right pointer: complex/accurate models
        - Objective: maximize value (accuracy × throughput)
        """
        # Start with full range of model sizes
        left_model = "nano"  # fastest, least accurate
        right_model = "xlarge"  # slowest, most accurate
        
        # Greedy: move based on which constraint is tighter
        # (bottleneck determines system performance)
        pass  # Implementation details
```

### 4. Pipeline Optimization

The two-pointer technique applies to ML pipeline optimization:

```python
def optimize_pipeline_stages(stage_latencies: List[int], 
                             stage_accuracies: List[float]) -> Tuple[int, int]:
    """
    Find optimal pipeline configuration.
    
    Trade-off:
    - More stages (wider): Higher accuracy but slower
    - Fewer stages (narrower): Faster but less accurate
    
    Similar to container: maximize value within constraints.
    """
    left = 0  # Minimum stages
    right = len(stage_latencies) - 1  # Maximum stages
    
    best_value = 0
    best_config = (0, 0)
    
    while left <= right:
        # Calculate current configuration value
        total_latency = sum(stage_latencies[left:right+1])
        avg_accuracy = sum(stage_accuracies[left:right+1]) / (right - left + 1)
        
        # Value function (customize based on requirements)
        value = avg_accuracy / total_latency  # Accuracy per unit time
        
        if value > best_value:
            best_value = value
            best_config = (left, right)
        
        # Greedy decision based on bottleneck
        if stage_latencies[left] > stage_latencies[right]:
            left += 1  # Remove slow stage from left
        else:
            right -= 1  # Remove slow stage from right
    
    return best_config
```

### Key Parallels

| Container Problem | ML Systems |
|-------------------|------------|
| Two heights | Two resource types (CPU/memory, speed/accuracy) |
| Width | Scale/throughput |
| Bottleneck (shorter line) | System bottleneck (slowest component) |
| Greedy optimization | Resource allocation strategy |
| O(N) efficiency | Efficient resource scanning |

The **greedy choice** principle is fundamental to both:
- **Container:** Move the pointer at the bottleneck (shorter line)
- **ML Systems:** Allocate resources to the bottleneck component first

## Interview Strategy

### How to Approach This in an Interview

**1. Clarify Requirements (1-2 minutes)**
```
Questions to ask:
- Can heights be negative? (No, from constraints)
- Can we modify the input array? (Not needed)
- What's the expected input size? (Up to 10^5)
- Any special cases? (minimum 2 elements)
```

**2. Start with Brute Force (2-3 minutes)**
```
"Let me start with a straightforward approach:
- Try all pairs of lines
- Calculate area for each
- Track maximum
- This is O(N²) but guarantees correctness"
```

**3. Identify Optimization (2-3 minutes)**
```
"The brute force is too slow for N=10^5. Let me think about optimization:
- Key insight: shorter line is always the bottleneck
- If we start wide and move inward, we can use greedy choice
- Always move the pointer at the shorter line
- This ensures we don't miss optimal solution
- Achieves O(N) time"
```

**4. Implement Optimal Solution (5-7 minutes)**
- Write clean, commented code
- Explain as you write
- Handle edge cases

**5. Test (2-3 minutes)**
- Walk through example
- Test edge cases (length 2, all same height)
- Verify complexity

### Common Mistakes to Avoid

1. **Wrong greedy choice:** Moving the taller pointer
   - This can only decrease area (width decreases, height can't increase)

2. **Off-by-one errors:** Using `<=` instead of `<` in while loop
   - Pointers should meet but not cross

3. **Incorrect area calculation:** Forgetting to use `min` for height
   - Water is limited by shorter line

4. **Missing edge cases:** Not handling arrays of length 2

5. **Complexity analysis:** Claiming O(N) but implementing O(N²)

### Follow-up Questions

**Q1: What if we can remove k lines to maximize area?**
```python
def max_area_remove_k(height: List[int], k: int) -> int:
    """
    Extension: Can remove up to k lines to maximize area.
    
    Approach:
    - Use sliding window of size (n - k)
    - For each window, find max area using two-pointer
    - Track global maximum
    
    Time: O(N × k) in worst case
    """
    n = len(height)
    if n - k < 2:
        return 0
    
    max_area = 0
    
    # Try all possible removals
    # More sophisticated: use DP or greedy heuristics
    # For interview: explain approach without full implementation
    
    return max_area
```

**Q2: What if lines have different costs, and we want to maximize area per unit cost?**
```python
def max_area_per_cost(height: List[int], costs: List[int]) -> Tuple[int, float]:
    """
    Maximize area / cost ratio.
    
    Approach:
    - Still use two-pointer for efficiency
    - Calculate area/cost for each configuration
    - Track best ratio
    """
    left = 0
    right = len(height) - 1
    best_ratio = 0
    best_area = 0
    
    while left < right:
        width = right - left
        h = min(height[left], height[right])
        area = width * h
        cost = costs[left] + costs[right]
        ratio = area / cost if cost > 0 else 0
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_area = area
        
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return best_area, best_ratio
```

**Q3: How would you parallelize this for huge datasets?**
```
Answer:
"The two-pointer approach is inherently sequential because each
step depends on the previous decision. However, for huge datasets:

1. Batch processing: Split input into chunks, find local max in each
2. MapReduce: Map phase finds local optima, Reduce phase combines
3. Approximation: Sample subset of lines for quick estimate
4. GPU acceleration: Use CUDA for brute force on GPU (might be faster than O(N) on CPU for moderate N)"
```

### Time Allocation (45-minute interview)

- Problem understanding: 2 min
- Brute force discussion: 3 min
- Optimal approach design: 3 min
- Implementation: 10 min
- Testing: 3 min
- Complexity analysis: 2 min
- Follow-ups/discussion: 7 min
- Buffer: 5 min

## Key Takeaways

✅ **Two-pointer technique** is essential for O(N) optimization in array problems

✅ **Greedy algorithms** work when local optimal choices lead to global optimum

✅ **Bottleneck analysis** is crucial - the limiting factor determines system behavior

✅ **Width vs height trade-off** appears in many real systems (scale vs quality, throughput vs latency)

✅ **Resource allocation** in ML follows similar greedy optimization principles

✅ **Start with brute force** in interviews to show you understand the problem

✅ **Optimize systematically** by identifying what makes brute force slow

✅ **Proof of correctness** matters - explain why greedy choice doesn't miss optimal solution

✅ **Production considerations** include input validation, metrics, error handling

✅ **Cross-domain application** - same principles apply to container problem, ML resource allocation, and compute allocation for speech models

### Mental Model

Think of this problem as:
- **Container:** Two-pointer greedy for max area
- **ML System:** Bottleneck resource limits system throughput
- **Speech System:** Compute allocation must address weakest component first

All three share the fundamental insight: **The bottleneck determines system capacity, so greedy optimization should target the bottleneck first.**

---

**Originally published at:** [arunbaby.com/dsa/0013-container-with-most-water](https://www.arunbaby.com/dsa/0013-container-with-most-water/)

*If you found this helpful, consider sharing it with others who might benefit.*

