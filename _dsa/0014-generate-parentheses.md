---
title: "Generate Parentheses"
day: 14
collection: dsa
categories:
  - dsa
tags:
  - backtracking
  - recursion
  - string
  - catalan-numbers
  - combination-generation
  - medium-easy
subdomain: "Recursion & Backtracking"
tech_stack: [Python]
scale: "O(4^N / √N) time, O(N) space"
companies: [Google, Meta, Amazon, Microsoft, Apple, Uber]
related_ml_day: 14
related_speech_day: 14
related_agents_day: 14
---

**Master backtracking to generate all valid combinations—the foundation of ensemble model selection and multi-model systems.**

## Problem Statement

Given `n` pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

### Examples

**Example 1:**
```
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
```

**Example 2:**
```
Input: n = 1
Output: ["()"]
```

**Example 3:**
```
Input: n = 2
Output: ["(())","()()"]
```

### Constraints

- `1 <= n <= 8`

## Understanding the Problem

This is a **canonical backtracking problem** that teaches us how to:
1. **Generate all valid combinations** from a search space
2. **Prune invalid paths early** (optimization)
3. **Build solutions incrementally** (recursive construction)
4. **Validate constraints** during generation

### What Makes Parentheses Valid?

A string of parentheses is valid if:
1. Every opening `(` has a corresponding closing `)`
2. At no point do we have more closing `)` than opening `(`
3. Total opening = total closing = `n`

**Examples:**
- Valid: `()`, `(())`, `()()`
- Invalid: `)(`, `()(`, `(()`

### Why This Problem Matters

1. **Backtracking pattern:** Core technique for combinatorial problems
2. **Constraint satisfaction:** Generate only valid solutions
3. **Tree exploration:** Navigate decision trees efficiently
4. **Real-world applications:**
   - Compiler design (expression parsing)
   - ML ensemble selection (choose model combinations)
   - Configuration generation (all valid system configs)

### The Catalan Number Connection

The number of valid parentheses strings with `n` pairs is the `n`-th **Catalan number**:

\[
C_n = \frac{1}{n+1}\binom{2n}{n} = \frac{(2n)!}{(n+1)!n!}
\]

| n | Valid combinations | Catalan number |
|---|-------------------|----------------|
| 1 | 1 | C₁ = 1 |
| 2 | 2 | C₂ = 2 |
| 3 | 5 | C₃ = 5 |
| 4 | 14 | C₄ = 14 |
| 5 | 42 | C₅ = 42 |
| 8 | 1430 | C₈ = 1430 |

This tells us the **size of our search space**—the number of solutions we need to generate.

## Approach 1: Brute Force - Generate All, Then Filter

### Intuition

Generate all possible strings of `2n` characters using `(` and `)`, then filter out the invalid ones.

### Implementation

```python
from itertools import product
from typing import List

def generateParenthesis_bruteforce(n: int) -> List[str]:
    """
    Brute force: generate all 2^(2n) combinations, filter valid ones.
    
    Time: O(2^(2n) × n) - generate all strings, validate each
    Space: O(2^(2n)) - store all combinations
    
    Why this approach?
    - Simple to understand
    - Shows the search space size
    - Demonstrates need for optimization
    
    Problem:
    - Extremely wasteful (generates many invalid strings)
    - Exponential in unoptimized space
    """
    def is_valid(s: str) -> bool:
        """Check if parentheses string is valid."""
        balance = 0
        for char in s:
            if char == '(':
                balance += 1
            else:
                balance -= 1
            
            # More closing than opening
            if balance < 0:
                return False
        
        # Must be balanced at end
        return balance == 0
    
    # Generate all possible strings of length 2n
    # Each position can be '(' or ')'
    all_combinations = []
    
    # Use binary representation: 0 = '(', 1 = ')'
    # Total: 2^(2n) combinations
    for i in range(2 ** (2 * n)):
        s = []
        num = i
        
        for _ in range(2 * n):
            if num % 2 == 0:
                s.append('(')
            else:
                s.append(')')
            num //= 2
        
        candidate = ''.join(s)
        if is_valid(candidate):
            all_combinations.append(candidate)
    
    return all_combinations


# Test
print(generateParenthesis_bruteforce(3))
# Output: ['((()))', '(()())', '(())()', '()(())', '()()()']
```

### Analysis

**Time Complexity: O(2^(2n) × n)**
- Generate 2^(2n) strings
- Validate each in O(n) time

**Space Complexity: O(2^(2n))**
- Store all combinations

**For n=8:**
- Generate: 2^16 = 65,536 strings
- Valid: only 1,430 (2.2%)
- **98% waste!**

This is clearly inefficient. We need a smarter approach.

## Approach 2: Backtracking (Optimal)

### The Key Insight

**Instead of generating all strings and filtering, generate only valid strings.**

We can build valid strings character by character, making decisions that maintain validity:

**Decision at each step:**
1. **Add `(`:** Only if we haven't used all `n` opening parentheses
2. **Add `)`:** Only if it won't make the string invalid (i.e., `open_count > close_count`)

This is **backtracking with constraint checking**.

### Backtracking Template

```python
def backtrack(current_state):
    if is_solution(current_state):
        add_to_results(current_state)
        return
    
    for choice in possible_choices:
        if is_valid_choice(choice, current_state):
            make_choice(choice)
            backtrack(new_state)
            undo_choice(choice)  # Backtrack
```

### Implementation

```python
def generateParenthesis(n: int) -> List[str]:
    """
    Optimal backtracking solution.
    
    Time: O(4^n / √n) - Catalan number complexity
    Space: O(n) - recursion depth
    
    Algorithm:
    1. Build string character by character
    2. At each step, decide: add '(' or ')'?
    3. Constraints:
       - Can add '(' if open_count < n
       - Can add ')' if close_count < open_count
    4. Base case: length = 2n (complete string)
    
    Why this works:
    - Only generates valid strings (no wasted work)
    - Prunes invalid branches early
    - Explores decision tree systematically
    """
    result = []
    
    def backtrack(current: str, open_count: int, close_count: int):
        """
        Build valid parentheses strings recursively.
        
        Args:
            current: String built so far
            open_count: Number of '(' used
            close_count: Number of ')' used
        """
        # Base case: we've used all n pairs
        if len(current) == 2 * n:
            result.append(current)
            return
        
        # Choice 1: Add opening parenthesis
        # Constraint: haven't used all n opening parens
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        
        # Choice 2: Add closing parenthesis
        # Constraint: won't create invalid string
        # (must have more opens than closes)
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)
    
    # Start with empty string
    backtrack('', 0, 0)
    return result
```

### Step-by-Step Visualization (n=3)

```
Start: "", open=0, close=0

                    ""
                    ↓
                   "("  (open=1, close=0)
              ┌─────┴─────┐
             "("          "()"
        (open=2)      (open=1, close=1)
         ↓                ↓
      ┌──"(("──┐       ┌─"()("─┐
     "((("    "(()"   "()()"  "()("
       ↓        ↓        ↓       ↓
     "((())"  "(()()"  "()(()"  ...
     
[Continue until all paths reach length 6]

Valid outputs:
1. "((()))"  - all opens first, then all closes
2. "(()())"  - interleaved pattern
3. "(())()"  - group of 2, then single pair
4. "()(())"  - single pair, then group of 2
5. "()()()"  - all separate pairs
```

### Decision Tree Analysis

At each node, we have up to 2 choices: add `(` or add `)`.

**Pruning happens when:**
- `open_count >= n` → can't add more `(`
- `close_count >= open_count` → can't add `)`

This dramatically reduces the search space:
- Brute force: 2^(2n) = 64 strings for n=3
- Backtracking: Only explores 5 valid paths
- **92% reduction!**

## Approach 3: Backtracking with String Builder (Memory Optimized)

### Optimization

Instead of creating new strings at each step (`current + '('`), use a list and modify in place.

```python
def generateParenthesis_optimized(n: int) -> List[str]:
    """
    Memory-optimized backtracking using list instead of string concatenation.
    
    Why?
    - String concatenation creates new objects (O(n) per operation)
    - List append/pop is O(1)
    - Reduces memory allocations
    
    Time: O(4^n / √n) - same as before
    Space: O(n) - reuse same list
    """
    result = []
    
    def backtrack(path: List[str], open_count: int, close_count: int):
        """
        Args:
            path: Mutable list of characters (instead of immutable string)
        """
        # Base case
        if len(path) == 2 * n:
            result.append(''.join(path))
            return
        
        # Add '('
        if open_count < n:
            path.append('(')
            backtrack(path, open_count + 1, close_count)
            path.pop()  # Backtrack (undo choice)
        
        # Add ')'
        if close_count < open_count:
            path.append(')')
            backtrack(path, open_count, close_count + 1)
            path.pop()  # Backtrack (undo choice)
    
    backtrack([], 0, 0)
    return result
```

### The Explicit Backtracking

Notice the pattern:
```python
path.append('(')      # Make choice
backtrack(...)        # Recurse
path.pop()            # Undo choice (backtrack)
```

This is the **essence of backtracking**: try a choice, explore its consequences, then undo it to try other choices.

## Approach 4: Iterative with Stack (No Recursion)

For completeness, here's an iterative version:

```python
def generateParenthesis_iterative(n: int) -> List[str]:
    """
    Iterative version using explicit stack.
    
    Converts recursion to iteration.
    Useful when recursion depth might be a concern.
    
    Time: O(4^n / √n)
    Space: O(4^n / √n) - for the stack
    """
    result = []
    
    # Stack stores: (current_string, open_count, close_count)
    stack = [('', 0, 0)]
    
    while stack:
        current, open_count, close_count = stack.pop()
        
        # Base case
        if len(current) == 2 * n:
            result.append(current)
            continue
        
        # Add choices to stack (reverse order for DFS)
        if close_count < open_count:
            stack.append((current + ')', open_count, close_count + 1))
        
        if open_count < n:
            stack.append((current + '(', open_count + 1, close_count))
    
    return result
```

## Implementation: Production-Grade Solution

```python
from typing import List, Set
from functools import lru_cache
import logging

class ParenthesesGenerator:
    """
    Production-ready parentheses generator with caching and validation.
    
    Features:
    - Multiple algorithms
    - Input validation
    - Result caching
    - Performance metrics
    """
    
    def __init__(self, algorithm: str = "backtracking"):
        """
        Initialize generator.
        
        Args:
            algorithm: "backtracking", "optimized", or "iterative"
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger(__name__)
        self.call_count = 0
        
        # Cache for memoization
        self._cache = {}
    
    def generate(self, n: int) -> List[str]:
        """
        Generate all valid parentheses combinations.
        
        Args:
            n: Number of pairs
            
        Returns:
            List of valid parentheses strings
            
        Raises:
            ValueError: If n is invalid
        """
        # Validate input
        if not isinstance(n, int):
            raise ValueError(f"n must be an integer, got {type(n)}")
        
        if n < 1 or n > 8:
            raise ValueError(f"n must be between 1 and 8, got {n}")
        
        # Check cache
        if n in self._cache:
            self.logger.debug(f"Cache hit for n={n}")
            return self._cache[n]
        
        # Generate based on algorithm
        if self.algorithm == "backtracking":
            result = self._backtracking(n)
        elif self.algorithm == "optimized":
            result = self._optimized(n)
        elif self.algorithm == "iterative":
            result = self._iterative(n)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Cache and return
        self._cache[n] = result
        self.call_count += 1
        
        self.logger.info(
            f"Generated {len(result)} combinations for n={n} "
            f"using {self.algorithm}"
        )
        
        return result
    
    def _backtracking(self, n: int) -> List[str]:
        """Standard backtracking implementation."""
        result = []
        
        def backtrack(current: str, open_count: int, close_count: int):
            if len(current) == 2 * n:
                result.append(current)
                return
            
            if open_count < n:
                backtrack(current + '(', open_count + 1, close_count)
            
            if close_count < open_count:
                backtrack(current + ')', open_count, close_count + 1)
        
        backtrack('', 0, 0)
        return result
    
    def _optimized(self, n: int) -> List[str]:
        """Optimized backtracking with list."""
        result = []
        
        def backtrack(path: List[str], open_count: int, close_count: int):
            if len(path) == 2 * n:
                result.append(''.join(path))
                return
            
            if open_count < n:
                path.append('(')
                backtrack(path, open_count + 1, close_count)
                path.pop()
            
            if close_count < open_count:
                path.append(')')
                backtrack(path, open_count, close_count + 1)
                path.pop()
        
        backtrack([], 0, 0)
        return result
    
    def _iterative(self, n: int) -> List[str]:
        """Iterative implementation."""
        result = []
        stack = [('', 0, 0)]
        
        while stack:
            current, open_count, close_count = stack.pop()
            
            if len(current) == 2 * n:
                result.append(current)
                continue
            
            if close_count < open_count:
                stack.append((current + ')', open_count, close_count + 1))
            
            if open_count < n:
                stack.append((current + '(', open_count + 1, close_count))
        
        return result
    
    @staticmethod
    def is_valid(s: str) -> bool:
        """
        Validate a parentheses string.
        
        Args:
            s: String to validate
            
        Returns:
            True if valid, False otherwise
        """
        balance = 0
        
        for char in s:
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            else:
                return False  # Invalid character
            
            if balance < 0:
                return False  # More closes than opens
        
        return balance == 0  # Must be balanced
    
    @staticmethod
    def catalan_number(n: int) -> int:
        """
        Calculate the n-th Catalan number.
        
        This is the expected number of valid combinations.
        
        Formula: C_n = (2n)! / ((n+1)! * n!)
        """
        if n <= 1:
            return 1
        
        # Calculate using dynamic programming to avoid overflow
        catalan = [0] * (n + 1)
        catalan[0] = catalan[1] = 1
        
        for i in range(2, n + 1):
            for j in range(i):
                catalan[i] += catalan[j] * catalan[i - 1 - j]
        
        return catalan[n]
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        return {
            "algorithm": self.algorithm,
            "cache_size": len(self._cache),
            "total_calls": self.call_count,
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test different algorithms
    for algo in ["backtracking", "optimized", "iterative"]:
        print(f"\n=== Testing {algo} ===")
        
        generator = ParenthesesGenerator(algorithm=algo)
        
        for n in [1, 2, 3, 4]:
            result = generator.generate(n)
            expected_count = generator.catalan_number(n)
            
            print(f"n={n}: {len(result)} combinations (expected: {expected_count})")
            if n <= 3:
                print(f"  {result}")
            
            # Validate all results
            assert all(generator.is_valid(s) for s in result)
            assert len(result) == expected_count
        
        print(f"Stats: {generator.get_stats()}")
```

## Testing

### Comprehensive Test Suite

```python
import pytest
from typing import List

class TestParenthesesGenerator:
    """Comprehensive test suite."""
    
    @pytest.fixture
    def generator(self):
        return ParenthesesGenerator(algorithm="backtracking")
    
    def test_base_cases(self, generator):
        """Test base cases."""
        assert generator.generate(1) == ["()"]
        assert len(generator.generate(2)) == 2
        assert len(generator.generate(3)) == 5
    
    def test_catalan_numbers(self, generator):
        """Test that output matches Catalan numbers."""
        test_cases = [
            (1, 1),
            (2, 2),
            (3, 5),
            (4, 14),
            (5, 42),
            (6, 132),
            (7, 429),
            (8, 1430),
        ]
        
        for n, expected_count in test_cases:
            result = generator.generate(n)
            assert len(result) == expected_count
            assert len(result) == generator.catalan_number(n)
    
    def test_all_valid(self, generator):
        """Test that all generated strings are valid."""
        for n in range(1, 9):
            result = generator.generate(n)
            
            for s in result:
                assert generator.is_valid(s), f"Invalid string: {s}"
                assert len(s) == 2 * n
    
    def test_no_duplicates(self, generator):
        """Test that there are no duplicate results."""
        for n in range(1, 9):
            result = generator.generate(n)
            assert len(result) == len(set(result)), f"Duplicates found for n={n}"
    
    def test_specific_cases(self, generator):
        """Test specific known results."""
        # n=2
        result = set(generator.generate(2))
        expected = {"(())", "()()"}
        assert result == expected
        
        # n=3
        result = set(generator.generate(3))
        expected = {"((()))", "(()())", "(())()", "()(())", "()()()"}
        assert result == expected
    
    def test_invalid_input(self, generator):
        """Test input validation."""
        with pytest.raises(ValueError):
            generator.generate(0)
        
        with pytest.raises(ValueError):
            generator.generate(9)
        
        with pytest.raises(ValueError):
            generator.generate(-1)
        
        with pytest.raises(ValueError):
            generator.generate("3")
    
    def test_caching(self, generator):
        """Test that caching works."""
        # First call
        result1 = generator.generate(5)
        
        # Second call should hit cache
        result2 = generator.generate(5)
        
        # Should return same result
        assert result1 == result2
        
        # Cache should contain n=5
        assert 5 in generator._cache
    
    def test_algorithms_equivalent(self):
        """Test that all algorithms produce same results."""
        n = 4
        
        bt = ParenthesesGenerator("backtracking").generate(n)
        opt = ParenthesesGenerator("optimized").generate(n)
        it = ParenthesesGenerator("iterative").generate(n)
        
        # Convert to sets for comparison (order doesn't matter)
        assert set(bt) == set(opt) == set(it)
    
    def test_validation_function(self, generator):
        """Test the is_valid function."""
        # Valid cases
        assert generator.is_valid("()")
        assert generator.is_valid("(())")
        assert generator.is_valid("()()")
        assert generator.is_valid("((()))")
        
        # Invalid cases
        assert not generator.is_valid("(")
        assert not generator.is_valid(")")
        assert not generator.is_valid(")(")
        assert not generator.is_valid("(()")
        assert not generator.is_valid("())")
        assert not generator.is_valid("((")
        assert not generator.is_valid("abc")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Complexity Analysis

### Time Complexity: O(4^n / √n)

This is the **n-th Catalan number** complexity.

**Why 4^n / √n?**

Using Stirling's approximation:

\[
C_n = \frac{1}{n+1}\binom{2n}{n} \approx \frac{4^n}{n^{3/2}\sqrt{\pi}}
\]

**Intuitive explanation:**
- At each step, we make a choice: `(` or `)`
- Naive bound: 2^(2n) (two choices, 2n steps)
- But constraints prune most branches
- Actual valid paths: roughly 4^n / √n

**For practical values:**

| n | Catalan C_n | 4^n / √n (approx) |
|---|-------------|-------------------|
| 1 | 1 | 4 |
| 2 | 2 | 5.7 |
| 3 | 5 | 11.6 |
| 4 | 14 | 32 |
| 5 | 42 | 102 |
| 8 | 1430 | 2,309 |

### Space Complexity: O(n)

**Recursion depth:** O(n)
- Maximum depth is 2n (length of string)
- Each recursive call adds a frame to the stack

**Result storage:** O(4^n / √n × n)
- Store C_n strings
- Each string has length 2n

**For the recursive solution:**
- Call stack: O(n)
- Temporary strings during construction: O(n)
- Output array: O(C_n × n)

## Production Considerations

### 1. Performance Optimization

```python
import time
from functools import wraps

def timing_decorator(func):
    """Measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        
        print(f"{func.__name__} took {(end - start) * 1000:.2f}ms")
        return result
    
    return wrapper


@timing_decorator
def benchmark_algorithms(n: int):
    """Compare algorithm performance."""
    results = {}
    
    for algo in ["backtracking", "optimized", "iterative"]:
        gen = ParenthesesGenerator(algorithm=algo)
        start = time.perf_counter()
        result = gen.generate(n)
        end = time.perf_counter()
        
        results[algo] = {
            "time_ms": (end - start) * 1000,
            "count": len(result)
        }
    
    return results


# Benchmark
for n in [5, 6, 7, 8]:
    print(f"\n=== n={n} ===")
    results = benchmark_algorithms(n)
    for algo, stats in results.items():
        print(f"{algo:12}: {stats['time_ms']:6.2f}ms ({stats['count']} results)")
```

### 2. Streaming Results

For large `n`, generate results one at a time instead of building the entire list:

```python
def generate_parentheses_stream(n: int):
    """
    Generator version - yields results one at a time.
    
    Benefits:
    - Lower memory usage
    - Can process results as they're generated
    - Early termination possible
    """
    def backtrack(current: str, open_count: int, close_count: int):
        if len(current) == 2 * n:
            yield current
            return
        
        if open_count < n:
            yield from backtrack(current + '(', open_count + 1, close_count)
        
        if close_count < open_count:
            yield from backtrack(current + ')', open_count, close_count + 1)
    
    yield from backtrack('', 0, 0)


# Usage
for i, parens in enumerate(generate_parentheses_stream(8)):
    print(f"{i+1}. {parens}")
    if i >= 10:  # Stop after first 10
        print("...")
        break
```

### 3. Parallel Generation

For very large `n`, parallelize by distributing different subtrees:

```python
from multiprocessing import Pool
from functools import partial

def generate_subtree(prefix: str, n: int) -> List[str]:
    """Generate all valid completions starting with prefix."""
    result = []
    
    # Count opens and closes in prefix
    open_count = prefix.count('(')
    close_count = prefix.count(')')
    
    def backtrack(current: str, open_c: int, close_c: int):
        if len(current) == 2 * n:
            result.append(current)
            return
        
        if open_c < n:
            backtrack(current + '(', open_c + 1, close_c)
        
        if close_c < open_c:
            backtrack(current + ')', open_c, close_c + 1)
    
    backtrack(prefix, open_count, close_count)
    return result


def generate_parallel(n: int, num_processes: int = 4) -> List[str]:
    """
    Parallel generation using multiprocessing.
    
    Strategy:
    1. Generate prefixes of length k
    2. Distribute prefixes to workers
    3. Each worker completes its subtree
    4. Merge results
    """
    if n <= 4:
        # Not worth parallelizing for small n
        return ParenthesesGenerator().generate(n)
    
    # Generate prefixes of length 6
    prefixes = []
    
    def gen_prefixes(current: str, open_c: int, close_c: int):
        if len(current) == 6:
            prefixes.append(current)
            return
        
        if open_c < n:
            gen_prefixes(current + '(', open_c + 1, close_c)
        
        if close_c < open_c:
            gen_prefixes(current + ')', open_c, close_c + 1)
    
    gen_prefixes('', 0, 0)
    
    # Process prefixes in parallel
    with Pool(num_processes) as pool:
        func = partial(generate_subtree, n=n)
        results = pool.map(func, prefixes)
    
    # Flatten results
    return [item for sublist in results for item in sublist]
```

## Connections to ML Systems

The **backtracking and combination generation** pattern from this problem directly applies to ML ensemble systems:

### 1. Model Ensemble Selection

**Problem:** Given N trained models, select the best subset for an ensemble.

**Similarity to Generate Parentheses:**
- **Parentheses:** Generate all valid combinations of `(` and `)`
- **Ensembles:** Generate all valid combinations of models

```python
def select_ensemble_models(
    models: List[Model],
    max_models: int,
    constraints: dict
) -> List[List[Model]]:
    """
    Select model combinations using backtracking.
    
    Similar to parentheses generation:
    - State: current model selection
    - Choices: add model or skip model
    - Constraints: max_models, diversity, latency budget
    - Pruning: skip combinations that violate constraints
    """
    result = []
    
    def backtrack(index: int, current_ensemble: List[Model], current_latency: float):
        # Base case: evaluated all models
        if index == len(models):
            if len(current_ensemble) > 0:
                result.append(current_ensemble[:])
            return
        
        # Choice 1: Include current model
        model = models[index]
        new_latency = current_latency + model.latency
        
        # Constraint checking (like parentheses validation)
        if (len(current_ensemble) < max_models and
            new_latency < constraints['max_latency'] and
            is_diverse_enough(current_ensemble, model)):
            
            current_ensemble.append(model)
            backtrack(index + 1, current_ensemble, new_latency)
            current_ensemble.pop()  # Backtrack
        
        # Choice 2: Skip current model
        backtrack(index + 1, current_ensemble, current_latency)
    
    backtrack(0, [], 0.0)
    return result


def is_diverse_enough(ensemble: List[Model], new_model: Model) -> bool:
    """Check if adding new_model maintains diversity."""
    # Ensure different model architectures
    architectures = set(m.architecture for m in ensemble)
    return new_model.architecture not in architectures
```

### 2. Hyperparameter Search

**Problem:** Search hyperparameter space for optimal configuration.

```python
def grid_search_backtracking(
    param_space: dict,
    validator: callable,
    max_trials: int
) -> List[dict]:
    """
    Hyperparameter search using backtracking.
    
    Similar to parentheses:
    - State: current hyperparameter selection
    - Choices: assign value to next hyperparameter
    - Pruning: skip configs that fail validation
    """
    results = []
    param_names = list(param_space.keys())
    
    def backtrack(index: int, current_config: dict):
        if len(results) >= max_trials:
            return  # Early termination
        
        if index == len(param_names):
            # Complete configuration
            if validator(current_config):
                results.append(current_config.copy())
            return
        
        # Try each value for current parameter
        param_name = param_names[index]
        for value in param_space[param_name]:
            current_config[param_name] = value
            
            # Prune: skip if clearly bad (early stopping)
            if is_promising(current_config):
                backtrack(index + 1, current_config)
            
            del current_config[param_name]
    
    backtrack(0, {})
    return results
```

### 3. Feature Combination Selection

**Problem:** Select best feature combinations for a model.

```python
def select_feature_combinations(
    features: List[str],
    min_features: int,
    max_features: int,
    correlation_threshold: float
) -> List[List[str]]:
    """
    Generate valid feature combinations.
    
    Constraints (like parentheses validity):
    - Size bounds: min_features <= |combo| <= max_features
    - Low correlation: features not too similar
    - Coverage: must cover different aspects
    """
    result = []
    
    def backtrack(index: int, current_features: List[str]):
        # Valid combination found
        if min_features <= len(current_features) <= max_features:
            if is_valid_feature_set(current_features, correlation_threshold):
                result.append(current_features[:])
        
        # Stop if at max or end
        if len(current_features) == max_features or index == len(features):
            return
        
        # Try including next feature
        feature = features[index]
        
        # Check if adding maintains validity
        if not conflicts_with(feature, current_features):
            current_features.append(feature)
            backtrack(index + 1, current_features)
            current_features.pop()
        
        # Try excluding next feature
        backtrack(index + 1, current_features)
    
    backtrack(0, [])
    return result
```

### Key Parallels

| Generate Parentheses | ML System Design |
|----------------------|------------------|
| Generate valid strings | Generate valid model combinations |
| Constraint: balanced parens | Constraint: latency/accuracy/diversity |
| Pruning: close > open | Pruning: violates SLA |
| Backtracking | Backtracking |
| Result: all valid strings | Result: all viable ensembles |

## Interview Strategy

### How to Approach in an Interview

**1. Clarify (1 min)**
```
- n is always >= 1?
- Need all combinations or just one?
- Any memory constraints?
- Sorted output required?
```

**2. Explain Intuition (2 min)**
```
"This is a classic backtracking problem. We build valid strings
character by character, making choices at each step:
- Add '(' if we haven't used all n
- Add ')' if it won't create invalid string

This is like exploring a decision tree, where each path represents
a sequence of choices."
```

**3. Discuss Approaches (2 min)**
```
"We could:
1. Brute force: Generate all 2^(2n) strings, filter valid ones
   - Too slow, O(2^(2n))
2. Backtracking: Only generate valid strings
   - Optimal, O(4^n / √n)
   - This is what I'll implement"
```

**4. Code (10 min)**
- Start with clear function signature
- Explain constraints as you code
- Add comments for clarity

**5. Test (3 min)**
- Walk through example: n=2
- Test edge case: n=1
- Mention complexity

**6. Follow-ups (5 min)**

### Common Mistakes

1. **Forgetting constraint checking**
   ```python
   # Wrong: might add ')' when invalid
   backtrack(current + ')')
   
   # Correct: check constraint first
   if close_count < open_count:
       backtrack(current + ')')
   ```

2. **Off-by-one errors**
   ```python
   # Wrong
   if open_count <= n:  # Should be <
   
   # Correct
   if open_count < n:
   ```

3. **Not handling base case**
   ```python
   # Need to check when to stop
   if len(current) == 2 * n:
       result.append(current)
       return
   ```

4. **Forgetting to backtrack in iterative version**
   - Must undo choices when backtracking

### Follow-up Questions

**Q1: Return only the first k valid combinations?**
```python
def generateParenthesis_first_k(n: int, k: int) -> List[str]:
    """Return first k valid combinations."""
    result = []
    
    def backtrack(current: str, open_count: int, close_count: int):
        if len(result) >= k:
            return  # Early termination
        
        if len(current) == 2 * n:
            result.append(current)
            return
        
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        
        if close_count < open_count and len(result) < k:
            backtrack(current + ')', open_count, close_count + 1)
    
    backtrack('', 0, 0)
    return result
```

**Q2: Generate parentheses with multiple types: `()`, `[]`, `{}`?**
```python
def generateParenthesis_multi_type(n: int) -> List[str]:
    """
    Generate with multiple bracket types.
    
    Additional constraint: must match types
    - '(' matches with ')'
    - '[' matches with ']'
    - '{' matches with '}'
    """
    result = []
    open_types = ['(', '[', '{']
    close_types = [')', ']', '}']
    match = {'(': ')', '[': ']', '{': '}'}
    
    def backtrack(current: str, stack: List[str]):
        if len(current) == 2 * n:
            if not stack:  # All matched
                result.append(current)
            return
        
        # Add opening bracket
        if len([c for c in current if c in open_types]) < n:
            for open_bracket in open_types:
                backtrack(current + open_bracket, stack + [open_bracket])
        
        # Add closing bracket
        if stack:
            last_open = stack[-1]
            close_bracket = match[last_open]
            backtrack(current + close_bracket, stack[:-1])
    
    backtrack('', [])
    return result
```

**Q3: What's the time complexity and why?**

Answer: O(4^n / √n), which is the n-th Catalan number. This comes from:
- Total valid strings = C_n = (1/(n+1)) * C(2n, n)
- Using Stirling's approximation: C_n ≈ 4^n / (n^(3/2) * √π)
- We generate each valid string once
- Building each string takes O(n) time
- Total: O(n × C_n) ≈ O(4^n / √n)

## Key Takeaways

✅ **Backtracking** is the optimal approach for generating all valid combinations

✅ **Constraint checking** during generation is more efficient than generate-and-filter

✅ **State tracking** (open_count, close_count) enables early pruning

✅ **Decision tree exploration** - each path represents a sequence of choices

✅ **Catalan numbers** describe the count of valid solutions

✅ **String vs list building** - lists are more memory efficient for backtracking

✅ **Caching** can avoid recomputation for repeated queries

✅ **Streaming results** (generators) reduce memory for large n

✅ **Same pattern applies** to ensemble selection, hyperparameter search, feature selection

✅ **Backtracking template** is universally applicable to combinatorial problems

### Mental Model

Think of this problem as:
- **Parentheses generation:** Decision tree of `(` and `)` choices with validity constraints
- **ML ensemble:** Decision tree of model selections with SLA constraints
- **Speech multi-model:** Decision tree of model combinations with latency/accuracy constraints

All use the same backtracking pattern: **make choice → check validity → recurse → undo choice**

---

**Originally published at:** [arunbaby.com/dsa/0014-generate-parentheses](https://www.arunbaby.com/dsa/0014-generate-parentheses/)

*If you found this helpful, consider sharing it with others who might benefit.*

