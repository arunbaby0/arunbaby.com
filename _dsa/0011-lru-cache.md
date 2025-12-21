---
title: "LRU Cache"
day: 11
collection: dsa
categories:
  - dsa
tags:
  - hash-map
  - linked-list
  - design
  - cache
  - lru
  - system-design
subdomain: Data Structures
tech_stack: [Python, Hash Map, Doubly Linked List, OrderedDict]
scale: "O(1) operations, production caching"
companies: [Google, Meta, Amazon, Microsoft, Netflix, LinkedIn]
related_ml_day: 11
related_speech_day: 11
related_agents_day: 11
---

**Master LRU cache design: O(1) get/put with hash map + doubly linked list. Critical for interviews and production caching systems.**

## Problem Statement

Design a data structure that follows the constraints of a **Least Recently Used (LRU) cache**.

Implement the `LRUCache` class:

- `LRUCache(int capacity)` Initialize the LRU cache with positive size capacity.
- `int get(int key)` Return the value of the key if the key exists, otherwise return -1.
- `put(int key, int value)` Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.

The functions `get` and `put` must each run in **O(1)** average time complexity.

### Examples

```
Input:
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]

Output:
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation:
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4
```

### Constraints

- `1 <= capacity <= 3000`
- `0 <= key <= 10^4`
- `0 <= value <= 10^5`
- At most `2 * 10^5` calls will be made to `get` and `put`

---

## Understanding LRU Cache

### What is LRU?

**Least Recently Used (LRU)** is a cache eviction policy that removes the least recently accessed item when the cache is full.

**Access = Read or Write**

When you:
- `get(key)` - The key becomes most recently used
- `put(key, value)` - The key becomes most recently used

### Why LRU? The Intuition

Imagine you're organizing files on your desk. You have limited space, so you stack recent documents on top. When you need space for a new document, you remove the one at the bottom (least recently used).

**The temporal locality principle**: Recently accessed data is likely to be accessed again soon.

**Example scenario:**
```
You visit websites: A → B → C → D → A → B

Notice that A and B are accessed multiple times.
LRU keeps these "hot" items in cache, evicting C or D if space is needed.
```

**Why not other policies?**

| Policy | How it works | Drawback |
|--------|-------------|----------|
| **FIFO** | Remove oldest inserted | Doesn't consider access patterns |
| **Random** | Remove random item | No intelligence, unpredictable |
| **LFU** | Remove least frequently used | Complex to implement, doesn't adapt to changing patterns |
| **LRU** | Remove least recently used | ✅ Simple, adaptive, good hit rates |

### Real-World Examples

**1. Browser Cache**
When you visit websites, your browser caches images and assets. If the cache fills up, it removes pages you haven't visited in a while (LRU).

**2. Database Query Cache**
Databases cache query results. Popular queries (accessed recently) stay in cache, while old queries are evicted.

**3. CDN Edge Caching**
Content Delivery Networks cache content at edge locations. Popular content (recently accessed) stays cached close to users.

**4. Operating System Memory**
When RAM is full, OS moves least recently used pages to disk (swap/page file).

### The Challenge: Achieving O(1) Operations

**The problem**: We need both:
- **Fast lookup** - O(1) to check if key exists
- **Fast reordering** - O(1) to move item to "most recent"
- **Fast eviction** - O(1) to remove "least recent"

**Why is this hard?**

If we use only one data structure:
- **Array**: Lookup O(n), reordering O(n) ❌
- **Hash Map**: Lookup O(1), but no ordering ❌
- **Linked List**: Reordering O(1), but lookup O(n) ❌

**The solution**: Combine both!
- **Hash Map**: For O(1) lookup
- **Doubly Linked List**: For O(1) reordering

### Understanding the Data Structure Choice

**Why Doubly Linked List?**

Let's think through what we need:

1. **Add to front (most recent)**: O(1)
   - Need to know the front ✓
   
2. **Remove from back (least recent)**: O(1)
   - Need to know the back ✓
   - Need `prev` pointer to update second-to-last node ✓
   
3. **Remove from middle** (when accessed): O(1)
   - Need `prev` pointer to update previous node ✓
   - Need `next` pointer to update next node ✓

**Singly linked list won't work** because:
- Can't update `prev.next` when removing a node from middle
- Would need to traverse from head to find previous node: O(n) ❌

**Doubly linked list** gives us:
```
prev ← [Node] → next

Can update both directions without traversal!
```

**Why Hash Map?**

We need O(1) lookup by key. Hash map provides this:
```python
cache[key]  # O(1) average case
```

Without hash map, we'd need to traverse the list: O(n) ❌

### The Dummy Node Trick

One of the most important patterns in linked list problems!

**Problem without dummy nodes:**
```python
# Edge cases everywhere!
if self.head is None:
    # Special handling for empty list
if node == self.head:
    # Special handling for removing head
if node == self.tail:
    # Special handling for removing tail
```

**Solution with dummy nodes:**
```python
# Dummy head and tail always exist
self.head = Node()  # Dummy
self.tail = Node()  # Dummy
self.head.next = self.tail
self.tail.prev = self.head

# Now we NEVER have to check for None!
# Always have prev and next pointers
```

**Why this works:**

```
Before (without dummy):
head=None  OR  head → [1] → [2] → None
                ↑ Special case!

After (with dummy):
Dummy Head ↔ [1] ↔ [2] ↔ Dummy Tail
           ↑ Always have structure!

Empty cache:
Dummy Head ↔ Dummy Tail
           ↑ Still valid!
```

**Benefits:**
1. No null checks needed
2. Same logic for all operations
3. Fewer bugs
4. Cleaner code

### Visualization

```
Initial: capacity=3, cache is empty

put(1, 'a')
Cache: [1='a']
       ↑ most recent

put(2, 'b')
Cache: [2='b'] -> [1='a']
       ↑ most recent

put(3, 'c')
Cache: [3='c'] -> [2='b'] -> [1='a']
       ↑ most recent         ↑ least recent

get(1)  # Access 1, move to front
Cache: [1='a'] -> [3='c'] -> [2='b']
       ↑ most recent         ↑ least recent

put(4, 'd')  # Cache full, evict 2 (LRU)
Cache: [4='d'] -> [1='a'] -> [3='c']
       ↑ most recent         ↑ least recent
```

---

## Solution 1: Hash Map + Doubly Linked List (Optimal)

### Intuition

To achieve **O(1)** for both get and put:

1. **Hash Map** - O(1) lookup by key
2. **Doubly Linked List** - O(1) insertion/deletion from any position

**Why doubly linked list?**
- Move node to front: O(1) (need prev pointer)
- Remove from back: O(1) (need prev pointer)
- Remove from middle: O(1) (need prev pointer)

### Data Structure Design

```
┌─────────────────────────────────────────────┐
│              LRU CACHE                       │
├─────────────────────────────────────────────┤
│                                             │
│  Hash Map (key -> node)                     │
│  ┌─────────────────────────────────────┐   │
│  │ key=1 -> Node(1, 'a')               │   │
│  │ key=2 -> Node(2, 'b')               │   │
│  │ key=3 -> Node(3, 'c')               │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  Doubly Linked List (access order)          │
│  ┌──────────────────────────────────────┐  │
│  │  Dummy Head <-> [3] <-> [2] <-> [1]  │  │
│  │              <-> Dummy Tail           │  │
│  │              ↑ MRU      ↑ LRU         │  │
│  └──────────────────────────────────────┘  │
│                                             │
└─────────────────────────────────────────────┘
```

### Implementation

Before we dive into code, let's understand the step-by-step approach:

**Step 1: Define the Node**
Each node needs:
- `key`: To identify the item
- `value`: The cached data
- `prev`: Pointer to previous node (for doubly linked list)
- `next`: Pointer to next node (for doubly linked list)

**Why store `key` in the node?**
When we evict the LRU node, we need its key to delete it from the hash map!

```python
class Node:
    """
    Doubly linked list node
    
    This is the building block of our cache.
    Each node stores a key-value pair and links to neighbors.
    
    Why doubly linked?
    - `prev` lets us remove nodes from anywhere in O(1)
    - `next` lets us traverse forward
    
    Think of it like a train car:
    [prev car] ← [this car] → [next car]
    """
    def __init__(self, key=0, value=0):
        self.key = key      # Needed to delete from hash map when evicting
        self.value = value  # The cached data
        self.prev = None    # Link to previous node
        self.next = None    # Link to next node

class LRUCache:
    """
    LRU Cache with O(1) get and put
    
    Data structures:
    - Hash map: key -> node (for O(1) lookup)
    - Doubly linked list: nodes in access order (for O(1) reorder)
    
    Layout:
    head <-> [MRU] <-> ... <-> [LRU] <-> tail
    
    Most recently used (MRU) is after head
    Least recently used (LRU) is before tail
    """
    
    def __init__(self, capacity: int):
        """
        Initialize LRU cache
        
        Time: O(1)
        Space: O(capacity)
        """
        self.capacity = capacity
        self.cache = {}  # key -> node
        
        # Dummy head and tail for easier operations
        self.head = Node()
        self.tail = Node()
        
        # Connect head and tail
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def get(self, key: int) -> int:
        """
        Get value by key
        
        If exists, move to front (most recently used)
        
        Time: O(1)
        Space: O(1)
        
        Why move to front on get()?
        Because accessing the key makes it "recently used"!
        LRU means "least RECENTLY USED", so we need to track when things are accessed.
        """
        # Step 1: Check if key exists (O(1) hash map lookup)
        if key not in self.cache:
            return -1  # Not found
        
        # Step 2: Get the node from hash map (O(1))
        node = self.cache[key]
        
        # Step 3: Move to front (O(1))
        # Why? Because we just accessed it, making it "most recently used"
        # 
        # Before:  head ↔ [A] ↔ [B] ↔ [C] ↔ tail
        #                      ↑ We want this
        #
        # After:   head ↔ [B] ↔ [A] ↔ [C] ↔ tail
        #                 ↑ Most recent now!
        
        self._remove(node)       # Remove from current position
        self._add_to_front(node)  # Add at front (most recent)
        
        return node.value
    
    def put(self, key: int, value: int) -> None:
        """
        Put key-value pair
        
        If key exists, update value and move to front
        If key doesn't exist, add to front
        If capacity exceeded, evict LRU (before tail)
        
        Time: O(1)
        Space: O(1)
        
        This is the heart of the LRU cache!
        """
        # Case 1: Key already exists
        # We need to UPDATE the value and move to front
        if key in self.cache:
            node = self.cache[key]
            node.value = value  # Update value
            
            # Move to front (most recently used)
            # Why? Because we just wrote to it!
            self._remove(node)
            self._add_to_front(node)
        
        # Case 2: Key doesn't exist - NEW insertion
        else:
            # Create new node
            node = Node(key, value)
            
            # Add to hash map (for O(1) lookup)
            self.cache[key] = node
            
            # Add to front of list (most recently used)
            self._add_to_front(node)
            
            # CRITICAL: Check if we exceeded capacity
            if len(self.cache) > self.capacity:
                # We have ONE TOO MANY items!
                # Must evict the LRU item (the one before tail)
                
                # Why tail.prev? 
                # Because dummy tail is at the end, so tail.prev is the actual LRU item
                lru_node = self.tail.prev
                
                # Remove from list
                self._remove(lru_node)
                
                # IMPORTANT: Also remove from hash map!
                # Many people forget this step in interviews!
                del self.cache[lru_node.key]  # lru_node.key tells us which key to delete
    
    def _remove(self, node: Node) -> None:
        """
        Remove node from doubly linked list
        
        Time: O(1)
        
        This is the magic of doubly linked lists!
        We can remove ANY node in O(1) if we have a reference to it.
        """
        # Get neighbors
        prev_node = node.prev
        next_node = node.next
        
        # Connect neighbors to each other, bypassing node
        # Before: prev ↔ node ↔ next
        # After:  prev ↔↔↔↔↔↔ next
        prev_node.next = next_node
        next_node.prev = prev_node
        
        # Note: We don't need to set node.prev or node.next to None
        # because we'll reuse this node or it will be garbage collected
    
    def _add_to_front(self, node: Node) -> None:
        """
        Add node to front (after head, before first real node)
        
        Time: O(1)
        
        We always add to the front because that's the "most recently used" position.
        
        Visual:
        Before: head ↔ [A] ↔ [B] ↔ tail
        After:  head ↔ [node] ↔ [A] ↔ [B] ↔ tail
                       ↑ Most recent!
        """
        # Step 1: Set node's pointers
        node.prev = self.head
        node.next = self.head.next  # This is the old first node
        
        # Step 2: Update neighbors to point to node
        # Order matters here! Update in the right sequence:
        # First: old first node's prev should point to new node
        self.head.next.prev = node
        # Second: head's next should point to new node
        self.head.next = node
        
        # Why this order?
        # If we did head.next = node first, we'd lose the reference to old first node!
    
    def __repr__(self):
        """String representation for debugging"""
        items = []
        current = self.head.next
        
        while current != self.tail:
            items.append(f"{current.key}={current.value}")
            current = current.next
        
        return f"LRUCache({self.capacity}): [" + " -> ".join(items) + "]"

# Example usage
cache = LRUCache(2)

cache.put(1, 1)
print(cache)  # [1=1]

cache.put(2, 2)
print(cache)  # [2=2 -> 1=1]

print(cache.get(1))  # Returns 1
print(cache)  # [1=1 -> 2=2]  (1 moved to front)

cache.put(3, 3)  # Evicts key 2
print(cache)  # [3=3 -> 1=1]

print(cache.get(2))  # Returns -1 (not found)

cache.put(4, 4)  # Evicts key 1
print(cache)  # [4=4 -> 3=3]

print(cache.get(1))  # Returns -1 (not found)
print(cache.get(3))  # Returns 3
print(cache.get(4))  # Returns 4
```

### Complexity Analysis

**Time Complexity:**
- `get`: **O(1)** - Hash map lookup + linked list reorder
- `put`: **O(1)** - Hash map insert + linked list operations

**Space Complexity:**
- **O(capacity)** - Store up to capacity nodes

---

## Solution 2: OrderedDict (Python Built-in)

Python's `collections.OrderedDict` maintains insertion order and provides `move_to_end()` for O(1) reordering.

```python
from collections import OrderedDict

class LRUCache:
    """
    LRU Cache using OrderedDict
    
    Simpler implementation but same complexity
    """
    
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: int) -> int:
        """
        Get value and move to end (most recent)
        
        Time: O(1)
        """
        if key not in self.cache:
            return -1
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        """
        Put key-value and move to end
        
        Time: O(1)
        """
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        
        # Evict LRU if over capacity
        if len(self.cache) > self.capacity:
            # popitem(last=False) removes first (oldest) item
            self.cache.popitem(last=False)
```

**Pros:**
- Clean and concise
- Good for interviews if allowed

**Cons:**
- Less educational (hides implementation details)
- May not be allowed in interviews

---

## Detailed Walkthrough

Let's trace through a complete example step by step:

```python
def trace_lru_cache():
    """
    Trace LRU cache operations with detailed output
    """
    cache = LRUCache(3)
    
    operations = [
        ('put', 1, 'apple'),
        ('put', 2, 'banana'),
        ('put', 3, 'cherry'),
        ('get', 1, None),
        ('put', 4, 'date'),
        ('get', 2, None),
        ('get', 3, None),
        ('put', 5, 'elderberry'),
    ]
    
    print("="*60)
    print("LRU CACHE TRACE (capacity=3)")
    print("="*60)
    
    for op in operations:
        if op[0] == 'put':
            _, key, value = op
            print(f"\nput({key}, '{value}')")
            cache.put(key, value)
        else:
            _, key, _ = op
            result = cache.get(key)
            print(f"\nget({key}) -> {result}")
        
        # Print cache state
        print(f"  Cache: {cache}")
        print(f"  Size: {len(cache.cache)}/{cache.capacity}")

trace_lru_cache()
```

**Output:**
```
============================================================
LRU CACHE TRACE (capacity=3)
============================================================

put(1, 'apple')
  Cache: [1=apple]
  Size: 1/3

put(2, 'banana')
  Cache: [2=banana -> 1=apple]
  Size: 2/3

put(3, 'cherry')
  Cache: [3=cherry -> 2=banana -> 1=apple]
  Size: 3/3

get(1) -> apple
  Cache: [1=apple -> 3=cherry -> 2=banana]
  Size: 3/3
  (1 moved to front)

put(4, 'date')
  Cache: [4=date -> 1=apple -> 3=cherry]
  Size: 3/3
  (2 evicted - was LRU)

get(2) -> -1
  Cache: [4=date -> 1=apple -> 3=cherry]
  Size: 3/3
  (2 not found)

get(3) -> cherry
  Cache: [3=cherry -> 4=date -> 1=apple]
  Size: 3/3
  (3 moved to front)

put(5, 'elderberry')
  Cache: [5=elderberry -> 3=cherry -> 4=date]
  Size: 3/3
  (1 evicted - was LRU)
```

---

## Common Mistakes & Edge Cases

### Mistake 1: Forgetting to Update on Get

```python
# ❌ WRONG: Don't update access order on get
def get(self, key):
    if key in self.cache:
        return self.cache[key].value  # Not moving to front!
    return -1

# ✅ CORRECT: Move to front on get
def get(self, key):
    if key not in self.cache:
        return -1
    
    node = self.cache[key]
    self._remove(node)
    self._add_to_front(node)
    return node.value
```

### Mistake 2: Not Using Dummy Nodes

```python
# ❌ WRONG: No dummy nodes - many edge cases
class LRUCache:
    def __init__(self, capacity):
        self.head = None  # Can be None!
        self.tail = None  # Can be None!

# ✅ CORRECT: Dummy nodes simplify logic
class LRUCache:
    def __init__(self, capacity):
        self.head = Node()  # Always exists
        self.tail = Node()  # Always exists
        self.head.next = self.tail
        self.tail.prev = self.head
```

### Mistake 3: Incorrect Removal Logic

```python
# ❌ WRONG: Forgetting to update both prev and next
def _remove(self, node):
    node.prev.next = node.next  # Only updates next!

# ✅ CORRECT: Update both connections
def _remove(self, node):
    node.prev.next = node.next
    node.next.prev = node.prev
```

### Edge Cases to Test

```python
def test_edge_cases():
    """Test important edge cases"""
    
    # Edge case 1: Capacity of 1
    print("Test 1: Capacity 1")
    cache = LRUCache(1)
    cache.put(1, 1)
    cache.put(2, 2)  # Should evict 1
    assert cache.get(1) == -1
    assert cache.get(2) == 2
    print("✓ Passed")
    
    # Edge case 2: Update existing key
    print("\nTest 2: Update existing key")
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(1, 10)  # Update 1
    assert cache.get(1) == 10
    print("✓ Passed")
    
    # Edge case 3: Get non-existent key
    print("\nTest 3: Get non-existent key")
    cache = LRUCache(2)
    assert cache.get(999) == -1
    print("✓ Passed")
    
    # Edge case 4: Fill to capacity
    print("\nTest 4: Fill to capacity")
    cache = LRUCache(3)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(3, 3)
    cache.put(4, 4)  # Should evict 1
    assert cache.get(1) == -1
    print("✓ Passed")
    
    # Edge case 5: Access pattern
    print("\nTest 5: Complex access pattern")
    cache = LRUCache(2)
    cache.put(2, 1)
    cache.put(1, 1)
    cache.put(2, 3)
    cache.put(4, 1)
    assert cache.get(1) == -1
    assert cache.get(2) == 3
    print("✓ Passed")
    
    print("\n" + "="*40)
    print("All edge case tests passed!")
    print("="*40)

test_edge_cases()
```

---

## Production-Ready Implementation

### Thread-Safe LRU Cache

```python
import threading
from collections import OrderedDict

class ThreadSafeLRUCache:
    """
    Thread-safe LRU cache for concurrent access
    
    Uses RLock (reentrant lock) to protect cache operations
    """
    
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.RLock()
        
        # Metrics
        self.hits = 0
        self.misses = 0
    
    def get(self, key: int) -> int:
        """Thread-safe get"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return -1
            
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        """Thread-safe put"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            
            self.cache[key] = value
            
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def get_stats(self):
        """Get cache statistics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'total': total,
                'hit_rate': hit_rate,
                'size': len(self.cache),
                'capacity': self.capacity
            }

# Test thread safety
import time
import random

def worker(cache, worker_id, operations=1000):
    """Worker thread that performs cache operations"""
    for _ in range(operations):
        key = random.randint(1, 20)
        
        if random.random() < 0.7:
            # 70% reads
            cache.get(key)
        else:
            # 30% writes
            cache.put(key, worker_id)

# Create cache and threads
cache = ThreadSafeLRUCache(capacity=10)
threads = []

print("Testing thread safety with 10 concurrent threads...")
start = time.time()

for i in range(10):
    t = threading.Thread(target=worker, args=(cache, i, 1000))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

elapsed = time.time() - start

print(f"\nCompleted in {elapsed:.2f}s")
print("\nCache Statistics:")
stats = cache.get_stats()
for key, value in stats.items():
    if key == 'hit_rate':
        print(f"  {key}: {value:.2%}")
    else:
        print(f"  {key}: {value}")
```

### LRU Cache with TTL (Time To Live)

```python
import time

class LRUCacheWithTTL:
    """
    LRU cache with time-based expiration
    
    Items expire after TTL seconds
    """
    
    def __init__(self, capacity: int, ttl: int = 300):
        """
        Args:
            capacity: Max number of items
            ttl: Time to live in seconds
        """
        self.capacity = capacity
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}  # key -> insertion time
    
    def get(self, key: int) -> int:
        """Get with expiration check"""
        if key not in self.cache:
            return -1
        
        # Check if expired
        if time.time() - self.timestamps[key] > self.ttl:
            # Expired, remove
            del self.cache[key]
            del self.timestamps[key]
            return -1
        
        # Not expired, move to end
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        """Put with timestamp"""
        if key in self.cache:
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
        
        # Evict LRU if over capacity
        if len(self.cache) > self.capacity:
            lru_key = next(iter(self.cache))
            del self.cache[lru_key]
            del self.timestamps[lru_key]
    
    def cleanup_expired(self):
        """Remove all expired items"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
        
        return len(expired_keys)

# Example
cache = LRUCacheWithTTL(capacity=3, ttl=5)

cache.put(1, 'apple')
cache.put(2, 'banana')

print("Immediately after insert:")
print(f"get(1) = {cache.get(1)}")  # 'apple'

print("\nWait 6 seconds...")
time.sleep(6)

print("After TTL expired:")
print(f"get(1) = {cache.get(1)}")  # -1 (expired)

print(f"\nCleaned up {cache.cleanup_expired()} expired items")
```

---

## Performance Benchmarking

```python
import time
import random
import matplotlib.pyplot as plt

def benchmark_lru_implementations():
    """
    Compare performance of different LRU implementations
    """
    capacities = [10, 100, 1000, 10000]
    implementations = {
        'Custom (Doubly Linked List)': LRUCache,
        'OrderedDict': lambda cap: LRUCacheOrderedDict(cap),
    }
    
    results = {name: [] for name in implementations}
    
    print("Benchmarking LRU Cache Implementations")
    print("="*60)
    
    for capacity in capacities:
        print(f"\nCapacity: {capacity}")
        
        for name, impl_class in implementations.items():
            cache = impl_class(capacity)
            
            # Generate workload
            num_operations = 10000
            operations = []
            
            for _ in range(num_operations):
                if random.random() < 0.7:
                    # 70% reads
                    key = random.randint(0, capacity * 2)
                    operations.append(('get', key))
                else:
                    # 30% writes
                    key = random.randint(0, capacity * 2)
                    value = random.randint(0, 1000)
                    operations.append(('put', key, value))
            
            # Benchmark
            start = time.perf_counter()
            
            for op in operations:
                if op[0] == 'get':
                    cache.get(op[1])
                else:
                    cache.put(op[1], op[2])
            
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            results[name].append(elapsed)
            print(f"  {name:30s}: {elapsed:6.2f}ms")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    for name, times in results.items():
        plt.plot(capacities, times, marker='o', label=name)
    
    plt.xlabel('Capacity')
    plt.ylabel('Time (ms) for 10,000 operations')
    plt.title('LRU Cache Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.savefig('lru_benchmark.png')
    plt.close()
    
    print("\n" + "="*60)
    print("Benchmark complete! Plot saved to lru_benchmark.png")

class LRUCacheOrderedDict:
    """OrderedDict implementation for comparison"""
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

benchmark_lru_implementations()
```

---

## Connection to ML Systems

### 1. Model Prediction Cache

```python
class ModelPredictionCache:
    """
    Cache model predictions to avoid recomputation
    
    Use case: Frequently requested predictions
    """
    
    def __init__(self, model, capacity=1000):
        self.model = model
        self.cache = LRUCache(capacity)
    
    def predict(self, features):
        """
        Predict with caching
        
        Args:
            features: tuple of feature values (must be hashable)
        
        Returns:
            prediction
        """
        # Create cache key from features
        cache_key = hash(features)
        
        # Try cache
        cached_prediction = self.cache.get(cache_key)
        if cached_prediction != -1:
            return cached_prediction
        
        # Cache miss: compute prediction
        prediction = self.model.predict([features])[0]
        
        # Store in cache
        self.cache.put(cache_key, prediction)
        
        return prediction

# Example
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train model
X_train = np.random.randn(100, 5)
y_train = (X_train.sum(axis=1) > 0).astype(int)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

# Create cached predictor
cached_model = ModelPredictionCache(model, capacity=100)

# Make predictions
for _ in range(1000):
    features = tuple(np.random.randint(0, 10, size=5))
    prediction = cached_model.predict(features)

print("Cache statistics:")
print(f"  Capacity: {cached_model.cache.capacity}")
print(f"  Current size: {len(cached_model.cache.cache)}")
```

### 2. Feature Store Cache

```python
class FeatureStoreCache:
    """
    Cache feature store lookups
    
    Feature stores can be slow (database/API calls)
    LRU cache reduces latency
    """
    
    def __init__(self, feature_store, capacity=10000):
        self.feature_store = feature_store
        self.cache = ThreadSafeLRUCache(capacity)
    
    def get_features(self, entity_id):
        """
        Get features for entity with caching
        
        Args:
            entity_id: Unique entity identifier
        
        Returns:
            Feature dictionary
        """
        # Try cache
        cached_features = self.cache.get(entity_id)
        if cached_features != -1:
            return cached_features
        
        # Cache miss: query feature store
        features = self.feature_store.query(entity_id)
        
        # Store in cache
        self.cache.put(entity_id, features)
        
        return features
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        return self.cache.get_stats()

# Example usage
class MockFeatureStore:
    """Mock feature store with latency"""
    def query(self, entity_id):
        time.sleep(0.001)  # Simulate 1ms latency
        return {'feature1': entity_id * 2, 'feature2': entity_id ** 2}

feature_store = MockFeatureStore()
cached_store = FeatureStoreCache(feature_store, capacity=1000)

# Simulate requests
start = time.time()
for _ in range(10000):
    entity_id = random.randint(1, 500)
    features = cached_store.get_features(entity_id)

elapsed = time.time() - start

print(f"Total time: {elapsed:.2f}s")
print("\nCache stats:")
stats = cached_store.get_cache_stats()
for key, value in stats.items():
    if key == 'hit_rate':
        print(f"  {key}: {value:.2%}")
    else:
        print(f"  {key}: {value}")
```

### 3. Embedding Cache

```python
class EmbeddingCache:
    """
    Cache embeddings for frequently queried items
    
    Useful for recommendation systems, search, etc.
    """
    
    def __init__(self, embedding_model, capacity=10000):
        self.model = embedding_model
        self.cache = LRUCache(capacity)
    
    def get_embedding(self, item_id):
        """
        Get embedding with caching
        
        Computing embeddings can be expensive (neural network inference)
        """
        # Try cache
        cached_embedding = self.cache.get(item_id)
        if cached_embedding != -1:
            return cached_embedding
        
        # Cache miss: compute embedding
        embedding = self.model.encode(item_id)
        
        # Store in cache
        self.cache.put(item_id, embedding)
        
        return embedding
    
    def batch_get_embeddings(self, item_ids):
        """
        Get embeddings for multiple items
        
        Separate cache hits from misses for efficient batch processing
        """
        embeddings = {}
        cache_misses = []
        
        # Check cache
        for item_id in item_ids:
            cached = self.cache.get(item_id)
            if cached != -1:
                embeddings[item_id] = cached
            else:
                cache_misses.append(item_id)
        
        # Batch compute misses
        if cache_misses:
            batch_embeddings = self.model.batch_encode(cache_misses)
            
            for item_id, embedding in zip(cache_misses, batch_embeddings):
                self.cache.put(item_id, embedding)
                embeddings[item_id] = embedding
        
        return [embeddings[item_id] for item_id in item_ids]
```

---

## Interview Tips

### Discussion Points

1. **Why not just use a hash map?**
   - Hash map gives O(1) lookup but doesn't track access order
   - Need additional data structure for ordering

2. **Why doubly linked list instead of array?**
   - Array: O(n) to remove from middle
   - Doubly linked list: O(1) to remove from any position

3. **Why dummy head/tail?**
   - Simplifies edge cases
   - No null checks needed
   - Consistent operations

4. **Can you use a single data structure?**
   - No, you need both:
     - Fast lookup: Hash map
     - Fast reordering: Linked list

### Follow-up Questions

**Q: How would you implement LFU (Least Frequently Used)?**

```python
class LFUCache:
    """
    Least Frequently Used cache
    
    Evicts item with lowest access frequency
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> (value, freq)
        self.freq_map = {}  # freq -> OrderedDict of keys
        self.min_freq = 0
    
    def get(self, key):
        if key not in self.cache:
            return -1
        
        value, freq = self.cache[key]
        
        # Increment frequency
        self._increment_freq(key, value, freq)
        
        return value
    
    def put(self, key, value):
        if self.capacity == 0:
            return
        
        if key in self.cache:
            # Update existing
            _, freq = self.cache[key]
            self._increment_freq(key, value, freq)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Evict LFU
                self._evict()
            
            self.cache[key] = (value, 1)
            
            if 1 not in self.freq_map:
                self.freq_map[1] = OrderedDict()
            
            self.freq_map[1][key] = None
            self.min_freq = 1
    
    def _increment_freq(self, key, value, freq):
        """Move key to higher frequency bucket"""
        # Remove from current frequency
        del self.freq_map[freq][key]
        
        if not self.freq_map[freq] and freq == self.min_freq:
            self.min_freq += 1
        
        # Add to next frequency
        new_freq = freq + 1
        if new_freq not in self.freq_map:
            self.freq_map[new_freq] = OrderedDict()
        
        self.freq_map[new_freq][key] = None
        self.cache[key] = (value, new_freq)
    
    def _evict(self):
        """Evict least frequently used (and least recently used within that frequency)"""
        # Get first key from min frequency bucket
        key_to_evict = next(iter(self.freq_map[self.min_freq]))
        
        del self.freq_map[self.min_freq][key_to_evict]
        del self.cache[key_to_evict]
```

**Q: How would you handle cache invalidation?**

```python
class LRUCacheWithInvalidation(LRUCache):
    """
    LRU cache with manual invalidation
    
    Useful when data changes externally
    """
    
    def invalidate(self, key):
        """Remove key from cache"""
        if key not in self.cache:
            return False
        
        node = self.cache[key]
        self._remove(node)
        del self.cache[key]
        
        return True
    
    def invalidate_pattern(self, pattern):
        """
        Invalidate keys matching pattern
        
        Example: invalidate_pattern('user_*')
        """
        import fnmatch
        
        keys_to_remove = [
            key for key in self.cache.keys()
            if fnmatch.fnmatch(str(key), pattern)
        ]
        
        for key in keys_to_remove:
            self.invalidate(key)
        
        return len(keys_to_remove)
```

---

## Key Takeaways

✅ **O(1) operations** - Hash map + doubly linked list  
✅ **Dummy nodes** - Simplify edge case handling  
✅ **Update on access** - Both get() and put() update recency  
✅ **Thread safety** - Use locks for concurrent access  
✅ **Real-world use** - Prediction cache, feature store, embeddings  

**Core Pattern:**
- Hash map for O(1) lookup
- Doubly linked list for O(1) reordering
- MRU at head, LRU at tail

---

**Originally published at:** [arunbaby.com/dsa/0011-lru-cache](https://www.arunbaby.com/dsa/0011-lru-cache/)

*If you found this helpful, consider sharing it with others who might benefit.*

