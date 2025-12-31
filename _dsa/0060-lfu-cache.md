---
title: "LFU Cache (Least Frequently Used)"
day: 60
related_ml_day: 60
related_speech_day: 60
related_agents_day: 60
collection: dsa
categories:
  - dsa
tags:
  - lfu
  - cache
  - doubly-linked-list
  - hash-table
  - hard
  - data-structure-design
difficulty: Hard
subdomain: "Data Structures"
tech_stack: Python
scale: "O(1) time for get and put, O(N) space"
companies: [Google, Amazon, Meta, Microsoft, Uber]
---


**"Designing an LFU Cache is the ultimate exercise in composite data structures—it forces you to synchronize multiple hash maps and linked lists to achieve O(1) performance for a complex priority problem."**

## 1. Introduction: The Evolution of Caching

In the world of high-performance computing, memory is your most precious resource. Whether you are building a web server, a database engine, or a machine learning inference pipeline, you will eventually hit the "Memory Wall." You cannot fit everything in RAM. You must choose what to keep and what to discard.

This choice is governed by **Cache Replacement Policies**.
- **The FIFO Approach (Queue)**: Discard what came in first. Simple, but often wrong because old data can be popular.
- **The LRU Approach (Least Recently Used)**: Discard what hasn't been used for the longest time. This works well for "Temporal Locality"—if you used it recently, you'll likely use it again soon.
- **The LFU Approach (Least Frequently Used)**: Discard what is used *least often*. This is superior for items that have long-term popularity but might be accessed sporadically.

Today we tackle the **LFU Cache**, one of the most challenging data structures to implement with O(1) efficiency. Unlike LRU, which needs a single list, LFU typically requires managing a dynamic collection of lists or a heap. We will explore why it is hard, how to solve it with a "Frequency-Tiered Doubly Linked List" (Linked-List-of-Linked-Lists), and how it connects to the future of persistent intelligence in AI.

### 1.1 Why This Problem Matters
-   **System Design Interview Staple**: It tests your ability to combine data structures.
-   **Real-world Impact**: Redis, CDN Edge Nodes, and CPU Branch Predictors all use variations of LFU.
-   **Complexity Theory**: It demonstrates how Amortized Analysis allows O(1) even for complex operations.

---

## 2. The Problem Statement

Implement the `LFUCache` class:

- `LFUCache(int capacity)`: Initializes the object with the capacity of the data structure.
- `int get(int key)`: Returns the value of the `key` if it exists in the cache. Otherwise, returns `-1`.
- `void put(int key, int value)`: Updates the value of the `key` if present, or inserts it if not.
  - If the cache reaches its `capacity`, it must invalidate and remove the **least frequently used** key.
  - If there is a tie (multiple keys with the same frequency), the **least recently used (LRU)** key among them must be removed.

**Constraint**: Both `get` and `put` must run in O(1) average time.

---

## 3. Why LFU is Hard: The O(1) Trap

If you've solved the **LRU Cache**, you know that a Hash Map combined with a Doubly Linked List (DLL) gives you O(1) because moving a node to the head of the list is a constant-time pointer operation.

In **LFU**, we have two axes of priority:
1. **Frequency**: How many times was the key accessed?
2. **Recency**: If frequencies are tied, which was used longest ago?

### The Naive (and Slow) Approach
You could use a **Min-Heap** where each node stores `(frequency, last_access_time, value)`.
- `get(key)`: O(log N) to update the heap.
- `put(key)`: O(log N) to insert/evict.
For a high-performance system (like a CDN handling millions of hits), O(log N) is often unacceptable. We need the constant-time performance of a hash table.

---

## 4. The Data Structure Design: The Frequency-Tiered DLL

To solve LFU in O(1), we must avoid sorting. Instead, we use a **linked list of linked lists**.

### 4.1 The Three Core Components
1. **`key_to_node` (Hash Map)**: Maps the `key` to a `Node` object. Each node stores its `value` and its current `frequency`.
2. **`freq_to_list` (Hash Map)**: Maps each `frequency` (an integer) to a **Doubly Linked List**. All keys in `freq_to_list[5]` have been accessed exactly 5 times.
3. **`min_freq` (Global Variable)**: Tracks the minimum frequency currently present in the cache. This is our "shortcut" to find the eviction candidate instantly.

### 4.2 The "Promotion" Flow
When a key is accessed:
1. Identify its current frequency `F`.
2. Remove it from the `freq_to_list[F]`.
3. If `freq_to_list[F]` is now empty and `F` was the `min_freq`, increment `min_freq`.
4. Move the key to `freq_to_list[F+1]`.
5. Update its internal frequency counter to `F+1`.

---

## 5. Implementation in Python

We first define a robust `Node` and `DoublyLinkedList` helper.

```python
class Node:
    """A single node representing a cache entry."""
    def __init__(self, key, value):
        self.key = key
        self.val = value
        self.freq = 1
        self.prev = None
        self.next = None

class DoublyLinkedList:
    """A standard DLL with dummy head and tail for easy removal."""
    def __init__(self):
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def add_to_front(self, node):
        """Adds a node directly after the dummy head (MRU position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
        self.size += 1

    def remove(self, node):
        """Removes a node from its current position in the list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1

    def pop_tail(self):
        """Removes and returns the node just before the dummy tail (LRU position)."""
        if self.size == 0:
            return None
        node = self.tail.prev
        self.remove(node)
        return node
```

### The LFUCache Orchestrator

```python
class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.min_freq = 0
        self.key_to_node = {}    # key -> Node
        self.freq_to_list = {}   # freq -> DoublyLinkedList

    def _update_node(self, node):
        """Internal helper to promote a node to a higher frequency tier."""
        old_freq = node.freq
        
        # 1. Remove from current frequency list
        self.freq_to_list[old_freq].remove(node)
        
        # 2. Update global min_freq if necessary
        if self.freq_to_list[old_freq].size == 0 and old_freq == self.min_freq:
            self.min_freq += 1
        
        # 3. Increment node frequency and add to new list
        node.freq += 1
        new_freq = node.freq
        if new_freq not in self.freq_to_list:
            self.freq_to_list[new_freq] = DoublyLinkedList()
        self.freq_to_list[new_freq].add_to_front(node)

    def get(self, key: int) -> int:
        if self.capacity == 0 or key not in self.key_to_node:
            return -1
        
        node = self.key_to_node[key]
        self._update_node(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return

        if key in self.key_to_node:
            # Case 1: Key exists, update value and promote frequency
            node = self.key_to_node[key]
            node.val = value
            self._update_node(node)
        else:
            # Case 2: New key
            if self.size >= self.capacity:
                # Eviction: Get the list for the current min_freq
                evict_candidate = self.freq_to_list[self.min_freq].pop_tail()
                del self.key_to_node[evict_candidate.key]
                self.size -= 1
            
            # Create and add the new node
            new_node = Node(key, value)
            self.key_to_node[key] = new_node
            
            # For a brand new node, frequency is always 1
            if 1 not in self.freq_to_list:
                self.freq_to_list[1] = DoublyLinkedList()
            self.freq_to_list[1].add_to_front(new_node)
            
            self.min_freq = 1
            self.size += 1
```

---

## 6. Implementation Deep Dive: Synchronizing the Maps

The beauty and the danger of this design lie in the synchronization. If you remove a node from a DLL but forget to update `key_to_node`, you have a **Memory Leak**. If you update `min_freq` incorrectly, you will evict the wrong item, leading to a **Bypass of the LFU logic**.

### 6.1 The "Dummy" Pattern
Why do we use dummy head and tail nodes in the DLL?
- It eliminates "if-else" checks for `node.next is None`.
- `self.head.next = self.tail` at initialization means the code for `add_to_front` works identical for the first node and the thousandth node.

### 6.2 The `min_freq` Reset
Crucially, when a *new* key is added, `min_freq` is **always** reset to 1. Many students fail by trying to "calculate" if 1 is really the min frequency. It is, by definition, because the new key is the newest frequency-1 item.

---

## 7. Comparative Performance Analysis

| Operation | LRU (LinkedHashMap) | LFU (Tiered DLL) | LFU (Min-Heap) |
| :--- | :--- | :--- | :--- |
| **Get** | O(1) | O(1) | O(log N) |
| **Put** | O(1) | O(1) | O(log N) |
| **Eviction** | O(1) | O(1) | O(1) |
| **Space** | O(N) | O(N) | O(N) |
| **Complexity** | Simple | High | Medium |

### Hidden Constants
While both LRU and Tiered-DLL LFU are O(1), the **Hidden Constant** for LFU is significantly higher. You are managing multiple hash map lookups.

---

## 8. Beyond LFU: Adaptive Replacement Cache (ARC)

Is LFU the ultimate policy? No.
LFU has a fatal flaw: its memory is "sticky." If a key was very popular in the morning but is never used again, it will stay in the cache forever because its frequency is too high.

This is why advanced systems like **PostgreSQL** and **ZFS** use **ARC (Adaptive Replacement Cache)**. 
- ARC maintains two lists: one for recency (LRU) and one for frequency (LFU).
- It dynamically resizes the "target size" for each list.

---

## 9. Real-World Applications

### 9.1 Database Buffer Pools
Databases like SQLite or MySQL use LFU-like logic to keep the "Indices" in memory while allowing the "Data Pages" to be evicted quickly.

### 9.2 Machine Learning Feature Stores
In the **ML System Design** world, we use LFU for **Feature Caching**.
- Globally popular features stay in the fastest cache layer.

### 9.3 Speech Technology
In **Speech Tech**, we use LFU to cache **Phonetic Embeddings**. Common words have their acoustic-to-text mappings cached in an LFU tier to save GPU inference time.

---

## 10. Interview Strategy

1. **Start with LRU**: Briefly explain why LRU is insufficient (it ignores long-term popularity).
2. **Propose the Tiered Map**: Draw a diagram showing `freq -> DLL`. This is the clearest way to explain O(1) LFU.
3. **Explain the Eviction Tie-break**: Mention that you use LRU *within* each frequency tier.
4. **Discuss min_freq**: Explain how you maintain the shortcut to the least frequent list.

---


---

## 11. Edge Cases and Resilience

When implementing LFU in a production environment (not just LeetCode), you face edge cases that break naive implementations.

### 11.1 The Zero-Capacity Cache
What if `capacity` is 0?
-   Naive code might try to `put()` and then immediately `evict()`.
-   If you initialize `min_freq = 1` blindly, you might crash when trying to access `freq_to_list[1]`.
-   **Production fix**: Always guard with `if capacity <= 0: return`.

### 11.2 Frequency Overflow
In a long-running Redis instance, a key might be accessed 2 billion times.
-   If your frequency counter is a 32-bit signed integer, it might wrap around to negative.
-   **Impact**: The "Most Popular" item suddenly becomes the "Least Frequent" (negative billion) and gets evicted immediately.
-   **Fix**: Implementation of "Frequency Aging" (Halving all frequencies every week) or using 64-bit counters.

### 11.3 Memory Alignment and Pointers
In Python, a `Node` object is heavy (PyObject structure). In C++, a `struct Node` is small.
-   However, the **Pointer Overhead** in LFU is 2x that of LRU.
-   LRU: `prev`, `next`.
-   LFU: `prev`, `next`, `key`, `value`, `freq`.
-   This reduced **Cache Locality** (CPU Cache, not LFU Cache) can make LFU slower in practice than LRU, even if Big-O is same.

---

## 12. Concurrency: Making LFU Thread-Safe

The implementation above is single-threaded. In a real web server (e.g., using Go or Java), multiple threads will call `get` and `put` simultaneously.

### 12.1 The Global Lock (Coarse-Grained)
The simplest fix is to wrap every method in `mutex.lock()`.
-   **Pros**: Correctness is guaranteed.
-   **Cons**: Massive contention. If thread A is evicting (updating 6 pointers), thread B is blocked. This destroys throughput.

### 12.2 Lock Stripping (Fine-Grained)
We can shard the LFU cache.
-   Create 16 separate `LFUCache` instances.
-   `shard_id = hash(key) % 16`.
-   Lock only the specific shard.
-   **Trade-off**: The "Least Frequently Used" is now only local to the shard, not global. You might evict a frequent item in Shard A while Shard B holds garbage. This is an acceptable trade-off for speed (used in **ConcurrentHashMap**).

### 12.3 Lock-Free LFU (The Holy Grail)
Research structures like **Window-TinyLFU** use probabilistic counters (Count-Min Sketch) which can be updated atomically without locks.
-   Instead of a precise "Frequency=19,203", we accept "Frequency ~ 19k".
-   This allows `get()` operations to be wait-free.

---

## 13. Advanced Variant: Window-TinyLFU

Most modern databases (like Cassandra and Dgraph) rely on **Caffeine**, a Java cache library that implements **Window-TinyLFU**.

### 13.1 The Problem with Standard LFU
LFU is slow to adapt to new trends.
-   Old viral content has `freq=10,000`.
-   New breaking news has `freq=10`.
-   The new content is evicted because `10 < 10,000`. This is bad.

### 13.2 The TinyLFU Architecture
1.  **Admission Window**: A small LRU region (1% of size) acts as a "Doorman." New items enter here.
2.  **Main Cache**: Segmented into Protected (hot) and Probation (cold) regions.
3.  **The Combat**: When the Window is full, the victim from the Window "fights" the victim from the Probation region.
    -   We compare their approximate frequencies using a **Count-Min Sketch**.
    -   If the newcomer is more popular than the incumbent, the incumbent is evicted.
    -   If the newcomer is weak, it is rejected (not cached at all).

This "Doorman" approach prevents "One-Hit Wonders" (items accessed once) from polluting the main cache history.

---

## 14. Complexity Analysis: A Mathematical Proof

Let's rigorously prove the Time Complexity.

### Space Complexity
-   `key_to_node`: Stores N nodes. `O(N)`.
-   `freq_to_list`: Stores N nodes distributed across lists. `O(N)`.
-   Total Space = `2N` pointers + `N` integers. -> **O(N)**.

### Time Complexity (Put)
1.  Check Hash Map: `O(1)` amortized.
2.  Create Node: `O(1)`.
3.  Link to DLL Head: `O(1)` (pointer assignment).
4.  If Full, Pop Tail: `O(1)` (pointer assignment).
5.  Remove from Map: `O(1)`.
6.  Update `min_freq`: `O(1)` (assignment).
7.  **Total**: **O(1)**.

### Time Complexity (Get)
1.  Check Hash Map: `O(1)`.
2.  Unlink Node: `O(1)` (pointer assignment).
3.  Link to New List: `O(1)`.
4.  **Total**: **O(1)**.

### Why Average vs Worst Case?
In Python/Java, hash map collisions can degrade to `O(K)` or `O(log K)`.
-   Therefore, strictly speaking, LFU is `O(log N)` in the worst case of Hash Collisions.
-   However, with a good cryptographic hash, we assume `O(1)`.

---

## 15. Summary of Optimization patterns

When designing high-performance data structures, we see recurring themes in LFU:
1.  **Space-Time Tradeoff**: We use 2x memory (Maps + Lists) to buy 100x speed (O(1) vs O(log N)).
2.  **Lazy Balancing**: We don't sort the lists. We assume simple appending (MRU) is "good enough" for the internal ordering.
3.  **Amortization**: We don't clean up empty frequency buckets immediately (unless we want to save space), allowing operations to flow faster.

---

## 16. Historical Context: From cache_mem to TinyLFU

The history of LFU is a history of trying to fix its "Stickiness."

### 1960s-1990s: Mathematical Idealism
In the early days of caching (IBM Mainframes), LFU was known as the "Optimum" policy under certain static probability distributions. If you know that 'A' is accessed 20% of the time, and 'B' 5% of the time, LFU is provably optimal.
-   **Problem**: Real-world distributions change.
-   **Solution**: Ignoring it. RAM was small; LRU was used because LFU was too expensive to implement (Heap overhead).

### 2000s: The LIRS/ARC Revolution
Researchers realized that "Scan Resistance" was the killer feature.
-   **ARC (Adaptive Replacement Cache)**: Invented by IBM. It tracked "Ghosts" (recently evicted items) to decide whether to enlarge the Frequency or Recency list.
-   **LIRS (Low Inter-reference Recency Set)**: Used in Linux Kernel and MySQL. It defines "hot" not by count, but by the "distance" between the last two accesses.

### 2015-Present: The Probabilistic Era (W-TinyLFU)
In 2015, Gil Einziger strictly solved the Space Overhead problem.
-   **Bloom Filters**: Using approximate counting (Count-Min Sketch) fits the frequency of millions of items into Kilobytes.
-   **Window-TinyLFU**: Used in **Caffeine** (Java), **Ristretto** (Go). This is currently the state-of-the-art. It combines a small LRU window (to capture new bursts) with a large LFU main region (to capture long-term popularity).

---

## 17. The Comprehensive Test Suite

When you write this in an interview code-pad, you need to verify it. Do not just write `put(1,1)`. Write a suite that targets the edge cases.

```python
import unittest

class TestLFUCache(unittest.TestCase):
    def test_basic_functional(self):
        # The standard Example
        lfu = LFUCache(2)
        lfu.put(1, 1)        # State: [1:1]
        lfu.put(2, 2)        # State: [2:1, 1:1]
        self.assertEqual(lfu.get(1), 1) # State: [1:2, 2:1]
        
        lfu.put(3, 3)        # Evicts 2 (freq 1). State: [3:1, 1:2]
        self.assertEqual(lfu.get(2), -1)
        self.assertEqual(lfu.get(3), 3) # State: [3:2, 1:2]
        
        lfu.put(4, 4)        # Tie-break. 1 has freq 2, 3 has freq 2.
                             # 1 was accessed at step 3. 3 was accessed at step 6.
                             # 1 is LRU among freq-2 items. Evict 1.
        self.assertEqual(lfu.get(1), -1)
        self.assertEqual(lfu.get(3), 3) # State: [3:3, 4:1]
        self.assertEqual(lfu.get(4), 4) # State: [4:2, 3:3]

    def test_frequency_promotion(self):
        # Ensure items move up the ladder correctly
        lfu = LFUCache(3)
        lfu.put(1, 10)
        
        # Access 100 times
        for i in range(100):
            lfu.get(1)
            
        self.assertEqual(lfu.key_to_node[1].freq, 101)
        
        # Add new items
        lfu.put(2, 20) # freq 1
        lfu.put(3, 30) # freq 1
        
        # This update should NOT evict 1, even though it's old
        lfu.put(4, 40) 
        # Cache full (3 items). Candidates: 2 (freq 1), 3 (freq 1). 1 (freq 101).
        # Evict 2 (LRU of freq 1).
        
        self.assertEqual(lfu.get(2), -1)
        self.assertEqual(lfu.get(1), 10)

    def test_zero_capacity(self):
        lfu = LFUCache(0)
        lfu.put(1, 1)
        self.assertEqual(lfu.get(1), -1)

    def test_update_value_does_not_reset_freq(self):
        lfu = LFUCache(2)
        lfu.put(1, 1)
        lfu.get(1) # freq 2
        
        lfu.put(1, 100) # update value. freq should be 3 now!
        self.assertEqual(lfu.key_to_node[1].freq, 3)
        self.assertEqual(lfu.get(1), 100)

    def test_tie_breaking(self):
        lfu = LFUCache(2)
        lfu.put(1, 1) # freq 1
        lfu.put(2, 2) # freq 1
        
        lfu.get(1) # 1 promoted to freq 2
        lfu.get(2) # 2 promoted to freq 2
        
        # Both freq 2. 1 is LRU because 2 was accessed last.
        lfu.put(3, 3) 
        
        self.assertEqual(lfu.get(1), -1) # 1 should be gone
        self.assertEqual(lfu.get(2), 2)  # 2 should stay

if __name__ == '__main__':
    unittest.main()
```

### 17.1 How to Debug LFU Instability
If you implement this and it fails randomly:
1.  **Check `min_freq` maintenance**: This is the #1 bug. If you remove the last node from `min_freq` list, do you definitely increment `min_freq`?
2.  **Check Empty List cleanup**: Do you delete the DLL from the hash map when it's empty? It's not strictly necessary but saves memory.
3.  **Check Node pointers**: When moving a node from List A to List B, do you reset `node.prev` and `node.next`? The `DoublyLinkedList.add_to_front` method should handle this overwriting safely, but verify.

---

## 18. Conclusion

When designing systems, LFU reminds us that **metadata** (frequency) is as important as the **data** itself.
Start with the Linked Hash Map. Add the Frequency dimension. Handle the edge cases.
You have now mastered one of the hardest data structures in the interview repertoire.

---

## 19. Key Takeaways

1. **States are Synchronized**: Designing complex structures is about maintaining invariants across multiple containers.
2. **Frequency vs. Recency**: LFU is about the "Volume" of access; LRU is about the "Proximity" of access.
3. **Future-Proofing**: As AI agents move toward long-term memory, the algorithms of LFU will become the foundation of "Thinking Architectures."


---

**Originally published at:** [arunbaby.com/dsa/0060-lfu-cache](https://www.arunbaby.com/dsa/0060-lfu-cache/)

*If you found this helpful, consider sharing it with others who might benefit.*
