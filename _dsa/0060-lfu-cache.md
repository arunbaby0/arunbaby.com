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
- **The FIFO Approach**: Discard what came in first. Simple, but often wrong.
- **The LRU Approach (Least Recently Used)**: Discard what hasn't been used for the longest time. This works well for "Temporal Locality"—if you used it recently, you'll likely use it again soon.
- **The LFU Approach (Least Frequently Used)**: Discard what is used *least often*. This is superior for items that have long-term popularity but might be accessed sporadically.

Today we tackle the **LFU Cache**, one of the most challenging data structures to implement with O(1) efficiency. We will explore why it is hard, how to solve it with a linked-list-of-linked-lists, and how it connects to the future of persistent intelligence in AI.

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

## 11. Key Takeaways

1. **States are Synchronized**: Designing complex structures is about maintaining invariants across multiple containers.
2. **Frequency vs. Recency**: LFU is about the "Volume" of access; LRU is about the "Proximity" of access.
3. **Future-Proofing**: As AI agents move toward long-term memory, the algorithms of LFU will become the foundation of "Thinking Architectures."

---

**Originally published at:** [arunbaby.com/dsa/0060-lfu-cache](https://www.arunbaby.com/dsa/0060-lfu-cache/)

*If you found this helpful, consider sharing it with others who might benefit.*
