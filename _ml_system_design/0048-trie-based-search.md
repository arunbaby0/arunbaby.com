---
title: "Trie-Based Search Systems (Typeahead)"
day: 48
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - search
  - autocomplete
  - typeahead
  - distributed-systems
  - trie
difficulty: Hard
subdomain: "Search Architecture"
tech_stack: Redis, Apache Solr, Custom C++
scale: "10 Billion queries/day, <20ms latency"
companies: Google, Amazon, Facebook, LinkedIn
related_dsa_day: 48
related_ml_day: 48
related_speech_day: 48
related_agents_day: 48
---

**"The user knows what they want. Your job is to tell them before they finish typing."**

## 1. Problem Statement

Typeahead (or Autocomplete) is the most heavily used feature on the internet.
-   **Volume**: Every keystroke triggers a query. If a user types "iphone case", that's 11 requests.
-   **Latency**: Must be < 50ms (Human perception of "instant").
-   **Freshness**: If "Earthquake in Japan" happens, it must appear in suggestions within minutes.

**The System Design**: Build a backend for a Google-style search bar that suggests top 10 completions based on prefix.

---

## 2. Understanding the Requirements

### 2.1 The Scale
-   **DAU**: 500 Million.
-   **Queries**: 5 billion searches / day.
-   **Keystrokes**: ~50 billion typeahead requests / day.
-   **QPS**: Peak 1 million QPS. (This requires massive horizontal scaling).

### 2.2 The Ranking Problem
We don't just want matching strings.
-   Prefix: "ja"
-   Candidates: "java", "japan", "javascript", "jacket".
-   We need to return top 5 sorted by `Score`.
    `Score = Transformation(Frequency, User_History, Location, Freshness)`

---

## 3. High-Level Architecture

We need a dedicated **Typeahead Service** separate from the main Search Service.

```
[Browser]
      | (GET /suggest?q=sys)
      v
[Load Balancer] --> [CDN (Caches popular: "fa"->"facebook")]
      |
      v
[Typeahead Aggregator]
      |
      +---> [Service Node 1 (A-C)]
      +---> [Service Node 2 (D-F)]
      +---> [Personalization Service (Reranking)]
```

---

## 4. Component Deep-Dives

### 4.1 The Data Structure: The Weighted Trie
In RAM, we store a specialized Trie.
Each Node contains:
1.  **Children Pointers**: `{'a': Node, 'b': Node}`.
2.  **Top-K Cache**: A pre-sorted list of the top 10 most popular completed queries that pass through this node.

**Why Pre-compute Top-K?**
If we wait until runtime to traverse the whole subtree of "sys" to find "system", "systolic"... it's too slow.
We aggregate scores offline. At node "s-y-s", we store `["system", "sysadmin"]` directly.
Time Complexity: $O(1)$ (Pointer chase + Return list).

### 4.2 Data Sharding
The Dictionary is too big for one machine.
-   **Option A: Range Partitioning (A-C, D-F)**
    -   *Risk*: Hotspots. 'S' is popular. 'X' is quiet.
-   **Option B: Hash Partitioning**
    -   `ShardID = Hash(prefix) % 100`.
    -   *Risk*: We can only hash the *prefix*. If user types "s", do we query all shards? No. Usually we shard by the first 2-3 characters.

---

## 5. Data Flow: The Log Processing Pipeline

How do queries get into the Trie?

1.  **Log Collection**: Browser emits "User searched: java" -> Kafka.
2.  **Aggregation (Spark/MapReduce)**:
    -   Window: Last 7 days.
    -   Group By `Query`. Count `Frequency`.
3.  **Trie Building (Offline)**:
    -   Worker reads aggregated list.
    -   Builds the Weighted Trie.
    -   Serializes to file (`trie_snapshot_v100.bin`).
4.  **Deployment**:
    -   Push file to S3.
    -   Typeahead Servers download and hot-swap memory.

---

## 6. Scaling Strategies

### 6.1 The "Trending" Problem (Real-time)
Offline builds happen weekly. But "Breaking News" needs to appear now.
**Hybrid Architecture**:
-   **Result = Merge(Static_Trie, Dynamic_Trie)**.
-   **Dynamic Trie**: Size limited (e.g., 200MB). Stores only high-velocity queries from the last 15 mins (via a Storm/Flink stream).

### 6.2 Sampling
We don't log every keystroke. We log the *final submission*.
Or we sample 1% of traffic for analytics.

---

## 7. Implementation: A Basic Typeahead Class

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.top_candidates = [] # Stores top 5 tuples (score, text)

    def insert(self, word, score):
        # Insert into cache if score is high enough
        self.top_candidates.append((score, word))
        self.top_candidates.sort(key=lambda x: -x[0])
        self.top_candidates = self.top_candidates[:5]

class TypeaheadSystem:
    def __init__(self):
        self.root = TrieNode()
        
    def add_query(self, query, score):
        node = self.root
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            # Crucial: Update the pre-computed top-k at every step
            node.insert(query, score)
            
    def suggest(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return [text for score, text in node.top_candidates]

# Usage
ta = TypeaheadSystem()
ta.add_query("amazon", 100)
ta.add_query("amazing", 50)
ta.add_query("am", 10)

print(ta.suggest("am")) 
# Returns ['amazon', 'amazing', 'am'] sorted by score (because we sorted on insert)
```

---

## 8. Monitoring & Metrics

1.  **Zero Results Rate**: % of prefixes that yield no suggestions.
2.  **Keystroke Latency**: p99 latency (must be < 50ms).
3.  **Selection Rate**: How often users click suggestion #3 vs completing the typing.
    -   High click rate on #1 = Good Ranking.

---

## 9. Failure Modes

1.  **The "Dirty Words" Problem**: The algorithm is unbiased. If users search for hate speech, the Trie suggests it.
    -   *Mitigation**: Strict "Blacklist" filter applied during the Offline Build phase.
2.  **Shard Failure**: If 'S' shard goes down, nobody can search 'System'.
    -   *Mitigation**: Replica sets (3 replicas per shard).

---

## 10. Real-World Case Study: Facebook Search

Facebook uses a system called **Unicorn**.
It's a graph-based search engine.
When you type "John", it doesn't just look for string "John".
It looks for: `Friends named John` > `Friends of Friends` > `Celebrities`.
They use **Social Graph Context** as a ranking signal, not just global frequency.

---

## 11. Cost Analysis

Storing 1 Billion queries in a naive Trie is RAM heavy.
**Optimization: Ternary Search Trees (TST)** or **Radix Trees**.
These compress the storage.
-   Nodes with single child are merged (`sys` -> `tem`).
-   Reduces RAM/Pointer overhead by 60%.

---

## 12. Key Takeaways

1.  **Pre-computation**: Never sort at query time. Sort at write time.
2.  **Prefix Sharding**: Hashing is messy for ranges. Use prefix-based sharding with careful heat-balancing.
3.  **Browser Caching**: The fastest query is the one served from `localStorage`.
4.  **Privacy**: Typeahead leaks intent. Encrypt the channel (HTTPS) and anonymize the logs.

---

**Originally published at:** [arunbaby.com/ml-system-design/0048-trie-based-search](https://www.arunbaby.com/ml-system-design/0048-trie-based-search/)

*If you found this helpful, consider sharing it with others who might benefit.*
