---
title: "Trie-Based Search Systems (Typeahead)"
day: 48
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - trie
  - search
  - autocomplete
  - system-design
  - ranking
difficulty: Hard
subdomain: "Search & Retrieval"
tech_stack: Redis, Apache Solr, Elasticsearch, Custom C++
scale: "Billions of daily queries, <20ms latency"
companies: Google, Amazon, Algolia, Facebook
related_dsa_day: 48
related_speech_day: 48
related_agents_day: 48
---

**"The best interfaces read your mind. Or at least, they complete your sentences."**

## 1. Introduction: The Magic of "Type..."

When you type "bey" into Google, it suggests "Beyoncé" instantly. When you type "sys" into your IDE, it suggests `System.out.println`. This is **Typeahead** (or Autocomplete).

It feels simple—just looking up strings—but implementing it at scale is one of the classic "hard" system design problems. Why?

1. **Velocity**: It happens on *every keystroke*. The query volume is 5-10x higher than regular search.
2. **Latency**: The suggestions must appear instantly (< 50ms) effectively "beating" the user's typing speed.
3. **Data Volume**: Google sees billions of unique queries. You can't just store a discrete list.
4. **Ranking**: "Java" could mean the island, the coffee, or the language. The system must pick the most relevant one based on popularity or context.

Today, we'll design a robust Typeahead system using the data structure we mastered in DSA: the **Trie**.

---

## 2. The Core Data Structure: The Weighted Trie

### 2.1 Why a Trie?

If you just stored all possible queries in a database table:
`SELECT * FROM queries WHERE text LIKE 'sys%' ORDER BY popularity DESC LIMIT 5`
This is an `O(N)` scan (or `O(log N)` with loops) that is painfully slow for millions of rows.

A **Trie** is perfect because:
1. **Prefix-based lookup**: Finding all words starting with "sys" requires traversing just 3 nodes (`s` -> `y` -> `s`).
2. **Compactness**: The prefixes "sys", "system", "systolic" share the same initial path, saving memory.

### 2.2 Adding "Weights" for Ranking

We don't just want valid completions; we want the *best* ones. We augment our Trie nodes to store **Top K** popular completions.

**Standard Trie Node:**
```
Node 'a':
  - children: {'b': Node, 'c': Node}
```

**Typeahead Trie Node:**
```
Node 'a':
  - children: {'b': Node, 'c': Node}
  - top_completions: [("amazon", 99), ("amazing", 50), ("apple", 45)]
```

By pre-calculating the top K results at *every* node, the runtime logic becomes trivial: "Go to node for the prefix 'a', and return the cached `top_completions` list." O(1) lookup time!

---

## 3. Optimizing for Scale

### 3.1 The Data Size Problem

A complete Trie of all Google searches would not fit in the RAM of a single machine.
- English words: ~200k.
- Proper nouns, locations, rare queries: ~Billions.

**Solution: Sharding**
We need to split the Trie across multiple servers.

1. **Range Sharding**:
   - Server A: Words starting with 'a' - 'c'
   - Server B: Words starting with 'd' - 'f'
   - *Problem*: 's' (system, school, star wars) is way more popular than 'x' (xylophone). Server 's' melts down.

2. **Hash Sharding (Better)**:
   - `Shard_ID = Hash(prefix) % Num_Servers`
   - *However*, we can only hash the *fixed-size prefix* (e.g., first 2 letters). `Hash("ap")` goes to Server 3.
   - All queries `ap...` ("apple", "application") are stored on Server 3.

### 3.2 Offline vs. Online Updates

**Offline Building (The MapReduce Approach):**
Most search terms don't change popularity instantly. "Facebook" is popular every day.
1. Aggregate logs weekly.
2. Build the entire specific Trie shards efficiently.
3. Push the new Trie files to the fleet.

**Online Updates (The Trending Problem):**
What about breaking news? "Earthquake in Japan" happens now.
We need a **Hybrid System**:
- **Static Trie**: Built weekly, immutable, highly optimized for read speed.
- **Dynamic Trie**: Small, in-memory Trie that updates in real-time (via Kafka stream of searches).
- **Result**: Merge results from Static + Dynamic at query time.

---

## 4. Sampling & Probabilistic Data Structures

Storing every single query ever typed is wasteful. "fawekjlfw" was typed once by a cat walking on a keyboard. We don't want to suggest that.

**Sampling**: Only index queries that have appeared > `N` times.
Using a **Count-Min Sketch** (a probabilistic counter) allows us to count frequency of billions of strings with very little memory. If the count exceeds a threshold, *then* we add it to our main Trie.

---

## 5. Client-Side Optimization

The fastest query is the one you strictly don't make.

1. **Debouncing**: Don't send a request for 'h', then 'he', then 'hel', then 'hell', then 'hello' in 100ms. Wait for a 50ms pause in typing before sending.
2. **Caching**:
   - Usage: User types "star", response is ["star wars", "star trek", "starbucks"].
   - User types "star w": The browser already knows "star wars" was a candidate. It can filter the *cached* list locally while waiting for the server.

---

## 6. Architecture Diagram

```
[User Browser]
      |
      | (GET /suggest?q=sys)
      v
[Load Balancer]
      |
      v
[Typeahead Service] <--- (Reads/Writes) ---> [Redis / Memcached (Hot Queries)]
      |
      | (Cache Miss)
      v
[Trie Service (Read-Only Replicas)]
      |
      | (Loads Trie Data)
      v
[S3 / Blob Storage] <--- (Writes Trie Files) --- [Offline Worker (Spark/MapReduce)]
                                                      ^
                                                      |
                                                 [Search Logs]
```

---

## 7. Comparison: Database vs. Trie

| Feature | SQL Database (`LIKE 'pre%'`) | Trie-Based System |
|---------|------------------------------|-------------------|
| **Latency** | High (Index scan) | Low (Pointer traversal) |
| **Prefix Matching** | Good with index | Native / Excellent |
| **Ranking** | Expensive sort at runtime | Pre-computed |
| **Updates** | Easy (INSERT) | Hard (Rebuilds needed) |
| **Memory** | High overhead | Compact (Shared prefixes) |

---

## 8. Summary & Takeaways

1. **Data Structures Matter**: The Trie isn't just a LeetCode problem; it's the industry standard for prefix search.
2. **Pre-computation is King**: Don't rank results while the user is waiting. Rank them offline and store the top 5 at the node.
3. **Shard Carefully**: Distributing data alphabeticallly creates "hot spots". Hashing is usually safer.
4. **Latency Budget**: With <50ms budgets, every millisecond counts. In-memory structures (Tries) generally beat disk-based databases here.

---

**Originally published at:** [arunbaby.com/ml-system-design/0048-trie-based-search](https://www.arunbaby.com/ml-system-design/0048-trie-based-search/)

*If you found this helpful, consider sharing it with others who might benefit.*
