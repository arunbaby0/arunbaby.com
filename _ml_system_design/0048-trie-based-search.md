---
title: "Trie-based Search Systems"
day: 48
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - trie
  - prefix-search
  - autocomplete
  - search-systems
  - data-structures
difficulty: Hard
subdomain: "Search Systems"
tech_stack: Python, Redis, Elasticsearch
scale: "Millions of entries, sub-10ms latency"
companies: Google, Microsoft, Amazon, LinkedIn
related_dsa_day: 48
related_speech_day: 48
related_agents_day: 48
---

**"Every keystroke triggers a prefix lookup—Tries make it instant."**

## 1. Problem Statement

Design a **trie-based search system** for autocomplete, spell-check, and prefix matching at scale. Support millions of entries with sub-10ms response times.

### Requirements

- **Prefix search**: Find all entries matching a prefix
- **Ranked results**: Return top-K by relevance/popularity
- **Real-time updates**: Add/remove entries dynamically
- **Fuzzy matching**: Handle typos and variations

## 2. Core Trie Implementation

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from heapq import nlargest

@dataclass
class TrieNode:
    children: Dict[str, 'TrieNode'] = None
    is_end: bool = False
    value: str = ""
    weight: float = 0.0  # For ranking
    count: int = 0  # Frequency
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}


class Trie:
    """Production-ready Trie with ranking support."""
    
    def __init__(self):
        self.root = TrieNode()
        self.size = 0
    
    def insert(self, word: str, weight: float = 1.0):
        """Insert word with weight for ranking."""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end:
            self.size += 1
        
        node.is_end = True
        node.value = word
        node.weight = max(node.weight, weight)
        node.count += 1
    
    def search_prefix(self, prefix: str, limit: int = 10) -> List[tuple]:
        """
        Find top-K matches for prefix.
        
        Returns: [(word, weight), ...]
        """
        node = self._find_node(prefix)
        if not node:
            return []
        
        # Collect all words under this prefix
        candidates = []
        self._collect_words(node, candidates)
        
        # Return top K by weight
        return nlargest(limit, candidates, key=lambda x: x[1])
    
    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Navigate to node for prefix."""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def _collect_words(self, node: TrieNode, results: List):
        """DFS to collect all words under node."""
        if node.is_end:
            results.append((node.value, node.weight))
        
        for child in node.children.values():
            self._collect_words(child, results)
    
    def delete(self, word: str) -> bool:
        """Remove word from Trie."""
        def _delete(node, word, depth):
            if depth == len(word):
                if not node.is_end:
                    return False
                node.is_end = False
                self.size -= 1
                return len(node.children) == 0
            
            char = word[depth]
            if char not in node.children:
                return False
            
            should_delete = _delete(node.children[char], word, depth + 1)
            
            if should_delete:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end
            
            return False
        
        return _delete(self.root, word.lower(), 0)
```

## 3. Autocomplete System

```python
from collections import OrderedDict
import time

class AutocompleteSystem:
    """Production autocomplete with caching and ranking."""
    
    def __init__(self, initial_data: List[tuple] = None):
        self.trie = Trie()
        self.cache = LRUCache(1000)
        
        if initial_data:
            for word, weight in initial_data:
                self.trie.insert(word, weight)
    
    def suggest(self, prefix: str, limit: int = 10) -> List[str]:
        """Get autocomplete suggestions."""
        prefix = prefix.lower().strip()
        
        if not prefix:
            return []
        
        # Check cache
        cache_key = f"{prefix}:{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Query Trie
        results = self.trie.search_prefix(prefix, limit)
        suggestions = [word for word, _ in results]
        
        # Cache result
        self.cache[cache_key] = suggestions
        
        return suggestions
    
    def record_selection(self, query: str, selected: str):
        """Boost weight when user selects a suggestion."""
        # Increase weight for learning
        node = self.trie._find_node(selected.lower())
        if node and node.is_end:
            node.weight *= 1.1
            node.count += 1
        
        # Invalidate cache
        self.cache.invalidate_prefix(query)


class LRUCache:
    """LRU cache for autocomplete results."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def __contains__(self, key):
        return key in self.cache
    
    def __getitem__(self, key):
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def __setitem__(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def invalidate_prefix(self, prefix: str):
        """Remove all cached entries starting with prefix."""
        to_remove = [k for k in self.cache if k.startswith(prefix)]
        for k in to_remove:
            del self.cache[k]
```

## 4. Fuzzy Matching with Edit Distance

```python
class FuzzyTrie(Trie):
    """Trie with fuzzy matching support."""
    
    def fuzzy_search(
        self, 
        query: str, 
        max_distance: int = 2,
        limit: int = 10
    ) -> List[tuple]:
        """
        Find words within edit distance of query.
        
        Uses bounded DFS with pruning.
        """
        query = query.lower()
        results = []
        
        def dfs(node, remaining, distance, path):
            # Pruning: exceeded distance budget
            if distance > max_distance:
                return
            
            if node.is_end:
                # Extra chars in word cost deletions
                total_distance = distance + len(remaining)
                if total_distance <= max_distance:
                    results.append((node.value, node.weight, total_distance))
            
            if not remaining:
                # Just deletions from now
                for char, child in node.children.items():
                    dfs(child, "", distance + 1, path + char)
                return
            
            current = remaining[0]
            rest = remaining[1:]
            
            for char, child in node.children.items():
                if char == current:
                    # Match - no cost
                    dfs(child, rest, distance, path + char)
                else:
                    # Substitution
                    dfs(child, rest, distance + 1, path + char)
                    # Insertion (skip this char in word)
                    dfs(child, remaining, distance + 1, path + char)
            
            # Deletion (skip this char in query)
            dfs(node, rest, distance + 1, path)
        
        dfs(self.root, query, 0, "")
        
        # Sort by distance, then weight
        results.sort(key=lambda x: (x[2], -x[1]))
        return [(word, weight) for word, weight, _ in results[:limit]]
```

## 5. Distributed Trie

```python
import hashlib

class DistributedTrie:
    """Shard Trie across multiple nodes."""
    
    def __init__(self, num_shards: int = 16):
        self.num_shards = num_shards
        self.shards = [Trie() for _ in range(num_shards)]
    
    def _get_shard(self, word: str) -> int:
        """Determine shard by first character hash."""
        first_char = word[0].lower() if word else 'a'
        return ord(first_char) % self.num_shards
    
    def insert(self, word: str, weight: float = 1.0):
        shard_id = self._get_shard(word)
        self.shards[shard_id].insert(word, weight)
    
    def search_prefix(self, prefix: str, limit: int = 10):
        """Search relevant shards."""
        if prefix:
            # Single shard for specific prefix
            shard_id = self._get_shard(prefix)
            return self.shards[shard_id].search_prefix(prefix, limit)
        else:
            # Search all shards for empty prefix
            results = []
            for shard in self.shards:
                results.extend(shard.search_prefix(prefix, limit))
            return nlargest(limit, results, key=lambda x: x[1])
```

## 6. Integration with ML Ranking

```python
class MLRankedAutocomplete:
    """Combine Trie retrieval with ML ranking."""
    
    def __init__(self, trie: Trie, ranker):
        self.trie = trie
        self.ranker = ranker  # ML model
    
    def suggest(self, prefix: str, context: Dict, limit: int = 10):
        """
        1. Retrieve candidates from Trie
        2. Rank with ML model
        """
        # Get more candidates than needed
        candidates = self.trie.search_prefix(prefix, limit * 5)
        
        if not candidates:
            return []
        
        # Extract features
        features = self._extract_features(candidates, context)
        
        # Score with ML model
        scores = self.ranker.predict(features)
        
        # Combine Trie weight with ML score
        ranked = [
            (word, 0.3 * trie_weight + 0.7 * ml_score)
            for (word, trie_weight), ml_score in zip(candidates, scores)
        ]
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in ranked[:limit]]
    
    def _extract_features(self, candidates, context):
        """Extract features for ML ranking."""
        features = []
        for word, trie_weight in candidates:
            features.append({
                'trie_weight': trie_weight,
                'word_length': len(word),
                'context_match': self._context_similarity(word, context),
                'recency': context.get('recency_scores', {}).get(word, 0)
            })
        return features
```

## 7. Connection to Word Search II

Both Word Search II and production Trie systems use the same core insight:
- **Prefix structure enables pruning**
- **Build once, query many times**
- **Early termination when prefix doesn't match**

## 8. Key Takeaways

1. **Trie = O(L) prefix lookup** regardless of dictionary size
2. **Weight/ranking transform** retrieval into recommendation
3. **Fuzzy matching** handles real-world typos
4. **Sharding** scales to billions of entries
5. **ML ranking** combines structure with learned relevance

---

**Originally published at:** [arunbaby.com/ml-system-design/0048-trie-based-search](https://www.arunbaby.com/ml-system-design/0048-trie-based-search/)

*If you found this helpful, consider sharing it with others who might benefit.*
