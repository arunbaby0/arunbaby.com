---
title: "Phonetic Trie for Speech Recognition"
day: 48
collection: speech_tech
categories:
  - speech-tech
tags:
  - phonetic-search
  - trie
  - pronunciation-dictionary
  - asr
  - speech-recognition
difficulty: Hard
subdomain: "Speech Search"
tech_stack: Python, G2P models
scale: "100K+ pronunciations, sub-5ms lookup"
companies: Google, Apple, Amazon, Nuance
related_dsa_day: 48
related_ml_day: 48
related_agents_day: 48
---

**"Match sounds, not spellings—phonetic Tries unlock voice search."**

## 1. Introduction

When users speak, they express sounds (phonemes), not characters. A phonetic Trie indexes pronunciations to enable voice search that matches what users *say*, not what they *spell*.

### Why Phonetic Search?

```
User says: "I want to order flowers"
ASR hears: "I want to order flours"

Text match: ❌ "flowers" ≠ "flours"  
Phonetic match: ✓ /F L AW1 ER0 Z/ = /F L AW1 ER0 Z/
```

## 2. Phonetic Trie Structure

```python
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

@dataclass
class PhoneticTrieNode:
    children: Dict[str, 'PhoneticTrieNode'] = field(default_factory=dict)
    words: Set[str] = field(default_factory=set)  # Multiple words can share pronunciation
    is_end: bool = False


class PhoneticTrie:
    """Trie indexed by phoneme sequences."""
    
    # CMU phoneme set (39 phonemes)
    PHONEMES = {
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
    }
    
    def __init__(self):
        self.root = PhoneticTrieNode()
        self.word_to_phones = {}  # Reverse lookup
    
    def insert(self, word: str, phonemes: List[str]):
        """
        Insert word with its pronunciation.
        
        Args:
            word: "flower"
            phonemes: ['F', 'L', 'AW1', 'ER0']
        """
        # Normalize phonemes (remove stress markers for matching)
        normalized = [self._normalize_phoneme(p) for p in phonemes]
        
        node = self.root
        for phoneme in normalized:
            if phoneme not in node.children:
                node.children[phoneme] = PhoneticTrieNode()
            node = node.children[phoneme]
        
        node.is_end = True
        node.words.add(word)
        self.word_to_phones[word] = phonemes
    
    def _normalize_phoneme(self, phoneme: str) -> str:
        """Remove stress markers (0, 1, 2)."""
        return ''.join(c for c in phoneme if not c.isdigit())
    
    def search_exact(self, phonemes: List[str]) -> Set[str]:
        """Find words matching exact phoneme sequence."""
        normalized = [self._normalize_phoneme(p) for p in phonemes]
        
        node = self.root
        for phoneme in normalized:
            if phoneme not in node.children:
                return set()
            node = node.children[phoneme]
        
        return node.words if node.is_end else set()
    
    def search_prefix(self, phonemes: List[str]) -> List[str]:
        """Find all words starting with phoneme prefix."""
        normalized = [self._normalize_phoneme(p) for p in phonemes]
        
        node = self.root
        for phoneme in normalized:
            if phoneme not in node.children:
                return []
            node = node.children[phoneme]
        
        # Collect all words under this prefix
        results = []
        self._collect(node, results)
        return results
    
    def _collect(self, node: PhoneticTrieNode, results: List):
        if node.is_end:
            results.extend(node.words)
        for child in node.children.values():
            self._collect(child, results)
```

## 3. Building from CMU Pronouncing Dictionary

```python
class CMUDictLoader:
    """Load CMU Pronouncing Dictionary into PhoneticTrie."""
    
    @staticmethod
    def load(dict_path: str) -> PhoneticTrie:
        """
        Load CMU dict format:
        FLOWER  F L AW1 ER0
        FLOWERS  F L AW1 ER0 Z
        """
        trie = PhoneticTrie()
        
        with open(dict_path, 'r', encoding='latin-1') as f:
            for line in f:
                if line.startswith(';;;'):
                    continue
                
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                word = parts[0]
                # Handle alternate pronunciations: FLOWER(1), FLOWER(2)
                if '(' in word:
                    word = word.split('(')[0]
                
                phonemes = parts[1:]
                trie.insert(word.lower(), phonemes)
        
        return trie
    
    @staticmethod
    def load_custom(entries: List[tuple]) -> PhoneticTrie:
        """Load custom word-pronunciation pairs."""
        trie = PhoneticTrie()
        for word, phonemes in entries:
            trie.insert(word.lower(), phonemes)
        return trie
```

## 4. Fuzzy Phonetic Matching

```python
class FuzzyPhoneticTrie(PhoneticTrie):
    """Phonetic Trie with edit distance support."""
    
    # Phoneme confusion matrix (common ASR errors)
    CONFUSIONS = {
        'N': ['M', 'NG'],
        'T': ['D', 'P'],
        'S': ['Z', 'SH'],
        'B': ['P', 'V'],
        'AE': ['EH', 'AH'],
        'IH': ['IY', 'EH'],
    }
    
    def fuzzy_search(
        self,
        phonemes: List[str],
        max_distance: int = 2,
        limit: int = 10
    ) -> List[tuple]:
        """
        Find words within phonetic edit distance.
        
        Returns: [(word, distance), ...]
        """
        normalized = [self._normalize_phoneme(p) for p in phonemes]
        results = []
        
        def dfs(node, remaining, distance, depth):
            if distance > max_distance:
                return
            
            if node.is_end:
                total = distance + len(remaining)
                if total <= max_distance:
                    for word in node.words:
                        results.append((word, total))
            
            if not remaining:
                for phoneme, child in node.children.items():
                    dfs(child, [], distance + 1, depth + 1)
                return
            
            current = remaining[0]
            rest = remaining[1:]
            
            for phoneme, child in node.children.items():
                if phoneme == current:
                    # Exact match
                    dfs(child, rest, distance, depth + 1)
                elif phoneme in self.CONFUSIONS.get(current, []):
                    # Likely confusion (lower cost)
                    dfs(child, rest, distance + 0.5, depth + 1)
                else:
                    # Substitution
                    dfs(child, rest, distance + 1, depth + 1)
                    # Insertion
                    dfs(child, remaining, distance + 1, depth + 1)
            
            # Deletion
            dfs(node, rest, distance + 1, depth)
        
        dfs(self.root, normalized, 0, 0)
        
        # Sort by distance
        results.sort(key=lambda x: x[1])
        
        # Deduplicate
        seen = set()
        unique = []
        for word, dist in results:
            if word not in seen:
                seen.add(word)
                unique.append((word, dist))
        
        return unique[:limit]
```

## 5. Integration with ASR

```python
class PhoneticSearchEngine:
    """Complete phonetic search with G2P fallback."""
    
    def __init__(self, dict_path: str):
        self.trie = CMUDictLoader.load(dict_path)
        self.g2p = G2PModel()  # Grapheme-to-phoneme model
    
    def search(
        self,
        query: str,
        fuzzy: bool = True,
        limit: int = 10
    ) -> List[str]:
        """
        Search by text (converts to phonemes first).
        """
        # Get phonemes for query
        phonemes = self._text_to_phonemes(query)
        
        if fuzzy:
            results = self.trie.fuzzy_search(phonemes, max_distance=2, limit=limit)
            return [word for word, _ in results]
        else:
            return list(self.trie.search_exact(phonemes))
    
    def search_phonemes(
        self,
        phonemes: List[str],
        fuzzy: bool = True,
        limit: int = 10
    ) -> List[str]:
        """
        Search directly by phoneme sequence (from ASR output).
        """
        if fuzzy:
            results = self.trie.fuzzy_search(phonemes, max_distance=2, limit=limit)
            return [word for word, _ in results]
        else:
            return list(self.trie.search_exact(phonemes))
    
    def _text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to phonemes using dictionary + G2P."""
        words = text.lower().split()
        all_phonemes = []
        
        for word in words:
            if word in self.trie.word_to_phones:
                all_phonemes.extend(self.trie.word_to_phones[word])
            else:
                # Use G2P model for OOV words
                all_phonemes.extend(self.g2p.predict(word))
        
        return all_phonemes
    
    def find_homophones(self, word: str) -> List[str]:
        """Find words with same pronunciation."""
        if word not in self.trie.word_to_phones:
            return []
        
        phonemes = self.trie.word_to_phones[word]
        matches = self.trie.search_exact(phonemes)
        
        return [w for w in matches if w != word]


class G2PModel:
    """Grapheme-to-Phoneme prediction for OOV words."""
    
    def predict(self, word: str) -> List[str]:
        """Predict phonemes for unknown word."""
        # Simplified: use rules or neural model
        # Real implementation would use seq2seq or transformer
        pass
```

## 6. Voice Search Application

```python
class VoiceSearchEngine:
    """Complete voice search with phonetic matching."""
    
    def __init__(self, catalog: List[Dict], dict_path: str):
        self.phonetic_engine = PhoneticSearchEngine(dict_path)
        self.catalog = {item['name'].lower(): item for item in catalog}
        self.catalog_trie = self._build_catalog_trie(catalog)
    
    def _build_catalog_trie(self, catalog):
        """Build phonetic index of catalog items."""
        trie = FuzzyPhoneticTrie()
        
        for item in catalog:
            name = item['name'].lower()
            phonemes = self.phonetic_engine._text_to_phonemes(name)
            trie.insert(name, phonemes)
        
        return trie
    
    def search(
        self,
        spoken_text: str,
        spoken_phonemes: List[str] = None
    ) -> List[Dict]:
        """
        Search catalog by spoken query.
        
        Args:
            spoken_text: ASR transcript
            spoken_phonemes: Optional phoneme sequence from ASR
        """
        # Use phonemes if available (more accurate)
        if spoken_phonemes:
            matches = self.catalog_trie.fuzzy_search(
                spoken_phonemes, max_distance=3, limit=10
            )
        else:
            phonemes = self.phonetic_engine._text_to_phonemes(spoken_text)
            matches = self.catalog_trie.fuzzy_search(
                phonemes, max_distance=3, limit=10
            )
        
        # Return catalog items
        results = []
        for name, distance in matches:
            if name in self.catalog:
                item = self.catalog[name].copy()
                item['match_score'] = 1.0 - (distance / 3.0)
                results.append(item)
        
        return results
```

## 7. Connection to Word Search II

Both use Trie + DFS with pruning:
- Word Search II: Match character paths in grid
- Phonetic Trie: Match phoneme paths in pronunciation space

Key insight: **Trie structure enables efficient prefix matching regardless of alphabet**.

## 8. Key Takeaways

1. **Index by phonemes**, not characters for voice search
2. **Normalize stress markers** for robust matching
3. **Support fuzzy matching** for ASR errors
4. **Use confusion matrix** to weight likely errors lower
5. **Integrate G2P** for out-of-vocabulary words

---

**Originally published at:** [arunbaby.com/speech-tech/0048-phonetic-trie](https://www.arunbaby.com/speech-tech/0048-phonetic-trie/)

*If you found this helpful, consider sharing it with others who might benefit.*
