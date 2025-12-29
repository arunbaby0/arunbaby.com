---
title: "Phonetic Trie"
day: 48
collection: speech_tech
categories:
  - speech-tech
tags:
  - asr
  - phonetic-search
  - trie
  - soundex
  - nlp
difficulty: Medium
subdomain: "ASR Decoding"
tech_stack: Python, Phonemizer, Metaphone
scale: "Searching 100M songs by voice"
companies: Spotify, Apple Music, Alexa
related_dsa_day: 48
related_ml_day: 48
related_agents_day: 48
---

**"Spelling is irrelevant. Sound is everything."**

## 1. Problem Statement

In text search (Google), if you type "Philemon", you expect to find "Philemon".
In voice search (Alexa), the user says `/f aɪ l i m ə n/`.
The ASR might transcribe this as:
-   "Philemon"
-   "File men"
-   "Fill a mon"
-   "Phi Lemon"

If your music database only has the string "Philemon", 3 out of 4 transcripts fail.
**The Problem**: How do we search a database using *sounds* rather than spellings?

---

## 2. Fundamentals: Phonetic Algorithms

To solve this, we need a canonical representation of "Islands of Sound".

### 2.1 Soundex (1918)
The oldest algorithm. Keeps the first letter, maps consonants to numbers, drops vowels.
-   `Robert` -> `R163`.
-   `Rupert` -> `R163`.
-   **Match!**

### 2.2 Metaphone / Double Metaphone (1990)
More sophisticated. Uses English pronunciation rules.
-   `Schmidt` -> `XMT` (X = 'sh' sound).
-   `Smith` -> `SM0`.

### 2.3 Neural Grapheme-to-Phoneme (G2P)
Modern approach. Use a Transformer to convert `Text -> Phonemes`.
-   `Taylor Swift` -> `T EY L ER S W IH F T`.

---

## 3. Architecture: The Dual-Index System

A voice search system maintains two inverted indexes.

1.  **Lexical Index (Standard)**: Maps words to IDs.
2.  **Phonetic Trie (The Solution)**: Maps *Phonetic Hashes* to IDs.

```
[User Says] -> [ASR] -> "File men"
                           |
                           v
                    [Phonetic Encoder (Metaphone)]
                           |
                           v
                        "FLMN"
                           |
                           v
                    [Phonetic Trie Search]
                           |
  Results: ["Philemon" (FLMN), "Philamon" (FLMN)]
```

---

## 4. Model Selection

For English: **Double Metaphone** is industry standard for retrieval because it handles ambiguity (returns Primary and Secondary encodings).
For Multilingual: **Neural G2P**. (Because Soundex logic fails on Chinese names).

---

## 5. Implementation: Building a Phonetic Search

We will implement a simple "Sound Search" using Python's `metaphone` library and a Trie.

```python
import phonetics # pip install phonetics
from collections import defaultdict

class PhoneticSearchEngine:
    def __init__(self):
        # Maps Sound Code -> List of actual words
        # This acts as our hash map / Trie
        self.index = defaultdict(list)
        
    def add_song(self, song_title):
        # 1. Compute the phonetic signature
        # dmetaphone returns tuple (Primary, Secondary)
        primary, secondary = phonetics.dmetaphone(song_title)
        
        # 2. Index both!
        if primary:
            self.index[primary].append(song_title)
        if secondary:
            self.index[secondary].append(song_title)
            
    def search(self, spoken_query):
        # 1. Convert user's transcript to sound code
        primary, secondary = phonetics.dmetaphone(spoken_query)
        
        results = set()
        if primary in self.index:
            results.update(self.index[primary])
        if secondary in self.index:
            results.update(self.index[secondary])
            
        return list(results)

# Usage
engine = PhoneticSearchEngine()
engine.add_song("Taylor Swift")
engine.add_song("Tailor Switch") # Confusingly similar

print(engine.search("Taler Swift")) 
# Output: ['Taylor Swift']
# Explanation: 'Taylor' -> TLER, 'Taler' -> TLER. Match.
```

---

## 6. Training Considerations

If using Neural G2P, you need a dictionary like **CMU Dict** (130k words with phonemes).
-   Training Loss: Cross Entropy on phonemes.
-   Accuracy metric: **PER (Phoneme Error Rate)**, not WER.

---

## 7. Production Deployment: Fuzzy Matching

"Exact Phonetic Match" is too strict.
-   Metaphone(`Stephen`) == Metaphone(`Steven`). (Good).
-   Metaphone(`Alexander`) != Metaphone(`Alexzander`). (Sometimes fails).

**Fuzzy Search**:
Instead of a HashMap, use a **Trie** of phonemes.
We can then perform **Levenshtein Distance** search *on the phoneme string*.
-   Find all songs where `Distance(Phonetic(Query), Phonetic(Target)) < 2`.

---

## 8. Scaling Strategies

Spotify has 100 Million tracks.
Linear scan of phonemes is impossible.
**Finite State Transducers (FST)**:
We compile the entire database of Songs -> Phonemes into a massive FST graph.
The User's voice input is also an FST.
The search is finding the **Intersection** of `User_FST` and `Database_FST`. This is extremely fast (microseconds for millions of items).
Ref: **OpenFST** library.

---

## 9. Quality Metrics

-   **MRR (Mean Reciprocal Rank)**: Did the correct song appear at #1?
-   **Robustness**: Test with "noisy" transcripts. Simulate ASR errors (`cat` -> `bat`) and check if retrieval still works.

---

## 10. Common Failure Modes

1.  **Homophones**: "Read" vs "Red". Phonetically identical `R EH D`.
    -   *Mitigation*: Use Context (Language Model). "Read a book" vs "Color red".
2.  **Proper Names**: New artist names (e.g., "6ix9ine") break standard phonetic rules.
    -   *Mitigation*: Manual "Pronunciation Injection" (Aliasing).

---

## 11. State-of-the-Art

**End-to-End Retrieval (E2E)**
Instead of `Audio -> Text -> Phonemes -> ID`.
We train a dual encoder:
1.  **Audio Encoder**: Embeds wav file into Vector `V_a`.
2.  **Text Encoder**: Embeds song titles into Vector `V_t`.
3.  **Search**: Find closest `V_t` to `V_a` in vector space.
This bypasses phonemes entirely! (Used by Google Now).

---

## 12. Key Takeaways

1.  **Text is lossy**: Converting Audio to Text loses information (intonation, ambiguity).
2.  **Canonicalization**: We map infinite variations of spelling to a finite set of sounds.
3.  **Trie Power**: A Phonetic Trie helps us find "Sounds like" matches efficiently.
4.  **Hybrid Approach**: Use Text Search + Phonetic Search + Vector Search together (Ensemble) for best results.

---

**Originally published at:** [arunbaby.com/speech-tech/0048-phonetic-trie](https://www.arunbaby.com/speech-tech/0048-phonetic-trie/)

*If you found this helpful, consider sharing it with others who might benefit.*
