---
title: "Phonetic Search in Speech"
day: 32
related_dsa_day: 32
related_ml_day: 32
related_agents_day: 32
collection: speech_tech
categories:
 - speech_tech
tags:
 - search
 - phonetics
 - asr
 - fuzzy-matching
subdomain: "Speech Retrieval"
tech_stack: [Elasticsearch, Soundex, Metaphone, Kaldi, G2P]
scale: "Millions of audio hours"
companies: [Spotify, Audible, YouTube, Apple Music]
---

**"Finding 'Jon' when the user types 'John', or 'Symphony' when they say 'Simfoni'."**

## 1. The Problem: Spelling vs. Sound

In text search, we assume the user knows the spelling. In speech applications (Voice Search, Podcast Search), users often search by **sound**.

**Challenges:**
1. **Homophones:** "Meat" vs "Meet", "Night" vs "Knight".
2. **Name Variations:** "Jon", "John", "Jhon".
3. **ASR Errors:** User said "Keras", ASR transcribed "Carrots".
4. **Out-of-Vocabulary (OOV):** Searching for a new artist name not in the ASR dictionary.

**Phonetic Search** indexes content by *how it sounds*, not just how it's spelled. This is critical for music search ("Play [Artist]"), contact search ("Call [Name]"), and podcast retrieval.

## 2. Classical Phonetic Algorithms

These algorithms map words to a phonetic code. Words with similar pronunciation get the same code.

### Soundex (1918)
Maps a name to a 4-character code (Letter + 3 Digits).
**Rules:**
1. Keep the first letter.
2. Remove vowels (a, e, i, o, u, y) and 'h', 'w'.
3. Map consonants to digits:
 - b, f, p, v `\to` 1
 - c, g, j, k, q, s, x, z `\to` 2
 - d, t `\to` 3
 - l `\to` 4
 - m, n `\to` 5
 - r `\to` 6
4. Merge adjacent duplicates.
5. Pad with zeros or truncate to length 4.

**Example:**
- **Smith:**
 - Keep 'S'.
 - m `\to` 5, i (drop), t `\to` 3, h (drop).
 - Result: S530.
- **Smyth:**
 - Keep 'S'.
 - m `\to` 5, y (drop), t `\to` 3, h (drop).
 - Result: S530.
- **Match!**

### Metaphone & Double Metaphone (1990, 2000)
Soundex is too simple (fails on "Phone" vs "Fawn"). Metaphone uses more complex rules based on English pronunciation.
- **Double Metaphone:** Returns two codes (Primary and Alternate) to handle ambiguity (e.g., foreign names).
 - "Schmidt" `\to` Primary: XMT (German), Alternate: SMT (Anglicized).

``python
import jellyfish

# Soundex
code1 = jellyfish.soundex("Smith")
code2 = jellyfish.soundex("Smyth")
print(f"Soundex: {code1} == {code2}") # True

# Metaphone
code3 = jellyfish.metaphone("Phone") # 'FN'
code4 = jellyfish.metaphone("Fawn") # 'FN'
print(f"Metaphone: {code3} == {code4}") # True

# Double Metaphone
primary, secondary = jellyfish.metaphone("Schmidt")
# Returns ('XMT', 'SMT')
``

## 3. Neural Phonetic Embeddings

Classical algorithms are rule-based and English-centric. They fail on names like "Siobhan" (pronounced "Shiv-on") unless explicitly coded.
**Neural approaches** learn phonetic similarity from data.

**Siamese Network on Phoneme Sequences:**
1. **G2P (Grapheme-to-Phoneme):** Convert word to phonemes.
 - "Phone" `\to` `F OW N`
 - "Fawn" `\to` `F AO N`
2. **Encoder:** Use an LSTM or Transformer to encode the phoneme sequence into a vector.
3. **Triplet Loss:** Train the model such that:
 - Distance(`Phone`, `Fawn`) is small (Positive pair).
 - Distance(`Phone`, `Cake`) is large (Negative pair).

**Architecture:**
``
Word A -> G2P -> Phonemes A -> [Bi-LSTM] -> Vector A
 ↓
 Similarity (Cosine)
 ↑
Word B -> G2P -> Phonemes B -> [Bi-LSTM] -> Vector B
``

**Benefit:**
- Handles cross-lingual sounds.
- Learns subtle variations (accents).
- Can be trained on "misspelled" search logs (e.g., users typing "Beyonce" as "Beyonse").

## 4. Fuzzy Search on ASR Lattices

When searching inside audio (e.g., "Find where they mentioned 'TensorFlow' in this podcast"), relying on the 1-best transcript is risky.
- Audio: "I used Keras today."
- ASR 1-best: "I used Carrots today."
- Search for "Keras" fails.

**ASR Lattice:** A graph of alternative transcriptions.
``
 /-- Carrots (0.6) --\
I bought today
 \-- Keras (0.4) ----/
``
The lattice contains "Keras" with a lower probability.

**Phonetic Indexing Strategy:**
1. **Lattice-to-Phonemes:** Convert all paths in the lattice to phoneme sequences.
 - Path 1: `K AE R AH T S` (Carrots)
 - Path 2: `K EH R AH S` (Keras)
2. **Index Phoneme N-grams:** Index 3-grams like `K EH R`, `EH R AH`, `R AH S`.
3. **Query Processing:**
 - Query: "TensorFlow" `\to` `T EH N S ER F L OW`.
 - Search for phoneme n-gram matches in the index.

**Elasticsearch Phonetic Token Filter:**
Elasticsearch has a built-in plugin to handle this.
``json
{
 "settings": {
 "analysis": {
 "filter": {
 "my_metaphone": {
 "type": "phonetic",
 "encoder": "metaphone",
 "replace": false
 }
 },
 "analyzer": {
 "my_analyzer": {
 "tokenizer": "standard",
 "filter": ["lowercase", "my_metaphone"]
 }
 }
 }
 }
}
``

## 5. Deep Dive: Grapheme-to-Phoneme (G2P)

To perform phonetic search, we need to convert text (Graphemes) to sound (Phonemes).

**Dictionary Lookup (CMU Dict):**
- `HELLO HH AH L OW`
- Fast, highly accurate for common words.
- **Fails** on OOV words (names, slang, new brands).

**Neural G2P:**
- **Seq2Seq Model:** Transformer trained to translate spelling to pronunciation.
- **Input:** `C H A T G P T`
- **Output:** `CH AE T JH IY P IY T IY`

``python
import torch
import torch.nn as nn

# Simplified Seq2Seq G2P Model
class G2PModel(nn.Module):
 def __init__(self, char_vocab_size, phone_vocab_size, hidden_dim=256):
 super().__init__()
 self.embedding = nn.Embedding(char_vocab_size, hidden_dim)
 self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
 self.fc = nn.Linear(hidden_dim, phone_vocab_size)
 
 def forward(self, x):
 # x: [batch, seq_len] (character indices)
 embedded = self.embedding(x)
 output, (hidden, cell) = self.lstm(embedded)
 # output: [batch, seq_len, hidden_dim]
 logits = self.fc(output)
 return logits
``

## 6. Deep Dive: Connectionist Temporal Classification (CTC) for Search

End-to-end ASR models (like Wav2Vec2) output a probability distribution over characters/phonemes for each time step (frame).

**Posteriorgram Search:**
Instead of decoding to text (which makes hard decisions), store the **Phonetic Posteriorgram** (matrix of Time `\times` Phoneme probabilities).

**Query:** "Alexa" (`AH L EH K S AH`)
**Search:**
1. Convert Query to a template matrix.
2. Slide the query template over the stored posteriorgram.
3. Compute **Dynamic Time Warping (DTW)** distance at each position.
4. If distance < threshold, return timestamp.

**Pros:**
- **No Vocabulary Limit:** Can search for words that didn't exist when the model was trained.
- **Robust:** Works even if the ASR is unsure (high entropy).

**Cons:**
- **Storage:** Storing float matrices is expensive compared to text.
- **Compute:** DTW is O(N \cdot M).

## 7. Deep Dive: Weighted Finite State Transducers (WFSTs)

In professional speech systems (like Kaldi), search is often implemented using WFSTs.

**Concept:**
- **FST:** A graph where edges have input labels, output labels, and weights.
- **Composition (`A \circ B`):** Combining two FSTs.

**Search as Composition:**
To search for a query `Q` in a lattice `L`:
1. Represent Query `Q` as an FST (accepts the query phonemes).
2. Represent Lattice `L` as an FST (paths are phoneme sequences).
3. Compute Intersection: `I = Q \circ L`.
4. If `I` is non-empty, the query exists in the lattice.
5. The shortest path in `I` gives the best time alignment and confidence score.

**Why WFSTs?**
- **Efficiency:** Operations like determinization and minimization optimize the graph for fast search.
- **Flexibility:** Can easily add "fuzzy" matching by composing with an Edit Distance FST (`E`).
 - Search = `Q \circ E \circ L`.

## 8. Deep Dive: Audio Fingerprinting vs. Phonetic Search

Don't confuse Phonetic Search with **Audio Fingerprinting** (e.g., Shazam).

| Feature | Audio Fingerprinting (Shazam) | Phonetic Search |
|:---|:---|:---|
| **Goal** | Identify *exact* recording | Find spoken words |
| **Input** | Audio snippet | Text query or Audio snippet |
| **Matching** | Spectrogram peaks (Constellation Map) | Phonemes / Words |
| **Robustness** | Robust to noise, but needs exact match | Robust to different speakers/accents |
| **Use Case** | "What song is this?" | "Find podcast about AI" |

## 9. Deep Dive: Case Study - E-commerce Voice Search

**Scenario:** User says "I want to buy a *Swarovski* necklace".
**Challenge:** "Swarovski" is hard to spell and pronounce. ASR might output "Swaroski", "Swarofski", "Svaroski".

**Solution:**
1. **Brand Name Expansion:**
 - Generate phonetic variants for all brand names offline.
 - Swarovski `\to` `S W AO R AA F S K IY`, `S W ER AA F S K IY`, etc.
2. **Fuzzy Indexing:**
 - Index the product catalog using these phonetic variants.
3. **Query Expansion:**
 - At runtime, generate phonetic variants for the user's query.
 - Search for all variants in the index.

## 10. Deep Dive: Evaluation Metrics

How do we measure if our phonetic search is working?

1. **Term Error Rate (TER):**
 - Specific to Keyword Search.
 - (False Negatives + False Positives) / Total Occurrences of Keyword.
2. **Mean Average Precision (MAP):**
 - Standard IR metric.
3. **Word Error Rate (WER):**
 - Standard ASR metric, but often uncorrelated with Search success.
 - ASR might get "to" vs "too" wrong (WER increase), but search doesn't care.
 - ASR might get "Taylor" vs "Tyler" wrong (WER increase), and search fails catastrophically.

## Implementation: Phonetic Search Engine

``python
import jellyfish
from collections import defaultdict

class PhoneticSearchEngine:
 def __init__(self):
 self.index = defaultdict(list)
 self.documents = {}
 
 def add_document(self, doc_id, text):
 self.documents[doc_id] = text
 
 # Tokenize
 tokens = text.split()
 for token in tokens:
 # Index by Soundex
 code = jellyfish.soundex(token)
 self.index[code].append(doc_id)
 
 # Index by Metaphone (better precision)
 meta = jellyfish.metaphone(token)
 self.index[meta].append(doc_id)
 
 # Index by Double Metaphone (handle ambiguity)
 # Note: jellyfish.metaphone is actually Double Metaphone in some versions
 
 def search(self, query):
 results = set()
 tokens = query.split()
 
 for token in tokens:
 # Try Soundex
 code = jellyfish.soundex(token)
 if code in self.index:
 results.update(self.index[code])
 
 # Try Metaphone
 meta = jellyfish.metaphone(token)
 if meta in self.index:
 results.update(self.index[meta])
 
 return [self.documents[did] for did in results]

# Usage
engine = PhoneticSearchEngine()
engine.add_document(1, "John Smith is a developer")
engine.add_document(2, "Jon Smyth is a coder")
engine.add_document(3, "Joan Smite is a manager")

print("Query: Jhon Smith")
results = engine.search("Jhon Smith")
for doc in results:
 print(f"- {doc}")
# Expected: Returns Doc 1 and Doc 2. Doc 3 might be excluded depending on algorithm.
``

## Top Interview Questions

**Q1: How do you handle names with multiple pronunciations?**
*Answer:*
Use a **Probabilistic Lexicon**.
- `Data` `\to` `D EY T AH` (0.6)
- `Data` `\to` `D AE T AH` (0.4)
Index both phonetic sequences. During search, match against either.

**Q2: Soundex vs. Metaphone vs. Neural?**
*Answer:*
- **Soundex:** Fast, low precision (many false positives), English only. Good for blocking/filtering.
- **Metaphone:** Better precision, handles more rules. Good for general text search.
- **Neural:** Best for cross-lingual, complex names, and noisy audio. Slower inference.

**Q3: How to search for a keyword in 10,000 hours of audio?**
*Answer:*
Do not run ASR on demand (too slow).
1. **Offline:** Run ASR/Phonetic decoding and index the output (Lattices or Text).
2. **Online:** Search the index (Inverted Index).
3. **Keyword Spotting (KWS):** If you only care about a specific wake word (e.g., "Hey Siri"), use a small streaming model on raw audio, not a search index.

**Q4: How to improve recall for "out of vocabulary" terms?**
*Answer:*
Use **Subword/Phonetic Indexing**. Instead of indexing whole words (which requires them to be in the dictionary), index character n-grams or phoneme n-grams. This allows partial matching even if the word itself is unknown.

**Q5: What is the difference between Keyword Spotting and Phonetic Search?**
*Answer:*
- **KWS:** Detects a *pre-defined* set of keywords in real-time audio (Binary Classification).
- **Phonetic Search:** Finds *arbitrary* queries in a large database of *recorded* audio (Information Retrieval).

## Key Takeaways

1. **Phonetic Search** bridges the gap between spelling and sound, essential for voice interfaces.
2. **Classical Algorithms** (Soundex, Metaphone) are effective baselines for text-based phonetic matching.
3. **Neural G2P + Embeddings** offer state-of-the-art accuracy and cross-lingual support.
4. **ASR Lattices** contain rich information (alternatives) that prevent search failures due to 1-best errors.
5. **Elasticsearch** has built-in support for phonetic matching, making it easy to deploy.

## Summary

| Aspect | Insight |
|:---|:---|
| **Core Problem** | Searching by sound, not spelling |
| **Algorithms** | Soundex, Metaphone, Neural Embeddings |
| **Indexing** | Phoneme n-grams, ASR Lattices |
| **Applications** | Voice Search, Podcast Indexing, Name Matching |

---

**Originally published at:** [arunbaby.com/speech-tech/0032-phonetic-search](https://www.arunbaby.com/speech-tech/0032-phonetic-search/)

*If you found this helpful, consider sharing it with others who might benefit.*
