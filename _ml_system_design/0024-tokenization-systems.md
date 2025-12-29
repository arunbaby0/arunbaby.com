---
title: "Tokenization Systems"
day: 24
related_dsa_day: 24
related_speech_day: 24
related_agents_day: 24
collection: ml_system_design
categories:
 - ml-system-design
tags:
 - nlp
 - tokenization
 - bpe
 - wordpiece
 - sentencepiece
subdomain: "NLP Preprocessing"
tech_stack: [Python, HuggingFace, SentencePiece]
scale: "Billion-scale corpus processing"
companies: [OpenAI, Google, Meta, HuggingFace]
---

**The critical preprocessing step that defines the vocabulary and capabilities of Large Language Models.**

## The Challenge: How Machines Read

Computers don't understand text; they understand numbers. To feed text into a neural network, we must break it down into smaller units and assign each unit a unique ID. This process is called **Tokenization**.

It sounds simple: just split by space, right?
- "I'm learning AI" -> ["I'm", "learning", "AI"]

But what about:
- "Don't" -> ["Do", "n't"]?
- "New York" -> ["New", "York"] or ["New York"]?
- "unhappiness" -> ["un", "happi", "ness"]?
- Chinese/Japanese text (no spaces)?
- Emojis?

If we just use words, our vocabulary size explodes (English has >1 million words). If we use characters, the sequences become too long and we lose semantic meaning.

Modern LLMs (GPT-4, Llama 3) use **Subword Tokenization**—a sweet spot between characters and words. In this post, we will design a production-grade tokenizer, exploring algorithms like BPE, WordPiece, and SentencePiece that power the AI revolution.

## High-Level Architecture: The Tokenization Pipeline

``ascii
+-----------+ +--------------+ +-----------------+
| Raw Text | --> | Normalizer | --> | Pre-Tokenizer |
+-----------+ +--------------+ +-----------------+
"Héllo!" "hello!" ["hello", "!"]
 |
 v
+-----------+ +--------------+ +-----------------+
| Token IDs | <-- | Post-Process | <-- | Model |
+-----------+ +--------------+ +-----------------+
[101, 7592, [CLS] hello ! BPE / WordPiece
 999, 102] [SEP]
``

## The Spectrum of Tokenization

### 1. Character-Level
- **Method:** Split every character. "Hello" -> ['H', 'e', 'l', 'l', 'o']
- **Vocab Size:** Small (~100-1000).
- **Pros:** No "Unknown" (OOV) tokens.
- **Cons:** Sequences are very long. "The" is 3 tokens. Models struggle to learn long-range dependencies.

### 2. Word-Level
- **Method:** Split by space/punctuation.
- **Vocab Size:** Massive (50k - 1M+).
- **Pros:** Tokens have semantic meaning.
- **Cons:** Huge embedding matrix (memory heavy). Cannot handle rare words ("Out of Vocabulary" problem).

### 3. Subword-Level (The Winner)
- **Method:** Break rare words into meaningful sub-units. "playing" -> "play" + "ing".
- **Vocab Size:** Controlled (30k - 100k).
- **Pros:** Handles unknown words by composition. Efficient sequence length.

## Algorithm 1: Byte Pair Encoding (BPE)

Used by GPT-2, GPT-3, and RoBERTa.

**Training (Learning the Vocabulary):**
1. Start with a vocabulary of all individual characters.
2. Represent the corpus as a sequence of characters.
3. Count the frequency of all adjacent pairs of tokens.
4. Merge the most frequent pair into a new token.
5. Repeat until the desired vocabulary size is reached.

**Example:**
Corpus: "hug", "pug", "pun", "bun"
1. Base vocab: ['b', 'g', 'h', 'n', 'p', 'u']
2. Pairs: ('u', 'g') appears in "hug", "pug". ('u', 'n') appears in "pun", "bun".
3. Merge 'u', 'g' -> 'ug'.
4. New vocab: ['b', 'h', 'n', 'p', 'ug']
5. Now "hug" is ['h', 'ug'].

**Inference (Tokenizing new text):**
Apply the learned merges in the same order.

## Algorithm 2: WordPiece

Used by BERT. Similar to BPE, but instead of merging the *most frequent* pair, it merges the pair that maximizes the **likelihood** of the training data (minimizes perplexity).

It effectively asks: "Which merge increases the probability of the data the most?"

## Algorithm 3: Unigram Language Model

Used by SentencePiece (ALBERT, T5).
Instead of starting small and merging (bottom-up), it starts with a massive vocabulary and **prunes** the least useful tokens (top-down).
It keeps the tokens that minimize the loss of a unigram language model over the corpus.

## System Design: SentencePiece

Most tokenizers assume input is space-separated. But what about Chinese? Or raw binary data?
**SentencePiece** treats the input as a raw stream of Unicode characters (including spaces). It replaces spaces with a special character (e.g., `_` or `<0x20>`) and then runs BPE/Unigram.

**Why is this a big deal?**
It makes the tokenization **reversible** (lossless).
`Detokenize(Tokenize(text)) == text`.
With standard split-by-space, you lose information about whether there were 1 or 2 spaces between words.

## Deep Dive: Implementing BPE from Scratch

Using a library is easy. Let's build BPE from scratch in Python to truly understand it.

``python
from collections import defaultdict

def get_stats(vocab):
 """
 Compute frequencies of adjacent pairs.
 vocab: dict of {"word space": frequency}
 """
 pairs = defaultdict(int)
 for word, freq in vocab.items():
 symbols = word.split()
 for i in range(len(symbols) - 1):
 pairs[symbols[i], symbols[i+1]] += freq
 return pairs

def merge_vocab(pair, v_in):
 """
 Merge the most frequent pair in the vocabulary.
 """
 v_out = {}
 bigram = ' '.join(pair)
 replacement = ''.join(pair)
 for word in v_in:
 w_out = word.replace(bigram, replacement)
 v_out[w_out] = v_in[word]
 return v_out

# 1. Initialize Vocabulary (Character level)
# We add </w> to mark end of word
vocab = {
 "l o w </w>": 5,
 "l o w e r </w>": 2,
 "n e w e s t </w>": 6,
 "w i d e s t </w>": 3
}

num_merges = 10
for i in range(num_merges):
 pairs = get_stats(vocab)
 if not pairs:
 break
 best = max(pairs, key=pairs.get)
 vocab = merge_vocab(best, vocab)
 print(f"Merge {i+1}: {best}")

# Output:
# Merge 1: ('e', 's') -> 'es'
# Merge 2: ('es', 't') -> 'est'
# Merge 3: ('est', '</w>') -> 'est</w>'
# ...
``

## The Hidden Step: Normalization

Before tokenization, we must **normalize** the text.
- **Unicode Normalization (NFKC):** "Héllo" vs "Hello".
- **Lowercasing:** (Optional). BERT-uncased does this. GPT preserves case.
- **Strip Accents:** "Naïve" -> "Naive".

**SentencePiece** is unique because it treats the input as a stream of bytes, so it doesn't need complex normalization rules. It just learns to merge bytes.

## Deep Dive: BPE vs. WordPiece vs. Unigram

It's easy to confuse these. Let's clarify the math.

### 1. BPE (Frequency-Based)
- **Objective:** Maximize the frequency of merged tokens.
- **Algorithm:**
 1. Count all symbol pairs.
 2. Merge the most frequent pair.
 3. Repeat.
- **Pros:** Simple, fast.
- **Cons:** Greedy. A merge now might prevent a better merge later.

### 2. WordPiece (Likelihood-Based)
- **Objective:** Maximize the likelihood of the training data.
- **Algorithm:**
 1. Count all symbol pairs.
 2. For each pair `(A, B)`, calculate the score: `freq(AB) / (freq(A) * freq(B))`.
 3. Merge the pair with the highest score.
- **Intuition:** This is **Pointwise Mutual Information (PMI)**. It prioritizes pairs that appear together *more often than chance*.
- **Example:** "th" is frequent, but "t" and "h" are also frequent individually. "q" and "u" are less frequent, but "q" is *almost always* followed by "u". WordPiece might prefer merging "qu" over "th".

### 3. Unigram (Probabilistic Pruning)
- **Objective:** Minimize the encoding length (entropy) of the text.
- **Algorithm:**
 1. Start with a massive vocabulary (e.g., all substrings).
 2. Train a Unigram Language Model on the current vocab.
 3. Calculate the "loss" (increase in perplexity) if we remove each token.
 4. Remove the bottom 20% of tokens that contribute least to the likelihood.
 5. Repeat until vocab size is reached.
- **Pros:** Probabilistic. Can output multiple segmentations with probabilities (useful for **Subword Regularization**).

## Advanced Topic: Byte-Level BPE (BBPE)

GPT-2 and GPT-3 use **Byte-Level BPE**. Why?

Standard BPE works on Unicode characters.
- **Problem:** There are 140,000+ Unicode characters (Emojis, Kanji, Cyrillic).
- If your base vocab is all Unicode chars, it's too big.
- If you exclude some, you get `[UNK]`.

**Solution:** Treat text as a stream of **Bytes** (UTF-8).
- Base vocab size is exactly 256 (0x00 to 0xFF).
- **Pros:**
 1. **Universal:** Can tokenize ANY string (even binary data, executables, images).
 2. **No [UNK]:** Every byte is in the vocab.
- **Cons:** Sequences are longer (UTF-8 characters can be 1-4 bytes).

**Implementation Detail:**
GPT-2 doesn't merge *across* categories. It won't merge a letter with a punctuation mark. This keeps tokens clean.

## Security: Tokenization Attacks

Did you know you can hack an LLM just by messing with tokenization?

### 1. The " SolidGoldMagikarp" Attack
As mentioned in the appendix, "glitch tokens" are tokens that exist in the vocab but were never seen during training.
- **Attack:** Inject these tokens into the prompt.
- **Effect:** The model's internal activations explode or go to zero, causing it to output gibberish or bypass safety filters.

### 2. Invisible Characters
- **Attack:** Insert invisible Unicode characters (e.g., Zero Width Space `\u200b`) inside a malicious word.
- **Text:** "Kill" -> "K\u200bill".
- **Tokenizer:** Sees "K", "\u200b", "ill".
- **Model:** Doesn't see the token "Kill", so safety filters (which look for specific token IDs) might fail.
- **Defense:** Normalize text (NFKC) to remove invisible characters before tokenization.

## Production Engineering: Serving Tokenizers at Scale

In Python, `tokenizer.encode("text")` takes 1ms. In C++, it takes 10 microseconds.
When serving a model at 100k QPS, Python tokenization is a bottleneck.

### 1. Rust / C++ Bindings
HuggingFace `tokenizers` is written in Rust.
- **Parallelism:** It can tokenize a batch of 1000 sentences in parallel using Rayon.
- **Memory Safety:** No segfaults.
- **Zero-Copy:** Passes pointers between Python and Rust to avoid copying strings.

### 2. Pre-computation
For static datasets (training), we pre-tokenize everything and store it as `int32` arrays (NumPy/Arrow).
- **Format:** `.arrow` or `.bin`.
- **Savings:** "Hello world" (11 bytes) -> `[15496, 995]` (8 bytes). Tokenized data is often smaller than raw text!

### 3. Vocabulary Management
- **Versioning:** Never change the tokenizer after training the model. If you change one ID, the model breaks.
- **Hash Check:** Store the MD5 hash of the `vocab.json` in the model config to prevent mismatches.

## Case Study: Multilingual Tokenization (XLM-R)

Facebook's XLM-RoBERTa supports 100 languages.
- **Vocab Size:** 250,000.
- **Sampling:** They sample training data from each language with `alpha = 0.3`.
 - `Prob ~ (Count)^0.3`.
 - This boosts low-resource languages (Swahili) and suppresses high-resource ones (English).
- **Result:** A single tokenizer that can handle "Hello" (English), "Bonjour" (French), and "नमस्ते" (Hindi).

## Tutorial: Training a SentencePiece Model

Let's get our hands dirty. How do you actually train a tokenizer for a new language?

### 1. Install SentencePiece
``bash
pip install sentencepiece
``

### 2. Prepare Data
You need a single text file with one sentence per line.
`data.txt`:
``text
Hello world
This is a test
...
``

### 3. Train the Model
``python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
 input='data.txt',
 model_prefix='m',
 vocab_size=1000,
 model_type='bpe', # or 'unigram', 'char', 'word'
 character_coverage=1.0, # 1.0 for English, 0.9995 for CJK
 user_defined_symbols=['<sep>', '<cls>']
)
``
This generates `m.model` (the binary) and `m.vocab` (the dictionary).

### 4. Load and Use
``python
sp = spm.SentencePieceProcessor()
sp.load('m.model')

# Encode
print(sp.encode_as_pieces('Hello world'))
# [' Hello', ' world']

# Decode
print(sp.decode_pieces([' Hello', ' world']))
# 'Hello world'
``

**Key Feature:** Reversibility.
`Decode(Encode(s)) == s`.
This is NOT true for BERT's WordPiece (which loses spaces).

## Deep Dive: HuggingFace `tokenizers` Library

The `tokenizers` library (written in Rust) is the engine under the hood of `transformers`.
It has 4 components:

1. **Normalizer:** Cleans text (NFD, Lowercase, Strip Accents).
2. **Pre-Tokenizer:** Splits text into "words" (Whitespace, Punctuation).
 - *Note:* SentencePiece skips this step.
3. **Model:** The core algorithm (BPE, WordPiece, Unigram).
4. **Post-Processor:** Adds special tokens (`[CLS]`, `[SEP]`).

### Building a Tokenizer from Scratch with HF

``python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. Initialize
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 2. Train
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
files = ["data.txt"]
tokenizer.train(files, trainer)

# 3. Save
tokenizer.save("tokenizer.json")
``

## Comparative Analysis: BPE vs. WordPiece across Languages

Which algorithm is better for which language?

| Language | BPE | WordPiece | Unigram |
| :--- | :--- | :--- | :--- |
| **English** | Good. Standard choice (GPT). | Good. Standard choice (BERT). | Good. |
| **Chinese** | Poor. Characters are words. | Poor. | **Best.** Can handle multi-char words without spaces. |
| **German** | Good. Handles compound words ("Donaudampfschiffahrt"). | Good. | Good. |
| **Code** | **Best.** Preserves whitespace. | Poor (strips whitespace). | Okay. |

**Verdict:**
- Use **BPE** for English and Code (GPT style).
- Use **Unigram (SentencePiece)** for Multilingual models (T5, XLM-R).

## Appendix G: The "Glitch Token" Crisis

In 2023, users found that asking ChatGPT about " SolidGoldMagikarp" caused it to hallucinate wildly or crash.
**Why?**
These strings were **User IDs** from the Reddit dataset used to train the tokenizer. They appeared frequently enough to become single tokens.
However, in the *training data* for the model, these IDs might have been filtered out (privacy).
So the model had a token in its vocabulary that it had **never seen** during training.
When it saw this token at inference time, its embeddings were random garbage, causing the model to break.

**Lesson:** Sanitize your tokenizer training data AND your model training data!

## Appendix B: Multilingual Tokenization Challenges

1. **Script Mixing:** If you train on 90% English and 10% Hindi, the Hindi tokens will be very short (characters), making inference slow and quality poor for Hindi.
 - **Fix:** Oversample low-resource languages during tokenizer training.
2. **No Spaces:** Chinese/Japanese/Thai don't use spaces.
 - **Fix:** SentencePiece is essential here. It doesn't rely on pre-tokenization (splitting by space).

## Appendix C: System Design Interview Transcript

**Interviewer:** "Design a tokenizer for a code generation model (like GitHub Copilot)."

**Candidate:** "Code is different from natural language.
1. **Whitespace matters:** In Python, indentation is syntax. We must use a lossless tokenizer like SentencePiece or Byte-Level BPE. We cannot strip spaces.
2. **Vocabulary:** We need tokens for common keywords (`def`, `class`, `return`) and common multi-character operators (`==`, `!=`, `->`).
3. **CamelCase:** Variables like `myVariableName` should probably be split as `my`, `Variable`, `Name`.
4. **Vocab Size:** I'd aim for 50k. Code has a lot of unique identifiers, but we want to learn the structure, not memorize variable names."

**Interviewer:** "How do you handle multiple languages (Python, Java, C++)?"

**Candidate:** "I would train the BPE on a balanced mix of corpora. If I train only on Python, the tokenizer will be inefficient for Java (e.g., it might split `System.out.println` into tiny pieces). I might also add language-specific special tokens like `<|python|>` to hint the model."

## The Future: Token-Free Models

If tokenization is so brittle, why do we use it?
Google's **ByT5** and **CANINE** architectures propose a radical idea: **No Tokenization.**

### ByT5 (Byte-Level T5)
- **Input:** Raw UTF-8 bytes.
- **Architecture:**
 - A heavy **Encoder** that processes bytes.
 - A light **Decoder**.
- **Pros:**
 - Robust to typos ("helo" is 1 byte away from "hello").
 - No OOV (Out of Vocabulary) issues.
 - Multilingual by default.
- **Cons:**
 - **Sequence Length:** "The cat" is 2 tokens (BPE) but 7 bytes. Attention is O(N^2), so 7x length = 49x compute.
 - **Fix:** ByT5 uses a "bottleneck" architecture to downsample the byte stream.

### CANINE (Character Architecture with No tokenization In Neural Encoders)
- Uses a hash-based embedding strategy.
- Instead of a lookup table `Emb[ID]`, it hashes the character codepoint to a vector.
- This allows it to handle an infinite vocabulary of Unicode characters without a massive embedding matrix.

## Visual Tokenization: ViT and VQ-GAN

Tokenization isn't just for text.
**Vision Transformers (ViT)** tokenize images.

1. **Patching:** Split a 224x224 image into 16x16 patches.
 - Result: 196 patches.
2. **Linear Projection:** Flatten each patch and map it to a vector.
 - These vectors are the "tokens".
3. **VQ-GAN (Vector Quantized GAN):**
 - Learns a "visual codebook" of 1024 shapes/textures.
 - An image is represented as a grid of indices `[34, 99, 102, ...]`.
 - This allows us to use GPT on images (DALL-E 1).

## Subword Regularization: BPE-Dropout

Standard BPE is deterministic. "apple" -> "ap", "ple".
But maybe "app", "le" is also valid.
**BPE-Dropout** randomly drops merges during training.
- **Training:** `x = "apple"`.
 - Epoch 1: `["ap", "ple"]`
 - Epoch 2: `["app", "le"]`
 - Epoch 3: `["a", "pp", "le"]`
- **Effect:** The model sees multiple segmentations of the same word. This makes it robust to noise and typos.
- **Result:** 2-3 BLEU score improvement on Machine Translation tasks.

## Appendix E: Comparison of Tokenizer Libraries

| Library | Language | Speed | Features |
| :--- | :--- | :--- | :--- |
| **HuggingFace Tokenizers** | Rust | ⚡️⚡️⚡️ | BPE, WordPiece, Unigram. The industry standard. |
| **SentencePiece (Google)** | C++ | ⚡️⚡️ | Unigram, BPE. Best for multilingual (no pre-tokenization). |
| **Tiktoken (OpenAI)** | Rust | ⚡️⚡️⚡️ | BPE. Optimized for GPT-4. Extremely fast. |
| **YouTokenToMe** | C++ | ⚡️⚡️ | BPE. Fast parallel training. |

**Recommendation:** Use **HuggingFace Tokenizers** for general NLP. Use **Tiktoken** if you are working with OpenAI models. Use **SentencePiece** if you are training a multilingual model from scratch.

## Appendix F: The "Space" Controversy

Should a space be a separate token?
- **BERT:** No. `_hello` (WordPiece uses `##` for suffixes).
- **GPT-2:** Yes. `Ġhello` (uses a special character for space).
- **T5:** Yes. `_hello` (uses underscore).
- **Llama:** Yes. `_hello`.

**Why it matters:**
If you treat space as a separate token, " hello" becomes `[" ", "hello"]`.
If you attach it, it becomes `["_hello"]`.
The latter is more efficient (1 token vs 2).
But what about " hello" (2 spaces)?
- Attached: `["_", "_hello"]`? Or `["__hello"]`?
- This edge case causes headaches in code generation (Python indentation).

## Conclusion

Tokenization is the first layer of the stack. If it's bad, your model is bad.
- **Too aggressive?** You lose meaning.
- **Too conservative?** You run out of memory.
- **Wrong algorithm?** You can't handle emojis or Chinese.

Understanding BPE and SentencePiece gives you the power to debug why your model thinks "2+2" is different from " 2 + 2".


---

**Originally published at:** [arunbaby.com/ml-system-design/0024-tokenization-systems](https://www.arunbaby.com/ml-system-design/0024-tokenization-systems/)

*If you found this helpful, consider sharing it with others who might benefit.*

