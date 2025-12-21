---
title: "Custom Language Modeling for ASR"
day: 50
collection: speech_tech
categories:
  - speech-tech
tags:
  - language-model
  - asr
  - n-gram
  - neural-lm
  - domain-adaptation
difficulty: Hard
subdomain: "ASR Systems"
tech_stack: Python, KenLM, PyTorch
scale: "Millions of training sentences, sub-10ms scoring"
companies: Google, Amazon, Apple, Microsoft, Nuance
related_dsa_day: 50
related_ml_day: 50
related_agents_day: 50
---

**"The right language model makes your ASR understand your domain."**

## 1. Introduction

Language models bias ASR toward likely word sequences. A medical ASR needs "hypertension" more than a banking ASR. Custom LMs bridge this domain gap.

### Why Custom LMs?

```
Generic ASR on medical dictation:
"The patient has high per tension" ❌

With medical LM:
"The patient has hypertension" ✓

The acoustic model heard similar sounds—
the language model resolved the ambiguity.
```

## 2. N-gram Language Models

```python
from collections import defaultdict
from typing import List, Dict, Tuple
import math

class NGramLM:
    """Traditional n-gram language model with smoothing."""
    
    def __init__(self, n: int = 3):
        self.n = n
        self.counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        self.vocab = set()
    
    def train(self, sentences: List[List[str]]):
        """Train on tokenized sentences."""
        for sentence in sentences:
            # Add BOS/EOS tokens
            tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            
            for token in sentence:
                self.vocab.add(token)
            
            # Count n-grams
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                word = tokens[i + self.n - 1]
                
                self.counts[context][word] += 1
                self.context_counts[context] += 1
    
    def probability(self, word: str, context: Tuple[str, ...]) -> float:
        """P(word | context) with Kneser-Ney smoothing."""
        # Simple add-k smoothing for demo
        k = 0.1
        count = self.counts[context][word]
        total = self.context_counts[context]
        
        return (count + k) / (total + k * len(self.vocab))
    
    def score_sentence(self, sentence: List[str]) -> float:
        """Log probability of sentence."""
        tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
        log_prob = 0.0
        
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i + self.n - 1])
            word = tokens[i + self.n - 1]
            
            prob = self.probability(word, context)
            log_prob += math.log(prob) if prob > 0 else float('-inf')
        
        return log_prob
    
    def perplexity(self, sentences: List[List[str]]) -> float:
        """Calculate perplexity on test set."""
        total_log_prob = 0.0
        total_words = 0
        
        for sentence in sentences:
            total_log_prob += self.score_sentence(sentence)
            total_words += len(sentence) + 1  # +1 for </s>
        
        return math.exp(-total_log_prob / total_words)
```

## 3. KenLM Integration

```python
import kenlm

class KenLMWrapper:
    """Wrapper for KenLM n-gram models."""
    
    def __init__(self, model_path: str):
        self.model = kenlm.Model(model_path)
    
    def score(self, text: str) -> float:
        """Log10 probability of text."""
        return self.model.score(text, bos=True, eos=True)
    
    def score_words(self, words: List[str]) -> List[Tuple[str, float]]:
        """Score each word in context."""
        scores = []
        state = kenlm.State()
        self.model.BeginSentenceWrite(state)
        
        for word in words:
            out_state = kenlm.State()
            prob = self.model.BaseScore(state, word, out_state)
            scores.append((word, prob))
            state = out_state
        
        return scores
    
    def perplexity(self, texts: List[str]) -> float:
        """Calculate perplexity."""
        total_log_prob = 0.0
        total_words = 0
        
        for text in texts:
            words = text.split()
            total_log_prob += self.score(text)
            total_words += len(words)
        
        # Convert from log10 to natural log
        return 10 ** (-total_log_prob / total_words)


def train_kenlm(corpus_path: str, output_path: str, order: int = 4):
    """Train KenLM model using lmplz."""
    import subprocess
    
    cmd = f"lmplz -o {order} < {corpus_path} > {output_path}.arpa"
    subprocess.run(cmd, shell=True, check=True)
    
    # Convert to binary for faster loading
    cmd = f"build_binary {output_path}.arpa {output_path}.bin"
    subprocess.run(cmd, shell=True, check=True)
    
    return f"{output_path}.bin"
```

## 4. Neural Language Models for ASR

```python
import torch
import torch.nn as nn

class NeuralLMForASR(nn.Module):
    """LSTM-based LM for ASR rescoring."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        logits = self.fc(output)
        return logits, hidden
    
    def score_sequence(self, token_ids: List[int]) -> float:
        """Score a token sequence."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor([token_ids[:-1]])
            target = torch.tensor([token_ids[1:]])
            
            logits, _ = self(x)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            scores = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
            return scores.sum().item()


class ASRRescorer:
    """Rescore ASR n-best lists with custom LM."""
    
    def __init__(
        self,
        am_weight: float = 1.0,
        lm_weight: float = 0.5,
        word_penalty: float = 0.0
    ):
        self.am_weight = am_weight
        self.lm_weight = lm_weight
        self.word_penalty = word_penalty
        self.lm = None
    
    def load_lm(self, model_path: str):
        """Load language model."""
        self.lm = KenLMWrapper(model_path)
    
    def rescore(self, nbest: List[Dict]) -> List[Dict]:
        """
        Rescore n-best list.
        
        nbest: [{'text': str, 'am_score': float}, ...]
        """
        for hyp in nbest:
            text = hyp['text']
            am_score = hyp['am_score']
            
            lm_score = self.lm.score(text)
            word_count = len(text.split())
            
            # Combine scores
            combined = (
                self.am_weight * am_score +
                self.lm_weight * lm_score +
                self.word_penalty * word_count
            )
            
            hyp['lm_score'] = lm_score
            hyp['combined_score'] = combined
        
        # Sort by combined score (higher = better)
        return sorted(nbest, key=lambda x: x['combined_score'], reverse=True)
```

## 5. Domain Adaptation

```python
class DomainAdaptedLM:
    """Interpolate generic and domain-specific LMs."""
    
    def __init__(
        self,
        generic_lm_path: str,
        domain_lm_path: str,
        interpolation_weight: float = 0.3
    ):
        self.generic_lm = KenLMWrapper(generic_lm_path)
        self.domain_lm = KenLMWrapper(domain_lm_path)
        self.weight = interpolation_weight  # Weight for domain LM
    
    def score(self, text: str) -> float:
        """Interpolated score."""
        generic_score = self.generic_lm.score(text)
        domain_score = self.domain_lm.score(text)
        
        # Linear interpolation in log space (approximately)
        return (1 - self.weight) * generic_score + self.weight * domain_score
    
    def optimize_weight(self, dev_texts: List[str]) -> float:
        """Find optimal interpolation weight on dev set."""
        best_weight = 0.0
        best_ppl = float('inf')
        
        for weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            self.weight = weight
            ppl = self._perplexity(dev_texts)
            
            if ppl < best_ppl:
                best_ppl = ppl
                best_weight = weight
        
        self.weight = best_weight
        return best_weight
    
    def _perplexity(self, texts: List[str]) -> float:
        total_score = sum(self.score(t) for t in texts)
        total_words = sum(len(t.split()) for t in texts)
        return 10 ** (-total_score / total_words)


def create_domain_lm(domain_corpus: str, generic_corpus: str, output_dir: str):
    """Build domain-adapted LM pipeline."""
    # Train domain LM
    domain_lm = train_kenlm(domain_corpus, f"{output_dir}/domain", order=4)
    
    # Train generic LM (or use existing)
    generic_lm = train_kenlm(generic_corpus, f"{output_dir}/generic", order=4)
    
    # Create interpolated model
    adapted_lm = DomainAdaptedLM(generic_lm, domain_lm)
    
    return adapted_lm
```

## 6. Contextual Biasing

```python
class ContextualBiasing:
    """Boost specific words/phrases during decoding."""
    
    def __init__(self):
        self.boost_phrases = {}  # phrase -> boost_weight
    
    def add_phrases(self, phrases: List[str], weight: float = 2.0):
        """Add phrases to boost."""
        for phrase in phrases:
            self.boost_phrases[phrase.lower()] = weight
    
    def compute_bias(self, text: str) -> float:
        """Compute bias score for text."""
        text_lower = text.lower()
        bias = 0.0
        
        for phrase, weight in self.boost_phrases.items():
            if phrase in text_lower:
                bias += weight
        
        return bias
    
    def rescore_with_bias(self, nbest: List[Dict]) -> List[Dict]:
        """Add contextual bias to rescoring."""
        for hyp in nbest:
            bias = self.compute_bias(hyp['text'])
            hyp['bias_score'] = bias
            hyp['combined_score'] = hyp.get('combined_score', 0) + bias
        
        return sorted(nbest, key=lambda x: x['combined_score'], reverse=True)


# Usage example
biasing = ContextualBiasing()
biasing.add_phrases([
    "machine learning",
    "neural network",
    "deep learning",
    "transformer"
], weight=3.0)

nbest = rescorer.rescore(nbest)
nbest = biasing.rescore_with_bias(nbest)
```

## 7. End-to-End Integration

```python
class ASRWithCustomLM:
    """Complete ASR system with custom LM."""
    
    def __init__(
        self,
        acoustic_model,
        language_model: DomainAdaptedLM,
        decoder,
        lm_weight: float = 0.5
    ):
        self.am = acoustic_model
        self.lm = language_model
        self.decoder = decoder
        self.lm_weight = lm_weight
        self.biasing = ContextualBiasing()
    
    def transcribe(self, audio, context_phrases: List[str] = None) -> str:
        """Transcribe with custom LM and optional biasing."""
        # Add context phrases
        if context_phrases:
            self.biasing.add_phrases(context_phrases)
        
        # Get acoustic scores
        am_output = self.am(audio)
        
        # Decode with beam search
        nbest = self.decoder.decode(am_output, n_best=10)
        
        # Rescore with LM
        for hyp in nbest:
            hyp['lm_score'] = self.lm.score(hyp['text'])
            hyp['combined'] = hyp['am_score'] + self.lm_weight * hyp['lm_score']
        
        # Apply biasing
        if context_phrases:
            nbest = self.biasing.rescore_with_bias(nbest)
        
        # Return best
        return max(nbest, key=lambda x: x['combined'])['text']
```

## 8. Connection to Alien Dictionary

Both extract ordering/structure from data:

| Alien Dictionary | Custom LM |
|-----------------|-----------|
| Word order → Char order | Text → Word probabilities |
| Explicit rules | Learned distributions |
| Deterministic | Probabilistic |
| Graph-based | Statistical/Neural |

Both infer hidden structure from observed sequences.

## 9. Key Takeaways

1. **Custom LMs are essential** for domain-specific ASR
2. **N-gram LMs** are fast and interpretable
3. **Neural LMs** capture longer dependencies
4. **Interpolation** combines generic and domain knowledge
5. **Contextual biasing** handles dynamic vocabularies

---

**Originally published at:** [arunbaby.com/speech-tech/0050-custom-language-modeling](https://www.arunbaby.com/speech-tech/0050-custom-language-modeling/)

*If you found this helpful, consider sharing it with others who might benefit.*
