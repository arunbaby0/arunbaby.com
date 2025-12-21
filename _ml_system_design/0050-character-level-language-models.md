---
title: "Character-Level Language Models"
day: 50
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - language-models
  - character-level
  - nlp
  - rnn
  - transformer
  - text-generation
difficulty: Hard
subdomain: "NLP Systems"
tech_stack: Python, PyTorch, Transformers
scale: "Vocabulary-free, any language, any domain"
companies: Google, OpenAI, Meta, DeepMind
related_dsa_day: 50
related_speech_day: 50
related_agents_day: 50
---

**"No vocabulary needed—predict one character at a time."**

## 1. Problem Statement

Design a **character-level language model** system that:
1. Generates text character by character
2. Handles any language without tokenization
3. Supports domain-specific fine-tuning
4. Enables spelling correction and completion

### Why Character-Level?

```
Token-level LM:
- Fixed vocabulary (50K-100K tokens)
- OOV words → [UNK]
- Language-specific tokenizers

Character-level LM:
- ~100-200 characters (all Unicode)
- No OOV—any word can be formed
- Language-agnostic
- Understands morphology naturally
```

## 2. Model Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class CharVocab:
    """Character vocabulary with special tokens."""
    
    def __init__(self, chars: str):
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
        special = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.chars = special + list(chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
    
    def __len__(self):
        return len(self.chars)
    
    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(c, self.char_to_idx[self.unk_token]) for c in text]
    
    def decode(self, indices: List[int]) -> str:
        return ''.join(self.idx_to_char.get(i, self.unk_token) for i in indices)


class CharLSTM(nn.Module):
    """LSTM-based character language model."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len) character indices
        Returns: logits (batch, seq_len, vocab_size), hidden
        """
        embed = self.embedding(x)  # (batch, seq, embed)
        output, hidden = self.lstm(embed, hidden)  # (batch, seq, hidden)
        logits = self.fc(output)  # (batch, seq, vocab)
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: str = 'cpu'):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        )


class CharTransformer(nn.Module):
    """Transformer-based character language model."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len
    
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len) character indices
        Returns: logits (batch, seq_len, vocab_size)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        embed = self.embedding(x) + self.pos_embedding(positions)
        
        # Causal mask for language modeling
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        output = self.transformer(embed, mask=mask)
        logits = self.fc(output)
        return logits
```

## 3. Training

```python
class CharLMTrainer:
    """Train character language models."""
    
    def __init__(self, model, vocab: CharVocab, device: str = 'cuda'):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.char_to_idx['<PAD>'])
    
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            x = batch['input'].to(self.device)
            y = batch['target'].to(self.device)
            
            optimizer.zero_grad()
            
            logits, _ = self.model(x)
            loss = self.criterion(logits.view(-1, len(self.vocab)), y.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['input'].to(self.device)
                y = batch['target'].to(self.device)
                
                logits, _ = self.model(x)
                loss = self.criterion(logits.view(-1, len(self.vocab)), y.view(-1))
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
```

## 4. Text Generation

```python
class CharLMGenerator:
    """Generate text from character LM."""
    
    def __init__(self, model, vocab: CharVocab, device: str = 'cuda'):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.model.eval()
    
    def generate(
        self,
        prompt: str = "",
        max_length: int = 200,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9
    ) -> str:
        """Generate text continuation."""
        if not prompt:
            prompt = self.vocab.bos_token
        
        input_ids = torch.tensor([self.vocab.encode(prompt)], device=self.device)
        
        generated = list(input_ids[0].cpu().numpy())
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                logits, hidden = self.model(input_ids, hidden)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated.append(next_token.item())
                
                if next_token.item() == self.vocab.char_to_idx.get(self.vocab.eos_token):
                    break
                
                input_ids = next_token
        
        return self.vocab.decode(generated)
    
    def complete(self, prefix: str, num_chars: int = 10) -> str:
        """Complete a word/phrase."""
        return self.generate(prefix, max_length=num_chars, temperature=0.7)
    
    def spell_correct(self, word: str, candidates: List[str]) -> str:
        """Score candidates and return best."""
        scores = []
        for candidate in candidates:
            score = self._score_sequence(candidate)
            scores.append((candidate, score))
        return max(scores, key=lambda x: x[1])[0]
    
    def _score_sequence(self, text: str) -> float:
        """Compute log probability of sequence."""
        input_ids = torch.tensor([self.vocab.encode(text)], device=self.device)
        
        with torch.no_grad():
            logits, _ = self.model(input_ids[:, :-1])
            log_probs = F.log_softmax(logits, dim=-1)
            
            target_ids = input_ids[:, 1:]
            sequence_log_prob = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            
            return sequence_log_prob.sum().item()
```

## 5. Use Cases

### 5.1 Spelling Correction

```python
def spell_correction_demo(generator: CharLMGenerator):
    """Use char LM for spelling correction."""
    misspelled = "recieve"
    candidates = ["receive", "recieve", "retrieve", "relieve"]
    
    corrected = generator.spell_correct(misspelled, candidates)
    print(f"Corrected: {misspelled} → {corrected}")
```

### 5.2 Text Completion

```python
def completion_demo(generator: CharLMGenerator):
    """Autocomplete demonstration."""
    prefix = "The quick brown fox ju"
    completion = generator.complete(prefix, num_chars=20)
    print(f"Completed: {completion}")
```

### 5.3 Domain-Specific Generation

```python
def domain_finetuning():
    """Fine-tune on domain-specific text."""
    # Load base model
    model = CharTransformer(vocab_size=100)
    model.load_state_dict(torch.load('base_model.pt'))
    
    # Fine-tune on medical text
    medical_data = load_medical_corpus()
    trainer = CharLMTrainer(model, vocab)
    
    for epoch in range(5):
        loss = trainer.train_epoch(medical_data, optimizer)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    # Now generates medical-style text
    generator = CharLMGenerator(model, vocab)
    print(generator.generate("Patient presents with"))
```

## 6. Comparison with Token-Level

| Aspect | Character-Level | Token-Level |
|--------|----------------|-------------|
| Vocabulary | ~100-200 | 50K-100K |
| OOV handling | Never OOV | UNK token |
| Sequence length | 5x longer | Shorter |
| Morphology | Natural | Learned |
| Training | Slower | Faster |
| Memory | Less parameters | More parameters |

## 7. Connection to Alien Dictionary

Both involve extracting ordering from sequences:

| Alien Dictionary | Character LM |
|-----------------|--------------|
| Word order → Char order | Text → Next char |
| Explicit constraints | Learned probabilities |
| Deterministic | Probabilistic |

Alien Dictionary = explicit constraint extraction  
Character LM = learned probability distribution

## 8. Key Takeaways

1. **Character-level = Vocabulary-free** - any word possible
2. **Better morphology understanding** - sees word structure
3. **Longer sequences** - need efficient architectures
4. **Use cases**: spelling, completion, generation
5. **Trade-off**: flexibility vs training efficiency

---

**Originally published at:** [arunbaby.com/ml-system-design/0050-character-level-language-models](https://www.arunbaby.com/ml-system-design/0050-character-level-language-models/)

*If you found this helpful, consider sharing it with others who might benefit.*
