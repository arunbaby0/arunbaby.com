---
title: "Context Window Management"
day: 12
collection: ai_agents
categories:
  - ai-agents
tags:
  - context-window
  - memory
  - caching
  - attention
  - kv-cache
difficulty: Medium-Easy
---

**"The Finite Canvas of Intelligence: Managing the Agent's RAM."**

## 1. Introduction: The RAM of the LLM

If Vector Search is the "Hard Drive" of an agent (Long-term, slow, massive), the **Context Window** is its **RAM** (Short-term, fast, expensive, volatile).

Every time you send a request to an LLM, you are sending a state snapshot. The model reads the inputs, performs inference, and generates an output. It does not "remember" the previous request unless you pass it in again.

In 2023, context windows were small (4k - 8k tokens).
In 2025, we have models with **1 Million+** tokens (Gemini 1.5, Claude 3).
This leads to a dangerous fallacy: *"Why manage context? Just stuff everything in."*

This is wrong for three reasons:
1.  **Cost:** Input tokens cost money. Re-sending a 100-page manual on every API call allows you to burn through your budget in minutes.
2.  **Latency:** Reading 1M tokens takes time (seconds to minutes). Real-time agents cannot wait 30 seconds to "read the manual" before answering "Hello."
3.  **Accuracy (The Dilution Effect):** The more garbage you put in the context, the less attention the model pays to the signal. This is known as the **Signal-to-Noise Ratio** problem.

Effective **Context Management** is the art of curating the perfect prompt—giving the model exactly what it needs to solve the *current* step, and nothing else.

---

## 2. The Physics of Attention: Why Length Matters

To understand why context matters, we must peek inside the Transformer architecture.

### 2.1 Quadratic Complexity: $O(N^2)$
The core mechanism of an LLM is **Self-Attention**. Every token attends to every other token to determine relationships.
*   10 tokens -> 100 interactions.
*   1,000 tokens -> 1,000,000 interactions.
*   100,000 tokens -> 10,000,000,000 interactions.

While optimizations (FlashAttention, Linear Attention) have reduced the constant factors, the fundamental truth remains: **Longer Context = More Compute**.

### 2.2 The KV Cache (Key-Value Cache)
When you chat with ChatGPT, why is the second message faster than the first?
Because of the **KV Cache**.
The model doesn't re-compute the attention matrices for the "History" part of the prompt. It stores them in GPU memory (VRAM).
*   *Implication for Agents:* If you change the **Systems Prompt** (the start of the context) every turn, you invalidate the cache. If you append to the end, you leverage the cache.
*   *Design Rule:* Keep your massive instructions static at the top. Append dynamic history at the bottom.

---

### 3. "Lost in the Middle" Phenomenon

A landmark paper (Liu et al., 2023) discovered a quirk in LLM psychology.
*   **Primacy Bias:** Models pay huge attention to the **Beginning** (System Prompt).
*   **Recency Bias:** Models pay huge attention to the **End** (User's last question).
*   **The Trough:** Information buried in the middle of a long context window is frequently ignored or hallucinated.

**Agent Strategy:**
If you retrieve 50 documents via RAG, do not just dump them in the middle.
*   Put the **Most Relevant** document at the very bottom (closest to the question).
*   Put the **Critical Instructions** ("Do not delete files") at the very top.

---

## 4. Context Management Strategies

How do we compress "A Lifetime of Memories" into a finite window?

### 4.1 Sliding Window (FIFO)
The simplest approach.
*   Keep the System Prompt.
*   Keep the last $N$ messages.
*   Drop the oldest messages as new ones arrive.
*   *Pros:* Simple. Constant cost.
*   *Cons:* "Amnesia." If the user said their name 50 turns ago, the agent forgets it.

### 4.2 Summarization (The Rolling Summary)
Instead of dropping old messages, convert them.
*   *State:* `[System, Summary, Recent_Msg_1, Recent_Msg_2]`
*   *Process:* When `Recent_Msgs` > 10:
    1.  Send `Summary + Recent_Msgs` to a cheap model (GPT-3.5).
    2.  Prompt: "Update the summary with new key details."
    3.  Replace `Summary` and clear `Recent_Msgs`.
*   *Pros:* Infinite duration.
*   *Cons:* Lossy. "Subtle details" are lost in summarization.

### 4.3 Selective Context (Filtering)
Use a specialized small model to act as a "Bouncer."
*   *Scenario:* You have 100 previous messages.
*   *Question:* "What was the error code?"
*   *Bouncer:* Scans the 100 messages. Identifies messages 42 and 43 as containing "Error".
*   *Context:* Constructs a prompt with ONLY messages 42 and 43.
*   *Pros:* High signal-to-noise ratio.
*   *Cons:* The Bouncer might miss the context.

---

## 5. The Game Changer: Context Caching

In late 2024, providers (Anthropic, Google, OpenAI) introduced **Prompt Caching** APIs.
*   *Old Way:* You pay to upload the 100-page manual on every API call.
*   *New Way:* You upload the manual once. You get a `cache_id`.
*   *Subsequent Calls:* You pass `cache_id`. The provider re-uses the pre-computed KV Cache on their GPU.
*   *Cost:* You pay a storage fee (cheap) but the input token fee drops by 90%.

**Impact on Agents:**
This enables **"Megaprompt Agents."** You can now give an agent a 200kb "Standard Operating Procedure" document in its system prompt without going bankrupt. It makes "Few-Shot Learning" (giving 100 examples) economically viable.

---

## 6. Code: A Context Buffer Implementation

Abstracting logic for a FIFO buffer with a "Pin" feature (keeping System Prompt).

```python
import tiktoken

class ContextBuffer:
    def __init__(self, max_tokens=4000, model="gpt-4"):
        self.max_tokens = max_tokens
        self.encoder = tiktoken.encoding_for_model(model)
        self.system_message = None
        self.history = [] # List of dicts

    def set_system_message(self, content):
        self.system_message = {"role": "system", "content": content}

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        self._prune()

    def _count_tokens(self, messages):
        text = "".join([m["content"] for m in messages])
        return len(self.encoder.encode(text))

    def _prune(self):
        """
        Enforce the token limit by dropping the oldest USER/ASSISTANT messages.
        Always keep the System Message.
        """
        current_tokens = self._count_tokens([self.system_message] + self.history)
        
        while current_tokens > self.max_tokens and len(self.history) > 0:
            # Remove the oldest message (FIFO)
            removed = self.history.pop(0)
            print(f"Pruning message from history: {removed['content'][:20]}...")
            
            # Recalculate
            current_tokens = self._count_tokens([self.system_message] + self.history)

    def get_messages(self):
        """Construct the final payload for the API."""
        if self.system_message:
            return [self.system_message] + self.history
        return self.history

# Usage
buffer = ContextBuffer(max_tokens=100) # Tiny limit for demo
buffer.set_system_message("You are a helpful assistant.")

buffer.add_message("user", "Hello! My name is Alice.")
buffer.add_message("assistant", "Hi Alice!")
buffer.add_message("user", "What is the capital of France?")
buffer.add_message("assistant", "Paris.")
buffer.add_message("user", "Tell me a very long story about dinosaurs...") 
# This last message is long, so it will trigger pruning of "Hello! My name is Alice."

print("Final Payload:", buffer.get_messages())
# Result: System Msg + The Dinosaur request. (Alice is forgotten).
```

---

## 7. Summary

Context is the scarcest resource in the AI economy.
*   **KV Caching** explains the physics of cost.
*   **Lost in the Middle** guides the placement of data.
*   **Context Caching APIs** are changing the economics.

A senior engineer treats Tokens like Bytes in network packets—optimizing, compressing, and caching them to build high-performance systems.

In the next post, we will explore **Multi-Step Reasoning**, where we use this context to perform complex logical deductions.
