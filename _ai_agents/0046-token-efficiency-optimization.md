---
title: "Token Efficiency Optimization"
day: 46
related_dsa_day: 46
related_ml_day: 46
related_speech_day: 46
collection: ai_agents
categories:
 - ai-agents
tags:
 - optimization
 - prompt-engineering
 - cost-reduction
 - latency
 - context-window
difficulty: Hard
subdomain: "Agent Performance"
tech_stack: Python, Tiktoken, GPT-4, Anthropic
scale: "Reducing costs by 50% at scale"
companies: OpenAI, Anthropic, Jasper, Copy.ai
---

**"The most expensive token is the one you didn't need to send."**

## 1. Introduction

In the world of AI Agents, **Tokens are Currency**.
- **Cost**: You pay per 1M input tokens. A verbose system prompt sent 100,000 times a day drains the budget.
- **Latency**: The more tokens the LLM reads, the slower the Time-To-First-Token (TTFT). Reading 10k tokens takes seconds.
- **Performance**: The "Lost in the Middle" phenomenon means LLMs forget instructions if the context is too stuffed.

**Token Efficiency Optimization** is the practice of compressing your agent's cognitive load without lobotomizing its intelligence. It is the "garbage collection" and "compression" of the Agentic world.

---

## 2. Core Concepts: The Anatomy of a Prompt

An Agent's context window isn't just a string. It has rigid sections, each with a "Token Tax":

1. **System Prompt**: The static instructions ("You are a helpful assistant..."). Sent *every single turn*.
2. **Tool Definitions**: The JSON schemas describing functions (`search_google`, `calculator`). Can be huge.
3. **Conversation History**: The dynamic chat log. Grows linearly (or super-linearly if tools are verbose).
4. **RAG Context**: The retrieved documents. Often the biggest chunk (3k-10k tokens).

Optimization attacks each of these layers.

---

## 3. Architecture Patterns for Efficiency

We treat the Context Window as a **Cache**.

### 3.1 The "Context Manager" Pattern
Instead of blindly appending `messages.append(response)`, we use a sophisticated manager class.
- **FIFO Buffer**: Keep last N turns.
- **Summarization**: When history > 10 turns, ask a cheap model (GPT-3.5) to summarize turns 1-5 into a single "Memory" string.
- **Tool Pruning**: Only inject tool definitions relevant to the current state.

### 3.2 Prompt Compression (The "Minification")
Much like minifying Javascript (`function variableName` -> `function a`), we can minify prompts.
- **Original**: "Please analyze the following text and determine the sentiment score between 0 and 1." (15 tokens)
- **Optimized**: "Analyze text. Return sentiment 0-1." (6 tokens)
- **Savings**: 60%.

---

## 4. Implementation Approaches

### Strategy A: Dynamic Tool Loading
If you have 50 tools, don't put 50 schemas in the system prompt.
Use a **Router**.
1. **Classifier Step**: "User asks about Math."
2. **Loader Step**: Load only `Calculator`, `WolframAlpha`.
3. **Execution Step**: Run Agent.

### Strategy B: System Prompt Refactoring
Standardize on terse, expert language. LLMs (especially GPT-4) understand compressed instructions.
Instead of "If the user does this, then you should do that", use "User: X -> Action: Y".

---

## 5. Code Examples: The Token Budget Manager

Here is a Python class that actively manages the context window ensuring we never overflow or overspend.

``python
import tiktoken

class TokenBudgetManager:
 def __init__(self, model="gpt-4", max_tokens=8000):
 self.max_tokens = max_tokens
 self.encoding = tiktoken.encoding_for_model(model)
 # Reserve 1000 tokens for the generated reply
 self.safety_margin = 1000 
 
 def count(self, text):
 return len(self.encoding.encode(text))
 
 def compress_history(self, history):
 """
 Compresses conversation history to fit budget.
 Strategy: Keep System Prompt + Last N messages. 
 """
 current_tokens = 0
 preserved_messages = []
 
 # 1. Always keep System Prompt (Critical Instructions)
 system_msg = next((m for m in history if m['role'] == 'system'), None)
 if system_msg:
 current_tokens += self.count(system_msg['content'])
 preserved_messages.append(system_msg)
 
 # 2. Add recent messages backwards until budget hit
 budget = self.max_tokens - current_tokens - self.safety_margin
 
 # Reverse excluding system
 chat_msgs = [m for m in history if m['role'] != 'system']
 
 for msg in reversed(chat_msgs):
 msg_tokens = self.count(msg['content'])
 if budget - msg_tokens >= 0:
 preserved_messages.insert(1, msg) # Insert after system
 budget -= msg_tokens
 else:
 break # Stop adding older messages
 
 return preserved_messages

# Usage
manager = TokenBudgetManager()
history = [
 {"role": "system", "content": "You are a concise agent..."},
 {"role": "user", "content": "Hello world"},
 # ... 50 more messages
]
optimized_history = manager.compress_history(history)
``

---

## 6. Production Considerations

### 6.1 Cost vs. Quality Curve
There is a "Pareto Frontiers" of optimization.
- Removing adjectives: No quality loss.
- Removing examples (Few-Shot): Slight quality loss.
- Removing constraints: High quality loss (Agent hallucinates).
**Rule of Thumb**: Never compress the **Safety Guidelines**.

### 6.2 Caching (The Semantic Cache)
We discussed caching in ML System Design. For Agents, **Semantic Caching** saves 100% of tokens.
Ref: [Cost Management](../ai-agents/0047-cost-management-for-agents).
If `User: "Hello"` is cached, we send 0 tokens to LLM.

---

## 7. Common Pitfalls

1. **JSON Schema Bloat**: Pydantic models with verbose descriptions.
 - *Fix*: Use terse descriptions. `Field(description="The user's age in years")` -> `Field(description="age (yrs)")`.
2. **HTML/XML Residue**: Scraping a website often leaves `<div>`, `class="..."`. These are junk tokens.
 - *Fix*: Use `html2text` or Markdown converters before injecting into context.
3. **Recursive Summarization**: If you summarize a summary of a summary, details wash out (Chinese Whispers).
 - *Fix*: Keep "Key Claims" separate from "Conversation Log".

---

## 8. Best Practices: The "Chain of Density"

A prompting technique from Salesforce Research.
Instead of asking for a summary once, ask the model to:
1. Write a summary.
2. Identify missing entities from the text.
3. Rewrite the summary to include those entities without increasing length.
4. Repeat 5 times.
This creates physically dense information blocks, maximizing information-per-token density.

---

## 9. Connections to Other Topics

This connects deeply to the **Transfer Learning** theme of .
- **Transfer Learning (ML)**: Freezes the backbone to save compute.
- **Token Optimization (Agents)**: Freezes/Compresses the System Prompt (the "backbone" of the agent's personality) to save IO.
- Both are about identifying the "Invariant" (core knowledge) vs the "Variant" (current input) and optimizing the ratio.

---

## 10. Real-World Examples

- **GitHub Copilot**: Uses "Fill-in-the-middle" models but aggressively prunes the surrounding code context to fit the most relevant file imports into the window.
- **AutoGPT**: Struggled famously with cost loops ($10 runs). Newer versions implement "sliding windows" and "memory summarization" by default.

---

## 11. Future Directions

- **Context Caching (Google Gemini 1.5)**: You pay once to upload a huge manual (1M tokens). Subsequent requests referencing that manual are cheap. This essentially "fine-tunes" the cache state.
- **Infinite Attention**: Architectures (like RingAttention) that make context length mathematically irrelevant, shifting the bottleneck purely to compute/cost.

---

## 12. Key Takeaways

1. **Count your tokens**: Use `tiktoken`. Don't guess. Length `\neq` Tokens.
2. **Minify everything**: Prompts, Schemas, HTML.
3. **Manage History**: It's a sliding window, not an infinite scroll.
4. **Density is Quality**: Concise prompts often yield smarter agents because the attention mechanism is less diluted.

---

**Originally published at:** [arunbaby.com/ai-agents/0046-token-efficiency-optimization](https://www.arunbaby.com/ai-agents/0046-token-efficiency-optimization/)

*If you found this helpful, consider sharing it with others who might benefit.*
