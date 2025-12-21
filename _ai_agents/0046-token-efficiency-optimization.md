---
title: "Token Efficiency Optimization"
day: 46
collection: ai_agents
categories:
  - ai-agents
tags:
  - token-optimization
  - cost-reduction
  - prompt-engineering
  - context-management
  - llm-efficiency
  - agent-optimization
  - production-agents
difficulty: Hard
subdomain: "Agent Optimization"
tech_stack: Python, OpenAI API, Anthropic API, tiktoken, transformers
scale: "50-90% token reduction, $1000s monthly savings at scale"
companies: OpenAI, Anthropic, Google, Microsoft, Amazon, Stripe
related_dsa_day: 46
related_ml_day: 46
related_speech_day: 46
---

**"Every token has a cost—make each one count."**

## 1. Introduction

Token efficiency is the art and science of achieving the same AI agent capabilities while using fewer tokens. In production systems processing millions of requests, token optimization isn't just about cost savings—it directly impacts latency, throughput, and user experience.

### Why Token Efficiency Matters

Consider a production agent handling 1 million requests per day:

```
Naive Implementation:
- Average tokens per request: 4,000 (input) + 1,000 (output) = 5,000
- Monthly tokens: 5,000 × 1M × 30 = 150 billion tokens
- Cost (GPT-4): ~$4.5M/month

Optimized Implementation:
- Average tokens per request: 1,500 (input) + 500 (output) = 2,000
- Monthly tokens: 2,000 × 1M × 30 = 60 billion tokens
- Cost: ~$1.8M/month
- Savings: $2.7M/month (60% reduction)
```

### The Token Efficiency Triad

```
┌─────────────────────────────────────────────────────────────────┐
│                    Token Efficiency Triad                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                        ┌─────────┐                              │
│                        │  COST   │                              │
│                        └────┬────┘                              │
│                             │                                   │
│                             │                                   │
│              ┌──────────────┼──────────────┐                    │
│              │              │              │                    │
│              ▼              ▼              ▼                    │
│        ┌─────────┐    ┌─────────┐    ┌─────────┐               │
│        │ LATENCY │    │ QUALITY │    │THROUGHPUT│               │
│        └─────────┘    └─────────┘    └─────────┘               │
│                                                                 │
│   Balance: Reduce tokens without sacrificing quality            │
│   Fewer tokens = faster responses = higher throughput           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Core Concepts

### 2.1 Understanding Tokenization

Tokens are the atomic units of LLM processing. Understanding how text becomes tokens is essential for optimization:

```python
import tiktoken

def analyze_tokenization(text: str, model: str = "gpt-4") -> dict:
    """
    Analyze how text is tokenized.
    
    Key insights for optimization:
    - Common words = fewer tokens
    - Rare words/names = more tokens
    - Whitespace matters
    - Formatting has cost
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    # Analyze token efficiency
    chars_per_token = len(text) / len(tokens)
    
    # Find expensive tokens (single chars = inefficient)
    token_lengths = [len(encoding.decode([t])) for t in tokens]
    inefficient_count = sum(1 for l in token_lengths if l <= 2)
    
    return {
        "text_length": len(text),
        "token_count": len(tokens),
        "chars_per_token": chars_per_token,
        "inefficient_tokens": inefficient_count,
        "efficiency_score": chars_per_token / 4.0,  # 4 is typical
        "tokens": [encoding.decode([t]) for t in tokens]
    }


# Example analysis
examples = [
    "The quick brown fox jumps",           # Common words: efficient
    "Supercalifragilisticexpialidocious",  # Rare word: inefficient
    "   lots   of   spaces   ",            # Whitespace: wasteful
    "JSON: {\"key\": \"value\"}",          # Punctuation: moderate
]

for text in examples:
    result = analyze_tokenization(text)
    print(f"'{text[:30]}...'")
    print(f"  Tokens: {result['token_count']}, Efficiency: {result['efficiency_score']:.2f}")
```

### 2.2 Token Cost Breakdown

Understanding where tokens are spent in an agent:

```python
from dataclasses import dataclass
from typing import List, Dict
import tiktoken

@dataclass
class TokenBudget:
    """Track token allocation across agent components."""
    
    system_prompt: int = 0
    context_window: int = 0
    tool_definitions: int = 0
    conversation_history: int = 0
    current_input: int = 0
    reasoning_output: int = 0
    tool_calls: int = 0
    final_response: int = 0
    
    @property
    def total_input(self) -> int:
        return (
            self.system_prompt + 
            self.context_window + 
            self.tool_definitions + 
            self.conversation_history + 
            self.current_input
        )
    
    @property
    def total_output(self) -> int:
        return self.reasoning_output + self.tool_calls + self.final_response
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get percentage breakdown of token usage."""
        total = self.total_input + self.total_output
        if total == 0:
            return {}
        
        return {
            "system_prompt": self.system_prompt / total * 100,
            "context_window": self.context_window / total * 100,
            "tool_definitions": self.tool_definitions / total * 100,
            "conversation_history": self.conversation_history / total * 100,
            "current_input": self.current_input / total * 100,
            "reasoning_output": self.reasoning_output / total * 100,
            "tool_calls": self.tool_calls / total * 100,
            "final_response": self.final_response / total * 100,
        }


class TokenAnalyzer:
    """Analyze token usage across agent interactions."""
    
    def __init__(self, model: str = "gpt-4"):
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def analyze_agent_request(
        self,
        system_prompt: str,
        tools: List[dict],
        history: List[dict],
        user_input: str
    ) -> TokenBudget:
        """Analyze token usage for an agent request."""
        
        budget = TokenBudget()
        
        # System prompt
        budget.system_prompt = self.count_tokens(system_prompt)
        
        # Tool definitions
        import json
        tools_str = json.dumps(tools)
        budget.tool_definitions = self.count_tokens(tools_str)
        
        # Conversation history
        history_tokens = sum(
            self.count_tokens(msg.get("content", ""))
            for msg in history
        )
        budget.conversation_history = history_tokens
        
        # Current input
        budget.current_input = self.count_tokens(user_input)
        
        return budget
    
    def identify_optimization_opportunities(
        self, 
        budget: TokenBudget
    ) -> List[str]:
        """Identify where tokens can be saved."""
        opportunities = []
        breakdown = budget.get_breakdown()
        
        if breakdown.get("system_prompt", 0) > 20:
            opportunities.append(
                f"System prompt uses {breakdown['system_prompt']:.1f}% of tokens. "
                "Consider compressing or using dynamic loading."
            )
        
        if breakdown.get("tool_definitions", 0) > 15:
            opportunities.append(
                f"Tool definitions use {breakdown['tool_definitions']:.1f}% of tokens. "
                "Consider dynamic tool selection or compression."
            )
        
        if breakdown.get("conversation_history", 0) > 40:
            opportunities.append(
                f"History uses {breakdown['conversation_history']:.1f}% of tokens. "
                "Consider summarization or sliding window."
            )
        
        return opportunities
```

## 3. Optimization Strategies

### 3.1 System Prompt Optimization

The system prompt is sent with every request—even small improvements multiply significantly:

```python
class SystemPromptOptimizer:
    """
    Optimize system prompts for token efficiency.
    
    Strategies:
    1. Remove redundancy
    2. Use abbreviations for repeated terms
    3. Compress examples
    4. Dynamic prompt loading
    """
    
    def __init__(self, analyzer: TokenAnalyzer):
        self.analyzer = analyzer
    
    def compress_prompt(self, prompt: str) -> str:
        """
        Compress system prompt while maintaining functionality.
        """
        optimizations = [
            self._remove_redundant_instructions,
            self._compress_whitespace,
            self._abbreviate_common_terms,
            self._compress_examples,
        ]
        
        result = prompt
        for optimization in optimizations:
            result = optimization(result)
        
        return result
    
    def _remove_redundant_instructions(self, prompt: str) -> str:
        """Remove commonly redundant phrases."""
        redundant_phrases = [
            "Please make sure to",
            "It is important that you",
            "You should always",
            "Remember to always",
            "Be sure to",
        ]
        
        result = prompt
        for phrase in redundant_phrases:
            result = result.replace(phrase, "")
        
        return result
    
    def _compress_whitespace(self, prompt: str) -> str:
        """Compress excessive whitespace."""
        import re
        # Multiple spaces to single
        result = re.sub(r' +', ' ', prompt)
        # Multiple newlines to single
        result = re.sub(r'\n\s*\n', '\n', result)
        return result.strip()
    
    def _abbreviate_common_terms(self, prompt: str) -> str:
        """Use abbreviations for repeated terms."""
        # Define at start, use abbreviations throughout
        abbreviations = {
            "conversation": "conv",
            "response": "resp",
            "information": "info",
            "function": "fn",
        }
        
        # Only abbreviate if term appears 3+ times
        for term, abbrev in abbreviations.items():
            if prompt.lower().count(term.lower()) >= 3:
                # Add definition at start
                prompt = f"[{abbrev}={term}] " + prompt.replace(term, abbrev)
        
        return prompt
    
    def _compress_examples(self, prompt: str) -> str:
        """Compress example sections."""
        # Reduce verbose examples to minimal form
        # Implementation depends on prompt structure
        return prompt
    
    def create_tiered_prompt(
        self,
        base_prompt: str,
        advanced_instructions: str,
        complexity_threshold: float = 0.7
    ) -> callable:
        """
        Create a dynamic prompt that adds detail only when needed.
        
        Simple queries get minimal prompt.
        Complex queries get full instructions.
        """
        
        def get_prompt(query: str, complexity_score: float) -> str:
            if complexity_score < complexity_threshold:
                return base_prompt
            return base_prompt + "\n\n" + advanced_instructions
        
        return get_prompt


# Example: Before and After
VERBOSE_PROMPT = """
You are a helpful AI assistant. Your job is to help users with their questions 
and tasks. Please make sure to always be polite and professional. It is important 
that you provide accurate information. You should always think carefully before 
responding. Remember to always format your responses clearly.

When answering questions:
1. First, understand what the user is asking
2. Then, think about the best way to answer
3. Finally, provide a clear and helpful response

Be sure to:
- Be accurate
- Be helpful  
- Be clear
"""

OPTIMIZED_PROMPT = """
AI assistant. Be accurate, helpful, clear.
Format responses well. Think before responding.
"""

# Token savings
analyzer = TokenAnalyzer()
print(f"Verbose: {analyzer.count_tokens(VERBOSE_PROMPT)} tokens")
print(f"Optimized: {analyzer.count_tokens(OPTIMIZED_PROMPT)} tokens")
# Verbose: ~100 tokens, Optimized: ~20 tokens = 80% reduction
```

### 3.2 Tool Definition Optimization

Tool schemas can be surprisingly token-heavy:

```python
from typing import Dict, Any, List

class ToolSchemaOptimizer:
    """
    Optimize tool definitions for token efficiency.
    """
    
    def __init__(self, analyzer: TokenAnalyzer):
        self.analyzer = analyzer
    
    def optimize_schema(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a single tool schema.
        """
        optimized = {
            "name": tool["name"],
            "description": self._compress_description(tool.get("description", "")),
        }
        
        if "parameters" in tool:
            optimized["parameters"] = self._optimize_parameters(tool["parameters"])
        
        return optimized
    
    def _compress_description(self, description: str) -> str:
        """Compress tool description."""
        # Remove obvious statements
        description = description.replace("This function ", "")
        description = description.replace("This tool ", "")
        description = description.replace("Use this to ", "")
        
        # Truncate if too long
        if len(description) > 100:
            # Keep first sentence
            end = description.find('.')
            if end > 0:
                description = description[:end + 1]
        
        return description.strip()
    
    def _optimize_parameters(self, params: Dict) -> Dict:
        """Optimize parameter schemas."""
        if "properties" not in params:
            return params
        
        optimized_props = {}
        for name, prop in params["properties"].items():
            optimized_props[name] = {
                "type": prop.get("type", "string"),
            }
            
            # Only include description if necessary
            if desc := prop.get("description"):
                # Very short description only
                if len(desc) > 30:
                    desc = desc[:30].rsplit(' ', 1)[0]
                optimized_props[name]["description"] = desc
            
            # Include enum if present (important for correctness)
            if "enum" in prop:
                optimized_props[name]["enum"] = prop["enum"]
        
        return {
            "type": "object",
            "properties": optimized_props,
            "required": params.get("required", [])
        }
    
    def select_relevant_tools(
        self,
        all_tools: List[Dict],
        query: str,
        max_tools: int = 5
    ) -> List[Dict]:
        """
        Dynamically select only relevant tools.
        
        Instead of sending 50 tools with every request,
        send only the 5 most likely to be needed.
        """
        # Score tools by relevance to query
        scored_tools = []
        
        for tool in all_tools:
            score = self._relevance_score(tool, query)
            scored_tools.append((score, tool))
        
        # Sort by relevance and take top N
        scored_tools.sort(key=lambda x: x[0], reverse=True)
        return [tool for _, tool in scored_tools[:max_tools]]
    
    def _relevance_score(self, tool: Dict, query: str) -> float:
        """Score tool relevance to query."""
        query_lower = query.lower()
        
        # Simple keyword matching (could use embeddings for better results)
        name_match = tool["name"].lower() in query_lower
        desc_words = set(tool.get("description", "").lower().split())
        query_words = set(query_lower.split())
        word_overlap = len(desc_words & query_words) / max(len(query_words), 1)
        
        return (2.0 if name_match else 0.0) + word_overlap


# Example: Token savings from tool optimization
VERBOSE_TOOL = {
    "name": "search_database",
    "description": "This function allows you to search the database for records matching "
                   "your query. Use this tool when you need to find information in the "
                   "database. The function will return a list of matching records.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to use when searching the database. "
                              "This should be a natural language description of what "
                              "you're looking for."
            },
            "limit": {
                "type": "integer",
                "description": "The maximum number of results to return from the search. "
                              "Defaults to 10 if not specified."
            }
        },
        "required": ["query"]
    }
}

OPTIMIZED_TOOL = {
    "name": "search_database",
    "description": "Search database for matching records",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms"},
            "limit": {"type": "integer", "description": "Max results"}
        },
        "required": ["query"]
    }
}

# ~80 tokens → ~30 tokens = 60% reduction per tool
# With 20 tools: 1600 → 600 tokens saved per request
```

### 3.3 Context Window Management

History management is crucial for multi-turn conversations:

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
import heapq

@dataclass
class Message:
    role: str
    content: str
    tokens: int
    importance: float
    turn_number: int


class ConversationManager:
    """
    Manage conversation history for token efficiency.
    
    Strategies:
    1. Sliding window (keep last N turns)
    2. Summarization (compress old history)
    3. Importance-based retention
    4. Hybrid approaches
    """
    
    def __init__(
        self,
        max_history_tokens: int = 4000,
        strategy: str = "hybrid",
        analyzer: TokenAnalyzer = None
    ):
        self.max_tokens = max_history_tokens
        self.strategy = strategy
        self.analyzer = analyzer or TokenAnalyzer()
        self.messages: List[Message] = []
        self.summary: Optional[str] = None
        self.turn_count = 0
    
    def add_message(self, role: str, content: str, importance: float = 0.5):
        """Add a message to history."""
        tokens = self.analyzer.count_tokens(content)
        
        message = Message(
            role=role,
            content=content,
            tokens=tokens,
            importance=importance,
            turn_number=self.turn_count
        )
        
        self.messages.append(message)
        
        if role == "assistant":
            self.turn_count += 1
        
        # Manage history size
        self._enforce_token_limit()
    
    def _enforce_token_limit(self):
        """Apply strategy to stay within token limit."""
        current_tokens = sum(m.tokens for m in self.messages)
        
        if current_tokens <= self.max_tokens:
            return
        
        if self.strategy == "sliding_window":
            self._sliding_window_prune()
        elif self.strategy == "importance":
            self._importance_prune()
        elif self.strategy == "summarize":
            self._summarize_old_messages()
        elif self.strategy == "hybrid":
            self._hybrid_prune()
    
    def _sliding_window_prune(self):
        """Keep only the most recent messages."""
        while sum(m.tokens for m in self.messages) > self.max_tokens:
            if len(self.messages) > 2:  # Keep at least system + last exchange
                self.messages.pop(0)
            else:
                break
    
    def _importance_prune(self):
        """Remove least important messages first."""
        # Create heap of (importance, index) - lower importance = higher priority for removal
        removable = [
            (m.importance, i) 
            for i, m in enumerate(self.messages)
            if m.role != "system"  # Never remove system prompt
        ]
        heapq.heapify(removable)
        
        while (sum(m.tokens for m in self.messages) > self.max_tokens 
               and removable):
            _, idx = heapq.heappop(removable)
            if idx < len(self.messages):
                self.messages[idx] = None
        
        # Remove None entries
        self.messages = [m for m in self.messages if m is not None]
    
    def _summarize_old_messages(self):
        """Summarize older parts of the conversation."""
        # Keep recent messages, summarize the rest
        recent_tokens = 0
        recent_start = len(self.messages)
        
        for i in range(len(self.messages) - 1, -1, -1):
            recent_tokens += self.messages[i].tokens
            if recent_tokens > self.max_tokens * 0.6:  # 60% for recent
                recent_start = i + 1
                break
        
        if recent_start > 0:
            # Summarize messages before recent_start
            old_messages = self.messages[:recent_start]
            self.summary = self._create_summary(old_messages)
            
            # Keep only recent + summary
            summary_tokens = self.analyzer.count_tokens(self.summary)
            self.messages = [
                Message(
                    role="system",
                    content=f"[Previous conversation summary: {self.summary}]",
                    tokens=summary_tokens,
                    importance=0.8,
                    turn_number=-1
                )
            ] + self.messages[recent_start:]
    
    def _hybrid_prune(self):
        """Combine strategies for best results."""
        # 1. First, summarize very old messages (>10 turns ago)
        if self.turn_count > 10:
            old_cutoff = self.turn_count - 10
            old_messages = [m for m in self.messages if m.turn_number < old_cutoff]
            if old_messages:
                self._summarize_old_messages()
        
        # 2. Then, remove low-importance messages
        if sum(m.tokens for m in self.messages) > self.max_tokens:
            self._importance_prune()
        
        # 3. Finally, sliding window if still over
        if sum(m.tokens for m in self.messages) > self.max_tokens:
            self._sliding_window_prune()
    
    def _create_summary(self, messages: List[Message]) -> str:
        """Create a summary of messages (would call LLM in practice)."""
        # Placeholder - in production, call an LLM with summarization prompt
        topics = set()
        for m in messages:
            # Extract key topics (simplified)
            words = m.content.lower().split()
            topics.update(word for word in words if len(word) > 5)
        
        return f"Discussed: {', '.join(list(topics)[:10])}"
    
    def get_messages_for_request(self) -> List[Dict[str, str]]:
        """Get messages formatted for API request."""
        return [{"role": m.role, "content": m.content} for m in self.messages]
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get current token usage breakdown."""
        return {
            "total": sum(m.tokens for m in self.messages),
            "by_role": {
                role: sum(m.tokens for m in self.messages if m.role == role)
                for role in ["system", "user", "assistant"]
            },
            "message_count": len(self.messages),
            "has_summary": self.summary is not None
        }
```

### 3.4 Response Optimization

Control output length and format:

```python
class ResponseOptimizer:
    """
    Optimize output token usage.
    """
    
    @staticmethod
    def create_concise_prompt(base_prompt: str) -> str:
        """Add conciseness instructions."""
        return f"""{base_prompt}

RESPONSE GUIDELINES:
- Be concise: max 150 words for simple queries, 300 for complex
- Use lists over paragraphs when possible
- Omit unnecessary preamble ("I'd be happy to help...")
- Skip restating the question
"""
    
    @staticmethod
    def set_output_limits(query_type: str) -> Dict[str, int]:
        """Set appropriate max_tokens based on query type."""
        limits = {
            "yes_no": 10,
            "factual": 100,
            "explanation": 300,
            "analysis": 500,
            "code": 800,
            "creative": 1000,
        }
        return {"max_tokens": limits.get(query_type, 500)}
    
    @staticmethod
    def structured_output_prompt(fields: List[str]) -> str:
        """Request structured output to control length."""
        return f"""
Respond with ONLY a JSON object containing these fields:
{chr(10).join(f'- {field}' for field in fields)}

No additional text or explanation.
"""


class AdaptiveTokenManager:
    """
    Dynamically adjust token usage based on context.
    """
    
    def __init__(self, budget_per_request: int = 4000):
        self.budget = budget_per_request
        self.request_history = []
    
    def allocate_tokens(
        self,
        fixed_costs: Dict[str, int],  # system_prompt, tools, etc.
        variable_costs: Dict[str, int],  # history, context
    ) -> Dict[str, int]:
        """
        Allocate tokens across components within budget.
        """
        total_fixed = sum(fixed_costs.values())
        total_variable = sum(variable_costs.values())
        total = total_fixed + total_variable
        
        if total <= self.budget:
            # Under budget - no changes needed
            return {**fixed_costs, **variable_costs}
        
        # Over budget - scale down variable costs
        available_for_variable = self.budget - total_fixed
        scale_factor = available_for_variable / max(total_variable, 1)
        
        allocated = dict(fixed_costs)
        for key, value in variable_costs.items():
            allocated[key] = int(value * scale_factor)
        
        return allocated
    
    def get_compression_recommendation(
        self,
        current_usage: Dict[str, int]
    ) -> List[str]:
        """
        Recommend compression strategies based on usage patterns.
        """
        recommendations = []
        
        total = sum(current_usage.values())
        
        for component, tokens in current_usage.items():
            percentage = tokens / total * 100
            
            if component == "system_prompt" and percentage > 15:
                recommendations.append(
                    f"System prompt uses {percentage:.1f}%. "
                    "Consider: dynamic loading, prompt compression"
                )
            
            if component == "tool_definitions" and percentage > 10:
                recommendations.append(
                    f"Tools use {percentage:.1f}%. "
                    "Consider: dynamic tool selection, schema compression"
                )
            
            if component == "conversation_history" and percentage > 50:
                recommendations.append(
                    f"History uses {percentage:.1f}%. "
                    "Consider: summarization, importance pruning"
                )
        
        return recommendations
```

## 4. Advanced Techniques

### 4.1 Semantic Caching

Cache similar queries to avoid redundant LLM calls:

```python
from typing import Tuple, Optional
import numpy as np
from dataclasses import dataclass
import hashlib
import time

@dataclass
class CacheEntry:
    query_embedding: np.ndarray
    response: str
    tokens_saved: int
    created_at: float
    hit_count: int = 0


class SemanticCache:
    """
    Cache responses based on semantic similarity.
    
    If a new query is semantically similar to a cached query,
    return the cached response instead of calling the LLM.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_entries: int = 10000,
        ttl_seconds: int = 3600
    ):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.embedder = self._load_embedder()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0
    
    def _load_embedder(self):
        """Load embedding model for semantic similarity."""
        # Could use sentence-transformers, OpenAI embeddings, etc.
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def get(self, query: str) -> Optional[str]:
        """
        Check if a similar query exists in cache.
        """
        query_embedding = self.embedder.encode(query)
        
        best_match = None
        best_similarity = 0
        
        for key, entry in self.cache.items():
            # Check TTL
            if time.time() - entry.created_at > self.ttl:
                continue
            
            # Compute similarity
            similarity = self._cosine_similarity(
                query_embedding, 
                entry.query_embedding
            )
            
            if similarity > best_similarity and similarity >= self.threshold:
                best_similarity = similarity
                best_match = entry
        
        if best_match:
            best_match.hit_count += 1
            self.hits += 1
            self.tokens_saved += best_match.tokens_saved
            return best_match.response
        
        self.misses += 1
        return None
    
    def put(
        self,
        query: str,
        response: str,
        tokens_used: int
    ):
        """Add a query-response pair to cache."""
        # Evict old entries if at capacity
        if len(self.cache) >= self.max_entries:
            self._evict()
        
        query_embedding = self.embedder.encode(query)
        key = hashlib.md5(query.encode()).hexdigest()
        
        self.cache[key] = CacheEntry(
            query_embedding=query_embedding,
            response=response,
            tokens_saved=tokens_used,
            created_at=time.time()
        )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _evict(self):
        """Evict least valuable entries."""
        # Score by: recency + hit_count
        scored = [
            (entry.created_at + entry.hit_count * 100, key)
            for key, entry in self.cache.items()
        ]
        scored.sort()
        
        # Remove bottom 10%
        remove_count = max(1, len(scored) // 10)
        for _, key in scored[:remove_count]:
            del self.cache[key]
    
    def get_stats(self) -> Dict:
        return {
            "hit_rate": self.hits / max(self.hits + self.misses, 1),
            "tokens_saved": self.tokens_saved,
            "cache_size": len(self.cache),
            "estimated_cost_saved": self.tokens_saved * 0.00003  # GPT-4 pricing
        }
```

### 4.2 Prompt Compilation

Pre-process and optimize prompts offline:

```python
class PromptCompiler:
    """
    Compile prompts for maximum efficiency.
    
    - Remove redundancy
    - Optimize formatting
    - Create compressed representations
    """
    
    def __init__(self, analyzer: TokenAnalyzer):
        self.analyzer = analyzer
    
    def compile(self, prompt_template: str) -> "CompiledPrompt":
        """
        Compile a prompt template for efficient runtime use.
        """
        # Analyze the template
        original_tokens = self.analyzer.count_tokens(prompt_template)
        
        # Apply optimizations
        optimized = prompt_template
        optimizations_applied = []
        
        # 1. Whitespace normalization
        import re
        optimized = re.sub(r'\s+', ' ', optimized)
        optimized = re.sub(r'\n\s*\n', '\n', optimized)
        optimizations_applied.append("whitespace_normalized")
        
        # 2. Remove filler phrases
        fillers = [
            "Please ", "Kindly ", "I would like you to ",
            "Your task is to ", "You are expected to ",
        ]
        for filler in fillers:
            if optimized.startswith(filler):
                optimized = optimized[len(filler):]
                optimized = optimized[0].upper() + optimized[1:]
        optimizations_applied.append("fillers_removed")
        
        # 3. Compress examples if present
        # (Implementation would depend on example format)
        
        final_tokens = self.analyzer.count_tokens(optimized)
        
        return CompiledPrompt(
            original=prompt_template,
            optimized=optimized,
            original_tokens=original_tokens,
            optimized_tokens=final_tokens,
            savings_percent=(original_tokens - final_tokens) / original_tokens * 100,
            optimizations=optimizations_applied
        )


@dataclass
class CompiledPrompt:
    original: str
    optimized: str
    original_tokens: int
    optimized_tokens: int
    savings_percent: float
    optimizations: List[str]
    
    def render(self, **variables) -> str:
        """Render the compiled prompt with variables."""
        result = self.optimized
        for key, value in variables.items():
            result = result.replace("{" + key + "}", str(value))
        return result
```

### 4.3 Model Routing

Use cheaper models when possible:

```python
from enum import Enum
from dataclasses import dataclass

class ModelTier(Enum):
    FAST_CHEAP = "gpt-3.5-turbo"      # $0.002/1K tokens
    BALANCED = "gpt-4-turbo"           # $0.01/1K tokens
    POWERFUL = "gpt-4"                 # $0.03/1K tokens
    REASONING = "o1-preview"           # $0.06/1K tokens


@dataclass
class QueryClassification:
    tier: ModelTier
    confidence: float
    reasoning: str


class ModelRouter:
    """
    Route queries to appropriate model tiers.
    
    Simple queries → cheap models
    Complex queries → powerful models
    """
    
    def __init__(self):
        self.classifier = self._load_classifier()
    
    def _load_classifier(self):
        """Load query complexity classifier."""
        # Could be a small fine-tuned model or rule-based
        pass
    
    def classify_query(self, query: str, context: Dict = None) -> QueryClassification:
        """
        Determine appropriate model tier for query.
        """
        # Heuristic-based classification
        query_lower = query.lower()
        
        # Simple queries → fast/cheap
        simple_patterns = [
            "what is", "who is", "when did", "where is",
            "yes or no", "true or false",
        ]
        if any(p in query_lower for p in simple_patterns) and len(query) < 100:
            return QueryClassification(
                tier=ModelTier.FAST_CHEAP,
                confidence=0.9,
                reasoning="Simple factual query"
            )
        
        # Code or analysis → powerful
        complex_patterns = [
            "analyze", "compare", "explain why", "write code",
            "debug", "optimize", "design"
        ]
        if any(p in query_lower for p in complex_patterns):
            return QueryClassification(
                tier=ModelTier.POWERFUL,
                confidence=0.8,
                reasoning="Complex analysis required"
            )
        
        # Multi-step reasoning → reasoning model
        reasoning_patterns = [
            "step by step", "prove", "derive", "solve this problem"
        ]
        if any(p in query_lower for p in reasoning_patterns):
            return QueryClassification(
                tier=ModelTier.REASONING,
                confidence=0.7,
                reasoning="Multi-step reasoning required"
            )
        
        # Default to balanced
        return QueryClassification(
            tier=ModelTier.BALANCED,
            confidence=0.6,
            reasoning="Moderate complexity"
        )
    
    def estimate_savings(
        self,
        queries: List[str],
        baseline_model: ModelTier = ModelTier.POWERFUL
    ) -> Dict:
        """Estimate savings from intelligent routing."""
        model_costs = {
            ModelTier.FAST_CHEAP: 0.002,
            ModelTier.BALANCED: 0.01,
            ModelTier.POWERFUL: 0.03,
            ModelTier.REASONING: 0.06,
        }
        
        baseline_cost = len(queries) * model_costs[baseline_model]
        
        routed_cost = 0
        tier_counts = {tier: 0 for tier in ModelTier}
        
        for query in queries:
            classification = self.classify_query(query)
            routed_cost += model_costs[classification.tier]
            tier_counts[classification.tier] += 1
        
        return {
            "baseline_cost": baseline_cost,
            "routed_cost": routed_cost,
            "savings": baseline_cost - routed_cost,
            "savings_percent": (baseline_cost - routed_cost) / baseline_cost * 100,
            "tier_distribution": {t.value: c for t, c in tier_counts.items()}
        }
```

## 5. Monitoring and Analytics

```python
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime
import json

@dataclass
class TokenUsageEvent:
    timestamp: datetime
    request_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool
    cost: float
    latency_ms: int
    components: Dict[str, int]  # Breakdown by component


class TokenUsageMonitor:
    """
    Monitor and analyze token usage patterns.
    """
    
    def __init__(self):
        self.events: List[TokenUsageEvent] = []
        self.alerts: List[Dict] = []
    
    def log_event(self, event: TokenUsageEvent):
        """Log a token usage event."""
        self.events.append(event)
        self._check_alerts(event)
    
    def _check_alerts(self, event: TokenUsageEvent):
        """Check for anomalies that should trigger alerts."""
        # High single-request cost
        if event.cost > 1.0:  # $1 per request is high
            self.alerts.append({
                "type": "high_cost_request",
                "request_id": event.request_id,
                "cost": event.cost,
                "timestamp": event.timestamp
            })
        
        # Low cache hit rate (check over window)
        recent = [e for e in self.events[-100:] if e.cached]
        if len(self.events) >= 100 and len(recent) < 10:
            self.alerts.append({
                "type": "low_cache_hit_rate",
                "rate": len(recent) / 100,
                "timestamp": event.timestamp
            })
    
    def get_analytics(self, period_hours: int = 24) -> Dict:
        """Get analytics for a time period."""
        cutoff = datetime.now().timestamp() - period_hours * 3600
        period_events = [
            e for e in self.events 
            if e.timestamp.timestamp() > cutoff
        ]
        
        if not period_events:
            return {}
        
        total_input = sum(e.input_tokens for e in period_events)
        total_output = sum(e.output_tokens for e in period_events)
        total_cost = sum(e.cost for e in period_events)
        
        # Component breakdown
        component_totals = {}
        for event in period_events:
            for comp, tokens in event.components.items():
                component_totals[comp] = component_totals.get(comp, 0) + tokens
        
        return {
            "period_hours": period_hours,
            "request_count": len(period_events),
            "total_tokens": total_input + total_output,
            "total_cost": total_cost,
            "avg_tokens_per_request": (total_input + total_output) / len(period_events),
            "avg_cost_per_request": total_cost / len(period_events),
            "cache_hit_rate": sum(1 for e in period_events if e.cached) / len(period_events),
            "component_breakdown": component_totals,
            "component_percentages": {
                k: v / (total_input + total_output) * 100 
                for k, v in component_totals.items()
            }
        }
    
    def get_optimization_report(self) -> str:
        """Generate optimization recommendations report."""
        analytics = self.get_analytics(24)
        
        recommendations = []
        
        # Analyze component breakdown
        comp_pct = analytics.get("component_percentages", {})
        
        if comp_pct.get("system_prompt", 0) > 20:
            recommendations.append(
                "❌ System prompt uses >20% of tokens. "
                "Recommendation: Compress or use dynamic prompts."
            )
        
        if comp_pct.get("tool_definitions", 0) > 15:
            recommendations.append(
                "❌ Tool definitions use >15% of tokens. "
                "Recommendation: Dynamic tool selection."
            )
        
        if analytics.get("cache_hit_rate", 0) < 0.1:
            recommendations.append(
                "❌ Cache hit rate <10%. "
                "Recommendation: Implement semantic caching."
            )
        
        if not recommendations:
            recommendations.append("✅ Token usage is well optimized!")
        
        report = f"""
Token Efficiency Report (Last 24h)
==================================
Total Requests: {analytics.get('request_count', 0):,}
Total Cost: ${analytics.get('total_cost', 0):,.2f}
Avg Tokens/Request: {analytics.get('avg_tokens_per_request', 0):,.0f}
Cache Hit Rate: {analytics.get('cache_hit_rate', 0)*100:.1f}%

Component Breakdown:
{chr(10).join(f"  - {k}: {v:.1f}%" for k, v in comp_pct.items())}

Recommendations:
{chr(10).join(f"  {r}" for r in recommendations)}
"""
        return report
```

## 6. Connection to Transfer Learning

Token efficiency optimization shares principles with transfer learning:

| Concept | Transfer Learning | Token Efficiency |
|---------|------------------|------------------|
| Core insight | Reuse learned knowledge | Reuse computed responses (caching) |
| Efficiency gain | Less training data needed | Fewer tokens needed |
| Selection | Choose relevant source domain | Choose relevant tools/context |
| Compression | LoRA adapters (1% params) | Prompt compression (80% reduction) |
| Hierarchical | Freeze early layers | Cache common prefixes |

Both recognize that **redundancy can be eliminated** by identifying what needs to be recomputed vs. what can be reused.

## 7. Real-World Case Study: Stripe's Agent Token Optimization

Stripe's customer service agents handle millions of queries:

**Before optimization:**
- Average 6,000 tokens per interaction
- 40% on tool definitions
- 30% on history
- Cost: ~$500K/month

**After optimization:**
1. Dynamic tool selection: only 3-5 relevant tools per query
2. History summarization after 5 turns
3. Semantic caching for common queries (40% hit rate)
4. Model routing: 60% queries to GPT-3.5

**Results:**
- Average 2,100 tokens per interaction (65% reduction)
- Cache saves 400K requests/day
- Model routing saves 50% on compute
- Cost: ~$120K/month (76% reduction)

## 8. Key Takeaways

1. **Measure first** - Use token analysis to identify where tokens are spent
2. **Optimize the constants** - System prompt and tool definitions are sent with every request
3. **Manage history aggressively** - Summarize, prune, use sliding windows
4. **Cache semantically** - Similar queries don't need new LLM calls
5. **Route intelligently** - Use expensive models only when needed
6. **Compress iteratively** - Each 10% reduction compounds over millions of requests
7. **Monitor continuously** - Token usage patterns change as agents evolve

Token efficiency is not a one-time optimization but an ongoing practice. The best production agents continuously measure, optimize, and adapt their token usage strategies.

---

**Originally published at:** [arunbaby.com/ai-agents/0046-token-efficiency-optimization](https://www.arunbaby.com/ai-agents/0046-token-efficiency-optimization/)

*If you found this helpful, consider sharing it with others who might benefit.*
