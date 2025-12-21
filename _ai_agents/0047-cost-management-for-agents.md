---
title: "Cost Management for AI Agents"
day: 47
collection: ai_agents
categories:
  - ai-agents
tags:
  - cost-management
  - llm-costs
  - optimization
  - token-budgeting
  - agent-economics
  - production-agents
difficulty: Hard
subdomain: "Agent Operations"
tech_stack: Python, OpenAI API, Anthropic API
scale: "Millions of requests, $10K-$1M monthly budgets"
companies: OpenAI, Anthropic, Google, Microsoft, Stripe
related_dsa_day: 47
related_ml_day: 47
related_speech_day: 47
---

**"An agent that's too expensive to run is an agent that won't run."**

## 1. Introduction

AI agents can be expensive—a poorly optimized agent processing 1M requests/day can cost $100K+/month. Cost management transforms agents from demos to sustainable production systems.

### The Cost Challenge

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Cost Breakdown                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLM API Costs (60-80%)                                         │
│  ├── Input tokens: $0.01-0.06 per 1K                           │
│  ├── Output tokens: $0.03-0.12 per 1K                          │
│  └── Reasoning models: $0.06-0.60 per 1K                       │
│                                                                 │
│  Tool Execution (10-20%)                                        │
│  ├── API calls to external services                            │
│  ├── Database queries                                          │
│  └── Compute for code execution                                │
│                                                                 │
│  Infrastructure (10-20%)                                        │
│  ├── Hosting, storage, networking                              │
│  └── Monitoring and logging                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Cost Tracking System

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import json

@dataclass
class CostEvent:
    """Record a cost-incurring event."""
    timestamp: datetime
    event_type: str  # 'llm_call', 'tool_call', 'embedding'
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    request_id: str = ""
    metadata: Dict = field(default_factory=dict)


class CostTracker:
    """Track and analyze agent costs."""
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        'claude-3-opus': {'input': 0.015, 'output': 0.075},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
        'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
    }
    
    def __init__(self):
        self.events: List[CostEvent] = []
        self.budgets: Dict[str, float] = {}
    
    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_id: str = ""
    ) -> CostEvent:
        """Record an LLM API call."""
        pricing = self.PRICING.get(model, {'input': 0.01, 'output': 0.03})
        
        cost = (
            (input_tokens / 1000) * pricing['input'] +
            (output_tokens / 1000) * pricing['output']
        )
        
        event = CostEvent(
            timestamp=datetime.now(),
            event_type='llm_call',
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            request_id=request_id
        )
        
        self.events.append(event)
        self._check_budget(cost)
        
        return event
    
    def set_budget(self, period: str, amount: float):
        """Set budget limit."""
        self.budgets[period] = amount
    
    def get_usage(self, hours: int = 24) -> Dict:
        """Get usage statistics."""
        cutoff = datetime.now().timestamp() - hours * 3600
        recent = [e for e in self.events if e.timestamp.timestamp() > cutoff]
        
        return {
            'total_cost': sum(e.cost_usd for e in recent),
            'total_input_tokens': sum(e.input_tokens for e in recent),
            'total_output_tokens': sum(e.output_tokens for e in recent),
            'request_count': len(recent),
            'by_model': self._group_by_model(recent)
        }
    
    def _group_by_model(self, events):
        result = {}
        for e in events:
            if e.model not in result:
                result[e.model] = {'cost': 0, 'calls': 0}
            result[e.model]['cost'] += e.cost_usd
            result[e.model]['calls'] += 1
        return result
    
    def _check_budget(self, cost):
        usage = self.get_usage(24)
        if 'daily' in self.budgets:
            if usage['total_cost'] > self.budgets['daily']:
                raise BudgetExceededError(
                    f"Daily budget exceeded: ${usage['total_cost']:.2f} > ${self.budgets['daily']:.2f}"
                )


class BudgetExceededError(Exception):
    pass
```

## 3. Cost Optimization Strategies

### 3.1 Model Selection

```python
class ModelRouter:
    """Route to appropriate model based on task complexity."""
    
    def __init__(self, cost_tracker: CostTracker):
        self.tracker = cost_tracker
    
    def select_model(self, query: str, context: Dict = None) -> str:
        """Choose model based on query complexity."""
        complexity = self._estimate_complexity(query, context)
        
        if complexity < 0.3:
            return 'gpt-3.5-turbo'  # Simple queries: $0.002/1K
        elif complexity < 0.7:
            return 'claude-3-sonnet'  # Medium: $0.009/1K
        else:
            return 'gpt-4-turbo'  # Complex: $0.02/1K
    
    def _estimate_complexity(self, query: str, context: Dict) -> float:
        """Estimate query complexity 0-1."""
        score = 0.0
        
        # Length-based
        if len(query) > 500:
            score += 0.2
        
        # Keyword-based
        complex_keywords = ['analyze', 'compare', 'explain why', 'code', 'debug']
        if any(kw in query.lower() for kw in complex_keywords):
            score += 0.3
        
        # Context-based
        if context and context.get('requires_reasoning'):
            score += 0.3
        
        return min(score, 1.0)
    
    def estimate_savings(self, queries: List[str]) -> Dict:
        """Estimate savings from routing vs always using GPT-4."""
        baseline_cost = 0
        routed_cost = 0
        
        for query in queries:
            # Baseline: always GPT-4
            baseline_cost += 0.02  # Rough per-query cost
            
            # Routed
            model = self.select_model(query)
            if model == 'gpt-3.5-turbo':
                routed_cost += 0.002
            elif model == 'claude-3-sonnet':
                routed_cost += 0.009
            else:
                routed_cost += 0.02
        
        return {
            'baseline': baseline_cost,
            'routed': routed_cost,
            'savings': baseline_cost - routed_cost,
            'savings_pct': (baseline_cost - routed_cost) / baseline_cost * 100
        }
```

### 3.2 Caching

```python
import hashlib
from typing import Optional, Tuple

class ResponseCache:
    """Cache LLM responses to avoid redundant calls."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[str, float, int]] = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def get(self, query: str, context: str = "") -> Optional[str]:
        """Get cached response if available."""
        key = self._make_key(query, context)
        
        if key in self.cache:
            response, timestamp, _ = self.cache[key]
            if datetime.now().timestamp() - timestamp < self.ttl:
                self.hits += 1
                return response
            else:
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, query: str, context: str, response: str, tokens_saved: int):
        """Cache a response."""
        if len(self.cache) >= self.max_size:
            self._evict()
        
        key = self._make_key(query, context)
        self.cache[key] = (response, datetime.now().timestamp(), tokens_saved)
    
    def _make_key(self, query: str, context: str) -> str:
        content = f"{query}|{context}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _evict(self):
        # Remove oldest 10%
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1][1]
        )
        for key, _ in sorted_items[:len(sorted_items)//10]:
            del self.cache[key]
    
    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        tokens_saved = sum(v[2] for v in self.cache.values() if v[2])
        
        return {
            'hit_rate': self.hits / total if total else 0,
            'cache_size': len(self.cache),
            'estimated_tokens_saved': tokens_saved * self.hits
        }
```

### 3.3 Request Batching

```python
import asyncio
from typing import List, Callable

class RequestBatcher:
    """Batch similar requests for efficiency."""
    
    def __init__(self, batch_size: int = 10, wait_ms: int = 100):
        self.batch_size = batch_size
        self.wait_ms = wait_ms
        self.pending: List[Tuple[str, asyncio.Future]] = []
        self.lock = asyncio.Lock()
    
    async def add_request(self, prompt: str) -> str:
        """Add request to batch, return result when processed."""
        future = asyncio.Future()
        
        async with self.lock:
            self.pending.append((prompt, future))
            
            if len(self.pending) >= self.batch_size:
                await self._process_batch()
        
        # Wait for result
        return await future
    
    async def _process_batch(self):
        """Process accumulated requests as a batch."""
        if not self.pending:
            return
        
        batch = self.pending[:self.batch_size]
        self.pending = self.pending[self.batch_size:]
        
        # Combine prompts
        prompts = [p for p, _ in batch]
        combined = "\n---\n".join(
            f"Request {i+1}: {p}" for i, p in enumerate(prompts)
        )
        
        # Single LLM call for batch
        # Cost: 1 call instead of N calls
        response = await self._call_llm(combined)
        
        # Parse and distribute results
        results = self._parse_batch_response(response, len(batch))
        for (_, future), result in zip(batch, results):
            future.set_result(result)
    
    async def _call_llm(self, prompt: str) -> str:
        # Actual LLM call
        pass
    
    def _parse_batch_response(self, response: str, count: int) -> List[str]:
        # Parse batched response into individual results
        pass
```

## 4. Budget Enforcement

```python
class BudgetEnforcer:
    """Enforce cost limits and quotas."""
    
    def __init__(self, tracker: CostTracker):
        self.tracker = tracker
        self.limits = {}
        self.alerts = []
    
    def set_limits(
        self,
        daily_limit: float = None,
        per_request_limit: float = None,
        hourly_limit: float = None
    ):
        """Set spending limits."""
        if daily_limit:
            self.limits['daily'] = daily_limit
        if per_request_limit:
            self.limits['per_request'] = per_request_limit
        if hourly_limit:
            self.limits['hourly'] = hourly_limit
    
    def check_before_request(self, estimated_cost: float) -> bool:
        """Check if request should proceed."""
        # Per-request limit
        if 'per_request' in self.limits:
            if estimated_cost > self.limits['per_request']:
                return False
        
        # Hourly limit
        if 'hourly' in self.limits:
            usage = self.tracker.get_usage(1)
            if usage['total_cost'] + estimated_cost > self.limits['hourly']:
                return False
        
        # Daily limit
        if 'daily' in self.limits:
            usage = self.tracker.get_usage(24)
            if usage['total_cost'] + estimated_cost > self.limits['daily']:
                return False
        
        return True
    
    def get_remaining_budget(self) -> Dict:
        """Get remaining budget for each period."""
        usage_hourly = self.tracker.get_usage(1)
        usage_daily = self.tracker.get_usage(24)
        
        return {
            'hourly': self.limits.get('hourly', float('inf')) - usage_hourly['total_cost'],
            'daily': self.limits.get('daily', float('inf')) - usage_daily['total_cost']
        }
    
    def downgrade_strategy(self) -> str:
        """Suggest model downgrade when approaching limits."""
        remaining = self.get_remaining_budget()
        
        if remaining['daily'] < 10:
            return 'gpt-3.5-turbo'  # Cheapest
        elif remaining['daily'] < 50:
            return 'claude-3-haiku'  # Very cheap
        elif remaining['daily'] < 100:
            return 'claude-3-sonnet'  # Moderate
        else:
            return 'gpt-4-turbo'  # Full power
```

## 5. Cost Reporting

```python
class CostReporter:
    """Generate cost reports and insights."""
    
    def __init__(self, tracker: CostTracker):
        self.tracker = tracker
    
    def daily_report(self) -> str:
        """Generate daily cost report."""
        usage = self.tracker.get_usage(24)
        
        report = f"""
╔══════════════════════════════════════════════════════╗
║              Daily Cost Report                        ║
╠══════════════════════════════════════════════════════╣
║ Total Cost:        ${usage['total_cost']:>10.2f}              ║
║ Total Requests:    {usage['request_count']:>10,}              ║
║ Input Tokens:      {usage['total_input_tokens']:>10,}              ║
║ Output Tokens:     {usage['total_output_tokens']:>10,}              ║
╠══════════════════════════════════════════════════════╣
║ Cost by Model:                                       ║"""
        
        for model, data in usage['by_model'].items():
            report += f"\n║   {model:20} ${data['cost']:>8.2f} ({data['calls']} calls)"
        
        report += "\n╚══════════════════════════════════════════════════════╝"
        
        return report
    
    def optimization_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations."""
        usage = self.tracker.get_usage(24)
        recommendations = []
        
        # Check model usage
        by_model = usage['by_model']
        if 'gpt-4' in by_model and by_model['gpt-4']['calls'] > 100:
            recommendations.append(
                "Consider routing simple queries to GPT-3.5 to save ~90% on those calls"
            )
        
        # Check token efficiency
        avg_tokens = (usage['total_input_tokens'] + usage['total_output_tokens']) / max(usage['request_count'], 1)
        if avg_tokens > 5000:
            recommendations.append(
                f"Average {avg_tokens:.0f} tokens/request is high. Consider prompt compression."
            )
        
        return recommendations
```

## 6. Connection to Model Serialization

Cost management relates to model serialization in key ways:
- **Smaller models = lower costs** (quantized models use less compute)
- **Edge deployment** reduces API costs entirely
- **Caching serialized responses** avoids recomputation

Both focus on **efficiency** at different levels of the stack.

## 7. Key Takeaways

1. **Track everything**: Every token, every call, every dollar
2. **Route intelligently**: Use expensive models only when needed
3. **Cache aggressively**: Similar queries shouldn't cost twice
4. **Set hard limits**: Budget enforcement prevents runaway costs
5. **Report and optimize**: Continuous monitoring reveals savings opportunities

---

**Originally published at:** [arunbaby.com/ai-agents/0047-cost-management-for-agents](https://www.arunbaby.com/ai-agents/0047-cost-management-for-agents/)

*If you found this helpful, consider sharing it with others who might benefit.*
