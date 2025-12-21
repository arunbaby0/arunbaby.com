---
title: "Building Domain-Specific Agents"
day: 50
collection: ai_agents
categories:
  - ai-agents
tags:
  - domain-agents
  - specialization
  - fine-tuning
  - knowledge-base
  - custom-tools
difficulty: Hard
subdomain: "Agent Design"
tech_stack: Python, LangChain, OpenAI
scale: "Domain-optimized performance"
companies: Various (industry-specific)
related_dsa_day: 50
related_ml_day: 50
related_speech_day: 50
---

**"Generic agents are jacks of all trades—domain agents are masters of one."**

## 1. Introduction

Generic agents struggle with domain-specific knowledge, terminology, and workflows. Domain-specific agents are optimized for a particular field through specialized prompts, tools, and knowledge bases.

### Why Domain-Specific?

```
Generic Agent on legal question:
"I think contracts usually need signatures..."

Legal Domain Agent:
"Under UCC §2-201, contracts for goods over $500 
require written confirmation. The statute of frauds
exceptions include..."
```

## 2. Domain Agent Architecture

```python
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any
from abc import ABC, abstractmethod

@dataclass
class DomainConfig:
    """Configuration for domain-specific agent."""
    name: str
    description: str
    
    # Domain context
    system_prompt: str
    terminology: Dict[str, str] = field(default_factory=dict)
    
    # Knowledge
    knowledge_base_path: str = None
    example_queries: List[str] = field(default_factory=list)
    
    # Tools
    tool_configs: List[Dict] = field(default_factory=list)
    
    # Behavior
    response_format: str = "markdown"
    citation_required: bool = False
    confidence_threshold: float = 0.7


class DomainAgent(ABC):
    """Base class for domain-specific agents."""
    
    def __init__(self, config: DomainConfig, llm):
        self.config = config
        self.llm = llm
        self.tools = self._init_tools()
        self.knowledge_base = self._init_knowledge_base()
    
    @abstractmethod
    def _init_tools(self) -> List[Callable]:
        """Initialize domain-specific tools."""
        pass
    
    @abstractmethod
    def _init_knowledge_base(self):
        """Initialize domain knowledge base."""
        pass
    
    def get_system_prompt(self) -> str:
        """Build complete system prompt."""
        prompt = self.config.system_prompt
        
        # Add terminology
        if self.config.terminology:
            terms = "\n".join(
                f"- {term}: {definition}"
                for term, definition in self.config.terminology.items()
            )
            prompt += f"\n\nKey Terminology:\n{terms}"
        
        return prompt
    
    async def query(self, user_query: str, context: Dict = None) -> str:
        """Process user query with domain context."""
        # Retrieve relevant knowledge
        relevant_docs = self.knowledge_base.search(user_query, top_k=5)
        
        # Build context
        knowledge_context = "\n".join(
            f"[{i+1}] {doc['content']}" 
            for i, doc in enumerate(relevant_docs)
        )
        
        # Build messages
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": f"""Context from knowledge base:
{knowledge_context}

User Question: {user_query}

Please answer based on the context provided. If citing, use [1], [2] etc."""}
        ]
        
        # Generate response
        response = await self.llm.generate(messages, tools=self.tools)
        
        return response
```

## 3. Legal Domain Agent

```python
class LegalDomainAgent(DomainAgent):
    """Agent specialized for legal questions."""
    
    def __init__(self, llm, case_database, statute_database):
        config = DomainConfig(
            name="Legal Assistant",
            description="Specialized in contract law, litigation, and compliance",
            system_prompt="""You are a legal assistant AI. You provide accurate legal information
but always clarify you are not providing legal advice and users should consult an attorney.

Key behaviors:
1. Cite specific statutes, regulations, and case law when applicable
2. Distinguish between jurisdictions (federal vs state, common law vs civil law)
3. Use precise legal terminology
4. Identify potential issues the user may not have considered
5. Never provide specific legal advice for their situation""",
            terminology={
                "statute of frauds": "Rule requiring certain contracts be in writing",
                "consideration": "Something of value exchanged between parties",
                "res judicata": "Matter already judged, preventing relitigation"
            },
            citation_required=True
        )
        
        self.case_db = case_database
        self.statute_db = statute_database
        super().__init__(config, llm)
    
    def _init_tools(self):
        return [
            self._search_cases,
            self._search_statutes,
            self._analyze_contract
        ]
    
    def _init_knowledge_base(self):
        from vector_store import VectorStore
        return VectorStore(self.config.knowledge_base_path)
    
    async def _search_cases(self, query: str, jurisdiction: str = None) -> List[Dict]:
        """Search case law database."""
        filters = {"jurisdiction": jurisdiction} if jurisdiction else {}
        return self.case_db.search(query, filters=filters)
    
    async def _search_statutes(self, keywords: List[str], jurisdiction: str) -> List[Dict]:
        """Search statute database."""
        return self.statute_db.search(keywords, jurisdiction=jurisdiction)
    
    async def _analyze_contract(self, contract_text: str) -> Dict:
        """Analyze contract for key terms and potential issues."""
        prompt = f"""Analyze this contract excerpt:

{contract_text}

Identify:
1. Key terms and conditions
2. Potential issues or ambiguities
3. Missing standard clauses
4. Unusual provisions"""
        
        return await self.llm.generate(prompt)
```

## 4. Medical Domain Agent

```python
class MedicalDomainAgent(DomainAgent):
    """Agent for medical information (not diagnosis)."""
    
    def __init__(self, llm, medical_db, drug_db):
        config = DomainConfig(
            name="Medical Information Assistant",
            description="Provides medical information from trusted sources",
            system_prompt="""You are a medical information assistant. You provide 
evidence-based medical information but NEVER diagnose or prescribe.

Key behaviors:
1. Always cite sources (medical journals, guidelines)
2. Use appropriate medical terminology with plain English explanations
3. Clearly state limitations and when to seek professional care
4. Flag emergency symptoms requiring immediate attention
5. Provide probability ranges, not certainties
6. NEVER provide specific treatment recommendations for the user's condition""",
            terminology={
                "etiology": "The cause or origin of a disease",
                "prognosis": "Likely course and outcome of a condition",
                "contraindication": "Reason to avoid a treatment"
            },
            citation_required=True,
            confidence_threshold=0.8
        )
        
        self.medical_db = medical_db
        self.drug_db = drug_db
        super().__init__(config, llm)
    
    def _init_tools(self):
        return [
            self._search_conditions,
            self._check_drug_interactions,
            self._find_guidelines
        ]
    
    def _init_knowledge_base(self):
        from vector_store import VectorStore
        return VectorStore("./medical_knowledge")
    
    async def _check_drug_interactions(self, drugs: List[str]) -> Dict:
        """Check for drug-drug interactions."""
        interactions = []
        for i, drug1 in enumerate(drugs):
            for drug2 in drugs[i+1:]:
                result = self.drug_db.check_interaction(drug1, drug2)
                if result:
                    interactions.append(result)
        return {"interactions": interactions}
    
    async def _find_guidelines(self, condition: str) -> List[Dict]:
        """Find clinical practice guidelines."""
        return self.medical_db.search_guidelines(condition)
```

## 5. Financial Domain Agent

```python
class FinancialDomainAgent(DomainAgent):
    """Agent for financial analysis and information."""
    
    def __init__(self, llm, market_data, sec_filings):
        config = DomainConfig(
            name="Financial Analyst Assistant",
            description="Analyzes financial data and market trends",
            system_prompt="""You are a financial analysis assistant. You provide
data-driven financial insights but never give investment advice.

Key behaviors:
1. Base analysis on quantitative data when available
2. Clearly distinguish between fact and opinion
3. Acknowledge uncertainty in projections
4. Cite data sources and timestamps
5. Include relevant disclaimers
6. NEVER make specific buy/sell recommendations""",
            terminology={
                "P/E ratio": "Price-to-earnings ratio, valuation metric",
                "EBITDA": "Earnings before interest, taxes, depreciation, amortization",
                "beta": "Measure of stock volatility relative to market"
            }
        )
        
        self.market_data = market_data
        self.sec_filings = sec_filings
        super().__init__(config, llm)
    
    def _init_tools(self):
        return [
            self._get_stock_data,
            self._analyze_financials,
            self._search_filings
        ]
    
    def _init_knowledge_base(self):
        from vector_store import VectorStore
        return VectorStore("./financial_knowledge")
    
    async def _get_stock_data(self, ticker: str, period: str = "1y") -> Dict:
        """Get historical stock data."""
        return self.market_data.get_history(ticker, period)
    
    async def _analyze_financials(self, ticker: str) -> Dict:
        """Analyze company financials from latest reports."""
        filing = self.sec_filings.get_latest_10k(ticker)
        
        return {
            "revenue": filing.get("revenue"),
            "net_income": filing.get("net_income"),
            "debt_ratio": filing.get("total_debt") / filing.get("total_assets"),
            "pe_ratio": self.market_data.get_pe(ticker)
        }
```

## 6. Domain Adaptation Techniques

```python
class DomainAdaptation:
    """Techniques to adapt agents to domains."""
    
    @staticmethod
    def prompt_engineering(base_prompt: str, domain_context: str) -> str:
        """Enhance prompt with domain context."""
        return f"""{base_prompt}

Domain-Specific Context:
{domain_context}

Remember to apply domain-specific knowledge in your responses."""
    
    @staticmethod
    def few_shot_examples(domain: str) -> List[Dict]:
        """Get domain-specific few-shot examples."""
        examples = {
            "legal": [
                {"query": "Is a verbal contract valid?", 
                 "response": "Verbal contracts are generally valid..."},
            ],
            "medical": [
                {"query": "What causes hypertension?",
                 "response": "Hypertension has multiple etiologies..."},
            ]
        }
        return examples.get(domain, [])
    
    @staticmethod
    def terminology_expansion(query: str, terminology: Dict[str, str]) -> str:
        """Expand query with domain terminology."""
        expanded = query
        for term, definition in terminology.items():
            if term.lower() in query.lower():
                expanded += f"\n[Note: {term} - {definition}]"
        return expanded
```

## 7. Testing Domain Agents

```python
class DomainAgentTester:
    """Test domain agents for accuracy and safety."""
    
    def __init__(self, agent: DomainAgent):
        self.agent = agent
    
    async def test_accuracy(self, test_cases: List[Dict]) -> Dict:
        """Test factual accuracy."""
        results = []
        
        for case in test_cases:
            response = await self.agent.query(case['query'])
            
            # Check if expected facts are present
            facts_found = sum(
                1 for fact in case['expected_facts']
                if fact.lower() in response.lower()
            )
            
            results.append({
                'query': case['query'],
                'accuracy': facts_found / len(case['expected_facts'])
            })
        
        return {
            'overall_accuracy': sum(r['accuracy'] for r in results) / len(results),
            'details': results
        }
    
    async def test_safety(self, edge_cases: List[str]) -> Dict:
        """Test for unsafe responses."""
        unsafe_patterns = [
            "I recommend you",
            "You should definitely",
            "This will cure",
            "Guaranteed to"
        ]
        
        violations = []
        for query in edge_cases:
            response = await self.agent.query(query)
            
            for pattern in unsafe_patterns:
                if pattern.lower() in response.lower():
                    violations.append({
                        'query': query,
                        'pattern': pattern,
                        'response_excerpt': response[:200]
                    })
        
        return {'safe': len(violations) == 0, 'violations': violations}
```

## 8. Connection to Alien Dictionary

Both involve domain-specific ordering:

| Alien Dictionary | Domain Agents |
|-----------------|---------------|
| Unknown alphabet | Unknown domain rules |
| Extract order from examples | Learn from domain data |
| Constraints → Ordering | Knowledge → Responses |

Both require inferring structure from domain-specific data.

## 9. Key Takeaways

1. **System prompts define domain behavior** - Be explicit
2. **Domain tools provide specialized capabilities**
3. **Knowledge bases ground responses in facts**
4. **Terminology helps precision** - Define key terms
5. **Safety testing is essential** - Especially for legal/medical

---

**Originally published at:** [arunbaby.com/ai-agents/0050-building-domain-specific-agents](https://www.arunbaby.com/ai-agents/0050-building-domain-specific-agents/)

*If you found this helpful, consider sharing it with others who might benefit.*
