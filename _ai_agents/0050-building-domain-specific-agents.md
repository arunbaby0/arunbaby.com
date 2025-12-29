---
title: "Building Domain-Specific Agents"
day: 50
related_dsa_day: 50
related_ml_day: 50
related_speech_day: 50
collection: ai_agents
categories:
 - ai-agents
tags:
 - vertical-ai
 - rag
 - fine-tuning
 - tools
 - agent-design
difficulty: Hard
subdomain: "Agent Application"
tech_stack: OpenAI Assistants API, Pinecone, LangChain
scale: "High-precision, high-stakes domains"
companies: Harvey.ai (Legal), Hippocratic AI (Medical), GitHub (Coding)
---

**"Don't build a generalist. Build a specialist."**

## 1. Introduction

Generic agents (ChatGPT) are "Jack of all trades, master of none".
- Ask them to write a poem? Great.
- Ask them to audit a Series A Term Sheet for "Liquidation Preference"? Dangerously vague.

**Domain-Specific Agents** are the future of SaaS. These agents are constrained to a "Vertical" (Law, Medicine, Finance, Coding).
They trade **Breadth** for **Depth** and **Reliability**.

---

## 2. Core Concepts: The Vertical Stack

To build a "Harvey.ai" (Legal Agent) or "Devin" (Coding Agent), you need 4 layers of specialization:

1. **Data Layer (RAG)**: Access to proprietary, non-public documents (e.g., 10k Court Cases).
2. **Tool Layer**: Access to specialized APIs (e.g., Westlaw Search, Terminal Access).
3. **Model Layer (Fine-Tuning)**: Knowing the syntax and "vibe" of the domain.
4. **Guardrail Layer**: Strict rules ("Never give financial advice").

---

## 3. Architecture Patterns: RAG vs Fine-Tuning

The biggest debate is: **"Do I fine-tune Llama-3 or just use RAG?"**

| Feature | RAG (Retrieval) | Fine-Tuning (Training) | Hybrid (Best) |
|---------|-----------------|------------------------|---------------|
| **Knowledge Source** | Dynamic (DB lookup) | Static (Weights) | Both |
| **Hallucination** | Low (Groundable) | Medium | Lowest |
| **Adaptability** | Instant (Update DB) | Slow (Retrain) | Mixed |
| **Style/Tone** | Weak | Strong | Strong |

**Design Choice**:
- Use **RAG** for *Facts* ("What is the termination clause?").
- Use **Fine-Tuning** for *Format/Behavior* ("Write a memo in the style of a Senior Partner").

---

## 4. Implementation Approaches

### 4.1 The Medical Scribe Pattern
1. **Input**: Doctor-Patient Audio.
2. **Specialized ASR**: (Speech).
3. **Extraction Agent**: Extracts "Symptoms", "Meds".
4. **Coding Agent**: Maps Symptoms -> ICD-10 Codes.
 - *Tool*: `lookup_icd10(query)`.
 - *Validation*: "Is 'Chest Pain' valid for a 5yo?"
5. **Output**: EMR Entry.

---

## 5. Code Examples: The Hybrid Agent

A simplified implementation using LangChain with a specialized tool and system prompt.

``python
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.prompts import SystemMessagePromptTemplate

# 1. Specialized Tool (The "Hands")
def search_case_law(query):
 """Searches the LexisNexis database for precedent."""
 # Mock return
 return "Case 12-404: Smith v. Jones established that..."

tools = [
 Tool(
 name="CaseLawSearch",
 func=search_case_law,
 description="Useful for finding legal precedents."
 )
]

# 2. Specialized Persona (The "Brain")
# This is where "Prompt Engineering" meets "Domain Expertise"
legal_prompt = """
You remain in character as a Senior Litigation Associate.
- Never invent cases.
- Always cite the Case ID.
- Use 'IRAC' format (Issue, Rule, Application, Conclusion).
- If uncertain, state "Further research required".
"""

# 3. The Agent
llm = ChatOpenAI(model="gpt-4", temperature=0) # Temp 0 for rigor
agent = initialize_agent(
 tools, 
 llm, 
 agent="zero-shot-react-description",
 agent_kwargs={
 "system_message": SystemMessagePromptTemplate.from_template(legal_prompt)
 }
)

# 4. Execution
agent.run("My landlord evicted me without notice. What are my rights in NY?")
``

---

## 6. Production Considerations

### 6.1 Evaluation (The "Bar Exam")
You cannot eval a Legal Agent with "Vibe Check".
You need a **Golden Dataset**:
- 100 real legal questions.
- 100 answers written by human lawyers.
- **Metric**: "Fact Recall", "Citation Accuracy".
- **Auto-Eval**: Use GPT-4 to grade the Agent's answer against the Human answer.

### 6.2 Liability Firewall
The Agent output should never go directly to the client.
It goes to a **Human in the Loop**.
UI Pattern: "Draft generated. **[Approve]** / **[Edit]**".

---

## 7. Common Pitfalls

1. **Over-retrieval**: Feeding the agent 50 pages of irrelevant case law confuses it.
 - *Fix*: Ranking/Reranking in RAG pipeline.
2. **Tone mismatch**: A medical agent sounding "excited" about a diagnosis.
 - *Fix*: Fine-tuning on medical datasets reduces the "Customer Service Voice" of base models.

---

## 8. Best Practices

1. **Ontology First**: Define the domain vocabulary. What exactly does "Churn" mean for *this* company?
2. **Few-Shot Prompting**: Include 5 examples of "Perfect Answers" in the prompt. This guides the style better than instructions.

---

## 9. Connections to Other Topics

This connects to **Alien Dictionary** (DSA ).
- To build a specialized agent, you must first learn the "Alien language" (Jargon) of the domain.
- You build the "Graph" of knowledge (Ontology) before you can traverse it.

---

## 10. Real-World Examples

- **GitHub Copilot**: Fine-tuned on code. It knows that `def` is likely followed by a function name. It formats indentation perfectly.
- **Harvey.ai**: Partnered with OpenAI to build a custom model trained on legal corpuses, specifically for Big Law firms.

---

## 11. Future Directions

- **Self-Improving Verticals**: The agent drafts a contract. The lawyer edits it. The edit is saved. The model is fine-tuned on the edit. The agent gets smarter every day (Data Flywheel).

---

## 12. Key Takeaways

1. **Depth over Breadth**: A tool that does one thing perfectly is valuable. A tool that does everything okay is a toy.
2. **Hybrid Architecture**: RAG + Fine-Tuning + Tools. You usually need all three.
3. **Eval is hard**: Building the "Bar Exam" for your agent is as hard as building the agent itself.

---

**Originally published at:** [arunbaby.com/ai-agents/0050-building-domain-specific-agents](https://www.arunbaby.com/ai-agents/0050-building-domain-specific-agents/)

*If you found this helpful, consider sharing it with others who might benefit.*
