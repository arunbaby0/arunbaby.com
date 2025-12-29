---
title: "Knowledge Graphs for Agents"
day: 51
related_dsa_day: 51
related_ml_day: 51
related_speech_day: 51
collection: ai_agents
categories:
 - ai-agents
tags:
 - knowledge-graphs
 - rag
 - entity-linking
 - graph-databases
 - reasoning
 - retrieval
difficulty: Hard
subdomain: "Structured Knowledge"
tech_stack: Neo4j, Amazon Neptune, NetworkX, Python
scale: "1B+ edges, low-latency retrieval, continuous updates"
companies: Google, Meta, Amazon, Microsoft
---

**"RAG gives you documents. A knowledge graph gives you *facts with structure*—and agents need structure to act reliably."**

## 1. Introduction

Agents fail in predictable ways:
- they forget constraints (“this vendor is not approved”)
- they confuse entities (“Apple” company vs fruit)
- they hallucinate relationships (“X acquired Y”) because the text is ambiguous
- they can’t maintain long, multi-hop consistency (“A depends on B depends on C”)

A **Knowledge Graph (KG)** helps because it turns unstructured knowledge into:
- entities (nodes)
- relationships (edges)
- types and constraints (schema)
- queryable, auditable facts

This matters for production agents because:
- it reduces hallucinations by grounding on explicit facts
- it enables multi-hop reasoning with bounded search
- it supports governance (who asserted this fact? when? from which source?)

Thematic link for today is **binary search and distributed algorithms**:
- graph retrieval at scale is a distributed systems problem
- many KG systems rely on indexing and boundary-based retrieval (not full scans)
- agents often combine “search in text” with “search in graph”, choosing boundaries (top-k, hop limits, trust thresholds) the same way we search for a partition in algorithmic problems

---

## 2. Core Concepts

### 2.1 What a knowledge graph is (and isn’t)

A KG is a directed labeled graph:
- nodes: entities (people, products, tickets, services, policies)
- edges: relations (OWNS, DEPENDS_ON, WORKS_AT, LOCATED_IN)
- properties: metadata (timestamps, confidence, provenance)

It is **not**:
- a single “database of truth” automatically correct
- a replacement for documents (you still need unstructured sources)
- a magic reasoner (you still need inference logic)

### 2.2 Triples and provenance

The simplest representation:
- `(subject, predicate, object)`

Example:
- `(ServiceA, DEPENDS_ON, ServiceB)`

In production you also store:
- `source_doc_id`
- `extraction_model_version`
- `confidence`
- `created_at`
- `valid_from/valid_to`

This turns the KG into something auditable and maintainable.

### 2.3 Schema matters (constraints reduce hallucinations)

Schema examples:
- `DEPENDS_ON` must connect `Service -> Service`
- `HAS_OWNER` must connect `Service -> Team`
- `HAS_SLA` must connect `Service -> SLA`

An agent can use schema to:
- validate candidate actions
- ask follow-up questions when required fields are missing

### 2.4 RDF vs Property Graph (two common worlds)

Knowledge graphs show up in two major representations:

**RDF (Resource Description Framework)**
- Data model: triples `(s, p, o)`
- Query language: SPARQL
- Great for: semantic web, ontologies, standardized vocabularies
- Strong on: formal semantics and interoperability

**Property graph**
- Data model: nodes and edges with properties
- Query languages: Cypher (Neo4j), Gremlin (TinkerPop), openCypher variants
- Great for: operational systems, flexible properties, developer ergonomics
- Strong on: traversal-style queries and performance patterns

For agents in product environments, property graphs are often chosen because:
- they map naturally to “entities with attributes”
- they’re easy to query for path/traversal workflows (“dependencies up to 3 hops”)

### 2.5 Ontologies: constraints beyond the graph

An ontology is a schema with semantics:
- classes (Team, Service, Vendor)
- relations (DEPENDS_ON, OWNED_BY)
- constraints (domains, ranges, cardinality)

Agents benefit from ontology constraints because:
- they reduce ambiguity (“OWNED_BY must point to Team”)
- they enable validation (“this triple is invalid”)
- they guide question asking (“owner missing; ask who owns this service”)

In practice, you don’t need a perfect ontology.
You need a **useful** ontology:
- small number of types
- clear ownership semantics
- a handful of relations that power your top agent workflows

### 2.6 Query languages and tool contracts

Agents should not be allowed to emit arbitrary graph queries in production.
Instead, you define a tool contract:
- input schema (validated)
- query templates (safe, indexed)
- output schema (typed)

Why this matters:
- prevents full scans (“MATCH (n) RETURN n”)
- prevents injection attacks in query strings
- makes caching possible (same template + args)

Example contracts:
- `get_owner(service_id) -> {team_id, team_name}`
- `get_dependencies(service_id, depth=2) -> {nodes, edges}`
- `policy_check(action, entity_id) -> {allowed: bool, rationale, citations}`

### 2.7 The “boundary” mindset in graph retrieval

Large graphs are not explored by “looking at everything”.
They are explored by **boundaries**:
- hop limit (depth 2 vs depth 10)
- top-k neighbors (ranked by confidence/trust)
- time window (facts valid “now” only)
- trust threshold (only facts with provenance from approved sources)

This is the same engineering instinct as the median partition algorithm:
don’t merge the universe; find the boundary that answers the question reliably and cheaply.

---

## 3. Architecture Patterns for Agents

### 3.1 RAG-only vs KG-only vs Hybrid

| Pattern | Strength | Weakness | Best for |
|--------|----------|----------|----------|
| RAG-only | flexible, easy ingestion | ambiguity, consistency issues | Q&A over docs |
| KG-only | precise facts, multi-hop | extraction effort, coverage gaps | policies, dependencies, catalogs |
| Hybrid (RAG + KG) | best of both | more system complexity | real production agents |

Typical production approach:
- RAG for **explanations** and long-form context
- KG for **constraints** and actionable facts

### 3.2 “KG as a tool” inside an agent

Agents should not “think” the KG; they should **query** it.
That means:
- a KG query tool (Cypher, SPARQL, Gremlin, or API)
- typed outputs (structured JSON)
- guardrails to prevent expensive scans

Example tool call:
- `get_service_dependencies(service_id, depth=2)`
- `find_owner(service_id)`
- `validate_policy(action, entity)`

This aligns with tool design principles: keep tools narrow, predictable, and observable.

---

## 4. Implementation Approaches

### 4.1 Building the KG: extraction pipeline

Source types:
- wikis, runbooks, RFCs
- ticket systems
- code repositories (IaC, service configs)
- monitoring catalogs

Pipeline:
1. ingest documents
2. chunk + embed (for RAG)
3. entity extraction + linking (NER + disambiguation)
4. relation extraction (triples)
5. validation against schema
6. write to KG store
7. attach provenance

The hard part is **entity resolution**:
“Payments” might refer to:
- a team name
- a service
- a product feature
and agents must not mix them.

### 4.1.1 Continuous updates (the KG is a living system)

In real orgs, the graph changes constantly:
- services are renamed
- ownership rotates
- dependencies change weekly

So you need a continuous ingestion model:
- event-driven updates from source-of-truth systems (Terraform, CMDB, on-call rota)
- periodic refresh from docs (wikis, runbooks)
- reconciliation jobs that detect inconsistencies (“service exists in code but not in KG”)

This turns KG construction into an MLOps/DataOps problem:
- versioning
- backfills
- rollbacks
- schema migrations

### 4.1.2 Verification layers (how you prevent “LLM-generated graph garbage”)

If you let an LLM freely extract triples, you will eventually get plausible nonsense.
Strong production pipelines add verification:

- **Schema validation**: domain/range checks, required fields.
- **Cross-source confirmation**: accept a triple only if it appears in 2+ sources or a trusted source.
- **Confidence thresholds**: require higher confidence for high-impact relations (OWNED_BY, APPROVED_VENDOR).
- **Human review queues**: route uncertain triples to subject-matter owners.

This is analogous to guardrails in agent tool calls:
structured knowledge is powerful, but wrong structured knowledge is dangerously persuasive.

### 4.2 Entity linking strategies

- dictionary + aliases (fast, brittle)
- embedding similarity (robust, needs good vectors)
- hybrid: alias match → embedding rerank → schema checks

### 4.2.1 Hard cases in entity linking (and how to handle them)

Entity linking fails in predictable situations:
- acronyms (“PS” could be “Payments Service” or “PostScript”)
- renamed entities (“Project Apollo” becomes “Project Helios”)
- overloaded words (“Mercury” planet vs internal tool)

Practical mitigations:
- maintain alias tables with ownership (teams own their aliases)
- store “effective time” for names (valid_from/valid_to)
- use context windows (doc section headers, repo paths, service namespaces)

### 4.3 Graph reasoning strategies for agents

Agents often need:
- multi-hop traversal (“if ServiceA depends on ServiceB, who owns B?”)
- path finding (“what’s the dependency path from X to Y?”)
- constraint checking (“is this vendor approved?”)

This is where classic DSA shows up: you’re basically running BFS/DFS over a constrained subgraph.

### 4.3.1 Query planning: from user question to graph query

A production agent should not “guess a Cypher query”.
It should follow a query planning pattern:

1. **Intent classification**: is the user asking for ownership, dependencies, policy checks, or explanations?
2. **Entity extraction**: which entities are mentioned (service names, vendors, teams)?
3. **Disambiguation**: map names to IDs using entity linking.
4. **Query template selection**: choose a safe, indexed query template.
5. **Bounded execution**: set hop limits, time windows, trust thresholds.
6. **Synthesis**: generate a response that explicitly separates facts (from KG) from narrative guidance (from docs).

If you implement this pipeline, your agent becomes predictable and debuggable.
If you skip it, you’ll get “demo magic” that fails under load.

---

## 5. Code Examples (Minimal KG + Agent-Style Retrieval)

This example builds a tiny KG in-memory and shows how an agent might use it as a tool.

``python
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set


@dataclass(frozen=True)
class Edge:
 src: str
 rel: str
 dst: str


class SimpleKG:
 def __init__(self) -> None:
 self.adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list) # node -> [(rel, neighbor)]
 self.edges: Set[Edge] = set()

 def add_edge(self, src: str, rel: str, dst: str) -> None:
 e = Edge(src, rel, dst)
 if e in self.edges:
 return
 self.edges.add(e)
 self.adj[src].append((rel, dst))

 def neighbors(self, node: str, rel: str | None = None) -> List[str]:
 out = []
 for r, dst in self.adj.get(node, []):
 if rel is None or rel == r:
 out.append(dst)
 return out

 def bfs_paths(self, start: str, goal: str, max_hops: int = 4) -> List[List[str]]:
 """
 Find paths from start to goal up to max_hops.
 In production, you'd limit expansions, enforce schemas, and add caching.
 """
 paths = []
 q = deque([(start, [start])])
 while q:
 node, path = q.popleft()
 if len(path) - 1 > max_hops:
 continue
 if node == goal:
 paths.append(path)
 continue
 for _, nxt in self.adj.get(node, []):
 if nxt in path:
 continue # avoid cycles
 q.append((nxt, path + [nxt]))
 return paths


# Example KG: services and ownership
kg = SimpleKG()
kg.add_edge("ServiceA", "DEPENDS_ON", "ServiceB")
kg.add_edge("ServiceB", "DEPENDS_ON", "ServiceC")
kg.add_edge("ServiceB", "OWNED_BY", "TeamPayments")
kg.add_edge("ServiceC", "OWNED_BY", "TeamInfra")


def tool_get_owner(service: str) -> str | None:
 owners = kg.neighbors(service, rel="OWNED_BY")
 return owners[0] if owners else None


def tool_dependency_owners(service: str, max_hops: int = 3) -> List[Tuple[str, str]]:
 """
 Returns (dependency_service, owner) pairs for all reachable dependencies.
 """
 seen = set([service])
 q = deque([(service, 0)])
 out = []
 while q:
 node, depth = q.popleft()
 if depth == max_hops:
 continue
 for dep in kg.neighbors(node, rel="DEPENDS_ON"):
 if dep in seen:
 continue
 seen.add(dep)
 out.append((dep, tool_get_owner(dep) or "UNKNOWN"))
 q.append((dep, depth + 1))
 return out


print(tool_dependency_owners("ServiceA"))
# [('ServiceB', 'TeamPayments'), ('ServiceC', 'TeamInfra')]
``

Why this example matters:
- agents want **bounded traversal** (max hops, no cycles)
- the “right” answer is often at a boundary: “search just deep enough”
- production KGs are massive, so you need indexes and distributed query engines

### 5.1 Example: KG-augmented RAG answer flow (how an agent should respond)

A strong production pattern is:
1. query KG for **constraints and identifiers**
2. retrieve docs via RAG for **explanations and runbooks**
3. generate an answer that cites both

Example question:
> “Who owns ServiceA and what should I do if it’s down?”

Agent flow:
- KG tool: `get_owner(ServiceA)` → `TeamPayments`
- KG tool: `get_dependencies(ServiceA, depth=2)` → `{ServiceB, ServiceC}`
- RAG tool: search runbooks for “ServiceA incident runbook” and “ServiceB incident runbook”

Then the agent produces:
- crisp ownership answer (from KG)
- dependency-aware debugging checklist (from KG + docs)
- citations/provenance pointers (runbook IDs, source docs)

This hybrid approach avoids a common failure mode:
RAG-only agents might give a good narrative, but miss the dependency chain and page the wrong team.

### 5.2 Tool design: return structured data, not paragraphs

KG tools should return JSON-like structures:
- stable IDs (service_id, team_id)
- typed relations
- confidence/provenance fields

If the tool returns prose, the agent will treat it like another document and hallucinate around it.
If the tool returns structured data, the agent can:
- validate fields
- branch logic (“if owner is UNKNOWN, ask follow-up”)
- cache responses

### 5.3 Evaluation: how you know the KG is helping

For agents, “it feels better” is not an evaluation.
You want task-level metrics such as:
- **constraint accuracy**: did the agent respect “approved vendor” lists?
- **entity resolution accuracy**: did it pick the right “Apple”?
- **routing accuracy**: did it page the correct on-call team?
- **multi-hop correctness**: did it preserve logical consistency across hops?

Typical evaluation design:
- create a gold set of questions with expected structured outputs
- run the agent with and without KG tools
- compare structured correctness + latency

KDs often improve correctness dramatically for constraint-heavy tasks, even if language fluency is unchanged.

---

## 6. Production Considerations

### 6.1 Latency and query safety
Agents are interactive.
KG queries must be:
- low-latency (p95 under a few hundred ms)
- protected from “full graph scans”

Techniques:
- precompute common neighborhoods (caching)
- enforce query budgets (node expansion limits)
- use typed query templates (prevent arbitrary Cypher injection)

### 6.2 Freshness and change management
Knowledge changes:
- ownership changes
- dependencies change (microservices evolve)
- policies update

So you need:
- incremental updates (CDC, stream ingestion)
- versioned facts (valid time)
- retractions and deprecations

### 6.2.1 Temporal graphs (answering “what was true then?”)

Agents frequently need temporal correctness:
- “Who owned this service during last month’s incident?”
- “Was vendor X approved when we signed the contract?”

If you only store the latest state, the agent will silently answer with *today’s truth* even when the question is about the past.
Two common designs:

- **Temporal edges**
 - store `valid_from` / `valid_to` on edges
 - query with a `timestamp` filter
 - pros: compact storage, fine-grained history
 - cons: queries require more indexing discipline

- **Versioned snapshots**
 - store periodic snapshots (daily/weekly)
 - pros: simple “point-in-time” queries
 - cons: storage-heavy at scale

For agent reliability, temporal modeling is not a “nice to have”; it prevents confident but wrong answers in incident analysis and compliance workflows.

### 6.3 Trust and provenance
Agents should weight facts by trust:
- “from Terraform state” > “from wiki” > “from an unreviewed ticket”

This reduces agent failure modes where the model confidently uses stale or wrong info.

### 6.3.1 Conflict resolution (when sources disagree)

Real graphs contain contradictions:
- wiki says owner is TeamA
- on-call rota says TeamB
- code repo indicates TeamC

A production agent should not “pick one randomly”.
Define a deterministic policy:
- rank sources by trust tier (system-of-record wins)
- prefer newer facts within the same trust tier
- if conflict persists, return a structured “conflict” response:
 - list candidates + their provenance
 - ask a clarifying question or route to a human owner

This is a major advantage of KGs over pure RAG: disagreements can be made explicit and handled systematically.

### 6.4 Storage choices: Neo4j vs Neptune vs “just Postgres”

You can build a KG on many backends. The right choice depends on:
- graph size (edges/nodes)
- query patterns (traversal vs lookup)
- operational constraints (managed service vs self-host)

Common options:

- **Neo4j (property graph)**
 - strengths: developer ergonomics (Cypher), traversal performance, mature tooling
 - trade-offs: operational scaling requires expertise; licensing considerations depending on edition

- **Amazon Neptune (managed graph DB)**
 - strengths: managed operations, integrates with AWS ecosystem
 - trade-offs: query latency patterns depend on access paths; careful modeling required

- **“Graph in Postgres”**
 - strengths: cheap, familiar, transactional
 - trade-offs: traversal queries can get slow; you’ll end up building indexes/caches anyway

For agents, a pragmatic path is:
1. start with a small, high-quality slice in a familiar store
2. move to a dedicated graph DB when traversal workloads and scale justify it

### 6.5 Indexing and retrieval: getting to p95 latency

Most agent queries are not “deep graph analytics”.
They are:
- ownership lookups
- dependency neighborhoods (depth 2–3)
- policy relations (who can do what)

To keep latency low:
- index by stable IDs (service_id, team_id)
- cache common neighborhoods (popular services)
- precompute transitive closures for bounded depths when feasible

Think of it as the graph equivalent of binary search:
you design indexes so you can jump to the relevant neighborhood without scanning the whole structure.

### 6.6 Sharding and distributed graphs (when you get big)

At 1B+ edges, single-node graphs struggle.
Distributed strategies:
- **shard by entity type** (services in one shard, people in another)
- **shard by tenant** (company/org boundaries)
- **shard by hash of node ID**

But distributed graphs introduce a classic distributed algorithms pain:
- traversals become cross-shard
- cross-shard latency dominates
- consistency becomes hard

Practical mitigation:
- design agent queries to stay within bounded neighborhoods
- replicate “hot” edges (like ownership) to reduce cross-shard hops

### 6.7 Access control (KGs can leak sensitive facts)

KGs can contain:
- PII (employee graphs)
- security permissions (who can access what)
- customer/vendor contracts

Agents should respect least privilege:
- row/edge-level ACLs
- query-time enforcement (the tool filters facts by caller permissions)
- logging and audits (“agent asked for X; it was denied”)

Without access control, a KG can become a “structured data leak” amplifier.

---

## 7. Common Pitfalls

1. **No entity resolution**: “Payments” refers to three things; agent mixes them.
2. **Schema-less graphs**: the KG becomes an untyped mess; queries become ambiguous.
3. **Over-extraction**: LLM extracts wrong triples; KG fills with plausible nonsense.
4. **Unbounded traversal**: agent asks for “everything connected to X”, times out, or returns irrelevant facts.
5. **Ignoring privacy**: KGs can store sensitive PII; governance and access control matter.

---

## 8. Best Practices

1. **Start narrow**: pick one domain (service catalog, policy compliance) and build a high-quality KG slice.
2. **Schema-first**: define types and relations before extraction.
3. **Provenance always**: every triple needs “why we believe this”.
4. **Hybrid retrieval**: use RAG for explanations, KG for constraints.
5. **Guard tool calls**: template queries, budgets, and typed outputs.

### 8.1 A concrete “enterprise agent” blueprint (KG + RAG + tools)

If you want to build something like an internal “SRE Copilot”, a robust architecture is:

1. **KG layer (facts and constraints)**
 - service ownership
 - dependencies and critical paths
 - SLAs and escalation policies
 - approved vendors / compliance constraints

2. **RAG layer (procedures and narrative)**
 - runbooks
 - incident retros
 - design docs

3. **Tool layer (actions)**
 - create ticket
 - page on-call
 - fetch dashboards
 - run safe diagnostics

4. **Policy layer**
 - enforce least privilege
 - block dangerous actions (e.g., “restart prod database”)
 - require human approval for high-risk changes

The agent’s job is then predictable:
- KG answers “what is true”
- RAG answers “what to do”
- tools execute “do it”
- policy ensures “do it safely”

### 8.2 Explainability: show the path, not just the answer

Agents become trustworthy when they can show *why*.
For KGs, the simplest and best explanation is often a path:
- “ServiceA depends on ServiceB, which is owned by TeamPayments”

In UI, show:
- the exact edges used (with provenance)
- timestamps (“last updated 2 days ago”)
- confidence and source (“from Terraform state”)

This is far more reliable than “the model says so”.

### 8.3 When to NOT use a KG

KDs are not free.
Avoid building a KG if:
- your domain is mostly unstructured and rapidly changing with no stable entities
- you can’t define a small ontology that captures value
- you don’t have a source of truth for critical relations

In those cases, start with RAG + lightweight structured extraction, and only evolve to a KG when your queries demand multi-hop precision.

---

## 9. Connections to Other Topics

This topic connects naturally across the day’s theme:
- The median partition idea teaches “boundary-based retrieval” instead of full scans—KG queries should do the same.
- Federated learning shows how distributed computation can respect privacy constraints; KGs often need similar governance and access control.
- Privacy-preserving speech highlights why sensitive data should not be centralized; KGs must be designed with least privilege and careful provenance to avoid leaking personal info through “structured facts”.

---

## 10. Real-World Examples

- **Google Knowledge Graph**: entity-centric search and disambiguation.
- **E-commerce catalogs**: products, brands, compatibility graphs.
- **Enterprise IT graphs**: services, dependencies, incidents, ownership, SLAs.
- **Security graphs**: users, permissions, resources—agents can use these for policy checks.

In agents, the common pattern is:
> query KG for “truthy” constraints, then let the LLM generate the narrative + action plan.

### 10.1 Example: Compliance / procurement agent

Task:
> “Can I use vendor X for project Y?”

KG helps because the decision depends on structured facts:
- vendor approvals by region
- contract status and expiry dates
- data residency constraints
- security questionnaire results

Agent flow:
1. entity link “vendor X” to `vendor_id`
2. KG query: `is_vendor_approved(vendor_id, region, data_class)`
3. if not approved, KG query: `recommended_alternatives(category, region)`
4. RAG: retrieve policy doc sections for rationale and citation

The output is both actionable and auditable:
- **Allowed/Denied** (deterministic)
- **Why** (citations to policy)
- **What to do next** (open procurement ticket, contact security)

This is exactly the kind of workload where KGs outperform “chat over documents”.

---

## 11. Future Directions

- **KG + embeddings**: store both symbolic edges and vector representations (graph embeddings, node2vec, GNNs).
- **Neuro-symbolic agents**: LLM proposes candidates, KG validates and constrains.
- **Continuous extraction with verification**: LLMs extract triples; deterministic systems verify via schema + cross-sources.

---

## 12. Key Takeaways

1. **Knowledge graphs give agents structure**: precise facts, multi-hop consistency, and governance.
2. **Hybrid wins**: RAG for context, KG for constraints and actions.
3. **Scaling is a distributed algorithms problem**: safe, bounded traversal + indexing are the difference between “demo” and “production”.

### 12.1 A “minimum viable KG” for agents (what to build first)

If you’re starting from scratch, don’t try to model the universe.
Build the smallest KG that makes your agent meaningfully better:

- **Entities**
 - `Service`, `Team`, `OnCallRotation`, `Runbook`, `Dashboard`, `SLA`
- **Relations**
 - `OWNED_BY(Service -> Team)`
 - `DEPENDS_ON(Service -> Service)`
 - `HAS_RUNBOOK(Service -> Runbook)`
 - `HAS_DASHBOARD(Service -> Dashboard)`
 - `HAS_SLA(Service -> SLA)`

Then add:
- provenance for each relation
- freshness timestamps
- access control for sensitive nodes (people/org charts)

Once this is in place, your agent can answer high-impact questions reliably:
- “Who owns this service?”
- “What are the critical dependencies?”
- “Where is the runbook/dashboard?”

That’s already far more actionable than generic RAG.

### 12.2 The “tool-first” discipline (how you keep agents honest)

A simple rule:
> If the answer should be deterministic, force a tool call.

Examples:
- ownership (KG tool)
- dependency list (KG tool)
- vendor approval (policy/KG tool)
- incident status (monitoring tool)

Reserve free-form generation for:
- explanations
- summaries
- recommended next steps (explicitly labeled as recommendations)

This division dramatically reduces hallucinations and makes your system auditable.

---

**Originally published at:** [arunbaby.com/ai-agents/0051-knowledge-graphs-for-agents](https://www.arunbaby.com/ai-agents/0051-knowledge-graphs-for-agents/)

*If you found this helpful, consider sharing it with others who might benefit.*

