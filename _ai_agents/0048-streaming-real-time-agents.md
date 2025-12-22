---
title: "Streaming Real-Time Agents"
day: 48
collection: ai_agents
categories:
  - ai-agents
tags:
  - streaming
  - real-time
  - ux
  - websockets
  - sse
difficulty: Hard
subdomain: "Agent Architecture"
tech_stack: Python, FastAPI, React, Server-Sent Events
scale: "Sub-100ms time-to-first-token"
companies: Vercel, LangChain, OpenAI
related_dsa_day: 48
related_ml_day: 48
related_speech_day: 48
---

**"Waiting is the hardest part. So don't make them wait."**

## 1. Introduction: The 10-Second Stare

You ask an advanced AI Agent: "Plan a travel itinerary for Tokyo, find hotels under $200, and book a reservation at a sushi place."
The agent goes to work.
- It searches for flights. (2 seconds)
- It reads 5 hotel reviews. (4 seconds)
- It checks restaurant availability. (3 seconds)
- It generates the final response. (3 seconds)

**Total time: 12 seconds.**
In the world of web UX, 12 seconds is an eternity. Users will think the app crashed and close the tab.

**The Solution: Streaming.**
Instead of waiting for the *final* answer, the agent should stream its "thoughts" and partial progress immediately.
0.1s: "Searching for flights..."
2.5s: "Found 3 flights. Now checking hotels..."
6.0s: "Hotel reviews analyzed. Looking for sushi..."

This transforms a "broken" experience into an engaging, magical one.

---

## 2. Theoretical Foundation: HTTP vs. Streaming

### 2.1 The Traditional Request/Response Cycle
Standard HTTP is **transactional**.
1. Client sends Request.
2. Server computes... (silence)...
3. Server sends Response (all at once).

This blocks the UI. The user sees a spinning loader.

### 2.2 Server-Sent Events (SSE) & WebSockets
For agents, we need **continuous** communication.
- **WebSockets**: Bi-directional. Overkill if the user just listens, but good for interruptions.
- **Server-Sent Events (SSE)**: Uni-directional (Server -> Client). Perfect for LLM streaming. Standard HTTP connection kept open.

---

## 3. Streaming "Thoughts" vs. "Tokens"

For a simple chatbot (like ChatGPT), you just stream the text tokens of the final answer.
`H`... `He`... `Hel`... `Hello`...

For an **Agent**, the stream is more complex. It contains different *types* of events:

1. **Thought Events**: "I need to use the Search Tool."
2. **Tool Input Events**: "Calling Google Search with query 'Tokyo Hotels'."
3. **Tool Output Events**: "Google returned: [Park Hyatt, APA Hotel...]"
4. **Final Answer Tokens**: "Based on my search, I recommend..."

If you only stream the final answer, the user sits in silence during steps 1-3.
If you stream everything as raw text, the user sees ugly distinct JSON or debug logs.

**The Solution: Structured Streaming Protocol**
We need to invent a mini-protocol for our stream.

```json
event: "status"
data: {"message": "Reading documents...", "icon": "📖"}

event: "thought"
data: {"text": "The user wants a cheap hotel, I should filter by price."}

event: "token"
data: "Based"

event: "token"
data: " on"
```

The Frontend UI parses these events:
- **Status events** update a small "Thinking..." indicator.
- **Token events** are appended to the main chat bubble.

---

## 4. Architecture Implementation

### 4.1 The Generator Pattern (Python)
In Python (FastAPI/Flask), we use **generators** (`yield`) to push data without closing the connection.

```python
# Conceptual implementation
async def stream_agent_execution(user_query):
    # 1. Thought phase
    yield format_event("status", "Planning tasks...")
    plan = await planner_llm.plan(user_query)
    
    # 2. Tool Execution phase
    for step in plan:
        yield format_event("status", f"Executing: {step.tool_name}...")
        result = await tools.execute(step)
        yield format_event("log", f"Tool output: {len(result)} bytes")
    
    # 3. Final Answer phase
    yield format_event("status", "Writing response...")
    async for token in response_llm.stream(context):
        yield format_event("token", token)
```

### 4.2 Handling "Backpressure"
Common Pitfall: The LLM generates tokens faster than the user's internet can download them (rare), or—more likely—the *Tool* is slow, causing the connection to time out.
- **Keep-Alive**: Send a "ping" event every few seconds if the Agent is waiting for a slow tool (like a 30s scraper). This prevents the browser or load balancer (Nginx/AWS LB) from killing the connection due to inactivity.

---

## 5. User Experience (UX) Patterns for Streaming

1. **The "Skeleton" Loader**: Before the first token arrives, show a pulsing layout structure.
2. **The "Thinking" Accordion**: Show a collapsed sections called "View Steps". Curious users can expand it to see the tool logs ("Searched Google", "Read PDF"). Casual users just see "Thinking..." and then the answer.
3. **Optimistic UI**: If the user asks "Draft an email", show the email editor opening *while* the agent is still generating the subject line.

---

## 6. Connection to Other Days

- **DSA (Word Search)**: Searching the agent's memory or tools list is a search problem.
- **ML System Design (Typeahead)**: Typeahead *is* a streaming interface. You stream characters, it streams suggestions. The latency constraints are similar (<100ms).

---

## 7. Summary

Real-time streaming is what separates "demo" agents from "production" agents.
It is an illusion of speed. The total time to complete the task might be the same, but the **Perceived Performance** is drastically better because the user sees progress immediately.

When building agents:
1. Don't just `return response`. `yield events`.
2. Define a clear protocol for your events (Status vs. Token).
3. Handle the silence gap while tools runs (Keep-Alives).

---

**Originally published at:** [arunbaby.com/ai-agents/0048-streaming-real-time-agents](https://www.arunbaby.com/ai-agents/0048-streaming-real-time-agents/)

*If you found this helpful, consider sharing it with others who might benefit.*
