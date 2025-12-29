---
title: "Streaming Real-Time Agents"
day: 48
related_dsa_day: 48
related_ml_day: 48
related_speech_day: 48
collection: ai_agents
categories:
 - ai-agents
tags:
 - streaming
 - websockets
 - sse
 - ux
 - latency
difficulty: Medium
subdomain: "Agent UX"
tech_stack: FastAPI, React, Server-Sent Events, LangChain
scale: "Concurrent streaming for 10k users"
companies: ChatGPT, Perplexity, Anthropic, Vercel
---

**"Waiting 10 seconds for a thoughtful answer is okay. Waiting 10 seconds for a blank screen is broken."**

## 1. Introduction

Human conversation is streamed. We start processing the first word of a sentence before the speaker finishes the paragraph.
Early LLM applications waited for the full generation (Stop Token) to complete before sending a JSON response.
- **Old Way**: Request -> Wait 15s -> Show 500 words. (User thinks app crashed).
- **New Way**: Request -> Wait 0.5s -> Show "The"... "quick"... "brown"...

For **AI Agents**, streaming is harder because they have "Thought Steps" (internal monologues) that the user shouldn't see, interspersed with "Final Answers".

---

## 2. Core Concepts: Protocols

How do we push data to the browser?

1. **Short Polling**: Client asks "Done?" every 1s. (Inefficient).
2. **WebSockets**: Bi-directional, full-duplex TCP. Good for real-time gaming, overkill for chat. Hard to load balance (stateful connections).
3. **Server-Sent Events (SSE)**: The standard for LLMs.
 - One-way HTTP connection.
 - Server keeps socket open and pushes `data: ...` chunks.
 - Simple to implement with standard Load Balancers.

---

## 3. Architecture Patterns: The Stream Transformer

We need an architecture that transforms raw LLM tokens into structured Agent Events.

``
[LLM (OpenAI)]
 | (Stream of Tokens)
 v
[Agent Parser Parsers logic]
 | Detects: "Action: Search" -> WAIT
 | Detects: "Observation: 42" -> WAIT
 | Detects: "Final Answer: The..." -> STREAM
 v
[Frontend (React)]
``

The key challenge: **Leakage**. We don't want to stream the raw JSON braces of a tool call to the user. We only want to stream the "final_answer".

---

## 4. Implementation Approaches

### 4.1 The Generator Pattern (Python)
Python `yield` is perfect for this.

``python
async def agent_stream(prompt):
 # 1. Start LLM Stream
 stream = await openai.ChatCompletion.create(..., stream=True)
 
 buffer = ""
 in_tool_mode = False
 
 async for chunk in stream:
 token = chunk.choices[0].delta.content
 buffer += token
 
 # 2. Logic to detect Tool Usage
 if "<tool>" in buffer:
 in_tool_mode = True
 yield event("status", "Thinking...")
 
 if not in_tool_mode:
 yield event("text", token)
``

### 4.2 The Client Consumer (React)
Using `fetch` with a `ReadableStream`.

``javascript
const response = await fetch('/api/agent');
const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
 const { done, value } = await reader.read();
 if (done) break;
 const chunk = decoder.decode(value);
 // Parse "event: text\ndata: hello"
 handleSSE(chunk);
}
``

---

## 5. Code Examples: FastAPI with SSE

Here is a robust backend implementation using `ssep` format.

``python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import asyncio

app = FastAPI()

async def event_generator():
 """
 Yields data in strict SSE format:
 data: {"key": "value"}\n\n
 """
 # Phase 1: Planning
 yield f"data: {json.dumps({'type': 'status', 'content': 'Searching Google...'})}\n\n"
 await asyncio.sleep(1) # Fake tool latency
 
 # Phase 2: Streaming Answer
 sentence = "The speed of light is 299,792 km/s."
 for word in sentence.split():
 yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
 await asyncio.sleep(0.1)

 # Phase 3: Done
 yield "data: [DONE]\n\n"

@app.get("/stream")
async def stream_endpoint():
 return StreamingResponse(event_generator(), media_type="text/event-stream")
``

---

## 6. Production Considerations

### 6.1 Buffering at the Edge
Nginx, Cloudflare, and AWS ALB love to "Buffer" responses to optimize compression (gzip).
They might wait until 1KB of data is accumulated before sending it to the user.
**Fix**:
- Set header `X-Accel-Buffering: no` (Nginx).
- Set header `Cache-Control: no-cache`.
- Disable Gzip for `/stream` endpoints.

### 6.2 Timeouts
Standard HTTP requests time out after 60s. An agent might take 2 minutes.
**Fix**: Configure your Load Balancer's `idle_timeout` to 300s+ for streaming paths.

---

## 7. Common Pitfalls

1. **JSON Truncation**: Trying to parse a JSON object via `json.loads()` while it's still streaming (incomplete).
 - *Fix*: Use a streaming JSON parser (like `json-stream` library) or only parse line-delimited chunks.
2. **Flash of Unstyled Content**: Streaming tokens causes the layout to shift violently (Cumulative Layout Shift).
 - *Fix*: Set a minimum height for the chat container.

---

## 8. Best Practices: "Skeleton Loaders" for Thoughts

Don't just stream text. Stream **Status Updates**.
Users love to see the "Brain":
- `[Status: Reading PDF...]`
- `[Status: calculating...]`
- `[Text: The answer is 42]`

This "Transparency" reduces perceived latency.

---

## 9. Connections to Other Topics

This connects to **Speech Model Export** (Speech).
- Both deal with **Streaming Latency**.
- In Speech, we show partial words (`he` -> `hel` -> `hello`).
- In Agents, we show partial thoughts.
- The UX challenge is identical: "Stability vs Speed".

---

## 10. Real-World Examples

- **Perplexity.ai**: The gold standard. They show the "Sources" appearing one by one, then the answer streams.
- **Vercel AI SDK**: A library that standardizes the "Stream Data Protocol", making it easy to hook Next.js to OpenAI streams.

---

## 11. Future Directions

- **Generative UI**: Streaming not just text, but React Components.
 - Agent streams: `<WeatherWidget temp="72" />`.
 - Browser renders the widget instantly.
- **Duplex Speech**: Streaming Audio In -> Streaming Audio Out (OpenAI GPT-4o). No text intermediate.

---

## 12. Key Takeaways

1. **SSE > WebSockets**: For 99% of Chat Agents, SSE is simpler and friendlier to firewalls.
2. **Edge Buffering is the Enemy**: If streaming isn't working, check your Nginx config.
3. **Visual Latency**: Aim for < 200ms TTFT (Time To First Token).
4. **Leakage Control**: Build a state machine to hide raw "Tool Calls" from the end user.

---

**Originally published at:** [arunbaby.com/ai-agents/0048-streaming-real-time-agents](https://www.arunbaby.com/ai-agents/0048-streaming-real-time-agents/)

*If you found this helpful, consider sharing it with others who might benefit.*
