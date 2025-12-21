---
title: "Streaming and Real-Time Agents"
day: 48
collection: ai_agents
categories:
  - ai-agents
tags:
  - streaming
  - real-time
  - server-sent-events
  - websockets
  - low-latency
difficulty: Hard
subdomain: "Agent Architecture"
tech_stack: Python, FastAPI, WebSockets, SSE
scale: "Sub-second first token, 1000s concurrent streams"
companies: OpenAI, Anthropic, Google, Microsoft
related_dsa_day: 48
related_ml_day: 48
related_speech_day: 48
---

**"Users shouldn't wait for the whole answer—stream it token by token."**

## 1. Introduction

Real-time agents stream responses as they're generated, providing immediate feedback. This transforms user experience from "waiting" to "watching the agent think."

### Why Streaming?

```
Non-streaming:
User: "Explain quantum computing"
[Wait 10 seconds...]
Agent: [Full 2000-word response appears]

Streaming:
User: "Explain quantum computing"
Agent: "Quantum" [50ms]
Agent: "computing" [100ms]
Agent: "is a type of..." [continuing...]
```

**Benefits:**
- First token in <500ms vs 5-10s for full response
- User can interrupt/redirect early
- Better perceived performance

## 2. Streaming Architecture

```python
from typing import AsyncIterator, Callable
import asyncio
from dataclasses import dataclass
from enum import Enum

class StreamEventType(Enum):
    TOKEN = "token"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    DONE = "done"


@dataclass
class StreamEvent:
    type: StreamEventType
    content: str = ""
    metadata: dict = None


class StreamingAgent:
    """Agent with streaming response support."""
    
    def __init__(self, llm_client, tools: list = None):
        self.llm = llm_client
        self.tools = {t.name: t for t in (tools or [])}
    
    async def run_streaming(
        self,
        query: str,
        context: dict = None
    ) -> AsyncIterator[StreamEvent]:
        """
        Execute agent with streaming output.
        
        Yields events as they occur:
        - Tokens as generated
        - Tool calls as detected
        - Tool results as executed
        """
        messages = self._build_messages(query, context)
        
        while True:
            # Stream LLM response
            full_response = ""
            tool_calls = []
            
            async for chunk in self.llm.stream(messages):
                if chunk.type == "token":
                    full_response += chunk.content
                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        content=chunk.content
                    )
                
                elif chunk.type == "tool_call":
                    tool_calls.append(chunk.tool_call)
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL,
                        content=chunk.tool_call.name,
                        metadata={"args": chunk.tool_call.args}
                    )
            
            # Handle tool calls
            if tool_calls:
                for call in tool_calls:
                    result = await self._execute_tool(call)
                    yield StreamEvent(
                        type=StreamEventType.TOOL_RESULT,
                        content=str(result),
                        metadata={"tool": call.name}
                    )
                    
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [call.to_dict()]
                    })
                    messages.append({
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": call.id
                    })
                
                # Continue generation after tool results
                continue
            else:
                # No more tool calls, done
                break
        
        yield StreamEvent(type=StreamEventType.DONE)
    
    async def _execute_tool(self, tool_call):
        """Execute a tool and return result."""
        tool = self.tools.get(tool_call.name)
        if not tool:
            return f"Error: Unknown tool {tool_call.name}"
        
        try:
            return await tool.execute(**tool_call.args)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _build_messages(self, query, context):
        messages = []
        if context and context.get("system"):
            messages.append({"role": "system", "content": context["system"]})
        messages.append({"role": "user", "content": query})
        return messages
```

## 3. Server-Sent Events (SSE)

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/agent/stream")
async def stream_agent(request: Request):
    """SSE endpoint for streaming agent responses."""
    data = await request.json()
    query = data.get("query", "")
    
    agent = StreamingAgent(llm_client, tools)
    
    async def event_generator():
        async for event in agent.run_streaming(query):
            # Format as SSE
            data = json.dumps({
                "type": event.type.value,
                "content": event.content,
                "metadata": event.metadata
            })
            yield f"data: {data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
```

## 4. WebSocket Streaming

```python
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/agent/ws")
async def websocket_agent(websocket: WebSocket):
    """WebSocket endpoint for bidirectional streaming."""
    await websocket.accept()
    agent = StreamingAgent(llm_client, tools)
    
    try:
        while True:
            # Receive query
            data = await websocket.receive_json()
            query = data.get("query", "")
            
            # Stream response
            async for event in agent.run_streaming(query):
                await websocket.send_json({
                    "type": event.type.value,
                    "content": event.content,
                    "metadata": event.metadata
                })
            
    except WebSocketDisconnect:
        pass


class WebSocketClient:
    """Client for WebSocket streaming."""
    
    def __init__(self, url: str):
        self.url = url
        self.ws = None
    
    async def connect(self):
        import websockets
        self.ws = await websockets.connect(self.url)
    
    async def query(self, text: str) -> AsyncIterator[StreamEvent]:
        """Send query and stream response."""
        await self.ws.send(json.dumps({"query": text}))
        
        while True:
            msg = await self.ws.recv()
            data = json.loads(msg)
            
            event = StreamEvent(
                type=StreamEventType(data["type"]),
                content=data.get("content", ""),
                metadata=data.get("metadata")
            )
            
            yield event
            
            if event.type == StreamEventType.DONE:
                break
    
    async def close(self):
        if self.ws:
            await self.ws.close()
```

## 5. Streaming with Tool Execution

```python
class StreamingToolAgent:
    """Handle tool calls mid-stream."""
    
    async def run_with_tools(
        self,
        query: str
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream tokens, pause for tools, resume.
        """
        messages = [{"role": "user", "content": query}]
        
        while True:
            buffer = ""
            tool_call_buffer = None
            
            async for chunk in self.llm.stream(messages):
                # Detect tool call start
                if "<tool_call>" in buffer + chunk.content:
                    # Pause token streaming
                    tool_call_buffer = ""
                    continue
                
                if tool_call_buffer is not None:
                    tool_call_buffer += chunk.content
                    
                    if "</tool_call>" in tool_call_buffer:
                        # Parse and execute tool
                        tool_call = self._parse_tool_call(tool_call_buffer)
                        
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL,
                            content=tool_call.name
                        )
                        
                        result = await self._execute_tool(tool_call)
                        
                        yield StreamEvent(
                            type=StreamEventType.TOOL_RESULT,
                            content=str(result)
                        )
                        
                        # Add to context
                        messages.append({
                            "role": "assistant",
                            "content": f"<tool_call>{tool_call_buffer}</tool_call>"
                        })
                        messages.append({
                            "role": "tool",
                            "content": str(result)
                        })
                        
                        tool_call_buffer = None
                        break
                else:
                    buffer += chunk.content
                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        content=chunk.content
                    )
            
            if tool_call_buffer is None:
                # No tool call, we're done
                break
        
        yield StreamEvent(type=StreamEventType.DONE)
```

## 6. Client-Side Handling

```javascript
// JavaScript SSE client
async function streamAgent(query, onToken, onToolCall, onDone) {
    const response = await fetch('/agent/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query})
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                
                switch (data.type) {
                    case 'token':
                        onToken(data.content);
                        break;
                    case 'tool_call':
                        onToolCall(data.content, data.metadata);
                        break;
                    case 'done':
                        onDone();
                        return;
                }
            }
        }
    }
}
```

## 7. Handling Cancellation

```python
class CancellableStreamingAgent:
    """Support user cancellation mid-stream."""
    
    def __init__(self, llm):
        self.llm = llm
        self.cancel_event = asyncio.Event()
    
    def cancel(self):
        """Cancel current stream."""
        self.cancel_event.set()
    
    async def run_streaming(self, query: str) -> AsyncIterator[StreamEvent]:
        self.cancel_event.clear()
        
        try:
            async for event in self._stream(query):
                if self.cancel_event.is_set():
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        content="Cancelled by user"
                    )
                    return
                
                yield event
        except asyncio.CancelledError:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content="Stream cancelled"
            )
    
    async def _stream(self, query: str):
        # Actual streaming implementation
        pass


# FastAPI with cancellation
@app.post("/agent/stream")
async def stream_with_cancel(request: Request):
    agent = CancellableStreamingAgent(llm)
    
    async def generator():
        async for event in agent.run_streaming(query):
            if await request.is_disconnected():
                agent.cancel()
                return
            yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(generator())
```

## 8. Connection to Trie-based Search

Both streaming and Trie search share:
- **Incremental results**: Get partial answers immediately
- **Early termination**: Stop when enough information found
- **Prefix matching**: Each token/phoneme extends the prefix

## 9. Key Takeaways

1. **Stream tokens** for better UX (first token <500ms)
2. **SSE for simple streaming**, WebSocket for bidirectional
3. **Handle tool calls** mid-stream gracefully
4. **Support cancellation** - users change their mind
5. **Buffer for parsing** - tool calls span multiple tokens

---

**Originally published at:** [arunbaby.com/ai-agents/0048-streaming-real-time-agents](https://www.arunbaby.com/ai-agents/0048-streaming-real-time-agents/)

*If you found this helpful, consider sharing it with others who might benefit.*
