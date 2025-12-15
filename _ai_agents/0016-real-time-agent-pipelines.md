---
title: "Real-Time Agent Pipelines"
day: 16
collection: ai_agents
categories:
  - ai-agents
tags:
  - real-time
  - latency
  - websockets
  - sse
  - streaming
  - asyncio
  - backpressure
  - tcp-udp
  - load-balancing
  - redis
difficulty: Medium
---

**"Speed is not a feature. Speed is the product."**

## 1. Introduction: The 200ms Wall

Historically, agent development focused on reasoning. Agents accepted a prompt, thought for 10 seconds, searched a vector database, formulated a plan, and eventually replied. For a Chatbot, an Email Assistant, or a Code Generator, 10 seconds is acceptable. The user is accustomed to waiting for a "complex" result.

However, for a **Voice Agent**, a **Trading Agent**, or a **Real-Time Gaming Agent**, 10 seconds is not just bad; it is broken.

Human conversation flows with a typical gap of **200 milliseconds** between speakers. This is the physiological "Turn-Taking Threshold".
*   **200ms:** A natural interface. The magic number for "Instant". This is the speed of a witty comeback or a simple acknowledgement ("Yeah", "Uh-huh").
*   **500ms:** Noticeable lag, but acceptable for internet calls (Zoom/Teams/Meet). We have trained ourselves to tolerate this "Satellite Delay" in the post-COVID era.
*   **1000ms+:** Frustrating. Users start interrupting ("Hello? Are you there?"). This breaks the illusion of presence. The user assumes the connection has dropped or the agent has crashed.

Building Real-Time Agents requires a complete paradigm shift from "Request-Response" (REST) to **Event-Driven Streaming** (WebSockets). It requires moving the metric of success from "Accuracy" to "Latency Jitter". It requires us to abandon the comfort of blocking code for the complexity of asynchronous event loops. In this post, we will engineer the pipeline for ultra-low latency intelligence.

---

## 2. The Mechanics of Latency: Where does the time go?

To solve latency, we must first understand it. Let's dissect a typical Agent turn in a naive implementation to see where the seconds bleed away.

### 2.1 The Waterfall (Serial Execution)
Imagine a standard HTTP pipeline implemented by a junior engineer:

1.  **Transport (Network Upload):** User sends audio -> Server.
    *   *Time:* 50ms (on Fiber) to 500ms (on 3G).
    *   *Constraint:* The server waits for the *entire* audio file to arrive before doing anything.
2.  **Transcription (STT):** Server receives the file. Calls OpenAI Whisper API.
    *   *Time:* 300ms (for a short sentence) to 2000ms (for a long monologue).
    *   *Constraint:* The STT engine processes the whole file at once.
3.  **Inference (LLM):** STT Text -> LLM. The LLM receives the prompt. It "thinks" (Time to First Token). It then generates the full response.
    *   *Time:* 2000ms.
    *   *Constraint:* The system waits for the *entire* text response to be generated before moving to the next step.
4.  **Synthesis (TTS):** LLM Text -> TTS Engine. The engine generates the audio waveform.
    *   *Time:* 500ms.
    *   *Constraint:* The system waits for the *entire* audio file to be generated.
5.  **Transport (Network Download):** Server -> User.
    *   *Time:* 50ms.

**Total:** $50 + 300 + 2000 + 500 + 50 = 2900ms$ (2.9 seconds).
This is nearly 3 seconds of dead air. In a voice conversation, 3 seconds is an eternity. It is the awkward silence that kills the vibe.

### 2.2 Streaming Pipelining (Parallel Execution)
To get to sub-500ms, we cannot execute these steps sequentially. We must **Pipeline** them. We must treat data as a continuous river, not a series of buckets. This is often called **Optimistic Execution**.

*   **User:** Speaks "Hello".
*   **STT (Stream):** The server processes audio chunks as they arrive.
    *   T=10ms: "H"
    *   T=20ms: "Hel"
    *   T=30ms: "Hello"
*   **LLM (Stream):** The LLM receives "Hello". It **Immediately** starts generating tokens. It does not wait for the sentence to be grammatically complete (though sometimes it should, which we will discuss in VAD).
*   **LLM Output (Stream):**
    *   T=100ms: Generates "Hi" (Token 1).
    *   T=150ms: Generates " there" (Token 2).
*   **TTS (Stream):** The TTS engine receives "Hi". It immediately synthesizes those phonemes into audio bytes.
*   **Network (Stream):** The server pushes the audio bytes for "Hi" to the user.
*   **User:** The user hears "Hi" while the LLM is *still generating* the rest of the sentence (" there, how are you?").

*   **Latency Cost:** We only pay the "Time to First Token" (TTFT) of each stage, not the full processing time.
*   **New Total:** $50 (Net) + 200 (STT-Partial) + 300 (LLM-TTFT) + 100 (TTS-Partial) = 650ms$.
*   *Result:* The user perceives an instant response.

---

## 3. Protocols: The Transport Layer

How do we move bits this fast over the public internet? The choice of protocol determines your baseline physics.

### 3.1 HTTP/REST (The Bad)
*   *Architecture:* Universal Request-Response. Client sends request, waits, server sends response.
*   *Overhead:* TCP 3-Way Handshake + TLS Handshake on *every* interaction. This adds ~100ms RTT just to say "Hello".
*   *Headers:* Every HTTP request carries bulky headers (Cookies, User-Agent, Accept-Encoding). Sending a 1KB audio chunk with 2KB of headers is inefficient (33% efficiency).
*   *Verdict:* Unusable for real-time audio. Good only for control signals (e.g. "Create Room").

### 3.2 Server-Sent Events (SSE) (The Okay)
*   *Architecture:* Client opens one HTTP connection (`Response-Type: text/event-stream`). Server keeps it open and pushes text events (`data: ...`) whenever it wants.
*   *Pros:* Great for Text Chat (ChatGPT uses this). Simple (Standard HTTP). Firewalls love it. Automatic reconnection logic in browsers.
*   *Cons:* **Unidirectional.** Server -> Client only. If the user interrupts, the client cannot use the same connection to send the interruption signal. It must open a *new* POST request to send data. This adds 1x RTT (Round Trip Time) latency to the interruption, making the agent feel "deaf" for a split second.

### 3.3 WebSockets (The King)
*   *Architecture:* Starts as HTTP, then "Upgrades" to raw TCP. Full Duplex. Both sides send binary/text frames at any time exactly when they happen.
*   *Pros:* Lowest overhead. Ideal for Audio/Video streams. No headers per frame.
*   *Cons:* **Stateful**. You cannot use standard Load Balancers (Round Robin) easily because the client is "sticky" to one server process. If that server dies, the call drops. Implementing "sticky sessions" or a "router layer" is required for scale.
*   *Framing:* WebSockets manage "Framing" (knowing where one message ends and the next begins) for you, unlike raw TCP where you have to implement length-prefixing.

### 3.4 WebRTC (The God Tier)
*   *Architecture:* UDP (User Datagram Protocol) via RTP (Real-time Transport Protocol).
*   *Physics:* TCP (WebSockets) guarantees delivery. If a packet is lost, TCP pauses everything to re-transmit it. This causes "Lag Spikes". UDP is "Fire and Forget". If a packet is lost, it is lost forever.
*   *Audio Logic:* In voice, a lost packet is a 20ms glitch. A delayed packet is a 500ms lag. We prefer **Glitch > Lag**. Therefore, UDP is superior.
*   *Cons:* Extremely complex to implement server-side (Requires DTLS encryption, ICE Candidate exchange, STUN/TURN servers for NAT traversal).
*   *Verdict:* Use WebSockets for text/simple audio agents. Use WebRTC for robust production voice (Zoom/Teams quality).

---

## 4. Asyncio: The Python Backend

Python is the language of AI, but naive Python (Flask/Django) kills real-time performance.
You **must** use `asyncio` (FastAPI/Quart/Litestar).

### 4.1 The Event Loop Mechanics: Single-Threaded Concurrency
In a blocking server (e.g., Flask with Sync Workers):
```python
def handle_request():
    audio = receive_audio() # BLOCKS the thread for 2 seconds waiting for network
    text = transcribe(audio) # BLOCKS the thread for processing
    return text
```
If you have 10 users, you need 10 threads. 1000 users = 1000 threads. The OS spends significant CPU time just context-switching between threads (saving/loading registers). You hit the "C10K Problem" (Connecting 10k users) very quickly.

In `asyncio`:
```python
async def handle_request():
    audio = await receive_audio() # Yields control.
    text = await transcribe(audio)
    return text
```
Here, `await` is a magical keyword. It tells the Event Loop: *"I am waiting for IO (Network/Disk). I release the CPU. Go do work for someone else. Wake me up when the data arrives."*
One single thread can handle 10,000 connections because at any given millisecond, 9,999 of them are just waiting for packets.

### 4.2 Handling Streams in Python: Async Generators
The most elegant pattern for pipelines is the **Async Generator**.

```python
async def llm_stream(prompt):
    # This call initiates the stream but doesn't block the whole function
    stream = await openai.chat.completions.create(model="gpt-4o", stream=True)
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token  # Emits token to the consumer immediately

async def tts_stream(text_iterator):
    buffer = ""
    async for token in text_iterator:
        buffer += token
        # Heuristic: Send to TTS only on sentence boundaries or commas
        # This optimizes for audio quality (intonation) vs latency
        if ends_sentence(buffer):
            audio = await synthesize(buffer)
            yield audio
            buffer = ""

# The Pipeline
async def handle_websocket(ws):
    await ws.accept()
    prompt = await ws.receive_text()
    
    # We pipe the generators together like Unix pipes
    # Data flows: Network -> iterator -> iterator -> Network
    text_stream = llm_stream(prompt)
    audio_stream = tts_stream(text_stream) 
    
    async for audio_chunk in audio_stream:
        await ws.send_bytes(audio_chunk)
```

This code is **lazy**. No processing happens until the final `async for` loop pulls data properly.

---

## 5. The Silent Killer: Backpressure

When you pipeline systems, speed mismatches occur. This leads to the **Producer-Consumer Problem**.
*   **LLM Generation:** 50 tokens/sec (Fast).
*   **TTS Synthesis:** Real-time (1 sec audio takes 1 sec to play).
*   **Network:** Variable (Cellular 4G implies jitter).
*   **Client Playback:** Real-time.

**Scenario:**
The LLM generates a 3-minute speech in 5 seconds. It pushes all that text to the TTS. The TTS generates 3 minutes of audio in 30 seconds. It pushes all that audio to the WebSocket buffer.
1.  **Memory Spike:** Server RAM fills up with buffered audio.
2.  **Latency Increase (The Death Spiral):** The user tries to interrupt ("Stop!"). The server receives "Stop". It stops the LLM. *But the network buffer is already filled with 3 minutes of audio.* TCP guarantees delivery. The user hears the agent talking for 3 minutes *after* they shouted "Stop!".

**Solution: Flow Control (Backpressure)**
*   **Buffer Cap:** Limit the output queue to 5 chunks. If the queue is full, `await` (pause) the producer.
    `await output_queue.put(audio)` -> This line blocks if queue is full. This naturally slows down the LLM to match the Network speed.
*   **Clearing:** On interruption (Barge-In), you must support a `clear_buffer()` command.
    *   This requires accessing the internal buffer of the WebSocket or simply sending a control signal to the client: `"IGNORE_PREVIOUS_AUDIO"`.
    *   Server-side, you drop all pending items in the `output_queue`.

---

## 6. Architecture Pattern: The Actor Model (Orchestrator)

For a robust pipeline, don't write one monolithic `handle_websocket` function. Use an **Orchestrator Pattern**. Break every component into an independent "Worker" that communicates via Queues.

*   **Input Queue:** Receives Audio Chunks from WS.
*   **VAD Worker:**
    *   Consumes Input.
    *   Detects "Speech Start" -> Emits "User Speaking" Event.
    *   Detects "Speech End" -> Emits "User Utterance" Event.
*   **STT Worker:** Consumes Utterance. Emits Text.
*   **Agent Brain:** Consumes Text. Maintains State. Decides Action (RAG/Tool). Emits Start Token.
*   **TTS Worker:** Consumes Tokens. Emits Audio.
*   **Output Queue:** Sends to WebSocket.

This decoupled architecture allows **Complex State Management**.
*   *Scenario:* User speaks while Agent is talking.
*   *VAD Worker:* Detects speech energy. Sends `INTERRUPT` event.
*   *Orchestrator:*
    *   Calls `TTS_Worker.cancel()`
    *   Calls `LLM_Worker.cancel()`
    *   Sends `ws.send_json({"type": "clear_audio"})` to client.
*   *Result:* Agent shuts up instantly, feeling "Listening".

---

## 7. Scaling WebSockets: The Redis Pub/Sub Layer

One server can handle 10k connections. But what if you need 1 Million?
You need multiple servers. But WebSockets are stateful.
*   User A connects to Server 1.
*   User B connects to Server 2.
*   Agent Logic runs on Server 3.

How does Agent Logic send audio to User A? It doesn't have the socket handle.

**Solution: Redis Pub/Sub (The Message Bus)**
1.  **Gateway Layer:** Holds the WebSockets. Does nothing but forward bytes.
2.  **Processing Layer:** Runs the AI Logic.
3.  **Communication:**
    *   User A sends audio. Gateway 1 publishes to Redis Channel `input:user_a`.
    *   Processor subscribes to `input:user_a`. Processes it.
    *   Processor publishes audio to `output:user_a`.
    *   Gateway 1 subscribes to `output:user_a`. Forwards bytes to WebSocket.
*   *Pros:* Infinite scaling. Processing layer can crash without dropping the connection.
*   *Cons:* Adds ~5-10ms latency (Redis RTT).

---

## 8. Code: A FastApi WebSocket Agent

Here is a more complete implementation demonstrating the connection lifecycle and asyncio gathering.

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import uuid
import json

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    session_id = str(uuid.uuid4())
    print(f"Session {session_id} started")
    
    # Queues for inter-task communication
    # maxsize=10 provides Backpressure!
    input_queue = asyncio.Queue(maxsize=100) 
    
    # Start the Processing Worker in background
    worker_task = asyncio.create_task(run_agent_pipeline(session_id, input_queue, websocket))
    
    try:
        while True:
            # 1. Receive packet
            # receive_bytes blocks until data arrives
            data = await websocket.receive_bytes()
            
            # Put into queue. If worker is slow, this will eventually block,
            # signaling the network layer to stop reading (TCP Window Scaling).
            await input_queue.put(data)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        worker_task.cancel() # Kill the worker
        print(f"Session {session_id} ended")

async def run_agent_pipeline(session_id, input_queue, ws):
    """
    Simulates the AI processing pipeline independent of the network loop.
    """
    print(f"Worker for {session_id} started")
    try:
        while True:
            audio_chunk = await input_queue.get()
            
            # Simulated VAD
            is_speech = process_vad(audio_chunk)
            if is_speech:
                 # Logic...
                 pass
                 
            # Simulated TTS Output
            # await ws.send_bytes(generated_audio)
            
    except asyncio.CancelledError:
        print("Worker cancelled")

def process_vad(chunk):
    # Mock VAD
    return True
```

---

## 9. Summary

Real-time is the frontier of Agent UX. It is where "Software Engineering" meets "AI". To build a great voice agent, you must stop thinking about "Models" and start thinking about "Systems".

*   **WebSockets/WebRTC** are mandatory for transport.
*   **Asyncio** is the mandatory runtime pattern for Python.
*   **Pipelining** (Streaming) hides the latency of individual components.
*   **Backpressure** handling ensures the agent doesn't sound "Laggy" during interruptions.
*   **Orchestration** decouples the brain from the mouth.

These real-time pipelines are the foundation for **Voice Agents**, which add layers of complexity regarding STT accuracy, Cost Analysis, and the physics of sound.
