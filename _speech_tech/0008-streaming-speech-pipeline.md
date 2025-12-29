---
title: "Streaming Speech Processing Pipeline"
day: 8
related_dsa_day: 8
related_ml_day: 8
related_agents_day: 8
collection: speech_tech
categories:
 - speech-tech
tags:
 - streaming
 - real-time
 - pipeline
 - latency
 - websockets
subdomain: Real-time Processing
tech_stack: [Python, WebSocket, asyncio, gRPC, PyAudio, sounddevice]
scale: "Real-time streaming"
companies: [Google, Amazon, Microsoft, Apple, Meta, Zoom]
---

**Build real-time speech processing pipelines that handle audio streams with minimal latency for live transcription and voice interfaces.**

## Introduction

**Streaming speech processing** handles audio in real-time as it's captured, without waiting for the entire recording.

**Why streaming matters:**
- **Low latency:** Start processing immediately (< 100ms)
- **Live applications:** Transcription, translation, voice assistants
- **Memory efficiency:** Process chunks, not entire recordings
- **Better UX:** Instant feedback to users

**Challenges:**
- Chunking audio correctly
- Managing state across chunks
- Handling network delays
- Synchronization issues

---

## Streaming Pipeline Architecture

``
┌─────────────┐
│ Microphone │
└──────┬──────┘
 │ Audio stream (PCM)
 ▼
┌──────────────────┐
│ Audio Chunker │ ← Split into chunks (e.g., 100ms)
└──────┬───────────┘
 │ Chunks
 ▼
┌──────────────────┐
│ Preprocessor │ ← Normalize, filter
└──────┬───────────┘
 │
 ▼
┌──────────────────┐
│ Feature Extract │ ← MFCC, Mel-spec
└──────┬───────────┘
 │
 ▼
┌──────────────────┐
│ ML Model │ ← ASR, classification
└──────┬───────────┘
 │
 ▼
┌──────────────────┐
│ Post-processing │ ← Smoothing, formatting
└──────┬───────────┘
 │
 ▼
┌──────────────────┐
│ Output │ ← Transcription, action
└──────────────────┘
``

---

## Audio Capture & Chunking

### Real-time Audio Capture

``python
import pyaudio
import numpy as np
from queue import Queue
import threading

class AudioStreamer:
 """
 Capture audio from microphone in real-time
 
 Buffers chunks for processing
 """
 
 def __init__(self, sample_rate=16000, chunk_duration_ms=100):
 self.sample_rate = sample_rate
 self.chunk_duration_ms = chunk_duration_ms
 self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
 
 self.audio_queue = Queue()
 self.stream = None
 self.running = False
 
 def _audio_callback(self, in_data, frame_count, time_info, status):
 """
 Callback called by PyAudio for each audio chunk
 
 Runs in separate thread
 """
 if status:
 print(f"Audio status: {status}")
 
 # Convert bytes to numpy array
 audio_data = np.frombuffer(in_data, dtype=np.int16)
 
 # Normalize to [-1, 1]
 audio_data = audio_data.astype(np.float32) / 32768.0
 
 # Add to queue
 self.audio_queue.put(audio_data)
 
 return (in_data, pyaudio.paContinue)
 
 def start(self):
 """Start capturing audio"""
 self.running = True
 
 p = pyaudio.PyAudio()
 
 self.stream = p.open(
 format=pyaudio.paInt16,
 channels=1,
 rate=self.sample_rate,
 input=True,
 frames_per_buffer=self.chunk_size,
 stream_callback=self._audio_callback
 )
 
 self.stream.start_stream()
 print(f"Started audio capture (chunk={self.chunk_duration_ms}ms)")
 
 def stop(self):
 """Stop capturing audio"""
 self.running = False
 if self.stream:
 self.stream.stop_stream()
 self.stream.close()
 print("Stopped audio capture")
 
 def get_chunk(self, timeout=1.0):
 """
 Get next audio chunk
 
 Returns: numpy array of audio samples
 """
 try:
 return self.audio_queue.get(timeout=timeout)
 except:
 return None

# Usage
streamer = AudioStreamer(sample_rate=16000, chunk_duration_ms=100)
streamer.start()

# Process chunks in real-time
while True:
 chunk = streamer.get_chunk()
 if chunk is not None:
 # Process chunk
 process_audio_chunk(chunk)
``

### Chunk Buffering Strategy

``python
class ChunkBuffer:
 """
 Buffer audio chunks with overlap
 
 Helps models that need context from previous chunks
 """
 
 def __init__(self, buffer_size=3, overlap_size=1):
 """
 Args:
 buffer_size: Number of chunks to keep
 overlap_size: Number of chunks to overlap
 """
 self.buffer_size = buffer_size
 self.overlap_size = overlap_size
 self.chunks = []
 
 def add_chunk(self, chunk):
 """Add new chunk to buffer"""
 self.chunks.append(chunk)
 
 # Keep only recent chunks
 if len(self.chunks) > self.buffer_size:
 self.chunks.pop(0)
 
 def get_buffered_audio(self):
 """
 Get concatenated audio with overlap
 
 Returns: numpy array
 """
 if not self.chunks:
 return None
 
 return np.concatenate(self.chunks)
 
 def get_latest_with_context(self):
 """
 Get latest chunk with context from previous chunks
 
 Useful for models that need history
 """
 if len(self.chunks) < 2:
 return self.chunks[-1] if self.chunks else None
 
 # Return last 'overlap_size + 1' chunks
 context_chunks = self.chunks[-(self.overlap_size + 1):]
 return np.concatenate(context_chunks)

# Usage
buffer = ChunkBuffer(buffer_size=3, overlap_size=1)

for chunk in audio_chunks:
 buffer.add_chunk(chunk)
 audio_with_context = buffer.get_latest_with_context()
 # Process audio with context
``

---

## WebSocket-Based Streaming

### Server Side

``python
import asyncio
import websockets
import json
import numpy as np
import time

class StreamingASRServer:
 """
 WebSocket server for streaming ASR
 
 Clients send audio chunks, server returns transcriptions
 """
 
 def __init__(self, model, port=8765):
 self.model = model
 self.port = port
 self.active_connections = set()
 
 async def handle_client(self, websocket, path):
 """Handle single client connection"""
 client_id = id(websocket)
 self.active_connections.add(websocket)
 print(f"Client {client_id} connected")
 
 try:
 async for message in websocket:
 # Decode message
 data = json.loads(message)
 
 if data['type'] == 'audio':
 # Process audio chunk
 audio_bytes = bytes.fromhex(data['audio'])
 audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
 
 # Run inference
 transcription = await self.process_chunk(audio_chunk)
 
 # Send result
 response = {
 'type': 'transcription',
 'text': transcription,
 'is_final': data.get('is_final', False)
 }
 await websocket.send(json.dumps(response))
 
 elif data['type'] == 'end':
 # Session ended
 break
 
 except websockets.exceptions.ConnectionClosed:
 print(f"Client {client_id} disconnected")
 
 finally:
 self.active_connections.remove(websocket)
 
 async def process_chunk(self, audio_chunk):
 """
 Process audio chunk
 
 Returns: Transcription text
 """
 # Extract features
 # Placeholder feature extractor (should match your model's expected input)
 features = extract_features(audio_chunk)
 
 # Run model inference
 transcription = self.model.predict(features)
 
 return transcription
 
 def start(self):
 """Start WebSocket server"""
 print(f"Starting ASR server on port {self.port}")
 
 start_server = websockets.serve(
 self.handle_client,
 'localhost',
 self.port
 )
 
 asyncio.get_event_loop().run_until_complete(start_server)
 asyncio.get_event_loop().run_forever()

# Usage
server = StreamingASRServer(model=asr_model, port=8765)
server.start()
``

### Client Side

``python
import asyncio
import websockets
import json
import numpy as np

class StreamingASRClient:
 """
 WebSocket client for streaming ASR
 
 Sends audio chunks and receives transcriptions
 """
 
 def __init__(self, server_url='ws://localhost:8765'):
 self.server_url = server_url
 self.websocket = None
 
 async def connect(self):
 """Connect to server"""
 self.websocket = await websockets.connect(self.server_url)
 print(f"Connected to {self.server_url}")
 
 async def send_audio_chunk(self, audio_chunk, is_final=False):
 """
 Send audio chunk to server
 
 Args:
 audio_chunk: numpy array
 is_final: Whether this is the last chunk
 """
 # Convert to bytes
 audio_bytes = audio_chunk.astype(np.float32).tobytes()
 audio_hex = audio_bytes.hex()
 
 # Create message
 message = {
 'type': 'audio',
 'audio': audio_hex,
 'is_final': is_final
 }
 
 # Send
 await self.websocket.send(json.dumps(message))
 
 async def receive_transcription(self):
 """
 Receive transcription from server
 
 Returns: Dict with transcription
 """
 response = await self.websocket.recv()
 return json.loads(response)
 
 async def close(self):
 """Close connection"""
 if self.websocket:
 await self.websocket.send(json.dumps({'type': 'end'}))
 await self.websocket.close()

# Usage
async def stream_audio():
 client = StreamingASRClient()
 await client.connect()
 
 # Stream audio chunks
 streamer = AudioStreamer()
 streamer.start()
 
 try:
 while True:
 chunk = streamer.get_chunk()
 if chunk is None:
 break
 
 # Send chunk
 await client.send_audio_chunk(chunk)
 
 # Receive transcription
 result = await client.receive_transcription()
 print(f"Transcription: {result['text']}")
 
 finally:
 streamer.stop()
 await client.close()

# Run
asyncio.run(stream_audio())
``

---

## Latency Optimization

### Latency Breakdown

``
Total Latency = Audio Capture + Network + Processing + Network + Display

Typical values:
- Audio capture: 10-50ms (chunk duration)
- Network (client → server): 10-30ms
- Feature extraction: 5-10ms
- Model inference: 20-100ms (depends on model)
- Network (server → client): 10-30ms
- Display: 1-5ms

Total: 56-225ms (aim for < 100ms)
``

### Optimization Strategies

``python
class OptimizedStreamingPipeline:
 """
 Optimized streaming pipeline
 
 Techniques:
 - Smaller chunks
 - Model quantization
 - Batch processing
 - Prefetching
 """
 
 def __init__(self, model, chunk_duration_ms=50):
 """
 Args:
 chunk_duration_ms: Smaller chunks = lower latency
 """
 self.model = model
 self.chunk_duration_ms = chunk_duration_ms
 
 # Prefetch buffer
 self.prefetch_buffer = asyncio.Queue(maxsize=3)
 
 # Start prefetching thread
 self.prefetch_task = None
 
 async def start_prefetching(self, audio_source):
 """
 Prefetch audio chunks
 
 Reduces waiting time
 """
 async for chunk in audio_source:
 await self.prefetch_buffer.put(chunk)
 
 async def process_stream(self, audio_source):
 """
 Process audio stream with optimizations
 """
 # Start prefetching
 self.prefetch_task = asyncio.create_task(
 self.start_prefetching(audio_source)
 )
 
 while True:
 # Get prefetched chunk (zero wait!)
 chunk = await self.prefetch_buffer.get()
 
 if chunk is None:
 break
 
 # Process chunk
 result = await self.process_chunk_optimized(chunk)
 
 yield result
 
 async def process_chunk_optimized(self, chunk):
 """
 Optimized chunk processing
 
 Uses quantized model for faster inference
 """
 # Extract features (optimized)
 features = self.extract_features_fast(chunk)
 
 # Run inference (quantized model)
 result = self.model.predict(features)
 
 return result
 
 def extract_features_fast(self, audio):
 """
 Fast feature extraction
 
 Uses caching and vectorization
 """
 # Vectorized operations are faster
 mfcc = librosa.feature.mfcc(
 y=audio,
 sr=16000,
 n_mfcc=13,
 hop_length=160 # Smaller hop = more features
 )
 
 return mfcc
``

---

## State Management

### Stateful Streaming

``python
class StatefulStreamingProcessor:
 """
 Maintain state across chunks
 
 Important for context-dependent models
 """
 
 def __init__(self, model):
 self.model = model
 self.state = None # Hidden state for RNN/LSTM models
 self.previous_chunks = []
 self.partial_results = []
 
 def process_chunk(self, audio_chunk):
 """
 Process chunk with state
 
 Returns: (result, is_complete)
 """
 # Extract features
 features = extract_features(audio_chunk)
 
 # Run model with state
 if hasattr(self.model, 'predict_stateful'):
 result, self.state = self.model.predict_stateful(
 features,
 previous_state=self.state
 )
 else:
 # Fallback: concatenate with previous chunks
 self.previous_chunks.append(audio_chunk)
 if len(self.previous_chunks) > 5:
 self.previous_chunks.pop(0)
 
 combined_audio = np.concatenate(self.previous_chunks)
 combined_features = extract_features(combined_audio)
 result = self.model.predict(combined_features)
 
 # Determine if result is complete
 is_complete = self.check_completeness(result)
 
 if is_complete:
 self.partial_results.append(result)
 
 return result, is_complete
 
 def check_completeness(self, result):
 """
 Check if result is a complete utterance
 
 Uses heuristics:
 - Pause detection
 - Confidence threshold
 - Length limits
 """
 # Simple heuristic: check for pause
 # (In practice, use more sophisticated methods)
 if hasattr(result, 'confidence') and result.confidence > 0.9:
 return True
 
 return False
 
 def reset_state(self):
 """Reset state (e.g., after complete utterance)"""
 self.state = None
 self.previous_chunks = []
 self.partial_results = []
``

---

## Error Handling & Recovery

### Robust Streaming

``python
class RobustStreamingPipeline:
 """
 Streaming pipeline with error handling
 
 Handles:
 - Network failures
 - Audio glitches
 - Model errors
 """
 
 def __init__(self, model):
 self.model = model
 self.error_count = 0
 self.max_errors = 10
 
 async def process_stream_robust(self, audio_source):
 """
 Process stream with error recovery
 """
 retry_count = 0
 max_retries = 3
 
 async for chunk in audio_source:
 try:
 # Process chunk
 result = await self.process_chunk_safe(chunk)
 
 # Reset retry count on success
 retry_count = 0
 
 yield result
 
 except AudioGlitchError as e:
 # Audio glitch: skip chunk
 print(f"Audio glitch detected: {e}")
 self.error_count += 1
 continue
 
 except ModelInferenceError as e:
 # Model error: retry with fallback
 print(f"Model inference failed: {e}")
 
 if retry_count < max_retries:
 retry_count += 1
 # Use simpler fallback model
 result = await self.fallback_inference(chunk)
 yield result
 else:
 # Give up after max retries
 print("Max retries exceeded, skipping chunk")
 retry_count = 0
 
 except Exception as e:
 # Unexpected error
 print(f"Unexpected error: {e}")
 self.error_count += 1
 
 if self.error_count > self.max_errors:
 raise RuntimeError("Too many errors, stopping stream")
 
 async def process_chunk_safe(self, chunk):
 """
 Process chunk with validation
 """
 # Validate chunk
 if not self.validate_chunk(chunk):
 raise AudioGlitchError("Invalid audio chunk")
 
 # Process
 try:
 features = extract_features(chunk)
 result = self.model.predict(features)
 return result
 except Exception as e:
 raise ModelInferenceError(f"Inference failed: {e}")
 
 def validate_chunk(self, chunk):
 """
 Validate audio chunk
 
 Checks for:
 - Correct length
 - Valid range
 - No NaN values
 """
 if chunk is None or len(chunk) == 0:
 return False
 
 if np.any(np.isnan(chunk)):
 return False
 
 if np.max(np.abs(chunk)) > 10: # Suspiciously large
 return False
 
 return True
 
 async def fallback_inference(self, chunk):
 """
 Fallback inference with simpler model
 
 Trades accuracy for reliability
 """
 # Use cached results or simple heuristics
 return {"text": "[processing...]", "confidence": 0.5}

class AudioGlitchError(Exception):
 pass

class ModelInferenceError(Exception):
 pass
``

---

## Connection to Model Serving (ML)

Streaming speech pipelines use model serving patterns:

``python
class StreamingSpeechServer:
 """
 Streaming speech server with model serving best practices
 
 Combines:
 - Model serving (ML)
 - Streaming audio (Speech)
 - Validation (DSA)
 """
 
 def __init__(self, model_path):
 # Load model (model serving pattern)
 self.model = self.load_model(model_path)
 
 # Validation (BST-like range checking)
 self.validator = AudioValidator()
 
 # Monitoring
 self.metrics = StreamingMetrics()
 
 def load_model(self, model_path):
 """Load model with caching (from model serving)"""
 import joblib
 return joblib.load(model_path)
 
 async def process_audio_stream(self, audio_chunks):
 """
 Process streaming audio
 
 Uses patterns from all topics
 """
 for chunk in audio_chunks:
 # Validate input (BST validation pattern)
 is_valid, violations = self.validator.validate(chunk)
 
 if not is_valid:
 print(f"Invalid chunk: {violations}")
 continue
 
 # Process chunk (model serving)
 start_time = time.time()
 result = self.model.predict(chunk)
 latency = time.time() - start_time
 
 # Monitor (model serving)
 self.metrics.record_prediction(latency)
 
 yield result

class AudioValidator:
 """Validate audio chunks (similar to BST validation)"""
 
 def __init__(self):
 # Define valid ranges (like BST min/max)
 self.amplitude_range = (-1.0, 1.0)
 self.length_range = (100, 10000) # samples
 
 def validate(self, chunk):
 """
 Validate chunk falls within ranges
 
 Like BST validation with [min, max] bounds
 """
 violations = []
 
 # Check amplitude range
 if np.min(chunk) < self.amplitude_range[0]:
 violations.append("Amplitude too low")
 if np.max(chunk) > self.amplitude_range[1]:
 violations.append("Amplitude too high")
 
 # Check length range
 if len(chunk) < self.length_range[0]:
 violations.append("Chunk too short")
 if len(chunk) > self.length_range[1]:
 violations.append("Chunk too long")
 
 return len(violations) == 0, violations
``

---

## Production Patterns

### 1. Multi-Channel Audio Streaming

``python
class MultiChannelStreamingProcessor:
 """
 Process multiple audio streams simultaneously
 
 Use case: Conference calls, multi-mic arrays
 """
 
 def __init__(self, num_channels=4):
 self.num_channels = num_channels
 self.channel_buffers = [ChunkBuffer() for _ in range(num_channels)]
 self.processors = [StreamingProcessor() for _ in range(num_channels)]
 
 async def process_multi_channel(self, channel_chunks: dict):
 """
 Process multiple channels in parallel
 
 Args:
 channel_chunks: Dict {channel_id: audio_chunk}
 
 Returns: Dict {channel_id: result}
 """
 import asyncio
 
 # Process channels in parallel
 tasks = []
 for channel_id, chunk in channel_chunks.items():
 task = self.processors[channel_id].process_chunk_async(chunk)
 tasks.append((channel_id, task))
 
 # Wait for all results
 results = {}
 for channel_id, task in tasks:
 result = await task
 results[channel_id] = result
 
 return results
 
 def merge_results(self, channel_results: dict):
 """
 Merge results from multiple channels
 
 E.g., speaker diarization, beam forming
 """
 # Simple merging: concatenate transcriptions
 merged_text = []
 
 for channel_id in sorted(channel_results.keys()):
 result = channel_results[channel_id]
 if result:
 merged_text.append(f"[Channel {channel_id}]: {result['text']}")
 
 return '\n'.join(merged_text)

# Usage
multi_processor = MultiChannelStreamingProcessor(num_channels=4)

# Stream audio from 4 microphones
async def process_meeting():
 while True:
 # Get chunks from all channels
 chunks = {
 0: mic1.get_chunk(),
 1: mic2.get_chunk(),
 2: mic3.get_chunk(),
 3: mic4.get_chunk()
 }
 
 # Process in parallel
 results = await multi_processor.process_multi_channel(chunks)
 
 # Merge and display
 merged = multi_processor.merge_results(results)
 print(merged)
``

### 2. Adaptive Chunk Size

``python
class AdaptiveChunkingProcessor:
 """
 Dynamically adjust chunk size based on network/compute conditions
 
 Smaller chunks: Lower latency but higher overhead
 Larger chunks: Higher latency but more efficient
 """
 
 def __init__(self, min_chunk_ms=50, max_chunk_ms=200):
 self.min_chunk_ms = min_chunk_ms
 self.max_chunk_ms = max_chunk_ms
 self.current_chunk_ms = 100 # Start with middle value
 self.latency_history = []
 
 def adjust_chunk_size(self, recent_latency_ms):
 """
 Adjust chunk size based on latency
 
 High latency → smaller chunks (more responsive)
 Low latency → larger chunks (more efficient)
 """
 self.latency_history.append(recent_latency_ms)
 
 if len(self.latency_history) < 10:
 return self.current_chunk_ms
 
 # Calculate average latency
 avg_latency = np.mean(self.latency_history[-10:])
 
 # Adjust chunk size
 if avg_latency > 150: # High latency
 # Reduce chunk size for better responsiveness
 self.current_chunk_ms = max(
 self.min_chunk_ms,
 self.current_chunk_ms - 10
 )
 print(f"↓ Reducing chunk size to {self.current_chunk_ms}ms")
 
 elif avg_latency < 50: # Very low latency
 # Increase chunk size for efficiency
 self.current_chunk_ms = min(
 self.max_chunk_ms,
 self.current_chunk_ms + 10
 )
 print(f"↑ Increasing chunk size to {self.current_chunk_ms}ms")
 
 return self.current_chunk_ms
 
 async def process_with_adaptive_chunking(self, audio_stream):
 """Process stream with adaptive chunk sizing"""
 for chunk in audio_stream:
 start_time = time.time()
 
 # Process chunk
 result = await self.process_chunk(chunk)
 
 # Calculate latency
 latency_ms = (time.time() - start_time) * 1000
 
 # Adjust chunk size for next iteration
 next_chunk_ms = self.adjust_chunk_size(latency_ms)
 
 yield result, next_chunk_ms
``

### 3. Buffering Strategy for Unreliable Networks

``python
class NetworkAwareStreamingBuffer:
 """
 Buffer audio to handle network issues
 
 Maintains smooth playback despite packet loss
 """
 
 def __init__(self, buffer_size_seconds=2.0, sample_rate=16000):
 self.buffer_size = int(buffer_size_seconds * sample_rate)
 self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
 self.write_pos = 0
 self.read_pos = 0
 self.underrun_count = 0
 self.overrun_count = 0
 
 def write_chunk(self, chunk):
 """
 Write audio chunk to buffer
 
 Returns: Success status
 """
 chunk_size = len(chunk)
 
 # Check for buffer overrun
 available_space = self.buffer_size - (self.write_pos - self.read_pos)
 if chunk_size > available_space:
 self.overrun_count += 1
 print("⚠️ Buffer overrun - dropping oldest data")
 # Drop oldest data
 self.read_pos = self.write_pos - self.buffer_size + chunk_size
 
 # Write to circular buffer
 for i, sample in enumerate(chunk):
 pos = (self.write_pos + i) % self.buffer_size
 self.buffer[pos] = sample
 
 self.write_pos += chunk_size
 return True
 
 def read_chunk(self, chunk_size):
 """
 Read audio chunk from buffer
 
 Returns: Audio chunk or None if underrun
 """
 # Check for buffer underrun
 available_data = self.write_pos - self.read_pos
 if available_data < chunk_size:
 self.underrun_count += 1
 print("⚠️ Buffer underrun - not enough data")
 return None
 
 # Read from circular buffer
 chunk = np.zeros(chunk_size, dtype=np.float32)
 for i in range(chunk_size):
 pos = (self.read_pos + i) % self.buffer_size
 chunk[i] = self.buffer[pos]
 
 self.read_pos += chunk_size
 return chunk
 
 def get_buffer_level(self):
 """Get current buffer fill level (0-1)"""
 available = self.write_pos - self.read_pos
 return available / self.buffer_size
 
 def get_stats(self):
 """Get buffer statistics"""
 return {
 'buffer_level': self.get_buffer_level(),
 'underruns': self.underrun_count,
 'overruns': self.overrun_count
 }

# Usage
buffer = NetworkAwareStreamingBuffer(buffer_size_seconds=2.0)

# Writer thread (receiving from network)
async def receive_audio():
 async for chunk in network_stream:
 buffer.write_chunk(chunk)
 
 # Adaptive buffering
 level = buffer.get_buffer_level()
 if level < 0.2:
 print("⚠️ Low buffer, may need to increase")

# Reader thread (processing)
async def process_audio():
 while True:
 chunk = buffer.read_chunk(chunk_size=1600) # 100ms at 16kHz
 if chunk is not None:
 result = await process_chunk(chunk)
 yield result
 else:
 await asyncio.sleep(0.01) # Wait for more data
``

---

## Advanced Optimization Techniques

### 1. Model Warm-Up

``python
class WarmUpStreamingProcessor:
 """
 Pre-warm model for lower latency on first request
 
 Cold start can add 100-500ms latency
 """
 
 def __init__(self, model):
 self.model = model
 self.is_warm = False
 
 def warm_up(self, sample_rate=16000):
 """
 Warm up model with dummy input
 
 Call during initialization
 """
 print("Warming up model...")
 
 # Create dummy audio chunk
 dummy_chunk = np.random.randn(int(sample_rate * 0.1)) # 100ms
 
 # Run inference to warm up
 for _ in range(3):
 _ = self.model.predict(dummy_chunk)
 
 self.is_warm = True
 print("Model warm-up complete")
 
 def process_chunk(self, chunk):
 """Process with warm-up check"""
 if not self.is_warm:
 self.warm_up()
 
 return self.model.predict(chunk)

# Usage
processor = WarmUpStreamingProcessor(model)
processor.warm_up() # Do this during server startup
``

### 2. GPU Batching for Throughput

``python
class GPUBatchProcessor:
 """
 Batch multiple streams for GPU efficiency
 
 GPUs are most efficient with batch processing
 """
 
 def __init__(self, model, max_batch_size=16, max_wait_ms=50):
 self.model = model
 self.max_batch_size = max_batch_size
 self.max_wait_ms = max_wait_ms
 self.pending_batches = []
 
 async def process_chunk_batched(self, chunk, stream_id):
 """
 Add chunk to batch and process when ready
 
 Returns: Future that resolves with result
 """
 future = asyncio.Future()
 self.pending_batches.append((chunk, stream_id, future))
 
 # Process batch if ready
 if len(self.pending_batches) >= self.max_batch_size:
 await self._process_batch()
 else:
 # Wait for more requests or timeout
 asyncio.create_task(self._process_batch_after_delay())
 
 return await future
 
 async def _process_batch(self):
 """Process accumulated batch on GPU"""
 if not self.pending_batches:
 return
 
 # Extract batch
 chunks = [item[0] for item in self.pending_batches]
 stream_ids = [item[1] for item in self.pending_batches]
 futures = [item[2] for item in self.pending_batches]
 
 # Pad to same length
 max_len = max(len(c) for c in chunks)
 padded_chunks = [
 np.pad(c, (0, max_len - len(c)), mode='constant')
 for c in chunks
 ]
 
 # Stack into batch
 batch = np.stack(padded_chunks)
 
 # Run batch inference on GPU
 results = self.model.predict_batch(batch)
 
 # Distribute results
 for result, future in zip(results, futures):
 future.set_result(result)
 
 # Clear batch
 self.pending_batches = []
 
 async def _process_batch_after_delay(self):
 """Process batch after timeout"""
 await asyncio.sleep(self.max_wait_ms / 1000.0)
 await self._process_batch()

# Usage
gpu_processor = GPUBatchProcessor(model, max_batch_size=16)

# Multiple concurrent streams
async def process_stream(stream_id):
 async for chunk in audio_streams[stream_id]:
 result = await gpu_processor.process_chunk_batched(chunk, stream_id)
 yield result

# Run multiple streams in parallel
await asyncio.gather(*[
 process_stream(i) for i in range(10)
])
``

### 3. Quantized Models for Edge Devices

``python
import torch
import torchaudio

class EdgeOptimizedStreamingASR:
 """
 Streaming ASR optimized for edge devices
 
 Uses INT8 quantization for faster inference
 """
 
 def __init__(self, model_path):
 # Load and quantize model
 self.model = torch.jit.load(model_path)
 self.model = torch.quantization.quantize_dynamic(
 self.model,
 {torch.nn.Linear, torch.nn.LSTM},
 dtype=torch.qint8
 )
 self.model.eval()
 
 def process_chunk_optimized(self, audio_chunk):
 """
 Process chunk with optimizations
 
 - INT8 quantization: 4x faster
 - No gradient computation
 - Minimal memory allocation
 """
 with torch.no_grad():
 # Convert to tensor
 audio_tensor = torch.from_numpy(audio_chunk).float()
 audio_tensor = audio_tensor.unsqueeze(0) # Add batch dim
 
 # Extract features (optimized)
 features = torchaudio.compliance.kaldi.mfcc(
 audio_tensor,
 sample_frequency=16000,
 num_ceps=13
 )
 
 # Run inference
 output = self.model(features)
 
 # Decode
 transcription = self.decode(output)
 
 return transcription
 
 def decode(self, output):
 """Simple greedy decoding"""
 # Get most likely tokens
 tokens = torch.argmax(output, dim=-1)
 
 # Convert to text (simplified)
 transcription = self.tokens_to_text(tokens)
 
 return transcription

# Benchmark: Quantized vs Full Precision
def benchmark_models():
 """Compare quantized vs full precision"""
 full_model = load_model('model_fp32.pt')
 quant_model = EdgeOptimizedStreamingASR('model_int8.pt')
 
 audio_chunk = np.random.randn(1600) # 100ms at 16kHz
 
 # Full precision
 start = time.time()
 for _ in range(100):
 _ = full_model.predict(audio_chunk)
 fp32_time = time.time() - start
 
 # Quantized
 start = time.time()
 for _ in range(100):
 _ = quant_model.process_chunk_optimized(audio_chunk)
 int8_time = time.time() - start
 
 print(f"FP32: {fp32_time:.2f}s")
 print(f"INT8: {int8_time:.2f}s")
 print(f"Speedup: {fp32_time / int8_time:.1f}x")
``

---

## Real-World Integration Examples

### 1. Zoom-like Meeting Transcription

``python
class MeetingTranscriptionService:
 """
 Real-time meeting transcription
 
 Similar to Zoom's live transcription
 """
 
 def __init__(self):
 self.asr_model = load_asr_model()
 self.active_sessions = {}
 
 def start_session(self, meeting_id):
 """Start transcription session"""
 self.active_sessions[meeting_id] = {
 'participants': {},
 'transcript': [],
 'start_time': time.time()
 }
 
 async def process_participant_audio(self, meeting_id, participant_id, audio_stream):
 """
 Process audio from single participant
 
 Returns: Real-time transcription
 """
 session = self.active_sessions[meeting_id]
 
 # Initialize participant
 if participant_id not in session['participants']:
 session['participants'][participant_id] = {
 'processor': StatefulStreamingProcessor(self.asr_model),
 'transcript_buffer': []
 }
 
 participant = session['participants'][participant_id]
 processor = participant['processor']
 
 async for chunk in audio_stream:
 # Process chunk
 result, is_complete = processor.process_chunk(chunk)
 
 if is_complete:
 # Add to transcript
 timestamp = time.time() - session['start_time']
 transcript_entry = {
 'participant_id': participant_id,
 'text': result['text'],
 'timestamp': timestamp,
 'confidence': result.get('confidence', 1.0)
 }
 
 session['transcript'].append(transcript_entry)
 
 yield transcript_entry
 
 def get_full_transcript(self, meeting_id):
 """Get complete meeting transcript"""
 if meeting_id not in self.active_sessions:
 return []
 
 transcript = self.active_sessions[meeting_id]['transcript']
 
 # Format as readable text
 formatted = []
 for entry in transcript:
 time_str = format_timestamp(entry['timestamp'])
 formatted.append(
 f"[{time_str}] Participant {entry['participant_id']}: {entry['text']}"
 )
 
 return '\n'.join(formatted)

def format_timestamp(seconds):
 """Format seconds as MM:SS"""
 minutes = int(seconds // 60)
 secs = int(seconds % 60)
 return f"{minutes:02d}:{secs:02d}"

# Usage
service = MeetingTranscriptionService()
service.start_session('meeting-123')

# Process audio from multiple participants
async def transcribe_meeting():
 participants = ['user1', 'user2', 'user3']
 
 # Process all participants in parallel
 tasks = [
 service.process_participant_audio(
 'meeting-123',
 participant_id,
 get_audio_stream(participant_id)
 )
 for participant_id in participants
 ]
 
 # Collect transcriptions
 # Collect tasks concurrently
 results = await asyncio.gather(*tasks, return_exceptions=True)
 for res in results:
 if isinstance(res, Exception):
 print(f"Stream error: {res}")
 else:
 for entry in res:
 print(f"[{entry['timestamp']:.1f}s] {entry['participant_id']}: {entry['text']}")
``

### 2. Voice Assistant Backend

``python
class VoiceAssistantPipeline:
 """
 Complete voice assistant pipeline
 
 ASR → NLU → Action → TTS
 """
 
 def __init__(self):
 self.asr = StreamingASR()
 self.nlu = IntentClassifier()
 self.action_executor = ActionExecutor()
 self.tts = TextToSpeech()
 
 async def process_voice_command(self, audio_stream):
 """
 Process voice command end-to-end
 
 Returns: Audio response
 """
 # 1. Speech Recognition
 transcription = await self.asr.transcribe_stream(audio_stream)
 print(f"User said: {transcription}")
 
 # 2. Natural Language Understanding
 intent = self.nlu.classify(transcription)
 print(f"Intent: {intent['name']} (confidence: {intent['confidence']:.2f})")
 
 # 3. Execute Action
 if intent['confidence'] > 0.7:
 response_text = await self.action_executor.execute(intent)
 else:
 response_text = "I'm not sure what you mean. Could you rephrase that?"
 
 # 4. Text-to-Speech
 response_audio = self.tts.synthesize(response_text)
 
 return {
 'transcription': transcription,
 'intent': intent,
 'response_text': response_text,
 'response_audio': response_audio
 }
 
 async def continuous_listening(self, audio_source):
 """
 Continuously listen for wake word + command
 
 Efficient always-on listening
 """
 wake_word_detector = WakeWordDetector('hey assistant')
 
 async for chunk in audio_source:
 # Check for wake word (lightweight model)
 if wake_word_detector.detect(chunk):
 print("🎤 Wake word detected!")
 
 # Start full ASR
 command_audio = await self.capture_command(audio_source, timeout=5.0)
 
 # Process command
 result = await self.process_voice_command(command_audio)
 
 # Play response
 play_audio(result['response_audio'])
 
 async def capture_command(self, audio_source, timeout=5.0):
 """Capture audio command after wake word"""
 command_chunks = []
 start_time = time.time()
 
 async for chunk in audio_source:
 command_chunks.append(chunk)
 
 # Check timeout
 if time.time() - start_time > timeout:
 break
 
 # Check for end of speech (silence)
 if self.is_silence(chunk):
 break
 
 return np.concatenate(command_chunks)
 
 def is_silence(self, chunk, threshold=0.01):
 """Detect if chunk is silence"""
 energy = np.sqrt(np.mean(chunk ** 2))
 return energy < threshold

# Usage
assistant = VoiceAssistantPipeline()

# Continuous listening
await assistant.continuous_listening(microphone_stream)
``

---

## Performance Metrics & SLAs

### Latency Tracking

``python
class StreamingLatencyTracker:
 """
 Track end-to-end latency for streaming pipeline
 
 Measures:
 - Audio capture latency
 - Network latency
 - Processing latency
 - Total latency
 """
 
 def __init__(self):
 self.metrics = {
 'capture_latency': [],
 'network_latency': [],
 'processing_latency': [],
 'total_latency': []
 }
 
 async def process_with_tracking(self, audio_chunk, capture_timestamp):
 """
 Process chunk with latency tracking
 
 Args:
 audio_chunk: Audio data
 capture_timestamp: When audio was captured
 
 Returns: (result, latency_breakdown)
 """
 # Network latency (time from capture to arrival)
 network_start = time.time()
 network_latency = (network_start - capture_timestamp) * 1000
 self.metrics['network_latency'].append(network_latency)
 
 # Processing latency
 processing_start = time.time()
 result = await self.process_chunk(audio_chunk)
 processing_end = time.time()
 processing_latency = (processing_end - processing_start) * 1000
 self.metrics['processing_latency'].append(processing_latency)
 
 # Total latency
 total_latency = (processing_end - capture_timestamp) * 1000
 self.metrics['total_latency'].append(total_latency)
 
 latency_breakdown = {
 'network_ms': network_latency,
 'processing_ms': processing_latency,
 'total_ms': total_latency
 }
 
 return result, latency_breakdown
 
 def get_latency_stats(self):
 """Get latency statistics"""
 stats = {}
 
 for metric_name, values in self.metrics.items():
 if values:
 stats[metric_name] = {
 'p50': np.percentile(values, 50),
 'p95': np.percentile(values, 95),
 'p99': np.percentile(values, 99),
 'mean': np.mean(values),
 'max': np.max(values)
 }
 
 return stats
 
 def check_sla(self, sla_ms=100):
 """
 Check if meeting SLA
 
 Returns: (is_meeting_sla, violation_rate)
 """
 if not self.metrics['total_latency']:
 return True, 0.0
 
 violations = sum(1 for lat in self.metrics['total_latency'] if lat > sla_ms)
 violation_rate = violations / len(self.metrics['total_latency'])
 
 is_meeting_sla = violation_rate < 0.01 # < 1% violations
 
 return is_meeting_sla, violation_rate

# Usage
tracker = StreamingLatencyTracker()

# Process with tracking
result, latency = await tracker.process_with_tracking(chunk, capture_time)

# Check SLA
is_ok, violation_rate = tracker.check_sla(sla_ms=100)
if not is_ok:
 print(f"⚠️ SLA violation rate: {violation_rate:.1%}")

# Get detailed stats
stats = tracker.get_latency_stats()
print(f"P95 latency: {stats['total_latency']['p95']:.1f}ms")
``

---

## Key Takeaways

✅ **Chunk audio correctly** - Balance latency vs context 
✅ **Manage state** - RNN/LSTM models need previous chunks 
✅ **Optimize latency** - Smaller chunks, quantization, prefetching 
✅ **Handle errors gracefully** - Network failures, audio glitches 
✅ **Validate inputs** - Like BST range checking 
✅ **Monitor performance** - Latency, error rate, throughput 
✅ **WebSocket for streaming** - Bidirectional, low-latency 

---

**Originally published at:** [arunbaby.com/speech-tech/0008-streaming-speech-pipeline](https://www.arunbaby.com/speech-tech/0008-streaming-speech-pipeline/)

*If you found this helpful, consider sharing it with others who might benefit.*

