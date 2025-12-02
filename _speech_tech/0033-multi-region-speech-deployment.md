---
title: "Multi-region Speech Deployment"
day: 33
collection: speech_tech
categories:
  - speech_tech
tags:
  - deployment
  - distributed-systems
  - asr
  - tts
  - edge-computing
subdomain: "Infrastructure"
tech_stack: [Kubernetes, gRPC, WebRTC, CDN, S3]
scale: "Global, multi-region, edge deployment"
companies: [Google, Amazon, Apple, Microsoft]
related_dsa_day: 33
related_ml_day: 33
---

**"Deploying speech models close to users for low-latency voice experiences."**

## 1. The Challenge: Speech Requires Real-Time Performance

Speech applications have **strict latency requirements**:
- **Voice Assistants:** Users expect responses in < 300ms.
- **Live Captioning:** Must transcribe as the speaker talks (< 100ms lag).
- **Voice Calls:** Any delay > 150ms is noticeable and disruptive.

**Why Multi-Region Deployment?**
- A single data center in Virginia serves users in Tokyo with 250ms network latency.
- Deploying ASR models in Tokyo reduces latency to 20ms.

## 2. Multi-Region Architecture Overview

**Architecture:**
```
        Global Load Balancer (AWS Route 53, Cloudflare)
                    |
    ---------------------------------
    |               |               |
  US-West         EU-West        Asia-Pacific
  (Oregon)        (Frankfurt)     (Tokyo)
    |               |               |
  ASR Model       ASR Model       ASR Model
  TTS Model       TTS Model       TTS Model
  Speaker ID      Speaker ID      Speaker ID
```

**Key Components:**
1. **Global Load Balancer:** Routes users to the nearest region (GeoDNS).
2. **Regional Clusters:** Each region has a full stack of speech models.
3. **Model Sync:** Ensures all regions serve the same model version.

## 3. GeoDNS Routing

**Concept:** Direct users to the closest data center based on their geographic location.

**Implementation:**
- **AWS Route 53:** Geolocation routing policy.
- **Cloudflare:** Automatic geo-routing via Anycast.

**Example (Route 53):**
```json
{
  "Name": "speech-api.example.com",
  "Type": "A",
  "GeoLocation": {
    "ContinentCode": "NA"
  },
  "ResourceRecords": [
    {"Value": "3.12.45.67"}  // US-West IP
  ]
}
```

Users in North America get routed to `3.12.45.67` (US-West).
Users in Europe get routed to EU-West.

**Pros:**
- Reduces latency by 80%+.
- Automatic failover (if a region goes down, route to the next closest).

**Cons:**
- DNS caching (TTL = 60s) can delay failover.

## 4. Edge Deployment for Ultra-Low Latency

For applications like voice calls or gaming, even 50ms is too much. Deploy models at the **edge** (CDN points of presence).

**Edge Locations:**
- AWS has 400+ edge locations.
- Cloudflare has 300+ PoPs.

**Architecture:**
```
User in Berlin → Cloudflare PoP (Berlin) → ASR Model (Frankfurt)
                       ↑
                  Model cached at edge
```

**Implementation (AWS Lambda@Edge):**
```python
import json

def lambda_handler(event, context):
    # Run lightweight ASR model at edge
    audio_data = event['body']
    transcript = edge_asr_model.predict(audio_data)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'transcript': transcript})
    }
```

**Trade-offs:**
- **Pros:** < 10ms latency.
- **Cons:** Edge environments have limited compute (no GPU). Must use quantized/lightweight models.

## 5. Deep Dive: Model Synchronization Across Regions

**Challenge:** You train a new ASR model in US-West. How do you deploy it to 10 regions?

### Strategy 1: Centralized Model Registry
A single S3 bucket (replicated globally) stores all model versions.

**Flow:**
1. Upload model to `s3://global-models/asr-v100.pt`.
2. S3 replicates to all regions (automated cross-region replication).
3. Each regional cluster pulls the latest model.

**AWS S3 Cross-Region Replication:**
```json
{
  "Role": "arn:aws:iam::123456789:role/s3-replication",
  "Rules": [
    {
      "Status": "Enabled",
      "Priority": 1,
      "Filter": {"Prefix": "models/"},
      "Destination": {
        "Bucket": "arn:aws:s3:::models-eu-west",
        "ReplicationTime": {"Status": "Enabled", "Time": {"Minutes": 15}}
      }
    }
  ]
}
```

Models replicate within 15 minutes.

### Strategy 2: Regional Model Stores
Each region has its own S3 bucket. A deployment pipeline copies models to all buckets.

**Terraform Script:**
```hcl
resource "aws_s3_bucket" "model_store" {
  for_each = toset(["us-west-2", "eu-west-1", "ap-southeast-1"])
  bucket   = "speech-models-${each.value}"
  region   = each.value
}

resource "aws_s3_bucket_object" "model" {
  for_each = aws_s3_bucket.model_store
  bucket   = each.value.id
  key      = "asr-v100.pt"
  source   = "models/asr-v100.pt"
}
```

**Pros:** Independent regions (failure in one doesn't affect others).
**Cons:** Deployment latency (sequential uploads).

## 6. Deep Dive: Handling Regional Data Compliance (GDPR)

**Problem:** EU regulations (GDPR) require that user audio data stays in the EU.

**Architecture:**
```
EU User → EU Load Balancer → EU ASR Model → EU Storage
   ↓
Audio NEVER leaves EU
```

**Implementation:**
- **Network Policies:** Block cross-region traffic from EU to US.
- **IAM Roles:** EU instances can only access EU S3 buckets.

```python
# In EU region only
AWS_REGION = "eu-west-1"
s3_client = boto3.client('s3', region_name=AWS_REGION)

# This will fail if model is in US bucket
model = s3_client.get_object(Bucket='models-us-west', Key='asr.pt')
# Error: Access Denied
```

**Separate Training Pipelines:**
- **EU Model:** Trained only on EU user data.
- **US Model:** Trained only on US user data.
- Models may have different vocabularies (accents, slang).

## 7. Deep Dive: Canary Deployment in Multi-Region

**Scenario:** Deploy a new TTS model to 5 regions.

**Strategy:**
1. **Deploy to 1% of servers in US-West** (canary).
2. Monitor for 24 hours (latency, error rate, voice quality scores).
3. If healthy, roll out to **all servers in US-West**.
4. If still healthy, roll out to **EU-West**, then **Asia-Pacific**.

**Kubernetes Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tts-v2-canary
  namespace: us-west
spec:
  replicas: 1  # 1% of 100 total replicas
  selector:
    matchLabels:
      app: tts
      version: v2
  template:
    spec:
      containers:
      - name: tts
        image: tts:v2
```

**Traffic Split (Istio):**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: tts-service
spec:
  hosts:
  - tts.example.com
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: tts
        subset: v2
  - route:
    - destination:
        host: tts
        subset: v1
      weight: 99
    - destination:
        host: tts
        subset: v2
      weight: 1
```

## 8. Deep Dive: Fallback and Disaster Recovery

**Scenario:** The EU-West data center goes offline (power outage).

### Fallback Strategy 1: Route to Nearest Healthy Region
```
EU User → (EU-West DOWN) → GeoDNS → US-East
```

**Cons:** Increased latency (50ms → 150ms).

### Fallback Strategy 2: Multi-Region Active-Active
Deploy to **2 regions per continent**.

```
EU User → EU-West (Primary) + EU-Central (Backup)
```

If EU-West fails, EU-Central takes over instantly.

**Cost:** 2x infrastructure in each continent.

## 9. Deep Dive: Caching at Edge and Regional Layers

Speech models are compute-intensive. Caching can reduce load.

**What to Cache:**
1. **TTS Output:** User says "What's the weather?" every morning.
   - Cache the audio file for "It's 72°F and sunny" for that user.
2. **Common Queries:** "Set a timer for 10 minutes" is a frequent request.
   - Precompute ASR + NLU results.

**Redis Caching:**
```python
import redis
import hashlib

redis_client = redis.Redis()

def tts_with_cache(text):
    cache_key = hashlib.md5(text.encode()).hexdigest()
    
    # Check cache
    cached_audio = redis_client.get(cache_key)
    if cached_audio:
        return cached_audio
    
    # Generate TTS
    audio = tts_model.synthesize(text)
    
    # Store in cache (TTL = 1 hour)
    redis_client.setex(cache_key, 3600, audio)
    
    return audio
```

**Pros:** Reduces TTS latency from 200ms to 5ms (cache hit).
**Cons:** Stale data (if model updates, cache must be invalidated).

## 10. Deep Dive: Model Compression for Edge Deployment

Edge devices (smartphones, smart speakers) have limited compute. Deploy **quantized models**.

**Quantization:**
- Convert FP32 weights to INT8.
- Model size: 500 MB → 125 MB.
- Inference speed: 2x faster.

**PyTorch Quantization:**
```python
import torch

# Load original model
model = torch.load('asr_fp32.pt')

# Quantize to INT8
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save
torch.save(quantized_model, 'asr_int8.pt')
```

**Accuracy Drop:** Typically < 1% WER increase.

## 11. Deep Dive: Monitoring Multi-Region Deployments

**Metrics to Track:**
1. **Latency per Region:** P50, P99 latency for ASR inference.
2. **Error Rate per Region:** % of failed requests.
3. **Model Version:** Which version is running in each region?
4. **Traffic Distribution:** % of traffic per region.

**Prometheus Metrics:**
```python
from prometheus_client import Counter, Histogram

asr_requests = Counter('asr_requests_total', 'Total ASR requests', ['region', 'model_version'])
asr_latency = Histogram('asr_latency_seconds', 'ASR latency', ['region'])

@app.post("/asr")
def transcribe(audio: bytes):
    region = get_region()
    model_version = get_model_version()
    
    asr_requests.labels(region=region, model_version=model_version).inc()
    
    with asr_latency.labels(region=region).time():
        transcript = asr_model.predict(audio)
    
    return {"transcript": transcript}
```

**Grafana Dashboard:**
- **Map View:** Show latency heatmap by region.
- **Alerts:** If EU-West p99 > 500ms, send alert.

## 12. Real-World Case Studies

### Case Study 1: Google Assistant Multi-Region
Google deploys ASR models to **20+ regions**.

**Architecture:**
- Each region has a cluster of GPU servers.
- Models stored in regional Google Cloud Storage buckets.
- Canary deployments in US-Central1 first, then global rollout.

**Result:** < 100ms latency for 95% of users globally.

### Case Study 2: Amazon Alexa Edge Deployment
Alexa uses a hybrid approach:
- **Wake Word Detection:** Runs on-device (edge).
- **Full ASR:** Runs in the cloud (regional clusters).

**Why?**
- Wake word detection is lightweight (< 1 MB model).
- Full ASR is heavy (> 500 MB model).

### Case Study 3: Zoom's Real-Time Transcription
Zoom deploys ASR models in **17 AWS regions**.

**Strategy:**
- Each meeting is routed to the closest region.
- If the region is overloaded, fallback to the next closest.
- Models updated weekly via blue-green deployment.

## Implementation: Multi-Region Speech API

**Step 1: Dockerize the ASR Model**
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY asr_model.pt .
COPY serve.py .
EXPOSE 8000
CMD ["python", "serve.py"]
```

**Step 2: Deploy to Multi-Region Kubernetes**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: asr-us-west
  namespace: us-west-2
spec:
  replicas: 10
  template:
    spec:
      containers:
      - name: asr
        image: my-registry/asr:v1
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: asr-eu-west
  namespace: eu-west-1
spec:
  replicas: 10
  template:
    spec:
      containers:
      - name: asr
        image: my-registry/asr:v1
        resources:
          limits:
            nvidia.com/gpu: 1
```

**Step 3: Global Load Balancer (Cloudflare)**
```bash
curl -X POST "https://api.cloudflare.com/client/v4/zones/{zone_id}/load_balancers" \
  -H "Authorization: Bearer {api_token}" \
  -d '{
    "name": "speech-api.example.com",
    "default_pools": ["us-west", "eu-west", "asia-pacific"],
    "region_pools": {
      "WNAM": ["us-west"],
      "EEUR": ["eu-west"],
      "SEAS": ["asia-pacific"]
    }
  }'
```

## Top Interview Questions

**Q1: How do you ensure low latency for global users?**
*Answer:*
- **GeoDNS:** Route users to the nearest region.
- **Edge Deployment:** Deploy lightweight models at CDN edge locations.
- **Caching:** Cache frequent TTS outputs and ASR results.

**Q2: What happens if a region goes down?**
*Answer:*
- **Failover:** GeoDNS automatically routes to the next closest healthy region.
- **Active-Active:** Deploy to multiple regions per continent for redundancy.
- **Monitoring:** Real-time health checks detect failures within seconds.

**Q3: How do you handle model updates across 10 regions?**
*Answer:*
- **Canary Deployment:** Roll out to 1% in one region first.
- **Progressive Rollout:** If healthy, deploy to all regions sequentially.
- **Automated Rollback:** If error rate spikes, revert to the previous version.

**Q4: How do you comply with GDPR for EU users?**
*Answer:*
- **Regional Isolation:** EU audio data never leaves EU servers.
- **Separate Models:** Train EU-specific models on EU data.
- **Network Policies:** Block cross-region traffic from EU to other regions.

**Q5: What's the difference between edge and regional deployment?**
*Answer:*
- **Regional:** Models run in data centers (full GPU, high compute).
- **Edge:** Models run at CDN PoPs (limited compute, quantized models).
- **Use Case:** Edge for ultra-low latency (< 10ms), Regional for high accuracy.

## 13. Deep Dive: Streaming ASR with WebRTC

For real-time applications (video calls, live captioning), audio streams chunk-by-chunk over WebRTC.

**Challenge:** Each audio chunk arrives every 20ms. ASR must process faster than real-time.

**Architecture:**
```
User Microphone
    ↓
WebRTC Stream (20ms chunks)
    ↓
Regional ASR Server (Streaming Model)
    ↓
Transcript (partial results every 100ms)
```

**Streaming ASR Implementation:**
```python
import grpc
from google.cloud import speech_v1p1beta1 as speech

def stream_transcribe():
    client = speech.SpeechClient()
    
    config = speech.StreamingRecognitionConfig(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='en-US',
        ),
        interim_results=True,  # Get partial results
    )
    
    # Generator for audio chunks
    def audio_generator():
        while True:
            chunk = get_audio_chunk()  # 20ms of audio
            yield speech.StreamingRecognizeRequest(audio_content=chunk)
    
    requests = audio_generator()
    responses = client.streaming_recognize(config, requests)
    
    for response in responses:
        for result in response.results:
            if result.is_final:
                print(f"Final: {result.alternatives[0].transcript}")
            else:
                print(f"Partial: {result.alternatives[0].transcript}")
```

**Key Requirement:** Model must process in < 20ms to keep up with audio stream.

## 14. Deep Dive: Voice Quality Monitoring

How do we measure if multi-region speech systems are working well?

**Metrics:**

**1. Word Error Rate (WER):**
\\[
\text{WER} = \frac{\text{Substitutions} + \text{Insertions} + \text{Deletions}}{\text{Total Words}}
\\]

**2. Mean Opinion Score (MOS) for TTS:**
- Human raters score voice quality from 1 (terrible) to 5 (excellent).
- Target: MOS > 4.0.

**3. Real-Time Factor (RTF):**
\\[
\text{RTF} = \frac{\text{Processing Time}}{\text{Audio Duration}}
\\]
- RTF < 1.0 means faster than real-time.
- Target for streaming: RTF < 0.5.

**Automated Testing:**
```python
import time

def measure_rtf(asr_model, audio_file):
    audio_duration = get_audio_duration(audio_file)  # e.g., 10 seconds
    
    start = time.time()
    transcript = asr_model.transcribe(audio_file)
    processing_time = time.time() - start
    
    rtf = processing_time / audio_duration
    print(f"RTF: {rtf:.2f}")
    
    if rtf > 1.0:
        print("WARNING: Model is slower than real-time!")
    
    return rtf
```

## 15. Deep Dive: Bandwidth Management

Streaming audio consumes significant bandwidth. Optimization is critical for mobile users.

**Audio Compression:**
- **Uncompressed PCM:** 16kHz, 16-bit = 256 kbps
- **Opus Codec:** 16 kbps (16x compression)
- **Speex:** 8 kbps (ultra-low bitrate)

**Implementation:**
```python
import pyaudio
import opuslib

def stream_compressed_audio():
    p = pyaudio.PyAudio()
    encoder = opuslib.Encoder(16000, 1, opuslib.APPLICATION_VOIP)
    
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=320)  # 20ms chunks
    
    while True:
        audio_chunk = stream.read(320)
        compressed = encoder.encode(audio_chunk, 320)
        
        # Send compressed chunk to server
        send_to_server(compressed)
```

**Trade-off:** Lower bitrate = worse audio quality = higher WER.

## 16. Deep Dive: Cost Optimization for Global Speech

Running GPU-based ASR globally is expensive. How do we reduce cost?

**Strategy 1: CPU Inference with ONNX Runtime**
Convert model from PyTorch to ONNX for optimized CPU inference.

```python
import torch
import onnx

# Export to ONNX
dummy_input = torch.randn(1, 80, 100)  # Mel spectrogram
torch.onnx.export(asr_model, dummy_input, "asr.onnx")

# Inference with ONNX Runtime (2-3x faster than PyTorch CPU)
import onnxruntime as ort

session = ort.InferenceSession("asr.onnx")
outputs = session.run(None, {"input": audio_features})
```

**Result:** CPU instances are 5x cheaper than GPU instances.

**Strategy 2: Autoscaling Based on Region-Specific Traffic**
Don't run 24/7 in all regions. Scale down at night.

**Traffic Patterns:**
- **US-West:** Peak at 2pm PST (5pm EST).
- **EU-West:** Peak at 2pm CET (8am EST).
- **Asia-Pacific:** Peak at 2pm JST (1am EST).

**Autoscaling Policy:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: asr-us-west
spec:
  scaleTargetRef:
    kind: Deployment
    name: asr-us-west
  minReplicas: 2   # Night
  maxReplicas: 50  # Day
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
```

**Strategy 3: Spot Instances for Batch Transcription**
For non-real-time workloads (podcast transcription), use spot instances.

```python
# Kubernetes Toleration for Spot Instances
spec:
  tolerations:
  - key: "spot"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
  nodeSelector:
    instance-type: "spot"
```

## 17. Deep Dive: Multi-Language Support

**Challenge:** Support 100+ languages across all regions.

**Option 1: Separate Models per Language**
- Deploy `asr-en`, `asr-fr`, `asr-es`, etc.
- **Pros:** Best accuracy per language.
- **Cons:** 100x storage and compute.

**Option 2: Multilingual Model**
- Single model trained on 100 languages.
- **Pros:** 1x storage, universal model.
- **Cons:** Slightly lower accuracy per language.

**Hybrid Approach:**
Deploy multilingual model globally. Deploy language-specific models in high-demand regions.

```python
def get_model(language_code, region):
    if language_code == 'en' and region == 'us-west':
        return load_model('asr-en-specialized')
    else:
        return load_model('asr-multilingual')
```

## 18. Deep Dive: Failover Testing and Chaos Engineering

**Problem:** How do you verify that failover actually works?

**Chaos Engineering Test:**
1. **Simulate Region Failure:** Terminate all pods in EU-West.
2. **Observe:** Do EU users get routed to US-East?
3. **Measure:** What's the latency increase? Any errors?

**Automated Failover Test:**
```python
import requests
import time

def test_failover():
    # Baseline: All regions healthy
    response = requests.get("https://speech-api.example.com/health")
    assert response.json()['all_healthy'] == True
    
    # Simulate EU-West failure
    kill_region('eu-west')
    
    time.sleep(10)  # Wait for DNS update
    
    # Test from EU client
    start = time.time()
    response = requests.post(
        "https://speech-api.example.com/asr",
        data=audio_data,
        headers={"X-Client-Location": "EU"}
    )
    latency = time.time() - start
    
    assert response.status_code == 200
    assert latency < 200  # Acceptable degraded latency
    
    print(f"Failover successful. Latency: {latency*1000:.0f}ms")
```

**Run Monthly:** Ensure team knows how to handle regional outages.

## 19. Deep Dive: Shadow Traffic for Model Validation

Before deploying a new ASR model to prod, validate it using shadow traffic.

**Shadow Traffic Strategy:**
1. Deploy new model alongside old model.
2. Send 100% of live traffic to both models.
3. Serve old model's response to users.
4. Log new model's response for comparison.

```python
import asyncio

async def shadow_predict(audio):
    # Primary prediction
    primary_task = asyncio.create_task(model_v1.predict(audio))
    
    # Shadow prediction (async, non-blocking)
    shadow_task = asyncio.create_task(model_v2.predict(audio))
    
    # Wait for primary
    primary_result = await primary_task
    
    # Log shadow result (don't wait)
    asyncio.create_task(log_shadow_result(shadow_task))
    
    return primary_result
```

**Comparison Metrics:**
- WER difference
- Latency difference
- Vocabulary coverage

## 20. Production War Stories

**War Story 1: The DNS Caching Disaster**
A team deployed to a new region. Updated DNS. But users kept going to the old region for 2 hours.
**Root Cause:** ISPs cached DNS records (TTL = 3600s).
**Lesson:** Set TTL = 60s for DNS records that might change.

**War Story 2: The Cross-Region Bandwidth Bill**
A team accidentally routed EU traffic to US servers for a week.
**Cost:** $50,000 in cross-region data transfer fees.
**Lesson:** Monitor traffic routing with dashboards.

**War Story 3: The GDPR Violation**
A bug caused EU user audio to be logged in US servers.
**Result:** GDPR fine, PR disaster.
**Lesson:** Implement fail-safe network policies (block cross-region traffic at firewall level).

## 21. Deep Dive: Network Optimization for Speech

Speech data requires significant bandwidth. Optimize network usage.

**Optimization 1: WebSocket Connection Pooling**
Reuse connections instead of creating new ones for each request.

```python
import websockets
import asyncio

class ConnectionPool:
    def __init__(self, uri, pool_size=10):
        self.uri = uri
        self.pool = asyncio.Queue(maxsize=pool_size)
        asyncio.create_task(self._fill_pool(pool_size))
    
    async def _fill_pool(self, size):
        for _ in range(size):
            conn = await websockets.connect(self.uri)
            await self.pool.put(conn)
    
    async def get_connection(self):
        return await self.pool.get()
    
    async def return_connection(self, conn):
        await self.pool.put(conn)

# Usage
pool = ConnectionPool('wss://speech-api.example.com')

async def stream_audio(audio_data):
    conn = await pool.get_connection()
    try:
        await conn.send(audio_data)
        response = await conn.recv()
        return response
    finally:
        await pool.return_connection(conn)
```

**Optimization 2: gRPC Streaming with Multiplexing**
gRPC multiplexes multiple streams over a single TCP connection.

```python
import grpc
from concurrent import futures

class SpeechService:
    def StreamingRecognize(self, request_iterator, context):
        for request in request_iterator:
            audio_chunk = request.audio_content
            transcript = asr_model.process_chunk(audio_chunk)
            yield SpeechResponse(transcript=transcript)

# Server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
add_SpeechServiceServicer_to_server(SpeechService(), server)
server.add_insecure_port('[::]:50051')
server.start()
```

## 22. Deep Dive: Latency SLAs and Penalties

Production systems often have strict latency SLAs.

**Example SLA:**
- **P50 latency:** < 50ms (99% of time)
- **P95 latency:** < 150ms
- **P99 latency:** < 300ms

**Penalty:** If P95 > 150ms for > 1 hour, pay $10,000.

**How to Meet SLAs:**
1. **Over-provision:** Run 30% more replicas than needed.
2. **Circuit Breaker:** If a region's latency spikes, stop routing traffic there.
3. **Fallback:** Route to next-closest region if primary is overloaded.

**Circuit Breaker Implementation:**
```python
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit tripped
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = 0
    
    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            raise e

# Usage
breaker = CircuitBreaker()

def call_asr_service(audio):
    return breaker.call(asr_model.transcribe, audio)
```

## 23. Production Deployment Checklist

Before deploying speech models to production, verify:

**Pre-Launch Checklist:**
- [ ] **Load Testing:** Test with 10x expected peak traffic.
- [ ] **Failover Drill:** Kill one region, verify traffic shifts.
- [ ] **Latency Benchmarks:** P99 < 300ms in all regions.
- [ ] **GDPR Compliance:** EU data stays in EU.
- [ ] **Monitoring Dashboards:** Grafana dashboards for each region.
- [ ] **Alerting Rules:** PagerDuty alerts for p99 > 500ms.
- [ ] **Runbooks:** Documented procedures for common incidents.
- [ ] **Rollback Plan:** Can revert to previous model in < 5 minutes.

**Load Testing Script:**
```python
import asyncio
import aiohttp
import time

async def stress_test(url, audio_file, num_requests=10000):
    async with aiohttp.ClientSession() as session:
        tasks = []
        start = time.time()
        
        for i in range(num_requests):
            task = session.post(url, data=open(audio_file, 'rb'))
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        duration = time.time() - start
        rps = num_requests / duration
        
        latencies = [r.headers.get('X-Latency-Ms') for r in responses]
        p99 = sorted([float(l) for l in latencies if l])[int(len(latencies) * 0.99)]
        
        print(f"RPS: {rps:.0f}")
        print(f"P99 Latency: {p99:.0f}ms")

# Run
asyncio.run(stress_test(
    'https://speech-api.example.com/asr',
    'test_audio.wav',
    num_requests=10000
))
```

## 24. Future Trends: Serverless Speech

**Trend:** Instead of running servers 24/7, use serverless (AWS Lambda, Google Cloud Run).

**Pros:**
- **Cost:** Pay only for actual usage (not idle time).
- **Auto-Scaling:** Scales to zero when not in use.

**Cons:**
- **Cold Start:** First request is slow (5-10 seconds to load model).
- **Memory Limits:** Lambda has 10GB max memory.

**Workaround: Provisioned Concurrency**
Keep N instances warm at all times.

```yaml
# AWS Lambda with Provisioned Concurrency
Resources:
  SpeechFunction:
    Type: AWS::Lambda::Function
    Properties:
      Runtime: python3.9
      Handler: app.handler
      MemorySize: 10240  # 10 GB
      Timeout: 60
  
  ProvisionedConcurrency:
    Type: AWS::Lambda::Alias
    Properties:
      FunctionName: !Ref SpeechFunction
      ProvisionedConcurrencyConfig:
        ProvisionedConcurrentExecutions: 10  # Keep 10 warm
```

## Key Takeaways

1. **Multi-Region is Essential:** Reduces latency and ensures high availability.
2. **GeoDNS:** Routes users to the nearest region automatically.
3. **Edge Deployment:** For ultra-low latency, deploy quantized models at the edge.
4. **Canary Deployments:** Reduce risk when rolling out new models.
5. **GDPR Compliance:** Regional isolation is critical for EU users.

## Summary

| Aspect | Insight |
|:---|:---|
| **Goal** | Low latency, high availability, global reach |
| **Architecture** | Multi-region clusters + GeoDNS + Edge caching |
| **Challenges** | Model sync, data compliance, disaster recovery |
| **Key Metrics** | Latency (p99), error rate, traffic distribution |

---

**Originally published at:** [arunbaby.com/speech-tech/0033-multi-region-speech-deployment](https://www.arunbaby.com/speech-tech/0033-multi-region-speech-deployment/)

*If you found this helpful, consider sharing it with others who might benefit.*
