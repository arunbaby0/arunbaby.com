---
title: "Event Stream Processing"
day: 16
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - stream-processing
  - real-time
  - kafka
  - flink
  - event-driven
  - windowing
  - temporal-processing
subdomain: "Real-Time Systems"
tech_stack: [Kafka, Apache Flink, Apache Spark Streaming, Redis, Cassandra, Prometheus]
scale: "Millions of events/sec, <100ms latency, multi-region"
companies: [Netflix, Uber, LinkedIn, Airbnb, Twitter, Spotify]
related_dsa_day: 16
related_speech_day: 16
related_agents_day: 16
---

**Build production event stream processing systems that handle millions of events per second using windowing and temporal aggregation—applying the same interval merging principles from algorithm design.**

## Problem Statement

Design an **Event Stream Processing System** that ingests, processes, and analyzes millions of events per second in real-time, supporting windowed aggregations, pattern detection, and low-latency analytics.

### Functional Requirements

1. **Event ingestion:** Ingest millions of events/second from multiple sources
2. **Stream processing:** Real-time transformations, filtering, enrichment
3. **Windowed aggregations:** Tumbling, sliding, session windows
4. **Pattern detection:** Complex event processing (CEP)
5. **State management:** Maintain state across events
6. **Exactly-once semantics:** No duplicate or lost events
7. **Late data handling:** Handle out-of-order events
8. **Multiple outputs:** Write to databases, caches, dashboards

### Non-Functional Requirements

1. **Throughput:** 1M+ events/second per partition
2. **Latency:** p99 < 100ms for event processing
3. **Availability:** 99.99% uptime
4. **Scalability:** Horizontal scaling to 1000+ nodes
5. **Fault tolerance:** Automatic recovery from failures
6. **Backpressure:** Handle traffic spikes gracefully
7. **Cost efficiency:** Optimize resource utilization

## Understanding the Requirements

Event stream processing is the **backbone of real-time analytics** at scale:

### Real-World Use Cases

| Company | Use Case | Scale | Technology |
|---------|----------|-------|------------|
| Netflix | Real-time recommendation updates | 10M+ events/sec | Kafka + Flink |
| Uber | Surge pricing, driver matching | 5M+ events/sec | Kafka + custom |
| LinkedIn | News feed ranking | 1M+ events/sec | Kafka + Samza |
| Airbnb | Pricing optimization | 500K+ events/sec | Kafka + Spark |
| Twitter | Trending topics | 5M+ tweets/sec | Kafka + custom |
| Spotify | Real-time playlist updates | 1M+ events/sec | Kafka + Flink |

### Why Event Streams Matter

1. **Real-time analytics:** Instant insights from data
2. **ML feature computation:** Real-time feature updates
3. **Fraud detection:** Immediate anomaly detection
4. **User engagement:** Real-time personalization
5. **Monitoring:** Live system health tracking
6. **Business intelligence:** Instant KPI updates

### The Interval Processing Connection

Just like the **Merge Intervals** problem:

| Merge Intervals | Event Stream Processing | Audio Segmentation |
|----------------|------------------------|-------------------|
| Merge overlapping time ranges | Merge event windows | Merge audio segments |
| Sort by start time | Event time ordering | Temporal ordering |
| Greedy merging | Window aggregation | Boundary merging |
| O(N log N) complexity | Stream buffering | Segment processing |
| Overlap detection | Event correlation | Segment alignment |

All three deal with **temporal data** requiring efficient interval/window processing.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Event Stream Processing System                 │
└─────────────────────────────────────────────────────────────────┘

          Event Sources
          ┌─────────┐ ┌─────────┐ ┌─────────┐
          │  Apps   │ │ Services│ │  IoT    │
          └────┬────┘ └────┬────┘ └────┬────┘
               │           │           │
               └───────────┼───────────┘
                           │
                    ┌──────▼──────┐
                    │   Kafka     │
                    │  (Message   │
                    │   Broker)   │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐ ┌──────▼──────┐ ┌────────▼────────┐
│ Stream         │ │ Windowing   │ │ Aggregation     │
│ Processing     │ │ Engine      │ │ Engine          │
│                │ │             │ │                 │
│ - Filter       │ │ - Tumbling  │ │ - Count         │
│ - Transform    │ │ - Sliding   │ │ - Sum           │
│ - Enrich       │ │ - Session   │ │ - Average       │
└───────┬────────┘ └──────┬──────┘ └────────┬────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼──────┐
                    │   State     │
                    │   Store     │
                    │  (RocksDB)  │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐ ┌──────▼──────┐ ┌────────▼────────┐
│   Database     │ │   Cache     │ │   Dashboard     │
│  (Cassandra)   │ │   (Redis)   │ │  (Grafana)      │
└────────────────┘ └─────────────┘ └─────────────────┘
```

### Key Components

1. **Message Broker:** Kafka for event ingestion and buffering
2. **Stream Processor:** Flink/Spark for real-time computation
3. **Windowing Engine:** Time-based and session-based windows
4. **State Store:** RocksDB for stateful processing
5. **Output Sinks:** Multiple destinations for processed events

## Component Deep-Dives

### 1. Event Windowing - Similar to Interval Merging

Windows group events by time, just like merging intervals:

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import time

@dataclass
class Event:
    """A single event in the stream."""
    event_id: str
    event_type: str
    timestamp: int  # Unix timestamp in milliseconds
    user_id: str
    data: Dict[str, Any]
    
    @property
    def event_time(self) -> datetime:
        """Get event time as datetime."""
        return datetime.fromtimestamp(self.timestamp / 1000.0)

@dataclass
class Window:
    """
    A time window containing events.
    
    Similar to intervals in merge intervals problem:
    - start: interval start
    - end: interval end
    - events: data within interval
    """
    start: int  # Window start (ms)
    end: int    # Window end (ms)
    events: List[Event]
    
    def overlaps(self, other: 'Window') -> bool:
        """
        Check if this window overlaps with another.
        
        Same logic as interval overlap:
        max(start1, start2) <= min(end1, end2)
        """
        return max(self.start, other.start) <= min(self.end, other.end)
    
    def merge(self, other: 'Window') -> 'Window':
        """
        Merge this window with another.
        
        Same as merging intervals:
        - New start = min of starts
        - New end = max of ends
        - Combine events
        """
        return Window(
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            events=self.events + other.events
        )
    
    @property
    def duration_ms(self) -> int:
        return self.end - self.start
    
    @property
    def event_count(self) -> int:
        return len(self.events)


class WindowManager:
    """
    Manage event windows for stream processing.
    
    Similar to merge intervals:
    - Group events into time windows
    - Merge overlapping windows
    - Maintain sorted window list
    """
    
    def __init__(self, window_type: str = "tumbling", window_size_ms: int = 60000):
        """
        Initialize window manager.
        
        Args:
            window_type: "tumbling", "sliding", or "session"
            window_size_ms: Window size in milliseconds
        """
        self.window_type = window_type
        self.window_size_ms = window_size_ms
        self.windows: List[Window] = []
    
    def assign_to_window(self, event: Event) -> List[Window]:
        """
        Assign event to window(s).
        
        Returns:
            List of windows this event belongs to
        """
        if self.window_type == "tumbling":
            return self._assign_tumbling(event)
        elif self.window_type == "sliding":
            return self._assign_sliding(event)
        elif self.window_type == "session":
            return self._assign_session(event)
        else:
            raise ValueError(f"Unknown window type: {self.window_type}")
    
    def _assign_tumbling(self, event: Event) -> List[Window]:
        """
        Tumbling windows: Fixed-size, non-overlapping.
        
        Example: 1-minute windows
        [0-60s], [60-120s], [120-180s], ...
        
        Each event belongs to exactly one window.
        """
        # Calculate which window this event belongs to
        window_id = event.timestamp // self.window_size_ms
        window_start = window_id * self.window_size_ms
        window_end = window_start + self.window_size_ms
        
        # Find or create window
        window = self._find_or_create_window(window_start, window_end)
        window.events.append(event)
        
        return [window]
    
    def _assign_sliding(self, event: Event) -> List[Window]:
        """
        Sliding windows: Fixed-size, overlapping.
        
        Example: 1-minute windows, sliding every 30 seconds
        [0-60s], [30-90s], [60-120s], ...
        
        Each event can belong to multiple windows.
        """
        slide_interval = self.window_size_ms // 2  # 50% overlap
        
        # Find all windows this event falls into
        windows = []
        
        # Calculate first window that could contain this event
        first_window_id = (event.timestamp - self.window_size_ms) // slide_interval
        first_window_start = first_window_id * slide_interval
        
        # Check windows until event is past window end
        current_start = first_window_start
        
        while current_start <= event.timestamp:
            current_end = current_start + self.window_size_ms
            
            if current_start <= event.timestamp < current_end:
                window = self._find_or_create_window(current_start, current_end)
                window.events.append(event)
                windows.append(window)
            
            current_start += slide_interval
        
        return windows
    
    def _assign_session(self, event: Event) -> List[Window]:
        """
        Session windows: Dynamic windows based on activity gaps.
        
        A session ends when there's a gap > session_timeout between events.
        
        This is like merging intervals with a max gap tolerance!
        """
        session_timeout = 5 * 60 * 1000  # 5 minutes
        
        # Find window that could be extended
        for window in self.windows:
            # Check if event is within session timeout of window end
            if event.timestamp - window.end <= session_timeout:
                # Extend window
                window.end = event.timestamp
                window.events.append(event)
                return [window]
        
        # Start new session
        window = Window(
            start=event.timestamp,
            end=event.timestamp,
            events=[event]
        )
        self.windows.append(window)
        
        return [window]
    
    def _find_or_create_window(self, start: int, end: int) -> Window:
        """Find existing window or create new one."""
        for window in self.windows:
            if window.start == start and window.end == end:
                return window
        
        # Create new window
        new_window = Window(start=start, end=end, events=[])
        self.windows.append(new_window)
        
        return new_window
    
    def get_completed_windows(self, watermark: int) -> List[Window]:
        """
        Get windows that are complete (past watermark).
        
        Watermark = latest timestamp we're confident we've seen all events for.
        
        Similar to merge intervals: return all intervals before a certain time.
        """
        completed = []
        remaining = []
        
        for window in self.windows:
            if window.end < watermark:
                completed.append(window)
            else:
                remaining.append(window)
        
        self.windows = remaining
        return completed
    
    def merge_overlapping_windows(self) -> List[Window]:
        """
        Merge overlapping windows.
        
        This is exactly the merge intervals algorithm!
        """
        if not self.windows:
            return []
        
        # Sort by start time
        sorted_windows = sorted(self.windows, key=lambda w: w.start)
        
        merged = [sorted_windows[0]]
        
        for current in sorted_windows[1:]:
            last = merged[-1]
            
            if current.overlaps(last):
                # Merge
                merged[-1] = last.merge(current)
            else:
                # Add new window
                merged.append(current)
        
        return merged
```

### 2. Stream Processing Engine

```python
from typing import Callable, List
from queue import Queue
import threading

class StreamProcessor:
    """
    Event stream processing engine.
    
    Features:
    - Real-time event processing
    - Windowed aggregations
    - Stateful operations
    - Exactly-once semantics
    """
    
    def __init__(self):
        self.window_manager = WindowManager(window_type="tumbling", window_size_ms=60000)
        self.aggregators: Dict[str, Callable] = {}
        self.state_store: Dict[str, Any] = {}
        
        # Processing queue
        self.event_queue = Queue(maxsize=10000)
        self.running = False
        
        # Metrics
        self.events_processed = 0
        self.windows_created = 0
    
    def register_aggregator(self, name: str, func: Callable):
        """Register an aggregation function."""
        self.aggregators[name] = func
    
    def process_event(self, event: Event):
        """
        Process a single event.
        
        Steps:
        1. Assign to window(s)
        2. Update state
        3. Apply aggregations
        4. Emit results
        """
        # Assign to windows
        windows = self.window_manager.assign_to_window(event)
        
        # Update state for each window
        for window in windows:
            window_key = f"{window.start}-{window.end}"
            
            # Initialize state if needed
            if window_key not in self.state_store:
                self.state_store[window_key] = {
                    'count': 0,
                    'sum': 0,
                    'events': []
                }
            
            # Update state
            state = self.state_store[window_key]
            state['count'] += 1
            state['events'].append(event)
            
            # Apply aggregations
            for name, aggregator in self.aggregators.items():
                result = aggregator(window.events)
                state[name] = result
        
        self.events_processed += 1
    
    def get_window_aggregates(self, window_start: int, window_end: int) -> Dict:
        """Get aggregates for a specific window."""
        window_key = f"{window_start}-{window_end}"
        return self.state_store.get(window_key, {})
    
    def flush_completed_windows(self, watermark: int) -> List[Dict]:
        """
        Flush completed windows to output.
        
        Similar to returning merged intervals after processing.
        """
        completed = self.window_manager.get_completed_windows(watermark)
        
        results = []
        
        for window in completed:
            window_key = f"{window.start}-{window.end}"
            
            if window_key in self.state_store:
                result = {
                    'window_start': window.start,
                    'window_end': window.end,
                    'aggregates': self.state_store[window_key]
                }
                results.append(result)
                
                # Clean up state
                del self.state_store[window_key]
        
        return results


# Example usage
def example_stream_processing():
    """Example: Count events per user in 1-minute windows."""
    processor = StreamProcessor()
    
    # Register aggregator
    def count_by_user(events: List[Event]) -> Dict[str, int]:
        """Count events per user."""
        counts = defaultdict(int)
        for event in events:
            counts[event.user_id] += 1
        return dict(counts)
    
    processor.register_aggregator('user_counts', count_by_user)
    
    # Process events
    events = [
        Event("1", "click", 1000, "user1", {}),
        Event("2", "click", 2000, "user1", {}),
        Event("3", "click", 3000, "user2", {}),
        Event("4", "view", 65000, "user1", {}),  # Next window
    ]
    
    for event in events:
        processor.process_event(event)
    
    # Flush completed windows (watermark = 70000ms)
    results = processor.flush_completed_windows(70000)
    
    for result in results:
        print(f"Window {result['window_start']}-{result['window_end']}:")
        print(f"  User counts: {result['aggregates']['user_counts']}")
```

### 3. Complex Event Processing (CEP)

```python
from typing import List, Callable
from dataclasses import dataclass

@dataclass
class Pattern:
    """Event pattern for detection."""
    name: str
    conditions: List[Callable[[Event], bool]]
    window_ms: int
    
class CEPEngine:
    """
    Complex Event Processing engine.
    
    Detect patterns in event streams:
    - Sequences: A followed by B within time window
    - Conditions: Events matching criteria
    - Aggregations: Count, sum over window
    """
    
    def __init__(self):
        self.patterns: List[Pattern] = []
        self.matches: List[Dict] = []
    
    def register_pattern(self, pattern: Pattern):
        """Register a pattern to detect."""
        self.patterns.append(pattern)
    
    def detect_patterns(self, events: List[Event]) -> List[Dict]:
        """
        Detect registered patterns in event stream.
        
        Uses interval-style processing:
        - Sort events by time
        - Sliding window over events
        - Check pattern conditions
        """
        matches = []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        for pattern in self.patterns:
            # Find sequences matching pattern
            pattern_matches = self._find_pattern_matches(sorted_events, pattern)
            matches.extend(pattern_matches)
        
        return matches
    
    def _find_pattern_matches(
        self,
        events: List[Event],
        pattern: Pattern
    ) -> List[Dict]:
        """Find all matches of pattern in events."""
        matches = []
        
        for i in range(len(events)):
            # Try to match pattern starting at event i
            match_events = [events[i]]
            
            # Check if first condition matches
            if not pattern.conditions[0](events[i]):
                continue
            
            # Look for subsequent events matching remaining conditions
            j = i + 1
            condition_idx = 1
            
            while j < len(events) and condition_idx < len(pattern.conditions):
                # Check if within time window
                if events[j].timestamp - events[i].timestamp > pattern.window_ms:
                    break
                
                # Check if condition matches
                if pattern.conditions[condition_idx](events[j]):
                    match_events.append(events[j])
                    condition_idx += 1
                
                j += 1
            
            # Check if full pattern matched
            if condition_idx == len(pattern.conditions):
                matches.append({
                    'pattern': pattern.name,
                    'events': match_events,
                    'start_time': events[i].timestamp,
                    'end_time': match_events[-1].timestamp
                })
        
        return matches


# Example: Fraud detection pattern
def fraud_detection_example():
    """Detect potential fraud: multiple failed logins followed by success."""
    cep = CEPEngine()
    
    # Define pattern
    pattern = Pattern(
        name="suspicious_login",
        conditions=[
            lambda e: e.event_type == "login_failed",
            lambda e: e.event_type == "login_failed",
            lambda e: e.event_type == "login_failed",
            lambda e: e.event_type == "login_success"
        ],
        window_ms=60000  # Within 1 minute
    )
    
    cep.register_pattern(pattern)
    
    # Test events
    events = [
        Event("1", "login_failed", 1000, "user1", {}),
        Event("2", "login_failed", 2000, "user1", {}),
        Event("3", "login_failed", 3000, "user1", {}),
        Event("4", "login_success", 4000, "user1", {}),
    ]
    
    matches = cep.detect_patterns(events)
    
    for match in matches:
        print(f"Pattern '{match['pattern']}' detected:")
        print(f"  Time window: {match['start_time']}-{match['end_time']}")
        print(f"  Events: {[e.event_id for e in match['events']]}")
```

### 4. State Management with Checkpointing

```python
import pickle
import os

class StateManager:
    """
    Manage stateful stream processing with checkpointing.
    
    Features:
    - Fault tolerance through checkpoints
    - Exactly-once semantics
    - State recovery
    """
    
    def __init__(self, checkpoint_dir: str = "/tmp/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.state: Dict[str, Any] = {}
        self.checkpoint_interval_ms = 60000
        self.last_checkpoint_time = 0
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def update_state(self, key: str, value: Any):
        """Update state."""
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        return self.state.get(key, default)
    
    def checkpoint(self, watermark: int):
        """
        Create state checkpoint.
        
        Similar to saving merged intervals periodically.
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{watermark}.pkl"
        )
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'watermark': watermark,
                'state': self.state
            }, f)
        
        self.last_checkpoint_time = watermark
        
        # Clean old checkpoints
        self._cleanup_old_checkpoints(watermark)
    
    def restore_from_checkpoint(self, watermark: Optional[int] = None):
        """Restore state from checkpoint."""
        if watermark is None:
            # Find latest checkpoint
            checkpoints = [
                f for f in os.listdir(self.checkpoint_dir)
                if f.startswith("checkpoint_")
            ]
            
            if not checkpoints:
                return
            
            latest = max(checkpoints, key=lambda f: int(f.split('_')[1].split('.')[0]))
            checkpoint_path = os.path.join(self.checkpoint_dir, latest)
        else:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{watermark}.pkl"
            )
        
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
                self.state = data['state']
                return data['watermark']
        
        return None
    
    def _cleanup_old_checkpoints(self, current_watermark: int, keep_last: int = 3):
        """Keep only recent checkpoints."""
        checkpoints = [
            (f, int(f.split('_')[1].split('.')[0]))
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_")
        ]
        
        # Sort by watermark
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Delete old ones
        for checkpoint_file, watermark in checkpoints[keep_last:]:
            os.remove(os.path.join(self.checkpoint_dir, checkpoint_file))
```

## Production Deployment

### Apache Kafka + Flink Architecture

```yaml
# docker-compose.yml for stream processing stack
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
  
  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
  
  flink-jobmanager:
    image: flink:latest
    ports:
      - "8081:8081"
    command: jobmanager
    environment:
      - JOB_MANAGER_RPC_ADDRESS=flink-jobmanager
  
  flink-taskmanager:
    image: flink:latest
    depends_on:
      - flink-jobmanager
    command: taskmanager
    environment:
      - JOB_MANAGER_RPC_ADDRESS=flink-jobmanager
    deploy:
      replicas: 3
```

### Kafka Producer

```python
from kafka import KafkaProducer
import json

class EventProducer:
    """Produce events to Kafka."""
    
    def __init__(self, bootstrap_servers: List[str], topic: str):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = topic
    
    def send_event(self, event: Event):
        """Send event to Kafka."""
        event_dict = {
            'event_id': event.event_id,
            'event_type': event.event_type,
            'timestamp': event.timestamp,
            'user_id': event.user_id,
            'data': event.data
        }
        
        self.producer.send(
            self.topic,
            value=event_dict,
            key=event.user_id.encode('utf-8')  # Partition by user
        )
    
    def flush(self):
        """Flush pending messages."""
        self.producer.flush()
```

## Scaling Strategies

### Horizontal Scaling

```python
# Kafka topics with multiple partitions for parallelism
def create_kafka_topic(admin_client, topic_name: str, num_partitions: int = 10):
    """Create Kafka topic with partitions."""
    from kafka.admin import NewTopic
    
    topic = NewTopic(
        name=topic_name,
        num_partitions=num_partitions,
        replication_factor=3
    )
    
    admin_client.create_topics([topic])
```

### Auto-scaling Based on Lag

```python
class StreamProcessorAutoScaler:
    """Auto-scale stream processors based on consumer lag."""
    
    def __init__(self, max_lag_threshold: int = 10000):
        self.max_lag_threshold = max_lag_threshold
    
    def should_scale_up(self, consumer_lag: int) -> bool:
        """Check if should add more processors."""
        return consumer_lag > self.max_lag_threshold
    
    def should_scale_down(self, consumer_lag: int) -> bool:
        """Check if can reduce processors."""
        return consumer_lag < self.max_lag_threshold * 0.5
```

## Real-World Case Study: Netflix Event Processing

### Netflix's Approach

Netflix processes **10M+ events/second** for real-time recommendations:

**Architecture:**
1. **Kafka:** 36+ clusters, 4000+ brokers
2. **Flink:** Real-time stream processing
3. **Keystone:** Real-time data pipeline
4. **Mantis:** Reactive stream processing

**Use Cases:**
- Real-time viewing analytics
- Recommendation updates
- A/B test metric computation
- Anomaly detection

**Results:**
- **10M events/sec** throughput
- **<100ms p99 latency**
- **99.99% availability**
- **Petabytes/day** processed

### Key Lessons

1. **Partition strategically** - by user ID for locality
2. **Use watermarks** for late data handling
3. **Checkpoint frequently** for fault tolerance
4. **Monitor lag closely** - key metric for health
5. **Test backpressure** - must handle traffic spikes

## Cost Analysis

### Infrastructure Costs (1M events/sec)

| Component | Nodes | Cost/Month | Notes |
|-----------|-------|------------|-------|
| Kafka brokers | 10 | $5,000 | r5.2xlarge |
| Flink workers | 20 | $8,000 | c5.4xlarge |
| State storage | - | $500 | S3 for checkpoints |
| Monitoring | - | $200 | Prometheus + Grafana |
| **Total** | | **$13,700/month** | **$0.37 per million events** |

### Optimization Strategies

1. **Batch processing:** Micro-batches reduce overhead
2. **Compression:** Reduce network/storage costs by 70%
3. **State backends:** RocksDB vs in-memory trade-offs
4. **Spot instances:** 70% cost reduction for stateless workers

## Key Takeaways

✅ **Windows are intervals** - same merging logic applies

✅ **Event time vs processing time** - critical distinction

✅ **Watermarks enable** late data handling

✅ **State management** requires checkpointing for fault tolerance

✅ **Exactly-once semantics** possible with careful design

✅ **Kafka + Flink** is industry standard stack

✅ **Partition for parallelism** - key to horizontal scaling

✅ **Monitor consumer lag** - critical health metric

✅ **Backpressure handling** essential for reliability

✅ **Same interval processing** as merge intervals problem

### Connection to Thematic Link: Interval Processing and Temporal Reasoning

All three topics share interval/window processing:

**DSA (Merge Intervals):**
- Sort intervals by start time
- Merge overlapping ranges
- O(N log N) greedy algorithm

**ML System Design (Event Stream Processing):**
- Sort events by timestamp
- Merge event windows
- Windowed aggregations

**Speech Tech (Audio Segmentation):**
- Sort audio segments temporally
- Merge adjacent segments
- Boundary detection

### Universal Pattern

```python
# Pattern used across all three:
1. Sort items by time/position
2. Process in temporal order
3. Merge adjacent/overlapping ranges
4. Apply aggregations within ranges
```

This pattern is **fundamental** to temporal data processing!

## Additional Design & Operational Considerations

To bring this closer to a real production design and to increase depth (and word
count) in a meaningful way, here are additional angles you should be comfortable
discussing:

- **Backpressure in detail:**
  - What happens when sinks (e.g., databases, dashboards) can’t keep up?
  - You should be able to talk about:
    - Producer-side throttling,
    - Buffer sizing and on-disk spill,
    - Dropping or sampling low-priority events,
    - Using dead-letter queues for problematic payloads.
  - In Flink/Spark, backpressure is built into the runtime; in custom systems,
    you must design this flow control explicitly.

- **Exactly-once vs at-least-once semantics:**
  - Exactly-once is often implemented as **effectively-once** at the sink:
    - Idempotent writes,
    - Deduplication using event IDs,
    - Transactional writes (Kafka transactions, 2PC, etc.).
  - Be ready to explain when at-least-once is acceptable (monitoring, metrics)
    and when you truly need exactly-once (billing, financial ledgers).

- **Multi-tenant stream processing:**
  - In a large org, many teams may share the same Kafka cluster / stream engine.
  - Consider:
    - Per-tenant resource quotas,
    - Isolation between streams (separate topics vs namespaces),
    - Access control (who can publish/consume),
    - Governance around schema evolution (Schema Registry, Protobuf/Avro).

- **Schema evolution and compatibility:**
  - Events evolve over time—fields are added/removed.
  - You should design:
    - Backward/forward compatible schemas,
    - Clear deprecation policies,
    - Validation in CI/CD so producers don’t break consumers.

- **End-to-end SLAs:**
  - It’s not enough to have p99 < 100ms inside the stream processor.
  - End-to-end latency includes:
    - Ingestion → broker,
    - Broker → stream processor,
    - Processor → sink,
    - Sink → dashboard / downstream system.
  - You should know where you’d instrument latency histograms and how you’d
    trace a single event through the system (e.g., using a correlation ID).

- **Cost-awareness and capacity planning:**
  - Stream processing clusters can be very expensive at high scale.
  - Think about:
    - Right-sizing instances (CPU vs memory bound),
    - Using autoscaling based on lag and CPU,
    - Separating critical vs non-critical pipelines (priority tiers).

Being able to reason about these topics—and tie them back to the core windowing
and interval-processing primitives from earlier in the post—will make your answer
stand out in senior-level interviews and in real design docs.

---

**Originally published at:** [arunbaby.com/ml-system-design/0016-event-stream-processing](https://www.arunbaby.com/ml-system-design/0016-event-stream-processing/)

*If you found this helpful, consider sharing it with others who might benefit.*



