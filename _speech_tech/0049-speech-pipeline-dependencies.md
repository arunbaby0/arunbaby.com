---
title: "Speech Pipeline Dependencies"
day: 49
collection: speech_tech
categories:
  - speech-tech
tags:
  - pipeline
  - dependencies
  - dag
  - speech-processing
  - orchestration
difficulty: Hard
subdomain: "Speech Systems"
tech_stack: Python, Airflow, Prefect
scale: "Multi-stage pipelines, batch and streaming"
companies: Google, Amazon, Apple, Nuance
related_dsa_day: 49
related_ml_day: 49
related_agents_day: 49
---

**"Speech pipelines are graphs—audio flows through ordered transformations."**

## 1. Introduction

Speech processing involves multiple interdependent stages: audio loading, feature extraction, VAD, ASR, NLU. Each stage depends on previous outputs, forming a DAG.

### Typical Speech Pipeline

```
┌─────────────┐
│ Audio Input │
└──────┬──────┘
       │
   ┌───▼───┐
   │ Resample│
   └───┬───┘
       │
   ┌───▼───┐    ┌───────────┐
   │  VAD  │───►│ Speaker ID │
   └───┬───┘    └───────────┘
       │
   ┌───▼───┐
   │Features│
   └───┬───┘
       │
   ┌───▼───┐    ┌───────────┐
   │  ASR  │───►│    NLU    │
   └───┬───┘    └───────────┘
       │
   ┌───▼───┐
   │Transcript│
   └─────────┘
```

## 2. Speech Pipeline DAG

```python
from dataclasses import dataclass
from typing import Dict, List, Any, Callable
from enum import Enum
import numpy as np

class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class PipelineStage:
    """A stage in the speech pipeline."""
    name: str
    func: Callable
    dependencies: List[str]
    config: Dict = None
    
    status: StageStatus = StageStatus.PENDING
    result: Any = None
    error: str = None


class SpeechPipeline:
    """DAG-based speech processing pipeline."""
    
    def __init__(self, name: str):
        self.name = name
        self.stages: Dict[str, PipelineStage] = {}
    
    def add_stage(
        self,
        name: str,
        func: Callable,
        dependencies: List[str] = None,
        config: Dict = None
    ):
        stage = PipelineStage(
            name=name,
            func=func,
            dependencies=dependencies or [],
            config=config or {}
        )
        self.stages[name] = stage
    
    def validate(self) -> bool:
        """Check for cycles using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {name: WHITE for name in self.stages}
        
        def has_cycle(name):
            if colors[name] == GRAY:
                return True
            if colors[name] == BLACK:
                return False
            
            colors[name] = GRAY
            for dep in self.stages[name].dependencies:
                if has_cycle(dep):
                    return True
            colors[name] = BLACK
            return False
        
        return not any(has_cycle(name) for name in self.stages)
    
    def get_execution_order(self) -> List[str]:
        """Topological sort for execution order."""
        in_degree = {name: len(stage.dependencies) 
                     for name, stage in self.stages.items()}
        
        # Build reverse dependency map
        dependents = {name: [] for name in self.stages}
        for name, stage in self.stages.items():
            for dep in stage.dependencies:
                dependents[dep].append(name)
        
        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return order
    
    def run(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """Execute pipeline on audio."""
        if not self.validate():
            raise ValueError("Pipeline has circular dependencies!")
        
        # Initial context
        context = {
            "audio": audio,
            "sample_rate": sample_rate
        }
        
        order = self.get_execution_order()
        
        for stage_name in order:
            stage = self.stages[stage_name]
            stage.status = StageStatus.RUNNING
            
            try:
                # Gather inputs from dependencies
                inputs = {}
                for dep in stage.dependencies:
                    inputs[dep] = self.stages[dep].result
                
                # Add context
                inputs["context"] = context
                inputs.update(stage.config)
                
                # Execute
                stage.result = stage.func(**inputs)
                stage.status = StageStatus.SUCCESS
                
            except Exception as e:
                stage.error = str(e)
                stage.status = StageStatus.FAILED
                raise
        
        return {name: stage.result for name, stage in self.stages.items()}
```

## 3. Common Speech Stages

```python
import librosa

def load_audio(context, **kwargs):
    """Load and normalize audio."""
    audio = context["audio"]
    sr = context["sample_rate"]
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return {"audio": audio, "sample_rate": sr, "duration": len(audio) / sr}


def resample(load_audio, target_sr=16000, **kwargs):
    """Resample to target rate."""
    audio = load_audio["audio"]
    sr = load_audio["sample_rate"]
    
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    return {"audio": audio, "sample_rate": target_sr}


def extract_features(resample, feature_type="mel", **kwargs):
    """Extract audio features."""
    audio = resample["audio"]
    sr = resample["sample_rate"]
    
    if feature_type == "mel":
        features = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
        features = librosa.power_to_db(features)
    elif feature_type == "mfcc":
        features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    return {"features": features, "feature_type": feature_type}


def voice_activity_detection(resample, threshold=0.01, **kwargs):
    """Detect speech segments."""
    audio = resample["audio"]
    sr = resample["sample_rate"]
    
    # Simple energy-based VAD
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    energy = np.mean(frames ** 2, axis=0)
    
    speech_frames = energy > threshold
    
    # Convert to time segments
    segments = []
    in_speech = False
    start = 0
    
    for i, is_speech in enumerate(speech_frames):
        if is_speech and not in_speech:
            start = i * hop_length / sr
            in_speech = True
        elif not is_speech and in_speech:
            end = i * hop_length / sr
            segments.append((start, end))
            in_speech = False
    
    if in_speech:
        segments.append((start, len(audio) / sr))
    
    return {"segments": segments, "speech_ratio": sum(e-s for s,e in segments) / (len(audio)/sr)}


def asr_transcription(extract_features, model=None, **kwargs):
    """Run ASR on features."""
    features = extract_features["features"]
    
    # Mock ASR (real would use Whisper, etc.)
    return {"transcript": "mock transcription", "confidence": 0.95}


def speaker_identification(extract_features, voice_activity_detection, model=None, **kwargs):
    """Identify speaker from features."""
    features = extract_features["features"]
    segments = voice_activity_detection["segments"]
    
    # Mock speaker ID
    return {"speaker_id": "speaker_001", "confidence": 0.87}
```

## 4. Building the Pipeline

```python
def create_asr_pipeline() -> SpeechPipeline:
    """Create complete ASR pipeline."""
    pipeline = SpeechPipeline("asr_pipeline")
    
    # Stage 1: Load audio
    pipeline.add_stage("load_audio", load_audio)
    
    # Stage 2: Resample
    pipeline.add_stage(
        "resample",
        resample,
        dependencies=["load_audio"],
        config={"target_sr": 16000}
    )
    
    # Stage 3a: VAD (parallel with features)
    pipeline.add_stage(
        "vad",
        voice_activity_detection,
        dependencies=["resample"]
    )
    
    # Stage 3b: Feature extraction (parallel with VAD)
    pipeline.add_stage(
        "features",
        extract_features,
        dependencies=["resample"],
        config={"feature_type": "mel"}
    )
    
    # Stage 4a: ASR (depends on features)
    pipeline.add_stage(
        "asr",
        asr_transcription,
        dependencies=["features"]
    )
    
    # Stage 4b: Speaker ID (depends on both)
    pipeline.add_stage(
        "speaker_id",
        speaker_identification,
        dependencies=["features", "vad"]
    )
    
    return pipeline


# Usage
pipeline = create_asr_pipeline()
audio = np.random.randn(16000 * 5)  # 5 seconds
results = pipeline.run(audio)

print(f"Transcript: {results['asr']['transcript']}")
print(f"Speaker: {results['speaker_id']['speaker_id']}")
```

## 5. Parallel Stage Execution

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncSpeechPipeline(SpeechPipeline):
    """Pipeline with parallel stage execution."""
    
    def __init__(self, name: str, max_workers: int = 4):
        super().__init__(name)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def run_async(self, audio: np.ndarray, sample_rate: int = 16000):
        """Execute pipeline with parallel stages."""
        context = {"audio": audio, "sample_rate": sample_rate}
        completed = set()
        
        while len(completed) < len(self.stages):
            # Find ready stages
            ready = [
                name for name, stage in self.stages.items()
                if name not in completed
                and all(dep in completed for dep in stage.dependencies)
            ]
            
            if not ready:
                break
            
            # Run ready stages in parallel
            tasks = [
                self._run_stage_async(name, context, completed)
                for name in ready
            ]
            
            await asyncio.gather(*tasks)
            completed.update(ready)
        
        return {name: stage.result for name, stage in self.stages.items()}
    
    async def _run_stage_async(self, name: str, context: Dict, completed: set):
        """Run single stage asynchronously."""
        stage = self.stages[name]
        
        inputs = {dep: self.stages[dep].result for dep in stage.dependencies}
        inputs["context"] = context
        inputs.update(stage.config)
        
        loop = asyncio.get_event_loop()
        stage.result = await loop.run_in_executor(
            self.executor,
            lambda: stage.func(**inputs)
        )
```

## 6. Streaming Pipeline

```python
class StreamingSpeechPipeline:
    """Pipeline for streaming audio processing."""
    
    def __init__(self, chunk_size: int = 1600):  # 100ms at 16kHz
        self.chunk_size = chunk_size
        self.buffer = np.array([])
        self.state = {}  # Carry state between chunks
    
    def process_chunk(self, chunk: np.ndarray) -> Dict:
        """Process a single audio chunk."""
        self.buffer = np.concatenate([self.buffer, chunk])
        
        results = {}
        
        # Feature extraction on chunk
        if len(self.buffer) >= self.chunk_size:
            features = self._extract_chunk_features(
                self.buffer[-self.chunk_size:]
            )
            results["features"] = features
            
            # VAD on chunk
            vad = self._chunk_vad(self.buffer[-self.chunk_size:])
            results["vad"] = vad
            
            # If enough speech, run ASR
            if vad["is_speech"] and len(self.buffer) >= self.chunk_size * 10:
                asr = self._incremental_asr(self.buffer)
                results["partial_transcript"] = asr
        
        # Trim buffer
        max_buffer = self.chunk_size * 50  # 5 seconds
        if len(self.buffer) > max_buffer:
            self.buffer = self.buffer[-max_buffer:]
        
        return results
    
    def _extract_chunk_features(self, chunk):
        # Incremental feature extraction
        pass
    
    def _chunk_vad(self, chunk):
        energy = np.mean(chunk ** 2)
        return {"is_speech": energy > 0.01, "energy": energy}
    
    def _incremental_asr(self, audio):
        # Incremental ASR
        pass
```

## 7. Connection to Course Schedule

Speech pipelines are exact applications of topological sort:

| Course Schedule | Speech Pipeline |
|----------------|-----------------|
| Course | Processing stage |
| Prerequisite | Data dependency |
| Valid schedule | Valid pipeline |
| Execution order | Processing order |

## 8. Key Takeaways

1. **Model as DAG** - dependencies are explicit
2. **Topological sort** determines execution order
3. **Parallel execution** where dependencies allow
4. **Streaming** requires careful state management
5. **Validation** catches circular dependencies early

---

**Originally published at:** [arunbaby.com/speech-tech/0049-speech-pipeline-dependencies](https://www.arunbaby.com/speech-tech/0049-speech-pipeline-dependencies/)

*If you found this helpful, consider sharing it with others who might benefit.*
