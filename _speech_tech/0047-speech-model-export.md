---
title: "Speech Model Export (ONNX/TFLite)"
day: 47
collection: speech_tech
categories:
  - speech-tech
tags:
  - model-export
  - onnx
  - tflite
  - edge-deployment
  - mobile-asr
difficulty: Hard
subdomain: "Model Deployment"
tech_stack: Python, PyTorch, TensorFlow, ONNX, TFLite
scale: "10-100x smaller models, sub-100ms mobile latency"
companies: Google, Apple, Meta, Microsoft
related_dsa_day: 47
related_ml_day: 47
related_agents_day: 47
---

**"A speech model that can't run on device can't scale."**

## 1. Introduction

Deploying speech models requires converting from training frameworks to optimized inference formats (ONNX, TFLite). This enables real-time speech on mobile and edge devices.

### Speech Deployment Requirements

- ASR: < 200ms end-to-end latency
- Wake word: < 50ms on device
- Mobile model size: < 50MB

## 2. ONNX Export

```python
import torch
import onnx
import onnxruntime as ort

class SpeechModelONNXExporter:
    """Export speech models to ONNX."""
    
    def export_encoder(
        self,
        model: torch.nn.Module,
        output_path: str,
        max_audio_length: int = 160000
    ):
        """Export speech encoder."""
        model.eval()
        sample_input = torch.randn(1, max_audio_length)
        
        dynamic_axes = {
            'audio': {0: 'batch', 1: 'length'},
            'features': {0: 'batch', 1: 'time'}
        }
        
        torch.onnx.export(
            model, sample_input, output_path,
            input_names=['audio'],
            output_names=['features'],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True
        )
        
        # Validate
        onnx.checker.check_model(onnx.load(output_path))
        return output_path
    
    def export_streaming(
        self,
        model: torch.nn.Module,
        output_path: str,
        chunk_size: int = 1600
    ):
        """Export for streaming inference."""
        streaming_model = StreamingWrapper(model, chunk_size)
        streaming_model.eval()
        
        chunk = torch.randn(1, chunk_size)
        state = torch.zeros(1, 512)
        
        torch.onnx.export(
            streaming_model,
            (chunk, state),
            output_path,
            input_names=['chunk', 'state'],
            output_names=['output', 'new_state'],
            opset_version=14
        )
        return output_path


class StreamingWrapper(torch.nn.Module):
    """Wrapper for streaming inference."""
    
    def __init__(self, model, chunk_size):
        super().__init__()
        self.model = model
        self.chunk_size = chunk_size
    
    def forward(self, chunk, state):
        full_input = torch.cat([state, chunk], dim=1)
        output = self.model(full_input)
        new_state = full_input[:, -512:]
        return output, new_state
```

## 3. TFLite Export

```python
import tensorflow as tf

class SpeechTFLiteExporter:
    """Export speech models to TFLite."""
    
    def convert_basic(self, model_path: str, output_path: str):
        """Basic conversion."""
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        return output_path
    
    def convert_quantized(
        self,
        model_path: str,
        output_path: str,
        quantization: str = 'dynamic'
    ):
        """Convert with quantization."""
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        
        if quantization == 'dynamic':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
        
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        return output_path


class TFLiteRunner:
    """Run TFLite speech models."""
    
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def infer(self, audio):
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            audio.astype('float32')
        )
        self.interpreter.invoke()
        return self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
```

## 4. Quantization

```python
class SpeechQuantizer:
    """Quantize speech models."""
    
    def quantize_pytorch_dynamic(self, model):
        """Dynamic quantization for 2-4x speedup."""
        model.eval()
        return torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8
        )
    
    def quantize_onnx(self, onnx_path: str, output_path: str):
        """Quantize ONNX model."""
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantize_dynamic(
            onnx_path,
            output_path,
            weight_type=QuantType.QInt8
        )
        return output_path
```

## 5. Validation

```python
class ExportValidator:
    """Validate exported models."""
    
    def validate_accuracy(self, original, exported_path, test_audio, tol=0.01):
        """Compare original vs exported outputs."""
        session = ort.InferenceSession(exported_path)
        input_name = session.get_inputs()[0].name
        
        errors = []
        for audio in test_audio:
            with torch.no_grad():
                orig_out = original(torch.tensor(audio).unsqueeze(0))
            
            exp_out = session.run(None, {input_name: audio[None]})[0]
            errors.append(abs(orig_out.numpy() - exp_out).mean())
        
        return {
            'mean_error': sum(errors) / len(errors),
            'passed': max(errors) < tol
        }
    
    def benchmark_latency(self, exported_path, audio_seconds=5.0, runs=100):
        """Measure inference latency."""
        import time
        
        session = ort.InferenceSession(exported_path)
        input_name = session.get_inputs()[0].name
        audio = np.random.randn(1, int(16000 * audio_seconds)).astype('float32')
        
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            session.run(None, {input_name: audio})
            times.append(time.perf_counter() - start)
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'p99_ms': np.percentile(times, 99) * 1000,
            'rtf': np.mean(times) / audio_seconds
        }
```

## 6. Platform-Specific Export

```python
class PlatformExporter:
    """Export for specific platforms."""
    
    @staticmethod
    def export_for_ios(onnx_path: str, output_path: str):
        """Convert to CoreML for iOS."""
        import coremltools as ct
        model = ct.converters.onnx.convert(model=onnx_path)
        model.save(output_path)
    
    @staticmethod
    def export_for_tensorrt(onnx_path: str, output_path: str):
        """Convert to TensorRT for NVIDIA."""
        import tensorrt as trt
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())
        
        config = builder.create_builder_config()
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        engine = builder.build_engine(network, config)
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
```

## 7. Connection to Model Serialization

Speech export extends general serialization with:
- Streaming support (state management)
- Variable-length handling (dynamic axes)
- Real-time validation (RTF benchmarks)

## 8. Key Takeaways

1. **ONNX for server**, TFLite for mobile
2. **Quantize for 2-4x speedup** with minimal accuracy loss
3. **Export streaming models** with explicit state
4. **Validate both accuracy and latency**
5. **Platform-specific optimization** (CoreML, TensorRT)

---

**Originally published at:** [arunbaby.com/speech-tech/0047-speech-model-export](https://www.arunbaby.com/speech-tech/0047-speech-model-export/)

*If you found this helpful, consider sharing it with others who might benefit.*
