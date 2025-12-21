---
title: "Model Serialization Systems"
day: 47
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - model-serialization
  - onnx
  - savedmodel
  - torchscript
  - model-export
  - model-deployment
  - mlops
difficulty: Hard
subdomain: "MLOps & Deployment"
tech_stack: Python, PyTorch, TensorFlow, ONNX, TorchScript
scale: "Models from MB to TB, sub-second load times"
companies: Google, Meta, Microsoft, NVIDIA, Hugging Face
related_dsa_day: 47
related_speech_day: 47
related_agents_day: 47
---

**"A model that can't be saved is a model that can't be deployed."**

## 1. Problem Statement

Design a **model serialization system** that saves trained ML models in portable, version-controlled formats and loads them efficiently for inference across different frameworks.

### Functional Requirements

1. **Serialization**: Convert models to persistent formats
2. **Deserialization**: Load models with minimal overhead
3. **Cross-Framework**: Support PyTorch ↔ TensorFlow ↔ ONNX
4. **Versioning**: Track versions with metadata
5. **Validation**: Verify loaded models match serialized versions

### Non-Functional Requirements

- **Loading time**: < 1 second for models up to 1GB
- **Storage efficiency**: < 20% overhead vs raw weights
- **Compatibility**: Support multiple framework versions

## 2. Understanding Model Serialization

### What Gets Serialized?

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Components                              │
├─────────────────────────────────────────────────────────────────┤
│  1. ARCHITECTURE - Layer types, connections, graph              │
│  2. WEIGHTS - Trained parameters (99% of size)                  │
│  3. METADATA - Input/output signatures, version info           │
│  4. PREPROCESSING - Tokenizers, scalers, encoders               │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Serialization Formats

### 3.1 Native PyTorch

```python
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

class PyTorchSerializer:
    """Handle PyTorch model serialization."""
    
    @staticmethod
    def save_state_dict(
        model: nn.Module,
        path: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Save state dict only (recommended for flexibility).
        
        Pros: Smaller, works across model versions
        Cons: Requires model class at load time
        """
        checkpoint = {
            'state_dict': model.state_dict(),
            'config': getattr(model, 'config', {}),
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
    
    @staticmethod
    def save_torchscript(model: nn.Module, path: str):
        """
        Save as TorchScript (recommended for production).
        
        Pros: No Python dependency, optimizable
        Cons: Not all ops supported
        """
        model.eval()
        scripted = torch.jit.script(model)
        scripted.save(path)
    
    @staticmethod
    def save_traced(
        model: nn.Module, 
        sample_input: torch.Tensor,
        path: str
    ):
        """
        Save traced model (for models with fixed control flow).
        """
        model.eval()
        traced = torch.jit.trace(model, sample_input)
        traced.save(path)
    
    @staticmethod
    def load(path: str, model_class=None, device='cpu'):
        """Load PyTorch model."""
        try:
            # Try TorchScript first
            return torch.jit.load(path, map_location=device)
        except:
            pass
        
        checkpoint = torch.load(path, map_location=device)
        
        if isinstance(checkpoint, nn.Module):
            return checkpoint
        
        if 'state_dict' in checkpoint and model_class:
            model = model_class(**checkpoint.get('config', {}))
            model.load_state_dict(checkpoint['state_dict'])
            return model
        
        raise ValueError("Cannot load model")
```

### 3.2 ONNX Format

```python
import onnx
import onnxruntime as ort
import numpy as np
from typing import List, Dict

class ONNXSerializer:
    """
    ONNX: Open Neural Network Exchange.
    
    Benefits:
    - Framework agnostic
    - Optimized runtime
    - Edge deployment ready
    """
    
    @staticmethod
    def export_pytorch(
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        input_names: List[str] = ['input'],
        output_names: List[str] = ['output'],
        dynamic_axes: Dict = None,
        opset_version: int = 14
    ):
        """Export PyTorch model to ONNX."""
        model.eval()
        
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True
        )
        
        # Validate
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        return output_path
    
    @staticmethod
    def load(path: str, providers: List[str] = None):
        """Load ONNX model for inference."""
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        return ort.InferenceSession(path, providers=providers)
    
    @staticmethod
    def infer(session, inputs: Dict[str, np.ndarray]):
        """Run ONNX inference."""
        input_names = [i.name for i in session.get_inputs()]
        ort_inputs = {name: inputs[name] for name in input_names}
        return session.run(None, ort_inputs)
    
    @staticmethod
    def optimize(input_path: str, output_path: str):
        """Optimize ONNX model."""
        import onnxoptimizer
        
        model = onnx.load(input_path)
        passes = onnxoptimizer.get_available_passes()
        optimized = onnxoptimizer.optimize(model, passes)
        onnx.save(optimized, output_path)
```

### 3.3 TensorFlow SavedModel

```python
import tensorflow as tf
from typing import Dict, Optional

class TensorFlowSerializer:
    """Handle TensorFlow model serialization."""
    
    @staticmethod
    def save(
        model: tf.keras.Model,
        path: str,
        include_optimizer: bool = False,
        signatures: Optional[Dict] = None
    ):
        """Save as SavedModel format."""
        if signatures:
            tf.saved_model.save(model, path, signatures=signatures)
        else:
            model.save(path, include_optimizer=include_optimizer)
    
    @staticmethod
    def load(path: str) -> tf.keras.Model:
        """Load SavedModel."""
        return tf.keras.models.load_model(path)
    
    @staticmethod
    def convert_to_tflite(
        saved_model_path: str,
        output_path: str,
        quantize: bool = False
    ):
        """Convert to TFLite for mobile/edge."""
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        return output_path
```

## 4. Model Registry

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import json
import hashlib
import shutil

@dataclass
class ModelVersion:
    """Represents a model version."""
    version: str
    path: str
    format: str
    created_at: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    checksum: str = ""
    status: str = "staged"  # staged, production, archived


class ModelRegistry:
    """Central registry for model versions."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Dict[str, ModelVersion]] = {}
        self._load_registry()
    
    def register(
        self,
        model_name: str,
        model_path: str,
        version: str,
        format: str,
        metrics: Dict[str, float] = None
    ) -> ModelVersion:
        """Register a new model version."""
        dest_dir = self.storage_path / model_name / version
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model
        source = Path(model_path)
        dest_path = dest_dir / source.name
        if source.is_file():
            shutil.copy2(source, dest_path)
        else:
            shutil.copytree(source, dest_path)
        
        # Compute checksum
        checksum = self._compute_checksum(dest_path)
        
        model_version = ModelVersion(
            version=version,
            path=str(dest_path),
            format=format,
            created_at=datetime.now(),
            metrics=metrics or {},
            checksum=checksum
        )
        
        if model_name not in self.models:
            self.models[model_name] = {}
        self.models[model_name][version] = model_version
        
        self._save_registry()
        return model_version
    
    def get_latest(self, model_name: str) -> Optional[ModelVersion]:
        """Get latest version."""
        if model_name not in self.models:
            return None
        
        versions = sorted(
            self.models[model_name].keys(),
            key=lambda v: [int(x) for x in v.split('.')]
        )
        return self.models[model_name][versions[-1]] if versions else None
    
    def get_production(self, model_name: str) -> Optional[ModelVersion]:
        """Get production version."""
        if model_name not in self.models:
            return None
        
        for v in self.models[model_name].values():
            if v.status == "production":
                return v
        return None
    
    def promote(self, model_name: str, version: str) -> bool:
        """Promote version to production."""
        if model_name not in self.models:
            return False
        
        # Demote current production
        for v in self.models[model_name].values():
            if v.status == "production":
                v.status = "archived"
        
        # Promote new version
        if version in self.models[model_name]:
            self.models[model_name][version].status = "production"
            self._save_registry()
            return True
        return False
    
    def _compute_checksum(self, path: Path) -> str:
        hasher = hashlib.sha256()
        if path.is_file():
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
        else:
            for file in sorted(path.rglob('*')):
                if file.is_file():
                    with open(file, 'rb') as f:
                        for chunk in iter(lambda: f.read(8192), b''):
                            hasher.update(chunk)
        return hasher.hexdigest()
    
    def _load_registry(self):
        registry_file = self.storage_path / 'registry.json'
        if registry_file.exists():
            with open(registry_file) as f:
                data = json.load(f)
            for model_name, versions in data.items():
                self.models[model_name] = {}
                for version, v_data in versions.items():
                    v_data['created_at'] = datetime.fromisoformat(v_data['created_at'])
                    self.models[model_name][version] = ModelVersion(**v_data)
    
    def _save_registry(self):
        registry_file = self.storage_path / 'registry.json'
        data = {}
        for model_name, versions in self.models.items():
            data[model_name] = {}
            for version, v in versions.items():
                data[model_name][version] = {
                    'version': v.version,
                    'path': v.path,
                    'format': v.format,
                    'created_at': v.created_at.isoformat(),
                    'metrics': v.metrics,
                    'checksum': v.checksum,
                    'status': v.status
                }
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
```

## 5. Large Model Serialization

```python
class ShardedSerializer:
    """Serialize large models in shards."""
    
    SHARD_SIZE = 2 * 1024**3  # 2GB
    
    def save_sharded(
        self,
        state_dict: Dict[str, torch.Tensor],
        output_dir: str
    ) -> List[str]:
        """Save model in 2GB shards."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        shards = []
        current_shard = {}
        current_size = 0
        shard_idx = 0
        index = {}
        
        for name, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            
            if current_size + tensor_size > self.SHARD_SIZE and current_shard:
                path = self._save_shard(output_dir, shard_idx, current_shard)
                shards.append(path)
                current_shard = {}
                current_size = 0
                shard_idx += 1
            
            current_shard[name] = tensor
            index[name] = shard_idx
            current_size += tensor_size
        
        if current_shard:
            path = self._save_shard(output_dir, shard_idx, current_shard)
            shards.append(path)
        
        # Save index
        with open(output_dir / 'index.json', 'w') as f:
            json.dump(index, f)
        
        return shards
    
    def _save_shard(self, output_dir, idx, data):
        path = output_dir / f"shard_{idx:05d}.pt"
        torch.save(data, str(path))
        return str(path)
    
    def load_sharded(self, model_dir: str, device='cpu'):
        """Load sharded model."""
        model_dir = Path(model_dir)
        
        with open(model_dir / 'index.json') as f:
            index = json.load(f)
        
        shard_indices = set(index.values())
        state_dict = {}
        
        for shard_idx in shard_indices:
            shard_path = model_dir / f"shard_{shard_idx:05d}.pt"
            shard_data = torch.load(str(shard_path), map_location=device)
            state_dict.update(shard_data)
        
        return state_dict
```

## 6. Quantized Serialization

```python
class QuantizedSerializer:
    """Serialize quantized models."""
    
    @staticmethod
    def quantize_dynamic(model: nn.Module, output_path: str):
        """Dynamic quantization for inference."""
        model.eval()
        
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )
        
        torch.jit.save(torch.jit.script(quantized), output_path)
        return quantized
    
    @staticmethod
    def quantize_onnx(model_path: str, output_path: str):
        """Quantize ONNX model."""
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantize_dynamic(
            model_path,
            output_path,
            weight_type=QuantType.QInt8
        )
        return output_path
```

## 7. Validation Layer

```python
class ModelValidator:
    """Validate serialized models."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    def validate(
        self,
        model_name: str,
        version: str,
        test_input,
        expected_output=None,
        tolerance=1e-5
    ) -> Dict[str, bool]:
        """Run all validation checks."""
        model_version = self.registry.get_latest(model_name)
        if not model_version:
            return {'exists': False}
        
        results = {'exists': True}
        
        # Integrity check
        results['integrity'] = self._check_integrity(model_version)
        
        # Schema check
        results['schema'] = self._check_schema(model_version)
        
        # Inference check
        if expected_output is not None:
            results['inference'] = self._check_inference(
                model_version, test_input, expected_output, tolerance
            )
        
        return results
    
    def _check_integrity(self, model_version: ModelVersion) -> bool:
        """Verify checksum."""
        current = self._compute_checksum(Path(model_version.path))
        return current == model_version.checksum
    
    def _check_schema(self, model_version: ModelVersion) -> bool:
        """Verify model loads correctly."""
        try:
            if model_version.format == 'pytorch':
                torch.jit.load(model_version.path)
            elif model_version.format == 'onnx':
                onnx.load(model_version.path)
                onnx.checker.check_model(onnx.load(model_version.path))
            return True
        except:
            return False
    
    def _check_inference(self, model_version, test_input, expected, tol):
        """Verify inference produces expected output."""
        if model_version.format == 'pytorch':
            model = torch.jit.load(model_version.path)
            model.eval()
            with torch.no_grad():
                actual = model(test_input)
            return torch.allclose(actual, expected, atol=tol)
        
        elif model_version.format == 'onnx':
            session = ort.InferenceSession(model_version.path)
            input_name = session.get_inputs()[0].name
            actual = session.run(None, {input_name: test_input.numpy()})[0]
            return np.allclose(actual, expected.numpy(), atol=tol)
        
        return False
```

## 8. Connection to Tree Serialization

Model serialization and tree serialization share core principles:

| Concept | Tree Serialization | Model Serialization |
|---------|-------------------|---------------------|
| Structure | Parent-child nodes | Layer graph |
| Values | Node values | Weight tensors |
| Null handling | Null markers | Optional layers |
| Reconstruction | From encoded string | From saved format |
| Validation | Structure check | Inference test |

Both solve: **preserving graph structure in linear storage**.

## 9. Key Takeaways

1. **Choose the right format**: Native for dev, ONNX for production
2. **Version everything together**: Weights + config + preprocessing
3. **Handle large models**: Shard for models > memory
4. **Validate before deploy**: Schema + integrity + inference checks
5. **Enable format conversion**: PyTorch ↔ ONNX ↔ TensorFlow

---

**Originally published at:** [arunbaby.com/ml-system-design/0047-model-serialization](https://www.arunbaby.com/ml-system-design/0047-model-serialization/)

*If you found this helpful, consider sharing it with others who might benefit.*
