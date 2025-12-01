---
title: "Model Ensembling"
day: 14
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - ensemble-learning
  - model-combination
  - voting
  - stacking
  - boosting
  - bagging
  - production-ml
subdomain: "Model Architecture"
tech_stack: [Python, scikit-learn, XGBoost, LightGBM, Ray, Kubernetes]
scale: "10+ models, 100K+ predictions/sec, multi-region deployment"
companies: [Netflix, Spotify, Google, Meta, Airbnb, Uber]
related_dsa_day: 14
related_speech_day: 14
---

**Build production ensemble systems that combine multiple models using backtracking strategies to explore optimal combinations.**

## Problem Statement

Design a **Model Ensembling System** that combines predictions from multiple ML models to achieve better accuracy, robustness, and reliability than any single model.

### Functional Requirements

1. **Model combination:** Aggregate predictions from N heterogeneous models
2. **Combination strategies:** Support voting, averaging, stacking, boosting
3. **Dynamic selection:** Choose best subset of models based on input characteristics
4. **Confidence scoring:** Provide uncertainty estimates
5. **Fallback handling:** Gracefully handle model failures
6. **A/B testing:** Compare ensemble vs individual models
7. **Model versioning:** Support multiple versions of same model
8. **Real-time inference:** Serve predictions with low latency

### Non-Functional Requirements

1. **Latency:** p95 < 100ms for inference
2. **Throughput:** 100K+ predictions/second
3. **Accuracy:** +5-10% improvement over single best model
4. **Availability:** 99.95% uptime (handle individual model failures)
5. **Scalability:** Support 100+ models in ensemble
6. **Cost efficiency:** Optimal resource usage
7. **Explainability:** Understand why ensemble made prediction

## Understanding the Requirements

Model ensembles are **widely used in production** because they:

1. **Improve accuracy:** Reduce bias and variance
2. **Increase robustness:** No single point of failure
3. **Handle uncertainty:** Better calibrated confidence scores
4. **Leverage diversity:** Different models capture different patterns

### When to Use Ensembles

**Good use cases:**
- **High-stakes predictions:** Fraud detection, medical diagnosis
- **Complex problems:** Multiple weak signals
- **Competitive ML:** Kaggle, research benchmarks
- **Production stability:** Reduce risk of single model failure

**Not ideal when:**
- **Latency critical:** <10ms requirements
- **Resource constrained:** Mobile/edge deployment
- **Interpretability required:** Individual model predictions needed
- **Simple problem:** Single model already achieves 99%+ accuracy

### Real-World Examples

| Company | Use Case | Ensemble Approach | Results |
|---------|----------|-------------------|---------|
| Netflix | Recommendation | Collaborative filtering + content-based + deep learning | +10% engagement |
| Spotify | Music recommendation | Audio features + CF + NLP + context | +15% listening time |
| Airbnb | Price prediction | GBM + Linear + Neural network | -5% RMSE |
| Uber | ETA prediction | LightGBM ensemble + traffic models | +12% accuracy |
| Kaggle Winners | Various | Stacked ensembles of 50-100 models | Consistent top ranks |

### The Backtracking Connection

Just like the **Generate Parentheses** problem:

| Generate Parentheses | Model Ensembling |
|----------------------|------------------|
| Generate valid string combinations | Generate valid model combinations |
| Constraints: balanced parens | Constraints: latency, diversity, accuracy |
| Backtracking to explore all paths | Backtracking to explore ensemble configurations |
| Prune invalid branches early | Prune underperforming combinations early |
| Result: all valid strings | Result: all viable ensembles |

**Core pattern:** Use backtracking to explore the space of possible model combinations and select the best one.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Ensemble System                             │
└─────────────────────────────────────────────────────────────────┘

                            ┌──────────────┐
                            │   Request    │
                            │   (Features) │
                            └──────┬───────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Ensemble Orchestrator     │
                    │   - Route to models         │
                    │   - Collect predictions     │
                    │   - Apply combination       │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼────────┐        ┌───────▼────────┐        ┌───────▼────────┐
│   Model 1      │        │   Model 2      │        │   Model N      │
│   (XGBoost)    │        │   (Neural Net) │        │   (Linear)     │
│                │        │                │        │                │
│  Pred: 0.85    │        │  Pred: 0.72    │        │  Pred: 0.79    │
└────────┬───────┘        └────────┬───────┘        └────────┬───────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Combiner                  │
                    │   - Voting / Averaging      │
                    │   - Stacking                │
                    │   - Weighted combination    │
                    └──────────────┬──────────────┘
                                   │
                            ┌──────▼───────┐
                            │  Final Pred  │
                            │   0.80       │
                            │  (conf 0.92) │
                            └──────────────┘
```

### Key Components

1. **Ensemble Orchestrator:** Routes requests, manages model execution
2. **Base Models:** Individual models (diverse architectures)
3. **Combiner:** Aggregates predictions using chosen strategy
4. **Meta-learner:** (Optional) Learns how to combine predictions
5. **Monitoring:** Tracks individual and ensemble performance

## Component Deep-Dives

### 1. Ensemble Orchestrator - Model Selection

The orchestrator decides which models to query using backtracking:

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from enum import Enum
import asyncio
import time

class ModelStatus(Enum):
    """Model health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"

@dataclass
class Model:
    """Represents a single model in the ensemble."""
    model_id: str
    model_type: str  # "xgboost", "neural_net", "linear", etc.
    version: str
    avg_latency_ms: float
    accuracy: float  # On validation set
    status: ModelStatus = ModelStatus.HEALTHY
    
    # For diversity
    architecture: str = ""
    training_data: str = ""
    
    async def predict(self, features: Dict) -> float:
        """Make prediction (async for parallel execution)."""
        # Simulate prediction
        await asyncio.sleep(self.avg_latency_ms / 1000.0)
        
        # In production: call actual model
        # return self.model.predict(features)
        
        # For demo: return dummy prediction
        return 0.5 + hash(self.model_id) % 50 / 100.0

@dataclass
class EnsembleConfig:
    """Configuration for ensemble."""
    max_models: int = 10
    max_latency_ms: float = 100.0
    min_diversity: float = 0.3  # Min difference in architecture
    combination_strategy: str = "voting"  # "voting", "averaging", "stacking"
    
@dataclass
class EnsembleResult:
    """Result from ensemble prediction."""
    prediction: float
    confidence: float
    models_used: List[str]
    latency_ms: float
    individual_predictions: Dict[str, float]


class EnsembleOrchestrator:
    """
    Orchestrates ensemble prediction using backtracking for model selection.
    
    Similar to Generate Parentheses:
    - Explore combinations of models
    - Prune combinations that violate constraints
    - Select optimal subset
    """
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models: List[Model] = []
        
    def add_model(self, model: Model):
        """Add a model to the ensemble."""
        self.models.append(model)
    
    def select_models_backtracking(
        self,
        features: Dict,
        max_latency: float
    ) -> List[Model]:
        """
        Select best subset of models using backtracking.
        
        Similar to Generate Parentheses backtracking:
        1. Start with empty selection
        2. Try adding each model
        3. Check constraints (latency, diversity)
        4. Recurse to try more models
        5. Backtrack if constraints violated
        
        Constraints:
        - Total latency <= max_latency
        - Model diversity >= min_diversity
        - Number of models <= max_models
        
        Returns:
            List of selected models
        """
        best_selection = []
        best_score = -float('inf')
        
        def calculate_diversity(models: List[Model]) -> float:
            """Calculate diversity score for model set."""
            if len(models) <= 1:
                return 1.0
            
            # Diversity = fraction of unique architectures
            unique_archs = len(set(m.architecture for m in models))
            return unique_archs / len(models)
        
        def estimate_accuracy(models: List[Model]) -> float:
            """Estimate ensemble accuracy from individual models."""
            if not models:
                return 0.0
            
            # Simple heuristic: weighted average with diversity bonus
            avg_acc = sum(m.accuracy for m in models) / len(models)
            diversity_bonus = calculate_diversity(models) * 0.1
            return avg_acc + diversity_bonus
        
        def backtrack(
            index: int,
            current_selection: List[Model],
            current_latency: float
        ):
            """
            Backtracking function to explore model combinations.
            
            Args:
                index: Current model index to consider
                current_selection: Models selected so far
                current_latency: Cumulative latency
            """
            nonlocal best_selection, best_score
            
            # Base case: evaluated all models
            if index == len(self.models):
                if current_selection:
                    score = estimate_accuracy(current_selection)
                    if score > best_score:
                        best_score = score
                        best_selection = current_selection[:]
                return
            
            model = self.models[index]
            
            # Skip unhealthy models
            if model.status != ModelStatus.HEALTHY:
                backtrack(index + 1, current_selection, current_latency)
                return
            
            # Choice 1: Include current model (if constraints satisfied)
            new_latency = current_latency + model.avg_latency_ms
            
            can_add = (
                len(current_selection) < self.config.max_models and
                new_latency <= max_latency and
                calculate_diversity(current_selection + [model]) >= self.config.min_diversity
            )
            
            if can_add:
                current_selection.append(model)
                backtrack(index + 1, current_selection, new_latency)
                current_selection.pop()  # Backtrack
            
            # Choice 2: Skip current model
            backtrack(index + 1, current_selection, current_latency)
        
        # Start backtracking
        backtrack(0, [], 0.0)
        
        # Ensure at least one model
        if not best_selection and self.models:
            # Fallback: use single best model
            best_selection = [max(self.models, key=lambda m: m.accuracy)]
        
        return best_selection
    
    async def predict(self, features: Dict) -> EnsembleResult:
        """
        Make ensemble prediction.
        
        Steps:
        1. Select models using backtracking
        2. Query selected models in parallel
        3. Combine predictions
        4. Return result with metadata
        """
        start_time = time.perf_counter()
        
        # Select models
        selected_models = self.select_models_backtracking(
            features,
            max_latency=self.config.max_latency_ms
        )
        
        # Query models in parallel (async)
        prediction_tasks = [
            model.predict(features)
            for model in selected_models
        ]
        
        predictions = await asyncio.gather(*prediction_tasks)
        
        # Build predictions map
        pred_map = {
            model.model_id: pred
            for model, pred in zip(selected_models, predictions)
        }
        
        # Combine predictions
        final_pred, confidence = self._combine_predictions(
            selected_models,
            predictions
        )
        
        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return EnsembleResult(
            prediction=final_pred,
            confidence=confidence,
            models_used=[m.model_id for m in selected_models],
            latency_ms=latency_ms,
            individual_predictions=pred_map
        )
    
    def _combine_predictions(
        self,
        models: List[Model],
        predictions: List[float]
    ) -> tuple[float, float]:
        """
        Combine predictions using configured strategy.
        
        Returns:
            (final_prediction, confidence)
        """
        if self.config.combination_strategy == "voting":
            # For binary classification: majority vote
            votes = [1 if p > 0.5 else 0 for p in predictions]
            final = sum(votes) / len(votes)
            confidence = abs(final - 0.5) * 2  # How confident is majority
            
        elif self.config.combination_strategy == "averaging":
            # Simple average
            final = sum(predictions) / len(predictions)
            
            # Confidence based on agreement
            variance = sum((p - final) ** 2 for p in predictions) / len(predictions)
            confidence = 1.0 / (1.0 + variance)  # High agreement = high confidence
            
        elif self.config.combination_strategy == "weighted_averaging":
            # Weight by model accuracy
            total_weight = sum(m.accuracy for m in models)
            final = sum(
                m.accuracy * p
                for m, p in zip(models, predictions)
            ) / total_weight
            
            # Weighted variance for confidence
            variance = sum(
                m.accuracy * (p - final) ** 2
                for m, p in zip(models, predictions)
            ) / total_weight
            confidence = 1.0 / (1.0 + variance)
            
        else:
            # Default: simple average
            final = sum(predictions) / len(predictions)
            confidence = 0.5
        
        return final, confidence
```

### 2. Combination Strategies

Different strategies for combining model predictions:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

class EnsembleCombiner:
    """Different strategies for combining model predictions."""
    
    @staticmethod
    def simple_voting(predictions: List[float], threshold: float = 0.5) -> float:
        """
        Majority voting for binary classification.
        
        Each model votes 0 or 1, return majority.
        """
        votes = [1 if p > threshold else 0 for p in predictions]
        return sum(votes) / len(votes)
    
    @staticmethod
    def weighted_voting(
        predictions: List[float],
        weights: List[float]
    ) -> float:
        """
        Weighted voting.
        
        Models with higher accuracy get more weight.
        """
        total_weight = sum(weights)
        return sum(w * p for w, p in zip(weights, predictions)) / total_weight
    
    @staticmethod
    def simple_averaging(predictions: List[float]) -> float:
        """Simple arithmetic mean."""
        return sum(predictions) / len(predictions)
    
    @staticmethod
    def geometric_mean(predictions: List[float]) -> float:
        """
        Geometric mean - useful when models have different scales.
        
        Formula: (p1 * p2 * ... * pn)^(1/n)
        """
        product = 1.0
        for p in predictions:
            product *= max(p, 1e-10)  # Avoid zero
        return product ** (1.0 / len(predictions))
    
    @staticmethod
    def rank_averaging(predictions: List[float]) -> float:
        """
        Average of ranks instead of raw predictions.
        
        Useful when models have different scales/calibrations.
        """
        # Sort predictions and assign ranks
        sorted_preds = sorted(enumerate(predictions), key=lambda x: x[1])
        ranks = [0] * len(predictions)
        
        for rank, (idx, _) in enumerate(sorted_preds):
            ranks[idx] = rank
        
        # Normalize ranks to [0, 1]
        avg_rank = sum(ranks) / len(ranks)
        return avg_rank / (len(ranks) - 1) if len(ranks) > 1 else 0.5


class StackingCombiner:
    """
    Stacking: Train a meta-model to combine base model predictions.
    
    This is the most powerful but also most complex approach.
    """
    
    def __init__(self):
        self.meta_model = LogisticRegression()
        self.is_trained = False
    
    def train(
        self,
        base_predictions: np.ndarray,  # Shape: (n_samples, n_models)
        true_labels: np.ndarray
    ):
        """
        Train meta-model on base model predictions.
        
        Args:
            base_predictions: Predictions from base models (holdout set)
            true_labels: True labels
        """
        self.meta_model.fit(base_predictions, true_labels)
        self.is_trained = True
    
    def predict(self, base_predictions: np.ndarray) -> np.ndarray:
        """
        Predict using meta-model.
        
        Args:
            base_predictions: Predictions from base models
            
        Returns:
            Final ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Meta-model not trained. Call train() first.")
        
        return self.meta_model.predict_proba(base_predictions)[:, 1]
    
    def get_model_importances(self) -> Dict[int, float]:
        """
        Get feature importances (which base models are most important).
        
        Returns:
            Dictionary mapping model index to importance
        """
        if not self.is_trained:
            return {}
        
        # For logistic regression, coefficients indicate importance
        coeffs = np.abs(self.meta_model.coef_[0])
        normalized = coeffs / coeffs.sum()
        
        return {i: float(imp) for i, imp in enumerate(normalized)}
```

### 3. Diversity Optimization

Diverse models make better ensembles. Here's how to measure and ensure diversity:

```python
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import numpy as np

class DiversityAnalyzer:
    """Analyze and optimize model diversity in ensemble."""
    
    @staticmethod
    def prediction_diversity(
        predictions: np.ndarray  # Shape: (n_samples, n_models)
    ) -> float:
        """
        Calculate diversity based on prediction disagreement.
        
        High diversity = models make different predictions.
        
        Returns:
            Diversity score in [0, 1]
        """
        n_models = predictions.shape[1]
        
        if n_models <= 1:
            return 0.0
        
        # Calculate pairwise correlation between model predictions
        correlations = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr, _ = spearmanr(predictions[:, i], predictions[:, j])
                correlations.append(corr)
        
        # Diversity = 1 - average correlation
        avg_correlation = np.mean(correlations)
        diversity = 1.0 - avg_correlation
        
        return max(0.0, diversity)
    
    @staticmethod
    def architectural_diversity(models: List[Model]) -> float:
        """
        Calculate diversity based on model architectures.
        
        Different architectures (XGBoost, NN, Linear) = high diversity.
        """
        if len(models) <= 1:
            return 0.0
        
        # Count unique architectures
        unique_archs = len(set(m.architecture for m in models))
        
        # Diversity = ratio of unique to total
        return unique_archs / len(models)
    
    @staticmethod
    def error_diversity(
        predictions: np.ndarray,  # Shape: (n_samples, n_models)
        true_labels: np.ndarray
    ) -> float:
        """
        Calculate diversity based on error patterns.
        
        Good diversity = models make errors on different samples.
        
        Returns:
            Error diversity score
        """
        n_samples, n_models = predictions.shape
        
        # Determine which samples each model gets wrong
        errors = (predictions > 0.5) != true_labels.reshape(-1, 1)
        
        # Calculate pairwise error overlap
        overlaps = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # What fraction of errors are shared?
                shared_errors = np.sum(errors[:, i] & errors[:, j])
                total_errors = np.sum(errors[:, i] | errors[:, j])
                
                if total_errors > 0:
                    overlap = shared_errors / total_errors
                    overlaps.append(overlap)
        
        # Diversity = 1 - average overlap
        avg_overlap = np.mean(overlaps) if overlaps else 0.5
        return 1.0 - avg_overlap
    
    @staticmethod
    def select_diverse_subset(
        models: List[Model],
        predictions: np.ndarray,  # Shape: (n_samples, n_models)
        k: int  # Number of models to select
    ) -> List[int]:
        """
        Select k most diverse models using greedy algorithm.
        
        Similar to backtracking but greedy instead of exhaustive.
        
        Algorithm:
        1. Start with best individual model
        2. Iteratively add model that maximizes diversity
        3. Stop when k models selected
        
        Returns:
            Indices of selected models
        """
        n_models = len(models)
        
        if k >= n_models:
            return list(range(n_models))
        
        # Start with best model
        accuracies = [m.accuracy for m in models]
        selected = [np.argmax(accuracies)]
        
        # Greedily add most diverse models
        for _ in range(k - 1):
            max_diversity = -1
            best_candidate = -1
            
            for candidate in range(n_models):
                if candidate in selected:
                    continue
                
                # Calculate diversity if we add this candidate
                test_selection = selected + [candidate]
                test_predictions = predictions[:, test_selection]
                
                diversity = DiversityAnalyzer.prediction_diversity(test_predictions)
                
                if diversity > max_diversity:
                    max_diversity = diversity
                    best_candidate = candidate
            
            if best_candidate >= 0:
                selected.append(best_candidate)
        
        return selected
```

### 4. Dynamic Ensemble Selection

Select different model subsets based on input characteristics:

```python
from sklearn.cluster import KMeans
from typing import Callable

class DynamicEnsembleSelector:
    """
    Dynamic ensemble selection: choose models based on input.
    
    Idea: Different models are good for different types of inputs.
    
    Example:
    - Linear models good for simple patterns
    - Neural nets good for complex patterns
    - Tree models good for categorical features
    """
    
    def __init__(self, models: List[Model], n_regions: int = 5):
        self.models = models
        self.n_regions = n_regions
        
        # Cluster validation set to identify regions
        self.clusterer = KMeans(n_clusters=n_regions, random_state=42)
        
        # Best models for each region
        self.region_models: Dict[int, List[int]] = {}
        
        self.is_trained = False
    
    def train(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_predictions: np.ndarray  # Shape: (n_samples, n_models)
    ):
        """
        Train selector on validation data.
        
        Steps:
        1. Cluster input space into regions
        2. For each region, find best models
        3. Store region -> models mapping
        """
        # Cluster input space
        self.clusterer.fit(X_val)
        clusters = self.clusterer.labels_
        
        # For each region, find best models
        for region in range(self.n_regions):
            region_mask = clusters == region
            region_y = y_val[region_mask]
            region_preds = model_predictions[region_mask]
            
            # Evaluate each model on this region
            model_scores = []
            
            for model_idx in range(len(self.models)):
                preds = region_preds[:, model_idx]
                
                # Calculate accuracy for this model in this region
                accuracy = np.mean((preds > 0.5) == region_y)
                model_scores.append((model_idx, accuracy))
            
            # Sort by accuracy and take top models
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 3 models for this region
            self.region_models[region] = [idx for idx, _ in model_scores[:3]]
        
        self.is_trained = True
    
    def select_models(self, features: np.ndarray) -> List[int]:
        """
        Select best models for given input.
        
        Args:
            features: Input features (single sample)
            
        Returns:
            Indices of selected models
        """
        if not self.is_trained:
            # Fallback: use all models
            return list(range(len(self.models)))
        
        # Determine which region this input belongs to
        region = self.clusterer.predict(features.reshape(1, -1))[0]
        
        # Return best models for this region
        return self.region_models.get(region, list(range(len(self.models))))
```

## Data Flow

### Prediction Pipeline

```
1. Request arrives with features
   └─> Feature preprocessing/validation

2. Model selection (backtracking or dynamic)
   └─> Identify optimal subset of models
   └─> Consider: latency budget, diversity, accuracy

3. Parallel inference
   └─> Query selected models concurrently
   └─> Set timeout for each model
   └─> Handle failures gracefully

4. Prediction combination
   └─> Apply combination strategy
   └─> Calculate confidence score

5. Post-processing
   └─> Calibration
   └─> Threshold optimization
   └─> Explanation generation

6. Return result
   └─> Final prediction
   └─> Confidence
   └─> Models used
   └─> Latency breakdown
```

### Training Pipeline

```
1. Train base models
   ├─> Different algorithms
   ├─> Different feature sets
   ├─> Different train/val splits
   └─> Ensure diversity

2. Generate meta-features (for stacking)
   └─> Cross-validation predictions
   └─> Avoid overfitting

3. Train meta-model
   └─> Learn optimal combination
   └─> Regularization to prevent overfitting

4. Evaluate ensemble
   └─> Compare to individual models
   └─> A/B test in production

5. Deploy
   └─> Canary rollout
   └─> Monitor performance
```

## Scaling Strategies

### Horizontal Scaling - Parallel Inference

```python
import ray

@ray.remote
class ModelServer:
    """Ray actor for serving a single model."""
    
    def __init__(self, model: Model):
        self.model = model
        # Load actual model weights
        # self.model_impl = load_model(model.model_id)
    
    def predict(self, features: Dict) -> float:
        """Make prediction."""
        # return self.model_impl.predict(features)
        return 0.5  # Dummy


class DistributedEnsemble:
    """Distributed ensemble using Ray."""
    
    def __init__(self, models: List[Model]):
        # Create Ray actor for each model
        self.model_servers = [
            ModelServer.remote(model)
            for model in models
        ]
        self.models = models
    
    async def predict(self, features: Dict) -> EnsembleResult:
        """Make distributed prediction."""
        # Query all models in parallel using Ray
        prediction_futures = [
            server.predict.remote(features)
            for server in self.model_servers
        ]
        
        # Wait for all predictions
        predictions = await asyncio.gather(*[
            asyncio.create_task(self._ray_to_asyncio(future))
            for future in prediction_futures
        ])
        
        # Combine predictions
        final_pred = sum(predictions) / len(predictions)
        
        return EnsembleResult(
            prediction=final_pred,
            confidence=0.8,
            models_used=[m.model_id for m in self.models],
            latency_ms=0.0,
            individual_predictions={}
        )
    
    @staticmethod
    async def _ray_to_asyncio(ray_future):
        """Convert Ray future to asyncio."""
        return ray.get(ray_future)
```

### Vertical Scaling - Model Compression

```python
class EnsembleOptimizer:
    """Optimize ensemble for production."""
    
    @staticmethod
    def knowledge_distillation(
        ensemble: EnsembleOrchestrator,
        X_train: np.ndarray,
        student_model: any
    ):
        """
        Distill ensemble into single student model.
        
        Benefits:
        - Single model = lower latency
        - Retains most of ensemble's accuracy
        - Easier deployment
        
        Process:
        1. Generate ensemble predictions on training data
        2. Train student model to mimic ensemble
        3. Use soft labels (probabilities) not hard labels
        """
        # Get ensemble predictions (soft labels)
        ensemble_preds = []
        
        for x in X_train:
            result = ensemble.predict(x)
            ensemble_preds.append(result.prediction)
        
        ensemble_preds = np.array(ensemble_preds)
        
        # Train student model
        student_model.fit(X_train, ensemble_preds)
        
        return student_model
    
    @staticmethod
    def prune_models(
        models: List[Model],
        predictions: np.ndarray,
        true_labels: np.ndarray,
        target_size: int
    ) -> List[int]:
        """
        Prune ensemble to target size while maintaining accuracy.
        
        Greedy algorithm:
        1. Start with full ensemble
        2. Iteratively remove least important model
        3. Stop when target size reached or accuracy drops
        
        Returns:
            Indices of models to keep
        """
        n_models = len(models)
        remaining = list(range(n_models))
        
        # Calculate baseline accuracy
        ensemble_preds = predictions[:, remaining].mean(axis=1)
        baseline_acc = np.mean((ensemble_preds > 0.5) == true_labels)
        
        while len(remaining) > target_size:
            min_impact = float('inf')
            model_to_remove = -1
            
            # Try removing each model
            for model_idx in remaining:
                test_remaining = [m for m in remaining if m != model_idx]
                
                if not test_remaining:
                    break
                
                # Evaluate ensemble without this model
                test_preds = predictions[:, test_remaining].mean(axis=1)
                test_acc = np.mean((test_preds > 0.5) == true_labels)
                
                # How much does accuracy drop?
                impact = baseline_acc - test_acc
                
                if impact < min_impact:
                    min_impact = impact
                    model_to_remove = model_idx
            
            if model_to_remove < 0:
                break
            
            # Remove least important model
            remaining.remove(model_to_remove)
            
            # Update baseline
            ensemble_preds = predictions[:, remaining].mean(axis=1)
            baseline_acc = np.mean((ensemble_preds > 0.5) == true_labels)
        
        return remaining
```

## Implementation: Complete System

```python
import logging
from typing import List, Dict, Optional
import numpy as np

class ProductionEnsemble:
    """
    Complete production ensemble system.
    
    Features:
    - Model selection using backtracking
    - Multiple combination strategies
    - Fallback handling
    - Performance monitoring
    - A/B testing support
    """
    
    def __init__(
        self,
        models: List[Model],
        config: EnsembleConfig,
        combiner_type: str = "weighted_averaging"
    ):
        self.orchestrator = EnsembleOrchestrator(config)
        
        # Add models to orchestrator
        for model in models:
            self.orchestrator.add_model(model)
        
        self.combiner_type = combiner_type
        self.logger = logging.getLogger(__name__)
        
        # Metrics
        self.prediction_count = 0
        self.total_latency = 0.0
        self.fallback_count = 0
    
    async def predict(
        self,
        features: Dict,
        explain: bool = False
    ) -> Dict:
        """
        Make ensemble prediction with optional explanation.
        
        Args:
            features: Input features
            explain: Whether to include explanation
            
        Returns:
            Dictionary with prediction and metadata
        """
        try:
            # Get ensemble prediction
            result = await self.orchestrator.predict(features)
            
            # Update metrics
            self.prediction_count += 1
            self.total_latency += result.latency_ms
            
            # Build response
            response = {
                "prediction": result.prediction,
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
                "models_used": result.models_used,
                "success": True
            }
            
            # Add explanation if requested
            if explain:
                response["explanation"] = self._generate_explanation(result)
            
            self.logger.info(
                f"Prediction: {result.prediction:.3f} "
                f"(confidence: {result.confidence:.3f}, "
                f"latency: {result.latency_ms:.1f}ms, "
                f"models: {len(result.models_used)})"
            )
            
            return response
            
        except Exception as e:
            # Fallback: use simple heuristic or cached result
            self.fallback_count += 1
            self.logger.error(f"Ensemble prediction failed: {e}")
            
            return {
                "prediction": 0.5,  # Neutral prediction
                "confidence": 0.0,
                "latency_ms": 0.0,
                "models_used": [],
                "success": False,
                "error": str(e)
            }
    
    def _generate_explanation(self, result: EnsembleResult) -> Dict:
        """
        Generate explanation for ensemble prediction.
        
        Returns:
            Dictionary with explanation details
        """
        # Analyze which models contributed most
        preds = list(result.individual_predictions.values())
        final_pred = result.prediction
        
        # Calculate agreement
        agreements = [
            1.0 - abs(p - final_pred)
            for p in preds
        ]
        
        # Sort models by agreement
        model_agreements = sorted(
            zip(result.models_used, agreements),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "final_prediction": final_pred,
            "model_contributions": [
                {
                    "model_id": model_id,
                    "agreement": agreement,
                    "prediction": result.individual_predictions[model_id]
                }
                for model_id, agreement in model_agreements
            ],
            "consensus_level": sum(agreements) / len(agreements) if agreements else 0.0
        }
    
    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        return {
            "prediction_count": self.prediction_count,
            "avg_latency_ms": (
                self.total_latency / self.prediction_count
                if self.prediction_count > 0 else 0.0
            ),
            "fallback_rate": (
                self.fallback_count / self.prediction_count
                if self.prediction_count > 0 else 0.0
            ),
            "models_available": len(self.orchestrator.models),
            "healthy_models": sum(
                1 for m in self.orchestrator.models
                if m.status == ModelStatus.HEALTHY
            )
        }


# Example usage
async def main():
    # Create models
    models = [
        Model("xgb_v1", "xgboost", "1.0", 15.0, 0.85, architecture="tree"),
        Model("nn_v1", "neural_net", "1.0", 25.0, 0.87, architecture="deep_learning"),
        Model("lr_v1", "linear", "1.0", 5.0, 0.80, architecture="linear"),
        Model("lgbm_v1", "lightgbm", "1.0", 12.0, 0.86, architecture="tree"),
        Model("rf_v1", "random_forest", "1.0", 20.0, 0.84, architecture="tree"),
    ]
    
    # Configure ensemble
    config = EnsembleConfig(
        max_models=3,
        max_latency_ms=50.0,
        min_diversity=0.3,
        combination_strategy="weighted_averaging"
    )
    
    # Create ensemble
    ensemble = ProductionEnsemble(models, config)
    
    # Make predictions
    features = {"feature1": 1.0, "feature2": 0.5}
    
    result = await ensemble.predict(features, explain=True)
    print(f"Prediction: {result}")
    
    # Get metrics
    metrics = ensemble.get_metrics()
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Real-World Case Study: Netflix Recommendation Ensemble

### Netflix's Approach

Netflix uses one of the most sophisticated ensemble systems in production:

**Architecture:**
1. **100+ base models:**
   - Collaborative filtering (matrix factorization)
   - Content-based filtering (metadata)
   - Deep learning (sequential models)
   - Contextual bandits (A/B testing integration)
   - Session-based models (recent activity)

2. **Ensemble strategy:**
   - Blending (weighted combination)
   - Separate ensembles for different contexts (homepage, search, continue watching)
   - Dynamic weights based on user segment

3. **Model selection:**
   - Not all models run for every request
   - Dynamic selection based on:
     * User type (new vs established)
     * Device (mobile vs TV vs web)
     * Time of day
     * Available data

4. **Combination:**
   - Learned weights (meta-learning)
   - Context-specific weights
   - Fallback to simpler models if latency budget exceeded

**Results:**
- **+10% engagement** vs single best model
- **p95 latency: 80ms** despite 100+ models
- **Cost optimization:** Only query necessary models
- **A/B testing:** Continuous experimentation with ensemble configs

### Key Lessons

1. **More models ≠ better:** Diminishing returns after ~20 diverse models
2. **Diversity matters more than individual accuracy**
3. **Dynamic selection crucial for latency**
4. **Meta-learning (stacking) outperforms simple averaging**
5. **Context-aware ensembles beat one-size-fits-all**

## Cost Analysis

### Cost Breakdown (1M predictions/day)

| Component | Single Model | Ensemble (5 models) | Savings/Cost |
|-----------|-------------|---------------------|--------------|
| **Compute** | $100/day | $300/day | +$200/day |
| **Latency (p95)** | 20ms | 50ms | +30ms |
| **Accuracy** | 85% | 91% | +6% |
| **False positives** | 15,000/day | 9,000/day | -6,000/day |

**Cost per false positive:** $10 (fraud loss, support tickets, etc.)

**ROI Calculation:**
- Additional compute cost: +$200/day
- Reduced false positives: 6,000 × $10 = $60,000/day saved
- **Net benefit: $59,800/day = $21.8M/year**

### Optimization Strategies

1. **Model pruning:** Remove redundant models
   - From 10 models → 5 models
   - Accuracy drop: <1%
   - Cost reduction: 50%

2. **Dynamic selection:** Query only needed models
   - Average models per prediction: 3 instead of 5
   - Cost reduction: 40%

3. **Knowledge distillation:** Distill ensemble into single model
   - Single model retains 95% of ensemble accuracy
   - Cost reduction: 80%
   - Latency reduction: 75%

4. **Caching:** Cache predictions for repeated queries
   - Cache hit rate: 30%
   - Cost reduction: 30%

## Key Takeaways

✅ **Ensembles improve accuracy by 5-15%** over single best model

✅ **Diversity is more important than individual model quality**

✅ **Backtracking explores model combinations** to find optimal subset

✅ **Dynamic selection reduces latency** while maintaining accuracy

✅ **Stacking (meta-learning) outperforms** simple averaging

✅ **Parallel inference is critical** for managing latency

✅ **Fallback handling ensures robustness** against individual model failures

✅ **Knowledge distillation captures** ensemble knowledge in single model

✅ **Real-time monitoring enables** adaptive ensemble strategies

✅ **Same backtracking pattern** as Generate Parentheses—explore combinations with constraints

### Connection to Thematic Link: Backtracking and Combination Strategies

All three topics share the same core pattern:

**DSA (Generate Parentheses):**
- Backtrack to explore all valid string combinations
- Prune invalid paths (close > open)
- Result: all valid parentheses strings

**ML System Design (Model Ensembling):**
- Backtrack to explore model combinations
- Prune combinations violating constraints (latency, diversity)
- Result: optimal ensemble configuration

**Speech Tech (Multi-model Speech Ensemble):**
- Backtrack to explore speech model combinations
- Prune based on accuracy/latency trade-offs
- Result: optimal multi-model speech system

The **universal pattern**: Generate combinations, validate constraints, prune invalid branches, select optimal solution.

---

**Originally published at:** [arunbaby.com/ml-system-design/0014-model-ensembling](https://www.arunbaby.com/ml-system-design/0014-model-ensembling/)

*If you found this helpful, consider sharing it with others who might benefit.*

