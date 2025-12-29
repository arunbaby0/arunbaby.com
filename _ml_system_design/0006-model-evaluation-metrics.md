---
title: "Model Evaluation Metrics"
day: 6
related_dsa_day: 6
related_speech_day: 6
related_agents_day: 6
collection: ml_system_design
categories:
 - ml-system-design
tags:
 - metrics
 - evaluation
 - model-performance
subdomain: Model Evaluation
tech_stack: [Python, Scikit-learn, TensorFlow, MLflow]
scale: "All models"
companies: [Google, Meta, Netflix, Uber, Airbnb]
---

**How to measure if your ML model is actually good, choosing the right metrics is as important as building the model itself.**

## Introduction

**Model evaluation metrics** are quantitative measures of model performance. Choosing the wrong metric can lead to models that optimize for the wrong objective.

**Why metrics matter:**
- **Define success:** What does "good" mean for your model?
- **Compare models:** Which of 10 models should you deploy?
- **Monitor production:** Detect when model degrades
- **Align with business:** ML metrics must connect to business KPIs

**What you'll learn:**
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Regression metrics (MSE, MAE, R²)
- Ranking metrics (NDCG, MAP, MRR)
- Choosing the right metric for your problem
- Production monitoring strategies

---

## Classification Metrics

### Binary Classification

**Confusion Matrix:** Foundation of all classification metrics.

``
 Predicted
 Pos Neg
Actual Pos TP FN
 Neg FP TN

TP: True Positive - Correctly predicted positive
TN: True Negative - Correctly predicted negative
FP: False Positive - Incorrectly predicted positive (Type I error)
FN: False Negative - Incorrectly predicted negative (Type II error)
``

#### Accuracy

``
Accuracy = (TP + TN) / (TP + TN + FP + FN)
``

**When to use:** Balanced datasets 
**When NOT to use:** Imbalanced datasets

**Example:**
``python
from sklearn.metrics import accuracy_score

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 0, 1]

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2%}") # 75.00%
``

**Accuracy Paradox:**
``python
# Dataset: 95% negative, 5% positive (highly imbalanced)
# Model always predicts negative → 95% accurate!
# But useless for detecting positive class
``

#### Precision

``
Precision = TP / (TP + FP)
``

**Interpretation:** Of all positive predictions, how many were actually positive?

**When to use:** Cost of false positives is high 
**Example:** Email spam detection (don't mark legitimate emails as spam)

#### Recall (Sensitivity, True Positive Rate)

``
Recall = TP / (TP + FN)
``

**Interpretation:** Of all actual positives, how many did we detect?

**When to use:** Cost of false negatives is high 
**Example:** Cancer detection (don't miss actual cases)

#### F1 Score

``
F1 = 2 * (Precision * Recall) / (Precision + Recall)
``

**Interpretation:** Harmonic mean of precision and recall

**When to use:** Need balance between precision and recall

**Implementation:**
``python
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]

# Compute metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n{cm}")
``

#### ROC Curve & AUC

**ROC (Receiver Operating Characteristic):** Plot of True Positive Rate vs False Positive Rate at different thresholds.

``python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# Predicted probabilities
y_true = [0, 0, 1, 1, 0, 1, 0, 1]
y_scores = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.3, 0.7]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Compute AUC
auc = roc_auc_score(y_true, y_scores)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print(f"AUC: {auc:.3f}")
``

**AUC Interpretation:**
- 1.0: Perfect classifier
- 0.5: Random classifier
- < 0.5: Worse than random (inverted predictions)

**When to use AUC:** When you want threshold-independent performance measure

#### Precision-Recall Curve

Better than ROC for imbalanced datasets.

``python
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Average precision
avg_precision = average_precision_score(y_true, y_scores)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
``

---

### Multi-Class Classification

**Macro vs Micro Averaging:**

``python
from sklearn.metrics import classification_report

y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 1, 2, 0, 2, 2]

# Classification report
report = classification_report(y_true, y_pred, target_names=['Class A', 'Class B', 'Class C'])
print(report)
``

**Macro Average:** Average of per-class metrics (treats all classes equally) 
**Micro Average:** Aggregate TP, FP, FN across all classes (favors frequent classes) 
**Weighted Average:** Weighted by class frequency

**When to use which:**
- **Macro:** All classes equally important
- **Micro:** Overall performance across all predictions
- **Weighted:** Account for class imbalance

---

## Regression Metrics

### Mean Squared Error (MSE)

``
MSE = (1/n) * Σ(y_true - y_pred)²
``

**Properties:**
- Penalizes large errors heavily (squared term)
- Always non-negative
- Same units as y²

``python
from sklearn.metrics import mean_squared_error
import numpy as np

y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")
``

### Root Mean Squared Error (RMSE)

``
RMSE = √MSE
``

**Properties:**
- Same units as y (interpretable)
- Sensitive to outliers

``python
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")
``

### Mean Absolute Error (MAE)

``
MAE = (1/n) * Σ|y_true - y_pred|
``

**Properties:**
- Linear penalty (all errors weighted equally)
- More robust to outliers than MSE
- Same units as y

``python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")
``

**MSE vs MAE:**
- Use **MSE** when large errors are especially bad
- Use **MAE** when all errors have equal weight

### R² Score (Coefficient of Determination)

``
R² = 1 - (SS_res / SS_tot)

where:
 SS_res = Σ(y_true - y_pred)² (residual sum of squares)
 SS_tot = Σ(y_true - y_mean)² (total sum of squares)
``

**Interpretation:**
- 1.0: Perfect predictions
- 0.0: Model performs as well as predicting mean
- < 0.0: Model worse than predicting mean

``python
from sklearn.metrics import r2_score
import numpy as np

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")
``

### Mean Absolute Percentage Error (MAPE)

``
MAPE = (100/n) * Σ|((y_true - y_pred) / y_true)|
``

**When to use:** When relative error matters more than absolute error

**Caveat:** Undefined when y_true = 0

``python
def mean_absolute_percentage_error(y_true, y_pred):
 """
 MAPE implementation
 
 Warning: Undefined when y_true contains zeros
 """
 y_true, y_pred = np.array(y_true), np.array(y_pred)
 
 # Avoid division by zero
 non_zero_mask = y_true != 0
 
 if not np.any(non_zero_mask):
 return np.inf
 
 return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

y_true = [100, 200, 150, 300]
y_pred = [110, 190, 160, 280]

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.2f}%")
``

---

## Ranking Metrics

For recommendation systems, search engines, etc.

### Normalized Discounted Cumulative Gain (NDCG)

Measures quality of ranking where position matters.

``python
from sklearn.metrics import ndcg_score

# Relevance scores for each item (higher = more relevant)
# Order matters: first item is ranked first, etc.
y_true = [[3, 2, 3, 0, 1, 2]] # True relevance
y_pred = [[2.8, 1.9, 2.5, 0.1, 1.2, 1.8]] # Predicted scores

# NDCG@k for different k values
for k in [3, 5, None]: # None means all items
 ndcg = ndcg_score(y_true, y_pred, k=k)
 label = f"NDCG@{k if k else 'all'}"
 print(f"{label}: {ndcg:.4f}")
``

**Interpretation:**
- 1.0: Perfect ranking
- 0.0: Worst possible ranking

**When to use:** Position-aware ranking (search, recommendations)

### Mean Average Precision (MAP)

``python
def average_precision(y_true, y_scores):
 """
 Compute Average Precision
 
 Args:
 y_true: Binary relevance (1 = relevant, 0 = not relevant)
 y_scores: Predicted scores
 
 Returns:
 Average precision
 """
 # Sort by scores (descending)
 sorted_indices = np.argsort(y_scores)[::-1]
 y_true_sorted = np.array(y_true)[sorted_indices]
 
 # Compute precision at each relevant item
 precisions = []
 num_relevant = 0
 
 for i, is_relevant in enumerate(y_true_sorted, 1):
 if is_relevant:
 num_relevant += 1
 precision_at_i = num_relevant / i
 precisions.append(precision_at_i)
 
 if not precisions:
 return 0.0
 
 return np.mean(precisions)

# Example
y_true = [1, 0, 1, 0, 1, 0]
y_scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

ap = average_precision(y_true, y_scores)
print(f"Average Precision: {ap:.4f}")
``

### Mean Reciprocal Rank (MRR)

Measures where the first relevant item appears.

``
MRR = (1/|Q|) * Σ(1 / rank_i)

where rank_i is the rank of first relevant item for query i
``

``python
def mean_reciprocal_rank(y_true_queries, y_pred_queries):
 """
 Compute MRR across multiple queries
 
 Args:
 y_true_queries: List of relevance lists (one per query)
 y_pred_queries: List of score lists (one per query)
 
 Returns:
 MRR score
 """
 reciprocal_ranks = []
 
 for y_true, y_scores in zip(y_true_queries, y_pred_queries):
 # Sort by scores
 sorted_indices = np.argsort(y_scores)[::-1]
 y_true_sorted = np.array(y_true)[sorted_indices]
 
 # Find first relevant item
 for rank, is_relevant in enumerate(y_true_sorted, 1):
 if is_relevant:
 reciprocal_ranks.append(1.0 / rank)
 break
 else:
 # No relevant item found
 reciprocal_ranks.append(0.0)
 
 return np.mean(reciprocal_ranks)

# Example: 3 queries
y_true_queries = [
 [0, 1, 0, 1], # Query 1: first relevant at position 2
 [1, 0, 0, 0], # Query 2: first relevant at position 1
 [0, 0, 1, 0], # Query 3: first relevant at position 3
]

y_pred_queries = [
 [0.2, 0.8, 0.3, 0.9],
 [0.9, 0.1, 0.2, 0.3],
 [0.1, 0.2, 0.9, 0.3],
]

mrr = mean_reciprocal_rank(y_true_queries, y_pred_queries)
print(f"MRR: {mrr:.4f}")
``

---

## Choosing the Right Metric

### Decision Framework

``python
class MetricSelector:
 """
 Help choose appropriate metric based on problem characteristics
 """
 
 def recommend_metric(
 self,
 task_type: str,
 class_balance: str = 'balanced',
 business_priority: str = None
 ) -> list[str]:
 """
 Recommend metrics based on problem characteristics
 
 Args:
 task_type: 'binary_classification', 'multiclass', 'regression', 'ranking'
 class_balance: 'balanced', 'imbalanced'
 business_priority: 'precision', 'recall', 'both', None
 
 Returns:
 List of recommended metrics
 """
 recommendations = []
 
 if task_type == 'binary_classification':
 if class_balance == 'balanced':
 recommendations.append('Accuracy')
 recommendations.append('ROC-AUC')
 else:
 recommendations.append('Precision-Recall AUC')
 recommendations.append('F1 Score')
 
 if business_priority == 'precision':
 recommendations.append('Precision (optimize threshold)')
 elif business_priority == 'recall':
 recommendations.append('Recall (optimize threshold)')
 elif business_priority == 'both':
 recommendations.append('F1 Score')
 
 elif task_type == 'multiclass':
 recommendations.append('Macro F1 (if classes equally important)')
 recommendations.append('Weighted F1 (if accounting for imbalance)')
 recommendations.append('Confusion Matrix (for detailed analysis)')
 
 elif task_type == 'regression':
 recommendations.append('RMSE (if penalizing large errors)')
 recommendations.append('MAE (if robust to outliers)')
 recommendations.append('R² (for explained variance)')
 
 elif task_type == 'ranking':
 recommendations.append('NDCG (for position-aware ranking)')
 recommendations.append('MAP (for information retrieval)')
 recommendations.append('MRR (for first relevant item)')
 
 return recommendations

# Usage
selector = MetricSelector()

# Example 1: Fraud detection (imbalanced, recall critical)
metrics = selector.recommend_metric(
 task_type='binary_classification',
 class_balance='imbalanced',
 business_priority='recall'
)
print("Fraud detection metrics:", metrics)

# Example 2: Search ranking
metrics = selector.recommend_metric(
 task_type='ranking'
)
print("Search ranking metrics:", metrics)
``

---

## Production Monitoring

### Metric Tracking System

``python
import time
from collections import deque
from typing import Dict, List

class MetricTracker:
 """
 Track metrics over time in production
 
 Use case: Monitor model performance degradation
 """
 
 def __init__(self, window_size=1000):
 self.window_size = window_size
 
 # Sliding windows for predictions and actuals
 self.predictions = deque(maxlen=window_size)
 self.actuals = deque(maxlen=window_size)
 self.timestamps = deque(maxlen=window_size)
 
 # Historical metrics
 self.metric_history = {
 'accuracy': [],
 'precision': [],
 'recall': [],
 'f1': [],
 'timestamp': []
 }
 
 def log_prediction(self, y_true, y_pred):
 """
 Log a prediction and its actual outcome
 """
 self.predictions.append(y_pred)
 self.actuals.append(y_true)
 self.timestamps.append(time.time())
 
 def compute_current_metrics(self) -> Dict:
 """
 Compute metrics over current window
 """
 if len(self.predictions) < 10:
 return {}
 
 from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
 
 try:
 metrics = {
 'accuracy': accuracy_score(self.actuals, self.predictions),
 'precision': precision_score(self.actuals, self.predictions, zero_division=0),
 'recall': recall_score(self.actuals, self.predictions, zero_division=0),
 'f1': f1_score(self.actuals, self.predictions, zero_division=0),
 'sample_count': len(self.predictions)
 }
 
 # Save to history
 for metric_name, value in metrics.items():
 if metric_name != 'sample_count':
 self.metric_history[metric_name].append(value)
 
 self.metric_history['timestamp'].append(time.time())
 
 return metrics
 
 except Exception as e:
 print(f"Error computing metrics: {e}")
 return {}
 
 def detect_degradation(self, baseline_metric: str = 'f1', threshold: float = 0.05) -> bool:
 """
 Detect if model performance has degraded
 
 Args:
 baseline_metric: Metric to monitor
 threshold: Alert if metric drops by this much from baseline
 
 Returns:
 True if degradation detected
 """
 history = self.metric_history.get(baseline_metric, [])
 
 if len(history) < 10:
 return False
 
 # Compare recent average to baseline (first 10% of history)
 baseline_size = max(10, len(history) // 10)
 baseline_avg = np.mean(history[:baseline_size])
 recent_avg = np.mean(history[-baseline_size:])
 
 degradation = baseline_avg - recent_avg
 
 return degradation > threshold

# Usage
tracker = MetricTracker(window_size=1000)

# Simulate predictions over time
for i in range(1500):
 # Simulate ground truth and prediction
 y_true = np.random.choice([0, 1], p=[0.7, 0.3])
 
 # Simulate model getting worse over time
 accuracy_degradation = min(0.1, i / 10000)
 if np.random.random() < (0.8 - accuracy_degradation):
 y_pred = y_true
 else:
 y_pred = 1 - y_true
 
 tracker.log_prediction(y_true, y_pred)
 
 # Compute metrics every 100 predictions
 if i % 100 == 0 and i > 0:
 metrics = tracker.compute_current_metrics()
 if metrics:
 print(f"Step {i}: F1 = {metrics['f1']:.3f}")
 
 if tracker.detect_degradation():
 print(f"⚠️ WARNING: Model degradation detected at step {i}")
``

---

## Model Calibration

**Calibration:** How well predicted probabilities match actual outcomes.

**Example of poor calibration:**
``python
# Model predicts 80% probability for 100 samples
# Only 40 of them are actually positive
# Model is overconfident! (80% predicted vs 40% actual)
``

### Calibration Plot

``python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_calibration_curve(y_true, y_prob, n_bins=10):
 """
 Plot calibration curve
 
 A well-calibrated model's curve follows the diagonal
 """
 prob_true, prob_pred = calibration_curve(
 y_true,
 y_prob,
 n_bins=n_bins,
 strategy='uniform'
 )
 
 plt.figure(figsize=(8, 6))
 plt.plot(prob_pred, prob_true, marker='o', label='Model')
 plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
 plt.xlabel('Mean Predicted Probability')
 plt.ylabel('Fraction of Positives')
 plt.title('Calibration Plot')
 plt.legend()
 plt.grid(True)
 plt.show()

# Example
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1] * 10 # 100 samples
y_prob = [0.2, 0.7, 0.8, 0.3, 0.9, 0.1, 0.6, 0.85, 0.15, 0.75] * 10

plot_calibration_curve(y_true, y_prob)
``

### Calibrating Models

Some models (e.g., SVMs, tree ensembles) output poorly calibrated probabilities.

``python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

# Train base model
base_model = RandomForestClassifier()
base_model.fit(X_train, y_train)

# Calibrate predictions
calibrated_model = CalibratedClassifierCV(
 base_model,
 method='sigmoid', # or 'isotonic'
 cv=5
)
calibrated_model.fit(X_train, y_train)

# Now probabilities are better calibrated
y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
``

**Calibration methods:**
- **Platt scaling (sigmoid):** Fits logistic regression on predictions
- **Isotonic regression:** Non-parametric, more flexible but needs more data

---

## Threshold Tuning

Classification models output probabilities. Choosing the decision threshold impacts precision/recall trade-off.

### Finding Optimal Threshold

``python
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

def find_optimal_threshold(y_true, y_prob, metric='f1'):
 """
 Find threshold that maximizes a metric
 
 Args:
 y_true: True labels
 y_prob: Predicted probabilities
 metric: 'f1', 'precision', 'recall', or custom function
 
 Returns:
 optimal_threshold, best_score
 """
 if metric == 'f1':
 # Compute F1 at different thresholds
 precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
 f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
 
 best_idx = np.argmax(f1_scores)
 return thresholds[best_idx] if best_idx < len(thresholds) else 0.5, f1_scores[best_idx]
 
 elif metric == 'precision':
 precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
 # Find threshold for minimum acceptable recall (e.g., 0.8)
 min_recall = 0.8
 valid_idx = recall >= min_recall
 if not any(valid_idx):
 return None, 0
 best_idx = np.argmax(precision[valid_idx])
 return thresholds[valid_idx][best_idx], precision[valid_idx][best_idx]
 
 elif metric == 'recall':
 precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
 # Find threshold for minimum acceptable precision (e.g., 0.9)
 min_precision = 0.9
 valid_idx = precision >= min_precision
 if not any(valid_idx):
 return None, 0
 best_idx = np.argmax(recall[valid_idx])
 return thresholds[valid_idx][best_idx], recall[valid_idx][best_idx]

# Example
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
y_prob = np.array([0.2, 0.7, 0.8, 0.3, 0.9, 0.1, 0.6, 0.85, 0.15, 0.75])

optimal_threshold, best_f1 = find_optimal_threshold(y_true, y_prob, metric='f1')
print(f"Optimal threshold: {optimal_threshold:.3f}, Best F1: {best_f1:.3f}")
``

### Threshold Selection Strategies

**1. Maximize F1 Score**
- Balanced precision and recall
- Good default choice

**2. Business-Driven**
``python
# Example: Fraud detection
# False negative (missed fraud) costs $500
# False positive (declined legit transaction) costs $10

def business_value_threshold(y_true, y_prob, fn_cost=500, fp_cost=10):
 """
 Find threshold that maximizes business value
 """
 best_threshold = 0.5
 best_value = float('-inf')
 
 for threshold in np.arange(0.1, 0.9, 0.01):
 y_pred = (y_prob >= threshold).astype(int)
 
 # Compute confusion matrix
 tn = ((y_true == 0) & (y_pred == 0)).sum()
 fp = ((y_true == 0) & (y_pred == 1)).sum()
 fn = ((y_true == 1) & (y_pred == 0)).sum()
 tp = ((y_true == 1) & (y_pred == 1)).sum()
 
 # Business value = savings from catching fraud - cost of false alarms
 value = tp * fn_cost - fp * fp_cost
 
 if value > best_value:
 best_value = value
 best_threshold = threshold
 
 return best_threshold, best_value

threshold, value = business_value_threshold(y_true, y_prob)
print(f"Best threshold: {threshold:.2f}, Business value: ${value:.2f}")
``

**3. Operating Point Selection**
``python
# Healthcare: Prioritize recall (don't miss diseases)
# Set minimum recall = 0.95, maximize precision subject to that

def threshold_for_min_recall(y_true, y_prob, min_recall=0.95):
 """Find threshold that achieves minimum recall while maximizing precision"""
 precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
 
 valid_indices = recall >= min_recall
 if not any(valid_indices):
 return None
 
 best_precision_idx = np.argmax(precision[valid_indices])
 threshold_idx = np.where(valid_indices)[0][best_precision_idx]
 
 return thresholds[threshold_idx] if threshold_idx < len(thresholds) else 0.0
``

---

## Handling Imbalanced Datasets

### Why Standard Metrics Fail

``python
# Dataset: 99% negative, 1% positive
y_true = [0] * 990 + [1] * 10
y_pred_dummy = [0] * 1000 # Always predict negative

from sklearn.metrics import accuracy_score, precision_score, recall_score

print(f"Accuracy: {accuracy_score(y_true, y_pred_dummy):.1%}") # 99%!
print(f"Precision: {precision_score(y_true, y_pred_dummy, zero_division=0):.1%}") # Undefined (0/0)
print(f"Recall: {recall_score(y_true, y_pred_dummy):.1%}") # 0%
``

**Accuracy is 99%** but model is useless!

### Better Metrics for Imbalanced Data

**1. Precision-Recall AUC**

Better than ROC-AUC for imbalanced data because it doesn't include TN (which dominates in imbalanced datasets).

``python
from sklearn.metrics import average_precision_score

ap = average_precision_score(y_true, y_scores)
print(f"Average Precision: {ap:.3f}")
``

**2. Cohen's Kappa**

Measures agreement between predicted and actual, adjusted for chance.

``python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's Kappa: {kappa:.3f}")

# Interpretation:
# < 0: No agreement
# 0-0.20: Slight
# 0.21-0.40: Fair
# 0.41-0.60: Moderate
# 0.61-0.80: Substantial
# 0.81-1.0: Almost perfect
``

**3. Matthews Correlation Coefficient (MCC)**

Takes all four confusion matrix values into account. Ranges from -1 to +1.

``python
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_true, y_pred)
print(f"MCC: {mcc:.3f}")

# Interpretation:
# +1: Perfect prediction
# 0: Random prediction
# -1: Perfect inverse prediction
``

**4. Class-Weighted Metrics**

``python
from sklearn.metrics import fbeta_score

# Emphasize recall (beta > 1) for imbalanced positive class
f2 = fbeta_score(y_true, y_pred, beta=2) # Recall weighted 2x more than precision
print(f"F2 Score: {f2:.3f}")
``

### Sampling Strategies

``python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Combine over-sampling and under-sampling
pipeline = ImbPipeline([
 ('oversample', SMOTE(sampling_strategy=0.5)), # Increase minority to 50% of majority
 ('undersample', RandomUnderSampler(sampling_strategy=1.0)) # Balance classes
])

X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
``

---

## Aligning ML Metrics with Business KPIs

### Example 1: E-commerce Recommendation System

**ML Metrics:**
- Precision@10: 0.65
- Recall@10: 0.45
- NDCG@10: 0.72

**Business KPIs:**
- Click-through rate (CTR): 3.5%
- Conversion rate: 1.2%
- Revenue per user: $45

**Alignment:**
``python
class BusinessMetricTracker:
 """
 Track both ML and business metrics
 
 Use case: Connect model performance to business impact
 """
 
 def __init__(self):
 self.ml_metrics = {}
 self.business_metrics = {}
 self.correlations = {}
 
 def log_session(
 self,
 ml_metrics: dict,
 business_metrics: dict
 ):
 """Log metrics for a user session"""
 for metric, value in ml_metrics.items():
 if metric not in self.ml_metrics:
 self.ml_metrics[metric] = []
 self.ml_metrics[metric].append(value)
 
 for metric, value in business_metrics.items():
 if metric not in self.business_metrics:
 self.business_metrics[metric] = []
 self.business_metrics[metric].append(value)
 
 def compute_correlations(self):
 """Compute correlation between ML and business metrics"""
 import numpy as np
 from scipy.stats import pearsonr
 
 for ml_metric in self.ml_metrics:
 for biz_metric in self.business_metrics:
 ml_values = np.array(self.ml_metrics[ml_metric])
 biz_values = np.array(self.business_metrics[biz_metric])
 
 if len(ml_values) == len(biz_values):
 corr, p_value = pearsonr(ml_values, biz_values)
 self.correlations[(ml_metric, biz_metric)] = {
 'correlation': corr,
 'p_value': p_value
 }
 
 return self.correlations

# Usage
tracker = BusinessMetricTracker()

# Log multiple sessions
for _ in range(100):
 tracker.log_session(
 ml_metrics={'precision': np.random.uniform(0.6, 0.7)},
 business_metrics={'ctr': np.random.uniform(0.03, 0.04)}
 )

correlations = tracker.compute_correlations()
print("ML Metric ↔ Business KPI Correlations:")
for (ml, biz), stats in correlations.items():
 print(f"{ml} ↔ {biz}: r={stats['correlation']:.3f}, p={stats['p_value']:.3f}")
``

### Example 2: Content Moderation

**ML Metrics:**
- Precision: 0.92 (92% of flagged content is actually bad)
- Recall: 0.78 (catch 78% of bad content)

**Business KPIs:**
- User reports: How many users still report bad content?
- User retention: Are false positives causing users to leave?
- Moderator workload: Hours spent reviewing flagged content

**Trade-off:**
- High recall → More bad content caught → Fewer user reports ✓
- But also → More false positives → Higher moderator workload ✗

``python
def estimate_moderator_cost(precision, recall, daily_content, hourly_rate=50):
 """
 Estimate cost of content moderation
 
 Args:
 precision: Model precision
 recall: Model recall
 daily_content: Number of content items per day
 hourly_rate: Cost per moderator hour
 
 Returns:
 Daily moderation cost
 """
 # Assume 1% of content is actually bad
 bad_content = daily_content * 0.01
 
 # Content flagged by model
 flagged = (bad_content * recall) / precision
 
 # Time to review (assume 30 seconds per item)
 review_hours = (flagged * 30) / 3600
 
 # Cost
 cost = review_hours * hourly_rate
 
 return cost, review_hours

# Compare different models
models = [
 {'name': 'Conservative', 'precision': 0.95, 'recall': 0.70},
 {'name': 'Balanced', 'precision': 0.90, 'recall': 0.80},
 {'name': 'Aggressive', 'precision': 0.85, 'recall': 0.90}
]

for model in models:
 cost, hours = estimate_moderator_cost(
 model['precision'],
 model['recall'],
 daily_content=100000
 )
 print(f"{model['name']}: ${cost:.2f}/day, {hours:.1f} hours/day")
``

---

## Common Pitfalls

### Pitfall 1: Data Leakage in Evaluation

``python
# WRONG: Fit preprocessing on entire dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Leakage! Test data info leaks into training

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# CORRECT: Fit only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit on train only
X_test_scaled = scaler.transform(X_test) # Transform test
``

### Pitfall 2: Using Wrong Metric for Problem

``python
# Wrong: Using accuracy for imbalanced fraud detection
# Fraud rate: 0.1%, model always predicts "not fraud"
# Accuracy: 99.9% ✓ (misleading!)
# Recall: 0% ✗ (useless!)

# Right: Use precision-recall, F1, or PR-AUC
``

### Pitfall 3: Ignoring Confidence Intervals

``python
# Model A: Accuracy = 85.2%
# Model B: Accuracy = 85.5%

# Is B really better? Need confidence intervals!

from scipy import stats

def accuracy_confidence_interval(y_true, y_pred, confidence=0.95):
 """Compute confidence interval for accuracy"""
 n = len(y_true)
 y_true = np.array(y_true)
 y_pred = np.array(y_pred)
 accuracy = (y_true == y_pred).sum() / n
 
 # Wilson score interval
 z = stats.norm.ppf((1 + confidence) / 2)
 denominator = 1 + z**2 / n
 center = (accuracy + z**2 / (2*n)) / denominator
 margin = z * np.sqrt(accuracy * (1 - accuracy) / n + z**2 / (4 * n**2)) / denominator
 
 return center - margin, center + margin

import numpy as np

# Example toy predictions for illustration
y_true_a = np.random.randint(0, 2, size=1000)
y_pred_a = np.random.randint(0, 2, size=1000)
y_true_b = np.random.randint(0, 2, size=1000)
y_pred_b = np.random.randint(0, 2, size=1000)

ci_a = accuracy_confidence_interval(y_true_a, y_pred_a)
acc_a = (y_true_a == y_pred_a).mean() * 100
print(f"Model A: {acc_a:.1f}% [{ci_a[0]*100:.1f}%, {ci_a[1]*100:.1f}%]")

ci_b = accuracy_confidence_interval(y_true_b, y_pred_b)
acc_b = (y_true_b == y_pred_b).mean() * 100
print(f"Model B: {acc_b:.1f}% [{ci_b[0]*100:.1f}%, {ci_b[1]*100:.1f}%]")

# If intervals overlap significantly, difference may not be meaningful
``

### Pitfall 4: Overfitting to Validation Set

``python
# WRONG: Repeatedly tuning on same validation set
for _ in range(100): # Many iterations
 model = train_model(X_train, y_train, hyperparams)
 val_score = evaluate(model, X_val, y_val)
 hyperparams = adjust_based_on_score(val_score) # Overfitting to val!

# CORRECT: Use nested cross-validation or holdout test set
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2)

# Tune on train_full (with inner CV)
best_model = grid_search_cv(X_train_full, y_train_full)

# Evaluate ONCE on test set
final_score = evaluate(best_model, X_test, y_test)
``

---

## Connection to Speech Systems

Model evaluation principles apply directly to speech/audio ML systems:

### TTS Quality Metrics

**Objective Metrics:**
- **Mel Cepstral Distortion (MCD):** Similar to MSE for regression
- **F0 RMSE:** Pitch prediction error
- **Duration Accuracy:** Similar to classification metrics for boundary detection

**Subjective Metrics:**
- **Mean Opinion Score (MOS):** Like human evaluation for content moderation
- **Must have confidence intervals:** Just like accuracy CIs above

### ASR Error Metrics

**Word Error Rate (WER):**
``
WER = (S + D + I) / N

S: Substitutions
D: Deletions
I: Insertions
N: Total words in reference
``

Similar to precision/recall trade-off:
- High substitutions → Low precision (predicting wrong words)
- High deletions → Low recall (missing words)

### Speaker Verification

Uses same binary classification metrics:
- **EER (Equal Error Rate):** Point where FPR = FNR
- **DCF (Detection Cost Function):** Business-driven threshold (like threshold tuning above)

``python
def compute_eer(y_true, y_scores):
 """
 Compute Equal Error Rate for speaker verification
 
 Similar to finding optimal threshold
 """
 from sklearn.metrics import roc_curve
 
 fpr, tpr, thresholds = roc_curve(y_true, y_scores)
 fnr = 1 - tpr
 
 # Find where FPR ≈ FNR
 eer_idx = np.argmin(np.abs(fpr - fnr))
 eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
 
 return eer, thresholds[eer_idx]

# Example: Speaker verification scores
y_true = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
y_scores = [0.9, 0.85, 0.7, 0.4, 0.3, 0.2, 0.8, 0.75, 0.5, 0.35]

eer, eer_threshold = compute_eer(y_true, y_scores)
print(f"EER: {eer:.2%} at threshold {eer_threshold:.3f}")
``

---

## Key Takeaways

✅ **No single best metric** - choice depends on problem and business context 
✅ **Accuracy misleading** for imbalanced datasets - use precision/recall/F1 
✅ **ROC-AUC** good for threshold-independent evaluation 
✅ **Precision-Recall** better than ROC for imbalanced data 
✅ **Regression metrics** - MSE for outlier sensitivity, MAE for robustness 
✅ **Ranking metrics** - NDCG for position-aware, MRR for first relevant item 
✅ **Production monitoring** - track metrics over time to detect degradation 
✅ **Align with business** - metrics must connect to business KPIs 

---

**Originally published at:** [arunbaby.com/ml-system-design/0006-model-evaluation-metrics](https://www.arunbaby.com/ml-system-design/0006-model-evaluation-metrics/)

*If you found this helpful, consider sharing it with others who might benefit.*

