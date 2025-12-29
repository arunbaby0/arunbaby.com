---
title: "Feature Engineering at Scale"
day: 7
related_dsa_day: 7
related_speech_day: 7
related_agents_day: 7
collection: ml_system_design
categories:
 - ml-system-design
tags:
 - feature-engineering
 - feature-store
 - data-processing
 - ml-pipeline
subdomain: Data & Features
tech_stack: [Python, Pandas, Spark, Feast, Tecton, Airflow]
scale: "Billions of features"
companies: [Google, Meta, Uber, Netflix, Airbnb]
---

**Feature engineering makes or breaks ML models, learn how to build scalable, production-ready feature pipelines that power real-world systems.**

## Introduction

**Feature engineering** is the process of transforming raw data into features that better represent the underlying problem to ML models.

**Why it matters:**
- **Makes models better:** Good features > complex models with bad features
- **Domain knowledge encoding:** Capture expert insights in features
- **Data quality:** Garbage in = garbage out
- **Production complexity:** 80% of ML engineering time is data/feature work

**Stat:** Andrew Ng says "Applied ML is basically feature engineering"

---

## Feature Engineering Pipeline Architecture

### High-Level Architecture

``
┌──────────────┐
│ Raw Data │ (Logs, DB, Streams)
└──────┬───────┘
 │
 ▼
┌──────────────────────────────┐
│ Feature Engineering Layer │
│ ┌─────────┐ ┌─────────┐ │
│ │Transform│ │ Compute │ │
│ │ Logic │ │ Engines │ │
│ └─────────┘ └─────────┘ │
└──────────┬───────────────────┘
 │
 ▼
┌──────────────────────────────┐
│ Feature Store │
│ ┌────────┐ ┌──────────┐ │
│ │ Online │ │ Offline │ │
│ │Features│ │ Features │ │
│ │(low ms)│ │ (batch) │ │
│ └────────┘ └──────────┘ │
└──────────┬───────────────────┘
 │
 ▼
┌──────────────────────────────┐
│ ML Models │
│ ┌─────────┐ ┌──────────┐ │
│ │Training │ │ Serving │ │
│ └─────────┘ └──────────┘ │
└──────────────────────────────┘
``

---

## Types of Features

### 1. Numerical Features

**Raw numerical values**

``python
import pandas as pd
import numpy as np

# Example dataset
df = pd.DataFrame({
 'age': [25, 30, 35, 40],
 'income': [50000, 75000, 100000, 125000],
 'num_purchases': [5, 12, 20, 15]
})

# Common transformations
df['age_squared'] = df['age'] ** 2
df['log_income'] = np.log(df['income'])
df['income_per_purchase'] = df['income'] / (df['num_purchases'] + 1) # +1 to avoid division by zero
``

### 2. Categorical Features

**Discrete values that represent categories**

#### One-Hot Encoding

``python
# Simple one-hot encoding
df_categorical = pd.DataFrame({
 'city': ['NYC', 'SF', 'LA', 'NYC', 'SF'],
 'device': ['mobile', 'desktop', 'mobile', 'tablet', 'desktop']
})

# One-hot encode
df_encoded = pd.get_dummies(df_categorical, columns=['city', 'device'])
print(df_encoded)
# city_LA city_NYC city_SF device_desktop device_mobile device_tablet
# 0 0 1 0 0 1 0
# 1 0 0 1 1 0 0
# ...
``

#### Label Encoding (for ordinal features)

``python
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({
 'size': ['small', 'medium', 'large', 'small', 'large']
})

le = LabelEncoder()
df['size_encoded'] = le.fit_transform(df['size'])
# small→0, medium→1, large→2
``

#### Target Encoding (Mean Encoding)

``python
def target_encode(df, column, target):
 """
 Replace category with mean of target variable
 
 Good for high-cardinality categoricals
 """
 means = df.groupby(column)[target].mean()
 return df[column].map(means)

# Example
df = pd.DataFrame({
 'city': ['NYC', 'SF', 'LA', 'NYC', 'SF', 'LA'],
 'conversion': [1, 0, 1, 1, 0, 0]
})

df['city_encoded'] = target_encode(df, 'city', 'conversion')
# NYC → 1.0 (2/2), SF → 0.0 (0/2), LA → 0.5 (1/2)
``

### 3. Text Features

**Transform text into numerical representations**

#### TF-IDF

``python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
 "machine learning is awesome",
 "deep learning is a subset of machine learning",
 "natural language processing is fun"
]

vectorizer = TfidfVectorizer(max_features=10)
tfidf_matrix = vectorizer.fit_transform(documents)

print(f"Shape: {tfidf_matrix.shape}")
print(f"Features: {vectorizer.get_feature_names_out()}")
``

#### Word Embeddings

``python
# Using pre-trained embeddings (e.g., Word2Vec, GloVe)
import gensim.downloader as api

# Load pre-trained model
word_vectors = api.load("glove-wiki-gigaword-100")

def text_to_embedding(text, word_vectors):
 """
 Average word vectors for text embedding
 """
 words = text.lower().split()
 vectors = [word_vectors[word] for word in words if word in word_vectors]
 
 if not vectors:
 return np.zeros(100)
 
 return np.mean(vectors, axis=0)

# Example
text = "machine learning"
embedding = text_to_embedding(text, word_vectors)
print(f"Embedding shape: {embedding.shape}") # (100,)
``

### 4. Time-Based Features

**Extract temporal patterns**

``python
import pandas as pd

df = pd.DataFrame({
 'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
})

# Extract time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_month'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_holiday'] = df['timestamp'].isin(holiday_dates).astype(int)

# Cyclical encoding (hour wraps around)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
``

**Cyclical encoding visualization:**

``
Hour encoding (linear):
0 ─ 6 ─ 12 ─ 18 ─ 24
 │
 └─> Problem: 0 and 24 are far apart numerically!

Hour encoding (cyclical):
 0/24
 │
 21──┼──3
 │ │ │
18 │ 6
 │ │ │
 15──┼──9
 12

Using sin/cos captures cyclical nature:
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
``

### 5. Aggregation Features

**Statistics over groups**

``python
# Example: user behavior features
user_sessions = pd.DataFrame({
 'user_id': [1, 1, 1, 2, 2, 3],
 'session_duration': [120, 300, 180, 450, 200, 350],
 'pages_viewed': [5, 12, 8, 20, 10, 15],
 'timestamp': pd.date_range('2024-01-01', periods=6, freq='D')
})

# Aggregate by user
user_features = user_sessions.groupby('user_id').agg({
 'session_duration': ['mean', 'std', 'min', 'max', 'sum'],
 'pages_viewed': ['mean', 'sum', 'count'],
 'timestamp': ['min', 'max'] # First/last session
}).reset_index()

# Flatten column names
user_features.columns = ['_'.join(col).strip('_') for col in user_features.columns.values]

# Time-windowed aggregations
user_sessions['date'] = user_sessions['timestamp'].dt.date

# Last 7 days features
last_7_days = user_sessions[
 user_sessions['timestamp'] >= (user_sessions['timestamp'].max() - pd.Timedelta(days=7))
]

user_features_7d = last_7_days.groupby('user_id').agg({
 'session_duration': 'mean',
 'pages_viewed': 'sum'
}).add_suffix('_7d')
``

---

## Advanced Feature Engineering Techniques

### 1. Interaction Features

**Capture relationships between features**

``python
from sklearn.preprocessing import PolynomialFeatures

# Simple example
df = pd.DataFrame({
 'feature_a': [1, 2, 3],
 'feature_b': [4, 5, 6]
})

# Polynomial features (includes interactions)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['feature_a', 'feature_b']])

# Creates: [a, b, a², ab, b²]
print(poly.get_feature_names_out())
# ['feature_a', 'feature_b', 'feature_a^2', 'feature_a feature_b', 'feature_b^2']

# Manual domain-specific interactions
df['price_per_sqft'] = df['price'] / df['sqft']
df['bedrooms_bathrooms_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)
``

### 2. Binning/Discretization

**Convert continuous to categorical**

``python
# Equal-width binning
df['age_bin'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], 
 labels=['child', 'young_adult', 'adult', 'senior'])

# Equal-frequency binning (quantiles)
df['income_quartile'] = pd.qcut(df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Custom bins based on domain knowledge
def categorize_temperature(temp):
 if temp < 32:
 return 'freezing'
 elif temp < 60:
 return 'cold'
 elif temp < 80:
 return 'mild'
 else:
 return 'hot'

df['temp_category'] = df['temperature'].apply(categorize_temperature)
``

### 3. Feature Crosses

**Combine multiple categorical features**

``python
# Simple feature cross
df['city_device'] = df['city'] + '_' + df['device']
# Creates: 'NYC_mobile', 'SF_desktop', etc.

# Multiple feature crosses
df['city_device_hour'] = df['city'] + '_' + df['device'] + '_' + df['hour_bin']

# Then one-hot encode the crosses
df_crossed = pd.get_dummies(df['city_device'], prefix='city_device')
``

### 4. Embedding Features

**Learn dense representations**

``python
import tensorflow as tf

def create_embedding_layer(vocab_size, embedding_dim):
 """
 Create embedding layer for categorical feature
 
 Useful for high-cardinality categoricals (e.g., user_id, item_id)
 """
 return tf.keras.layers.Embedding(
 input_dim=vocab_size,
 output_dim=embedding_dim,
 embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
 )

# Example: User embeddings
num_users = 10000
user_embedding_dim = 32

user_input = tf.keras.layers.Input(shape=(1,), name='user_id')
user_embedding = create_embedding_layer(num_users, user_embedding_dim)(user_input)
user_vec = tf.keras.layers.Flatten()(user_embedding)
``

---

## Feature Store Architecture

**Problem:** Features computed differently in training vs serving → prediction skew

**Solution:** Centralized feature store with unified computation

### Feature Store Components

``python
from dataclasses import dataclass
from typing import Callable, List
import numpy as np
import pandas as pd

@dataclass
class Feature:
 """Feature definition"""
 name: str
 transform_fn: Callable
 dependencies: List[str]
 batch_source: str # Where to get data for batch computation
 stream_source: str # Where to get data for real-time

class FeatureStore:
 """
 Simplified feature store
 
 Real systems: Feast, Tecton, AWS SageMaker Feature Store
 """
 
 def __init__(self):
 self.features = {}
 self.offline_store = {} # Batch features (historical)
 self.online_store = {} # Real-time features (low latency)
 
 def register_feature(self, feature: Feature):
 """Register feature definition"""
 self.features[feature.name] = feature
 
 def compute_batch_features(self, entity_ids: List[str], features: List[str]):
 """
 Compute features for training (batch)
 
 Returns: DataFrame with features
 """
 result = pd.DataFrame({'entity_id': entity_ids})
 
 for feature_name in features:
 feature = self.features[feature_name]
 
 # Load batch data
 data = self._load_batch_data(feature.batch_source, entity_id=None)
 
 # Compute feature
 result[feature_name] = feature.transform_fn(data)
 
 return result
 
 def get_online_features(self, entity_id: str, features: List[str]):
 """
 Get features for serving (real-time)
 
 Returns: Dict of feature values
 """
 result = {}
 
 for feature_name in features:
 # Check online store
 key = f"{entity_id}:{feature_name}"
 if key in self.online_store:
 result[feature_name] = self.online_store[key]
 else:
 # Compute on-the-fly (fallback)
 feature = self.features[feature_name]
 data = self._load_stream_data(feature.stream_source, entity_id)
 result[feature_name] = feature.transform_fn(data)
 
 return result
 
 def materialize_features(self, features: List[str]):
 """
 Pre-compute features and store in online store
 
 Batch job that runs periodically
 """
 for feature_name in features:
 feature = self.features[feature_name]
 
 # Compute for all entities
 all_entities = self._get_all_entities()
 
 for entity_id in all_entities:
 data = self._load_batch_data(feature.batch_source, entity_id)
 value = feature.transform_fn(data)
 
 # Store in online store
 key = f"{entity_id}:{feature_name}"
 self.online_store[key] = value
 
 def _load_batch_data(self, source, entity_id=None):
 # Load from data warehouse (e.g., BigQuery, Snowflake)
 pass
 
 def _load_stream_data(self, source, entity_id):
 # Load from stream (e.g., Kafka, Kinesis)
 pass
 
 def _get_all_entities(self):
 # Get all entity IDs
 pass

# Example usage
feature_store = FeatureStore()

# Register features
feature_store.register_feature(Feature(
 name='user_avg_purchase_amount_30d',
 transform_fn=lambda data: data['purchase_amount'].mean(),
 dependencies=['purchase_amount'],
 batch_source='dwh.purchases',
 stream_source='kafka.purchases'
))

# Training: Get batch features
training_features = feature_store.compute_batch_features(
 entity_ids=['user_1', 'user_2'],
 features=['user_avg_purchase_amount_30d']
)

# Serving: Get online features (< 10ms)
serving_features = feature_store.get_online_features(
 entity_id='user_1',
 features=['user_avg_purchase_amount_30d']
)
``

### Feature Store Benefits

**Training-Serving Consistency:**
``
Without Feature Store:
 Training: Compute features in Python/Spark
 Serving: Reimplement in Java/Go
 Result: Different implementations → prediction skew!

With Feature Store:
 Training: feature_store.get_offline_features()
 Serving: feature_store.get_online_features()
 Result: Same computation logic → consistent!
``

---

## Feature Engineering for Tree Traversal

Connecting to DSA(tree traversal):

### Hierarchical Features

``python
class CategoryTree:
 """
 Category hierarchy (like tree traversal)
 
 Example:
 Electronics
 / \
 Computers Phones
 / \ |
 Laptops Desktops Smartphones
 """
 
 def __init__(self):
 self.tree = {
 'Electronics': {
 'Computers': {
 'Laptops': {},
 'Desktops': {}
 },
 'Phones': {
 'Smartphones': {}
 }
 }
 }
 
 def get_category_path(self, category: str) -> List[str]:
 """
 Get path from root to category
 
 Uses DFS (similar to tree traversal)
 """
 def dfs(node, target, path):
 if node == target:
 return path + [node]
 
 if isinstance(node, dict):
 for child, subtree in node.items():
 result = dfs(subtree, target, path + [child])
 if result:
 return result
 
 return None
 
 for root, subtree in self.tree.items():
 path = dfs(subtree, category, [root])
 if path:
 return path
 
 return []
 
 def category_level_features(self, category: str):
 """
 Create features from category hierarchy
 
 level_1: Electronics
 level_2: Computers
 level_3: Laptops
 """
 path = self.get_category_path(category)
 
 features = {}
 for i, cat in enumerate(path):
 features[f'category_level_{i+1}'] = cat
 
 return features

# Example
cat_tree = CategoryTree()
features = cat_tree.category_level_features('Laptops')
print(features)
# {'category_level_1': 'Electronics', 
# 'category_level_2': 'Computers', 
# 'category_level_3': 'Laptops'}
``

---

## Connection to Speech Processing 

Feature engineering is critical in speech ML:

### Audio Feature Extraction Pipeline

``python
class AudioFeatureExtractor:
 """
 Extract features from audio (similar to general feature engineering)
 """
 
 def extract_spectral_features(self, audio):
 """
 Extract spectral features
 
 Similar to numerical feature engineering
 """
 import librosa
 
 # Mel-frequency cepstral coefficients
 mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
 
 # Spectral features
 spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=22050)
 spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=22050)
 
 # Aggregate over time (similar to aggregation features)
 features = {
 'mfcc_mean': np.mean(mfccs, axis=1),
 'mfcc_std': np.std(mfccs, axis=1),
 'spectral_centroid_mean': np.mean(spectral_centroid),
 'spectral_rolloff_mean': np.mean(spectral_rolloff)
 }
 
 return features
 
 def extract_prosodic_features(self, audio):
 """
 Extract prosody features (pitch, energy, duration)
 
 Domain-specific feature engineering
 """
 import librosa
 
 # Pitch (F0)
 f0, voiced_flag, voiced_probs = librosa.pyin(
 audio,
 fmin=librosa.note_to_hz('C2'),
 fmax=librosa.note_to_hz('C7')
 )
 
 # Energy
 energy = librosa.feature.rms(y=audio)
 
 # Duration features
 zero_crossings = librosa.feature.zero_crossing_rate(audio)
 
 features = {
 'pitch_mean': np.nanmean(f0),
 'pitch_std': np.nanstd(f0),
 'pitch_range': np.nanmax(f0) - np.nanmin(f0),
 'energy_mean': np.mean(energy),
 'energy_std': np.std(energy),
 'zcr_mean': np.mean(zero_crossings)
 }
 
 return features
``

---

## Production Best Practices

### 1. Feature Versioning

``python
class VersionedFeature:
 """Track feature versions"""
 
 def __init__(self, name, version, transform_fn):
 self.name = name
 self.version = version
 self.transform_fn = transform_fn
 self.created_at = datetime.now()
 
 def get_full_name(self):
 return f"{self.name}_v{self.version}"

# Example
user_age_v1 = VersionedFeature(
 name='user_age',
 version=1,
 transform_fn=lambda df: df['birth_year'].apply(lambda x: 2024 - x)
)

user_age_v2 = VersionedFeature(
 name='user_age',
 version=2,
 transform_fn=lambda df: (datetime.now().year - df['birth_year']).clip(0, 120)
)

# Models can specify feature version
model_features = {
 'user_age_v2', # Use version 2
 'income_v1'
}
``

### 2. Feature Monitoring

``python
class FeatureMonitor:
 """Monitor feature distributions"""
 
 def __init__(self):
 self.baseline_stats = {}
 
 def compute_stats(self, feature_name, values):
 """Compute feature statistics"""
 return {
 'mean': np.mean(values),
 'std': np.std(values),
 'min': np.min(values),
 'max': np.max(values),
 'nulls': np.isnan(values).sum(),
 'unique_count': len(np.unique(values))
 }
 
 def set_baseline(self, feature_name, values):
 """Set baseline statistics"""
 self.baseline_stats[feature_name] = self.compute_stats(feature_name, values)
 
 def check_drift(self, feature_name, values, threshold=0.1):
 """
 Check if feature distribution has drifted
 
 Returns: (has_drifted, drift_metrics)
 """
 if feature_name not in self.baseline_stats:
 return False, {}
 
 current_stats = self.compute_stats(feature_name, values)
 baseline_stats = self.baseline_stats[feature_name]
 
 # Check mean drift
 mean_drift = abs(current_stats['mean'] - baseline_stats['mean']) / (baseline_stats['std'] + 1e-8)
 
 # Check std drift
 std_ratio = current_stats['std'] / (baseline_stats['std'] + 1e-8)
 
 drift_metrics = {
 'mean_drift': mean_drift,
 'std_ratio': std_ratio,
 'null_rate_change': current_stats['nulls'] / len(values) - baseline_stats['nulls'] / len(values)
 }
 
 has_drifted = mean_drift > threshold or std_ratio < 0.5 or std_ratio > 2.0
 
 return has_drifted, drift_metrics

# Usage
monitor = FeatureMonitor()

# Set baseline during training
monitor.set_baseline('user_age', training_df['user_age'].values)

# Check for drift in production
has_drifted, metrics = monitor.check_drift('user_age', production_df['user_age'].values)
if has_drifted:
 print(f"⚠️ Feature drift detected: {metrics}")
``

### 3. Feature Documentation

``python
@dataclass
class FeatureDocumentation:
 """Document features for team collaboration"""
 name: str
 description: str
 owner: str
 creation_date: str
 dependencies: List[str]
 update_frequency: str # 'realtime', 'hourly', 'daily'
 sla_ms: int # SLA for feature computation
 example_values: List
 
 def to_markdown(self):
 """Generate markdown documentation"""
 return f"""
# Feature: {self.name}

**Description:** {self.description}

**Owner:** {self.owner}

**Created:** {self.creation_date}

**Update Frequency:** {self.update_frequency}

**SLA:** {self.sla_ms}ms

**Dependencies:** {', '.join(self.dependencies)}

**Example Values:** {self.example_values[:5]}
"""

# Example
feature_doc = FeatureDocumentation(
 name='user_purchase_frequency_30d',
 description='Number of purchases by user in last 30 days',
 owner='ml-team@company.com',
 creation_date='2024-01-15',
 dependencies=['purchase_events'],
 update_frequency='hourly',
 sla_ms=100,
 example_values=[0, 2, 5, 1, 3, 0, 7]
)

print(feature_doc.to_markdown())
``

---

## Feature Selection Techniques

**Problem:** Too many features can lead to:
- Overfitting
- Increased computation
- Reduced interpretability

### 1. Filter Methods

``python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.datasets import make_classification
import pandas as pd

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)

# ANOVA F-test
selector_f = SelectKBest(f_classif, k=10)
X_selected_f = selector_f.fit_transform(X, y)

# Get selected feature indices
selected_features_f = selector_f.get_support(indices=True)
print(f"Selected features (F-test): {selected_features_f}")

# Mutual Information
selector_mi = SelectKBest(mutual_info_classif, k=10)
X_selected_mi = selector_mi.fit_transform(X, y)

print(f"Original features: {X.shape[1]}")
print(f"Selected features: {X_selected_f.shape[1]}")
``

### 2. Wrapper Methods (Forward/Backward Selection)

``python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

# Forward selection
sfs = SequentialFeatureSelector(
 RandomForestClassifier(n_estimators=100),
 n_features_to_select=10,
 direction='forward',
 cv=5
)

sfs.fit(X, y)
selected_features = sfs.get_support(indices=True)
print(f"Forward selection features: {selected_features}")
``

### 3. Embedded Methods (L1 Regularization)

``python
from sklearn.linear_model import LassoCV
import numpy as np

# Lasso for feature selection
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X, y)

# Features with non-zero coefficients
importance = np.abs(lasso.coef_)
selected_features = np.where(importance > 0.01)[0]

print(f"Lasso selected {len(selected_features)} features")
print(f"Feature importance: {importance}")
``

### 4. Feature Importance from Models

``python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importance
importance = rf.feature_importances_
indices = np.argsort(importance)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importance[indices])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()

# Select top k features
k = 10
top_features = indices[:k]
print(f"Top {k} features: {top_features}")
``

---

## Automated Feature Engineering

### AutoFeat with Featuretools

``python
# Featuretools for automated feature engineering
import featuretools as ft
import pandas as pd

# Example: E-commerce transactions
customers = pd.DataFrame({
 'customer_id': [1, 2, 3],
 'age': [25, 35, 45],
 'city': ['NYC', 'SF', 'LA']
})

transactions = pd.DataFrame({
 'transaction_id': [1, 2, 3, 4, 5],
 'customer_id': [1, 1, 2, 2, 3],
 'amount': [100, 150, 200, 50, 300],
 'timestamp': pd.date_range('2024-01-01', periods=5, freq='D')
})

# Create entity set
es = ft.EntitySet(id='customer_transactions')

# Add entities
es = es.add_dataframe(
 dataframe_name='customers',
 dataframe=customers,
 index='customer_id'
)

es = es.add_dataframe(
 dataframe_name='transactions',
 dataframe=transactions,
 index='transaction_id',
 time_index='timestamp'
)

# Add relationship
es = es.add_relationship('customers', 'customer_id', 'transactions', 'customer_id')

# Generate features automatically
feature_matrix, feature_defs = ft.dfs(
 entityset=es,
 target_dataframe_name='customers',
 max_depth=2,
 verbose=True
)

print(f"Generated {len(feature_defs)} features automatically")
print(feature_matrix.head())

# Features like:
# - SUM(transactions.amount)
# - MEAN(transactions.amount)
# - COUNT(transactions)
# - MAX(transactions.timestamp)
``

### Custom Feature Generation

``python
class AutoFeatureGenerator:
 """
 Automatically generate mathematical transformations
 """
 
 def __init__(self, operations=['square', 'sqrt', 'log', 'reciprocal']):
 self.operations = operations
 
 def generate(self, df, numerical_columns):
 """
 Generate features by applying operations
 
 Args:
 df: DataFrame
 numerical_columns: Columns to transform
 
 Returns:
 DataFrame with original + generated features
 """
 result = df.copy()
 
 for col in numerical_columns:
 if 'square' in self.operations:
 result[f'{col}_squared'] = df[col] ** 2
 
 if 'sqrt' in self.operations:
 # Only for non-negative
 if (df[col] >= 0).all():
 result[f'{col}_sqrt'] = np.sqrt(df[col])
 
 if 'log' in self.operations:
 # Only for positive
 if (df[col] > 0).all():
 result[f'{col}_log'] = np.log(df[col])
 
 if 'reciprocal' in self.operations:
 # Avoid division by zero
 result[f'{col}_reciprocal'] = 1 / (df[col] + 1e-8)
 
 # Generate interactions
 for i, col1 in enumerate(numerical_columns):
 for col2 in numerical_columns[i+1:]:
 result[f'{col1}_times_{col2}'] = df[col1] * df[col2]
 result[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
 
 return result

# Usage
df = pd.DataFrame({
 'feature_a': [1, 2, 3, 4, 5],
 'feature_b': [10, 20, 30, 40, 50]
})

generator = AutoFeatureGenerator()
df_with_features = generator.generate(df, ['feature_a', 'feature_b'])

print(f"Original features: {df.shape[1]}")
print(f"After generation: {df_with_features.shape[1]}")
print(df_with_features.columns.tolist())
``

---

## Real-World Case Studies

### Case Study 1: Netflix Recommendation Features

``python
class NetflixFeatureEngine:
 """
 Feature engineering for content recommendation
 
 Based on public Netflix research papers
 """
 
 def engineer_user_features(self, user_history):
 """
 User behavioral features
 
 Args:
 user_history: DataFrame with user viewing history
 
 Returns:
 User features
 """
 features = {}
 
 # Viewing patterns
 features['total_watch_time'] = user_history['watch_duration'].sum()
 features['avg_watch_time'] = user_history['watch_duration'].mean()
 features['num_titles_watched'] = user_history['title_id'].nunique()
 
 # Time-based patterns
 user_history['hour'] = pd.to_datetime(user_history['timestamp']).dt.hour
 features['favorite_hour'] = user_history.groupby('hour').size().idxmax()
 features['weekend_ratio'] = (user_history['is_weekend'].sum() / len(user_history))
 
 # Genre preferences
 genre_counts = user_history['genre'].value_counts()
 features['favorite_genre'] = genre_counts.index[0] if len(genre_counts) > 0 else 'unknown'
 features['genre_diversity'] = user_history['genre'].nunique()
 
 # Completion rate
 features['completion_rate'] = (
 user_history['watch_duration'] / user_history['total_duration']
 ).mean()
 
 # Binge-watching behavior
 features['avg_sessions_per_day'] = user_history.groupby(
 user_history['timestamp'].dt.date
 ).size().mean()
 
 # Recency features
 last_watch = user_history['timestamp'].max()
 features['days_since_last_watch'] = (pd.Timestamp.now() - last_watch).days
 
 return features
 
 def engineer_content_features(self, content_metadata, user_interactions):
 """
 Content-based features
 
 Combine metadata + user engagement
 """
 features = {}
 
 # Popularity features
 features['view_count'] = len(user_interactions)
 features['unique_viewers'] = user_interactions['user_id'].nunique()
 features['avg_rating'] = user_interactions['rating'].mean()
 
 # Engagement features
 features['avg_completion_rate'] = (
 user_interactions['watch_duration'] / content_metadata['duration']
 ).mean()
 
 # Temporal features
 features['days_since_release'] = (
 pd.Timestamp.now() - pd.to_datetime(content_metadata['release_date'])
 ).days
 
 # Freshness score (decaying popularity)
 features['freshness_score'] = (
 features['view_count'] / (1 + np.log(1 + features['days_since_release']))
 )
 
 return features
``

### Case Study 2: Uber Demand Prediction Features

``python
class UberDemandFeatures:
 """
 Feature engineering for ride demand prediction
 
 Inspired by Uber's blog posts on ML
 """
 
 def engineer_spatial_features(self, location_data):
 """
 Spatial features for demand prediction
 """
 features = {}
 
 # Grid-based features
 features['grid_id'] = self.lat_lon_to_grid(
 location_data['lat'],
 location_data['lon']
 )
 
 # Distance to key locations
 features['dist_to_airport'] = self.haversine_distance(
 location_data['lat'], location_data['lon'],
 airport_lat, airport_lon
 )
 
 features['dist_to_downtown'] = self.haversine_distance(
 location_data['lat'], location_data['lon'],
 downtown_lat, downtown_lon
 )
 
 # Neighborhood features
 features['is_business_district'] = self.check_business_district(
 location_data['lat'], location_data['lon']
 )
 
 return features
 
 def engineer_temporal_features(self, timestamp):
 """
 Time-based features for demand
 """
 ts = pd.to_datetime(timestamp)
 
 features = {}
 
 # Basic time features
 features['hour'] = ts.hour
 features['day_of_week'] = ts.dayofweek
 features['is_weekend'] = ts.dayofweek in [5, 6]
 
 # Peak hours
 features['is_morning_rush'] = (7 <= ts.hour <= 9)
 features['is_evening_rush'] = (17 <= ts.hour <= 19)
 features['is_late_night'] = (23 <= ts.hour or ts.hour <= 5)
 
 # Special events
 features['is_holiday'] = self.check_holiday(ts)
 features['is_major_event_day'] = self.check_events(ts, location)
 
 # Weather features (if available)
 features['is_raining'] = self.get_weather(ts, 'rain')
 features['temperature'] = self.get_weather(ts, 'temp')
 
 return features
 
 def engineer_historical_features(self, location, timestamp, lookback_days=7):
 """
 Historical demand features
 """
 features = {}
 
 # Same hour, previous days
 for days_ago in [1, 7, 14]:
 past_timestamp = timestamp - pd.Timedelta(days=days_ago)
 features[f'demand_{days_ago}d_ago'] = self.get_historical_demand(
 location, past_timestamp
 )
 
 # Moving averages
 features['demand_7d_avg'] = self.get_avg_demand(
 location, timestamp, lookback_days=7
 )
 
 features['demand_7d_std'] = self.get_std_demand(
 location, timestamp, lookback_days=7
 )
 
 # Trend
 recent_demand = self.get_demand_series(location, timestamp, days=7)
 features['demand_trend'] = self.compute_trend(recent_demand)
 
 return features
 
 @staticmethod
 def haversine_distance(lat1, lon1, lat2, lon2):
 """Calculate distance between two points on Earth"""
 from math import radians, cos, sin, asin, sqrt
 
 # Convert to radians
 lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
 
 # Haversine formula
 dlat = lat2 - lat1
 dlon = lon2 - lon1
 a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
 c = 2 * asin(sqrt(a))
 
 # Radius of Earth in kilometers
 r = 6371
 
 return c * r
``

---

## Feature Engineering at Scale with Spark

``python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

class SparkFeatureEngine:
 """
 Scalable feature engineering with Apache Spark
 
 For datasets too large for pandas
 """
 
 def __init__(self):
 self.spark = SparkSession.builder \
 .appName("FeatureEngineering") \
 .getOrCreate()
 
 def aggregate_features(self, df, group_by_col, agg_col):
 """
 Compute aggregations at scale
 
 Args:
 df: Spark DataFrame
 group_by_col: Column to group by (e.g., 'user_id')
 agg_col: Column to aggregate (e.g., 'purchase_amount')
 
 Returns:
 DataFrame with aggregated features
 """
 agg_df = df.groupBy(group_by_col).agg(
 F.count(agg_col).alias(f'{agg_col}_count'),
 F.sum(agg_col).alias(f'{agg_col}_sum'),
 F.mean(agg_col).alias(f'{agg_col}_mean'),
 F.stddev(agg_col).alias(f'{agg_col}_std'),
 F.min(agg_col).alias(f'{agg_col}_min'),
 F.max(agg_col).alias(f'{agg_col}_max')
 )
 
 return agg_df
 
 def window_features(self, df, partition_col, order_col, value_col):
 """
 Compute window features (rolling aggregations)
 
 Example: 7-day rolling average
 """
 # Define window
 days_7 = 7 * 86400 # 7 days in seconds
 
 window_spec = Window \
 .partitionBy(partition_col) \
 .orderBy(F.col(order_col).cast('long')) \
 .rangeBetween(-days_7, 0)
 
 # Compute rolling features
 df_with_window = df.withColumn(
 f'{value_col}_7d_avg',
 F.avg(value_col).over(window_spec)
 ).withColumn(
 f'{value_col}_7d_sum',
 F.sum(value_col).over(window_spec)
 ).withColumn(
 f'{value_col}_7d_count',
 F.count(value_col).over(window_spec)
 )
 
 return df_with_window
 
 def lag_features(self, df, partition_col, order_col, value_col, lags=[1, 7, 30]):
 """
 Create lag features (previous values)
 """
 window_spec = Window \
 .partitionBy(partition_col) \
 .orderBy(order_col)
 
 for lag in lags:
 df = df.withColumn(
 f'{value_col}_lag_{lag}',
 F.lag(value_col, lag).over(window_spec)
 )
 
 return df

# Usage example
# spark_fe = SparkFeatureEngine()
# 
# # Load large dataset
# df = spark_fe.spark.read.parquet('s3://bucket/data/')
# 
# # Compute features at scale
# df_features = spark_fe.aggregate_features(df, 'user_id', 'purchase_amount')
# df_features = spark_fe.window_features(df, 'user_id', 'timestamp', 'purchase_amount')
``

---

## Cost Analysis & Optimization

### Feature Computation Cost

``python
class FeatureCostAnalyzer:
 """
 Analyze cost of feature computation
 
 Important for production systems
 """
 
 def __init__(self):
 self.feature_costs = {}
 
 def measure_cost(self, feature_name, compute_fn, data, iterations=100):
 """
 Measure computation cost
 
 Returns: (time_ms, memory_mb)
 """
 import time
 import tracemalloc
 
 # Measure time
 times = []
 for _ in range(iterations):
 start = time.perf_counter()
 compute_fn(data)
 end = time.perf_counter()
 times.append((end - start) * 1000) # ms
 
 avg_time = np.mean(times)
 
 # Measure memory
 tracemalloc.start()
 compute_fn(data)
 current, peak = tracemalloc.get_traced_memory()
 tracemalloc.stop()
 
 memory_mb = peak / 1024 / 1024
 
 self.feature_costs[feature_name] = {
 'time_ms': avg_time,
 'memory_mb': memory_mb,
 'cost_score': avg_time * memory_mb # Simple cost metric
 }
 
 return avg_time, memory_mb
 
 def recommend_features(self, feature_importance, cost_threshold=100):
 """
 Recommend features based on importance vs cost trade-off
 
 Args:
 feature_importance: Dict {feature_name: importance_score}
 cost_threshold: Maximum acceptable cost
 
 Returns:
 List of recommended features
 """
 recommendations = []
 
 for feature_name, importance in feature_importance.items():
 if feature_name not in self.feature_costs:
 continue
 
 cost = self.feature_costs[feature_name]['cost_score']
 
 # Value/cost ratio
 value_ratio = importance / (cost + 1e-8)
 
 if cost <= cost_threshold:
 recommendations.append({
 'feature': feature_name,
 'importance': importance,
 'cost': cost,
 'value_ratio': value_ratio
 })
 
 # Sort by value ratio
 recommendations.sort(key=lambda x: x['value_ratio'], reverse=True)
 
 return recommendations

# Example usage
analyzer = FeatureCostAnalyzer()

# Measure costs
analyzer.measure_cost('simple_sum', lambda df: df['col1'] + df['col2'], data)
analyzer.measure_cost('complex_agg', lambda df: df.groupby('id').agg({'col': ['mean', 'std', 'max']}), data)

# Get recommendations
feature_importance = {'simple_sum': 0.8, 'complex_agg': 0.3}
recommended = analyzer.recommend_features(feature_importance)
``

---

## Key Takeaways

✅ **Feature engineering is critical** - Often more impactful than model choice 
✅ **Feature stores solve consistency** - Same code for training and serving 
✅ **Domain knowledge matters** - Best features come from understanding the problem 
✅ **Monitor features in production** - Detect drift and data quality issues 
✅ **Version features** - Track changes, enable rollback 
✅ **Document everything** - Features are long-lived assets 
✅ **Like tree traversal** - Hierarchical features need DFS/BFS logic 

---

**Originally published at:** [arunbaby.com/ml-system-design/0007-feature-engineering](https://www.arunbaby.com/ml-system-design/0007-feature-engineering/)

*If you found this helpful, consider sharing it with others who might benefit.*

