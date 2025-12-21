---
title: "Data Preprocessing Pipeline Design"
day: 3
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - data-preprocessing
  - feature-engineering
  - data-quality
subdomain: Data Infrastructure
tech_stack: [Apache Spark, Apache Beam, Pandas, Dask]
scale: "1TB+ data/day"
companies: [Google, Meta, Netflix, Uber]
related_dsa_day: 3
related_speech_day: 3
related_agents_day: 3
---

**How to build production-grade pipelines that clean, transform, and validate billions of data points before training.**

## Introduction

Data preprocessing is the **most time-consuming yet critical** part of ML systems. Industry surveys show data scientists spend **60-80% of their time** on data preparation, cleaning, transforming, and validating data before training.

**Why it matters:**
- **Garbage in, garbage out:** Poor data quality → poor models
- **Scale:** Process terabytes/petabytes efficiently
- **Repeatability:** Same transformations in training & serving
- **Monitoring:** Detect data drift and quality issues

This post covers end-to-end preprocessing pipeline design at scale.

**What you'll learn:**
- Architecture for scalable preprocessing
- Data cleaning and validation strategies
- Feature engineering pipelines
- Training/serving skew prevention
- Monitoring and data quality
- Real-world examples from top companies

---

## Problem Definition

Design a scalable data preprocessing pipeline for a machine learning system.

### Functional Requirements

1. **Data Ingestion**
   - Ingest from multiple sources (databases, logs, streams)
   - Support batch and streaming data
   - Handle structured and unstructured data

2. **Data Cleaning**
   - Handle missing values
   - Remove duplicates
   - Fix inconsistencies
   - Outlier detection and handling

3. **Data Transformation**
   - Normalization/standardization
   - Encoding categorical variables
   - Feature extraction
   - Feature selection

4. **Data Validation**
   - Schema validation
   - Statistical validation
   - Anomaly detection
   - Data drift detection

5. **Feature Engineering**
   - Create derived features
   - Aggregations (time-based, user-based)
   - Interaction features
   - Embedding generation

### Non-Functional Requirements

1. **Scale**
   - Process 1TB+ data/day
   - Handle billions of records
   - Support horizontal scaling

2. **Latency**
   - Batch: Process daily data in < 6 hours
   - Streaming: < 1 second latency for real-time features

3. **Reliability**
   - 99.9% pipeline success rate
   - Automatic retries on failure
   - Data lineage tracking

4. **Consistency**
   - Same transformations in training and serving
   - Versioned transformation logic
   - Reproducible results

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                              │
├─────────────────────────────────────────────────────────────┤
│  Databases  │  Event Logs  │  File Storage  │  APIs         │
└──────┬──────┴──────┬──────┴───────┬─────────┴──────┬────────┘
       │             │              │                │
       └─────────────┼──────────────┼────────────────┘
                     ↓              ↓
              ┌─────────────────────────────┐
              │   Data Ingestion Layer      │
              │  (Kafka, Pub/Sub, Kinesis)  │
              └──────────────┬──────────────┘
                             ↓
              ┌─────────────────────────────┐
              │   Raw Data Storage          │
              │   (Data Lake: S3/GCS)       │
              └──────────────┬──────────────┘
                             ↓
              ┌─────────────────────────────┐
              │  Preprocessing Pipeline     │
              │                             │
              │  ┌──────────────────────┐   │
              │  │ 1. Data Validation   │   │
              │  └──────────────────────┘   │
              │  ┌──────────────────────┐   │
              │  │ 2. Data Cleaning     │   │
              │  └──────────────────────┘   │
              │  ┌──────────────────────┐   │
              │  │ 3. Feature Extraction│   │
              │  └──────────────────────┘   │
              │  ┌──────────────────────┐   │
              │  │ 4. Transformation    │   │
              │  └──────────────────────┘   │
              │  ┌──────────────────────┐   │
              │  │ 5. Quality Checks    │   │
              │  └──────────────────────┘   │
              │                             │
              │  (Spark/Beam/Airflow)       │
              └──────────────┬──────────────┘
                             ↓
              ┌─────────────────────────────┐
              │   Processed Data Storage    │
              │   (Feature Store/DW)        │
              └──────────────┬──────────────┘
                             ↓
              ┌─────────────────────────────┐
              │   Model Training            │
              │   & Serving                 │
              └─────────────────────────────┘
```

---

## Component 1: Data Validation

Validate data quality and schema before processing.

### Schema Validation

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import pandas as pd

class DataType(Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    TIMESTAMP = "timestamp"
    BOOLEAN = "boolean"

@dataclass
class FieldSchema:
    name: str
    dtype: DataType
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None

class SchemaValidator:
    """
    Validate data against expected schema
    
    Use case: Ensure incoming data matches expectations
    """
    
    def __init__(self, schema: List[FieldSchema]):
        self.schema = {field.name: field for field in schema}
    
    def validate(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate DataFrame against schema
        
        Returns:
            Dict of field_name → list of errors
        """
        errors = {}
        
        # Check for missing columns
        expected_cols = set(self.schema.keys())
        actual_cols = set(df.columns)
        missing = expected_cols - actual_cols
        if missing:
            errors['_schema'] = [f"Missing columns: {missing}"]
        
        # Validate each field
        for field_name, field_schema in self.schema.items():
            if field_name not in df.columns:
                continue
            
            field_errors = self._validate_field(df[field_name], field_schema)
            if field_errors:
                errors[field_name] = field_errors
        
        return errors
    
    def validate_record(self, record: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate a single record (dict) against schema
        """
        df = pd.DataFrame([record])
        return self.validate(df)
    
    def _validate_field(self, series: pd.Series, schema: FieldSchema) -> List[str]:
        """Validate a single field"""
        errors = []
        
        # Check nulls
        if not schema.nullable and series.isnull().any():
            null_count = series.isnull().sum()
            errors.append(f"Found {null_count} null values (not allowed)")
        
        # Check data type
        if schema.dtype == DataType.INT:
            if not pd.api.types.is_integer_dtype(series.dropna()):
                errors.append("Expected integer type")
        elif schema.dtype == DataType.FLOAT:
            if not pd.api.types.is_numeric_dtype(series.dropna()):
                errors.append("Expected numeric type")
        elif schema.dtype == DataType.STRING:
            if not pd.api.types.is_string_dtype(series.dropna()):
                errors.append("Expected string type")
        elif schema.dtype == DataType.BOOLEAN:
            if not pd.api.types.is_bool_dtype(series.dropna()):
                errors.append("Expected boolean type")
        elif schema.dtype == DataType.TIMESTAMP:
            if not pd.api.types.is_datetime64_any_dtype(series.dropna()):
                try:
                    pd.to_datetime(series.dropna())
                except Exception:
                    errors.append("Expected timestamp/datetime type")
        
        # Check value ranges
        if schema.min_value is not None:
            below_min = (series < schema.min_value).sum()
            if below_min > 0:
                errors.append(f"{below_min} values below minimum {schema.min_value}")
        
        if schema.max_value is not None:
            above_max = (series > schema.max_value).sum()
            if above_max > 0:
                errors.append(f"{above_max} values above maximum {schema.max_value}")
        
        # Check allowed values
        if schema.allowed_values is not None:
            invalid = ~series.isin(schema.allowed_values)
            invalid_count = invalid.sum()
            if invalid_count > 0:
                invalid_vals = series[invalid].unique()[:5]
                errors.append(
                    f"{invalid_count} values not in allowed set. "
                    f"Examples: {invalid_vals}"
                )
        
        return errors

# Usage
user_schema = [
    FieldSchema("user_id", DataType.INT, nullable=False, min_value=0),
    FieldSchema("age", DataType.INT, nullable=True, min_value=0, max_value=120),
    FieldSchema("country", DataType.STRING, nullable=False, 
                allowed_values=["US", "UK", "CA", "AU"]),
    FieldSchema("signup_date", DataType.TIMESTAMP, nullable=False)
]

validator = SchemaValidator(user_schema)
errors = validator.validate(user_df)

if errors:
    print("Validation errors found:")
    for field, field_errors in errors.items():
        print(f"  {field}: {field_errors}")
```

### Statistical Validation

```python
import numpy as np
from scipy import stats

class StatisticalValidator:
    """
    Detect statistical anomalies in data
    
    Compare current batch against historical baseline
    """
    
    def __init__(self, baseline_stats: Dict[str, Dict]):
        """
        Args:
            baseline_stats: Historical statistics per field
                {
                    'age': {'mean': 35.2, 'std': 12.5, 'median': 33},
                    'price': {'mean': 99.5, 'std': 25.0, 'median': 95}
                }
        """
        self.baseline = baseline_stats
    
    def validate(self, df: pd.DataFrame, threshold_sigma=3) -> List[str]:
        """
        Detect fields with distributions far from baseline
        
        Returns:
            List of warnings
        """
        warnings = []
        
        for field, baseline in self.baseline.items():
            if field not in df.columns:
                continue
            
            current = df[field].dropna()
            
            # Check mean shift
            current_mean = current.mean()
            expected_mean = baseline['mean']
            expected_std = baseline['std']
            
            denom = expected_std if expected_std > 1e-9 else 1e-9
            z_score = abs(current_mean - expected_mean) / denom
            
            if z_score > threshold_sigma:
                warnings.append(
                    f"{field}: Mean shifted significantly "
                    f"(current={current_mean:.2f}, "
                    f"baseline={expected_mean:.2f}, "
                    f"z-score={z_score:.2f})"
                )
            
            # Check distribution shift (KS test)
            baseline_samples = np.random.normal(
                baseline['mean'], 
                baseline['std'], 
                size=len(current)
            )
            
            ks_stat, p_value = stats.ks_2samp(current, baseline_samples)
            
            if p_value < 0.01:  # Significant difference
                warnings.append(
                    f"{field}: Distribution changed "
                    f"(KS statistic={ks_stat:.3f}, p={p_value:.3f})"
                )
        
        return warnings
```

---

## Component 2: Data Cleaning

Handle missing values, duplicates, and inconsistencies.

### Missing Value Handling

```python
class MissingValueHandler:
    """
    Handle missing values with different strategies
    """
    
    def __init__(self):
        self.imputers = {}
    
    def fit(self, df: pd.DataFrame, strategies: Dict[str, str]):
        """
        Fit imputation strategies
        
        Args:
            strategies: {column: strategy}
                strategy options: 'mean', 'median', 'mode', 'forward_fill', 'drop'
        """
        for col, strategy in strategies.items():
            if col not in df.columns:
                continue
            
            if strategy == 'mean':
                self.imputers[col] = df[col].mean()
            elif strategy == 'median':
                self.imputers[col] = df[col].median()
            elif strategy == 'mode':
                self.imputers[col] = df[col].mode()[0]
            # forward_fill and drop don't need fitting
    
    def transform(self, df: pd.DataFrame, strategies: Dict[str, str]) -> pd.DataFrame:
        """Apply imputation"""
        df = df.copy()
        
        for col, strategy in strategies.items():
            if col not in df.columns:
                continue
            
            if strategy in ['mean', 'median', 'mode']:
                df[col].fillna(self.imputers[col], inplace=True)
            
            elif strategy == 'forward_fill':
                df[col].fillna(method='ffill', inplace=True)
            
            elif strategy == 'backward_fill':
                df[col].fillna(method='bfill', inplace=True)
            
            elif strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
            
            elif strategy == 'constant':
                # Fill with a constant (e.g., 0, 'Unknown')
                fill_value = 0 if pd.api.types.is_numeric_dtype(df[col]) else 'Unknown'
                df[col].fillna(fill_value, inplace=True)
        
        return df
```

### Outlier Detection & Handling

```python
class OutlierHandler:
    """
    Detect and handle outliers
    """
    
    def detect_outliers_iqr(self, series: pd.Series, multiplier=1.5):
        """
        IQR method: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (series < lower_bound) | (series > upper_bound)
        
        return outliers
    
    def detect_outliers_zscore(self, series: pd.Series, threshold=3):
        """
        Z-score method: |z| > threshold
        """
        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = z_scores > threshold
        
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str], method='clip'):
        """
        Handle outliers
        
        Args:
            method: 'clip', 'remove', 'cap', 'transform'
        """
        df = df.copy()
        
        for col in columns:
            outliers = self.detect_outliers_iqr(df[col])
            
            if method == 'clip':
                # Clip to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower, upper)
            
            elif method == 'remove':
                # Remove outlier rows
                df = df[~outliers]
            
            elif method == 'cap':
                # Cap at 99th percentile
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(upper=upper)
            
            elif method == 'transform':
                # Log transform to reduce skew
                df[col] = np.log1p(df[col])
        
        return df
```

### Deduplication

```python
class Deduplicator:
    """
    Remove duplicate records
    """
    
    def deduplicate(
        self, 
        df: pd.DataFrame, 
        key_columns: List[str],
        keep='last',
        timestamp_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Remove duplicates
        
        Args:
            key_columns: Columns that define uniqueness
            keep: 'first', 'last', or False (remove all duplicates)
            timestamp_col: If provided, keep most recent
        """
        if timestamp_col:
            # Sort by timestamp descending, then drop duplicates keeping first
            df = df.sort_values(timestamp_col, ascending=False)
            df = df.drop_duplicates(subset=key_columns, keep='first')
        else:
            df = df.drop_duplicates(subset=key_columns, keep=keep)
        
        return df
```

---

## Component 3: Feature Engineering

Transform raw data into ML-ready features.

### Numerical Transformations

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class NumericalTransformer:
    """
    Apply numerical transformations
    """
    
    def __init__(self):
        self.scalers = {}
    
    def fit_transform(self, df: pd.DataFrame, transformations: Dict[str, str]):
        """
        Apply transformations
        
        transformations: {column: transformation_type}
            'standard': StandardScaler (mean=0, std=1)
            'minmax': MinMaxScaler (range [0, 1])
            'robust': RobustScaler (use median, IQR - robust to outliers)
            'log': Log transform
            'sqrt': Square root transform
        """
        df = df.copy()
        
        for col, transform_type in transformations.items():
            if col not in df.columns:
                continue
            
            if transform_type == 'standard':
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            
            elif transform_type == 'minmax':
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            
            elif transform_type == 'robust':
                scaler = RobustScaler()
                df[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            
            elif transform_type == 'log':
                df[col] = np.log1p(df[col])  # log(1 + x) to handle 0
            
            elif transform_type == 'sqrt':
                df[col] = np.sqrt(df[col])
            
            elif transform_type == 'boxcox':
                # Box-Cox transform (requires positive values)
                df[col], _ = stats.boxcox(df[col] + 1)  # +1 to handle 0
        
        return df
```

### Categorical Encoding

```python
class CategoricalEncoder:
    """
    Encode categorical variables
    """
    
    def __init__(self):
        self.encoders = {}
    
    def fit_transform(self, df: pd.DataFrame, encodings: Dict[str, str]):
        """
        Apply encodings
        
        encodings: {column: encoding_type}
            'onehot': One-hot encoding
            'label': Label encoding (0, 1, 2, ...)
            'target': Target encoding (mean of target per category)
            'frequency': Frequency encoding
            'ordinal': Ordinal encoding with custom order
        """
        df = df.copy()
        
        for col, encoding_type in encodings.items():
            if col not in df.columns:
                continue
            
            if encoding_type == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
                self.encoders[col] = list(dummies.columns)
            
            elif encoding_type == 'label':
                # Label encoding
                categories = df[col].unique()
                mapping = {cat: idx for idx, cat in enumerate(categories)}
                df[col] = df[col].map(mapping)
                self.encoders[col] = mapping
            
            elif encoding_type == 'frequency':
                # Frequency encoding
                freq = df[col].value_counts(normalize=True)
                df[col] = df[col].map(freq)
                self.encoders[col] = freq
        
        return df
```

### Temporal Features

```python
class TemporalFeatureExtractor:
    """
    Extract features from timestamps
    """
    
    def extract(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """
        Extract temporal features from timestamp column
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Basic temporal features
        df[f'{timestamp_col}_hour'] = df[timestamp_col].dt.hour
        df[f'{timestamp_col}_day_of_week'] = df[timestamp_col].dt.dayofweek
        df[f'{timestamp_col}_day_of_month'] = df[timestamp_col].dt.day
        df[f'{timestamp_col}_month'] = df[timestamp_col].dt.month
        df[f'{timestamp_col}_quarter'] = df[timestamp_col].dt.quarter
        df[f'{timestamp_col}_year'] = df[timestamp_col].dt.year
        
        # Derived features
        df[f'{timestamp_col}_is_weekend'] = df[f'{timestamp_col}_day_of_week'].isin([5, 6]).astype(int)
        df[f'{timestamp_col}_is_business_hours'] = df[f'{timestamp_col}_hour'].between(9, 17).astype(int)
        
        # Cyclical encoding (for periodic features like hour)
        df[f'{timestamp_col}_hour_sin'] = np.sin(2 * np.pi * df[f'{timestamp_col}_hour'] / 24)
        df[f'{timestamp_col}_hour_cos'] = np.cos(2 * np.pi * df[f'{timestamp_col}_hour'] / 24)
        
        return df
```

---

## Component 4: Pipeline Orchestration

Orchestrate the entire preprocessing workflow.

### Apache Beam Pipeline

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline using Apache Beam
    
    Handles:
    - Data validation
    - Cleaning
    - Feature engineering
    - Quality checks
    """
    
    def __init__(self, pipeline_options: PipelineOptions):
        self.options = pipeline_options
    
    def run(self, input_path: str, output_path: str):
        """
        Run preprocessing pipeline
        """
        with beam.Pipeline(options=self.options) as pipeline:
            (
                pipeline
                | 'Read Data' >> beam.io.ReadFromText(input_path)
                | 'Parse JSON' >> beam.Map(json.loads)
                | 'Validate Schema' >> beam.ParDo(ValidateSchemaFn())
                | 'Clean Data' >> beam.ParDo(CleanDataFn())
                | 'Extract Features' >> beam.ParDo(FeatureExtractionFn())
                | 'Quality Check' >> beam.ParDo(QualityCheckFn())
                | 'Write Output' >> beam.io.WriteToText(output_path)
            )

class ValidateSchemaFn(beam.DoFn):
    """Beam DoFn for schema validation"""
    
    def process(self, element):
        # Lazily initialize schema validator (avoid re-creating per element)
        if not hasattr(self, 'validator'):
            self.validator = SchemaValidator(get_schema())
        errors = self.validator.validate_record(element)
        
        if errors:
            # Log to dead letter queue
            yield beam.pvalue.TaggedOutput('invalid', (element, errors))
        else:
            yield element

class CleanDataFn(beam.DoFn):
    """Beam DoFn for data cleaning"""
    
    def process(self, element):
        # Handle missing values
        element = handle_missing(element)
        
        # Handle outliers
        element = handle_outliers(element)
        
        # Remove duplicates (stateful processing)
        # ...
        
        yield element
```

---

## Preventing Training/Serving Skew

**Critical problem:** Different preprocessing in training vs serving leads to poor model performance.

### Solution 1: Unified Preprocessing Library

```python
class PreprocessorV1:
    """
    Versioned preprocessing logic
    
    Same code used in training and serving
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Dict):
        self.config = config
        self.fitted_params = {}
    
    def fit(self, df: pd.DataFrame):
        """Fit on training data"""
        # Compute statistics needed for transform
        self.fitted_params['age_mean'] = df['age'].mean()
        self.fitted_params['price_scaler'] = MinMaxScaler().fit(df[['price']])
        # ...
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same transformations"""
        df = df.copy()
        
        # Use fitted parameters
        df['age_normalized'] = (df['age'] - self.fitted_params['age_mean']) / 10
        df['price_scaled'] = self.fitted_params['price_scaler'].transform(df[['price']])
        
        return df
    
    def save(self, path: str):
        """Save fitted preprocessor"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str):
        """Load fitted preprocessor"""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

# Training
preprocessor = PreprocessorV1(config)
preprocessor.fit(training_data)
preprocessor.save('models/preprocessor_v1.pkl')
X_train = preprocessor.transform(training_data)

# Serving
preprocessor = PreprocessorV1.load('models/preprocessor_v1.pkl')
X_serve = preprocessor.transform(serving_data)
```

### Solution 2: Feature Store

Store pre-computed features, ensuring consistency.

```python
class FeatureStore:
    """
    Centralized feature storage
    
    Benefits:
    - Features computed once, used everywhere
    - Versioned features
    - Point-in-time correct joins
    """
    
    def __init__(self, backend):
        self.backend = backend
    
    def write_features(
        self, 
        entity_id: str,
        features: Dict[str, Any],
        timestamp: datetime,
        feature_set_name: str,
        version: str
    ):
        """
        Write features for an entity
        """
        key = f"{feature_set_name}:{version}:{entity_id}:{timestamp}"
        self.backend.write(key, features)
    
    def read_features(
        self,
        entity_id: str,
        feature_set_name: str,
        version: str,
        as_of_timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Read features as of a specific timestamp
        
        Point-in-time correctness: Only use features available at inference time
        """
        # Query features created before as_of_timestamp
        features = self.backend.read_point_in_time(
            entity_id,
            feature_set_name,
            version,
            as_of_timestamp
        )
        
        return features
```

---

## Monitoring & Data Quality

Track data quality metrics over time.

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DataQualityMetrics:
    """Metrics for a data batch"""
    timestamp: datetime
    total_records: int
    null_counts: Dict[str, int]
    duplicate_count: int
    schema_errors: int
    outlier_counts: Dict[str, int]
    statistical_warnings: List[str]

class DataQualityMonitor:
    """
    Monitor data quality over time
    """
    
    def __init__(self, metrics_backend):
        self.backend = metrics_backend
    
    def compute_metrics(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Compute quality metrics for a batch"""
        
        metrics = DataQualityMetrics(
            timestamp=datetime.now(),
            total_records=len(df),
            null_counts={col: df[col].isnull().sum() for col in df.columns},
            duplicate_count=df.duplicated().sum(),
            schema_errors=0,  # From validation
            outlier_counts={},
            statistical_warnings=[]
        )
        
        # Detect outliers
        outlier_handler = OutlierHandler()
        for col in df.select_dtypes(include=[np.number]).columns:
            outliers = outlier_handler.detect_outliers_iqr(df[col])
            metrics.outlier_counts[col] = outliers.sum()
        
        return metrics
    
    def log_metrics(self, metrics: DataQualityMetrics):
        """Log metrics to monitoring system"""
        self.backend.write(metrics)
    
    def alert_on_anomalies(self, metrics: DataQualityMetrics):
        """Alert if metrics deviate significantly"""
        
        # Alert if > 5% nulls in critical fields
        critical_fields = ['user_id', 'timestamp', 'label']
        for field in critical_fields:
            null_rate = metrics.null_counts.get(field, 0) / metrics.total_records
            if null_rate > 0.05:
                self.send_alert(f"High null rate in {field}: {null_rate:.2%}")
        
        # Alert if > 10% duplicates
        dup_rate = metrics.duplicate_count / metrics.total_records
        if dup_rate > 0.10:
            self.send_alert(f"High duplicate rate: {dup_rate:.2%}")
```

---

## Real-World Examples

### Netflix: Data Preprocessing for Recommendations

**Scale:** Billions of viewing events/day

**Architecture:**
```
Event Stream (Kafka)
  ↓
Flink/Spark Streaming
  ↓
Feature Engineering
  - User viewing history aggregations
  - Time-based features
  - Content embeddings
  ↓
Feature Store (Cassandra)
  ↓
Model Training & Serving
```

**Key techniques:**
- Streaming aggregations (last 7 days views, etc.)
- Incremental updates to user profiles
- Point-in-time correct features

### Uber: Preprocessing for ETAs

**Challenge:** Predict arrival times using GPS data

**Pipeline:**
1. **Map Matching:** Snap GPS points to road network
2. **Outlier Removal:** Remove impossible speeds
3. **Feature Extraction:**
   - Time of day, day of week
   - Traffic conditions
   - Historical average speed
4. **Validation:** Check for data drift

**Latency:** < 100ms for real-time predictions

### Google: Search Ranking Data Pipeline

**Scale:** Process billions of queries and web pages

**Preprocessing steps:**
1. **Query normalization:** Lowercasing, tokenization, spelling correction
2. **Feature extraction from documents:**
   - PageRank scores
   - Content embeddings (BERT)
   - Click-through rate (CTR) features
3. **User context features:**
   - Location
   - Device type
   - Search history embeddings
4. **Join multiple data sources:**
   - User profile data
   - Document metadata
   - Real-time signals (freshness)

**Key insight:** Distributed processing using MapReduce/Dataflow for petabyte-scale data.

---

## Distributed Preprocessing with Spark

When data doesn't fit on one machine, use distributed frameworks.

### Spark Preprocessing Pipeline

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, stddev, count
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

class DistributedPreprocessor:
    """
    Large-scale preprocessing using Apache Spark
    
    Use case: Process 1TB+ data across cluster
    """
    
    def __init__(self):
        self.spark = SparkSession.builder \\
            .appName("MLPreprocessing") \\
            .getOrCreate()
    
    def load_data(self, path: str, format='parquet'):
        """Load data from distributed storage"""
        return self.spark.read.format(format).load(path)
    
    def clean_data(self, df):
        """Distributed data cleaning"""
        
        # Remove nulls
        df = df.dropna(subset=['user_id', 'timestamp'])
        
        # Handle outliers (clip at 99th percentile)
        for col_name in ['price', 'quantity']:
            quantile_99 = df.approxQuantile(col_name, [0.99], 0.01)[0]
            df = df.withColumn(
                col_name,
                when(col(col_name) > quantile_99, quantile_99).otherwise(col(col_name))
            )
        
        # Remove duplicates
        df = df.dropDuplicates(['user_id', 'item_id', 'timestamp'])
        
        return df
    
    def feature_engineering(self, df):
        """Distributed feature engineering"""
        
        # Time-based features
        df = df.withColumn('hour', hour(col('timestamp')))
        df = df.withColumn('day_of_week', dayofweek(col('timestamp')))
        df = df.withColumn('is_weekend', 
                          when(col('day_of_week').isin([1, 7]), 1).otherwise(0))
        
        # Aggregation features (window functions)
        from pyspark.sql.window import Window
        
        # User's average purchase price (last 30 days)
        window_30d = Window.partitionBy('user_id') \\
                          .orderBy(col('timestamp').cast('long')) \\
                          .rangeBetween(-30*24*3600, 0)
        
        df = df.withColumn('user_avg_price_30d', 
                          avg('price').over(window_30d))
        
        return df
    
    def normalize_features(self, df, numeric_cols):
        """Normalize numeric features"""
        
        # Assemble features into vector
        assembler = VectorAssembler(
            inputCols=numeric_cols,
            outputCol='features_raw'
        )
        
        # Standard scaling
        scaler = StandardScaler(
            inputCol='features_raw',
            outputCol='features_scaled',
            withMean=True,
            withStd=True
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, scaler])
        
        # Fit and transform
        model = pipeline.fit(df)
        df = model.transform(df)
        
        return df, model
    
    def save_preprocessed(self, df, output_path, model_path):
        """Save preprocessed data and fitted model"""
        
        # Save data (partitioned for efficiency)
        df.write.mode('overwrite') \\
          .partitionBy('date') \\
          .parquet(output_path)
        
        # Save preprocessing model for serving
        # model.save(model_path)

# Usage
preprocessor = DistributedPreprocessor()
df = preprocessor.load_data('s3://bucket/raw_data/')
df = preprocessor.clean_data(df)
df = preprocessor.feature_engineering(df)
df, model = preprocessor.normalize_features(df, ['price', 'quantity'])
preprocessor.save_preprocessed(df, 's3://bucket/processed/', 's3://bucket/models/')
```

---

## Advanced Feature Engineering Patterns

### 1. Time-Series Features

```python
class TimeSeriesFeatureExtractor:
    """
    Extract features from time-series data
    
    Use case: User engagement over time, sensor readings, stock prices
    """
    
    def extract_lag_features(self, df, value_col, lag_periods=[1, 7, 30]):
        """Create lagged features"""
        for lag in lag_periods:
            df[f'{value_col}_lag_{lag}'] = df.groupby('user_id')[value_col].shift(lag)
        return df
    
    def extract_rolling_statistics(self, df, value_col, windows=[7, 30]):
        """Rolling mean, std, min, max"""
        for window in windows:
            df[f'{value_col}_rolling_mean_{window}'] = \\
                df.groupby('user_id')[value_col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            df[f'{value_col}_rolling_std_{window}'] = \\
                df.groupby('user_id')[value_col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
        return df
    
    def extract_trend_features(self, df, value_col):
        """
        Trend: difference from moving average
        """
        df['rolling_mean_7'] = df.groupby('user_id')[value_col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df[f'{value_col}_trend'] = df[value_col] - df['rolling_mean_7']
        return df
```

### 2. Interaction Features

```python
class InteractionFeatureGenerator:
    """
    Create interaction features between variables
    
    Captures relationships not visible in individual features
    """
    
    def polynomial_features(self, df, cols, degree=2):
        """
        Create polynomial features
        
        Example: x, y → x, y, x², y², xy
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[cols])
        
        feature_names = poly.get_feature_names_out(cols)
        poly_df = pd.DataFrame(poly_features, columns=feature_names)
        
        return pd.concat([df, poly_df], axis=1)
    
    def ratio_features(self, df, numerator_cols, denominator_cols):
        """
        Create ratio features
        
        Example: revenue/cost, clicks/impressions (CTR)
        """
        for num_col in numerator_cols:
            for den_col in denominator_cols:
                df[f'{num_col}_per_{den_col}'] = df[num_col] / (df[den_col] + 1e-9)
        return df
    
    def categorical_interactions(self, df, cat_cols):
        """
        Combine categorical variables
        
        Example: city='SF', category='Tech' → 'SF_Tech'
        """
        if len(cat_cols) >= 2:
            df['_'.join(cat_cols)] = df[cat_cols].astype(str).agg('_'.join, axis=1)
        return df
```

### 3. Embedding Features

```python
class EmbeddingFeatureGenerator:
    """
    Generate embedding features from high-cardinality categoricals
    
    Use case: user_id, item_id, text
    """
    
    def train_category_embeddings(self, df, category_col, embedding_dim=50):
        """
        Train embeddings for categorical variable
        
        Uses skip-gram approach: predict co-occurring categories
        """
        from gensim.models import Word2Vec
        
        # Create sequences (e.g., user's purchase history)
        sequences = df.groupby('user_id')[category_col].apply(list).tolist()
        
        # Train Word2Vec
        model = Word2Vec(
            sentences=sequences,
            vector_size=embedding_dim,
            window=5,
            min_count=1,
            workers=4
        )
        
        # Get embeddings
        embeddings = {}
        for category in df[category_col].unique():
            if category in model.wv:
                embeddings[category] = model.wv[category]
        
        return embeddings
    
    def text_to_embeddings(self, df, text_col, model='sentence-transformers'):
        """
        Convert text to dense embeddings
        
        Use pre-trained models (BERT, etc.)
        """
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df[text_col].tolist())
        
        # Add as features
        for i in range(embeddings.shape[1]):
            df[f'{text_col}_emb_{i}'] = embeddings[:, i]
        
        return df
```

---

## Handling Data Drift

Data distributions change over time - models degrade if not monitored.

### Drift Detection

```python
from scipy.stats import ks_2samp, chi2_contingency

class DataDriftDetector:
    """
    Detect when data distribution changes
    """
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Args:
            reference_data: Historical "good" data (training distribution)
        """
        self.reference = reference_data
    
    def detect_numerical_drift(self, current_data: pd.DataFrame, col: str, threshold=0.05):
        """
        Kolmogorov-Smirnov test for numerical columns
        
        Returns:
            (drifted: bool, p_value: float)
        """
        ref_values = self.reference[col].dropna()
        curr_values = current_data[col].dropna()
        
        statistic, p_value = ks_2samp(ref_values, curr_values)
        
        drifted = p_value < threshold
        
        return drifted, p_value
    
    def detect_categorical_drift(self, current_data: pd.DataFrame, col: str, threshold=0.05):
        """
        Chi-square test for categorical columns
        """
        ref_dist = self.reference[col].value_counts(normalize=True)
        curr_dist = current_data[col].value_counts(normalize=True)
        
        # Align distributions
        all_categories = set(ref_dist.index) | set(curr_dist.index)
        ref_counts = [ref_dist.get(cat, 0) * len(self.reference) for cat in all_categories]
        curr_counts = [curr_dist.get(cat, 0) * len(current_data) for cat in all_categories]
        
        # Chi-square test
        contingency_table = [ref_counts, curr_counts]
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        drifted = p_value < threshold
        
        return drifted, p_value
    
    def detect_all_drifts(self, current_data: pd.DataFrame):
        """
        Check all columns for drift
        """
        drifts = {}
        
        # Numerical columns
        for col in current_data.select_dtypes(include=[np.number]).columns:
            drifted, p_value = self.detect_numerical_drift(current_data, col)
            if drifted:
                drifts[col] = {'type': 'numerical', 'p_value': p_value}
        
        # Categorical columns
        for col in current_data.select_dtypes(include=['object', 'category']).columns:
            drifted, p_value = self.detect_categorical_drift(current_data, col)
            if drifted:
                drifts[col] = {'type': 'categorical', 'p_value': p_value}
        
        return drifts

# Usage
detector = DataDriftDetector(training_data)
drifts = detector.detect_all_drifts(current_production_data)

if drifts:
    print("⚠️  Data drift detected in:", drifts.keys())
    # Trigger retraining or alert
```

---

## Production Best Practices

### 1. Idempotency

Ensure pipeline can be re-run safely without side effects.

```python
class IdempotentPipeline:
    """
    Pipeline that can be safely re-run
    """
    
    def process_batch(self, batch_id: str, input_path: str, output_path: str):
        """
        Process a batch idempotently
        """
        # Check if already processed
        if self.is_processed(batch_id):
            print(f"Batch {batch_id} already processed, skipping")
            return
        
        # Process
        data = self.load(input_path)
        processed = self.transform(data)
        
        # Write with batch ID
        self.save_with_checksum(processed, output_path, batch_id)
        
        # Mark as complete
        self.mark_processed(batch_id)
    
    def is_processed(self, batch_id: str) -> bool:
        """Check if batch already processed"""
        # Query metadata store
        return self.metadata_store.exists(batch_id)
    
    def mark_processed(self, batch_id: str):
        """Mark batch as processed"""
        self.metadata_store.write(batch_id, timestamp=datetime.now())
```

### 2. Data Versioning

Track versions of datasets and transformations.

```python
class VersionedDataset:
    """
    Version datasets for reproducibility
    """
    
    def save(self, df: pd.DataFrame, name: str, version: str):
        """
        Save versioned dataset
        
        Path: s3://bucket/{name}/{version}/data.parquet
        """
        path = f"s3://bucket/{name}/{version}/data.parquet"
        
        # Save data
        df.to_parquet(path)
        
        # Save metadata
        metadata = {
            'name': name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'num_rows': len(df),
            'num_cols': len(df.columns),
            'schema': df.dtypes.to_dict(),
            'checksum': self.compute_checksum(df)
        }
        
        self.save_metadata(name, version, metadata)
    
    def load(self, name: str, version: str) -> pd.DataFrame:
        """Load specific version"""
        path = f"s3://bucket/{name}/{version}/data.parquet"
        return pd.read_parquet(path)
```

### 3. Lineage Tracking

Track data transformations for debugging and compliance.

```python
class LineageTracker:
    """
    Track data lineage
    """
    
    def __init__(self):
        self.graph = {}
    
    def record_transformation(
        self, 
        input_datasets: List[str],
        output_dataset: str,
        transformation_code: str,
        parameters: Dict
    ):
        """
        Record a transformation
        """
        self.graph[output_dataset] = {
            'inputs': input_datasets,
            'transformation': transformation_code,
            'parameters': parameters,
            'timestamp': datetime.now()
        }
    
    def get_lineage(self, dataset: str) -> Dict:
        """
        Get full lineage of a dataset
        
        Returns tree of upstream datasets and transformations
        """
        if dataset not in self.graph:
            return {'dataset': dataset, 'inputs': []}
        
        node = self.graph[dataset]
        
        return {
            'dataset': dataset,
            'transformation': node['transformation'],
            'inputs': [self.get_lineage(inp) for inp in node['inputs']]
        }
```

---

## Common Preprocessing Challenges & Solutions

### Challenge 1: Imbalanced Classes

**Problem:** 95% of samples are class 0, 5% are class 1. Model always predicts class 0.

**Solutions:**

```python
class ImbalanceHandler:
    """
    Handle class imbalance
    """
    
    def upsample_minority(self, df, target_col):
        """
        Oversample minority class
        """
        from sklearn.utils import resample
        
        # Separate majority and minority classes
        df_majority = df[df[target_col] == 0]
        df_minority = df[df[target_col] == 1]
        
        # Upsample minority class
        df_minority_upsampled = resample(
            df_minority,
            replace=True,  # Sample with replacement
            n_samples=len(df_majority),  # Match majority class size
            random_state=42
        )
        
        # Combine
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
        
        return df_balanced
    
    def downsample_majority(self, df, target_col):
        """
        Undersample majority class
        """
        df_majority = df[df[target_col] == 0]
        df_minority = df[df[target_col] == 1]
        
        # Downsample majority class
        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=len(df_minority),
            random_state=42
        )
        
        df_balanced = pd.concat([df_majority_downsampled, df_minority])
        
        return df_balanced
    
    def smote(self, X, y):
        """
        Synthetic Minority Over-sampling Technique
        
        Generate synthetic samples for minority class
        """
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        return X_resampled, y_resampled
```

### Challenge 2: High-Cardinality Categoricals

**Problem:** User IDs have 10M unique values. One-hot encoding creates 10M columns.

**Solutions:**

```python
class HighCardinalityEncoder:
    """
    Handle high-cardinality categorical features
    """
    
    def target_encoding(self, df, cat_col, target_col):
        """
        Encode category by mean of target
        
        Example:
          city='SF' → mean(target | city='SF') = 0.65
          city='NY' → mean(target | city='NY') = 0.52
        
        Warning: Risk of overfitting. Use cross-validation encoding.
        """
        # Compute target mean per category
        target_means = df.groupby(cat_col)[target_col].mean()
        
        # Map
        df[f'{cat_col}_target_enc'] = df[cat_col].map(target_means)
        
        return df
    
    def frequency_encoding(self, df, cat_col):
        """
        Encode by frequency
        
        Common categories → higher values
        """
        freq = df[cat_col].value_counts(normalize=True)
        df[f'{cat_col}_freq'] = df[cat_col].map(freq)
        
        return df
    
    def hashing_trick(self, df, cat_col, n_features=100):
        """
        Hash categories into fixed number of buckets
        
        Pros: Fixed dimension
        Cons: Hash collisions
        """
        from sklearn.feature_extraction import FeatureHasher
        
        hasher = FeatureHasher(n_features=n_features, input_type='string')
        hashed = hasher.transform(df[[cat_col]].astype(str).values)
        
        # Convert to DataFrame
        hashed_df = pd.DataFrame(
            hashed.toarray(),
            columns=[f'{cat_col}_hash_{i}' for i in range(n_features)]
        )
        
        return pd.concat([df, hashed_df], axis=1)
```

### Challenge 3: Streaming Data Preprocessing

**Problem:** Need to preprocess real-time streams with low latency.

**Solution:**

```python
from kafka import KafkaConsumer, KafkaProducer
import json

class StreamingPreprocessor:
    """
    Real-time preprocessing for streaming data
    """
    
    def __init__(self):
        self.consumer = KafkaConsumer(
            'raw_events',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Load fitted preprocessor (from training)
        self.preprocessor = PreprocessorV1.load('models/preprocessor_v1.pkl')
    
    def process_stream(self):
        """
        Process events in real-time
        """
        for message in self.consumer:
            event = message.value
            
            # Preprocess
            processed = self.preprocess_event(event)
            
            # Validate
            if self.validate(processed):
                # Send to processed topic
                self.producer.send('processed_events', processed)
    
    def preprocess_event(self, event):
        """
        Preprocess single event (must be fast!)
        """
        # Convert to DataFrame
        df = pd.DataFrame([event])
        
        # Apply preprocessing
        df = self.preprocessor.transform(df)
        
        # Convert back to dict
        return df.to_dict('records')[0]
    
    def validate(self, event):
        """Quick validation"""
        required_fields = ['user_id', 'timestamp', 'features']
        return all(field in event for field in required_fields)
```

### Challenge 4: Privacy & Compliance (GDPR, CCPA)

**Problem:** Need to handle PII (Personally Identifiable Information).

**Solutions:**

```python
import hashlib

class PrivacyPreserver:
    """
    Handle PII in preprocessing
    """
    
    def anonymize_user_ids(self, df, id_col='user_id'):
        """
        Hash user IDs to anonymize
        """
        df[f'{id_col}_anonymized'] = df[id_col].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest()
        )
        df.drop(id_col, axis=1, inplace=True)
        return df
    
    def remove_pii(self, df, pii_cols=['email', 'phone', 'address']):
        """
        Remove PII columns
        """
        df.drop(pii_cols, axis=1, inplace=True, errors='ignore')
        return df
    
    def differential_privacy_noise(self, df, numeric_cols, epsilon=1.0):
        """
        Add Laplacian noise for differential privacy
        
        Args:
            epsilon: Privacy parameter (lower = more privacy, less utility)
        """
        for col in numeric_cols:
            sensitivity = df[col].max() - df[col].min()
            noise_scale = sensitivity / epsilon
            
            noise = np.random.laplace(0, noise_scale, size=len(df))
            df[col] = df[col] + noise
        
        return df
```

---

## Performance Optimization

### 1. Parallelize Transformations

```python
from multiprocessing import Pool
import numpy as np

class ParallelPreprocessor:
    """
    Parallelize preprocessing across CPU cores
    """
    
    def __init__(self, n_workers=4):
        self.n_workers = n_workers
    
    def process_parallel(self, df, transform_fn):
        """
        Apply transformation in parallel
        """
        # Split dataframe into chunks
        chunks = np.array_split(df, self.n_workers)
        
        # Process chunks in parallel
        with Pool(self.n_workers) as pool:
            processed_chunks = pool.map(transform_fn, chunks)
        
        # Combine results
        return pd.concat(processed_chunks)
```

### 2. Use Efficient Data Formats

```python
# Bad: CSV (slow to read/write, no compression)
df.to_csv('data.csv')  # 1 GB file, 60 seconds

# Better: Parquet (columnar, compressed)
df.to_parquet('data.parquet')  # 200 MB file, 5 seconds

# Best for streaming: Avro or Protocol Buffers
```

### 3. Cache Intermediate Results

```python
class CachedPreprocessor:
    """
    Cache preprocessing results
    """
    
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = cache_dir
    
    def process_with_cache(self, df, batch_id):
        """
        Check cache before processing
        """
        cache_path = f"{self.cache_dir}/{batch_id}.parquet"
        
        if os.path.exists(cache_path):
            print(f"Loading from cache: {batch_id}")
            return pd.read_parquet(cache_path)
        
        # Process
        processed = self.preprocess(df)
        
        # Save to cache
        processed.to_parquet(cache_path)
        
        return processed
```

---

## Key Takeaways

✅ **Data quality is critical** - bad data → bad models  
✅ **Schema validation** catches errors early before expensive processing  
✅ **Handle missing values** with domain-appropriate strategies (mean/median/forward-fill)  
✅ **Feature engineering** is where domain knowledge creates value  
✅ **Prevent training/serving skew** with unified preprocessing code  
✅ **Monitor data quality** continuously - detect drift and anomalies  
✅ **Use feature stores** for consistency and reuse at scale  
✅ **Distributed processing** (Spark/Beam) required for large-scale data  
✅ **Version datasets and transformations** for reproducibility  
✅ **Track data lineage** for debugging and compliance  
✅ **Handle class imbalance** with resampling or SMOTE  
✅ **Encode high-cardinality categoricals** with target/frequency encoding or hashing  
✅ **Optimize performance** with parallel processing, efficient formats, caching

---

**Originally published at:** [arunbaby.com/ml-system-design/0003-data-preprocessing](https://www.arunbaby.com/ml-system-design/0003-data-preprocessing/)

*If you found this helpful, consider sharing it with others who might benefit.*

