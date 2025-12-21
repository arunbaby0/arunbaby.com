---
title: "A/B Testing Systems for ML"
day: 4
collection: ml_system_design
categories:
  - ml-system-design
tags:
  - experimentation
  - ab-testing
  - metrics
  - statistical-testing
subdomain: Experimentation & Metrics
tech_stack: [Python, SQL, Apache Spark, Redis, Kafka]
scale: "10M+ users"
companies: [Google, Meta, Netflix, Airbnb, Uber]
related_dsa_day: 4
related_speech_day: 4
related_agents_day: 4
---

**How to design experimentation platforms that enable rapid iteration while maintaining statistical rigor at scale.**

## Introduction

A/B testing is the **backbone of data-driven decision making** in ML systems. Every major tech company runs thousands of experiments simultaneously to:

- Test new model versions
- Validate product changes
- Optimize user experience
- Measure feature impact

**Why it matters:**
- **Validate improvements:** Ensure new models actually perform better
- **Reduce risk:** Test changes on small cohorts before full rollout
- **Quantify impact:** Measure precise effect size, not just gut feeling
- **Enable velocity:** Run multiple experiments in parallel

**What you'll learn:**
- A/B testing architecture for ML systems
- Statistical foundations (hypothesis testing, power analysis)
- Experiment assignment and randomization
- Metrics tracking and analysis
- Guardrail metrics and quality assurance
- Real-world examples from tech giants

---

## Problem Definition

Design an A/B testing platform for ML systems.

### Functional Requirements

1. **Experiment Setup**
   - Create experiments with control/treatment variants
   - Define success metrics and guardrails
   - Set experiment parameters (duration, traffic allocation)
   - Support multi-variant testing (A/B/C/D)

2. **User Assignment**
   - Randomly assign users to variants
   - Ensure consistency (same user always sees same variant)
   - Support layered experiments
   - Handle new vs returning users

3. **Metrics Tracking**
   - Log user actions and outcomes
   - Compute experiment metrics in real-time
   - Track both primary and secondary metrics
   - Monitor guardrail metrics

4. **Statistical Analysis**
   - Calculate statistical significance
   - Compute confidence intervals
   - Detect early wins/losses
   - Generate experiment reports

### Non-Functional Requirements

1. **Scale**
   - Handle 10M+ users
   - Support 100+ concurrent experiments
   - Process billions of events/day

2. **Latency**
   - Assignment: < 10ms
   - Metrics updates: Near real-time (< 1 minute lag)

3. **Reliability**
   - 99.9% uptime
   - No data loss
   - Audit trail for all experiments

4. **Statistical Rigor**
   - Type I error (false positive) < 5%
   - Sufficient statistical power (80%+)
   - Multiple testing corrections

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Experimentation Platform                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Experiment Configuration Service             │  │
│  │  - Create experiments                                 │  │
│  │  - Define metrics                                     │  │
│  │  - Set parameters                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Assignment Service                           │  │
│  │  - Hash user_id → variant                            │  │
│  │  - Consistent assignment                              │  │
│  │  - Cache assignments                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Event Logging Service                        │  │
│  │  - Log user actions                                   │  │
│  │  - Track outcomes                                     │  │
│  │  - Stream to analytics                                │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Metrics Aggregation Service                  │  │
│  │  - Compute experiment metrics                         │  │
│  │  - Real-time dashboards                               │  │
│  │  - Statistical tests                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Analysis & Reporting Service                 │  │
│  │  - Statistical significance                           │  │
│  │  - Confidence intervals                               │  │
│  │  - Decision recommendations                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Data Flow:
User Request → Assignment → Show Variant → Log Events → Aggregate Metrics → Analyze
```

---

## Component 1: Experiment Assignment

Assign users to experiment variants consistently and randomly.

### Deterministic Assignment via Hashing

```python
import hashlib
from typing import List, Dict

class ExperimentAssigner:
    """
    Assign users to experiment variants
    
    Requirements:
    - Deterministic: Same user_id → same variant
    - Random: Uniform distribution across variants
    - Independent: Different experiments use different hash seeds
    """
    
    def __init__(self):
        self.experiments = {}  # experiment_id → config
    
    def create_experiment(
        self,
        experiment_id: str,
        variants: List[str],
        traffic_allocation: float = 1.0
    ):
        """
        Create new experiment
        
        Args:
            experiment_id: Unique experiment identifier
            variants: List of variant names (e.g., ['control', 'treatment'])
            traffic_allocation: Fraction of users to include (0.0 to 1.0)
        """
        self.experiments[experiment_id] = {
            'variants': variants,
            'traffic_allocation': traffic_allocation,
            'num_variants': len(variants)
        }
    
    def assign_variant(self, user_id: str, experiment_id: str) -> str:
        """
        Assign user to variant
        
        Uses consistent hashing for deterministic assignment
        
        Returns:
            Variant name or None if user not in experiment
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        
        # Hash user_id + experiment_id
        hash_input = f"{user_id}:{experiment_id}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        
        # Map to [0, 1]
        normalized = (hash_value % 10000) / 10000.0
        
        # Check if user is in experiment (traffic allocation)
        if normalized >= config['traffic_allocation']:
            return None  # User not in experiment
        
        # Assign to variant
        # Re-normalize to [0, 1] within allocated traffic
        variant_hash = normalized / config['traffic_allocation']
        variant_idx = int(variant_hash * config['num_variants'])
        
        return config['variants'][variant_idx]

# Usage
assigner = ExperimentAssigner()

# Create experiment: 50% control, 50% treatment, 100% of users
assigner.create_experiment(
    experiment_id='model_v2_test',
    variants=['control', 'treatment'],
    traffic_allocation=1.0
)

# Assign users
user_1_variant = assigner.assign_variant('user_123', 'model_v2_test')
print(f"User 123 assigned to: {user_1_variant}")

# Same user always gets same variant
assert assigner.assign_variant('user_123', 'model_v2_test') == user_1_variant
```

### Why Hashing Works

**Properties of MD5/SHA hashing:**
1. **Deterministic:** Same input → same output
2. **Uniform:** Output uniformly distributed
3. **Independent:** Different inputs → uncorrelated outputs

**Key insight:** Hash(user_id + experiment_id) acts as a random number generator with a fixed seed per user-experiment pair.

### Handling Traffic Allocation

```python
def assign_with_traffic_split(self, user_id: str, experiment_id: str) -> str:
    """
    Assign with partial traffic allocation
    
    Example: 10% of users in experiment
    - Hash to [0, 1]
    - If < 0.10 → assign to variant
    - Else → not in experiment
    """
    config = self.experiments[experiment_id]
    
    # Hash
    hash_input = f"{user_id}:{experiment_id}".encode('utf-8')
    hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
    normalized = (hash_value % 10000) / 10000.0
    
    # Traffic allocation check
    if normalized >= config['traffic_allocation']:
        return None
    
    # Within traffic, assign to variant
    # Scale normalized to [0, traffic_allocation] → [0, 1]
    variant_hash = normalized / config['traffic_allocation']
    variant_idx = int(variant_hash * config['num_variants'])
    
    return config['variants'][variant_idx]
```

---

## Component 2: Metrics Tracking

Track user actions and compute experiment metrics.

### Event Logging

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json

@dataclass
class ExperimentEvent:
    """Single experiment event"""
    user_id: str
    experiment_id: str
    variant: str
    event_type: str  # e.g., 'impression', 'click', 'purchase'
    timestamp: datetime
    metadata: dict = None
    
    def to_dict(self):
        return {
            'user_id': self.user_id,
            'experiment_id': self.experiment_id,
            'variant': self.variant,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }

class EventLogger:
    """
    Log experiment events
    
    In production: Stream to Kafka/Kinesis → Data warehouse
    """
    
    def __init__(self, output_file='experiment_events.jsonl'):
        self.output_file = output_file
    
    def log_event(self, event: ExperimentEvent):
        """
        Log single event
        
        In production: Send to message queue
        """
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')
    
    def log_assignment(self, user_id: str, experiment_id: str, variant: str):
        """Log when user is assigned to variant"""
        event = ExperimentEvent(
            user_id=user_id,
            experiment_id=experiment_id,
            variant=variant,
            event_type='assignment',
            timestamp=datetime.now()
        )
        self.log_event(event)
    
    def log_metric_event(
        self,
        user_id: str,
        experiment_id: str,
        variant: str,
        metric_name: str,
        metric_value: float
    ):
        """Log metric event (e.g., click, purchase)"""
        event = ExperimentEvent(
            user_id=user_id,
            experiment_id=experiment_id,
            variant=variant,
            event_type=metric_name,
            timestamp=datetime.now(),
            metadata={'value': metric_value}
        )
        self.log_event(event)

# Usage
logger = EventLogger()

# Log assignment
logger.log_assignment('user_123', 'model_v2_test', 'treatment')

# Log click
logger.log_metric_event('user_123', 'model_v2_test', 'treatment', 'click', 1.0)

# Log purchase
logger.log_metric_event('user_123', 'model_v2_test', 'treatment', 'purchase', 49.99)
```

### Metrics Aggregation

```python
from collections import defaultdict
import pandas as pd

class MetricsAggregator:
    """
    Aggregate experiment metrics from events
    
    Computes per-variant statistics
    """
    
    def __init__(self):
        self.variant_stats = defaultdict(lambda: defaultdict(list))
    
    def add_event(self, variant: str, metric_name: str, value: float):
        """Add metric value for variant"""
        self.variant_stats[variant][metric_name].append(value)
    
    def compute_metrics(self, experiment_id: str) -> pd.DataFrame:
        """
        Compute aggregated metrics per variant
        
        Returns DataFrame with columns:
        - variant
        - metric
        - count
        - mean
        - std
        - sum
        """
        results = []
        
        for variant, metrics in self.variant_stats.items():
            for metric_name, values in metrics.items():
                import numpy as np
                
                results.append({
                    'variant': variant,
                    'metric': metric_name,
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'sum': np.sum(values),
                    'min': np.min(values),
                    'max': np.max(values)
                })
        
        return pd.DataFrame(results)

# Usage
aggregator = MetricsAggregator()

# Simulate events
aggregator.add_event('control', 'ctr', 0.05)
aggregator.add_event('control', 'ctr', 0.04)
aggregator.add_event('treatment', 'ctr', 0.06)
aggregator.add_event('treatment', 'ctr', 0.07)

metrics_df = aggregator.compute_metrics('model_v2_test')
print(metrics_df)
```

---

## Component 3: Statistical Analysis

Determine if observed differences are statistically significant.

### T-Test for Continuous Metrics

```python
from scipy import stats
import numpy as np

class StatisticalAnalyzer:
    """
    Perform statistical tests on experiment data
    """
    
    def t_test(
        self,
        control_values: List[float],
        treatment_values: List[float],
        alpha: float = 0.05
    ) -> dict:
        """
        Two-sample t-test
        
        H0: mean(treatment) = mean(control)
        H1: mean(treatment) ≠ mean(control)
        
        Args:
            control_values: Metric values from control group
            treatment_values: Metric values from treatment group
            alpha: Significance level (default 0.05)
        
        Returns:
            Dictionary with test results
        """
        control = np.array(control_values)
        treatment = np.array(treatment_values)
        
        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(treatment, control)
        
        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control) - 1) * np.var(control, ddof=1) +
             (len(treatment) - 1) * np.var(treatment, ddof=1)) /
            (len(control) + len(treatment) - 2)
        )
        
        cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for difference
        se = pooled_std * np.sqrt(1/len(control) + 1/len(treatment))
        df = len(control) + len(treatment) - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        mean_diff = np.mean(treatment) - np.mean(control)
        ci_lower = mean_diff - t_critical * se
        ci_upper = mean_diff + t_critical * se
        
        # Relative lift
        relative_lift = (np.mean(treatment) / np.mean(control) - 1) * 100 if np.mean(control) > 0 else 0
        
        return {
            'control_mean': np.mean(control),
            'treatment_mean': np.mean(treatment),
            'absolute_diff': mean_diff,
            'relative_lift_pct': relative_lift,
            't_statistic': t_statistic,
            'p_value': p_value,
            'is_significant': p_value < alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'cohens_d': cohens_d,
            'sample_size_control': len(control),
            'sample_size_treatment': len(treatment)
        }

    def chi_square_test(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        alpha: float = 0.05
    ) -> dict:
        """
        Chi-square test for proportions (e.g., CTR, conversion rate)
        """
        # Construct contingency table
        contingency = np.array([
            [treatment_successes, treatment_total - treatment_successes],
            [control_successes, control_total - control_successes]
        ])
        
        # Chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        # Rates
        control_rate = control_successes / control_total if control_total > 0 else 0
        treatment_rate = treatment_successes / treatment_total if treatment_total > 0 else 0
        
        # Relative lift
        relative_lift = (treatment_rate / control_rate - 1) * 100 if control_rate > 0 else 0
        
        # CI for difference in proportions (Wald)
        p1 = treatment_rate
        p2 = control_rate
        se = np.sqrt(
            (p1 * (1 - p1) / max(treatment_total, 1)) +
            (p2 * (1 - p2) / max(control_total, 1))
        )
        z_critical = stats.norm.ppf(1 - alpha / 2)
        diff = p1 - p2
        ci_lower = diff - z_critical * se
        ci_upper = diff + z_critical * se
        
        return {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'absolute_diff': diff,
            'relative_lift_pct': relative_lift,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'is_significant': p_value < alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'sample_size_control': control_total,
            'sample_size_treatment': treatment_total
        }

# Usage
analyzer = StatisticalAnalyzer()

# Simulate metric data (e.g., session duration in seconds)
control_sessions = np.random.normal(120, 30, size=1000)  # mean=120s, std=30s
treatment_sessions = np.random.normal(125, 30, size=1000)  # mean=125s (5s improvement)

result = analyzer.t_test(control_sessions, treatment_sessions)

print(f"Control mean: {result['control_mean']:.2f}")
print(f"Treatment mean: {result['treatment_mean']:.2f}")
print(f"Relative lift: {result['relative_lift_pct']:.2f}%")
print(f"P-value: {result['p_value']:.4f}")
print(f"Significant: {result['is_significant']}")
print(f"95% CI: [{result['confidence_interval'][0]:.2f}, {result['confidence_interval'][1]:.2f}]")
```

### Chi-Square Test for Binary Metrics

```python
def chi_square_test(
    self,
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    alpha: float = 0.05
) -> dict:
    """
    Chi-square test for proportions (e.g., CTR, conversion rate)
    
    H0: p(treatment) = p(control)
    H1: p(treatment) ≠ p(control)
    """
    # Construct contingency table
    contingency = np.array([
        [treatment_successes, treatment_total - treatment_successes],
        [control_successes, control_total - control_successes]
    ])
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    # Compute rates
    control_rate = control_successes / control_total if control_total > 0 else 0
    treatment_rate = treatment_successes / treatment_total if treatment_total > 0 else 0
    
    # Relative lift
    relative_lift = (treatment_rate / control_rate - 1) * 100 if control_rate > 0 else 0
    
    # Confidence interval for difference in proportions
    p1 = treatment_rate
    p2 = control_rate
    
    se = np.sqrt(p1*(1-p1)/treatment_total + p2*(1-p2)/control_total)
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    diff = p1 - p2
    ci_lower = diff - z_critical * se
    ci_upper = diff + z_critical * se
    
    return {
        'control_rate': control_rate,
        'treatment_rate': treatment_rate,
        'absolute_diff': diff,
        'relative_lift_pct': relative_lift,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'is_significant': p_value < alpha,
        'confidence_interval': (ci_lower, ci_upper),
        'sample_size_control': control_total,
        'sample_size_treatment': treatment_total
    }

# Add to StatisticalAnalyzer class

# Usage
# Example: Click-through rate test
control_clicks = 450
control_impressions = 10000
treatment_clicks = 520
treatment_impressions = 10000

result = analyzer.chi_square_test(
    control_clicks, control_impressions,
    treatment_clicks, treatment_impressions
)

print(f"Control CTR: {result['control_rate']*100:.2f}%")
print(f"Treatment CTR: {result['treatment_rate']*100:.2f}%")
print(f"Relative lift: {result['relative_lift_pct']:.2f}%")
print(f"P-value: {result['p_value']:.4f}")
print(f"Significant: {result['is_significant']}")
```

---

## Sample Size Calculation & Power Analysis

Determine required sample size before running experiment.

### Power Analysis

```python
from scipy.stats import norm

class PowerAnalysis:
    """
    Calculate required sample size for experiments
    """
    
    def sample_size_for_proportions(
        self,
        baseline_rate: float,
        mde: float,  # Minimum Detectable Effect
        alpha: float = 0.05,
        power: float = 0.80
    ) -> int:
        """
        Calculate sample size needed to detect effect on proportion
        
        Args:
            baseline_rate: Current conversion rate (e.g., 0.05 for 5%)
            mde: Minimum relative effect to detect (e.g., 0.10 for 10% improvement)
            alpha: Significance level (Type I error rate)
            power: Statistical power (1 - Type II error rate)
        
        Returns:
            Required sample size per variant
        """
        # Target rate after improvement
        target_rate = baseline_rate * (1 + mde)
        
        # Z-scores
        z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed
        z_beta = norm.ppf(power)
        
        # Pooled proportion under H0
        p_avg = (baseline_rate + target_rate) / 2
        
        # Sample size formula
        numerator = (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) +
                    z_beta * np.sqrt(baseline_rate * (1 - baseline_rate) +
                                    target_rate * (1 - target_rate))) ** 2
        
        denominator = (target_rate - baseline_rate) ** 2
        
        n = numerator / denominator
        
        return int(np.ceil(n))
    
    def sample_size_for_means(
        self,
        baseline_mean: float,
        baseline_std: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> int:
        """
        Calculate sample size for continuous metric
        
        Args:
            baseline_mean: Current mean value
            baseline_std: Standard deviation
            mde: Minimum relative effect (e.g., 0.05 for 5% improvement)
            alpha: Significance level
            power: Statistical power
        
        Returns:
            Required sample size per variant
        """
        target_mean = baseline_mean * (1 + mde)
        effect_size = abs(target_mean - baseline_mean) / baseline_std
        
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    def experiment_duration(
        self,
        required_sample_size: int,
        daily_users: int,
        traffic_allocation: float = 0.5
    ) -> int:
        """
        Calculate experiment duration in days
        
        Args:
            required_sample_size: Sample size per variant
            daily_users: Daily active users
            traffic_allocation: Fraction of users in experiment
        
        Returns:
            Duration in days
        """
        users_per_day = daily_users * traffic_allocation
        days = required_sample_size / users_per_day
        
        return int(np.ceil(days))

# Usage
power = PowerAnalysis()

# Example: CTR improvement test
current_ctr = 0.05  # 5% baseline
mde = 0.10  # Want to detect 10% relative improvement (5% → 5.5%)

sample_size = power.sample_size_for_proportions(
    baseline_rate=current_ctr,
    mde=mde,
    alpha=0.05,
    power=0.80
)

print(f"Required sample size per variant: {sample_size:,}")

# If we have 100K daily users and allocate 50% to experiment
duration = power.experiment_duration(
    required_sample_size=sample_size,
    daily_users=100000,
    traffic_allocation=0.5
)

print(f"Experiment duration: {duration} days")
```

---

## Guardrail Metrics

Ensure experiments don't harm key business metrics.

### Implementing Guardrails

```python
class GuardrailChecker:
    """
    Monitor guardrail metrics during experiments
    
    Guardrails: Metrics that must not degrade
    """
    
    def __init__(self):
        self.guardrails = {}
    
    def define_guardrail(
        self,
        metric_name: str,
        threshold_type: str,  # 'relative' or 'absolute'
        threshold_value: float,
        direction: str  # 'decrease' or 'increase'
    ):
        """
        Define a guardrail metric
        
        Example:
          - Revenue must not decrease by more than 2%
          - Error rate must not increase by more than 0.5 percentage points
        """
        self.guardrails[metric_name] = {
            'threshold_type': threshold_type,
            'threshold_value': threshold_value,
            'direction': direction
        }
    
    def check_guardrails(
        self,
        control_metrics: dict,
        treatment_metrics: dict
    ) -> dict:
        """
        Check if treatment violates guardrails
        
        Returns:
            Dictionary of guardrail violations
        """
        violations = {}
        
        for metric_name, guardrail in self.guardrails.items():
            control_value = control_metrics.get(metric_name)
            treatment_value = treatment_metrics.get(metric_name)
            
            if control_value is None or treatment_value is None:
                continue
            
            # Calculate change
            if guardrail['threshold_type'] == 'relative':
                change = (treatment_value / control_value - 1) * 100
            else:  # absolute
                change = treatment_value - control_value
            
            # Check violation
            violated = False
            
            if guardrail['direction'] == 'decrease':
                # Metric should not decrease beyond threshold
                if change < -guardrail['threshold_value']:
                    violated = True
            else:  # increase
                # Metric should not increase beyond threshold
                if change > guardrail['threshold_value']:
                    violated = True
            
            if violated:
                violations[metric_name] = {
                    'control': control_value,
                    'treatment': treatment_value,
                    'change': change,
                    'threshold': guardrail['threshold_value'],
                    'type': guardrail['threshold_type']
                }
        
        return violations

# Usage
guardrails = GuardrailChecker()

# Define guardrails
guardrails.define_guardrail(
    metric_name='revenue_per_user',
    threshold_type='relative',
    threshold_value=2.0,  # Cannot decrease by more than 2%
    direction='decrease'
)

guardrails.define_guardrail(
    metric_name='error_rate',
    threshold_type='absolute',
    threshold_value=0.5,  # Cannot increase by more than 0.5 percentage points
    direction='increase'
)

# Check guardrails
control_metrics = {
    'revenue_per_user': 10.0,
    'error_rate': 1.0
}

treatment_metrics = {
    'revenue_per_user': 9.5,  # 5% decrease - violates guardrail!
    'error_rate': 1.2  # 0.2pp increase - OK
}

violations = guardrails.check_guardrails(control_metrics, treatment_metrics)

if violations:
    print("⚠️ Guardrail violations detected:")
    for metric, details in violations.items():
        print(f"  {metric}: {details['change']:.2f}% change (threshold: {details['threshold']}%)")
else:
    print("✅ All guardrails passed")
```

---

## Real-World Examples

### Netflix: Experimentation at Scale

**Scale:**
- 1000+ experiments running concurrently
- 200M+ users worldwide
- Multiple metrics per experiment

**Key innovations:**
- **Quasi-experimentation:** Use observational data when randomization not possible
- **Interleaving:** Test ranking algorithms by mixing results
- **Heterogeneous treatment effects:** Analyze impact per user segment

**Example metric:**
- **Stream starts per member:** How many shows/movies a user starts watching
- **Effective catalog size:** Number of unique titles watched (diversity metric)

### Google: Large-scale Testing

**Scale:**
- 10,000+ experiments per year
- 1B+ users
- Experiments across Search, Ads, YouTube, etc.

**Methodology:**
- **Layered experiments:** Run multiple experiments on same users (orthogonal layers)
- **Ramping:** Gradually increase traffic allocation
- **Long-running holdouts:** Keep small % in old version to measure long-term effects

**Example:**
Testing new ranking algorithm in Google Search:
- **Primary metric:** Click-through rate on top results
- **Guardrails:** Ad revenue, latency, user satisfaction
- **Duration:** 2-4 weeks
- **Traffic:** Start at 1%, ramp to 50%

---

## Advanced Topics

### Sequential Testing & Early Stopping

Stop experiments early when results are conclusive.

```python
import math
from scipy.stats import norm

class SequentialTesting:
    """
    Sequential probability ratio test (SPRT)
    
    Allows stopping experiment early while controlling error rates
    """
    
    def __init__(
        self,
        alpha=0.05,
        beta=0.20,
        mde=0.05  # Minimum detectable effect
    ):
        self.alpha = alpha  # Type I error rate
        self.beta = beta  # Type II error rate (1 - power)
        self.mde = mde
        
        # Calculate log-likelihood ratio bounds
        self.upper_bound = math.log((1 - beta) / alpha)
        self.lower_bound = math.log(beta / (1 - alpha))
    
    def should_stop(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int
    ) -> dict:
        """
        Check if experiment can be stopped
        
        Returns:
            {
                'decision': 'continue' | 'stop_treatment_wins' | 'stop_control_wins',
                'log_likelihood_ratio': float
            }
        """
        # Compute rates
        p_control = control_successes / control_total if control_total > 0 else 0
        p_treatment = treatment_successes / treatment_total if treatment_total > 0 else 0
        
        # Avoid edge cases
        p_control = max(min(p_control, 0.9999), 0.0001)
        p_treatment = max(min(p_treatment, 0.9999), 0.0001)
        
        # Log-likelihood ratio
        # H1: treatment is better by mde
        # H0: treatment = control
        
        p_h1 = p_control * (1 + self.mde)
        
        llr = 0
        
        # Contribution from treatment group
        llr += treatment_successes * math.log(p_h1 / p_control)
        llr += (treatment_total - treatment_successes) * math.log((1 - p_h1) / (1 - p_control))
        
        # Decision
        if llr >= self.upper_bound:
            return {'decision': 'stop_treatment_wins', 'log_likelihood_ratio': llr}
        elif llr <= self.lower_bound:
            return {'decision': 'stop_control_wins', 'log_likelihood_ratio': llr}
        else:
            return {'decision': 'continue', 'log_likelihood_ratio': llr}

# Usage
sequential = SequentialTesting(alpha=0.05, beta=0.20, mde=0.05)

# Check daily
for day in range(1, 15):
    control_clicks = day * 450
    control_impressions = day * 10000
    treatment_clicks = day * 500
    treatment_impressions = day * 10000
    
    result = sequential.should_stop(
        control_clicks, control_impressions,
        treatment_clicks, treatment_impressions
    )
    
    print(f"Day {day}: {result['decision']}")
    
    if result['decision'] != 'continue':
        print(f"🎉 Experiment can stop on day {day}!")
        break
```

### Multi-Armed Bandits

Allocate traffic dynamically to better-performing variants.

```python
import numpy as np

class ThompsonSampling:
    """
    Thompson Sampling for multi-armed bandit
    
    Dynamically allocate traffic to maximize reward
    while exploring alternatives
    """
    
    def __init__(self, num_variants):
        self.num_variants = num_variants
        
        # Beta distribution parameters for each variant
        # Beta(alpha, beta) represents posterior belief
        self.alpha = np.ones(num_variants)  # Success count + 1
        self.beta = np.ones(num_variants)  # Failure count + 1
    
    def select_variant(self) -> int:
        """
        Select variant to show to next user
        
        Sample from each variant's posterior and pick the best
        """
        # Sample from each variant's posterior distribution
        sampled_values = [
            np.random.beta(self.alpha[i], self.beta[i])
            for i in range(self.num_variants)
        ]
        
        # Select variant with highest sample
        return np.argmax(sampled_values)
    
    def update(self, variant: int, reward: float):
        """
        Update beliefs after observing reward
        
        Args:
            variant: Which variant was shown
            reward: 0 or 1 (failure or success)
        """
        if reward > 0:
            self.alpha[variant] += 1
        else:
            self.beta[variant] += 1
    
    def get_statistics(self):
        """Get current statistics for each variant"""
        stats = []
        
        for i in range(self.num_variants):
            # Mean of Beta(alpha, beta) = alpha / (alpha + beta)
            mean = self.alpha[i] / (self.alpha[i] + self.beta[i])
            
            # 95% credible interval
            samples = np.random.beta(self.alpha[i], self.beta[i], size=10000)
            ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])
            
            stats.append({
                'variant': i,
                'estimated_mean': mean,
                'total_samples': self.alpha[i] + self.beta[i] - 2,
                'successes': self.alpha[i] - 1,
                'credible_interval': (ci_lower, ci_upper)
            })
        
        return stats

# Usage
bandit = ThompsonSampling(num_variants=3)

# Simulate 10,000 users
for user in range(10000):
    # Select variant to show
    variant = bandit.select_variant()
    
    # Simulate user interaction (variant 2 is best: 6% CTR)
    true_ctrs = [0.04, 0.05, 0.06]
    clicked = np.random.random() < true_ctrs[variant]
    
    # Update beliefs
    bandit.update(variant, 1.0 if clicked else 0.0)

# Check statistics
stats = bandit.get_statistics()
for s in stats:
    print(f"Variant {s['variant']}: "
          f"Estimated CTR = {s['estimated_mean']:.3f}, "
          f"Samples = {s['total_samples']}, "
          f"95% CI = [{s['credible_interval'][0]:.3f}, {s['credible_interval'][1]:.3f}]")
```

### Variance Reduction: CUPED

Reduce variance by using pre-experiment covariates.

```python
class CUPED:
    """
    Controlled-experiment Using Pre-Experiment Data
    
    Reduces variance by adjusting for pre-experiment metrics
    """
    
    def __init__(self):
        pass
    
    def adjust_metric(
        self,
        y: np.ndarray,  # Post-experiment metric
        x: np.ndarray,  # Pre-experiment metric (covariate)
    ) -> np.ndarray:
        """
        Adjust post-experiment metric using pre-experiment data
        
        Adjusted metric: y_adj = y - theta * (x - E[x])
        
        Where theta is chosen to minimize variance of y_adj
        """
        # Compute optimal theta
        # theta = Cov(y, x) / Var(x)
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        cov_yx = np.mean((y - mean_y) * (x - mean_x))
        var_x = np.var(x, ddof=1)
        
        if var_x == 0:
            return y  # No adjustment possible
        
        theta = cov_yx / var_x
        
        # Adjust y
        y_adjusted = y - theta * (x - mean_x)
        
        return y_adjusted
    
    def compare_variants_with_cuped(
        self,
        control_post: np.ndarray,
        control_pre: np.ndarray,
        treatment_post: np.ndarray,
        treatment_pre: np.ndarray
    ) -> dict:
        """
        Compare variants using CUPED
        
        Returns improvement in statistical power
        """
        # Original comparison (without CUPED)
        from scipy import stats
        
        original_t, original_p = stats.ttest_ind(treatment_post, control_post)
        original_var = np.var(treatment_post) + np.var(control_post)
        
        # Adjust metrics
        all_pre = np.concatenate([control_pre, treatment_pre])
        all_post = np.concatenate([control_post, treatment_post])
        
        adjusted_post = self.adjust_metric(all_post, all_pre)
        
        # Split back
        n_control = len(control_post)
        control_post_adj = adjusted_post[:n_control]
        treatment_post_adj = adjusted_post[n_control:]
        
        # Adjusted comparison
        adjusted_t, adjusted_p = stats.ttest_ind(treatment_post_adj, control_post_adj)
        adjusted_var = np.var(treatment_post_adj) + np.var(control_post_adj)
        
        # Variance reduction
        variance_reduction = (original_var - adjusted_var) / original_var * 100
        
        return {
            'original_p_value': original_p,
            'adjusted_p_value': adjusted_p,
            'variance_reduction_pct': variance_reduction,
            'power_improvement': (original_var / adjusted_var) ** 0.5
        }

# Example: Using pre-experiment purchase history to reduce variance
control_pre = np.random.normal(100, 30, size=500)  # Past purchases
control_post = control_pre + np.random.normal(5, 20, size=500)  # Correlated

treatment_pre = np.random.normal(100, 30, size=500)
treatment_post = treatment_pre + np.random.normal(8, 20, size=500)  # Slightly better

cuped = CUPED()
result = cuped.compare_variants_with_cuped(
    control_post, control_pre,
    treatment_post, treatment_pre
)

print(f"Original p-value: {result['original_p_value']:.4f}")
print(f"Adjusted p-value: {result['adjusted_p_value']:.4f}")
print(f"Variance reduction: {result['variance_reduction_pct']:.1f}%")
print(f"Power improvement: {result['power_improvement']:.2f}x")
```

### Stratified Sampling

Ensure balance across important user segments.

```python
class StratifiedAssignment:
    """
    Assign users to experiments with stratification
    
    Ensures balanced assignment within strata (e.g., country, platform)
    """
    
    def __init__(self, num_variants=2):
        self.num_variants = num_variants
        self.strata_counters = {}  # stratum → variant counts
    
    def assign_variant(self, user_id: str, stratum: str) -> int:
        """
        Assign user to variant, ensuring balance within stratum
        
        Args:
            user_id: User identifier
            stratum: Stratum key (e.g., "US_iOS", "UK_Android")
        
        Returns:
            Variant index
        """
        # Initialize stratum if new
        if stratum not in self.strata_counters:
            self.strata_counters[stratum] = [0] * self.num_variants
        
        # Hash-based assignment (deterministic)
        import hashlib
        hash_input = f"{user_id}:{stratum}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        variant = hash_value % self.num_variants
        
        # Update counter
        self.strata_counters[stratum][variant] += 1
        
        return variant
    
    def get_balance_report(self) -> dict:
        """Check balance within each stratum"""
        report = {}
        
        for stratum, counts in self.strata_counters.items():
            total = sum(counts)
            proportions = [c / total for c in counts]
            
            # Check if balanced (each variant should have ~1/num_variants)
            expected = 1 / self.num_variants
            max_deviation = max(abs(p - expected) for p in proportions)
            
            report[stratum] = {
                'counts': counts,
                'proportions': proportions,
                'max_deviation': max_deviation,
                'balanced': max_deviation < 0.05  # Within 5% of expected
            }
        
        return report

# Usage
stratified = StratifiedAssignment(num_variants=2)

# Simulate user assignments
for i in range(10000):
    user_id = f"user_{i}"
    
    # Assign stratum based on user
    if i % 3 == 0:
        stratum = "US_iOS"
    elif i % 3 == 1:
        stratum = "US_Android"
    else:
        stratum = "UK_iOS"
    
    variant = stratified.assign_variant(user_id, stratum)

# Check balance
balance = stratified.get_balance_report()
for stratum, stats in balance.items():
    print(f"{stratum}: {stats['counts']}, balanced={stats['balanced']}")
```

---

## Multiple Testing Correction

When running many experiments, control family-wise error rate.

### Bonferroni Correction

```python
def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Bonferroni correction for multiple comparisons
    
    Adjusted alpha = alpha / num_tests
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate
    
    Returns:
        List of booleans (True = significant after correction)
    """
    num_tests = len(p_values)
    adjusted_alpha = alpha / num_tests
    
    return [p < adjusted_alpha for p in p_values]

# Example: Testing 10 variants
p_values = [0.04, 0.06, 0.03, 0.08, 0.02, 0.09, 0.07, 0.05, 0.01, 0.10]

significant_uncorrected = [p < 0.05 for p in p_values]
significant_corrected = bonferroni_correction(p_values, alpha=0.05)

print(f"Significant (uncorrected): {sum(significant_uncorrected)} / {len(p_values)}")
print(f"Significant (Bonferroni): {sum(significant_corrected)} / {len(p_values)}")
```

### False Discovery Rate (FDR) - Benjamini-Hochberg

```python
def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Benjamini-Hochberg procedure for FDR control
    
    Less conservative than Bonferroni
    
    Args:
        p_values: List of p-values
        alpha: Desired FDR level
    
    Returns:
        List of booleans (True = significant)
    """
    num_tests = len(p_values)
    
    # Sort p-values with original indices
    indexed_p_values = [(p, i) for i, p in enumerate(p_values)]
    indexed_p_values.sort()
    
    # Find largest k such that p[k] <= (k+1)/m * alpha
    significant_indices = set()
    
    for k in range(num_tests - 1, -1, -1):
        p_value, original_idx = indexed_p_values[k]
        threshold = (k + 1) / num_tests * alpha
        
        if p_value <= threshold:
            # Mark this and all smaller p-values as significant
            for j in range(k + 1):
                significant_indices.add(indexed_p_values[j][1])
            break
    
    # Create result list
    return [i in significant_indices for i in range(num_tests)]

# Compare to Bonferroni
fdr_significant = benjamini_hochberg(p_values, alpha=0.05)

print(f"Significant (FDR): {sum(fdr_significant)} / {len(p_values)}")
```

---

## Layered Experiments

Run multiple experiments simultaneously on orthogonal layers.

```python
class ExperimentLayer:
    """
    Single experiment layer
    """
    
    def __init__(self, layer_id: str, experiments: List[str]):
        self.layer_id = layer_id
        self.experiments = experiments
        self.num_experiments = len(experiments)
    
    def assign_experiment(self, user_id: str) -> str:
        """Assign user to one experiment in this layer"""
        import hashlib
        
        hash_input = f"{user_id}:{self.layer_id}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        
        experiment_idx = hash_value % self.num_experiments
        return self.experiments[experiment_idx]

class LayeredExperimentPlatform:
    """
    Platform supporting layered experiments
    
    Layers should be independent (orthogonal)
    """
    
    def __init__(self):
        self.layers = {}
    
    def add_layer(self, layer_id: str, experiments: List[str]):
        """Add experiment layer"""
        self.layers[layer_id] = ExperimentLayer(layer_id, experiments)
    
    def assign_user(self, user_id: str) -> dict:
        """
        Assign user to experiments across all layers
        
        Returns:
            Dict mapping layer_id → experiment_id
        """
        assignments = {}
        
        for layer_id, layer in self.layers.items():
            experiment = layer.assign_experiment(user_id)
            assignments[layer_id] = experiment
        
        return assignments

# Usage
platform = LayeredExperimentPlatform()

# Layer 1: Ranking algorithm tests
platform.add_layer(
    'ranking',
    ['ranking_baseline', 'ranking_ml_v1', 'ranking_ml_v2']
)

# Layer 2: UI tests (independent of ranking)
platform.add_layer(
    'ui',
    ['ui_old', 'ui_new_blue', 'ui_new_green']
)

# Layer 3: Recommendation tests
platform.add_layer(
    'recommendations',
    ['recs_baseline', 'recs_personalized']
)

# Assign user to experiments
user_experiments = platform.assign_user('user_12345')
print(f"User assigned to:")
for layer, experiment in user_experiments.items():
    print(f"  {layer}: {experiment}")

# User gets combination like:
# ranking: ranking_ml_v2
# ui: ui_new_blue
# recommendations: recs_personalized
```

---

## Airbnb's Experiment Framework

Real-world example of production experimentation.

**Key Components:**

1. **ERF (Experiment Reporting Framework)**
   - Centralized metric definitions
   - Automated metric computation
   - Standardized reporting

2. **CUPED for Variance Reduction**
   - Uses pre-experiment booking history
   - 50%+ variance reduction on key metrics
   - Dramatically reduces required sample size

3. **Quasi-experiments**
   - When randomization not possible (e.g., pricing tests)
   - Difference-in-differences analysis
   - Synthetic control methods

4. **Interference Handling**
   - Network effects (one user's treatment affects others)
   - Cluster randomization (randomize at city/market level)
   - Ego-cluster randomization

**Metrics Hierarchy:**

```
Primary Metrics (move-the-needle)
├── Bookings
├── Revenue
└── Guest Satisfaction

Secondary Metrics (understand mechanism)
├── Search engagement
├── Listing views
└── Message rate

Guardrail Metrics (protect)
├── Host satisfaction
├── Cancellation rate
└── Customer support tickets
```

---

## Key Takeaways

✅ **Randomization** via consistent hashing ensures unbiased assignment  
✅ **Statistical rigor** prevents false positives, require p < 0.05 and sufficient power  
✅ **Sample size** calculation upfront prevents underpowered experiments  
✅ **Guardrail metrics** protect against shipping harmful changes  
✅ **Real-time monitoring** enables early stopping for clear wins/losses  
✅ **Sequential testing** allows stopping early while controlling error rates  
✅ **Multi-armed bandits** dynamically optimize traffic allocation  
✅ **CUPED** reduces variance using pre-experiment data → smaller samples needed  
✅ **Stratified sampling** ensures balance across key user segments  
✅ **Multiple testing corrections** control error rates when running many experiments  
✅ **Layered experiments** increase experimentation velocity without conflicts  
✅ **Long-term holdouts** measure sustained impact vs novelty effects  

---

**Originally published at:** [arunbaby.com/ml-system-design/0004-ab-testing-systems](https://www.arunbaby.com/ml-system-design/0004-ab-testing-systems/)

*If you found this helpful, consider sharing it with others who might benefit.*

