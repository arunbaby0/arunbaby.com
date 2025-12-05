---
title: "Model Interpretability and Explainability (XAI)"
day: 39
collection: ml_system_design
categories:
  - ml_system_design
tags:
  - shap
  - lime
  - explainability
  - feature-importance
  - integrated-gradients
subdomain: "ML Ops"
tech_stack: [SHAP, LIME, Captum, Alibi]
scale: "Production Inference"
companies: [FICO, Google, Wells Fargo, Healthcare AI]
---

**"Trust, but verify. Why did the model say No?"**

## 1. The Black Box Problem

As models become more complex (Deep Learning, Ensembles), they become less interpretable.
-   **Linear Regression:** $y = 2x + 3$. We know exactly how $x$ affects $y$.
-   **ResNet-50:** 25 million parameters. Why is this image a "cat"?

**Why Interpretability Matters:**
1.  **Trust:** Users won't adopt AI if they don't understand it (e.g., doctors).
2.  **Debugging:** Why is the model failing on this specific edge case?
3.  **Regulation:** GDPR "Right to Explanation", Equal Credit Opportunity Act (ECOA).
4.  **Bias Detection:** Is the model relying on "Gender" or "Race" proxies?

## 2. Taxonomy of Explainability

### 1. Intrinsic vs. Post-hoc
-   **Intrinsic:** The model is self-explanatory (Decision Trees, Linear Models).
-   **Post-hoc:** A separate method explains a trained black-box model (SHAP, LIME).

### 2. Global vs. Local
-   **Global:** How does the model work overall? (Feature Importance).
-   **Local:** Why did the model make *this specific prediction*? (Individual SHAP values).

### 3. Model-Agnostic vs. Model-Specific
-   **Agnostic:** Works on any model (treats model as function $f(x)$).
-   **Specific:** Uses internal gradients or structure (Integrated Gradients, TreeSHAP).

## 3. Global Interpretability Techniques

### 1. Feature Importance (Permutation Importance)
**Idea:** Randomly shuffle a feature column. If model performance drops significantly, that feature is important.

**Algorithm:**
1.  Measure baseline accuracy.
2.  For each feature $j$:
    -   Shuffle column $j$ in the validation set.
    -   Measure new accuracy.
    -   Importance = Baseline - New Accuracy.

**Pros:** Model-agnostic, intuitive.
**Cons:** Ignores feature interactions. If features are correlated, shuffling one creates unrealistic data points.

### 2. Partial Dependence Plots (PDP)
**Idea:** Plot the average prediction as a function of one feature, marginalizing over all others.

$$PDP(x_j) = E_{x_{-j}}[f(x_j, x_{-j})]$$

**Pros:** Shows relationship (linear, quadratic, etc.).
**Cons:** Assumes independence between features.

## 4. Local Interpretability: LIME

**LIME (Local Interpretable Model-agnostic Explanations)**

**Intuition:**
The decision boundary of a neural net is complex globally, but **locally linear**.
LIME fits a simple linear model *around* the instance we want to explain.

**Algorithm:**
1.  Select instance $x$ to explain.
2.  Generate perturbed samples around $x$ (add noise).
3.  Get predictions for these samples using the black-box model.
4.  Weight samples by proximity to $x$.
5.  Train a weighted linear regression (Lasso) on these samples.
6.  The coefficients of the linear model are the explanations.

**Pros:** Works on images, text, tabular.
**Cons:** Unstable (different runs give different explanations). Sampling is hard in high dimensions.

## 5. Local Interpretability: SHAP (Shapley Additive Explanations)

**Intuition:** Game Theory.
Features are "players" in a cooperative game to produce the prediction. How do we fairly distribute the "payout" (prediction) among players?

**Shapley Value:**
The average marginal contribution of a feature value across all possible coalitions.

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} [f(S \cup \{j\}) - f(S)]$$

**Properties:**
1.  **Additivity:** Sum of SHAP values + Base Value = Prediction.
2.  **Consistency:** If a model changes so that a feature has higher impact, its SHAP value doesn't decrease.

**TreeSHAP:**
-   Fast algorithm for Tree ensembles (XGBoost, LightGBM).
-   $O(T \cdot L \cdot D^2)$ instead of exponential.

**Code Example:**
```python
import shap
import xgboost

# Train model
model = xgboost.XGBClassifier().fit(X_train, y_train)

# Explain
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test)
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
```

## 6. Deep Learning Interpretability

### 1. Saliency Maps (Gradients)
**Idea:** Compute gradient of output class score with respect to input image pixels.
$$M = \left| \frac{\partial y}{\partial x} \right|$$
-   High gradient = Changing this pixel changes the prediction a lot.

**Problem:** Gradients can be noisy (shattered gradients).

### 2. Integrated Gradients (IG)
**Idea:** Accumulate gradients along a path from a "baseline" (black image) to the input.
$$IG_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

**Pros:** Satisfies "Completeness" axiom (sum of attributions = difference in output).
**Cons:** Computationally expensive (requires 50-100 forward/backward passes).

### 3. Grad-CAM (Class Activation Mapping)
**Idea:** Use the feature maps of the last convolutional layer.
-   Compute gradients of class score w.r.t feature maps.
-   Global Average Pool gradients to get weights.
-   Weighted sum of feature maps = Heatmap.

**Pros:** High-level semantic explanation ("looking at the dog's ear").
**Cons:** Low resolution (limited by feature map size).

## 7. System Design: Explainability Service

**Scenario:** A bank uses an ML model for loan approval. Every decision must be explainable in real-time for the UI and stored for audit.

**Requirements:**
-   **Latency:** < 200ms for inference + explanation.
-   **Throughput:** 1000 RPS.
-   **Storage:** Store explanations for 7 years.

**Architecture:**

1.  **Inference Service:**
    -   Receives request.
    -   Runs model (XGBoost).
    -   Returns prediction.
    -   Asynchronously pushes (Input, Prediction) to Kafka.

2.  **Explanation Service (Consumer):**
    -   Consumes from Kafka.
    -   Runs **TreeSHAP** (fast enough for tabular).
    -   For Deep Learning, might use a simplified proxy model or pre-computed clusters.
    -   Stores SHAP values in **Cassandra** (Time-series/Wide-column).

3.  **API Gateway:**
    -   UI requests explanation by TransactionID.
    -   Service fetches from Cassandra.

**Optimization for Real-Time:**
-   **Background Calculation:** If explanation isn't needed *immediately* for the user, compute it offline.
-   **Approximation:** Use "FastTreeSHAP" or sample fewer coalitions.
-   **Caching:** Cache explanations for similar inputs.

## 8. Case Study: Debugging a Computer Vision Model

**Problem:** A "Wolf vs Husky" classifier has 99% accuracy but fails in the wild.

**Investigation:**
1.  Run LIME/Grad-CAM on the training set.
2.  **Discovery:** The model is looking at the **snow** in the background, not the animal.
    -   All Wolf photos in training data had snow.
    -   All Husky photos were indoors/grass.
3.  **Conclusion:** The model learned a "Snow Detector".

**Fix:**
-   Collect Wolf photos without snow.
-   Use data augmentation (background removal/swapping).
-   Penalize the model if it attends to the background (Attention Regularization).

## 9. Deep Dive: Counterfactual Explanations

**Idea:** "What is the smallest change to the input that would change the prediction?"
-   "You were denied the loan. If your income was $5000 higher, you would have been approved."

**Optimization Problem:**
Find $x'$ such that:
1.  $f(x') \neq f(x)$ (Prediction changes)
2.  $d(x, x')$ is minimized (Smallest change)
3.  $x'$ is plausible (Manifold constraint - don't suggest "Age = 200").

**Algorithm (DiCE - Diverse Counterfactual Explanations):**
-   Uses gradient descent to modify input $x$ to minimize loss + distance.

## 10. Deep Dive: Anchors

**Problem with LIME:** Linear explanations can be misleading if the boundary is highly non-linear.

**Anchors:** High-precision rules.
-   "If Income > 50k AND Debt < 5k, prediction is ALWAYS Approved."
-   Provides a "coverage" metric (how much of the input space does this rule cover?).

## 11. Ethical Considerations

**1. The Illusion of Explanability:**
-   Post-hoc explanations (LIME) are approximations. They might be wrong.
-   **Risk:** Users trusting a wrong explanation.

**2. Adversarial Attacks on Explanations:**
-   Attackers can manipulate the input such that the prediction stays the same (e.g., "Reject"), but the explanation changes (e.g., "Because of Age" -> "Because of Income").
-   **Fairwashing:** Making a biased model look fair by manipulating the explanation.

**3. Privacy Leakage:**
-   Explanations can leak information about the training data (Membership Inference).

## 12. Deep Dive: The Math of Shapley Values

Why is SHAP so popular? It's the *only* method that satisfies three key axioms:

**1. Local Accuracy (Efficiency):**
The sum of feature attributions must equal the difference between the prediction and the baseline.
$$ \sum_{j=1}^M \phi_j = f(x) - E[f(x)] $$

**2. Missingness:**
If a feature is missing (or has no effect), its attribution must be zero.
$$ x_j' = 0 \implies \phi_j = 0 $$

**3. Consistency (Monotonicity):**
If a model changes such that a feature's contribution increases (or stays same) regardless of other features, its Shapley value should not decrease.

**Proof Sketch:**
Shapley proved in 1953 that the weighted average of marginal contributions is the unique solution.
$$ \phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} [f(S \cup \{j\}) - f(S)] $$

**Interpretation:**
Imagine features entering a room one by one.
-   Room is empty: Prediction = Baseline.
-   Feature A enters: Prediction changes by $\Delta_A$.
-   Feature B enters: Prediction changes by $\Delta_B$.
-   Since order matters (interactions), we average over all possible entry orders ($N!$).

## 13. Deep Dive: Integrated Gradients (Path Integral)

For Deep Networks, computing Shapley values is too expensive ($2^N$ subsets). Integrated Gradients is a continuous approximation.

**Axiom: Sensitivity(a)**
If input $x$ and baseline $x'$ differ in one feature $i$ and have different predictions, then feature $i$ should have non-zero attribution.
-   Gradients violate this (e.g., ReLU saturation: gradient is 0 even if input > 0).

**Path Integral Formulation:**
We integrate gradients along a straight line path from baseline $x'$ to input $x$.
$$ IG_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha $$

**Implementation:**
Approximated by Riemann Sum:
1.  Generate $m$ steps (e.g., 50) between baseline and input.
    -   $x_k = x' + \frac{k}{m} (x - x')$.
2.  Compute gradients at each step.
3.  Average the gradients.
4.  Multiply by $(x_i - x'_i)$.

**Code:**
```python
def integrated_gradients(input_tensor, baseline, model, steps=50):
    # 1. Generate path
    alphas = torch.linspace(0, 1, steps)
    path = baseline + alphas[:, None] * (input_tensor - baseline)
    path.requires_grad = True
    
    # 2. Compute gradients
    preds = model(path)
    grads = torch.autograd.grad(torch.unbind(preds), path)[0]
    
    # 3. Average and multiply
    avg_grads = torch.mean(grads, dim=0)
    ig = (input_tensor - baseline) * avg_grads
    return ig
```

## 14. Deep Dive: TCAV (Testing with Concept Activation Vectors)

Most methods explain predictions using *input features* (pixels). TCAV explains using *concepts* (e.g., "stripes", "pointed ears").

**Idea:**
1.  Define a concept (e.g., "Stripes") by collecting examples (Zebras) and counter-examples.
2.  Train a linear classifier (CAV) to separate activations of "Stripes" vs "Random" in a hidden layer.
3.  The normal vector to the decision boundary is the **Concept Activation Vector ($v_C$)**.
4.  Compute directional derivative of prediction $f(x)$ along $v_C$.
    $$ S_{C, k}(x) = \nabla h_k(x) \cdot v_C $$
5.  If positive, the concept "Stripes" positively influenced the class "Zebra".

**Benefit:**
Allows asking: "Did the model predict 'Doctor' because of 'Male' concept?" (Bias detection).

## 15. System Design: Feature Store for Explainability

To explain a prediction made yesterday, we need the exact feature values from yesterday.

**Components:**
1.  **Online Feature Store (Redis):** Low latency for inference.
2.  **Offline Feature Store (Iceberg/Parquet):** Historical data for training.
3.  **Explainability Store:**
    -   Must link `PredictionID` -> `FeatureSnapshot`.
    -   **Option A:** Log payload to S3 (cheap, high latency).
    -   **Option B:** Time-travel query on Feature Store (complex).
    -   **Option C:** Log payload to Kafka -> ClickHouse (fast analytics).

**Architecture for Compliance (GDPR):**
-   User asks "Why?".
-   Query ClickHouse for `PredictionID`.
-   Retrieve `Features` + `ModelVersion`.
-   Load `ModelArtifact` from S3 (if not cached).
-   Run `KernelSHAP` (model-agnostic) on-the-fly.
-   Return JSON explanation.

## 16. Advanced: Mechanistic Interpretability

The frontier of XAI is understanding the *circuits* inside the model.

**Induction Heads (Anthropic):**
-   Found specific attention heads in Transformers responsible for "copying" previous tokens.
-   Explains in-context learning capabilities.

**Polysemantic Neurons:**
-   One neuron might activate for "cats" AND "cars".
-   **Superposition Hypothesis:** Models pack more features than neurons by using non-orthogonal directions.
-   **Sparse Autoencoders:** Used to disentangle these polysemantic neurons into interpretable features.

## 17. Case Study: Credit Risk Model (Regulatory Compliance)

**Constraint:**
-   US regulations (ECOA) require "Adverse Action Notices".
-   "We denied your loan because: 1. Income too low, 2. Debt too high."
-   Must be accurate and consistent.

**Solution:**
-   Use **Monotonicity Constraints** in XGBoost.
    -   Force "Income" to have positive relationship with score.
    -   Prevents counter-intuitive explanations ("Denied because income is too high").
-   Use **Global SHAP** to select top-k exclusion codes.
-   **Human-in-the-loop:** Review explanations for a sample of denials.

-   **Human-in-the-loop:** Review explanations for a sample of denials.

## 18. Deep Dive: Evaluating Explanations

How do we know if an explanation is "good"?

**1. Faithfulness (Fidelity):**
Does the explanation accurately reflect the model's logic?
-   **Metric:** **Insertion/Deletion Game**.
    -   Sort features by importance.
    -   Delete top-k features.
    -   Measure drop in prediction score.
    -   Steeper drop = More faithful.

**2. Plausibility (Human-interpretability):**
Does the explanation make sense to a human?
-   **Metric:** User studies. "Does this heatmap help you identify the class?"
-   **Conflict:** A faithful explanation (e.g., "Edge #405 activated") might not be plausible.

**3. Robustness (Stability):**
Do similar inputs yield similar explanations?
-   **Metric:** **Local Lipschitz Constant**.
    -   $\max \frac{||E(x) - E(x')||}{||x - x'||}$ for small perturbation.

## 19. Deep Dive: Adversarial Attacks on XAI

Attackers can fool explainers without fooling the model.

**Scaffolding Attack (Slack et al., 2020):**
-   Create a biased model $f_{biased}$ (e.g., rejects based on Race).
-   Create an unbiased model $f_{fair}$.
-   Create an **Out-of-Distribution (OOD) Detector**.
-   **Attack Model:**
    -   If input is OOD (which LIME/SHAP perturbations are!), use $f_{fair}$.
    -   If input is real distribution, use $f_{biased}$.
-   **Result:** LIME/SHAP sees the fair model. Users see the biased decisions.

**Mitigation:**
-   Use **Distribution-Aware** sampling for LIME/SHAP.
-   Audit the model on the real distribution, not just perturbed samples.

## 20. Deep Dive: Surrogate Models

When the black box is too complex, train a "Surrogate" that mimics it.

**Global Surrogate:**
-   Train a Decision Tree to predict the *output* of the Black Box.
-   **Pros:** Gives a global view of the logic.
-   **Cons:** Accuracy trade-off. The tree might be only 80% faithful to the Black Box.

**Local Surrogate (LIME):**
-   Train a linear model only on the neighborhood of $x$.
-   **Pros:** High fidelity locally.
-   **Cons:** No global view.

**Rule Extraction (Anchors):**
-   Find a rule $A$ such that $P(\text{Precision}(A) > 0.95)$ is high.
-   "If FICO > 700 and Income > 50k, prediction is Approved with 95% confidence."

## 21. System Design: Monitoring Explainability Drift

**Problem:** The model's logic might change over time (Concept Drift), or the data might shift (Covariate Shift), invalidating old explanations.

**Architecture:**
1.  **Explanation Logger:** Log SHAP values for every inference.
2.  **Drift Detection:**
    -   Compute distribution of SHAP values for top features.
    -   Compare $P(\phi_{age})_{today}$ vs $P(\phi_{age})_{training}$.
    -   Use **KL Divergence** or **KS Test**.
3.  **Alerting:**
    -   "Feature 'Income' is contributing 2x more to decisions today than last week."
    -   Indicates model might be latching onto a new correlation.

## 22. Case Study: Healthcare Diagnosis (Doctor-in-the-Loop)

**Scenario:** AI predicts Sepsis risk in ICU.

**Challenge:**
-   Doctors ignore "Black Box" alerts (Alert Fatigue).
-   Need "Why?" to verify.

**Solution:**
-   **Counterfactuals:** "Risk is 80%. If Lactate was < 2.0, Risk would be 40%."
-   **Similar Prototypes:** "This patient looks like Patient X and Patient Y who had Sepsis." (Case-based reasoning).
-   **Uncertainty Quantification:** "Risk is 80% $\pm$ 10%." (Aleatoric vs Epistemic uncertainty).

## 23. Interview Questions

**Q1: SHAP vs LIME?**
*Answer:*
-   **SHAP:** Theoretically optimal (Shapley values), consistent, global & local consistency. Slower.
-   **LIME:** Faster, intuitive (linear), but unstable and lacks theoretical guarantees.

**Q2: How to explain a CNN?**
*Answer:* Grad-CAM for high-level heatmap, Integrated Gradients for pixel-level attribution.

**Q3: What is the trade-off between Accuracy and Interpretability?**
*Answer:* Generally, simpler models (Linear, Trees) are interpretable but less accurate. Deep Nets are accurate but black boxes. XAI aims to bridge this gap.

**Q4: How to detect feature interaction?**
*Answer:* SHAP Interaction Values (generalization of Shapley values to pairs). Or Friedman's H-statistic.

## 24. Common Mistakes

1.  **Confusing Correlation with Causation:** Feature importance says "this feature is useful", not "this feature causes the outcome".
2.  **Ignoring Multicollinearity:** If two features are correlated, SHAP splits the credit. Dropping one might not drop performance, but it changes the explanation.
3.  **Over-trusting Saliency Maps:** Some saliency methods (like Guided Backprop) act like edge detectors and don't actually depend on the model parameters (Sanity Checks for Saliency Maps paper).

## 25. Further Reading

1.  **"Why Should I Trust You?" (Ribeiro et al., 2016):** The LIME paper.
2.  **"A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017):** The SHAP paper.
3.  **"Axiomatic Attribution for Deep Networks" (Sundararajan et al., 2017):** Integrated Gradients.
4.  **"Stop Explaining Black Box Machine Learning Models for High Stakes Decisions" (Rudin):** Argument for intrinsic interpretability.

## 26. Conclusion

Model Interpretability is no longer a nice-to-have; it's a requirement for responsible AI. While techniques like SHAP and LIME provide powerful tools to peek inside the black box, they are not silver bullets. Engineers must understand the limitations of these methods and design systems that prioritize transparency from the ground up. As we move to larger models (LLMs), explainability shifts from "feature attribution" to "chain of thought" and "mechanistic interpretability".

## 27. Summary

| Method | Type | Best For | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Feature Importance** | Global | Trees | Fast | Ignores interactions |
| **LIME** | Local | Any | Intuitive | Unstable |
| **SHAP** | Local/Global | Any | Consistent | Slow (exact) |
| **Integrated Gradients** | Local | Deep Nets | Axiomatic | Expensive |
| **Grad-CAM** | Local | CNNs | Visual | Low res |

---

**Originally published at:** [arunbaby.com/ml-system-design/0039-model-interpretability](https://www.arunbaby.com/ml-system-design/0039-model-interpretability/)
