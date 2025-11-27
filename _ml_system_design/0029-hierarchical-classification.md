---
title: "Hierarchical Classification Systems"
day: 29
collection: ml_system_design
categories:
  - ml_system_design
tags:
  - classification
  - taxonomy
  - hierarchical
  - multi-label
subdomain: "Classification & Taxonomy"
tech_stack: [TensorFlow, PyTorch, FAISS]
scale: "Billions of items, Millions of categories"
companies: [Amazon, Google Shopping, eBay, Wikipedia]
related_dsa_day: 29
related_ml_day: 29
related_speech_day: 29
---

**"Organizing the world's information into a structured hierarchy."**

## 1. What is Hierarchical Classification?

Hierarchical classification is the task of assigning an item to one or more nodes in a **taxonomy tree**.

**Example: Product Categorization**
```
Electronics
├── Computers
│   ├── Laptops
│   │   ├── Gaming Laptops
│   │   └── Business Laptops
│   └── Desktops
└── Mobile Devices
    ├── Smartphones
    └── Tablets
```

**Problem:** Given a product description "Dell XPS 15 with RTX 3050", classify it into:
- Electronics > Computers > Laptops > Business Laptops (or Gaming Laptops?)

**Challenges:**
1. **Large Taxonomies:** Amazon has 30,000+ categories.
2. **Multi-path:** An item can belong to multiple leaf nodes (e.g., "Wireless Gaming Mouse" → Computers/Accessories AND Gaming/Peripherals).
3. **Imbalanced Data:** "Electronics" has millions of items, "Vintage Typewriters" has 100.
4. **Hierarchy Violations:** A model might predict "Gaming Laptop" without predicting "Laptop" (parent).

## 2. Flat vs. Hierarchical Classification

| Aspect | Flat Classification | Hierarchical Classification |
|:---|:---|:---|
| **Model** | Single multi-class classifier | Tree-structured classifiers |
| **Predictions** | One label | Path in tree |
| **Training** | One model | Multiple models (global or local) |
| **Hierarchy** | Ignored | Exploited |
| **Example** | CIFAR-10 (10 classes) | ImageNet (1000 classes in hierarchy) |

**Why Hierarchical?**
- **Scalability:** Training a single model with 30,000 classes is intractable.
- **Interpretability:** Users navigate taxonomies ("Show me all Electronics > Computers").
- **Zero-shot:** New subcategories can be added without retraining the entire model.

## 3. Hierarchical Classification Approaches

### Approach 1: Global Classifier (Flat with Post-Processing)

Train a **single multi-class classifier** predicting all leaf nodes, then use the hierarchy to ensure consistency.

**Model:**
```python
# Predict all 30,000 leaf categories
logits = model(input)  # Shape: [batch, 30000]
probs = softmax(logits)

# Post-process: ensure parent probabilities >= child probabilities
for node in taxonomy.postorder():
    if node.children:
        node.prob = max(node.prob, max(child.prob for child in node.children))
```

**Pros:**
- Simple: One model.
- High accuracy if data is sufficient.

**Cons:**
- **Class Imbalance:** Popular categories dominate.
- **No hierarchy exploitation during training.**

### Approach 2: Local Classifiers Per Node (LCN)

Train a **separate classifier at each internal node** to choose among its children.

**Example:**
```
Root Classifier: Electronics vs. Clothing vs. Books
  ├─ Electronics Classifier: Computers vs. Mobile Devices
  │   ├─ Computers Classifier: Laptops vs. Desktops
```

**Inference:**
```python
node = root
while not node.is_leaf():
    probs = node.classifier(input)
    node = node.children[argmax(probs)]
return node
```

**Pros:**
- **Balanced:** Each classifier handles a small, focused problem.
- **Modular:** Can update one classifier without retraining others.

**Cons:**
- **Error Propagation:** If the root classifier is wrong, the entire path is wrong.
- **Many Models:** Need to train and deploy K models (where K = number of internal nodes).

### Approach 3: End-to-End Hierarchical Model

Use a **shared encoder** with **hierarchical output heads**.

**Architecture:**
```python
class HierarchicalClassifier(nn.Module):
    def __init__(self, taxonomy):
        super().__init__()
        self.encoder = ResNet()  # Shared
        self.heads = nn.ModuleDict({
            node.id: nn.Linear(2048, len(node.children))
            for node in taxonomy.internal_nodes()
        })
    
    def forward(self, x):
        features = self.encoder(x)
        outputs = {}
        for node_id, head in self.heads.items():
            outputs[node_id] = head(features)
        return outputs
```

**Loss:**
```python
total_loss = 0
for node in taxonomy.internal_nodes():
    if node in ground_truth_path:
        target = ground_truth_path[node].child_index
        total_loss += cross_entropy(outputs[node.id], target)
```

**Pros:**
- **Shared Representations:** Lower layers learn general features.
- **End-to-End Training:** Optimizes the entire path jointly.

**Cons:**
- **Complex:** Harder to debug and tune.
- **Memory:** All heads must fit in GPU memory.

## 4. Handling Multi-Label Hierarchy

Some items belong to **multiple paths** in the tree.

**Example:**
"Logitech Wireless Gaming Mouse"
- Path 1: Electronics > Computers > Accessories > Mouse
- Path 2: Electronics > Gaming > Peripherals > Mouse

**Approach: Multi-Task Learning**
- Treat each path as a separate task.
- Loss = Sum of losses for all valid paths.
```python
loss = sum(path_loss(model(x), path) for path in ground_truth_paths)
```

## 5. Hierarchy-Aware Loss Functions

### Loss 1: Hierarchical Softmax

Instead of standard softmax over 30,000 classes, factorize:
\\[
P(\text{Leaf} | x) = P(\text{Root} \to \text{Child}_1 | x) \times P(\text{Child}_1 \to \text{Child}_2 | x) \times \ldots
\\]

**Benefit:** Reduces computation from \\(O(K)\\) to \\(O(\log K)\\) where K is number of classes.

### Loss 2: Hierarchical Cross-Entropy (H-Loss)

Penalize mistakes based on the distance in the tree.
\\[
\mathcal{L} = \sum_{i=1}^{D} \alpha_i \cdot \text{CE}(\text{pred}_i, \text{true}_i)
\\]
where \\(D\\) is the depth and \\(\alpha_i\\) increases with depth (leaf nodes weighted more).

**Intuition:** Mistaking "Gaming Laptop" for "Business Laptop" (siblings) is less bad than mistaking it for "Tablet" (cousin).

### Loss 3: Symmetric KL Divergence

Encourage the model to predict **ancestor probabilities >= descendant probabilities**.
\\[
\mathcal{L}_{\text{consistency}} = \sum_{\text{parent, child}} \max(0, P(\text{child}) - P(\text{parent}))
\\]

## Deep Dive: Amazon's Product Taxonomy

Amazon has a **forest of taxonomies** (one per marketplace).
**Challenge:** Items listed in multiple marketplaces need consistent categorization.

**Solution: Transfer Learning**
1. Train a base model on US taxonomy (most data).
2. Fine-tune on UK, Japan, India taxonomies.
3. Use **adapter layers** to handle marketplace-specific categories.

**Scale:**
- **Items:** 350M+
- **Categories:** 30,000+
- **Languages:** 15+

## Deep Dive: Extreme Multi-Label Classification (XML)

When the number of labels is in the millions (e.g., Wikipedia categories), standard approaches fail.

**Approaches:**
1. **Embedding-Based (AnnexML, Bonsai, Parabel):**
   - Embed labels and inputs into the same space.
   - Use ANN (Approximate Nearest Neighbors) to retrieve top-K labels.
2. **Attention Mechanisms:**
   - Label-Attention: Attend to label descriptions during encoding.
3. **Tree Pruning:**
   - Prune unlikely branches early using a lightweight model.

**Parabel Architecture:**
```
1. Build a label tree (clustering similar labels).
2. Train classifiers at each node (like LCN).
3. Beam search during inference to explore top-K branches.
```

## Deep Dive: Google's Knowledge Graph Categories

Google Search uses a hierarchical taxonomy for entities.
**Example:**
```
Thing
├── Creative Work
│   └── Movie
└── Person
    └── Actor
```

**Challenge:** New entities appear daily (new movies, new people).
**Solution:**
- **Zero-Shot Classification:** Use a text encoder (BERT) to embed the entity description.
- **Nearest Ancestor:** Find the closest category in the embedding space.

## Deep Dive: Hierarchical Multi-Task Learning (HMTL)

In HMTL, tasks are organized in a hierarchy where:
- **Lower tasks** are easier (e.g., "Is this electronics?").
- **Higher tasks** are harder (e.g., "Is this a gaming laptop?").

**Architecture:**
```
Input → Shared Encoder → Task 1 (Electronics?) ──┐
                         ├→ Task 2 (Computers?)   │
                         └→ Task 3 (Laptops?)     │
                                                   ├→ Final Prediction
```

**Loss:**
\\[
\mathcal{L} = \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_2 + \lambda_3 \mathcal{L}_3
\\]
where \\(\lambda_i\\) are learned or hand-tuned.

## Deep Dive: Taxonomy Expansion (Adding New Categories)

**Problem:** A new category "Foldable Smartphones" needs to be added.

**Approach 1: Retrain from Scratch**
- Expensive and slow.

**Approach 2: Continual Learning**
- Fine-tune the model on new data while preserving old knowledge.
- **Challenge:** Catastrophic forgetting.
- **Solution:** Elastic Weight Consolidation (EWC) or rehearsal buffers.

**Approach 3: Few-Shot Learning**
- Train a meta-learner that can adapt to new categories with < 100 examples.
- Use **Prototypical Networks** or **MAML**.

## Deep Dive: Handling Imbalanced Hierarchies

**Problem:** "Laptops" has 1M examples, "Typewriters" has 50.

**Solutions:**
1. **Class Weighting:**
   \\[
   w_i = \frac{\text{total samples}}{\text{samples in class } i}
   \\]
2. **Focal Loss:**
   \\[
   \mathcal{L} = -\alpha (1 - p_t)^\gamma \log(p_t)
   \\]
   Focuses on hard-to-classify examples.
3. **Oversampling:**
   - Augment rare classes.
4. **Hierarchical Sampling:**
   - Sample uniformly at each **level** of the tree, not uniformly across all leaves.

## Deep Dive: Evaluation Metrics

### Metric 1: Hierarchical Precision and Recall

Standard precision/recall don't account for hierarchy.
**Hierarchical Precision:**
\\[
hP = \frac{|\text{predicted path} \cap \text{true path}|}{|\text{predicted path}|}
\\]

**Example:**
- True: Electronics > Computers > Laptops
- Predicted: Electronics > Computers > Desktops
- \\( hP = \frac{2}{3} \\) (got Electronics and Computers right, but wrong at Laptops level).

### Metric 2: Tree-Induced Distance

Measure the shortest path between predicted and true leaf in the tree.
\\[
d(\text{pred}, \text{true}) = \text{depth}(\text{LCA}(\text{pred}, \text{true}))
\\]

**Example:**
- pred = Gaming Laptop, true = Business Laptop
- LCA = Laptop
- Distance = depth(Laptop) = 2 (smaller is better)

### Metric 3: F1 at Different Levels

Compute F1 separately at each level of the hierarchy.
- **Level 0 (Root):** F1 = 100% (trivial).
- **Level 1:** F1 on {Electronics, Clothing, Books}.
- **Level 2:** F1 on {Computers, Mobile, etc.}.

## Deep Dive: Active Learning for Taxonomy Labeling

**Problem:** Labeling 10M products manually is expensive.

**Solution: Active Learning**
1. Train initial model on small labeled set.
2. **Query:** Find the most uncertain samples.
   - Entropy: \\( H = -\sum_i p_i \log p_i \\)
   - Least Confident: \\( 1 - \max_i p_i \\)
3. Human labels the queried samples.
4. Retrain and repeat.

**Hierarchical Active Learning:**
- Prioritize samples where the model is uncertain **at multiple levels** of the tree.

## Deep Dive: Hierarchical Attention Networks (HAN)

**Idea:** Use attention at each level of the hierarchy to focus on relevant features.

**Architecture:**
```python
class HierarchicalAttention(nn.Module):
    def __init__(self):
        self.word_attention = Attention()
        self.sentence_attention = Attention()
        self.document_attention = Attention()
    
    def forward(self, document):
        # Level 1: Words → Sentence Representation
        sentence_reps = []
        for sentence in document:
            word_reps = [self.embed(word) for word in sentence]
            sentence_rep = self.word_attention(word_reps)
            sentence_reps.append(sentence_rep)
        
        # Level 2: Sentences → Document Representation
        doc_rep = self.sentence_attention(sentence_reps)
        
        # Level 3: Document → Category
        category = self.document_attention(doc_rep)
        return category
```

## Deep Dive: Label Embedding and Matching

**Idea:** Embed both inputs and labels into the same space.

**Training:**
```python
input_emb = encoder(product_description)  # [batch, 512]
label_embs = label_encoder(all_labels)     # [30000, 512]

# Dot product similarity
scores = input_emb @ label_embs.T  # [batch, 30000]
loss = cross_entropy(scores, target)
```

**Inference:**
```python
# Use FAISS for fast top-K retrieval
top_k_labels = faiss_index.search(input_emb, k=10)
```

**Benefit:** Decouples the number of labels from model size. Can add new labels without retraining.

## Deep Dive: Hierarchical Reinforcement Learning (HRL) for Sequential Classification

**Problem:** Some taxonomies require a sequence of decisions.

**Example:**
"Classify this support ticket"
1. Department? (Sales, Support, Engineering)
2. If Support, Priority? (Low, Medium, High)
3. If High, Assign to? (Agent 1, Agent 2, Agent 3)

**Approach: Hierarchical RL**
- **High-Level Policy:** Chooses department.
- **Low-Level Policies:** Choose priority, assign agent.
- **Reward:** +1 if ticket is resolved quickly.

## Implementation Example: PyTorch Hierarchical Classifier

```python
import torch
import torch.nn as nn

class HierarchicalModel(nn.Module):
    def __init__(self, taxonomy, embedding_dim=768):
        super().__init__()
        self.taxonomy = taxonomy
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512)
        )
        
        # One classifier per internal node
        self.classifiers = nn.ModuleDict()
        for node in taxonomy.internal_nodes():
            self.classifiers[str(node.id)] = nn.Linear(512, len(node.children))
    
    def forward(self, x, return_all=False):
        features = self.encoder(x)
        
        if return_all:
            # Return logits for all nodes (for training)
            outputs = {}
            for node_id, classifier in self.classifiers.items():
                outputs[node_id] = classifier(features)
            return outputs
        else:
            # Greedy path prediction (for inference)
            node = self.taxonomy.root
            path = [node]
            
            while not node.is_leaf():
                logits = self.classifiers[str(node.id)](features)
                child_idx = torch.argmax(logits, dim=1)
                node = node.children[child_idx.item()]
                path.append(node)
            
            return path

def hierarchical_loss(outputs, true_path):
    loss = 0
    for node in true_path:
        if not node.is_leaf():
            logits = outputs[str(node.id)]
            target = true_path.index(node.get_child_on_path(true_path))
            loss += nn.CrossEntropyLoss()(logits, torch.tensor([target]))
    return loss
```

## Top Interview Questions

**Q1: How do you handle products that fit multiple categories?**
*Answer:*
Use **multi-label classification**. Train the model to predict multiple paths. At inference, use a threshold (e.g., predict all paths with confidence > 0.3).

**Q2: What if the taxonomy is updated (categories added/removed)?**
*Answer:*
Use **modular design** (local classifiers) so you can retrain only affected nodes. Or use **label embeddings** which allow adding new categories without retraining.

**Q3: How do you ensure hierarchy consistency?**
*Answer:*
Post-process predictions to ensure \\(P(\text{child}) \leq P(\text{parent})\\). During training, add a **consistency loss** term.

**Q4: How do you deal with extreme imbalance (popular vs. rare categories)?**
*Answer:*
- Focal loss to focus on hard examples.
- Hierarchical sampling (sample uniformly at each level).
- Data augmentation for rare categories.

## Key Takeaways

1. **Hierarchy Exploitation:** Use the tree structure during both training and inference.
2. **Local vs. Global:** Trade-off between modular (easy to update) and end-to-end (higher accuracy).
3. **Multi-Label:** Real-world taxonomies often have overlapping categories.
4. **Scalability:** For millions of classes, use embedding-based retrieval (ANN).
5. **Evaluation:** Standard metrics don't account for hierarchy; use hierarchical precision/recall.

## Summary

| Aspect | Insight |
|:---|:---|
| **Approaches** | Global, Local (LCN), End-to-End Hierarchical |
| **Loss Functions** | Hierarchical Softmax, H-Loss, Consistency Loss |
| **Scalability** | Label embeddings + FAISS for extreme multi-label |
| **Evaluation** | Hierarchical precision, tree-induced distance |

---

**Originally published at:** [arunbaby.com/ml-system-design/0029-hierarchical-classification](https://www.arunbaby.com/ml-system-design/0029-hierarchical-classification/)

*If you found this helpful, consider sharing it with others who might benefit.*


