---
title: "Ranking Systems at Scale"
day: 28
collection: ml_system_design
categories:
  - ml_system_design
tags:
  - ranking
  - recommender_systems
  - ltr
  - system_design
subdomain: "Search & Ranking"
tech_stack: [XGBoost, TensorFlow Ranking, Faiss]
scale: "1B+ Items, 100ms Latency"
companies: [Google, Netflix, Amazon, TikTok]
related_dsa_day: 28
related_speech_day: 28
related_agents_day: 28
---

**How does Google search 50 billion pages in 0.1 seconds? The answer is the "Ranking Funnel".**

## 1. Problem Definition

**Goal:** Given a user query (or context) and a massive corpus of items (1B+), return the top \(K\) most relevant items sorted by relevance.

**Constraints:**
-   **Latency:** < 200ms (P99).
-   **Throughput:** 100k QPS.
-   **Freshness:** New items should be searchable within minutes.

## 2. High-Level Architecture: The Funnel

We cannot score 1 billion items with a heavy BERT model. We use a **Multi-Stage Cascade**.

```
[Corpus: 1 Billion Items]
       |
       v
[Stage 1: Retrieval / Candidate Generation] -> Selects Top 10,000
       |  (Fast, Simple, High Recall)
       v
[Stage 2: L1 Ranking (Lightweight)] --------> Selects Top 500
       |  (Two-Tower, Logistic Regression)
       v
[Stage 3: L2 Ranking (Heavyweight)] --------> Selects Top 10
       |  (BERT, LambdaMART, DLRM)
       v
[Stage 4: Re-Ranking / Blending] -----------> Final Output
          (Diversity, Business Logic)
```

## 3. Stage 1: Retrieval (Candidate Generation)

**Objective:** High Recall. Don't miss the relevant items.
**Methods:**
1.  **Inverted Index (BM25):** Standard keyword search.
2.  **Vector Search (ANN):** Embed query and items. Use Faiss/ScaNN to find nearest neighbors.
3.  **Collaborative Filtering:** "Users who bought X also bought Y".

## 4. Stage 2: L1 Ranking (The Two-Tower Model)

**Objective:** Filter 10k -> 500.
**Model:** **Two-Tower Neural Network (Bi-Encoder)**.

```
   [User Features]          [Item Features]
         |                        |
    [Deep Net]               [Deep Net]
         |                        |
    [User Vector U]          [Item Vector V]
         \                      /
          \                    /
           \                  /
            Dot Product (U . V)
                    |
                  Score
```

**Why Two-Tower?**
-   **Cacheable:** Item vectors \(V\) can be precomputed and stored in a Vector DB.
-   **Fast:** Inference is just a Dot Product.

## 5. Stage 3: L2 Ranking (The Heavy Lifter)

**Objective:** Precision. Order the top 500 perfectly.
**Model:** **Cross-Encoder (BERT)** or **LambdaMART (GBDT)**.

**Feature Interaction:**
Unlike Two-Tower, Cross-Encoders feed User and Item features *together* into the network.
-   `Input: [User_Age, Item_Price, User_History, Item_Title]`
-   The network learns interactions: "Young users like cheap items".

**Loss Functions:**
-   **Pointwise:** RMSE / LogLoss. (Predict "Will click?").
-   **Pairwise:** RankNet / LambdaRank. (Predict "Is A better than B?").
-   **Listwise:** LambdaMART / Softmax. (Optimize NDCG directly).

## 6. Stage 4: Re-Ranking & Blending

The raw scores aren't enough. We need:
1.  **Diversity:** Don't show 10 shoes. Show shoes, socks, and laces. (MMR Algorithm).
2.  **Freshness:** Boost new items.
3.  **Business Logic:** Boost sponsored items (Ads).

## Deep Dive: Approximate Nearest Neighbors (ANN)

In Stage 1 (Retrieval), we need to find the top 1000 items closest to query vector \(Q\) among 1 billion item vectors \(V\).
Brute force scan is \(O(N)\). Too slow.
We use **ANN** algorithms.

### 1. HNSW (Hierarchical Navigable Small World)
-   **Structure:** A multi-layered graph.
-   **Top Layer:** Sparse long-range links (Highways).
-   **Bottom Layer:** Dense short-range links.
-   **Search:** Start at the top, greedily move closer to the target, drop down a layer, repeat.
-   **Pros:** Extremely fast, high recall.
-   **Cons:** High memory usage (stores the graph).

### 2. IVF (Inverted File Index)
-   **Training:** Cluster the vector space into \(C\) centroids (using K-Means).
-   **Indexing:** Assign each item to its closest centroid.
-   **Search:** Find the closest centroids to \(Q\), then scan only the items in those clusters.
-   **Pros:** Low memory (can be compressed with Product Quantization).

### 3. ScaNN (Google)
-   Optimizes the Anisotropic Quantization loss.
-   State-of-the-art performance for inner-product search (which is what we need for Dot Product).

## Deep Dive: Training the Two-Tower Model

How do we train the L1 Ranker?
We don't have explicit "negative" labels (items the user *didn't* like). We only have "clicks" (positives).

**Negative Sampling:**
For every positive pair \((U, I^+)\), we sample \(K\) negative items \(I^-\).
1.  **Random Negatives:** Pick random items from the catalog. (Easy, but too easy for the model).
2.  **Hard Negatives:** Pick items that the model *thought* were good but the user didn't click. (Harder, learns better boundaries).
3.  **In-Batch Negatives:** Use the positives from *other* users in the same batch as negatives for the current user. (Efficient, GPU friendly).

**Loss Function (Softmax Cross-Entropy):**
\[ L = -\log \frac{\exp(U \cdot I^+)}{\exp(U \cdot I^+) + \sum \exp(U \cdot I^-)} \]

## Deep Dive: Learning to Rank (LTR) Loss Functions

In Stage 3 (L2 Ranking), we care about the *order*.

### 1. Pointwise (RMSE / Sigmoid)
-   Treats each item independently.
-   "Predict the probability of click for Item A".
-   **Problem:** Doesn't care if Item A > Item B, only about the absolute score.

### 2. Pairwise (RankNet / LambdaRank)
-   Takes pairs \((A, B)\) where A is clicked and B is not.
-   Minimizes loss if \(Score(A) < Score(B)\).
-   **Problem:** Optimizes the number of inversions, not the position (NDCG). An error at rank 100 is treated same as error at rank 1.

### 3. Listwise (LambdaMART)
-   LambdaMART is a Gradient Boosted Decision Tree (GBDT) method.
-   It modifies the gradients of the pairwise loss by weighting them with the change in NDCG (\(\Delta NDCG\)).
-   **Intuition:** If swapping A and B causes a huge drop in NDCG, the gradient should be huge.
-   **Status:** Still the SOTA for tabular features in L2 ranking.

## Deep Dive: DLRM (Deep Learning Recommendation Model)

Facebook open-sourced DLRM. It's the standard for L2 Ranking.

**Architecture:**
1.  **Sparse Features (Categorical):** Mapped to dense embeddings.
2.  **Dense Features (Numerical):** Processed by an MLP (Bottom MLP).
3.  **Interaction:** Dot Product between embeddings and MLP output.
4.  **Top MLP:** Takes the interaction output and predicts CTR.

**Why DLRM?**
It explicitly models the interaction between sparse features (User ID, Item ID) and dense features (Age, Price).
It is optimized for training on massive clusters (Model Parallelism for embeddings, Data Parallelism for MLPs).

## Deep Dive: The Feature Store (Feast)

In Stage 4 (Feature Fetching), we need low latency.
We cannot run SQL queries on a Data Warehouse (Snowflake/BigQuery) in real-time.

**The Feature Store Solution:**
1.  **Offline Store:** (S3/Parquet). Used for training. Contains months of history.
2.  **Online Store:** (Redis/DynamoDB). Used for inference. Contains only the *latest* values.
3.  **Consistency:** The Feature Store ensures that the Offline and Online stores are in sync (Point-in-time correctness).

**Workflow:**
-   Data Engineering pipeline computes `user_last_50_clicks`.
-   Pushes to Offline Store (for training tomorrow's model).
-   Pushes to Online Store (for serving today's traffic).

## Deep Dive: A/B Testing for Ranking

How do we know the new model is better?
We run an **Interleaving Experiment**.

**Standard A/B Test:**
-   Group A sees Model A results.
-   Group B sees Model B results.
-   Compare CTR.
-   **Problem:** High variance. Users in Group A might just be happier people.

**Interleaving:**
-   Show *every* user a mix of Model A and Model B results.
-   `Result = [A1, B1, A2, B2, ...]`
-   If user clicks `A1`, Model A gets a point.
-   If user clicks `B1`, Model B gets a point.
-   **Pros:** Removes user variance. 100x more sensitive than standard A/B tests.

## Deep Dive: Real-time Inference (Triton)

Serving a BERT model in 50ms is hard.
**NVIDIA Triton Inference Server:**
-   **Dynamic Batching:** Groups incoming requests into a batch to maximize GPU utilization.
-   **Model Ensembling:** Runs Preprocessing (Python) -> Model (TensorRT) -> Postprocessing (Python) in a single pipeline.
-   **Concurrent Execution:** Runs multiple models on the same GPU.

## Deep Dive: DLRM (Deep Learning Recommendation Model)

Facebook open-sourced DLRM. It's the standard for L2 Ranking.

**Architecture:**
1.  **Sparse Features (Categorical):** Mapped to dense embeddings.
2.  **Dense Features (Numerical):** Processed by an MLP (Bottom MLP).
3.  **Interaction:** Dot Product between embeddings and MLP output.
4.  **Top MLP:** Takes the interaction output and predicts CTR.

**Why DLRM?**
It explicitly models the interaction between sparse features (User ID, Item ID) and dense features (Age, Price).
It is optimized for training on massive clusters (Model Parallelism for embeddings, Data Parallelism for MLPs).

## Deep Dive: The Feature Store (Feast)

In Stage 4 (Feature Fetching), we need low latency.
We cannot run SQL queries on a Data Warehouse (Snowflake/BigQuery) in real-time.

**The Feature Store Solution:**
1.  **Offline Store:** (S3/Parquet). Used for training. Contains months of history.
2.  **Online Store:** (Redis/DynamoDB). Used for inference. Contains only the *latest* values.
3.  **Consistency:** The Feature Store ensures that the Offline and Online stores are in sync (Point-in-time correctness).

**Workflow:**
-   Data Engineering pipeline computes `user_last_50_clicks`.
-   Pushes to Offline Store (for training tomorrow's model).
-   Pushes to Online Store (for serving today's traffic).

## Deep Dive: A/B Testing for Ranking

How do we know the new model is better?
We run an **Interleaving Experiment**.

**Standard A/B Test:**
-   Group A sees Model A results.
-   Group B sees Model B results.
-   Compare CTR.
-   **Problem:** High variance. Users in Group A might just be happier people.

**Interleaving:**
-   Show *every* user a mix of Model A and Model B results.
-   `Result = [A1, B1, A2, B2, ...]`
-   If user clicks `A1`, Model A gets a point.
-   If user clicks `B1`, Model B gets a point.
-   **Pros:** Removes user variance. 100x more sensitive than standard A/B tests.

## Deep Dive: Real-time Inference (Triton)

Serving a BERT model in 50ms is hard.
**NVIDIA Triton Inference Server:**
-   **Dynamic Batching:** Groups incoming requests into a batch to maximize GPU utilization.
-   **Model Ensembling:** Runs Preprocessing (Python) -> Model (TensorRT) -> Postprocessing (Python) in a single pipeline.
-   **Concurrent Execution:** Runs multiple models on the same GPU.

## Deep Dive: DLRM (Deep Learning Recommendation Model)

Facebook open-sourced DLRM. It's the standard for L2 Ranking.

**Architecture:**
1.  **Sparse Features (Categorical):** Mapped to dense embeddings.
2.  **Dense Features (Numerical):** Processed by an MLP (Bottom MLP).
3.  **Interaction:** Dot Product between embeddings and MLP output.
4.  **Top MLP:** Takes the interaction output and predicts CTR.

**Why DLRM?**
It explicitly models the interaction between sparse features (User ID, Item ID) and dense features (Age, Price).
It is optimized for training on massive clusters (Model Parallelism for embeddings, Data Parallelism for MLPs).

## Deep Dive: The Feature Store (Feast)

In Stage 4 (Feature Fetching), we need low latency.
We cannot run SQL queries on a Data Warehouse (Snowflake/BigQuery) in real-time.

**The Feature Store Solution:**
1.  **Offline Store:** (S3/Parquet). Used for training. Contains months of history.
2.  **Online Store:** (Redis/DynamoDB). Used for inference. Contains only the *latest* values.
3.  **Consistency:** The Feature Store ensures that the Offline and Online stores are in sync (Point-in-time correctness).

**Workflow:**
-   Data Engineering pipeline computes `user_last_50_clicks`.
-   Pushes to Offline Store (for training tomorrow's model).
-   Pushes to Online Store (for serving today's traffic).

## Deep Dive: A/B Testing for Ranking

How do we know the new model is better?
We run an **Interleaving Experiment**.

**Standard A/B Test:**
-   Group A sees Model A results.
-   Group B sees Model B results.
-   Compare CTR.
-   **Problem:** High variance. Users in Group A might just be happier people.

**Interleaving:**
-   Show *every* user a mix of Model A and Model B results.
-   `Result = [A1, B1, A2, B2, ...]`
-   If user clicks `A1`, Model A gets a point.
-   If user clicks `B1`, Model B gets a point.
-   **Pros:** Removes user variance. 100x more sensitive than standard A/B tests.

## Deep Dive: Real-time Inference (Triton)

Serving a BERT model in 50ms is hard.
**NVIDIA Triton Inference Server:**
-   **Dynamic Batching:** Groups incoming requests into a batch to maximize GPU utilization.
-   **Model Ensembling:** Runs Preprocessing (Python) -> Model (TensorRT) -> Postprocessing (Python) in a single pipeline.
-   **Concurrent Execution:** Runs multiple models on the same GPU.

## Deep Dive: DLRM (Deep Learning Recommendation Model)

Facebook open-sourced DLRM. It's the standard for L2 Ranking.

**Architecture:**
1.  **Sparse Features (Categorical):** Mapped to dense embeddings.
2.  **Dense Features (Numerical):** Processed by an MLP (Bottom MLP).
3.  **Interaction:** Dot Product between embeddings and MLP output.
4.  **Top MLP:** Takes the interaction output and predicts CTR.

**Why DLRM?**
It explicitly models the interaction between sparse features (User ID, Item ID) and dense features (Age, Price).
It is optimized for training on massive clusters (Model Parallelism for embeddings, Data Parallelism for MLPs).

## Deep Dive: The Feature Store (Feast)

In Stage 4 (Feature Fetching), we need low latency.
We cannot run SQL queries on a Data Warehouse (Snowflake/BigQuery) in real-time.

**The Feature Store Solution:**
1.  **Offline Store:** (S3/Parquet). Used for training. Contains months of history.
2.  **Online Store:** (Redis/DynamoDB). Used for inference. Contains only the *latest* values.
3.  **Consistency:** The Feature Store ensures that the Offline and Online stores are in sync (Point-in-time correctness).

**Workflow:**
-   Data Engineering pipeline computes `user_last_50_clicks`.
-   Pushes to Offline Store (for training tomorrow's model).
-   Pushes to Online Store (for serving today's traffic).

## Deep Dive: A/B Testing for Ranking

How do we know the new model is better?
We run an **Interleaving Experiment**.

**Standard A/B Test:**
-   Group A sees Model A results.
-   Group B sees Model B results.
-   Compare CTR.
-   **Problem:** High variance. Users in Group A might just be happier people.

**Interleaving:**
-   Show *every* user a mix of Model A and Model B results.
-   `Result = [A1, B1, A2, B2, ...]`
-   If user clicks `A1`, Model A gets a point.
-   If user clicks `B1`, Model B gets a point.
-   **Pros:** Removes user variance. 100x more sensitive than standard A/B tests.

## Deep Dive: Real-time Inference (Triton)

Serving a BERT model in 50ms is hard.
**NVIDIA Triton Inference Server:**
-   **Dynamic Batching:** Groups incoming requests into a batch to maximize GPU utilization.
-   **Model Ensembling:** Runs Preprocessing (Python) -> Model (TensorRT) -> Postprocessing (Python) in a single pipeline.
-   **Concurrent Execution:** Runs multiple models on the same GPU.

## Deep Dive: DLRM (Deep Learning Recommendation Model)

Facebook open-sourced DLRM. It's the standard for L2 Ranking.

**Architecture:**
1.  **Sparse Features (Categorical):** Mapped to dense embeddings.
2.  **Dense Features (Numerical):** Processed by an MLP (Bottom MLP).
3.  **Interaction:** Dot Product between embeddings and MLP output.
4.  **Top MLP:** Takes the interaction output and predicts CTR.

**Why DLRM?**
It explicitly models the interaction between sparse features (User ID, Item ID) and dense features (Age, Price).
It is optimized for training on massive clusters (Model Parallelism for embeddings, Data Parallelism for MLPs).

## Deep Dive: The Feature Store (Feast)

In Stage 4 (Feature Fetching), we need low latency.
We cannot run SQL queries on a Data Warehouse (Snowflake/BigQuery) in real-time.

**The Feature Store Solution:**
1.  **Offline Store:** (S3/Parquet). Used for training. Contains months of history.
2.  **Online Store:** (Redis/DynamoDB). Used for inference. Contains only the *latest* values.
3.  **Consistency:** The Feature Store ensures that the Offline and Online stores are in sync (Point-in-time correctness).

**Workflow:**
-   Data Engineering pipeline computes `user_last_50_clicks`.
-   Pushes to Offline Store (for training tomorrow's model).
-   Pushes to Online Store (for serving today's traffic).

## Deep Dive: A/B Testing for Ranking

How do we know the new model is better?
We run an **Interleaving Experiment**.

**Standard A/B Test:**
-   Group A sees Model A results.
-   Group B sees Model B results.
-   Compare CTR.
-   **Problem:** High variance. Users in Group A might just be happier people.

**Interleaving:**
-   Show *every* user a mix of Model A and Model B results.
-   `Result = [A1, B1, A2, B2, ...]`
-   If user clicks `A1`, Model A gets a point.
-   If user clicks `B1`, Model B gets a point.
-   **Pros:** Removes user variance. 100x more sensitive than standard A/B tests.

## Deep Dive: Real-time Inference (Triton)

Serving a BERT model in 50ms is hard.
**NVIDIA Triton Inference Server:**
-   **Dynamic Batching:** Groups incoming requests into a batch to maximize GPU utilization.
-   **Model Ensembling:** Runs Preprocessing (Python) -> Model (TensorRT) -> Postprocessing (Python) in a single pipeline.
-   **Concurrent Execution:** Runs multiple models on the same GPU.

## Deep Dive: Graph Neural Networks (GNNs) for Ranking

Pinterest uses **PinSage** (GraphSAGE).
-   **Graph:** Users and Items are nodes. Interactions (Clicks/Pins) are edges.
-   **Idea:** An item is defined by the users who pinned it. A user is defined by the items they pinned.
-   **Convolution:** Aggregate features from neighbors (and neighbors of neighbors).
-   **Benefit:** Captures higher-order structure. "Users who bought X also bought Y" is 1-hop. GNNs capture 2-hop and 3-hop signals.

## Deep Dive: Session-Based Recommendation

What if the user is anonymous (incognito)? We have no history.
We only have the current session: `[Item A, Item B, Item C, ...]`.
**Model:** GRU4Rec (Gated Recurrent Unit).
-   Input: Sequence of item embeddings.
-   Output: Predicted next item.
-   **Mechanism:** The hidden state evolves with each click, capturing the "Current Intent" (e.g., "Shopping for shoes").
-   **Loss:** Pairwise Ranking Loss (BPR).

## Deep Dive: Calibration (Why Probabilities Matter)

In Ranking, we often just care about the *order*.
But in **Ads Ranking**, the *absolute probability* matters.
**Why?**
-   **Bid:** Advertiser pays $1.00 per click.
-   **Expected Revenue:** \( \text{Bid} \times P(\text{Click}) \).
-   If model predicts 0.2 but real probability is 0.1, we over-estimate revenue and show the wrong ad.

**Calibration Techniques:**
1.  **Platt Scaling:** Fit a Logistic Regression on the output logits.
2.  **Isotonic Regression:** Fit a non-decreasing free-form line. (More flexible, requires more data).
3.  **Reliability Diagram:** Plot "Predicted Probability" vs "Actual Frequency". Ideally, it's a diagonal line \(y=x\).

## Deep Dive: Real-time Bidding (RTB)

In Ad Tech, ranking happens in an auction.
1.  **User visits page.**
2.  **Request sent to Ad Exchange.**
3.  **Exchange asks DSPs (Demand Side Platforms):** "How much for this user?"
4.  **DSP runs Ranking Model:** Predicts CTR and CVR (Conversion Rate).
5.  **Bid Calculation:** \(\text{Bid} = \text{CVR} \times \text{Value} \times \text{Pacing\_Factor}\).
6.  **Auction:** Highest bid wins.
7.  **Latency:** All this must happen in < 100ms.

## Deep Dive: Multi-Task Learning (MTL)

We often want to optimize multiple objectives:
-   **CTR:** Click-Through Rate.
-   **CVR:** Conversion Rate (Purchase).
-   **Dwell Time:** Time spent.

**Shared-Bottom Architecture:**
-   **Shared Layers:** Learn generic features (User embedding, Item embedding).
-   **Tower Layers:** Specific to each task.
    -   CTR Tower -> Sigmoid.
    -   CVR Tower -> Sigmoid.
-   **Loss:** \( L = w_1 L_{CTR} + w_2 L_{CVR} \).
-   **Benefit:** Transfer learning. Learning to predict clicks helps predict conversions (and vice versa).

## Deep Dive: Causal Inference in Ranking

The "Cold Start" problem is real. If we never show a new item, we never get data on it.
We treat this as a **Multi-Armed Bandit** problem.

### 1. Epsilon-Greedy
-   With probability \(1-\epsilon\), show the best item (Exploit).
-   With probability \(\epsilon\), show a random item (Explore).
-   **Pros:** Simple.
-   **Cons:** "Random" items might be terrible.

### 2. Upper Confidence Bound (UCB)
-   Calculate the mean CTR \(\mu\) and the variance \(\sigma\).
-   Score = \(\mu + \alpha \cdot \sigma\).
-   **Intuition:** Boost items we are uncertain about (high variance).
-   As we get more data, \(\sigma\) decreases, and we rely more on \(\mu\).

### 3. Thompson Sampling
-   Model the CTR as a Beta distribution \(Beta(\alpha, \beta)\).
-   Sample a value from this distribution for each item. Rank by sampled value.
-   **Pros:** Mathematically optimal for minimizing regret.

## Deep Dive: Bias and Fairness in Ranking

Ranking systems can amplify bias.
-   **Popularity Bias:** The rich get richer. Popular items get shown more, get more clicks, and become even more popular.
-   **Position Bias:** Top items get clicked because they are top.

**Mitigation:**
1.  **Inverse Propensity Weighting (IPW):**
    -   Weight clicks by the inverse probability of the user seeing the item.
    -   If item X was at rank 10 (low probability of being seen) but got clicked, it's a *very* strong signal.
2.  **Fairness Constraints:**
    -   Ensure that the top 10 results contain at least 2 items from "Small Creators".
    -   This is a constrained optimization problem.

## Deep Dive: Online Learning (FTRL)

In fast-moving domains (News, Ads), a model trained yesterday is stale.
We need **Online Learning**.

**FTRL (Follow The Regularized Leader):**
-   An optimization algorithm designed for sparse data and online updates.
-   It updates the weights after *every* batch of clicks.
-   Used heavily in Ad Click Prediction (Logistic Regression).
-   **Architecture:**
    -   **Wide Part:** FTRL (Memorization).
    -   **Deep Part:** DNN (Generalization).
    -   **Wide & Deep Learning (Google).**

## Deep Dive: Evaluation Metrics Math (NDCG)

Why do we use NDCG instead of Accuracy?
Because order matters.

**CG (Cumulative Gain):**
\[ CG_p = \sum_{i=1}^p rel_i \]
Sum of relevance scores. (Does not care about order).

**DCG (Discounted Cumulative Gain):**
\[ DCG_p = \sum_{i=1}^p \frac{rel_i}{\log_2(i+1)} \]
Penalizes relevant items appearing lower in the list.

**IDCG (Ideal DCG):**
The DCG of the *perfect* ordering.

**NDCG (Normalized DCG):**
\[ NDCG_p = \frac{DCG_p}{IDCG_p} \]
Values are between 0 and 1. Comparable across queries.

## Deep Dive: Feature Engineering for Ranking

Features are the lifeblood of ranking.

### 1. Counting Features
-   "How many times has this user clicked this category?"
-   "How many times has this item been bought in the last hour?"
-   **Implementation:** Count-Min Sketch or Redis counters.

### 2. Crossing Features
-   `User_Country x Item_Country`: (Match vs Mismatch).
-   `User_Gender x Item_Category`.

### 3. Sequence Features
-   "Last 50 items viewed".
-   Processed by a Transformer or GRU to generate a dynamic user embedding.

## System Design: The Latency Budget

We have 200ms total. How do we spend it?

| Component | Budget | Notes |
| :--- | :--- | :--- |
| **Network Overhead** | 30ms | Round trip to client. |
| **Query Understanding** | 20ms | BERT is slow; use DistilBERT or caching. |
| **Retrieval (ANN)** | 20ms | Parallelize across shards. |
| **L1 Ranking** | 30ms | Dot products are fast. |
| **Feature Fetching** | 40ms | The bottleneck! Fetching features for 500 items from Feature Store (Cassandra/Redis). |
| **L2 Ranking** | 50ms | Heavy model inference. |
| **Re-Ranking** | 10ms | Logic heavy, compute light. |

**Optimization:**
-   **Parallelism:** Run Retrieval and L1 for different sources (Ads, Organic) in parallel.
-   **Feature Caching:** Cache hot item features in local memory.

## Top Interview Questions

**Q1: How do you handle "Position Bias" in training data?**
*Answer:*
Users click the top result because it's top.
1.  **Randomization:** Shuffle the top 3 results for 1% of traffic.
2.  **Counterfactual Model:** Add `Position` as a feature during training. During inference, set `Position = 0` (or `Position = 1`) for all items. This asks the model: "How relevant would this be if it were at the top?"

**Q2: Why use Dot Product for L1 and Trees/BERT for L2?**
*Answer:*
-   **L1:** Needs to scan millions of items. Dot product allows using ANN indices (MIPS). Trees/BERT cannot be indexed easily.
-   **L2:** Needs precision on a small set (500). Trees/BERT capture non-linear feature interactions (e.g., "User likes Sci-Fi AND Item is Star Wars") better than a simple Dot Product.

**Q3: How do you evaluate a ranking system offline?**
*Answer:*
We use **Replay Evaluation**.
Take a historical log: User saw `[A, B, C]`, clicked `B`.
Run the new model on the context.
If the new model ranks `B` at position 1, it wins.
**Metric:** NDCG (Normalized Discounted Cumulative Gain) or MRR (Mean Reciprocal Rank).

## Key Takeaways & Features

| Feature Type | Examples |
| :--- | :--- |
| **User** | Age, Gender, Past Clicks, Search History |
| **Item** | Title, Price, Category, CTR (Click-Through Rate) |
| **Context** | Time of Day, Device, Location |
| **Interaction** | User-Category Affinity, Last-Viewed-Item Similarity |

**Positional Bias:**
Users click the top result simply because it's at the top.
**Fix:** Train with "Position" as a feature, but set `Position=0` during inference (Counterfactual Inference).

## 8. Evaluation Metrics

-   **Offline:**
    -   **NDCG@K:** Normalized Discounted Cumulative Gain. (Gold Standard).
    -   **MRR:** Mean Reciprocal Rank.
-   **Online:**
    -   **CTR:** Click-Through Rate.
    -   **Conversion Rate:** Purchases / Clicks.
    -   **Dwell Time:** Time spent on the page.

## 9. Failure Modes

1.  **The Cold Start Problem:** New items have no interaction data.
    -   *Fix:* Use Content-based retrieval (Embeddings) + Exploration (Bandits).
2.  **Feedback Loops:** The model only learns from what it shows. It never learns about items it *didn't* show.
    -   *Fix:* Epsilon-Greedy Exploration (Show random items 1% of the time).

## 10. Summary

| Stage | Model | Input Size | Latency |
| :--- | :--- | :--- | :--- |
| **Retrieval** | ANN / Inverted Index | 1 Billion | 10ms |
| **L1 Ranker** | Two-Tower / LR | 10,000 | 20ms |
| **L2 Ranker** | BERT / GBDT | 500 | 50ms |
| **Re-Ranker** | Heuristics | 50 | 5ms |

---

**Originally published at:** [arunbaby.com/ml-system-design/0028-ranking-systems-at-scale](https://www.arunbaby.com/ml-system-design/0028-ranking-systems-at-scale/)

*If you found this helpful, consider sharing it with others who might benefit.*


