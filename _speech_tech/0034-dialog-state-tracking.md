---
title: "Dialog State Tracking (DST)"
day: 34
related_dsa_day: 34
related_ml_day: 34
related_agents_day: 34
collection: speech_tech
categories:
 - speech_tech
tags:
 - nlp
 - dialog-systems
 - state-tracking
 - transformers
 - llm
subdomain: "Dialogue Systems"
tech_stack: [TRADE, BERT-DST, Rasa, HuggingFace]
scale: "Multi-turn, Multi-domain"
companies: [Amazon (Alexa), Google (Assistant), Apple (Siri), PolyAI]
---

**"The brain of a task-oriented dialogue system: remembering what the user wants."**

## 1. The Role of DST in Dialogue Systems

In a Task-Oriented Dialogue System (TODS), the pipeline typically looks like this:

1. **ASR:** Audio → Text ("I want a cheap Italian restaurant").
2. **NLU:** Text → Intent/Slots (`Intent: find_restaurant`, `Price: cheap`, `Cuisine: Italian`).
3. **DST (Dialog State Tracking):** Updates the **current state** of the conversation based on history.
4. **Policy (DPL):** Decides the next action (e.g., "Ask for location").
5. **NLG:** Action → Text ("Where are you located?").
6. **TTS:** Text → Audio.

**Why is DST hard?**
- **Multi-turn dependencies:**
 - User: "Book a table at Mario's."
 - System: "For how many?"
 - User: "Five." (DST must know "Five" refers to `people`, not `time` or `price`).
- **Corrections:**
 - User: "Actually, make it for six." (DST must update `people=6`).
- **Co-reference:**
 - User: "What's the address of **that** place?" (DST must resolve "that place" to "Mario's").

## 2. State Representation

The **Dialog State** is typically a set of `(Slot, Value)` pairs.

**Example State:**
``json
{
 "domain": "restaurant",
 "slots": {
 "cuisine": "italian",
 "price_range": "cheap",
 "area": "center",
 "people": "5"
 }
}
``

The goal of DST is to predict `S_t` given `S_{t-1}`, System Action `A_{t-1}`, and User Utterance `U_t`.

## 3. Approaches to DST

### 1. Rule-Based / Frame-Based
- **Logic:** If intent is `inform` and entity is `cuisine`, update `cuisine` slot.
- **Pros:** Interpretable, easy to start.
- **Cons:** Brittle. Fails on complex sentences ("I'm not looking for Italian anymore").

### 2. Classification-Based (Fixed Ontology)
- **Ontology:** A pre-defined list of all possible values for every slot.
 - `Cuisine`: [Italian, Chinese, Indian, ...]
- **Model:** A classifier that takes `(U_t, S_{t-1})` and outputs a probability distribution over the ontology for each slot.
- **Pros:** Simple classification problem.
- **Cons:** Cannot handle **out-of-vocabulary (OOV)** values (e.g., a new restaurant name).

### 3. Generation-Based (Open Vocabulary)
- **Model:** Seq2Seq model (Transformer) that generates the value string.
- **Input:** "I want to eat at [Restaurant Name]."
- **Output:** Generates "[Restaurant Name]" char-by-char or token-by-token.
- **Pros:** Handles OOV values.
- **Cons:** Can generate invalid hallucinations.

## 4. Modern Architectures

### 1. TRADE (Transferable Multi-Domain State Generator)
- **Problem:** How to handle multiple domains (Hotel, Train, Taxi) without training separate models?
- **Architecture:**
 - **Utterance Encoder:** BiGRU / BERT encodes user text.
 - **Slot Gate:** Decides if a slot is `PTR` (generate from text), `DONTCARE`, or `NONE`.
 - **State Generator:** Copy mechanism (Pointer Network) to copy words from user utterance into the slot value.
- **Key Feature:** Zero-shot transfer to new domains by sharing parameters.

### 2. BERT-DST
- Uses a pre-trained BERT to encode the dialogue context.
- For each slot, it predicts a span `(start_index, end_index)` in the utterance that corresponds to the value.
- **Pros:** Extremely accurate for extractive values.

### 3. LLM-based DST (In-Context Learning)
- **Prompt:**
 ``
 Conversation:
 User: I need a hotel.
 System: Where?
 User: In Cambridge.
 
 Current State JSON:
 {"service": "hotel", "slots": {"area": "cambridge"}}
 ``
- **Pros:** No training required. Handles complex reasoning.
- **Cons:** High latency and cost.

## 5. Challenges in DST

### 1. Slot Carryover
- User: "Find a hotel in the north." (`area=north`)
- System: "I found 3 hotels."
- User: "Book the first one."
- **DST:** Must carry over `area=north` to the booking intent, plus resolve "first one".

### 2. Slot Value Normalization
- User says: "six thirty", "6:30", "half past six".
- DST must normalize all to `18:30`.

### 3. Zero-Shot Domain Adaptation
- Train on "Restaurants".
- Deploy on "Hospitals".
- The model should understand "I need a cardiologist" implies `department=cardiology` based on semantic similarity to "I need Italian food".

## 6. Evaluation Metrics

### 1. Joint Goal Accuracy (JGA)
- The percentage of turns where **ALL** slots in the state are predicted correctly.
- Strict metric. If you get 4/5 slots right, JGA is 0.

### 2. Slot Accuracy
- The percentage of individual slots predicted correctly.
- Usually much higher than JGA (> 95% vs > 50% JGA).

## 7. Deep Dive: Multi-Domain DST (MultiWOZ)

**MultiWOZ** is the standard benchmark dataset.
- **Domains:** 7 (Restaurant, Hotel, Attraction, Train, Taxi, Hospital, Police).
- **Complexity:** Users switch domains mid-conversation.
 - "Book a hotel in the center. Also, I need a taxi to get there."

**Cross-Domain Constraints:**
- The destination of the taxi must match the address of the hotel.
- DST must track these implicit constraints.

## 8. Deep Dive: Handling "Don't Care"

- User: "I want a restaurant, any price is fine."
- DST Update: `price_range = dontcare`.
- **Database Query:** `SELECT * FROM restaurants` (ignore price filter).
- This is distinct from `price_range = none` (user hasn't specified yet).

## 9. System Design: Low-Latency DST

In production, DST adds latency to every turn.

**Optimization:**
1. **Candidate Selection:** Only update slots relevant to the current domain.
2. **Caching:** Cache the state object. Only process the *delta* (new utterance).
3. **Distillation:** Distill BERT-Large into DistilBERT or TinyBERT for 10x speedup.

## 10. Real-World Case Study: Google Duplex

Google Duplex (AI that calls restaurants) requires extreme state tracking.
- **State:** Not just slots, but also "negotiation state".
- **Scenario:**
 - AI: "Table for 7pm?"
 - Human: "We only have 8pm."
 - AI (DST): Update `time=20:00`, check if acceptable against user constraints.

## 11. Top Interview Questions

**Q1: How do you handle slot value correction?**
*Answer:* The model must attend to the entire history. If the user says "No, I meant X", the model sees the previous "X" and the negation, updating the state to overwrite the old value.

**Q2: Pointer Networks vs. Classification?**
*Answer:* Pointer networks (copy mechanism) are better for names, times, and open sets. Classification is better for small fixed sets (Price: cheap/moderate/expensive).

**Q3: How does DST interact with the Policy?**
*Answer:* DST outputs the state `S_t`. The Policy takes `S_t` and database results to decide the action `A_t`.

## 12. Summary

| Component | Description |
| :--- | :--- |
| **Input** | User Utterance + Dialogue History |
| **Output** | Structured State (Slots & Values) |
| **Key Models** | TRADE, BERT-DST, LLMs |
| **Metric** | Joint Goal Accuracy (JGA) |
| **Challenge** | Context dependency, OOV values |

## 13. Deep Dive: TRADE Architecture Details

**TRADE (Transferable Multi-Domain State Generator)** is a landmark paper in DST.

**Key Components:**

1. **Utterance Encoder:**
 - Uses Bi-GRU to encode the user utterance `U_t` and dialogue history `H_t`.
 - Output: Context vectors `H_{ctx}`.

2. **State Encoder:**
 - Encodes the slot names (e.g., "hotel-price", "train-destination").
 - This allows the model to understand semantic similarity between slots across domains.

3. **Slot Gate (Classifier):**
 - For each slot `j`, predicts `P_{gate} \in \{PTR, NONE, DONTCARE\}`.
 - `PTR`: The value is in the utterance (generate it).
 - `NONE`: The slot is not mentioned.
 - `DONTCARE`: The user said "any".

4. **Copy Mechanism (Pointer Generator):**
 - If `P_{gate} = PTR`, the model generates the value by copying tokens from the utterance.
 - `P_{vocab} = P_{gen} P_{vocab} + (1 - P_{gen}) P_{copy}`.
 - This allows handling OOV words (e.g., rare restaurant names).

**Why it works for Zero-Shot:**
- It learns a general "slot-filling" behavior.
- If trained on "Restaurant-Price", it can transfer to "Hotel-Price" because the concept of "Price" is semantically similar.

## 14. Deep Dive: BERT-DST Implementation

**BERT-DST** treats state tracking as a **Reading Comprehension** task (SQuAD style).

**Input Format:**
`[CLS] [SLOT] price [SEP] [USER] I want a cheap place [SEP] [SYS] What price? [SEP]`

**Output:**
- **Start Logits:** Probability of each token being the start of the value.
- **End Logits:** Probability of each token being the end of the value.
- **Class Logits:** `NONE`, `DONTCARE`, `SPAN`.

**Code Snippet (HuggingFace Transformers):**

``python
import torch
from transformers import BertForQuestionAnswering

class BertDST(torch.nn.Module):
 def __init__(self):
 super().__init__()
 self.bert = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
 self.classifier = torch.nn.Linear(768, 3) # NONE, DONTCARE, SPAN

 def forward(self, input_ids, attention_mask):
 outputs = self.bert(input_ids, attention_mask=attention_mask)
 start_logits = outputs.start_logits
 end_logits = outputs.end_logits
 
 # Use [CLS] token embedding for classification
 cls_embedding = outputs.hidden_states[-1][:, 0, :]
 class_logits = self.classifier(cls_embedding)
 
 return start_logits, end_logits, class_logits
``

## 15. Deep Dive: LLM-based DST (In-Context Learning)

With GPT-4, we can do DST without training.

**Prompt Engineering Strategy:**

1. **Instruction:** "You are a helpful assistant tracking the state of a dialogue. Output JSON."
2. **Ontology Definition:** "Possible slots: restaurant-food, restaurant-area, restaurant-price."
3. **Few-Shot Examples:** Provide 3-5 examples of difficult cases (corrections, co-reference).
4. **Chain-of-Thought (CoT):** Ask the model to explain *why* it updated a slot.

**Example Prompt:**
``
User: "I want a cheap place."
Reasoning: User specified price constraint.
State: {"price": "cheap"}

User: "Actually, I don't care about price."
Reasoning: User corrected previous constraint.
State: {"price": "dontcare"}

User: "Find me a place in the center."
Reasoning: User added area constraint.
State: {"price": "dontcare", "area": "center"}
``

**Fine-Tuning (LoRA):**
For lower latency/cost, fine-tune a smaller model (Llama-3-8B) on the MultiWOZ dataset. It can match GPT-4 performance at 1/10th the cost.

## 16. Deep Dive: Handling Non-Categorical Slots

Some slots are not simple strings.

**1. Time:**
- User: "I want to leave at 5."
- User: "I want to arrive by 5."
- DST must distinguish `leave_at` vs `arrive_by`.
- Normalization: "5pm", "17:00", "five in the afternoon" -> `17:00`.

**2. Boolean:**
- User: "Does it have internet?"
- DST: `internet=True`.
- User: "I don't need parking."
- DST: `parking=False` (or `dontcare` depending on schema).

**3. Numbers (People):**
- User: "Me and my wife." -> `people=2`.
- User: "A party of five." -> `people=5`.
- Requires NLU reasoning or a number parser.

## 17. Deep Dive: Data Augmentation for DST

DST data is expensive to collect (requires expert annotation).

**Technique 1: Slot Substitution**
- Original: "I want a [Italian] restaurant in the [Center]."
- Augmented: "I want a [Chinese] restaurant in the [North]."
- Replace values using the ontology.

**Technique 2: Back-Translation**
- English -> French -> English.
- "I'm looking for a cheap hotel" -> "Je cherche un hôtel bon marché" -> "I am searching for an inexpensive hotel."
- Increases linguistic diversity.

**Technique 3: User Simulator**
- Build a rule-based User Simulator that interacts with the system.
- Generate millions of synthetic dialogues.
- Train DST on synthetic data + fine-tune on real data.

## 18. Deep Dive: Error Recovery

What if DST gets it wrong?

**Confidence Scores:**
- If DST predicts `cuisine=italian` with 0.4 confidence.
- **Policy Action:** "Did you say you wanted Italian food?" (Confirmation).
- If confidence > 0.9, proceed silently.

**N-Best Lists:**
- DST shouldn't just output the top state.
- Output top-N hypotheses.
- The Policy can use the full distribution to make better decisions.

## 19. System Design: Scalable DST Service

**Architecture:**
- **Stateless Service:** The DST service should not store state.
- **Input:** `(History, Current_Utterance)`.
- **Output:** `New_State`.
- **Storage:** Redis stores the session state.

**Latency Budget:**
- Total Turn Latency: 200ms.
- ASR: 50ms.
- **DST: 50ms.**
- Policy: 20ms.
- NLG+TTS: 80ms.

**Optimization:**
- **Quantization:** INT8 quantization of BERT/Trade models.
- **Caching:** Cache common utterances ("Yes", "No", "Thank you") -> No-op state update.

## 20. Advanced: Multi-Modal DST

In AR/VR or Smart Displays, context includes screen clicks.

- **Scenario:** User points at a screen and says "How much is this one?"
- **Input:** Audio + Gaze/Touch coordinates.
- **Resolution:**
 - DST receives `click_event(item_id=123)`.
 - Resolves "this one" to `item_id=123`.
 - Updates state: `focused_item=123`.

## 21. Summary

| Component | Description |
| :--- | :--- |
| **Input** | User Utterance + Dialogue History |
| **Output** | Structured State (Slots & Values) |
| **Key Models** | TRADE, BERT-DST, LLMs |
| **Metric** | Joint Goal Accuracy (JGA) |
| **Challenge** | Context dependency, OOV values |
| **Challenge** | Context dependency, OOV values |
| **Future** | Multi-modal, Zero-shot transfer |

## 22. Deep Dive: Schema-Guided Dialogue (SGD)

The **SGD dataset** (Google) pushed DST towards zero-shot generalization.

**Key Idea:**
- Instead of hardcoding slots, provide a **Schema Description** (natural language description of slots).
- Model Input: `User: "I want a hotel."` + `Schema: "Hotel-Area: The location of the hotel."`
- Model Task: Does the utterance match the schema description?

**Benefit:**
- To add a new domain ("Flight"), just write descriptions. No training data needed.

## 23. Deep Dive: Reinforcement Learning for DST

Standard DST is trained with Supervised Learning (Cross-Entropy).
**Problem:** It optimizes for per-turn accuracy, not long-term success.

**RL Approach:**
- **State:** Dialogue History.
- **Action:** DST Update.
- **Reward:** +1 if the conversation ends successfully (user books hotel), -1 if user hangs up.
- **Algorithm:** Policy Gradient (REINFORCE) or PPO.

**Result:** The model learns to be robust. Even if it misses a slot in turn 2, it might recover in turn 4 to maximize the final reward.

## 24. Deep Dive: Latency Optimization Techniques

DST is often the bottleneck.

**Technique 1: Cascade Architecture**
- Use a tiny model (DistilBERT) for 90% of turns.
- If confidence < threshold, call the large model (GPT-4 / BERT-Large).

**Technique 2: Parallel Execution**
- Run DST in parallel with ASR? No, DST needs text.
- Run DST in parallel with **NLU Intent Classification**.
- While DST computes state, NLU computes "Is the user angry?" (Sentiment).

**Technique 3: State Delta Prediction**
- Instead of predicting the full state every turn, predict the **operation**:
 - `UPDATE(price, cheap)`
 - `DELETE(area)`
 - `KEEP(others)`

## 25. Deep Dive: Privacy in DST

DST stores user preferences. This is PII (Personally Identifiable Information).

**Risks:**
- `phone_number`, `credit_card`, `home_address`.

**Mitigation:**
1. **PII Redaction:** Replace sensitive entities with tokens before storage.
 - "Call 555-0199" -> "Call [PHONE_NUMBER]".
 - DST tracks `phone=[PHONE_NUMBER]`.
 - The actual number is stored in a secure, ephemeral context, not the logs.
2. **Federated Learning:** Train DST on user devices. Only send gradients to the server.

## 26. Code: Simple Rule-Based DST

For simple use cases, don't use a Transformer.

``python
class RuleBasedDST:
 def __init__(self):
 self.state = {}
 self.ontology = {
 "price": ["cheap", "moderate", "expensive"],
 "area": ["north", "south", "east", "west", "center"]
 }

 def update(self, utterance):
 utterance = utterance.lower()
 
 # Heuristic: Check for keywords
 for slot, values in self.ontology.items():
 for value in values:
 if f" {value} " in f" {utterance} ":
 self.state[slot] = value
 
 # Heuristic: Negation ("not cheap")
 if "not cheap" in utterance and self.state.get("price") == "cheap":
 del self.state["price"]
 self.state["price_not"] = "cheap"
 
 return self.state

# Usage
dst = RuleBasedDST()
print(dst.update("I want a cheap restaurant"))
# {'price': 'cheap'}
print(dst.update("Actually, not cheap"))
# {'price_not': 'cheap'}
``

## 27. Summary

| Component | Description |
| :--- | :--- |
| **Input** | User Utterance + Dialogue History |
| **Output** | Structured State (Slots & Values) |
| **Key Models** | TRADE, BERT-DST, LLMs |
| **Metric** | Joint Goal Accuracy (JGA) |
| **Challenge** | Context dependency, OOV values |
| **Challenge** | Context dependency, OOV values |
| **Future** | Multi-modal, Zero-shot transfer |

## 28. Deep Dive: Evaluation Datasets Beyond MultiWOZ

While MultiWOZ is the standard, it has flaws (noisy annotations).

**1. CrossWOZ (Chinese):**
- First large-scale Chinese Cross-Domain dataset.
- More complex dependencies than MultiWOZ.

**2. SGD (Schema-Guided Dialogue):**
- 16,000+ dialogues across 20 domains.
- Designed to test zero-shot transfer to unseen domains.

**3. TreeDST:**
- Models dialogue state as a tree structure rather than a flat list of slots.
- Handles hierarchical dependencies (`flight` -> `return_flight`).

## 29. Deep Dive: User Simulation for DST Training

How do we build a User Simulator?

**Agenda-Based User Simulator (ABUS):**
- **Goal:** The user has a goal `(inform: cuisine=italian, request: address)`.
- **Stack:** The user keeps a stack of actions.
- **Policy:**
 - If System says "What cuisine?", User pops `inform: cuisine=italian`.
 - If System says "Address is 123 Main St", User pops `request: address`.
- **Error Model:** Introduce noise (ASR errors, synonym replacement) to make it realistic.

**Neural User Simulator:**
- Train a Seq2Seq model on real data to predict `User_Utterance` given `System_Utterance` and `User_Goal`.
- More natural, but harder to control.

## 30. Deep Dive: Interactive Learning

Can the DST improve by talking to users?

**Scenario:**
- User: "I want a table at The Golden Dragon."
- DST: `restaurant=Golden Dragon` (Confidence 0.4).
- System: "Did you mean The Golden Dragon?"
- User: "Yes."
- **Update:** Add `(User Utterance, State)` to training set.

**Bandit Feedback:**
- If the user completes the task successfully, the entire trajectory was likely correct.
- Use this implicit feedback to fine-tune the model.

## 31. Deep Dive: Deployment on Edge Devices

Running BERT-DST on a phone?

**Challenges:**
- **Size:** BERT-Base is 400MB.
- **Latency:** Must be < 50ms.

**Solutions:**
1. **MobileBERT / TinyBERT:** Compressed architectures (15-20MB).
2. **TFLite / ONNX Runtime:** Optimized inference engines for ARM CPUs.
3. **Dynamic Quantization:** Convert weights to INT8 at runtime.

**Example (TFLite Conversion):**
``python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('dst_model.tflite', 'wb') as f:
 f.write(tflite_model)
``

with open('dst_model.tflite', 'wb') as f:
 f.write(tflite_model)
``

## 32. Deep Dive: Contextual Bandit for DST Optimization

**Problem:** Which DST model should we use for a given user?
- Power users → Fast rule-based DST.
- Complex queries → BERT-DST.
- Ambiguous queries → LLM-DST.

**Contextual Bandit Approach:**
- **Context:** User history, query complexity score.
- **Arms:** [Rule-Based, BERT-DST, LLM-DST].
- **Reward:** 1 if task succeeds, 0 if user abandons.
- **Algorithm:** Thompson Sampling or LinUCB.

**Result:** Dynamically route queries to the most cost-effective model while maximizing success rate.

## 33. Case Study: Rasa Open Source DST

**Rasa** is the most popular open-source dialogue framework.

**DST Component: "Slot Filling"**
- Uses a **Diet Classifier** (Dual Intent Entity Transformer).
- **Architecture:**
 - Shared Transformer encoder for both Intent and Entity.
 - Separate CRF heads for slot tagging.
- **Training:** Rasa NLU data format (Markdown).

**Example:**
``markdown
## intent:book_restaurant
- I want to book a table at [Mario's](restaurant_name) for [5](people) people
- Reserve [2](people) seats at [The Golden Dragon](restaurant_name)
``

**Deployment:**
- Rasa Action Server handles DST + Policy + NLG.
- Can be deployed on-prem (no cloud dependency).

## 34. Future Trends: Unified Dialogue Models

**Current:** Separate modules (ASR → NLU → DST → Policy → NLG → TTS).
**Future:** **End-to-End Dialogue Models**.

**AudioLM / SpeechGPT:**
- Input: Audio tokens.
- Output: Audio tokens (system response).
- **DST is implicit** in the latent state of the Transformer.

**Advantages:**
- No error propagation between modules.
- Can handle paralinguistics (tone, emotion) directly.

**Challenges:**
- Lack of interpretability. How do we debug if we can't see the state?
- **Hybrid Approach:** Use E2E for generation, but extract state for logging/debugging.

## 35. Further Reading

1. **"TRADE: Transferable Multi-Domain State Generator" (Wu et al., 2019):** The TRADE paper.
2. **"Schema-Guided Dialogue State Tracking" (Rastogi et al., 2020):** Zero-shot DST.
3. **"MultiWOZ 2.1: A Consolidated Multi-Domain Dialogue Dataset" (Eric et al., 2020):** The standard benchmark.
4. **"Recent Advances in Deep Learning Based Dialogue Systems" (Chen et al., 2021):** Comprehensive survey.
5. **"Rasa: Open Source Language Understanding and Dialogue Management" (Bocklisch et al., 2017):** Rasa architecture.

- **"Rasa: Open Source Language Understanding and Dialogue Management" (Bocklisch et al., 2017):** Rasa architecture.

- **"Rasa: Open Source Language Understanding and Dialogue Management" (Bocklisch et al., 2017):** Rasa architecture.

## 36. Ethical Considerations

**1. Bias in Slot Values:**
- If training data has `doctor=male` 90% of the time, DST might incorrectly resolve "the doctor" to male pronouns.
- **Fix:** Balanced data collection and fairness-aware training.

**2. Manipulation:**
- A malicious DST could intentionally misinterpret user requests to push certain products.
- **Example:** User: "Cheap hotel" → DST: `price=expensive` (to maximize commission).
- **Safeguard:** Audit logs, user feedback loops.

**3. Transparency:**
- Users should know when they're talking to a bot vs. human.
- **Regulation:** California Bot Disclosure Law requires bots to identify themselves.

## 37. Conclusion

Dialog State Tracking is the **memory** of conversational AI. Without accurate state tracking, dialogue systems would be stateless, forcing users to repeat themselves every turn. The evolution from rule-based systems to BERT-DST to LLM-based in-context learning represents a fundamental shift in how we build dialogue systems. Modern DST systems must handle multi-domain conversations, zero-shot transfer to new domains, and real-time updates with sub-50ms latency. As we move toward unified end-to-end dialogue models, DST will become implicit rather than explicit, but the core challenge remains: **understanding what the user wants, even when they don't say it directly**. The future of DST lies in multi-modal understanding (combining speech, vision, and touch), personalization (learning user preferences over time), and explainability (being able to justify why the system believes the user wants X).

## 38. Summary

| Component | Description |
| :--- | :--- |
| **Input** | User Utterance + Dialogue History |
| **Output** | Structured State (Slots & Values) |
| **Key Models** | TRADE, BERT-DST, LLMs |
| **Metric** | Joint Goal Accuracy (JGA) |
| **Challenge** | Context dependency, OOV values |
| **Future** | Multi-modal, Zero-shot transfer |

---

**Originally published at:** [arunbaby.com/speech-tech/0034-dialog-state-tracking](https://www.arunbaby.com/speech-tech/0034-dialog-state-tracking/)
