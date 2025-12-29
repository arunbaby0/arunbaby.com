---
title: "Structured Output Patterns"
day: 35
related_dsa_day: 35
related_ml_day: 35
related_speech_day: 35
collection: ai_agents
categories:
 - ai-agents
tags:
 - structured-output
 - json-mode
 - pydantic
 - function-calling
 - schema-design
difficulty: Medium
---

**"Make agents predictable: enforce schemas, validate outputs, and recover automatically when the model slips."**

## 1. Introduction: The “AI Contract”

### 1.1 The History of the AI Contract
In the early days of LLMs (2020-2022), developers spent 80% of their time writing "Garbage Disposal" code—complex regex functions designed to clean up the conversation. You would ask for a list of fruits and get: *"Sure, I love fruits! Here they are: 1. Apple, 2. Banana... hope this helps!"*. This was the "Dark Ages" of agentic engineering. Today we handle this through **Hard Constraints**.

### 1.2 The Precision/Creativity Trade-off
There is a fundamental tension in LLMs. The more you force them to follow a structure (Precision), the less "Creative" they become. For an agent checking a bank balance, you want 100.0% precision and 0.0% creativity. Structured output is the dial you use to turn down the model's "hallucination engine" and turn up its "logic engine."

### 1.3 Why JSON Won the AI War
While LLMs can output XML or YAML, **JSON** is the undisputed champion. It is token-efficient, natively supported by all major model providers (OpenAI, Gemini), and perfectly matches the `dict` and `class` structures of modern programming languages like Python and TypeScript.

---

## 2. Roadmap & Junior Engineer Goals

Structured output is the "Final Frontier" of agentic reliability.

### Step 1: The Scraper
Learn to use Pydantic to extract data from messy text. This is the "Hello World" of structured output.

### Step 2: The Action-Taker
Connect your Pydantic schemas to actual Python functions. When the agent outputs a `ChargeCard` object, the money actually moves. This requires **High-Stakes Validation**.

### Step 3: The Architect
Design a recursive, self-healing system where agents can update their own schemas and communicate via complex DAGs.

---

## 3. Implementing a "Drafting" Loop for JSON

Sometimes a single turn is not enough to get a complex JSON right.

**The Workflow:**
1. **Draft:** Agent A outputs a draft JSON.
2. **Audit:** An automated script checks for "Null" fields or type mismatches.
3. **Refactor:** The script sends a prompt back: *"Your JSON had 2 empty fields. Based on the source text, can you fill those in? Here is your previous attempt: [JSON]"*
4. **Final:** The agent outputs the complete, verified JSON.

---

## 4. The "Schema-First" vs. "Prompt-First" Debate

Should you define your logic in the JSON Schema or the System Prompt?

* **Schema-First:** Put all constraints in the `description` fields of the Pydantic model.
* **Prompt-First:** Put the logic in the main instruction block and keep the schema "Dumb."
* **Verdict:** **Schema-First** is more reliable. Language Models pay more attention to tokens that are close to where they are writing. By putting the instruction in the field description, you ensure the model "remembers" it as it generates the value.

---

## 5. Handling "Token Overflows" in JSON

What if the LLM's JSON is 5,000 tokens but your `max_tokens` is 4,000? The JSON will be cut off at the end, usually looking like `... "key": "val`.

**The Fix: Partial Parsing.**
Use a library like `partialjson` or `json-stream`. These can parse a "Broken" JSON string and return as many valid keys as were successfully generated before the cutoff. This prevents the entire request from being a total loss.

---

## 6. Security: The "JSON Bomb" and Denial of Service

Malicious agents (or poisoned data) can output thousands of nested brackets: `[[[[[[[[...]]]]]]]]`.

**The Defense:**
1. **Max Depth:** Set a hard limit (e.g., 5 levels) in your JSON recursive parser.
2. **Schema Enforcement:** Never use `Dict[str, Any]` in your production code. Always define every field.

---

## 7. Advanced Pattern: The "CoT + JSON" Hybrid

One of the biggest mistakes developers make is asking for *pure* JSON (essentially asking for the result without the thinking). When you force a model to output only JSON, you are training it to skip its "Reasoning" step.

**The Fix: The Reasoning Envelope.**
Include a `thought_process` field as the **first** key in your schema. Because LLMs process tokens sequentially, this forces the model to "explain" its logic *before* it picks the final value for the data fields.

**Example Schema:**
``json
{
 "thought_process": "The user mentioned they are 25 but were born in 1998, which is consistent. They live in Paris, which is in France.",
 "age": 25,
 "country_code": "FR",
 "confidence_score": 0.95
}
``
By allowing this "Internal Monologue," you increase the accuracy of the structured data by up to 30%, especially for complex extraction tasks.

---

## 8. Schema Design Patterns for Agents

### 3.1 The "Envelope" Pattern
Wrap your data in a metadata layer. This allows your code to distinguish between an "Agent Answer" and an "Agent Error" using the same consistent JSON structure.
``json
{
 "status": "success",
 "payload": { ... },
 "error_msg": null,
 "execution_time_ms": 450
}
``

### 3.2 The "Polymorphic" Tool
Instead of having 10 different tools (which bloats the context window), use one `perform_action` tool that takes a `Union` of different types (e.g., `CreateUser`, `UpdateUser`).

### 3.3 The "Recursive Schema" Pattern
For tasks like summarizing a document's table of contents, use a schema that allows a node to have child nodes.
``python
class Header(BaseModel):
 title: str
 sub_headers: List["Header"] = [] # Recursive!
``
This is essential for capturing hierarchical information without losing the parent-child relationships.

---

## 9. Token Efficiency: The Math of JSON Keys

In a traditional API, key names like `user_authentication_timestamp_registered` are fine—they add clarity. In AI agents, **Keys cost money.**

### 4.1 The Compression Ratio
If you are processing 1 million requests a day:
* Key `"account_balance_in_usd"` = 25 tokens.
* Key `"bal"` = 3 tokens.
* **Savings:** 22 tokens per request. Over 1M requests, that's **22 Million tokens** saved. At `10 per million tokens, you just saved `220/day with a single rename.

### 4.2 The "Schema-Mapping" Solution
You can use a "Compressed Schema" for the LLM (with 1-letter keys) and then have a small Python function that maps `a -> "account_name"`, `b -> "balance"` before the data hits your database. This gives you the best of both worlds: extreme token efficiency and human-readable code.

---

Use **Pydantic** classes to define your schemas. Pydantic handles the translation to JSON-Schema (which the LLM actually sees) and ensures that the model's output is parsed into a clean Python object with full type safety.

**The Mini-Prompt Power:**
The `Field(description="...")` parameter in Pydantic is your most powerful tool.
``python
class UserProfile(BaseModel):
 mood: str = Field(description="One of [HAPPY, SAD, ANGRY]. Default to HAPPY if unclear.")
``
The LLM provider (like OpenAI or Anthropic) takes these descriptions and injects them directly into the "System Instructions." This is often more effective than writing a separate 5-page PDF of instructions because the instruction sits **exactly where the model is deciding what to write**.

---

---

## 10. Case Study: The "Messy Legal Contract" Parser

Structured output is the only way to convert unstructured PDFs into a SQL database. A legal agent reads a 100-page lease, extracts clauses into a recursive JSON structure, validates them against the source, and directly executes `INSERT` statements into a database.

---

---

## 11. JSON Mode vs. Function Calling: Which one to use?

This is a common point of confusion for junior engineers. Both features force the model to output JSON, but they do it in different ways.

### 8.1 JSON Mode (The "Soft" Constraint)
In "JSON Mode," you simply tell the model: *"You must output valid JSON."* You then provide your schema in the system prompt.
* **Pros:** Flexible. The model can be slightly more conversational if allowed.
* **Cons:** The model can still hallucinate a key or misspell a field name. You are responsible for all validation logic in your code.

### 11.2 Function Calling (The "Hard" Constraint)
In Function Calling, the schema is passed as a **Separate Parameter** in the API call (the `tools` or `functions` array).
* **Pros:** The model is trained specifically to fill these "Slots." Many providers also use constrained sampling (see **Grammar-Based Sampling** in Section 13) to keep outputs valid.
* **Cons:** Slightly more restrictive. If your schema is too complex, the model might get confused and fail to "call" the function at all.

**Verdict:** Use **Function Calling** for 3rd-party tool use or database updates. Use **JSON Mode** for internal reasoning or data extraction where the exact key names are less critical than the data types.

---

## 12. Schema Evolution: Handling Versioning in Production

What happens when you need to add a `middle_name` field to a user profile that's used by 1,000 agents currently running?

### 9.1 The "Backwards Compatibility" Rule
Never delete a mandatory field. If you must remove a field, make it optional first, wait for all agents to update, and then remove it 30 days later.

### 9.2 The "Schema Registry" Pattern
In a large team, store your Pydantic models in a central library.
* **Version 1:** `class UserV1(BaseModel): name: str`
* **Version 2:** `class UserV2(BaseModel): name: str, age: int`
By keeping both versions in code, you can use a **Transformation Agent** to "Up-convert" any older checkpoints (State Management) from V1 to V2 as they are loaded.

---

## 13. Grammar-Based Sampling: The "Outlines" Revolution

For open-source models, you can use **Grammar-Based Sampling** (Outlines/Guidance). This modifies the model's logits at every token, making it **mathematically impossible** for the model to output invalid JSON. This moves your reliability from 99% to 100%.

---

## 14. The JSON Schema Masterclass: Field Descriptions

The `description` field in your Pydantic model is a **Mini-Prompt**. Most LLM providers inject these descriptions into the system instructions. A precise description like *"Integer between 1 and 100. If unknown, return -1"* is more effective than a 5-page PDF of instructions.

---

## 15. Security: Type Safety & JSON Bombs

Don't trust the agent's JSON.
* **Sanitization:** Guard against XSS markers in string values.
* **Recursion Limits:** Set a max depth in your parser to prevent "Recursive JSON Bombs" from crashing your server.
* **Strict Typing:** Use Pydantic's `strict=True` to prevent the model from successfully outputting `"10"` when an `int` 10 is required.

---

## 16. Multi-Modal Structured Output

Link AI Vision to UI Graphics. An agent looks at a photo, outputs the `[x, y]` coordinates of an object in JSON, and your React frontend uses that data to render an interactive label over the image.

---

---

## 17. The "Plan" Schema: Structuring Agent Thought (DAGs)

For complex agents, you don't just want one step. You want a **Plan**.

**The DAG Schema:**
``python
class Step(BaseModel):
 id: int
 tool: str
 args: dict
 dependencies: List[int] = []

class Plan(BaseModel):
 steps: List[Step]
``
By structuring the plan as a **Directed Acyclic Graph (DAG)**, your orchestrator can look at the `dependencies` list and execute steps 1, 3, and 5 in parallel if they don't depend on each other. This reduces your total task latency by up to 50%.

---

## 18. Data Integrity: The "Double Verification" Pattern

Even with structured output, the LLM can still "Lie" (Hallucinate) within the JSON.

**The Workflow:**
1. **Agent A (The Extractor):** Takes an image and outputs `{ "price": 99.99 }`.
2. **Agent B (The Auditor):** Takes the *same* image and the *JSON* from Agent A.
3. **Task:** "Verify that the price in the JSON matches the image exactly. If not, output a corrected JSON."
4. **Result:** This "Cross-Check" is the highest form of reliability in current AI engineering.

---

## 19. Performance Detail: Handling Dates & Decimals

LLMs are notoriously bad at formatting Dates and Decimals inside JSON.
* **Fix:** Force the agent to output dates as ISO-8601 strings and let your Python code handle the conversion to `datetime` objects.
* **Fix:** Use `float` with 2 decimal place constraints in your schema to prevent the model from outputting scientific notation for currency.

---

## 20. The "Self-Documenting" Agent

Ask your agent to output a JSON object that describes its own capabilities. This allows your frontend to dynamically render "Help Tooltips" or "Command Menus" without you having to hardcode them.

---

---

## 21. Security: Sanitizing JSON Values (Parameter Injection)

Just because the output is JSON doesn't mean it's safe.

**The Vulnerability:**
Imagine your agent outputs a JSON that you then pass directly into a SQL query:
``json
{"user_id": "123; DROP TABLE users;"}
``
**The Fix: Strict Validation.**
In Pydantic, use `UUID` or `EmailStr` types instead of raw strings. This forces the parser to reject anything that doesn't follow the exact format, killing injection attacks before they touch your database.

---

## 22. Future Trends: Native Structural Encoding

We are moving away from "Prompting" for JSON and toward **Native Constraints**.

### 25.1 Speculative Decoding for JSON
Models are being trained to predict the *structure* of a JSON block much faster than the *content*. This allows for ultra-fast "Structural Prefills" where the model generates the keys almost instantly.

### 25.2 Multi-Scale Schemas
Instead of one giant 1MB schema, agents will use **Adaptive Schemas**. The orchestrator detects the task and sends only the specific "Branch" of the schema needed, saving 90% on input tokens.

---

## 23. The "Unit Test" for Schemas: How to Test Your Agent

How do you know if your Pydantic model is "Agent-Friendly"?

**The Mock Test:**
1. **Generate:** Use a top-tier model (GPT-4o) to generate 100 "Edge Case" JSONs that follow your schema.
2. **Validate:** Try to parse those JSONs into your Pydantic objects.
3. **Refine:** If the model struggles to generate valid data for a specific field, the field's `description` is likely too vague. Update the description and repeat.

---

## 24. FAQ: Advanced Troubleshooting

**Q: My agent keeps outputting 'JSON' as a string inside the value. Why?**
A: This is a "Conversational Drift" error. The model is confusing the *format* of the output with the *content*.
* **Fix:** Add a negative constraint: *"Do not include the word 'JSON' inside the data values unless specifically requested."*

**Q: How do I handle circular dependencies in my schema?**
A: **Don't.** Circular dependencies (e.g., A contains B, B contains A) will crash almost any JSON parser. Instead, use "References" (IDs) and have the agent output a flat list of objects that link to each other.

**Q: Is YAML better for small models?**
A: Sometimes. YAML is more human-readable and takes fewer tokens (no braces). However, because YAML relies on indentation, even a single space error can break the entire file. For **reliability**, JSON is still the winner.

**Q: Can I use structured output to generate code?**
A: Yes. Instead of raw text, have the agent output a JSON object: `{ "language": "python", "code": "...", "unit_tests": "..." }`. This allows you to programmatically isolate the code and run it in a sandbox.

**Q: Why does my agent hallucinate keys that aren't in the schema?**
A: This is often caused by "Leakage" from the model's training data. If the model has seen a thousand "Invoice" schemas on GitHub, it will try to add `tax_rate` to your schema even if you didn't ask for it. Force it to stop with `extra='forbid'` (see the security + strict typing guidance in Section 15).

---

## 25. Complex Example: The "E-commerce Support Ticket" Schema

Let's look at a production-grade schema for a ticket-routing agent.

``python
class Ticket(BaseModel):
 thought: str = Field(..., description="Reasoning about the user's intent")
 priority: int = Field(1, ge=1, le=5)
 category: Literal["Billing", "Technical", "Shipping", "General"]
 entities: Dict[str, str] = Field(default_factory=dict, description="Extracted IDs, emails, etc")
 next_step: str = Field(..., description="Action to take: [REPLY, ESCALATE, WAIT]")
``

### 30.1 Why this works
1. **Reasoning First:** The `thought` key forces the model to categorize correctly.
2. **Range Constraints:** `ge=1, le=5` prevents the model from outputting `priority: 99`.
3. **Strict Categories:** `Literal` ensures the database doesn't crash on a "Billing support" category it doesn't recognize.

---

## 26. The Future: "Schema-Driven" Small Language Models (SLMs)

We are seeing a trend where 1.5B and 7B parameter models (like Phi-3 or Llama-3-8B) are being fine-tuned **specifically** on structured output.

* **The Result:** A 7B model that is 100% accurate at JSON formatting can replace a 1T model (GPT-4) for 90% of extraction tasks.
* **The Strategy:** As a junior engineer, always start with a big model to "Design the Schema," but then try to "Downgrade" to a smaller model to save 99% of your costs once the schema is stable.

---

## 27. Handling Non-Latin Characters and Localization in JSON

LLMs can sometimes struggle with JSON formatting when the content is in Chinese, Arabic, or Emoji.

* **The Problem:** Unicode characters can take multiple tokens, sometimes "confusing" the model's bracket-matching logic.
* **The Fix:** Use `ensure_ascii=False` when testing your local parsers, and ensure your system prompt explicitly allows UTF-8 in string values.

---

## 28. Benchmarking Model Accuracy for JSON

Not all models are created equal for structured data.

| Model | JSON Accuracy (Complex Nested) | Tool Call Reliability |
| :--- | :--- | :--- |
| **GPT-4o** | 99% | Excellent |
| **Claude 3.5 Sonnet** | 98.5% | Exceptional |
| **Llama-3-70B** | 95% | Good |
| **Mistral Large 2** | 94% | Good |
| **Gemini 1.5 Pro** | 97% | Excellent (Large Context) |

* **Pro Tip:** If your agent is failing to output JSON, it's often more effective to **Change the Model** than to write a 10-page prompt.

---

## 29. JSON vs. Python Code Generation: When to Use Which?

Sometimes, JSON isn't enough. If you need the agent to perform a complex calculation (e.g., *"Calculate the mortgage interest over 30 years with a variable rate"*), a JSON object with a `result` field will likely be wrong.

* **The Python Pattern:** Instead of asking for JSON, ask the agent to output a **Python Script**.
* **The Workflow:**
 1. Agent writes a Python function.
 2. The orchestrator runs the code in a **Secure Sandbox**.
 3. The result of the code is the "Answer."
* **Verdict:** Use **JSON** for data state and tool parameters. Use **Code Generation** for math, logic, and data transformation.

---

## 30. The "Self-Correction" Loop in Detail

As a junior engineer, your first `while` loop should be a **JSON Recovery Loop**.

``python
retries = 3
while retries > 0:
 try:
 raw_output = llm.generate(prompt)
 data = UserSchema.model_validate_json(raw_output)
 return data # Success!
 except ValidationError as e:
 # Pass the SPECIFIC pydantic error back to the model
 prompt += f"\nError in your last JSON: {str(e)}. Please correct it."
 retries -= 1
``
This simple loop increases your system's reliability from 90% to 99.9%. By providing the model with the exact line and type of the error, you are giving it the "Map" it needs to fix itself.

---

## 31. Large Scale Schema Management (Prompt Governance)

In a company with 50 different agents, you cannot just copy-paste schemas. You need a **Schema Registry**.

1. **Central Library:** Store your Pydantic models in a shared Git repo.
2. **Versioning:** Tag your schemas (e.g., `invoice_schema_v2`).
3. **Automatic Testing:** Every time you update the schema, run a script that tests 5 different LLMs against it to ensure no regressions in accuracy.

---

## 32. Monitoring "Schema Drift": The Silent Failure

Sometimes, an LLM provider updates their model and it starts adding extra whitespace or different escaping characters to its JSON.

**The Fix: The "Consistency" Dashboard.**
Track the "Parsing Failure Rate" in your observability tool. If the rate jumps from 0.1% to 2% on a Tuesday morning, you know that the model provider has changed something behind the scenes, and you need to update your **Retry Logic** or your **Validation Schema**.

---

## 33. Key Takeaways & Junior Engineer Roadmap

Structured output is the "Final Frontier" of agentic reliability.

### Step 1: The Scraper
Learn to use Pydantic to extract data from messy text. This is the "Hello World" of structured output.

### Step 2: The Action-Taker
Connect your Pydantic schemas to actual Python functions. When the agent outputs a `ChargeCard` object, the money actually moves. This requires **High-Stakes Validation**.

### Step 3: The Architect
Design a recursive, self-healing system where agents can update their own schemas and communicate via complex DAGs.

**Final Project Idea:** build a "News Investigator."
1. **Input:** A broad topic (e.g., "AI Regulation").
2. **Agent:** Searches the web, parses 10 articles into a structured `ResearchNode` schema.
3. **Orchestrator:** Validates the nodes, identifies missing info, and sends a "Delta Schema" back to the agent to fill in the gaps.
4. **Output:** A 100% verified, structured Knowledge Graph.

**Congratulations!** You've completed the communication trilogy. You now know how to design roles, manage state, and communicate precisely.

---

## 34. Double Logic Link: Interfaces and Mapping

In the DSA track, we solve **Design Twitter**. This is about defining **Interfaces**. A Tweet is a structured object. If your API doesn't follow the schema, the system crashes. Structured output is the "Interface of the Agent."

In the ML track, we look at **Recommendation Systems**. RecSys takes messy user history and maps it into **Structured Embeddings**. Structured output is how we map a user's natural language bio into structured "Interest Categories" that a RecSys can consume.


---

**Originally published at:** [arunbaby.com/ai-agents/0035-structured-output-patterns](https://www.arunbaby.com/ai-agents/0035-structured-output-patterns/)

*If you found this helpful, consider sharing it with others who might benefit.*

