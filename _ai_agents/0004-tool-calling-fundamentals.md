---
title: "Tool Calling Fundamentals"
day: 4
collection: ai_agents
categories:
  - ai-agents
tags:
  - tools
  - function-calling
  - apis
  - security
  - pydantic
difficulty: Easy
---

**"Giving the Brain Hands to Act: The Interface Between Intelligence and Infrastructure."**

## 1. Introduction: From Text to Action

An LLM isolated in a box is just a text generator. It can hallucinatively describe the weather, but it cannot check the actual temperature. To become an **Agent**, it needs to interact with the outside world. This is achieved through **Tool Calling** (often called Function Calling).

Tool calling is the bridge between the **Probabilistic World** of AI (where 2+2 might equal 5 if the context is weird) and the **Deterministic World** of Software (where 2+2 is always 4). It is the mechanism that transforms an LLM from a "Chatbot" into a "Controller."

In this post, we will dissect the anatomy of a tool call, explore standard protocols like JSON Schema, and build a robust Python runtime that allows an agent to safely execute code.

---

## 2. The Mechanics of Tool Calling

How does a model "call" a function? It doesn't. It generates text that *you* parse.

### 2.1 The Lifecycle of an Action
Let's trace a user request: *"What is the weather in Tokyo?"* through the system.

1.  **Tool Definition:** You provide the model with a "menu" of functions in the System Prompt (usually in JSON format).
    *   *Prompt:* "You have a tool `get_weather(city: str)`."
2.  **Reasoning:** The model analyzes the user request against the menu. It determines that `get_weather` is the right match for "weather in Tokyo."
3.  **Generation:** The model generates a special token or formatted string (e.g., a JSON object) representing the intent to call the function:
    ```json
    { "tool": "get_weather", "arguments": { "city": "Tokyo" } }
    ```
4.  **Pause (Stop Sequence):** The inference engine recognizes that the model has output a tool call and *stops* generating. It returns control to your Python script.
5.  **Execution (The Runtime):**
    *   Your code (the "Orchestrator") parses the JSON.
    *   It validates the arguments (Is "Tokyo" a string?).
    *   It calls the *actual* Python function `requests.get(...)`.
    *   It captures the result: `{"temp": 18, "condition": "Cloudy"}`.
6.  **Context Injection:** You do not show the result to the user yet. You append a new message to the chat history:
    *   `Role: Tool`, `Content: {"temp": 18, "condition": "Cloudy"}`.
7.  **Final Response:** You invoke the model again. It sees the tool output and generates the final natural language answer: *"It's 18°C and cloudy in Tokyo."*

### 2.2 JSON Schema: The Protocol
Standardization is key. The industry has converged on **JSON Schema** (OpenAPI) to define tools. Models like GPT-4 are fine-tuned to read this specific format.

```json
{
  "name": "search_database",
  "description": "Searches the customer database for orders.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The exact SQL query to run."
      },
      "limit": {
        "type": "integer",
        "description": "Max results to return",
        "default": 10
      }
    },
    "required": ["query"]
  }
}
```

*   **Crucial Tip:** The `description` field is the instruction for the LLM. Be verbose here. "Search for orders" is bad. "Search for orders by customer ID or date using standard SQL syntax. Tables: orders, customers" is good. The description *is* the prompt for that tool.

---

## 3. Design Patterns for Reliability

LLMs are clumsy. They make mistakes. Your runtime must be defensive.

### 3.1 Pydantic Validation (The Shield)
LLMs hallucinate arguments. They might call `get_weather(location="Tokyo")` when your function expects `city="Tokyo"`.
*   **Pattern:** Wrap every tool in a **Pydantic Model**.
*   **Logic:** Before executing, pass the LLM's arguments into the Pydantic model.
*   **Self-Correction:** If validation fails, **catch the error and return it to the LLM**.
    *   *System Response:* `Error: Invalid argument 'location'. Did you mean 'city'?`
    *   *LLM:* "Ah, sorry." (Tries again with `city`).
This Self-Correction loop is vital. Without it, your agent crashes on 10% of queries.

### 3.2 Robust Outputs (Error Propagation)
What happens if your API returns a 500-line stack trace?
*   **Problem:** It runs up your token bill and fills the context window with noise. The agent uses the stack trace to hallucinate a weird answer.
*   **Fix:** Catch exceptions in the tool wrapper. Return a concise, safe string.
    *   *Bad:* `Traceback (most recent call last): File "main.py"...`
    *   *Good:* `Error: The database connection timed out. Please try again later.`
*   **Why:** This allows the Agent to reason about the failure ("Okay, I'll apologize to the user") rather than getting confused.

### 3.3 Atomic vs. Mega Tools
*   **Mega Tool:** `manage_user(action, id, data)` - Hard for the LLM to understand all the permutations. It has to guess the schema for `data` based on `action`.
*   **Atomic Tools:** `create_user`, `delete_user`, `update_email`.
*   **Rule:** Smaller, specific tools reduce hallucination rates. Adhere to the Single Responsibility Principle.

---

## 4. Scaling: The "Too Many Tools" Problem

GPT-4 has a context limit. If you have 5,000 internal APIs (like AWS or Stripe), you cannot paste all 5,000 JSON schemas into the prompt. It would cost $5 per query.

### 4.1 Solution: Tool RAG (Retrieval)
Treat tool definitions like documents.
1.  **Embed** the descriptions of all 5,000 tools into a Vector Database.
2.  When a user query comes in ("How do I refund a charge?"), embed the query.
3.  **Search** the Vector DB for the top 5 most relevant tools (`stripe_refund`, `stripe_get_charge`, etc.).
4.  **Inject** only those 5 definitions into the context prompt.
This allows agents to have infinite toolkits.

### 4.2 Handling Asynchronous Actions
Some tools take time (e.g., `generate_video`, `provision_server`). You can't keep the HTTP connection open for 10 minutes.
*   **Pattern:**
    1.  Agent calls `start_job()`.
    2.  Tool returns immediately: `{"job_id": "123", "status": "pending"}`.
    3.  Agent reasons: "I have started the job. I will check back later."
    4.  Agent (or a scheduler) periodically calls `check_status(job_id)`.

---

## 5. Security: The Sandbox

The most dangerous tool is `run_python_code`.
If an agent can run code, it can `os.system('rm -rf /')` or `os.environ['AWS_KEY']`.

### 5.1 Sandboxing Strategies
**NEVER run agent code on your host.**
1.  **Transational Containers:** Spin up a Docker container for the session. Destroy it after.
2.  **WebAssembly (Wasm):** Run code in a browser-like sandbox (e.g., Pyodide).
3.  **Cloud Sandboxes:** Use services like **E2B** or **Modal** that provide secure, isolated VMs specifically for AI agents.

### 5.2 Permission Scopes & HITL
*   **Read-Only:** Give agents "Viewer" access by default.
*   **Human-in-the-Loop (HITL):** Tag dangerous tools (`delete_db`, `send_money`) as `requires_approval=True`.
    *   When the agent calls it, the runtime pauses.
    *   The user gets a pop-up: "Agent wants to delete DB. Allow?"
    *   Only on "Yes" does the code execute.

---

## 6. Code: A Semantic Tool Router

Let's build a simple tool executor in Python using Type Hints.

```python
import inspect
import json
from typing import Callable, Any

class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, func: Callable):
        """Decorator to register a function as a tool."""
        self.tools[func.__name__] = func
        return func

    def get_definitions(self):
        """Generate JSON schemas for the LLM."""
        definitions = []
        for name, func in self.tools.items():
            # Basic introspection (in production, use Pydantic)
            sig = inspect.signature(func)
            doc = func.__doc__ or "No description."
            definitions.append({
                "name": name,
                "description": doc,
                "parameters": str(sig) # Simplified for demo
            })
        return definitions

    def execute(self, tool_name: str, **kwargs):
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."
        
        try:
            # Here we would add Pydantic validation
            return self.tools[tool_name](**kwargs)
        except TypeError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Runtime Error: {str(e)}"

# Usage
registry = ToolRegistry()

@registry.register
def get_weather(city: str, unit: str = "celsius"):
    """Fetches weather for a city. Args: city (str), unit (str)"""
    # Simulate API call
    return f"Weather in {city} is 20 degrees {unit}"

@registry.register
def search_web(query: str):
    """Searches the internet."""
    return f"Results for {query}..."

# 1. Get definitions to send to LLM
print("Sending to LLM:", json.dumps(registry.get_definitions(), indent=2))

# 2. Simulate LLM deciding to call a tool (The Model Output)
response_tool = "get_weather"
response_args = {"city": "Paris", "unit": "celsius"}

# 3. Execute
print(f"Executing {response_tool}...")
result = registry.execute(response_tool, **response_args)
print("Result:", result)
```

---

## 7. Summary

Tool Calling is what makes AI useful.
*   It requires **Standardized Interfaces** (JSON Schema).
*   It demands **Defensive Coding** (Validation, Error Handling).
*   It necessitates **Strict Security** (Sandboxing, Permissions).

By mastering these fundamentals, you can build agents that don't just talk, but *do*—transforming business workflows from manual drudgery to autonomous execution.

In the next post, we will tackle the final challenge of the agent loop: **Memory Architectures**—how to remember what you did yesterday.
