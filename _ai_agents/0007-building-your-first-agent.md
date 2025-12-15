---
title: "Building Your First Agent"
day: 7
collection: ai_agents
categories:
  - ai-agents
tags:
  - tutorial
  - python
  - openai
  - react
  - basics
difficulty: Easy
---

**"Hello World? No, Hello Agent."**

## 1. Introduction: Demystifying the Magic

Enough theory. We've talked about ReAct loops, Memory buffers, and Framework architectures. Now, we build.

In this tutorial, we will build a **Research Agent** from scratch using **Raw Python**.
We will **NOT** use LangChain, LangGraph, or any other framework. Why? Because to master the frameworks, you must first understand the pain they solve. You need to feel the friction of managing chat history, parsing JSON, and handling errors manually. Only then will you appreciate the abstractions.

**The Mission:** Build an agent that can:
1.  Look up the current weather for a specific city.
2.  Search Wikipedia for fun facts about that city.
3.  Synthesize a "Travel Report" combining both.
4.  Handle errors gracefully (e.g., if the weather API fails).

By the end of this post, you will have a functioning `Agent` class that you can run locally.

---

## 2. The Setup

You will need:
*   Python 3.10+
*   `openai` library
*   `requests` (for our mock tools)
*   `termcolor` (to make the logs pretty)

```bash
pip install openai requests termcolor
```

## 3. Step 1: Defining the "Hands" (Tools)

Our agent is only as good as its tools. Let's define two simple Python functions.
In a real production system, these would call live APIs. For this tutorial, we will mock the returns to ensure deterministic behavior so you can reproduce the results.

```python
import json

# Mock Weather Tool (Real API requires keys)
def get_weather(city: str):
    """Get the current weather for a given city."""
    print(f"\n  > [Tool] Checking weather for {city}...")
    
    # Mock data to simulate API responses
    # In a real app, use requests.get(f"api.weather.com?q={city}")
    if "Tokyo" in city: 
        return json.dumps({"temp": 25, "condition": "Sunny", "humidity": "60%"})
    if "London" in city: 
        return json.dumps({"temp": 15, "condition": "Rainy", "humidity": "90%"})
    if "New York" in city: 
        return json.dumps({"temp": 20, "condition": "Cloudy", "humidity": "50%"})
        
    return json.dumps({"temp": 22, "condition": "Unknown", "note": "City not found in mock DB"})

# Mock Wikipedia Tool
def search_wikipedia(query: str):
    """Search Wikipedia for a query."""
    print(f"\n  > [Tool] Searching Wikipedia for {query}...")
    
    # Mock data
    return f"Wikipedia Summary: {query} is a famous location known for its culture, history, and vibrant economy."
```

## 4. Step 2: Defining the "Brain" (The Tool Schema)

We need to tell the LLM (GPT-4o) about these tools. We use the OpenAI `tools` schema. This is the "API Documentation" that the model reads to understand *how* to call our functions.

A critical part of "Prompt Engineering" is actually "Schema Engineering." The more descriptive your parameter descriptions are, the better the agent performs.

```python
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city. Always use the full city name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string", 
                        "description": "The city name, e.g. San Francisco or Tokyo"
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for broad historical and cultural information about a place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "The search term"
                    }
                },
                "required": ["query"],
            },
        },
    }
]
```

## 5. Step 3: The "Heart" (The Execution Loop)

This is the magic. We need a `while` loop that orchestrates the **Perception-Reasoning-Action (PRA)** cycle.

### 5.1 The Logic Flow
1.  **Initialize Memory:** Start with a System Prompt defining the Persona.
2.  **Call LLM:** Send the history to OpenAI.
3.  **Check Stop Reason:** Did the model generate text (`stop`) or ask for a tool (`tool_calls`)?
4.  **If Tool:** 
    *   Parse the function name and arguments from the JSON.
    *   Execute the Python function.
    *   Inject the result back into history as a `tool` role message.
    *   **Loop again** (so the model can see the result and decide what to do next).
5.  **If Text:** Print the answer and break the loop.

### 5.2 The Raw Implementation

```python
import os
from openai import OpenAI
from termcolor import colored

# Ensure you have your key set: export OPENAI_API_KEY="sk-..."
client = OpenAI() 

def run_agent(user_query):
    # 1. Initialize Memory
    messages = [
        {"role": "system", "content": "You are a helpful travel assistant. You have access to weather and wikipedia tools. You must use them to look up real information. Answer comprehensively."},
        {"role": "user", "content": user_query}
    ]

    print(colored(f"User: {user_query}", "green"))

    # The ReAct Loop
    step_count = 0
    while step_count < 10: # Safety break to prevent infinite loops
        # 2. CALL LLM
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto" 
        )
        
        msg = response.choices[0].message
        messages.append(msg) # CRITICAL: Update memory with the assistant's thought

        # 3. DECIDE: Did the model ask for a tool?
        if msg.tool_calls:
            print(colored(f"Agent Thought: I need to use tools...", "yellow"))
            
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                # Parse JSON arguments safely
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    print("Error: Model generated invalid JSON")
                    continue
                
                # 4. ACT: Execute the tool
                result = ""
                try:
                    if func_name == "get_weather":
                        result = get_weather(args["city"])
                    elif func_name == "search_wikipedia":
                        result = search_wikipedia(args["query"])
                    else:
                        result = f"Error: Tool {func_name} not found."
                except Exception as e:
                    result = f"Error executing tool: {str(e)}"
                
                # 5. OBSERVE: Feed result back to LLM
                # We must link the result to the specific tool_call_id
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        else:
            # No tool calls? The model is done. It has "Thought" enough.
            print(colored(f"Agent: {msg.content}", "cyan"))
            break
            
        step_count += 1

# Run it
if __name__ == "__main__":
    run_agent("Plan a trip to Tokyo for me. What's the weather and what is it famous for?")
```

### 5.3 Execution Trace
If you run this, you will see the logs:
1.  **Iter 0:** User asks.
2.  **Iter 1:** LLM outputs `ToolCall: get_weather('Tokyo')`.
3.  **Code:** Executes `get_weather`. Appends `{"temp": 25...}` to messages.
4.  **Iter 2:** LLM sees history + weather. Outputs `ToolCall: search_wikipedia('Tokyo')`.
5.  **Code:** Executes `search_wikipedia`. Appends result.
6.  **Iter 3:** LLM sees history + weather + wiki. Decides "I have enough info." Generates the final travel report. Loop breaks.

---

## 6. Refactoring: The Class-Based Agent (Production Ready)

The script above is fine for a demo, but messy for production. Let's refactor it into an Object-Oriented structure. This helps us manage **State** (memory) better foundation for frameworks like LangGraph.

### The Agent Class

```python
class Agent:
    def __init__(self, system_prompt, tools, model="gpt-4o"):
        self.system_prompt = system_prompt
        # Map function names to the actual callable functions
        self.tools = {t.__name__: t for t in tools} 
        self.model = model
        self.messages = []
        
    def _get_schema(self):
        """Helper to return the JSON schema."""
        # Ideally, generate this dynamically from self.tools
        return tools_schema 

    def run(self, query):
        """Main execution entry point."""
        self.messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": query}]
        
        steps = 0
        while steps < 10: 
            response = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self._get_schema()
            )
            msg = response.choices[0].message
            self.messages.append(msg)
            
            # If the model produced text, return it (unless it also called tools)
            if not msg.tool_calls:
                return msg.content
            
            # Execute Tools
            for tool_call in msg.tool_calls:
                self._execute_tool(tool_call)
            
            steps += 1
            
        return "Error: Max steps reached."

    def _execute_tool(self, tool_call):
        """Executes a single tool call and updates memory."""
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"  [Exec] {name}({args})")
        
        if name in self.tools:
            try:
                result = self.tools[name](**args)
            except Exception as e:
                result = f"Error: {e}"
        else:
            result = "Error: Tool not found"
            
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

# New Usage
my_agent = Agent(
    system_prompt="You are a research assistant. Be concise.",
    tools=[get_weather, search_wikipedia]
)
final_answer = my_agent.run("Tell me about London weather and history.")
print(f"Final Answer: {final_answer}")
```

## 7. The "Why Frameworks Exist" Moment

As you write this code, you start to notice friction points. This is why frameworks like **LangChain** and **LangGraph** exist. They solve the robust engineering problems you haven't faced yet:

1.  **Schema Auto-Generation:** Writing that `tools_schema` JSON manually is painful and error-prone. LangChain auto-generates it from your Python function type hints.
2.  **Memory Management:** Our `messages` list grows indefinitely. If the user chats for an hour, you'll hit the Context Window limit. Frameworks handle "Summary Memory" or "Sliding Window" automatically.
3.  **Broadcasting:** What if you want to stream the "Thinking..." tokens to a Frontend UI? Implementing SSE (Server Sent Events) for the intermediate thoughts is complex.
4.  **Parallelization:** GPT-4 can call 5 tools at once. Our loop executes them sequentially. A robust framework executes them in `asyncio.gather`.

## 8. Summary & Next Steps

Building an agent is not magic. It is just **Looping over an LLM**.
*   **Prompt** (Perception)
*   **LLM** (Reasoning)
*   **Code** (Action)
*   **Loop** (Feedback)

You have now built a Level 3 Agent. You understand the raw mechanics. When you switch to **LangGraph** in the next section, you will understand that it is simply a state machine wrapping this exact `while` loop logic with better persistence and typing.

With the basics built, we can now look at **Agent Workflow Patterns**, arranging these agents into "Editors," "Planners," and "Reviewers" to solve much harder problems.
