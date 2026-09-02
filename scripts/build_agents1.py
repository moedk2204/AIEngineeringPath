import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nbutil import md, code, write

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "06-Agents-LangChain-LangGraph")

# ---------------------------------------------------------------- 01
write(os.path.join(OUT, "01-langchain-fundamentals.ipynb"), [
md('''
# 01 - How LangChain Actually Works

LangChain has a reputation for being a confusing pile of abstractions. Most of that
confusion comes from tutorials written for LangChain 0.x, whose central object —
`AgentExecutor` — no longer exists in modern LangChain.

This notebook covers the small number of pieces that actually matter in **LangChain 1.x**.

By the end you will be able to:
1. Say what LangChain gives you that calling a provider SDK directly does not.
2. Use messages, models, prompts, tools and structured output.
3. Read a tool call off a model response and execute it.
4. Recognise deprecated 0.x patterns when you meet them in old tutorials.

> **No API key needed.** Every cell runs against a scripted stand-in model
> (`offline_llm.py`). Swap in a real provider at any point with `real_model()`.
'''),
code('''
import sys, os
sys.path.insert(0, os.path.abspath("."))     # so `offline_llm` is importable

from offline_llm import ScriptedChatModel, ai, tool_call
import langchain
print("langchain version:", langchain.__version__)
'''),
md('''
## 1. What LangChain is for

Strip away the marketing and LangChain does three useful things:

1. **One interface over every provider.** The same code runs against OpenAI, Anthropic,
   Google or a local Ollama model. Swapping provider is a one-line change.
2. **A standard message and tool-call format.** Every provider has its own JSON shape for
   tool calls. LangChain normalises them, so your tool-handling code is written once.
3. **Composable pieces.** Prompts, models, parsers and tools snap together.

That is genuinely useful. What it is **not** is a reason to use LangChain for a single
API call — for that, the provider's own SDK is simpler. LangChain earns its place when you
have tools, multiple steps, or want provider portability.

## 2. Models

`init_chat_model` builds a model from a `"provider:model"` string.

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4.1-mini")
llm = init_chat_model("anthropic:claude-opus-5")
llm = init_chat_model("google_genai:gemini-2.5-flash")
llm = init_chat_model("ollama:llama3.2")          # local, free, no key
```

Everything downstream is identical regardless of which line you picked. That portability
is the single best reason to use LangChain.

For this notebook we use a scripted stand-in so nothing needs a key.
'''),
code('''
llm = ScriptedChatModel(responses=[
    ai("Machine learning is a way of writing programs that learn patterns from data "
       "instead of being given explicit rules."),
])

response = llm.invoke("Explain machine learning in one sentence.")
print(type(response).__name__)
print()
print(response.content)
'''),
md('''
## 3. Messages

A conversation is a list of typed messages. There are four you will use constantly:

| Message | Who wrote it | Purpose |
|---|---|---|
| `SystemMessage` | you | standing instructions, persona, rules |
| `HumanMessage` | the user | the request |
| `AIMessage` | the model | the reply — **and any tool calls** |
| `ToolMessage` | your code | the result of running a tool |

The whole agent loop in notebook 03 is just these four types in a cycle.
'''),
code('''
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

conversation = [
    SystemMessage("You are a concise data science tutor. Never write more than two sentences."),
    HumanMessage("What is overfitting?"),
]

llm = ScriptedChatModel(responses=[
    ai("Overfitting is when a model memorises the training data instead of the pattern. "
       "You spot it when training loss keeps falling while validation loss rises."),
])

reply = llm.invoke(conversation)
conversation.append(reply)

for m in conversation:
    print(f"{type(m).__name__:15} {m.content[:80]}")
'''),
md('''
> **The model is stateless.** It has no memory between calls — you resend the whole list
> every time. "Memory" in an agent framework means *managing that list for you*. That is
> what LangGraph's checkpointer does in notebook 02.

## 4. Prompt templates

A template is a string with slots, so you write the prompt once and fill it per request.
'''),
code('''
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in at most {max_words} words."),
    ("human", "{question}"),
])

messages = prompt.invoke({
    "role": "machine learning tutor",
    "max_words": 30,
    "question": "What does a learning rate do?",
})

for m in messages.messages:
    print(f"{type(m).__name__:15} {m.content}")
'''),
md('''
## 5. Chaining with `|`

The pipe operator connects components: the output of one becomes the input of the next.
This is LangChain Expression Language (LCEL).
'''),
code('''
from langchain_core.output_parsers import StrOutputParser

llm = ScriptedChatModel(responses=[
    ai("It controls how big a step the optimiser takes on each update."),
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "role": "machine learning tutor",
    "max_words": 30,
    "question": "What does a learning rate do?",
})
print(repr(result))       # a plain string, not an AIMessage - the parser unwrapped it
'''),
md('''
Read `prompt | llm | StrOutputParser()` as "fill the template, send it to the model,
then pull the text out". Every chain also gets `.stream()` and `.batch()` for free.

**When to stop chaining.** LCEL is a straight line: A then B then C. The moment you need a
**loop** (call a tool, look at the result, maybe call another) or a **branch** (route to a
different path depending on the answer), you have outgrown chains. That is exactly the
point where LangGraph starts — notebook 02.

## 6. Tools

A tool is a Python function the model is allowed to ask for. The `@tool` decorator turns
a function into one.
'''),
code('''
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    fake = {"beirut": "28C and sunny", "london": "12C and raining", "cairo": "35C and clear"}
    return fake.get(city.lower(), f"No data for {city}")

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

print("name       :", get_weather.name)
print("description:", get_weather.description)
print("args schema:", get_weather.args)
'''),
md('''
> **The docstring is not documentation — it is the prompt.** The model decides whether to
> call your tool based on its name, description and argument names. A vague docstring
> produces a tool the model never calls, or calls at the wrong moment. This is the single
> highest-leverage thing you control when an agent misbehaves.

### The model does not run your tool

This trips up everyone once. The model **requests** a call; **your code** executes it and
sends the result back. Nothing runs automatically.
'''),
code('''
llm = ScriptedChatModel(responses=[
    ai("", [tool_call("get_weather", city="Beirut")]),
])

model_with_tools = llm.bind_tools([get_weather, multiply])
response = model_with_tools.invoke([HumanMessage("What is the weather in Beirut?")])

print("content    :", repr(response.content))    # empty - the model wants a tool first
print("tool_calls :", response.tool_calls)
'''),
code('''
# Your code executes it and returns a ToolMessage.
call = response.tool_calls[0]
result = get_weather.invoke(call["args"])

tool_message = ToolMessage(content=str(result), tool_call_id=call["id"])
print("executed   :", call["name"], call["args"])
print("result     :", result)
print("wrapped as :", type(tool_message).__name__)
'''),
md('''
That cycle — **model asks → you execute → you send the result back → model answers** — is
the entire idea of an agent. Everything in notebook 03 is that loop, automated.

## 7. Structured output

Free text is hard to program against. `with_structured_output` makes the model fill in a
schema instead, and you get back a validated Python object.
'''),
code('''
from pydantic import BaseModel, Field
from typing import Literal

class Ticket(BaseModel):
    """A support ticket extracted from a customer message."""
    category: Literal["billing", "technical", "account", "other"] = Field(
        description="Which team should handle this")
    urgency: Literal["low", "medium", "high"] = Field(description="How fast it needs a reply")
    summary: str = Field(description="One-line summary of the problem")

print("Schema the model must fill:")
for name, field in Ticket.model_fields.items():
    print(f"  {name:10} {field.annotation}")

print("""
With a real model:

    structured = llm.with_structured_output(Ticket)
    ticket = structured.invoke("I have been charged twice this month and nobody replied!")
    ticket.category    # 'billing'
    ticket.urgency     # 'high'

You get a validated Ticket object, not a string you have to parse.""")
'''),
md('''
**Use structured output whenever the result feeds other code.** Classification, extraction,
routing decisions, filling a database row. Parsing free text with regexes is a bug waiting
to happen.

## 8. Recognising deprecated LangChain 0.x code

You will meet a lot of it. `tool-calling-legacy/exercise1.py` in this folder is written in
the old style. Here is the translation table:

| LangChain 0.x (deprecated) | LangChain 1.x |
|---|---|
| `from langchain.agents import AgentExecutor` | `from langchain.agents import create_agent` |
| `create_tool_calling_agent(llm, tools, prompt)` | `create_agent(model, tools, system_prompt=...)` |
| `AgentExecutor(agent=..., tools=...)` | `create_agent(...)` returns a ready graph |
| `ConversationBufferMemory` | a LangGraph **checkpointer** + `thread_id` |
| `LLMChain(llm=..., prompt=...)` | `prompt \\| llm` |
| `initialize_agent(...)` | `create_agent(...)` |

```python
# --- OLD (do not write this) ---
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, memory=ConversationBufferMemory())
executor.invoke({"input": "..."})

# --- NEW ---
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(model, tools, system_prompt="...", checkpointer=InMemorySaver())
agent.invoke({"messages": [{"role": "user", "content": "..."}]},
             config={"configurable": {"thread_id": "user-123"}})
```

The big conceptual change: **an agent is now a graph**, not an executor object. That is why
notebook 02 is about graphs before notebook 03 builds an agent.

---
## Exercises

1. Write a `@tool` called `search_papers(topic: str, year: int)`. Print `.args` and
   `.description`. Now rewrite the docstring to be vague and compare — what would the model
   lose?
2. Build a chain `prompt | llm | StrOutputParser()` that summarises a paragraph in exactly
   three bullet points. Test the template with `.invoke()` alone first.
3. Extend `Ticket` with a `customer_sentiment` field. Which fields need a `description` for
   the model to fill them correctly, and which are obvious from the name?
4. Open `tool-calling-legacy/exercise1.py` and rewrite it using `create_agent`. Use the
   table above. You will remove more lines than you add.
5. **Stretch:** copy `.sample.env` to `.env`, add one real API key, and re-run sections 2-7
   with `real_model()` instead of `ScriptedChatModel`. Nothing else should need to change —
   that is the portability claim, tested.

## Next

Notebook 02 introduces LangGraph, which is what you use once a straight-line chain is not
enough.
'''),
])

# ---------------------------------------------------------------- 02
write(os.path.join(OUT, "02-langgraph-basics.ipynb"), [
md('''
# 02 - LangGraph Basics

A chain is a straight line. Real applications need **loops** (try a tool, look at the
result, maybe try another) and **branches** (route this question to a different path).

LangGraph gives you a state machine: **state** that flows through **nodes** connected by
**edges**. Modern LangChain agents are built on it — `create_agent` returns a LangGraph
graph — so understanding this notebook is what makes notebook 03 readable rather than magic.

By the end you will be able to:
1. Define graph state with a `TypedDict` and a reducer.
2. Add nodes and edges, and route conditionally.
3. Compile, invoke, stream and visualise a graph.
4. Add persistent memory with a checkpointer and `thread_id`.

> **No API key needed.** Most of this notebook is plain Python functions — that is the
> point. A node does not have to call an LLM.
'''),
code('''
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

import langgraph
print("langgraph ready")
'''),
md('''
## 1. State

Every graph has one shared **state** object. Each node receives it and returns a partial
update, which LangGraph merges in.

By default a returned key **overwrites** the old value. That is wrong for a message list —
you want to *append*. A **reducer** controls the merge, and `add_messages` is the built-in
one for conversations.
'''),
code('''
class CounterState(TypedDict):
    value: int              # no reducer -> each node overwrites this
    log: Annotated[list, lambda old, new: old + new]     # reducer -> appends

def double(state: CounterState) -> dict:
    return {"value": state["value"] * 2, "log": [f"doubled to {state['value'] * 2}"]}

def add_ten(state: CounterState) -> dict:
    return {"value": state["value"] + 10, "log": [f"added 10 -> {state['value'] + 10}"]}

builder = StateGraph(CounterState)
builder.add_node("double", double)
builder.add_node("add_ten", add_ten)
builder.add_edge(START, "double")
builder.add_edge("double", "add_ten")
builder.add_edge("add_ten", END)

graph = builder.compile()
result = graph.invoke({"value": 5, "log": []})

print("final value:", result["value"])
print("log:")
for line in result["log"]:
    print("  ", line)
'''),
md('''
Notice the difference: `value` was replaced at each step (5 → 10 → 20), while `log`
accumulated. **That distinction is the most common source of LangGraph bugs** — a message
list without `add_messages` silently loses history.

## 2. The anatomy of a graph

| Piece | What it is |
|---|---|
| **State** | a `TypedDict` describing everything that flows through |
| **Node** | a function `state -> partial state update` |
| **Edge** | "after this node, go to that one" |
| **Conditional edge** | a function that *decides* which node comes next |
| `START` / `END` | the entry and exit points |
| `.compile()` | validates the wiring and returns a runnable graph |

A node is an ordinary Python function. It does not have to touch an LLM — it can query a
database, call an API, or do arithmetic.

## 3. Conditional edges — routing

This is the thing a chain cannot do. A routing function inspects the state and returns the
**name** of the next node.
'''),
code('''
class TicketState(TypedDict):
    text: str
    category: str
    response: str

def classify(state: TicketState) -> dict:
    """In a real app this would be an LLM call with structured output."""
    text = state["text"].lower()
    if any(w in text for w in ["charge", "refund", "invoice", "bill"]):
        category = "billing"
    elif any(w in text for w in ["error", "crash", "bug", "broken"]):
        category = "technical"
    else:
        category = "general"
    return {"category": category}

def handle_billing(state: TicketState) -> dict:
    return {"response": "Routed to the billing team. Expect a reply within 4 hours."}

def handle_technical(state: TicketState) -> dict:
    return {"response": "Routed to engineering. Please attach your log file."}

def handle_general(state: TicketState) -> dict:
    return {"response": "Thanks for getting in touch - a human will reply shortly."}

def route(state: TicketState) -> Literal["billing", "technical", "general"]:
    """The routing function returns the NAME of the next node."""
    return state["category"]

builder = StateGraph(TicketState)
builder.add_node("classify", classify)
builder.add_node("billing", handle_billing)
builder.add_node("technical", handle_technical)
builder.add_node("general", handle_general)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route)      # <- the branch
for node in ["billing", "technical", "general"]:
    builder.add_edge(node, END)

support_graph = builder.compile()

for text in ["I was charged twice this month!",
             "The export button crashes the app",
             "Do you have an office in Beirut?"]:
    out = support_graph.invoke({"text": text, "category": "", "response": ""})
    print(f"{out['category']:10} | {out['response']}")
'''),
md('''
## 4. Visualising the graph

A compiled graph can draw itself. Reading the picture is the fastest way to spot wiring
mistakes.
'''),
code('''
print(support_graph.get_graph().draw_mermaid())
'''),
md('''
Paste that into [mermaid.live](https://mermaid.live) to see it rendered. In Jupyter with
`pygraphviz` installed you can also use:

```python
from IPython.display import Image
Image(support_graph.get_graph().draw_mermaid_png())
```

## 5. Loops — what makes it an *agent*

A conditional edge can point **backwards**. That creates a cycle, which is how an agent
keeps calling tools until it is done. This is the structure `create_agent` builds for you.
'''),
code('''
class RetryState(TypedDict):
    attempts: int
    solved: bool
    history: Annotated[list, lambda old, new: old + new]

def attempt(state: RetryState) -> dict:
    n = state["attempts"] + 1
    solved = n >= 3                       # pretend it succeeds on the third try
    return {"attempts": n, "solved": solved,
            "history": [f"attempt {n}: {'success' if solved else 'failed'}"]}

def should_retry(state: RetryState) -> Literal["attempt", "__end__"]:
    if state["solved"]:
        return END
    if state["attempts"] >= 5:            # ALWAYS bound your loops
        return END
    return "attempt"

builder = StateGraph(RetryState)
builder.add_node("attempt", attempt)
builder.add_edge(START, "attempt")
builder.add_conditional_edges("attempt", should_retry)

retry_graph = builder.compile()
out = retry_graph.invoke({"attempts": 0, "solved": False, "history": []})

for line in out["history"]:
    print(" ", line)
print("\\nsolved:", out["solved"], "after", out["attempts"], "attempts")
'''),
md('''
> **Always bound your loops.** An agent whose exit condition never fires will burn tokens
> until it hits a recursion limit. LangGraph defaults to 25 steps, but write your own
> ceiling — as the `attempts >= 5` line does above.

## 6. Streaming

`.invoke()` waits for the whole run. `.stream()` yields as it goes, which is what you need
for a UI that shows progress.
'''),
code('''
print("stream_mode='updates' - what each node changed:\\n")
for chunk in support_graph.stream(
        {"text": "The export button crashes", "category": "", "response": ""},
        stream_mode="updates"):
    for node_name, update in chunk.items():
        print(f"  {node_name:10} -> {update}")

print("\\nstream_mode='values' - the whole state after each step:\\n")
for snapshot in support_graph.stream(
        {"text": "The export button crashes", "category": "", "response": ""},
        stream_mode="values"):
    print("  ", snapshot)
'''),
md('''
| `stream_mode` | Yields | Use for |
|---|---|---|
| `"updates"` | just what each node changed | logging, showing progress |
| `"values"` | the full state after each step | debugging |
| `"messages"` | LLM tokens as they generate | a chat UI that types |

## 7. Memory — checkpointers and `thread_id`

This is how "the agent remembers" actually works, and it replaces LangChain 0.x's
`ConversationBufferMemory`.

A **checkpointer** saves the state after every step. A **`thread_id`** names a conversation.
Same `thread_id` means the state is loaded back; a different one starts fresh.
'''),
code('''
from langchain.messages import HumanMessage, AIMessage

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]     # the reducer that appends

def respond(state: ChatState) -> dict:
    turn = len([m for m in state["messages"] if isinstance(m, HumanMessage)])
    return {"messages": [AIMessage(f"(reply {turn}) I have {len(state['messages'])} messages in context.")]}

builder = StateGraph(ChatState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

chat = builder.compile(checkpointer=InMemorySaver())      # <- memory switched on

alice = {"configurable": {"thread_id": "alice"}}

for text in ["Hello", "What is my name?", "Still there?"]:
    out = chat.invoke({"messages": [HumanMessage(text)]}, config=alice)
    print(f"user: {text:20} agent: {out['messages'][-1].content}")

print("\\nTotal messages remembered on alice's thread:", len(out["messages"]))
'''),
code('''
# A different thread_id is a different conversation - nothing is shared.
bob = {"configurable": {"thread_id": "bob"}}
out_bob = chat.invoke({"messages": [HumanMessage("Hi")]}, config=bob)

print("bob's thread :", len(out_bob["messages"]), "messages")
print("alice's thread:", len(chat.get_state(alice).values["messages"]), "messages")
'''),
md('''
> **`InMemorySaver` disappears when the process exits.** For anything real use a durable
> checkpointer — `langgraph-checkpoint-sqlite` for a single machine,
> `langgraph-checkpoint-postgres` for a server. The code change is one line.

### Inspecting and rewinding

Because every step is checkpointed, you can look at the state, and even go back to an
earlier point and take a different path.
'''),
code('''
state = chat.get_state(alice)
print("next node to run:", state.next)
print("messages so far :", len(state.values["messages"]))

print("\\ncheckpoint history (newest first):")
for i, snapshot in enumerate(chat.get_state_history(alice)):
    print(f"  {i}: {len(snapshot.values['messages'])} messages")
    if i >= 4:
        break
'''),
md('''
---
## Exercises

1. Remove the reducer from `log` in section 1 (make it a plain `list`). Run it again and
   explain what happens to the log.
2. Add a fourth category `"sales"` to the support graph, with its own node and keyword
   rule. Redraw the mermaid diagram to confirm the wiring.
3. In the retry graph, change `attempt` so it never succeeds and remove the
   `attempts >= 5` guard. What happens, and what error do you get? Now put the guard back.
4. Build a two-node graph where node A fetches a number and node B decides — via a
   conditional edge — whether to go back to A or finish. Bound it at 10 iterations.
5. Add a `user_name` field to `ChatState` and have the node fill it from the first message.
   Confirm it persists across turns on the same `thread_id` but not across different ones.
6. **Stretch:** install `langgraph-checkpoint-sqlite` and swap `InMemorySaver` for
   `SqliteSaver`. Restart the kernel, re-run with the same `thread_id`, and confirm the
   conversation survived.

## Next

Notebook 03 builds a real tool-using agent — first with `create_agent` in four lines, then
hand-built as a `StateGraph` so you can see that those four lines produce exactly the graph
you just learned to draw.
'''),
])
