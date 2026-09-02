import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nbutil import md, code, write

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "06-Agents-LangChain-LangGraph")

HEAD = '''
import sys, os
sys.path.insert(0, os.path.abspath("."))

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from offline_llm import ScriptedChatModel, ai, tool_call
print("ready")
'''

TOOLS = '''
@tool
def search_docs(query: str) -> str:
    """Search the internal knowledge base for a topic. Use for company policy questions."""
    kb = {
        "refund": "Refunds are issued within 14 days of purchase, no questions asked.",
        "shipping": "Standard shipping takes 3-5 business days. Express is next-day.",
        "warranty": "All hardware carries a 2-year warranty from the delivery date.",
    }
    for key, value in kb.items():
        if key in query.lower():
            return value
    return "No matching document found."

@tool
def get_order_status(order_id: str) -> str:
    """Look up the delivery status of an order by its ID."""
    orders = {"A-1001": "shipped, arriving Tuesday",
              "A-1002": "processing",
              "A-1003": "delivered on 12 March"}
    return orders.get(order_id.upper(), f"No order found with ID {order_id}")

@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '119.98 * 0.15'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:          # never exec() raw model output
        return "Refused: expression contains disallowed characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Error: {exc}"

TOOLS = [search_docs, get_order_status, calculate]
print("tools:", [t.name for t in TOOLS])
'''

# ---------------------------------------------------------------- 03
write(os.path.join(OUT, "03-build-a-real-agent.ipynb"), [
md('''
# 03 - Building a Real Agent with LangGraph

Now the two notebooks come together. You will build the same tool-using agent **twice**:

1. With `create_agent` — four lines.
2. By hand as a `StateGraph` — about thirty lines.

The second version is the point. `create_agent` is a convenience wrapper that builds
exactly the graph you are about to draw yourself, and knowing that is the difference
between using an agent and being able to debug one.

> **No API key needed.** A scripted model drives the loop so you can watch every step.
> Everything works identically with a real model — swap in `real_model()`.
'''),
code(HEAD),
md('''
## 1. The tools

Three tools a customer-support agent would plausibly need.
'''),
code(TOOLS),
md('''
> **Note the guard in `calculate`.** The model chooses the string that gets evaluated.
> Never pass model output to `eval`, `exec`, `os.system` or a raw SQL string without
> validating it. Treat every tool argument as untrusted user input — because it is.

## 2. Version one — `create_agent`
'''),
code('''
from langchain.agents import create_agent

# The script the stand-in model will follow: look up the order, then answer.
llm = ScriptedChatModel(responses=[
    ai("", [tool_call("get_order_status", order_id="A-1001")]),
    ai("Your order A-1001 has shipped and should arrive on Tuesday."),
])

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt="You are a support agent. Use tools to look up facts. Never guess.",
)

result = agent.invoke({"messages": [HumanMessage("Where is my order A-1001?")]})

for m in result["messages"]:
    kind = type(m).__name__
    if kind == "AIMessage" and m.tool_calls:
        print(f"{kind:13} -> calls {m.tool_calls[0]['name']}({m.tool_calls[0]['args']})")
    else:
        print(f"{kind:13} {m.content}")
'''),
md('''
Four lines of setup, and the full loop ran: the model asked for a tool, LangGraph executed
it, fed the `ToolMessage` back, and the model wrote the final answer.

### What `create_agent` actually returns

A compiled LangGraph graph. Look at it.
'''),
code('''
print(agent.get_graph().draw_mermaid())
'''),
md('''
Read that diagram against notebook 02: a **model** node, a **tools** node, a **conditional
edge** from the model that either goes to tools or to `END`, and an edge from tools **back**
to the model. That backward edge is the loop.

## 3. Version two — the same agent, by hand

Now build it yourself. This is the whole of "what an agent is".
'''),
code('''
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

tools_by_name = {t.name: t for t in TOOLS}

def call_model(state: AgentState) -> dict:
    """Node 1: ask the LLM what to do next."""
    system = SystemMessage("You are a support agent. Use tools to look up facts. Never guess.")
    response = model_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState) -> dict:
    """Node 2: run whatever the LLM asked for, return one ToolMessage per call."""
    last = state["messages"][-1]
    results = []
    for call in last.tool_calls:
        output = tools_by_name[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """The conditional edge: did the model ask for a tool, or is it done?"""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

builder = StateGraph(AgentState)
builder.add_node("model", call_model)
builder.add_node("tools", call_tools)

builder.add_edge(START, "model")
builder.add_conditional_edges("model", should_continue)
builder.add_edge("tools", "model")               # <- the loop back

hand_built = builder.compile()
print(hand_built.get_graph().draw_mermaid())
'''),
code('''
llm = ScriptedChatModel(responses=[
    ai("", [tool_call("search_docs", query="refund policy")]),
    ai("You can return it for a full refund within 14 days of purchase."),
])
model_with_tools = llm.bind_tools(TOOLS)

out = hand_built.invoke({"messages": [HumanMessage("Can I get a refund?")]})

for m in out["messages"]:
    kind = type(m).__name__
    if kind == "AIMessage" and m.tool_calls:
        print(f"{kind:13} -> {m.tool_calls[0]['name']}({m.tool_calls[0]['args']})")
    else:
        print(f"{kind:13} {m.content}")
'''),
md('''
**That is an agent.** Two nodes and a conditional edge that loops. Everything else — memory,
streaming, human approval, multi-agent — is elaboration on this skeleton.

`ToolNode` from `langgraph.prebuilt` replaces `call_tools` with a tested implementation
that also handles errors and parallel calls:

```python
builder.add_node("tools", ToolNode(TOOLS))
```

## 4. Multi-step reasoning

The loop runs as many times as the model needs. Here it uses two tools in sequence.
'''),
code('''
llm = ScriptedChatModel(responses=[
    ai("", [tool_call("get_order_status", "c1", order_id="A-1003")]),
    ai("", [tool_call("calculate", "c2", expression="119.98 * 0.15")]),
    ai("Order A-1003 was delivered on 12 March. A 15% restocking fee on $119.98 "
       "is $18.00, so you would be refunded $101.98."),
])
model_with_tools = llm.bind_tools(TOOLS)

out = hand_built.invoke({"messages": [
    HumanMessage("Order A-1003 - when did it arrive, and what is 15% of $119.98?")
]})

print(f"{len(out['messages'])} messages, {sum(1 for m in out['messages'] if type(m).__name__ == 'ToolMessage')} tool calls\\n")
for m in out["messages"]:
    kind = type(m).__name__
    if kind == "AIMessage" and m.tool_calls:
        print(f"  {kind:13} -> {m.tool_calls[0]['name']}({m.tool_calls[0]['args']})")
    else:
        print(f"  {kind:13} {str(m.content)[:90]}")
'''),
md('''
The graph went model → tools → model → tools → model, deciding at each pass. **Nobody wrote
that sequence.** That is what distinguishes an agent from a chain, and it is also why
agents are harder to test — the path is not fixed.

## 5. Memory

Same mechanism as notebook 02: a checkpointer plus a `thread_id`.
'''),
code('''
llm = ScriptedChatModel(responses=[
    ai("", [tool_call("get_order_status", order_id="A-1002")]),
    ai("Order A-1002 is still processing."),
    ai("You asked about order A-1002 a moment ago."),
])
model_with_tools = llm.bind_tools(TOOLS)

memory_agent = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "customer-42"}}

r1 = memory_agent.invoke({"messages": [HumanMessage("Status of A-1002?")]}, config=config)
print("turn 1:", r1["messages"][-1].content)

r2 = memory_agent.invoke({"messages": [HumanMessage("What did I just ask about?")]}, config=config)
print("turn 2:", r2["messages"][-1].content)

print(f"\\nthread now holds {len(r2['messages'])} messages")
'''),
md('''
## 6. Streaming the agent's work

Users will not wait in silence while an agent runs three tools. Stream the steps.
'''),
code('''
llm = ScriptedChatModel(responses=[
    ai("", [tool_call("search_docs", query="warranty")]),
    ai("Hardware is covered by a 2-year warranty from delivery."),
])
model_with_tools = llm.bind_tools(TOOLS)

for chunk in hand_built.stream(
        {"messages": [HumanMessage("How long is the warranty?")]},
        stream_mode="updates"):
    for node, update in chunk.items():
        msg = update["messages"][-1]
        if getattr(msg, "tool_calls", None):
            print(f"[{node}] calling {msg.tool_calls[0]['name']}...")
        else:
            print(f"[{node}] {str(msg.content)[:80]}")
'''),
md('''
## 7. Structured output from an agent

An agent that returns free text is hard to plug into other systems. `response_format`
makes the final answer a validated object.
'''),
code('''
from pydantic import BaseModel, Field

class SupportAnswer(BaseModel):
    """The final structured answer from the support agent."""
    answer: str = Field(description="The reply to show the customer")
    used_tools: list[str] = Field(description="Names of tools consulted")
    needs_human: bool = Field(description="True if a human agent must follow up")

print("""With a real model:

    agent = create_agent(
        model=real_model("openai:gpt-4.1-mini"),
        tools=TOOLS,
        system_prompt="You are a support agent.",
        response_format=SupportAnswer,
    )
    result = agent.invoke({"messages": [HumanMessage("Where is order A-1001?")]})
    result["structured_response"].needs_human      # -> False

The agent runs its tool loop as usual, then fills the schema at the end.""")
'''),
md('''
## 8. `create_agent` or hand-built?

| Use `create_agent` | Build a `StateGraph` |
|---|---|
| one agent, one tool loop | several agents handing off |
| the standard ReAct pattern | custom routing before or after the model |
| you want it working today | a human approval step mid-run |
| — | non-LLM nodes (database, validation, retries) |

**Start with `create_agent`.** Drop to a `StateGraph` when you hit something it will not
express — and now you know that dropping down is a small step, not a rewrite.

---
## Exercises

1. Add a `create_ticket(summary: str, urgency: str)` tool that appends to a Python list.
   Script the model to call it, and confirm the list changed.
2. Replace the hand-written `call_tools` with `ToolNode(TOOLS)`. Confirm the output is
   identical.
3. Make one tool raise an exception. What happens with your `call_tools`? What happens with
   `ToolNode`? Which behaviour do you want in production?
4. Add a `max_steps` counter to `AgentState` and make `should_continue` end the run after
   4 tool calls. Why does every production agent need this?
5. Script the model to call **two** tools in a single `AIMessage` (pass two entries to
   `tool_calls`). Does your `call_tools` handle it? Does `ToolNode`?
6. **Stretch:** add a `validate` node between `tools` and `model` that rejects a
   `ToolMessage` longer than 500 characters and replaces it with a truncation notice.
   This is real context management — see module 07.

## Next

Notebook 04 covers the patterns that turn this into something you would actually deploy:
human approval, multi-agent handoff, error handling and observability.
'''),
])

# ---------------------------------------------------------------- 04
write(os.path.join(OUT, "04-agent-patterns-and-production.ipynb"), [
md('''
# 04 - Agent Patterns & Getting to Production

Notebook 03 built a working agent. This notebook covers what stands between a working
agent and one you would let customers touch.

1. Human-in-the-loop approval for dangerous actions.
2. Multi-agent supervision.
3. Error handling and bounded loops.
4. Observability and cost.
5. When not to build an agent at all.
'''),
code(HEAD),
code(TOOLS),
md('''
## 1. Human-in-the-loop

Some tool calls should not happen without a person saying yes: refunding money, sending
email, deleting records, writing to production.

LangGraph's `interrupt()` **pauses the graph mid-run**, saves the state to the checkpointer,
and returns control to your code. You resume with a `Command`. This only works with a
checkpointer, because the run has to survive the pause.
'''),
code('''
from langgraph.types import interrupt, Command

@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a refund to a customer. Requires human approval."""
    return f"Refunded ${amount:.2f} for order {order_id}"

class ApprovalState(TypedDict):
    messages: Annotated[list, add_messages]
    pending: dict

def propose(state: ApprovalState) -> dict:
    """The agent decides to refund - but does not do it yet."""
    return {"pending": {"order_id": "A-1001", "amount": 119.98},
            "messages": [AIMessage("I recommend refunding order A-1001 for $119.98.")]}

def human_gate(state: ApprovalState) -> dict:
    """Pause here and wait for a person."""
    decision = interrupt({
        "question": "Approve this refund?",
        "details": state["pending"],
    })
    return {"messages": [AIMessage(f"Human decision: {decision}")]}

def execute(state: ApprovalState) -> dict:
    last = state["messages"][-1].content
    if "approve" in last.lower():
        result = issue_refund.invoke(state["pending"])
        return {"messages": [AIMessage(result)]}
    return {"messages": [AIMessage("Refund cancelled - no money moved.")]}

builder = StateGraph(ApprovalState)
builder.add_node("propose", propose)
builder.add_node("human_gate", human_gate)
builder.add_node("execute", execute)
builder.add_edge(START, "propose")
builder.add_edge("propose", "human_gate")
builder.add_edge("human_gate", "execute")
builder.add_edge("execute", END)

approval_graph = builder.compile(checkpointer=InMemorySaver())
print("built")
'''),
code('''
config = {"configurable": {"thread_id": "refund-1"}}

# Run until the interrupt fires.
result = approval_graph.invoke({"messages": [HumanMessage("I want a refund for A-1001")],
                                "pending": {}}, config=config)

print("PAUSED. The graph is waiting for a human.")
print("interrupt payload:", result["__interrupt__"][0].value)
print("\\nnext node when resumed:", approval_graph.get_state(config).next)
'''),
code('''
# A person approves. Resume with Command(resume=...).
final = approval_graph.invoke(Command(resume="approve"), config=config)

for m in final["messages"]:
    print(f"  {type(m).__name__:11} {m.content}")
'''),
code('''
# The same graph, rejected. Note the fresh thread_id - a new conversation.
config2 = {"configurable": {"thread_id": "refund-2"}}
approval_graph.invoke({"messages": [HumanMessage("Refund A-1001")], "pending": {}}, config=config2)
rejected = approval_graph.invoke(Command(resume="reject"), config=config2)

print(rejected["messages"][-1].content)
'''),
md('''
> **The state survived the pause.** That is the whole point of the checkpointer. A real
> deployment can interrupt on Monday, put the request in a review queue, and resume on
> Tuesday from a different process.

LangChain also ships `HumanInTheLoopMiddleware`, which adds this to `create_agent` without
a custom graph:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model=llm, tools=TOOLS,
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"issue_refund": True})],
    checkpointer=InMemorySaver(),
)
```

## 2. Multi-agent — a supervisor routing to specialists

One agent with fifteen tools gets confused and expensive. Split it: small focused agents,
and a supervisor that routes.
'''),
code('''
class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str

def supervisor(state: TeamState) -> dict:
    """Decides which specialist handles this. A real one uses structured output."""
    text = state["messages"][-1].content.lower()
    if any(w in text for w in ["order", "delivery", "shipped", "tracking"]):
        route = "orders_agent"
    elif any(w in text for w in ["refund", "policy", "warranty", "return"]):
        route = "policy_agent"
    else:
        route = "general_agent"
    return {"route": route}

def orders_agent(state: TeamState) -> dict:
    return {"messages": [AIMessage("[orders] Order A-1001 shipped, arriving Tuesday.")]}

def policy_agent(state: TeamState) -> dict:
    return {"messages": [AIMessage("[policy] Refunds are available within 14 days.")]}

def general_agent(state: TeamState) -> dict:
    return {"messages": [AIMessage("[general] Let me connect you with a human.")]}

builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor)
for name, fn in [("orders_agent", orders_agent), ("policy_agent", policy_agent),
                 ("general_agent", general_agent)]:
    builder.add_node(name, fn)
    builder.add_edge(name, END)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", lambda s: s["route"])

team = builder.compile()

for q in ["Where is my order?", "What is your refund policy?", "Do you sponsor events?"]:
    out = team.invoke({"messages": [HumanMessage(q)], "route": ""})
    print(f"{q:32} -> {out['messages'][-1].content}")
'''),
code('''
print(team.get_graph().draw_mermaid())
'''),
md('''
**Each specialist is itself an agent** — swap `orders_agent` for a full `create_agent` graph
with only order tools, and `policy_agent` for one with only document search. Small tool
sets mean better tool choice and cheaper calls.

> **Do not reach for multi-agent early.** It multiplies the number of LLM calls and makes
> failures much harder to trace. Split only when one agent's tool list has genuinely grown
> unmanageable — usually past about ten tools.

## 3. Error handling and bounded loops

Three failure modes will bite you in production.

### Tools raise exceptions
'''),
code('''
from langgraph.prebuilt import ToolNode

@tool
def flaky_lookup(key: str) -> str:
    """Look something up in a service that sometimes fails."""
    raise ConnectionError("upstream service unavailable")

class ToolState(TypedDict):
    messages: Annotated[list, add_messages]

def tool_graph(handle_errors: bool):
    """A one-node graph so we can run ToolNode - it must live inside a graph."""
    b = StateGraph(ToolState)
    b.add_node("tools", ToolNode([flaky_lookup], handle_tool_errors=handle_errors))
    b.add_edge(START, "tools")
    b.add_edge("tools", END)
    return b.compile()

failing_call = AIMessage(content="", tool_calls=[
    {"name": "flaky_lookup", "args": {"key": "x"}, "id": "c1", "type": "tool_call"}])

for handle in [True, False]:
    try:
        out = tool_graph(handle).invoke({"messages": [failing_call]})
        print(f"handle_tool_errors={handle}")
        print(f"   -> became a ToolMessage: {out['messages'][-1].content[:70]}...")
    except Exception as exc:
        print(f"handle_tool_errors={handle}")
        print(f"   -> raised {type(exc).__name__}: {exc}  (the whole run dies)")
'''),
md('''
Without `handle_tool_errors`, one flaky API takes down the whole run. With it, the model
sees the error text and can try something else — which is usually what you want.

### Runaway loops
'''),
code('''
class LoopState(TypedDict):
    steps: int

def work(state: LoopState) -> dict:
    return {"steps": state["steps"] + 1}

def keep_going(state: LoopState) -> Literal["work", "__end__"]:
    return END if state["steps"] >= 3 else "work"

b = StateGraph(LoopState)
b.add_node("work", work)
b.add_edge(START, "work")
b.add_conditional_edges("work", keep_going)
bounded = b.compile()

print("with an exit condition:", bounded.invoke({"steps": 0}))

# Now the same graph with no exit condition, guarded only by the recursion limit.
b2 = StateGraph(LoopState)
b2.add_node("work", work)
b2.add_edge(START, "work")
b2.add_conditional_edges("work", lambda s: "work")     # never ends
runaway = b2.compile()

try:
    runaway.invoke({"steps": 0}, config={"recursion_limit": 10})
except Exception as exc:
    print("\\nwithout one:", type(exc).__name__)
    print(" ", str(exc)[:110])
'''),
md('''
**Two layers of protection, and you want both:**
1. `recursion_limit` in the config — a hard backstop (default 25).
2. Your own counter in the state — a meaningful limit with a graceful message.

### Untrusted tool arguments

Worth repeating from notebook 03: the model chooses your tool arguments, and a user can
influence the model. Validate every argument, allowlist rather than blocklist, and never
hand model output to `eval`, `exec`, a shell, or string-built SQL. An agent with a
`run_sql` tool and no allowlist is a data breach waiting for a prompt injection.

## 4. Observability and cost

Agents are hard to debug because the path changes every run. You need traces.

**LangSmith** is the built-in option — set four environment variables and every run is
recorded, with each node, prompt, tool call and token count:

```bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=my-agent
```

Those are already in this folder's `.sample.env`. No code change is needed — LangChain
picks them up.

### Where the money goes

Every loop iteration resends the **entire** message history. A ten-step agent run does not
cost ten calls' worth of input tokens — it costs roughly the sum of a growing history, which
is quadratic in the number of steps.
'''),
code('''
print(f"{'step':>5} {'history':>9} {'cumulative input tokens':>26}")
cumulative, history = 0, 400          # ~400 tokens of system prompt + question
for step in range(1, 11):
    cumulative += history
    print(f"{step:>5} {history:>9} {cumulative:>26,}")
    history += 250                    # each tool call + result adds to the history

print(f"\\n10 steps cost about {cumulative:,} input tokens, not {400 * 10:,}.")
print("This is why bounding the loop is a cost control, not just a safety measure.")
'''),
md('''
**Four levers that actually reduce cost:**

1. **Bound the loop.** The table above is the argument.
2. **Fewer, better tools.** Ten tools means ten schemas in the prompt on every single call.
3. **Trim history.** Summarise or drop old turns — `SummarizationMiddleware` does this for
   `create_agent`.
4. **Right-size the model.** Route easy turns to a small model, hard ones to a large one.

## 5. Do you need an agent at all?

Ask these before you build one:

- **Is the sequence of steps fixed?** Then write a chain, or plain Python. An agent
  deciding what you already know is latency and cost with extra failure modes.
- **Is there exactly one tool?** Call it directly.
- **Can a wrong action be undone?** If not, a human gate is mandatory (section 1).
- **Can you tell whether the answer was right?** If not, you cannot improve it and you
  should not ship it.

| Situation | Build |
|---|---|
| Fixed steps, known order | a chain — or ordinary code |
| One LLM call, structured result | `with_structured_output` |
| Fixed branch on a classification | a `StateGraph` with a conditional edge |
| Open-ended, model picks the tools | an agent |
| Several distinct skill areas | a supervisor + specialist agents |

The most common mistake in this whole module is building an agent for a problem that a
chain solves — slower, dearer, and less predictable, for no gain.

## 6. Production checklist

Before an agent touches real users:

- [ ] Loop bounded by both `recursion_limit` and a counter in state
- [ ] `handle_tool_errors=True`, or explicit try/except in every tool
- [ ] Every tool argument validated; no `eval`/`exec`/shell/raw SQL on model output
- [ ] Human approval on anything irreversible
- [ ] A durable checkpointer (SQLite or Postgres), not `InMemorySaver`
- [ ] Tracing switched on
- [ ] Per-conversation token budget, and an alert when it is exceeded
- [ ] Timeouts on every external call
- [ ] Secrets in environment variables, never in the notebook
- [ ] An evaluation set of ~20 real questions with expected behaviour, run before each deploy

That last one is the one everybody skips and everybody regrets.

---
## Exercises

1. Add a second interrupt to the approval graph that also asks for a reason when the refund
   is rejected. Confirm both pauses work on the same thread.
2. Turn `orders_agent` into a real `create_agent` with only `get_order_status`, wired into
   the supervisor graph. Does the supervisor still route correctly?
3. Set `handle_tool_errors=False` and re-run the flaky tool. Where does the exception
   surface, and what would a user see?
4. Add a `token_budget` field to an agent's state, increment an estimate on every model
   node, and end the run when it exceeds a cap.
5. Take the cost table and recompute it for a 20-step run. At $2 per million input tokens,
   what does one conversation cost? What does 10,000 of them cost?
6. **Stretch:** write 10 test questions for the support agent with expected tool calls.
   Run them as a loop and report how many chose the right tool. That is an eval set — the
   thing that lets you change a prompt without breaking production.

## Where to go next

- **Module 07** — how to structure and deploy this as a real service.
- The `.sample.env` in this folder lists every key these notebooks can use.
- LangGraph docs: <https://docs.langchain.com/oss/python/langgraph/overview>
'''),
])
