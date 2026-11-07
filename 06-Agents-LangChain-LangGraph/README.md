# Module 06 — Agents with LangChain & LangGraph

**Goal:** build agents you could actually deploy — and understand the loop well enough to
debug one at 3am.

## The problem this module solves

Most agent tutorials you will find teach **LangChain 0.x**: `AgentExecutor`,
`create_tool_calling_agent`, `ConversationBufferMemory`. Those APIs are deprecated.
Modern LangChain (1.x) is built on **LangGraph**, and the mental model is different:
an agent is a *graph*, not an executor object.

`tool-calling-legacy/` keeps an example of the old style, because you will meet it in
older tutorials and need to recognise it. Notebook 01 has the translation table.

## Order

| # | Notebook | Focus | Needs a key? |
|---|---|---|---|
| 00 | `00-huggingface-tool-calling.ipynb` | tool calling with raw Hugging Face models | yes |
| 01 | `01-langchain-fundamentals.ipynb` | **models, messages, prompts, tools, structured output** | **no** |
| 02 | `02-langgraph-basics.ipynb` | **state, nodes, edges, routing, loops, memory** | **no** |
| 03 | `03-build-a-real-agent.ipynb` | **`create_agent`, then the same agent hand-built** | **no** |
| 04 | `04-agent-patterns-and-production.ipynb` | **human approval, multi-agent, errors, cost** | **no** |

Also here:
- `attention-from-scratch/` — self-attention, multi-head and causal attention in raw
  PyTorch. Read these when you want to know what is inside a transformer.
- `tool-calling-legacy/` — the deprecated 0.x style, for recognition only.

## No API key required

Notebooks 01–04 run against `offline_llm.py`, a scripted stand-in model. The real
LangChain and LangGraph machinery runs — the agent loop, tool execution, state, routing,
checkpointing — only the model's replies are pre-written.

That means students can learn the mechanics for free, and it makes the notebooks
deterministic, so a cell that worked yesterday works today.

When you want real answers:

```python
from offline_llm import real_model
llm = real_model("openai:gpt-4.1-mini")        # or anthropic: / google_genai: / ollama:
```

Copy `.sample.env` to `.env` and add the one key you have. `ollama:` runs locally and is
free — the best option if you have no key at all.

## Setup

```bash
pip install langchain langgraph python-dotenv
pip install langchain-openai          # or -anthropic / -google-genai / -ollama
```

Verified against **langchain 1.2.0** and **langgraph 1.0.5**. If your version differs and
an import fails, check the [migration guide](https://docs.langchain.com/oss/python/migrate/langgraph-v1).

## The five ideas

1. **An agent is a loop.** Model → tool → model → tool → answer. Two nodes and a
   conditional edge. Notebook 03 builds it by hand so this stops being abstract.
2. **`create_agent` is a wrapper.** It returns a LangGraph graph you could have written
   yourself. Print `agent.get_graph().draw_mermaid()` and look.
3. **Memory is a checkpointer plus a `thread_id`.** Not a magic memory object.
4. **Tool descriptions are prompts.** A vague docstring is the most common cause of an
   agent choosing the wrong tool.
5. **Bound every loop.** It is a safety control and a cost control. Notebook 04 shows the
   quadratic token growth that makes this matter.

## Security — read this before shipping anything

The model chooses your tool arguments, and a user can influence the model. Therefore:

- **Never** pass model output to `eval`, `exec`, a shell, or a string-built SQL query.
- Allowlist what a tool accepts; do not blocklist what it rejects.
- Put a human gate in front of anything irreversible — refunds, emails, deletions, writes
  to production.
- An agent with a `run_sql` tool and no allowlist is a data breach waiting for a prompt
  injection.

Notebook 03's `calculate` tool shows the minimum: validate the characters before
evaluating anything.

## Self-check

- [ ] Draw the two-node agent loop from memory.
- [ ] Explain what `add_messages` does and what breaks without it.
- [ ] Explain the difference between a chain and a graph, and when you need the graph.
- [ ] Say how memory works, in terms of checkpointer and `thread_id`.
- [ ] Given a task, decide agent vs chain vs a single structured call — and justify it.
- [ ] Name three things that must be true before an agent touches real users.

## Course references

Both are free, official courses from LangChain Academy and use the modern 1.x APIs this
module teaches.

| Course | Length | Use it for |
|---|---|---|
| [Introduction to LangChain — Python](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python) | ~1.5h, 30 lessons | Notebooks 00–01: models, tools, building agents, debugging with LangSmith |
| [Introduction to LangGraph — Python](https://academy.langchain.com/courses/intro-to-langgraph) | ~6h, 55 lessons | Notebooks 02–04: state, memory, human-in-the-loop, sub-graphs, deployment |
