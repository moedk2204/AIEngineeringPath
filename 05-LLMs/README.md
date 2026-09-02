# Module 05 — Large Language Models

**Goal:** run, adapt and build on pretrained language models — and know which of those
three you actually need.

## Before you start

Module 04. An LLM is a neural network, and notebooks 01 and 06 there are what make the
words "weights", "fine-tuning" and "inference" mean something concrete.

## Order

| # | Notebook | Focus |
|---|---|---|
| 00 | `00-llm-fundamentals.ipynb` | **tokens, embeddings, decoding — the vocabulary** |
| 01 | `01-intro-to-transformers-library.ipynb` | pipelines, models, tokenizers |
| 02 | `02-running-models-on-gpu.ipynb` | devices, dtypes, memory |
| 03 | `03-finetuning-a-model.ipynb` | adapting a pretrained model to your data |
| 04 | `04-rag-application-local.ipynb` | retrieval-augmented generation over PDFs |

Also here:
- `applications/` — a simple inference script and a Gradio chat UI
- `assignment/` — a practice assignment
- `data/us_census/` — PDFs used by the RAG notebook

## The decision that matters most

Four ways to make a model know about *your* data. People reach for the last one first,
which is almost always wrong.

| Approach | Cost | Use when |
|---|---|---|
| **Prompting** | free | the model already knows; you just need the right instruction |
| **RAG** (notebook 04) | low | facts change, are private, or are too many to fit in a prompt |
| **Fine-tuning** (notebook 03) | medium | you need a *style*, *format* or *task* the base model does not do |
| **Training from scratch** | enormous | essentially never |

> **RAG teaches facts; fine-tuning teaches behaviour.** Fine-tuning a model so it "knows"
> your company handbook is the most common and most expensive mistake in this field — the
> handbook changes next month, and the model does not. Use RAG.

## Setup

```bash
pip install transformers torch sentence-transformers
pip install langchain-community faiss-cpu pypdf     # for the RAG notebook
pip install gradio                                   # for applications/2-gradio-chat.py
```

`requirements-extra.txt` lists the additional packages these notebooks use.

**Model downloads.** The first run of any notebook downloads weights from Hugging Face —
hundreds of MB, sometimes more. They cache in `~/.cache/huggingface`. Behind a corporate
proxy this may fail; see the SSL note in the root `SETUP.md`.

**No GPU?** Everything runs on CPU, slowly. Fine-tuning (notebook 03) is the one that
genuinely wants a GPU — use Google Colab for it.

**API keys.** Notebooks 01–04 use open models and need no key. `assignment/.sample.env`
lists the keys the assignment uses.

## Known gaps in this module

Worth knowing what you are not being taught here, so you can go looking:

- **Evaluation.** How do you know the output was good? There is no accuracy score. Look up
  LLM-as-judge, and building a small eval set of real questions with expected answers.
  This is the single biggest gap between a demo and a product.
- **Prompt engineering as a discipline** — few-shot examples, structured output, chain of
  thought.
- **Chunking strategy for RAG.** Notebook 04 uses one strategy; chunk size and overlap
  change retrieval quality more than the choice of model does.
- **Quantization** — running an 8B model on a laptop with 4-bit weights.
- **Cost and latency budgets** — covered in module 06, notebook 04.
- **Safety** — prompt injection, jailbreaks, PII leakage.

## Self-check

- [ ] Explain what a token is and why two similar sentences can have very different costs.
- [ ] Given a business problem, choose prompting vs RAG vs fine-tuning and defend it.
- [ ] Explain what temperature does, and when you want it at 0.
- [ ] Draw the RAG pipeline: document → chunks → embeddings → retrieval → prompt → answer.
- [ ] Name the biggest failure mode of RAG. (Retrieval, not generation — nearly always.)
