import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nbutil import md, code, write

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "05-LLMs")

write(os.path.join(OUT, "00-llm-fundamentals.ipynb"), [
md('''
# 00 - LLM Fundamentals: Tokens, Embeddings, Decoding

Before you call a single `pipeline()`, learn the three concepts that explain almost every
surprising thing an LLM does:

1. **Tokens** — models do not see text, and this drives your bill.
2. **Embeddings** — meaning as geometry. This is what makes RAG possible.
3. **Decoding** — the model outputs a probability distribution, and *you* choose how to
   sample from it. This is why the same prompt gives different answers.

Everything runs on small models (GPT-2 sized) on CPU.

> **First run downloads weights** — a few hundred MB, cached in `~/.cache/huggingface`.
> Later runs are instant.
'''),
code('''
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
print("torch", torch.__version__)
'''),
md('''
## 1. Tokens

A model has a fixed vocabulary of a few tens of thousands of **tokens** — sub-word pieces.
Your text is chopped into those pieces and each becomes an integer. That is all the model
ever sees.
'''),
code('''
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(f"vocabulary size: {tokenizer.vocab_size:,} tokens")

text = "Machine learning is transforming healthcare."
ids = tokenizer.encode(text)
pieces = [tokenizer.decode([i]) for i in ids]

print(f"\\ntext   : {text}")
print(f"tokens : {pieces}")
print(f"ids    : {ids}")
print(f"\\n{len(text)} characters -> {len(ids)} tokens")
'''),
md('''
Notice the leading spaces inside the tokens. In GPT-2's vocabulary `" learning"` and
`"learning"` are **different tokens**. That is why a stray space can change a model's
behaviour.

### Not all text costs the same
'''),
code('''
import pandas as pd

samples = {
    "plain English":     "The cat sat on the mat and looked out of the window.",
    "technical jargon":  "Backpropagation computes gradients via the chain rule.",
    "code":              "def train(model, X, y):\\n    return model.fit(X, y)",
    "a rare word":       "Antidisestablishmentarianism is a long word.",
    "numbers":           "Order 84719 shipped on 2026-03-14 for $1,299.99",
    "Arabic":            "\\u0627\\u0644\\u062a\\u0639\\u0644\\u0645 \\u0627\\u0644\\u0622\\u0644\\u064a \\u064a\\u063a\\u064a\\u0631 \\u0627\\u0644\\u0639\\u0627\\u0644\\u0645",
}

rows = []
for label, s in samples.items():
    n = len(tokenizer.encode(s))
    rows.append({"kind": label, "characters": len(s), "tokens": n,
                 "chars per token": round(len(s) / n, 2)})

pd.DataFrame(rows)
'''),
md('''
**Read the last column.** English prose gets ~4 characters per token. Code, rare words,
numbers and non-Latin scripts get far fewer — sometimes one character per token.

Three consequences you will meet in practice:

1. **Cost.** APIs bill per token. The same paragraph in Arabic can cost several times what
   it costs in English.
2. **Context limits.** A "32k context window" holds far less Arabic or code than English.
3. **Arithmetic is hard for LLMs.** `1299.99` may split into four unrelated tokens. The
   model never sees the number — only the fragments. This is why you give a model a
   calculator tool (module 06) rather than trusting its mental arithmetic.
'''),
code('''
print("How a number gets shredded:")
for number in ["7", "42", "1299", "1299.99"]:
    print(f"  {number:9} -> {[tokenizer.decode([i]) for i in tokenizer.encode(number)]}")
'''),
md('''
> **Estimating tokens.** A rough rule for English is `tokens ~ characters / 4`, or
> `tokens ~ words * 1.3`. Use the real tokenizer whenever the number matters.

## 2. Embeddings — meaning as geometry

An embedding turns a piece of text into a vector, positioned so that **similar meanings
sit close together**. This is the mechanism behind semantic search, and therefore behind
RAG (notebook 04).
'''),
code('''
from transformers import AutoModel

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL)
embed_model.eval()

def embed(texts: list[str]) -> np.ndarray:
    """Mean-pool the token vectors, then normalise to unit length."""
    batch = embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = embed_model(**batch).last_hidden_state          # (batch, tokens, dim)

    mask = batch["attention_mask"].unsqueeze(-1).float()      # ignore padding
    pooled = (out * mask).sum(1) / mask.sum(1)
    return torch.nn.functional.normalize(pooled, dim=1).numpy()

vecs = embed(["hello world"])
print("one sentence ->", vecs.shape, "numbers")
'''),
code('''
sentences = [
    "How do I get a refund?",
    "I want my money back",
    "What is your return policy?",
    "The weather is nice today",
    "It is sunny outside",
]

E = embed(sentences)
similarity = E @ E.T          # vectors are normalised, so a dot product IS cosine similarity

frame = pd.DataFrame(similarity.round(2),
                     index=[s[:26] for s in sentences],
                     columns=[f"s{i}" for i in range(len(sentences))])
frame
'''),
code('''
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(similarity, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(sentences)))
ax.set_yticks(range(len(sentences)))
ax.set_xticklabels([f"s{i}" for i in range(len(sentences))])
ax.set_yticklabels([s[:30] for s in sentences])
for i in range(len(sentences)):
    for j in range(len(sentences)):
        ax.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center",
                color="white" if similarity[i, j] > 0.6 else "black", fontsize=9)
plt.colorbar(im, label="cosine similarity")
plt.title("Two clusters, and no shared keywords inside either")
plt.tight_layout()
plt.show()
'''),
md('''
**The important row is `"How do I get a refund?"` against `"I want my money back"`.**
They share no significant words — a keyword search scores them zero. The embedding scores
them high, because it encodes *meaning*, not spelling.

That is the whole trick behind semantic search.

### Semantic search in nine lines

This is RAG's retrieval step, without a vector database.
'''),
code('''
documents = [
    "Refunds are issued within 14 days of purchase, no questions asked.",
    "Standard shipping takes 3-5 business days; express is next-day.",
    "All hardware carries a 2-year warranty from the delivery date.",
    "Our support team is available Monday to Friday, 9am to 6pm.",
    "You can change your delivery address before the order ships.",
]
doc_vectors = embed(documents)

def search(query: str, k: int = 2):
    scores = embed([query])[0] @ doc_vectors.T
    for rank in np.argsort(scores)[::-1][:k]:
        print(f"  {scores[rank]:.3f}  {documents[rank]}")

for q in ["Can I get my money back?", "When will my package arrive?", "Is it under guarantee?"]:
    print(f'"{q}"')
    search(q)
    print()
'''),
md('''
> **RAG in one sentence:** embed your documents once, embed the user's question, retrieve
> the nearest few, and paste them into the prompt. Notebook 04 does exactly this with a
> real vector store and PDFs.
>
> **And the failure mode:** if retrieval returns the wrong documents, no model can save the
> answer. When RAG gives a bad answer, look at what was retrieved *before* you blame the
> model or the prompt.

## 3. Decoding — where the randomness comes from

A language model does exactly one thing: given some tokens, produce a probability
distribution over the next token. It never "chooses" — **your decoding settings do**.
'''),
code('''
from transformers import AutoModelForCausalLM

lm_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
lm = AutoModelForCausalLM.from_pretrained("distilgpt2")
lm.eval()

prompt = "The capital city of France is"
inputs = lm_tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    logits = lm(**inputs).logits[0, -1]        # scores for the NEXT token

probs = torch.softmax(logits, dim=-1)
top = torch.topk(probs, 8)

print(f'"{prompt}" ...\\n')
for score, idx in zip(top.values, top.indices):
    print(f"  {score.item():6.2%}  {lm_tokenizer.decode([idx])!r}")
'''),
md('''
That distribution is the model's entire output. Everything after this is a sampling policy.

### Temperature

Temperature divides the scores before they become probabilities. Low temperature sharpens
the distribution (confident, repetitive); high temperature flattens it (varied, riskier).
'''),
code('''
def distribution_at(temperature: float, k: int = 6):
    scaled = torch.softmax(logits / temperature, dim=-1)
    top = torch.topk(scaled, k)
    return [(lm_tokenizer.decode([i]).strip() or "_", v.item()) for v, i in
            zip(top.values, top.indices)]

fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
for ax, temp in zip(axes, [0.1, 0.7, 1.0, 2.0]):
    pairs = distribution_at(temp)
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    ax.bar(range(len(values)), values, color="#4C78A8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(f"temperature = {temp}")
    ax.set_ylabel("probability")
plt.suptitle("The same logits, four sampling temperatures")
plt.tight_layout()
plt.show()
'''),
md('''
At `0.1` the top token takes nearly all the mass — the model becomes near-deterministic.
At `2.0` the mass spreads out and unlikely tokens become plausible choices.

| Setting | Use for |
|---|---|
| `temperature=0` (greedy) | extraction, classification, code, anything you will parse |
| `0.3 – 0.7` | most assistant and Q&A work |
| `0.9 – 1.2` | brainstorming, creative writing |
| above `1.5` | rarely useful |

**`temperature=0` is the setting people forget.** If an LLM feeds other code, you almost
always want the same input to give the same output.
'''),
code('''
inputs = lm_tokenizer("Machine learning is", return_tensors="pt")

print("GREEDY (temperature 0) - same every time:")
for _ in range(2):
    out = lm.generate(**inputs, max_new_tokens=18, do_sample=False,
                      pad_token_id=lm_tokenizer.eos_token_id)
    print("  ", lm_tokenizer.decode(out[0], skip_special_tokens=True))

print("\\nSAMPLING (temperature 1.0, top_p 0.9) - different every time:")
for seed in [0, 1]:
    torch.manual_seed(seed)
    out = lm.generate(**inputs, max_new_tokens=18, do_sample=True,
                      temperature=1.0, top_p=0.9,
                      pad_token_id=lm_tokenizer.eos_token_id)
    print("  ", lm_tokenizer.decode(out[0], skip_special_tokens=True))
'''),
md('''
> **distilgpt2 is small and its output is not very good.** That is fine — the *mechanism*
> is identical in a frontier model. What changes with scale is the quality of the
> distribution, not how you sample from it.

### `top_p` and `top_k`

Two ways to cut off the tail before sampling:

- **`top_k=50`** — only ever consider the 50 most likely tokens.
- **`top_p=0.9`** (nucleus) — consider the smallest set of tokens whose probabilities sum
  to 0.9. Adaptive: few candidates when the model is confident, many when it is not.

`top_p` is the better default. Together with temperature it is most of what you can tune.

## 4. The context window

Everything — system prompt, history, retrieved documents, the question, and the answer
being generated — shares one fixed token budget.
'''),
code('''
print(f"distilgpt2 context window: {lm.config.n_positions} tokens")
print(f"Modern models: 128,000 to 1,000,000+\\n")

budget = 8000
parts = {"system prompt": 300, "retrieved documents": 4000,
         "conversation history": 2000, "user question": 150}

used = sum(parts.values())
for name, n in parts.items():
    print(f"  {name:24} {n:>6,}")
print(f"  {'-' * 24} {'-' * 6}")
print(f"  {'used':24} {used:>6,}")
print(f"  {'left for the answer':24} {budget - used:>6,}  <- out of {budget:,}")
'''),
md('''
This is why long conversations get expensive and eventually break: the history grows, the
room for an answer shrinks. Module 06, notebook 04 shows the cost curve, and the fixes —
summarise old turns, retrieve fewer documents, bound the loop.

---
## Exercises

1. Tokenize a paragraph in your own first language and compare characters-per-token against
   English. What does that mean for what an API charges you?
2. Add three sentences of your own to the similarity matrix — two that mean the same thing
   in different words, one unrelated. Does the geometry match your intuition?
3. Break the semantic search: write a query that *should* match a document but does not.
   What kind of query fails? (Try acronyms, negations, and very short queries.)
4. Generate from the same prompt at `temperature=0.2`, `0.8` and `1.5`, five samples each.
   Describe how the failure mode changes as temperature rises.
5. Set `top_p=0.5` and `top_p=1.0` at a fixed temperature of 1.0. Which produces more
   coherent text, and why?
6. **Stretch:** implement greedy decoding yourself in a loop — call the model, `argmax` the
   logits, append the token, repeat. Confirm it matches `do_sample=False`. This is the
   generation loop, and it is smaller than you expect.

## Next

Notebook 01 introduces the `transformers` library properly — pipelines, model classes and
tokenizers — now that you know what a token is.
'''),
])
