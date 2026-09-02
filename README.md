# ML Learning Path — from NumPy to a deployed AI service

A self-contained course for students learning machine learning and AI engineering in
Python. Seven modules, in order, each with a README of its own.

---

## The path

| Module | What it covers | Notebooks | Suggested time |
|---|---|---|---|
| **01 – ML Libraries** | NumPy, pandas, Matplotlib | 4 | ~1 week |
| **02 – Supervised Learning** | the full workflow, plus the theory behind it | 9 | ~3 weeks |
| **03 – Unsupervised Learning** | clustering, PCA, recommendation | 6 | ~2 weeks |
| **04 – Deep Learning** | backprop by hand, TensorFlow, PyTorch, CNNs | 10 | ~3 weeks |
| **05 – LLMs** | tokens, embeddings, transformers, fine-tuning, RAG | 5 | ~2 weeks |
| **06 – Agents** | LangChain, LangGraph, real tool-using agents | 5 | ~2 weeks |
| **07 – Production** | how to structure and ship it, with a working example | — | ~1 week |

Work in order. Each module assumes the one before it.

---

## Module 01 — ML Libraries

Exercise notebooks: each cell states a task, you write the code.

`01-numpy-introduction` · `02-numpy-exercises` · `03-pandas-exercises` · `04-matplotlib-exercises`

**Done when:** you can load a CSV, clean it, group it and plot a comparison without
looking anything up.

## Module 02 — Supervised Learning

Notebook 01 is the end-to-end workflow. Notebooks 02–08 each take one thing it used
without explaining, and explain it properly. Notebook 09 is the capstone.

| # | Notebook | The question it answers |
|---|---|---|
| 01 | `01-end-to-end-workflow` | How does a full scikit-learn project look? |
| 02 | `02-preprocessing-and-scaling` | Why do KNN and SVM need scaling, and trees don't? |
| 03 | `03-overfitting-and-learning-curves` | Too simple, too complex, or short of data? |
| 04 | `04-cross-validation-and-leakage` | Is my score real, or did I leak the test set? |
| 05 | `05-classification-metrics` | Accuracy is 97% and the model is useless — why? |
| 06 | `06-regression-and-regularization` | What does `Ridge`/`Lasso` do to the coefficients? |
| 07 | `07-trees-forests-and-boosting` | How does a tree split? Bagging vs boosting? |
| 08 | `08-hyperparameter-tuning` | How do I tune without fooling myself? |
| 09 | `09-capstone-project` | Can you run the whole loop yourself? |

**The five ideas:** scale distance-based models · a single split is a lottery ticket ·
every fitted transformer belongs in a `Pipeline` · the metric comes from the problem ·
the tuning score is optimistic.

## Module 03 — Unsupervised Learning

No `y`, and no accuracy to hide behind.

`00-from-supervised-to-unsupervised` · `01-kmeans-from-scratch` ·
`02-clustering-pca-dbscan-hierarchical` · `03-collaborative-recommendation` ·
`04-news-clustering-nlp` · `05-churn-eda`

Worked solutions for 01–04 are in `solutions/`. **Attempt first.**

## Module 04 — Deep Learning

| # | Notebook | Needs |
|---|---|---|
| 00 | `00-when-to-use-deep-learning` — should you use a neural net at all? | sklearn |
| 01 | `01-neural-net-from-scratch-numpy` — **forward, backprop, gradient descent by hand** | numpy |
| 02–05 | perceptron, TensorFlow, Fashion-MNIST, CNN | tensorflow |
| 06 | `06-pytorch-fundamentals` — tensors, autograd, the training loop | torch |
| 07 | `07-cnn-pytorch` | torch |
| 08 | `08-training-diagnostics-and-regularization` — dropout, weight decay, early stopping | torch |
| 09 | `09-transfer-learning` — what you will actually do in practice | tensorflow |

Notebooks 00 and 01 need neither framework. **Notebook 01 is the most important one in
the module** — write backprop once and `model.fit()` stops being magic.

## Module 05 — LLMs

`00-llm-fundamentals` (tokens, embeddings, decoding) · `01-intro-to-transformers-library` ·
`02-running-models-on-gpu` · `03-finetuning-a-model` · `04-rag-application-local`

**The decision that matters:** prompting → RAG → fine-tuning, in that order of preference.
RAG teaches facts; fine-tuning teaches behaviour. Fine-tuning a model to "know" your
handbook is the expensive mistake.

## Module 06 — Agents with LangChain & LangGraph

`01-langchain-fundamentals` · `02-langgraph-basics` · `03-build-a-real-agent` ·
`04-agent-patterns-and-production`

**These run without an API key.** A scripted stand-in model (`offline_llm.py`) drives the
real LangGraph machinery, so the agent loop, tool execution, memory and routing all
genuinely run — only the model's replies are pre-written. Swap in a real provider with one
line when you want real answers.

Notebook 03 builds the same agent twice: once with `create_agent`, once by hand as a
`StateGraph`. The second is the point.

## Module 07 — Production Project Structure

Not notebooks — a written guide plus `example-project/`, a working ML service:
train a model, serve it over FastAPI, validate input, test it, containerise it.

**The one rule:** training and serving are different programs with different lifecycles.

```bash
cd 07-Production-Project-Structure/example-project
python scripts/train.py && pytest -q && uvicorn heart_api.main:app --reload
```

---

## Setup

See [`SETUP.md`](SETUP.md). Modules 01–03 need only the base stack:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
jupyter lab
```

Modules 04–07 each add their own dependencies — see the module README. Install them when
you get there, not up front.

All notebooks load data with a relative path like `../data/heart-disease.csv`, so
**launch Jupyter from the folder that contains this README**, and open notebooks from
their own module folder.

---

## Datasets in `data/`

| File | Used by | Task |
|---|---|---|
| `heart-disease.csv` | modules 02, 04, 07 | binary classification |
| `car-sales-extended-missing-data.csv` | module 02 | regression, missing values |
| `car-sales.csv`, `car-sales-missing-data.csv` | module 01 | pandas practice |
| `Mall_Customers.csv` | module 03 | clustering |
| `Country-data.csv` | module 03 | clustering + PCA |
| `movies.csv`, `ratings.csv`, `links.csv`, `tags.csv` | module 03 | recommendation |
| `news_data.tsv` | module 03 | text clustering |
| `churn.csv`, `batch_churn.csv` | module 03, capstone extension | classification |

Modules 05 and 06 keep their own data alongside their notebooks.

---

## Notes for whoever is teaching this

- **The exercises are the course.** The notebooks are scaffolding around them.
- **Make students do 02/04.** Leakage is the mistake that makes a portfolio project
  worthless, and it is invisible until someone shows them the noise demo.
- **02/05 section 3** — moving a decision threshold on purpose — is the highest-value ten
  minutes in the whole path.
- **04/01 cannot be skipped.** Every student who skips writing backprop has to come back
  to it later.
- **04/00 and 04/08 teach honesty.** Both show results that do *not* flatter deep
  learning. That is deliberate: the numbers are real, and students should see that
  measuring beats assuming.
- **In 03, insist the clusters get named.** A student who cannot name a cluster has
  produced a plot, not a segmentation.
- **06 runs without keys**, so a whole class can do it with no budget and no signups.
- Most new notebooks are generated from `scripts/build_*.py`. Edit the script and re-run it
  to change content across a module; editing a `.ipynb` directly is fine for one-offs.

---

## Attribution

- Module 01 and notebook `02/01` are practice notebooks from the
  [Zero to Mastery ML course](https://github.com/mrdbourke/zero-to-mastery-ml).
- Written for this path: `02/02`–`02/09`, `03/00`, `04/00`, `04/01`, `04/06`, `04/08`,
  `05/00`, `06/01`–`06/04`, `06/offline_llm.py`, and all of module 07.

### Free resources to go deeper

- [INRIA scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc) — by the
  scikit-learn core developers. The best free follow-up to module 02.
- [An Introduction to Statistical Learning (ISLP)](https://www.statlearning.com/) —
  free PDF + Python labs. The maths under module 02.
- [Kaggle Learn](https://www.kaggle.com/learn) — short, in-browser reinforcement.
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) —
  interactive gradient-descent visualisations.
- [StatQuest](https://www.youtube.com/c/joshstarmer) — visual explanations of every
  concept in module 02.
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/overview) — for module 06.
- *Designing Machine Learning Systems*, Chip Huyen — for module 07.
