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

## Video resources

Every link below was checked and resolves. Use them as lookups for when a concept does not
land — not as a syllabus. The trap for beginners is watching more video instead of writing
code.

### Start here — pick ONE full course

| Course | Length | Why this one |
|---|---|---|
| [Machine Learning for Everybody](https://www.youtube.com/watch?v=i_LwzRVP7bg) — freeCodeCamp | ~4h | Best single starting point. Concepts + code, no maths prerequisites |
| [ML with Python and Scikit-Learn — Full Course](https://www.youtube.com/watch?v=hDKCxebp88A) — freeCodeCamp | ~18h | Closest match to modules 01–02: linear/logistic → trees → forests → boosting |
| [Scikit-learn Crash Course](https://www.youtube.com/watch?v=0B5eIE_1vpU) — freeCodeCamp | ~2h | If you just need the library, fast |
| [ML Tutorial Python playlist](https://www.youtube.com/playlist?list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw) — codebasics | short episodes | For anyone who bounces off long videos |

### Module 02 — supervised learning

The [StatQuest Machine Learning playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)
is the best free explanation of these concepts anywhere. Watch each video **before** the
notebook it lines up with:

| Notebook | Watch first |
|---|---|
| `02/01` end-to-end | [Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) |
| `02/02` scaling | [Support Vector Machines Pt 1](https://www.youtube.com/watch?v=efR1C6CvhmE) — shows why distance matters |
| `02/03` overfitting | [Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — **the most useful 7 minutes here** |
| `02/04` cross-validation | [Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) |
| `02/05` metrics | [The Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o), then [ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJemM) |
| `02/06` regularization | [Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) |
| `02/07` trees | [Decision Trees](https://www.youtube.com/watch?v=_L39rN6gz7Y) → [Part 2](https://www.youtube.com/watch?v=wpNl-JwwplA) → [Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) → [Gradient Boost](https://www.youtube.com/watch?v=3CC4N4z3GJc) |
| `02/08` tuning | [GridSearchCV](https://www.youtube.com/watch?v=HdlDYng8g9s) — codebasics |

Also useful: [Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA) and
[How to evaluate a classifier in scikit-learn](https://www.youtube.com/watch?v=85dtiMz9tSo) (Data School).

### Module 03 — unsupervised learning

| Notebook | Watch first |
|---|---|
| `03/00` the mental shift | [K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA), then [PCA in 5 minutes](https://www.youtube.com/watch?v=HMOI_lkzW08) |
| `03/01` k-means from scratch | [K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA) — watch it twice; you are about to implement this |
| `03/02` PCA, DBSCAN, hierarchical | [PCA Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ) → [DBSCAN](https://www.youtube.com/watch?v=RDZUdRSDOok) → [Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo) |
| `03/03` recommendation | [How does Netflix recommend movies?](https://www.youtube.com/watch?v=ZspR5PZemcs) — Luis Serrano, or [The Math Behind Recommender Systems](https://www.youtube.com/watch?v=n3RKsY2H-NE) |
| `03/04` text clustering | [TF-IDF, clearly explained](https://www.youtube.com/watch?v=OymqCnh-APA) — ritvikmath |

Once clustering makes sense, these two are the natural next step for visualising
high-dimensional data: [t-SNE](https://www.youtube.com/watch?v=NEaUSP4YerM) and
[UMAP](https://www.youtube.com/watch?v=eN0wFzBA4Sc), both StatQuest.

### Module 04 — deep learning

**Watch the 3Blue1Brown series first, in order.** It is the best explanation of neural
networks that exists, and chapters 3–4 are exactly what `04/01` makes you write by hand.

| # | Video |
|---|---|
| 1 | [But what *is* a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) |
| 2 | [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) |
| 3 | [Backpropagation, intuitively](https://www.youtube.com/watch?v=Ilg3gGewQ5U) |
| 4 | [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) |

([the full playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi))

Then, per notebook:

| Notebook | Watch first |
|---|---|
| `04/01` net from scratch | [The Essential Main Ideas of Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw) → [Backpropagation Main Ideas](https://www.youtube.com/watch?v=IN2XmBhILt4) → [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — all StatQuest |
| `04/02`–`04/05` TensorFlow | [TensorFlow 2.0 Complete Course](https://www.youtube.com/watch?v=tPYj3fFJGjk) — freeCodeCamp, ~7h |
| `04/05` + `04/07` CNNs | [Image Classification with CNNs](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest, or [How CNNs work](https://www.youtube.com/watch?v=FmpDIaiMIeA) — Brandon Rohrer |
| `04/06` PyTorch | [PyTorch for Deep Learning — Full Course](https://www.youtube.com/watch?v=V_xro1bcAuA) — freeCodeCamp, ~25h (dip in, do not watch it all) |
| `04/08` regularization | [Regularization](https://www.youtube.com/watch?v=6g0t3Phly2M) → [Why Regularization Reduces Overfitting](https://www.youtube.com/watch?v=NyG-7nRpsW8) → [Understanding Dropout](https://www.youtube.com/watch?v=ARq74QuavAo) — all Andrew Ng |
| `04/09` transfer learning | [Transfer Learning](https://www.youtube.com/watch?v=yofjFQddwHE) — Andrew Ng |

### Interactive instead of video — often better

- [Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) —
  ~3h, runs in the browser, nothing to install.
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) —
  interactive gradient-descent and loss widgets.
- [INRIA scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/) — by the
  scikit-learn core developers. The deepest free follow-up to module 02.
- [Andrew Ng's ML Specialization](https://www.coursera.org/specializations/machine-learning-introduction) —
  free to audit (choose **Audit**, not Enroll).

### Suggested order for a complete beginner

1. **Machine Learning for Everybody** (~4h) in one sitting — no notes, just absorb the vocabulary.
2. Module **01** notebooks.
3. Module **02**, watching the matching StatQuest video before each notebook.
4. **Kaggle Intro to ML** afterwards, as reinforcement.

For modules 03 and 04, watch per-notebook rather than front-loading — except the
3Blue1Brown series, which is worth watching straight through before you touch module 04.

---

## Attribution

- Module 01 and notebook `02/01` are practice notebooks from the
  [Zero to Mastery ML course](https://github.com/mrdbourke/zero-to-mastery-ml).
- Written for this path: `02/02`–`02/09`, `03/00`, `04/00`, `04/01`, `04/06`, `04/08`,
  `05/00`, `06/01`–`06/04`, `06/offline_llm.py`, and all of module 07.

### Reading, beyond the videos above

- [An Introduction to Statistical Learning (ISLP)](https://www.statlearning.com/) —
  free PDF + Python labs. The maths under module 02.
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/overview) — for module 06.
- *Designing Machine Learning Systems*, Chip Huyen — for module 07.
