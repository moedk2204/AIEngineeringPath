# Module 07 — How a Production ML Project Is Structured

Every notebook up to here has been a notebook. This module is about the other 90% of the
job: turning a model into something a team can run, test, deploy and fix at 3am.

There is a working example in `example-project/` — train a model, serve it over HTTP, test
it, containerise it. Read this page first, then go read that code.

---

## The one rule that matters

> **Training and serving are different programs with different lifecycles.**

Training is slow, runs occasionally, needs the whole dataset, and produces an **artifact**.
Serving is fast, runs constantly, needs one row at a time, and **consumes** that artifact.

Almost every structural mistake in ML engineering comes from blurring those two. If your
API imports `train.py`, or your model retrains on startup, or `predict()` reads the CSV —
stop and separate them.

```
   data  ──▶  scripts/train.py  ──▶  models/model.joblib  ──▶  the API  ──▶  users
              (runs sometimes)         (the artifact)          (runs always)
```

## The standard layout

```
project/
├── README.md                  what it is, how to run it, in 20 lines
├── requirements.txt           pinned versions
├── Makefile                   the commands a newcomer needs
├── Dockerfile                 how it ships
├── .env.example               every env var, no real values
├── .gitignore                 .env, .venv, model artifacts
│
├── data/                      never committed if it is large or private
├── models/                    build output, not source - gitignored
├── notebooks/                 exploration only; nothing imports from here
│
├── src/your_package/          THE ACTUAL APPLICATION
│   ├── __init__.py            version string
│   ├── config.py              all settings, read from the environment
│   ├── schemas.py             request/response contracts (Pydantic)
│   ├── model.py               load the artifact, make predictions
│   ├── api.py                 HTTP routes - thin
│   ├── main.py                app factory + startup/shutdown
│   └── logging_conf.py        structured logging
│
├── scripts/
│   └── train.py               produces models/model.joblib
│
└── tests/
    ├── conftest.py            shared fixtures
    ├── test_schemas.py        validation
    ├── test_model.py          the model service, no HTTP
    └── test_api.py            the contract, through a real client
```

### Why `src/`

Putting your package under `src/` means it must be **installed or on the path** to import.
That sounds like a nuisance; it is actually the point. It stops "works on my machine
because I happened to be in the right directory", and it makes the test suite exercise the
same import path production uses.

### Why notebooks are quarantined

Notebooks are for exploring. They run out of order, hold hidden state, and diff badly.
**Nothing in `src/` may import from `notebooks/`.** When notebook code becomes worth
keeping, move it into `src/` and write a test for it.

---

## The layers, and why each exists

| Layer | File | Job | Test |
|---|---|---|---|
| **Config** | `config.py` | one place for every setting, from env | rarely |
| **Schema** | `schemas.py` | reject bad input at the edge | fast, no I/O |
| **Service** | `model.py` | load artifact, predict | unit, no HTTP |
| **Route** | `api.py` | HTTP in, HTTP out — nothing else | via a test client |
| **App** | `main.py` | wire it together, load once at startup | smoke test |

The value of this split shows up when something changes:

- Swap scikit-learn for PyTorch → only `model.py` changes.
- Move from FastAPI to a batch job → `api.py` goes, `model.py` is reused as-is.
- Change the decision threshold → a config value, not a code deploy.

If a change forces you to edit four files, the layering is wrong.

---

## The eight things that separate a demo from a service

### 1. Configuration comes from the environment

No hardcoded paths, no keys in source. The **same image** runs in dev and prod — only the
environment differs. See `config.py`.

### 2. The model loads once, at startup

Loading inside a request handler adds hundreds of milliseconds to every call. Load into
`app.state` during the lifespan, and hold it. See `main.py`.

### 3. Validate at the edge

A Pydantic schema turns a malformed request into a clean `422` with a helpful message,
instead of a `500` from somewhere deep inside pandas. `age: int = Field(ge=0, le=120)`
is worth more than a page of defensive `if` statements.

### 4. A health endpoint that tells the truth

Orchestrators poll `/health`. It must be cheap and must never throw. Note the example
starts **even when the model is missing** and reports `degraded` — a container that refuses
to boot tells you nothing; a degraded one tells you exactly what is wrong.

### 5. Version every artifact

The bundle stores a version string and its metrics alongside the pipeline, and every
prediction response carries `model_version`. When someone asks "why did this prediction
change last Tuesday", that field is the entire answer.

### 6. A deploy gate in training

`scripts/train.py` refuses to write the artifact if the held-out AUC is below a threshold.
A bad model that never reaches `models/` cannot be deployed by accident.

### 7. Structured logs

`print()` is not logging. Emit JSON lines with a level, a logger name and a timestamp, so a
log aggregator can filter them. See `logging_conf.py`.

### 8. Tests that run in seconds

Schema and service tests need no model file and no network. They will catch most of your
mistakes before the slow tests even start.

---

## Deployment, in the order you should learn it

1. **`uvicorn` on your machine** — `make serve`, open `/docs`, click the endpoints.
2. **Docker** — `make docker`. Now it runs the same way everywhere.
3. **A managed container host** — Cloud Run, Render, Fly.io, Azure Container Apps. Push the
   image, get a URL. This is where most ML services should stop.
4. **Kubernetes** — only when you have a team to run it, and a reason.

Skipping to step 4 is the most common over-engineering mistake in this field.

### The Dockerfile details that matter

- **Multi-stage build** — compile dependencies in one stage, copy only the result. Smaller
  image, smaller attack surface.
- **Never run as root** — `useradd` and `USER appuser`.
- **Pin the base image** — `python:3.12-slim`, not `python:latest`.
- **A `HEALTHCHECK`** — so the orchestrator knows the difference between "running" and
  "working".
- **Copy `requirements.txt` before the source** — dependencies change rarely, source
  changes constantly. Get the layer cache the right way round.

---

## What this example deliberately leaves out

Real systems need these; they are out of scope for a first example, and knowing they are
missing is part of the lesson.

| Missing | What you would add |
|---|---|
| Authentication | an API key header or OAuth, and rate limiting |
| A database | store predictions for audit and later analysis |
| Monitoring | Prometheus metrics: latency, error rate, prediction distribution |
| **Drift detection** | alert when live inputs stop resembling training data |
| CI/CD | GitHub Actions: test → build → push → deploy |
| Model registry | MLflow or similar, instead of a joblib file in a folder |
| Experiment tracking | which run produced this model, with what data |
| Load testing | how many requests per second before it falls over |

**Drift is the one people forget.** A model does not degrade because the code broke — it
degrades because the world changed and nobody was watching the inputs.

---

## Try it

```bash
cd example-project
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt

python scripts/train.py            # writes models/model.joblib
pytest -q                          # 13 tests
uvicorn heart_api.main:app --reload --port 8000
```

Then open <http://localhost:8000/docs> — FastAPI generates interactive documentation from
the schemas, and you can send a request from the browser.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,
       "thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

---

## Exercises

1. **Break it on purpose.** Delete `models/model.joblib` and start the app. What does
   `/health` say? What does `/predict` return? Why is that better than crashing?
2. Add a `GET /model/info` endpoint returning the version, training date and stored
   metrics. Which file(s) do you have to touch? (Should be two.)
3. Change `decision_threshold` in `.env` from `0.35` to `0.7` and restart. Confirm
   `flagged` changes with no code edit. Connect this to module 02, notebook 05.
4. Add a test asserting that a patient with obviously low-risk values gets a lower
   probability than a high-risk one. Why is this a *better* test than asserting an exact
   number?
5. Add an `X-API-Key` header check as a FastAPI dependency. Return 401 when it is wrong.
   Where does the expected key come from — and where must it *not* come from?
6. **Stretch:** write a GitHub Actions workflow that installs dependencies, runs the tests,
   and builds the Docker image on every push.
7. **Stretch:** take the agent from module 06 and wrap it in this same structure — routes,
   schemas, a service layer, tests. The layering does not change just because the model is
   an LLM.

## Going further

- FastAPI docs: <https://fastapi.tiangolo.com/>
- *Designing Machine Learning Systems*, Chip Huyen — the best book on this subject.
- The FastAPI docs' "Bigger Applications" and SQL database guides, then Alembic for
  migrations, are the natural next steps beyond this example.
