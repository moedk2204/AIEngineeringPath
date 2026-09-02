# Heart Risk API — a minimal production ML service

A complete, working example of the structure described in `../README.md`.
Trains a classifier on the heart-disease dataset and serves it over HTTP.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python scripts/train.py           # -> models/model.joblib
pytest -q                         # 13 tests
uvicorn heart_api.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> for interactive docs generated from the schemas.

> If `uvicorn` cannot find `heart_api`, set `PYTHONPATH=src` (the `Makefile` and
> `Dockerfile` both do this).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness + whether a model is loaded |
| `POST` | `/predict` | score one patient |
| `POST` | `/predict/batch` | score up to 1000 patients |
| `GET` | `/docs` | interactive API documentation |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,
       "thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

```json
{"probability": 0.6544, "flagged": true, "threshold": 0.35,
 "model_version": "20260902-014737"}
```

## What each file is for

| File | Responsibility |
|---|---|
| `scripts/train.py` | produces the artifact; **never imported by the API** |
| `src/heart_api/config.py` | every setting, read from the environment |
| `src/heart_api/schemas.py` | request/response contracts; rejects bad input |
| `src/heart_api/model.py` | the only file that knows about scikit-learn |
| `src/heart_api/api.py` | HTTP routes — thin, no business logic |
| `src/heart_api/main.py` | app factory; loads the model once at startup |
| `tests/` | schema, service and contract tests |

## Notes worth reading in the code

- `main.py` starts the app **even without a model file**, and `/health` reports
  `degraded`. Failing to boot tells an operator nothing.
- `model.py` pins `FEATURE_ORDER`. A reordered column is a silent accuracy bug, not a
  crash — `test_model.py` asserts it matches the schema.
- `train.py` refuses to save a model scoring below 0.75 AUC. A bad model that never
  reaches `models/` cannot be deployed by mistake.
- `decision_threshold` is **config**, not code. Changing who gets flagged is a business
  decision (module 02, notebook 05) and should not need a deploy.
- Every response carries `model_version`, so any prediction can be traced to the run that
  produced it.

## Docker

```bash
docker build -t heart-risk-api .
docker run -p 8000:8000 heart-risk-api
# or
docker compose up --build
```

Train **before** building — the image copies `models/` in. Baking the artifact into the
image makes the deploy reproducible; the alternative is fetching it at startup from a
model registry or object store.
