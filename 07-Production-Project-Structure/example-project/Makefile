.PHONY: install train test serve docker clean

install:
	python -m venv .venv && .venv/bin/pip install -r requirements.txt

train:
	python scripts/train.py

test:
	pytest -q

serve:
	uvicorn heart_api.main:app --reload --port 8000

docker:
	docker build -t heart-risk-api . && docker run -p 8000:8000 heart-risk-api

clean:
	rm -rf __pycache__ .pytest_cache models/*.joblib
