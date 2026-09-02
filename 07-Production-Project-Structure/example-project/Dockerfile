# Multi-stage: build deps in one layer, ship only what runs.
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
WORKDIR /app

# Never run as root in a container.
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY models/ ./models/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=models/model.joblib

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "heart_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
