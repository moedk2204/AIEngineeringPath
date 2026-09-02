"""
The model service: loading, and turning validated input into a prediction.

This is deliberately the ONLY file that knows about scikit-learn. The API layer
talks to `ModelService`, not to a pipeline - so you can swap sklearn for PyTorch
or a remote endpoint without touching a single route.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

# The exact column order the pipeline was trained on. Getting this wrong is a silent
# accuracy bug, not a crash - which is why it lives in code, not in someone's memory.
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before the model is available."""


class ModelService:
    def __init__(self, model_path: Path, threshold: float = 0.5) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._pipeline = None
        self._version: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def version(self) -> str | None:
        return self._version

    def load(self) -> None:
        """Load once at startup, never per request - loading is slow."""
        if not self.model_path.exists():
            logger.error("model file not found at %s", self.model_path)
            return
        bundle = joblib.load(self.model_path)
        self._pipeline = bundle["pipeline"]
        self._version = bundle.get("version", "unknown")
        logger.info("loaded model version %s from %s", self._version, self.model_path)

    def predict(self, records: list[dict]) -> list[dict]:
        if not self.is_loaded:
            raise ModelNotLoadedError("model is not loaded")

        frame = pd.DataFrame(records)[FEATURE_ORDER]   # enforce column order
        probabilities = self._pipeline.predict_proba(frame)[:, 1]

        return [
            {
                "probability": round(float(p), 4),
                "flagged": bool(p >= self.threshold),
                "threshold": self.threshold,
                "model_version": self._version or "unknown",
            }
            for p in probabilities
        ]
