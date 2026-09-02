"""
Training is a SCRIPT, not a notebook, and not part of the API.

Run it, get a versioned artifact, ship the artifact. The API never trains.
Separating these two is the single most important structural decision in an ML project.

    python scripts/train.py
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heart_api.model import FEATURE_ORDER          # one source of truth for columns

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "heart-disease.csv"
OUT = ROOT / "models" / "model.joblib"

CATEGORICAL = ["cp", "restecg", "slope", "thal", "ca"]
NUMERIC = [c for c in FEATURE_ORDER if c not in CATEGORICAL]


def build_pipeline() -> Pipeline:
    """Preprocessing lives INSIDE the pipeline - module 02, notebook 04."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    return Pipeline([
        ("prep", pre),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       class_weight="balanced", random_state=42)),
    ])


def main() -> int:
    df = pd.read_csv(DATA)
    X, y = df[FEATURE_ORDER], df["target"]
    logger.info("loaded %d rows, %d features", len(X), X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = build_pipeline()

    cv = cross_val_score(pipeline, X_train, y_train,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="roc_auc")
    logger.info("cross-validated ROC-AUC: %.3f +/- %.3f", cv.mean(), cv.std())

    pipeline.fit(X_train, y_train)

    # The honest number: the held-out set, used once.
    test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    logger.info("held-out test ROC-AUC:    %.3f", test_auc)
    print(classification_report(y_test, pipeline.predict(X_test),
                                target_names=["no disease", "disease"], digits=3))

    # A deploy gate. If the model is not good enough, do NOT write the artifact.
    if test_auc < 0.75:
        logger.error("test AUC %.3f below the 0.75 gate - refusing to save", test_auc)
        return 1

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "version": version,
        "features": FEATURE_ORDER,
        "metrics": {"cv_roc_auc": round(float(cv.mean()), 4),
                    "test_roc_auc": round(float(test_auc), 4)},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, OUT)

    (OUT.parent / "metrics.json").write_text(json.dumps(
        {"version": version, "cv_roc_auc": round(float(cv.mean()), 4),
         "test_roc_auc": round(float(test_auc), 4)}, indent=2))

    logger.info("saved model version %s to %s", version, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
