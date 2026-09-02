"""Unit tests for the model service, independent of HTTP."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heart_api.model import FEATURE_ORDER, ModelNotLoadedError, ModelService

from sample_data import VALID_PATIENT


def test_predicting_before_load_raises():
    service = ModelService(Path("does/not/exist.joblib"))
    assert not service.is_loaded
    with pytest.raises(ModelNotLoadedError):
        service.predict([VALID_PATIENT])


def test_missing_file_does_not_crash_load():
    service = ModelService(Path("does/not/exist.joblib"))
    service.load()                    # logs an error, does not raise
    assert not service.is_loaded


def test_feature_order_matches_the_schema():
    """Guards the silent-accuracy-bug: schema fields and model columns must agree."""
    from heart_api.schemas import PatientFeatures
    assert set(FEATURE_ORDER) == set(PatientFeatures.model_fields)
