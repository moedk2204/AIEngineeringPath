"""Validation tests. These need no model file and run in milliseconds."""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heart_api.schemas import PatientFeatures

from sample_data import VALID_PATIENT


def test_valid_patient_parses():
    p = PatientFeatures(**VALID_PATIENT)
    assert p.age == 63


@pytest.mark.parametrize("field,bad_value", [
    ("age", 200),          # above the ge/le bound
    ("sex", 2),            # not in the Literal
    ("chol", 5),           # below the bound
    ("thal", 9),           # not in the Literal
])
def test_out_of_range_is_rejected(field, bad_value):
    payload = {**VALID_PATIENT, field: bad_value}
    with pytest.raises(ValidationError):
        PatientFeatures(**payload)


def test_missing_field_is_rejected():
    payload = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
    with pytest.raises(ValidationError):
        PatientFeatures(**payload)
