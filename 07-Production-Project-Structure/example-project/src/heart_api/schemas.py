"""
The API contract. Pydantic validates every request before your code sees it,
so a bad payload is a clean 422 instead of a 500 from deep inside sklearn.
"""
from typing import Literal

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """One patient. Field constraints reject impossible values at the edge."""

    age: int = Field(ge=0, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(description="0 = female, 1 = male")
    cp: Literal[0, 1, 2, 3] = Field(description="Chest pain type")
    trestbps: float = Field(ge=50, le=260, description="Resting blood pressure (mm Hg)")
    chol: float = Field(ge=100, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(description="Resting ECG result")
    thalach: float = Field(ge=60, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(description="Exercise-induced angina")
    oldpeak: float = Field(ge=0, le=10, description="ST depression")
    slope: Literal[0, 1, 2] = Field(description="Slope of peak exercise ST segment")
    ca: Literal[0, 1, 2, 3, 4] = Field(description="Number of major vessels")
    thal: Literal[0, 1, 2, 3] = Field(description="Thalassemia result")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
            }]
        }
    }


class Prediction(BaseModel):
    probability: float = Field(ge=0, le=1, description="Probability of heart disease")
    flagged: bool = Field(description="True if above the review threshold")
    threshold: float = Field(description="Threshold used for this decision")
    model_version: str = Field(description="Which model produced this")


class BatchRequest(BaseModel):
    patients: list[PatientFeatures] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
