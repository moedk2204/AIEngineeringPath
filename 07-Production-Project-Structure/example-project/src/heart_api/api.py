"""HTTP routes. Thin - they validate, delegate, and translate errors into status codes."""
import logging

from fastapi import APIRouter, HTTPException, Request

from .schemas import (BatchRequest, BatchResponse, HealthResponse,
                      PatientFeatures, Prediction)
from .model import ModelNotLoadedError

logger = logging.getLogger(__name__)
router = APIRouter()


def _service(request: Request):
    return request.app.state.model_service


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    """Liveness probe. Orchestrators poll this - keep it cheap and never let it throw."""
    service = _service(request)
    return HealthResponse(
        status="ok" if service.is_loaded else "degraded",
        model_loaded=service.is_loaded,
        model_version=service.version,
    )


@router.post("/predict", response_model=Prediction, tags=["inference"])
def predict(patient: PatientFeatures, request: Request) -> Prediction:
    try:
        result = _service(request).predict([patient.model_dump()])[0]
    except ModelNotLoadedError:
        # 503, not 500: the service is temporarily unable, and a retry may succeed.
        raise HTTPException(status_code=503, detail="Model not loaded")
    return Prediction(**result)


@router.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(payload: BatchRequest, request: Request) -> BatchResponse:
    try:
        results = _service(request).predict([p.model_dump() for p in payload.patients])
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    logger.info("scored batch of %d", len(results))
    return BatchResponse(predictions=[Prediction(**r) for r in results])
