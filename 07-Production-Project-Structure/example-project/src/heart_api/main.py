"""
Application factory and lifespan.

The model is loaded ONCE at startup and held on `app.state`. Loading it inside a
request handler would add hundreds of milliseconds to every call.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .api import router
from .config import settings
from .logging_conf import configure_logging
from .model import ModelService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    logger.info("starting %s v%s", settings.app_name, __version__)

    service = ModelService(settings.model_path, settings.decision_threshold)
    service.load()
    app.state.model_service = service

    if not service.is_loaded:
        # Start anyway so /health can report 'degraded' - a container that refuses to
        # boot tells you nothing. A degraded one tells you exactly what is wrong.
        logger.warning("started WITHOUT a model - /predict will return 503")

    yield
    logger.info("shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Flags patients who may need a cardiology review.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
