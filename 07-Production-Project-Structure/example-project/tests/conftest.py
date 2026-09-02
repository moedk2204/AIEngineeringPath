"""Shared pytest fixtures. Also puts `src/` on the path so `heart_api` imports."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from heart_api.main import create_app


@pytest.fixture(scope="session")
def client():
    """A real HTTP client against the app, with startup/shutdown run properly."""
    with TestClient(create_app()) as c:
        yield c
