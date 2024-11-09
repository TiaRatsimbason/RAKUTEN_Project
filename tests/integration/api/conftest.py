import pytest
from fastapi.testclient import TestClient
from src.api.app import app  # Remplacez avec le chemin correct de votre application FastAPI

@pytest.fixture
def client():
    return TestClient(app)