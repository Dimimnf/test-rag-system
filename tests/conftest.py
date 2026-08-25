import os

import pytest
from fastapi.testclient import TestClient


# Тестовый ключ нужен только для валидации Settings и не используется в сети.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_MODEL", "test-model")

from main import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    """Возвращает синхронный клиент тестового FastAPI-приложения."""
    with TestClient(app) as test_client:
        yield test_client
