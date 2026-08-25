from fastapi.testclient import TestClient
import pytest

import routers.document as document_router
from services.ai.llm import LlmProviderError
from services.ai.rag import EmptyDocumentError, RagNotReadyError


def test_uploads_text_document(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(document_router.rag_service, "change", lambda path: 2)

    response = client.post(
        "/documents",
        files={"file": ("rules.txt", b"Document text", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json() == {"filename": "rules.txt", "chunks": 2}


def test_uploads_pdf_document(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(document_router.rag_service, "change", lambda path: 1)

    response = client.post(
        "/documents",
        files={"file": ("rules.pdf", b"%PDF-test", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["chunks"] == 1


def test_rejects_unsupported_file(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("rules.docx", b"content", "application/octet-stream")},
    )

    assert response.status_code == 415


def test_rejects_mismatched_mime_type(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("rules.pdf", b"content", "text/plain")},
    )

    assert response.status_code == 415


def test_rejects_file_larger_than_five_megabytes(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("rules.txt", b"x" * (5 * 1024 * 1024 + 1), "text/plain")},
    )

    assert response.status_code == 413


def test_rejects_document_without_text(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_empty_document(path: object) -> int:
        raise EmptyDocumentError("Документ не содержит извлекаемого текста")

    monkeypatch.setattr(document_router.rag_service, "change", raise_empty_document)

    response = client.post(
        "/documents",
        files={"file": ("scan.pdf", b"%PDF-scan", "application/pdf")},
    )

    assert response.status_code == 422


def test_answers_question(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        document_router.llm_service,
        "answer",
        lambda question: "Ответ из документа",
    )

    response = client.post(
        "/documents/questions",
        json={"question": "Что указано?"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "Ответ из документа"}


def test_rejects_empty_question(client: TestClient) -> None:
    response = client.post(
        "/documents/questions",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_reports_missing_active_document(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_not_ready(question: str) -> str:
        raise RagNotReadyError("Активный документ ещё не проиндексирован")

    monkeypatch.setattr(document_router.llm_service, "answer", raise_not_ready)

    response = client.post(
        "/documents/questions",
        json={"question": "Что указано?"},
    )

    assert response.status_code == 409


def test_maps_provider_error_to_bad_gateway(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_provider_error(question: str) -> str:
        raise LlmProviderError("provider unavailable")

    monkeypatch.setattr(document_router.llm_service, "answer", raise_provider_error)

    response = client.post(
        "/documents/questions",
        json={"question": "Что указано?"},
    )

    assert response.status_code == 502
