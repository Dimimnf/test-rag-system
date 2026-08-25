from pathlib import Path

import pytest
from langchain_core.documents import Document

from services.ai.rag.document_processor import DocumentProcessor
from services.ai.rag.exceptions import EmptyDocumentError, UnsupportedDocumentError


def test_processes_text_document(tmp_path: Path) -> None:
    document = tmp_path / "document.txt"
    document.write_text("Первый раздел.\n\nВторой раздел.", encoding="utf-8")

    chunks = DocumentProcessor().process(document)

    assert chunks
    assert "Первый раздел" in chunks[0].page_content


def test_dispatches_pdf_to_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document = tmp_path / "document.pdf"
    document.write_bytes(b"%PDF-test")
    processor = DocumentProcessor()
    loaded = [Document(page_content="Текст PDF")]
    monkeypatch.setattr(processor, "_load", lambda path: loaded)

    chunks = processor.process(document)

    assert chunks[0].page_content == "Текст PDF"


def test_rejects_document_without_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "scan.pdf"
    document.write_bytes(b"%PDF-test")
    processor = DocumentProcessor()
    monkeypatch.setattr(
        processor,
        "_load",
        lambda path: [Document(page_content="   ")],
    )

    with pytest.raises(EmptyDocumentError):
        processor.process(document)


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    document = tmp_path / "document.docx"
    document.write_bytes(b"content")

    with pytest.raises(UnsupportedDocumentError):
        DocumentProcessor().process(document)
