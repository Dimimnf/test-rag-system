from pathlib import Path
from typing import cast

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from services.ai.rag.exceptions import RagNotReadyError
from services.ai.rag.service import Rag
from services.ai.rag.vector_store import VectorStoreRepository


class FakeProcessor:
    """Возвращает заранее подготовленные чанки без чтения документа."""

    def __init__(self, chunks: list[Document]) -> None:
        self.chunks = chunks
        self.received: Path | None = None

    def process(self, document: Path) -> list[Document]:
        """Запоминает путь и возвращает тестовые чанки."""
        self.received = document
        return self.chunks


class FakeRepository:
    """Запоминает операции фасада без обращения к Chroma."""

    def __init__(self) -> None:
        self.replaced: tuple[Path, list[Document]] | None = None
        self.requested_k: int | None = None
        self.retriever = cast(VectorStoreRetriever, object())

    def replace(self, source: Path, chunks: list[Document]) -> None:
        """Запоминает замену активного документа."""
        self.replaced = (source, chunks)

    def get_retriever(self, k: int) -> VectorStoreRetriever:
        """Запоминает лимит и возвращает тестовый retriever."""
        self.requested_k = k
        return self.retriever


def test_rag_facade_coordinates_processor_and_repository(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("Текст", encoding="utf-8")
    chunks = [Document(page_content="Текст")]
    processor = FakeProcessor(chunks)
    repository = FakeRepository()
    rag = Rag(processor=processor, repository=repository)  # type: ignore[arg-type]

    count = rag.change(source)
    retriever = rag.get_retriever(k=4)

    assert count == 1
    assert processor.received == source.resolve()
    assert repository.replaced == (source.resolve(), chunks)
    assert repository.requested_k == 4
    assert retriever is repository.retriever


def test_repository_rejects_invalid_k(tmp_path: Path) -> None:
    repository = VectorStoreRepository(data_directory=tmp_path)

    with pytest.raises(ValueError):
        repository.get_retriever(0)


def test_repository_requires_active_marker(tmp_path: Path) -> None:
    repository = VectorStoreRepository(data_directory=tmp_path)

    with pytest.raises(RagNotReadyError):
        repository.get_retriever(1)
