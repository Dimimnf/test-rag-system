from typing import Any, cast

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from services.ai.llm.exceptions import LlmProviderError
from services.ai.llm.prompt import format_context
from services.ai.llm.service import Llm
from services.ai.rag import Rag


class FakeRetriever:
    """Возвращает подготовленный контекст и запоминает вопрос."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.question: str | None = None

    def invoke(self, question: str) -> list[Document]:
        """Возвращает документы для переданного вопроса."""
        self.question = question
        return self.documents


class FakeRag:
    """Выдаёт fake retriever и запоминает параметр поиска."""

    def __init__(self, retriever: FakeRetriever) -> None:
        self.retriever = retriever
        self.k: int | None = None

    def get_retriever(self, k: int = 1) -> FakeRetriever:
        """Возвращает подготовленный retriever."""
        self.k = k
        return self.retriever


class FakeModel:
    """Запоминает prompt и возвращает заданный ответ."""

    def __init__(self, response: AIMessage | Exception) -> None:
        self.response = response
        self.prompt: Any = None

    def invoke(self, prompt: Any) -> AIMessage:
        """Возвращает ответ или имитирует ошибку провайдера."""
        self.prompt = prompt
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_formats_documents_with_explicit_separator() -> None:
    context = format_context(
        [Document(page_content="Первый"), Document(page_content="Второй")]
    )

    assert context == "Первый\n\n---\n\nВторой"


def test_generates_answer_from_retrieved_context() -> None:
    retriever = FakeRetriever([Document(page_content="Срок: 5 дней")])
    rag = FakeRag(retriever)
    model = FakeModel(AIMessage(content="Пять дней."))
    llm = Llm(
        rag=cast(Rag, rag),
        model=cast(BaseChatModel, model),
    )

    answer = llm.answer(" Какой срок? ")
    messages = model.prompt.to_messages()

    assert answer == "Пять дней."
    assert rag.k == 3
    assert retriever.question == "Какой срок?"
    assert "Срок: 5 дней" in messages[0].content
    assert "Какой срок?" in messages[1].content


def test_maps_model_failure_to_provider_error() -> None:
    retriever = FakeRetriever([Document(page_content="Контекст")])
    model = FakeModel(RuntimeError("provider unavailable"))
    llm = Llm(
        rag=cast(Rag, FakeRag(retriever)),
        model=cast(BaseChatModel, model),
    )

    with pytest.raises(LlmProviderError):
        llm.answer("Вопрос")
