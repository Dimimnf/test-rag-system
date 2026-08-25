from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from config import get_settings
from logger import get_logger
from services.ai.llm.exceptions import InvalidLlmResponseError, LlmProviderError
from services.ai.llm.model import create_chat_model
from services.ai.llm.prompt import QUESTION_ANSWER_PROMPT, format_context
from services.ai.rag import Rag, RagNotReadyError, rag_service


logger = get_logger(__name__)


class Llm:
    """Координирует поиск контекста и генерацию ответа моделью."""

    def __init__(
        self,
        rag: Rag = rag_service,
        model: BaseChatModel | None = None,
        model_factory: Callable[[], BaseChatModel] = create_chat_model,
    ) -> None:
        """Создаёт LLM-фасад с заменяемыми RAG и chat-моделью.

        Args:
            rag: Фасад для получения retriever активного документа.
            model: Готовая chat-модель, обычно используемая в тестах.
            model_factory: Ленивая фабрика модели для рабочего окружения.
        """
        self._rag = rag
        self._model = model
        self._model_factory = model_factory

    def _get_model(self) -> BaseChatModel:
        """Лениво создаёт и переиспользует chat-модель."""
        if self._model is None:
            self._model = self._model_factory()
        return self._model

    @staticmethod
    def _extract_answer(message: AIMessage) -> str:
        """Извлекает текст из стандартного ответа LangChain.

        Args:
            message: Ответ chat-модели.

        Returns:
            Непустой текст ответа.
        """
        content = message.content
        if isinstance(content, str):
            answer = content.strip()
        else:
            # Совместимые провайдеры могут вернуть текст отдельными блоками.
            answer = "\n".join(
                block if isinstance(block, str) else str(block.get("text", ""))
                for block in content
            ).strip()
        if not answer:
            raise InvalidLlmResponseError("Модель вернула пустой ответ")
        return answer

    def answer(self, question: str) -> str:
        """Находит релевантный контекст и генерирует по нему ответ.

        Args:
            question: Непустой вопрос пользователя.

        Returns:
            Текст ответа OpenAI-compatible модели.
        """
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Вопрос не должен быть пустым")

        try:
            settings = get_settings()
            retriever = self._rag.get_retriever(k=settings.rag_top_k)
            documents = retriever.invoke(normalized_question)
            prompt = QUESTION_ANSWER_PROMPT.invoke(
                {
                    "context": format_context(documents),
                    "question": normalized_question,
                }
            )
            response = self._get_model().invoke(prompt)
            if not isinstance(response, AIMessage):
                raise InvalidLlmResponseError("Модель вернула ответ неизвестного типа")
            return self._extract_answer(response)
        except RagNotReadyError:
            raise
        except (InvalidLlmResponseError, LlmProviderError):
            raise
        except Exception as error:
            logger.exception("Ошибка при генерации ответа")
            raise LlmProviderError("Не удалось получить ответ модели") from error


llm_service = Llm()
