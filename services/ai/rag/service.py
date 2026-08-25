import threading
from pathlib import Path

from langchain_core.vectorstores import VectorStoreRetriever

from logger import get_logger
from services.ai.rag.document_processor import DocumentProcessor
from services.ai.rag.vector_store import VectorStoreRepository


logger = get_logger(__name__)


class Rag:
    """Координирует обработку документа и доступ к векторному индексу."""

    def __init__(
        self,
        processor: DocumentProcessor | None = None,
        repository: VectorStoreRepository | None = None,
    ) -> None:
        """Создаёт RAG-фасад с заменяемыми зависимостями.

        Args:
            processor: Обработчик исходных документов.
            repository: Репозиторий persistent-векторов.
        """
        self._processor = processor or DocumentProcessor()
        self._repository = repository or VectorStoreRepository()
        self._lock = threading.RLock()

    def change(self, document: str | Path) -> int:
        """Полностью заменяет активный документ и возвращает число чанков.

        Args:
            document: Путь к новому TXT или PDF.

        Returns:
            Количество добавленных в индекс чанков.
        """
        source = Path(document).resolve()
        with self._lock:
            logger.info("Начинается замена активного документа")
            chunks = self._processor.process(source)
            self._repository.replace(source, chunks)
            logger.info("Активный документ заменён, чанков: %s", len(chunks))
            return len(chunks)

    def get_retriever(self, k: int = 1) -> VectorStoreRetriever:
        """Возвращает retriever активного документа.

        Args:
            k: Максимальное количество результатов поиска.

        Returns:
            Retriever persistent Chroma-индекса.
        """
        with self._lock:
            return self._repository.get_retriever(k)


rag_service = Rag()
