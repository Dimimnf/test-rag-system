import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from logger import get_logger
from services.ai.rag.exceptions import RagNotReadyError


logger = get_logger(__name__)
PROJECT_DIRECTORY = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
COLLECTION_NAME = "active_document"


class VectorStoreRepository:
    """Управляет файлами и persistent Chroma-индексом активного документа."""

    def __init__(
        self,
        data_directory: Path = DATA_DIRECTORY,
        embeddings: Embeddings | None = None,
    ) -> None:
        """Создаёт репозиторий с настраиваемым каталогом и embeddings.

        Args:
            data_directory: Каталог persistent-данных сервиса.
            embeddings: Готовая модель embeddings для тестов или переиспользования.
        """
        self._data_directory = data_directory
        self._chroma_directory = data_directory / "chroma"
        self._upload_directory = data_directory / "uploads"
        self._marker = data_directory / "active_document"
        self._embeddings = embeddings
        self._vector_store: Chroma | None = None

    def _get_embeddings(self) -> Embeddings:
        """Лениво создаёт и возвращает модель локальных embeddings."""
        if self._embeddings is None:
            # Модель загружается только при первом индексировании или поиске.
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._embeddings

    def _delete_collection(self) -> None:
        """Удаляет предыдущую коллекцию и связанные локальные файлы."""
        if self._chroma_directory.exists():
            vector_store = self._vector_store or Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self._get_embeddings(),
                persist_directory=str(self._chroma_directory),
            )
            try:
                vector_store.delete_collection()
            except ValueError:
                logger.info("Предыдущая коллекция Chroma уже отсутствует")

        self._vector_store = None
        self._marker.unlink(missing_ok=True)
        for active_document in self._upload_directory.glob("active.*"):
            active_document.unlink(missing_ok=True)

    def replace(self, source: Path, chunks: list[Document]) -> None:
        """Заменяет исходник и Chroma-индекс активного документа.

        Args:
            source: Проверенный локальный файл документа.
            chunks: Подготовленные непустые чанки.
        """
        self._upload_directory.mkdir(parents=True, exist_ok=True)
        incoming = self._upload_directory / f".incoming{source.suffix.lower()}"
        active = self._upload_directory / f"active{source.suffix.lower()}"

        try:
            shutil.copy2(source, incoming)
            # Marker удаляется до замены, чтобы незавершённый индекс не считался готовым.
            self._delete_collection()
            incoming.replace(active)
            for chunk in chunks:
                chunk.metadata = {**chunk.metadata, "source": str(active)}

            self._vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self._get_embeddings(),
                collection_name=COLLECTION_NAME,
                persist_directory=str(self._chroma_directory),
            )
            # Marker публикуется последним после успешной записи Chroma.
            self._marker.write_text(COLLECTION_NAME, encoding="utf-8")
        finally:
            incoming.unlink(missing_ok=True)

    def get_retriever(self, k: int) -> VectorStoreRetriever:
        """Возвращает retriever готового индекса с указанным лимитом.

        Args:
            k: Максимальное количество релевантных чанков.

        Returns:
            Retriever активной коллекции Chroma.
        """
        if k <= 0:
            raise ValueError("Количество результатов k должно быть больше нуля")
        if not self._marker.is_file():
            raise RagNotReadyError("Активный документ ещё не проиндексирован")

        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self._get_embeddings(),
                persist_directory=str(self._chroma_directory),
                create_collection_if_not_exists=False,
            )
        return self._vector_store.as_retriever(search_kwargs={"k": k})
