from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import get_logger
from services.ai.rag.exceptions import (
    DocumentReadError,
    EmptyDocumentError,
    UnsupportedDocumentError,
)


logger = get_logger(__name__)
SUPPORTED_SUFFIXES = {".pdf", ".txt"}


class DocumentProcessor:
    """Извлекает текст из поддерживаемых документов и создаёт чанки."""

    def __init__(
        self,
        splitter: RecursiveCharacterTextSplitter | None = None,
    ) -> None:
        """Создаёт обработчик с настраиваемым разделителем текста.

        Args:
            splitter: Разделитель документов на чанки.
        """
        self._splitter = splitter or RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            separators=["\n\n", "\n", ".", " "],
        )

    @staticmethod
    def _load(document: Path) -> list[Document]:
        """Загружает страницы TXT или PDF в документы LangChain.

        Args:
            document: Путь к локальному документу.

        Returns:
            Страницы с извлечённым текстом и метаданными.
        """
        suffix = document.suffix.lower()
        if suffix == ".txt":
            loader = TextLoader(
                str(document),
                encoding="utf-8",
                autodetect_encoding=True,
            )
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(document))
        else:
            raise UnsupportedDocumentError(
                "Поддерживаются только документы TXT и PDF"
            )

        try:
            return loader.load()
        except Exception as error:
            raise DocumentReadError("Не удалось прочитать документ") from error

    def process(self, document: str | Path) -> list[Document]:
        """Проверяет документ, извлекает текст и возвращает непустые чанки.

        Args:
            document: Путь к подготовленному локальному файлу.

        Returns:
            Непустые чанки документа для векторизации.
        """
        source = Path(document).resolve()
        if not source.is_file():
            raise DocumentReadError(f"Файл документа не найден: {source}")
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise UnsupportedDocumentError(
                "Поддерживаются только документы TXT и PDF"
            )

        logger.info("Начинается обработка документа формата %s", source.suffix)
        pages = self._load(source)
        chunks = [
            chunk
            for chunk in self._splitter.split_documents(pages)
            if chunk.page_content.strip()
        ]
        if not chunks:
            raise EmptyDocumentError("Документ не содержит извлекаемого текста")

        logger.info("Документ разделён на чанки, количество: %s", len(chunks))
        return chunks
