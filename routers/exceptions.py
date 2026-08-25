from fastapi import HTTPException, status


class UnsupportedDocumentHttpError(HTTPException):
    """Возвращает 415 для неподдерживаемого формата или MIME-типа."""

    def __init__(self, detail: str) -> None:
        """Создаёт ошибку с причиной отклонения документа."""
        super().__init__(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=detail,
        )


class UploadTooLargeHttpError(HTTPException):
    """Возвращает 413 при превышении допустимого размера upload."""

    def __init__(self, detail: str) -> None:
        """Создаёт ошибку с описанием ограничения размера."""
        super().__init__(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=detail,
        )


class InvalidDocumentHttpError(HTTPException):
    """Возвращает 422 для повреждённого документа или документа без текста."""

    def __init__(self, detail: str) -> None:
        """Создаёт ошибку с причиной невозможности индексирования."""
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=detail,
        )


class RagNotReadyHttpError(HTTPException):
    """Возвращает 409, если активный RAG-индекс ещё не создан."""

    def __init__(self, detail: str) -> None:
        """Создаёт ошибку с текущим состоянием RAG-индекса."""
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)


class LlmProviderHttpError(HTTPException):
    """Возвращает 502 при ошибке OpenAI-compatible провайдера."""

    def __init__(self) -> None:
        """Создаёт безопасную ошибку без деталей внешнего провайдера."""
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Не удалось получить ответ модели",
        )


class DocumentIndexingHttpError(HTTPException):
    """Возвращает 500 при непредвиденной ошибке индексирования."""

    def __init__(self) -> None:
        """Создаёт безопасную внутреннюю ошибку индексирования."""
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось проиндексировать документ",
        )


class QuestionProcessingHttpError(HTTPException):
    """Возвращает 500 при непредвиденной ошибке обработки вопроса."""

    def __init__(self) -> None:
        """Создаёт безопасную внутреннюю ошибку обработки вопроса."""
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось обработать вопрос",
        )
