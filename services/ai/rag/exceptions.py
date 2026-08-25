class EmptyDocumentError(ValueError):
    """Сообщает, что документ не содержит пригодного для индексации текста."""


class UnsupportedDocumentError(ValueError):
    """Сообщает, что формат документа не поддерживается."""


class DocumentReadError(ValueError):
    """Сообщает, что содержимое документа не удалось прочитать."""


class RagNotReadyError(RuntimeError):
    """Сообщает, что активный RAG-индекс ещё не создан."""
