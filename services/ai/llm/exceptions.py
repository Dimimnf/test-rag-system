class LlmProviderError(RuntimeError):
    """Сообщает об ошибке OpenAI-compatible провайдера."""


class InvalidLlmResponseError(LlmProviderError):
    """Сообщает, что модель вернула ответ без текста."""
