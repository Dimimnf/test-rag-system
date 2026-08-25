from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config import get_settings


def create_chat_model() -> BaseChatModel:
    """Создаёт chat-модель для настроенного OpenAI-compatible API."""
    settings = get_settings()
    parameters = {
        "api_key": settings.openai_api_key.get_secret_value(),
        "model": settings.openai_model,
        "temperature": 0,
    }
    if settings.openai_base_url is not None:
        parameters["base_url"] = str(settings.openai_base_url)
    return ChatOpenAI(**parameters)
