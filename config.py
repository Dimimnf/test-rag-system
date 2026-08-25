from functools import lru_cache

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Хранит настройки RAG-сервиса, загружаемые из окружения."""

    openai_api_key: SecretStr = Field(
        alias="OPENAI_API_KEY",
        title="Ключ OpenAI API",
        description="Секретный ключ OpenAI или совместимого провайдера.",
        examples=["sk-example"],
    )
    openai_model: str = Field(
        alias="OPENAI_MODEL",
        min_length=1,
        title="Модель генерации",
        description="Название chat-модели у выбранного провайдера.",
        examples=["gpt-4o-mini"],
    )
    openai_base_url: HttpUrl | None = Field(
        default=None,
        alias="OPENAI_BASE_URL",
        title="Адрес OpenAI-compatible API",
        description="Базовый URL стороннего провайдера; для OpenAI можно не задавать.",
        examples=["https://api.openai.com/v1"],
    )
    rag_top_k: int = Field(
        default=3,
        alias="RAG_TOP_K",
        ge=1,
        le=10,
        title="Количество чанков",
        description="Число релевантных чанков, передаваемых модели.",
        examples=[3],
    )
    upload_max_bytes: int = Field(
        default=5 * 1024 * 1024,
        alias="UPLOAD_MAX_BYTES",
        ge=1,
        title="Максимальный размер документа",
        description="Максимальный размер загружаемого файла в байтах.",
        examples=[5242880],
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает единый проверенный экземпляр настроек приложения."""
    return Settings()
