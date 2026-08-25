from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    """Описывает запрос с вопросом по активному документу."""

    question: str = Field(
        min_length=1,
        max_length=4000,
        title="Вопрос",
        description="Вопрос, ответ на который нужно найти в активном документе.",
        examples=["Какой срок согласования указан в документе?"],
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Удаляет внешние пробелы и отклоняет пустой вопрос.

        Args:
            value: Исходный вопрос из тела запроса.

        Returns:
            Вопрос без внешних пробелов.
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("Вопрос не должен быть пустым")
        return normalized
