from pydantic import BaseModel, Field


class UploadDocumentResponse(BaseModel):
    """Описывает результат успешной индексации документа."""

    filename: str = Field(
        title="Имя файла",
        description="Исходное имя загруженного документа.",
        examples=["regulations.pdf"],
    )
    chunks: int = Field(
        ge=1,
        title="Количество чанков",
        description="Количество фрагментов, сохранённых в Chroma.",
        examples=[12],
    )


class QuestionResponse(BaseModel):
    """Описывает ответ модели по активному документу."""

    answer: str = Field(
        min_length=1,
        title="Ответ",
        description="Ответ, сформированный только по найденному контексту.",
        examples=["Срок согласования составляет пять рабочих дней."],
    )
