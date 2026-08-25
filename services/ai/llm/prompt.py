from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """\
Вы — помощник, отвечающий на вопросы по загруженному документу.
Используйте только приведённый контекст и не добавляйте внешние знания.
Если контекста недостаточно для точного ответа, прямо сообщите об этом.

Контекст:
{context}
"""

QUESTION_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Вопрос: {question}"),
    ]
)


def format_context(documents: list[Document]) -> str:
    """Объединяет найденные документы в разделённый контекст.

    Args:
        documents: Релевантные документы от retriever.

    Returns:
        Текст контекста в порядке релевантности.
    """
    return "\n\n---\n\n".join(
        document.page_content.strip()
        for document in documents
        if document.page_content.strip()
    )
