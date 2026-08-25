from fastapi import FastAPI

from routers import document_router


app = FastAPI(
    title="RAG Document Service",
    description="Поиск и генерация ответов по одному активному TXT или PDF."
)
app.include_router(document_router)


def main() -> None:
    """Запускает ASGI-приложение в режиме разработки."""
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
