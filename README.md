# RAG Document Service

FastAPI-сервис загружает один текстовый документ, сохраняет его чанки в
persistent Chroma и отвечает на вопросы через OpenAI-compatible API, используя
только найденный в документе контекст.

## Возможности

- загрузка TXT и PDF с текстовым слоем размером до 5 МБ;
- полная замена предыдущего документа при новой загрузке;
- локальные embeddings `sentence-transformers/all-MiniLM-L6-v2`;
- сохранение векторного индекса Chroma между перезапусками;
- настройка модели, ключа и API endpoint через `.env`;
- документированный OpenAPI-контракт по адресу `/docs`.

## Алгоритм

### Индексирование

1. `POST /documents` проверяет расширение, MIME-тип и размер файла.
2. `TextLoader` или `PyPDFLoader` извлекает текст.
3. `RecursiveCharacterTextSplitter` создаёт чанки по 600 символов с overlap 80.
4. `HuggingFaceEmbeddings` преобразует чанки в локальные embeddings.
5. Chroma полностью заменяет коллекцию единственного активного документа.
6. Marker готовности публикуется только после успешной записи индекса.

### Ответ на вопрос

1. `POST /documents/questions` принимает непустой вопрос.
2. Chroma retriever находит `RAG_TOP_K` наиболее близких чанков.
3. Чанки объединяются в контекст и вместе с вопросом передаются в prompt.
4. `ChatOpenAI` обращается к настроенному OpenAI-compatible API.
5. System prompt запрещает использовать сведения за пределами контекста.

## Архитектура

```text
routers/document.py
    -> services/document/upload.py
    -> services/ai/rag/service.py
        -> document_processor.py
        -> vector_store.py
    -> services/ai/llm/service.py
        -> prompt.py
        -> model.py

schemas/document/
    request.py
    response.py
```

`Rag` и `Llm` являются небольшими публичными фасадами. Их зависимости можно
заменять в тестах без обращения к Chroma, Hugging Face или внешнему API.

## Установка

Требуется Python 3.12+ и [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Создайте `.env` на основе `.env.example`:

```dotenv
OPENAI_API_KEY=sk-example
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
RAG_TOP_K=3
UPLOAD_MAX_BYTES=5242880
```

Для официального OpenAI `OPENAI_BASE_URL` можно удалить. При первом
индексировании приложение скачает локальную модель embeddings.

## Запуск

```bash
uv run uvicorn main:app --reload
```

Swagger UI будет доступен по адресу `http://127.0.0.1:8000/docs`.

## Примеры запросов

Загрузка TXT:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@regulations.txt;type=text/plain"
```

Загрузка PDF:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@regulations.pdf;type=application/pdf"
```

Вопрос по активному документу:

```bash
curl -X POST http://127.0.0.1:8000/documents/questions \
  -H "Content-Type: application/json" \
  -d '{"question":"Какой срок согласования указан в документе?"}'
```

## Ошибки и ограничения

- `409` возвращается, если вопрос задан до индексирования документа.
- `413` возвращается для файла больше 5 МБ.
- `415` возвращается для неподдерживаемого расширения или MIME-типа.
- `422` возвращается для пустого, повреждённого или сканированного PDF без текста.
- `502` возвращается при ошибке OpenAI-compatible провайдера.
- OCR, несколько документов, история диалога и streaming не реализованы.

## Тесты

```bash
uv run pytest
```

Тесты используют mocks и не требуют API key, загрузки модели или доступа к сети.
