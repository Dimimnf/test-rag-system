from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from config import get_settings
from logger import get_logger
from schemas.document import (
    QuestionRequest,
    QuestionResponse,
    UploadDocumentResponse,
)
from services.ai.llm import LlmProviderError, llm_service
from services.ai.rag import (
    DocumentReadError,
    EmptyDocumentError,
    RagNotReadyError,
    UnsupportedDocumentError,
    rag_service,
)
from services.document import (
    UnsupportedUploadError,
    UploadTooLargeError,
    save_upload,
    validate_upload,
)


logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Документы"])


@router.post(
    "",
    response_model=UploadDocumentResponse,
    summary="Загрузить документ",
    description="Заменяет активный документ и создаёт новый Chroma-индекс.",
    response_description="Результат индексации документа.",
    responses={
        413: {"description": "Файл превышает ограничение 5 МБ."},
        415: {"description": "Формат или MIME-тип не поддерживается."},
        422: {"description": "Документ повреждён или не содержит текста."},
    },
)
async def upload_document(
    file: Annotated[
        UploadFile,
        File(description="Текстовый файл TXT или PDF с текстовым слоем до 5 МБ."),
    ],
) -> UploadDocumentResponse:
    """Принимает и полностью переиндексирует единственный активный документ."""
    temporary_path: Path | None = None
    try:
        filename, suffix = validate_upload(file)
        settings = get_settings()
        temporary_path = await save_upload(file, suffix, settings.upload_max_bytes)
        # Индексирование синхронное и не должно блокировать event loop FastAPI.
        chunks = await run_in_threadpool(rag_service.change, temporary_path)
        return UploadDocumentResponse(filename=filename, chunks=chunks)
    except (UnsupportedUploadError, UnsupportedDocumentError) as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(error),
        ) from error
    except UploadTooLargeError as error:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(error),
        ) from error
    except (EmptyDocumentError, DocumentReadError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception("Не удалось обработать загруженный документ")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось проиндексировать документ",
        ) from error
    finally:
        # Временный upload удаляется при любом результате индексирования.
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        await file.close()


@router.post(
    "/questions",
    response_model=QuestionResponse,
    summary="Задать вопрос по документу",
    description="Ищет релевантные чанки и генерирует ответ только по их тексту.",
    response_description="Ответ по содержимому активного документа.",
    responses={
        409: {"description": "Активный документ ещё не проиндексирован."},
        502: {"description": "OpenAI-compatible провайдер недоступен."},
    },
)
async def ask_document(request: QuestionRequest) -> QuestionResponse:
    """Возвращает ответ модели на вопрос по активному документу."""
    try:
        # Retrieval и вызов модели выполняются вне основного event loop.
        answer = await run_in_threadpool(llm_service.answer, request.question)
        return QuestionResponse(answer=answer)
    except RagNotReadyError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except LlmProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Не удалось получить ответ модели",
        ) from error
    except Exception as error:
        logger.exception("Не удалось ответить на вопрос по документу")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось обработать вопрос",
        ) from error
