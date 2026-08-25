from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile


READ_CHUNK_SIZE = 64 * 1024
ALLOWED_CONTENT_TYPES = {
    ".pdf": {"application/pdf", "application/x-pdf"},
    ".txt": {"text/plain"},
}


class UnsupportedUploadError(ValueError):
    """Сообщает, что формат или MIME-тип upload не поддерживается."""


class UploadTooLargeError(ValueError):
    """Сообщает, что upload превышает допустимый размер."""


def validate_upload(file: UploadFile) -> tuple[str, str]:
    """Проверяет имя и MIME-тип загруженного документа.

    Args:
        file: Файл из multipart-запроса.

    Returns:
        Исходное имя файла и нормализованное расширение.
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    content_type = (file.content_type or "").split(";", maxsplit=1)[0].lower()
    if suffix not in ALLOWED_CONTENT_TYPES:
        raise UnsupportedUploadError("Поддерживаются только файлы TXT и PDF")
    if content_type not in ALLOWED_CONTENT_TYPES[suffix]:
        raise UnsupportedUploadError(
            "MIME-тип файла не соответствует его расширению"
        )
    return filename, suffix


async def save_upload(file: UploadFile, suffix: str, max_bytes: int) -> Path:
    """Сохраняет upload порциями во временный файл с ограничением размера.

    Args:
        file: Загруженный файл FastAPI.
        suffix: Проверенное расширение документа.
        max_bytes: Допустимый размер файла в байтах.

    Returns:
        Путь к созданному временному файлу.
    """
    total_size = 0
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_path = Path(temporary_file.name)
            while chunk := await file.read(READ_CHUNK_SIZE):
                # Лимит проверяется до записи очередного блока на диск.
                total_size += len(chunk)
                if total_size > max_bytes:
                    raise UploadTooLargeError("Размер файла превышает 5 МБ")
                temporary_file.write(chunk)
        return temporary_path
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
