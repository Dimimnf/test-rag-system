import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIRECTORY = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIRECTORY / "app.log"
MAX_LOG_SIZE = 2 * 1024 * 1024
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_HANDLER_MARKER = "_local_app_log_handler"


def get_logger(name: str = "app") -> logging.Logger:
    """Создаёт именованный logger для записи в файл и консоль.

    Args:
        name: Имя logger, отображаемое в сообщениях.

    Returns:
        Настроенный экземпляр logger.
    """
    configured_logger = logging.getLogger(name)
    configured_logger.setLevel(logging.INFO)
    configured_logger.propagate = False

    if any(
        getattr(handler, _HANDLER_MARKER, False)
        for handler in configured_logger.handlers
    ):
        return configured_logger

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=1,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    setattr(file_handler, _HANDLER_MARKER, True)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    setattr(console_handler, _HANDLER_MARKER, True)

    configured_logger.addHandler(file_handler)
    configured_logger.addHandler(console_handler)
    return configured_logger


logger = get_logger()
