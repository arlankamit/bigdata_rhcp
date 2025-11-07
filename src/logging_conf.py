# -*- coding: utf-8 -*-
import logging, structlog, sys, os

def setup_logging():
    """
    Прод-вариант:
    - JSON Renderer + iso timestamp
    - совместим с uvicorn/fastapi
    - можно переключить формат через LOG_FORMAT=plain (для локалки)
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = os.getenv("LOG_FORMAT", "json").lower()  # json | plain

    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level, logging.INFO),
        format="%(message)s" if fmt == "json" else "%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # приглушаем шум от uvicorn.access при JSON
    for name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        logging.getLogger(name).setLevel(getattr(logging, level, logging.INFO))
        if fmt == "json" and name == "uvicorn.access":
            logging.getLogger(name).propagate = False

    timestamper = structlog.processors.TimeStamper(fmt="iso")
    processors = [
        structlog.contextvars.merge_contextvars,
        timestamper,
        structlog.processors.add_log_level,
    ]
    if fmt == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.processors.KeyValueRenderer(key_order=["event","level","timestamp"]))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level, logging.INFO)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    )
