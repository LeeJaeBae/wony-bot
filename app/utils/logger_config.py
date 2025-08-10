"""Logger configuration for WonyBot"""

import logging
import sys
from pathlib import Path

def setup_logging(level=logging.INFO, to_console: bool = False, to_file: bool = True):
    """Setup logging configuration.
    
    By default, logs go to file only to avoid polluting CLI chat output.
    Set to_console=True to also log to stdout (e.g. during debugging).
    """

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    # Reconfigure handlers safely to avoid duplicates and control console/file outputs
    existing_handlers = list(root_logger.handlers)
    # Remove StreamHandler if console logging is disabled
    for h in existing_handlers:
        if isinstance(h, logging.StreamHandler) and not to_console:
            root_logger.removeHandler(h)
        if isinstance(h, logging.FileHandler) and not to_file:
            root_logger.removeHandler(h)

    # Ensure desired handlers exist
    want_console = to_console and not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
    want_file = to_file and not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    if want_console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        sh.setLevel(level)
        root_logger.addHandler(sh)
    if want_file:
        fh = logging.FileHandler(log_dir / "wonybot.log", encoding='utf-8')
        fh.setFormatter(formatter)
        fh.setLevel(level)
        root_logger.addHandler(fh)

    # Update existing handlers' levels
    for h in root_logger.handlers:
        h.setLevel(level)

    # Set specific loggers (keep debug detail but they won't go to console unless enabled)
    logging.getLogger("app.services.memory_manager").setLevel(logging.DEBUG)
    logging.getLogger("app.services.enhanced_memory").setLevel(logging.DEBUG)
    logging.getLogger("app.services.chat").setLevel(logging.DEBUG)

    # Reduce noise from other libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    return logging.getLogger(__name__)