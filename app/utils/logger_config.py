"""Logger configuration for WonyBot"""

import logging
import sys
from pathlib import Path

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "wonybot.log", encoding='utf-8')
        ]
    )
    
    # Set specific loggers
    logging.getLogger("app.services.memory_manager").setLevel(logging.DEBUG)
    logging.getLogger("app.services.enhanced_memory").setLevel(logging.DEBUG)
    logging.getLogger("app.services.chat").setLevel(logging.DEBUG)
    
    # Reduce noise from other libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)