"""Configuration management for WonyBot"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:20b"
    ollama_timeout: int = 300  # 5 minutes timeout for long responses
    
    # Database Configuration
    database_url: str = "sqlite+aiosqlite:///./wony_bot.db"
    
    # Application Settings
    app_name: str = "WonyBot"
    debug: bool = False
    log_level: str = "INFO"
    default_persona: str = "wony"  # Default persona to use
    
    # Conversation Settings
    max_history_length: int = 10  # Maximum messages to keep in context
    streaming_enabled: bool = True  # Enable streaming responses
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    prompts_dir: Path = base_dir / "prompts"
    data_dir: Path = base_dir / "data"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Create necessary directories
settings.prompts_dir.mkdir(exist_ok=True)
settings.data_dir.mkdir(exist_ok=True)