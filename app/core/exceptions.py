"""Custom exceptions for WonyBot"""

from typing import Optional, Any, Dict


class WonyBotError(Exception):
    """Base exception for all WonyBot errors"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses"""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ConfigurationError(WonyBotError):
    """Raised when configuration is invalid or missing"""
    pass


class OllamaError(WonyBotError):
    """Base exception for Ollama-related errors"""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when cannot connect to Ollama"""
    
    def __init__(self, host: str, message: Optional[str] = None):
        super().__init__(
            message or f"Cannot connect to Ollama at {host}",
            error_code="OLLAMA_CONNECTION_ERROR",
            details={"host": host}
        )


class OllamaModelError(OllamaError):
    """Raised when model is not available or fails"""
    
    def __init__(self, model: str, message: Optional[str] = None):
        super().__init__(
            message or f"Model {model} is not available",
            error_code="OLLAMA_MODEL_ERROR",
            details={"model": model}
        )


class DatabaseError(WonyBotError):
    """Base exception for database-related errors"""
    pass


class SessionNotFoundError(DatabaseError):
    """Raised when session is not found"""
    
    def __init__(self, session_id: str):
        super().__init__(
            f"Session {session_id} not found",
            error_code="SESSION_NOT_FOUND",
            details={"session_id": session_id}
        )


class RAGError(WonyBotError):
    """Base exception for RAG-related errors"""
    pass


class DocumentLoadError(RAGError):
    """Raised when document cannot be loaded"""
    
    def __init__(self, file_path: str, reason: str):
        super().__init__(
            f"Cannot load document {file_path}: {reason}",
            error_code="DOCUMENT_LOAD_ERROR",
            details={"file_path": file_path, "reason": reason}
        )


class EmbeddingError(RAGError):
    """Raised when embedding generation fails"""
    
    def __init__(self, text_length: int, reason: str):
        super().__init__(
            f"Embedding generation failed: {reason}",
            error_code="EMBEDDING_ERROR",
            details={"text_length": text_length, "reason": reason}
        )


class SearchError(RAGError):
    """Raised when search operation fails"""
    
    def __init__(self, query: str, reason: str):
        super().__init__(
            f"Search failed for query '{query}': {reason}",
            error_code="SEARCH_ERROR",
            details={"query": query, "reason": reason}
        )


class MemoryError(WonyBotError):
    """Raised when memory operations fail"""
    pass


class MemoryOverflowError(MemoryError):
    """Raised when memory buffer overflows"""
    
    def __init__(self, buffer_size: int, max_size: int):
        super().__init__(
            f"Memory buffer overflow: {buffer_size} > {max_size}",
            error_code="MEMORY_OVERFLOW",
            details={"buffer_size": buffer_size, "max_size": max_size}
        )


class SecurityError(WonyBotError):
    """Raised when security violation is detected"""
    
    def __init__(self, message: str, threat_type: Optional[str] = None):
        super().__init__(
            message,
            error_code="SECURITY_VIOLATION",
            details={"threat_type": threat_type or "unknown"}
        )


class ValidationError(WonyBotError):
    """Raised when input validation fails"""
    
    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Validation failed for {field}: {reason}",
            error_code="VALIDATION_ERROR",
            details={"field": field, "value": str(value), "reason": reason}
        )