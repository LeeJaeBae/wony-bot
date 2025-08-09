"""Dependency Injection Container for WonyBot"""

from typing import Optional, Dict, Any
import asyncio
from functools import lru_cache
import logging

from app.config import settings
from app.core.database import init_db
from app.core.ollama import OllamaClient
from app.core.session import SessionManager
from app.services.chat import ChatService
from app.services.prompt import PromptService
from app.services.memory_manager import ConversationMemoryManager
from app.rag import RAGChain

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Singleton container for all services with lazy initialization"""
    
    _instance: Optional['ServiceContainer'] = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        """Private constructor - use get_instance() instead"""
        if ServiceContainer._instance is not None:
            raise RuntimeError("Use ServiceContainer.get_instance() instead of direct instantiation")
        
        # Service instances (lazy loaded)
        self._services: Dict[str, Any] = {}
        self._initialized = False
        
    @classmethod
    async def get_instance(cls) -> 'ServiceContainer':
        """Get or create singleton instance"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.initialize()
        return cls._instance
    
    async def initialize(self):
        """Initialize core services"""
        if self._initialized:
            return
        
        logger.info("Initializing ServiceContainer...")
        
        # Initialize database
        await init_db()
        
        # Mark as initialized
        self._initialized = True
        logger.info("ServiceContainer initialized successfully")
    
    @lru_cache(maxsize=1)
    def get_ollama_client(self) -> OllamaClient:
        """Get or create Ollama client"""
        if 'ollama' not in self._services:
            self._services['ollama'] = OllamaClient()
            logger.debug("Created OllamaClient instance")
        return self._services['ollama']
    
    @lru_cache(maxsize=1)
    def get_session_manager(self) -> SessionManager:
        """Get or create session manager"""
        if 'session_manager' not in self._services:
            self._services['session_manager'] = SessionManager()
            logger.debug("Created SessionManager instance")
        return self._services['session_manager']
    
    @lru_cache(maxsize=1)
    def get_prompt_service(self) -> PromptService:
        """Get or create prompt service"""
        if 'prompt_service' not in self._services:
            self._services['prompt_service'] = PromptService()
            logger.debug("Created PromptService instance")
        return self._services['prompt_service']
    
    def get_rag_chain(self, collection_name: str = "documents") -> RAGChain:
        """Get or create RAG chain for specific collection"""
        key = f'rag_chain_{collection_name}'
        if key not in self._services:
            self._services[key] = RAGChain(collection_name=collection_name)
            logger.debug(f"Created RAGChain instance for collection: {collection_name}")
        return self._services[key]
    
    def get_chat_service(self, enable_memory: bool = True, collection_name: str = "documents") -> ChatService:
        """Get or create chat service with proper dependencies"""
        key = f'chat_service_{enable_memory}_{collection_name}'
        if key not in self._services:
            rag_chain = self.get_rag_chain(collection_name) if enable_memory else None
            self._services[key] = ChatService(
                enable_memory=enable_memory,
                rag_chain=rag_chain
            )
            logger.debug(f"Created ChatService instance (memory={enable_memory})")
        return self._services[key]
    
    def get_memory_manager(self, collection_name: str = "chat_history") -> ConversationMemoryManager:
        """Get or create memory manager"""
        key = f'memory_manager_{collection_name}'
        if key not in self._services:
            rag_chain = self.get_rag_chain(collection_name)
            self._services[key] = ConversationMemoryManager(
                rag_chain=rag_chain,
                auto_save=True
            )
            logger.debug(f"Created ConversationMemoryManager for collection: {collection_name}")
        return self._services[key]
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up ServiceContainer...")
        
        # Cleanup services that need it
        for service_name, service in self._services.items():
            if hasattr(service, 'cleanup'):
                try:
                    await service.cleanup()
                    logger.debug(f"Cleaned up {service_name}")
                except Exception as e:
                    logger.error(f"Failed to cleanup {service_name}: {e}")
        
        # Clear services
        self._services.clear()
        self._initialized = False
        
        # Clear singleton instance
        ServiceContainer._instance = None
        
        logger.info("ServiceContainer cleanup complete")
    
    @classmethod
    async def reset(cls):
        """Reset the container (useful for testing)"""
        if cls._instance:
            await cls._instance.cleanup()
        cls._instance = None


# Convenience function for getting container
async def get_container() -> ServiceContainer:
    """Get the service container instance"""
    return await ServiceContainer.get_instance()