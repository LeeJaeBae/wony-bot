"""Chat service for handling conversations"""

from typing import Optional, AsyncGenerator
from uuid import UUID
import time
from app.core.ollama import OllamaClient
from app.core.session import SessionManager
from app.models.schemas import Message, MessageRole, ChatRequest, ChatResponse
from app.config import settings
from app.services.memory_manager import ConversationMemoryManager, ImportanceLevel
from app.rag import RAGChain
import logging

logger = logging.getLogger(__name__)

class ChatService:
    """Service for managing chat interactions"""
    
    def __init__(self, enable_memory: bool = True, rag_chain: Optional[RAGChain] = None):
        self.ollama = OllamaClient()
        self.session_manager = SessionManager()
        self.enable_memory = enable_memory
        
        # Initialize memory manager if enabled
        if enable_memory:
            self.memory_manager = ConversationMemoryManager(
                rag_chain=rag_chain,
                auto_save=True,
                importance_threshold=ImportanceLevel.LOW  # 더 많은 대화 저장
            )
            logger.info("Memory manager enabled for chat service")
        else:
            self.memory_manager = None
        
    async def chat(
        self,
        message: str,
        session_id: Optional[UUID] = None,
        system_prompt: Optional[str] = None,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Process a chat message and return response"""
        
        # Create or get session
        if session_id:
            session = await self.session_manager.get_session(session_id)
            if not session:
                # Create new session if ID doesn't exist
                session = await self.session_manager.create_session()
                session_id = session.id
        else:
            session = await self.session_manager.create_session()
            session_id = session.id
        
        # Add user message to history
        await self.session_manager.add_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=message
        )
        
        # Get recent messages for context
        messages = await self.session_manager.get_recent_messages(
            session_id=session_id,
            limit=settings.max_history_length
        )
        
        # Add system prompt if provided
        if system_prompt:
            messages.insert(0, Message(
                role=MessageRole.SYSTEM,
                content=system_prompt
            ))
        
        # Generate response
        response_text = ""
        start_time = time.time()
        
        async for chunk in self.ollama.chat(messages, stream=stream):
            response_text += chunk
            if stream:
                yield chunk
        
        # Calculate duration and approximate tokens
        duration_ms = int((time.time() - start_time) * 1000)
        tokens_used = len(response_text.split()) * 1.3  # Rough approximation
        
        # Save assistant response
        assistant_message = await self.session_manager.add_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=response_text,
            tokens_used=int(tokens_used),
            meta_data={"duration_ms": duration_ms}
        )
        
        # Auto-save to memory if enabled
        if self.memory_manager:
            try:
                # Save user message if important
                user_msg = Message(role=MessageRole.USER, content=message)
                await self.memory_manager.save_to_memory(
                    message=user_msg,
                    context=messages[:-1] if len(messages) > 1 else None,
                    session_id=session_id
                )
                
                # Save assistant response if important
                assistant_msg = Message(role=MessageRole.ASSISTANT, content=response_text)
                await self.memory_manager.save_to_memory(
                    message=assistant_msg,
                    context=messages,
                    session_id=session_id
                )
                
                logger.debug("Checked messages for memory storage")
            except Exception as e:
                logger.error(f"Failed to save to memory: {e}")
        
        if not stream:
            yield response_text
        
        # Return session ID for reference
        yield f"\n[session:{session_id}]"
    
    async def get_session_history(self, session_id: UUID) -> Optional[list]:
        """Get the chat history for a session"""
        session = await self.session_manager.get_session(session_id)
        if session:
            return session.messages
        return None
    
    async def list_sessions(self, limit: int = 10):
        """List recent chat sessions"""
        return await self.session_manager.list_sessions(limit=limit)
    
    async def clear_session(self, session_id: UUID) -> bool:
        """Clear messages from a session"""
        return await self.session_manager.clear_messages(session_id)
    
    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session completely"""
        return await self.session_manager.delete_session(session_id)