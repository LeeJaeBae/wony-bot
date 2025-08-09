"""Session management for chat conversations"""

from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from app.core.database import async_session, SessionModel, MessageModel
from app.models.schemas import Session, Message, MessageRole

class SessionManager:
    """Manage chat sessions and message history"""
    
    async def create_session(self, name: Optional[str] = None) -> Session:
        """Create a new chat session"""
        async with async_session() as db:
            session_model = SessionModel(
                id=str(uuid4()),
                name=name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(session_model)
            await db.commit()
            
            return Session(
                id=UUID(session_model.id),
                name=session_model.name,
                created_at=session_model.created_at,
                updated_at=session_model.updated_at,
                messages=[]
            )
    
    async def get_session(self, session_id: UUID) -> Optional[Session]:
        """Get a session by ID with all messages"""
        async with async_session() as db:
            result = await db.execute(
                select(SessionModel)
                .where(SessionModel.id == str(session_id))
                .options(selectinload(SessionModel.messages))
            )
            session_model = result.scalar_one_or_none()
            
            if not session_model:
                return None
            
            messages = [
                Message(
                    id=UUID(msg.id),
                    session_id=UUID(msg.session_id),
                    role=MessageRole(msg.role),
                    content=msg.content,
                    tokens_used=msg.tokens_used,
                    created_at=msg.created_at,
                    metadata=msg.meta_data
                )
                for msg in sorted(session_model.messages, key=lambda x: x.created_at)
            ]
            
            return Session(
                id=UUID(session_model.id),
                name=session_model.name,
                created_at=session_model.created_at,
                updated_at=session_model.updated_at,
                messages=messages
            )
    
    async def list_sessions(self, limit: int = 10) -> List[Session]:
        """List recent sessions"""
        async with async_session() as db:
            result = await db.execute(
                select(SessionModel)
                .order_by(desc(SessionModel.updated_at))
                .limit(limit)
            )
            sessions = result.scalars().all()
            
            return [
                Session(
                    id=UUID(s.id),
                    name=s.name,
                    created_at=s.created_at,
                    updated_at=s.updated_at,
                    messages=[]
                )
                for s in sessions
            ]
    
    async def add_message(
        self,
        session_id: UUID,
        role: MessageRole,
        content: str,
        tokens_used: Optional[int] = None,
        meta_data: Optional[dict] = None
    ) -> Message:
        """Add a message to a session"""
        async with async_session() as db:
            message_model = MessageModel(
                id=str(uuid4()),
                session_id=str(session_id),
                role=role.value,
                content=content,
                tokens_used=tokens_used,
                created_at=datetime.utcnow(),
                meta_data=meta_data
            )
            db.add(message_model)
            
            # Update session's updated_at
            result = await db.execute(
                select(SessionModel).where(SessionModel.id == str(session_id))
            )
            session = result.scalar_one_or_none()
            if session:
                session.updated_at = datetime.utcnow()
            
            await db.commit()
            
            return Message(
                id=UUID(message_model.id),
                session_id=UUID(message_model.session_id),
                role=MessageRole(message_model.role),
                content=message_model.content,
                tokens_used=message_model.tokens_used,
                created_at=message_model.created_at,
                metadata=message_model.meta_data
            )
    
    async def get_recent_messages(
        self,
        session_id: UUID,
        limit: int = 10
    ) -> List[Message]:
        """Get recent messages from a session"""
        async with async_session() as db:
            result = await db.execute(
                select(MessageModel)
                .where(MessageModel.session_id == str(session_id))
                .order_by(desc(MessageModel.created_at))
                .limit(limit)
            )
            messages = result.scalars().all()
            
            # Return in chronological order
            return [
                Message(
                    id=UUID(msg.id),
                    session_id=UUID(msg.session_id),
                    role=MessageRole(msg.role),
                    content=msg.content,
                    tokens_used=msg.tokens_used,
                    created_at=msg.created_at,
                    metadata=msg.meta_data
                )
                for msg in reversed(messages)
            ]
    
    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session and all its messages"""
        async with async_session() as db:
            result = await db.execute(
                select(SessionModel).where(SessionModel.id == str(session_id))
            )
            session = result.scalar_one_or_none()
            
            if session:
                await db.delete(session)
                await db.commit()
                return True
            
            return False
    
    async def clear_messages(self, session_id: UUID) -> bool:
        """Clear all messages from a session"""
        async with async_session() as db:
            result = await db.execute(
                select(MessageModel).where(MessageModel.session_id == str(session_id))
            )
            messages = result.scalars().all()
            
            for msg in messages:
                await db.delete(msg)
            
            await db.commit()
            return len(messages) > 0