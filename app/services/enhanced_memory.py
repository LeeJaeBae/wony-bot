"""Enhanced Memory System for WonyBot - Always Remember!"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID
import logging
from app.models.schemas import Message, MessageRole
from app.services.memory_manager import ConversationMemoryManager, ImportanceLevel

logger = logging.getLogger(__name__)

class EnhancedMemoryManager(ConversationMemoryManager):
    """강화된 메모리 매니저 - 모든 중요 정보를 확실히 저장"""
    
    def __init__(self, *args, **kwargs):
        # 기본 임계값을 LOW로 설정해서 더 많이 저장
        kwargs['importance_threshold'] = ImportanceLevel.LOW
        super().__init__(*args, **kwargs)
        self.force_save_keywords = [
            "기억", "내일", "일정", "약속", "중요", "꼭", "반드시",
            "remember", "tomorrow", "schedule", "important"
        ]
        logger.info("Enhanced Memory Manager initialized with aggressive saving")
    
    async def save_to_memory(
        self,
        message: Message,
        context: List[Message] = None,
        session_id: Optional[UUID] = None,
        force: bool = False
    ) -> bool:
        """무조건 저장하는 강화된 메모리 저장"""
        
        # 키워드가 있으면 무조건 저장
        content_lower = message.content.lower()
        for keyword in self.force_save_keywords:
            if keyword in content_lower:
                force = True
                logger.info(f"Force saving due to keyword: {keyword}")
                break
        
        # 질문이면 무조건 저장
        if "?" in message.content or "？" in message.content:
            force = True
            logger.info("Force saving question")
        
        # 답변도 저장 (질문에 대한 답변일 경우)
        if context and len(context) > 0:
            last_msg = context[-1]
            if "?" in last_msg.content:
                force = True
                logger.info("Force saving answer to question")
        
        # 부모 클래스의 save_to_memory 호출
        result = await super().save_to_memory(
            message=message,
            context=context,
            session_id=session_id,
            force=force
        )
        
        if result:
            logger.info(f"✅ Memory saved successfully for session {session_id}")
        else:
            logger.warning(f"⚠️ Memory not saved for session {session_id}")
        
        return result
    
    async def get_relevant_memories(
        self,
        query: str,
        session_id: Optional[UUID] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """관련 메모리 검색 - 세션 기반으로 우선 검색"""
        
        memories = []
        
        # 현재 세션의 메모리 우선 검색
        if session_id and self.rag_chain:
            session_filter = {"session_id": str(session_id)}
            session_memories = await self.query_memories(
                query=query,
                top_k=top_k,
                tag_filter=None,
                importance_filter=None
            )
            memories.extend(session_memories)
            logger.info(f"Found {len(session_memories)} memories for session {session_id}")
        
        # 전체 메모리에서도 검색
        if self.rag_chain:
            all_memories = await self.query_memories(
                query=query,
                top_k=top_k,
                tag_filter=None,
                importance_filter=None  
            )
            
            # 중복 제거
            memory_ids = {m.get('id') for m in memories}
            for mem in all_memories:
                if mem.get('id') not in memory_ids:
                    memories.append(mem)
        
        logger.info(f"Total {len(memories)} relevant memories found")
        return memories