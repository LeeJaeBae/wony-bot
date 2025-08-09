#!/usr/bin/env python3
"""Test memory saving functionality"""

import asyncio
import logging
from app.services.chat import ChatService
from app.rag import RAGChain
from app.core.database import init_db

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_memory():
    # Initialize database
    await init_db()
    
    # Initialize RAG chain
    rag_chain = RAGChain(collection_name="chat_history")
    
    # Initialize chat service with memory
    chat_service = ChatService(enable_memory=True, rag_chain=rag_chain)
    
    print("Testing memory system...")
    print(f"Memory manager initialized: {chat_service.memory_manager is not None}")
    print(f"Auto-save enabled: {chat_service.memory_manager.auto_save if chat_service.memory_manager else False}")
    print(f"Importance threshold: {chat_service.memory_manager.importance_threshold.name if chat_service.memory_manager else 'N/A'}")
    
    # Test message
    test_messages = [
        "내일 오전 10시에 중요한 회의가 있어",
        "파이썬 코드 에러가 나는데 도와줘",
        "안녕하세요"
    ]
    
    for msg in test_messages:
        print(f"\n처리중: {msg}")
        response = ""
        async for chunk in chat_service.chat(message=msg, stream=False):
            if not chunk.startswith("\n[session:"):
                response = chunk
        print(f"응답: {response[:100]}...")
    
    # Check memory stats
    if chat_service.memory_manager:
        stats = chat_service.memory_manager.get_memory_stats()
        print("\n메모리 통계:")
        print(f"  버퍼 크기: {stats['buffer_size']}")
        print(f"  중요도 분포: {stats.get('importance_distribution', {})}")
        print(f"  태그 분포: {stats.get('tag_distribution', {})}")

if __name__ == "__main__":
    asyncio.run(test_memory())