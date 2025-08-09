#!/usr/bin/env python3
"""Test enhanced memory system"""

import asyncio
import logging
from uuid import uuid4
from app.services.chat import ChatService
from app.rag import RAGChain
from app.core.database import init_db
from app.utils.logger_config import setup_logging

# Setup logging
setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_memory_saving():
    """Test memory saving and retrieval"""
    
    print("\n" + "="*50)
    print("🧪 Enhanced Memory System Test")
    print("="*50 + "\n")
    
    # Initialize
    await init_db()
    rag_chain = RAGChain(collection_name="chat_history")
    chat_service = ChatService(enable_memory=True, rag_chain=rag_chain)
    
    # Test messages
    test_conversations = [
        ("내일 강릉 여행 가기로 했어", "강릉 여행 일정 저장"),
        ("여자친구 이름은 만만이야", "개인 정보 저장"),
        ("오전 10시에 회의 있어", "일정 정보 저장"),
        ("내일 일정 뭐야?", "저장된 정보 확인")
    ]
    
    session_id = uuid4()
    print(f"📌 Test Session ID: {session_id}\n")
    
    for user_msg, test_name in test_conversations:
        print(f"🧪 Test: {test_name}")
        print(f"👤 User: {user_msg}")
        
        # Send message
        response = ""
        async for chunk in chat_service.chat(
            message=user_msg,
            session_id=session_id,
            stream=False
        ):
            if not chunk.startswith("\n[session:"):
                response = chunk
        
        print(f"🤖 Wony: {response[:100]}...")
        
        # Check memory buffer
        if chat_service.memory_manager:
            stats = chat_service.memory_manager.get_memory_stats()
            print(f"💾 Memory Buffer: {stats['buffer_size']} entries")
        
        print("-" * 30 + "\n")
        await asyncio.sleep(1)  # Small delay
    
    # Test memory retrieval
    print("\n" + "="*50)
    print("🔍 Memory Retrieval Test")
    print("="*50 + "\n")
    
    if chat_service.memory_manager:
        # Search for saved memories
        memories = await chat_service.memory_manager.query_memories(
            query="강릉 여행 만만이 회의",
            top_k=10
        )
        
        print(f"📚 Found {len(memories)} memories:\n")
        for i, mem in enumerate(memories, 1):
            content = mem.get('content', '')[:100]
            metadata = mem.get('metadata', {})
            print(f"{i}. {content}...")
            print(f"   - Importance: {metadata.get('importance', 'N/A')}")
            print(f"   - Tags: {metadata.get('tags', 'N/A')}")
            print()
    
    # Final stats
    if chat_service.memory_manager:
        final_stats = chat_service.memory_manager.get_memory_stats()
        print("\n📊 Final Memory Stats:")
        print(f"   - Buffer Size: {final_stats['buffer_size']}")
        print(f"   - Auto-Save: {final_stats['auto_save']}")
        print(f"   - Threshold: {final_stats['importance_threshold']}")
        
        if final_stats.get('importance_distribution'):
            print("\n   Importance Distribution:")
            for level, count in final_stats['importance_distribution'].items():
                print(f"      - {level}: {count}")

if __name__ == "__main__":
    asyncio.run(test_memory_saving())