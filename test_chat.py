#!/usr/bin/env python
"""Test hierarchical chat system"""

import asyncio
from app.services.chat import ChatService
from app.rag import RAGChain

async def test_hierarchical_chat():
    """Test hierarchical chat service"""
    
    # Initialize services
    rag_chain = RAGChain()
    chat_service = ChatService(
        enable_memory=True,
        rag_chain=rag_chain,
        enable_agents=True,
        use_hierarchical=True  # Enable hierarchical system
    )
    
    print("✅ Chat service initialized with hierarchical agents")
    
    # Test with a simple message
    print("\n🔹 Testing simple message:")
    simple_result = ""
    async for chunk in chat_service.chat("안녕하세요!", stream=False):
        if not chunk.startswith("\n[session:"):
            simple_result = chunk
    print(f"Response: {simple_result[:100]}...")
    
    # Test with complex task
    print("\n🔹 Testing complex task with hierarchical system:")
    complex_task = "AI 에이전트 시스템의 성능을 분석하고 개선 방안을 제시해줘"
    
    if chat_service.hierarchical_manager:
        result = await chat_service.chat_with_hierarchical_agents(
            message=complex_task,
            require_consensus=True
        )
        
        print(f"Status: {result.get('status')}")
        if 'consensus' in result:
            print(f"Consensus reached: {result['consensus'].get('reached')}")
            if result['consensus'].get('reached'):
                print(f"Approval rate: {result['consensus'].get('approval_rate', 0):.1%}")
    else:
        print("❌ Hierarchical manager not available")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_hierarchical_chat())