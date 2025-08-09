#!/usr/bin/env python3
"""Example usage of WonyBot programmatically"""

import asyncio
from app.services.chat import ChatService
from app.services.prompt import PromptService
from app.core.database import init_db

async def main():
    # Initialize database
    await init_db()
    
    # Create services
    chat_service = ChatService()
    prompt_service = PromptService()
    
    # Get developer prompt
    prompt = await prompt_service.get_prompt("developer")
    
    # Send a message
    print("🤖 WonyBot: 대화를 시작합니다...")
    print("-" * 50)
    
    response = ""
    async for chunk in chat_service.chat(
        message="Python으로 피보나치 수열을 생성하는 함수를 만들어줘",
        system_prompt=prompt.content if prompt else None,
        stream=True
    ):
        if not chunk.startswith("\n[session:"):
            print(chunk, end="", flush=True)
            response += chunk
        else:
            session_id = chunk.strip()[9:-1]
            print(f"\n\n💾 Session saved: {session_id}")
    
    print("\n" + "-" * 50)
    print("✅ 대화 완료!")

if __name__ == "__main__":
    asyncio.run(main())