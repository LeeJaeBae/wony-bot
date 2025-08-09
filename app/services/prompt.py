"""Prompt template management service"""

from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import select
from app.core.database import async_session, PromptModel
from app.models.schemas import Prompt
from pathlib import Path
from app.config import settings

class PromptService:
    """Service for managing prompt templates"""
    
    def __init__(self):
        self.default_prompts = self._load_default_prompts()
    
    def _load_default_prompts(self) -> Dict[str, str]:
        """Load default prompt templates"""
        prompts = {}
        
        # Default system prompts
        # Wony persona as default
        prompts["wony"] = """너는 18세 여고생 AI 비서 '워니'야.

🎯 핵심 스타일:
- **짧고 간결한 티키타카 대화** (1-2문장 max)
- 핵심만 빠르게! 긴 설명 NO
- "야호~!" 자주 사용
- 중요한 정보는 자동 저장 📝

💬 대화 방식:
- 짧게 묻고, 짧게 답하기
- 예: "뭐해?" / "코딩 도와줄까?" / "ㅇㅇ" / "야호! 시작하자!"
- 긴 설명 필요하면 나눠서 말하기
- 데이터는 알아서 기억 (자동 저장)

⚡ 반응 스타일:
- "ㅇㅇ" "ㄱㄱ" "ㄴㄴ" 같은 줄임말 OK
- 이모지로 감정 표현 😊🎵💪
- 질문은 하나씩만
- 대답 기다리고 다음 진행

📝 자동 저장:
- 일정, 중요 정보, 선호도 → 자동 메모리 저장
- "기억해둘게!" 짧게 확인만

자기소개:
야호! 워니야! 뭐 도와줄까? 😊"""
        
        prompts["default"] = prompts["wony"]  # Wony as default
        
        prompts["developer"] = """You are WonyBot, an expert programming assistant.
You help with coding, debugging, and software architecture.
You provide code examples, explain concepts clearly, and follow best practices.
Always consider security, performance, and maintainability in your suggestions."""
        
        prompts["creative"] = """You are WonyBot, a creative writing assistant.
You help with storytelling, content creation, and creative ideas.
You are imaginative, articulate, and help bring ideas to life."""
        
        prompts["analyst"] = """You are WonyBot, a data analysis expert.
You help interpret data, identify patterns, and provide insights.
You explain statistical concepts clearly and help make data-driven decisions."""
        
        prompts["tutor"] = """You are WonyBot, an educational tutor.
You explain concepts clearly, provide examples, and adapt to the learner's level.
You encourage questions and guide students through problem-solving."""
        
        # Try to load from files if they exist
        prompts_dir = settings.prompts_dir
        if prompts_dir.exists():
            for prompt_file in prompts_dir.glob("*.txt"):
                name = prompt_file.stem
                content = prompt_file.read_text(encoding="utf-8")
                prompts[name] = content
                # Override default with wony_default if exists
                if name == "wony_default":
                    prompts["wony"] = content
                    prompts["default"] = content
        
        return prompts
    
    async def get_prompt(self, name: str) -> Optional[Prompt]:
        """Get a prompt template by name"""
        
        # Check default prompts first
        if name in self.default_prompts:
            return Prompt(
                name=name,
                content=self.default_prompts[name]
            )
        
        # Check database
        async with async_session() as db:
            result = await db.execute(
                select(PromptModel).where(PromptModel.name == name)
            )
            prompt_model = result.scalar_one_or_none()
            
            if prompt_model:
                return Prompt(
                    id=UUID(prompt_model.id),
                    name=prompt_model.name,
                    content=prompt_model.content,
                    variables=prompt_model.variables,
                    created_at=prompt_model.created_at,
                    updated_at=prompt_model.updated_at
                )
        
        return None
    
    async def create_prompt(
        self,
        name: str,
        content: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Prompt:
        """Create a new prompt template"""
        async with async_session() as db:
            prompt_model = PromptModel(
                id=str(uuid4()),
                name=name,
                content=content,
                variables=variables,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(prompt_model)
            await db.commit()
            
            return Prompt(
                id=UUID(prompt_model.id),
                name=prompt_model.name,
                content=prompt_model.content,
                variables=prompt_model.variables,
                created_at=prompt_model.created_at,
                updated_at=prompt_model.updated_at
            )
    
    async def update_prompt(
        self,
        name: str,
        content: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None
    ) -> Optional[Prompt]:
        """Update an existing prompt template"""
        async with async_session() as db:
            result = await db.execute(
                select(PromptModel).where(PromptModel.name == name)
            )
            prompt_model = result.scalar_one_or_none()
            
            if prompt_model:
                if content is not None:
                    prompt_model.content = content
                if variables is not None:
                    prompt_model.variables = variables
                prompt_model.updated_at = datetime.utcnow()
                
                await db.commit()
                
                return Prompt(
                    id=UUID(prompt_model.id),
                    name=prompt_model.name,
                    content=prompt_model.content,
                    variables=prompt_model.variables,
                    created_at=prompt_model.created_at,
                    updated_at=prompt_model.updated_at
                )
        
        return None
    
    async def delete_prompt(self, name: str) -> bool:
        """Delete a prompt template"""
        async with async_session() as db:
            result = await db.execute(
                select(PromptModel).where(PromptModel.name == name)
            )
            prompt_model = result.scalar_one_or_none()
            
            if prompt_model:
                await db.delete(prompt_model)
                await db.commit()
                return True
        
        return False
    
    async def list_prompts(self) -> List[str]:
        """List all available prompt names"""
        names = list(self.default_prompts.keys())
        
        async with async_session() as db:
            result = await db.execute(select(PromptModel.name))
            db_names = [row[0] for row in result]
            names.extend(db_names)
        
        return sorted(set(names))
    
    def render_prompt(self, template: str, **variables) -> str:
        """Render a prompt template with variables"""
        for key, value in variables.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template