"""Specialized agents for different tasks"""

import asyncio
from typing import Dict, Any, Optional, List
import json
import re
from datetime import datetime
import logging

from app.agents.base_agent import BaseAgent, AgentRole, AgentStatus
from app.core.ollama import OllamaClient
from app.models.schemas import Message, MessageRole

logger = logging.getLogger(__name__)


async def _consume_chat_stream(chat_call) -> str:
    """Consume an Ollama chat call that may be a coroutine or an async iterator.
    Supports both patterns used in tests and runtime.
    """
    # If the call returned a coroutine, await it to get the iterator/result
    try:
        import inspect
        if inspect.iscoroutine(chat_call):
            chat_call = await chat_call
    except Exception:
        # Fallback: proceed with whatever object we have
        pass

    # If the result is an async iterable, iterate and concatenate
    response_text = ""
    if hasattr(chat_call, "__aiter__"):
        async for chunk in chat_call:
            response_text += chunk
        return response_text

    # If it's a plain string (unlikely), return as is
    if isinstance(chat_call, str):
        return chat_call

    # If it's awaitable (AsyncMock may be awaitable), try awaiting once to get text
    try:
        return await chat_call
    except Exception:
        # Last resort: string cast
        return str(chat_call)

class ResearchAgent(BaseAgent):
    """Agent specialized in research and information gathering"""
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__(
            name="Research Assistant",
            role=AgentRole.RESEARCHER,
            description="Expert at finding and analyzing information",
            capabilities=[
                "Information research",
                "Fact-checking",
                "Source verification",
                "Topic investigation",
                "Knowledge synthesis"
            ]
        )
        self.ollama = ollama_client
    
    async def process_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process research task"""
        try:
            task_id = await self.start_task(task, context)
            
            # Prepare research prompt
            research_prompt = f"""As a research specialist, please help with the following task:

Task: {task}

Please provide:
1. Key findings and information
2. Important facts and data
3. Relevant context and background
4. Sources or references (if applicable)
5. Summary of findings

Be thorough, accurate, and cite sources when possible."""
            
            # Create messages for Ollama
            messages = [
                Message(role=MessageRole.SYSTEM, content=self.system_prompt),
                Message(role=MessageRole.USER, content=research_prompt)
            ]
            
            # Get response from Ollama (supports coroutine or async iterator)
            response = await _consume_chat_stream(self.ollama.chat(messages, stream=False))
            
            # Process and structure the response
            result = {
                'status': 'success',
                'agent': self.name,
                'task': task,
                'findings': response,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.complete_task(result)
            return result
            
        except Exception as e:
            logger.error(f"Research agent error: {e}")
            self.status = AgentStatus.ERROR
            return {
                'status': 'error',
                'agent': self.name,
                'error': str(e)
            }


class CodeAgent(BaseAgent):
    """Agent specialized in coding and programming tasks"""
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__(
            name="Code Assistant",
            role=AgentRole.CODER,
            description="Expert at writing and debugging code",
            capabilities=[
                "Code writing",
                "Debugging",
                "Code review",
                "Refactoring",
                "Algorithm design",
                "API development"
            ]
        )
        self.ollama = ollama_client
    
    async def process_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process coding task"""
        try:
            task_id = await self.start_task(task, context)
            
            # Extract language if specified
            language = context.get('language', 'python') if context else 'python'
            
            # Prepare coding prompt
            coding_prompt = f"""As a coding expert, please help with the following task:

Task: {task}
Language: {language}

Please provide:
1. Clean, well-commented code
2. Explanation of the approach
3. Any important considerations
4. Example usage (if applicable)

Follow best practices and ensure the code is production-ready."""
            
            # Create messages for Ollama
            messages = [
                Message(role=MessageRole.SYSTEM, content=self.system_prompt),
                Message(role=MessageRole.USER, content=coding_prompt)
            ]
            
            # Get response from Ollama
            response = await _consume_chat_stream(self.ollama.chat(messages, stream=False))
            
            # Extract code blocks
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', response, re.DOTALL)
            
            result = {
                'status': 'success',
                'agent': self.name,
                'task': task,
                'response': response,
                'code_blocks': code_blocks,
                'language': language,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.complete_task(result)
            return result
            
        except Exception as e:
            logger.error(f"Code agent error: {e}")
            self.status = AgentStatus.ERROR
            return {
                'status': 'error',
                'agent': self.name,
                'error': str(e)
            }


class AnalysisAgent(BaseAgent):
    """Agent specialized in data analysis and insights"""
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__(
            name="Analysis Assistant",
            role=AgentRole.ANALYST,
            description="Expert at analyzing data and providing insights",
            capabilities=[
                "Data analysis",
                "Pattern recognition",
                "Statistical analysis",
                "Trend identification",
                "Insight generation",
                "Report creation"
            ]
        )
        self.ollama = ollama_client
    
    async def process_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process analysis task"""
        try:
            task_id = await self.start_task(task, context)
            
            # Prepare analysis prompt
            data = context.get('data', '') if context else ''
            analysis_prompt = f"""As a data analyst, please help with the following task:

Task: {task}
{f'Data: {data}' if data else ''}

Please provide:
1. Key observations and patterns
2. Statistical insights (if applicable)
3. Trends and correlations
4. Actionable recommendations
5. Summary of findings

Be thorough and data-driven in your analysis."""
            
            # Create messages for Ollama
            messages = [
                Message(role=MessageRole.SYSTEM, content=self.system_prompt),
                Message(role=MessageRole.USER, content=analysis_prompt)
            ]
            
            # Get response from Ollama
            response = await _consume_chat_stream(self.ollama.chat(messages, stream=False))
            
            result = {
                'status': 'success',
                'agent': self.name,
                'task': task,
                'analysis': response,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.complete_task(result)
            return result
            
        except Exception as e:
            logger.error(f"Analysis agent error: {e}")
            self.status = AgentStatus.ERROR
            return {
                'status': 'error',
                'agent': self.name,
                'error': str(e)
            }


class SummaryAgent(BaseAgent):
    """Agent specialized in summarization and key point extraction"""
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__(
            name="Summary Assistant",
            role=AgentRole.SUMMARIZER,
            description="Expert at creating concise summaries",
            capabilities=[
                "Text summarization",
                "Key point extraction",
                "Executive summaries",
                "Meeting notes",
                "Document digests"
            ]
        )
        self.ollama = ollama_client
    
    async def process_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process summarization task"""
        try:
            task_id = await self.start_task(task, context)
            
            # Get text to summarize
            text = context.get('text', task) if context else task
            summary_type = context.get('type', 'general') if context else 'general'
            
            # Prepare summary prompt
            summary_prompt = f"""As a summarization expert, please help with the following task:

Text/Task: {text}
Summary Type: {summary_type}

Please provide:
1. Executive summary (2-3 sentences)
2. Key points (bullet list)
3. Important details
4. Action items (if any)
5. Conclusion

Keep the summary clear, concise, and comprehensive."""
            
            # Create messages for Ollama
            messages = [
                Message(role=MessageRole.SYSTEM, content=self.system_prompt),
                Message(role=MessageRole.USER, content=summary_prompt)
            ]
            
            # Get response from Ollama
            response = await _consume_chat_stream(self.ollama.chat(messages, stream=False))
            
            result = {
                'status': 'success',
                'agent': self.name,
                'task': task,
                'summary': response,
                'summary_type': summary_type,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.complete_task(result)
            return result
            
        except Exception as e:
            logger.error(f"Summary agent error: {e}")
            self.status = AgentStatus.ERROR
            return {
                'status': 'error',
                'agent': self.name,
                'error': str(e)
            }


class CreativeAgent(BaseAgent):
    """Agent specialized in creative tasks and ideation"""
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__(
            name="Creative Assistant",
            role=AgentRole.CREATIVE,
            description="Expert at creative thinking and ideation",
            capabilities=[
                "Creative writing",
                "Brainstorming",
                "Idea generation",
                "Story creation",
                "Design concepts",
                "Problem solving"
            ]
        )
        self.ollama = ollama_client
    
    async def process_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process creative task"""
        try:
            task_id = await self.start_task(task, context)
            
            # Get creative parameters
            style = context.get('style', 'creative') if context else 'creative'
            tone = context.get('tone', 'professional') if context else 'professional'
            
            # Prepare creative prompt
            creative_prompt = f"""As a creative expert, please help with the following task:

Task: {task}
Style: {style}
Tone: {tone}

Please provide:
1. Creative solutions/ideas
2. Unique perspectives
3. Innovative approaches
4. Multiple variations (if applicable)
5. Implementation suggestions

Be creative, think outside the box, and provide original ideas."""
            
            # Create messages for Ollama
            messages = [
                Message(role=MessageRole.SYSTEM, content=self.system_prompt),
                Message(role=MessageRole.USER, content=creative_prompt)
            ]
            
            # Get response from Ollama
            response = ""
            async for chunk in self.ollama.chat(messages, stream=False):
                response += chunk
            
            result = {
                'status': 'success',
                'agent': self.name,
                'task': task,
                'creative_output': response,
                'style': style,
                'tone': tone,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.complete_task(result)
            return result
            
        except Exception as e:
            logger.error(f"Creative agent error: {e}")
            self.status = AgentStatus.ERROR
            return {
                'status': 'error',
                'agent': self.name,
                'error': str(e)
            }