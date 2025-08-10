"""Base Agent class for WonyBot agents"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import logging
from app.models.schemas import Message, MessageRole

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles/types"""
    RESEARCHER = "researcher"
    CODER = "coder"
    ANALYST = "analyst"
    SUMMARIZER = "summarizer"
    CREATIVE = "creative"
    GENERAL = "general"


class AgentStatus(Enum):
    """Agent status"""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(
        self,
        name: str,
        role: AgentRole,
        description: str,
        capabilities: List[str],
        system_prompt: Optional[str] = None
    ):
        """Initialize base agent
        
        Args:
            name: Agent name
            role: Agent role/type
            description: Agent description
            capabilities: List of agent capabilities
            system_prompt: Custom system prompt for the agent
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.description = description
        self.capabilities = capabilities
        self.system_prompt = system_prompt or self._get_default_prompt()
        self.status = AgentStatus.IDLE
        self.created_at = datetime.now()
        self.last_activity = None
        self.task_history = []
        self.current_task = None
        
        logger.info(f"Agent '{name}' ({role.value}) initialized")
    
    def _get_default_prompt(self) -> str:
        """Get default system prompt based on role"""
        prompts = {
            AgentRole.RESEARCHER: """You are a Research Agent specialized in:
- Finding and analyzing information
- Fact-checking and verification
- Summarizing research findings
- Providing citations and sources
Always provide accurate, well-researched information.""",
            
            AgentRole.CODER: """You are a Coding Agent specialized in:
- Writing clean, efficient code
- Debugging and fixing issues
- Code review and optimization
- Multiple programming languages
Always follow best practices and write well-commented code.""",
            
            AgentRole.ANALYST: """You are an Analysis Agent specialized in:
- Data analysis and interpretation
- Pattern recognition
- Statistical analysis
- Creating insights from data
Always provide clear, data-driven insights.""",
            
            AgentRole.SUMMARIZER: """You are a Summary Agent specialized in:
- Creating concise summaries
- Extracting key points
- Organizing information hierarchically
- Maintaining context accuracy
Always create clear, comprehensive summaries.""",
            
            AgentRole.CREATIVE: """You are a Creative Agent specialized in:
- Creative writing and ideation
- Brainstorming solutions
- Generating innovative ideas
- Artistic and creative tasks
Always think outside the box and be creative.""",
            
            AgentRole.GENERAL: """You are a General Purpose Agent capable of:
- Handling various tasks
- Adapting to different requirements
- Providing helpful assistance
- Problem-solving
Always be helpful and adaptive."""
        }
        return prompts.get(self.role, prompts[AgentRole.GENERAL])
    
    @abstractmethod
    async def process_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a task
        
        Args:
            task: Task description
            context: Optional context for the task
            
        Returns:
            Result dictionary with status and output
        """
        pass
    
    async def start_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Start processing a task
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        self.current_task = {
            'id': task_id,
            'task': task,
            'context': context,
            'started_at': datetime.now(),
            'status': 'processing'
        }
        self.status = AgentStatus.WORKING
        self.last_activity = datetime.now()
        
        logger.info(f"Agent '{self.name}' started task: {task_id}")
        return task_id
    
    async def complete_task(self, result: Dict[str, Any]) -> None:
        """Mark current task as completed
        
        Args:
            result: Task result
        """
        if self.current_task:
            self.current_task['completed_at'] = datetime.now()
            self.current_task['status'] = 'completed'
            self.current_task['result'] = result
            self.task_history.append(self.current_task)
            self.current_task = None
        
        self.status = AgentStatus.IDLE
        self.last_activity = datetime.now()
        
        logger.info(f"Agent '{self.name}' completed task")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status
        
        Returns:
            Status dictionary
        """
        return {
            'id': self.id,
            'name': self.name,
            'role': self.role.value,
            'status': self.status.value,
            'current_task': self.current_task,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'tasks_completed': len(self.task_history)
        }
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities
        
        Returns:
            List of capabilities
        """
        return self.capabilities
    
    def can_handle(self, task: str) -> bool:
        """Check if agent can handle a task
        
        Args:
            task: Task description
            
        Returns:
            True if agent can handle the task
        """
        task_lower = task.lower()
        
        # Check if any capability keyword is in the task
        for capability in self.capabilities:
            if any(keyword in task_lower for keyword in capability.lower().split()):
                return True
        
        # Role-specific checks
        role_keywords = {
            AgentRole.RESEARCHER: ['research', 'find', 'search', 'investigate', 'lookup'],
            AgentRole.CODER: ['code', 'program', 'debug', 'implement', 'develop'],
            AgentRole.ANALYST: ['analyze', 'analyse', 'data', 'statistics', 'pattern'],
            AgentRole.SUMMARIZER: ['summarize', 'summary', 'brief', 'outline', 'key points'],
            AgentRole.CREATIVE: ['create', 'design', 'imagine', 'brainstorm', 'idea']
        }
        
        if self.role in role_keywords:
            return any(keyword in task_lower for keyword in role_keywords[self.role])
        
        return False
    
    def reset(self) -> None:
        """Reset agent state"""
        self.status = AgentStatus.IDLE
        self.current_task = None
        self.last_activity = None
        logger.info(f"Agent '{self.name}' reset")