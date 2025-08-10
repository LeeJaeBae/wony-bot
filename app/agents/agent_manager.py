"""Agent Manager for coordinating multiple agents"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json

from app.agents.base_agent import BaseAgent, AgentRole, AgentStatus
from app.agents.specialized_agents import (
    ResearchAgent,
    CodeAgent,
    AnalysisAgent,
    SummaryAgent,
    CreativeAgent
)
from app.core.ollama import OllamaClient

logger = logging.getLogger(__name__)


class AgentManager:
    """Manages and coordinates multiple agents"""
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        """Initialize Agent Manager
        
        Args:
            ollama_client: Optional Ollama client instance
        """
        self.ollama = ollama_client or OllamaClient()
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue = []
        self.completed_tasks = []
        self.active_tasks = {}
        
        # Initialize default agents
        self._initialize_default_agents()
        
        logger.info("Agent Manager initialized with default agents")
    
    def _initialize_default_agents(self):
        """Initialize default specialized agents"""
        # Create specialized agents
        self.agents['researcher'] = ResearchAgent(self.ollama)
        self.agents['coder'] = CodeAgent(self.ollama)
        self.agents['analyst'] = AnalysisAgent(self.ollama)
        self.agents['summarizer'] = SummaryAgent(self.ollama)
        self.agents['creative'] = CreativeAgent(self.ollama)
        
        logger.info(f"Initialized {len(self.agents)} default agents")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID
        
        Args:
            agent_id: Agent ID or role name
            
        Returns:
            Agent instance or None
        """
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents
        
        Returns:
            List of agent information
        """
        agent_list = []
        for agent_id, agent in self.agents.items():
            agent_list.append({
                'id': agent_id,
                'name': agent.name,
                'role': agent.role.value,
                'status': agent.status.value,
                'capabilities': agent.capabilities,
                'tasks_completed': len(agent.task_history)
            })
        return agent_list
    
    def find_best_agent(self, task: str) -> Optional[BaseAgent]:
        """Find the best agent for a task
        
        Args:
            task: Task description
            
        Returns:
            Best suited agent or None
        """
        # Check each agent's capability to handle the task
        candidates = []
        for agent_id, agent in self.agents.items():
            if agent.can_handle(task) and agent.status == AgentStatus.IDLE:
                candidates.append(agent)
        
        if not candidates:
            # If no idle agents, check all agents
            for agent_id, agent in self.agents.items():
                if agent.can_handle(task):
                    candidates.append(agent)
        
        # Return the first available candidate
        # In future, could implement more sophisticated selection
        return candidates[0] if candidates else None
    
    async def assign_task(
        self,
        task: str,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Assign a task to an agent
        
        Args:
            task: Task description
            agent_id: Optional specific agent ID
            context: Optional task context
            
        Returns:
            Task assignment result
        """
        # Find agent
        if agent_id:
            agent = self.get_agent(agent_id)
            if not agent:
                return {
                    'status': 'error',
                    'error': f'Agent {agent_id} not found'
                }
        else:
            agent = self.find_best_agent(task)
            if not agent:
                return {
                    'status': 'error',
                    'error': 'No suitable agent found for this task'
                }
        
        # Check if agent is available
        if agent.status == AgentStatus.WORKING:
            # Add to queue if agent is busy
            self.task_queue.append({
                'task': task,
                'agent_id': agent.id,
                'context': context,
                'queued_at': datetime.now()
            })
            return {
                'status': 'queued',
                'agent': agent.name,
                'queue_position': len(self.task_queue)
            }
        
        # Process task
        logger.info(f"Assigning task to {agent.name}: {task}")
        result = await agent.process_task(task, context)
        
        # Track completed task
        self.completed_tasks.append({
            'task': task,
            'agent': agent.name,
            'result': result,
            'completed_at': datetime.now()
        })
        
        return result
    
    async def process_parallel_tasks(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process multiple tasks in parallel
        
        Args:
            tasks: List of task dictionaries with 'task' and optional 'agent_id'
            
        Returns:
            List of results
        """
        # Create async tasks
        async_tasks = []
        for task_info in tasks:
            task = task_info.get('task')
            agent_id = task_info.get('agent_id')
            context = task_info.get('context')
            
            async_tasks.append(
                self.assign_task(task, agent_id, context)
            )
        
        # Process in parallel
        results = await asyncio.gather(*async_tasks)
        return results
    
    async def delegate_complex_task(
        self,
        complex_task: str,
        breakdown: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Delegate a complex task to multiple agents
        
        Args:
            complex_task: Complex task description
            breakdown: Optional task breakdown
            
        Returns:
            Combined results
        """
        # If no breakdown provided, analyze the task
        if not breakdown:
            # Use analyst to break down the task
            analysis_result = await self.assign_task(
                f"Break down this complex task into subtasks: {complex_task}",
                agent_id='analyst'
            )
            
            # Extract subtasks from analysis (simplified)
            # In a real implementation, would parse the response more carefully
            breakdown = [complex_task]  # Fallback to original task
        
        # Assign subtasks to appropriate agents
        subtask_assignments = []
        for subtask in breakdown:
            agent = self.find_best_agent(subtask)
            if agent:
                subtask_assignments.append({
                    'task': subtask,
                    'agent_id': agent.id
                })
        
        # Process subtasks in parallel
        results = await self.process_parallel_tasks(subtask_assignments)
        
        # Summarize results
        summary_context = {
            'text': json.dumps(results, indent=2),
            'type': 'task_results'
        }
        summary = await self.assign_task(
            f"Summarize the results of this complex task: {complex_task}",
            agent_id='summarizer',
            context=summary_context
        )
        
        return {
            'status': 'success',
            'complex_task': complex_task,
            'subtask_results': results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent status or None
        """
        agent = self.get_agent(agent_id)
        return agent.get_status() if agent else None
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all agents and manager
        
        Returns:
            Complete status report
        """
        return {
            'agents': [agent.get_status() for agent in self.agents.values()],
            'queue_length': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'active_agents': sum(1 for a in self.agents.values() if a.status == AgentStatus.WORKING)
        }
    
    async def process_queue(self):
        """Process queued tasks"""
        while self.task_queue:
            # Get next task from queue
            queued_task = self.task_queue.pop(0)
            
            # Try to assign it
            result = await self.assign_task(
                queued_task['task'],
                queued_task.get('agent_id'),
                queued_task.get('context')
            )
            
            # If still queued, put it back
            if result.get('status') == 'queued':
                self.task_queue.insert(0, queued_task)
                break
    
    def reset_all_agents(self):
        """Reset all agents to idle state"""
        for agent in self.agents.values():
            agent.reset()
        logger.info("All agents reset to idle state")
    
    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task history
        
        Args:
            limit: Number of tasks to return
            
        Returns:
            List of completed tasks
        """
        return self.completed_tasks[-limit:] if self.completed_tasks else []