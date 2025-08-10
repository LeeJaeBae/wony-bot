"""Orchestrator Agent - Central coordinator for hierarchical agent system"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json
import logging
from dataclasses import dataclass, field
from collections import defaultdict

from app.agents.base_agent import BaseAgent, AgentRole, AgentStatus
from app.agents.agent_manager import AgentManager
from app.core.ollama import OllamaClient

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    VOTING = "voting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Task representation"""
    id: str
    description: str
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    votes: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            'id': self.id,
            'description': self.description,
            'priority': self.priority.value,
            'status': self.status.value,
            'assigned_to': self.assigned_to,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
            'votes': self.votes
        }


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent - Central coordinator
    
    Responsibilities:
    1. Task decomposition and planning
    2. Work distribution to worker agents
    3. Coordination of collective decision making
    4. Performance monitoring and optimization
    5. Conflict resolution
    """
    
    def __init__(
        self,
        agent_manager: AgentManager,
        ollama_client: Optional[OllamaClient] = None,
        consensus_threshold: float = 0.6
    ):
        """Initialize Orchestrator Agent
        
        Args:
            agent_manager: Agent manager instance
            ollama_client: Optional Ollama client
            consensus_threshold: Threshold for consensus (0.0 to 1.0)
        """
        super().__init__(
            name="Orchestrator",
            role=AgentRole.GENERAL,
            description="Central coordinator for hierarchical agent system",
            capabilities=[
                "Task planning and decomposition",
                "Work distribution",
                "Consensus coordination",
                "Performance monitoring",
                "Conflict resolution"
            ],
            system_prompt="""You are the Orchestrator Agent, responsible for:
1. Breaking down complex tasks into subtasks
2. Assigning work to appropriate worker agents
3. Coordinating collective decision making
4. Monitoring performance and optimizing distribution
5. Resolving conflicts and ensuring consensus

Always think strategically and ensure efficient task completion."""
        )
        
        self.agent_manager = agent_manager
        self.ollama = ollama_client or OllamaClient()
        self.consensus_threshold = consensus_threshold
        
        # Task management
        self.task_queue: List[Task] = []
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        
        # Performance tracking
        self.agent_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'tasks_completed': 0,
            'success_rate': 1.0,
            'average_time': 0.0,
            'specialties': []
        })
        
        # Decision history
        self.decision_history: List[Dict[str, Any]] = []
        
        logger.info("Orchestrator Agent initialized with consensus threshold: %.2f", consensus_threshold)
    
    async def process_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a task through orchestration
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Orchestration result
        """
        try:
            task_id = await self.start_task(task, context)
            
            # Step 1: Decompose task
            subtasks = await self.decompose_task(task, context)
            logger.info(f"Decomposed task into {len(subtasks)} subtasks")
            
            # Step 2: Create task objects
            tasks = []
            for i, subtask in enumerate(subtasks):
                task_obj = Task(
                    id=f"{task_id}-{i}",
                    description=subtask['description'],
                    priority=TaskPriority(subtask.get('priority', 3))
                )
                tasks.append(task_obj)
                self.task_queue.append(task_obj)
            
            # Step 3: Distribute tasks
            distribution_results = await self.distribute_tasks(tasks)
            
            # Step 4: Monitor execution
            execution_results = await self.monitor_execution(tasks)
            
            # Step 5: Collective decision on results
            final_decision = await self.coordinate_consensus(execution_results)
            
            # Step 6: Compile final result
            result = {
                'status': 'success',
                'orchestrator': self.name,
                'original_task': task,
                'subtasks': len(tasks),
                'distribution': distribution_results,
                'execution': execution_results,
                'consensus': final_decision,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.complete_task(result)
            return result
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            self.status = AgentStatus.ERROR
            return {
                'status': 'error',
                'orchestrator': self.name,
                'error': str(e)
            }
    
    async def decompose_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Decompose complex task into subtasks
        
        Args:
            task: Main task description
            context: Optional context
            
        Returns:
            List of subtasks
        """
        from app.models.schemas import Message, MessageRole
        
        decomposition_prompt = f"""As the Orchestrator, decompose this task into subtasks:

Task: {task}
{f"Context: {json.dumps(context)}" if context else ""}

Please provide a JSON list of subtasks with:
1. description: Clear description of the subtask
2. priority: Priority level (1-5, where 5 is highest)
3. estimated_time: Estimated time in minutes
4. required_role: Best agent role for this task (researcher/coder/analyst/summarizer/creative)

Format:
[
    {{
        "description": "...",
        "priority": 3,
        "estimated_time": 10,
        "required_role": "researcher"
    }},
    ...
]

Provide ONLY the JSON array, no additional text."""
        
        messages = [
            Message(role=MessageRole.SYSTEM, content=self.system_prompt),
            Message(role=MessageRole.USER, content=decomposition_prompt)
        ]
        
        response = ""
        async for chunk in self.ollama.chat(messages, stream=False):
            response += chunk
        
        # Parse JSON response
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                subtasks = json.loads(json_match.group())
            else:
                # Fallback to simple task
                subtasks = [{
                    "description": task,
                    "priority": 3,
                    "estimated_time": 30,
                    "required_role": "general"
                }]
        except json.JSONDecodeError:
            logger.warning("Failed to parse subtasks JSON, using fallback")
            subtasks = [{
                "description": task,
                "priority": 3,
                "estimated_time": 30,
                "required_role": "general"
            }]
        
        return subtasks
    
    async def distribute_tasks(
        self,
        tasks: List[Task]
    ) -> Dict[str, Any]:
        """Distribute tasks to appropriate agents
        
        Args:
            tasks: List of tasks to distribute
            
        Returns:
            Distribution results
        """
        distribution = {}
        
        for task in tasks:
            # Find best agent for the task
            best_agent = self.agent_manager.find_best_agent(task.description)
            
            if best_agent:
                # Assign task
                task.assigned_to = best_agent.id
                task.status = TaskStatus.ASSIGNED
                self.active_tasks[task.id] = task
                
                distribution[task.id] = {
                    'task': task.description,
                    'assigned_to': best_agent.name,
                    'agent_role': best_agent.role.value,
                    'priority': task.priority.value
                }
                
                logger.info(f"Assigned task {task.id} to {best_agent.name}")
            else:
                logger.warning(f"No suitable agent found for task {task.id}")
                distribution[task.id] = {
                    'task': task.description,
                    'assigned_to': None,
                    'status': 'unassigned'
                }
        
        return distribution
    
    async def monitor_execution(
        self,
        tasks: List[Task]
    ) -> Dict[str, Any]:
        """Monitor task execution by worker agents
        
        Args:
            tasks: Tasks to monitor
            
        Returns:
            Execution results
        """
        results = {}
        
        # Execute tasks in parallel
        async_tasks = []
        for task in tasks:
            if task.assigned_to:
                async_tasks.append(self._execute_task(task))
        
        if async_tasks:
            task_results = await asyncio.gather(*async_tasks)
            
            for task, result in zip(tasks, task_results):
                task.result = result
                task.status = TaskStatus.VOTING if result['status'] == 'success' else TaskStatus.FAILED
                results[task.id] = result
        
        return results
    
    async def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result
        """
        try:
            # Get assigned agent
            agent = self.agent_manager.get_agent(task.assigned_to)
            if not agent:
                return {'status': 'error', 'error': 'Agent not found'}
            
            # Execute task
            start_time = datetime.now()
            result = await self.agent_manager.assign_task(
                task=task.description,
                agent_id=task.assigned_to
            )
            
            # Update performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_performance(task.assigned_to, result['status'] == 'success', execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def coordinate_consensus(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate collective decision making
        
        Args:
            results: Execution results from all agents
            
        Returns:
            Consensus decision
        """
        # Prepare for voting
        voting_items = []
        for task_id, result in results.items():
            if result.get('status') == 'success':
                voting_items.append({
                    'task_id': task_id,
                    'result': result,
                    'votes': {}
                })
        
        if not voting_items:
            return {
                'consensus_reached': False,
                'reason': 'No successful results to vote on'
            }
        
        # Collect votes from all agents
        all_agents = self.agent_manager.list_agents()
        for item in voting_items:
            for agent in all_agents:
                # Each agent votes on the result
                vote = await self._get_agent_vote(
                    agent['id'],
                    item['task_id'],
                    item['result']
                )
                item['votes'][agent['id']] = vote
        
        # Calculate consensus
        consensus_results = []
        for item in voting_items:
            total_votes = len(item['votes'])
            positive_votes = sum(1 for v in item['votes'].values() if v)
            consensus_ratio = positive_votes / total_votes if total_votes > 0 else 0
            
            consensus_results.append({
                'task_id': item['task_id'],
                'consensus_ratio': consensus_ratio,
                'consensus_reached': consensus_ratio >= self.consensus_threshold,
                'positive_votes': positive_votes,
                'total_votes': total_votes,
                'votes': item['votes']
            })
        
        # Record decision
        decision = {
            'consensus_reached': any(r['consensus_reached'] for r in consensus_results),
            'timestamp': datetime.now().isoformat(),
            'results': consensus_results,
            'threshold': self.consensus_threshold
        }
        
        self.decision_history.append(decision)
        
        return decision
    
    async def _get_agent_vote(
        self,
        agent_id: str,
        task_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """Get vote from an agent on a result
        
        Args:
            agent_id: Agent ID
            task_id: Task ID
            result: Result to vote on
            
        Returns:
            True for approve, False for reject
        """
        # Simplified voting logic
        # In a real system, each agent would evaluate based on their expertise
        
        # For now, use simple heuristics
        if result.get('status') != 'success':
            return False
        
        # Agents tend to approve results from their domain
        agent = self.agent_manager.get_agent(agent_id)
        if agent and result.get('agent') == agent.name:
            return True
        
        # Random approval based on result quality (simplified)
        import random
        return random.random() > 0.3
    
    def _update_performance(
        self,
        agent_id: str,
        success: bool,
        execution_time: float
    ):
        """Update agent performance metrics
        
        Args:
            agent_id: Agent ID
            success: Whether task was successful
            execution_time: Task execution time in seconds
        """
        metrics = self.agent_performance[agent_id]
        
        # Update metrics
        metrics['tasks_completed'] += 1
        
        # Update success rate (moving average)
        current_rate = metrics['success_rate']
        metrics['success_rate'] = (current_rate * (metrics['tasks_completed'] - 1) + 
                                   (1.0 if success else 0.0)) / metrics['tasks_completed']
        
        # Update average time (moving average)
        current_avg = metrics['average_time']
        metrics['average_time'] = (current_avg * (metrics['tasks_completed'] - 1) + 
                                   execution_time) / metrics['tasks_completed']
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all agents
        
        Returns:
            Performance report
        """
        return {
            'orchestrator': self.name,
            'total_tasks': len(self.completed_tasks),
            'active_tasks': len(self.active_tasks),
            'queued_tasks': len(self.task_queue),
            'agent_performance': dict(self.agent_performance),
            'consensus_history': len(self.decision_history),
            'last_decisions': self.decision_history[-5:] if self.decision_history else []
        }
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get current task status
        
        Returns:
            Task status information
        """
        return {
            'queue': [t.to_dict() for t in self.task_queue],
            'active': {k: v.to_dict() for k, v in self.active_tasks.items()},
            'completed': [t.to_dict() for t in self.completed_tasks[-10:]]
        }