"""Hierarchical Agent Manager with Orchestrator and Consensus System"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import json
import uuid

from app.agents.agent_manager import AgentManager
from app.agents.orchestrator import OrchestratorAgent, TaskPriority
from app.agents.consensus import ConsensusSystem, VoteType, VoteOption, Proposal
from app.agents.base_agent import AgentStatus
from app.core.ollama import OllamaClient

logger = logging.getLogger(__name__)


class HierarchicalAgentManager:
    """
    Hierarchical Agent Manager
    
    Architecture:
    - Orchestrator: Central coordinator that manages tasks
    - Worker Agents: Execute assigned tasks
    - Consensus System: Collective decision making
    
    Features:
    1. Task decomposition and planning by Orchestrator
    2. Automatic work distribution to specialized agents
    3. Collective decision making through voting
    4. Performance monitoring and optimization
    5. Inter-agent communication
    """
    
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        consensus_threshold: float = 0.6,
        vote_type: VoteType = VoteType.SIMPLE_MAJORITY,
        enable_auto_scaling: bool = True
    ):
        """Initialize Hierarchical Agent Manager
        
        Args:
            ollama_client: Optional Ollama client
            consensus_threshold: Threshold for consensus decisions
            vote_type: Default voting mechanism
            enable_auto_scaling: Enable automatic agent scaling
        """
        self.ollama = ollama_client or OllamaClient()
        
        # Initialize base agent manager with worker agents
        self.agent_manager = AgentManager(ollama_client=self.ollama)
        
        # Initialize Orchestrator
        self.orchestrator = OrchestratorAgent(
            agent_manager=self.agent_manager,
            ollama_client=self.ollama,
            consensus_threshold=consensus_threshold
        )
        
        # Initialize Consensus System
        self.consensus_system = ConsensusSystem(
            default_vote_type=vote_type,
            default_timeout=60
        )
        
        # Configuration
        self.enable_auto_scaling = enable_auto_scaling
        self.consensus_threshold = consensus_threshold
        
        # Communication channels (message queues)
        self.message_queues: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance metrics
        self.system_metrics = {
            'tasks_processed': 0,
            'consensus_decisions': 0,
            'average_consensus_time': 0.0,
            'task_success_rate': 1.0
        }
        
        # Initialize agent weights based on performance
        self._initialize_agent_weights()
        
        logger.info("Hierarchical Agent Manager initialized")
    
    def _initialize_agent_weights(self):
        """Initialize voting weights for agents based on their roles"""
        # Orchestrator has higher weight
        self.consensus_system.set_agent_weight('orchestrator', 2.0)
        
        # Specialized agents have equal weight
        for agent_id in self.agent_manager.agents.keys():
            self.consensus_system.set_agent_weight(agent_id, 1.0)
    
    async def process_complex_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        require_consensus: bool = True
    ) -> Dict[str, Any]:
        """Process a complex task through the hierarchical system
        
        Args:
            task: Task description
            context: Optional context
            require_consensus: Whether to require consensus on results
            
        Returns:
            Processing result with consensus decision
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Orchestrator decomposes and plans the task
            logger.info("Orchestrator processing task...")
            orchestration_result = await self.orchestrator.process_task(task, context)
            
            if orchestration_result['status'] != 'success':
                return orchestration_result
            
            # Step 2: If consensus is required, create proposal
            if require_consensus:
                proposal_id = str(uuid.uuid4())
                proposal = await self.consensus_system.create_proposal(
                    proposal_id=proposal_id,
                    title=f"Task Result: {task[:50]}...",
                    description=json.dumps(orchestration_result, indent=2),
                    proposer='orchestrator',
                    deadline=datetime.now() + timedelta(minutes=5)
                )
                
                # Step 3: Collect votes from all agents
                await self._collect_votes(proposal_id, orchestration_result)
                
                # Step 4: Tally votes and get consensus
                voting_result = await self.consensus_system.tally_votes(
                    proposal_id=proposal_id,
                    required_voters=list(self.agent_manager.agents.keys()) + ['orchestrator']
                )
                
                # Update metrics
                self.system_metrics['consensus_decisions'] += 1
                consensus_time = (datetime.now() - start_time).total_seconds()
                self._update_average_consensus_time(consensus_time)
                
                # Step 5: Prepare final result
                final_result = {
                    'status': 'success' if voting_result.consensus_reached else 'partial',
                    'task': task,
                    'orchestration': orchestration_result,
                    'consensus': {
                        'reached': voting_result.consensus_reached,
                        'approval_rate': voting_result.approval_rate,
                        'participation_rate': voting_result.participation_rate,
                        'decision': voting_result.result.value,
                        'votes': len(voting_result.votes)
                    },
                    'execution_time': (datetime.now() - start_time).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # No consensus required, return orchestration result
                final_result = {
                    'status': 'success',
                    'task': task,
                    'orchestration': orchestration_result,
                    'consensus': {'required': False},
                    'execution_time': (datetime.now() - start_time).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Update metrics
            self.system_metrics['tasks_processed'] += 1
            self._update_success_rate(final_result['status'] == 'success')
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing complex task: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'task': task,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _collect_votes(
        self,
        proposal_id: str,
        orchestration_result: Dict[str, Any]
    ):
        """Collect votes from all agents on the orchestration result
        
        Args:
            proposal_id: Proposal ID for voting
            orchestration_result: Result to vote on
        """
        # Orchestrator votes first
        orchestrator_vote = await self._evaluate_result(
            'orchestrator',
            orchestration_result
        )
        await self.consensus_system.cast_vote(
            proposal_id=proposal_id,
            voter_id='orchestrator',
            option=orchestrator_vote['option'],
            reason=orchestrator_vote['reason']
        )
        
        # Collect votes from worker agents
        for agent_id in self.agent_manager.agents.keys():
            agent_vote = await self._evaluate_result(
                agent_id,
                orchestration_result
            )
            await self.consensus_system.cast_vote(
                proposal_id=proposal_id,
                voter_id=agent_id,
                option=agent_vote['option'],
                reason=agent_vote['reason']
            )
    
    async def _evaluate_result(
        self,
        agent_id: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Have an agent evaluate a result and provide vote
        
        Args:
            agent_id: Agent evaluating the result
            result: Result to evaluate
            
        Returns:
            Vote decision with reason
        """
        # Simple evaluation logic
        # In a real system, each agent would analyze based on their expertise
        
        # Check if the agent was involved in the task
        execution_results = result.get('execution', {})
        agent_involved = any(
            agent_id in str(r.get('agent', ''))
            for r in execution_results.values()
        )
        
        # Agents tend to approve if:
        # 1. They were involved and it succeeded
        # 2. The overall consensus from subtasks is positive
        # 3. No critical failures
        
        has_failures = any(
            r.get('status') == 'error'
            for r in execution_results.values()
        )
        
        if has_failures:
            return {
                'option': VoteOption.REJECT,
                'reason': 'Task execution had failures'
            }
        
        if agent_involved:
            return {
                'option': VoteOption.APPROVE,
                'reason': 'Successfully contributed to task'
            }
        
        # Default: approve if most subtasks succeeded
        success_rate = sum(
            1 for r in execution_results.values()
            if r.get('status') == 'success'
        ) / max(len(execution_results), 1)
        
        if success_rate >= 0.7:
            return {
                'option': VoteOption.APPROVE,
                'reason': f'High success rate: {success_rate:.1%}'
            }
        else:
            return {
                'option': VoteOption.ABSTAIN,
                'reason': 'Insufficient information to decide'
            }
    
    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message: Dict[str, Any]
    ):
        """Send message between agents
        
        Args:
            from_agent: Sender agent ID
            to_agent: Receiver agent ID
            message: Message content
        """
        if to_agent not in self.message_queues:
            self.message_queues[to_agent] = []
        
        self.message_queues[to_agent].append({
            'from': from_agent,
            'to': to_agent,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.debug(f"Message sent: {from_agent} -> {to_agent}")
    
    async def broadcast_message(
        self,
        from_agent: str,
        message: Dict[str, Any]
    ):
        """Broadcast message to all agents
        
        Args:
            from_agent: Sender agent ID
            message: Message to broadcast
        """
        all_agents = list(self.agent_manager.agents.keys()) + ['orchestrator']
        
        for agent_id in all_agents:
            if agent_id != from_agent:
                await self.send_message(from_agent, agent_id, message)
        
        logger.info(f"Broadcast from {from_agent} to all agents")
    
    def get_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get messages for an agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of messages
        """
        messages = self.message_queues.get(agent_id, [])
        # Clear queue after reading
        if agent_id in self.message_queues:
            self.message_queues[agent_id] = []
        return messages
    
    def _update_average_consensus_time(self, new_time: float):
        """Update average consensus time metric"""
        current_avg = self.system_metrics['average_consensus_time']
        count = self.system_metrics['consensus_decisions']
        
        if count <= 1:
            self.system_metrics['average_consensus_time'] = new_time
        else:
            self.system_metrics['average_consensus_time'] = (
                (current_avg * (count - 1) + new_time) / count
            )
    
    def _update_success_rate(self, success: bool):
        """Update task success rate metric"""
        current_rate = self.system_metrics['task_success_rate']
        count = self.system_metrics['tasks_processed']
        
        if count <= 1:
            self.system_metrics['task_success_rate'] = 1.0 if success else 0.0
        else:
            self.system_metrics['task_success_rate'] = (
                (current_rate * (count - 1) + (1.0 if success else 0.0)) / count
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status
        
        Returns:
            System status information
        """
        return {
            'orchestrator': {
                'status': self.orchestrator.status.value,
                'performance': self.orchestrator.get_performance_report(),
                'tasks': self.orchestrator.get_task_status()
            },
            'workers': self.agent_manager.get_all_status(),
            'consensus': {
                'active_proposals': len(self.consensus_system.active_proposals),
                'voting_history': self.consensus_system.get_voting_history(5)
            },
            'metrics': self.system_metrics,
            'message_queues': {
                agent_id: len(queue)
                for agent_id, queue in self.message_queues.items()
            }
        }
    
    def get_agent_hierarchy(self) -> Dict[str, Any]:
        """Get agent hierarchy structure
        
        Returns:
            Hierarchy information
        """
        return {
            'orchestrator': {
                'name': self.orchestrator.name,
                'role': 'coordinator',
                'status': self.orchestrator.status.value,
                'capabilities': self.orchestrator.capabilities
            },
            'workers': [
                {
                    'id': agent_id,
                    'name': agent.name,
                    'role': agent.role.value,
                    'status': agent.status.value,
                    'capabilities': agent.capabilities
                }
                for agent_id, agent in self.agent_manager.agents.items()
            ],
            'consensus_system': {
                'vote_type': self.consensus_system.default_vote_type.value,
                'threshold': self.consensus_threshold,
                'weights': self.consensus_system.agent_weights
            }
        }
    
    async def scale_workers(self, target_count: int):
        """Scale the number of worker agents
        
        Args:
            target_count: Target number of workers
        """
        if not self.enable_auto_scaling:
            logger.warning("Auto-scaling is disabled")
            return
        
        current_count = len(self.agent_manager.agents)
        
        if target_count > current_count:
            # Need to add more agents
            # This would involve creating new agent instances
            logger.info(f"Scaling up from {current_count} to {target_count} workers")
        elif target_count < current_count:
            # Need to remove agents
            # This would involve gracefully shutting down agents
            logger.info(f"Scaling down from {current_count} to {target_count} workers")
        else:
            logger.info("No scaling needed")
    
    async def emergency_stop(self):
        """Emergency stop all operations"""
        logger.warning("Emergency stop initiated")
        
        # Reset all agents to idle
        self.orchestrator.reset()
        self.agent_manager.reset_all_agents()
        
        # Clear all active proposals
        self.consensus_system.active_proposals.clear()
        self.consensus_system.current_votes.clear()
        
        # Clear message queues
        self.message_queues.clear()
        
        logger.info("Emergency stop completed")