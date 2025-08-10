"""Consensus System - Collective decision making for agents"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import logging
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


class VoteType(Enum):
    """Types of voting mechanisms"""
    SIMPLE_MAJORITY = "simple_majority"      # >50% approval
    SUPER_MAJORITY = "super_majority"        # >66% approval  
    UNANIMOUS = "unanimous"                  # 100% approval
    WEIGHTED = "weighted"                     # Weighted by expertise
    RANKED_CHOICE = "ranked_choice"          # Ranked preference voting
    CONSENSUS = "consensus"                   # Continue until consensus


class VoteOption(Enum):
    """Vote options"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    DELEGATE = "delegate"


@dataclass
class Proposal:
    """Proposal for voting"""
    id: str
    title: str
    description: str
    proposer: str
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'proposer': self.proposer,
            'created_at': self.created_at.isoformat(),
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'metadata': self.metadata
        }


@dataclass
class Vote:
    """Individual vote"""
    voter_id: str
    option: VoteOption
    weight: float = 1.0
    reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'voter_id': self.voter_id,
            'option': self.option.value,
            'weight': self.weight,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class VotingResult:
    """Voting result"""
    proposal_id: str
    votes: List[Vote]
    vote_type: VoteType
    result: VoteOption
    consensus_reached: bool
    approval_rate: float
    participation_rate: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'proposal_id': self.proposal_id,
            'votes': [v.to_dict() for v in self.votes],
            'vote_type': self.vote_type.value,
            'result': self.result.value,
            'consensus_reached': self.consensus_reached,
            'approval_rate': self.approval_rate,
            'participation_rate': self.participation_rate,
            'timestamp': self.timestamp.isoformat()
        }


class ConsensusSystem:
    """
    Consensus System for collective decision making
    
    Features:
    1. Multiple voting mechanisms
    2. Weighted voting based on expertise
    3. Delegation support
    4. Voting history and audit trail
    5. Conflict resolution
    """
    
    def __init__(
        self,
        default_vote_type: VoteType = VoteType.SIMPLE_MAJORITY,
        default_timeout: int = 60  # seconds
    ):
        """Initialize Consensus System
        
        Args:
            default_vote_type: Default voting mechanism
            default_timeout: Default voting timeout in seconds
        """
        self.default_vote_type = default_vote_type
        self.default_timeout = default_timeout
        
        # Active proposals
        self.active_proposals: Dict[str, Proposal] = {}
        
        # Voting records
        self.current_votes: Dict[str, List[Vote]] = {}
        self.voting_history: List[VotingResult] = []
        
        # Agent weights for weighted voting
        self.agent_weights: Dict[str, float] = {}
        
        # Delegation mapping
        self.delegations: Dict[str, str] = {}  # voter_id -> delegate_id
        
        logger.info(f"Consensus System initialized with {default_vote_type.value} voting")
    
    async def create_proposal(
        self,
        proposal_id: str,
        title: str,
        description: str,
        proposer: str,
        deadline: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Proposal:
        """Create a new proposal for voting
        
        Args:
            proposal_id: Unique proposal ID
            title: Proposal title
            description: Detailed description
            proposer: ID of proposing agent
            deadline: Optional voting deadline
            metadata: Additional metadata
            
        Returns:
            Created proposal
        """
        proposal = Proposal(
            id=proposal_id,
            title=title,
            description=description,
            proposer=proposer,
            deadline=deadline,
            metadata=metadata or {}
        )
        
        self.active_proposals[proposal_id] = proposal
        self.current_votes[proposal_id] = []
        
        logger.info(f"Created proposal: {title} (ID: {proposal_id})")
        return proposal
    
    async def cast_vote(
        self,
        proposal_id: str,
        voter_id: str,
        option: VoteOption,
        reason: Optional[str] = None
    ) -> Vote:
        """Cast a vote on a proposal
        
        Args:
            proposal_id: Proposal to vote on
            voter_id: ID of voting agent
            option: Vote option
            reason: Optional reason for vote
            
        Returns:
            Cast vote
        """
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        # Check for delegation
        actual_voter = self.delegations.get(voter_id, voter_id)
        
        # Get weight for weighted voting
        weight = self.agent_weights.get(actual_voter, 1.0)
        
        # Create vote
        vote = Vote(
            voter_id=actual_voter,
            option=option,
            weight=weight,
            reason=reason
        )
        
        # Remove any previous vote by this voter
        self.current_votes[proposal_id] = [
            v for v in self.current_votes[proposal_id]
            if v.voter_id != actual_voter
        ]
        
        # Add new vote
        self.current_votes[proposal_id].append(vote)
        
        logger.info(f"Vote cast: {actual_voter} -> {option.value} on {proposal_id}")
        return vote
    
    async def tally_votes(
        self,
        proposal_id: str,
        vote_type: Optional[VoteType] = None,
        required_voters: Optional[List[str]] = None
    ) -> VotingResult:
        """Tally votes and determine result
        
        Args:
            proposal_id: Proposal to tally
            vote_type: Voting mechanism to use
            required_voters: List of required voters
            
        Returns:
            Voting result
        """
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        votes = self.current_votes.get(proposal_id, [])
        vote_type = vote_type or self.default_vote_type
        
        # Calculate participation
        if required_voters:
            participation_rate = len(votes) / len(required_voters)
        else:
            participation_rate = 1.0  # Assume all interested parties voted
        
        # Tally based on vote type
        if vote_type == VoteType.SIMPLE_MAJORITY:
            result = self._tally_simple_majority(votes)
        elif vote_type == VoteType.SUPER_MAJORITY:
            result = self._tally_super_majority(votes)
        elif vote_type == VoteType.UNANIMOUS:
            result = self._tally_unanimous(votes)
        elif vote_type == VoteType.WEIGHTED:
            result = self._tally_weighted(votes)
        elif vote_type == VoteType.CONSENSUS:
            result = self._tally_consensus(votes)
        else:
            result = self._tally_simple_majority(votes)
        
        # Create voting result
        voting_result = VotingResult(
            proposal_id=proposal_id,
            votes=votes,
            vote_type=vote_type,
            result=result['decision'],
            consensus_reached=result['consensus'],
            approval_rate=result['approval_rate'],
            participation_rate=participation_rate
        )
        
        # Archive result
        self.voting_history.append(voting_result)
        
        # Clean up if consensus reached
        if result['consensus']:
            del self.active_proposals[proposal_id]
            del self.current_votes[proposal_id]
        
        logger.info(f"Voting result: {result['decision'].value} "
                   f"(approval: {result['approval_rate']:.2%})")
        
        return voting_result
    
    def _tally_simple_majority(self, votes: List[Vote]) -> Dict[str, Any]:
        """Tally using simple majority (>50%)"""
        if not votes:
            return {
                'decision': VoteOption.ABSTAIN,
                'consensus': False,
                'approval_rate': 0.0
            }
        
        approve_count = sum(1 for v in votes if v.option == VoteOption.APPROVE)
        total_count = len([v for v in votes if v.option != VoteOption.ABSTAIN])
        
        if total_count == 0:
            return {
                'decision': VoteOption.ABSTAIN,
                'consensus': False,
                'approval_rate': 0.0
            }
        
        approval_rate = approve_count / total_count
        
        return {
            'decision': VoteOption.APPROVE if approval_rate > 0.5 else VoteOption.REJECT,
            'consensus': approval_rate > 0.5,
            'approval_rate': approval_rate
        }
    
    def _tally_super_majority(self, votes: List[Vote]) -> Dict[str, Any]:
        """Tally using super majority (>66%)"""
        if not votes:
            return {
                'decision': VoteOption.ABSTAIN,
                'consensus': False,
                'approval_rate': 0.0
            }
        
        approve_count = sum(1 for v in votes if v.option == VoteOption.APPROVE)
        total_count = len([v for v in votes if v.option != VoteOption.ABSTAIN])
        
        if total_count == 0:
            return {
                'decision': VoteOption.ABSTAIN,
                'consensus': False,
                'approval_rate': 0.0
            }
        
        approval_rate = approve_count / total_count
        
        return {
            'decision': VoteOption.APPROVE if approval_rate > 0.66 else VoteOption.REJECT,
            'consensus': approval_rate > 0.66,
            'approval_rate': approval_rate
        }
    
    def _tally_unanimous(self, votes: List[Vote]) -> Dict[str, Any]:
        """Tally requiring unanimous approval"""
        if not votes:
            return {
                'decision': VoteOption.ABSTAIN,
                'consensus': False,
                'approval_rate': 0.0
            }
        
        non_abstain = [v for v in votes if v.option != VoteOption.ABSTAIN]
        if not non_abstain:
            return {
                'decision': VoteOption.ABSTAIN,
                'consensus': False,
                'approval_rate': 0.0
            }
        
        all_approve = all(v.option == VoteOption.APPROVE for v in non_abstain)
        
        return {
            'decision': VoteOption.APPROVE if all_approve else VoteOption.REJECT,
            'consensus': all_approve,
            'approval_rate': 1.0 if all_approve else 0.0
        }
    
    def _tally_weighted(self, votes: List[Vote]) -> Dict[str, Any]:
        """Tally using weighted voting"""
        if not votes:
            return {
                'decision': VoteOption.ABSTAIN,
                'consensus': False,
                'approval_rate': 0.0
            }
        
        approve_weight = sum(v.weight for v in votes if v.option == VoteOption.APPROVE)
        reject_weight = sum(v.weight for v in votes if v.option == VoteOption.REJECT)
        total_weight = approve_weight + reject_weight
        
        if total_weight == 0:
            return {
                'decision': VoteOption.ABSTAIN,
                'consensus': False,
                'approval_rate': 0.0
            }
        
        approval_rate = approve_weight / total_weight
        
        return {
            'decision': VoteOption.APPROVE if approval_rate > 0.5 else VoteOption.REJECT,
            'consensus': approval_rate > 0.5,
            'approval_rate': approval_rate
        }
    
    def _tally_consensus(self, votes: List[Vote]) -> Dict[str, Any]:
        """Tally aiming for consensus (no strong objections)"""
        if not votes:
            return {
                'decision': VoteOption.ABSTAIN,
                'consensus': False,
                'approval_rate': 0.0
            }
        
        approve_count = sum(1 for v in votes if v.option == VoteOption.APPROVE)
        reject_count = sum(1 for v in votes if v.option == VoteOption.REJECT)
        abstain_count = sum(1 for v in votes if v.option == VoteOption.ABSTAIN)
        
        total = len(votes)
        
        # Consensus reached if no more than 10% strongly object
        strong_objection_threshold = 0.1
        
        if reject_count / total <= strong_objection_threshold:
            decision = VoteOption.APPROVE
            consensus = True
        else:
            decision = VoteOption.REJECT
            consensus = False
        
        approval_rate = approve_count / total if total > 0 else 0.0
        
        return {
            'decision': decision,
            'consensus': consensus,
            'approval_rate': approval_rate
        }
    
    def set_agent_weight(self, agent_id: str, weight: float):
        """Set voting weight for an agent
        
        Args:
            agent_id: Agent ID
            weight: Voting weight (typically 0.0 to 1.0)
        """
        self.agent_weights[agent_id] = weight
        logger.info(f"Set weight for {agent_id}: {weight}")
    
    def delegate_vote(self, voter_id: str, delegate_id: str):
        """Delegate voting power to another agent
        
        Args:
            voter_id: Agent delegating vote
            delegate_id: Agent receiving delegation
        """
        self.delegations[voter_id] = delegate_id
        logger.info(f"{voter_id} delegated vote to {delegate_id}")
    
    def revoke_delegation(self, voter_id: str):
        """Revoke vote delegation
        
        Args:
            voter_id: Agent revoking delegation
        """
        if voter_id in self.delegations:
            del self.delegations[voter_id]
            logger.info(f"{voter_id} revoked delegation")
    
    def get_proposal_status(self, proposal_id: str) -> Dict[str, Any]:
        """Get current status of a proposal
        
        Args:
            proposal_id: Proposal ID
            
        Returns:
            Proposal status
        """
        if proposal_id not in self.active_proposals:
            return {'status': 'not_found'}
        
        proposal = self.active_proposals[proposal_id]
        votes = self.current_votes.get(proposal_id, [])
        
        return {
            'proposal': proposal.to_dict(),
            'votes_cast': len(votes),
            'current_votes': [v.to_dict() for v in votes],
            'status': 'active'
        }
    
    def get_voting_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent voting history
        
        Args:
            limit: Number of recent results to return
            
        Returns:
            List of voting results
        """
        return [r.to_dict() for r in self.voting_history[-limit:]]