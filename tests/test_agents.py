"""Test agents functionality"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.agents.base_agent import BaseAgent, AgentRole, AgentStatus
from app.agents.specialized_agents import ResearchAgent, CodeAgent, AnalysisAgent
from app.agents.agent_manager import AgentManager
from app.core.ollama import OllamaClient
from app.models.schemas import Message, MessageRole


@pytest.fixture
def mock_ollama():
    """Create mock Ollama client"""
    mock = Mock(spec=OllamaClient)
    mock.chat = AsyncMock()
    return mock


@pytest.fixture
def agent_manager(mock_ollama):
    """Create agent manager with mock Ollama"""
    return AgentManager(ollama_client=mock_ollama)


class TestBaseAgent:
    """Test base agent functionality"""
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        class TestAgent(BaseAgent):
            async def process_task(self, task, context=None):
                return {'status': 'success'}
        
        agent = TestAgent(
            name="Test Agent",
            role=AgentRole.GENERAL,
            description="Test agent",
            capabilities=["testing"]
        )
        
        assert agent.name == "Test Agent"
        assert agent.role == AgentRole.GENERAL
        assert agent.status == AgentStatus.IDLE
        assert "testing" in agent.capabilities
    
    def test_can_handle_task(self):
        """Test task capability matching"""
        class TestAgent(BaseAgent):
            async def process_task(self, task, context=None):
                return {'status': 'success'}
        
        agent = TestAgent(
            name="Research Agent",
            role=AgentRole.RESEARCHER,
            description="Research agent",
            capabilities=["research", "analysis"]
        )
        
        assert agent.can_handle("Please research this topic")
        assert agent.can_handle("Do some analysis")
        assert not agent.can_handle("Write some code")


class TestSpecializedAgents:
    """Test specialized agents"""
    
    @pytest.mark.asyncio
    async def test_research_agent(self, mock_ollama):
        """Test research agent"""
        mock_ollama.chat.return_value = AsyncMock()
        mock_ollama.chat.return_value.__aiter__.return_value = ["Research findings"]
        
        agent = ResearchAgent(mock_ollama)
        result = await agent.process_task("Research AI trends")
        
        assert result['status'] == 'success'
        assert result['agent'] == "Research Assistant"
        assert 'findings' in result
    
    @pytest.mark.asyncio
    async def test_code_agent(self, mock_ollama):
        """Test code agent"""
        mock_ollama.chat.return_value = AsyncMock()
        mock_ollama.chat.return_value.__aiter__.return_value = ["```python\nprint('Hello')\n```"]
        
        agent = CodeAgent(mock_ollama)
        result = await agent.process_task("Write a hello world program")
        
        assert result['status'] == 'success'
        assert result['agent'] == "Code Assistant"
        assert 'response' in result
        assert 'code_blocks' in result
    
    @pytest.mark.asyncio
    async def test_analysis_agent(self, mock_ollama):
        """Test analysis agent"""
        mock_ollama.chat.return_value = AsyncMock()
        mock_ollama.chat.return_value.__aiter__.return_value = ["Analysis results"]
        
        agent = AnalysisAgent(mock_ollama)
        result = await agent.process_task("Analyze this data")
        
        assert result['status'] == 'success'
        assert result['agent'] == "Analysis Assistant"
        assert 'analysis' in result


class TestAgentManager:
    """Test agent manager"""
    
    def test_initialization(self, agent_manager):
        """Test agent manager initialization"""
        assert len(agent_manager.agents) == 5
        assert 'researcher' in agent_manager.agents
        assert 'coder' in agent_manager.agents
        assert 'analyst' in agent_manager.agents
        assert 'summarizer' in agent_manager.agents
        assert 'creative' in agent_manager.agents
    
    def test_list_agents(self, agent_manager):
        """Test listing agents"""
        agents = agent_manager.list_agents()
        assert len(agents) == 5
        
        for agent in agents:
            assert 'id' in agent
            assert 'name' in agent
            assert 'role' in agent
            assert 'status' in agent
            assert 'capabilities' in agent
    
    def test_find_best_agent(self, agent_manager):
        """Test finding best agent for task"""
        # Research task
        agent = agent_manager.find_best_agent("Please research AI trends")
        assert agent is not None
        assert agent.role == AgentRole.RESEARCHER
        
        # Coding task
        agent = agent_manager.find_best_agent("Write a Python function")
        assert agent is not None
        assert agent.role == AgentRole.CODER
        
        # Analysis task
        agent = agent_manager.find_best_agent("Analyze this data")
        assert agent is not None
        assert agent.role == AgentRole.ANALYST
    
    @pytest.mark.asyncio
    async def test_assign_task(self, agent_manager, mock_ollama):
        """Test assigning task to agent"""
        mock_ollama.chat.return_value = AsyncMock()
        mock_ollama.chat.return_value.__aiter__.return_value = ["Task completed"]
        
        result = await agent_manager.assign_task(
            task="Research machine learning",
            agent_id="researcher"
        )
        
        assert result['status'] == 'success'
        assert 'agent' in result
    
    @pytest.mark.asyncio
    async def test_assign_task_auto_select(self, agent_manager, mock_ollama):
        """Test auto-selecting agent for task"""
        mock_ollama.chat.return_value = AsyncMock()
        mock_ollama.chat.return_value.__aiter__.return_value = ["Code generated"]
        
        result = await agent_manager.assign_task(
            task="Write a Python function to sort a list"
        )
        
        assert result['status'] == 'success'
        assert result['agent'] == "Code Assistant"
    
    def test_get_status(self, agent_manager):
        """Test getting agent status"""
        status = agent_manager.get_all_status()
        
        assert 'agents' in status
        assert 'queue_length' in status
        assert 'completed_tasks' in status
        assert 'active_agents' in status
        
        assert len(status['agents']) == 5
        assert status['queue_length'] == 0
        assert status['active_agents'] == 0


@pytest.mark.asyncio
async def test_chat_service_with_agents():
    """Test chat service with agents integration"""
    from app.services.chat import ChatService
    from app.rag import RAGChain
    
    with patch('app.services.chat.OllamaClient') as mock_ollama_class:
        mock_ollama = Mock()
        mock_ollama.chat = AsyncMock()
        mock_ollama.chat.return_value.__aiter__.return_value = ["Agent response"]
        mock_ollama_class.return_value = mock_ollama
        
        # Create chat service with agents
        chat_service = ChatService(
            enable_memory=False,
            rag_chain=None,
            enable_agents=True
        )
        
        assert chat_service.agent_manager is not None
        
        # Test chat with agent
        result = await chat_service.chat_with_agent(
            message="Research AI trends",
            agent_id="researcher"
        )
        
        assert result['status'] == 'success'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])