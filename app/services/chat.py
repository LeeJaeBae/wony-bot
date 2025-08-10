"""Chat service for handling conversations"""

from typing import Optional, AsyncGenerator, Dict, Any
from uuid import UUID
import time
from app.core.ollama import OllamaClient
from app.core.session import SessionManager
from app.models.schemas import Message, MessageRole, ChatRequest, ChatResponse
from app.config import settings
from app.services.memory_manager import ConversationMemoryManager, ImportanceLevel
from app.services.enhanced_memory import EnhancedMemoryManager
from app.rag import RAGChain
from app.agents.agent_manager import AgentManager
from app.agents.hierarchical_manager import HierarchicalAgentManager
import logging
import json

logger = logging.getLogger(__name__)

class ChatService:
    """Service for managing chat interactions"""
    
    def __init__(self, enable_memory: bool = True, rag_chain: Optional[RAGChain] = None, 
                 enable_agents: bool = True, use_hierarchical: bool = True):
        self.ollama = OllamaClient()
        self.session_manager = SessionManager()
        self.enable_memory = enable_memory
        self.enable_agents = enable_agents
        self.use_hierarchical = use_hierarchical
        
        # Initialize enhanced memory manager if enabled
        if enable_memory:
            self.memory_manager = EnhancedMemoryManager(
                rag_chain=rag_chain,
                auto_save=True,
                importance_threshold=ImportanceLevel.LOW  # 더 많은 대화 저장
            )
            logger.info("Enhanced Memory Manager enabled for chat service")
        else:
            self.memory_manager = None
        
        # Initialize agent system
        if enable_agents:
            if use_hierarchical:
                # Use hierarchical agent system by default
                self.hierarchical_manager = HierarchicalAgentManager(ollama_client=self.ollama)
                self.agent_manager = self.hierarchical_manager.agent_manager
                logger.info("Hierarchical Agent System enabled with Orchestrator and consensus")
            else:
                # Use basic agent manager
                self.agent_manager = AgentManager(ollama_client=self.ollama)
                self.hierarchical_manager = None
                logger.info("Basic Agent Manager enabled with specialized agents")
        else:
            self.agent_manager = None
            self.hierarchical_manager = None
        
    async def chat(
        self,
        message: str,
        session_id: Optional[UUID] = None,
        system_prompt: Optional[str] = None,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Process a chat message and return response"""
        
        # Create or get session
        if session_id:
            session = await self.session_manager.get_session(session_id)
            if not session:
                # Create new session if ID doesn't exist
                session = await self.session_manager.create_session()
                session_id = session.id
        else:
            session = await self.session_manager.create_session()
            session_id = session.id
        
        # Add user message to history
        await self.session_manager.add_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=message
        )
        
        # Get recent messages for context
        messages = await self.session_manager.get_recent_messages(
            session_id=session_id,
            limit=settings.max_history_length
        )
        
        # Add memory context to system prompt
        memory_context = ""
        if self.memory_manager and hasattr(self.memory_manager, 'get_relevant_memories'):
            try:
                # 관련 메모리 검색
                relevant_memories = await self.memory_manager.get_relevant_memories(
                    query=message,
                    session_id=session_id,
                    top_k=3
                )
                if relevant_memories:
                    memory_context = "\n\n💾 이전 대화 기억:\n"
                    for mem in relevant_memories[:3]:
                        content = mem.get('content', '')[:200]
                        memory_context += f"- {content}\n"
                    logger.info(f"Added {len(relevant_memories)} memories to context")
            except Exception as e:
                logger.error(f"Failed to get memories: {e}")
        
        # Add system prompt with memory context (even when no explicit system_prompt)
        if system_prompt or memory_context:
            prompt_with_memory = system_prompt or ""
            if memory_context:
                prompt_with_memory += memory_context
            if prompt_with_memory:
                messages.insert(0, Message(
                    role=MessageRole.SYSTEM,
                    content=prompt_with_memory
                ))
        
        # Generate response
        response_text = ""
        start_time = time.time()
        
        async for chunk in self.ollama.chat(messages, stream=stream):
            response_text += chunk
            if stream:
                yield chunk
        
        # Calculate duration and approximate tokens
        duration_ms = int((time.time() - start_time) * 1000)
        tokens_used = len(response_text.split()) * 1.3  # Rough approximation
        
        # Save assistant response
        assistant_message = await self.session_manager.add_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=response_text,
            tokens_used=int(tokens_used),
            meta_data={"duration_ms": duration_ms}
        )
        
        # Auto-save to memory if enabled
        if self.memory_manager:
            try:
                # Save user message if important
                user_msg = Message(role=MessageRole.USER, content=message)
                await self.memory_manager.save_to_memory(
                    message=user_msg,
                    context=messages[:-1] if len(messages) > 1 else None,
                    session_id=session_id
                )
                
                # Save assistant response if important
                assistant_msg = Message(role=MessageRole.ASSISTANT, content=response_text)
                await self.memory_manager.save_to_memory(
                    message=assistant_msg,
                    context=messages,
                    session_id=session_id
                )
                
                logger.debug("Checked messages for memory storage")
            except Exception as e:
                logger.error(f"Failed to save to memory: {e}")
        
        if not stream:
            # Return response once
            yield response_text
            # Then return session id marker
            yield f"\n[session:{session_id}]"
        else:
            # In streaming mode, only emit session id once at the end
            yield f"\n[session:{session_id}]"
    
    async def get_session_history(self, session_id: UUID) -> Optional[list]:
        """Get the chat history for a session"""
        session = await self.session_manager.get_session(session_id)
        if session:
            return session.messages
        return None
    
    async def list_sessions(self, limit: int = 10):
        """List recent chat sessions"""
        return await self.session_manager.list_sessions(limit=limit)
    
    async def clear_session(self, session_id: UUID) -> bool:
        """Clear messages from a session"""
        return await self.session_manager.clear_messages(session_id)
    
    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session completely"""
        return await self.session_manager.delete_session(session_id)
    
    async def chat_with_agent(
        self,
        message: str,
        agent_id: Optional[str] = None,
        session_id: Optional[UUID] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a chat message using an agent
        
        Args:
            message: User message/task
            agent_id: Optional specific agent ID
            session_id: Optional session ID
            context: Optional context for the agent
            
        Returns:
            Agent response
        """
        if not self.agent_manager:
            return {
                'status': 'error',
                'error': 'Agent system not enabled'
            }
        
        # Find or assign agent
        result = await self.agent_manager.assign_task(
            task=message,
            agent_id=agent_id,
            context=context
        )
        
        # Save to session if provided
        if session_id and result.get('status') == 'success':
            await self.session_manager.add_message(
                session_id=session_id,
                role=MessageRole.USER,
                content=message
            )
            
            # Format agent response
            agent_response = f"[Agent: {result.get('agent', 'Unknown')}]\n"
            if 'findings' in result:
                agent_response += result['findings']
            elif 'response' in result:
                agent_response += result['response']
            elif 'analysis' in result:
                agent_response += result['analysis']
            elif 'summary' in result:
                agent_response += result['summary']
            elif 'creative_output' in result:
                agent_response += result['creative_output']
            else:
                agent_response += json.dumps(result, indent=2)
            
            await self.session_manager.add_message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=agent_response
            )
        
        return result
    
    def get_agent_manager(self) -> Optional[AgentManager]:
        """Get the agent manager instance"""
        return self.agent_manager
    
    def get_hierarchical_manager(self) -> Optional[HierarchicalAgentManager]:
        """Get the hierarchical manager instance"""
        return self.hierarchical_manager
    
    async def chat_with_hierarchical_agents(
        self,
        message: str,
        session_id: Optional[UUID] = None,
        require_consensus: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a complex task using hierarchical agent system
        
        Args:
            message: User message/task
            session_id: Optional session ID
            require_consensus: Whether to require consensus
            context: Optional context
            
        Returns:
            Hierarchical processing result
        """
        if not self.hierarchical_manager:
            return {
                'status': 'error',
                'error': 'Hierarchical agent system not enabled'
            }
        
        # Process with hierarchical system
        result = await self.hierarchical_manager.process_complex_task(
            task=message,
            context=context,
            require_consensus=require_consensus
        )
        
        # Save to session if provided
        if session_id and result.get('status') in ['success', 'partial']:
            await self.session_manager.add_message(
                session_id=session_id,
                role=MessageRole.USER,
                content=message
            )
            
            # Format hierarchical response
            response = f"[Hierarchical Agent System]\n"
            response += f"📋 Task: {message}\n"
            response += f"✅ Status: {result['status']}\n"
            
            if 'consensus' in result and result['consensus'].get('reached'):
                response += f"🗳️ Consensus: {result['consensus']['approval_rate']:.1%} approval\n"
            
            if 'orchestration' in result:
                response += f"\n📊 Results:\n"
                orchestration = result['orchestration']
                if 'execution' in orchestration:
                    for task_id, exec_result in orchestration['execution'].items():
                        if exec_result.get('status') == 'success':
                            response += f"  • {task_id}: ✅ Completed\n"
                        else:
                            response += f"  • {task_id}: ❌ Failed\n"
            
            await self.session_manager.add_message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=response
            )
        
        return result