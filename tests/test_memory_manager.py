"""Tests for ConversationMemoryManager"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, patch

from app.services.memory_manager import (
    ConversationMemoryManager,
    ImportanceLevel,
    MemoryEntry
)
from app.models.schemas import Message, MessageRole


class TestMemoryManager:
    """Test suite for ConversationMemoryManager"""
    
    @pytest.fixture
    def memory_manager(self):
        """Create a memory manager instance for testing"""
        mock_rag = Mock()
        mock_rag.embedding_manager = Mock()
        mock_rag.embedding_manager.embed_text = Mock(return_value=[0.1] * 384)
        mock_rag.retriever = Mock()
        mock_rag.retriever.add_documents_with_index = Mock()
        
        return ConversationMemoryManager(
            rag_chain=mock_rag,
            auto_save=True,
            importance_threshold=ImportanceLevel.MEDIUM
        )
    
    @pytest.mark.unit
    def test_importance_analysis_keywords(self, memory_manager):
        """Test importance analysis based on keywords"""
        # Test critical keywords
        message = Message(role=MessageRole.USER, content="이것은 매우 중요한 정보입니다. 꼭 기억해주세요.")
        importance, tags = memory_manager.analyze_importance(message)
        assert importance == ImportanceLevel.HIGH or importance == ImportanceLevel.CRITICAL
        assert "critical" in tags
        
        # Test task keywords
        message = Message(role=MessageRole.USER, content="오늘 해야 할 일: 보고서 작성")
        importance, tags = memory_manager.analyze_importance(message)
        assert importance.value >= ImportanceLevel.MEDIUM.value
        assert "task" in tags
        
        # Test low importance
        message = Message(role=MessageRole.USER, content="안녕하세요")
        importance, tags = memory_manager.analyze_importance(message)
        assert importance == ImportanceLevel.LOW
    
    @pytest.mark.unit
    def test_importance_analysis_questions(self, memory_manager):
        """Test that questions are marked as important"""
        message = Message(role=MessageRole.USER, content="RAG 시스템은 어떻게 작동하나요?")
        importance, tags = memory_manager.analyze_importance(message)
        assert importance.value >= ImportanceLevel.MEDIUM.value
        assert "question" in tags
    
    @pytest.mark.unit
    def test_extract_key_information(self, memory_manager):
        """Test information extraction from messages"""
        message = Message(
            role=MessageRole.USER, 
            content='GPT-OSS는 "OpenAI의 오픈 소스 모델"입니다. URL: https://example.com'
        )
        
        extracted = memory_manager.extract_key_information(message)
        
        assert extracted["raw_content"] == message.content
        assert "quotes" in extracted
        assert "OpenAI의 오픈 소스 모델" in extracted["quotes"]
        # URL detection through regex
        assert "https://example.com" in message.content
    
    @pytest.mark.unit
    def test_memory_buffer_overflow_prevention(self, memory_manager):
        """Test that memory buffer doesn't grow indefinitely"""
        # Set smaller limits for testing
        memory_manager.MAX_BUFFER_SIZE = 10
        memory_manager.BUFFER_CLEANUP_THRESHOLD = 8
        
        # Add many entries
        for i in range(20):
            entry = MemoryEntry(
                content=f"Memory {i}",
                importance=ImportanceLevel.LOW if i < 10 else ImportanceLevel.HIGH,
                context="",
                tags=["test"],
                timestamp=datetime.now() - timedelta(hours=i),
                session_id=uuid4()
            )
            memory_manager.memory_buffer.append(entry)
        
        # Simulate adding one more entry that triggers cleanup
        message = Message(role=MessageRole.USER, content="Important message")
        memory_manager.analyze_importance(message)
        
        # Check buffer was cleaned
        assert len(memory_manager.memory_buffer) <= memory_manager.MAX_BUFFER_SIZE
    
    @pytest.mark.asyncio
    async def test_save_to_memory_threshold(self, memory_manager):
        """Test that only messages above threshold are saved"""
        # Low importance message
        low_msg = Message(role=MessageRole.USER, content="Hi")
        saved = await memory_manager.save_to_memory(low_msg)
        assert saved is False  # Below MEDIUM threshold
        
        # High importance message
        high_msg = Message(role=MessageRole.USER, content="중요: 내일 회의가 있습니다. 꼭 참석해주세요!")
        saved = await memory_manager.save_to_memory(high_msg)
        assert saved is True  # Above MEDIUM threshold
    
    @pytest.mark.asyncio
    async def test_save_to_memory_with_force(self, memory_manager):
        """Test forcing save regardless of importance"""
        low_msg = Message(role=MessageRole.USER, content="Simple message")
        saved = await memory_manager.save_to_memory(low_msg, force=True)
        assert saved is True  # Forced save
    
    @pytest.mark.asyncio
    async def test_cleanup_old_memories(self, memory_manager):
        """Test cleaning up old memories"""
        # Add old and new entries
        old_entry = MemoryEntry(
            content="Old memory",
            importance=ImportanceLevel.LOW,
            context="",
            tags=["old"],
            timestamp=datetime.now() - timedelta(days=40),
            session_id=uuid4()
        )
        
        new_entry = MemoryEntry(
            content="New memory",
            importance=ImportanceLevel.HIGH,
            context="",
            tags=["new"],
            timestamp=datetime.now(),
            session_id=uuid4()
        )
        
        memory_manager.memory_buffer = [old_entry, new_entry]
        
        # Clean memories older than 30 days
        removed = await memory_manager.cleanup_old_memories(days=30)
        
        assert removed == 1
        assert len(memory_manager.memory_buffer) == 1
        assert memory_manager.memory_buffer[0].content == "New memory"
    
    @pytest.mark.unit
    def test_build_context_string(self, memory_manager):
        """Test context string building"""
        messages = [
            Message(role=MessageRole.USER, content="First question"),
            Message(role=MessageRole.ASSISTANT, content="First answer"),
            Message(role=MessageRole.USER, content="Second question"),
        ]
        
        context = memory_manager._build_context_string(messages, max_messages=2)
        
        assert "Assistant: First answer" in context
        assert "User: Second question" in context
        assert "First question" not in context  # Only last 2 messages
    
    @pytest.mark.asyncio
    async def test_summarize_session(self, memory_manager):
        """Test session summarization"""
        messages = [
            Message(role=MessageRole.USER, content="What is RAG?"),
            Message(role=MessageRole.ASSISTANT, content="RAG is Retrieval-Augmented Generation"),
            Message(role=MessageRole.USER, content="TODO: Implement caching"),
            Message(role=MessageRole.ASSISTANT, content="I'll help with that"),
        ]
        
        summary = await memory_manager.summarize_session(messages, session_id=uuid4())
        
        assert "Session Summary" in summary
        assert "Total Messages: 4" in summary
        assert "Key Questions (1):" in summary
        assert "Identified Tasks (1):" in summary
    
    @pytest.mark.unit
    def test_get_memory_stats(self, memory_manager):
        """Test memory statistics"""
        # Add some entries
        for i in range(5):
            memory_manager.memory_buffer.append(
                MemoryEntry(
                    content=f"Memory {i}",
                    importance=ImportanceLevel.MEDIUM,
                    context="",
                    tags=["test", "unit"],
                    timestamp=datetime.now(),
                    session_id=None
                )
            )
        
        stats = memory_manager.get_memory_stats()
        
        assert stats["buffer_size"] == 5
        assert stats["buffer_max_size"] == memory_manager.MAX_BUFFER_SIZE
        assert stats["buffer_usage_percent"] == (5 / memory_manager.MAX_BUFFER_SIZE) * 100
        assert stats["auto_save"] is True
        assert stats["importance_threshold"] == "MEDIUM"