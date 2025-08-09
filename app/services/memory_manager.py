"""Conversation Memory Manager - Auto-saves important information to RAG"""

from typing import List, Dict, Any, Optional, Tuple
import re
import asyncio
from datetime import datetime
from uuid import UUID
import logging
from dataclasses import dataclass
from enum import Enum

from app.models.schemas import Message, MessageRole
from app.rag import RAGChain, VectorStore, EmbeddingManager, DocumentLoader
from app.rag.document_loader import Document

logger = logging.getLogger(__name__)

class ImportanceLevel(Enum):
    """Importance levels for messages"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class MemoryEntry:
    """A memory entry to be saved"""
    content: str
    importance: ImportanceLevel
    context: str
    tags: List[str]
    timestamp: datetime
    session_id: Optional[UUID] = None
    
class ConversationMemoryManager:
    """Manages automatic saving of important conversation content to RAG"""
    
    # Memory management constants
    MAX_BUFFER_SIZE = 1000  # Maximum entries in memory buffer
    BUFFER_CLEANUP_THRESHOLD = 800  # Start cleanup when reaching this size
    
    # Keywords indicating important information
    IMPORTANCE_KEYWORDS = {
        "critical": ["중요", "꼭", "반드시", "필수", "critical", "important", "must", "essential"],
        "task": ["해야", "할 일", "작업", "task", "todo", "need to", "should"],
        "definition": ["는", "이란", "정의", "is", "means", "definition"],
        "instruction": ["하세요", "해주세요", "하는 방법", "how to", "please", "instructions"],
        "personal": ["내", "나의", "저의", "my", "personal", "custom"],
        "reminder": ["기억", "잊지", "remember", "don't forget", "remind"],
        "fact": ["사실", "정보", "fact", "information", "data"],
        "preference": ["좋아", "싫어", "선호", "prefer", "like", "dislike"]
    }
    
    # Patterns for extracting structured information
    EXTRACTION_PATTERNS = {
        "definition": r"(.+?)(?:는|이란|is|means)\s+(.+)",
        "task": r"(?:해야 할 일|todo|task):\s*(.+)",
        "instruction": r"(?:방법|how to):\s*(.+)",
        "fact": r"(?:사실|fact):\s*(.+)"
    }
    
    def __init__(
        self,
        rag_chain: Optional[RAGChain] = None,
        auto_save: bool = True,
        importance_threshold: ImportanceLevel = ImportanceLevel.MEDIUM
    ):
        """Initialize memory manager"""
        self.rag_chain = rag_chain
        self.auto_save = auto_save
        self.importance_threshold = importance_threshold
        self.memory_buffer = []
        self.collection_name = "chat_history"
        
        logger.info(f"Memory Manager initialized (auto_save={auto_save}, threshold={importance_threshold.name})")
    
    def analyze_importance(self, message: Message, context: List[Message] = None) -> Tuple[ImportanceLevel, List[str]]:
        """Analyze the importance of a message"""
        
        content = message.content.lower()
        tags = []
        importance_score = 0
        
        # Check for importance keywords
        for category, keywords in self.IMPORTANCE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content:
                    importance_score += 1
                    tags.append(category)
                    break
        
        # Check for questions (usually important)
        if "?" in message.content or "？" in message.content:
            importance_score += 1
            tags.append("question")
        
        # Check for commands or instructions
        if any(cmd in content for cmd in ["!", "해주", "please", "하세요"]):
            importance_score += 1
            tags.append("command")
        
        # Check message length (longer messages often more important)
        if len(message.content) > 200:
            importance_score += 1
            tags.append("detailed")
        
        # Check for numbers, dates, URLs (factual information)
        if re.search(r'\d+|http[s]?://|www\.', message.content):
            importance_score += 1
            tags.append("factual")
        
        # Determine importance level
        if importance_score >= 4:
            importance = ImportanceLevel.CRITICAL
        elif importance_score >= 3:
            importance = ImportanceLevel.HIGH
        elif importance_score >= 2:
            importance = ImportanceLevel.MEDIUM
        else:
            importance = ImportanceLevel.LOW
        
        # Context analysis (if previous messages provided)
        if context and len(context) > 1:
            # Check if this is a response to a question
            for prev_msg in context[-3:]:
                if prev_msg.role == MessageRole.USER and "?" in prev_msg.content:
                    importance = max(importance, ImportanceLevel.HIGH)
                    tags.append("answer")
                    break
        
        return importance, list(set(tags))
    
    def extract_key_information(self, message: Message) -> Dict[str, Any]:
        """Extract structured information from message"""
        
        extracted = {
            "raw_content": message.content,
            "structured_info": {}
        }
        
        # Try to extract structured patterns
        for pattern_name, pattern in self.EXTRACTION_PATTERNS.items():
            match = re.search(pattern, message.content, re.IGNORECASE | re.DOTALL)
            if match:
                extracted["structured_info"][pattern_name] = match.groups()
        
        # Extract entities (simple approach - can be enhanced with NER)
        # Extract quoted text
        quotes = re.findall(r'"([^"]*)"', message.content)
        if quotes:
            extracted["quotes"] = quotes
        
        # Extract code blocks
        code_blocks = re.findall(r'```(.*?)```', message.content, re.DOTALL)
        if code_blocks:
            extracted["code"] = code_blocks
        
        # Extract lists
        list_items = re.findall(r'^\s*[-*]\s+(.+)$', message.content, re.MULTILINE)
        if list_items:
            extracted["list_items"] = list_items
        
        return extracted
    
    async def save_to_memory(
        self,
        message: Message,
        context: List[Message] = None,
        session_id: Optional[UUID] = None,
        force: bool = False
    ) -> bool:
        """Save important message to memory (RAG)"""
        
        # Analyze importance
        importance, tags = self.analyze_importance(message, context)
        
        # Check if should save
        if not force and importance.value < self.importance_threshold.value:
            logger.debug(f"Message importance ({importance.name}) below threshold, skipping")
            return False
        
        # Extract key information
        extracted = self.extract_key_information(message)
        
        # Create memory entry
        memory_entry = MemoryEntry(
            content=message.content,
            importance=importance,
            context=self._build_context_string(context) if context else "",
            tags=tags,
            timestamp=datetime.now(),
            session_id=session_id
        )
        
        # Add to buffer with size management
        self.memory_buffer.append(memory_entry)
        
        # Prevent memory overflow
        if len(self.memory_buffer) > self.MAX_BUFFER_SIZE:
            logger.warning(f"Memory buffer overflow, cleaning up old entries")
            # Keep only the most recent entries, prioritizing higher importance
            sorted_buffer = sorted(
                self.memory_buffer,
                key=lambda x: (x.importance.value, x.timestamp.timestamp() if hasattr(x.timestamp, 'timestamp') else 0),
                reverse=True
            )
            self.memory_buffer = sorted_buffer[:self.BUFFER_CLEANUP_THRESHOLD]
            logger.info(f"Cleaned memory buffer to {len(self.memory_buffer)} entries")
        
        # Save to RAG if auto-save enabled
        if self.auto_save and self.rag_chain:
            await self._save_to_rag(memory_entry, extracted)
            logger.info(f"Saved memory with importance {importance.name} and tags {tags}")
            return True
        
        return False
    
    async def _save_to_rag(self, entry: MemoryEntry, extracted: Dict[str, Any]):
        """Save memory entry to RAG system"""
        
        if not self.rag_chain:
            logger.warning("RAG chain not initialized, cannot save memory")
            return
        
        # Format content for RAG
        formatted_content = self._format_memory_content(entry, extracted)
        
        # Create document
        doc = Document(
            content=formatted_content,
            metadata={
                "type": "conversation_memory",
                "importance": entry.importance.name,
                "tags": ", ".join(entry.tags),
                "timestamp": entry.timestamp.isoformat(),
                "session_id": str(entry.session_id) if entry.session_id else None,
                "source": "chat_conversation"
            }
        )
        
        # Generate embedding
        embedding = self.rag_chain.embedding_manager.embed_text(doc.content)
        
        # Save to vector store
        self.rag_chain.retriever.add_documents_with_index(
            documents=[doc.content],
            embeddings=[embedding],
            metadatas=[doc.metadata],
            collection_name=self.collection_name
        )
    
    def _format_memory_content(self, entry: MemoryEntry, extracted: Dict[str, Any]) -> str:
        """Format memory content for storage"""
        
        parts = []
        
        # Add timestamp and importance
        parts.append(f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M')}] [{entry.importance.name}]")
        
        # Add tags
        if entry.tags:
            parts.append(f"Tags: {', '.join(entry.tags)}")
        
        # Add main content
        parts.append(f"\nContent: {entry.content}")
        
        # Add context if available
        if entry.context:
            parts.append(f"\nContext: {entry.context}")
        
        # Add structured information
        if extracted.get("structured_info"):
            parts.append("\nStructured Information:")
            for key, value in extracted["structured_info"].items():
                parts.append(f"  - {key}: {value}")
        
        # Add quotes if any
        if extracted.get("quotes"):
            parts.append("\nQuoted Text:")
            for quote in extracted["quotes"]:
                parts.append(f'  - "{quote}"')
        
        return "\n".join(parts)
    
    def _build_context_string(self, messages: List[Message], max_messages: int = 3) -> str:
        """Build context string from recent messages"""
        
        if not messages:
            return ""
        
        # Take last few messages for context
        recent = messages[-max_messages:] if len(messages) > max_messages else messages
        
        context_parts = []
        for msg in recent:
            role = msg.role.value.capitalize()
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            context_parts.append(f"{role}: {content}")
        
        return " | ".join(context_parts)
    
    async def process_conversation(
        self,
        messages: List[Message],
        session_id: Optional[UUID] = None
    ) -> int:
        """Process entire conversation and save important parts"""
        
        saved_count = 0
        
        for i, message in enumerate(messages):
            # Get context (previous messages)
            context = messages[:i] if i > 0 else None
            
            # Try to save
            saved = await self.save_to_memory(
                message=message,
                context=context,
                session_id=session_id
            )
            
            if saved:
                saved_count += 1
        
        logger.info(f"Processed conversation: saved {saved_count}/{len(messages)} messages")
        return saved_count
    
    async def summarize_session(
        self,
        messages: List[Message],
        session_id: Optional[UUID] = None
    ) -> str:
        """Create a summary of the conversation session"""
        
        if not messages:
            return "No messages to summarize"
        
        # Extract key points
        key_points = []
        questions = []
        answers = []
        tasks = []
        
        for i, msg in enumerate(messages):
            if msg.role == MessageRole.USER and "?" in msg.content:
                questions.append(msg.content)
            elif msg.role == MessageRole.ASSISTANT and i > 0 and messages[i-1].role == MessageRole.USER:
                if "?" in messages[i-1].content:
                    answers.append(msg.content[:200])
            
            # Extract tasks
            if any(keyword in msg.content.lower() for keyword in ["해야", "할 일", "todo", "task"]):
                tasks.append(msg.content[:100])
        
        # Build summary
        summary_parts = [
            f"Session Summary - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total Messages: {len(messages)}"
        ]
        
        if questions:
            summary_parts.append(f"\nKey Questions ({len(questions)}):")
            for q in questions[:5]:  # Top 5 questions
                summary_parts.append(f"  • {q[:100]}")
        
        if tasks:
            summary_parts.append(f"\nIdentified Tasks ({len(tasks)}):")
            for t in tasks[:5]:
                summary_parts.append(f"  • {t}")
        
        summary = "\n".join(summary_parts)
        
        # Save summary to RAG
        if self.rag_chain:
            doc = Document(
                content=summary,
                metadata={
                    "type": "session_summary",
                    "session_id": str(session_id) if session_id else None,
                    "timestamp": datetime.now().isoformat(),
                    "message_count": len(messages)
                }
            )
            
            embedding = self.rag_chain.embedding_manager.embed_text(doc.content)
            self.rag_chain.retriever.add_documents_with_index(
                documents=[doc.content],
                embeddings=[embedding],
                metadatas=[doc.metadata],
                collection_name=self.collection_name
            )
        
        return summary
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        
        stats = {
            "buffer_size": len(self.memory_buffer),
            "buffer_max_size": self.MAX_BUFFER_SIZE,
            "buffer_usage_percent": (len(self.memory_buffer) / self.MAX_BUFFER_SIZE) * 100,
            "auto_save": self.auto_save,
            "importance_threshold": self.importance_threshold.name,
            "collection": self.collection_name
        }
        
        # Count by importance
        importance_counts = {}
        for entry in self.memory_buffer:
            level = entry.importance.name
            importance_counts[level] = importance_counts.get(level, 0) + 1
        
        stats["importance_distribution"] = importance_counts
        
        # Count by tags
        tag_counts = {}
        for entry in self.memory_buffer:
            for tag in entry.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        stats["tag_distribution"] = tag_counts
        
        return stats
    
    async def query_memories(
        self,
        query: str,
        top_k: int = 5,
        importance_filter: Optional[ImportanceLevel] = None,
        tag_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Query saved memories"""
        
        if not self.rag_chain:
            logger.warning("RAG chain not initialized")
            return []
        
        # Build metadata filter
        filter_metadata = {}
        if importance_filter:
            filter_metadata["importance"] = importance_filter.name
        
        # Perform search
        results = self.rag_chain.search(
            query=query,
            top_k=top_k,
            search_type="hybrid",
            filter_metadata=filter_metadata if filter_metadata else None
        )
        
        # Additional tag filtering if needed
        if tag_filter:
            filtered = []
            for result in results:
                tags = result["metadata"].get("tags", "").split(", ")
                if any(tag in tags for tag in tag_filter):
                    filtered.append(result)
            results = filtered
        
        return results
    
    async def cleanup_old_memories(self, days: int = 30) -> int:
        """Clean up old memories from buffer and optionally from RAG"""
        
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Clean buffer
        old_buffer = self.memory_buffer
        self.memory_buffer = [
            entry for entry in self.memory_buffer
            if entry.timestamp > cutoff_date
        ]
        
        removed = len(old_buffer) - len(self.memory_buffer)
        if removed > 0:
            logger.info(f"Removed {removed} old entries from memory buffer")
        
        return removed