"""RAG Chain for question answering with retrieved context"""

from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
from app.rag.vector_store import VectorStore
from app.rag.embeddings import EmbeddingManager
from app.rag.retriever import HybridRetriever
from app.rag.document_loader import DocumentLoader, Document
from app.core.ollama import OllamaClient
from app.models.schemas import Message, MessageRole
from pathlib import Path

logger = logging.getLogger(__name__)

class RAGChain:
    """Complete RAG pipeline for document-based question answering"""
    
    # RAG prompt templates
    RAG_SYSTEM_PROMPT = """You are WonyBot, an AI assistant that answers questions based on provided context.

IMPORTANT RULES:
1. Answer ONLY based on the provided context
2. If the context doesn't contain the answer, say "제공된 정보에서 답을 찾을 수 없습니다"
3. Always cite the source (filename, page, etc.) when available
4. Be concise but complete in your answers
5. Use the same language as the user's question"""
    
    RAG_USER_PROMPT = """## Context:
{context}

## Question:
{question}

## Answer:
Based on the provided context, here is my answer:"""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        document_loader: Optional[DocumentLoader] = None,
        ollama_client: Optional[OllamaClient] = None,
        collection_name: str = "documents"
    ):
        """Initialize RAG chain components"""
        
        # Initialize components
        self.vector_store = vector_store or VectorStore()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.document_loader = document_loader or DocumentLoader()
        self.ollama_client = ollama_client or OllamaClient()
        self.collection_name = collection_name
        
        # Initialize retriever
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager
        )
        
        logger.info("RAG Chain initialized")
    
    def index_document(
        self,
        file_path: str,
        chunk: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Index a single document"""
        
        try:
            # Load document
            document = self.document_loader.load_file(file_path)
            
            # Add custom metadata if provided
            if metadata:
                document.metadata.update(metadata)
            
            # Process document (chunking)
            if chunk:
                documents = self.document_loader.chunk_document(document)
            else:
                documents = [document]
            
            # Generate embeddings
            texts = [doc.content for doc in documents]
            embeddings = self.embedding_manager.embed_batch(texts, show_progress=True)
            
            # Prepare metadata
            metadatas = [doc.metadata for doc in documents]
            
            # Add to vector store and BM25 index
            self.retriever.add_documents_with_index(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                collection_name=self.collection_name
            )
            
            logger.info(f"Indexed {len(documents)} chunks from {file_path}")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            raise
    
    def index_directory(
        self,
        directory: str,
        glob_pattern: str = "**/*",
        recursive: bool = True,
        chunk: bool = True
    ) -> int:
        """Index all documents in a directory"""
        
        try:
            # Load all documents
            documents = self.document_loader.load_directory(
                directory=directory,
                glob_pattern=glob_pattern,
                recursive=recursive
            )
            
            if not documents:
                logger.warning(f"No documents found in {directory}")
                return 0
            
            # Process all documents
            all_chunks = []
            for doc in documents:
                if chunk:
                    chunks = self.document_loader.chunk_document(doc)
                    all_chunks.extend(chunks)
                else:
                    all_chunks.append(doc)
            
            # Generate embeddings in batches
            texts = [doc.content for doc in all_chunks]
            embeddings = self.embedding_manager.embed_batch(texts, show_progress=True)
            
            # Prepare metadata
            metadatas = [doc.metadata for doc in all_chunks]
            
            # Add to vector store and BM25 index
            self.retriever.add_documents_with_index(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                collection_name=self.collection_name
            )
            
            logger.info(f"Indexed {len(all_chunks)} chunks from {len(documents)} documents")
            return len(all_chunks)
            
        except Exception as e:
            logger.error(f"Failed to index directory: {e}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        
        # Perform search based on type
        if search_type == "hybrid":
            results = self.retriever.hybrid_search(
                query=query,
                collection_name=self.collection_name,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
        elif search_type == "vector":
            results = self.retriever.vector_search(
                query=query,
                collection_name=self.collection_name,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
        elif search_type == "keyword":
            results = self.retriever.keyword_search(
                query=query,
                collection_name=self.collection_name,
                top_k=top_k
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.document,
                "metadata": result.metadata,
                "score": result.score,
                "source": result.source
            })
        
        return formatted_results
    
    async def ask(
        self,
        question: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        stream: bool = True,
        include_sources: bool = True
    ) -> AsyncGenerator[str, None]:
        """Ask a question using RAG"""
        
        # Search for relevant context
        search_results = self.search(
            query=question,
            top_k=top_k,
            search_type=search_type
        )
        
        if not search_results:
            yield "죄송합니다. 관련된 정보를 찾을 수 없습니다."
            return
        
        # Build context from search results
        context_parts = []
        sources = set()
        
        for i, result in enumerate(search_results, 1):
            # Add document content
            context_parts.append(f"[Document {i}]\n{result['content']}")
            
            # Collect sources
            if "source" in result["metadata"]:
                sources.add(result["metadata"]["source"])
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        user_prompt = self.RAG_USER_PROMPT.format(
            context=context,
            question=question
        )
        
        # Generate response
        messages = [
            Message(role=MessageRole.SYSTEM, content=self.RAG_SYSTEM_PROMPT),
            Message(role=MessageRole.USER, content=user_prompt)
        ]
        
        # Stream response
        response_text = ""
        async for chunk in self.ollama_client.chat(messages, stream=stream):
            response_text += chunk
            if stream:
                yield chunk
        
        # Add sources if requested
        if include_sources and sources:
            sources_text = "\n\n📚 Sources:\n" + "\n".join([f"• {s}" for s in sources])
            yield sources_text
        
        if not stream:
            yield response_text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        
        stats = {
            "vector_store": self.vector_store.get_collection_stats(self.collection_name),
            "embedding_cache": self.embedding_manager.get_cache_stats(),
            "collections": self.vector_store.list_collections()
        }
        
        return stats
    
    def clear_index(self, collection_name: Optional[str] = None):
        """Clear the vector index"""
        
        collection = collection_name or self.collection_name
        self.vector_store.clear_collection(collection)
        logger.info(f"Cleared collection '{collection}'")
    
    def export_index(
        self,
        output_path: Optional[str] = None,
        collection_name: Optional[str] = None
    ) -> str:
        """Export index to JSON"""
        
        collection = collection_name or self.collection_name
        return self.vector_store.export_collection(collection, output_path)