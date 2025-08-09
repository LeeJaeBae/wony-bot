"""Vector store management using ChromaDB"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
import hashlib
import json
from pathlib import Path
from app.config import settings as app_settings
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector database operations with ChromaDB"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize ChromaDB client"""
        self.persist_dir = persist_directory or str(app_settings.data_dir / "chroma")
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Collections for different document types
        self.collections = {}
        self._init_collections()
        
    def _init_collections(self):
        """Initialize or get existing collections"""
        collection_configs = {
            "documents": {
                "metadata": {"hnsw:space": "cosine"},
                "description": "General documents (PDF, DOCX, TXT, MD)"
            },
            "chat_history": {
                "metadata": {"hnsw:space": "cosine"},
                "description": "Chat conversation history"
            },
            "web_content": {
                "metadata": {"hnsw:space": "cosine"},
                "description": "Web pages and online content"
            },
            "code": {
                "metadata": {"hnsw:space": "cosine"},
                "description": "Source code and technical documentation"
            }
        }
        
        for name, config in collection_configs.items():
            try:
                self.collections[name] = self.client.get_or_create_collection(
                    name=name,
                    metadata=config["metadata"]
                )
                logger.info(f"Collection '{name}' initialized: {config['description']}")
            except Exception as e:
                logger.error(f"Failed to initialize collection '{name}': {e}")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = "documents"
    ) -> List[str]:
        """Add documents with embeddings to the vector store"""
        
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        # Generate IDs if not provided
        if ids is None:
            ids = [self._generate_id(doc) for doc in documents]
        
        # Ensure metadatas is provided
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        
        # Add to ChromaDB
        try:
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to '{collection_name}'")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        collection_name: str = "documents",
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar documents using vector similarity"""
        
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        try:
            # Perform similarity search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata if filter_metadata else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "ids": results["ids"][0] if results["ids"] else []
            }
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
    
    def update_document(
        self,
        document_id: str,
        document: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: str = "documents"
    ):
        """Update an existing document"""
        
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        try:
            update_params = {"ids": [document_id]}
            
            if document is not None:
                update_params["documents"] = [document]
            if embedding is not None:
                update_params["embeddings"] = [embedding]
            if metadata is not None:
                update_params["metadatas"] = [metadata]
            
            collection.update(**update_params)
            logger.info(f"Updated document {document_id} in '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            raise
    
    def delete_documents(
        self,
        ids: List[str],
        collection_name: str = "documents"
    ):
        """Delete documents by IDs"""
        
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        try:
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def get_collection_stats(self, collection_name: str = "documents") -> Dict[str, Any]:
        """Get statistics about a collection"""
        
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        try:
            count = collection.count()
            
            return {
                "name": collection_name,
                "count": count,
                "persist_directory": self.persist_dir
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"name": collection_name, "count": 0, "error": str(e)}
    
    def clear_collection(self, collection_name: str = "documents"):
        """Clear all documents from a collection"""
        
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        try:
            # Delete and recreate the collection
            self.client.delete_collection(name=collection_name)
            self._init_collections()
            logger.info(f"Cleared collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for a document based on its content"""
        # Use SHA256 hash of content for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"doc_{content_hash}_{uuid4().hex[:8]}"
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        return list(self.collections.keys())
    
    def export_collection(
        self,
        collection_name: str = "documents",
        output_path: Optional[str] = None
    ) -> str:
        """Export a collection to JSON"""
        
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        # Get all documents
        results = collection.get(include=["documents", "metadatas", "embeddings"])
        
        # Prepare export data
        export_data = {
            "collection": collection_name,
            "count": len(results["ids"]),
            "documents": []
        }
        
        for i in range(len(results["ids"])):
            export_data["documents"].append({
                "id": results["ids"][i],
                "document": results["documents"][i] if results["documents"] else None,
                "metadata": results["metadatas"][i] if results["metadatas"] else None,
                # Embeddings are large, optionally exclude them
                # "embedding": results["embeddings"][i] if results["embeddings"] else None
            })
        
        # Save to file
        if output_path is None:
            output_path = str(app_settings.data_dir / f"{collection_name}_export.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {export_data['count']} documents to {output_path}")
        return output_path