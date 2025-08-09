"""Hybrid retriever combining vector and keyword search"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import re
from dataclasses import dataclass
import logging
from app.rag.vector_store import VectorStore
from app.rag.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with document and score"""
    document: str
    metadata: Dict[str, Any]
    score: float
    source: str  # 'vector' or 'keyword'
    
class HybridRetriever:
    """Combines vector similarity and keyword search for better retrieval"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """Initialize hybrid retriever"""
        
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        
        # BM25 components
        self.bm25_index = {}  # collection_name -> BM25Okapi
        self.bm25_docs = {}   # collection_name -> List[str]
        self.bm25_metadata = {}  # collection_name -> List[Dict]
        
        logger.info(f"HybridRetriever initialized with weights: vector={vector_weight}, keyword={keyword_weight}")
    
    def build_bm25_index(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        collection_name: str = "documents"
    ):
        """Build BM25 index for keyword search"""
        
        # Tokenize documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        # Create BM25 index
        self.bm25_index[collection_name] = BM25Okapi(tokenized_docs)
        self.bm25_docs[collection_name] = documents
        self.bm25_metadata[collection_name] = metadatas
        
        logger.info(f"Built BM25 index for {len(documents)} documents in '{collection_name}'")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def vector_search(
        self,
        query: str,
        collection_name: str = "documents",
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform vector similarity search"""
        
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(query)
        
        # Search in vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            collection_name=collection_name,
            n_results=top_k,
            filter_metadata=filter_metadata
        )
        
        # Convert to SearchResult objects
        search_results = []
        for i, doc in enumerate(results["documents"]):
            search_results.append(
                SearchResult(
                    document=doc,
                    metadata=results["metadatas"][i] if i < len(results["metadatas"]) else {},
                    score=1.0 - results["distances"][i] if i < len(results["distances"]) else 0.0,
                    source="vector"
                )
            )
        
        return search_results
    
    def keyword_search(
        self,
        query: str,
        collection_name: str = "documents",
        top_k: int = 10
    ) -> List[SearchResult]:
        """Perform BM25 keyword search"""
        
        if collection_name not in self.bm25_index:
            logger.warning(f"No BM25 index for collection '{collection_name}'")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        bm25 = self.bm25_index[collection_name]
        scores = bm25.get_scores(query_tokens)
        
        # Get top-k documents
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Create SearchResult objects
        search_results = []
        docs = self.bm25_docs[collection_name]
        metadatas = self.bm25_metadata[collection_name]
        
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                search_results.append(
                    SearchResult(
                        document=docs[idx],
                        metadata=metadatas[idx] if idx < len(metadatas) else {},
                        score=float(scores[idx]),
                        source="keyword"
                    )
                )
        
        return search_results
    
    def hybrid_search(
        self,
        query: str,
        collection_name: str = "documents",
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword search"""
        
        # Get results from both searches
        vector_results = self.vector_search(
            query=query,
            collection_name=collection_name,
            top_k=top_k * 2,  # Get more results for merging
            filter_metadata=filter_metadata
        )
        
        keyword_results = self.keyword_search(
            query=query,
            collection_name=collection_name,
            top_k=top_k * 2
        )
        
        # Merge and deduplicate results
        merged_results = self._merge_results(vector_results, keyword_results)
        
        # Rerank if requested
        if rerank and merged_results:
            merged_results = self._rerank_results(query, merged_results)
        
        # Return top-k
        return merged_results[:top_k]
    
    def _merge_results(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Merge and deduplicate search results"""
        
        # Normalize scores
        vector_results = self._normalize_scores(vector_results)
        keyword_results = self._normalize_scores(keyword_results)
        
        # Create document map for deduplication
        doc_map = {}
        
        # Add vector results with weighted scores
        for result in vector_results:
            doc_hash = hashlib.sha256(result.document.encode()).hexdigest()[:16]
            if doc_hash not in doc_map:
                result.score *= self.vector_weight
                doc_map[doc_hash] = result
            else:
                # Combine scores if document already exists
                doc_map[doc_hash].score += result.score * self.vector_weight
        
        # Add keyword results with weighted scores
        for result in keyword_results:
            doc_hash = hashlib.sha256(result.document.encode()).hexdigest()[:16]
            if doc_hash not in doc_map:
                result.score *= self.keyword_weight
                doc_map[doc_hash] = result
            else:
                # Combine scores if document already exists
                doc_map[doc_hash].score += result.score * self.keyword_weight
        
        # Sort by combined score
        merged = list(doc_map.values())
        merged.sort(key=lambda x: x.score, reverse=True)
        
        return merged
    
    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Normalize scores to 0-1 range"""
        
        if not results:
            return results
        
        # Get min and max scores
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            for r in results:
                r.score = 1.0
            return results
        
        # Normalize
        for r in results:
            r.score = (r.score - min_score) / (max_score - min_score)
        
        return results
    
    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        diversity_factor: float = 0.3
    ) -> List[SearchResult]:
        """Rerank results using embedding similarity and MMR"""
        
        if not results:
            return results
        
        # Get embeddings for reranking
        documents = [r.document for r in results]
        reranked = self.embedding_manager.rerank(query, documents)
        
        # Update scores based on reranking
        score_map = {doc: score for idx, score, doc in reranked}
        
        for result in results:
            if result.document in score_map:
                # Combine original score with rerank score
                result.score = (result.score + score_map[result.document]) / 2
        
        # Apply MMR for diversity
        if diversity_factor > 0:
            results = self._apply_mmr(results, diversity_factor)
        
        # Sort by final score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _apply_mmr(
        self,
        results: List[SearchResult],
        diversity_factor: float = 0.3
    ) -> List[SearchResult]:
        """Apply Maximum Marginal Relevance for result diversity"""
        
        if len(results) <= 1:
            return results
        
        # Get embeddings for all documents
        documents = [r.document for r in results]
        embeddings = self.embedding_manager.embed_text(documents)
        
        # Convert to numpy arrays
        embeddings = [np.array(emb) for emb in embeddings]
        
        # Initialize selected results
        selected = [results[0]]  # Start with highest scoring
        selected_embeddings = [embeddings[0]]
        remaining = list(range(1, len(results)))
        
        # Iteratively select diverse results
        while remaining and len(selected) < len(results):
            best_idx = None
            best_score = -float('inf')
            
            for idx in remaining:
                # Calculate relevance score
                relevance = results[idx].score
                
                # Calculate similarity to already selected documents
                max_sim = 0
                for sel_emb in selected_embeddings:
                    sim = self.embedding_manager.similarity(embeddings[idx], sel_emb)
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr_score = (1 - diversity_factor) * relevance - diversity_factor * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(results[best_idx])
                selected_embeddings.append(embeddings[best_idx])
                remaining.remove(best_idx)
        
        return selected
    
    def add_documents_with_index(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        collection_name: str = "documents"
    ) -> List[str]:
        """Add documents to both vector store and BM25 index"""
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name
        )
        
        # Update BM25 index
        if collection_name in self.bm25_index:
            # Append to existing index
            existing_docs = self.bm25_docs[collection_name]
            existing_metadata = self.bm25_metadata[collection_name]
            
            all_docs = existing_docs + documents
            all_metadata = existing_metadata + metadatas
        else:
            all_docs = documents
            all_metadata = metadatas
        
        self.build_bm25_index(all_docs, all_metadata, collection_name)
        
        return doc_ids

# Add missing import
import hashlib