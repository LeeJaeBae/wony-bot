"""Embedding generation and management"""

from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import hashlib
import json
from pathlib import Path
from app.config import settings as app_settings
import logging
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embedding generation with caching and batch processing"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize embedding model"""
        
        # Set device (MPS for M1/M2 Macs, CUDA for NVIDIA, CPU fallback)
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Set up caching
        self.cache_dir = Path(cache_dir or app_settings.data_dir / "embeddings_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings.json"
        self._load_cache()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Model info
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded model '{model_name}' with {self.embedding_dim} dimensions")
    
    def _load_cache(self):
        """Load embedding cache from disk"""
        self.cache = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    cache_data = json.load(f)
                    # Convert lists back to numpy arrays
                    for key, value in cache_data.items():
                        self.cache[key] = np.array(value, dtype=np.float32)
                logger.info(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            cache_data = {
                key: value.tolist() for key, value in self.cache.items()
            }
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f)
            logger.debug(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.sha256(f"{self.model_name}:{text}".encode()).hexdigest()
    
    def embed_text(
        self,
        text: Union[str, List[str]],
        use_cache: bool = True,
        normalize: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings for text"""
        
        # Handle single text
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        embeddings = []
        texts_to_encode = []
        cache_indices = []
        
        # Check cache
        for i, t in enumerate(texts):
            if use_cache:
                cache_key = self._get_cache_key(t)
                if cache_key in self.cache:
                    embeddings.append(self.cache[cache_key])
                    cache_indices.append(i)
                    continue
            
            texts_to_encode.append(t)
        
        # Generate embeddings for uncached texts
        if texts_to_encode:
            logger.info(f"Generating embeddings for {len(texts_to_encode)} texts")
            
            try:
                # Batch encode
                new_embeddings = self.model.encode(
                    texts_to_encode,
                    convert_to_tensor=False,
                    normalize_embeddings=normalize,
                    show_progress_bar=len(texts_to_encode) > 10
                )
                
                # Convert to numpy if needed
                if torch.is_tensor(new_embeddings):
                    new_embeddings = new_embeddings.cpu().numpy()
                
                # Add to cache
                if use_cache:
                    for t, emb in zip(texts_to_encode, new_embeddings):
                        cache_key = self._get_cache_key(t)
                        self.cache[cache_key] = emb
                    self._save_cache()
                
                # Merge cached and new embeddings in correct order
                result_embeddings = []
                new_idx = 0
                for i in range(len(texts)):
                    if i in cache_indices:
                        # Use cached embedding
                        idx = cache_indices.index(i)
                        result_embeddings.append(embeddings[idx])
                    else:
                        # Use new embedding
                        result_embeddings.append(new_embeddings[new_idx])
                        new_idx += 1
                
                embeddings = result_embeddings
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                raise
        
        # Convert to list format for ChromaDB
        embeddings = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]
        
        return embeddings[0] if single_input else embeddings
    
    async def embed_text_async(
        self,
        text: Union[str, List[str]],
        use_cache: bool = True,
        normalize: bool = True
    ) -> Union[List[float], List[List[float]]]:
        """Async wrapper for embedding generation"""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.embed_text,
            text,
            use_cache,
            normalize
        )
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        use_cache: bool = True,
        normalize: bool = True,
        show_progress: bool = True
    ) -> List[List[float]]:
        """Embed texts in batches for better performance"""
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if show_progress:
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = self.embed_text(
                batch,
                use_cache=use_cache,
                normalize=normalize
            )
            
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_size_mb = 0
        if self.cache_file.exists():
            cache_size_mb = self.cache_file.stat().st_size / (1024 * 1024)
        
        return {
            "cached_embeddings": len(self.cache),
            "cache_size_mb": round(cache_size_mb, 2),
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device
        }
    
    def similarity(
        self,
        embedding1: Union[List[float], np.ndarray],
        embedding2: Union[List[float], np.ndarray]
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        
        # Convert to numpy arrays
        if isinstance(embedding1, list):
            embedding1 = np.array(embedding1)
        if isinstance(embedding2, list):
            embedding2 = np.array(embedding2)
        
        # Normalize
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Cosine similarity
        return float(np.dot(embedding1, embedding2))
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float, str]]:
        """Rerank documents based on similarity to query"""
        
        # Get query embedding
        query_embedding = np.array(self.embed_text(query))
        
        # Get document embeddings
        doc_embeddings = [np.array(emb) for emb in self.embed_text(documents)]
        
        # Calculate similarities
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            sim = self.similarity(query_embedding, doc_emb)
            similarities.append((i, sim, documents[i]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k:
            return similarities[:top_k]
        
        return similarities