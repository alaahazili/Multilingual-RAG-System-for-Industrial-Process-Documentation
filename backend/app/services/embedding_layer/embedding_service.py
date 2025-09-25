"""
Enhanced Embedding Service - Embedding Layer Component

Provides high-quality vector embeddings using intfloat/multilingual-e5-large
with support for both Hugging Face Inference API and self-hosted models.
"""

import os
import requests
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

# Global model cache to prevent reloading
_MODEL_CACHE = {}


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service."""
    model_name: str = "intfloat/multilingual-e5-large"
    vector_dimension: int = 1024
    use_hf_api: bool = True  # Use Hugging Face Inference API
    api_token: Optional[str] = None
    api_url: Optional[str] = None
    timeout_s: int = 300  # Increased timeout to 5 minutes for large batches
    normalize_embeddings: bool = True
    batch_size: int = 32


class EmbeddingService:
    """Enhanced embedding service with multilingual support and optimized performance."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config: EmbeddingConfig = None):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: EmbeddingConfig = None):
        if not self._initialized:
            self.config = config or EmbeddingConfig()
            self._model = None
            self._api_client = None
            
            # Initialize based on configuration
            if self.config.use_hf_api:
                self._init_hf_api_client()
            else:
                self._init_self_hosted_model()
            
            self._initialized = True
            logger.info(f"EmbeddingService initialized (singleton) with config: {self.config.model_name}")
    
    def _init_hf_api_client(self):
        """Initialize Hugging Face Inference API client."""
        self.api_url = self.config.api_url or f"https://api-inference.huggingface.co/models/{self.config.model_name}"
        # Try HF_TOKEN_EMBED first, then fallback to HF_TOKEN
        self.api_token = self.config.api_token or os.getenv("HF_TOKEN_EMBED") or os.getenv("HF_TOKEN", "")
        
        if not self.api_token:
            raise RuntimeError("HF_TOKEN_EMBED or HF_TOKEN env var is required to use Hugging Face Inference API")
        
        self._headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "x-wait-for-model": "true",
        }
        logger.info(f"Initialized HF Inference API client for {self.config.model_name}")
    
    def _init_self_hosted_model(self):
        """Initialize self-hosted sentence transformer model with caching."""
        try:
            # Check if model is already in global cache
            cache_key = f"embedding_{self.config.model_name}"
            if cache_key in _MODEL_CACHE:
                logger.info(f"Using cached embedding model: {self.config.model_name}")
                self._model = _MODEL_CACHE[cache_key]
            else:
                logger.info(f"Loading embedding model into cache: {self.config.model_name}")
                self._model = SentenceTransformer(self.config.model_name)
                # Cache the model globally
                _MODEL_CACHE[cache_key] = self._model
                logger.info(f"Embedding model cached successfully: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load self-hosted model: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.config.vector_dimension
    
    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """Embed document passages for storage."""
        return self.embed_texts(texts, kind="passage")
    
    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """Embed queries for search."""
        return self.embed_texts(texts, kind="query")
    
    def embed_texts(self, texts: List[str], kind: str = "passage") -> List[List[float]]:
        """Embed texts with proper E5 prompts."""
        if not texts:
            return []
        
        if kind not in {"passage", "query"}:
            raise ValueError("kind must be 'passage' or 'query'")
        
        # Add E5 prompts
        prompted_texts = [f"{kind}: {text}" for text in texts]
        
        if self.config.use_hf_api:
            return self._embed_with_hf_api(prompted_texts)
        else:
            return self._embed_with_self_hosted(prompted_texts)
    
    def _embed_with_hf_api(self, texts: List[str]) -> List[List[float]]:
        """Embed using Hugging Face Inference API."""
        try:
            payload = {
                "inputs": texts,
                "options": {"wait_for_model": True}
            }
            
            response = requests.post(
                self.api_url,
                headers=self._headers,
                json=payload,
                timeout=self.config.timeout_s
            )
            response.raise_for_status()
            
            embeddings = response.json()
            if not isinstance(embeddings, list) or not embeddings:
                raise RuntimeError(f"Unexpected HF API response: {str(embeddings)[:200]}")
            
            # Normalize embeddings if configured
            if self.config.normalize_embeddings:
                embeddings = self._normalize_embeddings(embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"HF API embedding failed: {e}")
            raise
    
    def _embed_with_self_hosted(self, texts: List[str]) -> List[List[float]]:
        """Embed using self-hosted model."""
        try:
            # Process in batches for memory efficiency
            all_embeddings = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                batch_embeddings = self._model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize_embeddings
                )
                
                # Convert to list of lists
                batch_embeddings = batch_embeddings.tolist()
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Self-hosted embedding failed: {e}")
            raise
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit vectors."""
        normalized = []
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            if norm > 0:
                normalized.append([x / norm for x in emb])
            else:
                normalized.append(emb)
        return normalized
    
    def create_embedding(self, text: str, content_type: str = "general") -> List[float]:
        """Create a single passage embedding."""
        return self.embed_passages([text])[0]
    
    def create_query_embedding(self, text: str) -> List[float]:
        """Create a single query embedding."""
        return self.embed_queries([text])[0]
    
    def batch_embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch embed documents with metadata preservation."""
        if not documents:
            return []
        
        # Extract texts for embedding
        texts = [doc.get("content", "") for doc in documents]
        
        # Create embeddings
        embeddings = self.embed_passages(texts)
        
        # Combine with metadata
        embedded_docs = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            embedded_doc = doc.copy()
            embedded_doc["embedding"] = embedding
            # Use a simple integer ID for Qdrant compatibility
            embedded_doc["embedding_id"] = i + 1
            embedded_docs.append(embedded_doc)
        
        return embedded_docs
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            "model_name": self.config.model_name,
            "vector_dimension": self.config.vector_dimension,
            "use_hf_api": self.config.use_hf_api,
            "normalize_embeddings": self.config.normalize_embeddings,
            "batch_size": self.config.batch_size
        }
