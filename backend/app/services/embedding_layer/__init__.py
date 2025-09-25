"""
Embedding Layer - Vector Generation & Indexing Module

This module handles:
- High-quality vector embeddings using intfloat/multilingual-e5-large
- Optimized chunking for semantic matching
- Qdrant vector database integration
- Multilingual support (English + French)
- Metadata filtering and cosine similarity search
"""

from .embedding_service import EmbeddingService, EmbeddingConfig
from .qdrant_store import QdrantStore, QdrantConfig
from .chunk_optimizer import ChunkOptimizer, ChunkConfig
from .multilingual_processor import MultilingualProcessor

__all__ = [
    "EmbeddingService",
    "EmbeddingConfig", 
    "QdrantStore",
    "QdrantConfig",
    "ChunkOptimizer",
    "ChunkConfig",
    "MultilingualProcessor"
]


