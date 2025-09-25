"""
Retrieval Layer - Vector Search Module

Provides advanced semantic search capabilities:
- Query embedding with intfloat/multilingual-e5-large
- Top-k search in Qdrant (configurable, 3-10 results)
- Optional reranking with bge-reranker for better relevance
- Metadata filtering by document, section, language
- Enhanced search with semantic similarity and technical content matching
"""

from .retrieval_service import RetrievalService, RetrievalConfig
from .reranker import Reranker, RerankerConfig
from .search_optimizer import SearchOptimizer, SearchConfig
from .metadata_filter import MetadataFilter, FilterConfig

__all__ = [
    "RetrievalService", "RetrievalConfig",
    "Reranker", "RerankerConfig", 
    "SearchOptimizer", "SearchConfig",
    "MetadataFilter", "FilterConfig"
]


