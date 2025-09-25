"""
Retrieval Service - Retrieval Layer Component

Main service that orchestrates the complete retrieval pipeline:
- Query embedding with intfloat/multilingual-e5-large
- Top-k search in Qdrant (configurable, 3-10 results)
- Optional reranking for better relevance
- Metadata filtering capabilities
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from sentence_transformers import CrossEncoder

from ..embedding_layer import EmbeddingService, EmbeddingConfig
from ..embedding_layer.qdrant_store import QdrantStore, QdrantConfig
from .reranker import Reranker, RerankerConfig
from .search_optimizer import SearchOptimizer, SearchConfig
from .metadata_filter import MetadataFilter, FilterConfig

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval service."""
    # Search parameters
    top_k: int = 10  # Default top-k results
    min_score_threshold: float = 0.5  # Lower threshold for better recall
    max_score_threshold: float = 0.95  # Maximum similarity score
    
    # Reranking
    use_reranker: bool = True  # Enable/disable reranking
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # Reranker model
    rerank_top_k: int = 5  # Number of results to rerank
    
    # Search optimization
    enable_search_optimization: bool = True
    enable_metadata_filtering: bool = True
    
    # Embedding service
    embedding_config: Optional[EmbeddingConfig] = None
    qdrant_config: Optional[QdrantConfig] = None


class RetrievalService:
    """Main retrieval service with advanced search capabilities."""
    
    def __init__(self, config: RetrievalConfig = None):
        self.config = config or RetrievalConfig()
        
        # Initialize components
        self.embedding_service = EmbeddingService(self.config.embedding_config)
        self.qdrant_store = QdrantStore(self.config.qdrant_config)
        
        # Initialize optional components
        self.reranker = None
        if self.config.use_reranker:
            self.reranker = Reranker(RerankerConfig(model_name=self.config.reranker_model))
        
        self.search_optimizer = None
        if self.config.enable_search_optimization:
            self.search_optimizer = SearchOptimizer(SearchConfig())
        
        self.metadata_filter = None
        if self.config.enable_metadata_filtering:
            self.metadata_filter = MetadataFilter(FilterConfig())
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        document_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
        facility_filter: Optional[str] = None,
        equipment_tags: Optional[List[str]] = None,
        has_equations: Optional[bool] = None,
        has_tables: Optional[bool] = None,
        technical_terms: Optional[List[str]] = None,
        enable_reranking: Optional[bool] = None,
        **kwargs  # Allow additional filter parameters
    ) -> Dict[str, Any]:
        """
        Perform comprehensive semantic search with optional reranking.
        
        Args:
            query: User query text
            top_k: Number of results to return (default: config.top_k)
            score_threshold: Minimum similarity score (default: config.min_score_threshold)
            document_filter: Filter by specific document
            section_filter: Filter by specific section
            language_filter: Filter by language (en/fr)
            facility_filter: Filter by facility
            equipment_tags: Filter by equipment tags
            has_equations: Filter by presence of equations
            has_tables: Filter by presence of tables
            enable_reranking: Override reranking setting
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            # Use config defaults if not specified
            top_k = top_k or self.config.top_k
            score_threshold = score_threshold or self.config.min_score_threshold
            enable_reranking = enable_reranking if enable_reranking is not None else self.config.use_reranker
            
            logger.info(f"Starting search for query: '{query[:50]}...'")
            
            # Step 1: Create query embedding
            query_embedding = self.embedding_service.create_query_embedding(query)
            logger.debug(f"Created query embedding: {len(query_embedding)} dimensions")
            
            # Step 2: Apply metadata filtering
            query_filter = None
            if self.metadata_filter:
                query_filter = self.metadata_filter.create_filter(
                    document=document_filter,
                    section=section_filter,
                    language=language_filter,
                    facility=facility_filter,
                    equipment_tags=equipment_tags,
                    has_equations=has_equations,
                    has_tables=has_tables,
                    technical_terms=technical_terms
                )
            
            # Step 3: Perform initial vector search
            initial_results = self.qdrant_store.search(
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
            
            logger.info(f"Initial search returned {len(initial_results)} results")
            
            # Step 4: Apply search optimization if enabled
            if self.search_optimizer and initial_results:
                initial_results = self.search_optimizer.optimize_results(
                    results=initial_results,
                    query=query,
                    top_k=top_k
                )
                logger.debug("Applied search optimization")
            
            # Step 5: Apply reranking if enabled and we have results
            final_results = initial_results
            reranking_applied = False
            
            if enable_reranking and self.reranker and initial_results:
                # Take top results for reranking
                rerank_candidates = initial_results[:self.config.rerank_top_k]
                reranked_results = self.reranker.rerank(
                    query=query,
                    passages=[result["text"] for result in rerank_candidates]
                )
                
                # Update scores and reorder
                for i, result in enumerate(rerank_candidates):
                    result["reranked_score"] = reranked_results[i]["score"]
                    result["original_score"] = result["score"]
                    result["score"] = reranked_results[i]["score"]
                
                # Reorder by reranked scores
                rerank_candidates.sort(key=lambda x: x["score"], reverse=True)
                
                # Combine reranked top results with remaining results
                remaining_results = initial_results[self.config.rerank_top_k:]
                final_results = rerank_candidates + remaining_results
                reranking_applied = True
                
                logger.info(f"Applied reranking to top {len(rerank_candidates)} results")
            
            # Step 6: Prepare response
            response = {
                "query": query,
                "results": final_results[:top_k],
                "total_results": len(final_results),
                "search_metadata": {
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                    "reranking_applied": reranking_applied,
                    "filters_applied": {
                        "document": document_filter,
                        "section": section_filter,
                        "language": language_filter,
                        "facility": facility_filter,
                        "equipment_tags": equipment_tags,
                        "has_equations": has_equations,
                        "has_tables": has_tables,
                        "technical_terms": technical_terms
                    }
                }
            }
            
            logger.info(f"Search completed: {len(final_results[:top_k])} results returned")
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": str(e),
                "search_metadata": {}
            }
    
    def search_by_technical_content(
        self,
        query: str,
        equipment_tags: List[str],
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Search specifically for technical content with equipment tags."""
        return self.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            equipment_tags=equipment_tags
        )
    
    def search_by_document(
        self,
        query: str,
        document_name: str,
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Search within a specific document."""
        return self.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            document_filter=document_name
        )
    
    def search_by_section(
        self,
        query: str,
        section_name: str,
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Search within a specific section."""
        return self.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            section_filter=section_name
        )
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the retrieval service."""
        collection_info = self.qdrant_store.get_collection_info()
        
        return {
            "embedding_model": self.embedding_service.get_model_info(),
            "qdrant_collection": collection_info,
            "reranker_enabled": self.reranker is not None,
            "reranker_model": self.config.reranker_model if self.reranker else None,
            "search_optimization_enabled": self.search_optimizer is not None,
            "metadata_filtering_enabled": self.metadata_filter is not None,
            "default_top_k": self.config.top_k,
            "default_score_threshold": self.config.min_score_threshold
        }
