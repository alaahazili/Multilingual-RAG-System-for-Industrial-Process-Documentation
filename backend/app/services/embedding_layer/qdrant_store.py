"""
Enhanced Qdrant Store - Embedding Layer Component

Provides high-performance vector storage with:
- 1024-dimensional normalized vectors
- Metadata filtering by document, chapter, section, language
- Cosine similarity search with high recall
- Optimized indexing for technical documents
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    Replica,
    ShardKey,
)

logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector store."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "documents"
    vector_size: int = 1024
    distance_metric: Distance = Distance.COSINE
    on_disk_payload: bool = True
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    replication_factor: int = 1


class QdrantStore:
    """Enhanced Qdrant vector store with advanced filtering and search capabilities."""
    
    def __init__(self, config: QdrantConfig = None):
        self.config = config or QdrantConfig()
        self.client = QdrantClient(
            host=self.config.host,
            port=self.config.port,
        )
        self.collection_name = self.config.collection_name
        self.vector_size = self.config.vector_size
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the collection exists with proper configuration."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self._create_collection()
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    def _create_collection(self):
        """Create collection with optimized parameters for technical documents."""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance="Cosine",  # Fixed: Use correct case
                on_disk=True,
                hnsw_config={
                    "m": self.config.hnsw_m,
                    "ef_construct": self.config.hnsw_ef_construct,
                }
            ),
            optimizers_config={
                "default_segment_number": 2,
                "memmap_threshold": 20000,
            },
            replication_factor=self.config.replication_factor,
            on_disk_payload=self.config.on_disk_payload,
        )
    
    def upsert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Upsert documents with embeddings and metadata."""
        try:
            points = []
            
            for i, doc in enumerate(documents):
                embedding = doc.get("embedding")
                if not embedding or len(embedding) != self.vector_size:
                    logger.warning(f"Invalid embedding for document {doc.get('chunk_id', 'unknown')}")
                    continue
                
                # Create point with enhanced metadata
                point = PointStruct(
                    id=doc.get("embedding_id", i + 1),  # Use integer ID
                    vector=embedding,
                    payload={
                        "text": doc.get("content", ""),
                        "chunk_id": doc.get("chunk_id", ""),
                        "document": doc.get("document", ""),
                        "section": doc.get("section_title", ""),
                        "section_number": doc.get("section_number", ""),
                        "page_start": doc.get("page_start", 0),
                        "page_end": doc.get("page_end", 0),
                        "language": doc.get("language", "en"),
                        "document_type": doc.get("document_type", ""),
                        "facility": doc.get("facility", ""),
                        "token_count": doc.get("token_count", 0),
                        # Enhanced technical metadata
                        "has_equations": doc.get("has_equations", False),
                        "has_tables": doc.get("has_tables", False),
                        "equipment_tags": doc.get("equipment_tags", []),
                        "technical_terms": doc.get("technical_terms", []),
                        "measurements": doc.get("measurements", []),
                        # Legacy metadata for compatibility
                        "semantic_features": {
                            "has_equations": doc.get("has_equations", False),
                            "has_tables": doc.get("has_tables", False),
                            "technical_terms": doc.get("technical_terms", [])
                        },
                        "technical_content": {
                            "equipment_tags": doc.get("equipment_tags", []),
                            "measurements": doc.get("measurements", [])
                        },
                        "cross_references": doc.get("cross_references", []),
                        "metadata": doc.get("metadata", {}),
                    }
                )
                points.append(point)
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                logger.info(f"Upserted {len(points)} documents to {self.collection_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            return False
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        query_filter: Optional[Filter] = None,
        search_params: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Enhanced search with metadata filtering and cosine similarity."""
        try:
            # Default search parameters for high recall
            if search_params is None:
                search_params = {
                    "hnsw_ef": 128,  # Higher ef for better recall
                    "exact": False,
                }
            
            response = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                search_params=search_params,
                with_payload=True,
                with_vectors=False,
            )
            
            # Convert to standardized format
            results = []
            for result in response:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                    "text": result.payload.get("text", ""),
                    "chunk_id": result.payload.get("chunk_id", ""),
                    "document": result.payload.get("document", ""),
                    "section": result.payload.get("section", ""),
                    "page_start": result.payload.get("page_start", 0),
                    "page_end": result.payload.get("page_end", 0),
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_metadata(
        self,
        query_vector: List[float],
        document: Optional[str] = None,
        section: Optional[str] = None,
        language: Optional[str] = None,
        facility: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search with metadata filtering."""
        conditions = []
        
        if document:
            conditions.append(FieldCondition(key="document", match=MatchValue(value=document)))
        
        if section:
            conditions.append(FieldCondition(key="section", match=MatchValue(value=section)))
        
        if language:
            conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
        
        if facility:
            conditions.append(FieldCondition(key="facility", match=MatchValue(value=facility)))
        
        query_filter = None
        if conditions:
            query_filter = Filter(must=conditions)
        
        return self.search(
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter
        )
    
    def search_by_technical_content(
        self,
        query_vector: List[float],
        equipment_tags: Optional[List[str]] = None,
        has_equations: Optional[bool] = None,
        has_tables: Optional[bool] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search by technical content features."""
        conditions = []
        
        if equipment_tags:
            conditions.append(
                FieldCondition(
                    key="technical_content.equipment_tags",
                    match=MatchAny(any=equipment_tags)
                )
            )
        
        if has_equations is not None:
            conditions.append(
                FieldCondition(
                    key="semantic_features.has_equations",
                    match=MatchValue(value=has_equations)
                )
            )
        
        if has_tables is not None:
            conditions.append(
                FieldCondition(
                    key="semantic_features.has_tables",
                    match=MatchValue(value=has_tables)
                )
            )
        
        query_filter = None
        if conditions:
            query_filter = Filter(must=conditions)
        
        return self.search(
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter
        )
    
    def list_documents(self) -> List[str]:
        """List all unique documents in the collection."""
        try:
            response = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            documents = set()
            for point in response[0]:
                if point.payload and "document" in point.payload:
                    documents.add(point.payload["document"])
            return sorted(list(documents))
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information and statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            
            return {
                "name": info.name,
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance),
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": str(info.status),
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def delete_document(self, document_name: str) -> bool:
        """Delete all chunks for a specific document."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="document", match=MatchValue(value=document_name))]
                )
            )
            logger.info(f"Deleted document: {document_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_name}: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all data from the collection."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=[])
            )
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
