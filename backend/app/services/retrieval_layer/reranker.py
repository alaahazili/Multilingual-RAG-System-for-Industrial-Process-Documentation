"""
Reranker Service - Retrieval Layer Component

Provides intelligent re-ranking of search results using BAAI/bge-reranker-v2-m3
to improve search quality and relevance.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Global model cache to prevent reloading
_MODEL_CACHE = {}


@dataclass
class RerankerConfig:
    """Configuration for reranker service."""
    model_name: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 512
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 8
    normalize_scores: bool = True
    cache_dir: str = "models_cache/reranker"


class Reranker:
    """Intelligent reranker for improving search result quality."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config: RerankerConfig = None):
        if cls._instance is None:
            cls._instance = super(Reranker, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: RerankerConfig = None):
        if not self._initialized:
            self.config = config or RerankerConfig()
            self._model = None
            self._initialized = False
            
            # Set device
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device
            
            # Set cache directory
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load model
            self._load_model()
            
            self._initialized = True
            logger.info(f"Reranker initialized (singleton) with model: {self.config.model_name}")
    
    def _load_model(self):
        """Load the reranker model with caching."""
        try:
            # Check if model is already in global cache
            cache_key = f"reranker_{self.config.model_name}"
            if cache_key in _MODEL_CACHE:
                logger.info(f"Using cached reranker model: {self.config.model_name}")
                self._model = _MODEL_CACHE[cache_key]
                self._initialized = True
                return
            
            # Check if model exists locally
            model_path = self.cache_dir / self.config.model_name.split("/")[-1]
            if model_path.exists() and (model_path / "config.json").exists():
                logger.info(f"Loading reranker model from cache: {model_path}")
                self._model = CrossEncoder(
                    model_name_or_path=str(model_path),
                    max_length=self.config.max_length,
                    device=self.device
                )
            else:
                logger.info(f"Downloading reranker model: {self.config.model_name}")
                logger.info("This will happen only once. Model will be cached for future use.")
                
                # Download and cache the model
                self._model = CrossEncoder(
                    model_name_or_path=self.config.model_name,
                    max_length=self.config.max_length,
                    device=self.device,
                    cache_folder=str(self.cache_dir)
                )
                
                logger.info(f"Model downloaded and cached to: {model_path}")
            
            # Cache the model globally
            _MODEL_CACHE[cache_key] = self._model
            logger.info(f"Reranker model cached successfully: {self.config.model_name}")
            
            self._initialized = True
            logger.info(f"Reranker model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self._initialized = False
    
    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank passages based on their relevance to the query.
        
        Args:
            query: The search query
            passages: List of passages to rerank
            top_k: Number of top results to return (None for all)
            
        Returns:
            List of reranked results with scores
        """
        if not self._initialized or not self._model:
            logger.warning("Reranker not initialized, returning original order")
            return [
                {"text": passage, "score": 1.0 - (i * 0.1), "rank": i + 1}
                for i, passage in enumerate(passages)
            ]
        
        if not passages:
            return []
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, passage] for passage in passages]
            
            # Get scores from cross-encoder
            scores = self._model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )
            
            # Handle different return types from the model
            if hasattr(scores, 'tolist'):
                # Convert numpy array to list
                scores = scores.tolist()
            elif not isinstance(scores, list):
                # Convert single score to list
                scores = [scores]
            
            # Ensure all scores are valid floats and handle any remaining issues
            processed_scores = []
            for score in scores:
                try:
                    if hasattr(score, 'item'):
                        # Handle numpy scalars
                        processed_scores.append(float(score.item()))
                    else:
                        processed_scores.append(float(score) if score is not None else 0.0)
                except (ValueError, TypeError):
                    processed_scores.append(0.0)
            
            scores = processed_scores
            
            # Normalize scores if configured
            if self.config.normalize_scores:
                scores = self._normalize_scores(scores)
            
            # Create results with scores and ranks
            results = []
            for i, (passage, score) in enumerate(zip(passages, scores)):
                results.append({
                    "text": passage,
                    "score": float(score),
                    "rank": i + 1
                })
            
            # Sort by score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Update ranks after sorting
            for i, result in enumerate(results):
                result["rank"] = i + 1
            
            # Return top_k if specified
            if top_k is not None:
                results = results[:top_k]
            
            logger.debug(f"Reranked {len(passages)} passages")
            return results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order with default scores
            return [
                {"text": passage, "score": 1.0 - (i * 0.1), "rank": i + 1}
                for i, passage in enumerate(passages)
            ]
    
    def rerank_with_metadata(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results while preserving metadata.
        
        Args:
            query: The search query
            results: List of search results with metadata
            top_k: Number of top results to return
            
        Returns:
            List of reranked results with preserved metadata
        """
        if not results:
            return []
        
        # Extract passages for reranking
        passages = [result.get("text", "") for result in results]
        
        # Perform reranking
        reranked_passages = self.rerank(query, passages, top_k)
        
        # Map reranked results back to original results with metadata
        reranked_results = []
        for reranked in reranked_passages:
            # Find original result with matching text
            original_result = None
            for result in results:
                if result.get("text", "") == reranked["text"]:
                    original_result = result.copy()
                    break
            
            if original_result:
                # Update with reranked score and rank
                original_result["reranked_score"] = reranked["score"]
                original_result["reranked_rank"] = reranked["rank"]
                original_result["original_score"] = original_result.get("score", 0.0)
                original_result["score"] = reranked["score"]  # Use reranked score as primary
                reranked_results.append(original_result)
        
        return reranked_results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return scores
        
        import numpy as np
        scores_array = np.array(scores)
        
        # Min-max normalization
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        
        if max_score > min_score:
            normalized = (scores_array - min_score) / (max_score - min_score)
        else:
            normalized = np.ones_like(scores_array)
        
        return normalized.tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model."""
        return {
            "model_name": self.config.model_name,
            "initialized": self._initialized,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "normalize_scores": self.config.normalize_scores,
            "device": self.config.device
        }
    
    def is_available(self) -> bool:
        """Check if reranker is available and initialized."""
        return self._initialized and self._model is not None
