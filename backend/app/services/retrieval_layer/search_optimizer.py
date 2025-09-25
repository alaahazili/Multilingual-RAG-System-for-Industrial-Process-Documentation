"""
Search Optimizer - Retrieval Layer Component

Provides search result optimization capabilities:
- Result deduplication and filtering
- Score boosting for technical content
- Context-aware result ranking
- Query expansion and refinement
"""

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for search optimization."""
    enable_deduplication: bool = True
    enable_score_boosting: bool = True
    enable_context_ranking: bool = True
    min_text_length: int = 50  # Minimum text length for valid results
    max_text_length: int = 2000  # Maximum text length for results
    duplicate_threshold: float = 0.9  # Similarity threshold for deduplication
    technical_boost_factor: float = 1.2  # Score boost for technical content


class SearchOptimizer:
    """Search optimizer for enhancing search result quality."""
    
    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        
        # Technical content patterns
        self.technical_patterns = [
            r'\b[A-Z]{2}-\d{3}-[A-Z]{3}\b',  # Equipment tags
            r'\d+\.\d+\s*[A-Za-z]+\b',  # Measurements
            r'\b[A-Za-z]+\s*=\s*[0-9A-Za-z+\-*/()\s.]+\b',  # Equations
            r'\b(flow|pressure|temperature|level|control|valve|pump|tank)\b',  # Technical terms
        ]
        
        # Section patterns for context
        self.section_patterns = [
            r'^\s*\d+\.\d+\.\d+\s+',  # Section numbers
            r'^\s*\d+\.\d+\s+',
            r'^\s*\d+\s+',
        ]
    
    def optimize_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Optimize search results with various enhancement techniques.
        
        Args:
            results: List of search results
            query: Original search query
            top_k: Number of results to return
            
        Returns:
            Optimized list of results
        """
        if not results:
            return results
        
        optimized_results = results.copy()
        
        # Step 1: Filter by text quality
        if self.config.enable_deduplication:
            optimized_results = self._filter_by_quality(optimized_results)
        
        # Step 2: Remove duplicates
        if self.config.enable_deduplication:
            optimized_results = self._remove_duplicates(optimized_results)
        
        # Step 3: Apply score boosting
        if self.config.enable_score_boosting:
            optimized_results = self._apply_score_boosting(optimized_results, query)
        
        # Step 4: Context-aware ranking
        if self.config.enable_context_ranking:
            optimized_results = self._apply_context_ranking(optimized_results, query)
        
        # Step 5: Sort by final scores
        optimized_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Step 6: Return top_k results
        return optimized_results[:top_k]
    
    def _filter_by_quality(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results by text quality criteria."""
        filtered_results = []
        
        for result in results:
            text = result.get("text", "")
            
            # Check text length
            if len(text) < self.config.min_text_length:
                continue
            
            if len(text) > self.config.max_text_length:
                # Truncate long texts
                result["text"] = text[:self.config.max_text_length] + "..."
            
            # Check for meaningful content
            if self._has_meaningful_content(text):
                filtered_results.append(result)
        
        logger.debug(f"Filtered {len(results)} -> {len(filtered_results)} results by quality")
        return filtered_results
    
    def _has_meaningful_content(self, text: str) -> bool:
        """Check if text has meaningful content."""
        # Remove whitespace and common words
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Check for technical content
        has_technical = any(re.search(pattern, text, re.IGNORECASE) for pattern in self.technical_patterns)
        
        # Check for section structure
        has_structure = any(re.search(pattern, text) for pattern in self.section_patterns)
        
        # Check minimum word count
        word_count = len(cleaned.split())
        
        return word_count >= 10 or has_technical or has_structure
    
    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or very similar results."""
        unique_results = []
        seen_texts: Set[str] = set()
        
        for result in results:
            text = result.get("text", "").strip()
            
            # Create a normalized version for comparison
            normalized_text = self._normalize_text_for_comparison(text)
            
            # Check if we've seen this text before
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                unique_results.append(result)
        
        logger.debug(f"Deduplicated {len(results)} -> {len(unique_results)} results")
        return unique_results
    
    def _normalize_text_for_comparison(self, text: str) -> str:
        """Normalize text for duplicate detection."""
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Take first 100 characters for comparison
        return normalized[:100]
    
    def _apply_score_boosting(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Apply score boosting for technical content and query relevance."""
        boosted_results = []
        
        for result in results:
            boosted_result = result.copy()
            text = result.get("text", "")
            original_score = result.get("score", 0.0)
            
            # Calculate boost factors
            technical_boost = self._calculate_technical_boost(text)
            query_relevance_boost = self._calculate_query_relevance_boost(text, query)
            
            # Apply boosts
            final_score = original_score * technical_boost * query_relevance_boost
            
            boosted_result["score"] = min(final_score, 1.0)  # Cap at 1.0
            boosted_result["technical_boost"] = technical_boost
            boosted_result["query_relevance_boost"] = query_relevance_boost
            
            boosted_results.append(boosted_result)
        
        return boosted_results
    
    def _calculate_technical_boost(self, text: str) -> float:
        """Calculate score boost based on technical content."""
        boost = 1.0
        
        # Count technical patterns
        technical_count = 0
        for pattern in self.technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technical_count += len(matches)
        
        # Apply boost based on technical content
        if technical_count > 0:
            boost = min(boost * (1 + technical_count * 0.1), self.config.technical_boost_factor)
        
        return boost
    
    def _calculate_query_relevance_boost(self, text: str, query: str) -> float:
        """Calculate score boost based on query relevance."""
        boost = 1.0
        
        # Convert to lowercase for comparison
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Split into words
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        
        # Calculate word overlap
        overlap = len(query_words.intersection(text_words))
        total_query_words = len(query_words)
        
        if total_query_words > 0:
            overlap_ratio = overlap / total_query_words
            boost = 1.0 + (overlap_ratio * 0.3)  # Up to 30% boost
        
        return boost
    
    def _apply_context_ranking(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Apply context-aware ranking based on section structure and metadata."""
        context_results = []
        
        for result in results:
            context_result = result.copy()
            text = result.get("text", "")
            
            # Check for section headers
            has_section_header = any(re.search(pattern, text) for pattern in self.section_patterns)
            
            # Check metadata for context
            section_title = result.get("section", "")
            document = result.get("document", "")
            
            # Apply context boost
            context_boost = 1.0
            if has_section_header:
                context_boost *= 1.1
            
            if section_title and any(word.lower() in section_title.lower() for word in query.split()):
                context_boost *= 1.15
            
            # Apply context boost to score
            context_result["score"] = min(context_result.get("score", 0.0) * context_boost, 1.0)
            context_result["context_boost"] = context_boost
            
            context_results.append(context_result)
        
        return context_results
    
    def get_optimization_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about optimization results."""
        if not results:
            return {}
        
        scores = [result.get("score", 0.0) for result in results]
        technical_boosts = [result.get("technical_boost", 1.0) for result in results]
        query_boosts = [result.get("query_relevance_boost", 1.0) for result in results]
        context_boosts = [result.get("context_boost", 1.0) for result in results]
        
        return {
            "total_results": len(results),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "avg_technical_boost": sum(technical_boosts) / len(technical_boosts) if technical_boosts else 1.0,
            "avg_query_boost": sum(query_boosts) / len(query_boosts) if query_boosts else 1.0,
            "avg_context_boost": sum(context_boosts) / len(context_boosts) if context_boosts else 1.0,
        }


