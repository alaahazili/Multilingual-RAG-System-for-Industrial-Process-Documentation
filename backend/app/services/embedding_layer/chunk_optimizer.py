"""
Chunk Optimizer - Embedding Layer Component

Optimizes document chunks for better semantic matching:
- 200-500 tokens max per chunk (optimized for technical docs)
- 120 token overlap for enhanced context continuity
- Semantic boundary detection
- Technical content preservation
- Cross-reference handling
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for optimized chunking."""
    min_tokens: int = 200
    max_tokens: int = 500
    target_tokens: int = 400  # Optimized for technical docs
    overlap_tokens: int = 120  # Enhanced context continuity
    preserve_semantic_boundaries: bool = True
    preserve_technical_content: bool = True
    preserve_cross_references: bool = True


class ChunkOptimizer:
    """Optimizes chunks for better semantic matching and embedding quality."""
    
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        
        # Semantic boundary patterns
        self.semantic_boundaries = [
            r'^\s*\d+\.\d+\.\d+\s+',  # Section numbers
            r'^\s*\d+\.\d+\s+',
            r'^\s*\d+\s+',
            r'^\s*[A-Z]\.\d+\s+',
            r'^\s*[A-Z][a-z]+:',  # Labels like "Note:", "Warning:"
            r'^\s*[A-Z][A-Z\s]+$',  # All caps headers
        ]
        
        # Technical content patterns
        self.technical_patterns = [
            r'[A-Z]{2}-\d{3}-[A-Z]{3}',  # Equipment tags
            r'\d+\.\d+\s*[A-Za-z]+',  # Measurements
            r'[A-Za-z]+\s*=\s*[0-9A-Za-z+\-*/()\s.]+',  # Equations
            r'\|.*\|',  # Tables
        ]
        
        # Cross-reference patterns
        self.cross_ref_patterns = [
            r'see\s+section\s+\d+\.\d+',
            r'refer\s+to\s+table\s+\d+',
            r'as\s+shown\s+in\s+figure\s+\d+',
            r'according\s+to\s+specification\s+\d+',
        ]
    
    def optimize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Optimize existing chunks for better semantic matching."""
        optimized_chunks = []
        
        for chunk in chunks:
            content = chunk.get("content", "")
            if not content:
                continue
            
            # Split into optimal sub-chunks if needed
            sub_chunks = self._split_into_optimal_chunks(content, chunk)
            optimized_chunks.extend(sub_chunks)
        
        return optimized_chunks
    
    def _split_into_optimal_chunks(self, content: str, original_chunk: Dict) -> List[Dict]:
        """Split content into optimally sized chunks."""
        words = content.split()
        
        if len(words) <= self.config.max_tokens:
            # Content is already within optimal range
            return [self._create_optimized_chunk(content, original_chunk)]
        
        chunks = []
        start = 0
        
        while start < len(words):
            # Calculate end position
            end = min(start + self.config.target_tokens, len(words))
            
            # Find optimal break point
            if end < len(words):
                end = self._find_optimal_break_point(words, start, end)
            
            # Extract chunk
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            # Ensure minimum size
            if len(chunk_words) >= self.config.min_tokens:
                optimized_chunk = self._create_optimized_chunk(chunk_text, original_chunk)
                optimized_chunk["chunk_index"] = len(chunks)
                optimized_chunk["total_chunks"] = (len(words) + self.config.target_tokens - 1) // self.config.target_tokens
                chunks.append(optimized_chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.config.overlap_tokens)
            
            # Prevent infinite loop
            if start >= len(words):
                break
        
        return chunks
    
    def _find_optimal_break_point(self, words: List[str], start: int, end: int) -> int:
        """Find the optimal break point for semantic coherence."""
        # Look for semantic boundaries first
        for i in range(end - 1, start, -1):
            if self._is_semantic_boundary(words[i]):
                return i
        
        # Look for sentence endings
        for i in range(end - 1, start, -1):
            if words[i].endswith(('.', '!', '?')):
                return i + 1
        
        # Look for paragraph breaks
        for i in range(end - 1, start, -1):
            if words[i] == '\n\n':
                return i + 1
        
        # Look for technical content boundaries
        for i in range(end - 1, start, -1):
            if self._is_technical_boundary(words[i]):
                return i
        
        # Default to original end
        return end
    
    def _is_semantic_boundary(self, word: str) -> bool:
        """Check if a word represents a semantic boundary."""
        for pattern in self.semantic_boundaries:
            if re.match(pattern, word):
                return True
        return False
    
    def _is_technical_boundary(self, word: str) -> bool:
        """Check if a word represents a technical content boundary."""
        # Check for technical patterns
        for pattern in self.technical_patterns:
            if re.search(pattern, word):
                return True
        
        # Check for cross-references
        for pattern in self.cross_ref_patterns:
            if re.search(pattern, word, re.IGNORECASE):
                return True
        
        return False
    
    def _create_optimized_chunk(self, content: str, original_chunk: Dict) -> Dict:
        """Create an optimized chunk with enhanced metadata."""
        optimized_chunk = original_chunk.copy()
        optimized_chunk["content"] = content
        optimized_chunk["token_count"] = len(content.split())
        optimized_chunk["optimized"] = True
        
        # Add semantic analysis
        optimized_chunk["semantic_features"] = self._extract_semantic_features(content)
        
        # Add technical content analysis
        optimized_chunk["technical_content"] = self._extract_technical_content(content)
        
        # Add cross-reference analysis
        optimized_chunk["cross_references"] = self._extract_cross_references(content)
        
        return optimized_chunk
    
    def _extract_semantic_features(self, content: str) -> Dict:
        """Extract semantic features from content."""
        features = {
            "has_section_numbers": bool(re.search(r'\d+\.\d+', content)),
            "has_equipment_tags": bool(re.search(r'[A-Z]{2}-\d{3}', content)),
            "has_measurements": bool(re.search(r'\d+\.\d+\s*[A-Za-z]+', content)),
            "has_equations": bool(re.search(r'[A-Za-z]+\s*=\s*[0-9A-Za-z+\-*/()\s.]+', content)),
            "has_tables": bool(re.search(r'\|.*\|', content)),
        }
        return features
    
    def _extract_technical_content(self, content: str) -> Dict:
        """Extract technical content information."""
        technical_content = {
            "equipment_tags": re.findall(r'[A-Z]{2}-\d{3}-[A-Z]{3}', content),
            "measurements": re.findall(r'\d+\.\d+\s*[A-Za-z]+', content),
            "equations": re.findall(r'[A-Za-z]+\s*=\s*[0-9A-Za-z+\-*/()\s.]+', content),
            "units": re.findall(r'\b[A-Za-z]+\b', content),  # Basic unit detection
        }
        return technical_content
    
    def _extract_cross_references(self, content: str) -> List[str]:
        """Extract cross-references from content."""
        cross_refs = []
        for pattern in self.cross_ref_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            cross_refs.extend(matches)
        return cross_refs
    
    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """Get statistics about optimized chunks."""
        if not chunks:
            return {}
        
        token_counts = [chunk.get("token_count", 0) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "optimized_chunks": len([c for c in chunks if c.get("optimized", False)]),
            "semantic_boundaries": len([c for c in chunks if c.get("semantic_features", {}).get("has_section_numbers", False)]),
            "technical_content": len([c for c in chunks if any(c.get("technical_content", {}).values())]),
        }
