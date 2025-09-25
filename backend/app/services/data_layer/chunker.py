"""
Document Chunker - Data Layer Component

Simplified chunker that works with HTML processor.
Handles dynamic chunking with:
- Structure-based chunking (~512-1024 tokens per chunk)
- Overlap management (100-200 tokens)
- Metadata preservation (section numbers, titles, page numbers)
- Content-aware boundaries
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for document chunking."""
    target_tokens: int = 768  # Target tokens per chunk
    overlap_tokens: int = 150  # Overlap between chunks
    min_chunk_tokens: int = 200  # Minimum chunk size
    max_chunk_tokens: int = 1024  # Maximum chunk size
    preserve_sections: bool = True  # Preserve section boundaries
    include_metadata: bool = True  # Include metadata in chunks


@dataclass
class DocumentChunk:
    """Represents a single document chunk with metadata."""
    content: str
    chunk_id: str
    page_start: int
    page_end: int
    section_title: Optional[str] = None
    section_number: Optional[str] = None
    section_level: int = 0
    metadata: Dict = None
    token_count: int = 0
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
        
        # Add technical content detection
        self.metadata.update({
            "has_equations": self._detect_equations(),
            "has_tables": self._detect_tables(),
            "equipment_tags": self._extract_equipment_tags(),
            "technical_terms": self._extract_technical_terms(),
            "measurements": self._extract_measurements(),
            "language": self._detect_language()
        })
    
    def _detect_equations(self) -> bool:
        """Detect if chunk contains mathematical equations."""
        equation_patterns = [
            r'[A-Za-z]\s*=\s*[0-9A-Za-z+\-*/()\s.]+',  # Basic equations
            r'[0-9]+\s*[+\-*/]\s*[0-9]+',  # Arithmetic
            r'[A-Za-z]+\s*[+\-*/]\s*[A-Za-z]+',  # Variable operations
        ]
        return any(re.search(pattern, self.content) for pattern in equation_patterns)
    
    def _detect_tables(self) -> bool:
        """Detect if chunk contains table-like content."""
        table_patterns = [
            r'\|\s*[^|]+\s*\|',  # Pipe-separated tables
            r'\s{3,}[A-Za-z]+\s{3,}',  # Space-separated columns
            r'[A-Za-z]+\s+[0-9]+\s+[A-Za-z]+',  # Data rows
        ]
        return any(re.search(pattern, self.content) for pattern in table_patterns)
    
    def _extract_equipment_tags(self) -> List[str]:
        """Extract equipment tags from content."""
        equipment_patterns = [
            r'\b[A-Z]{2}-\d{3}-[A-Z]{3}\b',  # XX-123-XXX format
            r'\b[A-Z]{2}-\d{3}\b',  # XX-123 format
            r'\b[A-Z]{2}\d{3}\b',  # XX123 format
        ]
        tags = []
        for pattern in equipment_patterns:
            tags.extend(re.findall(pattern, self.content))
        return list(set(tags))
    
    def _extract_technical_terms(self) -> List[str]:
        """Extract technical terms from content."""
        technical_terms = [
            'flow', 'pressure', 'temperature', 'level', 'control', 'valve', 
            'pump', 'tank', 'pipeline', 'station', 'capacity', 'efficiency',
            'specification', 'parameter', 'setting', 'operation', 'maintenance'
        ]
        found_terms = []
        content_lower = self.content.lower()
        for term in technical_terms:
            if term in content_lower:
                found_terms.append(term)
        return found_terms
    
    def _extract_measurements(self) -> List[str]:
        """Extract measurement units and values."""
        measurement_patterns = [
            r'\d+\.?\d*\s*(bar|psi|pa|kpa|mpa|°c|°f|k|m³/h|l/min|gpm|m|ft|mm|in)',
            r'\d+\.?\d*\s*(percent|%|ppm|ppb)',
            r'\d+\.?\d*\s*(volts|v|amperes|a|watts|w|hertz|hz)'
        ]
        measurements = []
        for pattern in measurement_patterns:
            measurements.extend(re.findall(pattern, self.content, re.IGNORECASE))
        return list(set(measurements))
    
    def _detect_language(self) -> str:
        """Simple language detection based on common words."""
        english_words = ['the', 'and', 'or', 'for', 'with', 'from', 'this', 'that']
        french_words = ['le', 'la', 'les', 'et', 'ou', 'pour', 'avec', 'de', 'ce', 'cette']
        
        content_lower = self.content.lower()
        english_count = sum(1 for word in english_words if word in content_lower)
        french_count = sum(1 for word in french_words if word in content_lower)
        
        if french_count > english_count:
            return 'french'
        else:
            return 'english'


class DocumentChunker:
    """Main document chunking class."""
    
    def __init__(self, config: ChunkConfig = None):
        """Initialize the chunker with configuration."""
        self.config = config or ChunkConfig()
        logger.info(f"Initialized DocumentChunker with config: {self.config}")
    
    def chunk_document(self, document_content: str, metadata: Dict = None) -> List[DocumentChunk]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            document_content: The full document text
            metadata: Additional metadata for the document
            
        Returns:
            List of DocumentChunk objects
        """
        if metadata is None:
            metadata = {}
        
        logger.info(f"Starting document chunking. Content length: {len(document_content)}")
        
        # Split content into chunks
        chunks = self._split_content(document_content)
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk_content in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_id": f"chunk_{i+1}"
            })
            
            chunk = DocumentChunk(
                content=chunk_content,
                chunk_id=f"chunk_{i+1}",
                page_start=metadata.get("page_start", 1),
                page_end=metadata.get("page_end", 1),
                section_title=metadata.get("section_title", ""),
                section_number=metadata.get("section_number", ""),
                section_level=metadata.get("section_level", 0),
                metadata=chunk_metadata,
                token_count=len(chunk_content.split())
            )
            document_chunks.append(chunk)
        
        logger.info(f"Created {len(document_chunks)} chunks")
        return document_chunks
    
    def _split_content(self, content: str) -> List[str]:
        """Split content into chunks based on configuration."""
        words = content.split()
        chunks = []
        
        if len(words) <= self.config.max_chunk_tokens:
            # Content fits in single chunk
            chunks.append(content)
        else:
            # Split into multiple chunks with overlap
            start = 0
            while start < len(words):
                end = min(start + self.config.target_tokens, len(words))
                chunk_words = words[start:end]
                chunks.append(' '.join(chunk_words))
                
                # Calculate next start position with overlap
                overlap_start = max(0, end - self.config.overlap_tokens)
                start = overlap_start if start < overlap_start else end
        
        return chunks
    
    def get_chunking_stats(self) -> Dict[str, Any]:
        """Get chunking statistics and configuration."""
        return {
            "chunker_type": "DocumentChunker",
            "config": {
                "target_tokens": self.config.target_tokens,
                "overlap_tokens": self.config.overlap_tokens,
                "min_chunk_tokens": self.config.min_chunk_tokens,
                "max_chunk_tokens": self.config.max_chunk_tokens,
                "preserve_sections": self.config.preserve_sections,
                "include_metadata": self.config.include_metadata
            },
            "features": [
                "Dynamic token-based chunking",
                "Configurable overlap",
                "Metadata preservation",
                "Technical content detection",
                "Equipment tag extraction",
                "Measurement extraction",
                "Language detection"
            ]
        }
