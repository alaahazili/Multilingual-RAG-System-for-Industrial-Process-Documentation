"""
Data Layer - Document Processing Module

This module handles:
- HTML parsing and extraction
- Text cleaning and normalization
- Dynamic chunking with metadata
- Structure detection and preservation
- Table and formula processing
"""

from .html_processor import HTMLProcessor
from .chunker import DocumentChunker
from .table_formula_processor import TableFormulaProcessor

__all__ = [
    "HTMLProcessor",
    "DocumentChunker",
    "TableFormulaProcessor"
]
