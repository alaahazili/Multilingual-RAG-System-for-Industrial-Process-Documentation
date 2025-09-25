"""
Metadata Filter - Retrieval Layer Component

Provides metadata filtering capabilities for search results:
- Filter by document, section, language, facility
- Filter by technical content (equipment tags, equations, tables)
- Complex filter combinations
- Query filter generation for Qdrant
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for metadata filtering."""
    enable_document_filtering: bool = True
    enable_section_filtering: bool = True
    enable_language_filtering: bool = True
    enable_facility_filtering: bool = True
    enable_technical_filtering: bool = True
    strict_filtering: bool = False  # Use AND instead of OR for multiple filters


class MetadataFilter:
    """Metadata filter for creating complex search filters."""
    
    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
    
    def create_filter(
        self,
        document: Optional[str] = None,
        section: Optional[str] = None,
        language: Optional[str] = None,
        facility: Optional[str] = None,
        equipment_tags: Optional[List[str]] = None,
        has_equations: Optional[bool] = None,
        has_tables: Optional[bool] = None,
        technical_terms: Optional[List[str]] = None,
        document_type: Optional[str] = None,
        page_range: Optional[tuple] = None,
    ) -> Optional[Filter]:
        """
        Create a Qdrant filter based on metadata criteria.
        
        Args:
            document: Filter by specific document name
            section: Filter by section title or number
            language: Filter by language (en/fr)
            facility: Filter by facility name
            equipment_tags: Filter by equipment tags
            has_equations: Filter by presence of equations
            has_tables: Filter by presence of tables
            document_type: Filter by document type
            page_range: Filter by page range (start_page, end_page)
            
        Returns:
            Qdrant Filter object or None if no filters
        """
        conditions = []
        
        # Document filtering
        if self.config.enable_document_filtering and document:
            conditions.append(
                FieldCondition(key="document", match=MatchValue(value=document))
            )
        
        # Section filtering
        if self.config.enable_section_filtering and section:
            conditions.append(
                FieldCondition(key="section", match=MatchValue(value=section))
            )
        
        # Language filtering
        if self.config.enable_language_filtering and language:
            conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language))
            )
        
        # Facility filtering
        if self.config.enable_facility_filtering and facility:
            conditions.append(
                FieldCondition(key="facility", match=MatchValue(value=facility))
            )
        
        # Document type filtering
        if document_type:
            conditions.append(
                FieldCondition(key="document_type", match=MatchValue(value=document_type))
            )
        
        # Equipment tags filtering
        if self.config.enable_technical_filtering and equipment_tags:
            conditions.append(
                FieldCondition(
                    key="equipment_tags",
                    match=MatchAny(any=equipment_tags)
                )
            )
        
        # Equations filtering
        if self.config.enable_technical_filtering and has_equations is not None:
            conditions.append(
                FieldCondition(
                    key="has_equations",
                    match=MatchValue(value=has_equations)
                )
            )
        
        # Tables filtering
        if self.config.enable_technical_filtering and has_tables is not None:
            conditions.append(
                FieldCondition(
                    key="has_tables",
                    match=MatchValue(value=has_tables)
                )
            )
        
        # Technical terms filtering
        if self.config.enable_technical_filtering and technical_terms:
            conditions.append(
                FieldCondition(
                    key="technical_terms",
                    match=MatchAny(any=technical_terms)
                )
            )
        
        # Page range filtering
        if page_range and len(page_range) == 2:
            start_page, end_page = page_range
            if start_page is not None:
                conditions.append(
                    FieldCondition(
                        key="page_start",
                        range={"gte": start_page}
                    )
                )
            if end_page is not None:
                conditions.append(
                    FieldCondition(
                        key="page_end",
                        range={"lte": end_page}
                    )
                )
        
        # Create filter based on conditions
        if not conditions:
            return None
        
        if self.config.strict_filtering:
            # Use AND logic (all conditions must match)
            return Filter(must=conditions)
        else:
            # Use OR logic (any condition can match)
            return Filter(should=conditions)
    
    def create_document_filter(self, document_name: str) -> Filter:
        """Create a filter for a specific document."""
        return Filter(
            must=[FieldCondition(key="document", match=MatchValue(value=document_name))]
        )
    
    def create_section_filter(self, section_name: str) -> Filter:
        """Create a filter for a specific section."""
        return Filter(
            must=[FieldCondition(key="section", match=MatchValue(value=section_name))]
        )
    
    def create_language_filter(self, language: str) -> Filter:
        """Create a filter for a specific language."""
        return Filter(
            must=[FieldCondition(key="language", match=MatchValue(value=language))]
        )
    
    def create_facility_filter(self, facility_name: str) -> Filter:
        """Create a filter for a specific facility."""
        return Filter(
            must=[FieldCondition(key="facility", match=MatchValue(value=facility_name))]
        )
    
    def create_technical_filter(
        self,
        equipment_tags: Optional[List[str]] = None,
        has_equations: Optional[bool] = None,
        has_tables: Optional[bool] = None
    ) -> Filter:
        """Create a filter for technical content."""
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
        
        if not conditions:
            return Filter(must=[])
        
        return Filter(must=conditions)
    
    def create_combined_filter(
        self,
        filters: List[Filter],
        use_and_logic: bool = True
    ) -> Filter:
        """
        Combine multiple filters with AND or OR logic.
        
        Args:
            filters: List of Filter objects to combine
            use_and_logic: True for AND logic, False for OR logic
            
        Returns:
            Combined Filter object
        """
        if not filters:
            return Filter(must=[])
        
        if len(filters) == 1:
            return filters[0]
        
        # Extract conditions from all filters
        all_conditions = []
        for filter_obj in filters:
            if hasattr(filter_obj, 'must') and filter_obj.must:
                all_conditions.extend(filter_obj.must)
            if hasattr(filter_obj, 'should') and filter_obj.should:
                all_conditions.extend(filter_obj.should)
        
        if not all_conditions:
            return Filter(must=[])
        
        if use_and_logic:
            return Filter(must=all_conditions)
        else:
            return Filter(should=all_conditions)
    
    def validate_filter(self, filter_obj: Filter) -> bool:
        """Validate if a filter object is properly constructed."""
        try:
            # Check if filter has valid structure
            if not hasattr(filter_obj, 'must') and not hasattr(filter_obj, 'should'):
                return False
            
            # Check if conditions are valid
            conditions = []
            if hasattr(filter_obj, 'must') and filter_obj.must:
                conditions.extend(filter_obj.must)
            if hasattr(filter_obj, 'should') and filter_obj.should:
                conditions.extend(filter_obj.should)
            
            for condition in conditions:
                if not hasattr(condition, 'key') or not hasattr(condition, 'match'):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Filter validation failed: {e}")
            return False
    
    def get_filter_info(self, filter_obj: Filter) -> Dict[str, Any]:
        """Get information about a filter object."""
        if not filter_obj:
            return {"type": "none", "conditions": 0}
        
        conditions = []
        if hasattr(filter_obj, 'must') and filter_obj.must:
            conditions.extend(filter_obj.must)
        if hasattr(filter_obj, 'should') and filter_obj.should:
            conditions.extend(filter_obj.should)
        
        filter_info = {
            "type": "must" if hasattr(filter_obj, 'must') and filter_obj.must else "should",
            "conditions": len(conditions),
            "condition_details": []
        }
        
        for condition in conditions:
            condition_info = {
                "key": getattr(condition, 'key', 'unknown'),
                "match_type": type(condition.match).__name__ if hasattr(condition, 'match') else 'unknown'
            }
            filter_info["condition_details"].append(condition_info)
        
        return filter_info
