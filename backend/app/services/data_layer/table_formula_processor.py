"""
Table and Formula Processor - Data Layer Component

Specialized handling for:
- Table structure analysis and formatting
- Mathematical formula parsing and validation
- Technical specification extraction
- Engineering calculations preservation
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TableStructure:
    """Represents a structured table with metadata."""
    table_id: str
    headers: List[str]
    rows: List[List[str]]
    metadata: Dict
    table_type: str  # "specification", "calculation", "reference", etc.


@dataclass
class FormulaStructure:
    """Represents a mathematical formula with metadata."""
    formula_id: str
    expression: str
    variables: List[str]
    constants: List[str]
    formula_type: str  # "equation", "calculation", "specification", etc.
    units: Optional[str] = None


class TableFormulaProcessor:
    """Enhanced processor for tables and mathematical formulas."""
    
    def __init__(self):
        # Table type detection patterns
        self.table_patterns = {
            "specification": [
                r"Specification|Spec|Technical\s+Data|Parameters",
                r"Equipment|Device|Component|System",
                r"Dimensions|Size|Capacity|Rating"
            ],
            "calculation": [
                r"Calculation|Formula|Equation|Math",
                r"Result|Output|Computed|Derived"
            ],
            "reference": [
                r"Reference|Table|Chart|Data",
                r"Standard|Code|Regulation|Requirement"
            ]
        }
        
        # Formula type detection patterns
        self.formula_patterns = {
            "engineering": [
                r"[A-Za-z]+\s*=\s*[A-Za-z0-9+\-*/()\s.]+",  # Engineering equations
                r"Power|Force|Pressure|Temperature|Flow",  # Engineering terms
            ],
            "mathematical": [
                r"[A-Za-z]+\s*=\s*[0-9+\-*/()\s.]+",  # Mathematical equations
                r"Sum|Product|Average|Ratio|Percentage",  # Mathematical terms
            ],
            "specification": [
                r"\d+(?:\.\d+)?\s*[A-Za-z]+",  # Measurements with units
                r"Min|Max|Range|Limit|Threshold",  # Specification terms
            ]
        }
    
    def process_tables(self, tables: List[Dict]) -> List[TableStructure]:
        """Process and structure tables."""
        processed_tables = []
        
        for table in tables:
            try:
                table_data = table.get("data", [])
                if not table_data:
                    continue
                
                # Extract headers and rows
                headers = table_data[0] if table_data else []
                rows = table_data[1:] if len(table_data) > 1 else []
                
                # Determine table type
                table_type = self._classify_table(headers, rows)
                
                # Create structured table
                structured_table = TableStructure(
                    table_id=table.get("table_id", "unknown"),
                    headers=headers,
                    rows=rows,
                    metadata={
                        "position": table.get("position", {}),
                        "rows_count": len(rows),
                        "columns_count": len(headers),
                        "text_representation": table.get("text_representation", "")
                    },
                    table_type=table_type
                )
                
                processed_tables.append(structured_table)
                
            except Exception as e:
                logger.warning(f"Failed to process table: {e}")
                continue
        
        return processed_tables
    
    def process_formulas(self, formulas: List[Dict]) -> List[FormulaStructure]:
        """Process and structure mathematical formulas."""
        processed_formulas = []
        
        for formula in formulas:
            try:
                formula_text = formula.get("text", "")
                if not formula_text:
                    continue
                
                # Extract variables and constants
                variables = self._extract_variables(formula_text)
                constants = self._extract_constants(formula_text)
                
                # Determine formula type
                formula_type = self._classify_formula(formula_text)
                
                # Extract units if present
                units = self._extract_units(formula_text)
                
                # Create structured formula
                structured_formula = FormulaStructure(
                    formula_id=formula.get("formula_id", "unknown"),
                    expression=formula_text,
                    variables=variables,
                    constants=constants,
                    formula_type=formula_type,
                    units=units
                )
                
                processed_formulas.append(structured_formula)
                
            except Exception as e:
                logger.warning(f"Failed to process formula: {e}")
                continue
        
        return processed_formulas
    
    def _classify_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Classify table type based on headers and content."""
        header_text = " ".join(headers).lower()
        row_text = " ".join([" ".join(row) for row in rows[:3]]).lower()  # First 3 rows
        
        combined_text = f"{header_text} {row_text}"
        
        for table_type, patterns in self.table_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return table_type
        
        return "general"
    
    def _classify_formula(self, formula_text: str) -> str:
        """Classify formula type based on content."""
        formula_lower = formula_text.lower()
        
        for formula_type, patterns in self.formula_patterns.items():
            for pattern in patterns:
                if re.search(pattern, formula_lower, re.IGNORECASE):
                    return formula_type
        
        return "general"
    
    def _extract_variables(self, formula_text: str) -> List[str]:
        """Extract variables from formula."""
        # Look for single letters or words that could be variables
        variables = []
        
        # Single letter variables (common in math)
        single_letter_vars = re.findall(r'\b[A-Za-z]\b', formula_text)
        variables.extend(single_letter_vars)
        
        # Multi-letter variables (common in engineering)
        multi_letter_vars = re.findall(r'\b[A-Za-z]{2,}\b', formula_text)
        variables.extend(multi_letter_vars)
        
        # Remove common words that aren't variables
        non_variables = {'and', 'or', 'the', 'for', 'with', 'in', 'on', 'at', 'to', 'of', 'by'}
        variables = [var for var in variables if var.lower() not in non_variables]
        
        return list(set(variables))  # Remove duplicates
    
    def _extract_constants(self, formula_text: str) -> List[str]:
        """Extract numerical constants from formula."""
        # Find all numbers (including decimals)
        constants = re.findall(r'\d+(?:\.\d+)?', formula_text)
        return constants
    
    def _extract_units(self, formula_text: str) -> Optional[str]:
        """Extract units from formula."""
        # Common unit patterns
        unit_patterns = [
            r'\b(?:m|mm|cm|km|ft|in|yd)\b',  # Length units
            r'\b(?:kg|g|lb|ton)\b',  # Weight units
            r'\b(?:bar|Pa|kPa|MPa|psi)\b',  # Pressure units
            r'\b(?:°C|°F|K)\b',  # Temperature units
            r'\b(?:m³|L|gal|ft³)\b',  # Volume units
            r'\b(?:kW|W|hp)\b',  # Power units
            r'\b(?:rpm|Hz)\b',  # Speed/frequency units
        ]
        
        for pattern in unit_patterns:
            match = re.search(pattern, formula_text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def format_table_for_embedding(self, table: TableStructure) -> str:
        """Format table for embedding in text."""
        lines = []
        
        # Add table header
        lines.append(f"[TABLE: {table.table_type.upper()}]")
        
        # Add headers
        if table.headers:
            lines.append(" | ".join(table.headers))
            lines.append("-" * len(lines[-1]))
        
        # Add rows
        for row in table.rows:
            lines.append(" | ".join(row))
        
        lines.append("[/TABLE]")
        
        return "\n".join(lines)
    
    def format_formula_for_embedding(self, formula: FormulaStructure) -> str:
        """Format formula for embedding in text."""
        parts = []
        
        # Add formula type
        parts.append(f"[FORMULA: {formula.formula_type.upper()}]")
        
        # Add expression
        parts.append(formula.expression)
        
        # Add variables if present
        if formula.variables:
            parts.append(f"Variables: {', '.join(formula.variables)}")
        
        # Add units if present
        if formula.units:
            parts.append(f"Units: {formula.units}")
        
        parts.append("[/FORMULA]")
        
        return " | ".join(parts)
    
    def extract_technical_specifications(self, tables: List[TableStructure]) -> Dict:
        """Extract technical specifications from tables."""
        specifications = {}
        
        for table in tables:
            if table.table_type == "specification":
                for row in table.rows:
                    if len(row) >= 2:
                        parameter = row[0].strip()
                        value = row[1].strip()
                        specifications[parameter] = value
        
        return specifications
    
    def validate_formula_syntax(self, formula_text: str) -> Dict:
        """Validate mathematical formula syntax."""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check for balanced parentheses
        if formula_text.count('(') != formula_text.count(')'):
            validation_result["is_valid"] = False
            validation_result["errors"].append("Unbalanced parentheses")
        
        # Check for valid operators
        operators = re.findall(r'[+\-*/=]', formula_text)
        if not operators:
            validation_result["warnings"].append("No mathematical operators found")
        
        # Check for valid variable names
        variables = re.findall(r'\b[A-Za-z]+\b', formula_text)
        for var in variables:
            if len(var) == 1 and var.lower() in ['i', 'j', 'k', 'n', 'x', 'y', 'z']:
                continue  # Common mathematical variables
            if len(var) > 10:
                validation_result["warnings"].append(f"Long variable name: {var}")
        
        return validation_result


