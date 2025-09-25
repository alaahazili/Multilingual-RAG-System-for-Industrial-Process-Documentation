"""
Multilingual Processor - Embedding Layer Component

Provides multilingual support for English and French:
- Language detection and classification
- Content preprocessing for different languages
- Cross-language semantic matching
- Technical terminology preservation
"""

import re
from typing import List, Dict, Optional, Tuple, Any
import logging
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

logger = logging.getLogger(__name__)

# Set seed for consistent language detection
DetectorFactory.seed = 0


class MultilingualProcessor:
    """Handles multilingual content processing for embedding optimization."""
    
    def __init__(self):
        # Language-specific patterns
        self.language_patterns = {
            "en": {
                "technical_terms": [
                    r"specification", r"equipment", r"system", r"control",
                    r"pressure", r"temperature", r"flow", r"capacity",
                    r"operation", r"maintenance", r"procedure", r"requirement"
                ],
                "measurements": [
                    r"\d+\.\d+\s*(?:bar|psi|°C|°F|m³|L|gal|kW|hp|rpm)"
                ],
                "equipment_tags": [
                    r"[A-Z]{2}-\d{3}-[A-Z]{3}", r"[A-Z]{2}-\d{3}"
                ]
            },
            "fr": {
                "technical_terms": [
                    r"spécification", r"équipement", r"système", r"contrôle",
                    r"pression", r"température", r"débit", r"capacité",
                    r"opération", r"maintenance", r"procédure", r"exigence"
                ],
                "measurements": [
                    r"\d+\.\d+\s*(?:bar|psi|°C|°F|m³|L|gal|kW|ch|tr/min)"
                ],
                "equipment_tags": [
                    r"[A-Z]{2}-\d{3}-[A-Z]{3}", r"[A-Z]{2}-\d{3}"
                ]
            }
        }
        
        # Common technical terms across languages
        self.common_technical_terms = [
            r"PID", r"PLC", r"SCADA", r"HMI", r"RTU", r"DCS",
            r"ASME", r"API", r"ISO", r"ASTM", r"DIN", r"BS",
            r"MPa", r"GPa", r"N/m²", r"Pa", r"ft", r"in",
        ]
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        try:
            # Clean text for better detection
            clean_text = self._clean_for_detection(text)
            
            if not clean_text:
                return "en"  # Default to English
            
            detected_lang = detect(clean_text)
            
            # Map to supported languages
            if detected_lang in ["en", "fr"]:
                return detected_lang
            else:
                # Default to English for unsupported languages
                return "en"
                
        except LangDetectException:
            logger.warning("Language detection failed, defaulting to English")
            return "en"
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return "en"
    
    def _clean_for_detection(self, text: str) -> str:
        """Clean text for language detection."""
        # Remove technical terms and numbers that might confuse detection
        cleaned = re.sub(r'[A-Z]{2}-\d{3}', '', text)
        cleaned = re.sub(r'\d+\.\d+\s*[A-Za-z]+', '', cleaned)
        cleaned = re.sub(r'[A-Z]{2,}', '', cleaned)  # Remove all caps terms
        
        # Keep only alphabetic characters and spaces
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def process_content(self, text: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Process content for multilingual embedding optimization."""
        if language is None:
            language = self.detect_language(text)
        
        processed = {
            "original_text": text,
            "language": language,
            "technical_terms": self._extract_technical_terms(text, language),
            "measurements": self._extract_measurements(text, language),
            "equipment_tags": self._extract_equipment_tags(text),
            "common_terms": self._extract_common_terms(text),
            "processed_text": self._preprocess_for_embedding(text, language),
        }
        
        return processed
    
    def _extract_technical_terms(self, text: str, language: str) -> List[str]:
        """Extract language-specific technical terms."""
        terms = []
        
        # Language-specific terms
        if language in self.language_patterns:
            for pattern in self.language_patterns[language]["technical_terms"]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                terms.extend(matches)
        
        # Common technical terms
        for pattern in self.common_technical_terms:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend(matches)
        
        return list(set(terms))  # Remove duplicates
    
    def _extract_measurements(self, text: str, language: str) -> List[str]:
        """Extract measurements with units."""
        measurements = []
        
        if language in self.language_patterns:
            for pattern in self.language_patterns[language]["measurements"]:
                matches = re.findall(pattern, text)
                measurements.extend(matches)
        
        return list(set(measurements))
    
    def _extract_equipment_tags(self, text: str) -> List[str]:
        """Extract equipment tags (language-independent)."""
        tags = []
        
        for pattern in [r"[A-Z]{2}-\d{3}-[A-Z]{3}", r"[A-Z]{2}-\d{3}"]:
            matches = re.findall(pattern, text)
            tags.extend(matches)
        
        return list(set(tags))
    
    def _extract_common_terms(self, text: str) -> List[str]:
        """Extract common technical terms across languages."""
        terms = []
        
        for pattern in self.common_technical_terms:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend(matches)
        
        return list(set(terms))
    
    def _preprocess_for_embedding(self, text: str, language: str) -> str:
        """Preprocess text for better embedding quality."""
        processed = text
        
        # Normalize whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Preserve technical terms
        processed = self._preserve_technical_terms(processed, language)
        
        # Normalize measurements
        processed = self._normalize_measurements(processed)
        
        return processed.strip()
    
    def _preserve_technical_terms(self, text: str, language: str) -> str:
        """Preserve technical terms for better semantic matching."""
        # Note: TECH tags removed to avoid artifacts in final output
        # Technical terms are still preserved but without special markers
        return text
    
    def _normalize_measurements(self, text: str) -> str:
        """Normalize measurement formats."""
        # Normalize spacing around units
        text = re.sub(r'(\d+\.?\d*)\s*([A-Za-z]+)', r'\1 \2', text)
        
        # Standardize common units
        unit_mappings = {
            r'\bdeg\b': '°',
            r'\bdegC\b': '°C',
            r'\bdegF\b': '°F',
            r'\bcubic\s+meters?\b': 'm³',
            r'\bliters?\b': 'L',
            r'\bkilowatts?\b': 'kW',
            r'\bhorsepower\b': 'hp',
        }
        
        for pattern, replacement in unit_mappings.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def batch_process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple documents for multilingual support."""
        processed_docs = []
        
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue
            
            # Process content
            processed_content = self.process_content(content)
            
            # Update document with multilingual information
            processed_doc = doc.copy()
            processed_doc.update(processed_content)
            
            processed_docs.append(processed_doc)
        
        return processed_docs
    
    def get_language_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about language distribution."""
        language_counts = {}
        total_docs = len(documents)
        
        for doc in documents:
            language = doc.get("language", "en")
            language_counts[language] = language_counts.get(language, 0) + 1
        
        return {
            "total_documents": total_docs,
            "language_distribution": language_counts,
            "supported_languages": ["en", "fr"],
            "primary_language": max(language_counts.items(), key=lambda x: x[1])[0] if language_counts else "en"
        }
    
    def create_cross_language_query(self, query: str, target_language: str) -> str:
        """Create a cross-language query for better matching."""
        # This is a simple approach - in production, you might use translation APIs
        query_language = self.detect_language(query)
        
        if query_language == target_language:
            return query
        
        # For now, return the original query with language markers
        # In a full implementation, you would translate the query
        return f"[{query_language}->{target_language}] {query}"

