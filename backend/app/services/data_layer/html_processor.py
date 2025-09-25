"""
Enhanced Data Layer Pipeline - HTML PROCESSOR VERSION
Features:
- Process single HTML file instead of multiple PDFs
- CORRECT section detection based on HTML structure (h1, h2, h3, h4 tags)
- FIXED equipment tag extraction from HTML content
- IMPROVED technical term extraction with case handling
- ACCURATE section hierarchy and metadata
- Save to JSON for later embedding layer processing
"""
import asyncio
import sys
import os
import logging
import traceback
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import tiktoken
from bs4 import BeautifulSoup

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from dotenv import load_dotenv
load_dotenv()


def setup_comprehensive_logging():
    """Setup comprehensive logging system for HTML Data Layer"""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure comprehensive logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),
            # Main log file
            logging.FileHandler(f"logs/enhanced_data_layer_html_{timestamp}.log", mode='w', encoding='utf-8'),
            # Error-only log file
            logging.FileHandler(f"logs/enhanced_data_layer_html_errors_{timestamp}.log", mode='w', encoding='utf-8')
        ]
    )
    
    # Create specific loggers
    main_logger = logging.getLogger('enhanced_data_layer_html_main')
    html_logger = logging.getLogger('enhanced_data_layer_html_parser')
    section_logger = logging.getLogger('enhanced_data_layer_html_section')
    chunking_logger = logging.getLogger('enhanced_data_layer_html_chunking')
    error_logger = logging.getLogger('enhanced_data_layer_html_errors')
    
    # Set levels
    main_logger.setLevel(logging.INFO)
    html_logger.setLevel(logging.DEBUG)
    section_logger.setLevel(logging.DEBUG)
    chunking_logger.setLevel(logging.DEBUG)
    error_logger.setLevel(logging.ERROR)
    
    return main_logger, html_logger, section_logger, chunking_logger, error_logger


def log_error_details(error_logger: logging.Logger, stage: str, error: Exception, context: Dict = None):
    """Log detailed error information"""
    error_logger.error(f"=== ERROR IN {stage.upper()} ===")
    error_logger.error(f"Error Type: {type(error).__name__}")
    error_logger.error(f"Error Message: {str(error)}")
    error_logger.error(f"Context: {context or 'None'}")
    error_logger.error(f"Stack Trace:")
    error_logger.error(traceback.format_exc())
    error_logger.error("=" * 50)


def extract_document_metadata(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract document metadata from HTML"""
    metadata = {
        "title": "",
        "document_number": "",
        "revision": "",
        "issue_date": "",
        "project": "",
        "client": ""
    }
    
    try:
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()
        
        # Extract document info from table
        doc_info_table = soup.find('table')
        if doc_info_table:
            rows = doc_info_table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    key = cells[0].get_text().strip().lower()
                    value = cells[1].get_text().strip()
                    
                    if "document no" in key:
                        metadata["document_number"] = value
                    elif "revision" in key:
                        metadata["revision"] = value
                    elif "issue date" in key:
                        metadata["issue_date"] = value
                    elif "project" in key:
                        metadata["project"] = value
                    elif "client" in key:
                        metadata["client"] = value
        
    except Exception as e:
        logging.warning(f"Failed to extract document metadata: {e}")
    
    return metadata


def detect_section_hierarchy_html(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """
    HTML FIXED: Section detection based on HTML structure
    - Uses h1, h2, h3, h4 tags for section hierarchy
    - Extracts section numbers and titles
    - Preserves content structure
    """
    sections = []
    
    try:
        # Find all heading tags
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        
        for i, heading in enumerate(headings):
            # Get section number from span with class "section-number"
            section_number_span = heading.find('span', class_='section-number')
            section_number = ""
            if section_number_span:
                section_number = section_number_span.get_text().strip().rstrip('.')
            else:
                # Try to extract from heading text
                heading_text = heading.get_text().strip()
                number_match = re.match(r'^(\d+(?:\.\d+)*)', heading_text)
                if number_match:
                    section_number = number_match.group(1)
            
            # Get section title (remove section number if present)
            section_title = heading.get_text().strip()
            if section_number:
                # Remove section number from title
                title_without_number = re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', section_title)
                section_title = title_without_number.strip()
            
            # Determine section level based on heading tag
            tag_name = heading.name
            if tag_name == 'h1':
                section_level = 1
            elif tag_name == 'h2':
                section_level = 2
            elif tag_name == 'h3':
                section_level = 3
            elif tag_name == 'h4':
                section_level = 4
            else:
                section_level = 1
            
            # Get section content (everything until next heading of same or higher level)
            content_elements = []
            current_element = heading.next_sibling
            
            while current_element:
                if hasattr(current_element, 'name'):
                    if current_element.name in ['h1', 'h2', 'h3', 'h4']:
                        # Check if this is a heading of same or higher level
                        if current_element.name == tag_name or (
                            tag_name == 'h2' and current_element.name == 'h1' or
                            tag_name == 'h3' and current_element.name in ['h1', 'h2'] or
                            tag_name == 'h4' and current_element.name in ['h1', 'h2', 'h3']
                        ):
                            break
                
                content_elements.append(current_element)
                current_element = current_element.next_sibling
            
            # Convert content elements to text
            content_text = ""
            for element in content_elements:
                if hasattr(element, 'get_text'):
                    content_text += element.get_text() + "\n"
                elif isinstance(element, str):
                    content_text += element + "\n"
            
            # Clean up content
            content_text = re.sub(r'\n\s*\n', '\n\n', content_text)
            content_text = content_text.strip()
            
            if content_text:  # Only add sections with content
                section_data = {
                    'number': section_number,
                    'title': section_title,
                    'level': section_level,
                    'tag': tag_name,
                    'content': content_text,
                    'id': heading.get('id', ''),
                    'subsections': []
                }
                sections.append(section_data)
        
    except Exception as e:
        logging.error(f"Error in section detection: {e}")
        traceback.print_exc()
    
    return sections


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (GPT-4 tokenizer)"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4


def split_text_with_overlap(text: str, max_tokens: int = 500, overlap_tokens: int = 120) -> List[str]:
    """
    Split text into chunks with overlap, preserving tables and formulas
    EXACTLY: Split if >500 tokens, with 120-token overlap
    """
    chunks = []
    
    # If text is small enough, return as single chunk
    text_tokens = count_tokens(text)
    if text_tokens <= max_tokens:
        return [text]
    
    # Split by sentences first to avoid breaking mid-sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # If adding this sentence would exceed limit
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            if overlap_tokens > 0:
                # Find overlap from end of previous chunk
                overlap_text = ""
                overlap_count = 0
                for prev_sentence in reversed(sentences[:sentences.index(sentence)]):
                    if overlap_count + count_tokens(prev_sentence) <= overlap_tokens:
                        overlap_text = prev_sentence + " " + overlap_text
                        overlap_count += count_tokens(prev_sentence)
                    else:
                        break
                
                current_chunk = overlap_text + sentence
                current_tokens = overlap_count + sentence_tokens
            else:
                current_chunk = sentence
                current_tokens = sentence_tokens
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def detect_language(text: str) -> str:
    """Detect language of text"""
    # Simple language detection based on common words
    french_words = ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'est', 'sont', 'pour', 'avec', 'dans', 'sur']
    english_words = ['the', 'and', 'for', 'with', 'in', 'on', 'at', 'to', 'of', 'a', 'an', 'is', 'are']
    
    text_lower = text.lower()
    french_count = sum(1 for word in french_words if word in text_lower)
    english_count = sum(1 for word in english_words if word in text_lower)
    
    if french_count > english_count:
        return "fr"
    else:
        return "en"


def extract_equipment_tags_html(text: str) -> List[str]:
    """Extract equipment tags from HTML text"""
    # HTML FIXED: More precise patterns for equipment tags
    equipment_patterns = [
        # Equipment tags with full format: 07-PU-001, 07-TK-001
        r'\b\d{2}-[A-Z]{2,3}-\d{3}\b',
        # Simple equipment tags: 0-UV, 0-HV, 1-PU
        r'\b\d{1,2}-[A-Z]{2,3}\b(?!-\d{3})',  # Negative lookahead to exclude drawing refs
        # Equipment tags without numbers: PU-001, TK-001
        r'\b[A-Z]{2,3}-\d{3}\b',
        # Splitter boxes: 0-SB-00
        r'\b\d{1,2}-[A-Z]{2}-\d{2}\b',
    ]
    
    equipment_tags = []
    for pattern in equipment_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        equipment_tags.extend(matches)
    
    # HTML FIXED: Filter out drawing references
    filtered_tags = []
    for tag in equipment_tags:
        # Exclude drawing references like PAC-0-PR-PID-000, PAC-00-IC-PHI-000
        if not re.match(r'PAC-\d+-[A-Z]+-[A-Z]+-\d+', tag, re.IGNORECASE):
            filtered_tags.append(tag)
    
    return list(set(filtered_tags))


def extract_technical_terms_html(text: str) -> List[str]:
    """Extract technical terms from HTML text"""
    # HTML FIXED: Improved patterns with case handling
    technical_patterns = [
        # Process terms
        r'\b(?:Flow|Pressure|Temperature|Level|Speed|Capacity|Efficiency|Density|BPL|K-Grade)\b',
        # Control systems
        r'\b(?:PID|PLC|DCS|SCADA|HMI|VSD|BPL|GSW)\b',
        # Units
        r'\b(?:bar|kPa|MPa|°C|rpm|kW|hp|m³|kg|tonnes|tph)\b',
        # Equipment types
        r'\b(?:Slurry|Phosphate|Pipeline|Pump|Tank|Valve|Agitator|Choke|Station)\b',
        # Additional PAC-specific terms
        r'\b(?:Batch|Interface|Flush|Dump|Pond|Thickener|Wash|Plant)\b',
    ]
    
    technical_terms = []
    for pattern in technical_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        technical_terms.extend(matches)
    
    # HTML FIXED: Normalize case and remove duplicates
    normalized_terms = []
    seen_terms = set()
    for term in technical_terms:
        normalized = term.lower()
        if normalized not in seen_terms:
            normalized_terms.append(term)
            seen_terms.add(normalized)
    
    return normalized_terms


def extract_tables_from_html(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract tables from HTML and convert to structured format"""
    tables = []
    
    try:
        table_elements = soup.find_all('table')
        
        for i, table in enumerate(table_elements):
            table_data = {
                'table_id': f'table_{i + 1}',
                'headers': [],
                'rows': [],
                'text_representation': ''
            }
            
            # Extract headers
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
                    table_data['headers'] = headers
            
            # Extract rows
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')
            else:
                rows = table.find_all('tr')[1:] if table.find('thead') else table.find_all('tr')
            
            for row in rows:
                cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                if cells:
                    table_data['rows'].append(cells)
            
            # Create text representation (Markdown format)
            if table_data['headers']:
                table_data['text_representation'] += ' | '.join(table_data['headers']) + '\n'
                table_data['text_representation'] += '|' + '|'.join(['---' for _ in table_data['headers']]) + '|\n'
            
            for row in table_data['rows']:
                table_data['text_representation'] += ' | '.join(row) + '\n'
            
            tables.append(table_data)
            
    except Exception as e:
        logging.warning(f"Failed to extract tables: {e}")
    
    return tables


async def process_html_file(html_file: Path, html_logger: logging.Logger, 
                           section_logger: logging.Logger, chunking_logger: logging.Logger, 
                           error_logger: logging.Logger) -> List[Dict[str, Any]]:
    """Process a single HTML file with proper section detection and metadata extraction"""
    file_chunks = []
    errors = []
    
    try:
        html_logger.info(f"Starting HTML processing: {html_file.name}")
        html_logger.info(f"File size: {html_file.stat().st_size / 1024:.2f} KB")
        
        # Generate unique document ID
        doc_id = f"{html_file.stem.lower()}_v1"
        
        # Step 1: Parse HTML
        html_logger.debug(f"Step 1: Parsing HTML document")
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        html_logger.info(f"  [SUCCESS] Parsed HTML document")
        
        # Step 2: Extract document metadata
        html_logger.debug(f"Step 2: Extracting document metadata")
        document_metadata = extract_document_metadata(soup)
        html_logger.info(f"  [SUCCESS] Extracted metadata: {document_metadata}")
        
        # Step 3: Extract tables
        html_logger.debug(f"Step 3: Extracting tables")
        tables = extract_tables_from_html(soup)
        html_logger.info(f"  [SUCCESS] Extracted {len(tables)} tables")
        
        # Step 4: HTML FIXED section detection
        section_logger.debug(f"Step 4: Detecting section hierarchy with HTML patterns")
        try:
            sections = detect_section_hierarchy_html(soup)
            section_logger.info(f"  [SUCCESS] Detected {len(sections)} sections")
            
            # Log section details
            for i, section in enumerate(sections):
                section_logger.debug(f"    Section {i+1}: {section['number']} - '{section['title']}' (Level {section['level']})")
                section_logger.debug(f"      Content length: {len(section['content'])} chars")
                
        except Exception as e:
            error_context = {
                "file": html_file.name,
                "html_length": len(html_content)
            }
            log_error_details(error_logger, "SECTION_DETECTION", e, error_context)
            errors.append(f"Section detection failed: {e}")
            # Create single section as fallback
            sections = [{
                'number': '1',
                'title': 'Main Content',
                'level': 1,
                'tag': 'h1',
                'content': soup.get_text(),
                'id': '',
                'subsections': []
            }]
        
        # Step 5: Enhanced chunking with overlap
        chunking_logger.debug(f"Step 5: Creating enhanced chunks with overlap")
        
        chunk_id = 1
        for section in sections:
            try:
                section_text = section['content']
                if not section_text.strip():
                    chunking_logger.warning(f"    Empty section {section['number']}, skipping")
                    continue
                
                # Split section into chunks with overlap (EXACTLY: Split if >500 tokens, with 120-token overlap)
                section_chunks = split_text_with_overlap(section_text, max_tokens=500, overlap_tokens=120)
                
                chunking_logger.debug(f"    Section {section['number']}: {len(section_chunks)} chunks created")
                
                for chunk_idx, chunk_text in enumerate(section_chunks):
                    if not chunk_text.strip():
                        continue
                    
                    # Extract metadata with HTML FIXED functions
                    equipment_tags = extract_equipment_tags_html(chunk_text)
                    technical_terms = extract_technical_terms_html(chunk_text)
                    language = detect_language(chunk_text)
                    
                    chunk_data = {
                        "text": chunk_text,
                        "metadata": {
                            "doc_id": doc_id,
                            "document": html_file.stem,
                            "document_title": document_metadata.get("title", ""),
                            "document_number": document_metadata.get("document_number", ""),
                            "revision": document_metadata.get("revision", ""),
                            "issue_date": document_metadata.get("issue_date", ""),
                            "project": document_metadata.get("project", ""),
                            "client": document_metadata.get("client", ""),
                            "chapter": section['number'],
                            "section_title": section['title'],
                            "section_level": section['level'],
                            "section_tag": section['tag'],
                            "section_id": section['id'],
                            "language": language,
                            "facility": "PAC",
                            "equipment_tags": equipment_tags,
                            "has_equations": "=" in chunk_text or "+" in chunk_text or "×" in chunk_text or "÷" in chunk_text or "Pressure" in chunk_text or "Flow" in chunk_text or "Density" in chunk_text,
                            "has_tables": "|" in chunk_text or "Table" in chunk_text or "Sequence" in chunk_text,
                            "technical_terms": technical_terms,
                            "chunk_id": f"{doc_id}_chunk_{chunk_id}",
                            "chunk_index": chunk_idx + 1,
                            "total_chunks_in_section": len(section_chunks),
                            "token_count": count_tokens(chunk_text),
                            "processing_errors": errors,
                            # Add page information (estimated based on section)
                            "page_start": int(section['number'].split('.')[0]) if section['number'].split('.')[0].isdigit() else 1,
                            "page_end": int(section['number'].split('.')[0]) if section['number'].split('.')[0].isdigit() else 1
                        }
                    }
                    
                    file_chunks.append(chunk_data)
                    chunk_id += 1
                    
                    chunking_logger.debug(f"      Chunk {chunk_idx + 1}: {len(chunk_text)} chars, {count_tokens(chunk_text)} tokens, {len(equipment_tags)} equipment tags")
                    
            except Exception as e:
                error_context = {
                    "file": html_file.name,
                    "section_number": section.get('number', 'Unknown'),
                    "section_title": section.get('title', 'Unknown')
                }
                log_error_details(error_logger, "CHUNK_CREATION", e, error_context)
                errors.append(f"Chunk creation failed for section {section.get('number', 'Unknown')}: {e}")
                continue
        
        chunking_logger.info(f"  [SUCCESS] Created {len(file_chunks)} chunks for {html_file.name}")
        
    except Exception as e:
        error_context = {
            "file": html_file.name,
            "file_size": html_file.stat().st_size if html_file.exists() else 0
        }
        log_error_details(error_logger, "HTML_PROCESSING", e, error_context)
        errors.append(f"HTML processing failed: {e}")
    
    return file_chunks, errors


async def run_enhanced_data_layer_html_pipeline():
    """Main HTML Data Layer pipeline"""
    # Setup logging
    main_logger, html_logger, section_logger, chunking_logger, error_logger = setup_comprehensive_logging()
    
    main_logger.info("=" * 80)
    main_logger.info("ENHANCED DATA LAYER PIPELINE - HTML PROCESSOR VERSION")
    main_logger.info("=" * 80)
    
    # Configuration
    html_file = "General_Control_Philosophy.html"
    output_file = f"enhanced_chunks_html_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    start_time = datetime.now()
    
    # Check HTML file
    html_path = Path(html_file)
    if not html_path.exists():
        main_logger.error(f"HTML file not found: {html_file}")
        return
    
    main_logger.info(f"Processing HTML file: {html_file}")
    
    # Process HTML file
    all_chunks = []
    all_errors = []
    
    try:
        file_chunks, file_errors = await process_html_file(
            html_path, html_logger, section_logger, chunking_logger, error_logger
        )
        
        if file_chunks:
            all_chunks.extend(file_chunks)
            main_logger.info(f"[SUCCESS] {html_file}: {len(file_chunks)} chunks created")
        else:
            main_logger.error(f"[FAILED] {html_file}: No chunks created")
        
        if file_errors:
            all_errors.extend([f"{html_file}: {error}" for error in file_errors])
            
    except Exception as e:
        error_context = {"file": html_file}
        log_error_details(error_logger, "FILE_PROCESSING", e, error_context)
        all_errors.append(f"{html_file}: Processing failed - {e}")
    
    # Generate comprehensive report
    end_time = datetime.now()
    duration = end_time - start_time
    
    main_logger.info(f"\n{'='*80}")
    main_logger.info("ENHANCED DATA LAYER PIPELINE - HTML PROCESSOR VERSION COMPLETED")
    main_logger.info(f"{'='*80}")
    
    # Summary statistics
    if all_chunks:
        total_text_length = sum(len(chunk["text"]) for chunk in all_chunks)
        avg_chunk_length = total_text_length / len(all_chunks)
        total_tokens = sum(chunk["metadata"]["token_count"] for chunk in all_chunks)
        avg_tokens = total_tokens / len(all_chunks)
        total_equipment_tags = sum(len(chunk["metadata"]["equipment_tags"]) for chunk in all_chunks)
        chunks_with_equations = sum(1 for chunk in all_chunks if chunk["metadata"]["has_equations"])
        chunks_with_tables = sum(1 for chunk in all_chunks if chunk["metadata"]["has_tables"])
        
        # Section hierarchy statistics
        unique_sections = set(chunk["metadata"]["chapter"] for chunk in all_chunks)
        max_section_level = max(chunk["metadata"]["section_level"] for chunk in all_chunks)
        
        main_logger.info(f"📊 HTML PROCESSOR SUMMARY STATISTICS:")
        main_logger.info(f"  Total chunks created: {len(all_chunks)}")
        main_logger.info(f"  Total text length: {total_text_length:,} characters")
        main_logger.info(f"  Average chunk length: {avg_chunk_length:.1f} characters")
        main_logger.info(f"  Total tokens: {total_tokens:,}")
        main_logger.info(f"  Average tokens per chunk: {avg_tokens:.1f}")
        main_logger.info(f"  Unique sections: {len(unique_sections)}")
        main_logger.info(f"  Maximum section level: {max_section_level}")
        main_logger.info(f"  Total equipment tags: {total_equipment_tags}")
        main_logger.info(f"  Chunks with equations: {chunks_with_equations}")
        main_logger.info(f"  Chunks with tables: {chunks_with_tables}")
        main_logger.info(f"  Total errors: {len(all_errors)}")
        main_logger.info(f"  Execution time: {duration}")
        
        # Save chunks to JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "total_chunks": len(all_chunks),
                        "total_text_length": total_text_length,
                        "total_tokens": total_tokens,
                        "unique_sections": len(unique_sections),
                        "max_section_level": max_section_level,
                        "total_errors": len(all_errors),
                        "execution_time": str(duration),
                        "timestamp": datetime.now().isoformat()
                    },
                    "chunks": all_chunks,
                    "errors": all_errors
                }, f, indent=2, ensure_ascii=False)
            
            main_logger.info(f"✅ HTML processor chunks saved to: {output_file}")
            
        except Exception as e:
            main_logger.error(f"❌ Failed to save chunks: {e}")
            log_error_details(error_logger, "CHUNK_SAVING", e)
    
    else:
        main_logger.error("❌ No chunks created! Pipeline failed.")
    
    # Error summary
    if all_errors:
        main_logger.error(f"\n❌ ERROR SUMMARY ({len(all_errors)} errors):")
        for i, error in enumerate(all_errors, 1):
            main_logger.error(f"  {i}. {error}")
    
    main_logger.info(f"\n{'='*80}")
    main_logger.info("ENHANCED DATA LAYER PIPELINE - HTML PROCESSOR VERSION END")
    main_logger.info(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(run_enhanced_data_layer_html_pipeline())


class HTMLProcessor:
    """HTML Document Processor for RAG Pipeline"""
    
    def __init__(self):
        """Initialize the HTML processor with logging"""
        self.main_logger, self.html_logger, self.section_logger, self.chunking_logger, self.error_logger = setup_comprehensive_logging()
    
    async def process_html_file(self, html_file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single HTML file and return chunks with metadata
        
        Args:
            html_file_path: Path to the HTML file
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        html_file = Path(html_file_path)
        
        try:
            self.main_logger.info(f"🚀 Starting HTML processing: {html_file.name}")
            
            # Process the HTML file using the existing function
            chunks, errors = await process_html_file(
                html_file=html_file,
                html_logger=self.html_logger,
                section_logger=self.section_logger,
                chunking_logger=self.chunking_logger,
                error_logger=self.error_logger
            )
            
            self.main_logger.info(f"✅ HTML processing completed. Generated {len(chunks)} chunks.")
            return chunks
            
        except Exception as e:
            self.error_logger.error(f"❌ Error processing HTML file: {e}")
            self.error_logger.error(traceback.format_exc())
            raise
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "processor_type": "HTMLProcessor",
            "features": [
                "HTML parsing with BeautifulSoup",
                "Section hierarchy detection",
                "Table extraction",
                "Equipment tag extraction",
                "Technical term extraction",
                "Language detection",
                "Dynamic chunking with overlap"
            ]
        }
