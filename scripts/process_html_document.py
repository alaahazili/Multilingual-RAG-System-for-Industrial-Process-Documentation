#!/usr/bin/env python3
"""
Process HTML Document - Document Preparation for RAG
Processes the General_Control_Philosophy.html document to prepare it for efficient search and retrieval.

REQUIREMENTS MET:
✅ Document Source: General_Control_Philosophy.html
✅ Preprocessing: 
   - Extract text (headings, paragraphs, lists, tables, formulas)
   - Preserve tables in Markdown format
   - Preserve formulas in readable text format
✅ Chunking: Split into meaningful pieces (~500 tokens per chunk)
✅ Metadata Preservation:
   - Section name / number
   - Page reference (estimated from section)
   - Raw text (including formulas/tables)
   - Equipment tags, technical terms, language detection
"""

import asyncio
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.data_layer.html_processor import HTMLProcessor


def setup_logging():
    """Setup logging for the processing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


async def process_html_document():
    """Process the HTML document and create chunks"""
    logger = setup_logging()
    
    logger.info("🚀 STARTING HTML DOCUMENT PROCESSING")
    logger.info("=" * 60)
    
    try:
        # Initialize HTML processor
        logger.info("📄 Initializing HTML processor...")
        processor = HTMLProcessor()
        
        # Define input and output paths
        input_file = Path("data/documents/General_Control_Philosophy.html")
        output_file = Path("data/processed_chunks.json")
        
        # Check if input file exists
        if not input_file.exists():
            logger.error(f"❌ Input file not found: {input_file}")
            return False
        
        logger.info(f"📖 Processing file: {input_file}")
        logger.info(f"💾 Output will be saved to: {output_file}")
        
        # Process the HTML file
        logger.info("🔄 Processing HTML document...")
        start_time = datetime.now()
        
        chunks = await processor.process_html_file(str(input_file))
        
        processing_time = datetime.now() - start_time
        logger.info(f"✅ Processing completed in {processing_time}")
        
        # Create output data structure
        output_data = {
            "metadata": {
                "total_chunks": len(chunks),
                "total_text_length": sum(len(chunk["text"]) for chunk in chunks),
                "total_tokens": sum(chunk["metadata"].get("token_count", 0) for chunk in chunks),
                "unique_sections": len(set(chunk["metadata"].get("section_title", "") for chunk in chunks)),
                "max_section_level": max(chunk["metadata"].get("section_level", 0) for chunk in chunks),
                "total_errors": 0,
                "execution_time": str(processing_time),
                "timestamp": datetime.now().isoformat()
            },
            "chunks": chunks,
            "errors": []
        }
        
        # Save processed data
        logger.info("💾 Saving processed chunks...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Log statistics
        logger.info("📊 Processing Statistics:")
        logger.info(f"  - Total chunks: {len(chunks)}")
        logger.info(f"  - Total text length: {output_data['metadata']['total_text_length']:,} characters")
        logger.info(f"  - Total tokens: {output_data['metadata']['total_tokens']:,}")
        logger.info(f"  - Unique sections: {output_data['metadata']['unique_sections']}")
        logger.info(f"  - Max section level: {output_data['metadata']['max_section_level']}")
        logger.info(f"  - Processing time: {processing_time}")
        
        # Show sample chunks
        logger.info("\n📝 Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            logger.info(f"  Chunk {i+1}:")
            logger.info(f"    - Section: {chunk['metadata'].get('section_title', 'N/A')}")
            logger.info(f"    - Tokens: {chunk['metadata'].get('token_count', 0)}")
            logger.info(f"    - Text preview: {chunk['text'][:100]}...")
            logger.info("")
        
        logger.info("🎉 HTML document processing completed successfully!")
        logger.info(f"✅ Output saved to: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing HTML document: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def main():
    """Main function"""
    success = await process_html_document()
    
    if success:
        print("\n🎉 SUCCESS! HTML document has been processed.")
        print("📁 Next step: Create embeddings using the processed chunks.")
    else:
        print("\n❌ FAILED! Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
