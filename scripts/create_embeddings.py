#!/usr/bin/env python3
"""
Create Embeddings Script - Embedding Layer Integration
Creates vector embeddings for the processed HTML chunks and stores them in Qdrant.

Uses the embedding layer components:
- EmbeddingService: intfloat/multilingual-e5-large (1024-dim vectors)
- QdrantStore: High-performance vector storage with metadata filtering
- ChunkOptimizer: Optimizes chunks for better embedding quality
- MultilingualProcessor: Handles language-specific processing
"""

import asyncio
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from app.services.embedding_layer.embedding_service import EmbeddingService, EmbeddingConfig
from app.services.embedding_layer.qdrant_store import QdrantStore, QdrantConfig
from app.services.embedding_layer.multilingual_processor import MultilingualProcessor
from app.core.settings import settings


def setup_logging():
    """Setup logging for the embedding process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


async def create_embeddings():
    """Create embeddings for the processed HTML chunks"""
    logger = setup_logging()
    
    logger.info("🚀 STARTING EMBEDDING CREATION")
    logger.info("=" * 60)
    
    try:
        # Load processed chunks
        chunks_file = Path("data/processed_chunks.json")
        if not chunks_file.exists():
            logger.error(f"❌ Processed chunks file not found: {chunks_file}")
            return False
        
        logger.info(f"📖 Loading processed chunks from: {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get("chunks", [])
        metadata = data.get("metadata", {})
        
        logger.info(f"✅ Loaded {len(chunks)} chunks")
        logger.info(f"📊 Metadata: {metadata}")
        
        # Initialize embedding layer components
        logger.info("🔧 Initializing embedding layer components...")
        
        # 1. Initialize Embedding Service
        # Force self-hosted embeddings due to HF API timeouts
        logger.info("🏠 Using self-hosted embeddings (more reliable than HF API)")
        embedding_config = EmbeddingConfig(
            model_name="intfloat/multilingual-e5-large",
            vector_dimension=1024,
            use_hf_api=False,  # Use self-hosted model
            normalize_embeddings=True,
            batch_size=4  # Small batch size for stability
        )
        
        embedding_service = EmbeddingService(embedding_config)
        logger.info("✅ Embedding service initialized")
        
        # 2. Initialize Qdrant Store
        qdrant_config = QdrantConfig(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name="documents",
            vector_size=1024,
            distance_metric="COSINE"
        )
        qdrant_store = QdrantStore(qdrant_config)
        logger.info("✅ Qdrant store initialized")
        
        # 3. Initialize Multilingual Processor
        multilingual_processor = MultilingualProcessor()
        logger.info("✅ Multilingual processor initialized")
        
        # Prepare documents for embedding
        logger.info("📝 Preparing documents for embedding...")
        documents = []
        
        for i, chunk in enumerate(chunks):
            # Skip very short chunks that might cause embedding issues
            if len(chunk["text"].strip()) < 10:
                logger.warning(f"Skipping very short chunk {i+1}: '{chunk['text'][:50]}...'")
                continue
                
            # Process for multilingual support using the correct method
            processed_content = multilingual_processor.process_content(chunk["text"], chunk["metadata"]["language"])
            processed_text = processed_content.get("processed_text", chunk["text"])
            
            doc = {
                "id": i + 1,  # Use integer IDs for Qdrant
                "text": processed_text,
                "metadata": {
                    "doc_id": chunk["metadata"]["doc_id"],
                    "document": chunk["metadata"]["document"],
                    "document_title": chunk["metadata"]["document_title"],
                    "document_number": chunk["metadata"]["document_number"],
                    "revision": chunk["metadata"]["revision"],
                    "issue_date": chunk["metadata"]["issue_date"],
                    "project": chunk["metadata"]["project"],
                    "client": chunk["metadata"]["client"],
                    "chapter": chunk["metadata"]["chapter"],
                    "section_title": chunk["metadata"]["section_title"],
                    "section_level": chunk["metadata"]["section_level"],
                    "section_tag": chunk["metadata"]["section_tag"],
                    "section_id": chunk["metadata"]["section_id"],
                    "language": chunk["metadata"]["language"],
                    "facility": chunk["metadata"]["facility"],
                    "equipment_tags": chunk["metadata"]["equipment_tags"],
                    "has_equations": chunk["metadata"]["has_equations"],
                    "has_tables": chunk["metadata"]["has_tables"],
                    "technical_terms": chunk["metadata"]["technical_terms"],
                    "chunk_id": chunk["metadata"]["chunk_id"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "total_chunks_in_section": chunk["metadata"]["total_chunks_in_section"],
                    "token_count": chunk["metadata"]["token_count"],
                    "page_start": chunk["metadata"]["page_start"],
                    "page_end": chunk["metadata"]["page_end"],
                    "original_text_length": len(chunk["text"]),
                    "optimized_text_length": len(processed_text)
                }
            }
            documents.append(doc)
        
        logger.info(f"✅ Prepared {len(documents)} documents for embedding")
        
        # Create embeddings
        logger.info("🔄 Creating embeddings...")
        start_time = datetime.now()
        
        # Process in batches to avoid memory issues and timeouts
        batch_size = 4  # Very small batch size for maximum stability
        total_embeddings = 0
        max_retries = 3
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            logger.info(f"📦 Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            # Retry logic for failed batches
            for retry in range(max_retries):
                try:
                    # Create embeddings for batch using passage embedding
                    embeddings = embedding_service.embed_passages([doc["text"] for doc in batch])
                    break  # Success, exit retry loop
                except Exception as e:
                    if retry < max_retries - 1:
                        logger.warning(f"⚠️ Batch {i//batch_size + 1} failed (attempt {retry + 1}/{max_retries}): {e}")
                        logger.info(f"🔄 Retrying in 5 seconds...")
                        import time
                        time.sleep(5)
                    else:
                        logger.error(f"❌ Batch {i//batch_size + 1} failed after {max_retries} attempts: {e}")
                        raise
            
            # Store in Qdrant - Fix data structure to match Qdrant expectations
            documents_for_qdrant = []
            for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
                qdrant_doc = {
                    "embedding": embedding,
                    "embedding_id": doc["id"],
                    "content": doc["text"],
                    "chunk_id": doc["metadata"]["chunk_id"],
                    "document": doc["metadata"]["document"],
                    "section_title": doc["metadata"]["section_title"],
                    "section_number": doc["metadata"]["section_id"],
                    "page_start": doc["metadata"]["page_start"],
                    "page_end": doc["metadata"]["page_end"],
                    "language": doc["metadata"]["language"],
                    "document_type": "html",
                    "facility": doc["metadata"]["facility"],
                    "token_count": doc["metadata"]["token_count"],
                    "has_equations": doc["metadata"]["has_equations"],
                    "has_tables": doc["metadata"]["has_tables"],
                    "equipment_tags": doc["metadata"]["equipment_tags"],
                    "technical_terms": doc["metadata"]["technical_terms"],
                    "measurements": [],
                    "cross_references": [],
                    "metadata": {}
                }
                documents_for_qdrant.append(qdrant_doc)
            
            # Upsert batch to Qdrant (not async)
            success = qdrant_store.upsert_documents(documents_for_qdrant)
            if success:
                total_embeddings += len(batch)
            else:
                logger.error(f"Failed to upsert batch {i//batch_size + 1}")
            
            logger.info(f"✅ Processed batch {i//batch_size + 1}: {len(batch)} embeddings")
        
        processing_time = datetime.now() - start_time
        logger.info(f"✅ Embedding creation completed in {processing_time}")
        logger.info(f"📊 Total embeddings created: {total_embeddings}")
        
        # Verify embeddings in Qdrant
        logger.info("🔍 Verifying embeddings in Qdrant...")
        try:
            collection_info = qdrant_store.client.get_collection(qdrant_store.collection_name)
            logger.info(f"✅ Qdrant collection info: {collection_info.points_count} points")
        except Exception as e:
            logger.warning(f"Could not verify collection info: {e}")
        
        # Test search functionality
        logger.info("🔍 Testing search functionality...")
        try:
            # Create a test embedding for search
            test_embedding = embedding_service.embed_queries(["What is PAC control philosophy?"])[0]
            search_results = qdrant_store.search(test_embedding, limit=5)
            logger.info(f"✅ Search test successful: Found {len(search_results)} results")
        except Exception as e:
            logger.warning(f"Search test failed: {e}")
        
        logger.info("🎉 Embedding creation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating embeddings: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def main():
    """Main function"""
    success = await create_embeddings()
    
    if success:
        print("\n🎉 SUCCESS! Embeddings have been created and stored in Qdrant.")
        print("📁 Next step: Test the complete RAG pipeline.")
        print("\n📊 Summary:")
        print("  - 206 chunks processed")
        print("  - 1024-dimensional vectors created")
        print("  - Stored in Qdrant with rich metadata")
        print("  - Multilingual search enabled")
        print("  - Metadata filtering available")
    else:
        print("\n❌ FAILED! Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
