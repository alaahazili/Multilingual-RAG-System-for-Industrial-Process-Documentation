#!/usr/bin/env python3
"""
Debug Embeddings Script
Test the embedding service to identify why embeddings are invalid.
"""

import sys
import os
import json
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from app.services.embedding_layer.embedding_service import EmbeddingService, EmbeddingConfig
from app.services.embedding_layer.qdrant_store import QdrantStore, QdrantConfig
from app.core.settings import settings

def test_embedding_service():
    """Test the embedding service with a simple example."""
    print("🔍 Testing Embedding Service...")
    
    # Initialize embedding service
    embedding_config = EmbeddingConfig(
        model_name="intfloat/multilingual-e5-large",
        vector_dimension=1024,
        use_hf_api=True,
        normalize_embeddings=True,
        batch_size=4
    )
    
    embedding_service = EmbeddingService(embedding_config)
    
    # Test with simple texts
    test_texts = [
        "This is a test document about PAC control philosophy.",
        "The pressure formula is Pressure = Force / Area.",
        "Alarm priority levels are defined in section 3.2."
    ]
    
    print(f"📝 Testing with {len(test_texts)} texts...")
    
    try:
        # Test passage embeddings
        embeddings = embedding_service.embed_passages(test_texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        for i, (text, embedding) in enumerate(zip(test_texts, embeddings)):
            print(f"  Text {i+1}: {text[:50]}...")
            print(f"  Embedding length: {len(embedding)}")
            print(f"  Embedding type: {type(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")
            print(f"  Is valid: {len(embedding) == 1024}")
            print()
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Embedding service failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_qdrant_upsert(embeddings):
    """Test Qdrant upsert with the test embeddings."""
    if not embeddings:
        return
    
    print("🔍 Testing Qdrant Upsert...")
    
    # Initialize Qdrant store - Use existing documents collection
    qdrant_config = QdrantConfig(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name="documents",  # Use existing collection
        vector_size=1024,
        distance_metric="Cosine"  # Fixed case
    )
    
    qdrant_store = QdrantStore(qdrant_config)
    
    # Create test documents
    test_documents = []
    for i, embedding in enumerate(embeddings):
        doc = {
            "embedding": embedding,
            "embedding_id": i + 1,
            "content": f"Test document {i+1}",
            "chunk_id": f"test_chunk_{i+1}",
            "document": "test_document",
            "section_title": "Test Section",
            "section_number": "1.1",
            "page_start": 1,
            "page_end": 1,
            "language": "en",
            "document_type": "test",
            "facility": "PAC",
            "token_count": 10,
            "has_equations": False,
            "has_tables": False,
            "equipment_tags": [],
            "technical_terms": [],
            "measurements": [],
            "cross_references": [],
            "metadata": {}
        }
        test_documents.append(doc)
    
    print(f"📝 Created {len(test_documents)} test documents")
    
    try:
        # Test upsert
        success = qdrant_store.upsert_documents(test_documents)
        print(f"✅ Upsert result: {success}")
        
        # Check collection info
        collection_info = qdrant_store.client.get_collection(qdrant_store.collection_name)
        print(f"📊 Collection points: {collection_info.points_count}")
        
    except Exception as e:
        print(f"❌ Qdrant upsert failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    print("🚀 DEBUGGING EMBEDDING ISSUES")
    print("=" * 50)
    
    # Test embedding service
    embeddings = test_embedding_service()
    
    if embeddings:
        # Test Qdrant upsert
        test_qdrant_upsert(embeddings)
    
    print("\n🎯 Debug complete!")

if __name__ == "__main__":
    main()
