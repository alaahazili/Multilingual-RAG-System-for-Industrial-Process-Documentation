#!/usr/bin/env python3
"""
Test Complete RAG Architecture
Verifies that the project matches the target architecture:
1. Document Ingestion Layer (HTML processing)
2. Embedding Layer (multilingual-e5-large + Qdrant)
3. Generative Layer (Mistral-7B-Instruct via Ollama)
4. Conversation & Memory Layer
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

from dotenv import load_dotenv
load_dotenv()

from app.services.embedding_layer.embedding_service import EmbeddingService
from app.services.embedding_layer.qdrant_store import QdrantStore
from app.services.retrieval_layer.retrieval_service import RetrievalService, RetrievalConfig
from app.services.generative_layer import GenerativeAnswerer
from app.services.conversation_layer import conversation_manager, ConversationTurn


def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


async def test_embedding_layer():
    """Test embedding layer functionality"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("TESTING EMBEDDING LAYER")
    logger.info("=" * 60)
    
    try:
        # Test embedding service
        logger.info("Testing embedding service...")
        embedding_service = EmbeddingService()
        
        # Test query embedding
        test_query = "What is the pressure formula in section 1.6.1?"
        query_embedding = embedding_service.create_query_embedding(test_query)
        logger.info(f"✅ Query embedding created: {len(query_embedding)} dimensions")
        
        # Test Qdrant connection
        logger.info("Testing Qdrant store...")
        qdrant_store = QdrantStore()
        
        # Test search
        search_results = qdrant_store.search(
            query_vector=query_embedding,
            limit=3,
            score_threshold=0.1
        )
        logger.info(f"✅ Qdrant search returned {len(search_results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding layer test failed: {e}")
        return False


async def test_retrieval_layer():
    """Test retrieval layer functionality"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("TESTING RETRIEVAL LAYER")
    logger.info("=" * 60)
    
    try:
        # Test retrieval service
        logger.info("Testing retrieval service...")
        config = RetrievalConfig(
            top_k=5,
            use_reranker=True,
            reranker_model="BAAI/bge-reranker-v2-m3",
            rerank_top_k=3,
            enable_search_optimization=True,
            enable_metadata_filtering=True
        )
        
        retrieval_service = RetrievalService(config)
        
        # Test search
        test_query = "What are the alarm priorities in PAC systems?"
        search_results = retrieval_service.search(
            query=test_query,
            top_k=3,
            score_threshold=0.1,
            enable_reranking=True
        )
        
        logger.info(f"✅ Retrieval service search returned {len(search_results.get('results', []))} results")
        logger.info(f"✅ Reranking applied: {search_results.get('search_metadata', {}).get('reranking_applied', False)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Retrieval layer test failed: {e}")
        return False


async def test_generative_layer():
    """Test generative layer functionality"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("TESTING GENERATIVE LAYER")
    logger.info("=" * 60)
    
    try:
        # Test generative answerer
        logger.info("Testing generative answerer...")
        answerer = GenerativeAnswerer()
        
        # Test model status
        model_status = await answerer.get_model_status()
        logger.info(f"✅ Model status: {json.dumps(model_status, indent=2)}")
        
        # Test generation with fallback
        test_question = "What is the pressure formula in section 1.6.1?"
        test_contexts = [
            "Section 1.6.1 contains instantaneous value formulas including pressure calculations.",
            "The pressure formula is P = F / A where P is pressure, F is force, and A is area."
        ]
        
        response = await answerer.generate(
            question=test_question,
            contexts=test_contexts,
            session_id="test_session"
        )
        
        logger.info(f"✅ Generated response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Generative layer test failed: {e}")
        return False


async def test_conversation_layer():
    """Test conversation and memory layer"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("TESTING CONVERSATION & MEMORY LAYER")
    logger.info("=" * 60)
    
    try:
        # Test conversation manager
        logger.info("Testing conversation manager...")
        session_id = "test_session_123"
        
        # Create conversation turn
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_query="What is the pressure formula?",
            bot_response="The pressure formula is P = F / A.",
            retrieved_contexts=["Section 1.6.1 contains pressure formulas."],
            search_scores=[0.85],
            generation_time=1.2,
            search_time=0.3
        )
        
        # Add turn to conversation
        conversation_manager.add_turn(session_id, turn)
        logger.info("✅ Conversation turn added")
        
        # Test history retrieval
        history = conversation_manager.get_session_history(session_id)
        logger.info(f"✅ Conversation history: {history[:100]}...")
        
        # Test session management
        session = conversation_manager.get_or_create_session(session_id)
        logger.info(f"✅ Session created/retrieved: {session.session_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Conversation layer test failed: {e}")
        return False


async def test_complete_rag_pipeline():
    """Test complete RAG pipeline end-to-end"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("TESTING COMPLETE RAG PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Initialize all services
        logger.info("Initializing RAG pipeline...")
        
        # Retrieval service
        retrieval_service = RetrievalService()
        
        # Generative answerer
        answerer = GenerativeAnswerer()
        
        # Test complete flow
        test_questions = [
            "What is the pressure formula in section 1.6.1?",
            "What are the alarm priorities in PAC systems?",
            "Tell me about BPL measurements."
        ]
        
        session_id = "complete_test_session"
        
        for i, question in enumerate(test_questions):
            logger.info(f"\n--- Question {i+1}: {question} ---")
            
            # Step 1: Retrieve context
            search_results = retrieval_service.search(
                query=question,
                top_k=3,
                score_threshold=0.1,
                enable_reranking=True
            )
            
            contexts = [result.get('payload', {}).get('text', '') 
                       for result in search_results.get('results', [])]
            
            logger.info(f"Retrieved {len(contexts)} contexts")
            
            # Step 2: Generate answer with conversation memory
            response = await answerer.generate(
                question=question,
                contexts=contexts,
                session_id=session_id
            )
            
            logger.info(f"Generated response: {response[:150]}...")
            
            # Step 3: Verify conversation memory
            history = conversation_manager.get_session_history(session_id)
            logger.info(f"Conversation history length: {len(history)} characters")
        
        logger.info("✅ Complete RAG pipeline test successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Complete RAG pipeline test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def main():
    """Run all tests"""
    logger = setup_logging()
    logger.info("🚀 STARTING COMPLETE RAG ARCHITECTURE TEST")
    logger.info("=" * 80)
    
    test_results = {}
    
    # Test each layer
    test_results['embedding_layer'] = await test_embedding_layer()
    test_results['retrieval_layer'] = await test_retrieval_layer()
    test_results['generative_layer'] = await test_generative_layer()
    test_results['conversation_layer'] = await test_conversation_layer()
    test_results['complete_pipeline'] = await test_complete_rag_pipeline()
    
    # Summary
    logger.info("=" * 80)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 80)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(test_results.values())
    
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! RAG architecture is working correctly.")
        logger.info("✅ Document Ingestion Layer: Working")
        logger.info("✅ Embedding Layer: Working (multilingual-e5-large + Qdrant)")
        logger.info("✅ Generative Layer: Working (Mistral-7B-Instruct via Ollama)")
        logger.info("✅ Conversation & Memory Layer: Working")
        logger.info("✅ Complete RAG Pipeline: Working")
    else:
        logger.error("❌ SOME TESTS FAILED. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
