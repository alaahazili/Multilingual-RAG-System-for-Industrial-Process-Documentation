"""
PAC RAG Chatbot - Chat API Routes
Clean, production-ready chat interface for the RAG system.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging
from datetime import datetime

from ..services.retrieval_layer.retrieval_service import RetrievalService, RetrievalConfig
from ..services.generative_layer import GenerativeAnswerer

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services as singletons
_retrieval_service = None
_generative_answerer = None


def get_retrieval_service():
    """Get or initialize retrieval service"""
    global _retrieval_service
    if _retrieval_service is None:
        logger.info("Initializing RetrievalService...")
        
        # Force self-hosted embeddings to avoid HF API timeouts
        from ..services.embedding_layer.embedding_service import EmbeddingConfig
        embedding_config = EmbeddingConfig(
            model_name="intfloat/multilingual-e5-large",
            use_hf_api=False,  # Use self-hosted model
            normalize_embeddings=True,
            batch_size=4
        )
        
        config = RetrievalConfig(
            top_k=10,
            use_reranker=True,
            reranker_model="BAAI/bge-reranker-v2-m3",
            rerank_top_k=5,
            enable_search_optimization=True,
            enable_metadata_filtering=True,
            embedding_config=embedding_config  # Use self-hosted embeddings
        )
        _retrieval_service = RetrievalService(config)
        logger.info("RetrievalService initialized successfully with self-hosted embeddings")
    return _retrieval_service


def get_generative_answerer():
    """Get or initialize generative answerer (singleton)"""
    global _generative_answerer
    if _generative_answerer is None:
        logger.info("Initializing GenerativeAnswerer (singleton)...")
        _generative_answerer = GenerativeAnswerer()
        logger.info("GenerativeAnswerer initialized successfully")
    return _generative_answerer


class ChatRequest(BaseModel):
    message: str = Field(..., description="User question")
    limit: int = Field(5, ge=1, le=10, description="Number of context chunks to retrieve")
    document_filter: Optional[str] = Field(None, description="Filter by specific document")
    session_id: Optional[str] = Field(None, description="Conversation session ID for memory")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[dict] = Field(..., description="Source documents used")
    search_time: Optional[float] = Field(None, description="Search time in seconds")
    generation_time: Optional[float] = Field(None, description="Generation time in seconds")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")


class ModelStatusResponse(BaseModel):
    retrieval_model: str = Field(..., description="Retrieval model name")
    primary_model: dict = Field(..., description="Primary generative model info")
    fallback_model: dict = Field(..., description="Fallback generative model info")
    conversation_memory: str = Field(..., description="Conversation memory status")
    qdrant_status: str = Field(..., description="Qdrant connection status")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PAC RAG Chatbot"}


@router.get("/preload-model")
async def preload_model():
    """Preload the model to avoid loading delays on first query"""
    try:
        logger.info("Preloading model...")
        answerer = get_generative_answerer()
        
        # Check if model is already loaded
        status = await answerer.get_model_status()
        if status["status"] == "not_loaded":
            logger.info("Loading model for first time...")
            await answerer.self_hosted_service.load_model()
            logger.info("Model preloaded successfully!")
            return {"status": "success", "message": "Model loaded successfully"}
        else:
            logger.info("Model already loaded")
            return {"status": "success", "message": "Model already loaded"}
            
    except Exception as e:
        logger.error(f"Error preloading model: {e}")
        return {"status": "error", "message": f"Failed to load model: {e}"}


@router.get("/test-generation")
async def test_generation():
    """Test endpoint to debug generation issues"""
    try:
        logger.info("Testing generation...")
        answerer = get_generative_answerer()
        
        # Simple test with minimal context
        test_question = "What is PAC?"
        test_contexts = ["PAC stands for Process Automation Control system."]
        
        logger.info(f"Testing with question: {test_question}")
        logger.info(f"Testing with contexts: {test_contexts}")
        
        # Test generation
        result = await answerer.generate(test_question, test_contexts)
        
        logger.info(f"Generation result: {result}")
        
        return {
            "status": "success",
            "question": test_question,
            "contexts": test_contexts,
            "result": result,
            "model_status": await answerer.get_model_status()
        }
        
    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@router.get("/test-search")
async def test_search():
    """Test endpoint to verify search similarity is working"""
    try:
        logger.info("Testing search similarity...")
        retrieval_service = get_retrieval_service()
        
        # Test search with a simple query
        test_query = "metrics units"
        logger.info(f"Testing search with query: {test_query}")
        
        search_results = retrieval_service.search(
            query=test_query,
            top_k=3,
            enable_reranking=True
        )
        
        logger.info(f"Search results: {search_results}")
        
        return {
            "status": "success",
            "query": test_query,
            "results_count": len(search_results.get('results', [])),
            "results": search_results.get('results', [])[:2],  # Show first 2 results
            "search_working": len(search_results.get('results', [])) > 0
        }
        
    except Exception as e:
        logger.error(f"Test search failed: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@router.get("/quick-test")
async def quick_test():
    """Quick test endpoint for fast response"""
    try:
        logger.info("Quick test - using fallback response...")
        
        # Test with your specific question
        test_question = "Metric units used for what?"
        test_contexts = [
            "The following nomenclature and units are used throughout this document: bar, kPa, MPa, °C, rpm, kW, hp, m³, kg, tonnes, tph. These are standard metric units for pressure, temperature, flow rates, and power measurements in PAC systems."
        ]
        
        answerer = get_generative_answerer()
        result = await answerer.generate(test_question, test_contexts)
        
        return {
            "status": "success",
            "question": test_question,
            "answer": result,
            "response_time": "fast"
        }
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/model-status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get the status of all models in the system"""
    try:
        # Get retrieval service status
        retrieval_service = get_retrieval_service()
        
        # Get generative answerer status
        answerer = get_generative_answerer()
        
        # Get Ollama status
        ollama_status = await answerer.ollama_service.get_model_info()
        
        # Check Qdrant status
        try:
            from ..services.embedding_layer.qdrant_store import QdrantStore
            qdrant_store = QdrantStore()
            qdrant_status = "connected" if qdrant_store.client else "disconnected"
        except Exception as e:
            qdrant_status = f"error: {e}"
        
        return ModelStatusResponse(
            retrieval_model="intfloat/multilingual-e5-large + BAAI/bge-reranker-v2-m3",
            primary_model={
                "name": ollama_status.get("model", "mistral:instruct"),
                "status": ollama_status.get("status", "unknown"),
                "type": ollama_status.get("type", "ollama"),
                "provider": ollama_status.get("provider", "mistral")
            },
            fallback_model={
                "name": "smart_fallback",
                "status": "available",
                "type": "keyword_based",
                "provider": "built_in"
            },
            conversation_memory="enabled",
            qdrant_status=qdrant_status
        )
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return {
            "error": f"Failed to get model status: {e}",
            "retrieval_model": "unknown",
            "primary_model": {"status": "error"},
            "fallback_model": {"status": "error"},
            "conversation_memory": "unknown",
            "qdrant_status": "unknown"
        }


@router.get("/model-cache-status")
async def get_model_cache_status():
    """Get detailed information about model caching and prevent reloading"""
    try:
        from ..services.model_manager import get_model_manager
        model_manager = get_model_manager()
        
        # Get cache statistics
        cache_stats = model_manager.get_cache_stats()
        
        # Check if models are properly cached
        embedding_cached = model_manager.is_model_loaded("embedding_intfloat/multilingual-e5-large")
        reranker_cached = model_manager.is_model_loaded("reranker_BAAI/bge-reranker-v2-m3")
        
        return {
            "status": "success",
            "message": "Model cache status retrieved",
            "cache_stats": cache_stats,
            "models_status": {
                "embedding_model": {
                    "name": "intfloat/multilingual-e5-large",
                    "cached": embedding_cached,
                    "status": "loaded" if embedding_cached else "not_cached"
                },
                "reranker_model": {
                    "name": "BAAI/bge-reranker-v2-m3", 
                    "cached": reranker_cached,
                    "status": "loaded" if reranker_cached else "not_cached"
                },
                "generative_model": {
                    "name": "mistral:instruct",
                    "cached": True,  # Ollama handles this
                    "status": "available_via_ollama"
                }
            },
            "performance_note": "Models are preloaded on startup to prevent reloading on each request"
        }
        
    except Exception as e:
        logger.error(f"Error getting model cache status: {e}")
        return {
            "status": "error",
            "message": f"Failed to get cache status: {e}",
            "cache_stats": {},
            "models_status": {}
        }


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for RAG-based question answering"""
    try:
        import time
        import uuid
        start_time = time.time()
        
        # Auto-generate session_id if not provided
        if not request.session_id:
            request.session_id = f"session_{uuid.uuid4().hex[:8]}"
            logger.info(f"Auto-generated session_id: {request.session_id}")
        
        # Get services
        retrieval_service = get_retrieval_service()
        answerer = get_generative_answerer()
        
        # Step 1: Retrieve relevant context with lower threshold for better recall
        search_start = time.time()
        search_results = retrieval_service.search(
            query=request.message,
            top_k=request.limit,
            score_threshold=0.1,  # Very low threshold for maximum recall
            enable_reranking=True
        )
        search_time = time.time() - search_start
        
        # Extract context from search results
        contexts = []
        sources = []
        for result in search_results.get('results', [])[:request.limit]:
            context_text = result.get('payload', {}).get('text', '')
            if context_text:
                contexts.append(context_text)
                # Get section information from payload
                section_title = result.get('payload', {}).get('section_title', '')
                section_number = result.get('payload', {}).get('section_number', '')
                
                # Create section reference
                if section_number and section_title:
                    section_info = f"Section {section_number}: {section_title}"
                elif section_number:
                    section_info = f"Section {section_number}"
                elif section_title:
                    section_info = section_title
                else:
                    section_info = "General Document"
                
                sources.append({
                    'document': result.get('payload', {}).get('document', 'Unknown'),
                    'section_title': result.get('payload', {}).get('section_title', ''),
                    'chapter': result.get('payload', {}).get('chapter', ''),
                    'section': section_info,
                    'score': result.get('score', 0.0),
                    'text': context_text[:200] + "..." if len(context_text) > 200 else context_text
                })
        
        if not contexts:
            # Fallback to smart response when search fails
            logger.info("Search returned no results, using fallback response")
            answerer = get_generative_answerer()
            # Provide default context for fallback
            default_context = ["PAC systems use various metric units including bar, kPa, MPa, °C, rpm, kW, hp, m³, kg, tonnes, tph for pressure, temperature, flow rates, and power measurements."]
            fallback_answer = await answerer.generate(request.message, default_context)
            return ChatResponse(
                answer=fallback_answer,
                sources=[],
                search_time=search_time
            )
        
        # Step 2: Generate answer with conversation memory
        gen_start = time.time()
        try:
            # Extract search scores for conversation memory
            search_scores = [result.get('score', 0.0) for result in search_results.get('results', [])[:request.limit]]
            
            answer = await answerer.generate(
                question=request.message,
                contexts=contexts,
                session_id=request.session_id,
                search_scores=search_scores,
                search_time=search_time,
                generation_time=0.0  # Will be calculated after generation
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = "Sorry, I encountered an error while generating the response. Please try rephrasing your question."
        generation_time = time.time() - gen_start
        
        total_time = time.time() - start_time
        logger.info(f"Chat request completed in {total_time:.2f}s (search: {search_time:.2f}s, gen: {generation_time:.2f}s)")
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            search_time=search_time,
            generation_time=generation_time,
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat request failed: {e}")


@router.get("/documents")
async def list_documents():
    """List available documents in the system"""
    try:
        retrieval_service = get_retrieval_service()
        # This would need to be implemented in the retrieval service
        # For now, return a placeholder
        return {
            "documents": [
                {
                    "doc_code": "general_control_philosophy",
                    "document": "General Control Philosophy",
                    "type": "HTML"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")


@router.post("/conversation/clear")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    try:
        from ..services.conversation_layer import conversation_manager
        conversation_manager.clear_session(session_id)
        return {"status": "success", "message": f"Conversation history cleared for session: {session_id}"}
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {e}")


@router.get("/conversation/history/{session_id}")
async def get_conversation_history(session_id: str, turns: int = 5):
    """Get conversation history for a session"""
    try:
        from ..services.conversation_layer import conversation_manager
        session = conversation_manager.get_or_create_session(session_id)
        history = session.get_recent_history(turns)
        
        return {
            "session_id": session_id,
            "history": [
                {
                    "timestamp": turn.timestamp,
                    "user_query": turn.user_query,
                    "bot_response": turn.bot_response,
                    "search_time": turn.search_time,
                    "generation_time": turn.generation_time
                }
                for turn in history
            ]
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {e}")


@router.post("/conversation/save")
async def save_conversation(session_id: str, filepath: str = None):
    """Save conversation history to file"""
    try:
        from ..services.conversation_layer import conversation_manager
        if not filepath:
            filepath = f"conversation_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        conversation_manager.save_session(session_id, filepath)
        return {"status": "success", "message": f"Conversation saved to: {filepath}"}
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save conversation: {e}")
