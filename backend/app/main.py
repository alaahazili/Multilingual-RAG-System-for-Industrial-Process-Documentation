"""
PAC RAG Chatbot - FastAPI Application
Clean, production-ready FastAPI application for the RAG chatbot.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

from .api.health import router as health_router
from .api.chat_routes import router as chat_router
from .api.ui import router as ui_router
from .core.logging import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="PAC RAG Chatbot",
        description="A production-ready RAG system for PAC control philosophy documents",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(health_router, prefix="/api", tags=["Health"])
    app.include_router(chat_router, prefix="/api", tags=["Chat"])
    app.include_router(ui_router, tags=["UI"])

    # Mount static files if they exist
    static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.on_event("startup")
    async def startup_event():
        """Application startup event"""
        logger.info("🚀 PAC RAG Chatbot starting up...")
        logger.info("✅ FastAPI application initialized")
        logger.info("✅ CORS middleware configured")
        logger.info("✅ API routes registered")
        
        # Preload models to prevent reloading on first request
        try:
            logger.info("🔄 Preloading models for optimal performance...")
            
            # Preload embedding model
            from .services.embedding_layer.embedding_service import EmbeddingService, EmbeddingConfig
            embedding_config = EmbeddingConfig(
                model_name="intfloat/multilingual-e5-large",
                use_hf_api=False,  # Force self-hosted
                normalize_embeddings=True,
                batch_size=4
            )
            embedding_service = EmbeddingService(embedding_config)
            logger.info("✅ Embedding model preloaded")
            
            # Preload reranker model
            from .services.retrieval_layer.reranker import Reranker, RerankerConfig
            reranker_config = RerankerConfig(
                model_name="BAAI/bge-reranker-v2-m3",
                max_length=512,
                device="auto",
                batch_size=8,
                normalize_scores=True
            )
            reranker = Reranker(reranker_config)
            logger.info("✅ Reranker model preloaded")
            
            # Check Ollama availability
            try:
                logger.info("🔄 Checking Ollama availability...")
                from .api.chat_routes import get_generative_answerer
                answerer = get_generative_answerer()
                
                # Check Ollama availability
                try:
                    await answerer.ollama_service.check_availability()
                    logger.info("✅ Ollama (Mistral-7B-Instruct) available!")
                except Exception as e:
                    logger.warning(f"⚠️ Ollama not available: {e}")
            except Exception as e:
                logger.warning(f"⚠️ Ollama check failed: {e}")
            
            logger.info("✅ All models preloaded successfully - no more reloading!")
            
        except Exception as e:
            logger.error(f"❌ Error preloading models: {e}")
            logger.warning("⚠️ Models will be loaded on first request (slower performance)")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event"""
        logger.info("🛑 PAC RAG Chatbot shutting down...")

    @app.get("/", tags=["Root"])
    def root():
        """Root endpoint with API information"""
        return {
            "message": "PAC RAG Chatbot API",
            "version": "1.0.0",
            "description": "Production-ready RAG system for PAC control philosophy documents",
            "endpoints": {
                "docs": "/docs",
                "health": "/api/health",
                "chat": "/api/chat",
                "model_status": "/api/model-status",
                "ui": "/"
            },
            "features": [
                "HTML Document Processing",
                "Multilingual E5 Embeddings",
                "Qdrant Vector Search",
                "BGE Reranker",
                "Mistral-7B-Instruct Generation (Ollama)",
                "Conversation Memory",
                "Real-time Chat Interface"
            ]
        }

    return app


# Create the application instance
app = create_app()


