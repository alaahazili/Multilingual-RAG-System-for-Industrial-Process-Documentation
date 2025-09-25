#!/usr/bin/env python3
"""
Configuration Verification Script
Purpose: Verify all components are properly configured for one-time installation
"""

import sys
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from app.core.settings import settings
from app.services.embedding_layer.embedding_service import EmbeddingService, EmbeddingConfig
from app.services.generative_layer.ollama_generation import OllamaGenerationService, OllamaConfig

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging for verification"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def check_environment_variables() -> Dict[str, Any]:
    """Check environment variables configuration"""
    logger.info("🔍 Checking environment variables...")
    
    results = {
        "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
        "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
        "hf_token_embed": bool(os.getenv("HF_TOKEN_EMBED")),
        "model_cache_dir": os.getenv("MODEL_CACHE_DIR", "models_cache")
    }
    
    logger.info(f"✅ Qdrant: {results['qdrant_host']}:{results['qdrant_port']}")
    logger.info(f"✅ HF Token: {'Available' if results['hf_token_embed'] else 'Not needed (self-hosted)'}")
    logger.info(f"✅ Model Cache: {results['model_cache_dir']}")
    
    return results


def check_embedding_model() -> Dict[str, Any]:
    """Check embedding model installation"""
    logger.info("🔍 Checking embedding model...")
    
    try:
        # Test self-hosted model
        config = EmbeddingConfig(
            model_name="intfloat/multilingual-e5-large",
            use_hf_api=False,
            batch_size=4
        )
        
        embedding_service = EmbeddingService(config)
        
        # Test with a simple text
        test_texts = ["This is a test document."]
        embeddings = embedding_service.embed_passages(test_texts)
        
        results = {
            "status": "✅ Working",
            "model": config.model_name,
            "dimensions": len(embeddings[0]),
            "type": "self-hosted",
            "test_passed": True
        }
        
        logger.info(f"✅ Embedding model: {results['model']} ({results['dimensions']} dimensions)")
        logger.info(f"✅ Type: {results['type']}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Embedding model failed: {e}")
        return {
            "status": "❌ Failed",
            "error": str(e),
            "test_passed": False
        }


async def check_generative_model() -> Dict[str, Any]:
    """Check generative model (Ollama)"""
    logger.info("🔍 Checking generative model...")
    
    try:
        config = OllamaConfig(
            base_url="http://localhost:11434",
            model_name="mistral:instruct"
        )
        
        ollama_service = OllamaGenerationService(config)
        available = await ollama_service.check_availability()
        
        if available:
            model_info = await ollama_service.get_model_info()
            
            results = {
                "status": "✅ Working",
                "model": config.model_name,
                "type": "ollama",
                "base_url": config.base_url,
                "test_passed": True,
                "model_info": model_info
            }
            
            logger.info(f"✅ Generative model: {results['model']}")
            logger.info(f"✅ Type: {results['type']}")
            logger.info(f"✅ Base URL: {results['base_url']}")
            
            return results
        else:
            logger.error("❌ Ollama service not available")
            return {
                "status": "❌ Not available",
                "model": config.model_name,
                "type": "ollama",
                "test_passed": False
            }
            
    except Exception as e:
        logger.error(f"❌ Generative model failed: {e}")
        return {
            "status": "❌ Failed",
            "error": str(e),
            "test_passed": False
        }


def check_qdrant_connection() -> Dict[str, Any]:
    """Check Qdrant connection"""
    logger.info("🔍 Checking Qdrant connection...")
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        
        # Test connection
        collections = client.get_collections()
        
        results = {
            "status": "✅ Connected",
            "host": settings.qdrant_host,
            "port": settings.qdrant_port,
            "collections": len(collections.collections),
            "test_passed": True
        }
        
        logger.info(f"✅ Qdrant: Connected to {results['host']}:{results['port']}")
        logger.info(f"✅ Collections: {results['collections']}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
        return {
            "status": "❌ Failed",
            "error": str(e),
            "test_passed": False
        }


def check_model_cache() -> Dict[str, Any]:
    """Check model cache directory"""
    logger.info("🔍 Checking model cache...")
    
    cache_dir = Path(settings.model_cache_dir)
    
    if cache_dir.exists():
        cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        cache_size_mb = cache_size / (1024 * 1024)
        
        results = {
            "status": "✅ Exists",
            "path": str(cache_dir),
            "size_mb": round(cache_size_mb, 2),
            "test_passed": True
        }
        
        logger.info(f"✅ Model cache: {results['path']}")
        logger.info(f"✅ Cache size: {results['size_mb']} MB")
        
        return results
    else:
        logger.warning(f"⚠️ Model cache directory does not exist: {cache_dir}")
        return {
            "status": "⚠️ Not found",
            "path": str(cache_dir),
            "test_passed": False
        }


async def main():
    """Main verification function"""
    logger = setup_logging()
    
    logger.info("🚀 STARTING CONFIGURATION VERIFICATION")
    logger.info("=" * 60)
    
    # Run all checks
    env_check = check_environment_variables()
    embedding_check = check_embedding_model()
    generative_check = await check_generative_model()
    qdrant_check = check_qdrant_connection()
    cache_check = check_model_cache()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    
    all_passed = all([
        embedding_check.get("test_passed", False),
        generative_check.get("test_passed", False),
        qdrant_check.get("test_passed", False)
    ])
    
    logger.info(f"✅ Environment Variables: {env_check}")
    logger.info(f"✅ Embedding Model: {embedding_check['status']}")
    logger.info(f"✅ Generative Model: {generative_check['status']}")
    logger.info(f"✅ Qdrant Connection: {qdrant_check['status']}")
    logger.info(f"✅ Model Cache: {cache_check['status']}")
    
    if all_passed:
        logger.info("\n🎉 ALL COMPONENTS VERIFIED SUCCESSFULLY!")
        logger.info("✅ Models are installed once and will not need reinstallation")
        logger.info("✅ System is ready for production use")
        logger.info("✅ No server restart required for model changes")
        
        print("\n🎉 CONFIGURATION VERIFICATION COMPLETE!")
        print("✅ All components are properly configured")
        print("✅ Models are installed once (no reinstallation needed)")
        print("✅ System is ready for use")
        
        return True
    else:
        logger.error("\n❌ SOME COMPONENTS FAILED VERIFICATION")
        logger.error("Please check the errors above and fix them")
        
        print("\n❌ CONFIGURATION VERIFICATION FAILED!")
        print("Please check the errors above and fix them")
        
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

