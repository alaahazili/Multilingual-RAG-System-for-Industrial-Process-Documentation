"""
Model Manager Service

Centralized model management to prevent reloading models on every startup.
Implements singleton pattern and global caching for all models used in the system.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


class ModelManager:
    """Centralized model manager with singleton pattern and caching."""
    
    _instance = None
    _initialized = False
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._models = {}
            self._initialized = True
            logger.info("ModelManager initialized (singleton)")
    
    def register_model(self, model_key: str, model_instance: Any, model_type: str = "unknown"):
        """Register a model instance in the global cache."""
        if model_key in self._models:
            logger.warning(f"Model {model_key} already registered, overwriting")
        
        self._models[model_key] = {
            "instance": model_instance,
            "type": model_type,
            "loaded": True
        }
        logger.info(f"Model {model_key} ({model_type}) registered in cache")
    
    def get_model(self, model_key: str) -> Optional[Any]:
        """Get a cached model instance."""
        if model_key in self._models:
            model_info = self._models[model_key]
            if model_info["loaded"]:
                logger.debug(f"Retrieved cached model: {model_key}")
                return model_info["instance"]
            else:
                logger.warning(f"Model {model_key} is not loaded")
                return None
        else:
            logger.warning(f"Model {model_key} not found in cache")
            return None
    
    def is_model_loaded(self, model_key: str) -> bool:
        """Check if a model is loaded and cached."""
        return model_key in self._models and self._models[model_key]["loaded"]
    
    def list_models(self) -> Dict[str, Any]:
        """List all cached models."""
        return {
            key: {
                "type": info["type"],
                "loaded": info["loaded"]
            }
            for key, info in self._models.items()
        }
    
    def clear_cache(self):
        """Clear all cached models (use with caution)."""
        self._models.clear()
        logger.info("Model cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the model cache."""
        total_models = len(self._models)
        loaded_models = sum(1 for info in self._models.values() if info["loaded"])
        
        return {
            "total_models": total_models,
            "loaded_models": loaded_models,
            "cache_size": len(self._models),
            "models": self.list_models()
        }


# Global model manager instance
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    return model_manager

