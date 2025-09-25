from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Configuration
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    
    # Hugging Face API (for embeddings only)
    hf_token_embed: str = Field(default="", validation_alias="HF_TOKEN_EMBED")
    hf_inference_model: str = Field(
        default="intfloat/multilingual-e5-large",
        validation_alias="HF_INFERENCE_MODEL",
    )
    
    # Qdrant Vector Database
    qdrant_host: str = Field(default="localhost", validation_alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, validation_alias="QDRANT_PORT")
    qdrant_collection: str = Field(default="documents", validation_alias="QDRANT_COLLECTION")
    
    # Vector Configuration
    vector_size: int = 1024  # E5-large dimension
    
    # Model Cache Directory
    model_cache_dir: str = Field(default="models_cache", validation_alias="MODEL_CACHE_DIR")
    
    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", validation_alias="LOG_FILE")

    class Config:
        env_file = ".env"
        case_sensitive = False
        protected_namespaces = ('settings_',)


settings = Settings()


