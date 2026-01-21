"""Configuration management for the application."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Azure AI Content Understanding
    cu_endpoint: str
    cu_key: str
    cu_api_version: str = "2025-11-01"
    
    # Azure AI Search
    search_endpoint: str
    search_index_name: str = "video-scenes"
    search_api_key: str
    
    # Azure OpenAI (for embeddings)
    embedding_endpoint: Optional[str] = None
    embedding_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    embedding_deployment_name: str = "text-embedding-3-small"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
