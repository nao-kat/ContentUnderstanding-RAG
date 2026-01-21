"""Configuration management for the application."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Azure AI Content Understanding
    cu_endpoint: str = ""
    cu_key: str = ""
    cu_api_version: str = "2025-11-01"
    
    # Azure AI Search
    search_endpoint: str = ""
    search_index_name: str = "video-scenes"
    search_api_key: str = ""
    
    # Azure OpenAI (for embeddings)
    embedding_endpoint: Optional[str] = None
    embedding_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    embedding_deployment_name: str = "text-embedding-3-small"


# Global settings instance
def get_settings() -> Settings:
    """Get settings instance (lazy loaded)."""
    return Settings()


# Default settings for backward compatibility
settings = get_settings()
