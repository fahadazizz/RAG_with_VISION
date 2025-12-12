from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    
    # Pinecone Vector Store
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(alias="PINECONE_INDEX_NAME")
    
    # Embedding Model
    # embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_model_name: str = Field(default="embeddinggemma:latest")

    
    # LLM
    llm_model_name: str = Field(default="kimi-k2-thinking:cloud")
    llm_base_url: str = Field(default="http://localhost:11434")
    llm_temperature: float = Field(default=0.7)

    
    # Chunking
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    
    # RAG Retrieval
    rag_top_k: int = Field(default=2)
    
    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=True)
    
    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Clear cache and reload settings."""
    get_settings.cache_clear()
    return get_settings()
