from typing import List
from functools import lru_cache

from langchain_ollama import OllamaEmbeddings
from config import get_settings


class EmbeddingModel:
    
    def __init__(
        self,
    ):
        settings = get_settings()
        
        self.model_name = settings.embedding_model_name
        # self.device = "cpu"
        
        self._embeddings = OllamaEmbeddings(
            model=self.model_name,
            base_url=settings.llm_base_url,
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self._embeddings.embed_documents(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query.
        Optimized for query embedding (may use different pooling).
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector
        """
        return self._embeddings.embed_query(query)
    
    def get_langchain_embeddings(self) -> OllamaEmbeddings:
        return self._embeddings


@lru_cache()
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()