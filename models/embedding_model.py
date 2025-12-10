from typing import List
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from config import get_settings


class EmbeddingModel:
    
    def __init__(
        self,
        model_name: str | None = None,
    ):
        settings = get_settings()
        
        self.model_name = model_name or settings.embedding_model_name
        self.device = "cpu"
        
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
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
    
    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get the underlying LangChain embeddings object.
        Useful for direct integration with LangChain components.
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        return self._embeddings


@lru_cache()
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()