from typing import List
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from config import get_settings


class EmbeddingModel:
    
    def __init__(
        self,
    ):
        # We enforce the CLIP model here to ensure alignment with the image model
        self.model_name = "sentence-transformers/clip-ViT-L-14"
        self.device = "cpu"
        
        print(f"Loading Text Embedding Model: {self.model_name}...")
        
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True} 
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        """
        return self._embeddings.embed_documents(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query.
        """
        return self._embeddings.embed_query(query)
    
    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        return self._embeddings


@lru_cache()
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()