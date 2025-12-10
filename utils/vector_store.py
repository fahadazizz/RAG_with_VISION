from typing import List, Optional, Tuple
from functools import lru_cache

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import get_settings
from models.embedding_model import get_embedding_model


class VectorStoreManager:
    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
    ):
        settings = get_settings()
        
        self.api_key = api_key or settings.pinecone_api_key
        self.index_name = index_name or settings.pinecone_index_name
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME not found in environment")
        
        # Initialize embedding model
        self._embedding_model = get_embedding_model()
        
        # Initialize Pinecone client
        self._pc = Pinecone(api_key=self.api_key)
        
        # Initialize vector store
        self._vector_store = PineconeVectorStore(
            index=self._pc.Index(self.index_name),
            embedding=self._embedding_model.get_langchain_embeddings(),
        )
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents to add per batch
            
        Returns:
            List of document IDs
        """
        all_ids = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = self._vector_store.add_documents(batch)
            all_ids.extend(ids)
        
        return all_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        return self._vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        return self._vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )
    
    def delete_by_filter(self, filter: dict) -> None:
        """
        Delete documents matching a filter.
        
        Args:
            filter: Metadata filter for documents to delete
        """
        # Get the Pinecone index directly for deletion
        index = self._pc.Index(self.index_name)
        index.delete(filter=filter)
    
    def get_retriever(self, k: int = 5, filter: Optional[dict] = None):
        """
        Get a LangChain retriever for use in chains.
        
        Args:
            k: Number of documents to retrieve
            filter: Optional metadata filter
            
        Returns:
            LangChain retriever
        """
        search_kwargs = {"k": k}
        if filter:
            search_kwargs["filter"] = filter
        
        return self._vector_store.as_retriever(
            search_kwargs=search_kwargs
        )


@lru_cache()
def get_vector_store() -> VectorStoreManager:
    """
    Get a cached vector store manager instance.
    
    Returns:
        Cached VectorStoreManager instance
    """
    return VectorStoreManager()
