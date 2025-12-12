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

    def add_image_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> List[str]:
        """
        Add documents with pre-computed embeddings (for Images) directly to Pinecone.
        Bypasses the default text embedding model.
        
        Args:
            documents: List of Document objects
            embeddings: List of embedding vectors (must match len(documents))
            batch_size: Batch size
            
        Returns:
            List of IDs
        """
        import uuid
        
        index = self._pc.Index(self.index_name)
        all_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]
            
            vectors = []
            batch_ids = []
            
            for doc, emb in zip(batch_docs, batch_embs):
                doc_id = str(uuid.uuid4())
                batch_ids.append(doc_id)
                
                # Metadata must be simple types for Pinecone
                metadata = doc.metadata.copy()
                metadata["text"] = doc.page_content 
                # Ensure image_paths is not a list (Pinecone supports list of strings, but let's be safe)
                # Actually Pinecone supports list[str].
                
                vectors.append({
                    "id": doc_id, 
                    "values": emb, 
                    "metadata": metadata
                })
            
            index.upsert(vectors=vectors)
            all_ids.extend(batch_ids)
            
        print(f"Upserted {len(all_ids)} image vectors directly.")
        return all_ids
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
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

        print(f"doing similairty search with score: query='{query}' k={k}")
        return self._vector_store.similarity_search_with_score(
            query=query,
            k=k,
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


@lru_cache()
def get_vector_store() -> VectorStoreManager:
    """
    Get a cached vector store manager instance.
    
    Returns:
        Cached VectorStoreManager instance
    """
    return VectorStoreManager()
