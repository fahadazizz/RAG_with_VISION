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
    
    def multimodal_search(
        self,
        text_query: str | None = None,
        image_query_path: str | None = None,
        k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Perform multimodal retrieval.
        - Text Query: Embed with text model.
        - Image Query: Embed with CLIP.
        - Both: Fuse embeddings (mean of normalized vectors).
        
        Args:
            text_query: Optional text query
            image_query_path: Optional path to query image
            k: Top k results
            
        Returns:
            List of (Document, score)
        """
        import numpy as np
        
        text_emb = None
        if text_query:
            text_emb = self._embedding_model.embed_query(text_query)
            norm = np.linalg.norm(text_emb)
            if norm > 0:
                text_emb = (np.array(text_emb) / norm).tolist()
                
        image_emb = None
        if image_query_path:
            from models.clip_model import get_clip_model
            clip = get_clip_model()
            image_emb = clip.get_image_embedding(image_query_path)
            norm = np.linalg.norm(image_emb)
            if norm > 0:
                image_emb = (np.array(image_emb) / norm).tolist()
                
        final_vec = []
        if text_emb and image_emb:
            fused = (np.array(text_emb) + np.array(image_emb)) / 2.0
            norm = np.linalg.norm(fused)
            if norm > 0:
                final_vec = (fused / norm).tolist()
            else:
                final_vec = fused.tolist()
                
        elif text_emb:
            final_vec = text_emb
        elif image_emb:
            final_vec = image_emb
        else:
            return [] 
            
        print(f"Multimodal Search: Text='{text_query}' Image='{image_query_path}'")
        
        try:
            index = self._pc.Index(self.index_name)
            results = index.query(
                vector=final_vec,
                top_k=k,
                include_metadata=True
            )
            
            docs = []
            for match in results.matches:
                metadata = match.metadata or {}
                content = metadata.pop("text", "") 
                doc = Document(page_content=content, metadata=metadata)
                docs.append((doc, match.score))
                
            return docs
            
        except Exception as e:
            print(f"Error in multimodal search: {e}")
            return []


@lru_cache()
def get_vector_store() -> VectorStoreManager:
    """
    Get a cached vector store manager instance.
    
    Returns:
        Cached VectorStoreManager instance
    """
    return VectorStoreManager()
