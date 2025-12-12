from typing import List, Optional
from dataclasses import dataclass

from langchain_core.documents import Document

from config import get_settings
from utils.vector_store import get_vector_store


@dataclass
class RetrievalResult:
    document: Document
    score: float
    
    @property
    def content(self) -> str:
        return self.document.page_content
    
    @property
    def metadata(self) -> dict:
        return self.document.metadata


class RAGRetriever:
    def __init__(
        self,
    ):
        settings = get_settings()
        
        self._vector_store = get_vector_store()
        self._top_k = settings.rag_top_k
    
    def retrieve(
        self,
        query: str,
        image_query_path: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using Multimodal Fusion or Standard Search.
        
        Args:
            query: User text query
            image_query_path: Optional path to query image
            
        Returns:
            List of top documents with scores
        """
        if image_query_path:
            # use multimodal search
            results = self._vector_store.multimodal_search(
                text_query=query,
                image_query_path=image_query_path,
                k=self._top_k
            )
            print(f"Retrieved {len(results)} documents via Multimodal Fusion")
        else:
            # Standard text search
            results = self._vector_store.similarity_search_with_score(
                query=query,
                k=self._top_k,
            )
            print(f"Retrieved {len(results)} documents via Cosine Similarity")

        for i, (doc, score) in enumerate(results):
             filename = doc.metadata.get('filename', 'Unknown')
             page = f", Page {doc.metadata.get('page')}" if 'page' in doc.metadata else ""
             print(f" - Doc {i}: score={score:.4f}, source={filename}{page}")
        
        return [
            RetrievalResult(document=doc, score=score)
            for doc, score in results
        ]
    
    def format_context(self, results: List[RetrievalResult]) -> str:
        """
        Format retrieved results into a context string.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Formatted string for LLM prompt
        """
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.metadata.get("filename", "Unknown")
            page_info = f", Page {result.metadata.get('page')}" if "page" in result.metadata else ""
            
            context_parts.append(
                f"[Source {i}: {source}{page_info}]\n{result.content}"
            )
        
        return "\n\n---\n\n".join(context_parts)


def get_retriever() -> RAGRetriever:
    return RAGRetriever()
