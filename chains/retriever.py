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
        self._score_threshold = settings.rag_score_threshold
        self._rerank_top_k = settings.rag_rerank_top_k
    
    def retrieve(
        self,
        query: str,
        filter: Optional[dict] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve documents with score filtering and reranking.
        
        Process:
        1. Get top_k documents from vector store
        2. Filter by score threshold
        3. Sort by score and return top rerank_top_k
        
        Args:
            query: User query
            filter: Optional metadata filter
            
        Returns:
            List of top reranked documents with scores
        """
        # Get initial results with scores
        results = self._vector_store.similarity_search_with_score(
            query=query,
            k=self._top_k,
            filter=filter,
        )
        
        retrieval_results = [
            RetrievalResult(document=doc, score=score)
            for doc, score in results
        ]
        
        filtered_results = [
            r for r in retrieval_results
            if r.score >= self._score_threshold
        ]
        
        sorted_results = sorted(
            filtered_results,
            key=lambda x: x.score,
            reverse=True
        )
        
        return sorted_results[:self._rerank_top_k]
    
    def get_context_string(
        self,
        query: str,
        filter: Optional[dict] = None,
        include_sources: bool = True,
    ) -> str:
        """
        Get formatted context string for LLM augmentation.
        Uses retrieve() which always applies score filtering and reranking.
        
        Args:
            query: User query
            filter: Optional metadata filter
            include_sources: Whether to include source information
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query=query, filter=filter)
        
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            if include_sources:
                source = result.metadata.get("filename", "Unknown")
                context_parts.append(
                    f"[Source {i}: {source}]\n{result.content}"
                )
            else:
                context_parts.append(f"[{i}] {result.content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_sources(
        self,
        query: str,
        filter: Optional[dict] = None,
    ) -> List[dict]:
        """
        Get source information for retrieved documents.
        Uses retrieve() which always applies score filtering and reranking.
        
        Args:
            query: User query
            filter: Optional metadata filter
            
        Returns:
            List of source metadata dictionaries
        """
        results = self.retrieve(query=query, filter=filter)
        return [result.metadata for result in results]


def get_retriever() -> RAGRetriever:
    """Factory function to get a retriever instance."""
    return RAGRetriever()
