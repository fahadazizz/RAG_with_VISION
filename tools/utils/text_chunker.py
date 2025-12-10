from typing import Optional
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from tools.utils.text_cleaner import TextCleaner, clean_document_text


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[Document]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
                     (should contain 'filename' and 'timestamp')
            
        Returns:
            List of LangChain Document objects with chunked content
        """
        base_metadata = metadata or {}
        doc = Document(page_content=text, metadata=base_metadata)
        chunks = self._splitter.split_documents([doc])
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def chunk_with_context(
        self,
        text: str,
        context_window: int = 100,
        metadata: Optional[dict] = None,
    ) -> list[Document]:
        """
        Split text into chunks with additional context from neighboring chunks.
        This improves retrieval quality by providing surrounding context.
        
        Args:
            text: Input text to chunk
            context_window: Number of characters of context from neighbors
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of Document objects with context-enhanced content
        """
        chunks = self.chunk(text, metadata)
        
        if len(chunks) <= 1:
            return chunks
        
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.page_content
            
            # Add context from previous chunk
            if i > 0:
                prev_content = chunks[i - 1].page_content
                prev_context = prev_content[-context_window:]
                content = f"[Previous context: ...{prev_context}]\n\n{content}"
            
            # Add context from next chunk
            if i < len(chunks) - 1:
                next_content = chunks[i + 1].page_content
                next_context = next_content[:context_window]
                content = f"{content}\n\n[Next context: {next_context}...]"
            
            enhanced_chunk = Document(
                page_content=content,
                metadata={**chunk.metadata, "has_context": True}
            )
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks

