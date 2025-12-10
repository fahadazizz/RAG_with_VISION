from datetime import datetime
from langchain_core.documents import Document
from tools.utils.text_chunker import TextChunker
from tools.text_cleaner import clean_document_text


def chunk_documents(
    text: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    clean_text: bool = True,
) -> list[Document]:
    """
    Clean and chunk a document for RAG ingestion.
    
    Args:
        text: Raw document text
        filename: Name of the source file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        clean_text: Whether to apply text cleaning
        
    Returns:
        List of Document chunks ready for embedding
    """
    metadata = {
        "filename": filename,
        "timestamp": datetime.now().isoformat(),
    }
    
    if clean_text:
        text = clean_document_text(text)
    
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk(text, metadata)
    
    return chunks
