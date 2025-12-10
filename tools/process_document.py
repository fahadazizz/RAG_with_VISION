from datetime import datetime
from langchain_core.documents import Document
from tools.load_documents import load_document
from tools.text_cleaner import clean_document_text
from tools.utils.text_chunker import TextChunker


def process_document(
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    clean_text: bool = True,
) -> list[Document]:
    """
    Complete document processing pipeline: load → clean → chunk.
    This is the main entry point for document ingestion.
    
    Args:
        source: File path or URL
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        clean_text: Whether to apply text cleaning
        
    Returns:
        List of Document chunks ready for embedding
    """
    # Load document
    documents = load_document(source)
    
    # Extract filename from source
    if source.startswith(("http://", "https://")):
        filename = source
    else:
        from pathlib import Path
        filename = Path(source).name
    
    # Process all document pages
    all_chunks = []
    timestamp = datetime.now().isoformat()
    
    for doc in documents:
        text = doc.page_content
        
        # Clean text if requested
        if clean_text:
            text = clean_document_text(text)
        
        # Skip empty documents
        if not text.strip():
            continue
        
        # Create metadata
        metadata = {
            "filename": filename,
            "timestamp": timestamp,
        }
        
        # Preserve page number if present
        if "page" in doc.metadata:
            metadata["page"] = doc.metadata["page"]
        
        # Chunk the text
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk(text, metadata)
        all_chunks.extend(chunks)
    
    return all_chunks
