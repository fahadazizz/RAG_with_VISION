from datetime import datetime
from pathlib import Path
from langchain_core.documents import Document

from config import get_settings
from tools.process_document import process_document
from tools.utils.document_loaders import DocumentLoaderFactory
from utils.vector_store import get_vector_store


class DocumentAgent:
    def __init__(self):
        settings = get_settings()
        self._vector_store = get_vector_store()
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap
    
    def ingest_file(self, file_path: str, clean_text: bool = True) -> dict:
        """
        Ingest a file into the vector store.
        
        Args:
            file_path: Path to the file (PDF or DOCX)
            clean_text: Whether to apply text cleaning
            
        Returns:
            Ingestion result with document count and IDs
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not DocumentLoaderFactory.is_supported(file_path):
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        chunks = process_document(
            source=file_path,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            clean_text=clean_text,
        )
        
        ids = self._vector_store.add_documents(chunks)
        
        return {
            "status": "success",
            "filename": path.name,
            "chunks_created": len(chunks),
            "document_ids": ids,
            "timestamp": datetime.now().isoformat(),
        }
    
    def ingest_url(self, url: str, clean_text: bool = True) -> dict:
        """
        Ingest content from a URL into the vector store.
        
        Args:
            url: URL to fetch content from
            clean_text: Whether to apply text cleaning
            
        Returns:
            Ingestion result with document count and IDs
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL format. Must start with http:// or https://")
        
        chunks = process_document(
            source=url,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            clean_text=clean_text,
        )
        
        ids = self._vector_store.add_documents(chunks)
        
        return {
            "status": "success",
            "source": url,
            "chunks_created": len(chunks),
            "document_ids": ids,
            "timestamp": datetime.now().isoformat(),
        }
    
    def delete_file(self, filename: str) -> dict:
        """Delete all chunks associated with a filename."""
        try:
            self._vector_store.delete_by_filter({"filename": filename})
            return {
                "status": "success",
                "deleted_file": filename,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


def get_document_agent() -> DocumentAgent:
    return DocumentAgent()
