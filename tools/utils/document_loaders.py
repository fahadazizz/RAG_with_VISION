import os
from typing import Optional
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import WebBaseLoader


class MetaDATAExtractor():
  
    def _create_metadata(self, filename: str) -> dict:
        """
        Create standard metadata with filename and timestamp.
        
        Args:
            filename: Name of the source file
            
        Returns:
            Metadata dictionary
        """
        return {
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
        }


class PDFLoader(MetaDATAExtractor):
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.filename = Path(file_path).name
    
    def load(self) -> list[Document]:
        """
        Load PDF document.
        
        Returns:
            List of Document objects (one per page)
        """
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        
        metadata = self._create_metadata(self.filename)
        for doc in documents:
            page_num = doc.metadata.get("page", 0)
            doc.metadata = {
                **metadata,
                "page": page_num,
            }
        
        return documents


class DOCXLoader(MetaDATAExtractor):
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.filename = Path(file_path).name
    
    def load(self) -> list[Document]:
        """
        Load DOCX document.
        
        Returns:
            List of Document objects
        """
        loader = Docx2txtLoader(self.file_path)
        documents = loader.load()
        
        # Update metadata
        metadata = self._create_metadata(self.filename)
        for doc in documents:
            doc.metadata = metadata
        
        return documents


class URLLoader(MetaDATAExtractor):
    
    def __init__(self, url: str):
        self.url = url
    
    def load(self) -> list[Document]:
        """
        Load content from URL.
        
        Returns:
            List of Document objects
        """
        loader = WebBaseLoader(
            web_paths=[self.url],
            bs_kwargs={
                "parse_only": None, 
            }
        )
        documents = loader.load()
        
        # Update metadata with URL as filename
        metadata = self._create_metadata(self.url)
        for doc in documents:
            doc.metadata = metadata
        
        return documents


class DocumentLoaderFactory:
    
    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
    
    @classmethod
    def get_loader(cls, source: str) -> BaseDocumentLoader:
        """
        Get appropriate loader for the source.
        
        Args:
            source: File path or URL
            
        Returns:
            Appropriate document loader
            
        Raises:
            ValueError: If source type is not supported
        """
        # Check if it's a URL
        if source.startswith(("http://", "https://")):
            return URLLoader(source)
        
        path = Path(source)
        extension = path.suffix.lower()
        
        if extension == ".pdf":
            return PDFLoader(source)
        elif extension == ".docx":
            return DOCXLoader(source)
        else:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {cls.SUPPORTED_EXTENSIONS}"
            )
    
    @classmethod
    def is_supported(cls, source: str) -> bool:
        """
        Check if a source is supported.
        
        Args:
            source: File path or URL
            
        Returns:
            True if supported, False otherwise
        """
        if source.startswith(("http://", "https://")):
            return True
        
        extension = Path(source).suffix.lower()
        return extension in cls.SUPPORTED_EXTENSIONS

