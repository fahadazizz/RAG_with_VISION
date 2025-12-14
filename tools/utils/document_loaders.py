import os
import io
from typing import Optional
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
import pymupdf4llm
from PIL import Image


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
        self.img_dir = Path("static/images")
        self.img_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[Document]:
        """
        Load PDF with text and images using pymupdf4llm.
        Extracts clean markdown text and saves images to disk for CLIP processing.
        
        Returns:
            List of Documents (one per page)
        """
        pdf_img_dir = self.img_dir / Path(self.filename).stem
        pdf_img_dir.mkdir(parents=True, exist_ok=True)
        
        md_text_with_images = pymupdf4llm.to_markdown(
            doc=self.file_path,
            page_chunks=True,      
            write_images=True,     
            image_path=str(pdf_img_dir),  
            image_format="png",    
            dpi=200               
        )
        
        # Process the results
        documents = []
        metadata = self._create_metadata(self.filename)
        
        for page_data in md_text_with_images:
            page_num = page_data.get("page", 0) + 1  
            page_text = page_data.get("text", "")
            
            # Build page metadata
            page_metadata = {
                **metadata,
                "page": page_num,
                "source": self.file_path,
            }
            
            page_images = []
            img_pattern = f"{Path(self.filename).stem}-{page_num}-"
            
            for img_file in pdf_img_dir.glob(f"{img_pattern}*.png"):
                page_images.append(str(img_file))
            
            if page_images:
                page_metadata["image_paths"] = page_images
            
            doc = Document(
                page_content=page_text,
                metadata=page_metadata
            )
            documents.append(doc)
        
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
        
        metadata = self._create_metadata(self.filename)
        for doc in documents:
            doc.metadata = metadata
        
        return documents


class DocumentLoaderFactory:
    
    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
    
    @classmethod
    def get_loader(cls, source: str):
        """
        Get appropriate loader for the source.
        
        Args:
            source: File path
            
        Returns:
            Appropriate document loader
            
        Raises:
            ValueError: If source type is not supported
        """
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
            source: File path
            
        Returns:
            True if supported, False otherwise
        """
        extension = Path(source).suffix.lower()
        return extension in cls.SUPPORTED_EXTENSIONS

