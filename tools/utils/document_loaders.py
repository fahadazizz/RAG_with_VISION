import os
from typing import Optional
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
import fitz 
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
    
    def _extract_images(self) -> dict:
        """
        Extract images from PDF and save to disk.
        
        Returns:
            Dictionary mapping page_index (0-based) to list of image paths.
        """
        images_map = {}
        doc = fitz.open(self.file_path)
        
        for page_index, page in enumerate(doc):
            image_list = page.get_images(full=True)
            page_images = []
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    image_filename = f"{self.filename}_p{page_index+1}_i{img_index+1}.{image_ext}"
                    image_path = self.img_dir / image_filename
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    page_images.append(str(image_path))
                except Exception as e:
                    print(f"Error extracting image {img_index} on page {page_index}: {e}")
            
            if page_images:
                images_map[page_index] = page_images
                
        return images_map

    def load(self) -> list[Document]:
        """
        Load PDF with text, images, and OCR using LangChain's PyMuPDFLoader.
        Also manually extracts and saves images to disk for multimodal usage.
        
        Returns:
            List of Documents (one per page)
        """
        loader = PyMuPDFLoader(
            self.file_path,
            mode="page",
            images_inner_format="markdown-img",
            images_parser=RapidOCRBlobParser(),
        )
        documents = loader.load()
        
        images_by_page = self._extract_images()
        
        metadata = self._create_metadata(self.filename)
        for doc in documents:
            existing_meta = doc.metadata
            
            page_idx = existing_meta.get("page", 0) 
            
            final_metadata = {
                **metadata,
                **existing_meta,
            }
            
            if page_idx in images_by_page:
                final_metadata["image_paths"] = images_by_page[page_idx]
            
            if "page" in existing_meta:
                 final_metadata["page"] = existing_meta["page"] + 1
            
            doc.metadata = final_metadata
            doc.page_content = doc.page_content 
        
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
        print("loading document")
        loader = Docx2txtLoader(self.file_path)
        print("document laoded")
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
        loader = WebBaseLoader(self.url)
        documents = loader.load()
        
        metadata = self._create_metadata(self.url)
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
            source: File path or URL
            
        Returns:
            Appropriate document loader
            
        Raises:
            ValueError: If source type is not supported
        """
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

