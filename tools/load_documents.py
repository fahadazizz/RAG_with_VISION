from langchain_core.documents import Document
from tools.utils.document_loaders import DocumentLoaderFactory


def load_document(source: str) -> list[Document]:
    """
    Load a document from any supported source (PDF, DOCX, URL).
    
    Args:
        source: File path or URL
        
    Returns:
        List of Document objects
    """
    loader = DocumentLoaderFactory.get_loader(source)
    return loader.load()
